"""Inspect static project artifacts and notebook outputs without executing code."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from .schema import Finding, ProjectInventory


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()[:12]


def check_evidence(inventory: ProjectInventory) -> tuple[list[Finding], dict[str, object]]:
    findings: list[Finding] = []
    executed = 0
    output_blocks = 0
    for notebook in inventory.notebooks:
        try:
            payload = json.loads(notebook.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            findings.append(Finding("evidence", "medium", "Unreadable notebook", f"Could not parse {notebook.name}.", recommendation="Commit valid notebooks and run notebook validation in CI."))
            continue
        for cell in payload.get("cells", []):
            if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
                executed += 1
            output_blocks += len(cell.get("outputs", []))
    if inventory.notebooks and output_blocks == 0:
        findings.append(Finding("evidence", "high", "Notebooks have no saved execution evidence", "The repository cannot demonstrate that notebook workflows completed successfully.", recommendation="Execute notebooks in CI, commit concise representative outputs or exported reports, metrics.json, and explanation evidence."))
    if not inventory.output_files:
        findings.append(Finding("evidence", "medium", "No saved output artifacts found", "Inputs, predictions, and explanations cannot be reviewed together.", recommendation="Save representative evidence with raw input, truth, prediction, confidence, explanation, fidelity, model hash, and data hash."))
    return findings, {"executed_code_cells": executed, "saved_output_blocks": output_blocks, "data_hashes": {path.name: file_hash(path) for path in inventory.data_files[:10]}}
