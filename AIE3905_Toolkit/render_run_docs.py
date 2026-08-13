"""Render a checked-in run-evidence document from generated metrics artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render(project: Path) -> None:
    metrics_path = project / "outputs" / "metrics.json"
    evidence_path = project / "outputs" / "representative_evidence.json"
    if not metrics_path.exists() or not evidence_path.exists():
        raise FileNotFoundError(f"Missing execution artifacts for {project.name}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    lines = ["# Generated Run Evidence", "", "This file is generated from the current trusted execution. Do not edit it manually.", "", "## Metrics", ""]
    for key, value in metrics.get("metrics", {}).items(): lines.append(f"- **{key}:** {value}")
    lines += ["", "## Validation", "", f"- **Split:** {metrics.get('split', 'project-specific documented split')}", f"- **Model-validity gate:** {metrics.get('validity_gate_passed', 'not applicable')}", "", "## Representative evidence", "", f"- **Sample ID:** {evidence['sample_id']}", f"- **Expected output:** {evidence['expected_output']}", f"- **Predicted output:** {evidence['predicted_output']}", f"- **Explanation method:** {evidence['explanation_method']}", f"- **Model hash:** {evidence['model_hash']}", f"- **Data hash:** {evidence['data_hash']}"]
    (project / "RUN_EVIDENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for project in ROOT.iterdir():
        if project.is_dir() and (project / "outputs" / "metrics.json").exists(): render(project)
