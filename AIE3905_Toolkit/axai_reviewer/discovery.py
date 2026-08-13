"""Project discovery without importing or executing student code."""
from __future__ import annotations

from pathlib import Path

from .schema import ProjectInventory

IGNORED = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}
DOC_SUFFIXES = {".md", ".rst", ".txt", ".pdf", ".docx"}
DATA_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}
OUTPUT_SUFFIXES = {".png", ".svg", ".html", ".json", ".csv"}


def discover_project(project_root: str | Path) -> ProjectInventory:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project directory not found: {root}")
    inventory = ProjectInventory(root=root)
    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix == ".py":
            inventory.code_files.append(path)
            if path.name in {"app.py", "main.py", "study.py", "train.py", "run.py"}:
                inventory.entry_points.append(path)
        elif suffix == ".ipynb":
            inventory.notebooks.append(path)
        elif suffix in DOC_SUFFIXES:
            inventory.documents.append(path)
        if suffix in DATA_SUFFIXES and "output" not in path.parts:
            inventory.data_files.append(path)
        if suffix in OUTPUT_SUFFIXES and (any(part.startswith("output") for part in path.parts) or "artifact" in path.parts or "report" in path.name.lower()):
            inventory.output_files.append(path)
    return inventory


def read_text(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if path.suffix.lower() not in {".py", ".md", ".rst", ".txt", ".json"}:
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)
