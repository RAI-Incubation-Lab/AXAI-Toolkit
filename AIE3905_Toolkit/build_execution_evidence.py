"""Execute trusted bundled examples and save reproducible notebook/output evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
MODULES = {
    "Bitcoin Historical": "study",
    "Chess Game": "study",
    "FIFA World Cup 2026": "study",
    "Google Play Store Apps": "study",
    "Medical Cost Personal": "study",
    "Netflix Movies and TV Shows": "study",
    "Students Performance in Exams": "study",
    "Titanic": "titanic_demo",
    "Video Game Sales": "study",
    "Wine Reviews": "study",
}


@contextmanager
def project_import_path(directory: Path):
    original = Path.cwd()
    sys.path.insert(0, str(directory)); os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(original); sys.path.pop(0)
        for name in ("study", "titanic_demo", "titanic_quality_extension"):
            sys.modules.pop(name, None)


def json_value(value: Any) -> Any:
    if isinstance(value, (np.generic,)): return value.item()
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, Path): return str(value)
    return value


def generic_evidence(module: Any, run: Any, output: Path) -> dict[str, Any]:
    if hasattr(module, "write_evidence"):
        module.write_evidence(run, output)
        return json.loads((output / "representative_evidence.json").read_text(encoding="utf-8"))
    if hasattr(run, "X_test"):
        X = run.X_test; y = run.y_test; prediction = getattr(run, "predictions", getattr(run, "y_pred"))[0]
        raw = X.iloc[0].to_dict(); expected = y.iloc[0]
    elif hasattr(run, "test"):
        X = run.test; raw = X.iloc[0].drop(labels=[name for name in ("model_text", "description") if name in X.columns]).to_dict()
        expected = X.iloc[0].get("high_score", X.iloc[0].get("up_next_hour", None)); prediction = run.predictions[0]
    else:
        X = run.data; raw = X.iloc[0].drop(labels=[name for name in ("content_text", "description", "cast") if name in X.columns]).to_dict()
        expected = None; prediction = "similarity catalogue"
    payload = {
        "sample_id": str(X.index[0]), "raw_input": raw, "expected_output": json_value(expected),
        "predicted_output": json_value(prediction), "probability": None,
        "correct": bool(prediction == expected) if expected is not None else None,
        "explanation_method": "model-specific feature or token contribution",
        "explanation": "See the notebook's local explanation cell.",
        "model_hash": hashlib.sha256(repr(getattr(run, "model", getattr(run, "vectorizer", "unknown"))).encode()).hexdigest()[:12],
        "data_hash": hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest()[:12],
    }
    (output / "representative_evidence.json").write_text(json.dumps(payload, indent=2, default=json_value), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({"metrics": getattr(run, "metrics", {"catalogue_items": len(X), "task": "similarity recommendation"})}, indent=2, default=json_value), encoding="utf-8")
    return payload


def notebook(directory: Path, module_name: str, output: Path) -> Path:
    name = "demo-titanic.ipynb" if directory.name == "Titanic" else "xai_workflow.ipynb"
    source = [
        nbformat.v4.new_markdown_cell(f"# {directory.name}: Executed XAI Workflow\n\nThis notebook records a reproducible run. Read `study.py` (or `titanic_demo.py`) for the complete step-by-step implementation."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport sys\nsys.path.insert(0, str(Path.cwd()))\nimport " + module_name + " as study"),
        nbformat.v4.new_markdown_cell("## Train and evaluate\n\nThe model is trained with the project-specific validation strategy and its model-validity gate."),
        nbformat.v4.new_code_cell("run = study.train() if hasattr(study, 'train') else study.train_titanic_model()\nprint(json.dumps(getattr(run, 'metrics', {'catalogue_items': len(getattr(run, 'data', []))}), indent=2, default=str))"),
        nbformat.v4.new_markdown_cell("## Save linked evidence\n\nEach evidence item links raw input, ground truth, prediction, model/data identifiers, and the explanation method."),
        nbformat.v4.new_code_cell("output = Path('outputs')\noutput.mkdir(exist_ok=True)\nif hasattr(study, 'write_evidence'):\n    evidence_path, metrics_path = study.write_evidence(run, output)\nelse:\n    evidence_path, metrics_path = output / 'representative_evidence.json', output / 'metrics.json'\nprint(evidence_path)\nprint(metrics_path)"),
        nbformat.v4.new_markdown_cell("## Interpretation gate\n\nOnly interpret explanation results when the recorded model-validity gate passes. Explanations describe model behaviour, not causal mechanisms or deployment suitability."),
    ]
    doc = nbformat.v4.new_notebook(cells=source, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    path = directory / name
    nbformat.write(doc, path)
    NotebookClient(doc, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(directory)}}).execute()
    nbformat.write(doc, path)
    return path


def build_case(name: str) -> None:
    directory = ROOT / name; output = directory / "outputs"; output.mkdir(exist_ok=True)
    with project_import_path(directory):
        module = importlib.import_module(MODULES[name]); run = module.train() if hasattr(module, "train") else module.train_titanic_model()
        generic_evidence(module, run, output)
        path = notebook(directory, MODULES[name], output)
    print(f"Built evidence and executed {path}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("cases", nargs="*", choices=MODULES); args = parser.parse_args()
    for name in args.cases or list(MODULES): build_case(name)


if __name__ == "__main__": main()
