from __future__ import annotations

import json
from pathlib import Path

from axai_reviewer.main import audit_project
from example_generator import create_example


def test_audit_finds_missing_evidence(tmp_path: Path) -> None:
    project = tmp_path / "student_project"
    project.mkdir()
    (project / "study.py").write_text("from sklearn.model_selection import train_test_split\n", encoding="utf-8")
    (project / "README.md").write_text("A classification dataset.", encoding="utf-8")
    (project / "lesson.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "execution_count": None, "outputs": []}]}), encoding="utf-8")
    result = audit_project(project)
    titles = {finding.title for finding in result.findings}
    assert "No model-validity baseline found" in titles
    assert "Notebooks have no saved execution evidence" in titles
    assert result.scores["overall"] < 100


def test_generator_creates_standalone_csv_example(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("feature,target\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")
    output = create_example(tmp_path / "new_example", source, "target", "classification")
    assert (output / "study.py").exists()
    assert (output / "MODEL_CARD.md").exists()
    assert (output / source.name).exists()
