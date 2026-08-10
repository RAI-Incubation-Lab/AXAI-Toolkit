"""Data structures shared by the AXAI student-project reviewer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Finding:
    area: str
    severity: str
    title: str
    detail: str
    evidence: str = ""
    recommendation: str = ""


@dataclass
class ProjectInventory:
    root: Path
    code_files: list[Path] = field(default_factory=list)
    notebooks: list[Path] = field(default_factory=list)
    documents: list[Path] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)
    entry_points: list[Path] = field(default_factory=list)


@dataclass
class AuditResult:
    project: str
    inventory: ProjectInventory
    findings: list[Finding]
    scores: dict[str, float]
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "inventory": {
                key: [str(path) for path in value] if isinstance(value, list) else str(value)
                for key, value in asdict(self.inventory).items()
            },
            "findings": [asdict(item) for item in self.findings],
            "scores": self.scores,
            "summary": self.summary,
        }
