"""Transparent, deliberately simple project-readiness scoring."""
from __future__ import annotations

from collections import Counter

from .schema import Finding

AREAS = ("documentation", "validation", "performance", "explainability", "fairness", "counterfactual", "evidence")
PENALTY = {"high": 35.0, "medium": 15.0, "low": 6.0}


def score(findings: list[Finding]) -> dict[str, float]:
    grouped: dict[str, list[Finding]] = {area: [] for area in AREAS}
    for finding in findings:
        grouped.setdefault(finding.area, []).append(finding)
    results: dict[str, float] = {}
    for area, items in grouped.items():
        results[area] = round(max(0.0, 100.0 - sum(PENALTY.get(item.severity, 10.0) for item in items)), 1)
    results["overall"] = round(sum(results[area] for area in AREAS) / len(AREAS), 1)
    results["high_findings"] = float(Counter(item.severity for item in findings)["high"])
    return results
