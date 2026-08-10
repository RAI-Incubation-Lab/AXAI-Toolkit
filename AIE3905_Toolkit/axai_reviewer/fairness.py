"""Checks for diagnostic, rather than overclaimed, fairness reporting."""
from __future__ import annotations

from .schema import Finding


def check_fairness(code: str, documentation: str) -> list[Finding]:
    text = f"{code}\n{documentation}".lower()
    findings: list[Finding] = []
    if "fair" not in text and "group_audit" not in text:
        return [Finding("fairness", "medium", "No subgroup diagnostic found", "The project contains no evidence that performance is inspected across meaningful groups.", recommendation="Add a clearly scoped subgroup diagnostic, or justify why protected/group attributes are unavailable or inappropriate.")]
    metrics = sum(term in text for term in ("true_positive_rate", "false_positive_rate", "precision", "recall", "selection_rate"))
    if metrics < 3:
        findings.append(Finding("fairness", "medium", "Fairness evidence is narrow", "Selection rate or accuracy alone cannot support a fairness conclusion.", recommendation="Report subgroup size, TPR, FPR, precision/recall, disparity ratios, and label the result as diagnostic."))
    if "confidence interval" not in text and "bootstrap" not in text:
        findings.append(Finding("fairness", "low", "Uncertainty is not reported", "Small groups can produce unstable subgroup metrics.", recommendation="Add a minimum group-size warning and bootstrap confidence intervals where a group analysis is shown."))
    return findings
