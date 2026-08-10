"""Checks for ethical and feasible counterfactual explanations."""
from __future__ import annotations

from .schema import Finding

PROTECTED_TERMS = ("sex", "gender", "race", "ethnicity", "age", "disability")


def check_counterfactuals(code: str) -> list[Finding]:
    lower = code.lower()
    if "counterfactual" not in lower:
        return []
    findings: list[Finding] = []
    if any(term in lower for term in PROTECTED_TERMS) and "sensitivity" not in lower:
        findings.append(Finding("counterfactual", "high", "Counterfactual search may alter protected attributes", "Protected attributes should be used only for sensitivity analysis, not as recommendations.", recommendation="Separate actionable changes, protected-attribute sensitivity, infeasible candidates, target_achieved, and constraint validation."))
    if "target_achieved" not in lower:
        findings.append(Finding("counterfactual", "medium", "Counterfactual success is not explicit", "A returned candidate may not actually reach the requested outcome.", recommendation="Save a target_achieved boolean and do not present unsuccessful candidates as counterfactuals."))
    return findings
