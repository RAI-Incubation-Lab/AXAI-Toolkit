"""Static evidence checks for model validity before explanation interpretation."""
from __future__ import annotations

from .schema import Finding


def check_performance(code: str, documentation: str) -> list[Finding]:
    findings: list[Finding] = []
    all_text = f"{code}\n{documentation}".lower()
    has_baseline = "dummyclassifier" in all_text or "dummyregressor" in all_text or "baseline" in all_text
    if not has_baseline:
        findings.append(Finding("performance", "high", "No model-validity baseline found", "Explanations of a model that does not beat a simple baseline can be technically correct but educationally misleading.", recommendation="Compare against DummyClassifier/DummyRegressor and gate interpretation on a documented improvement."))
    classification = "classifier" in all_text or "classification" in all_text
    if classification and not ("roc_auc" in all_text or "average_precision" in all_text or "pr_auc" in all_text):
        findings.append(Finding("performance", "medium", "Probability quality is not evaluated", "Classification explanations often concern probability changes, but no ROC-AUC or PR-AUC evidence was found.", recommendation="Report ROC-AUC and PR-AUC when probabilities are used; state why they are unsuitable if not."))
    if "timeseriessplit" not in all_text and ("timestamp" in all_text or "datetime" in all_text or "time series" in all_text):
        findings.append(Finding("performance", "medium", "Time-aware validation was not detected", "Random validation can use future information for a historical prediction task.", recommendation="Use chronological or rolling validation and document the forecast horizon."))
    return findings
