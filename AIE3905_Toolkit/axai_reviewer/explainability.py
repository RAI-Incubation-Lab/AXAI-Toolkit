"""Checks that explanations are queryable, faithful, and tied to evidence."""
from __future__ import annotations

from .schema import Finding


def check_explainability(code: str, documentation: str) -> list[Finding]:
    text = f"{code}\n{documentation}".lower()
    findings: list[Finding] = []
    has_method = "shap" in text or "lime" in text or "permutation_importance" in text
    if not has_method:
        findings.append(Finding("explainability", "high", "No explanation method found", "The project does not show how a prediction is explained.", recommendation="Add global and local explanation evidence appropriate to the model and task."))
    if "predictions.csv" in text and "predict_fn" not in text and "pipeline.predict" not in text:
        findings.append(Finding("explainability", "high", "Static predictions appear non-queryable", "A table of fixed predictions cannot support perturbed-input explanations or counterfactual search.", recommendation="Expose a stable predict_fn or label this as a non-queryable black box with limited explanation claims."))
    if "lime" in text and "encoded_frame" in text:
        findings.append(Finding("explainability", "high", "LIME may perturb encoded categories", "One-hot features can form invalid combinations when perturbed independently.", recommendation="Run LIME in the original feature space, call the complete pipeline, declare categorical values, and record local fidelity."))
    if "lime" in text and "score" not in text:
        findings.append(Finding("explainability", "medium", "LIME fidelity is not recorded", "A local surrogate explanation is not interpretable without its fit quality.", recommendation="Save explanation.score and reject or flag low-fidelity explanations."))
    if "shap_values" in text and "sample" not in text:
        findings.append(Finding("explainability", "medium", "SHAP sampling rule is not visible", "Explaining every test item can be slow and irreproducible in an interactive dashboard.", recommendation="Use a seeded representative sample, cache results, show progress, and record sample size in evidence."))
    return findings
