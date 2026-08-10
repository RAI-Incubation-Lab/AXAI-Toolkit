"""Documentation checks for model cards and teaching projects."""
from __future__ import annotations

from .schema import Finding, ProjectInventory

REQUIRED_TOPICS = {
    "purpose": ("intended use", "learning goal", "objective"),
    "data": ("data source", "dataset", "provenance"),
    "metrics": ("metric", "accuracy", "mae", "r2", "f1"),
    "limitations": ("limitation", "caution", "risk", "not suitable"),
    "explanations": ("shap", "lime", "explain"),
}


def check_documentation(inventory: ProjectInventory, text: str) -> list[Finding]:
    names = {path.name.lower() for path in inventory.documents}
    findings: list[Finding] = []
    if "readme.md" not in names:
        findings.append(Finding("documentation", "high", "README is missing", "A student cannot identify setup and intended learning outcomes.", recommendation="Add a README with installation, data, task, and run instructions."))
    if "model_card.md" not in names:
        findings.append(Finding("documentation", "medium", "Model card is missing", "The model's scope and limitations are not recorded.", recommendation="Add MODEL_CARD.md with intended use, data, metrics, explanation limits, and fairness caveats."))
    lowered = text.lower()
    for topic, terms in REQUIRED_TOPICS.items():
        if not any(term in lowered for term in terms):
            findings.append(Finding("documentation", "medium", f"Documentation lacks {topic}", f"No evidence of a {topic} section was found.", recommendation=f"Document {topic} explicitly and tie it to the saved run evidence."))
    return findings
