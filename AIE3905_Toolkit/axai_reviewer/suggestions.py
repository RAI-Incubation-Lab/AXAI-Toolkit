"""Turn findings into a short, prioritized teaching action list."""
from __future__ import annotations

from .schema import Finding

ORDER = {"high": 0, "medium": 1, "low": 2}


def prioritized_suggestions(findings: list[Finding], limit: int = 8) -> list[Finding]:
    return sorted(findings, key=lambda item: (ORDER.get(item.severity, 3), item.area, item.title))[:limit]
