"""Stable Markdown and JSON reports suitable for classroom feedback."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import AuditResult
from .suggestions import prioritized_suggestions


def markdown_report(result: AuditResult) -> str:
    lines = [f"# AXAI Project Review: {result.project}", "", "## Readiness scores", ""]
    for key, value in result.scores.items():
        if key != "high_findings":
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value:.1f}/100")
    lines += ["", "## Evidence inventory", ""]
    for name in ("code_files", "notebooks", "documents", "data_files", "output_files", "entry_points"):
        lines.append(f"- **{name.replace('_', ' ').title()}:** {len(getattr(result.inventory, name))}")
    for key, value in result.summary.items():
        if key != "data_hashes":
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    lines += ["", "## Priority recommendations", ""]
    for item in prioritized_suggestions(result.findings):
        lines += [f"### [{item.severity.upper()}] {item.title}", item.detail, "", f"**Recommendation:** {item.recommendation}", ""]
    if not result.findings:
        lines.append("No static concerns were detected. Confirm results with an executed, isolated review run.")
    return "\n".join(lines)


def write_reports(result: AuditResult, output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "axai_review"
    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    json_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return markdown_path, json_path
