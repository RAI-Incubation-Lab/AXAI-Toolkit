"""Public API and command-line entry point for AXAI project audits."""
from __future__ import annotations

import argparse
from pathlib import Path

from .counterfactual import check_counterfactuals
from .discovery import discover_project, read_text
from .documentation import check_documentation
from .evidence import check_evidence
from .explainability import check_explainability
from .fairness import check_fairness
from .leakage import check_leakage
from .performance import check_performance
from .report import write_reports
from .runner import run_study
from .schema import AuditResult
from .scoring import score


def audit_project(project_root: str | Path, *, execute: bool = False, timeout: int = 180) -> AuditResult:
    inventory = discover_project(project_root)
    code = read_text(inventory.code_files)
    documentation = read_text(inventory.documents)
    findings = []
    findings += check_documentation(inventory, documentation)
    findings += check_leakage(code)
    findings += check_performance(code, documentation)
    findings += check_explainability(code, documentation)
    findings += check_fairness(code, documentation)
    findings += check_counterfactuals(code)
    evidence_findings, summary = check_evidence(inventory)
    findings += evidence_findings
    if execute:
        findings += run_study(inventory.root, timeout=timeout)
    return AuditResult(inventory.root.name, inventory, findings, score(findings), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review explainability evidence in a student project without importing its code.")
    parser.add_argument("project", help="Path to a student project directory")
    parser.add_argument("--output", default=None, help="Directory for axai_review.md and axai_review.json")
    parser.add_argument("--execute", action="store_true", help="Explicitly run trusted study.py with a timeout")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    result = audit_project(args.project, execute=args.execute, timeout=args.timeout)
    markdown, json_report = write_reports(result, args.output or Path(args.project) / "review_output")
    print(f"Wrote {markdown}")
    print(f"Wrote {json_report}")
    print(f"Overall readiness: {result.scores['overall']:.1f}/100")


if __name__ == "__main__":
    main()
