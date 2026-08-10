"""One stable teacher-facing entry point for AXAI toolkit operations."""
from __future__ import annotations

import argparse
from pathlib import Path

from axai_reviewer.main import audit_project
from axai_reviewer.report import write_reports
from example_generator import create_example


def audit(args: argparse.Namespace) -> None:
    result = audit_project(args.project, execute=args.execute, timeout=args.timeout)
    markdown, json_report = write_reports(result, args.output or Path(args.project) / "review_output")
    print(f"Overall readiness: {result.scores['overall']:.1f}/100")
    print(f"Markdown report: {markdown}")
    print(f"JSON report: {json_report}")


def audit_all(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for child in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if (child / "study.py").exists() or (child / "README.md").exists():
            result = audit_project(child, execute=False)
            markdown, _ = write_reports(result, child / "review_output")
            print(f"{child.name}: {result.scores['overall']:.1f}/100 -> {markdown}")


def create(args: argparse.Namespace) -> None:
    output = create_example(args.destination, args.data, args.target, args.task, not args.no_copy_data)
    print(f"Created standalone example: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AXAI toolkit harness for project review and new examples.")
    subparsers = parser.add_subparsers(required=True)
    audit_parser = subparsers.add_parser("audit", help="Audit one student project")
    audit_parser.add_argument("project")
    audit_parser.add_argument("--output")
    audit_parser.add_argument("--execute", action="store_true", help="Run trusted study.py explicitly")
    audit_parser.add_argument("--timeout", type=int, default=180)
    audit_parser.set_defaults(func=audit)
    all_parser = subparsers.add_parser("audit-all", help="Audit every example below a root directory")
    all_parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parent)
    all_parser.set_defaults(func=audit_all)
    create_parser = subparsers.add_parser("create-example", help="Generate a standalone CSV example")
    create_parser.add_argument("destination")
    create_parser.add_argument("--data", required=True)
    create_parser.add_argument("--target", required=True)
    create_parser.add_argument("--task", choices=["classification", "regression"], required=True)
    create_parser.add_argument("--no-copy-data", action="store_true")
    create_parser.set_defaults(func=create)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
