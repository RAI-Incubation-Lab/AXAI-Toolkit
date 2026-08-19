# -*- coding: utf-8 -*-
"""AXAI-Toolkit 命令行入口。

支持：
- axai scan <path>            静态扫描 Prompt/PII/偏见风险
- axai test --entry <file>    动态测试占位命令
- axai fix --prompt-file <f>  生成并应用 Prompt 加固补丁
- axai telemetry <status|enable|disable>
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import telemetry as telemetry_module
from .rai.ast_linter import lint_python_file
from .rai.probes import run_static_prompt_scan
from .remediation.prompt_patch import apply_prompt_patch, generate_prompt_patch

app = typer.Typer(help="AXAI-Toolkit: AI transparency, safety and compliance scanner.")
console = Console()

TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".env"}


def _iter_text_files(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            yield p


@app.command()
def scan(
    path: str = typer.Argument(".", help="要扫描的文件或目录"),
    export_html: Optional[str] = typer.Option(None, "--export-html", help="导出 HTML 报告路径"),
    export_json: Optional[str] = typer.Option(None, "--export-json", help="导出 JSON 报告路径"),
):
    """对 AI 项目 / Prompt 文件执行静态风险扫描。"""
    target = Path(path)
    if not target.exists():
        console.print(f"[red]路径不存在: {target}[/red]")
        raise typer.Exit(1)

    findings = []
    total_score = 100.0
    ast_count = 0
    for file in _iter_text_files(target):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        result = run_static_prompt_scan(text)
        if result["score"] < 100:
            findings.append(
                {
                    "type": "static-scan",
                    "detail": f"{file}: score={result['score']}, "
                    f"prompt_risks={len(result['prompt_risks'])}, "
                    f"pii={len(result['pii'])}, bias={len(result['bias'])}",
                }
            )
        total_score = min(total_score, result["score"])

        # AST Linter：扫描 Python 源码中的 Prompt 拼接/注入风险
        if file.suffix.lower() == ".py":
            for ast_finding in lint_python_file(file):
                ast_count += 1
                findings.append(
                    {
                        "type": ast_finding.get("type", "ast-lint"),
                        "detail": f"{file}:{ast_finding.get('line', '?')} {ast_finding.get('detail', '')}",
                    }
                )
                total_score = max(0.0, total_score - 5)

    table = Table(title="AXAI Scan Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Scanned path", str(target))
    table.add_row("Files scanned", str(len(list(_iter_text_files(target)))))
    table.add_row("Findings", str(len(findings)))
    table.add_row("Score", f"{total_score:.1f}/100")
    console.print(table)

    if findings:
        console.print("[yellow]Findings:[/yellow]")
        for item in findings:
            console.print(f"  - {item['detail']}")

    if export_html:
        from .reporting.html_report import export_html_report, generate_html_report

        report = generate_html_report(
            title="AXAI Static Scan Report",
            summary=f"Scan path: {target}",
            findings=findings,
            scores={
                "Security": max(0, total_score),
                "PII": max(0, 100 - 10 * sum(len(run_static_prompt_scan(p.read_text(encoding='utf-8', errors='ignore'))['pii']) for p in _iter_text_files(target) if p.suffix.lower() in TEXT_SUFFIXES)),
                "Bias": max(0, 100 - 10 * sum(len(run_static_prompt_scan(p.read_text(encoding='utf-8', errors='ignore'))['bias']) for p in _iter_text_files(target) if p.suffix.lower() in TEXT_SUFFIXES)),
                "AST": max(0, 100 - 10 * ast_count),
                "Prompt": max(0, total_score),
                "Overall": max(0, total_score),
            },
        )
        out = export_html_report(report, export_html)
        console.print(f"[green]HTML report saved: {out}[/green]")

    if export_json:
        from .reporting.json_report import export_json_report, generate_json_report

        report_json = generate_json_report(
            title="AXAI Static Scan Report",
            summary=f"Scan path: {target}",
            findings=findings,
            scores={
                "Security": max(0, total_score),
                "PII": max(0, 100 - 10 * sum(len(run_static_prompt_scan(p.read_text(encoding='utf-8', errors='ignore'))['pii']) for p in _iter_text_files(target) if p.suffix.lower() in TEXT_SUFFIXES)),
                "Bias": max(0, 100 - 10 * sum(len(run_static_prompt_scan(p.read_text(encoding='utf-8', errors='ignore'))['bias']) for p in _iter_text_files(target) if p.suffix.lower() in TEXT_SUFFIXES)),
                "AST": max(0, 100 - 10 * ast_count),
                "Prompt": max(0, total_score),
                "Overall": max(0, total_score),
            },
        )
        out_json = export_json_report(report_json, export_json)
        console.print(f"[green]JSON report saved: {out_json}[/green]")


@app.command()
def test(
    entry: str = typer.Option(..., "--entry", help="Agent/模型入口，例如 agent.py:run_agent"),
    suite: str = typer.Option("xai,safety,bias", "--suite", help="测试套件，逗号分隔"),
    export_html: Optional[str] = typer.Option(None, "--export-html", help="导出 HTML 报告路径"),
):
    """对 Agent/Prompt 入口执行自动化 XAI 与安全红队测试（教学占位实现）。"""
    console.print(f"[cyan]Test entry:[/cyan] {entry}")
    console.print(f"[cyan]Suite:[/cyan] {suite}")
    console.print("[yellow]该命令为教学版占位实现；接入真实模型后会自动执行越狱/偏见/PII 探针。[/yellow]")

    if export_html:
        from .reporting.html_report import export_html_report, generate_html_report

        report = generate_html_report(
            title="AXAI Dynamic Test Report",
            summary=f"Entry: {entry}, Suite: {suite}",
            findings=[{"type": "placeholder", "detail": "真实模型探针待接入"}],
            scores={"Safety": 80, "XAI": 70, "Bias": 75},
        )
        out = export_html_report(report, export_html)
        console.print(f"[green]HTML report saved: {out}[/green]")


@app.command()
def fix(
    prompt_file: Optional[str] = typer.Option(None, "--prompt-file", help="原始 Prompt 文件路径，默认 system_prompt.txt"),
    output_file: Optional[str] = typer.Option(None, "--output", help="加固后 Prompt 输出路径"),
    apply_patch: bool = typer.Option(False, "--apply-prompt-patch", help="直接写出加固后的 Prompt"),
):
    """生成处方级 Prompt 加固补丁。"""
    src = Path(prompt_file) if prompt_file else Path("system_prompt.txt")
    if not src.exists():
        console.print(f"[red]文件不存在: {src}[/red]")
        raise typer.Exit(1)
    original = src.read_text(encoding="utf-8")
    patch = generate_prompt_patch(original)
    console.print("[bold green]Prompt Hardening Patch:[/bold green]")
    console.print(patch["diff"])
    if apply_patch:
        out = apply_prompt_patch(str(src), output_file)
        console.print(f"[green]Hardened prompt written to: {out}[/green]")


@app.command()
def telemetry(
    action: str = typer.Argument("status", help="status / enable / disable"),
):
    """管理隐私友好型匿名遥测。"""
    if action == "status":
        state = "enabled" if telemetry_module.is_enabled() else "disabled"
        console.print(f"Telemetry: [bold]{state}[/bold]")
    elif action == "enable":
        telemetry_module.enable()
        console.print("Telemetry enabled (anonymized metadata only).")
    elif action == "disable":
        telemetry_module.disable()
        console.print("Telemetry disabled.")
    else:
        console.print("[red]Unknown action. Use status/enable/disable.[/red]")
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()
