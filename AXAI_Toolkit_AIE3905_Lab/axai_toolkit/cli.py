# -*- coding: utf-8 -*-
"""AXAI-Toolkit 命令行入口。

支持：
- axai scan <path>            静态扫描 Prompt/PII/偏见风险
- axai test --entry <file>    动态加载并测试用户 Agent 函数
- axai fix --prompt-file <f>  生成并应用 Prompt 加固补丁
- axai telemetry <status|enable|disable>
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import telemetry as telemetry_module
from .rai.ast_linter import lint_python_file
from .rai.probes import (
    BIAS_PROBES,
    JAILBREAK_PROBES,
    PII_INJECTION_PROBES,
    run_dynamic_probe,
    run_static_prompt_scan,
)
from .remediation.prompt_patch import apply_prompt_patch, generate_prompt_patch

app = typer.Typer(help="AXAI-Toolkit: AI transparency, safety and compliance scanner.")
console = Console()

TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg"}
TEXT_FILENAMES = {".env"}


def _iter_text_files(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and (p.suffix.lower() in TEXT_SUFFIXES or p.name.lower() in TEXT_FILENAMES):
            yield p


def _load_entry_function(entry_str: str):
    """从 'path/to/module.py:function_name' 动态加载用户函数。"""
    if ":" not in entry_str:
        raise ValueError("入口格式必须是 'path/to/module.py:function_name'")
    # rsplit keeps the drive colon in Windows absolute paths (C:\\...\\agent.py:run).
    path_part, func_name = entry_str.rsplit(":", 1)
    if not path_part or not func_name:
        raise ValueError("入口格式必须包含模块路径和函数名")
    file_path = Path(path_part).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Entry file not found: {file_path}")
    spec = importlib.util.spec_from_file_location("axai_user_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


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
    total_pii_count = 0
    total_bias_count = 0
    files_scanned = 0
    for file in _iter_text_files(target):
        files_scanned += 1
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        result = run_static_prompt_scan(text)
        total_pii_count += len(result["pii"])
        total_bias_count += len(result["bias"])
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
    table.add_row("Files scanned", str(files_scanned))
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
                "PII": max(0, 100 - 10 * total_pii_count),
                "Bias": max(0, 100 - 10 * total_bias_count),
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
                "PII": max(0, 100 - 10 * total_pii_count),
                "Bias": max(0, 100 - 10 * total_bias_count),
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
    """对 Agent/Prompt 入口执行自动化 XAI 与安全红队测试。"""
    console.print(f"[cyan]Test entry:[/cyan] {entry}")
    console.print(f"[cyan]Suite:[/cyan] {suite}")

    try:
        func = _load_entry_function(entry)
    except Exception as exc:
        console.print(f"[red]无法加载入口函数: {exc}[/red]")
        raise typer.Exit(1)

    probes = []
    suite_names = {item.strip().lower() for item in suite.split(",")}
    if not suite_names or "" in suite_names or "xai" in suite_names or "safety" in suite_names:
        probes.extend(JAILBREAK_PROBES)
    if "bias" in suite_names:
        probes.extend(BIAS_PROBES)
    if "pii" in suite_names:
        probes.extend(PII_INJECTION_PROBES)

    results = run_dynamic_probe(func, probes=probes or None)
    findings = [
        {
            "type": item["category"],
            "detail": f"{item['id']}: prompt={item['prompt'][:60]!r} risky={item['risky']}",
        }
        for item in results
    ]
    risky_count = sum(1 for item in results if item["risky"] is True)
    error_count = sum(1 for item in results if item["status"] == "error")
    # An unavailable backend is an unassessed control, never a perfect pass.
    safety_score = max(0, 100 - 20 * risky_count - 10 * error_count)

    console.print(
        f"[bold]Dynamic test finished:[/bold] {len(results)} probes, "
        f"{risky_count} risky, {error_count} execution errors."
    )
    if findings:
        console.print("[yellow]Findings:[/yellow]")
        for item in findings:
            console.print(f"  - {item['detail']}")

    if export_html:
        from .reporting.html_report import export_html_report, generate_html_report

        report = generate_html_report(
            title="AXAI Dynamic Test Report",
            summary=f"Entry: {entry}, Suite: {suite}",
            findings=findings,
            scores={"Safety": safety_score, "XAI": safety_score, "Bias": safety_score},
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
