# -*- coding: utf-8 -*-
"""JSON 报告导出。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def generate_json_report(
    title: str,
    summary: str,
    findings: list[dict],
    scores: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """生成 JSON 格式的报告字符串。"""
    report = {
        "title": title,
        "summary": summary,
        "findings": findings,
        "scores": scores or {},
    }
    if extra:
        report.update(extra)
    return json.dumps(report, ensure_ascii=False, indent=2)


def export_json_report(
    report_json: str,
    output_path: str,
) -> Path:
    """将 JSON 报告写入文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json, encoding="utf-8")
    return path
