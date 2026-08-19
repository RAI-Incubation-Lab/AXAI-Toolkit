# -*- coding: utf-8 -*-
"""报告生成模块：HTML 审计报告、JSON 导出。"""
from .badge import generate_badge_markdown, generate_badge_url  # noqa: F401
from .html_report import (  # noqa: F401
    export_html_report,
    generate_html_report,
)
from .json_report import (  # noqa: F401
    export_json_report,
    generate_json_report,
)

__all__ = [
    "generate_html_report",
    "export_html_report",
    "generate_json_report",
    "export_json_report",
    "generate_badge_markdown",
    "generate_badge_url",
]
