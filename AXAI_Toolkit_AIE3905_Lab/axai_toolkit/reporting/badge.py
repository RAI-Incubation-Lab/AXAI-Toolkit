# -*- coding: utf-8 -*-
"""GitHub 徽章生成（静态 Shields.io 风格）。"""
from __future__ import annotations


def generate_badge_markdown(
    score: float,
    label: str = "AXAI",
    color: str = "blue",
) -> str:
    """生成一个简单的 Shields.io Markdown 徽章。"""
    text = f"{score:.0f}/100"
    return f"![{label}](https://img.shields.io/badge/{label}-{text}-{color})"


def generate_badge_url(
    score: float,
    label: str = "AXAI",
    color: str = "blue",
) -> str:
    """生成徽章图片 URL。"""
    text = f"{score:.0f}%2F100"
    return f"https://img.shields.io/badge/{label}-{text}-{color}"
