# -*- coding: utf-8 -*-
"""处方级修复：Prompt 补丁与 PII 脱敏中间件生成。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..llm.prompt_guard import diff_text, generate_guardrail_code, harden_prompt


def generate_prompt_patch(original_prompt: str) -> dict:
    """生成 Prompt 加固补丁。"""
    hardened = harden_prompt(original_prompt)
    return {
        "original": original_prompt,
        "hardened": hardened,
        "diff": diff_text(original_prompt, hardened),
        "summary": "已添加系统提示词边界、防注入指令与安全拒绝策略。",
    }


def apply_prompt_patch(
    prompt_file: str,
    output_file: Optional[str] = None,
) -> Path:
    """读取 Prompt 文件，生成加固版本并写回/另存。"""
    src = Path(prompt_file)
    original = src.read_text(encoding="utf-8")
    patch = generate_prompt_patch(original)
    if output_file is None:
        output_file = str(src.with_name(src.stem + "_hardened" + src.suffix))
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patch["hardened"], encoding="utf-8")
    return out


def generate_pii_middleware_code() -> str:
    """生成 PII 脱敏中间件代码。"""
    return generate_guardrail_code()
