# -*- coding: utf-8 -*-
"""静态 AST Linter：扫描 Python 代码中的 Prompt 拼接 / 注入风险。

对应 Proposal 中的“静态 AST Linter (扫描 Prompt 拼接漏洞)”。
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Union

PROMPT_KEYWORDS = ("prompt", "message", "system", "instruction", "query", "user_input")

# 常见危险外部输入来源
RISKY_SOURCES = ("input(", "request.", "params", "body", "os.environ", "getenv")


def _is_prompt_like(name: str) -> bool:
    lowered = name.lower()
    return any(kw in lowered for kw in PROMPT_KEYWORDS)


def lint_python_source(source: str, filename: str = "<string>") -> list[dict]:
    """扫描一段 Python 源码，返回风险发现列表。"""
    findings = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return [
            {
                "type": "ast-syntax-error",
                "file": filename,
                "detail": "无法解析 Python 源码，可能是非 Python 文件或语法错误。",
            }
        ]

    for node in ast.walk(tree):
        # 检测 f-string 中包含外部插值的 Prompt
        if isinstance(node, ast.JoinedStr):
            text_parts = [
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            ]
            has_formatted = any(isinstance(part, ast.FormattedValue) for part in node.values)
            if has_formatted and any(_is_prompt_like(part) for part in text_parts):
                findings.append(
                    {
                        "type": "prompt-fstring",
                        "file": filename,
                        "line": getattr(node, "lineno", None),
                        "detail": "Prompt 使用 f-string 直接拼接变量，存在提示词注入风险。",
                    }
                )

        # 检测字符串与外部输入/敏感来源的加号拼接
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = node.left
            right = node.right
            source_names = set()
            if isinstance(left, ast.Name):
                source_names.add(left.id)
            if isinstance(right, ast.Name):
                source_names.add(right.id)
            if any(_is_prompt_like(name) for name in source_names):
                risky = any(
                    (isinstance(n, ast.Name) and any(src in n.id.lower() for src in ("input", "request", "params", "body")))
                    or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "input")
                    for n in (left, right)
                )
                if risky:
                    findings.append(
                        {
                            "type": "prompt-concatenation",
                            "file": filename,
                            "line": getattr(node, "lineno", None),
                            "detail": "Prompt 使用字符串拼接外部输入，建议改用安全模板或增加输入校验。",
                        }
                    )

        # 检测直接调用 input() 并用于 prompt 变量
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_prompt_like(target.id):
                    value = node.value
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "input":
                        findings.append(
                            {
                                "type": "prompt-from-input",
                                "file": filename,
                                "line": getattr(node, "lineno", None),
                                "detail": f"Prompt 直接来自 input()，建议增加过滤或审计。",
                            }
                        )

    return findings


def lint_python_file(path: Union[str, Path]) -> list[dict]:
    """扫描一个 Python 文件。"""
    path = Path(path)
    source = path.read_text(encoding="utf-8", errors="ignore")
    return lint_python_source(source, filename=str(path))
