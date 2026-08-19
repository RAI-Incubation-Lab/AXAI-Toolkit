# -*- coding: utf-8 -*-
"""示例 15：AST Linter 扫描 Prompt 拼接漏洞。

运行方式：
    python examples/15_ast_linter_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axai_toolkit.rai.ast_linter import lint_python_source

RISKY_CODE = '''
def build_prompt(user_input: str):
    prompt = f"System: You are a helper. User said: {user_input}"
    return prompt
'''


def main():
    findings = lint_python_source(RISKY_CODE, filename="risky_agent.py")
    print("AST 风险发现：")
    for item in findings:
        print(f"  - {item['type']} line {item.get('line', '?')}: {item['detail']}")
    if not findings:
        print("  未发现风险")


if __name__ == "__main__":
    main()
