# -*- coding: utf-8 -*-
"""示例 13：SDK 装饰器无侵入监控（对应 Proposal 的 from axai import trace_agent）。

运行方式：
    python examples/13_sdk_decorator_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axai import AuditConfig, trace_agent


@trace_agent(config=AuditConfig(check_faithfulness=True, detect_pii=True, guard_sql=True))
def demo_agent(user_prompt: str):
    # 模拟一个极简 Agent
    if "drop table" in user_prompt.lower():
        return "I cannot execute that."
    return f"processed: {user_prompt}"


def main():
    result = demo_agent("Please call 13800138000 or test@example.com")
    print("函数返回值：", result)
    print("审计轨迹：", result.get("__axai_trace__", {}) if isinstance(result, dict) else "N/A")


if __name__ == "__main__":
    main()
