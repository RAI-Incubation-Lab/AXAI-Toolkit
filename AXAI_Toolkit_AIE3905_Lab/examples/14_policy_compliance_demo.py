# -*- coding: utf-8 -*-
"""示例 14：Policy-to-Code 合规评分。

运行方式：
    python examples/14_policy_compliance_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axai_toolkit.metrics.rai import POLICY_SUITES, run_policy_suite


def main():
    print("可用政策套件：", list(POLICY_SUITES))
    result = run_policy_suite(
        "nist_ai_rmf",
        {
            "Govern": True,
            "Map": True,
            "Measure": False,
            "Manage": False,
        },
    )
    print("合规得分：", result["compliance_score"])
    print("未通过项：", result["failed"])
    print("检查明细：")
    for item in result["items"]:
        print(f"  - {item['name']}: {item['description']} -> {'通过' if item['passed'] else '未通过'}")


if __name__ == "__main__":
    main()
