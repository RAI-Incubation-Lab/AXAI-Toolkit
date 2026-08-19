# -*- coding: utf-8 -*-
"""示例 11：LLM 思维链忠实度与 RAG Grounding Ratio。

运行方式：
    python examples/11_llm_faithfulness_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axai_toolkit.llm.faithfulness import evaluate_cot_faithfulness
from axai_toolkit.llm.grounding import grounding_ratio


def main():
    # 1. RAG 幻觉检测
    claims = [
        "公司的 2024 年收入增长了 20%。",
        "CEO 在发布会上宣布了新产品。",
    ]
    evidence = [
        "2024 年公司收入同比增长 20%。",
        "CEO 在年度发布会上介绍了新产品线。",
    ]
    rag_result = grounding_ratio(claims, evidence, threshold=0.3)
    print("Grounding Ratio:", rag_result["grounding_ratio"])
    print("幻觉风险:", rag_result["hallucinated_claims"])

    # 2. CoT 忠实度
    steps = [
        "公司收入从 100 万增长到 120 万。",
        "增长额为 20 万。",
        "因此增长率为 20%。",
    ]
    answer = "公司收入增长率为 20%。"
    cot_result = evaluate_cot_faithfulness(steps, answer, evidence=evidence)
    print("CoT 忠实度分数:", cot_result["faithfulness_score"])
    print("覆盖度:", cot_result["coverage"], "一致性:", cot_result["consistency"])


if __name__ == "__main__":
    main()
