# -*- coding: utf-8 -*-
"""RAG 证据引用真伪度（Grounding Ratio）与简单蕴含评估。"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np

# 支持中文连续片段、英文单词（长度 >= 2）与数字/下划线组合
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]{2,}")


def simple_entailment(claim: str, evidence: str) -> float:
    """基于词重叠的简化蕴含分。

    真实场景可替换为模型蕴含分类器或向量语义相似度。
    """
    claim_terms = set(TOKEN_PATTERN.findall(claim.lower()))
    evidence_terms = set(TOKEN_PATTERN.findall(evidence.lower()))
    if not claim_terms:
        return 0.0
    overlap = len(claim_terms & evidence_terms)
    return overlap / len(claim_terms)


def grounding_ratio(
    claims: list[str],
    evidence_chunks: list[str],
    entailment_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = 0.4,
) -> dict:
    """逐句校验回答中的命题是否被检索到的证据支撑。

    Returns
    -------
    dict
        包含每条 claim 的得分、平均 grounding ratio 与幻觉风险 claim。
    """
    if entailment_fn is None:
        entailment_fn = simple_entailment

    per_claim = []
    hallucinated = []
    for claim in claims:
        best_score = max(
            (entailment_fn(claim, chunk) for chunk in evidence_chunks),
            default=0.0,
        )
        per_claim.append({"claim": claim, "score": float(best_score)})
        if best_score < threshold:
            hallucinated.append({"claim": claim, "score": float(best_score)})

    if not claims:
        ratio = 1.0
    else:
        ratio = float(np.mean([item["score"] for item in per_claim]))

    return {
        "grounding_ratio": ratio,
        "per_claim": per_claim,
        "hallucinated_claims": hallucinated,
        "threshold": threshold,
    }
