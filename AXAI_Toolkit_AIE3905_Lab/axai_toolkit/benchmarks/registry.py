# -*- coding: utf-8 -*-
"""全球权威 XAI/RAI 评测体系注册表。

用于教学时快速对照 AXAI-Toolkit 与外部 Benchmark 的关系。
"""
from __future__ import annotations

BENCHMARK_REGISTRY = {
    "quantus": {
        "name": "Quantus",
        "domain": "XAI (Classic/DL)",
        "focus": ["Faithfulness", "Robustness", "Complexity", "Localisation"],
        "scenario": "深度学习图像/传统模型特征归因评估",
    },
    "openxai": {
        "name": "OpenXAI",
        "domain": "XAI (Tabular/Post-hoc)",
        "focus": ["Explanation fidelity", "Stability", "Fairness"],
        "scenario": "金融风控、医疗表格模型解释验证",
    },
    "eraser": {
        "name": "ERASER",
        "domain": "XAI (NLP)",
        "focus": ["Comprehensiveness", "Sufficiency"],
        "scenario": "文本分类、人工标注 Rationale 对照",
    },
    "faithcot": {
        "name": "FaithCoT-Bench",
        "domain": "XAI (LLM CoT)",
        "focus": ["CoT faithfulness", "Unfaithfulness detection"],
        "scenario": "大模型思维链真伪性检验",
    },
    "decodingtrust": {
        "name": "DecodingTrust",
        "domain": "Responsible AI",
        "focus": ["Toxicity", "Bias", "Adversarial robustness", "Privacy", "Ethics"],
        "scenario": "大模型安全防护、深度红队漏洞挖掘",
    },
    "trustllm": {
        "name": "TrustLLM",
        "domain": "Responsible AI",
        "focus": ["Truthfulness", "Safety", "Fairness", "Robustness", "Privacy"],
        "scenario": "模型综合能力评测、合规准入评估",
    },
    "helm": {
        "name": "Stanford HELM",
        "domain": "Responsible AI / Holistic Evaluation",
        "focus": ["Accuracy", "Bias", "Toxicity", "Copyright", "Efficiency"],
        "scenario": "大模型全景评测与选型",
    },
    "harmbench": {
        "name": "HarmBench",
        "domain": "Safety",
        "focus": ["Jailbreak attacks", "Malicious content"],
        "scenario": "自动化红队越狱测试",
    },
    "ragas": {
        "name": "RAGAS",
        "domain": "Response Quality / RAG",
        "focus": ["Faithfulness", "Answer relevance", "Context recall"],
        "scenario": "RAG 知识库问答质量评测",
    },
    "truelens": {
        "name": "TruLens",
        "domain": "Response Quality / RAG",
        "focus": ["Ground truth", "Answer relevance", "Context relevance"],
        "scenario": "RAG 应用可观测性",
    },
    "nist_ai_rmf": {
        "name": "NIST AI RMF 1.0",
        "domain": "Governance & Compliance",
        "focus": ["Govern", "Map", "Measure", "Manage"],
        "scenario": "企业 AI 风险管理、合规审计",
    },
    "iso_42001": {
        "name": "ISO/IEC 42001",
        "domain": "Governance & Compliance",
        "focus": ["AI management system", "Lifecycle governance"],
        "scenario": "AI 管理体系认证",
    },
    "eu_ai_act": {
        "name": "EU AI Act",
        "domain": "Governance & Compliance",
        "focus": ["High-risk AI", "Transparency", "Conformity assessment"],
        "scenario": "高风险 AI 合规一致性",
    },
}


def list_benchmarks() -> list[str]:
    """返回所有支持的 Benchmark 名称。"""
    return sorted(BENCHMARK_REGISTRY.keys())


def get_benchmark(name: str) -> dict:
    """按 key 获取 Benchmark 信息。"""
    key = name.strip().lower()
    if key not in BENCHMARK_REGISTRY:
        raise KeyError(f"未知 Benchmark: {name}。可用: {list_benchmarks()}")
    return BENCHMARK_REGISTRY[key]
