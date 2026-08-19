# -*- coding: utf-8 -*-
"""解释质量评估指标。

这些指标适合课堂教学，帮助学生从不同角度比较解释方法：
- Faithfulness：解释是否忠实地反映模型行为。
- Stability：解释是否对小扰动稳定。
- Complexity：解释是否足够简洁。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _extract_values(importance) -> np.ndarray:
    """从常见解释结果 dict 中提取归因值数组。"""
    if isinstance(importance, dict):
        if "values" in importance:
            return np.asarray(importance["values"], dtype=float)
        if "coefficients" in importance:
            return np.asarray(importance["coefficients"], dtype=float)
        if "importances_mean" in importance:
            return np.asarray(importance["importances_mean"], dtype=float)
    raise ValueError("无法从 importance 中提取数值，请传入包含 values/coefficients/importances_mean 的 dict")


def _predict_score(model, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 2:
            return float(proba[0, 1])
        return float(proba.max())
    return float(model.predict(x)[0])


def faithfulness(
    model,
    instance: np.ndarray,
    importance,
    X_mean: Optional[np.ndarray] = None,
    top_k: int = 3,
) -> float:
    """Faithfulness：移除 top-k 重要特征后，预测分数的下降幅度。

    下降越大，说明该解释确实抓住了模型依赖的重要特征。
    """
    instance = np.asarray(instance, dtype=float).reshape(-1)
    values = _extract_values(importance)
    if len(values) != len(instance):
        raise ValueError("importance 的长度必须与 instance 一致")

    if X_mean is None:
        X_mean = np.zeros_like(instance)
    X_mean = np.asarray(X_mean, dtype=float).reshape(-1)

    top_indices = np.argsort(np.abs(values))[::-1][:top_k]
    modified = instance.copy()
    modified[top_indices] = X_mean[top_indices]

    original_score = _predict_score(model, instance)
    modified_score = _predict_score(model, modified)
    return original_score - modified_score


def stability(
    model,
    instance: np.ndarray,
    importance_func: Callable[[np.ndarray], dict],
    n_samples: int = 20,
    perturbation_std: float = 0.02,
    random_state: int = 42,
) -> float:
    """Stability：在原样本附近加入小扰动，比较解释结果的相似度。

    返回平均余弦相似度，越接近 1 表示越稳定。
    """
    rng = np.random.default_rng(random_state)
    instance = np.asarray(instance, dtype=float).reshape(-1)
    original_values = _extract_values(importance_func(instance))

    similarities = []
    for _ in range(n_samples):
        perturbed = instance + rng.normal(
            0.0, perturbation_std, size=instance.shape
        )
        perturbed_values = _extract_values(importance_func(perturbed))
        denom = np.linalg.norm(original_values) * np.linalg.norm(perturbed_values)
        if denom < 1e-12:
            similarities.append(1.0)
        else:
            similarities.append(
                float(np.dot(original_values, perturbed_values) / denom)
            )
    return float(np.mean(similarities))


def complexity(
    importance,
    threshold: float = 1e-3,
) -> dict:
    """Complexity：衡量解释的简洁程度。

    返回使用的特征数量（大于阈值的归因值个数）以及归一化熵。
    特征数量越少、熵越低，通常表示解释越简洁。
    """
    values = _extract_values(importance)
    active = int(np.sum(np.abs(values) > threshold))
    probs = np.abs(values) / (np.sum(np.abs(values)) + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    n = len(values)
    normalized_entropy = entropy / np.log(n) if n > 1 else 0.0
    return {
        "active_features": active,
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
    }
