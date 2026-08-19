# -*- coding: utf-8 -*-
"""教学用 SHAP（SHapley Additive exPlanations）实现。

这里不依赖外部 shap 库，而是用均值填充的背景分布 + Shapley 值公式，
帮助学生理解 SHAP 背后的合作博弈思想。
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np


def _predict_score(
    model,
    x: np.ndarray,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """得到模型在 x 上的预测分数。"""
    x = np.asarray(x, dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if class_index is None:
            if proba.shape[1] == 2:
                class_index = 1
            else:
                class_index = 0
        return proba[:, class_index]
    return model.predict(x)


def exact_shapley(
    model,
    X: np.ndarray,
    instance: np.ndarray,
    feature_names: Optional[list[str]] = None,
    class_index: Optional[int] = None,
) -> dict:
    """精确计算 Shapley 值（适用于特征数较少的数据集）。

    Parameters
    ----------
    model : sklearn/任意模型
        需要具有 predict 或 predict_proba 方法。
    X : np.ndarray
        背景数据集，用于估计缺失特征的填充值。
    instance : np.ndarray
        待解释样本。
    feature_names : list[str], optional
        特征名称。
    class_index : int, optional
        分类问题中要解释的类别索引；默认二分类取正类。
    """
    X = np.asarray(X, dtype=float)
    instance = np.asarray(instance, dtype=float).reshape(-1)
    n_features = X.shape[1]
    if instance.shape[0] != n_features:
        raise ValueError("instance 与 X 的特征数不一致")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    baseline = X.mean(axis=0)

    def value_function(mask: tuple[int, ...]) -> float:
        x = baseline.copy()
        for i in mask:
            x[i] = instance[i]
        return float(_predict_score(model, x.reshape(1, -1), class_index)[0])

    # 缓存所有子集的取值，避免重复计算
    cache: dict[tuple[int, ...], float] = {}

    def cached_value(mask: tuple[int, ...]) -> float:
        if mask not in cache:
            cache[mask] = value_function(mask)
        return cache[mask]

    shapley_values = np.zeros(n_features)
    all_features = set(range(n_features))

    for i in range(n_features):
        others = list(all_features - {i})
        total = 0.0
        for r in range(n_features):
            for subset_tuple in itertools.combinations(others, r):
                subset = tuple(sorted(subset_tuple))
                subset_with_i = tuple(sorted(subset + (i,)))
                v_with = cached_value(subset_with_i)
                v_without = cached_value(subset)
                weight = (
                    math.factorial(r)
                    * math.factorial(n_features - r - 1)
                    / math.factorial(n_features)
                )
                total += weight * (v_with - v_without)
        shapley_values[i] = total

    return {
        "feature_names": feature_names,
        "values": shapley_values,
        "base_value": float(_predict_score(model, baseline.reshape(1, -1), class_index)[0]),
        "instance": instance,
    }


def shapley_sampling(
    model,
    X: np.ndarray,
    instance: np.ndarray,
    feature_names: Optional[list[str]] = None,
    class_index: Optional[int] = None,
    n_samples: int = 200,
    random_state: int = 42,
) -> dict:
    """用排列采样近似计算 Shapley 值，适合特征较多的场景。"""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    instance = np.asarray(instance, dtype=float).reshape(-1)
    n_features = X.shape[1]
    if instance.shape[0] != n_features:
        raise ValueError("instance 与 X 的特征数不一致")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    baseline = X.mean(axis=0)
    contributions = np.zeros(n_features)

    for _ in range(n_samples):
        perm = rng.permutation(n_features)
        current = baseline.copy()
        prev_score = float(_predict_score(model, current.reshape(1, -1), class_index)[0])
        for idx in perm:
            current = current.copy()
            current[idx] = instance[idx]
            new_score = float(_predict_score(model, current.reshape(1, -1), class_index)[0])
            contributions[idx] += new_score - prev_score
            prev_score = new_score

    shapley_values = contributions / n_samples

    return {
        "feature_names": feature_names,
        "values": shapley_values,
        "base_value": float(_predict_score(model, baseline.reshape(1, -1), class_index)[0]),
        "instance": instance,
    }
