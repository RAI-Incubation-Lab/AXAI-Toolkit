# -*- coding: utf-8 -*-
"""反事实解释（Counterfactual Explanation）的简单贪心实现。"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _predict_class(model, x: np.ndarray) -> int:
    if hasattr(model, "predict"):
        return int(model.predict(x.reshape(1, -1))[0])
    if hasattr(model, "predict_proba"):
        return int(np.argmax(model.predict_proba(x.reshape(1, -1))[0]))
    raise TypeError("模型必须具有 predict 或 predict_proba 方法")


def greedy_counterfactual(
    model,
    instance: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    target_class: Optional[int] = None,
    max_changes: int = 5,
    steps_per_feature: int = 10,
    random_state: int = 42,
) -> dict:
    """贪心搜索一个尽可能接近原始样本的反事实。

    Parameters
    ----------
    model : sklearn 分类器
        已训练的分类模型。
    instance : np.ndarray
        待解释样本。
    X : np.ndarray
        训练数据特征，用于估计每个特征的变化范围。
    y : np.ndarray
        训练数据标签。
    target_class : int, optional
        希望反事实被预测为的类别；默认是当前预测类别的另一类。
    max_changes : int
        最多修改的特征数。
    steps_per_feature : int
        每个特征尝试的步数。
    random_state : int
        随机种子。

    Returns
    -------
    dict
        包含反事实样本、是否找到、修改的特征等。
    """
    X = np.asarray(X, dtype=float)
    instance = np.asarray(instance, dtype=float).reshape(-1)
    y = np.asarray(y)

    current_class = _predict_class(model, instance)
    if target_class is None:
        target_class = 1 - current_class
    if target_class == current_class:
        return {
            "counterfactual": instance,
            "found": True,
            "target_class": target_class,
            "original_class": current_class,
            "changed_features": [],
        }

    rng = np.random.default_rng(random_state)
    # 用各类别特征均值之差作为修改方向的参考
    class_means = np.array([X[y == c].mean(axis=0) for c in np.unique(y)])
    if target_class >= len(class_means):
        raise ValueError("target_class 超出数据集中的类别范围")
    target_mean = class_means[target_class]

    # 按当前样本与目标类别均值差异最大的特征依次尝试
    diff = np.abs(target_mean - instance)
    feature_order = np.argsort(diff)[::-1]

    candidate = instance.copy()
    changed = []
    for feature in feature_order:
        if len(changed) >= max_changes:
            break
        direction = np.sign(target_mean[feature] - instance[feature])
        if direction == 0:
            direction = 1
        feature_std = X[:, feature].std() + 1e-8
        best_trial = None
        for t in np.linspace(0, 2.0, steps_per_feature):
            trial = candidate.copy()
            trial[feature] = instance[feature] + direction * t * feature_std
            if _predict_class(model, trial) == target_class:
                best_trial = trial
                break
        if best_trial is not None:
            candidate = best_trial
            changed.append(int(feature))
            if _predict_class(model, candidate) == target_class:
                break

    found = _predict_class(model, candidate) == target_class
    return {
        "counterfactual": candidate,
        "found": found,
        "target_class": target_class,
        "original_class": current_class,
        "changed_features": changed,
        "original_instance": instance,
    }
