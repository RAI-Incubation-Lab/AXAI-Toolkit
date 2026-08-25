# -*- coding: utf-8 -*-
"""反事实解释（Counterfactual Explanation）的简单贪心实现。"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _predict_class(model, x: np.ndarray) -> Any:
    if hasattr(model, "predict"):
        return model.predict(x.reshape(1, -1))[0]
    if hasattr(model, "predict_proba"):
        index = int(np.argmax(model.predict_proba(x.reshape(1, -1))[0]))
        return model.classes_[index] if hasattr(model, "classes_") else index
    raise TypeError("模型必须具有 predict 或 predict_proba 方法")


def _target_score(model, x: np.ndarray, target_class: Any) -> float:
    """Return the target-class probability, or a class-match score as fallback."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x.reshape(1, -1))[0]
        if hasattr(model, "classes_"):
            matches = np.where(np.asarray(model.classes_) == target_class)[0]
            if len(matches) != 1:
                raise ValueError("target_class 不在模型类别中")
            return float(probabilities[int(matches[0])])
        if isinstance(target_class, (int, np.integer)) and 0 <= target_class < len(probabilities):
            return float(probabilities[int(target_class)])
    return float(_predict_class(model, x) == target_class)


def greedy_counterfactual(
    model,
    instance: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    target_class: Optional[Any] = None,
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
    class_labels = np.unique(y)
    if target_class is None:
        alternatives = [label for label in class_labels if label != current_class]
        if not alternatives:
            raise ValueError("反事实解释需要至少两个类别")
        target_class = alternatives[0]
    if target_class == current_class:
        return {
            "counterfactual": instance,
            "found": True,
            "target_class": target_class,
            "original_class": current_class,
            "changed_features": [],
        }

    if target_class not in class_labels:
        raise ValueError("target_class 超出数据集中的类别范围")
    target_mean = X[y == target_class].mean(axis=0)

    candidate = instance.copy()
    if max_changes < 1 or steps_per_feature < 2:
        raise ValueError("max_changes 必须至少为 1，steps_per_feature 必须至少为 2")
    changed: list[int] = []
    available = set(range(X.shape[1]))
    # At every round retain the best *partial* change.  The previous
    # implementation committed a change only after one feature alone flipped
    # the class, so it could never find a valid multi-feature counterfactual.
    for _ in range(max_changes):
        if _predict_class(model, candidate) == target_class or not available:
            break
        best_trial = None
        best_feature = None
        best_score = _target_score(model, candidate, target_class)
        for feature in available:
            for fraction in np.linspace(0.0, 1.5, steps_per_feature)[1:]:
                trial = candidate.copy()
                trial[feature] = candidate[feature] + fraction * (target_mean[feature] - candidate[feature])
                score = _target_score(model, trial, target_class)
                if score > best_score + 1e-12:
                    best_trial, best_feature, best_score = trial, feature, score
        if best_trial is None:
            break
        candidate = best_trial
        changed.append(int(best_feature))
        available.remove(best_feature)

    found = _predict_class(model, candidate) == target_class
    return {
        "counterfactual": candidate,
        "found": found,
        "target_class": target_class,
        "original_class": current_class,
        "changed_features": changed,
        "original_instance": instance,
    }
