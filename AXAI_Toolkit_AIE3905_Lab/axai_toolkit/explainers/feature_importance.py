# -*- coding: utf-8 -*-
"""模型无关的特征重要性方法：置换重要性、部分依赖图。"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, r2_score


def _score(model, X: np.ndarray, y: np.ndarray) -> float:
    """根据模型类型自动选择评分函数。"""
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(X)[:, 1] if model.classes_.size == 2 else model.predict(X)
        if model.classes_.size == 2:
            return accuracy_score(y, (pred >= 0.5).astype(int))
        return accuracy_score(y, pred)
    return r2_score(y, model.predict(X))


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list[str]] = None,
    n_repeats: int = 5,
    random_state: int = 42,
    score_func: Optional[Callable] = None,
) -> dict:
    """计算置换特征重要性。

    原理：随机打乱某一列特征会破坏该特征与标签的关系；
    若模型性能下降越多，说明该特征越重要。
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if score_func is None:
        score_func = _score

    baseline = score_func(model, X, y)
    importances = np.zeros((n_repeats, n_features))

    X_permuted = X.copy()
    for repeat in range(n_repeats):
        for col in range(n_features):
            X_permuted[:, col] = rng.permutation(X[:, col])
            importances[repeat, col] = baseline - score_func(model, X_permuted, y)
            X_permuted[:, col] = X[:, col]  # 还原

    return {
        "feature_names": feature_names,
        "importances_mean": importances.mean(axis=0),
        "importances_std": importances.std(axis=0),
        "baseline_score": baseline,
    }


def partial_dependence(
    model,
    X: np.ndarray,
    feature_index: int,
    grid_points: int = 20,
) -> dict:
    """计算单个特征的部分依赖（Partial Dependence）。

    通过将数据集中该特征替换为网格值，观察平均预测如何变化。
    """
    X = np.asarray(X, dtype=float)
    feature_index = int(feature_index)
    if feature_index < 0 or feature_index >= X.shape[1]:
        raise ValueError("feature_index 超出范围")

    values = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), grid_points)
    avg_preds = []

    if hasattr(model, "predict_proba"):
        # 对二分类取正类概率；多分类取所有类别平均概率幅值
        if hasattr(model, "classes_") and model.classes_.size == 2:
            for v in values:
                X_rep = X.copy()
                X_rep[:, feature_index] = v
                avg_preds.append(model.predict_proba(X_rep)[:, 1].mean())
        else:
            for v in values:
                X_rep = X.copy()
                X_rep[:, feature_index] = v
                avg_preds.append(model.predict_proba(X_rep).mean())
    else:
        for v in values:
            X_rep = X.copy()
            X_rep[:, feature_index] = v
            avg_preds.append(model.predict(X_rep).mean())

    return {
        "feature_index": feature_index,
        "values": values,
        "average_prediction": np.asarray(avg_preds),
    }
