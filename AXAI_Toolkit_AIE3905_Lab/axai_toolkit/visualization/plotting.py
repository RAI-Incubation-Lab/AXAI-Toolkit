# -*- coding: utf-8 -*-
"""可视化辅助函数。"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(
    importance: dict,
    ax=None,
    title: str = "Feature Importance",
):
    """绘制特征重要性条形图。

    Parameters
    ----------
    importance : dict
        包含 feature_names 与 importances_mean（或 values）的字典。
    """
    feature_names = importance.get("feature_names")
    if "importances_mean" in importance:
        values = np.asarray(importance["importances_mean"])
    else:
        values = np.asarray(importance["values"])
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(values))]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, len(values) * 0.5)))
    order = np.argsort(values)
    ax.barh(np.array(feature_names)[order], values[order])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_partial_dependence(
    pd_result: dict,
    feature_name: Optional[str] = None,
    ax=None,
):
    """绘制单特征部分依赖曲线。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pd_result["values"], pd_result["average_prediction"], marker="o")
    ax.set_title("Partial Dependence")
    ax.set_xlabel(feature_name or f"feature_{pd_result['feature_index']}")
    ax.set_ylabel("Average prediction")
    ax.grid(alpha=0.3)
    return ax


def plot_lime_weights(
    explanation: dict,
    num_features: int = 5,
    ax=None,
):
    """绘制 LIME 局部权重条形图。"""
    feature_names = explanation["feature_names"]
    coefs = np.asarray(explanation["coefficients"])
    order = np.argsort(np.abs(coefs))[::-1][:num_features]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, len(order) * 0.5)))

    labels = [feature_names[i] for i in order]
    weights = [coefs[i] for i in order]
    colors = ["#d62728" if w > 0 else "#1f77b4" for w in weights]
    ax.barh(labels, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("LIME Local Weights")
    ax.set_xlabel("Weight")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_saliency_map(
    image: np.ndarray,
    saliency: np.ndarray,
    ax=None,
):
    """并排显示原图与 Saliency Map。"""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(saliency, cmap="hot")
    ax[1].set_title("Saliency Map")
    ax[1].axis("off")
    return ax


def plot_gradcam_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    ax=None,
    alpha: float = 0.5,
):
    """在原图上叠加 Grad-CAM 热力图。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.set_title("Grad-CAM")
    ax.axis("off")
    return ax


def plot_counterfactual_comparison(
    original: np.ndarray,
    counterfactual: np.ndarray,
    feature_names: Optional[list[str]] = None,
    ax=None,
):
    """比较原始样本与反事实样本的特征取值。"""
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(original))]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(original))
    width = 0.35
    ax.bar(x - width / 2, original, width, label="Original", color="#1f77b4")
    ax.bar(x + width / 2, counterfactual, width, label="Counterfactual", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Feature value")
    ax.legend()
    ax.set_title("Original vs Counterfactual")
    ax.grid(axis="y", alpha=0.3)
    return ax
