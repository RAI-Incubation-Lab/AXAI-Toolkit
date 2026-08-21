# -*- coding: utf-8 -*-
"""一个轻量级 LIME（Local Interpretable Model-agnostic Explanations）实现。

该实现面向教学，省略了部分工程细节，但保留了 LIME 的核心思想：
在待解释样本附近采样扰动，训练一个可解释的局部代理模型。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge


class LimeTabularExplainer:
    """表格数据的 LIME 解释器。

    Parameters
    ----------
    training_data : np.ndarray
        用于估计特征分布的训练数据（形状 n_samples x n_features）。
    feature_names : list[str], optional
        特征名称。
    mode : str
        "classification" 或 "regression"。
    random_state : int
        随机种子。
    kernel_width : float
        距离核函数的宽度，越大则局部邻域越宽。
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[list[str]] = None,
        mode: str = "classification",
        random_state: int = 42,
        kernel_width: float = 0.75,
    ):
        self.training_data = np.asarray(training_data, dtype=float)
        self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.training_data.shape[1])]
        self.mode = mode
        self.random_state = random_state
        self.kernel_width = kernel_width
        self.means = self.training_data.mean(axis=0)
        self.stds = self.training_data.std(axis=0) + 1e-8

    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        """返回二分类正类概率；多分类返回预测类别的概率；对回归返回预测值。"""
        X = np.asarray(X, dtype=float)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2:
                if proba.shape[1] == 2:
                    return proba[:, 1]
                return proba.max(axis=1)
            return proba
        return model.predict(X)

    def explain_instance(
        self,
        model,
        instance: np.ndarray,
        num_samples: int = 500,
        top_labels: Optional[int] = None,
    ) -> dict:
        """解释单个样本的预测。

        Returns
        -------
        dict
            包含局部代理模型系数、截距、扰动样本、样本权重、预测概率等。
        """
        instance = np.asarray(instance, dtype=float).reshape(1, -1)
        if instance.shape[1] != self.training_data.shape[1]:
            raise ValueError("instance 的特征数必须与 training_data 一致")

        rng = np.random.default_rng(self.random_state)
        n_features = self.training_data.shape[1]

        # 1. 在实例附近生成扰动样本
        perturbations = np.zeros((num_samples, n_features))
        for i in range(num_samples):
            perturbations[i] = instance[0] + rng.normal(
                0.0, self.stds, size=n_features
            )

        # 2. 计算样本权重：距离越近权重越大
        normalized_perturbations = (perturbations - self.means) / self.stds
        normalized_instance = (instance - self.means) / self.stds
        distances = np.linalg.norm(
            normalized_perturbations - normalized_instance, axis=1
        )
        weights = np.exp(-(distances**2) / self.kernel_width**2)

        # 3. 获取模型在扰动样本上的预测
        if self.mode == "classification":
            predictions = self._predict_proba(model, perturbations)
        else:
            predictions = model.predict(perturbations)

        # 4. 在标准化特征空间上训练局部加权线性模型
        local_model = Ridge(alpha=1.0)
        local_model.fit(normalized_perturbations, predictions, sample_weight=weights)

        # 5. 整理解释结果
        if self.mode == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(instance)[0]
            predicted_class = int(np.argmax(proba))
            predicted_proba = float(proba[predicted_class])
        else:
            predicted_class = None
            predicted_proba = float(model.predict(instance)[0])

        explanation = {
            "feature_names": self.feature_names,
            "coefficients": local_model.coef_,
            "intercept": local_model.intercept_,
            "perturbations": perturbations,
            "weights": weights,
            "local_predictions": predictions,
            "predicted_class": predicted_class,
            "predicted_proba": predicted_proba,
            "mode": self.mode,
        }
        return explanation

    def as_list(self, explanation: dict, num_features: int = 5) -> list[tuple[str, float]]:
        """将解释结果转换为 (特征名, 权重) 列表，按权重绝对值降序排列。"""
        coefs = np.asarray(explanation["coefficients"])
        order = np.argsort(np.abs(coefs))[::-1][:num_features]
        return [(self.feature_names[i], float(coefs[i])) for i in order]
