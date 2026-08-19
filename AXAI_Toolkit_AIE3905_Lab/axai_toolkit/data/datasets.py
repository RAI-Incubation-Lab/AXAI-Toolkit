# -*- coding: utf-8 -*-
"""教学用数据集加载与合成函数。

所有数据集均为合成数据或本地生成数据，不依赖网络下载。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# 信用风险教学示例使用的特征名（便于课堂讲解业务含义）
CREDIT_FEATURE_NAMES = [
    "income",
    "credit_history",
    "debt_ratio",
    "payment_status",
    "account_age",
    "utilization",
]


def make_classification_data(
    n_samples: int = 300,
    n_features: int = 6,
    n_informative: int = 4,
    n_redundant: int = 1,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """生成一个适合课堂展示的合成二分类数据集。

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵。
    y : np.ndarray, shape (n_samples,)
        二分类标签（0/1）。
    feature_names : list[str]
        特征名称列表。
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=0.02,
        random_state=random_state,
    )
    # 用更贴近课堂故事的名称替换默认名称
    if n_features == len(CREDIT_FEATURE_NAMES):
        feature_names = CREDIT_FEATURE_NAMES
    else:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def load_demo_classification(
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """加载课堂演示用的二分类数据，并划分训练集/测试集。

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    X, y, feature_names = make_classification_data(random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names


def make_regression_data(
    n_samples: int = 300,
    n_features: int = 5,
    n_informative: int = 3,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """生成一个适合课堂展示的合成回归数据集。"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def load_demo_regression(
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """加载课堂演示用的回归数据集，并划分训练集/测试集。"""
    X, y, feature_names = make_regression_data(random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


def load_toy_images(
    n_samples: int = 120,
    image_size: int = 16,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """生成两个类别的玩具灰度图像。

    类别 0：垂直白色条带；类别 1：水平白色条带。
    图像尺寸为 (image_size, image_size)，像素值范围 [0, 1]。

    Returns
    -------
    images : np.ndarray, shape (n_samples, image_size, image_size)
        灰度图像，可直接用于展示或转换为 PyTorch Tensor。
    labels : np.ndarray, shape (n_samples,)
        整数标签 0/1。
    class_names : list[str]
        类别名称。
    """
    rng = np.random.default_rng(random_state)
    half = n_samples // 2
    images = np.zeros((n_samples, image_size, image_size), dtype=np.float32)

    # 类别 0：垂直条带
    col_start = image_size // 2 - 2
    col_end = image_size // 2 + 2
    for i in range(half):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        img[:, col_start:col_end] = 1.0
        img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
        images[i] = np.clip(img, 0.0, 1.0)

    # 类别 1：水平条带
    row_start = image_size // 2 - 2
    row_end = image_size // 2 + 2
    for i in range(half, n_samples):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        img[row_start:row_end, :] = 1.0
        img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
        images[i] = np.clip(img, 0.0, 1.0)

    labels = np.array([0] * half + [1] * (n_samples - half), dtype=np.int64)
    class_names = ["vertical_bar", "horizontal_bar"]
    return images, labels, class_names
