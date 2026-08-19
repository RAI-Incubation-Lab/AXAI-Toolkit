# -*- coding: utf-8 -*-
"""通用小工具函数。"""
from __future__ import annotations

from typing import Optional

import numpy as np


def ensure_2d(X) -> np.ndarray:
    """确保输入为二维数组；若为一维则转为单行二维数组。"""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("输入必须是 1D 或 2D 数组")
    return arr


def get_feature_names(
    X,
    feature_names: Optional[list[str]] = None,
) -> list[str]:
    """返回特征名称列表。"""
    X = np.asarray(X)
    n_features = X.shape[1] if X.ndim == 2 else X.shape[0]
    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names 长度与特征数量不一致")
    return list(feature_names)


def safe_import_torch():
    """尝试导入 PyTorch，未安装时给出教学提示。"""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
