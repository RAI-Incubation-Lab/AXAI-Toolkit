# -*- coding: utf-8 -*-
"""Saliency Map（显著图）：通过输入梯度解释深度学习模型的预测。"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _to_module(model):
    """兼容 SimpleCNN 包装器和原生 torch.nn.Module。"""
    if hasattr(model, "net"):
        return model.net
    return model


def saliency_map(
    model,
    input_tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """计算输入图像的 Saliency Map。

    Parameters
    ----------
    model : torch.nn.Module 或 SimpleCNN
        已训练的图像分类模型。
    input_tensor : torch.Tensor
        形状为 (C,H,W) 或 (1,C,H,W) 的输入图像张量。
    target_class : int, optional
        要解释的目标类别；默认取模型预测的类别。

    Returns
    -------
    np.ndarray
        形状为 (H,W) 的显著图，数值已归一化到 [0, 1]。
    """
    import torch

    module = _to_module(model)
    module.eval()

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.clone().requires_grad_(True)

    output = module(input_tensor)
    if target_class is None:
        target_class = int(output.argmax(dim=1).item())
    score = output[0, target_class]

    module.zero_grad()
    score.backward()
    grad = input_tensor.grad[0].abs()

    if grad.shape[0] == 1:
        saliency = grad[0]
    else:
        saliency = grad.max(dim=0).values

    saliency = saliency.detach().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency
