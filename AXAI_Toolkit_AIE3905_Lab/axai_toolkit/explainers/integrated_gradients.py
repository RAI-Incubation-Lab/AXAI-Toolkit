# -*- coding: utf-8 -*-
"""Integrated Gradients（积分梯度）解释方法。"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _to_module(model):
    if hasattr(model, "net"):
        return model.net
    return model


def integrated_gradients(
    model,
    input_tensor,
    target_class: Optional[int] = None,
    baseline=None,
    steps: int = 50,
) -> np.ndarray:
    """计算输入图像的积分梯度归因。

    Parameters
    ----------
    model : torch.nn.Module 或 SimpleCNN
        已训练模型。
    input_tensor : torch.Tensor
        形状为 (C,H,W) 或 (1,C,H,W)。
    target_class : int, optional
        要解释的目标类别。
    baseline : torch.Tensor, optional
        基线图像，默认全 0。
    steps : int
        积分路径上的采样步数。

    Returns
    -------
    np.ndarray
        与输入图像同形状（去掉 batch 维）的归因图。
    """
    import torch

    module = _to_module(model)
    module.eval()

    add_batch = False
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
        add_batch = True

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    elif baseline.dim() == 3:
        baseline = baseline.unsqueeze(0)

    if target_class is None:
        with torch.no_grad():
            target_class = int(module(input_tensor).argmax(dim=1).item())

    grad_accumulator = torch.zeros_like(input_tensor, dtype=torch.float32)
    for alpha in np.linspace(0.0, 1.0, steps):
        scaled = baseline + alpha * (input_tensor - baseline)
        scaled = scaled.clone().requires_grad_(True)
        output = module(scaled)
        score = output[0, target_class]
        module.zero_grad()
        score.backward()
        grad_accumulator += scaled.grad

    avg_grad = grad_accumulator / steps
    attribution = (input_tensor - baseline) * avg_grad
    attribution = attribution[0].detach().cpu().numpy()

    if add_batch and attribution.shape[0] == 1:
        attribution = attribution[0]
    return attribution
