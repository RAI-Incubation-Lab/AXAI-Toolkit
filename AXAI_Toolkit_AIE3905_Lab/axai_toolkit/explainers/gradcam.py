# -*- coding: utf-8 -*-
"""Grad-CAM（Gradient-weighted Class Activation Mapping）实现。

Grad-CAM 使用目标卷积层的激活图与梯度加权，生成类别相关的热力图。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class GradCAMExplainer:
    """Grad-CAM 解释器。

    Parameters
    ----------
    model : torch.nn.Module 或 SimpleCNN
        已训练模型。
    target_layer : torch.nn.Module
        希望可视化的卷积层，通常是最后一个卷积层。
    """

    def __init__(self, model, target_layer):
        import torch  # noqa: F401

        if hasattr(model, "net"):
            model = model.net
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        self._hooks.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """生成 Grad-CAM 热力图。

        Returns
        -------
        np.ndarray
            与输入图像同尺寸的归一化热力图，数值范围 [0, 1]。
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("未捕获到目标层的梯度/激活，请检查 target_layer 是否为卷积层")

        # 权重 = 梯度在空间维度上的平均
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # 加权求和 -> ReLU -> 归一化
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # 上采样到输入尺寸
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


def gradcam(
    model,
    input_tensor,
    target_layer,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """便捷函数：计算 Grad-CAM 热力图。"""
    explainer = GradCAMExplainer(model, target_layer)
    try:
        return explainer.generate(input_tensor, target_class=target_class)
    finally:
        explainer.remove_hooks()
