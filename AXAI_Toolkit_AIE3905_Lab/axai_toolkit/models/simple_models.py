# -*- coding: utf-8 -*-
"""用于 XAI 课堂演示的简单模型封装。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegression:
    """训练逻辑回归分类器。"""
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X, y)
    return model


def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: Optional[int] = 4,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """训练浅层决策树，便于可视化树结构。"""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    return model


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """训练随机森林分类器。"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


class SimpleCNN:
    """一个极简 CNN，用于教学演示 Saliency Map 与 Grad-CAM。

    输入为单通道灰度图像，输出为二分类 logits。
    """

    def __init__(self, image_size: int = 16, num_classes: int = 2):
        import torch
        from torch import nn

        self.image_size = image_size
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * (image_size // 4) * (image_size // 4), num_classes),
        )

    def __call__(self, x):
        return self.net(x)

    def forward(self, x):
        return self.net(x)

    def train(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 0.01,
        device: Optional[str] = None,
    ) -> dict:
        """在玩具图像上训练模型，并返回每轮平均损失。"""
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        X = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            avg_loss = total_loss / len(dataset)
            history.append(avg_loss)

        # 为了便于后续解释演示，统一将模型移回 CPU
        self.net = model.cpu()
        return {"loss": history}


def train_small_cnn(
    images: np.ndarray,
    labels: np.ndarray,
    image_size: Optional[int] = None,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.01,
    random_state: int = 42,
    device: Optional[str] = None,
) -> SimpleCNN:
    """训练一个小型 CNN，返回 SimpleCNN 包装对象。

    该函数需要 PyTorch。若未安装 PyTorch，会给出明确提示。
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "训练 CNN 需要安装 PyTorch，请运行：pip install torch torchvision"
        ) from exc

    torch.manual_seed(random_state)
    if image_size is None:
        image_size = images.shape[1] if images.ndim == 3 else images.shape[2]

    cnn = SimpleCNN(image_size=image_size, num_classes=len(np.unique(labels)))
    cnn.train(
        images,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    return cnn
