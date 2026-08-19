# -*- coding: utf-8 -*-
"""示例 6：Saliency Map 图像解释。

需要安装 PyTorch。运行方式：
    python examples/06_saliency_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from axai_toolkit.data.datasets import load_toy_images
from axai_toolkit.explainers.saliency import saliency_map
from axai_toolkit.models.simple_models import train_small_cnn
from axai_toolkit.visualization.plotting import plot_saliency_map


def main():
    images, labels, class_names = load_toy_images(n_samples=120, image_size=16)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = train_small_cnn(
        train_images,
        train_labels,
        image_size=16,
        epochs=5,
        random_state=42,
    )
    idx = 1
    image = test_images[idx]
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    saliency = saliency_map(model, tensor)

    print(f"测试图像真实类别: {class_names[test_labels[idx]]}")
    plot_saliency_map(image, saliency)
    plt.tight_layout()
    plt.savefig("saliency_map.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
