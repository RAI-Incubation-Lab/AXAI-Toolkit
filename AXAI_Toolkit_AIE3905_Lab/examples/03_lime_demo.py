# -*- coding: utf-8 -*-
"""示例 3：LIME 局部解释。

运行方式：
    python examples/03_lime_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.lime import LimeTabularExplainer
from axai_toolkit.models.simple_models import train_random_forest
from axai_toolkit.visualization.plotting import plot_lime_weights


def main():
    X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
    model = train_random_forest(X_train, y_train, n_estimators=80, random_state=42)

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="classification",
        random_state=42,
    )
    idx = 3
    instance = X_test[idx]
    explanation = explainer.explain_instance(model, instance, num_samples=500)

    print(f"样本 {idx} 的预测类别: {explanation['predicted_class']}")
    print("LIME 局部权重：")
    for name, weight in explainer.as_list(explanation, num_features=5):
        print(f"  {name:>15s}: {weight:+.4f}")

    plot_lime_weights(explanation, num_features=5)
    plt.tight_layout()
    plt.savefig("lime_weights.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
