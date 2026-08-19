# -*- coding: utf-8 -*-
"""示例 5：SHAP 值解释。

运行方式：
    python examples/05_shap_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.shap import exact_shapley
from axai_toolkit.models.simple_models import train_random_forest
from axai_toolkit.visualization.plotting import plot_feature_importance


def main():
    X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
    model = train_random_forest(X_train, y_train, n_estimators=80, random_state=42)

    idx = 2
    instance = X_test[idx]
    explanation = exact_shapley(model, X_train, instance, feature_names=feature_names)

    print(f"样本 {idx} 的 SHAP 值：")
    for name, value in zip(explanation["feature_names"], explanation["values"]):
        print(f"  {name:>15s}: {value:+.4f}")

    plot_feature_importance(explanation, title=f"SHAP Values for Sample {idx}")
    plt.tight_layout()
    plt.savefig("shap_values.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
