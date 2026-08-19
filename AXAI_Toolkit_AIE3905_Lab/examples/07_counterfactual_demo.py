# -*- coding: utf-8 -*-
"""示例 7：反事实解释。

运行方式：
    python examples/07_counterfactual_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.counterfactual import greedy_counterfactual
from axai_toolkit.models.simple_models import train_logistic_regression
from axai_toolkit.visualization.plotting import plot_counterfactual_comparison


def main():
    X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
    model = train_logistic_regression(X_train, y_train)

    # 找一个原始预测为 0 的测试样本
    idx = int((model.predict(X_test) == 0).argmax())
    instance = X_test[idx]

    result = greedy_counterfactual(
        model,
        instance,
        X_train,
        y_train,
        max_changes=4,
        steps_per_feature=10,
        random_state=42,
    )

    print(f"原始类别: {result['original_class']}, 目标类别: {result['target_class']}")
    print(f"是否找到反事实: {result['found']}")
    print(f"修改的特征下标: {result['changed_features']}")
    if result["found"]:
        print("反事实样本：")
        for name, orig, cf in zip(
            feature_names, result["original_instance"], result["counterfactual"]
        ):
            mark = " *" if orig != cf else ""
            print(f"  {name:>15s}: {orig:+.3f} -> {cf:+.3f}{mark}")

        plot_counterfactual_comparison(
            result["original_instance"],
            result["counterfactual"],
            feature_names=feature_names,
        )
        plt.tight_layout()
        plt.savefig("counterfactual.png", dpi=150, bbox_inches="tight")
        plt.show()
    else:
        print("未找到反事实，请尝试增大 max_changes 或 steps_per_feature。")


if __name__ == "__main__":
    main()
