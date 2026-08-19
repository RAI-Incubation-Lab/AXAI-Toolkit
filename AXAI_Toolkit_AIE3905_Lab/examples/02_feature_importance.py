# -*- coding: utf-8 -*-
"""示例 2：置换特征重要性与部分依赖。

运行方式：
    python examples/02_feature_importance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.feature_importance import partial_dependence, permutation_importance
from axai_toolkit.models.simple_models import train_random_forest
from axai_toolkit.visualization.plotting import (
    plot_feature_importance,
    plot_partial_dependence,
)
import matplotlib.pyplot as plt


def main():
    X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
    model = train_random_forest(X_train, y_train, n_estimators=80, random_state=42)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        feature_names=feature_names,
        n_repeats=5,
        random_state=42,
    )
    print("置换重要性（均值 ± 标准差）：")
    for name, mean, std in zip(
        result["feature_names"], result["importances_mean"], result["importances_std"]
    ):
        print(f"  {name:>15s}: {mean:+.4f} ± {std:.4f}")

    plot_feature_importance(result, title="Permutation Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 部分依赖图：以第一个特征为例
    pd_result = partial_dependence(model, X_test, feature_index=0)
    plot_partial_dependence(pd_result, feature_name=feature_names[0])
    plt.tight_layout()
    plt.savefig("partial_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
