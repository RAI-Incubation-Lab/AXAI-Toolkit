# -*- coding: utf-8 -*-
"""示例 8：解释质量评估。

运行方式：
    python examples/08_metrics_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.lime import LimeTabularExplainer
from axai_toolkit.metrics.quality import complexity, faithfulness, stability
from axai_toolkit.models.simple_models import train_random_forest


def main():
    X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
    model = train_random_forest(X_train, y_train, n_estimators=80, random_state=42)

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="classification",
        random_state=42,
    )
    idx = 5
    instance = X_test[idx]
    explanation = explainer.explain_instance(model, instance, num_samples=300)

    # Faithfulness：移除 LIME 认为最重要的 3 个特征后，预测概率下降多少
    faith = faithfulness(
        model,
        instance,
        explanation,
        X_mean=X_train.mean(axis=0),
        top_k=3,
    )

    # Stability：在样本附近扰动后，LIME 解释的余弦相似度
    stab = stability(
        model,
        instance,
        lambda x: explainer.explain_instance(model, x, num_samples=200),
        n_samples=10,
        perturbation_std=0.02,
        random_state=42,
    )

    # Complexity：解释使用的特征数量与归一化熵
    comp = complexity(explanation, threshold=1e-3)

    print(f"样本 {idx} 的预测类别: {explanation['predicted_class']}")
    print(f"Faithfulness (top-3 删除后概率下降): {faith:.4f}")
    print(f"Stability (平均余弦相似度): {stab:.4f}")
    print(f"Complexity (活跃特征数): {comp['active_features']}")
    print(f"Complexity (归一化熵): {comp['normalized_entropy']:.4f}")


if __name__ == "__main__":
    main()
