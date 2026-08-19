# -*- coding: utf-8 -*-
"""示例 1：快速上手——数据、模型与最简单的模型解释。

运行方式：
    python examples/01_intro.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.models.simple_models import train_logistic_regression


def main():
    X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
    model = train_logistic_regression(X_train, y_train)

    y_pred = model.predict(X_test)
    print("逻辑回归测试集准确率：", accuracy_score(y_test, y_pred))
    print("\n特征名称：", feature_names)

    # 最简单的模型解释：逻辑回归系数
    print("\n逻辑回归系数（正值促进类别 1，负值促进类别 0）：")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name:>15s}: {coef:+.3f}")

    # 查看一个测试样本
    idx = 0
    print(f"\n测试样本 {idx} 的真实标签: {y_test[idx]}, 预测标签: {model.predict(X_test[idx: idx+1])[0]}")
    print("样本特征：")
    for name, value in zip(feature_names, X_test[idx]):
        print(f"  {name:>15s}: {value:.3f}")


if __name__ == "__main__":
    main()
