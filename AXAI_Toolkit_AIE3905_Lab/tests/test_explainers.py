# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.counterfactual import greedy_counterfactual
from axai_toolkit.explainers.feature_importance import partial_dependence, permutation_importance
from axai_toolkit.explainers.lime import LimeTabularExplainer
from axai_toolkit.explainers.shap import exact_shapley
from axai_toolkit.models.simple_models import train_logistic_regression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
    model = train_logistic_regression(X_train, y_train)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "model": model,
    }


def test_permutation_importance(data):
    result = permutation_importance(
        data["model"],
        data["X_test"],
        data["y_test"],
        feature_names=data["feature_names"],
        n_repeats=2,
    )
    assert result["importances_mean"].shape == (len(data["feature_names"]),)
    assert result["importances_std"].shape == result["importances_mean"].shape


def test_partial_dependence(data):
    result = partial_dependence(data["model"], data["X_test"], feature_index=0, grid_points=5)
    assert len(result["values"]) == 5
    assert len(result["average_prediction"]) == 5


def test_multiclass_partial_dependence_is_not_constant_probability_average():
    X, y = make_classification(
        n_samples=180, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, random_state=3,
    )
    model = LogisticRegression(max_iter=1000, multi_class="auto").fit(X, y)
    result = partial_dependence(model, X, feature_index=0, grid_points=6, class_index=1)
    assert result["class_index"] == 1
    assert not np.allclose(result["average_prediction"], 1 / 3)


def test_lime_explanation(data):
    explainer = LimeTabularExplainer(
        data["X_train"],
        feature_names=data["feature_names"],
        random_state=42,
    )
    explanation = explainer.explain_instance(
        data["model"], data["X_test"][0], num_samples=100
    )
    assert explanation["coefficients"].shape == (len(data["feature_names"]),)
    assert len(explainer.as_list(explanation, num_features=3)) == 3


def test_exact_shapley(data):
    explanation = exact_shapley(
        data["model"],
        data["X_train"],
        data["X_test"][0],
        feature_names=data["feature_names"],
    )
    assert explanation["values"].shape == (len(data["feature_names"]),)
    assert np.isfinite(explanation["values"]).all()


def test_counterfactual(data):
    instance = data["X_test"][0]
    result = greedy_counterfactual(
        data["model"],
        instance,
        data["X_train"],
        data["y_train"],
        max_changes=5,
        steps_per_feature=5,
    )
    # 至少应返回一个反事实结果对象
    assert "counterfactual" in result
    assert result["counterfactual"].shape == instance.shape


def test_counterfactual_accepts_nonzero_class_labels():
    X, y, _ = load_demo_classification()
    y = np.where(y == 0, 10, 20)
    model = train_logistic_regression(X, y)
    result = greedy_counterfactual(model, X[0], X, y, target_class=20)
    assert result["target_class"] == 20
