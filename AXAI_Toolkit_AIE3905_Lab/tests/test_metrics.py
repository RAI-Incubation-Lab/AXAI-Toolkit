# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.lime import LimeTabularExplainer
from axai_toolkit.metrics.quality import complexity, faithfulness, stability
from axai_toolkit.models.simple_models import train_logistic_regression


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


def test_faithfulness_returns_float(data):
    explainer = LimeTabularExplainer(
        data["X_train"],
        feature_names=data["feature_names"],
        random_state=42,
    )
    explanation = explainer.explain_instance(
        data["model"], data["X_test"][0], num_samples=100
    )
    value = faithfulness(
        data["model"],
        data["X_test"][0],
        explanation,
        X_mean=data["X_train"].mean(axis=0),
        top_k=2,
    )
    assert isinstance(value, float)


def test_stability_returns_float(data):
    explainer = LimeTabularExplainer(
        data["X_train"],
        feature_names=data["feature_names"],
        random_state=42,
    )
    value = stability(
        data["model"],
        data["X_test"][0],
        lambda x: explainer.explain_instance(data["model"], x, num_samples=80),
        n_samples=5,
        random_state=42,
    )
    assert -1.0 <= value <= 1.0


def test_complexity(data):
    explainer = LimeTabularExplainer(
        data["X_train"],
        feature_names=data["feature_names"],
        random_state=42,
    )
    explanation = explainer.explain_instance(
        data["model"], data["X_test"][0], num_samples=100
    )
    result = complexity(explanation, threshold=1e-3)
    assert result["active_features"] >= 0
    assert 0.0 <= result["normalized_entropy"] <= 1.0
