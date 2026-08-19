# -*- coding: utf-8 -*-
"""Simple machine learning models for XAI demonstrations."""
from .simple_models import (  # noqa: F401
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
    train_small_cnn,
)

__all__ = [
    "train_logistic_regression",
    "train_decision_tree",
    "train_random_forest",
    "train_small_cnn",
]
