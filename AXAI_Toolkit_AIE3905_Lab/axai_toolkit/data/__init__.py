# -*- coding: utf-8 -*-
"""Data loading and synthetic dataset generators for teaching."""
from .datasets import (  # noqa: F401
    load_demo_classification,
    load_demo_regression,
    load_toy_images,
    make_classification_data,
    make_regression_data,
)

__all__ = [
    "load_demo_classification",
    "load_demo_regression",
    "load_toy_images",
    "make_classification_data",
    "make_regression_data",
]
