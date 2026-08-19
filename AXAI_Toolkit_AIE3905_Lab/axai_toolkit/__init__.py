# -*- coding: utf-8 -*-
"""AXAI Toolkit: an educational Explainable AI toolkit.

本工具箱用于本科课堂讲解可解释人工智能（XAI）的基础概念与常用方法。
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    agents,
    benchmarks,
    llm,
    rai,
    remediation,
    reporting,
    sdk,
    telemetry,
)
from .sdk import AuditConfig, trace_agent  # noqa: F401
from .data.datasets import (  # noqa: F401
    load_demo_classification,
    load_demo_regression,
    load_toy_images,
    make_classification_data,
    make_regression_data,
)
from .models.simple_models import (  # noqa: F401
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
    train_small_cnn,
)

__all__ = [
    "__version__",
    "agents",
    "benchmarks",
    "llm",
    "rai",
    "remediation",
    "reporting",
    "sdk",
    "telemetry",
    "AuditConfig",
    "trace_agent",
    "load_demo_classification",
    "load_demo_regression",
    "load_toy_images",
    "make_classification_data",
    "make_regression_data",
    "train_logistic_regression",
    "train_decision_tree",
    "train_random_forest",
    "train_small_cnn",
]
