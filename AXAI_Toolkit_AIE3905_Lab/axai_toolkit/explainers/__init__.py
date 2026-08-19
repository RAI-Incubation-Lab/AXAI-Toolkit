# -*- coding: utf-8 -*-
"""Explainability methods implemented for teaching."""
from .feature_importance import (  # noqa: F401
    partial_dependence,
    permutation_importance,
)
from .lime import LimeTabularExplainer  # noqa: F401
from .shap import exact_shapley, shapley_sampling  # noqa: F401
from .counterfactual import greedy_counterfactual  # noqa: F401
from .integrated_gradients import integrated_gradients  # noqa: F401
from .saliency import saliency_map  # noqa: F401
from .gradcam import GradCAMExplainer, gradcam  # noqa: F401

__all__ = [
    "permutation_importance",
    "partial_dependence",
    "LimeTabularExplainer",
    "exact_shapley",
    "shapley_sampling",
    "greedy_counterfactual",
    "integrated_gradients",
    "saliency_map",
    "gradcam",
    "GradCAMExplainer",
]
