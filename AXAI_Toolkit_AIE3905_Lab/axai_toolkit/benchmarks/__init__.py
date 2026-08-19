# -*- coding: utf-8 -*-
"""权威 Benchmark 注册表与评测体系映射。"""
from .registry import (  # noqa: F401
    BENCHMARK_REGISTRY,
    get_benchmark,
    list_benchmarks,
)

__all__ = ["BENCHMARK_REGISTRY", "get_benchmark", "list_benchmarks"]
