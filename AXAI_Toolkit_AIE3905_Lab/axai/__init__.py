# -*- coding: utf-8 -*-
"""AXAI 兼容命名空间。

使得 Proposal 中的用法可以直接工作：

    from axai import trace_agent, AuditConfig
"""
from axai_toolkit.sdk import AuditConfig, trace_agent  # noqa: F401

__all__ = ["AuditConfig", "trace_agent"]
