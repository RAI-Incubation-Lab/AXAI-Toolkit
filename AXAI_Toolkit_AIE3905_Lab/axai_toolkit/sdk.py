# -*- coding: utf-8 -*-
"""AXAI SDK：无侵入式 Agent / 函数追踪装饰器。

对应 Proposal 中的：
    from axai import trace_agent, AuditConfig

    @trace_agent(config=AuditConfig(check_faithfulness=True, detect_pii=True, guard_sql=True))
    def my_university_agent(user_prompt: str):
        ...
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .rai.probes import detect_pii

# 简单 SQL 注入关键词
SQL_RISK_KEYWORDS = [
    "drop table",
    "delete from",
    "insert into",
    "update set",
    "alter table",
    "select * from",
    "--",
    ";",
]


@dataclass
class AuditConfig:
    """审计配置。"""

    check_faithfulness: bool = True
    detect_pii: bool = True
    guard_sql: bool = True
    allowed_tools: Optional[set[str]] = None
    log_trace: bool = True
    extra: dict = field(default_factory=dict)


def _check_sql(text: str) -> list[str]:
    lowered = text.lower()
    return [kw for kw in SQL_RISK_KEYWORDS if kw in lowered]


def trace_agent(config: Optional[AuditConfig] = None):
    """无侵入式装饰器：记录调用、检测 PII / SQL 风险，并附加审计轨迹。

    用法：
        @trace_agent(AuditConfig(detect_pii=True, guard_sql=True))
        def my_agent(user_prompt: str):
            return "result"
    """
    if config is None:
        config = AuditConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            trace = {
                "function": func.__name__,
                "args": [],
                "kwargs": {},
                "pii_findings": [],
                "sql_risks": [],
                "duration": 0.0,
                "result_preview": None,
            }

            # 仅记录可安全序列化的参数，避免记录敏感大对象
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    trace["args"].append(arg)
            for key, value in kwargs.items():
                if isinstance(value, (str, int, float, bool)):
                    trace["kwargs"][key] = value

            if config.detect_pii:
                for value in [*trace["args"], *trace["kwargs"].values()]:
                    if isinstance(value, str):
                        trace["pii_findings"].extend(detect_pii(value))

            if config.guard_sql:
                for value in [*trace["args"], *trace["kwargs"].values()]:
                    if isinstance(value, str):
                        trace["sql_risks"].extend(_check_sql(value))

            result = func(*args, **kwargs)
            trace["duration"] = time.time() - start
            trace["result_preview"] = str(result)[:200]

            if config.log_trace:
                # 将审计轨迹挂到函数返回值上，方便教学查看
                if isinstance(result, dict):
                    result["__axai_trace__"] = trace
                elif hasattr(result, "__dict__"):
                    try:
                        result.__dict__["__axai_trace__"] = trace
                    except Exception:
                        pass

            return result

        return wrapper

    return decorator
