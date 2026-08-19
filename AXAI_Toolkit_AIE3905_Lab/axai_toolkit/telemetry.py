# -*- coding: utf-8 -*-
"""隐私友好型匿名遥测模块。

默认关闭。即使用户开启，也仅上报匿名化、非业务数据。
"""
from __future__ import annotations

import hashlib
import platform
import uuid
from pathlib import Path

_CFG_PATH = Path(__file__).resolve().parent / ".telemetry_config"


def _machine_hash() -> str:
    """生成不可逆的机器匿名标识。"""
    raw = platform.node() + platform.system() + platform.machine()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def is_enabled() -> bool:
    """检查遥测是否开启。"""
    if not _CFG_PATH.exists():
        return False
    return _CFG_PATH.read_text(encoding="utf-8").strip().lower() == "enabled"


def disable() -> None:
    """显式关闭遥测。"""
    _CFG_PATH.write_text("disabled", encoding="utf-8")


def enable() -> None:
    """显式开启遥测（默认不推荐开启）。"""
    _CFG_PATH.write_text("enabled", encoding="utf-8")


def anonymized_payload(framework: str = "unknown") -> dict:
    """构造仅含匿名元数据的遥测 payload。"""
    return {
        "machine_id_hash": _machine_hash(),
        "session_id": str(uuid.uuid4()),
        "os_type": platform.system(),
        "python_version": platform.python_version(),
        "framework": framework,
        "event": "axai_scan",
        # 严禁在此添加 prompt、业务数据或文件内容
    }
