# -*- coding: utf-8 -*-
"""RAI 指标：处方级可修复性、合规度等。"""
from __future__ import annotations

from typing import Iterable, Optional

# 可执行政策套件（Policy Suites）示例
POLICY_SUITES: dict[str, list[tuple[str, str]]] = {
    "nist_ai_rmf": [
        ("Govern", "是否建立了 AI 治理角色与职责"),
        ("Map", "是否识别了 AI 系统的使用场景与风险"),
        ("Measure", "是否对 AI 风险进行了量化评估"),
        ("Manage", "是否制定了风险处置与持续监控流程"),
    ],
    "eu_ai_act_high_risk": [
        ("Risk Management", "高风险 AI 是否建立风险管理系统"),
        ("Data Governance", "训练数据是否满足质量与隐私要求"),
        ("Transparency", "是否向用户提供透明度说明"),
        ("Human Oversight", "是否提供人工监督机制"),
    ],
    "china_genai_filing": [
        ("Security Assessment", "是否完成生成式 AI 安全评估"),
        ("Content Safety", "是否有内容安全过滤机制"),
        ("User Notification", "是否向用户提示 AI 生成内容"),
        ("Data Protection", "是否遵守个人信息保护要求"),
    ],
}


def get_policy_suite(name: str) -> list[tuple[str, str]]:
    """获取一个政策套件。"""
    key = name.strip().lower()
    if key not in POLICY_SUITES:
        raise KeyError(f"未知政策套件: {name}。可用: {list(POLICY_SUITES)}")
    return POLICY_SUITES[key]


def run_policy_suite(
    name: str,
    check_results: Optional[dict[str, bool]] = None,
) -> dict:
    """运行一个政策套件并返回合规得分。

    Parameters
    ----------
    name : str
        政策套件名称，例如 "nist_ai_rmf"。
    check_results : dict[str, bool], optional
        检查项名称到是否通过的映射。缺失的检查项不会被臆测为通过，且会令
        总体合规分数标记为 ``None``，防止在没有证据时生成虚假的合规结论。
    """
    suite = get_policy_suite(name)
    check_results = check_results or {}
    expected_names = [item[0] for item in suite]
    unknown = sorted(set(check_results) - set(expected_names))
    invalid = sorted(name for name, value in check_results.items() if not isinstance(value, bool))
    if unknown:
        raise ValueError(f"检查结果含未知项目: {unknown}")
    if invalid:
        raise TypeError(f"检查结果必须为 bool: {invalid}")

    missing = [name for name in expected_names if name not in check_results]
    checks = [(name, check_results.get(name, False)) for name in expected_names]
    result = compliance_score(checks)
    if missing:
        result["compliance_score"] = None
        result["assessment_status"] = "incomplete"
        result["summary"] = "检查证据不完整，未生成总体合规分数。"
    else:
        result["assessment_status"] = "assessed"
    result["suite"] = name
    result["items"] = [
        {"name": item[0], "description": item[1], "passed": check_results.get(item[0])}
        for item in suite
    ]
    result["missing"] = missing
    return result


def prescriptive_remediation_index(
    findings: Iterable[dict],
) -> dict:
    """处方级可修复性指数（PRI）。

    衡量扫描发现的问题中有多少可以自动生成修复建议/补丁。
    """
    findings = list(findings)
    total = len(findings)
    if total == 0:
        return {
            "pri": 100.0,
            "fixable": 0,
            "total": 0,
            "summary": "未发现问题，PRI 为 100。",
        }
    fixable = sum(
        1
        for item in findings
        if item.get("fixable", False) or item.get("remediation")
    )
    pri = 100.0 * fixable / total
    return {
        "pri": float(pri),
        "fixable": fixable,
        "total": total,
        "summary": f"{fixable}/{total} 个问题可自动修复。",
    }


def compliance_score(checks: Iterable[tuple[str, bool]]) -> dict:
    """将法规/政策检查项转换为合规得分。"""
    checks = list(checks)
    if not checks:
        return {"compliance_score": 0.0, "passed": 0, "total": 0}
    passed = sum(1 for _, ok in checks if ok)
    return {
        "compliance_score": 100.0 * passed / len(checks),
        "passed": passed,
        "total": len(checks),
        "failed": [name for name, ok in checks if not ok],
    }
