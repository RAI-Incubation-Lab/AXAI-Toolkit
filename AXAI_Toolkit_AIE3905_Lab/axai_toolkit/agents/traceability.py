# -*- coding: utf-8 -*-
"""Agent 工具调用因果图与轨迹分析。

使用轻量字典/集合表示 DAG，不强制依赖 networkx。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentStep:
    """Agent 单步轨迹。"""

    step_id: str
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    parent: Optional[str] = None


@dataclass
class AgentTrace:
    """Agent 完整执行轨迹。"""

    steps: list[AgentStep] = field(default_factory=list)

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)


def build_tool_dag(trace: AgentTrace) -> dict[str, list[str]]:
    """根据 parent 关系构建工具调用 DAG。

    Returns
    -------
    dict[str, list[str]]
        节点 ID 到其直接子节点列表的映射。
    """
    dag: dict[str, list[str]] = {step.step_id: [] for step in trace.steps}
    for step in trace.steps:
        if step.parent and step.parent in dag:
            dag[step.parent].append(step.step_id)
    return dag


def detect_cycles(trace: AgentTrace) -> list[list[str]]:
    """检测轨迹中是否存在循环依赖（简化版 DFS）。

    教学实现返回所有包含环的路径。
    """
    dag = build_tool_dag(trace)
    visited = set()
    stack = []
    cycles = []

    def dfs(node: str, path: list[str]) -> None:
        if node in stack:
            start = stack.index(node)
            cycles.append(stack[start:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        stack.append(node)
        for child in dag.get(node, []):
            dfs(child, path + [child])
        stack.pop()

    for node in dag:
        dfs(node, [node])
    return cycles


def compute_minimal_path(
    trace: AgentTrace,
    start: Optional[str] = None,
    goal: Optional[str] = None,
) -> list[str]:
    """计算从起点到终点的最短必要路径（BFS）。"""
    if not trace.steps:
        return []
    if start is None:
        start = trace.steps[0].step_id
    if goal is None:
        goal = trace.steps[-1].step_id

    dag = build_tool_dag(trace)
    from collections import deque

    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for child in dag.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append((child, path + [child]))
    return []


def redundancy_score(trace: AgentTrace) -> float:
    """计算冗余调用比例。

    重复工具调用次数 / 总调用次数；越高表示越可能存在无效循环或冗余。
    """
    if not trace.steps:
        return 0.0
    actions = [step.action for step in trace.steps if step.action]
    if not actions:
        return 0.0
    repeated = sum(1 for i, action in enumerate(actions) if action in actions[:i])
    return repeated / len(actions)
