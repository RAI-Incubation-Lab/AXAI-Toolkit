# -*- coding: utf-8 -*-
"""示例 12：Agent 工具调用因果图与权限审计。

运行方式：
    python examples/12_agent_audit_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axai_toolkit.agents.audit import audit_tool_permissions
from axai_toolkit.agents.traceability import (
    AgentStep,
    AgentTrace,
    build_tool_dag,
    detect_cycles,
    redundancy_score,
)


def main():
    trace = AgentTrace()
    trace.add_step(AgentStep("s1", thought="用户想查天气", action="search", parent=None))
    trace.add_step(AgentStep("s2", thought="需要再确认", action="search", parent="s1"))
    trace.add_step(AgentStep("s3", thought="尝试写入文件", action="write_file", parent="s2"))
    trace.add_step(AgentStep("s4", thought="完成任务", action="respond", parent="s3"))

    dag = build_tool_dag(trace)
    print("DAG:", dag)
    print("循环:", detect_cycles(trace))
    print("冗余分数:", redundancy_score(trace))

    audit = audit_tool_permissions(trace, allowed_tools={"search", "respond"})
    print("权限审计分数:", audit["score"])
    print("违规:", audit["violations"])


if __name__ == "__main__":
    main()
