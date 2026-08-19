# AXAI-Toolkit：下一代应用型可解释与负责任 AI 评测治理框架

本工具箱面向大学课堂与开源生态，既包含经典机器学习 / 深度学习 XAI 教学模块，也包含大模型（LLM）、RAG、智能体（Agent）与 Responsible AI（RAI）的评测与治理能力。核心功能不依赖外部网络，可直接在本地 Python 环境运行。

> 愿景：让任何开发者通过一行命令或几行代码，即可完成 AI 应用的透明度体检、安全合规漏洞扫描、因果推理追踪，并获得“处方级”修复建议。

## 目录

- [简介](#简介)
- [功能模块](#功能模块)
- [安装方法](#安装方法)
- [CLI 快速开始](#cli-快速开始)
- [Python 快速开始](#python-快速开始)
- [教学建议](#教学建议)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

## 简介

AXAI-Toolkit 旨在帮助学生和开发者理解：

- 经典 XAI：LIME、SHAP、Saliency、Grad-CAM、Integrated Gradients、反事实解释；
- 解释质量评估：Faithfulness、Stability、Complexity；
- LLM 可解释性：思维链忠实度（Causal CoT Faithfulness）、RAG 幻觉检测（Grounding Ratio）；
- Agent 透明度：工具调用 DAG、循环检测、权限越权审计；
- Responsible AI：越狱探针、偏见探针、PII 检测、Prompt 加固、合规评分。

工具箱采用“先传统机器学习，后大模型/Agent”的教学路径，并提供 CLI、SDK、Streamlit 交互式演示与 HTML 审计报告。

## 功能模块

| 模块 | 说明 |
| ---- | ---- |
| `axai_toolkit.data` | 生成/加载教学用数据集，包括表格数据与图像数据 |
| `axai_toolkit.models` | 训练简单的可解释基线模型（逻辑回归、决策树、小型 CNN） |
| `axai_toolkit.explainers` | 经典 XAI：LIME、SHAP、Saliency、Grad-CAM、Integrated Gradients、Counterfactual |
| `axai_toolkit.metrics` | 解释质量与 RAI 指标：Faithfulness、Stability、Complexity、PRI、Compliance、Policy Suites |
| `axai_toolkit.rai` | 30 条 RAI 探针：越狱、偏见、PII 注入 + AST Linter |
| `axai_toolkit.llm` | LLM 解释：CoT 忠实度、RAG Grounding、Prompt 加固 |
| `axai_toolkit.agents` | Agent 审计：工具调用 DAG、循环检测、权限越权 |
| `axai_toolkit.sdk` | `trace_agent` 装饰器与 `AuditConfig`，支持 `from axai import ...` |
| `axai_toolkit.reporting` | HTML / JSON 诊断报告 + GitHub 徽章 |
| `axai_toolkit.remediation` | 处方级修复：Prompt 补丁、PII 脱敏中间件 |
| `axai_toolkit.benchmarks` | 权威 Benchmark 注册表（Quantus、OpenXAI、DecodingTrust、TrustLLM、HELM 等） |
| `axai_toolkit.telemetry` | 隐私友好型匿名遥测（默认关闭） |
| `axai_toolkit.cli` | `axai scan/test/fix/telemetry` 命令行入口 |
| `examples` | 可直接运行的示例脚本 |
| `tests` | 单元测试 |
| `docs` | 安装指南、用户/教学指南、开发者指南 |

## 安装方法

推荐使用 Anaconda 创建独立环境：

```bash
# 进入工具箱目录
cd AXAI_Toolkit

# 创建 conda 环境（建议 Python 3.11）
conda create -n axai python=3.11 -y

# 激活环境
conda activate axai

# 安装依赖
pip install -r requirements.txt

# 若希望以可编辑方式安装本工具包
pip install -e .
```

> 图像深度学习示例需要安装 PyTorch（CPU 版即可）。若课堂环境无法安装 PyTorch，
> 仍可运行除 `04_gradcam_demo.py`、`06_saliency_demo.py` 和 `09_integrated_gradients_demo.py` 之外的全部示例。

## CLI 快速开始

安装本工具包后，可以直接使用 `axai` 命令：

```bash
# 1. 对 AI 项目 / Prompt 文件目录执行静态扫描
axai scan ./my_student_project/ --export-html report.html --export-json report.json

# 2. 动态红队测试（教学占位实现，可接入真实模型）
axai test --entry agent.py:run_agent --suite xai,safety,bias --export-html audit_report.html

# 3. Prompt 自动加固并查看 diff
axai fix --prompt-file system_prompt.txt --apply-prompt-patch

# 4. 遥测管理（默认关闭）
axai telemetry status
axai telemetry disable
```

如果尚未安装为全局命令，也可以运行：

```bash
python -m axai_toolkit.cli scan .
```

## Python 快速开始

### 0. 检查环境

```bash
python examples/00_check_environment.py
```

Windows 一键脚本（可选）：

```bat
run_tests.bat
start_app.bat
```

### 1. 经典表格数据 + 模型无关解释

```bash
python examples/02_feature_importance.py
python examples/03_lime_demo.py
```

### 2. 图像数据 + 深度学习解释

```bash
python examples/04_gradcam_demo.py
python examples/06_saliency_demo.py
python examples/09_integrated_gradients_demo.py
```

### 3. RAI / LLM / Agent 教学示例

```bash
python examples/10_rai_scan_demo.py
python examples/11_llm_faithfulness_demo.py
python examples/12_agent_audit_demo.py
python examples/13_sdk_decorator_demo.py
python examples/14_policy_compliance_demo.py
python examples/15_ast_linter_demo.py
```

SDK 一行接入：

```python
from axai import trace_agent, AuditConfig

@trace_agent(config=AuditConfig(detect_pii=True, guard_sql=True))
def my_agent(user_prompt: str):
    return {"output": "ok"}
```

### 4. 交互式课堂演示

```bash
streamlit run examples/app.py
```

### 5. 运行测试

```bash
python -m pytest tests -v
```

## 教学建议

- **第 1 周**：机器学习与可解释性概述，运行 `examples/01_intro.py`。
- **第 2 周**：特征重要性、部分依赖图，运行 `examples/02_feature_importance.py`。
- **第 3 周**：LIME 与局部解释，运行 `examples/03_lime_demo.py`。
- **第 4 周**：SHAP 与博弈论解释，运行 `examples/05_shap_demo.py`。
- **第 5 周**：深度学习 Saliency / Grad-CAM / Integrated Gradients，运行 `examples/04_gradcam_demo.py`、`examples/06_saliency_demo.py`、`examples/09_integrated_gradients_demo.py`。
- **第 6 周**：反事实解释与解释评估，运行 `examples/07_counterfactual_demo.py`、`examples/08_metrics_demo.py`。
- **第 7 周**：RAI 安全扫描与 PII 检测，运行 `examples/10_rai_scan_demo.py`。
- **第 8 周**：LLM 思维链忠实度与 RAG 幻觉检测，运行 `examples/11_llm_faithfulness_demo.py`。
- **第 9 周**：Agent 工具调用追踪与权限审计，运行 `examples/12_agent_audit_demo.py`。
- **第 10 周**：综合项目，学生完成一份包含 XAI/RAI/Policy 的 AI 系统审计报告，可结合 `examples/13_sdk_decorator_demo.py`、`examples/14_policy_compliance_demo.py`、`examples/15_ast_linter_demo.py`。

## 项目结构

```
AXAI_Toolkit/
├── README.md
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── requirements.txt
├── run_tests.bat
├── setup.py
├── start_app.bat
├── .github/
│   └── workflows/
│       └── axai-scan.yml
├── .pre-commit-config.yaml
├── axai/
│   └── __init__.py
├── axai_toolkit/
│   ├── __init__.py
│   ├── cli.py
│   ├── sdk.py
│   ├── telemetry.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── audit.py
│   │   └── traceability.py
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── registry.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py
│   ├── explainers/
│   │   ├── __init__.py
│   │   ├── feature_importance.py
│   │   ├── lime.py
│   │   ├── shap.py
│   │   ├── saliency.py
│   │   ├── gradcam.py
│   │   ├── integrated_gradients.py
│   │   └── counterfactual.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── faithfulness.py
│   │   ├── grounding.py
│   │   └── prompt_guard.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── quality.py
│   │   └── rai.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── simple_models.py
│   ├── rai/
│   │   ├── __init__.py
│   │   ├── ast_linter.py
│   │   └── probes.py
│   ├── remediation/
│   │   ├── __init__.py
│   │   └── prompt_patch.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── badge.py
│   │   ├── html_report.py
│   │   └── json_report.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── common.py
│   └── visualization/
│       ├── __init__.py
│       └── plotting.py
├── examples/
│   ├── 00_check_environment.py
│   ├── 01_intro.py
│   ├── 02_feature_importance.py
│   ├── 03_lime_demo.py
│   ├── 04_gradcam_demo.py
│   ├── 05_shap_demo.py
│   ├── 06_saliency_demo.py
│   ├── 07_counterfactual_demo.py
│   ├── 08_metrics_demo.py
│   ├── 09_integrated_gradients_demo.py
│   ├── 10_rai_scan_demo.py
│   ├── 11_llm_faithfulness_demo.py
│   ├── 12_agent_audit_demo.py
│   ├── 13_sdk_decorator_demo.py
│   ├── 14_policy_compliance_demo.py
│   ├── 15_ast_linter_demo.py
│   └── app.py
├── docs/
│   ├── INSTALL.md
│   ├── USER_GUIDE.md
│   └── DEVELOPER_GUIDE.md
├── scripts/
│   └── read_docx.py
└── tests/
    ├── test_explainers.py
    ├── test_metrics.py
    └── test_rai_llm_agents.py
```

## 常见问题

- **Q: 没有 GPU 可以运行吗？**
  A: 可以。所有示例均可在 CPU 上运行，PyTorch 使用 CPU 版即可。

- **Q: 无法安装 PyTorch？**
  A: 可跳过图像相关示例，表格数据的解释方法不依赖 PyTorch。

- **Q: 如何添加自己的数据集？**
  A: 将 CSV 文件放入 `data/` 或自定义加载函数，参考 `axai_toolkit/data/datasets.py`。

## 许可证

本项目仅用于课堂教学与学术研究，所有生成文件均保存在 `AXAI_Toolkit` 文件夹内。
