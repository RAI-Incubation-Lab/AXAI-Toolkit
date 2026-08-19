# 构建者与维护者指南（Developer Guide）

本文档整合了项目结构、模块说明、Benchmark 对照、扩展方式、路线图与代码审查记录，适用于工具包构建者 / 维护者。

## 1. 项目结构

```text
AXAI_Toolkit/
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
├── .github/workflows/axai-scan.yml
├── .pre-commit-config.yaml
├── axai/                  # 兼容命名空间：from axai import trace_agent
├── axai_toolkit/
│   ├── cli.py
│   ├── sdk.py
│   ├── telemetry.py
│   ├── agents/
│   ├── benchmarks/
│   ├── data/
│   ├── explainers/
│   ├── llm/
│   ├── metrics/
│   ├── models/
│   ├── rai/
│   ├── remediation/
│   ├── reporting/
│   ├── utils/
│   └── visualization/
├── examples/
├── docs/
├── scripts/
└── tests/
```

## 2. 开发环境准备

```bash
cd /d D:\FILE\CUHKSZ\project\Marcus\XAI\AXAI_Toolkit
conda create -n axai python=3.11 -y
conda activate axai
pip install -r requirements.txt
pip install -e .
```

验证：

```bash
python -m pytest tests -v
axai --help
python -c "from axai import trace_agent, AuditConfig"
```

## 3. 常用开发命令

```bash
# 测试
python -m pytest tests -v

# 运行全部示例
python examples/00_check_environment.py
python examples/10_rai_scan_demo.py
python examples/11_llm_faithfulness_demo.py
python examples/12_agent_audit_demo.py
python examples/13_sdk_decorator_demo.py
python examples/14_policy_compliance_demo.py
python examples/15_ast_linter_demo.py

# 交互式演示
streamlit run examples/app.py

# CLI
axai scan . --export-html report.html --export-json report.json
axai fix --prompt-file README.md
axai telemetry status
```

## 4. 模块说明

| 模块 | 职责 |
| ---- | ---- |
| `axai_toolkit.data` | 教学数据生成与加载 |
| `axai_toolkit.models` | 简单 ML / CNN 模型 |
| `axai_toolkit.explainers` | 经典 XAI：LIME、SHAP、Saliency、Grad-CAM、Integrated Gradients、Counterfactual |
| `axai_toolkit.metrics` | Faithfulness、Stability、Complexity、PRI、Compliance、Policy Suites |
| `axai_toolkit.rai` | 30 条 RAI 探针 + AST Linter |
| `axai_toolkit.llm` | CoT 忠实度、RAG Grounding、Prompt 加固 |
| `axai_toolkit.agents` | Agent 工具调用 DAG、循环检测、权限审计 |
| `axai_toolkit.sdk` | `trace_agent` 装饰器与 `AuditConfig` |
| `axai_toolkit.reporting` | HTML / JSON 报告 + GitHub 徽章 |
| `axai_toolkit.remediation` | Prompt 补丁、PII 脱敏中间件 |
| `axai_toolkit.benchmarks` | 外部 Benchmark 注册表 |
| `axai_toolkit.cli` | `axai` 命令行入口 |
| `axai_toolkit.telemetry` | 匿名遥测（默认关闭） |

## 5. 权威 Benchmark 对照

| 评测体系 | 领域 | AXAI 对应能力 |
| -------- | ---- | ------------- |
| Quantus | 经典/深度 XAI | `metrics.quality` |
| OpenXAI | 表格后验 XAI | `explainers` + `metrics.quality` |
| ERASER | NLP XAI | `llm.faithfulness` |
| FaithCoT-Bench | LLM CoT | `llm.faithfulness.counterfactual_mutation` |
| DecodingTrust | RAI | `rai.probes` |
| TrustLLM / HELM | RAI | `rai.probes` + `metrics.rai` |
| NIST AI RMF / ISO 42001 | 治理合规 | `metrics.rai.compliance_score` |
| RAGAS / TruLens | RAG 质量 | `llm.grounding` |
| HarmBench | Safety | `rai.probes.JAILBREAK_PROBES` |

代码查看：

```python
from axai_toolkit.benchmarks import list_benchmarks, get_benchmark
```

## 6. 扩展指南

### 6.1 新增 RAI 探针

在 `axai_toolkit/rai/probes.py` 的 `JAILBREAK_PROBES`、`BIAS_PROBES` 或 `PII_INJECTION_PROBES` 中添加条目。

### 6.2 新增 XAI 方法

1. 在 `axai_toolkit/explainers/` 新建模块。
2. 在 `explainers/__init__.py` 导出。
3. 添加示例与测试。
4. 更新 `README.md` 与本文档。

### 6.3 新增 CLI 子命令

在 `axai_toolkit/cli.py` 使用 `@app.command()` 添加。

### 6.4 新增 Policy Suite

在 `axai_toolkit/metrics/rai.py` 的 `POLICY_SUITES` 中新增条目，并使用 `run_policy_suite()` 运行。

### 6.5 扩展 SDK

在 `axai_toolkit/sdk.py` 中扩展 `AuditConfig` 或 `trace_agent` 的审计逻辑。

### 6.6 新增报告格式

在 `axai_toolkit/reporting/` 中新增模块，并同步到 `reporting/__init__.py`。

## 7. 代码审查记录

### 已完成

- CLI：`scan / test / fix / telemetry`
- 30 条对抗探针
- AST Linter
- SDK 装饰器
- HTML / JSON 报告
- GitHub 徽章
- Policy Suites
- GitHub Actions / pre-commit
- 构建者与学生文档整合

### 待后续迭代

- 接入真实 LLM API（OpenAI / DeepSeek）
- Cloudflare Workers / Supabase 遥测上报
- 班级排行榜 / 大屏看板
- 官方 LangChain / LlamaIndex / Dify / CrewAI Connectors
- PDF 报告导出
- 机制可解释性 / SAE 深度集成

## 8. 路线图

### Phase 1：高校课堂极限冷启动（Month 1-2）

- 100+ 学生、30+ 异构 AI 项目试验
- 打磨 3 分钟上手指南
- 沉淀对抗用例库
- 班级排行榜看板

### Phase 2：开源重构与全球社区首发（Month 3-5）

- 重构 Monorepo
- arXiv 技术报告 / Demo 论文
- Hacker News、Reddit、HuggingFace 首发

### Phase 3：生态渠道与开发者工作流集成（Month 6-9）

- GitHub Actions 市场插件
- LangChain / LlamaIndex / Dify / CrewAI Connectors
- 全球贡献者计划

### Phase 4：行业基准与企业合规平台（Month 10-12+）

- 年度评测白皮书
- 企业合规审计扩展包
- 成为事实标准

## 9. 打包与发布

```bash
# 本地可编辑安装
pip install -e .

# 构建分发包
pip install build
python -m build
```

发布前检查：

- `python -m pytest tests -v`
- `python examples/00_check_environment.py`
- `axai scan .` 可运行
- `python -c "from axai import trace_agent, AuditConfig"` 可导入
- 所有新增文件均在 `AXAI_Toolkit` 文件夹内

## 10. 注意事项

- 所有代码与文档使用 UTF-8 编码。
- 不要在本项目之外创建或修改文件。
- 新增依赖请同步更新 `requirements.txt`、`setup.py`、`pyproject.toml`。
- 遥测默认关闭，禁止上报 Prompt 或业务数据。
