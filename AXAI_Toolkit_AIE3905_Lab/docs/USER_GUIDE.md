# 用户与教学指南（User & Teaching Guide）

本文档整合了学生使用指引、课堂教学安排、综合项目与评分建议，适用于学生、助教和授课教师。

## 1. 环境准备

默认学生已安装 Anaconda。请先按 `INSTALL.md` 完成环境配置：

```bash
cd /d D:\FILE\CUHKSZ\project\Marcus\XAI\AXAI_Toolkit
conda create -n axai python=3.11 -y
conda activate axai
pip install -r requirements.txt
pip install -e .
```

验证环境：

```bash
python examples/00_check_environment.py
```

## 2. 快速开始

```bash
# 经典 XAI
python examples/01_intro.py
python examples/02_feature_importance.py
python examples/03_lime_demo.py
python examples/05_shap_demo.py

# 图像解释（需要 PyTorch）
python examples/04_gradcam_demo.py
python examples/06_saliency_demo.py
python examples/09_integrated_gradients_demo.py

# RAI / LLM / Agent / SDK / Policy / AST
python examples/10_rai_scan_demo.py
python examples/11_llm_faithfulness_demo.py
python examples/12_agent_audit_demo.py
python examples/13_sdk_decorator_demo.py
python examples/14_policy_compliance_demo.py
python examples/15_ast_linter_demo.py
```

交互式课堂演示：

```bash
streamlit run examples/app.py
```

## 3. CLI 使用

```bash
# 扫描项目或 Prompt
axai scan . --export-html report.html --export-json report.json

# 动态测试占位命令
axai test --entry agent.py:run_agent --suite xai,safety,bias --export-html audit.html

# Prompt 加固
axai fix --prompt-file system_prompt.txt --apply-prompt-patch

# 遥测管理
axai telemetry status
axai telemetry disable
```

如果 `axai` 不可用，可使用：

```bash
python -m axai_toolkit.cli ...
```

## 4. Python 调用示例

```python
from axai import trace_agent, AuditConfig
from axai_toolkit.data.datasets import load_demo_classification
from axai_toolkit.explainers.lime import LimeTabularExplainer
from axai_toolkit.models.simple_models import train_random_forest

# 经典 XAI
X_train, X_test, y_train, y_test, feature_names = load_demo_classification()
model = train_random_forest(X_train, y_train)
explainer = LimeTabularExplainer(X_train, feature_names=feature_names)
explanation = explainer.explain_instance(model, X_test[0])
print(explainer.as_list(explanation))

# SDK 无侵入监控
@trace_agent(config=AuditConfig(detect_pii=True, guard_sql=True))
def my_agent(user_prompt: str):
    return {"output": "ok"}
```

## 5. 课堂实验安排（10 周）

| 周次 | 主题 | 示例 |
| ---- | ---- | ---- |
| 第 1 周 | 机器学习与可解释性入门 | `examples/01_intro.py` |
| 第 2 周 | 特征重要性与部分依赖 | `examples/02_feature_importance.py` |
| 第 3 周 | LIME 局部解释 | `examples/03_lime_demo.py` |
| 第 4 周 | SHAP 值 | `examples/05_shap_demo.py` |
| 第 5 周 | 深度学习解释 | `examples/04_gradcam_demo.py`、`06_saliency_demo.py`、`09_integrated_gradients_demo.py` |
| 第 6 周 | 反事实与解释评估 | `examples/07_counterfactual_demo.py`、`08_metrics_demo.py` |
| 第 7 周 | RAI 安全扫描 | `examples/10_rai_scan_demo.py` |
| 第 8 周 | LLM 思维链与 RAG | `examples/11_llm_faithfulness_demo.py` |
| 第 9 周 | Agent 审计 | `examples/12_agent_audit_demo.py` |
| 第 10 周 | SDK / Policy / 综合项目 | `examples/13_sdk_decorator_demo.py`、`14_policy_compliance_demo.py`、`15_ast_linter_demo.py` |

## 6. 综合项目要求

学生可选择以下方向之一：

1. 表格数据 XAI 分析
2. 图像模型解释
3. LLM / RAG 安全与忠实度审计
4. Agent 权限与工具调用审计
5. SDK 装饰器与 Policy 合规审计

建议流程：

1. 运行对应示例，理解输出。
2. 编写或修改自己的脚本。
3. 使用至少两种解释/审计方法。
4. 使用至少两个指标，例如：
   - `faithfulness`
   - `stability`
   - `complexity`
   - `prescriptive_remediation_index`
   - `compliance_score`
5. 生成 HTML / JSON 报告。
6. 撰写报告并说明结果意义。

## 7. 评分建议

| 维度 | 分值 | 说明 |
| ---- | ---- | ---- |
| 问题定义 | 15 | 是否清楚说明为什么要解释/审计模型 |
| 方法选择 | 20 | 方法是否适合数据与模型 |
| 实验设计 | 20 | 是否合理比较不同方法 |
| 可视化 | 15 | 图表是否清晰、规范 |
| 评估与讨论 | 20 | 是否使用指标评价结果并讨论局限 |
| 代码质量 | 10 | 代码可读、可复现、有注释 |

加分项：

- 新增一种 RAI 探针或 XAI 方法
- 使用真实公开数据集 / 真实 LLM API
- 增加 GitHub Actions 示例
- 将 Streamlit 演示扩展为可上传数据的应用

## 8. 常见问题

### 8.1 找不到 `axai_toolkit`

确保在 `AXAI_Toolkit` 根目录运行，并已执行：

```bash
pip install -e .
```

### 8.2 `axai` 命令不存在

使用：

```bash
python -m axai_toolkit.cli ...
```

### 8.3 图像示例无法运行

安装 CPU 版 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 8.4 运行测试

```bash
python -m pytest tests -v
```

## 9. 提交前检查清单

- [ ] 所有代码可以运行
- [ ] 使用了至少一种 XAI 方法
- [ ] 使用了至少一种 RAI / 安全 / 审计方法
- [ ] 有输出截图或 HTML / JSON 报告
- [ ] 在报告中解释了结果的意义
- [ ] 没有修改项目文件夹以外的内容
