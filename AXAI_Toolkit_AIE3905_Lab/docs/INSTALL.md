# 安装指南

本文档说明如何在本地计算机上安装并运行 AXAI Toolkit。

## 环境要求

- Anaconda 或 Miniconda
- Python 3.9 或更高版本（建议 3.11）
- pip
- 可选：PyTorch（CPU 版即可，用于图像解释示例）

## 安装步骤

### 1. 进入工具箱目录

```bash
cd D:\FILE\CUHKSZ\project\Marcus\XAI\AXAI_Toolkit_AIE3905_Lab
```

### 2. 创建 conda 环境（强烈推荐）

```bash
conda create -n axai python=3.11 -y
conda activate axai
```

> 如果 `conda activate` 不可用，请先运行：
> ```bash
> conda init
> ```
> 然后重新打开终端，或直接使用 Anaconda Prompt。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 3.1 安装 CLI（可选但推荐）

```bash
pip install -e .
```

这样可以直接使用 `axai` 命令。如果不安装，也可以使用 `python -m axai_toolkit.cli`。

如果需要运行深度学习图像解释示例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. 验证安装

```bash
python -c "import axai_toolkit; print(axai_toolkit.__version__)"
```

如果输出 `0.1.0`，说明安装成功。

同时验证 SDK 命名空间：

```bash
python -c "from axai import trace_agent, AuditConfig; print('SDK OK')"
```

### 5. 验证 CLI

```bash
axai --help
```

如果 `axai` 命令不可用，可以运行：

```bash
python -m axai_toolkit.cli --help
```

## 运行示例

在 `AXAI_Toolkit_AIE3905_Lab` 根目录下运行：

```bash
python examples/01_intro.py
python examples/02_feature_importance.py
python examples/03_lime_demo.py
python examples/05_shap_demo.py
python examples/07_counterfactual_demo.py
python examples/08_metrics_demo.py
```

图像示例（需要 PyTorch）：

```bash
python examples/04_gradcam_demo.py
python examples/06_saliency_demo.py
python examples/09_integrated_gradients_demo.py
```

RAI / LLM / Agent 示例：

```bash
python examples/10_rai_scan_demo.py
python examples/11_llm_faithfulness_demo.py
python examples/12_agent_audit_demo.py
```

CLI 示例：

```bash
python -m axai_toolkit.cli scan examples/10_rai_scan_demo.py
python -m axai_toolkit.cli fix --prompt-file docs/INSTALL.md
```

交互式演示：

```bash
streamlit run examples/app.py
```

## 常见问题

- **pip 安装慢**：可配置国内镜像，例如 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。
- **PyTorch 安装失败**：跳过 `examples/04_gradcam_demo.py`、`examples/06_saliency_demo.py` 和 `examples/09_integrated_gradients_demo.py`，其余示例均可运行。
- **运行示例时提示找不到 `axai_toolkit`**：请确保在 `AXAI_Toolkit` 根目录下运行，示例脚本已自动将根目录加入 `sys.path`。
