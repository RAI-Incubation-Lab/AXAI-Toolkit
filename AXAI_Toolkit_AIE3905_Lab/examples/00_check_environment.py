# -*- coding: utf-8 -*-
"""环境自检脚本：检查运行 AXAI Toolkit 所需的核心依赖。"""
import importlib
import sys

REQUIRED = [
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "matplotlib",
    "PIL",
    "typer",
    "rich",
    "pydantic",
]
OPTIONAL = ["torch", "streamlit"]


def main():
    print(f"Python 版本: {sys.version}")
    ok = True
    for package in REQUIRED:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            ok = False
    for package in OPTIONAL:
        try:
            importlib.import_module(package)
            print(f"[OK] {package} (optional)")
        except ImportError:
            print(f"[INFO] {package} 未安装（可选）")

    if not ok:
        print("\n请先运行: pip install -r requirements.txt")
        sys.exit(1)
    print("\n环境检查通过，可以开始使用 AXAI Toolkit。")


if __name__ == "__main__":
    main()
