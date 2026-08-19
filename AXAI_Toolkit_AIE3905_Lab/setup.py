# -*- coding: utf-8 -*-
"""AXAI Toolkit setup script."""
from setuptools import find_packages, setup

setup(
    name="axai-toolkit",
    version="0.1.0",
    description="A teaching-oriented Explainable AI (XAI) toolkit for undergraduate courses.",
    author="AXAI Course Team",
    packages=find_packages(include=["axai", "axai.*", "axai_toolkit", "axai_toolkit.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "pandas>=1.5",
        "matplotlib>=3.6",
        "Pillow>=9.0",
        "streamlit>=1.20",
        "typer>=0.9",
        "rich>=13.0",
        "pydantic>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "axai=axai_toolkit.cli:main",
        ],
    },
    extras_require={
        "dev": ["pytest>=7.0"],
        "deep": ["torch>=1.13", "torchvision>=0.14"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
