@echo off
rem 启动 AXAI Toolkit Streamlit 交互式演示
cd /d %~dp0
streamlit run examples/app.py
pause
