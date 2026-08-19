@echo off
rem 运行 AXAI Toolkit 单元测试
cd /d %~dp0
python -m pytest tests -v
pause
