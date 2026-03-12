@echo off
REM 机器学习 Web UI 启动脚本 (Windows)

echo 🚀 启动机器学习 Web UI...
echo.

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python
    pause
    exit /b 1
)

REM 检查依赖
echo 📦 检查依赖...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ⚠️  未安装 Streamlit，正在安装...
    pip install -r requirements.txt
)

REM 切换到 web_app 目录
cd /d "%~dp0web_app"

echo.
echo ✅ 启动 Streamlit 应用...
echo 🌐 访问地址：http://localhost:8501
echo.
echo 💡 提示：按 Ctrl+C 停止应用
echo.

REM 启动 Streamlit
python -m streamlit run app.py
