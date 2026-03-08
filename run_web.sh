#!/bin/bash
# 机器学习 Web UI 启动脚本

echo "🚀 启动机器学习 Web UI..."
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 Python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  未安装 Streamlit，正在安装..."
    pip3 install -r requirements.txt
fi

# 切换到 web_app 目录
cd "$(dirname "$0")/web_app"

echo ""
echo "✅ 启动 Streamlit 应用..."
echo "🌐 访问地址：http://localhost:8501"
echo ""
echo "💡 提示：按 Ctrl+C 停止应用"
echo ""

# 启动 Streamlit
python3 -m streamlit run app.py --server.headless false
