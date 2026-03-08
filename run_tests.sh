#!/bin/bash

# =============================================================================
# 运行测试脚本
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Machine Learning Project - Tests   ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# 检查虚拟环境
if [ -d ".venv" ]; then
    echo -e "${YELLOW}激活虚拟环境...${NC}"
    source .venv/bin/activate
fi

# 安装测试依赖
echo -e "${YELLOW}检查测试依赖...${NC}"
pip install -q pytest pytest-cov pytest-asyncio

# 运行测试
echo ""
echo -e "${YELLOW}运行测试...${NC}"
echo ""

pytest tests/ \
    --cov=model_code \
    --cov=web_app \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-fail-under=80 \
    -v \
    --tb=short \
    "$@"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  ✅ 所有测试通过！                  ${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "查看 HTML 覆盖率报告：${YELLOW}open htmlcov/index.html${NC}"
else
    echo ""
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}  ❌ 测试失败！                       ${NC}"
    echo -e "${RED}======================================${NC}"
    exit 1
fi
