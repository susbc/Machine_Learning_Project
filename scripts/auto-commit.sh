#!/bin/bash
# 自动上传代码到 GitHub
# 用法：./scripts/auto-commit.sh "提交信息"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查 Git 状态
echo "📊 检查 Git 状态..."
git status --short

# 添加所有更改
echo "➕ 添加所有更改..."
git add -A

# 检查是否有更改
if git diff --staged --quiet; then
    echo "✅ 没有更改需要提交"
    exit 0
fi

# 提交信息
COMMIT_MSG="${1:-auto: 自动提交代码更新 $(date '+%Y-%m-%d %H:%M:%S')}"

# 提交
echo "💾 提交更改..."
git commit -m "$COMMIT_MSG"

# 推送
echo "🚀 推送到 GitHub..."
git push origin main

echo "✅ 完成！"
