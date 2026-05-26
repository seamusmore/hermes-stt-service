#!/bin/bash
# STT 服务代码提交脚本
# 用法：./commit.sh "提交说明"

set -e

cd /mnt/stt-service

# 检查是否有修改
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ 没有需要提交的修改"
    exit 0
fi

# 显示修改内容
echo "=== 待提交的修改 ==="
git status --short

# 提交
if [ -n "$1" ]; then
    echo ""
    echo "=== 提交代码 ==="
    git add -A
    git commit -m "$1"
    echo "✅ 提交成功！"
else
    echo "❌ 错误：请提供提交说明"
    echo "用法：./commit.sh \"提交说明\""
    exit 1
fi

# 显示提交历史
echo ""
echo "=== 最近提交 ==="
git log --oneline -3
