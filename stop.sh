#!/bin/bash
# 公共 STT 服务停止脚本

set -e

SERVICE_DIR="/home/admin/.hermes/services/stt-service"
PID_FILE="$SERVICE_DIR/stt-service.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止 STT 服务 (PID: $PID)..."
        kill $PID
        rm -f "$PID_FILE"
        echo "✅ 服务已停止"
    else
        echo "⚠️  服务未运行 (PID 文件存在但进程不存在)"
        rm -f "$PID_FILE"
    fi
else
    echo "⚠️  PID 文件不存在，尝试通过进程名停止..."
    pkill -f "uvicorn app:app" || echo "未找到相关进程"
fi
