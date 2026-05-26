#!/bin/bash
# 公共 STT 服务启动脚本

set -e

SERVICE_DIR="/mnt/stt-service"
VENV="/home/admin/.hermes/hermes-agent/venv"
PID_FILE="$SERVICE_DIR/stt-service.pid"
LOG_FILE="$SERVICE_DIR/stt-service.log"

cd "$SERVICE_DIR"

# 激活虚拟环境
source "$VENV/bin/activate"

# 检查依赖
echo "检查依赖..."
pip install -q -r requirements.txt

# 启动服务
echo "启动 STT 服务..."
echo "引擎：${STT_ENGINE:-whisper}"
echo "模型：${STT_MODEL:-default}"
echo "日志文件：$LOG_FILE"

# 后台运行
nohup python -m uvicorn app:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

sleep 2

# 检查是否启动成功
if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
    echo "✅ STT 服务已启动 (PID: $(cat $PID_FILE))"
    echo "访问：http://localhost:8001/docs"
    echo "健康检查：curl http://localhost:8001/health"
else
    echo "❌ 服务启动失败，查看日志：$LOG_FILE"
    exit 1
fi
