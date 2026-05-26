# 公共 STT 服务

为 Hermes 和 OpenClaw 提供统一的语音转文字 HTTP API 服务。

## 🚀 快速启动

### 方式一：手动启动（开发/测试）

```bash
cd /mnt/stt-service
source /home/admin/.hermes/hermes-agent/venv/bin/activate
./start.sh
```

### 方式二：systemd 服务（生产环境）

```bash
# 安装服务
sudo ln -s /mnt/stt-service/stt-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable stt-service
sudo systemctl start stt-service

# 查看状态
sudo systemctl status stt-service

# 查看日志
sudo journalctl -u stt-service -f

# 重启服务
sudo systemctl restart stt-service
```

## 📡 API 端点

服务地址：`http://localhost:8001`

### 1. 健康检查

```bash
curl http://localhost:8001/health
```

返回：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "base",
  "version": "1.0.0"
}
```

### 2. 转录音频

```bash
curl -X POST http://localhost:8001/transcribe \
  -F "file=@audio.ogg" \
  -F "language=zh"
```

参数：
- `file`: 音频文件（必填）
- `language`: 语言代码（可选，默认自动检测）
- `model`: 模型名称（可选，默认服务配置）

返回：
```json
{
  "success": true,
  "text": "你好，这是转录的文本",
  "provider": "local",
  "model": "base",
  "language": "zh",
  "duration_ms": 5000,
  "processing_time_ms": 3200
}
```

### 3. 查看可用模型

```bash
curl http://localhost:8001/models
```

返回：
```json
{
  "available": ["tiny", "base", "small", "medium", "large"],
  "current": "base"
}
```

## ⚙️ 配置

通过环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STT_SERVICE_PORT` | 8001 | 服务端口 |
| `STT_SERVICE_HOST` | 0.0.0.0 | 监听地址 |
| `STT_SERVICE_WORKERS` | 2 | Worker 数量 |
| `STT_MODEL` | base | 默认模型 (tiny/base/small/medium/large) |
| `STT_LANGUAGE` | zh | 默认语言 |
| `MAX_AUDIO_SIZE_MB` | 25 | 最大文件大小 (MB) |
| `STT_REQUEST_TIMEOUT` | 60 | 请求超时 (秒) |

### 示例配置

```bash
# 在 ~/.bashrc 或 systemd 服务中设置
export STT_MODEL=base
export STT_LANGUAGE=zh
export STT_SERVICE_WORKERS=4
```

## 🔌 客户端使用示例

### Python 客户端

```python
import requests

def transcribe_audio(file_path: str, server_url: str = "http://localhost:8001"):
    """转录音频文件"""
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{server_url}/transcribe",
            files={"file": f},
            data={"language": "zh"}
        )
    
    result = response.json()
    if result["success"]:
        return result["text"]
    else:
        raise Exception(result["error"])

# 使用
text = transcribe_audio("voice_message.ogg")
print(f"识别结果：{text}")
```

### Hermes Agent 集成

在 Hermes 中调用 STT 服务：

```python
import requests
from pathlib import Path

def stt_transcribe(audio_path: str) -> dict:
    """调用公共 STT 服务"""
    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                "http://localhost:8001/transcribe",
                files={"file": f},
                timeout=60
            )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### OpenClaw 集成

在 OpenClaw 的 `openclaw.json` 中添加配置：

```json
{
  "stt_service": {
    "enabled": true,
    "url": "http://localhost:8001",
    "model": "base",
    "language": "zh",
    "timeout": 60
  }
}
```

## 📊 性能参考

| 模型 | 大小 | 速度 (1 分钟音频) | 中文准确率 | 推荐场景 |
|------|------|------------------|-----------|---------|
| tiny | 39M | ~5 秒 | 85%+ | 日常对话 |
| base | 74M | ~10 秒 | 90%+ | ✅ 推荐 |
| small | 244M | ~30 秒 | 93%+ | 高精度 |
| medium | 769M | ~60 秒 | 95%+ | 专业场景 |
| large | 1550M | ~90 秒 | 96%+ | 最高精度 |

## 🔧 故障排查

### 服务无法启动

```bash
# 查看日志
cat /mnt/stt-service/stt-service.log

# 检查端口占用
lsof -i :8001

# 手动测试
cd /mnt/stt-service
source /home/admin/.hermes/hermes-agent/venv/bin/activate
python app.py
```

### 模型加载失败

```bash
# 检查模型缓存
ls -la ~/.hermes/whisper_cache/

# 重新下载模型
export HF_ENDPOINT=https://hf-mirror.com
cd ~/.hermes/whisper_cache
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Systran/faster-whisper-base', cache_dir='.')
"
```

### 并发性能问题

```bash
# 增加 worker 数量
export STT_SERVICE_WORKERS=4

# 或者调整 systemd 配置
sudo systemctl edit stt-service
# 添加：Environment="STT_SERVICE_WORKERS=4"
sudo systemctl restart stt-service
```

## 📝 版本历史

- **v1.0.0** (2026-04-18): 初始版本
  - FastAPI HTTP 服务
  - faster-whisper 本地转录
  - 支持多模型切换
  - 并发处理支持

## 🎯 下一步

- [ ] 添加 API 认证
- [ ] 添加请求限流
- [ ] 添加批量转录接口
- [ ] 添加 WebSocket 流式转录
- [ ] 添加音频预处理（降噪、增益）
