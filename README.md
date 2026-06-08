# STT 语音转写服务

为 Hermes 提供语音转文字 HTTP API。

## 快速启动

```bash
# systemd（生产）
sudo systemctl restart stt-service

# 手动启动
cd /mnt/stt-service
source /home/admin/.hermes/hermes-agent/venv/bin/activate
STT_ENGINE=sensevoice uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
```

## API

| 端点 | 说明 |
|------|------|
| `GET /health` | 健康检查 |
| `GET /models` | 可用模型 |
| `GET /engines` | 可用引擎 |
| `POST /transcribe` | 转录音频 |
| `GET /cache/status` | 缓存状态 |

## 引擎

| 引擎 | 后端 | 缓存路径 |
|------|------|----------|
| `sensevoice` | funasr + torch FP32 | `~/.hermes/sensevoice_cache/` |
| `whisper` | faster-whisper | `~/.hermes/whisper_cache/` |

切换引擎：改 systemd `Environment=STT_ENGINE=` 后重启。

## 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STT_ENGINE` | `sensevoice` | 引擎选择 |
| `STT_LANGUAGE` | `zh` | 默认语言 |
| `MAX_AUDIO_SIZE_MB` | `25` | 最大文件大小 |
