# STT 语音转写服务

为 Hermes 提供本地语音转文字 HTTP API，支持 SenseVoice（中文优化 + 情感检测）和 Whisper（多语言）。

## 快速启动

```bash
# systemd（生产）
sudo systemctl restart stt-service

# 手动启动
cd /mnt/stt-service
source venv/bin/activate
pip install -r requirements.txt
STT_ENGINE=sensevoice uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
```

首次启动 sensevoice 引擎时自动从 modelscope 下载模型（~900MB，2-3 分钟），之后本地缓存命中秒级启动。

## API

| 端点 | 说明 |
|------|------|
| `GET /health` | 健康检查（含 `idle_seconds` 空闲时间） |
| `GET /models` | 可用模型 |
| `GET /engines` | 可用引擎 |
| `POST /transcribe` | 转录音频 |
| `GET /cache/status` | 缓存状态 |
| `GET /cache/models` | 已下载模型列表 |
| `POST /cache/download` | 下载模型 |
| `DELETE /cache/models/{name}` | 删除模型 |

## 引擎

| 引擎 | 后端 | 内存 | 缓存路径 |
|------|------|------|----------|
| `sensevoice` | funasr + torch FP32 + mmap | RSS ~530MB | `models/sensevoice/` |
| `whisper` | faster-whisper | ~370MB（base） | `models/whisper/` |

sensevoice 通过 mmap 懒加载，物理内存占用可被内核回收。空闲超时后自动卸载模型页。

切换引擎：改 systemd `Environment=STT_ENGINE=` 后重启。

## 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STT_ENGINE` | `sensevoice` | 引擎选择（sensevoice / whisper） |
| `STT_LANGUAGE` | `zh` | 默认语言 |
| `STT_IDLE_TIMEOUT` | `600` | 空闲超时秒数，超时卸载模型释放内存 |
| `STT_MODEL_DIR` | `/mnt/stt-service/models` | 模型缓存根目录 |
| `MAX_AUDIO_SIZE_MB` | `25` | 最大文件大小 |
| `STT_SERVICE_WORKERS` | `1` | uvicorn worker 数 |

## 新机器部署

```bash
git clone https://github.com/seamusmore/hermes-stt-service.git /mnt/stt-service
cd /mnt/stt-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 安装 systemd 服务
sudo cp stt-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stt-service
```

sensevoice 模型首次启动自动下载，whisper 模型通过 `POST /cache/download` 手动下载后自动切换。

## 维护

```bash
# 查看日志
sudo journalctl -u stt-service -f

# 查看缓存
curl http://localhost:8001/cache/models

# 空闲状态（idle_seconds 超时后自动卸载）
curl http://localhost:8001/health
```
