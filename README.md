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

## mmap 内存管理

SenseVoice 模型文件 893MB（FP32），传统 `torch.load` 全量载入进程堆，固定占用 ~1.4GB RSS。

本项目通过 monkey-patch funasr 的加载函数，改用 `torch.load(mmap=True)` + `param.data` 直接赋值：

| 方式 | RSS | 数据位置 | 内存紧张时 |
|------|-----|---------|-----------|
| 全量加载 | ~1360 MB | 进程堆（AnonPages） | swap 到磁盘 |
| mmap | ~530 MB | page cache（Cached） | 内核直接丢页 |

**关键区别：** page cache 中的页是干净的（有磁盘备份），内核可在内存紧张时瞬间回收，无需 swap。下次推理时缺页中断自动按需读回，仅增加首次延迟 ~4s。

结合空闲超时机制（默认 10 分钟），空闲时自动 `unload()` 释放所有模型页，物理内存占用量显著降低。

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
