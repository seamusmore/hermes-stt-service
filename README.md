# STT 语音转写服务

为 Hermes / OpenClaw 提供本地语音转文字 HTTP API，支持三引擎：SenseVoice-Q8（推荐，228MB INT8）、SenseVoice-torch（FP32 + mmap）、faster-whisper（多语言）。

## 快速启动

```bash
# systemd（生产）
sudo systemctl restart stt-service

# 手动启动
cd /mnt/stt-service
source /home/admin/.hermes/hermes-agent/venv/bin/activate
STT_ENGINE=sensevoice-q8 uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
```

## API

| 端点 | 说明 |
|------|------|
| `GET /health` | 健康检查（含 `idle_seconds` 空闲时间、`status` healthy/degraded） |
| `GET /models` | 可用模型列表 + 当前模型 |
| `GET /engines` | 可用引擎列表 |
| `POST /transcribe` | 转录音频，表单参数 `file` + `language`（可选） |
| `GET /cache/status` | 缓存大小与磁盘剩余空间 |
| `GET /cache/models` | 已下载模型列表 |
| `POST /cache/download` | 下载 whisper 模型（sensevoice 首次启动自动下载） |
| `DELETE /cache/models/{name}` | 删除指定模型缓存 |
| `DELETE /cache/clear` | 清空所有缓存 |

## 引擎

| 引擎 | 后端 | 模型 | 进程 RSS | 模型文件 |
|------|------|------|:---:|:---:|
| `sensevoice-q8` ★ | sherpa-onnx + INT8 + mmap | SenseVoice-Small 8bit | ~490 MB | 228 MB |
| `sensevoice` | funasr + torch FP32 + mmap | iic/SenseVoiceSmall | ~530 MB | 893 MB |
| `whisper` | faster-whisper | base/small/medium/large | ~370 MB（base） | 142–3000 MB |

默认引擎 `sensevoice-q8`（环境变量 `STT_ENGINE=sensevoice-q8`），中文优化 + 语种/情感/事件检测。

**冷启动自动下载**：三个引擎首次加载时如果模型文件不在本地，自动从 ModelScope / HF 镜像拉取，无需手动下载。

## 代码结构

```
/mnt/stt-service/
├── app.py                       # FastAPI 应用（启动、API 端点、缓存管理）
├── engines/
│   ├── __init__.py              # BaseEngine 模板方法 + 注册表
│   │   ├── _ensure_model_loaded()  ← 唯一加载入口，基类统一
│   │   ├── _check_model_cached()   ← 子类告知缓存是否存在
│   │   ├── _load_model()           ← 子类实现具体加载
│   │   └── _auto_download()        ← 基类 ModelScope / whisper 覆盖走 HF
│   ├── sensevoice_q8.py        # 两个方法：check + load
│   ├── sensevoice_torch.py     # 两个方法：check + load
│   └── whisper_engine.py       # 两个方法 + _auto_download 覆盖
├── models/                      # 模型缓存（STT_MODEL_DIR）
│   ├── sensevoice-q8/          # model.int8.onnx 228MB + tokens.txt
│   └── sensevoice/             # model.pt 893MB + config 等
├── stt-service.service          # systemd unit（生产用）
├── requirements.txt
├── test_stt.py
├── manage_cache.py
└── README.md
```

新增引擎仅需继承 `BaseEngine`，实现 `_check_model_cached()` 和 `_load_model()` 两个方法，其余流程由基类模板方法自动管理。

## mmap 内存管理

SenseVoice 两个引擎都通过 monkey-patch 注入 mmap 加载：

| 引擎 | mmap 策略 | 效果 |
|------|----------|------|
| `sensevoice` | patch `funasr.load_pretrained_model` → `torch.load(mmap=True)` | RSS 从 1.4GB 降至 530MB |
| `sensevoice-q8` | patch `sherpa_onnx.OfflineRecognizer.from_sense_voice` → `onnxruntime.SessionOptions.enable_mmap=True` | RSS 从 694MB 降至 490MB |

mmap 页由内核 page cache 管理，内存紧张时可直接丢弃（无需 swap），下次推理缺页中断按需读回。

空闲超时（`STT_IDLE_TIMEOUT`，默认 600s）后自动 `unload()` 释放模型页。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STT_ENGINE` | `sensevoice-q8` | 引擎选择（sensevoice-q8 / sensevoice / whisper） |
| `STT_LANGUAGE` | `zh` | 默认语言 |
| `STT_IDLE_TIMEOUT` | `600` | 空闲超时（秒），超时卸载模型 |
| `STT_MODEL_DIR` | `/mnt/stt-service/models` | 模型缓存根目录 |
| `STT_SERVICE_PORT` | `8001` | 监听端口 |
| `MAX_AUDIO_SIZE_MB` | `25` | 最大音频文件大小 |
| `OMP_NUM_THREADS` | `2` | ONNX 推理线程数 |

## 💡 复用 Hermes Agent 的 venv

生产环境不创建独立 venv，直接复用 Hermes Agent 的 Python 环境（funasr、torch、sherpa-onnx 等依赖均在 Hermes venv 中）：

```
Environment="PATH=/home/admin/.hermes/hermes-agent/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/admin/.hermes/hermes-agent/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
```

好处：省去维护独立 venv 的开销，依赖版本与 Hermes Agent 保持一致。

## 运维

```bash
# 状态与日志
systemctl status stt-service
sudo journalctl -u stt-service -f

# 健康检查
curl -s http://localhost:8001/health | python3 -m json.tool

# 查看当前模型
curl -s http://localhost:8001/models

# 查看缓存
curl -s http://localhost:8001/cache/status | python3 -m json.tool

# 切换引擎：改 systemd 后重启
sudo systemctl edit stt-service  # 或编辑 /etc/systemd/system/stt-service.service
sudo systemctl daemon-reload
sudo systemctl restart stt-service

# 查看进程内存
grep -E "VmRSS|RssAnon|RssFile" /proc/$(ss -tlnp | grep 8001 | grep -oP 'pid=\K\d+')/status
```

