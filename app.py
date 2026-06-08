#!/usr/bin/env python3
"""
公共 STT HTTP 服务
为 Hermes 和 OpenClaw 提供统一的语音转文字 API

支持多引擎切换（whisper / sensevoice），通过 STT_ENGINE 环境变量选择。

启动方式：
    cd /mnt/stt-service
    source /home/admin/.hermes/hermes-agent/venv/bin/activate
    STT_ENGINE=sensevoice uvicorn app:app --host 0.0.0.0 --port 8001

API 端点：
    POST /transcribe - 转录音频文件
    GET  /health    - 健康检查
    GET  /models    - 可用模型列表
    GET  /engines   - 可用引擎列表
    GET  /cache/status  - 缓存状态
    GET  /cache/models  - 已下载模型列表
    POST /cache/download - 下载模型
    DELETE /cache/models/{model} - 删除模型
    DELETE /cache/clear - 清理所有缓存
"""

import os
import sys
import time
import threading
import logging
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# 确保项目根目录在 sys.path，支持在任何目录下启动服务
_svc_root = Path(__file__).resolve().parent
if str(_svc_root) not in sys.path:
    sys.path.insert(0, str(_svc_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stt-service")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

SERVICE_PORT = int(os.getenv("STT_SERVICE_PORT", "8001"))
SERVICE_HOST = os.getenv("STT_SERVICE_HOST", "0.0.0.0")
WORKERS = int(os.getenv("STT_SERVICE_WORKERS", "1"))

# 引擎配置：whisper | sensevoice
STT_ENGINE = os.getenv("STT_ENGINE", "sensevoice")
STT_MODEL = os.getenv("STT_MODEL", "")  # 空=引擎默认模型
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "zh")
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE_MB", "25")) * 1024 * 1024
REQUEST_TIMEOUT = int(os.getenv("STT_REQUEST_TIMEOUT", "60"))

# 空闲超时（秒）：超时后自动卸载模型，释放 page cache
IDLE_TIMEOUT = int(os.getenv("STT_IDLE_TIMEOUT", "600"))

# 模型缓存根目录（可通过 STT_MODEL_DIR 环境变量覆盖）
CACHE_DIR = Path(os.getenv("STT_MODEL_DIR", "/mnt/stt-service/models"))
WHISPER_CACHE_DIR = CACHE_DIR / "whisper"

SUPPORTED_FORMATS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
    ".wav", ".webm", ".ogg", ".aac", ".flac"
}

# ---------------------------------------------------------------------------
# 引擎加载
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    """获取/初始化当前引擎（单例）"""
    global _engine
    if _engine is not None:
        return _engine

    from engines import get_engine_class, list_engines

    logger.info(f"Initializing engine: {STT_ENGINE}")
    engine_cls = get_engine_class(STT_ENGINE)
    _engine = engine_cls(cache_dir=CACHE_DIR)
    return _engine


def _resolve_model() -> str:
    """确定使用的模型名"""
    if STT_MODEL:
        return STT_MODEL
    engine = _get_engine()
    return engine.info().default_model


# ---------------------------------------------------------------------------
# 空闲超时 —— 超时后自动卸载模型释放 page cache
# ---------------------------------------------------------------------------

_last_activity = time.time()
_idle_lock = threading.Lock()
_idle_monitor_running = False


def _touch_activity():
    """记录一次活动，重置空闲计时器"""
    global _last_activity
    with _idle_lock:
        _last_activity = time.time()


def _get_idle_seconds() -> float:
    with _idle_lock:
        return time.time() - _last_activity


def _idle_monitor():
    """后台线程：每 30s 检查空闲状态，超时则卸载模型"""
    global _idle_monitor_running
    _idle_monitor_running = True
    logger.info(f"Idle monitor started (timeout={IDLE_TIMEOUT}s)")
    while _idle_monitor_running:
        time.sleep(30)
        if not _idle_monitor_running:
            break
        idle_secs = _get_idle_seconds()
        if idle_secs >= IDLE_TIMEOUT:
            engine = _get_engine()
            if engine.model_loaded:
                logger.info(f"Idle {idle_secs:.0f}s >= {IDLE_TIMEOUT}s, unloading model...")
                engine.unload()
                logger.info("Model unloaded (idle timeout)")


def _stop_idle_monitor():
    global _idle_monitor_running
    _idle_monitor_running = False


# ---------------------------------------------------------------------------
# 缓存管理（兼容旧 whisper 缓存 + 新引擎缓存）
# ---------------------------------------------------------------------------

# Whisper 模型仓库映射（兼容旧缓存）
MODEL_REPO_MAP = {
    "tiny":   "Systran/faster-whisper-tiny",
    "base":   "Systran/faster-whisper-base",
    "small":  "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large":  "Systran/faster-whisper-large-v3",
}

MODEL_SIZE_MAP = {
    "tiny": 100, "base": 142, "small": 500,
    "medium": 1500, "large": 3000,
}


def get_cache_status() -> dict:
    """获取缓存状态"""
    all_dirs = []
    total_size = 0

    for cache_root in [CACHE_DIR, WHISPER_CACHE_DIR]:
        if not cache_root.exists():
            continue
        for f in cache_root.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

    total_size_mb = round(total_size / (1024 * 1024), 2)

    try:
        import shutil
        available = shutil.disk_usage(CACHE_DIR if CACHE_DIR.exists() else Path.home())
        available_space_mb = round(available.free / (1024 * 1024), 2)
    except Exception:
        available_space_mb = 0

    # 探测已下载的 whisper 模型
    downloaded_models = []
    for model_name, repo_id in MODEL_REPO_MAP.items():
        for cache_root in [CACHE_DIR, WHISPER_CACHE_DIR]:
            repo_dir_name = f"models--{repo_id.replace('/', '--')}"
            if (cache_root / repo_dir_name).exists():
                downloaded_models.append(model_name)
                break

    # 探测 sensevoice 缓存
    sv_cache = CACHE_DIR / "sensevoice"
    if sv_cache.exists():
        downloaded_models.append("sensevoice")

    return {
        "cache_dir": str(CACHE_DIR),
        "total_size_mb": total_size_mb,
        "models_downloaded": downloaded_models,
        "available_space_mb": available_space_mb,
    }


def list_downloaded_models() -> list:
    downloaded = []
    for model_name, repo_id in MODEL_REPO_MAP.items():
        repo_dir_name = f"models--{repo_id.replace('/', '--')}"
        found = False
        for cache_root in [CACHE_DIR, WHISPER_CACHE_DIR]:
            if (cache_root / repo_dir_name).exists():
                downloaded.append({
                    "name": model_name,
                    "repo_id": repo_id,
                    "size_mb": MODEL_SIZE_MAP.get(model_name, 0),
                    "downloaded": True,
                })
                found = True
                break
        if not found:
            downloaded.append({
                "name": model_name,
                "repo_id": repo_id,
                "size_mb": MODEL_SIZE_MAP.get(model_name, 0),
                "downloaded": False,
            })

    # sensevoice
    sv_cache = CACHE_DIR / "sensevoice"
    downloaded.append({
        "name": "sensevoice",
        "repo_id": "iic/SenseVoiceSmall (funasr+torch)",
        "size_mb": 893,
        "downloaded": sv_cache.exists(),
    })
    return downloaded


def download_model(model_name: str) -> dict:
    """下载模型"""
    # Whisper 模型
    if model_name in MODEL_REPO_MAP:
        repo_id = MODEL_REPO_MAP[model_name]
        try:
            old_offline = os.environ.get("HF_HUB_OFFLINE", "0")
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            from huggingface_hub import snapshot_download

            logger.info(f"Downloading model {model_name} ({repo_id})...")
            cache_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(WHISPER_CACHE_DIR),
                local_dir_use_symlinks=False,
            )
            os.environ["HF_HUB_OFFLINE"] = old_offline

            logger.info(f"✅ Model {model_name} downloaded to {cache_path}")
            return {
                "success": True,
                "model": model_name,
                "repo_id": repo_id,
                "cache_path": cache_path,
                "size_mb": MODEL_SIZE_MAP.get(model_name, 0),
            }
        except Exception as e:
            os.environ["HF_HUB_OFFLINE"] = old_offline
            logger.error(f"Failed to download model {model_name}: {e}", exc_info=True)
            return {"success": False, "error": f"Download failed: {e}"}

    # SenseVoice-torch 通过 funasr 首次加载时自动下载
    if model_name.startswith("sensevoice"):
        return {
            "success": False,
            "error": "SenseVoice-torch model is auto-downloaded on first load via funasr. "
                     "Just start the service with STT_ENGINE=sensevoice.",
        }

    return {"success": False, "error": f"Unknown model: {model_name}"}


def delete_model(model_name: str) -> dict:
    """删除模型"""
    import shutil

    if model_name in MODEL_REPO_MAP:
        repo_id = MODEL_REPO_MAP[model_name]
        repo_dir_name = f"models--{repo_id.replace('/', '--')}"
        for cache_root in [CACHE_DIR, WHISPER_CACHE_DIR]:
            repo_dir = cache_root / repo_dir_name
            if repo_dir.exists():
                size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
                size_mb = round(size / (1024 * 1024), 2)
                shutil.rmtree(repo_dir)
                logger.info(f"✅ Model {model_name} deleted ({size_mb}MB freed)")
                return {"success": True, "model": model_name, "freed_mb": size_mb}
        return {"success": False, "error": f"Model {model_name} not found in cache"}

    if model_name.startswith("sensevoice"):
        sv_cache = CACHE_DIR / "sensevoice"
        if sv_cache.exists():
            size = sum(f.stat().st_size for f in sv_cache.rglob("*") if f.is_file())
            size_mb = round(size / (1024 * 1024), 2)
            shutil.rmtree(sv_cache)
            return {"success": True, "model": model_name, "freed_mb": size_mb}
        return {"success": False, "error": f"Model {model_name} not found in cache"}

    return {"success": False, "error": f"Unknown model: {model_name}"}


def clear_all_cache() -> dict:
    """清理所有缓存（保留旧 whisper_cache 不动）"""
    import shutil

    if not CACHE_DIR.exists():
        return {"success": False, "error": "Cache directory does not exist"}

    try:
        size = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
        size_mb = round(size / (1024 * 1024), 2)
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Cache cleared ({size_mb}MB freed)")
        return {"success": True, "freed_mb": size_mb, "cache_dir": str(CACHE_DIR)}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        return {"success": False, "error": f"Clear failed: {e}"}


# ---------------------------------------------------------------------------
# FastAPI 生命周期
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("STT Service starting...")
    logger.info(f"Engine: {STT_ENGINE}, Language: {STT_LANGUAGE}")
    logger.info(f"Cache dir: {CACHE_DIR}")
    logger.info(f"Idle timeout: {IDLE_TIMEOUT}s")

    try:
        engine = _get_engine()
        model = _resolve_model()
        logger.info(f"Loading model: {model}")
        engine.load_model(model)
        _touch_activity()
        logger.info(f"✅ STT Service ready (engine={STT_ENGINE}, model={model})")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

    # 启动空闲超时监控线程
    monitor_thread = threading.Thread(target=_idle_monitor, daemon=True, name="idle-monitor")
    monitor_thread.start()

    yield

    _stop_idle_monitor()
    logger.info("STT Service shutting down...")


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(
    title="公共 STT 服务",
    description="语音转文字 HTTP API - 多引擎支持 (whisper / sensevoice)",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class TranscribeResponse(BaseModel):
    success: bool
    text: str
    provider: str = "local"
    model: str
    engine: str
    language: str
    duration_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    emotion: Optional[str] = None  # SenseVoice 附加
    event: Optional[str] = None    # SenseVoice 附加
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_loaded: bool
    model_name: Optional[str]
    idle_seconds: float = 0
    version: str


class ModelsResponse(BaseModel):
    available: list
    current: str


class EnginesResponse(BaseModel):
    available: list
    current: str


class CacheStatusResponse(BaseModel):
    cache_dir: str
    total_size_mb: float
    models_downloaded: list
    available_space_mb: float


class ModelInfo(BaseModel):
    name: str
    repo_id: str
    size_mb: float
    downloaded: bool


class DownloadRequest(BaseModel):
    model: str


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    engine = _get_engine()
    return HealthResponse(
        status="healthy" if engine.model_loaded else "degraded",
        engine=STT_ENGINE,
        model_loaded=engine.model_loaded,
        model_name=engine.model_name,
        idle_seconds=_get_idle_seconds(),
        version="2.0.0",
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    engine = _get_engine()
    return ModelsResponse(
        available=engine.list_available_models(),
        current=engine.model_name or _resolve_model(),
    )


@app.get("/engines", response_model=EnginesResponse)
async def list_engines():
    from engines import list_engines as _list_engines
    return EnginesResponse(
        available=_list_engines(),
        current=STT_ENGINE,
    )


@app.get("/cache/status")
async def get_cache_status_endpoint():
    return get_cache_status()


@app.get("/cache/models")
async def list_cached_models():
    return {"models": list_downloaded_models()}


@app.post("/cache/download")
async def download_model_endpoint(request: DownloadRequest):
    result = download_model(request.model)
    if result["success"]:
        return JSONResponse(status_code=200, content=result)
    else:
        return JSONResponse(status_code=400, content=result)


@app.delete("/cache/models/{model_name}")
async def delete_model_endpoint(model_name: str):
    result = delete_model(model_name)
    if result["success"]:
        return JSONResponse(status_code=200, content=result)
    else:
        return JSONResponse(status_code=400, content=result)


@app.delete("/cache/clear")
async def clear_cache_endpoint():
    result = clear_all_cache()
    if result["success"]:
        return JSONResponse(status_code=200, content=result)
    else:
        return JSONResponse(status_code=400, content=result)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件"),
    language: Optional[str] = Form(None, description="语言代码 (默认自动检测)"),
    model: Optional[str] = Form(None, description="模型名称"),
):
    """
    转录音频文件

    - **file**: 音频文件 (支持 mp3, wav, ogg, m4a, webm, flac 等)
    - **language**: 语言代码 (zh=中文, en=英语，默认自动检测)
    - **model**: 模型名称 (可选，默认使用服务配置的模型)

    返回转录文本和元数据，SenseVoice 引擎额外返回情感识别结果
    """
    start_time = time.time()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    try:
        content = await file.read()
        if len(content) > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {len(content) / (1024*1024):.1f}MB (max {MAX_AUDIO_SIZE / (1024*1024):.0f}MB)",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        # 获取引擎
        engine = _get_engine()

        # 重置空闲计时器
        _touch_activity()

        # 确定模型（如果请求指定了不同模型，需要重新加载）
        use_model = model if model else _resolve_model()
        try:
            engine.load_model(use_model)
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return TranscribeResponse(
                success=False, text="", model=use_model,
                engine=STT_ENGINE, language=language or STT_LANGUAGE,
                error=f"Model loading failed: {e}",
            )

        # 执行转录
        logger.info(f"Transcribing: {file.filename} ({len(content)} bytes) [engine={STT_ENGINE}, model={use_model}]")

        use_language = language if language else STT_LANGUAGE
        result = engine.transcribe(temp_path, language=use_language)

        logger.info(
            f"Transcription complete: {file.filename} -> {len(result.text)} chars "
            f"(processing: {result.processing_time_ms}ms)"
            + (f", emotion: {result.emotion}" if result.emotion else "")
        )

        return TranscribeResponse(
            success=True,
            text=result.text,
            provider="local",
            model=use_model,
            engine=STT_ENGINE,
            language=result.language or use_language,
            duration_ms=result.duration_ms,
            processing_time_ms=result.processing_time_ms,
            emotion=result.emotion,
            event=result.event,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return TranscribeResponse(
            success=False, text="",
            model=model or _resolve_model(),
            engine=STT_ENGINE,
            language=language or STT_LANGUAGE,
            error=f"Transcription failed: {e}",
        )
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("公共 STT 服务 v2.0")
    logger.info("=" * 60)
    logger.info(f"Host: {SERVICE_HOST}")
    logger.info(f"Port: {SERVICE_PORT}")
    logger.info(f"Engine: {STT_ENGINE}")
    logger.info(f"Model: {STT_MODEL or '(default)'}")
    logger.info(f"Language: {STT_LANGUAGE}")
    logger.info(f"Max file size: {MAX_AUDIO_SIZE / (1024*1024):.0f}MB")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        workers=WORKERS,
    )
