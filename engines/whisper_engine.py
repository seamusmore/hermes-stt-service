"""
faster-whisper 引擎实现

基于 Systran/faster-whisper 的 Whisper 模型推理，
支持 tiny/base/small/medium/large 五种规格。
"""

from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Optional

from . import BaseEngine, TranscribeResult, EngineInfo, register_engine

logger = logging.getLogger("stt-service")


@register_engine("whisper")
class WhisperEngine(BaseEngine):
    """faster-whisper 引擎"""

    # 模型名称 -> HuggingFace repo
    MODEL_MAP = {
        "tiny":   "Systran/faster-whisper-tiny",
        "base":   "Systran/faster-whisper-base",
        "small":  "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large":  "Systran/faster-whisper-large-v3",
    }

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="whisper",
            display_name="faster-whisper (Whisper)",
            models=list(self.MODEL_MAP.keys()),
            default_model="base",
            supports_language_param=True,
            supports_streaming=False,
        )

    # 旧缓存目录（兼容）
    LEGACY_CACHE_DIR = Path.home() / ".hermes" / "whisper_cache"

    def _resolve_download_root(self) -> str:
        """确定模型下载/查找目录：优先旧缓存，其次新缓存"""
        for d in [self.LEGACY_CACHE_DIR, self.cache_dir]:
            if d.exists():
                return str(d)
        # 都不存在，用旧路径（首次下载时自动创建）
        return str(self.LEGACY_CACHE_DIR)

    def load_model(self, model_name: str) -> None:
        if self._model is not None and self._model_name == model_name:
            return

        logger.info(f"[whisper] Loading model: {model_name}")
        start = time.time()

        from faster_whisper import WhisperModel

        # 设置 HuggingFace 镜像和离线模式
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_OFFLINE"] = "1"

        full_name = self.MODEL_MAP.get(model_name, model_name)
        download_root = self._resolve_download_root()
        Path(download_root).mkdir(parents=True, exist_ok=True)

        self._model = WhisperModel(
            full_name,
            device="cpu",
            compute_type="int8",
            download_root=download_root,
            local_files_only=True,
        )
        self._model_name = model_name

        elapsed = time.time() - start
        logger.info(f"[whisper] Model loaded in {elapsed:.2f}s")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
        start = time.time()

        if self._model is None:
            raise RuntimeError("Model not loaded")

        kwargs = {"beam_size": 5}
        if language and language != "auto":
            kwargs["language"] = language

        segments, info = self._model.transcribe(audio_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments)

        processing_ms = int((time.time() - start) * 1000)
        duration_ms = int(info.duration * 1000) if hasattr(info, "duration") else None

        return TranscribeResult(
            text=text,
            language=language or getattr(info, "language", "auto"),
            duration_ms=duration_ms,
            processing_time_ms=processing_ms,
        )
