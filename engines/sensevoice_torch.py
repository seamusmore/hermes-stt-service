"""
SenseVoice 引擎实现（funasr + torch FP16）

model.pt 已替换为 FP16 权重（447MB），funasr 直接加载即可。
切换精度：替换 model.pt → model_fp32_backup.pt / model_int8.pt → model.pt
"""

from __future__ import annotations
import time
import re
import gc
import logging
from pathlib import Path
from typing import Optional

import soundfile as sf

from . import BaseEngine, TranscribeResult, EngineInfo, register_engine

logger = logging.getLogger("stt-service")

_RICH_TAG_RE = re.compile(r"<\|[^|]*\|>")


@register_engine("sensevoice-torch")
class SenseVoiceTorchEngine(BaseEngine):
    """SenseVoice-Small 引擎 (funasr, model.pt 即最终精度)"""

    MODEL_DIR = "sensevoice_torch"

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="sensevoice-torch",
            display_name="SenseVoice-Small (funasr+torch)",
            models=["sensevoice-torch"],
            default_model="sensevoice-torch",
            supports_language_param=False,
            supports_streaming=False,
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        from funasr import AutoModel

        model_dir = str(self.cache_dir / self.MODEL_DIR)
        logger.info("[sensevoice-torch] Loading from %s", model_dir)

        self._model = AutoModel(
            model=model_dir,
            disable_update=True,
            device="cpu",
            ncpu=2,
        )
        self._model_name = "sensevoice-torch"
        logger.info("[sensevoice-torch] Model ready")

    def load_model(self, model_name: str) -> None:
        self._ensure_model_loaded()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        result = self._model.generate(
            input=audio_path,
            language=language or "auto",
            use_itn=True,
        )
        processing_ms = int((time.time() - start) * 1000)

        if not result:
            return TranscribeResult(
                text="", language=language or "auto",
                processing_time_ms=processing_ms,
            )

        item = result[0]
        raw_text = item.get("text", "")

        emotion = None
        event = None
        emo_match = re.search(r"<\|EMO_(\w+)\|>", raw_text)
        if emo_match:
            emotion = emo_match.group(1).lower()
        event_tags = re.findall(r"<\|((?!EMO_)[A-Z_]+)\|>", raw_text)
        if event_tags:
            event = ",".join(event_tags).lower()

        text = _RICH_TAG_RE.sub("", raw_text).strip()

        return TranscribeResult(
            text=text,
            language=language or "auto",
            processing_time_ms=processing_ms,
            emotion=emotion,
            event=event,
        )

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            gc.collect()
            logger.info("[sensevoice-torch] Model unloaded")
