"""
SenseVoice Q8 引擎实现（sherpa-onnx + INT8 量化 ONNX）

model.int8.onnx 为 INT8 量化权重（~228MB），通过 onnxruntime 推理。
Monkey patch onnxruntime 的 SessionOptions.enable_mmap，让模型文件以 mmap
方式映射（不进堆），效果等同于旧引擎对 funasr 做的 torch.load(mmap=True)。
"""

from __future__ import annotations
import time
import re
import gc
import logging
from pathlib import Path
from typing import Optional

from . import BaseEngine, TranscribeResult, EngineInfo, register_engine

logger = logging.getLogger("stt-service")

_RICH_TAG_RE = re.compile(r"<\|[^|]*\|>")


# ---------------------------------------------------------------------------
# Monkey patch onnxruntime：强制 enable_mmap
# ---------------------------------------------------------------------------

def _patch_ort_mmap():
    """替换 onnxruntime.InferenceSession.__init__，强制启用 mmap 加载模型。"""
    import onnxruntime as ort

    _original_init = ort.InferenceSession.__init__

    def _patched_init(self, path_or_bytes, sess_options=None,
                      providers=None, provider_options=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        sess_options.enable_mmap = True
        _original_init(self, path_or_bytes,
                       sess_options=sess_options,
                       providers=providers,
                       provider_options=provider_options,
                       **kwargs)

    ort.InferenceSession.__init__ = _patched_init
    return _original_init


def _restore_ort(_original_init):
    """恢复 onnxruntime 原始 __init__"""
    import onnxruntime as ort
    ort.InferenceSession.__init__ = _original_init


@register_engine("sensevoice-q8")
class SenseVoiceQ8Engine(BaseEngine):
    """SenseVoice-Small Q8 引擎 (sherpa-onnx, INT8)"""

    MODEL_DIR = "sensevoice-q8"

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="sensevoice-q8",
            display_name="SenseVoice-Small Q8 (sherpa-onnx INT8)",
            models=["sensevoice-q8"],
            default_model="sensevoice-q8",
            supports_language_param=True,
            supports_streaming=False,
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        import sherpa_onnx

        model_dir = self.cache_dir / self.MODEL_DIR
        model_path = str(model_dir / "model.int8.onnx")
        tokens_path = str(model_dir / "tokens.txt")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please download the Q8 model first."
            )

        logger.info("[sensevoice-q8] Loading from %s", model_path)

        _orig_ort = _patch_ort_mmap()
        try:
            self._model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_path,
                tokens=tokens_path,
                num_threads=2,
                use_itn=True,
                language="auto",
            )
        finally:
            _restore_ort(_orig_ort)

        self._model_name = "sensevoice-q8"
        logger.info("[sensevoice-q8] Model ready (INT8, ~228MB)")

    def load_model(self, model_name: str) -> None:
        self._ensure_model_loaded()

    def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        import soundfile as sf

        start = time.time()

        # sherpa-onnx SenseVoice 使用 auto 自动检测语言
        samples, sr = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples[:, 0]  # 取第一声道

        stream = self._model.create_stream()
        stream.accept_waveform(sr, samples)
        self._model.decode_stream(stream)

        processing_ms = int((time.time() - start) * 1000)

        raw_text = stream.result.text

        # 提取情感和事件标签
        emotion = None
        event = None
        emo_match = re.search(r"<\|EMO_(\w+)\|>", raw_text)
        if emo_match:
            emotion = emo_match.group(1).lower()
        event_tags = re.findall(r"<\|((?!EMO_)[A-Z_]+)\|>", raw_text)
        if event_tags:
            event = ",".join(event_tags).lower()

        # 清理标签得到纯文本
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
            logger.info("[sensevoice-q8] Model unloaded")
