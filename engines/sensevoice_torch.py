"""
SenseVoice 引擎实现（funasr + torch FP32）

model.pt 为 FP32 权重（893MB），通过 mmap 懒加载，RSS ~530MB。
内核可在内存紧张时回收模型页（干净页无需 swap），空闲超时后自动 unload 释放。
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


def _patch_funasr_loader():
    """替换 funasr 的 load_pretrained_model，用 mmap 避免全量加载到堆。"""
    import torch
    import funasr.auto.auto_model as auto_mod

    _original = auto_mod.load_pretrained_model

    def _mmap_loader(path, model, ignore_init_mismatch=True,
                     map_location="cpu", oss_bucket=None,
                     scope_map=None, excludes=None, **kwargs):
        # 用 mmap 加载，避免 893MB 全进堆
        checkpoint = torch.load(path, map_location=map_location, mmap=True)
        src_state = checkpoint.get("state_dict",
                    checkpoint.get("model_state_dict",
                    checkpoint.get("model", checkpoint)))

        if isinstance(scope_map, str):
            scope_map = scope_map.split(",")
        scope_map = (scope_map or []) + ["module.", "None"]

        if excludes is not None and isinstance(excludes, str):
            excludes = excludes.split(",")

        matched = 0
        for name, param in model.named_parameters():
            if excludes:
                skip = any(name.startswith(ex) for ex in excludes)
                if skip:
                    continue

            k_src = name
            for i in range(0, len(scope_map), 2):
                sp = scope_map[i] if scope_map[i].lower() != "none" else ""
                dp = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""
                if dp == "" and (sp + name) in src_state:
                    k_src = sp + name
                elif name.startswith(dp) and name.replace(dp, sp, 1) in src_state:
                    k_src = name.replace(dp, sp, 1)

            if k_src in src_state:
                src_tensor = src_state[k_src]
                if ignore_init_mismatch and param.shape != src_tensor.shape:
                    continue
                # 直接赋值，不复制：param.data 指向 mmap 张量的底层存储
                param.data = src_tensor
                matched += 1

        logger.info("[sensevoice] Loading ckpt: %s, matched=%d params (mmap)",
                    path, matched)

    auto_mod.load_pretrained_model = _mmap_loader
    return _original


def _restore_funasr_loader(original):
    import funasr.auto.auto_model as auto_mod
    auto_mod.load_pretrained_model = original


@register_engine("sensevoice")
class SenseVoiceEngine(BaseEngine):
    """SenseVoice-Small 引擎 (funasr+torch, mmap)"""

    MODEL_DIR = "sensevoice"

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="sensevoice",
            display_name="SenseVoice-Small (funasr+torch)",
            models=["sensevoice"],
            default_model="sensevoice",
            supports_language_param=False,
            supports_streaming=False,
            modelscope_repo="iic/SenseVoiceSmall",
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        from funasr import AutoModel

        model_dir = str(self.cache_dir / self.MODEL_DIR)
        model_pt = Path(model_dir) / "model.pt"

        if not model_pt.exists():
            self._auto_download(Path(model_dir), str(model_pt))

        logger.info("[sensevoice] Loading from %s (mmap)", model_dir)

        _orig_loader = _patch_funasr_loader()
        try:
            self._model = AutoModel(
                model=model_dir,
                disable_update=True,
                device="cpu",
                ncpu=2,
            )
        finally:
            _restore_funasr_loader(_orig_loader)

        self._model_name = "sensevoice"
        logger.info("[sensevoice] Model ready")

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
            logger.info("[sensevoice] Model unloaded")
