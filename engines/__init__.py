"""
STT 引擎抽象层

模板方法模式：
- _ensure_model_loaded()  基类统一入口
- _check_model_cached()   子类告知缓存是否存在
- _load_model()            子类实现具体加载
- _auto_download()         基类 ModelScope 默认，whisper 覆盖走 HF
"""

from __future__ import annotations
import os
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("stt-service")


@dataclass
class TranscribeResult:
    text: str
    language: str = ""
    duration_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    emotion: Optional[str] = None
    event: Optional[str] = None


@dataclass
class EngineInfo:
    name: str
    display_name: str
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    supports_language_param: bool = True
    supports_streaming: bool = False
    modelscope_repo: str = ""


class BaseEngine(ABC):
    """STT 引擎基类 — 模板方法模式

    _ensure_model_loaded() 定义统一流程：
      已加载 → 跳过
      缓存检查 (_check_model_cached)
      缺失 → 自动下载 (_auto_download)
      加载 (_load_model)
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._model = None
        self._model_name: Optional[str] = None

    # ---- 抽象方法 ----

    @abstractmethod
    def info(self) -> EngineInfo:
        ...

    @abstractmethod
    def _check_model_cached(self) -> bool:
        ...

    @abstractmethod
    def _load_model(self) -> None:
        ...

    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
        ...

    # ---- 模板方法 ----

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if not self._check_model_cached():
            self._auto_download()
        self._load_model()

    def load_model(self, model_name: str) -> None:
        if self._model is not None and self._model_name == model_name:
            return
        self._model_name = model_name
        self._ensure_model_loaded()

    # ---- 可覆盖：自动下载 ----

    def _auto_download(self) -> None:
        """从 ModelScope 下载并展平（whisper 覆盖走 HF）"""
        import shutil
        import tempfile
        from modelscope.hub.snapshot_download import snapshot_download

        repo_id = self.info().modelscope_repo
        if not repo_id:
            raise FileNotFoundError(
                f"Model not cached and no ModelScope repo for {self.info().name}."
            )

        model_dir = self.cache_dir / self.info().name
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] Auto-downloading from ModelScope: %s", self.info().name, repo_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = snapshot_download(repo_id, cache_dir=tmp_dir)
            src = Path(downloaded)
            if src.exists():
                for f in src.iterdir():
                    dst = model_dir / f.name
                    if not dst.exists():
                        if f.is_dir():
                            shutil.copytree(str(f), str(dst))
                        else:
                            shutil.copy2(str(f), str(dst))
                        logger.info("[%s] Copied: %s", self.info().name, f.name)
            else:
                for root, dirs, files in os.walk(tmp_dir):
                    for f in files:
                        src_file = Path(root) / f
                        if not (model_dir / f).exists():
                            shutil.copy2(str(src_file), str(model_dir / f))
                            logger.info("[%s] Copied: %s", self.info().name, f)

        if not self._check_model_cached():
            raise FileNotFoundError(f"Auto-download completed but model not found for {self.info().name}")
        logger.info("[%s] Auto-download complete", self.info().name)

    # ---- 属性 ----

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def list_available_models(self) -> list[str]:
        return self.info().models

    def health_detail(self) -> dict:
        return {
            "engine": self.info().name,
            "model_loaded": self.model_loaded,
            "model_name": self._model_name,
        }


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

_ENGINE_CLASSES: dict[str, type[BaseEngine]] = {}


def register_engine(name: str):
    def wrapper(cls: type[BaseEngine]):
        _ENGINE_CLASSES[name] = cls
        return cls
    return wrapper


def get_engine_class(name: str) -> type[BaseEngine]:
    if name not in _ENGINE_CLASSES:
        _lazy_import_engines()
    if name not in _ENGINE_CLASSES:
        raise ValueError(f"Unknown engine: {name}. Available: {list(_ENGINE_CLASSES.keys())}")
    return _ENGINE_CLASSES[name]


def list_engines() -> list[str]:
    _lazy_import_engines()
    return list(_ENGINE_CLASSES.keys())


def _lazy_import_engines():
    import os as _os
    engine = _os.environ.get("STT_ENGINE", "sensevoice-q8")
    _IMPORT_MAP = {
        "whisper": ".whisper_engine",
        "sensevoice": ".sensevoice_torch",
        "sensevoice-q8": ".sensevoice_q8",
    }
    mod = _IMPORT_MAP.get(engine)
    if mod:
        import importlib
        importlib.import_module(mod, package="engines")
    else:
        from . import whisper_engine  # noqa: F401
        from . import sensevoice_torch  # noqa: F401
        from . import sensevoice_q8  # noqa: F401
