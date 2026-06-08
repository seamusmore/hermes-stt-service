"""
STT 引擎抽象层

提供统一的 BaseEngine 接口，不同 STT 后端（faster-whisper、funasr/SenseVoice）
实现各自的 Engine 子类，app.py 通过环境变量 STT_ENGINE 切换。

新增引擎只需：
1. 在 engines/ 下新建文件，继承 BaseEngine
2. 在 _IMPORT_MAP 中注册
3. 设置 STT_ENGINE=你的引擎名
"""

from __future__ import annotations
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("stt-service")

# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class TranscribeResult:
    """统一转录结果"""
    text: str
    language: str = ""
    duration_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    # SenseVoice 附加信息
    emotion: Optional[str] = None
    event: Optional[str] = None


@dataclass
class EngineInfo:
    """引擎描述信息"""
    name: str
    display_name: str
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    supports_language_param: bool = True
    supports_streaming: bool = False


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseEngine(ABC):
    """STT 引擎基类"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._model = None
        self._model_name: Optional[str] = None

    @abstractmethod
    def info(self) -> EngineInfo:
        """返回引擎描述"""
        ...

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """加载模型（单例，重复调用同模型名跳过）"""
        ...

    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            language: 语言代码，None=自动检测
        Returns:
            TranscribeResult
        """
        ...

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def list_available_models(self) -> list[str]:
        """列出该引擎支持的模型名"""
        return self.info().models

    def health_detail(self) -> dict:
        """健康检查细节"""
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
    """引擎类装饰器，注册到全局表"""
    def wrapper(cls: type[BaseEngine]):
        _ENGINE_CLASSES[name] = cls
        return cls
    return wrapper


def get_engine_class(name: str) -> type[BaseEngine]:
    """按名称获取引擎类"""
    if name not in _ENGINE_CLASSES:
        # 延迟导入，触发注册
        _lazy_import_engines()
    if name not in _ENGINE_CLASSES:
        raise ValueError(
            f"Unknown engine: {name}. Available: {list(_ENGINE_CLASSES.keys())}"
        )
    return _ENGINE_CLASSES[name]


def list_engines() -> list[str]:
    """列出所有已注册引擎名"""
    _lazy_import_engines()
    return list(_ENGINE_CLASSES.keys())


def _lazy_import_engines():
    """延迟导入当前需要的引擎模块，触发 @register_engine 装饰器"""
    import os
    engine = os.environ.get("STT_ENGINE", "sensevoice")
    _IMPORT_MAP = {
        "whisper": ".whisper_engine",
        "sensevoice": ".sensevoice_torch",
        "sensevoice-q8": ".sensevoice_q8",
    }
    # 只导入目标引擎
    mod = _IMPORT_MAP.get(engine)
    if mod:
        import importlib
        importlib.import_module(mod, package="engines")
    else:
        # fallback: 导入所有
        from . import whisper_engine  # noqa: F401
        from . import sensevoice_torch  # noqa: F401
        from . import sensevoice_q8  # noqa: F401
