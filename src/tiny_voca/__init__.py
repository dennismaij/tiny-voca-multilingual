"""tiny-voca: a small two-voice TTS trained from scratch.

The speech model loads no pretrained weights. A pretrained phoneme recogniser
is used only by the annotation step, and an ASR model only by evaluation.
"""

from .config import Config
from .model import TinyVoca
from .text import (
    SUPPORTED_LANGUAGES,
    activate_language,
    active_language,
    set_language,
)
from .runtime import TinyVocaPipeline, get_pipeline

__all__ = [
    "Config",
    "TinyVoca",
    "TinyVocaPipeline",
    "get_pipeline",
    "activate_language",
    "active_language",
    "set_language",
    "SUPPORTED_LANGUAGES",
]
