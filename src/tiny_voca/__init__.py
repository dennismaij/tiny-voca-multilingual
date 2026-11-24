"""tiny-voca: a small two-voice TTS trained from scratch.

The speech model loads no pretrained weights. A pretrained phoneme recogniser
is used only by the annotation step, and an ASR model only by evaluation.
"""

from .config import Config

try:
    from .model import TinyVoca
    from .runtime import TinyVocaPipeline, get_pipeline
except ImportError:
    TinyVoca = None
    TinyVocaPipeline = None
    get_pipeline = None

from .text import (
    SUPPORTED_LANGUAGES,
    activate_language,
    active_language,
    set_language,
)

__all__ = [
    "Config",
    "TinyVoca",
    "TinyVocaPipeline",
    "get_pipeline",
    "activate_language",
    "active_language",
    "set_language",
    "SUPPORTED_LANGUAGES",
    "register_word_override",
    "register_alias",
    "load_pronunciation_dictionary",
    "clear_custom_overrides",
    "preprocess_text_for_pronunciation",
]
