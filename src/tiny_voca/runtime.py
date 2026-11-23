"""Multi-language runtime pipeline for tiny-voca.

Provides isolated language activation so that only the selected language (Dutch,
British English, or American English) is loaded and used for speech synthesis.
"""

from pathlib import Path
import json
import numpy as np
import soundfile as sf
import torch

from .config import Config
from .text import (
    LANGUAGE_CONFIGS,
    SUPPORTED_LANGUAGES,
    activate_language,
    active_language,
    phonemize,
)
from .infer import load_model, speak_text, limit_peak


class TinyVocaPipeline:
    """An activated language pipeline for text-to-speech synthesis."""

    def __init__(self, language: str, checkpoint: str | Path | None = None,
                 onnx_model: str | Path | None = None, device: str = "cpu"):
        language = language.lower()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")

        self.language = language
        self.device = device
        self.checkpoint_path = Path(checkpoint) if checkpoint else None
        self.onnx_path = Path(onnx_model) if onnx_model else None

        # 1. Activate ONLY the selected language
        activate_language(language)

        # 2. Load model (PyTorch or ONNX)
        self.model = None
        self.ort_session = None
        self.onnx_metadata = None
        self.sample_rate = 24000

        if self.onnx_path:
            import onnxruntime as ort
            self.ort_session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
            meta_json = self.onnx_path.with_suffix(".json")
            if meta_json.exists():
                self.onnx_metadata = json.loads(meta_json.read_text(encoding="utf-8"))
                self.sample_rate = self.onnx_metadata.get("sample_rate", 24000)
        elif self.checkpoint_path:
            self.model = load_model(self.checkpoint_path, device=self.device, check_frontend=False)
            self.sample_rate = self.model.config.sample_rate

    def synthesize(self, text: str, voice: str = "F2", speed: float = 1.0,
                   pitch_shift: float = 0.0, pitch_range: float = 1.0,
                   seed: int = 0) -> tuple[np.ndarray, int]:
        """Synthesize text in the active language to a float32 numpy audio waveform."""
        # Ensure this pipeline's language is active
        if active_language() != self.language:
            activate_language(self.language)

        if self.ort_session is not None:
            # ONNX Runtime path
            p = phonemize(text)
            tokens_in = np.array([p.ids], dtype=np.int64)
            stress_in = np.array([p.stress], dtype=np.int64)
            voices = self.onnx_metadata.get("voices", ["F2", "M2"]) if self.onnx_metadata else ["F2", "M2"]
            voice_idx = voices.index(voice) if voice in voices else 0
            voice_in = np.array([voice_idx], dtype=np.int64)
            speed_in = np.array([speed], dtype=np.float32)
            pitch_shift_in = np.array([pitch_shift], dtype=np.float32)
            pitch_range_in = np.array([pitch_range], dtype=np.float32)

            feed = {
                "tokens": tokens_in,
                "stress": stress_in,
                "voice": voice_in,
                "speed": speed_in,
                "pitch_shift": pitch_shift_in,
                "pitch_range": pitch_range_in,
            }
            outputs = self.ort_session.run(None, feed)
            audio = outputs[0][0]  # shape: [samples]
            return audio.astype(np.float32), self.sample_rate

        elif self.model is not None:
            # PyTorch path
            raw, _ = speak_text(
                self.model, text, voice, speed=speed,
                pitch_shift=pitch_shift, pitch_range=pitch_range,
                seed=seed, peak=None
            )
            wave = limit_peak(raw).numpy()
            return wave, self.sample_rate

        else:
            raise RuntimeError("No checkpoint (.pt) or ONNX model (.onnx) loaded.")

    def speak_to_file(self, text: str, output_path: str | Path, voice: str = "F2",
                      speed: float = 1.0, pitch_shift: float = 0.0,
                      pitch_range: float = 1.0, seed: int = 0):
        """Synthesize text and write directly to an audio file (WAV/FLAC)."""
        wave, sr = self.synthesize(text, voice, speed, pitch_shift, pitch_range, seed)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, wave, sr)
        return out_path


def get_pipeline(language: str, checkpoint: str | Path | None = None,
                 onnx_model: str | Path | None = None,
                 device: str = "mps" if torch.backends.mps.is_available() else "cpu") -> TinyVocaPipeline:
    """Convenience factory to activate and return a language pipeline."""
    return TinyVocaPipeline(language=language, checkpoint=checkpoint, onnx_model=onnx_model, device=device)
