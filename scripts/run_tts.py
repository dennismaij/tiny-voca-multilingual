#!/usr/bin/env python3
"""Run tiny-voca TTS with modular language activation.

Allows activating ONLY the desired language (Dutch, British English, or American English)
and generating speech to an audio file using either PyTorch or ONNX Runtime.

Examples:
    # Synthesize Dutch speech with ONNX model:
    python scripts/run_tts.py --lang nl --onnx exported/dutch.onnx --text "Hallo allemaal!" --output out_nl.wav

    # Synthesize British English speech with PyTorch checkpoint on MPS:
    python scripts/run_tts.py --lang en-gb --checkpoint checkpoints/gb/model.pt --text "Good afternoon." --output out_gb.wav

    # Inspect active language phonemization without audio synthesis:
    python scripts/run_tts.py --lang nl --text "Het weer is prachtig." --dry-run
"""

import argparse
from pathlib import Path
import sys

import tiny_voca
from tiny_voca.text import phonemize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", "--lang", choices=tiny_voca.SUPPORTED_LANGUAGES, required=True,
                        help="Language to activate: 'nl' (Dutch), 'en-gb' (British English), 'en-us' (American English)")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--output", "-o", type=Path, help="Path for output audio file (e.g. speech.wav)")
    parser.add_argument("--checkpoint", type=Path, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", type=Path, help="Path to exported ONNX model (.onnx)")
    parser.add_argument("--voice", choices=("F2", "M2"), default="F2", help="Voice ID (F2 or M2)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (1.0 = normal)")
    parser.add_argument("--pitch-shift", type=float, default=0.0, help="Pitch shift in log F0 (e.g. 0.058 = +1 semitone)")
    parser.add_argument("--pitch-range", type=float, default=1.0, help="Pitch variation scale")
    parser.add_argument("--device", default="mps", help="Device for PyTorch inference ('mps', 'cuda', 'cpu')")
    parser.add_argument("--dry-run", action="store_true", help="Print activated language symbols and phonemes without loading neural model")
    args = parser.parse_args()

    # 1. Activate ONLY the selected language
    tiny_voca.activate_language(args.language)
    print(f"[tiny-voca] Activated language: '{tiny_voca.active_language()}'")

    # 2. Phonemize text in active language
    p = phonemize(args.text)
    print(f"[tiny-voca] Text: \"{args.text}\"")
    print(f"[tiny-voca] Phonemes: {' '.join(p.symbols)}")
    print(f"[tiny-voca] Token count: {len(p.ids)}")

    if args.dry_run:
        print("[tiny-voca] Dry-run complete. Language activated successfully.")
        return

    if not args.output:
        parser.error("--output is required unless running with --dry-run")

    if not args.checkpoint and not args.onnx:
        parser.error("Must specify either --checkpoint (.pt) or --onnx (.onnx) for speech synthesis")

    # 3. Create pipeline & synthesize
    pipeline = tiny_voca.get_pipeline(
        language=args.language,
        checkpoint=args.checkpoint,
        onnx_model=args.onnx,
        device=args.device,
    )
    out_file = pipeline.speak_to_file(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        speed=args.speed,
        pitch_shift=args.pitch_shift,
        pitch_range=args.pitch_range,
    )
    print(f"[tiny-voca] Audio successfully written to: {out_file}")


if __name__ == "__main__":
    main()
