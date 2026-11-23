"""End-to-end smoke test for tiny-voca on Apple Silicon (MPS).

Runs the full pipeline on a mini-dataset:
1. Generates 20 synthetic sentences for 2 voices (F2, M2)
2. Extracts acoustic features (tiny_voca.prepare)
3. Performs CTC forced alignment with Wav2Vec2 (tiny_voca.annotate)
4. Trains for 5 steps on MPS (tiny_voca.train)
5. Exports to ONNX and tests ONNX Runtime inference
"""

import argparse
import io
import json
import math
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tempfile
import time
import wave

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from tiny_voca import text
from tiny_voca.config import Config
from tiny_voca.train import export_model
from tiny_voca.model import TinyVoca
from tiny_voca.prepare import prepare_split
from tiny_voca.train import TrainingSystem, optimize_step, collate, balanced_indices, BucketSampler
from tiny_voca.data import AnnotatedDataset
from tiny_voca.align import PhoneRecognizer, AlignmentSettings, align_utterance
from tiny_voca.annotate import gate

DUTCH_SMOKE_SENTENCES = [
    "Hallo wereld, dit is een test voor de Nederlandse spraaksynthese.",
    "De snelle bruine vos springt over de luie hond bij de molen.",
    "Het weer in Amsterdam is vandaag bijzonder aangenaam met veel zonneschijn.",
    "Morgen gaan we gezellig fietsen door de duinen en koffie drinken.",
    "Waarom hebben we vandaag geen verse groenten gekocht op de markt?",
    "Zij vertelden dat het onderzoek naar de fonemen succesvol was afgerond.",
    "Het Koninkrijk der Nederlanden is een soevereine staat in West-Europa.",
    "Gezelligheid kent geen tijd als we samen een stroopwafel eten.",
    "Wat kost een kaartje voor de trein naar Schiphol Airport?",
    "We hebben een klein huisje met een prachtige tuin aan de gracht.",
    "De kinderen spelen vrolijk buiten op het schoolplein.",
    "Kun je alsjeblieft het raam even dichtdoen want het tocht hier?",
    "Technologische vernieuwing helpt ons sneller te werken.",
    "De kat slaapt rustig op de vensterbank in de namiddagzon.",
    "Zij luisterden aandachtig naar het mooie concert in het theater.",
    "Lekker eten en een goed gesprek maken de avond compleet.",
    # Validation sentences (distinct texts, zero leakage)
    "Vandaag begint een nieuw hoofdstuk in onze samenwerking.",
    "De bibliotheek heeft duizenden interessante boeken over geschiedenis.",
    "Fietsen langs de kustlijn is een geweldige ervaring in de herfst.",
    "Goede communicatie is de sleutel tot een succesvol project.",
]

BRITISH_SMOKE_SENTENCES = [
    "Hello world, this is a test for British English speech synthesis.",
    "The quick brown fox jumps over the lazy dog near the old mill.",
    "The weather in London is rather splendid today with plenty of sunshine.",
    "Tomorrow we shall go for a delightful bicycle ride through the countryside.",
    "Why have we not purchased fresh vegetables from the market today?",
    "They mentioned that the phoneme research was completed with great success.",
    "The United Kingdom is an island nation situated in north-western Europe.",
    "Having a warm cup of Earl Grey tea with a biscuit is truly wonderful.",
    "How much does a train ticket to Edinburgh Waverley station cost?",
    "We have a lovely cottage with a beautiful garden by the river.",
    "The cheerful children are playing games outside in the playground.",
    "Could you please close the window slightly as there is a draft?",
    "Technological advancements allow us to work considerably faster.",
    "The cat is sleeping peacefully on the windowsill in the afternoon sun.",
    "They listened attentively to the classical concert at the hall.",
    "Splendid food and engaging conversation made the evening perfect.",
    # Validation sentences (distinct texts, zero leakage)
    "Today marks the beginning of an exciting journey for all of us.",
    "The university library contains thousands of rare and fascinating books.",
    "Walking along the coastline is a magnificent experience in autumn.",
    "Clear and honest communication remains the cornerstone of every project.",
]


def run_smoke_test(language="nl", device="mps"):
    print(f"\n==================================================================")
    print(f"  STARTING TINY-VOCA SMOKE TEST: Language='{language}', Device='{device}'")
    print(f"==================================================================")

    # 1. Set language in text frontend
    print(f"\n--- STEP 1: Configuring Language Frontend ({language}) ---")
    text.set_language(language)
    print(f"Active voice: {text.ESPEAK_VOICE}")
    print(f"Active symbol inventory: {len(text.SYMBOLS)} tokens")
    cfg = Config()
    print(f"Config initialized: vocab_size={cfg.vocab_size}, sample_rate={cfg.sample_rate}Hz")

    # 2. Select sentences & split
    sentences = DUTCH_SMOKE_SENTENCES if language == "nl" else BRITISH_SMOKE_SENTENCES
    train_texts = sentences[:16]
    val_texts = sentences[16:]

    work_dir = Path(tempfile.mkdtemp(prefix=f"tiny_voca_smoke_{language}_"))
    print(f"Working temporary directory: {work_dir}")

    try:
        # Load Piper voice for fast synthetic teacher generation
        from huggingface_hub import hf_hub_download
        import piper
        from scripts.generate_audio import PIPER_DEFAULT_VOICES

        piper_voices = {}
        defaults = PIPER_DEFAULT_VOICES[language]
        print(f"\n--- STEP 2: Generating Synthetic Audio for 2 Voices (F2, M2) ---")
        for voice_id in ("F2", "M2"):
            repo, subpath = defaults[voice_id]
            print(f"Loading Piper voice {voice_id} ({subpath})...")
            m = hf_hub_download(repo, subpath)
            c = hf_hub_download(repo, subpath + ".json")
            piper_voices[voice_id] = piper.PiperVoice.load(m, config_path=c)

        # Generate audio files
        audio_dir = work_dir / "raw_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        splits = {"train": train_texts, "validation": val_texts}
        split_records = {"train": [], "validation": []}

        for split_name, texts in splits.items():
            for idx, sent in enumerate(texts):
                for voice_id in ("F2", "M2"):
                    clip_id = f"{split_name}_{idx}_{voice_id}"
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wav_file:
                        piper_voices[voice_id].synthesize_wav(sent, wav_file)
                    buf.seek(0)
                    wave_data, sr = sf.read(buf, dtype="float32")
                    if sr != cfg.sample_rate:
                        wave_data = resample_poly(wave_data, cfg.sample_rate, sr)

                    # Normalize peak to 0.95
                    peak = float(np.max(np.abs(wave_data)))
                    if peak > 0:
                        wave_data = wave_data * (0.95 / peak)

                    out_flac = audio_dir / f"{clip_id}.flac"
                    sf.write(out_flac, wave_data, cfg.sample_rate, subtype="PCM_16")

                    record = {
                        "id": clip_id,
                        "text_id": f"{split_name}_{idx}",
                        "voice_id": voice_id,
                        "text": sent,
                        "split": split_name,
                        "audio": str(out_flac),
                    }
                    split_records[split_name].append(record)
            print(f"Generated {len(split_records[split_name])} clips for {split_name}.")

        # 3. Acoustic Feature Preparation (tiny_voca.prepare)
        print(f"\n--- STEP 3: Preparing Acoustic Features (wave, mel, pYIN pitch, energy) ---")
        features_dir = work_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "validation"):
            source_jsonl = work_dir / f"source_{split_name}.jsonl"
            source_jsonl.write_text("".join(json.dumps(r) + "\n" for r in split_records[split_name]))

            args_prep = argparse.Namespace(
                inputs=[source_jsonl],
                output=features_dir,
                split=split_name,
                text_column="text",
                speaker_column="voice_id",
                resume=False,
            )
            prepare_split(args_prep, cfg)
        print(f"Acoustic features successfully computed and cached in {features_dir}")

        # 4. Forced Alignment Annotation (tiny_voca.annotate)
        print(f"\n--- STEP 4: Forced Alignment on '{device}' with Wav2Vec2 ---")
        recognizer = PhoneRecognizer(device=device, local_files_only=False)
        settings = AlignmentSettings()
        annotations_dir = work_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "validation"):
            split_jsonl = features_dir / f"{split_name}.jsonl"
            rows = [json.loads(l) for l in split_jsonl.read_text().splitlines() if l.strip()]
            annotated_rows = []
            for row in rows:
                sample = torch.load(features_dir / row["path"], weights_only=True)
                p = text.phonemize(row["text"])
                durations, qc = align_utterance(recognizer, p, sample["wave"].numpy(), sample["prosody"][:, 1].numpy(), cfg, settings)
                annotated_rows.append({
                    "id": row["id"],
                    "speaker": row["speaker"],
                    "text_hash": row["text_hash"],
                    "tokens": list(p.ids),
                    "stress": list(p.stress),
                    "durations": durations,
                    "frames": row["frames"],
                    "path": row["path"],
                    "unknown": list(p.unknown),
                    "qc": qc,
                    "frontend": text.frontend_fingerprint(),
                    "acoustic": cfg.acoustic_fingerprint,
                    "annotation": "smoke_test_annotation_fingerprint",
                })
            out_annot = annotations_dir / f"{split_name}.jsonl"
            out_annot.write_text("".join(json.dumps(r) + "\n" for r in annotated_rows))
            # Write minimal annotation metadata
            (annotations_dir / "annotations.json").write_text(json.dumps({
                "identity": {"acoustic": cfg.acoustic_fingerprint, "frontend": text.frontend_fingerprint()},
                "fingerprint": "smoke_test_annotation_fingerprint",
                "acoustic_root": str(features_dir),
                "complete": True,
            }))
            print(f"Annotated {len(annotated_rows)} clips for {split_name}.")

        # 5. Training Step Verification on MPS
        print(f"\n--- STEP 5: Running 5 Training Steps on '{device}' ---")
        train_ds = AnnotatedDataset(annotations_dir / "train.jsonl", cfg)
        val_ds = AnnotatedDataset(annotations_dir / "validation.jsonl", cfg)

        from tiny_voca.losses import Discriminators
        from tiny_voca.train import WEIGHTS
        system = TrainingSystem(cfg).to(device)
        critics = Discriminators().to(device)

        opt = torch.optim.AdamW(system.parameters(), lr=1e-3)
        c_opt = torch.optim.AdamW(critics.parameters(), lr=1e-3)

        recipe = {
            "learning_rate": 1e-3, "warmup": 2, "lr_half_life": 1000.0, "grad_clip": 10.0, "gan_start": 999,
            "crop_samples": 2048, "crops": 1, "teacher_hold": 10, "teacher_ramp": 10, "teacher_final": 0.0,
            "weights": WEIGHTS, "gan_ratio": 0.7, "gan_ramp": 2, "feature_matching": 30.0, "gan_scale_max": 2000.0,
            "branched_from": None
        }

        system.train()
        for step in range(5):
            batch = collate([train_ds[i] for i in range(2)], device)
            metrics = optimize_step(system, critics, opt, c_opt, batch, step, recipe)
            print(f"  Step {step + 1}/5: loss={metrics['loss']:.4f}, mel_loss={metrics['mel']:.4f}, dur_loss={metrics['duration']:.4f}, grad_norm={metrics['grad_norm']:.4f}")

        # 6. ONNX Export Verification
        print(f"\n--- STEP 6: Exporting Trained Checkpoint to ONNX ---")
        inference_pt = work_dir / "inference.pt"
        export_model(system.model, inference_pt, 5, annotation="smoke_test")

        from tiny_voca.export import main as export_onnx_main
        onnx_file = work_dir / "tiny_voca_smoke.onnx"
        sys_argv_backup = sys.argv
        sys.argv = ["export", "--checkpoint", str(inference_pt), "--output", str(onnx_file)]
        export_onnx_main()
        sys.argv = sys_argv_backup
        print(f"ONNX model exported to {onnx_file} ({onnx_file.stat().st_size / 1024 / 1024:.2f} MB)")

        # Verify ONNX Runtime inference
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        test_p = text.phonemize("Dit is een snelle test.") if language == "nl" else text.phonemize("This is a quick test.")
        tokens_in = np.array([test_p.ids], dtype=np.int64)
        stress_in = np.array([test_p.stress], dtype=np.int64)
        voice_in = np.array([0], dtype=np.int64)
        speed_in = np.array([1.0], dtype=np.float32)
        pitch_shift = np.array([0.0], dtype=np.float32)
        pitch_range = np.array([1.0], dtype=np.float32)

        audio_out, dur_out = session.run(None, {
            "tokens": tokens_in,
            "stress": stress_in,
            "voice": voice_in,
            "speed": speed_in,
            "pitch_shift": pitch_shift,
            "pitch_range": pitch_range,
        })
        print(f"ONNX inference verified! Generated waveform: shape={audio_out.shape}, duration={audio_out.shape[1]/cfg.sample_rate:.2f}s")
        print(f"\n==================================================================")
        print(f"  SMOKE TEST PASSED SUCCESSFULLY FOR {language.upper()} ON {device.upper()}!")
        print(f"==================================================================\n")
        return True

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        text.set_language("en-us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="nl", choices=["nl", "en-gb", "en-us"])
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    success = run_smoke_test(language=args.language, device=args.device)
    if not success:
        sys.exit(1)
