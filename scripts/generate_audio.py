"""Generate multi-voice recordings with a local teacher TTS (Piper or Qwen3-TTS).

Supports:
- Apple Silicon M4 Max acceleration (CPU / MPS)
- Multiple languages: Dutch (nl), British English (en-gb), and American English (en-us)
- Teacher TTS engines: Piper (ultra-fast ONNX) and Qwen3-TTS
- Multilingual Whisper screening (wer <= 0.15)
- Peak normalisation to 0.95 and 16-bit 24kHz FLAC output
- Parquet and JSONL manifest output matching tiny_voca.prepare schema
"""

import argparse
import concurrent.futures
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import wave

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

TARGET_SAMPLE_RATE = 24000

PIPER_DEFAULT_VOICES = {
    "nl": {
        "F2": ("rhasspy/piper-voices", "nl/nl_NL/mls/medium/nl_NL-mls-medium.onnx"),
        "M2": ("rhasspy/piper-voices", "nl/nl_NL/pim/medium/nl_NL-pim-medium.onnx"),
    },
    "en-gb": {
        "F2": ("rhasspy/piper-voices", "en/en_GB/alba/medium/en_GB-alba-medium.onnx"),
        "M2": ("rhasspy/piper-voices", "en/en_GB/alan/medium/en_GB-alan-medium.onnx"),
    },
    "en-us": {
        "F2": ("rhasspy/piper-voices", "en/en_US/lessac/medium/en_US-lessac-medium.onnx"),
        "M2": ("rhasspy/piper-voices", "en/en_US/ryan/medium/en_US-ryan-medium.onnx"),
    },
}


def normalize_text_for_wer(text, language="en"):
    """Language-sensitive text normalization for ASR WER checking."""
    text = text.lower()
    text = re.sub(r"[\.,\?!:;\"\(\)—\-_]", " ", text)
    if language.startswith("en"):
        contractions = {"can't": "cannot", "won't": "will not", "n't": " not", "'re": " are", "'s": " is", "'d": " would", "'ll": " will", "'ve": " have"}
        for k, v in contractions.items():
            text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_piper_voice(repo_id, model_subpath):
    from huggingface_hub import hf_hub_download
    import piper
    model_path = hf_hub_download(repo_id, model_subpath)
    config_path = hf_hub_download(repo_id, model_subpath + ".json")
    return piper.PiperVoice.load(model_path, config_path=config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripts", type=Path, required=True, help="Input scripts JSONL file")
    parser.add_argument("--root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--language", default="nl", choices=["nl", "en-gb", "en-us"])
    parser.add_argument("--engine", default="piper", choices=["piper", "qwen"])
    parser.add_argument("--voices", type=Path, help="Optional voices.json config")
    parser.add_argument("--references", type=Path, help="Directory with voice cloning references (for Qwen)")
    parser.add_argument("--filter-asr", action="store_true", help="Run Whisper ASR quality filter (default: disabled for Piper)")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size (small, medium, large-v3-turbo)")
    parser.add_argument("--max-wer", type=float, default=0.15, help="Maximum allowed Word Error Rate (default: 0.15)")
    parser.add_argument("--limit", type=int, help="Only process the first N scripts")
    parser.add_argument("--export-parquet", action="store_true", help="Also export Parquet shards for tiny_voca.prepare")
    parser.add_argument("--deadline-file", type=Path)
    parser.add_argument("--deadline-margin", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()

    audio_dir = args.root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.root / "state.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS clips(id TEXT PRIMARY KEY, status TEXT, attempts INTEGER, payload TEXT)")
    db.commit()

    rows = [json.loads(line) for line in args.scripts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if "character_count" in rows[0]:
        rows = sorted(rows, key=lambda r: (r.get("character_count", 0), r.get("id", "")))
    if args.limit:
        rows = rows[:args.limit]

    voice_ids = ("F2", "M2")
    jobs = [{"id": f"{row['id']}_{voice}", "voice": voice, "row": row} for row in rows for voice in voice_ids]
    started = time.time()

    def out_of_time():
        if (args.root / "STOP").exists():
            return True
        if not args.deadline_file:
            return False
        try:
            return time.time() >= json.loads(args.deadline_file.read_text())["deadline"] - args.deadline_margin
        except (OSError, ValueError, KeyError):
            return False

    def report(stage):
        counts = dict(db.execute("SELECT status, count(*) FROM clips GROUP BY status").fetchall())
        hours = sum(json.loads(p)["duration_seconds"] for (p,) in db.execute("SELECT payload FROM clips WHERE status='accepted'")) / 3600
        value = {
            "stage": stage,
            "target_clips": len(jobs),
            "counts": counts,
            "accepted_audio_hours": round(hours, 3),
            "elapsed_seconds": round(time.time() - started),
            "updated": time.time(),
        }
        temp_progress = args.root / "progress.json.tmp"
        temp_progress.write_text(json.dumps(value, indent=2))
        os.replace(temp_progress, args.root / "progress.json")
        print(json.dumps(value), flush=True)

    # Initialize Teacher Model
    piper_voices = {}
    qwen_model, qwen_prompts = None, {}

    if args.engine == "piper":
        lang_defaults = PIPER_DEFAULT_VOICES.get(args.language, PIPER_DEFAULT_VOICES["en-us"])
        print(f"Loading Piper teacher voices for language '{args.language}'...")
        for v in voice_ids:
            repo, subpath = lang_defaults[v]
            piper_voices[v] = load_piper_voice(repo, subpath)
            print(f"  Loaded {v} from {subpath}")
    elif args.engine == "qwen":
        device = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading Qwen3-TTS on {device}...")
        from qwen_tts import Qwen3TTSModel
        qwen_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=device, dtype=torch.bfloat16)
        if args.voices and args.references:
            refs = json.loads(args.voices.read_text())
            for v in voice_ids:
                qwen_prompts[v] = qwen_model.create_voice_clone_prompt(
                    ref_audio=str(args.references / f"{v}.wav"),
                    ref_text=refs[v]["text"],
                    x_vector_only_mode=False,
                )[0]

    # Initialize Multilingual Whisper only if requested
    asr = None
    if args.filter_asr:
        from faster_whisper import WhisperModel
        whisper_device = "cpu"
        print(f"Loading multilingual Whisper ('{args.whisper_model}') on {whisper_device}...")
        asr = WhisperModel(args.whisper_model, device=whisper_device, compute_type="int8", cpu_threads=4)
        asr_lang = "nl" if args.language == "nl" else "en"

    def synthesize_clip(job, seed):
        text = job["row"]["text"]
        voice = job["voice"]
        if args.engine == "piper":
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                piper_voices[voice].synthesize_wav(text, wav_file)
            buf.seek(0)
            wave_data, sample_rate = sf.read(buf, dtype="float32")
        else:
            waves, sample_rate = qwen_model.generate_voice_clone(
                text=[text], language=["Dutch" if args.language == "nl" else "English"],
                voice_clone_prompt=[qwen_prompts[voice]], max_new_tokens=600
            )
            wave_data = np.asarray(waves[0], dtype=np.float32)

        # Resample to 24000 Hz if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            wave_data = resample_poly(wave_data, TARGET_SAMPLE_RATE, sample_rate)
        return wave_data, TARGET_SAMPLE_RATE

    def process_job(job, attempt, seed):
        start_time = time.time()
        wave_data, rate = synthesize_clip(job, seed)
        gen_duration = time.time() - start_time
        duration = len(wave_data) / rate
        peak = float(np.max(np.abs(wave_data))) if len(wave_data) else 0.0

        reasons = []
        if not np.isfinite(wave_data).all() or peak < 1e-5:
            reasons.append("invalid_or_silent_audio")
        if duration < 0.5 or duration > 35:
            reasons.append("duration")
        word_count = len(job["row"]["text"].split())
        if duration > 0 and not 0.5 <= word_count / duration <= 7.0:
            reasons.append("speaking_rate")

        transcript, error = "", 0.0
        if not reasons and asr is not None:
            from jiwer import wer
            w16 = resample_poly(wave_data.astype("float32"), 16000, rate)
            segments, _ = asr.transcribe(w16, language=asr_lang, beam_size=1, condition_on_previous_text=False)
            transcript = " ".join(s.text.strip() for s in segments)
            ref_norm = normalize_text_for_wer(job["row"]["text"], args.language)
            hyp_norm = normalize_text_for_wer(transcript, args.language) or "<empty>"
            error = wer(ref_norm, hyp_norm)
            if error > args.max_wer:
                reasons.append(f"asr_mismatch (WER {error:.2f} > {args.max_wer})")

        out_flac = audio_dir / f"{job['id']}.flac"
        if np.isfinite(wave_data).all() and peak > 0:
            normalized_wave = wave_data * (0.95 / peak)
            sf.write(out_flac, normalized_wave, rate, subtype="PCM_16")

        payload = {
            "id": job["id"],
            "text_id": str(job["row"]["id"]),
            "voice_id": job["voice"],
            "text": job["row"]["text"],
            "split": job["row"].get("split", "train"),
            "audio": f"audio/{job['id']}.flac",
            "duration_seconds": round(duration, 3),
            "generation_seconds": round(gen_duration, 3),
            "asr_transcript": transcript,
            "normalized_wer": round(float(error), 4),
            "review_reasons": reasons,
            "teacher_engine": args.engine,
            "language": args.language,
        }
        status = "review" if reasons else "accepted"
        db.execute(
            "INSERT INTO clips(id, status, attempts, payload) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, attempts=excluded.attempts, payload=excluded.payload",
            (job["id"], status, attempt, json.dumps(payload, ensure_ascii=False)),
        )
        db.commit()

    print(f"Starting generation of {len(jobs)} total clips for {len(rows)} texts...")
    stopped = False
    for attempt in (1, 2):
        existing = {uid: (status, n) for uid, status, n in db.execute("SELECT id, status, attempts FROM clips")}
        pending = [j for j in jobs if (attempt == 1 and j["id"] not in existing)
                   or (attempt == 2 and j["id"] in existing and existing[j["id"]][0] == "review" and existing[j["id"]][1] < 2)]
        print(f"Attempt {attempt}: {len(pending)} clips pending...")
        for i, job in enumerate(pending):
            if out_of_time():
                stopped = True
                break
            seed = args.seed + attempt * 10000 + i
            process_job(job, attempt, seed)
            if (i + 1) % 50 == 0 or (i + 1) == len(pending):
                report("generating")
        if stopped:
            break

    accepted = [json.loads(p) for (p,) in db.execute("SELECT payload FROM clips WHERE status='accepted' ORDER BY id")]
    review = [json.loads(p) for (p,) in db.execute("SELECT payload FROM clips WHERE status!='accepted' ORDER BY id")]

    (args.root / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted), encoding="utf-8"
    )
    (args.root / "review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review), encoding="utf-8"
    )

    for split_name in ("train", "validation", "test"):
        split_rows = [r for r in accepted if r.get("split") == split_name]
        if split_rows:
            (args.root / f"{split_name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split_rows), encoding="utf-8"
            )

    if args.export_parquet and accepted:
        import pyarrow as pa
        import pyarrow.parquet as pq
        parquet_rows = []
        for r in accepted:
            flac_path = args.root / r["audio"]
            parquet_rows.append({
                "id": r["id"],
                "text": r["text"],
                "voice_id": r["voice_id"],
                "split": r["split"],
                "audio": {"bytes": flac_path.read_bytes(), "path": r["audio"]},
            })
        pq.write_table(pa.Table.from_pylist(parquet_rows), args.root / "dataset.parquet")
        for split_name in ("train", "validation", "test"):
            subset = [r for r in parquet_rows if r["split"] == split_name]
            if subset:
                pq.write_table(pa.Table.from_pylist(subset), args.root / f"{split_name}.parquet")
        print(f"Exported {len(parquet_rows)} clips to {args.root / 'dataset.parquet'}")

    report("paused" if stopped else "complete")
    print(f"Done! {len(accepted)} clips accepted, {len(review)} sent to review.")


if __name__ == "__main__":
    main()
