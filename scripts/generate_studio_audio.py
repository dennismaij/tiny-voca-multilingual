"""Generate studio-grade multi-voice dataset using broadcast neural TTS (edge-tts).

Produces 24 kHz 16-bit FLAC recordings matching tiny_voca schema with:
- Studio-quality natural human vocal timbre (zero VITS vocoder buzz)
- 0.95 peak normalisation
- Async concurrent generation with exponential backoff retry and 2-pass reconciliation
- SQLite state tracking for seamless resumption
- Partitioned JSONL manifests (train, validation, test)
"""

import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import time

import edge_tts
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 24000

STUDIO_VOICES = {
    "nl": {
        "F2": "nl-NL-ColetteNeural",
        "M2": "nl-NL-MaartenNeural",
    },
    "en-gb": {
        "F2": "en-GB-LibbyNeural",
        "M2": "en-GB-RyanNeural",
    },
    "en-us": {
        "F2": "en-US-JennyNeural",
        "M2": "en-US-GuyNeural",
    },
}


async def fetch_audio_with_retry(text: str, voice: str, max_retries: int = 5) -> tuple[np.ndarray, int]:
    """Fetch audio bytes from Edge TTS with retry and jittered exponential backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            comm = edge_tts.Communicate(text, voice)
            buf = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            if not buf:
                raise ValueError("Empty audio buffer received from edge-tts")
            wave_data, sample_rate = sf.read(io.BytesIO(buf), dtype="float32")
            if sample_rate != TARGET_SAMPLE_RATE:
                wave_data = resample_poly(wave_data, TARGET_SAMPLE_RATE, sample_rate)
            return wave_data, TARGET_SAMPLE_RATE
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep_sec = 0.5 + random.uniform(0.2, 0.8) * (2 ** attempt)
                await asyncio.sleep(sleep_sec)
    raise last_err or RuntimeError("Failed to fetch audio after max retries")


async def main_async(args):
    audio_dir = args.root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.root / "state.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS clips(id TEXT PRIMARY KEY, status TEXT, attempts INTEGER, payload TEXT)"
    )
    db.commit()

    rows = [json.loads(line) for line in args.scripts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    voice_ids = ("F2", "M2")
    voice_map = STUDIO_VOICES[args.language]
    jobs = [{"id": f"{row['id']}_{voice}", "voice": voice, "row": row} for row in rows for voice in voice_ids]

    sem = asyncio.Semaphore(args.concurrency)

    async def process_job(job, attempt_num):
        voice = job["voice"]
        voice_name = voice_map[voice]
        text = job["row"]["text"]

        async with sem:
            # Add small random jitter so requests don't hit the socket at the exact same microsecond
            await asyncio.sleep(random.uniform(0.05, 0.25))
            t_start = time.time()
            try:
                wave_data, rate = await fetch_audio_with_retry(text, voice_name)
                gen_duration = time.time() - t_start
                duration = len(wave_data) / rate
                peak = float(np.max(np.abs(wave_data))) if len(wave_data) else 0.0

                reasons = []
                if not np.isfinite(wave_data).all() or peak < 1e-5:
                    reasons.append("invalid_or_silent_audio")
                if duration < 0.4 or duration > 35:
                    reasons.append("duration")

                out_flac = audio_dir / f"{job['id']}.flac"
                if np.isfinite(wave_data).all() and peak > 0:
                    normalized = wave_data * (0.95 / peak)
                    sf.write(out_flac, normalized, rate, subtype="PCM_16")

                status = "review" if reasons else "accepted"
                payload = {
                    "id": job["id"],
                    "text_id": str(job["row"]["id"]),
                    "voice_id": voice,
                    "text": text,
                    "split": job["row"].get("split", "train"),
                    "audio": f"audio/{job['id']}.flac",
                    "duration_seconds": round(duration, 3),
                    "generation_seconds": round(gen_duration, 3),
                    "review_reasons": reasons,
                    "teacher_engine": "edge_studio",
                    "teacher_voice": voice_name,
                    "language": args.language,
                }
                db.execute(
                    "INSERT INTO clips(id, status, attempts, payload) VALUES(?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET status=excluded.status, attempts=excluded.attempts, payload=excluded.payload",
                    (job["id"], status, attempt_num, json.dumps(payload, ensure_ascii=False)),
                )
                db.commit()
            except Exception as err:
                payload = {
                    "id": job["id"],
                    "text_id": str(job["row"]["id"]),
                    "voice_id": voice,
                    "text": text,
                    "split": job["row"].get("split", "train"),
                    "audio": "",
                    "duration_seconds": 0.0,
                    "generation_seconds": round(time.time() - t_start, 3),
                    "review_reasons": [str(err)],
                    "teacher_engine": "edge_studio",
                    "language": args.language,
                }
                db.execute(
                    "INSERT INTO clips(id, status, attempts, payload) VALUES(?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET status=excluded.status, payload=excluded.payload",
                    (job["id"], "failed", attempt_num, json.dumps(payload, ensure_ascii=False)),
                )
                db.commit()

    # Pass 1 & Pass 2
    for pass_num in (1, 2):
        existing = {uid: status for uid, status in db.execute("SELECT id, status FROM clips")}
        pending = [j for j in jobs if existing.get(j["id"]) != "accepted"]
        if not pending:
            print(f"Pass {pass_num}: All {len(jobs)} clips already accepted!")
            break
        print(f"Pass {pass_num}: Processing {len(pending)} pending clips...")
        progress_counter = 0
        t0 = time.time()

        async def monitored_job(j):
            nonlocal progress_counter
            await process_job(j, pass_num)
            progress_counter += 1
            if progress_counter % 50 == 0 or progress_counter == len(pending):
                elapsed = time.time() - t0
                speed = progress_counter / max(elapsed, 1e-3)
                remaining = (len(pending) - progress_counter) / max(speed, 1e-3)
                print(
                    f"[{progress_counter}/{len(pending)}] ({speed:.2f} clips/s, {elapsed:.1f}s elapsed, ~{remaining/60:.1f}m left) "
                    f"Latest: {j['id']}",
                    flush=True,
                )

        await asyncio.gather(*[monitored_job(j) for j in pending])

    # Write manifests
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

    hours = sum(r["duration_seconds"] for r in accepted) / 3600
    print(f"\nComplete! Accepted: {len(accepted)} clips ({hours:.2f} hours audio) | Review/Failed: {len(review)} clips.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts", type=Path, required=True, help="Input scripts JSONL file")
    parser.add_argument("--root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--language", default="nl", choices=["nl", "en-gb", "en-us"])
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent async downloads")
    parser.add_argument("--limit", type=int, help="Limit number of prompt scripts")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
