#!/usr/bin/env python3
"""Download and clean diverse sentence prompts for tiny-voca dataset generation.

Fetches clean, conversational, human-written sentences from Tatoeba per language,
filters out symbols/digits/URLs, ensures length boundaries, and writes JSONL
files with train/validation/test splits.
"""

import argparse
import bz2
import json
from pathlib import Path
import re
import urllib.request

from tiny_voca import activate_language
from tiny_voca.text import phonemize, PHONE_SET


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Remove surrounding quotes if paired
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    # Normalize typography
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text


def is_valid_sentence(text: str) -> bool:
    if len(text) < 25 or len(text) > 140:
        return False
    # Avoid numbers/digits to prevent reading ambiguities
    if re.search(r"[0-9]", text):
        return False
    # Avoid URLs, emails, special markdown / HTML characters
    if re.search(r"http|www|\.com|@|[<>{}\[\]_~\\/|+=*#$^]", text):
        return False
    # Must end with standard sentence punctuation
    if not text.rstrip()[-1] in ".!?":
        return False
    return True


def fetch_tatoeba_sentences(lang_code: str, target_count: int = 4500):
    url = f"https://downloads.tatoeba.org/exports/per_language/{lang_code}/{lang_code}_sentences.tsv.bz2"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    sentences = []
    seen = set()

    print(f"Downloading {lang_code} corpus from Tatoeba...")
    with urllib.request.urlopen(req, timeout=45) as resp:
        decompressor = bz2.BZ2Decompressor()
        buffer = b""
        while len(sentences) < target_count:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            buffer += decompressor.decompress(chunk)
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for line in lines:
                parts = line.decode("utf-8", errors="ignore").split("\t")
                if len(parts) >= 3:
                    text = clean_text(parts[2])
                    if is_valid_sentence(text):
                        norm = text.lower()
                        if norm not in seen:
                            seen.add(norm)
                            sentences.append(text)
                            if len(sentences) >= target_count:
                                break
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", "--lang", choices=("nl", "en-gb", "en-us"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file (e.g. data/prompts_nl.jsonl)")
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=200)
    args = parser.parse_args()

    total_target = args.train_count + args.val_count + args.test_count
    tatoeba_code = "nld" if args.language == "nl" else "eng"

    sentences = fetch_tatoeba_sentences(tatoeba_code, target_count=total_target + 200)
    if len(sentences) < total_target:
        print(f"Warning: Only fetched {len(sentences)} sentences, requested {total_target}")

    activate_language(args.language)
    print(f"Filtering and verifying phoneme coverage for {args.language}...")

    valid_rows = []
    covered_phones = set()
    for idx, text in enumerate(sentences):
        try:
            p = phonemize(text)
            if len(p.ids) > 180 or len(p.ids) < 6:
                continue
            if p.unknown:
                continue
            covered_phones.update(s for s in p.symbols if s in PHONE_SET)
            valid_rows.append(text)
            if len(valid_rows) >= total_target:
                break
        except Exception:
            continue

    print(f"Valid sentences selected: {len(valid_rows)}")
    print(f"Phoneme inventory coverage: {len(covered_phones)}/{len(PHONE_SET)} phones")

    # Split into train / validation / test
    train_texts = valid_rows[:args.train_count]
    val_texts = valid_rows[args.train_count:args.train_count + args.val_count]
    test_texts = valid_rows[args.train_count + args.val_count:args.train_count + args.val_count + args.test_count]

    records = []
    for split_name, subset in (("train", train_texts), ("validation", val_texts), ("test", test_texts)):
        for idx, text in enumerate(subset):
            records.append({
                "id": f"{args.language}_{split_name}_{idx:05d}",
                "text": text,
                "split": split_name,
                "character_count": len(text),
                "language": args.language,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"Wrote {len(records)} prompts to {args.output} ({len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test)")


if __name__ == "__main__":
    main()
