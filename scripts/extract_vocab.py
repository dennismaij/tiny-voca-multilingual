"""Extract and validate IPA phoneme vocabulary from a target language corpus.

Usage:
    python scripts/extract_vocab.py --text data/corpus_nl.txt --language nl
    python scripts/extract_vocab.py --text data/corpus_en_gb.txt --language en-gb
"""

import argparse
import collections
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata

ALIGNER_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def get_aligner_vocab():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(ALIGNER_MODEL, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_phones_from_text(text_lines, language_voice):
    espeak_bin = shutil.which("espeak-ng")
    if not espeak_bin:
        raise RuntimeError("espeak-ng is not installed. Install via 'brew install espeak-ng'.")

    full_text = "\n".join(line.strip() for line in text_lines if line.strip())
    if not full_text:
        return collections.Counter()

    proc = subprocess.run(
        [espeak_bin, "-v", language_voice, "-q", "--ipa", "--sep=_", "--stdin"],
        input=full_text + "\n",
        text=True,
        capture_output=True,
        check=True,
    )

    phones = collections.Counter()
    for word in proc.stdout.split():
        for unit in word.split("_"):
            unit = unit.replace("ˈ", "").replace("ˌ", "").strip()
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                phones[unit] += 1
    return phones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True, help="Text file (.txt or .jsonl with 'text' field)")
    parser.add_argument("--language", default="nl", choices=["nl", "en-gb", "en-us"], help="Target language")
    parser.add_argument("--output", type=Path, help="Optional output JSON path for the vocabulary config")
    args = parser.parse_args()

    lines = []
    if args.text.suffix == ".jsonl":
        for line in args.text.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "text" in data:
                        lines.append(data["text"])
                    elif isinstance(data, str):
                        lines.append(data)
                except json.JSONDecodeError:
                    lines.append(line)
    else:
        lines = args.text.read_text(encoding="utf-8").splitlines()

    print(f"Loaded {len(lines)} lines from {args.text}. Phonemizing with eSpeak NG voice '{args.language}'...")
    counter = extract_phones_from_text(lines, args.language)
    print(f"Extracted {len(counter)} distinct raw phoneme units from corpus.")

    print(f"Downloading/loading aligner vocabulary from {ALIGNER_MODEL}...")
    aligner_vocab = get_aligner_vocab()

    present_in_aligner = []
    missing_from_aligner = []
    for phone, count in counter.most_common():
        if phone in aligner_vocab:
            present_in_aligner.append((phone, count))
        else:
            missing_from_aligner.append((phone, count))

    print(f"\n--- Coverage Summary for '{args.language}' ---")
    print(f"Total distinct phonemes: {len(counter)}")
    print(f"Directly recognized by forced aligner: {len(present_in_aligner)} / {len(counter)} ({len(present_in_aligner)/max(len(counter), 1):.1%})")

    if missing_from_aligner:
        print("\nPhonemes needing FALLBACK mapping for alignment:")
        for phone, count in missing_from_aligner:
            print(f"  '{phone}' (occurred {count} times)")

    # Prepare vocabulary export
    specials = ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"]
    punctuation = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"]
    all_phones = sorted(counter.keys())
    symbols = specials + punctuation + all_phones

    config_data = {
        "language": args.language,
        "vocab_size": len(symbols),
        "specials": specials,
        "punctuation": punctuation,
        "phones": all_phones,
        "symbols": symbols,
        "aligner_missing": [p for p, _ in missing_from_aligner],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSaved vocabulary config to {args.output}")


if __name__ == "__main__":
    main()
