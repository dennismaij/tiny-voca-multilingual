#!/usr/bin/env bash
# Training runner for UK English (en-gb) Studio Model on Apple Silicon MPS
set -euo pipefail

PROJECT_DIR="/Volumes/PRO-G40/Github/Tuny-voca"
cd "$PROJECT_DIR"
mkdir -p checkpoints/british_studio

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting UK English (en-gb) Studio Training..."
exec "$PROJECT_DIR/.venv/bin/python" -m tiny_voca.train \
    --language en-gb \
    --train annotations/british_studio/train.jsonl \
    --validation annotations/british_studio/validation.jsonl \
    --output checkpoints/british_studio \
    --device mps \
    --steps 100000 \
    --max-seconds 172800 \
    --save-every 1000 \
    --keep-every 10000 \
    --log-every 100
