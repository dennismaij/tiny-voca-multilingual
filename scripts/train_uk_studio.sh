#!/usr/bin/env bash
# Training runner for UK English (en-gb) Studio Model on Apple Silicon MPS
set -euo pipefail

PROJECT_DIR="/Volumes/PRO-G40/Github/Tuny-voca"
cd "$PROJECT_DIR"
mkdir -p checkpoints/british_studio

if pgrep -f "tiny_voca.train.*checkpoints/british_studio" >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] UK English studio training is already running. Exiting."
    exit 0
fi

RESUME_ARGS=()
if [ -f "checkpoints/british_studio/last.pt" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found existing checkpoint. Resuming from checkpoints/british_studio/last.pt..."
    RESUME_ARGS=(--resume checkpoints/british_studio/last.pt)
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting UK English (en-gb) Studio Training from scratch..."
fi

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
    --log-every 100 \
    "${RESUME_ARGS[@]}"
