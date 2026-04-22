#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-minimax/MiniMax-M2.7}"
MAX_ITER="${MAX_ITERATIONS:-10}"

WORKDIR="${WORKDIR:-/tmp/tiny-transformer-e2e}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

PROMPT="Create a tiny PyTorch transformer script in tiny_transformer.py with a quick forward-pass sanity test, run it, and report output."

echo "[e2e] model=$MODEL workdir=$WORKDIR"

if ! command -v ml-intern >/dev/null 2>&1; then
  echo "[e2e] ml-intern not on PATH. Install with: uv tool install -e ."
  exit 1
fi

# Optional guardrails for common models.
if [[ "$MODEL" == minimax/* ]] && [[ -z "${MINIMAX_API_KEY:-}" ]]; then
  echo "[e2e] MINIMAX_API_KEY not set; skipping run for model $MODEL"
  exit 0
fi
if [[ "$MODEL" == zai/* ]] && [[ -z "${ZAI_API_KEY:-}" ]]; then
  echo "[e2e] ZAI_API_KEY not set; skipping run for model $MODEL"
  exit 0
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[e2e] HF_TOKEN not set; ml-intern may prompt. Continuing..."
fi

ml-intern --model "$MODEL" --max-iterations "$MAX_ITER" --no-stream "$PROMPT"

test -f tiny_transformer.py
python3 tiny_transformer.py

echo "[e2e] tiny transformer flow completed"
