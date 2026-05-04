#!/usr/bin/env bash
# LeWM v3: EMA target encoder (the missing piece for JEPA)
set -e

PROJECT_DIR="/home/jgbla/repos/ml-intern/lewm-computer-use"
cd "$PROJECT_DIR"

RUN_NAME="lewm_v3_ema_$(date +%Y%m%d_%H%M)"
OUT_DIR="outputs/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  LeWM v3 Training + EMA Target Encoder"
echo "  Run: $RUN_NAME"
echo "  $(date)"
echo "============================================"
echo ""
echo "Key fix over v2:"
echo "  - EMA target encoder (momentum=0.996)"
echo "  - Target encoded with EMA weights, stop grad"
echo "  - Per-step LR scheduler (was per-epoch, broken)"
echo ""
echo "Retained from v2:"
echo "  - Episode-aware sampling"
echo "  - Episode-level train/val split"
echo "  - Multi-position prediction loss"
echo "  - Data augmentation"
echo ""

nohup python3 -u scripts/train_lewm.py \
    --data data/mind2web_all.h5 \
    --epochs 100 --lr 5e-5 --weight-decay 1e-2 \
    --batch-size 4 --grad-accum 2 \
    --img-size 128 --ctx-len 3 --embed-dim 192 \
    --encoder-scale tiny --dropout 0.15 \
    --output-dir "$OUT_DIR" \
    > "$OUT_DIR/train.log" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Log: tail -f $OUT_DIR/train.log"
echo ""
echo "After training:"
echo "  python3 scripts/eval_lewm.py --checkpoint $OUT_DIR/best_model.pt --data data/mind2web_all.h5"
