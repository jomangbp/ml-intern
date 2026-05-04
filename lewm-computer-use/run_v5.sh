#!/usr/bin/env bash
# LeWM v5: Smaller model, combined data, no EMA, strong regularization
set -e

PROJECT_DIR="/home/jgbla/repos/ml-intern/lewm-computer-use"
cd "$PROJECT_DIR"

RUN_NAME="lewm_v5_final_$(date +%Y%m%d_%H%M)"
OUT_DIR="outputs/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  LeWM v5 Final Training"
echo "  Run: $RUN_NAME"
echo "  $(date)"
echo "============================================"
echo ""
echo "Design:"
echo "  - Combined 21K frames (MW + CLI), 3K episodes"
echo "  - 12.3M params (embed 128, pred depth 3)"
echo "  - No EMA target encoder (stop-grad on online targets)"
echo "  - Strong L2: weight_decay=5e-2"
echo "  - Dropout=0.2, λ=0.15"
echo "  - Multi-position prediction loss"
echo "  - Data augmentation"
echo ""

nohup python3 -u scripts/train_lewm.py \
    --data data/combined_mw_cli.h5 \
    --epochs 100 --lr 5e-5 --weight-decay 5e-2 \
    --batch-size 4 --grad-accum 2 \
    --img-size 128 --ctx-len 3 --embed-dim 128 \
    --predictor-depth 3 --encoder-scale tiny --dropout 0.2 \
    --lambd 0.15 \
    --output-dir "$OUT_DIR" \
    > "$OUT_DIR/train.log" 2>&1 &

PID=$!
echo "PID: $PID"
echo "  tail -f $OUT_DIR/train.log"
echo ""
echo "Eval:"
echo "  python3 scripts/eval_lewm.py --checkpoint $OUT_DIR/best_model.pt --data data/combined_mw_cli.h5"
