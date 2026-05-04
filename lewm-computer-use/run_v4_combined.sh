#!/usr/bin/env bash
# LeWM v4: Combined MW+CLI data, smaller model, stronger regularization
set -e

PROJECT_DIR="/home/jgbla/repos/ml-intern/lewm-computer-use"
cd "$PROJECT_DIR"

RUN_NAME="lewm_v4_combined_$(date +%Y%m%d_%H%M)"
OUT_DIR="outputs/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  LeWM v4: Combined MW+CLI, Small Model"
echo "  Run: $RUN_NAME"
echo "  $(date)"
echo "============================================"
echo ""
echo "Strategy:"
echo "  - Combined dataset: 21K frames, 3K episodes (MW + CLI)"
echo "  - predictor_depth=3 (was 6): 50% less predictor params"
echo "  - embed_dim=128 (was 192): smaller latent space"
echo "  - weight_decay=5e-2 (was 1e-2): strong L2 penalty"
echo "  - dropout=0.2 (was 0.15): more dropout"
echo "  - lambd=0.15 (was 0.09): more SIGReg weight"
echo "  - No EMA (hurt on small data)"
echo ""

nohup python3 -u scripts/train_lewm.py \
    --data data/combined_mw_cli.h5 \
    --epochs 100 --lr 5e-5 --weight-decay 5e-2 \
    --batch-size 4 --grad-accum 2 \
    --img-size 128 --ctx-len 3 --embed-dim 128 \
    --encoder-scale tiny --dropout 0.2 \
    --lambd 0.15 \
    --output-dir "$OUT_DIR" \
    > "$OUT_DIR/train.log" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Log: tail -f $OUT_DIR/train.log"
