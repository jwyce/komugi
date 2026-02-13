#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-/workspace/data}"
MODELS_DIR="${2:-/workspace/models}"
EPOCHS="${3:-50}"
DEVICE="${4:-cuda}"
NUM_GENERATIONS="${5:-5}"

NUM_BLOCKS=10
CHANNELS=128
BATCH_SIZE=256
LR=0.001

mkdir -p "$MODELS_DIR"

echo "Training loop: $NUM_GENERATIONS generations, $EPOCHS epochs each"
echo "Data: $DATA_DIR | Models: $MODELS_DIR | Device: $DEVICE"
echo ""

for gen in $(seq 0 $((NUM_GENERATIONS - 1))); do
    echo "=========================================="
    echo "Generation $gen"
    echo "=========================================="

    DATA_FILE="$DATA_DIR/gen${gen}.jsonl"
    if [ ! -f "$DATA_FILE" ]; then
        echo "Error: $DATA_FILE not found. Upload self-play data first."
        exit 1
    fi

    CHECKPOINT_DIR="/workspace/checkpoints/gen${gen}"
    mkdir -p "$CHECKPOINT_DIR"

    # Warm-start from previous generation's final checkpoint
    RESUME_FLAG=""
    if [ "$gen" -gt 0 ]; then
        PREV_DIR="/workspace/checkpoints/gen$((gen - 1))"
        PREV_CKPT=$(ls -t "$PREV_DIR"/model_epoch_*.pt 2>/dev/null | head -1)
        if [ -n "$PREV_CKPT" ] && [ -f "$PREV_CKPT" ]; then
            RESUME_FLAG="--resume $PREV_CKPT"
            echo "Warm-starting from $PREV_CKPT"
        fi
    fi

    echo "Training on $(wc -l < "$DATA_FILE") positions..."
    python train.py \
        --data "$DATA_FILE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --device "$DEVICE" \
        --save-every 10 \
        --output-dir "$CHECKPOINT_DIR" \
        --num-blocks "$NUM_BLOCKS" \
        --channels "$CHANNELS" \
        $RESUME_FLAG

    FINAL_CHECKPOINT="$CHECKPOINT_DIR/model_epoch_${EPOCHS}.pt"
    if [ ! -f "$FINAL_CHECKPOINT" ]; then
        FINAL_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/model_epoch_*.pt 2>/dev/null | head -1)
    fi

    if [ -z "$FINAL_CHECKPOINT" ] || [ ! -f "$FINAL_CHECKPOINT" ]; then
        echo "Error: no checkpoint found after training gen $gen"
        exit 1
    fi

    OUTPUT_MODEL="$MODELS_DIR/gungi_v${gen}.onnx"
    echo "Exporting $FINAL_CHECKPOINT -> $OUTPUT_MODEL"
    python export_onnx.py \
        --checkpoint "$FINAL_CHECKPOINT" \
        --output "$OUTPUT_MODEL" \
        --num-blocks "$NUM_BLOCKS" \
        --channels "$CHANNELS"

    echo "Generation $gen complete: $(du -h "$OUTPUT_MODEL" | cut -f1)"
    echo ""
done

echo "=========================================="
echo "Done. $NUM_GENERATIONS models in $MODELS_DIR"
ls -lh "$MODELS_DIR"/gungi_v*.onnx
echo "=========================================="
