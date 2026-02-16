#!/bin/bash
set -euo pipefail

GAMES="${1:-5000}"
SIMS="${2:-400}"
EPOCHS="${3:-50}"
DEVICE="${4:-cuda}"

DATA_DIR="/workspace/data"
MODELS_DIR="/workspace/models"
CHECKPOINTS_DIR="/workspace/checkpoints"

NUM_BLOCKS=10
CHANNELS=128
BATCH_SIZE=256
LR=0.001
MAX_MOVES=300
THREADS=$(nproc)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1

PHASES=(
    "beginner:10"
    "intermediate:10"
    "advanced:10"
)

mkdir -p "$DATA_DIR" "$MODELS_DIR"

TOTAL_GENS=0
for phase in "${PHASES[@]}"; do
    gens="${phase##*:}"
    TOTAL_GENS=$((TOTAL_GENS + gens))
done

echo "================================================================"
echo "KOMUGI FULL TRAINING PIPELINE"
echo "================================================================"
echo "Games/gen: $GAMES | Sims: $SIMS | Epochs: $EPOCHS | Threads: $THREADS | GPUs: $NUM_GPUS"
echo "Phases: ${PHASES[*]}"
echo "Total generations: $TOTAL_GENS"
echo "================================================================"
echo ""

GLOBAL_GEN=0
PREV_MODEL=""
PREV_PHASE=""

run_generation() {
    local mode="$1"
    local gen_in_phase="$2"
    local phase_name="$3"
    local gen_label="${phase_name}_gen${gen_in_phase}"

    if [ -f "$MODELS_DIR/${gen_label}.done" ]; then
        echo "[$gen_label] Already complete (found .done marker), skipping..."
        local done_model="$MODELS_DIR/${gen_label}.onnx"
        if [ -f "$done_model" ]; then
            PREV_MODEL="$done_model"
        fi
        GLOBAL_GEN=$((GLOBAL_GEN + 1))
        return
    fi

    echo ""
    echo "=========================================="
    echo "[$gen_label] Generation $((GLOBAL_GEN + 1))/$TOTAL_GENS — mode=$mode"
    echo "=========================================="

    local data_file="$DATA_DIR/${gen_label}.jsonl"
    local ckpt_dir="$CHECKPOINTS_DIR/$gen_label"
    mkdir -p "$ckpt_dir"

    local model_arg="-"
    if [ -n "$PREV_MODEL" ] && [ -f "$PREV_MODEL" ]; then
        model_arg="$PREV_MODEL"
        echo "Self-play with model: $PREV_MODEL"
    else
        echo "Self-play with heuristic policy (no model yet)"
    fi

    local sp_elapsed=0
    if [ -f "$data_file" ] && [ -s "$data_file" ]; then
        local positions
        positions=$(wc -l < "$data_file")
        echo "Self-play data already exists: $data_file ($positions positions), skipping..."
    else
        echo "Generating $GAMES games ($mode, $SIMS sims, $THREADS threads)..."
        local sp_start=$SECONDS
        selfplay "$GAMES" "$data_file" "$SIMS" "$model_arg" "$mode" "$THREADS"
        sp_elapsed=$((SECONDS - sp_start))
        local positions
        positions=$(wc -l < "$data_file")
        echo "Self-play done: $positions positions in ${sp_elapsed}s"
    fi

    local resume_flag=""
    if [ -n "$PREV_MODEL" ] && [ "$gen_in_phase" -gt 0 ]; then
        local prev_gen_label="${phase_name}_gen$((gen_in_phase - 1))"
        local prev_ckpt_dir="$CHECKPOINTS_DIR/$prev_gen_label"
        if [ -d "$prev_ckpt_dir" ]; then
            local prev_ckpt
            prev_ckpt=$(ls -t "$prev_ckpt_dir"/model_epoch_*.pt 2>/dev/null | head -1 || true)
            if [ -n "$prev_ckpt" ] && [ -f "$prev_ckpt" ]; then
                resume_flag="--resume $prev_ckpt"
                echo "Warm-starting from $prev_ckpt"
            fi
        fi
    fi

    local data_window="$data_file"
    local window_size=3
    for prev_g in $(seq $((gen_in_phase - 1)) -1 $((gen_in_phase - window_size + 1))); do
        [ "$prev_g" -lt 0 ] && break
        local prev_data="$DATA_DIR/${phase_name}_gen${prev_g}.jsonl"
        [ -f "$prev_data" ] && data_window="$prev_data,$data_window"
    done

    local preprocessed_dir="$DATA_DIR/${gen_label}_preprocessed"
    echo "Preprocessing $positions positions (window: $data_window)..."
    local prep_start=$SECONDS
    python preprocess.py "$data_window" "$preprocessed_dir"
    local prep_elapsed=$((SECONDS - prep_start))
    echo "Preprocessing done in ${prep_elapsed}s"

    echo "Training $EPOCHS epochs on $positions positions (${NUM_GPUS} GPUs)..."
    local train_start=$SECONDS
    torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py \
        --data "$preprocessed_dir" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --device "$DEVICE" \
        --save-every 10 \
        --output-dir "$ckpt_dir" \
        --num-blocks "$NUM_BLOCKS" \
        --channels "$CHANNELS" \
        $resume_flag
    local train_elapsed=$((SECONDS - train_start))
    echo "Training done in ${train_elapsed}s"

    local final_ckpt="$ckpt_dir/model_epoch_${EPOCHS}.pt"
    if [ ! -f "$final_ckpt" ]; then
        final_ckpt=$(ls -t "$ckpt_dir"/model_epoch_*.pt 2>/dev/null | head -1)
    fi

    if [ -z "$final_ckpt" ] || [ ! -f "$final_ckpt" ]; then
        echo "ERROR: no checkpoint found after training $gen_label"
        exit 1
    fi

    local output_model="$MODELS_DIR/${gen_label}.onnx"
    echo "Exporting $final_ckpt -> $output_model"
    python export_onnx.py \
        --checkpoint "$final_ckpt" \
        --output "$output_model" \
        --num-blocks "$NUM_BLOCKS" \
        --channels "$CHANNELS"

    PREV_MODEL="$output_model"
    echo "Model promoted: $output_model"

    touch "$MODELS_DIR/${gen_label}.done"
    GLOBAL_GEN=$((GLOBAL_GEN + 1))

    local stale_gen=$((gen_in_phase - window_size))
    if [ "$stale_gen" -ge 0 ]; then
        local stale_data="$DATA_DIR/${phase_name}_gen${stale_gen}.jsonl"
        if [ -f "$stale_data" ]; then
            echo "Cleaning up stale data: $stale_data ($(du -h "$stale_data" | cut -f1))"
            rm -f "$stale_data"
        fi
        local stale_prep="$DATA_DIR/${phase_name}_gen${stale_gen}_preprocessed"
        if [ -d "$stale_prep" ]; then
            echo "Cleaning up stale preprocessed: $stale_prep"
            rm -rf "$stale_prep"
        fi
    fi
    # Also clean current gen's preprocessed dir (no longer needed after training)
    if [ -d "$preprocessed_dir" ]; then
        echo "Cleaning up preprocessed dir: $preprocessed_dir"
        rm -rf "$preprocessed_dir"
    fi

    echo "[$gen_label] Complete: $(du -h "$output_model" | cut -f1) model, ${sp_elapsed}s self-play + ${train_elapsed}s training"
}

PIPELINE_START=$SECONDS

for phase in "${PHASES[@]}"; do
    mode="${phase%%:*}"
    gens="${phase##*:}"

    echo ""
    echo "################################################################"
    echo "PHASE: $mode ($gens generations)"
    echo "################################################################"

    if [ -n "$PREV_PHASE" ] && [ "$PREV_PHASE" != "$mode" ]; then
        echo "Cleaning up previous phase data ($PREV_PHASE)..."
        rm -f "$DATA_DIR"/${PREV_PHASE}_gen*.jsonl "$DATA_DIR"/${PREV_PHASE}_gen*.pgn
        rm -rf "$DATA_DIR"/${PREV_PHASE}_gen*_preprocessed
    fi
    PREV_PHASE="$mode"

    for g in $(seq 0 $((gens - 1))); do
        run_generation "$mode" "$g" "$mode"
    done
done

cp "$PREV_MODEL" "$MODELS_DIR/komugi_final.onnx" 2>/dev/null || true

TOTAL_ELAPSED=$((SECONDS - PIPELINE_START))
HOURS=$((TOTAL_ELAPSED / 3600))
MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================"
echo "Total time: ${HOURS}h ${MINS}m"
echo "Generations trained: $GLOBAL_GEN"
echo "Final model: $MODELS_DIR/komugi_final.onnx"
echo ""
echo "Models:"
ls -lh "$MODELS_DIR"/*.onnx
echo "================================================================"
