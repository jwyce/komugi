#!/bin/bash
set -euo pipefail

GAMES="${1:-5000}"
SIMS="${2:-800}"
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
echo "Games/gen: $GAMES | Sims: $SIMS | Epochs: $EPOCHS | Threads: $THREADS"
echo "Phases: ${PHASES[*]}"
echo "Total generations: $TOTAL_GENS"
echo "================================================================"
echo ""

GLOBAL_GEN=0
PREV_MODEL=""

run_generation() {
    local mode="$1"
    local gen_in_phase="$2"
    local phase_name="$3"
    local gen_label="${phase_name}_gen${gen_in_phase}"

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

    local effective_sims=$SIMS
    if [ "$gen_in_phase" -ge 7 ]; then
        effective_sims=$((SIMS * 2))
    elif [ "$gen_in_phase" -ge 3 ]; then
        effective_sims=$(( (SIMS * 3 + 1) / 2 ))
    fi

    echo "Generating $GAMES games ($mode, $effective_sims sims, $THREADS threads)..."
    local sp_start=$SECONDS
    selfplay "$GAMES" "$data_file" "$effective_sims" "$model_arg" "$mode" "$THREADS"
    local sp_elapsed=$((SECONDS - sp_start))
    local positions
    positions=$(wc -l < "$data_file")
    echo "Self-play done: $positions positions in ${sp_elapsed}s"

    local resume_flag=""
    if [ -n "$PREV_MODEL" ]; then
        local prev_ckpt_dir
        prev_ckpt_dir=$(ls -dt "$CHECKPOINTS_DIR"/*/model_epoch_*.pt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
        if [ -n "$prev_ckpt_dir" ]; then
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

    echo "Training $EPOCHS epochs on $positions positions (window: $data_window)..."
    local train_start=$SECONDS
    python train.py \
        --data "$data_window" \
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

    if [ -n "$PREV_MODEL" ] && [ -f "$PREV_MODEL" ]; then
        local gate_games=20
        local gate_sims=200
        local new_stderr
        new_stderr=$(mktemp)
        local old_stderr
        old_stderr=$(mktemp)

        selfplay "$gate_games" /dev/null "$gate_sims" "$output_model" "$mode" "$THREADS" 2>"$new_stderr"
        local new_decisive
        new_decisive=$(grep -cE "WhiteWin|BlackWin" "$new_stderr" || true)

        selfplay "$gate_games" /dev/null "$gate_sims" "$PREV_MODEL" "$mode" "$THREADS" 2>"$old_stderr"
        local old_decisive
        old_decisive=$(grep -cE "WhiteWin|BlackWin" "$old_stderr" || true)

        rm -f "$new_stderr" "$old_stderr"

        if [ "$new_decisive" -ge "$old_decisive" ]; then
            echo "GATING: New model promoted ($new_decisive decisive >= $old_decisive)"
            PREV_MODEL="$output_model"
        else
            echo "GATING: Old model retained ($old_decisive decisive > $new_decisive)"
        fi
    else
        PREV_MODEL="$output_model"
    fi

    GLOBAL_GEN=$((GLOBAL_GEN + 1))

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
