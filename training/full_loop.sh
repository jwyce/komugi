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
THREADS="${THREADS:-$(nproc)}"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
BEST_CKPT_POLICY_WEIGHT="${BEST_CKPT_POLICY_WEIGHT:-1.0}"
BEST_CKPT_VALUE_WEIGHT="${BEST_CKPT_VALUE_WEIGHT:-1.0}"
SELFPLAY_SHARDS="${SELFPLAY_SHARDS:-1}"
SELFPLAY_SHARD_THREADS="${SELFPLAY_SHARD_THREADS:-0}"
SELFPLAY_GPUS_PER_SHARD="${SELFPLAY_GPUS_PER_SHARD:-0}"

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

select_best_checkpoint() {
    local ckpt_dir="$1"
    local metrics_file="$ckpt_dir/metrics.csv"
    python3 - "$ckpt_dir" "$metrics_file" "$BEST_CKPT_POLICY_WEIGHT" "$BEST_CKPT_VALUE_WEIGHT" <<'PY'
import csv
import math
import os
import sys

ckpt_dir, metrics_file, policy_w, value_w = sys.argv[1:5]
policy_w = float(policy_w)
value_w = float(value_w)

if not os.path.exists(metrics_file):
    print("")
    raise SystemExit(0)

best_score = math.inf
best_path = ""

with open(metrics_file, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        epoch_raw = (row.get("epoch") or "").strip()
        val_policy_raw = (row.get("val_policy_loss") or "").strip()
        val_value_raw = (row.get("val_value_loss") or "").strip()
        if not epoch_raw or not val_policy_raw or not val_value_raw:
            continue
        try:
            epoch = int(float(epoch_raw))
            val_policy = float(val_policy_raw)
            val_value = float(val_value_raw)
        except ValueError:
            continue
        if not (math.isfinite(val_policy) and math.isfinite(val_value)):
            continue
        candidate = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
        if not os.path.exists(candidate):
            continue
        score = policy_w * val_policy + value_w * val_value
        if score < best_score:
            best_score = score
            best_path = candidate

print(best_path)
PY
}

gpu_list_for_shard() {
    local start_idx="$1"
    local count="$2"
    local total="$3"
    local out=""
    local i
    for ((i = 0; i < count; i++)); do
        local gpu=$(( (start_idx + i) % total ))
        if [ -z "$out" ]; then
            out="$gpu"
        else
            out="$out,$gpu"
        fi
    done
    printf '%s\n' "$out"
}

renumber_pgn_games() {
    local pgn_path="$1"
    python3 - "$pgn_path" <<'PY'
import re
import sys

path = sys.argv[1]
counter = 0
out = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("[Game "):
            counter += 1
            out.append(f'[Game "{counter}"]\n')
        else:
            out.append(line)

with open(path, "w", encoding="utf-8") as fh:
    fh.writelines(out)
PY
}

run_selfplay_games() {
    local games="$1"
    local output_file="$2"
    local sims="$3"
    local model_arg="$4"
    local mode="$5"
    local default_threads="$6"

    if [ "$SELFPLAY_SHARDS" -le 1 ]; then
        selfplay "$games" "$output_file" "$sims" "$model_arg" "$mode" "$default_threads"
        return
    fi

    local shard_count="$SELFPLAY_SHARDS"
    if [ "$shard_count" -gt "$games" ]; then
        shard_count="$games"
    fi

    local shard_threads="$SELFPLAY_SHARD_THREADS"
    if [ "$shard_threads" -le 0 ]; then
        shard_threads=$(( (default_threads + shard_count - 1) / shard_count ))
    fi
    [ "$shard_threads" -lt 1 ] && shard_threads=1

    local gpus_per_shard="$SELFPLAY_GPUS_PER_SHARD"
    if [ "$gpus_per_shard" -le 0 ]; then
        if [ "$NUM_GPUS" -ge "$shard_count" ]; then
            gpus_per_shard=$(( NUM_GPUS / shard_count ))
            [ "$gpus_per_shard" -lt 1 ] && gpus_per_shard=1
        else
            gpus_per_shard=1
        fi
    fi

    echo "Running sharded self-play: shards=$shard_count threads_per_shard=$shard_threads gpus_per_shard=$gpus_per_shard"

    local base_games=$(( games / shard_count ))
    local remainder=$(( games % shard_count ))
    local -a shard_json_files=()
    local -a shard_pgn_files=()
    local -a shard_logs=()
    local -a shard_pids=()

    local sid
    for ((sid = 0; sid < shard_count; sid++)); do
        local shard_games="$base_games"
        if [ "$sid" -lt "$remainder" ]; then
            shard_games=$(( shard_games + 1 ))
        fi
        if [ "$shard_games" -le 0 ]; then
            continue
        fi

        local shard_json="${output_file%.jsonl}.shard${sid}.jsonl"
        local shard_pgn="${output_file%.jsonl}.shard${sid}.pgn"
        local shard_log="${output_file%.jsonl}.shard${sid}.log"
        rm -f "$shard_json" "$shard_pgn" "$shard_log"

        local gpu_env=""
        if [ "$NUM_GPUS" -gt 0 ] && [ "$model_arg" != "-" ]; then
            local gpu_start=$(( (sid * gpus_per_shard) % NUM_GPUS ))
            local gpu_list
            gpu_list=$(gpu_list_for_shard "$gpu_start" "$gpus_per_shard" "$NUM_GPUS")
            gpu_env="$gpu_list"
            echo "[shard $sid] games=$shard_games threads=$shard_threads CUDA_VISIBLE_DEVICES=$gpu_list"
            CUDA_VISIBLE_DEVICES="$gpu_env" selfplay "$shard_games" "$shard_json" "$sims" "$model_arg" "$mode" "$shard_threads" >"$shard_log" 2>&1 &
        else
            echo "[shard $sid] games=$shard_games threads=$shard_threads (CPU inference)"
            selfplay "$shard_games" "$shard_json" "$sims" "$model_arg" "$mode" "$shard_threads" >"$shard_log" 2>&1 &
        fi

        shard_json_files+=("$shard_json")
        shard_pgn_files+=("$shard_pgn")
        shard_logs+=("$shard_log")
        shard_pids+=("$!")
    done

    local failed=0
    local pid
    for pid in "${shard_pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "ERROR: one or more self-play shards failed"
        local log
        for log in "${shard_logs[@]}"; do
            if [ -f "$log" ]; then
                echo "--- tail: $log ---"
                tail -40 "$log" || true
            fi
        done
        exit 1
    fi

    local f
    : > "$output_file"
    for f in "${shard_json_files[@]}"; do
        cat "$f" >> "$output_file"
    done

    local merged_pgn="${output_file%.jsonl}.pgn"
    : > "$merged_pgn"
    for f in "${shard_pgn_files[@]}"; do
        if [ -f "$f" ]; then
            cat "$f" >> "$merged_pgn"
            printf '\n' >> "$merged_pgn"
        fi
    done
    renumber_pgn_games "$merged_pgn"

    for f in "${shard_json_files[@]}" "${shard_pgn_files[@]}" "${shard_logs[@]}"; do
        rm -f "$f"
    done
}

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
        run_selfplay_games "$GAMES" "$data_file" "$SIMS" "$model_arg" "$mode" "$THREADS"
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
            prev_ckpt=$(select_best_checkpoint "$prev_ckpt_dir")
            if [ -z "$prev_ckpt" ]; then
                prev_ckpt=$(ls -t "$prev_ckpt_dir"/model_epoch_*.pt 2>/dev/null | head -1 || true)
            fi
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

    local final_ckpt
    final_ckpt=$(select_best_checkpoint "$ckpt_dir")
    if [ -n "$final_ckpt" ] && [ -f "$final_ckpt" ]; then
        echo "Selected best checkpoint from validation metrics: $final_ckpt"
    else
        final_ckpt="$ckpt_dir/model_epoch_${EPOCHS}.pt"
        if [ ! -f "$final_ckpt" ]; then
            final_ckpt=$(ls -t "$ckpt_dir"/model_epoch_*.pt 2>/dev/null | head -1)
        fi
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
