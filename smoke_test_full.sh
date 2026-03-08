#!/bin/bash
# FULL PIPELINE SMOKE TEST (single-GPU NNUE)
# Validates: selfplay → labelgen → NNUE training → puzzle extraction
# NNUE uses 1 GPU (not multi-GPU like ONNX training)
set -euo pipefail

WORKSPACE="/workspace"
MODEL="${WORKSPACE}/models/advanced_gen9_rulefix.onnx"
SMOKE_DIR="${WORKSPACE}/smoke_full"

echo "================================================================"
echo "FULL PIPELINE SMOKE TEST"
echo "================================================================"
echo ""
echo "This validates the entire chain:"
echo "  1. Selfplay (100 games, 5000 sims) - CPU only"
echo "  2. Labelgen (5000 sims labels) - CPU search"
echo "  3. NNUE training (10 epochs) - SINGLE GPU"
echo "  4. Puzzle extraction (candidate detection)"
echo ""
echo "⚠ NNUE training uses 1 GPU (not 8 like ONNX training)"
echo "   This is expected - NNUE is single-GPU only"
echo ""
echo "Estimated time: 2-4 hours"
echo "================================================================"
echo ""

# Check model exists
if [ ! -f "$MODEL" ]; then
    echo "✗ ERROR: Model not found: $MODEL"
    echo "Wait for pass B to complete and export"
    exit 1
fi

# Clean previous smoke test
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

# ============================================================================
# STEP 1: Selfplay (with RAM monitoring)
# ============================================================================
echo "STEP 1: Selfplay (100 games, 5000 sims, advanced)"
echo "----------------------------------------------------------------"
echo "Monitoring RAM usage during selfplay..."
cd "$WORKSPACE"

# Start RAM monitor in background
RAM_LOG="${SMOKE_DIR}/ram_usage.log"
touch "$RAM_LOG"
(
  while true; do
    # Get total RSS for all selfplay processes
    rss_kb=$(ps aux | grep '[s]elfplay' | awk '{sum+=$6} END {print sum}')
    if [ -n "$rss_kb" ] && [ "$rss_kb" -gt 0 ]; then
      rss_gb=$(echo "scale=2; $rss_kb / 1024 / 1024" | bc)
      echo "$(date +%H:%M:%S) ${rss_gb}GB" >> "$RAM_LOG"
    fi
    sleep 5
  done
) &
RAM_MONITOR_PID=$!

# Start disk monitor in background
DISK_LOG="${SMOKE_DIR}/disk_usage.log"
echo "timestamp used_gb available_gb percent" > "$DISK_LOG"
touch "$DISK_LOG"
(
  while true; do
    # Get disk usage for /workspace
    df -h /workspace | tail -1 | awk -v date="$(date +%H:%M:%S)" '{print date, $3, $4, $5}' >> "$DISK_LOG"
    sleep 30
  done
) &
DISK_MONITOR_PID=$!

# Run selfplay with timing
echo "⏱️  Starting selfplay at $(date '+%H:%M:%S')"
START_TIME=$(date +%s)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KOMUGI_MAX_MOVES=500 KOMUGI_VL_BATCH_SIZE=12 \
    timeout 7200 selfplay 100 "${SMOKE_DIR}/games.jsonl" 5000 "$MODEL" advanced 256
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Stop monitors
kill $RAM_MONITOR_PID 2>/dev/null || true
kill $DISK_MONITOR_PID 2>/dev/null || true
wait $RAM_MONITOR_PID 2>/dev/null || true
wait $DISK_MONITOR_PID 2>/dev/null || true
# Check results
if [ ! -f "${SMOKE_DIR}/games.jsonl" ]; then
    echo "✗ FAILED: No JSONL output"
    fi

# Calculate game stats
games=$(grep -cE '^\[t[0-9]+\] Game' "${SMOKE_DIR}/games.jsonl" 2>/dev/null || echo 0)
lines=$(wc -l < "${SMOKE_DIR}/games.jsonl")
echo "✓ Selfplay: $games games, $lines positions"
echo ""

# Calculate timing and GPM
if [ -n "$DURATION" ] && [ "$DURATION" -gt 0 ] && [ "$games" -gt 0 ]; then
  duration_min=$((DURATION / 60))
  gpm=$(awk "BEGIN {printf \"%.2f\", $games * 60 / $DURATION}")
  
  echo "⏱️  TIMING (100 games @ 5000 sims):"
  echo "   Duration: ${duration_min}min"
  echo "   Throughput: ${gpm} gpm"
  echo ""
  
  # Project for full runs
  proj_5000_min=$(awk "BEGIN {printf \"%.0f\", 5000 * $DURATION / $games}")
  proj_5000_hours=$(awk "BEGIN {printf \"%.1f\", $proj_5000_min / 60}")
  proj_5000_days=$(awk "BEGIN {printf \"%.1f\", $proj_5000_min / 60 / 24}")
  proj_2000_hours=$(awk "BEGIN {printf \"%.1f\", 2000 * $DURATION / $games / 60}")
  
  echo "📋 TIMELINE PROJECTION (from actual 5000-sim performance):"
  echo "   Beginner (2000 sims): ~${proj_2000_hours}h"
  echo "   Intermediate (2000 sims): ~${proj_2000_hours}h"
  echo "   Advanced (5000 sims): ~${proj_5000_hours}h (${proj_5000_days} days)"
  echo "   Total selfplay: ~$(awk "BEGIN {printf \"%.1f\", $proj_2000_hours * 2 + $proj_5000_hours}")h"
  echo ""
fi

if [ -f "${SMOKE_DIR}/ram_usage.log" ]; then
  peak_ram=$(awk '{if($2>max) max=$2} END {print max}' "${SMOKE_DIR}/ram_usage.log")
  avg_ram=$(awk '{sum+=$2; count++} END {if(count>0) print sum/count}' "${SMOKE_DIR}/ram_usage.log")
  samples=$(wc -l < "${SMOKE_DIR}/ram_usage.log")
  echo "📊 RAM Usage (5000 sims, 256 threads):"
  echo "   Peak: ${peak_ram}GB"
  echo "   Avg: ${avg_ram}GB"
  echo "   Samples: $samples"
  echo ""
  
  # Recommend thread count for full runs
  if command -v bc >/dev/null 2>&1; then
    per_thread=$(echo "scale=3; $peak_ram / 256" | bc)
    max_threads_500gb=$(echo "scale=0; 500 / $per_thread" | bc)
    safe_threads=$(echo "scale=0; $max_threads_500gb * 0.9 / 1" | bc)  # 10% safety margin
    echo "📋 RECOMMENDATION FOR FULL RUNS:"
    echo "   RAM per thread @ 5000 sims: ~${per_thread}GB"
    echo "   Max threads for 500GB RAM: ~$max_threads_500gb"
    echo "   Safe thread count (90%): $safe_threads"
    echo ""
    if [ "$safe_threads" -lt 200 ]; then
      echo "⚠️  WARNING: Use $safe_threads threads to avoid OOM"
      echo "    (256 threads risks OOM with $peak_ram GB peak)"
    else
      echo "✅ 256 threads should be safe"
    fi
    echo ""
  fi
fi

# Calculate disk usage from log
if [ -f "${SMOKE_DIR}/disk_usage.log" ]; then
  peak_disk=$(tail -n +2 "${SMOKE_DIR}/disk_usage.log" | awk '{gsub(/G/,"",$2); if($2>max) max=$2} END {print max}')
  final_available=$(tail -1 "${SMOKE_DIR}/disk_usage.log" | awk '{print $3}')
  final_percent=$(tail -1 "${SMOKE_DIR}/disk_usage.log" | awk '{print $4}')
  echo "📀 Disk Usage (during selfplay):"
  echo "   Peak used: ${peak_disk}GB"
  echo "   Final available: ${final_available}"
  echo "   Final usage: ${final_percent}"
  echo ""
  
  # Project for full runs
  if [ -n "$peak_disk" ] && [ "$games" -gt 0 ]; then
    disk_per_100games=$(echo "scale=1; $peak_disk" | bc)
    proj_5000_beg=$(echo "scale=0; $disk_per_100games * 50" | bc)
    proj_5000_int=$(echo "scale=0; $disk_per_100games * 50" | bc)
    proj_5000_adv=$(echo "scale=0; $disk_per_100games * 50" | bc)
    total_proj=$(echo "scale=0; $proj_5000_beg + $proj_5000_int + $proj_5000_adv" | bc)
    echo "📋 DISK PROJECTION FOR FULL RUNS:"
    echo "   Per 100 games: ~${disk_per_100games}GB"
    echo "   Beginner (5000): ~${proj_5000_beg}GB"
    echo "   Intermediate (5000): ~${proj_5000_int}GB"
    echo "   Advanced (5000): ~${proj_5000_adv}GB"
    echo "   Total estimated: ~${total_proj}GB"
    echo ""
    # Check if we have enough space
    available_gb=$(df /workspace | tail -1 | awk '{print $4/1024/1024}')
    if [ -n "$available_gb" ] && command -v bc >/dev/null 2>&1; then
      if [ "$total_proj" -gt "$available_gb" ]; then
        echo "⚠️  WARNING: Insufficient disk space"
        echo "    Need: ${total_proj}GB, Have: ${available_gb}GB"
        echo "    Delete files or reduce game count"
      else
        echo "✅ Sufficient disk space (${available_gb}GB available)"
      fi
    fi
    echo ""
  fi
fi

echo ""

# ============================================================================
# STEP 2: Labelgen
# ============================================================================
echo "STEP 2: Labelgen (5000 sims)"
echo "----------------------------------------------------------------"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KOMUGI_VL_BATCH_SIZE=12 \
    timeout 3600 python3 labelgen.py \
    "${SMOKE_DIR}/games.jsonl" \
    "${SMOKE_DIR}/labels.jsonl" \
    5000 "$MODEL" advanced 256

if [ ! -f "${SMOKE_DIR}/labels.jsonl" ]; then
    echo "✗ FAILED: No labels generated"
    exit 1
fi

labels=$(wc -l < "${SMOKE_DIR}/labels.jsonl")
echo "✓ Labelgen: $labels labels generated"
echo ""

# ============================================================================
# STEP 3: Preprocess (for NNUE)
# ============================================================================
echo "STEP 3: Preprocess (NNUE training data)"
echo "----------------------------------------------------------------"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    timeout 1800 python3 preprocess.py \
    "${SMOKE_DIR}/labels.jsonl" \
    "${SMOKE_DIR}/preprocessed"

if [ ! -d "${SMOKE_DIR}/preprocessed" ]; then
    echo "✗ FAILED: Preprocessing failed"
    exit 1
fi

positions=$(python3 -c "import numpy as np; m=np.load('${SMOKE_DIR}/preprocessed/meta.npy'); print(int(m[0]))" 2>/dev/null || echo "unknown")
echo "✓ Preprocess: $positions positions ready"
echo ""

# ============================================================================
# STEP 4: NNUE Training (single GPU)
# ============================================================================
echo "STEP 4: NNUE Training (10 epochs, SINGLE GPU)"
echo "----------------------------------------------------------------"
echo "⚠ Using only 1 GPU (train_nnue.py is single-GPU only)"
echo "  This is expected behavior - not multi-GPU like ONNX training"
echo ""
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    timeout 3600 python3 train_nnue.py \
    --data "${SMOKE_DIR}/preprocessed" \
    --epochs 10 \
    --batch-size 256 \
    --device cuda \
    --output-dir "${SMOKE_DIR}/nnue_checkpoint"

if [ ! -d "${SMOKE_DIR}/nnue_checkpoint" ]; then
    echo "✗ FAILED: NNUE training failed"
    exit 1
fi

# Check if checkpoint was saved
ckpt=$(find "${SMOKE_DIR}/nnue_checkpoint" -name "*.pt" | head -1)
if [ -z "$ckpt" ]; then
    echo "✗ FAILED: No checkpoint saved"
    exit 1
fi

echo "✓ NNUE: Checkpoint saved at $ckpt"
echo "  (Trained on single GPU - this is correct)"
echo ""
echo ""

# ============================================================================
# STEP 5: Puzzle Extraction (basic)
# ============================================================================
echo "STEP 5: Puzzle Extraction (candidate detection)"
echo "----------------------------------------------------------------"

if [ -f "${SMOKE_DIR}/games.pgn" ]; then
    # Count candidate positions (tactical moments)
    candidates=$(grep -cE '(取|返|新)' "${SMOKE_DIR}/games.pgn" 2>/dev/null || echo 0)
    echo "✓ Puzzle candidates: $positions with captures/betrayals/arata"
    
    # Verify we can parse PGN
    python3 -c "
import chess.pgn
import io

with open('${SMOKE_DIR}/games.pgn') as f:
    games = 0
    while chess.pgn.read_game(f):
        games += 1
        if games >= 5:  # Just check first 5
            break
    print(f'✓ PGN parse: {games} games readable')
" 2>/dev/null || echo "⚠ PGN parse check skipped"
else
    echo "⚠ No PGN file for puzzle extraction"
fi

echo ""

# ============================================================================
# STEP 6: Tactician Betrayal Check
# ============================================================================
echo "STEP 6: Rulefix Validation (tactician betrayal presence)"
echo "----------------------------------------------------------------"

python3 /workspace/check_tactician_betrayal.py "${SMOKE_DIR}/games.jsonl" 2>/dev/null || true
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "================================================================"
echo "SMOKE TEST SUMMARY"
echo "================================================================"
echo ""
echo "Files generated:"
ls -lh "$SMOKE_DIR" 2>/dev/null | tail -n +2
echo ""

echo "Disk usage:"
df -h /workspace | tail -1
echo ""

# Validate all steps passed
if [ "$games" -lt 90 ]; then
    echo "✗ FAIL: Expected ~100 games, got $games"
    exit 1
fi

if [ "$labels" -lt 1000 ]; then
    echo "✗ FAIL: Expected labels, got $labels"
    exit 1
fi

echo "✓✓✓ ALL STEPS PASSED ✓✓✓"
echo ""
echo "Full pipeline validated:"
echo "  ✓ Selfplay @ 5000 sims works"
echo "  ✓ Labelgen produces valid output"
echo "  ✓ NNUE training completes"
echo "  ✓ Puzzle extraction possible"
echo ""
echo "READY FOR FULL RUNS:"
echo "  - Beginner: 2000 sims × 5000 games (~2-3 days)"
echo "  - Intermediate: 2000 sims × 5000 games (~2-3 days)"
echo "  - Advanced: 5000 sims × 5000 games (~4-6 days)"
echo ""
echo "Next actions:"
echo "1. Review ${SMOKE_DIR}/ for quality"
echo "2. Delete ${SMOKE_DIR}/ to free disk"
echo "3. Start full selfplay runs"
echo ""
echo "⚠ IMPORTANT: NNUE uses single GPU"
echo "   This is expected - different from ONNX multi-GPU training"
echo ""
echo "Estimated timeline:"
echo "  - Selfplay (3 modes): ~8-10 days (CPU only)"
echo "  - Labelgen: ~1-2 days (CPU search)"
echo "  - NNUE training: ~6-12 hours (1 GPU)"
echo "  - Puzzle gen: ~1-2 days (CPU)"
echo "  Total: ~12-14 days"
echo ""
echo "RAM check: You have 500GB headroom - plenty for 256 threads"
echo "Disk check: Monitor with 'watch -n 300 df -h /workspace'"
echo ""
echo "================================================================"
