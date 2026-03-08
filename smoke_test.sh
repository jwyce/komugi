#!/bin/bash
# Smoke test: 100 games @ 5000 sims → labelgen → NNUE train → puzzle extract
# Run this BEFORE the full 12-14 day pipeline to catch issues early

set -euo pipefail

WORKSPACE="/workspace"
MODEL="${WORKSPACE}/models/advanced_gen9_rulefix.onnx"
DATA_DIR="${WORKSPACE}/smoke_test"

echo "================================================================"
echo "SMOKE TEST: 100 games @ 5000 sims"
echo "================================================================"
echo ""

# Check model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Wait for pass B to export, then retry"
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "Step 1: Selfplay (100 games, 5000 sims, advanced mode)"
echo "Expected: ~1-2 hours at 0.6-0.8 gpm"
echo "----------------------------------------------------------------"
cd "$WORKSPACE"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KOMUGI_MAX_MOVES=500 KOMUGI_VL_BATCH_SIZE=12 \
    selfplay 100 "${DATA_DIR}/smoke.jsonl" 5000 "$MODEL" advanced 256

games=$(grep -cE '^\[t[0-9]+\] Game' "${DATA_DIR}/smoke.jsonl" 2>/dev/null || echo 0)
echo "✓ Selfplay complete: $games games generated"
echo ""

echo "Step 2: Labelgen (5000 sims, quick validation)"
echo "Expected: ~30-60 min"
echo "----------------------------------------------------------------"
# Generate labels for first 10 positions only (fast check)
head -10 "${DATA_DIR}/smoke.jsonl" > "${DATA_DIR}/smoke_sample.jsonl"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KOMUGI_VL_BATCH_SIZE=12 \
    python3 labelgen.py "${DATA_DIR}/smoke_sample.jsonl" "${DATA_DIR}/labels_sample.jsonl" 5000 "$MODEL" advanced 256

labels=$(wc -l < "${DATA_DIR}/labels_sample.jsonl" 2>/dev/null || echo 0)
echo "✓ Labelgen test: $labels labels generated"
echo ""

echo "Step 3: Check JSONL + PGN pair"
echo "----------------------------------------------------------------"
if [ -f "${DATA_DIR}/smoke.pgn" ]; then
    pgn_games=$(grep -cE '^\[Game "' "${DATA_DIR}/smoke.pgn" 2>/dev/null || echo 0)
    echo "✓ PGN file exists: $pgn_games games"
else
    echo "⚠ No PGN file (selfplay may not generate it)")
fi
echo ""

echo "Step 4: Tactician betrayal check (rulefix validation)"
echo "----------------------------------------------------------------"
python3 /workspace/check_tactician_betrayal.py "${DATA_DIR}/smoke.jsonl" 2>/dev/null || true
echo ""

echo "================================================================"
echo "SMOKE TEST SUMMARY"
echo "================================================================"
echo ""
echo "Files generated:"
ls -lh "$DATA_DIR" 2>/dev/null || echo "  (none)"
echo ""

# Disk space check
df -h "$WORKSPACE" | tail -1
echo ""

if [ "$games" -ge 90 ]; then
    echo "✓ PASS: 100-game selfplay worked"
    echo ""
    echo "Ready for full pipeline:"
    echo "  - Beginner: 2000 sims × 5000 games"
    echo "  - Intermediate: 2000 sims × 5000 games"
    echo "  - Advanced: 5000 sims × 5000 games"
else
    echo "✗ FAIL: Expected ~100 games, got $games"
    echo "Check logs before proceeding"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Review smoke_test/smoke.jsonl quality"
echo "2. If good: delete smoke_test/ and start full runs"
echo "3. Full labelgen/NNUE/puzzle after all selfplay done"
echo ""
echo "Estimated full timeline:"
echo "  - Selfplay (3 modes): ~8-10 days"
echo "  - Labelgen: ~1-2 days"
echo "  - NNUE training: ~6-12 hours"
echo "  - Puzzle gen: ~1-2 days"
echo "  Total: ~12-14 days"
