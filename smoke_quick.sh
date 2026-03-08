#!/bin/bash
# Quick ad-hoc smoke test (no file writes, just validation)
# Run this anytime to check if selfplay works at 5000 sims

MODEL="/workspace/models/advanced_gen9_rulefix.onnx"

echo "Quick selfplay smoke test (10 games, 5000 sims)"
echo "================================================"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: $MODEL not found"
    exit 1
fi

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KOMUGI_MAX_MOVES=500 KOMUGI_VL_BATCH_SIZE=12 \
    timeout 1800 selfplay 10 /tmp/smoke_quick.jsonl 5000 "$MODEL" advanced 256

if [ $? -eq 0 ] && [ -f /tmp/smoke_quick.jsonl ]; then
    games=$(grep -cE '^\[t[0-9]+\] Game' /tmp/smoke_quick.jsonl)
    echo "✓ Success: $games games in ~30 min (expected ~0.6 gpm)"
    rm /tmp/smoke_quick.jsonl
else
    echo "✗ Failed or timed out"
    exit 1
fi
