#!/bin/bash
#
# End-to-end smoke test for the full NNUE pipeline.
# Validates: label generation → training → export → Rust inference → WASM build.
#
# Usage: bash tests/test_nnue_e2e.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

E2E_DIR="/tmp/komugi_e2e_$$"
LABELS_FILE="$E2E_DIR/labels.jsonl"
CKPT_DIR="$E2E_DIR/ckpt"
NNUE_FILE="$E2E_DIR/test.nnue"

cleanup() {
    rm -rf "$E2E_DIR"
}
trap cleanup EXIT

mkdir -p "$E2E_DIR" "$CKPT_DIR"

PASS=0
FAIL=0
PHASE_START=0

phase_start() {
    echo ""
    echo "================================================================"
    echo "PHASE $1: $2"
    echo "================================================================"
    PHASE_START=$SECONDS
}

phase_end() {
    local elapsed=$((SECONDS - PHASE_START))
    echo "✓ Phase $1 passed (${elapsed}s)"
    PASS=$((PASS + 1))
}

phase_fail() {
    echo "✗ Phase $1 FAILED: $2"
    FAIL=$((FAIL + 1))
}

# ============================================================================
# Phase 1: Label Generation
# ============================================================================
phase_start 1 "Label Generation"

echo "Building labelgen binary..."
cargo build --release --bin labelgen 2>&1 | tail -5

echo "Generating labels: 3 games, 50 play sims, 100 eval sims, beginner, 2 threads..."
./target/release/labelgen 3 "$LABELS_FILE" 50 100 "" beginner 2

if [ ! -f "$LABELS_FILE" ]; then
    phase_fail 1 "labels file not created"
    exit 1
fi

LABEL_COUNT=$(wc -l < "$LABELS_FILE")
if [ "$LABEL_COUNT" -lt 1 ]; then
    phase_fail 1 "labels file is empty"
    exit 1
fi
echo "Generated $LABEL_COUNT labeled positions"

echo "Validating label schema..."
python3 training/validate_labels.py "$LABELS_FILE"

phase_end 1

# ============================================================================
# Phase 2: NNUE Training
# ============================================================================
phase_start 2 "NNUE Training"

echo "Training NNUE for 3 epochs..."
python3 training/train_nnue.py \
    --data "$LABELS_FILE" \
    --epochs 3 \
    --batch-size 32 \
    --output "$CKPT_DIR" \
    --device cpu \
    --val-split 0.0 \
    --num-workers 0 \
    --save-every 1

FINAL_CKPT="$CKPT_DIR/nnue_final.pt"
if [ ! -f "$FINAL_CKPT" ]; then
    phase_fail 2 "nnue_final.pt not created"
    exit 1
fi

CKPT_SIZE=$(stat -f%z "$FINAL_CKPT" 2>/dev/null || stat -c%s "$FINAL_CKPT" 2>/dev/null)
echo "Checkpoint: $FINAL_CKPT ($CKPT_SIZE bytes)"

phase_end 2

# ============================================================================
# Phase 3: Export to .nnue
# ============================================================================
phase_start 3 "Export to .nnue"

echo "Exporting checkpoint to .nnue binary..."
python3 training/export_nnue.py \
    --checkpoint "$FINAL_CKPT" \
    --output "$NNUE_FILE"

if [ ! -f "$NNUE_FILE" ]; then
    phase_fail 3 ".nnue file not created"
    exit 1
fi

NNUE_SIZE=$(stat -f%z "$NNUE_FILE" 2>/dev/null || stat -c%s "$NNUE_FILE" 2>/dev/null)
echo "NNUE file: $NNUE_FILE ($NNUE_SIZE bytes)"

EXPECTED_MIN=$((3 * 1024 * 1024))
EXPECTED_MAX=$((4 * 1024 * 1024))
if [ "$NNUE_SIZE" -lt "$EXPECTED_MIN" ] || [ "$NNUE_SIZE" -gt "$EXPECTED_MAX" ]; then
    phase_fail 3 ".nnue file size $NNUE_SIZE outside expected range [${EXPECTED_MIN}-${EXPECTED_MAX}]"
    exit 1
fi

phase_end 3

# ============================================================================
# Phase 4: Rust Inference Test
# ============================================================================
phase_start 4 "Rust Inference Test"

echo "Running NNUE E2E load + eval test..."
NNUE_TEST_PATH="$NNUE_FILE" cargo test -p komugi-engine -- nnue::test_e2e_load_and_eval --nocapture

phase_end 4

# ============================================================================
# Phase 5: WASM Build
# ============================================================================
phase_start 5 "WASM Build"

if ! command -v wasm-pack &>/dev/null; then
    echo "wasm-pack not found, skipping WASM build"
    echo "(install: cargo install wasm-pack)"
    PASS=$((PASS + 1))
else
    mkdir -p models
    cp "$NNUE_FILE" models/gungi.nnue
    echo "Copied .nnue to models/gungi.nnue"

    echo "Building WASM..."
    wasm-pack build crates/komugi-wasm --target web 2>&1 | tail -10

    if [ ! -f "crates/komugi-wasm/pkg/komugi_wasm_bg.wasm" ]; then
        phase_fail 5 "WASM output not found"
        exit 1
    fi

    WASM_SIZE=$(stat -f%z "crates/komugi-wasm/pkg/komugi_wasm_bg.wasm" 2>/dev/null || stat -c%s "crates/komugi-wasm/pkg/komugi_wasm_bg.wasm" 2>/dev/null)
    echo "WASM binary: $WASM_SIZE bytes"

    phase_end 5
fi

# ============================================================================
# Summary
# ============================================================================
TOTAL_ELAPSED=$((SECONDS))

echo ""
echo "================================================================"
echo "E2E SMOKE TEST COMPLETE"
echo "================================================================"
echo "Passed: $PASS / $((PASS + FAIL))"
echo "Time:   ${TOTAL_ELAPSED}s"
echo "Labels: $LABEL_COUNT positions"
echo "NNUE:   $NNUE_SIZE bytes"
echo "================================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi

exit 0
