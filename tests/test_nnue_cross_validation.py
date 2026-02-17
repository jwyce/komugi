#!/usr/bin/env python3
"""
Cross-validation test: Verify Rust and Python NNUE feature extraction produce identical results.

Reads FENs from tests/fixtures/nnue_test_positions.txt, extracts features using both
Rust (via cargo run) and Python (via nnue_features module), and compares them.
"""

import subprocess
import json
import sys
from pathlib import Path

# Add training directory to path so we can import nnue_features
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

import nnue_features


def run_rust_extraction(fen_file: str) -> dict:
    """
    Run the Rust binary to extract features from all FENs in a file.
    Returns dict: {fen -> {"white": [...], "black": [...]}}
    """
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "nnue_features",
            "--",
            fen_file,
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Rust binary failed: {result.stderr}")
        sys.exit(1)

    # Parse JSON output
    try:
        data = json.loads(result.stdout)
        rust_features = {}
        for item in data:
            fen = item["fen"]
            rust_features[fen] = {
                "white": item["white"],
                "black": item["black"],
            }
        return rust_features
    except json.JSONDecodeError as e:
        print(f"Failed to parse Rust output as JSON: {e}")
        print(f"Output: {result.stdout}")
        sys.exit(1)


def read_test_fens(fen_file: str) -> list[str]:
    """Read FENs from fixture file, skipping comments and empty lines."""
    fens = []
    with open(fen_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                fens.append(line)
    return fens


def main():
    fen_file = Path(__file__).parent / "fixtures" / "nnue_test_positions.txt"

    if not fen_file.exists():
        print(f"FEN fixture file not found: {fen_file}")
        sys.exit(1)

    print(f"Reading FENs from {fen_file}")
    fens = read_test_fens(str(fen_file))
    print(f"Found {len(fens)} test positions\n")

    # Extract features using Rust
    print("Running Rust feature extraction...")
    rust_features = run_rust_extraction(str(fen_file))
    print(f"Rust extracted {len(rust_features)} positions\n")

    # Extract features using Python
    print("Running Python feature extraction...")
    python_features = {}
    for fen in fens:
        try:
            white_features = nnue_features.extract_features(fen, "w")
            black_features = nnue_features.extract_features(fen, "b")
            python_features[fen] = {
                "white": white_features,
                "black": black_features,
            }
        except Exception as e:
            print(f"Python extraction failed for FEN: {fen}")
            print(f"Error: {e}")
            sys.exit(1)

    print(f"Python extracted {len(python_features)} positions\n")

    # Compare results
    print("Comparing Rust vs Python features...")
    mismatches = []

    for fen in fens:
        if fen not in rust_features:
            print(f"ERROR: FEN not in Rust results: {fen}")
            sys.exit(1)

        if fen not in python_features:
            print(f"ERROR: FEN not in Python results: {fen}")
            sys.exit(1)

        rust_white = rust_features[fen]["white"]
        rust_black = rust_features[fen]["black"]
        python_white = python_features[fen]["white"]
        python_black = python_features[fen]["black"]

        # Compare white perspective
        if rust_white != python_white:
            mismatches.append(
                {
                    "fen": fen,
                    "perspective": "white",
                    "rust_count": len(rust_white),
                    "python_count": len(python_white),
                    "rust_features": rust_white[:10],  # First 10 for debugging
                    "python_features": python_white[:10],
                }
            )

        # Compare black perspective
        if rust_black != python_black:
            mismatches.append(
                {
                    "fen": fen,
                    "perspective": "black",
                    "rust_count": len(rust_black),
                    "python_count": len(python_black),
                    "rust_features": rust_black[:10],  # First 10 for debugging
                    "python_features": python_black[:10],
                }
            )

    # Report results
    print(f"\n{'=' * 70}")
    if mismatches:
        print(f"FAILED: {len(mismatches)} mismatches found\n")
        for i, mismatch in enumerate(mismatches, 1):
            print(f"Mismatch {i}:")
            print(f"  FEN: {mismatch['fen']}")
            print(f"  Perspective: {mismatch['perspective']}")
            print(f"  Rust count: {mismatch['rust_count']}")
            print(f"  Python count: {mismatch['python_count']}")
            print(f"  Rust (first 10): {mismatch['rust_features']}")
            print(f"  Python (first 10): {mismatch['python_features']}")
            print()
        sys.exit(1)
    else:
        print(f"✓ SUCCESS: All {len(fens)} positions validated successfully!")
        print(f"  Rust and Python feature extraction produce identical results.")
        sys.exit(0)


if __name__ == "__main__":
    main()
