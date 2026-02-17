#!/usr/bin/env python3
"""
Validate JSONL label generation output.

Schema:
{
  "fen": "string — full gungi FEN",
  "outcome": "float — 1.0 (white win), -1.0 (black win), 0.0 (draw)",
  "search_value": "float — MCTS root value at 5000 sims",
  "policy": [["move_san", probability], ...],
  "move_number": "int — ply in original game"
}
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_fen(fen: str) -> bool:
    """Check if FEN is parseable (basic validation)."""
    if not isinstance(fen, str) or not fen.strip():
        return False

    parts = fen.split()
    # Gungi FEN has specific structure - at minimum should have multiple parts
    return len(parts) >= 2


def validate_record(record: Dict[str, Any], line_num: int) -> Tuple[bool, List[str]]:
    """
    Validate a single label record.

    Returns: (is_valid, list_of_errors)
    """
    errors = []

    # Check all required fields present
    required_fields = {"fen", "outcome", "search_value", "policy", "move_number"}
    missing = required_fields - set(record.keys())
    if missing:
        errors.append(f"Line {line_num}: Missing fields: {missing}")
        return False, errors

    # Validate FEN
    if not isinstance(record["fen"], str):
        errors.append(
            f"Line {line_num}: 'fen' must be string, got {type(record['fen']).__name__}"
        )
    elif not validate_fen(record["fen"]):
        errors.append(f"Line {line_num}: 'fen' is not parseable: {record['fen'][:50]}")

    # Validate outcome
    if not isinstance(record["outcome"], (int, float)):
        errors.append(
            f"Line {line_num}: 'outcome' must be float, got {type(record['outcome']).__name__}"
        )
    elif record["outcome"] not in {-1.0, 0.0, 1.0}:
        errors.append(
            f"Line {line_num}: 'outcome' must be in {{-1.0, 0.0, 1.0}}, got {record['outcome']}"
        )

    # Validate search_value
    if not isinstance(record["search_value"], (int, float)):
        errors.append(
            f"Line {line_num}: 'search_value' must be float, got {type(record['search_value']).__name__}"
        )
    elif not (-50000.0 <= record["search_value"] <= 50000.0):
        errors.append(
            f"Line {line_num}: 'search_value' must be finite in [-50000, 50000], got {record['search_value']}"
        )
    elif not (
        isinstance(record["search_value"], float)
        or isinstance(record["search_value"], int)
    ):
        errors.append(f"Line {line_num}: 'search_value' is not finite")

    # Validate policy
    if not isinstance(record["policy"], list):
        errors.append(
            f"Line {line_num}: 'policy' must be list, got {type(record['policy']).__name__}"
        )
    else:
        policy_sum = 0.0
        for i, item in enumerate(record["policy"]):
            if not isinstance(item, list) or len(item) != 2:
                errors.append(
                    f"Line {line_num}: policy[{i}] must be [move_san, probability], got {item}"
                )
                continue

            move_san, prob = item
            if not isinstance(move_san, str):
                errors.append(
                    f"Line {line_num}: policy[{i}][0] (move_san) must be string, got {type(move_san).__name__}"
                )
            if not isinstance(prob, (int, float)):
                errors.append(
                    f"Line {line_num}: policy[{i}][1] (probability) must be float, got {type(prob).__name__}"
                )
            elif prob < 0.0 or prob > 1.0:
                errors.append(
                    f"Line {line_num}: policy[{i}][1] probability must be in [0.0, 1.0], got {prob}"
                )
            else:
                policy_sum += prob

        # Check probabilities sum to ~1.0 (allow small floating point error)
        if record["policy"] and abs(policy_sum - 1.0) > 0.01:
            errors.append(
                f"Line {line_num}: policy probabilities sum to {policy_sum:.4f}, expected ~1.0"
            )

    # Validate move_number
    if not isinstance(record["move_number"], int):
        errors.append(
            f"Line {line_num}: 'move_number' must be int, got {type(record['move_number']).__name__}"
        )
    elif record["move_number"] < 0:
        errors.append(
            f"Line {line_num}: 'move_number' must be non-negative, got {record['move_number']}"
        )

    # Check for unexpected fields (encoding should NOT be present)
    unexpected = set(record.keys()) - required_fields
    if unexpected:
        errors.append(f"Line {line_num}: Unexpected fields: {unexpected}")

    return len(errors) == 0, errors


def validate_jsonl(filepath: str) -> Tuple[int, int, List[str]]:
    """
    Validate JSONL file.

    Returns: (valid_count, total_count, list_of_errors)
    """
    path = Path(filepath)
    if not path.exists():
        return 0, 0, [f"File not found: {filepath}"]

    valid_count = 0
    total_count = 0
    all_errors = []

    try:
        with open(path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                total_count += 1
                line = line.strip()

                if not line:
                    all_errors.append(f"Line {line_num}: Empty line")
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    all_errors.append(f"Line {line_num}: Invalid JSON: {e}")
                    continue

                is_valid, errors = validate_record(record, line_num)
                if is_valid:
                    valid_count += 1
                else:
                    all_errors.extend(errors)

    except IOError as e:
        return 0, 0, [f"Error reading file: {e}"]

    return valid_count, total_count, all_errors


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_labels.py <path/to/labels.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]
    valid_count, total_count, errors = validate_jsonl(filepath)

    if errors:
        for error in errors:
            print(error)
        print()

    print(f"{valid_count}/{total_count} records valid")

    # Exit with non-zero if any errors or file not found
    sys.exit(0 if (valid_count == total_count and total_count > 0) else 1)


if __name__ == "__main__":
    main()
