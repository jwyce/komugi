#!/usr/bin/env python3

import argparse
import glob
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Candidate:
    score: float
    fen: str
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an initial puzzle dataset from self-play JSONL records."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input JSONL files or glob patterns (e.g. /workspace/data/*.jsonl)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=50000, help="Max puzzles to write")
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.12,
        help="Minimum policy probability gap between top1 and top2",
    )
    parser.add_argument(
        "--min-top1",
        type=float,
        default=0.20,
        help="Minimum top1 policy probability",
    )
    parser.add_argument(
        "--min-move-number",
        type=int,
        default=6,
        help="Ignore positions before this move number",
    )
    parser.add_argument(
        "--no-drop",
        action="store_true",
        help="Exclude drop-move puzzles",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Exclude capture-move puzzles",
    )
    parser.add_argument(
        "--allow-other",
        action="store_true",
        help="Allow non-drop/non-capture solution moves",
    )
    parser.add_argument(
        "--include-draws",
        action="store_true",
        help="Include draw-labeled records (default: decisive only)",
    )
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        elif Path(pattern).exists():
            files.append(pattern)
    return sorted(set(files))


def top_two_policy(
    policy: list[Any],
) -> tuple[tuple[str, float], tuple[str, float]] | None:
    parsed: list[tuple[str, float]] = []
    for item in policy:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        san = str(item[0])
        try:
            prob = float(item[1])
        except (TypeError, ValueError):
            continue
        parsed.append((san, prob))
    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda x: x[1], reverse=True)
    return parsed[0], parsed[1]


def classify_move(san: str) -> str:
    if san.startswith("新"):
        return "drop"
    if "取" in san:
        return "capture"
    return "other"


def difficulty_from_gap(gap: float) -> str:
    if gap >= 0.35:
        return "easy"
    if gap >= 0.22:
        return "medium"
    return "hard"


def candidate_score(gap: float, top1: float, move_type: str, move_number: int) -> float:
    score = gap * 1.8 + top1
    if move_type == "capture":
        score += 0.10
    elif move_type == "drop":
        score += 0.08
    score += min(move_number, 150) / 2000.0
    return score


def main() -> int:
    args = parse_args()
    files = expand_inputs(args.input)
    if not files:
        raise SystemExit("No input files found")

    heap: list[tuple[float, int, Candidate]] = []
    dedup_fens: set[str] = set()
    seen = 0
    kept = 0

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                seen += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fen = str(rec.get("fen", ""))
                if not fen or fen in dedup_fens:
                    continue

                try:
                    move_number = int(rec.get("move_number", 0))
                except (TypeError, ValueError):
                    continue
                if move_number < args.min_move_number:
                    continue

                try:
                    outcome = float(rec.get("outcome", 0.0))
                except (TypeError, ValueError):
                    continue
                if not args.include_draws and abs(outcome) < 0.5:
                    continue

                policy = rec.get("policy")
                if not isinstance(policy, list):
                    continue
                top_two = top_two_policy(policy)
                if not top_two:
                    continue
                (top1_san, top1_prob), (top2_san, top2_prob) = top_two

                gap = top1_prob - top2_prob
                if top1_prob < args.min_top1 or gap < args.min_gap:
                    continue

                move_type = classify_move(top1_san)
                allowed = (
                    (move_type == "drop" and not args.no_drop)
                    or (move_type == "capture" and not args.no_capture)
                    or (move_type == "other" and args.allow_other)
                )
                if not allowed:
                    continue

                score = candidate_score(gap, top1_prob, move_type, move_number)
                puzzle = {
                    "fen": fen,
                    "solution": top1_san,
                    "played_move": str(rec.get("played_move", "")),
                    "runner_up": top2_san,
                    "policy_top1": round(top1_prob, 6),
                    "policy_top2": round(top2_prob, 6),
                    "policy_gap": round(gap, 6),
                    "move_type": move_type,
                    "difficulty": difficulty_from_gap(gap),
                    "outcome": outcome,
                    "move_number": move_number,
                    "source": Path(file_path).name,
                }
                cand = Candidate(score=score, fen=fen, record=puzzle)

                if len(heap) < args.limit:
                    heapq.heappush(heap, (score, kept, cand))
                    dedup_fens.add(fen)
                    kept += 1
                    continue

                if heap and score > heap[0][0]:
                    _, _, evicted = heapq.heapreplace(heap, (score, kept, cand))
                    dedup_fens.discard(evicted.fen)
                    dedup_fens.add(fen)
                    kept += 1

    ranked = [entry[2] for entry in heap]
    ranked.sort(key=lambda c: c.score, reverse=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for cand in ranked:
            out.write(json.dumps(cand.record, ensure_ascii=False) + "\n")

    print(f"Scanned records: {seen}")
    print(f"Puzzles written: {len(ranked)}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
