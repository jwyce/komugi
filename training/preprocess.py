import math
import re
import sys

import orjson

import numpy as np

from model import (
    NUM_INPUT_PLANES,
    PIECE_KANJI_TO_INDEX,
    POLICY_SIZE,
    board_move_to_index,
    drop_move_to_index,
)

ENCODING_SIZE = NUM_INPUT_PLANES * 9 * 9
COORD_RE = re.compile(r"\((\d+)-(\d+)-(\d+)\)")


def san_to_policy_index(san: str):
    clean = san.strip().rstrip("#=")
    if clean.endswith("\u7d42"):
        clean = clean[:-1]
    if not clean:
        return None
    is_drop = clean.startswith("\u65b0")
    if is_drop:
        clean = clean[1:]
    if not clean:
        return None
    piece_kanji = clean[0]
    piece_type_idx = PIECE_KANJI_TO_INDEX.get(piece_kanji)
    if piece_type_idx is None:
        return None
    coords = [(int(r), int(f), int(t)) for r, f, t in COORD_RE.findall(clean)]
    if not coords:
        return None
    if is_drop or len(coords) == 1:
        to_rank, to_file, _ = coords[-1]
        try:
            return drop_move_to_index(piece_type_idx, to_rank, to_file)
        except ValueError:
            return None
    from_rank, from_file, _ = coords[0]
    to_rank, to_file, _ = coords[-1]
    try:
        return board_move_to_index(from_rank, from_file, to_rank, to_file)
    except ValueError:
        return None


def main():
    input_paths = sys.argv[1]
    output_dir = sys.argv[2]

    paths = [p.strip() for p in input_paths.split(",") if p.strip()]

    n = 0
    for path in paths:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    n += 1
    print(f"Preprocessing {n} positions from {len(paths)} file(s)...", flush=True)

    import os
    import shutil

    tmp_dir = output_dir + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    enc = np.memmap(
        f"{tmp_dir}/encodings.npy",
        dtype="float32",
        mode="w+",
        shape=(n, ENCODING_SIZE),
    )
    pol = np.memmap(
        f"{tmp_dir}/policies.npy",
        dtype="float32",
        mode="w+",
        shape=(n, POLICY_SIZE),
    )
    val = np.memmap(f"{tmp_dir}/values.npy", dtype="float32", mode="w+", shape=(n,))

    i = 0
    for path in paths:
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = orjson.loads(line)
                enc[i] = record["encoding"]

                policy = np.zeros(POLICY_SIZE, dtype="float32")
                for entry in record.get("policy", []):
                    if isinstance(entry, list) and len(entry) == 2:
                        san, prob = entry
                    elif isinstance(entry, dict):
                        san, prob = entry.get("san"), entry.get("prob")
                    else:
                        continue
                    if not isinstance(san, str) or prob is None:
                        continue
                    try:
                        prob = float(prob)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(prob) or prob < 0:
                        continue
                    idx = san_to_policy_index(san)
                    if idx is not None:
                        policy[idx] += prob
                total = policy.sum()
                if total > 0:
                    policy /= total
                pol[i] = policy

                val[i] = float(record["outcome"])
                i += 1
                if i % 100000 == 0:
                    print(f"  {i}/{n} ({100 * i // n}%)", flush=True)

    enc.flush()
    pol.flush()
    val.flush()

    np.save(f"{tmp_dir}/meta.npy", np.array([n, ENCODING_SIZE, POLICY_SIZE]))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.rename(tmp_dir, output_dir)
    print(f"Done: {n} positions -> {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
