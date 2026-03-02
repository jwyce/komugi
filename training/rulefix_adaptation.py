#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import subprocess
from pathlib import Path


MODES = ("beginner", "intermediate", "advanced")


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def best_checkpoint(
    ckpt_dir: Path, policy_w: float = 1.0, value_w: float = 1.0
) -> Path | None:
    metrics = ckpt_dir / "metrics.csv"
    if not metrics.exists():
        return latest_checkpoint(ckpt_dir)

    best_score = math.inf
    best_path: Path | None = None
    with metrics.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                epoch = int(float((row.get("epoch") or "").strip()))
                val_policy = float((row.get("val_policy_loss") or "").strip())
                val_value = float((row.get("val_value_loss") or "").strip())
            except ValueError:
                continue
            if not (math.isfinite(val_policy) and math.isfinite(val_value)):
                continue
            candidate = ckpt_dir / f"model_epoch_{epoch}.pt"
            if not candidate.exists():
                continue
            score = policy_w * val_policy + value_w * val_value
            if score < best_score:
                best_score = score
                best_path = candidate

    return best_path if best_path is not None else latest_checkpoint(ckpt_dir)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    checkpoints = sorted(
        ckpt_dir.glob("model_epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return checkpoints[0] if checkpoints else None


def detect_num_gpus() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except Exception:
        return 1
    lines = [line for line in out.splitlines() if line.strip()]
    return max(1, len(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-gen9 rulefix adaptation for all modes"
    )
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--tag", default="gen9_rulefix")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--games-beginner", type=int, default=1200)
    parser.add_argument("--games-intermediate", type=int, default=1200)
    parser.add_argument("--games-advanced", type=int, default=1500)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--source-model-template", default="{models_dir}/{mode}_gen9.onnx"
    )
    parser.add_argument(
        "--source-checkpoint-template",
        default="",
        help="Optional .pt warm-start path template, e.g. {checkpoints_dir}/{mode}_gen9/model_epoch_50.pt",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workspace = Path(args.workspace)
    data_dir = workspace / "data"
    models_dir = workspace / "models"
    checkpoints_dir = workspace / "checkpoints"
    train_dir = workspace / "training"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    games_by_mode = {
        "beginner": args.games_beginner,
        "intermediate": args.games_intermediate,
        "advanced": args.games_advanced,
    }

    gpus = detect_num_gpus()
    print(f"Detected GPUs: {gpus}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Tag: {args.tag}")

    for mode in args.modes:
        games = games_by_mode[mode]
        gen_label = f"{mode}_{args.tag}"
        model_in = Path(
            args.source_model_template.format(
                mode=mode,
                models_dir=models_dir,
                checkpoints_dir=checkpoints_dir,
                workspace=workspace,
                tag=args.tag,
            )
        )
        if not model_in.exists() and not args.dry_run:
            raise FileNotFoundError(f"Missing source model for {mode}: {model_in}")

        data_file = data_dir / f"{gen_label}.jsonl"
        prep_dir = data_dir / f"{gen_label}_preprocessed"
        ckpt_dir = checkpoints_dir / gen_label
        out_model = models_dir / f"{gen_label}.onnx"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 72)
        print(f"[{gen_label}] self-play")
        run_cmd(
            [
                "selfplay",
                str(games),
                str(data_file),
                str(args.sims),
                str(model_in),
                mode,
                str(args.threads),
            ],
            cwd=workspace,
            dry_run=args.dry_run,
        )

        print(f"[{gen_label}] preprocess")
        run_cmd(
            ["python", "preprocess.py", str(data_file), str(prep_dir)],
            cwd=train_dir,
            dry_run=args.dry_run,
        )

        resume_arg: list[str] = []
        if args.source_checkpoint_template:
            resume_path = Path(
                args.source_checkpoint_template.format(
                    mode=mode,
                    models_dir=models_dir,
                    checkpoints_dir=checkpoints_dir,
                    workspace=workspace,
                    tag=args.tag,
                )
            )
            if resume_path.exists() or args.dry_run:
                resume_arg = ["--resume", str(resume_path)]

        print(f"[{gen_label}] train")
        train_cmd = [
            "python",
            "train.py",
            "--data",
            str(prep_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--device",
            str(args.device),
            "--save-every",
            str(args.save_every),
            "--output-dir",
            str(ckpt_dir),
            "--num-blocks",
            str(args.num_blocks),
            "--channels",
            str(args.channels),
        ] + resume_arg

        if gpus > 1:
            train_cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(gpus),
            ] + train_cmd[1:]

        run_cmd(train_cmd, cwd=train_dir, dry_run=args.dry_run)

        chosen_ckpt = best_checkpoint(ckpt_dir)
        if chosen_ckpt is None and not args.dry_run:
            raise RuntimeError(f"No checkpoint produced for {gen_label}")

        print(f"[{gen_label}] export onnx")
        run_cmd(
            [
                "python",
                "export_onnx.py",
                "--checkpoint",
                str(chosen_ckpt or (ckpt_dir / "model_epoch_1.pt")),
                "--output",
                str(out_model),
                "--num-blocks",
                str(args.num_blocks),
                "--channels",
                str(args.channels),
            ],
            cwd=train_dir,
            dry_run=args.dry_run,
        )

        print(f"[{gen_label}] complete -> {out_model}")

    print("=" * 72)
    print("Rulefix adaptation complete")


if __name__ == "__main__":
    main()
