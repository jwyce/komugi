"""
NNUE training script for gungi.

Reads label-gen JSONL (fen, search_value, outcome), extracts sparse features
via nnue_features.py, trains with lambda-interpolated value targets.

Usage:
    python training/train_nnue.py --data labels.jsonl --epochs 100
"""

import argparse
import csv
import importlib
import json
import math
import random
from pathlib import Path

np = importlib.import_module("numpy")
torch = importlib.import_module("torch")
F = importlib.import_module("torch.nn.functional")
_torch_utils_data = importlib.import_module("torch.utils.data")
DataLoader = _torch_utils_data.DataLoader
Dataset = _torch_utils_data.Dataset

from nnue_features import TOTAL_FEATURES, extract_features
from nnue_model import NnueModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_value(sv, K: float = 400.0):
    """Normalize search value: 2/(1+exp(-sv/K)) - 1.

    For MCTS Q-values in [-1,1] with default K=400 this compresses values
    near zero. Decrease K (e.g. 1.0) for stronger signal from search_value.
    """
    return 2.0 / (1.0 + torch.exp(-sv / K)) - 1.0


# ============================================================================
# Dataset
# ============================================================================


class NnueDataset(Dataset):
    """Dataset for NNUE training from label-gen JSONL.

    Loads all positions into memory. For each position extracts sparse
    feature indices for both perspectives, stores compactly as lists.
    Dense tensors are created on-the-fly in __getitem__.
    """

    def __init__(self, data_paths: list) -> None:
        self.white_indices = []
        self.black_indices = []
        _stm = []
        _sv = []
        _outcome = []

        for path in data_paths:
            self._load_file(path, _stm, _sv, _outcome)

        self._stm = np.array(_stm, dtype=np.float32)
        self._sv = np.array(_sv, dtype=np.float32)
        self._outcome = np.array(_outcome, dtype=np.float32)
        print(f"loaded {len(self)} positions from {len(data_paths)} file(s)")

    def _load_file(self, path: str, _stm: list, _sv: list, _outcome: list) -> None:
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                fen = record["fen"]

                # Extract sparse features for both perspectives
                w_idx = extract_features(fen, "w")
                b_idx = extract_features(fen, "b")

                # Side to move from FEN (3rd field)
                turn = fen.split()[2]
                stm_val = 0.0 if turn == "w" else 1.0

                # Search value is already from STM perspective
                sv = float(record["search_value"])

                # Outcome from white perspective -> convert to STM
                outcome = float(record["outcome"])
                if turn == "b":
                    outcome = -outcome

                self.white_indices.append(w_idx)
                self.black_indices.append(b_idx)
                _stm.append(stm_val)
                _sv.append(sv)
                _outcome.append(outcome)
                count += 1
        print(f"  {path}: {count} positions")

    def __len__(self) -> int:
        return len(self._stm)

    def __getitem__(self, idx: int):
        # Scatter sparse indices into dense binary tensor
        w_feat = torch.zeros(TOTAL_FEATURES, dtype=torch.float32)
        for i in self.white_indices[idx]:
            w_feat[i] = 1.0

        b_feat = torch.zeros(TOTAL_FEATURES, dtype=torch.float32)
        for i in self.black_indices[idx]:
            b_feat[i] = 1.0

        return (
            w_feat,
            b_feat,
            torch.tensor(self._stm[idx], dtype=torch.float32),
            torch.tensor(self._sv[idx], dtype=torch.float32),
            torch.tensor(self._outcome[idx], dtype=torch.float32),
        )


# ============================================================================
# Training / Evaluation
# ============================================================================


def train_epoch(model, loader, optimizer, device, lam: float, K: float):
    model.train()
    total_loss = 0.0
    count = 0

    for w_feat, b_feat, stm, sv, outcome in loader:
        w_feat = w_feat.to(device)
        b_feat = b_feat.to(device)
        stm = stm.to(device)
        sv = sv.to(device)
        outcome = outcome.to(device)

        pred = model(w_feat, b_feat, stm)
        target = lam * normalize_value(sv, K) + (1.0 - lam) * outcome
        loss = F.mse_loss(pred, target)

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"NaN/Inf loss detected: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def evaluate(model, loader, device, lam: float, K: float):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for w_feat, b_feat, stm, sv, outcome in loader:
            w_feat = w_feat.to(device)
            b_feat = b_feat.to(device)
            stm = stm.to(device)
            sv = sv.to(device)
            outcome = outcome.to(device)

            pred = model(w_feat, b_feat, stm)
            target = lam * normalize_value(sv, K) + (1.0 - lam) * outcome
            loss = F.mse_loss(pred, target)

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(model, optimizer, scheduler, epoch, output_dir: Path, name=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (name or f"nnue_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )
    return path


# ============================================================================
# CLI
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NNUE from label-gen JSONL")
    parser.add_argument("--data", required=True, nargs="+", help="JSONL file(s)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.75,
        help="Blend weight: lambda*normalize(sv) + (1-lambda)*outcome",
    )
    parser.add_argument(
        "--K",
        type=float,
        default=400.0,
        help="Normalization temperature (decrease for MCTS Q-values in [-1,1])",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output", default="training/nnue_checkpoints")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    return parser


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    # Load data
    dataset = NnueDataset(args.data)
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Model + optimizer
    model = NnueModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"resumed from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    # CSV metrics
    metrics_path = output_dir / "nnue_metrics.csv"
    metrics_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    csv_file = open(metrics_path, "a" if metrics_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not metrics_exists:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        csv_file.flush()

    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"NNUE training: {train_size} train, {val_size} val, "
        f"{param_count:,} params, {args.epochs} epochs"
    )
    print(f"  lambda={args.lam}, K={args.K}, lr={args.lr}, batch={args.batch_size}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, args.lam, args.K
        )
        scheduler.step()

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, args.lam, args.K)

        lr = scheduler.get_last_lr()[0]
        msg = f"epoch={epoch} train_loss={train_loss:.6f} lr={lr:.8f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.6f}"
        print(msg)

        csv_writer.writerow(
            [epoch, train_loss, val_loss if val_loss is not None else "", lr]
        )
        csv_file.flush()

        if epoch % args.save_every == 0:
            path = save_checkpoint(model, optimizer, scheduler, epoch, output_dir)
            print(f"saved: {path}")

    # Final checkpoint
    final_path = save_checkpoint(
        model, optimizer, scheduler, args.epochs, output_dir, name="nnue_final.pt"
    )
    print(f"saved final: {final_path}")
    csv_file.close()


if __name__ == "__main__":
    main()
