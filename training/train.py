import argparse
import csv
import glob
import importlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

np = importlib.import_module("numpy")
torch = importlib.import_module("torch")
F = importlib.import_module("torch.nn.functional")
_torch_utils_data = importlib.import_module("torch.utils.data")
DataLoader = _torch_utils_data.DataLoader
Dataset = _torch_utils_data.Dataset
DistributedSampler = importlib.import_module(
    "torch.utils.data.distributed"
).DistributedSampler
DDP = importlib.import_module("torch.nn.parallel").DistributedDataParallel
dist = importlib.import_module("torch.distributed")

from model import (
    NUM_INPUT_PLANES,
    PIECE_KANJI_TO_INDEX,
    POLICY_SIZE,
    GungiNet,
    board_move_to_index,
    drop_move_to_index,
)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return local_rank() == 0


COORD_RE = re.compile(r"\((\d+)-(\d+)-(\d+)\)")


def build_mirror_permutation() -> list[int]:
    perm = list(range(POLICY_SIZE))
    board_size = 9
    num_squares = board_size * board_size
    drop_offset = num_squares * num_squares

    # Board move index: from_sq * 81 + to_sq.
    for from_rank in range(board_size):
        for from_file in range(board_size):
            for to_rank in range(board_size):
                for to_file in range(board_size):
                    from_sq = from_rank * board_size + from_file
                    to_sq = to_rank * board_size + to_file
                    orig_idx = from_sq * num_squares + to_sq
                    mir_from_sq = from_rank * board_size + (board_size - 1 - from_file)
                    mir_to_sq = to_rank * board_size + (board_size - 1 - to_file)
                    mir_idx = mir_from_sq * num_squares + mir_to_sq
                    perm[orig_idx] = mir_idx

    # Drop move index: 6561 + piece * 81 + to_sq.
    for piece in range(14):
        for to_rank in range(board_size):
            for to_file in range(board_size):
                to_sq = to_rank * board_size + to_file
                orig_idx = drop_offset + piece * num_squares + to_sq
                mir_to_sq = to_rank * board_size + (board_size - 1 - to_file)
                mir_idx = drop_offset + piece * num_squares + mir_to_sq
                perm[orig_idx] = mir_idx

    return perm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def san_to_policy_index(san: str) -> int | None:
    clean = san.strip()
    clean = clean.rstrip("#=")
    if clean.endswith("終"):
        clean = clean[:-1]
    if not clean:
        return None

    is_drop = clean.startswith("新")
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


def parse_policy_entry(entry: Any) -> tuple[str, float] | None:
    if isinstance(entry, list) and len(entry) == 2:
        san, prob = entry
    elif isinstance(entry, tuple) and len(entry) == 2:
        san, prob = entry
    elif isinstance(entry, dict):
        san = entry.get("san")
        prob = entry.get("prob")
    else:
        return None

    if not isinstance(san, str):
        return None
    if prob is None:
        return None
    try:
        prob_value = float(prob)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(prob_value) or prob_value < 0:
        return None
    return san, prob_value


class SelfPlayDataset(Dataset):
    def __init__(self, preprocessed_dir: str, augment_symmetry: bool = True) -> None:
        self.augment_symmetry = augment_symmetry
        self.mirror_perm = torch.tensor(build_mirror_permutation(), dtype=torch.long)
        meta = np.load(f"{preprocessed_dir}/meta.npy")
        n = int(meta[0])
        enc_size = int(meta[1])
        pol_size = int(meta[2])
        self.encodings = np.memmap(
            f"{preprocessed_dir}/encodings.npy",
            dtype="float32",
            mode="r",
            shape=(n, enc_size),
        )
        self.policies = np.memmap(
            f"{preprocessed_dir}/policies.npy",
            dtype="float32",
            mode="r",
            shape=(n, pol_size),
        )
        self.values = np.memmap(
            f"{preprocessed_dir}/values.npy",
            dtype="float32",
            mode="r",
            shape=(n,),
        )
        self._len = n
        print(f"loaded preprocessed data: {n} positions from {preprocessed_dir}")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        encoding = torch.from_numpy(self.encodings[idx].copy()).reshape(
            NUM_INPUT_PLANES, 9, 9
        )
        policy_target = torch.from_numpy(self.policies[idx].copy())
        value_target = torch.tensor([self.values[idx]], dtype=torch.float32)

        if self.augment_symmetry and random.random() < 0.5:
            encoding = encoding.flip(2)
            policy_target = policy_target[self.mirror_perm]

        return encoding, policy_target, value_target


def train_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0

    for encoding, policy_target, value_target in loader:
        encoding = encoding.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        policy_logits, value = model(encoding)

        policy_log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_target * policy_log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value.squeeze(1), value_target.squeeze(1))
        loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pl = policy_loss.item()
        vl = value_loss.item()
        if not (math.isfinite(pl) and math.isfinite(vl)):
            raise RuntimeError(f"NaN/Inf detected: policy_loss={pl}, value_loss={vl}")
        total_policy_loss += pl
        total_value_loss += vl

    denom = max(len(loader), 1)
    return total_policy_loss / denom, total_value_loss / denom


def evaluate(
    model,
    loader,
    device,
):
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0

    with torch.no_grad():
        for encoding, policy_target, value_target in loader:
            encoding = encoding.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            policy_logits, value = model(encoding)
            policy_log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_target * policy_log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value.squeeze(1), value_target.squeeze(1))

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    denom = max(len(loader), 1)
    return total_policy_loss / denom, total_value_loss / denom


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"model_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Gungi policy+value model from self-play JSONL"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Preprocessed data directory (output of preprocess.py)",
    )
    parser.add_argument(
        "--augment-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable random left-right symmetry augmentation",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", default="training/checkpoints")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint .pt file to resume/warm-start from",
    )
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Path to CSV file for metrics logging (default: {output_dir}/metrics.csv)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        rank = local_rank()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

    dataset = SelfPlayDataset(args.data, augment_symmetry=args.augment_symmetry)
    if is_main_process():
        print(f"loaded {len(dataset)} positions")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size = len(dataset)
        val_size = 0

    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = GungiNet(num_blocks=args.num_blocks, channels=args.channels).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank()])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = args.metrics_file
    if metrics_file is None:
        metrics_file = str(output_dir / "metrics.csv")
    metrics_file = Path(metrics_file)

    csv_file = None
    csv_writer = None
    metrics_file_exists = metrics_file.exists() and metrics_file.stat().st_size > 0

    writer = None
    if args.tensorboard and is_main_process():
        tb_log_dir = output_dir / "tb_logs"
        SummaryWriter = importlib.import_module("torch.utils.tensorboard").SummaryWriter
        writer = SummaryWriter(log_dir=str(tb_log_dir))

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        raw_model = model.module if use_ddp else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        resumed_epoch = checkpoint.get("epoch", 0)
        # Warm-start: load weights but reset epoch counter (training on new data)
        start_epoch = 1
        if is_main_process():
            print(
                f"resumed from {args.resume} (epoch {resumed_epoch}), resetting to epoch 1"
            )

    if is_main_process():
        csv_file = open(metrics_file, "a" if metrics_file_exists else "w", newline="")
        csv_writer = csv.writer(csv_file)
        if not metrics_file_exists:
            csv_writer.writerow(
                [
                    "epoch",
                    "train_policy_loss",
                    "train_value_loss",
                    "val_policy_loss",
                    "val_value_loss",
                    "lr",
                ]
            )
            csv_file.flush()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_policy_loss, train_value_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        scheduler.step()

        val_policy_loss = None
        val_value_loss = None
        if val_loader is not None:
            val_policy_loss, val_value_loss = evaluate(model, val_loader, device)

        if is_main_process():
            message = (
                f"epoch={epoch} "
                f"train_policy_loss={train_policy_loss:.6f} "
                f"train_value_loss={train_value_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.8f}"
            )
            if val_policy_loss is not None:
                message += (
                    f" val_policy_loss={val_policy_loss:.6f}"
                    f" val_value_loss={val_value_loss:.6f}"
                )
            print(message)

            val_policy_loss_val = val_policy_loss if val_policy_loss is not None else ""
            val_value_loss_val = val_value_loss if val_value_loss is not None else ""
            lr = scheduler.get_last_lr()[0]
            csv_writer.writerow(
                [
                    epoch,
                    train_policy_loss,
                    train_value_loss,
                    val_policy_loss_val,
                    val_value_loss_val,
                    lr,
                ]
            )
            csv_file.flush()

            if writer is not None:
                writer.add_scalar("train/policy_loss", train_policy_loss, epoch)
                writer.add_scalar("train/value_loss", train_value_loss, epoch)
                writer.add_scalar("lr", lr, epoch)
                if val_policy_loss is not None:
                    writer.add_scalar("val/policy_loss", val_policy_loss, epoch)
                    writer.add_scalar("val/value_loss", val_value_loss, epoch)

            if epoch % args.save_every == 0:
                raw_model = model.module if use_ddp else model
                path = save_checkpoint(
                    raw_model, optimizer, scheduler, epoch, output_dir
                )
                print(f"saved checkpoint: {path}")

    if is_main_process():
        csv_file.close()
        if writer is not None:
            writer.close()
        raw_model = model.module if use_ddp else model
        final_path = save_checkpoint(
            raw_model, optimizer, scheduler, args.epochs, output_dir
        )
        print(f"saved final checkpoint: {final_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
