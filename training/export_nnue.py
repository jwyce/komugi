"""
Export trained NNUE checkpoint to quantized .nnue binary.

Binary format matches Rust nnue_format.rs (NnueParams::from_bytes):
  Header (20B): b"GNUE" + version(u32) + total_features(u32) + hidden1(u32) + hidden2(u32)
  L0: ft_bias i16[256] + ft_weights i16[total_features * 256]
  L1: l1_bias i32[32] + l1_weights i8[512 * 32]
  L2: l2_bias i32(scalar) + l2_weights i8[32]
  All values little-endian.

Quantization:
  L0 weights/bias: float * QA(255) -> i16
  L1/L2 weights: float * QB(64) -> i8
  L1/L2 biases: float * QA^2 * QB -> i32
"""

import argparse
import importlib
import struct
from pathlib import Path

torch = importlib.import_module("torch")
np = importlib.import_module("numpy")

from nnue_model import NnueModel, QA, QB


def clamp_round(tensor, scale, lo, hi):
    return torch.round(tensor * scale).clamp(lo, hi)


def export_nnue(checkpoint_path: Path, output_path: Path) -> None:
    model = NnueModel()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    total_features = model.total_features
    hidden1 = model.hidden1
    hidden2 = model.hidden2

    # --- L0 (Feature Transformer) ---
    # PyTorch: ft.weight [hidden1, total_features], ft.bias [hidden1]
    # Binary:  ft_weights [total_features, hidden1] row-major, ft_bias [hidden1]
    ft_bias_q = clamp_round(model.ft.bias.data, QA, -32768, 32767).to(torch.int16)
    ft_weight_q = clamp_round(model.ft.weight.data.t(), QA, -32768, 32767).to(
        torch.int16
    )  # [total_features, hidden1]

    # --- L1 ---
    # PyTorch: l1.weight [hidden2, hidden1*2], l1.bias [hidden2]
    # Binary:  l1_weights [hidden1*2, hidden2] row-major, l1_bias [hidden2]
    # Bias scale: input is QA^2 (from SCReLU), weights scaled by QB
    l1_bias_scale = QA * QA * QB
    l1_bias_q = clamp_round(model.l1.bias.data, l1_bias_scale, -(2**31), 2**31 - 1).to(
        torch.int32
    )
    l1_weight_q = clamp_round(model.l1.weight.data.t(), QB, -128, 127).to(
        torch.int8
    )  # [512, 32]

    # --- L2 ---
    # PyTorch: l2.weight [1, hidden2], l2.bias [1]
    # Binary:  l2_weights [hidden2], l2_bias scalar
    l2_bias_scale = QA * QA * QB
    l2_weight_q = clamp_round(model.l2.weight.data.squeeze(0), QB, -128, 127).to(
        torch.int8
    )  # [32]
    l2_bias_val = int(
        clamp_round(model.l2.bias.data, l2_bias_scale, -(2**31), 2**31 - 1)
        .to(torch.int32)
        .item()
    )

    # --- Write binary ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        # Header
        f.write(b"GNUE")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", total_features))
        f.write(struct.pack("<I", hidden1))
        f.write(struct.pack("<I", hidden2))

        # L0
        f.write(ft_bias_q.numpy().tobytes())
        f.write(ft_weight_q.numpy().tobytes())

        # L1
        f.write(l1_bias_q.numpy().tobytes())
        f.write(l1_weight_q.numpy().tobytes())

        # L2
        f.write(struct.pack("<i", l2_bias_val))
        f.write(l2_weight_q.numpy().tobytes())

    file_size = output_path.stat().st_size
    expected_size = (
        20
        + hidden1 * 2  # ft_bias i16
        + total_features * hidden1 * 2  # ft_weights i16
        + hidden2 * 4  # l1_bias i32
        + hidden1 * 2 * hidden2  # l1_weights i8
        + 4  # l2_bias i32
        + hidden2  # l2_weights i8
    )
    if file_size != expected_size:
        raise RuntimeError(f"size mismatch: {file_size} != {expected_size}")

    print(f"exported {output_path} ({file_size:,} bytes)")
    print(f"  total_features={total_features} hidden1={hidden1} hidden2={hidden2}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NNUE checkpoint to quantized .nnue binary"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .nnue file")
    args = parser.parse_args()
    export_nnue(Path(args.checkpoint), Path(args.output))


if __name__ == "__main__":
    main()
