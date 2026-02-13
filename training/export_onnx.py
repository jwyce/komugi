import argparse
import importlib
from pathlib import Path

torch = importlib.import_module("torch")

from model import GungiNet, NUM_INPUT_PLANES


def load_model(checkpoint_path: Path, num_blocks: int, channels: int) -> GungiNet:
    model = GungiNet(num_blocks=num_blocks, channels=channels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(
    checkpoint_path: Path, output_path: Path, num_blocks: int, channels: int
) -> None:
    model = load_model(checkpoint_path, num_blocks=num_blocks, channels=channels)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, NUM_INPUT_PLANES, 9, 9, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["board_state"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board_state": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export trained GungiNet checkpoint to ONNX"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    export_onnx(
        checkpoint_path, output_path, num_blocks=args.num_blocks, channels=args.channels
    )
    print(f"exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()
