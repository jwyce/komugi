# Gungi Training

PyTorch training pipeline for a policy+value network using self-play JSONL data.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

## Generate self-play data

```bash
cargo run --release --bin selfplay -- 100 data/selfplay.jsonl 800
```

## Train model

```bash
python training/train.py --data data/selfplay.jsonl --epochs 50 --batch-size 256 --lr 0.001
```

Checkpoints are saved to `training/checkpoints/`.

## Export ONNX

```bash
python training/export_onnx.py --checkpoint training/checkpoints/model_epoch_50.pt --output models/gungi_v1.onnx
```

## Data schema expected per JSONL line

- `encoding`: flat length-9639 list (reshaped to `119 x 9 x 9`)
- `policy`: list of `[san, prob]` entries
- `outcome`: scalar in `[-1, 1]`

`san` format is parsed from engine SAN (for example, `新兵(5-5-1)` for drops or `兵(7-5-1)(6-5-1)` for board moves).

## NNUE Training (Distillation)

After training the ResNet teacher, distill it into a smaller NNUE student:

### Generate labels with high-quality evaluations
```bash
cargo run --release --bin labelgen -- 1000 labels.jsonl 400 5000 model.onnx beginner 4
```

### Train NNUE
```bash
python training/train_nnue.py --data labels.jsonl --epochs 50 --batch-size 4096 --lambda 0.5
```

### Export to .nnue format
```bash
python training/export_nnue.py --checkpoint checkpoints_nnue/nnue_final.pt --output models/gungi.nnue
```

The exported .nnue file (~3.5MB) can be embedded in WASM for client-side inference.
