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

## Post-Gen9 Rulefix Adaptation

When the rules change after `*_gen9` is already trained, do not restart. Recreate the instance with the newest Docker image, copy over last teacher artifacts, then run adaptation passes.

### 1) Recreate + copy artifacts

- Rebuild and push the updated training image.
- Recreate the instance from that image.
- Copy these teachers into `/workspace/models/`:
  - `beginner_gen9.onnx`
  - `intermediate_gen9.onnx`
  - `advanced_gen9.onnx`
- Optionally copy matching checkpoint trees for warm-start (`/workspace/checkpoints/{mode}_gen9/`).

### 2) Run adaptation script

Recommended for current rule deltas: two-pass advanced adaptation (coverage + stabilization).

```bash
python training/rulefix_adaptation.py \
  --workspace /workspace \
  --tag gen9_rulefix \
  --games-beginner 1200 \
  --games-intermediate 1200 \
  --games-advanced 5000 \
  --advanced-two-pass \
  --games-advanced-second-pass 2000 \
  --sims 400 \
  --epochs 25
```

If you copied `.pt` checkpoints, add warm-start template:

```bash
python training/rulefix_adaptation.py \
  --workspace /workspace \
  --tag gen9_rulefix \
  --source-checkpoint-template "{checkpoints_dir}/{mode}_gen9/model_epoch_50.pt"
```

Outputs (with `--advanced-two-pass`):

- `/workspace/models/beginner_gen9_rulefix.onnx`
- `/workspace/models/intermediate_gen9_rulefix.onnx`
- `/workspace/models/advanced_gen9_rulefix_a.onnx`
- `/workspace/models/advanced_gen9_rulefix_b.onnx`
- `/workspace/models/advanced_gen9_rulefix.onnx` (alias of pass B)

### 3) Continue normal distillation flow

Use the adapted teachers for label generation and NNUE training (same high-sim labelgen plan):

- Beginner labels from `beginner_gen9_rulefix.onnx`
- Intermediate labels from `intermediate_gen9_rulefix.onnx`
- Advanced labels from `advanced_gen9_rulefix.onnx`

## Puzzle Dataset Seeding

Generate an initial puzzle candidate set from self-play JSONL:

```bash
python training/generate_puzzles.py \
  --input /workspace/data/*.jsonl \
  --output /workspace/data/puzzles_seed_50k.jsonl \
  --limit 50000
```

Useful filters:

- `--min-gap 0.15` for stricter tactic selection
- `--min-move-number 10` to skip opening noise
- `--no-drop` or `--no-capture` to bias puzzle types
- `--allow-other` to include non-capture/non-drop tactical moves

## Puzzle Stage-2 Verification (Multi-Move)

Stage-1 (`generate_puzzles.py`) finds strong single-position candidates. Stage-2 verifies
continuation lines against best defense and buckets puzzles into `easy` / `medium` / `hard`
using target line length and move-uniqueness thresholds.

### Build verifier binary

```bash
cargo build --release -p komugi-engine --bin puzzle_verify
```

### Verify Stage-1 candidates

```bash
./target/release/puzzle_verify \
  --input /workspace/data/puzzles_seed_50k.jsonl \
  --output-prefix /workspace/data/puzzles_stage2 \
  --depth 4 \
  --num-pv 3 \
  --easy-min-ply 1 \
  --easy-max-ply 3 \
  --medium-min-ply 5 \
  --medium-max-ply 11 \
  --hard-min-ply 13 \
  --hard-max-ply 19
```

Outputs:

- `/workspace/data/puzzles_stage2_all.jsonl`
- `/workspace/data/puzzles_stage2_easy.jsonl`
- `/workspace/data/puzzles_stage2_medium.jsonl`
- `/workspace/data/puzzles_stage2_hard.jsonl`

Notes:

- `--easy/medium/hard-min/max-ply` control verified line length ranges (in plies).
- `--depth` and `--num-pv` trade quality vs runtime.
- Increase strictness with `--*-attacker-gap` and `--*-min-final-eval`.
