# komugi

A Rust engine for gungi, the fictional board game from Hunter x Hunter.

Generates and validates legal moves, evaluates positions, and trains via self-play using an AlphaZero-style pipeline. The branching factor ranges from 200 to 700+ legal moves per position, far exceeding chess (~35).

## Project Structure

```
komugi-core/      Game logic, move generation, FEN/SAN, position representation.
                  Zero-allocation hot paths.

komugi-engine/    Search (alpha-beta + MCTS), classical eval, neural network
                  inference (ONNX), self-play data generation, position encoding
                  (119 planes).

komugi-wasm/      WASM bindings for the browser. Powers gungi.io.
```

## Game Modes

| Mode             | Description                                              |
|------------------|----------------------------------------------------------|
| **Intro**        | Simplified ruleset for learning                          |
| **Beginner**     | Standard rules, tier 1-2 pieces                          |
| **Intermediate** | Draft phase + marshal stacking                           |
| **Advanced**     | Full ruleset with tier 3 pieces                          |

## Notation

SAN uses kanji for move types:

| Kanji | Meaning | Description    |
|-------|---------|----------------|
| 新    | Arata   | Drop           |
| 取    | Toru    | Capture        |
| 付    | Tsuke   | Stack          |
| 返    | Kaesu   | Betray         |
| 終    | Owari   | Draft end      |

FEN format is byte-identical to the gungi.js TypeScript reference implementation.

## Quick Start

Run tests:

```sh
cargo test -p komugi-core -p komugi-engine
```

Run self-play:

```sh
cargo run --release --bin selfplay -- [games] [output.jsonl] [sims] [model?] [mode] [threads]
```

| Arg           | Description                                         |
|---------------|-----------------------------------------------------|
| `games`       | Number of games to play                             |
| `output.jsonl`| Output file for training data                       |
| `sims`        | MCTS simulations per move                           |
| `model?`      | Path to ONNX model, or `-` for classical eval       |
| `mode`        | Game mode: `intro`, `beginner`, `intermediate`, `advanced` |
| `threads`     | Number of parallel games                            |

## Training Pipeline

The self-play loop follows the AlphaZero pattern: generate data, train, export, repeat.

**1. Generate data (classical eval):**

```sh
cargo run --release --bin selfplay -- 500 data/gen0.jsonl 400 - beginner
```

**2. Train the network:**

```sh
cd training && python train.py --data ../data/gen0.jsonl --epochs 50 --device cuda
```

**3. Export to ONNX:**

```sh
python export_onnx.py --checkpoint checkpoints/model_epoch_50.pt --output ../models/gungi_v0.onnx
```

**4. Generate data with the trained model:**

```sh
cargo run --release --bin selfplay --features neural -- 500 data/gen1.jsonl 400 models/gungi_v0.onnx beginner
```

Repeat steps 2-4 for each generation.

## Neural Network

ResNet architecture:

- 10 residual blocks, 128 channels
- ~3M parameters
- Input: 119 planes on a 9x9 board
- Policy head: 7695 output dimensions
- Value head: scalar win probability

## Position Encoding

119 input planes over a 9x9 grid:

| Planes | Description                |
|--------|----------------------------|
| 84     | Piece placement (by type, color, tier) |
| 28     | Hand pieces (pieces available to drop)  |
| 3      | Tower height at each position           |
| 1      | Game phase (draft vs. game)             |
| 1      | Side to move                            |
| 1      | Max tier is 3 (advanced mode flag)      |
| 1      | Marshal stacking allowed                |

## Dependencies

- Rust 2021 edition
- Python 3.10+ with PyTorch 2.0+ (training only)

## License

MIT
