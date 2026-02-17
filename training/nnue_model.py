"""
NNUE model for gungi position evaluation.

Architecture:
  Feature Transformer (L0): TOTAL_FEATURES → 256, shared weights, dual perspectives
  Concat [stm_acc, opponent_acc] → 512
  SCReLU → L1 (512 → 32) → SCReLU → L2 (32 → 1)

SCReLU: clamp(x, 0, 1)² in float training.
During quantized inference: clamp(x, 0, QA)² with QA-scaled accumulators.
"""

import importlib

torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")

from nnue_features import TOTAL_FEATURES

# Quantization constants (must match Rust nnue_format.rs)
QA = 255  # Feature transformer / accumulator scale
QB = 64  # Hidden/output layer scale
SCALE = 400  # Centipawn scale


def screlu(x):
    """Squared Clipped ReLU: clamp(x, 0, 1)²."""
    return torch.clamp(x, 0.0, 1.0).square()


class NnueModel(nn.Module):
    """NNUE evaluation network with dual-perspective feature transformer.

    Value-only (no policy head). Designed for sparse binary input features.
    """

    def __init__(
        self,
        total_features: int = TOTAL_FEATURES,
        hidden1: int = 256,
        hidden2: int = 32,
    ):
        super().__init__()
        self.total_features = total_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        # Feature Transformer (L0): shared for both perspectives
        self.ft = nn.Linear(total_features, hidden1)
        # L1: [stm_acc, opp_acc] → hidden2
        self.l1 = nn.Linear(hidden1 * 2, hidden2)
        # L2: hidden2 → scalar
        self.l2 = nn.Linear(hidden2, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.ft.weight, nonlinearity="relu")
        nn.init.zeros_(self.ft.bias)
        nn.init.kaiming_normal_(self.l1.weight, nonlinearity="relu")
        nn.init.zeros_(self.l1.bias)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(
        self,
        white_features: "torch.Tensor",
        black_features: "torch.Tensor",
        stm: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Args:
            white_features: [B, total_features] dense binary
            black_features: [B, total_features] dense binary
            stm: [B] side-to-move (0=white, 1=black)

        Returns:
            [B] raw eval from STM perspective
        """
        # Feature transformer: shared weights, applied per perspective
        acc_w = self.ft(white_features)  # [B, 256]
        acc_b = self.ft(black_features)  # [B, 256]

        # Arrange: STM first, opponent second
        is_black = stm.unsqueeze(1).bool()
        acc_stm = torch.where(is_black, acc_b, acc_w)
        acc_opp = torch.where(is_black, acc_w, acc_b)

        x = torch.cat([acc_stm, acc_opp], dim=1)  # [B, 512]
        x = screlu(x)

        x = self.l1(x)
        x = screlu(x)

        return self.l2(x).squeeze(1)
