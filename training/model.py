import importlib

torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")


BOARD_SIZE = 9
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
NUM_PIECE_TYPES = 14
NUM_INPUT_PLANES = 119

DROP_MOVE_OFFSET = NUM_SQUARES * NUM_SQUARES
POLICY_SIZE = DROP_MOVE_OFFSET + NUM_PIECE_TYPES * NUM_SQUARES

PIECE_KANJI_TO_INDEX = {
    "帥": 0,
    "大": 1,
    "中": 2,
    "小": 3,
    "侍": 4,
    "槍": 5,
    "馬": 6,
    "忍": 7,
    "砦": 8,
    "兵": 9,
    "砲": 10,
    "弓": 11,
    "筒": 12,
    "謀": 13,
}

PIECE_INDEX_TO_KANJI = {v: k for k, v in PIECE_KANJI_TO_INDEX.items()}


def square_to_index(rank: int, file: int) -> int:
    if rank < 1 or rank > BOARD_SIZE or file < 1 or file > BOARD_SIZE:
        raise ValueError("rank/file out of bounds")
    return (rank - 1) * BOARD_SIZE + (file - 1)


def index_to_square(index: int) -> tuple[int, int]:
    if index < 0 or index >= NUM_SQUARES:
        raise ValueError("square index out of bounds")
    rank = (index // BOARD_SIZE) + 1
    file = (index % BOARD_SIZE) + 1
    return rank, file


def board_move_to_index(
    from_rank: int, from_file: int, to_rank: int, to_file: int
) -> int:
    from_idx = square_to_index(from_rank, from_file)
    to_idx = square_to_index(to_rank, to_file)
    return from_idx * NUM_SQUARES + to_idx


def drop_move_to_index(piece_type_idx: int, to_rank: int, to_file: int) -> int:
    if piece_type_idx < 0 or piece_type_idx >= NUM_PIECE_TYPES:
        raise ValueError("piece_type_idx out of bounds")
    to_idx = square_to_index(to_rank, to_file)
    return DROP_MOVE_OFFSET + piece_type_idx * NUM_SQUARES + to_idx


def index_to_move(index: int) -> dict:
    if index < 0 or index >= POLICY_SIZE:
        raise ValueError("policy index out of bounds")
    if index < DROP_MOVE_OFFSET:
        from_idx = index // NUM_SQUARES
        to_idx = index % NUM_SQUARES
        from_rank, from_file = index_to_square(from_idx)
        to_rank, to_file = index_to_square(to_idx)
        return {
            "is_drop": False,
            "from_rank": from_rank,
            "from_file": from_file,
            "to_rank": to_rank,
            "to_file": to_file,
            "piece_type_idx": None,
        }

    drop_index = index - DROP_MOVE_OFFSET
    piece_type_idx = drop_index // NUM_SQUARES
    to_idx = drop_index % NUM_SQUARES
    to_rank, to_file = index_to_square(to_idx)
    return {
        "is_drop": True,
        "from_rank": None,
        "from_file": None,
        "to_rank": to_rank,
        "to_file": to_file,
        "piece_type_idx": piece_type_idx,
    }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)


class GungiNet(nn.Module):
    def __init__(
        self, num_blocks: int = 10, channels: int = 128, policy_size: int = POLICY_SIZE
    ) -> None:
        super().__init__()
        self.policy_size = policy_size

        self.stem_conv = nn.Conv2d(
            NUM_INPUT_PLANES, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(channels)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(channels * BOARD_SIZE * BOARD_SIZE, policy_size)

        self.value_conv = nn.Conv2d(
            channels, 32, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.value_bn = nn.BatchNorm2d(32)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.backbone(x)

        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_relu(policy)
        policy = torch.flatten(policy, start_dim=1)
        policy_logits = self.policy_fc(policy)

        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_relu(value)
        value = torch.flatten(value, start_dim=1)
        value = self.value_fc1(value)
        value = torch.relu(value)
        value = self.value_fc2(value)
        value = self.value_tanh(value)

        return policy_logits, value
