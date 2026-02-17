"""
NNUE sparse binary feature extraction for gungi.

Mirrors Rust encoding.rs exactly. Features are sparse indices, not dense planes.
"""

import re
from typing import Optional

# ============================================================================
# Constants (must match Rust encoding.rs)
# ============================================================================

# Piece types (order matches Rust PieceType enum)
PIECE_TYPES = [
    "Marshal",  # 0
    "General",  # 1
    "LieutenantGeneral",  # 2
    "MajorGeneral",  # 3
    "Warrior",  # 4
    "Lancer",  # 5
    "Rider",  # 6
    "Spy",  # 7
    "Fortress",  # 8
    "Soldier",  # 9
    "Cannon",  # 10
    "Archer",  # 11
    "Musketeer",  # 12
    "Tactician",  # 13
]

# FEN code to piece type index
FEN_CODE_TO_PIECE_TYPE = {
    "m": 0,  # Marshal
    "g": 1,  # General
    "i": 2,  # LieutenantGeneral
    "j": 3,  # MajorGeneral
    "w": 4,  # Warrior
    "n": 5,  # Lancer
    "r": 6,  # Rider
    "s": 7,  # Spy
    "f": 8,  # Fortress
    "d": 9,  # Soldier
    "c": 10,  # Cannon
    "a": 11,  # Archer
    "k": 12,  # Musketeer
    "t": 13,  # Tactician
}

# Color codes
COLOR_WHITE = 0
COLOR_BLACK = 1

# Board dimensions
BOARD_SIZE = 9
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE  # 81

# Max hand counts (must match Rust HAND_MAX_COUNTS)
MAX_HAND_COUNTS = {
    0: 2,  # Marshal
    1: 2,  # General
    2: 2,  # LieutenantGeneral
    3: 4,  # MajorGeneral
    4: 4,  # Warrior
    5: 6,  # Lancer
    6: 4,  # Rider
    7: 4,  # Spy
    8: 4,  # Fortress
    9: 8,  # Soldier
    10: 2,  # Cannon
    11: 4,  # Archer
    12: 2,  # Musketeer
    13: 2,  # Tactician
}

# Feature index ranges (sparse NNUE, not dense planes)
# Board features: piece_type(14) × color(2) × square(81) × tier(3)
BOARD_FEATURES_PER_TIER = 14 * 2 * NUM_SQUARES  # 2268
BOARD_FEATURES_TOTAL = BOARD_FEATURES_PER_TIER * 3  # 6804

# Hand features (thermometer encoding)
# For each piece type (14) and color (2), encode count as thermometer
# Total thermometer bits: (1+1+1+2+2+2+2+2+2+7+2+2+1+1) × 2 colors = 28 × 2 = 56 total
HAND_FEATURES_START = BOARD_FEATURES_TOTAL  # 6804
_HAND_BITS = sum(MAX_HAND_COUNTS.values()) * 2
HAND_FEATURES_TOTAL = _HAND_BITS  # 56

# Global features
GLOBAL_FEATURES_START = HAND_FEATURES_START + HAND_FEATURES_TOTAL  # 6860
GLOBAL_FEATURES = 4  # phase, side_to_move, max_tier_3, marshal_stacking

TOTAL_FEATURES = GLOBAL_FEATURES_START + GLOBAL_FEATURES  # 6864

# ============================================================================
# FEN Parsing
# ============================================================================


def parse_fen(fen: str) -> dict:
    """
    Parse gungi FEN string into components.

    Format: board/hand turn mode drafting move_number
    Example: "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"

    Returns dict with:
        - board: list of 81 squares (rank 1 to 9, file 1 to 9)
        - hand_white: dict of piece_type -> count
        - hand_black: dict of piece_type -> count
        - turn: 'w' or 'b'
        - mode: 0=intro, 1=beginner, 2=intermediate, 3=advanced
        - drafting: [white_drafting, black_drafting]
    """
    parts = fen.split()
    if len(parts) != 6:
        raise ValueError(f"FEN must have 6 fields, got {len(parts)}")

    board_str, hand_str, turn_str, mode_str, drafting_str, move_num = parts

    # Parse board (9 ranks, rank 1 first)
    ranks = board_str.split("/")
    if len(ranks) != 9:
        raise ValueError(f"Expected 9 ranks, got {len(ranks)}")

    for rank_str in ranks:
        square_count = 0
        parts = rank_str.split("|")
        for part in parts:
            if ":" in part:
                # Tower: counts as 1 square
                square_count += 1
            else:
                # Single pieces or empty squares
                for ch in part:
                    if ch.isdigit():
                        square_count += int(ch)
                    else:
                        square_count += 1

        if square_count != 9:
            raise ValueError(f"Rank has {square_count} squares, expected 9")

    # Parse hand pieces
    hand_parts = hand_str.split("/")
    if len(hand_parts) != 2:
        raise ValueError(f"Expected 2 hands, got {len(hand_parts)}")

    hand_white = parse_hand(hand_parts[0], is_white=True)
    hand_black = parse_hand(hand_parts[1], is_white=False)

    # Parse turn
    if turn_str not in ("w", "b"):
        raise ValueError(f"Turn must be 'w' or 'b', got {turn_str}")

    # Parse mode (0=intro, 1=beginner, 2=intermediate, 3=advanced)
    try:
        mode = int(mode_str)
        if mode not in (0, 1, 2, 3):
            raise ValueError(f"Mode must be 0-3, got {mode}")
    except ValueError:
        raise ValueError(f"Invalid mode: {mode_str}")

    # Parse drafting flags
    drafting = [False, False]
    if drafting_str != "-":
        if "w" in drafting_str:
            drafting[0] = True
        if "b" in drafting_str:
            drafting[1] = True

    return {
        "hand_white": hand_white,
        "hand_black": hand_black,
        "turn": turn_str,
        "mode": mode,
        "drafting": drafting,
    }


def parse_hand(hand_str: str, is_white: bool) -> dict:
    """Parse hand piece string (e.g., 'J2N2R2D1' or '-')."""
    hand = {}

    if hand_str == "-":
        return hand

    # Parse pairs: piece_code + count
    for i in range(0, len(hand_str), 2):
        if i + 1 >= len(hand_str):
            raise ValueError(f"Invalid hand format: {hand_str}")

        piece_code = hand_str[i]
        count_char = hand_str[i + 1]

        # Normalize to lowercase for lookup (uppercase = white, lowercase = black)
        piece_code_lower = piece_code.lower()
        if piece_code_lower not in FEN_CODE_TO_PIECE_TYPE:
            raise ValueError(f"Invalid piece code in hand: {piece_code}")

        if not count_char.isdigit():
            raise ValueError(f"Invalid count in hand: {count_char}")

        # Verify color matches
        is_uppercase = piece_code.isupper()
        if is_white and not is_uppercase:
            raise ValueError(
                f"White hand should have uppercase pieces, got {piece_code}"
            )
        if not is_white and is_uppercase:
            raise ValueError(
                f"Black hand should have lowercase pieces, got {piece_code}"
            )

        piece_type = FEN_CODE_TO_PIECE_TYPE[piece_code_lower]
        count = int(count_char)
        hand[piece_type] = count

    return hand

    # Parse pairs: piece_code + count
    for i in range(0, len(hand_str), 2):
        if i + 1 >= len(hand_str):
            raise ValueError(f"Invalid hand format: {hand_str}")

        piece_code = hand_str[i]
        count_char = hand_str[i + 1]

        # Normalize to lowercase for lookup
        piece_code_lower = piece_code.lower()
        if piece_code_lower not in FEN_CODE_TO_PIECE_TYPE:
            raise ValueError(f"Invalid piece code in hand: {piece_code}")

        if not count_char.isdigit():
            raise ValueError(f"Invalid count in hand: {count_char}")

        piece_type = FEN_CODE_TO_PIECE_TYPE[piece_code_lower]
        count = int(count_char)
        hand[piece_type] = count

    return hand


# ============================================================================
# Feature Extraction
# ============================================================================


def extract_features(fen: str, perspective: str) -> list[int]:
    """
    Extract NNUE sparse binary features from FEN string.

    Args:
        fen: Full gungi FEN string
        perspective: 'w' for White, 'b' for Black

    Returns:
        Sorted list of active feature indices
    """
    if perspective not in ("w", "b"):
        raise ValueError(f"Perspective must be 'w' or 'b', got {perspective}")

    parsed = parse_fen(fen)
    features = set()

    # Determine if we're flipping perspective
    flip = perspective == "b"

    # ========================================================================
    # Board features (piece placement)
    # ========================================================================

    # Parse board into towers (stacks of pieces)
    # Board is stored as flat list of piece codes (bottom to top per square)
    # We need to reconstruct towers
    towers = [[] for _ in range(81)]

    # The board string encodes towers with ':' separators
    # We need to re-parse to get tower structure
    board_str = fen.split()[0]
    ranks = board_str.split("/")

    square_idx = 0
    for rank_idx, rank_str in enumerate(ranks):
        # Files go right-to-left (file 9 leftmost, file 1 rightmost)
        # So we need to reverse the rank string parsing
        parts = rank_str.split("|")
        # First, parse all parts to get the pieces
        parsed_parts = []
        for part in parts:
            if ":" in part:
                # Tower: pieces from bottom to top on a single square
                tower = part.split(":")
                parsed_parts.append(("tower", tower))
            else:
                # Single pieces or empty squares
                for ch in part:
                    if ch.isdigit():
                        parsed_parts.append(("empty", int(ch)))
                    else:
                        parsed_parts.append(("piece", ch))

        # Now reverse the order to account for right-to-left files
        parsed_parts.reverse()

        # Process the reversed parts
        for item_type, item_data in parsed_parts:
            if item_type == "tower":
                tower = item_data
                for tier, piece_code in enumerate(tower):
                    piece_code_lower = piece_code.lower()
                    piece_type = FEN_CODE_TO_PIECE_TYPE[piece_code_lower]
                    color = COLOR_BLACK if piece_code.islower() else COLOR_WHITE
                    towers[square_idx].append((piece_type, color, tier))
                square_idx += 1
            elif item_type == "piece":
                piece_code = item_data
                piece_type = FEN_CODE_TO_PIECE_TYPE[piece_code.lower()]
                color = COLOR_BLACK if piece_code.islower() else COLOR_WHITE
                towers[square_idx].append((piece_type, color, 0))
                square_idx += 1
            elif item_type == "empty":
                square_idx += item_data

    # Add board features for each piece in each tower
    for square_idx, tower in enumerate(towers):
        for tier, (piece_type, color, _) in enumerate(tower):
            if tier >= 3:
                break  # Only 3 tiers

            # Calculate feature index
            # Board: piece_type(14) × color(2) × square(81) × tier(3)
            # Index = tier * (14*2*81) + color * (14*81) + piece_type * 81 + square

            sq = square_idx
            if flip:
                # Flip perspective: only flip rank, not file
                rank = sq // 9
                file = sq % 9
                rank = 8 - rank
                sq = rank * 9 + file
                # Flip color
                color = 1 - color

            # Match Rust ordering: ((piece_type * 2 + color) * 81 + square) * 3 + tier
            feature_idx = (((piece_type * 2 + color) * NUM_SQUARES + sq) * 3) + tier
            features.add(feature_idx)

    # ========================================================================
    # Hand features (thermometer encoding)
    # ========================================================================

    hand_white = parsed["hand_white"]
    hand_black = parsed["hand_black"]

    hand_offsets = [0]
    for pt in range(14):
        hand_offsets.append(hand_offsets[-1] + MAX_HAND_COUNTS[pt])

    hand_span = sum(MAX_HAND_COUNTS.values())

    for rel_color in [0, 1]:
        # Convert relative color to absolute color based on perspective
        if flip:
            abs_color = COLOR_BLACK if rel_color == 0 else COLOR_WHITE
        else:
            abs_color = COLOR_WHITE if rel_color == 0 else COLOR_BLACK

        hand = hand_white if abs_color == COLOR_WHITE else hand_black

        for piece_type in range(14):
            max_count = MAX_HAND_COUNTS[piece_type]
            offset = hand_offsets[piece_type]
            count = hand.get(piece_type, 0)
            count = min(count, max_count)

            rel_base = HAND_FEATURES_START + rel_color * hand_span
            for i in range(count):
                features.add(rel_base + offset + i)

    # ========================================================================
    # Global features
    # ========================================================================

    # Phase: 0 = drafting, 1 = game
    in_draft = parsed["drafting"][0] or parsed["drafting"][1]
    if in_draft:
        features.add(GLOBAL_FEATURES_START + 0)

    # Side to move
    side_to_move = COLOR_WHITE if parsed["turn"] == "w" else COLOR_BLACK
    if flip:
        side_to_move = 1 - side_to_move
    if side_to_move == COLOR_WHITE:
        features.add(GLOBAL_FEATURES_START + 1)

    # Max tier is 3 (advanced mode)
    if parsed["mode"] == 3:
        features.add(GLOBAL_FEATURES_START + 2)

    # Marshal stacking allowed (intermediate or advanced)
    if parsed["mode"] in (2, 3):
        features.add(GLOBAL_FEATURES_START + 3)

    return sorted(list(features))


# ============================================================================
# Self-test
# ============================================================================


def test_extract_features():
    """Self-test on known positions."""

    # Test 1: Intro position
    intro_fen = "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1/j2n2r2d1 w 0 - 1"
    features_w = extract_features(intro_fen, "w")
    features_b = extract_features(intro_fen, "b")

    assert isinstance(features_w, list), "Features should be a list"
    assert isinstance(features_b, list), "Features should be a list"
    assert len(features_w) > 0, "Should have features"
    assert len(features_b) > 0, "Should have features"
    assert all(isinstance(f, int) for f in features_w), "All features should be ints"
    assert all(isinstance(f, int) for f in features_b), "All features should be ints"
    assert all(0 <= f < TOTAL_FEATURES for f in features_w), "Features out of range"
    assert all(0 <= f < TOTAL_FEATURES for f in features_b), "Features out of range"
    assert features_w == sorted(features_w), "Features should be sorted"
    assert features_b == sorted(features_b), "Features should be sorted"

    print(
        f"✓ Intro position: {len(features_w)} features (white), {len(features_b)} features (black)"
    )

    # Test 2: Beginner position
    beginner_fen = "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
    features_w = extract_features(beginner_fen, "w")
    features_b = extract_features(beginner_fen, "b")

    assert len(features_w) > 0, "Should have features"
    assert len(features_b) > 0, "Should have features"
    assert all(0 <= f < TOTAL_FEATURES for f in features_w), "Features out of range"
    assert all(0 <= f < TOTAL_FEATURES for f in features_b), "Features out of range"

    print(
        f"✓ Beginner position: {len(features_w)} features (white), {len(features_b)} features (black)"
    )

    # Test 3: Intermediate position (draft phase)
    intermediate_fen = "9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
    features_w = extract_features(intermediate_fen, "w")
    features_b = extract_features(intermediate_fen, "b")

    assert len(features_w) > 0, "Should have features"
    assert len(features_b) > 0, "Should have features"
    assert all(0 <= f < TOTAL_FEATURES for f in features_w), "Features out of range"
    assert all(0 <= f < TOTAL_FEATURES for f in features_b), "Features out of range"

    print(
        f"✓ Intermediate position: {len(features_w)} features (white), {len(features_b)} features (black)"
    )

    # Test 4: Advanced position
    advanced_fen = "9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 wb 1"
    features_w = extract_features(advanced_fen, "w")
    features_b = extract_features(advanced_fen, "b")

    assert len(features_w) > 0, "Should have features"
    assert len(features_b) > 0, "Should have features"
    assert all(0 <= f < TOTAL_FEATURES for f in features_w), "Features out of range"
    assert all(0 <= f < TOTAL_FEATURES for f in features_b), "Features out of range"

    print(
        f"✓ Advanced position: {len(features_w)} features (white), {len(features_b)} features (black)"
    )

    # Test 5: Verify TOTAL_FEATURES constant
    print(f"\n✓ TOTAL_FEATURES = {TOTAL_FEATURES}")
    print(
        f"  - Board features: {BOARD_FEATURES_TOTAL} (14 types × 2 colors × 81 squares × 3 tiers)"
    )
    print(f"  - Hand features: {HAND_FEATURES_TOTAL} (thermometer)")
    print(
        f"  - Global features: {GLOBAL_FEATURES} (phase, side, max_tier_3, marshal_stack)"
    )

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_extract_features()
