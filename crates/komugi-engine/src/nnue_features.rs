use komugi_core::{Color, PieceType, Position, SetupMode, SQUARES};

const PIECE_TYPE_COUNT: usize = 14;
const REL_COLOR_COUNT: usize = 2;
const SQUARE_COUNT: usize = 81;
const TIER_COUNT: usize = 3;

const BOARD_FEATURE_COUNT: usize = PIECE_TYPE_COUNT * REL_COLOR_COUNT * SQUARE_COUNT * TIER_COUNT;
const HAND_MAX_COUNTS: [usize; PIECE_TYPE_COUNT] = [2, 2, 2, 4, 4, 6, 4, 4, 4, 8, 2, 4, 2, 2];
const HAND_FEATURE_COUNT: usize = 2 * hand_feature_span();
const GLOBAL_FEATURE_COUNT: usize = 4;

const HAND_FEATURE_START: usize = BOARD_FEATURE_COUNT;
const GLOBAL_FEATURE_START: usize = HAND_FEATURE_START + HAND_FEATURE_COUNT;

const PHASE_FEATURE_INDEX: usize = GLOBAL_FEATURE_START;
const SIDE_TO_MOVE_FEATURE_INDEX: usize = GLOBAL_FEATURE_START + 1;
const MAX_TIER_3_FEATURE_INDEX: usize = GLOBAL_FEATURE_START + 2;
const MARSHAL_STACKING_FEATURE_INDEX: usize = GLOBAL_FEATURE_START + 3;

pub const TOTAL_FEATURES: usize = BOARD_FEATURE_COUNT + HAND_FEATURE_COUNT + GLOBAL_FEATURE_COUNT;

pub fn extract_features(position: &Position, perspective: Color) -> Vec<u16> {
    let mut features = Vec::with_capacity(256);

    for square in SQUARES {
        let Some(tower) = position.board.get(square) else {
            continue;
        };

        let square_idx = perspective_square_index(square.rank, square.file, perspective);
        for (tier_idx, slot) in tower
            .pieces()
            .iter()
            .enumerate()
            .take(usize::from(tower.height()))
        {
            let Some(piece) = slot else {
                continue;
            };
            let rel_color = relative_color(piece.color, perspective);
            let idx = board_feature_index(piece.piece_type, rel_color, square_idx, tier_idx);
            features.push(idx as u16);
        }
    }

    for &rel_color in &[0u8, 1u8] {
        let color = absolute_color(rel_color, perspective);
        for piece_type in PieceType::ALL {
            let count = hand_count(position, piece_type, color) as usize;
            let max_count = hand_max_count(piece_type);
            let capped = count.min(max_count);
            for level in 0..capped {
                let idx = hand_feature_index(piece_type, rel_color, level);
                features.push(idx as u16);
            }
        }
    }

    if position.in_draft() {
        features.push(PHASE_FEATURE_INDEX as u16);
    }
    if position.turn == perspective {
        features.push(SIDE_TO_MOVE_FEATURE_INDEX as u16);
    }
    if matches!(position.mode, SetupMode::Advanced) {
        features.push(MAX_TIER_3_FEATURE_INDEX as u16);
    }
    if matches!(position.mode, SetupMode::Intermediate | SetupMode::Advanced) {
        features.push(MARSHAL_STACKING_FEATURE_INDEX as u16);
    }

    features.sort_unstable();
    features
}

const fn hand_feature_span() -> usize {
    let mut idx = 0;
    let mut total = 0;
    while idx < HAND_MAX_COUNTS.len() {
        total += HAND_MAX_COUNTS[idx];
        idx += 1;
    }
    total
}

fn hand_max_count(piece_type: PieceType) -> usize {
    HAND_MAX_COUNTS[piece_type as usize]
}

fn hand_count(position: &Position, piece_type: PieceType, color: Color) -> u8 {
    let count: u16 = position
        .hand
        .iter()
        .filter(|hp| hp.piece_type == piece_type && hp.color == color)
        .map(|hp| u16::from(hp.count))
        .sum();
    count.min(u16::from(u8::MAX)) as u8
}

fn relative_color(piece_color: Color, perspective: Color) -> u8 {
    if piece_color == perspective {
        0
    } else {
        1
    }
}

fn absolute_color(rel_color: u8, perspective: Color) -> Color {
    match (rel_color, perspective) {
        (0, color) => color,
        (1, Color::White) => Color::Black,
        (1, Color::Black) => Color::White,
        _ => panic!("invalid relative color"),
    }
}

fn perspective_square_index(rank: u8, file: u8, perspective: Color) -> usize {
    let rank_idx = usize::from(rank - 1);
    let file_idx = usize::from(file - 1);
    match perspective {
        Color::White => rank_idx * 9 + file_idx,
        Color::Black => (8 - rank_idx) * 9 + file_idx,
    }
}

fn board_feature_index(
    piece_type: PieceType,
    rel_color: u8,
    square_idx: usize,
    tier_idx: usize,
) -> usize {
    (((piece_type as usize * REL_COLOR_COUNT + usize::from(rel_color)) * SQUARE_COUNT + square_idx)
        * TIER_COUNT)
        + tier_idx
}

fn hand_feature_index(piece_type: PieceType, rel_color: u8, level: usize) -> usize {
    let rel_base = HAND_FEATURE_START + usize::from(rel_color) * hand_feature_span();
    let piece_base = HAND_MAX_COUNTS
        .iter()
        .take(piece_type as usize)
        .sum::<usize>();
    rel_base + piece_base + level
}

#[cfg(test)]
mod tests {
    use super::*;
    use komugi_core::{Piece, SetupMode, Square};

    #[test]
    fn feature_layout_count_sanity() {
        assert_eq!(BOARD_FEATURE_COUNT, 14 * 2 * 81 * 3);
        assert_eq!(HAND_FEATURE_COUNT, 100);
        assert_eq!(TOTAL_FEATURES, 6908);

        let position = Position::new(SetupMode::Advanced);
        let features = extract_features(&position, Color::White);
        assert!(features.windows(2).all(|w| w[0] <= w[1]));
        assert!(features
            .iter()
            .all(|&idx| usize::from(idx) < TOTAL_FEATURES));
    }

    #[test]
    fn perspective_flip_mirrors_rank_and_relative_color() {
        let mut position = Position::new(SetupMode::Advanced);
        position.drafting_rights = [false, false];
        position.turn = Color::White;
        for hp in &mut position.hand {
            hp.count = 0;
        }

        let white_square = Square::new_unchecked(2, 4);
        let black_square = Square::new_unchecked(7, 4);
        position
            .board
            .put(
                Piece {
                    piece_type: PieceType::Marshal,
                    color: Color::White,
                },
                white_square,
            )
            .expect("valid square");
        position
            .board
            .put(
                Piece {
                    piece_type: PieceType::Soldier,
                    color: Color::Black,
                },
                black_square,
            )
            .expect("valid square");

        let white_features = extract_features(&position, Color::White);
        let black_features = extract_features(&position, Color::Black);

        let white_piece_white_perspective = board_feature_index(
            PieceType::Marshal,
            0,
            perspective_square_index(2, 4, Color::White),
            0,
        ) as u16;
        let white_piece_black_perspective = board_feature_index(
            PieceType::Marshal,
            1,
            perspective_square_index(2, 4, Color::Black),
            0,
        ) as u16;
        let black_piece_white_perspective = board_feature_index(
            PieceType::Soldier,
            1,
            perspective_square_index(7, 4, Color::White),
            0,
        ) as u16;
        let black_piece_black_perspective = board_feature_index(
            PieceType::Soldier,
            0,
            perspective_square_index(7, 4, Color::Black),
            0,
        ) as u16;

        assert!(white_features.contains(&white_piece_white_perspective));
        assert!(black_features.contains(&white_piece_black_perspective));
        assert!(white_features.contains(&black_piece_white_perspective));
        assert!(black_features.contains(&black_piece_black_perspective));
    }

    #[test]
    fn hand_thermometer_encoding_emits_all_thresholds() {
        let mut position = Position::new(SetupMode::Advanced);
        position.drafting_rights = [false, false];
        for hp in &mut position.hand {
            hp.count = 0;
        }

        set_hand_count(&mut position, Color::White, PieceType::Lancer, 3);
        set_hand_count(&mut position, Color::Black, PieceType::Lancer, 2);

        let features = extract_features(&position, Color::White);
        let own_l1 = hand_feature_index(PieceType::Lancer, 0, 0) as u16;
        let own_l2 = hand_feature_index(PieceType::Lancer, 0, 1) as u16;
        let own_l3 = hand_feature_index(PieceType::Lancer, 0, 2) as u16;
        let own_l4 = hand_feature_index(PieceType::Lancer, 0, 3) as u16;
        let opp_l1 = hand_feature_index(PieceType::Lancer, 1, 0) as u16;
        let opp_l2 = hand_feature_index(PieceType::Lancer, 1, 1) as u16;
        let opp_l3 = hand_feature_index(PieceType::Lancer, 1, 2) as u16;

        assert!(features.contains(&own_l1));
        assert!(features.contains(&own_l2));
        assert!(features.contains(&own_l3));
        assert!(!features.contains(&own_l4));
        assert!(features.contains(&opp_l1));
        assert!(features.contains(&opp_l2));
        assert!(!features.contains(&opp_l3));
    }

    fn set_hand_count(position: &mut Position, color: Color, piece_type: PieceType, count: u8) {
        for hp in &mut position.hand {
            if hp.color == color && hp.piece_type == piece_type {
                hp.count = count;
            }
        }
    }
}
