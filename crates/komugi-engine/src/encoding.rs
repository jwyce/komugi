use komugi_core::{Color, PieceType, Position, SetupMode, SQUARES};

const BOARD_AREA: usize = 81;
const PIECE_PLANES_PER_TIER: usize = 28;
const HAND_PLANE_START: usize = 84;
const TOWER_GE_1_PLANE: usize = 112;
const TOWER_GE_2_PLANE: usize = 113;
const TOWER_EQ_3_PLANE: usize = 114;
const PHASE_PLANE: usize = 115;
const SIDE_TO_MOVE_PLANE: usize = 116;
const MAX_TIER_3_PLANE: usize = 117;
const MARSHAL_STACK_PLANE: usize = 118;

pub const NUM_PLANES: usize = 119;
pub const BOARD_SIZE: usize = 9;
pub const ENCODING_SIZE: usize = NUM_PLANES * BOARD_SIZE * BOARD_SIZE;

pub fn encode_position(position: &Position) -> Vec<f32> {
    let mut tensor = vec![0.0f32; ENCODING_SIZE];

    for square in SQUARES {
        if let Some(tower) = position.board.get(square) {
            set_square(
                &mut tensor,
                TOWER_GE_1_PLANE,
                usize::from(square.rank - 1),
                usize::from(square.file - 1),
                1.0,
            );

            let height = tower.height();
            if height >= 2 {
                set_square(
                    &mut tensor,
                    TOWER_GE_2_PLANE,
                    usize::from(square.rank - 1),
                    usize::from(square.file - 1),
                    1.0,
                );
            }
            if height == 3 {
                set_square(
                    &mut tensor,
                    TOWER_EQ_3_PLANE,
                    usize::from(square.rank - 1),
                    usize::from(square.file - 1),
                    1.0,
                );
            }

            let pieces = tower.pieces();
            for (tier, slot) in pieces.iter().enumerate().take(usize::from(height)) {
                let Some(piece) = slot else {
                    continue;
                };
                let plane = tier * PIECE_PLANES_PER_TIER
                    + piece_type_color_offset(piece.piece_type, piece.color);
                set_square(
                    &mut tensor,
                    plane,
                    usize::from(square.rank - 1),
                    usize::from(square.file - 1),
                    1.0,
                );
            }
        }
    }

    for piece_type in PieceType::ALL {
        for color in [Color::White, Color::Black] {
            let plane = HAND_PLANE_START + piece_type_color_offset(piece_type, color);
            let max_count = max_hand_count(piece_type) as f32;
            let count = hand_count(position, piece_type, color);
            let value = f32::from(count) / max_count;
            fill_plane(&mut tensor, plane, value);
        }
    }

    fill_plane(
        &mut tensor,
        PHASE_PLANE,
        if position.in_draft() { 1.0 } else { 0.0 },
    );
    fill_plane(
        &mut tensor,
        SIDE_TO_MOVE_PLANE,
        if position.turn == Color::White {
            1.0
        } else {
            0.0
        },
    );

    let is_advanced = matches!(position.mode, SetupMode::Advanced);
    let marshal_can_stack = matches!(position.mode, SetupMode::Advanced | SetupMode::Intermediate);
    fill_plane(
        &mut tensor,
        MAX_TIER_3_PLANE,
        if is_advanced { 1.0 } else { 0.0 },
    );
    fill_plane(
        &mut tensor,
        MARSHAL_STACK_PLANE,
        if marshal_can_stack { 1.0 } else { 0.0 },
    );

    tensor
}

fn piece_type_color_offset(piece_type: PieceType, color: Color) -> usize {
    piece_type as usize * 2 + color as usize
}

fn hand_count(position: &Position, piece_type: PieceType, color: Color) -> u8 {
    let max_count = max_hand_count(piece_type);
    let count: u16 = position
        .hand
        .iter()
        .filter(|hp| hp.piece_type == piece_type && hp.color == color)
        .map(|hp| u16::from(hp.count))
        .sum();
    count.min(u16::from(max_count)) as u8
}

fn fill_plane(tensor: &mut [f32], plane: usize, value: f32) {
    let start = plane * BOARD_AREA;
    let end = start + BOARD_AREA;
    for cell in &mut tensor[start..end] {
        *cell = value;
    }
}

fn set_square(tensor: &mut [f32], plane: usize, rank_idx: usize, file_idx: usize, value: f32) {
    let idx = plane * BOARD_AREA + rank_idx * BOARD_SIZE + file_idx;
    tensor[idx] = value;
}

pub const NUM_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
pub const DROP_MOVE_OFFSET: usize = NUM_SQUARES * NUM_SQUARES;
pub const POLICY_SIZE: usize = DROP_MOVE_OFFSET + 14 * NUM_SQUARES;

pub fn move_to_policy_index(mv: &komugi_core::Move) -> usize {
    let to_idx = square_index(mv.to.square);
    match mv.move_type {
        komugi_core::MoveType::Arata => DROP_MOVE_OFFSET + mv.piece as usize * NUM_SQUARES + to_idx,
        _ => {
            let from = mv.from.expect("non-drop move must have from square");
            let from_idx = square_index(from.square);
            from_idx * NUM_SQUARES + to_idx
        }
    }
}

pub fn square_index(sq: komugi_core::Square) -> usize {
    usize::from(sq.rank - 1) * BOARD_SIZE + usize::from(sq.file - 1)
}

fn max_hand_count(piece_type: PieceType) -> u8 {
    match piece_type {
        PieceType::Marshal => 1,
        PieceType::General => 1,
        PieceType::LieutenantGeneral => 1,
        PieceType::MajorGeneral => 2,
        PieceType::Warrior => 2,
        PieceType::Lancer => 2,
        PieceType::Rider => 2,
        PieceType::Spy => 2,
        PieceType::Fortress => 2,
        PieceType::Soldier => 7,
        PieceType::Cannon => 2,
        PieceType::Archer => 2,
        PieceType::Musketeer => 1,
        PieceType::Tactician => 1,
    }
}
