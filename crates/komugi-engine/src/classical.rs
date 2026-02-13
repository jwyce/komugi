use komugi_core::{
    board::Board,
    constants::SQUARES,
    eval::Evaluator,
    position::Position,
    types::{Color, Piece, PieceType, Score, Square},
};

/// Centipawn piece values from gungi reference.
/// Marshal = 0 because capturing it ends the game (handled separately).
const PIECE_VALUES: [i32; 14] = [
    0,   // Marshal
    900, // General
    800, // LieutenantGeneral
    500, // MajorGeneral
    400, // Warrior
    500, // Lancer
    500, // Rider
    400, // Spy
    300, // Fortress
    100, // Soldier
    500, // Cannon
    400, // Archer
    300, // Musketeer
    400, // Tactician
];

/// Piece-square table: bonus for central control and forward advancement.
/// Indexed [rank_from_own_side 0..9][distance_from_center 0..4].
/// rank_from_own_side: 0 = back rank, 8 = furthest forward.
/// distance_from_center: 0 = center (file 5), 4 = edge (file 1 or 9).
const PST_ADVANCEMENT: [[i32; 5]; 9] = [
    //  center → edge
    [0, 0, 0, -5, -10],  // back rank
    [5, 3, 2, 0, -5],    // rank 1
    [10, 8, 5, 2, 0],    // rank 2
    [20, 15, 10, 5, 2],  // rank 3
    [30, 25, 20, 10, 5], // rank 4 (center)
    [35, 30, 25, 15, 8], // rank 5
    [30, 25, 20, 10, 5], // rank 6
    [20, 15, 10, 5, 0],  // rank 7
    [10, 8, 5, 2, 0],    // rank 8 (furthest forward)
];

/// Marshal should stay safe (back ranks preferred).
const PST_MARSHAL: [[i32; 5]; 9] = [
    [20, 15, 10, 5, 0], // back rank: safe
    [10, 8, 5, 2, -5],
    [0, 0, -5, -10, -15],
    [-10, -15, -20, -25, -30],
    [-30, -35, -40, -45, -50],
    [-40, -45, -50, -55, -60],
    [-50, -55, -60, -65, -70],
    [-60, -65, -70, -75, -80],
    [-70, -75, -80, -85, -90],
];

/// Fortress prefers defensive positions (back/mid ranks).
const PST_FORTRESS: [[i32; 5]; 9] = [
    [15, 12, 10, 5, 0],
    [12, 10, 8, 3, 0],
    [10, 8, 5, 2, 0],
    [5, 3, 2, 0, -2],
    [0, 0, -2, -5, -8],
    [-5, -5, -8, -10, -12],
    [-10, -10, -12, -15, -18],
    [-15, -15, -18, -20, -22],
    [-20, -20, -22, -25, -28],
];

/// Tower control bonus per tier (index 0 unused, tiers 1-3).
const TOWER_CONTROL_BONUS: [i32; 4] = [0, 0, 25, 50];

/// Bonus for tower height owned by one color.
const TOWER_HEIGHT_BONUS: [i32; 4] = [0, 0, 15, 35];

/// Bonus per friendly piece adjacent to marshal.
const MARSHAL_SHIELD_BONUS: i32 = 10;

#[derive(Debug, Clone, Copy)]
pub struct ClassicalEval;

impl ClassicalEval {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ClassicalEval {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for ClassicalEval {
    fn evaluate(&self, position: &Position) -> Score {
        let board = &position.board;

        let mut white_score: i32 = 0;
        let mut black_score: i32 = 0;

        let mut white_marshal_sq: Option<Square> = None;
        let mut black_marshal_sq: Option<Square> = None;

        for sq in SQUARES {
            let Some(tower) = board.get(sq) else {
                continue;
            };

            let top = tower.top();
            let height = tower.height();

            let mut white_in_tower = 0u8;
            let mut black_in_tower = 0u8;

            for piece in tower.iter() {
                let value = piece_value(piece.piece_type);
                let pst = pst_bonus(&piece, sq);

                match piece.color {
                    Color::White => {
                        white_score += value + pst;
                        white_in_tower += 1;
                        if piece.piece_type == PieceType::Marshal {
                            white_marshal_sq = Some(sq);
                        }
                    }
                    Color::Black => {
                        black_score += value + pst;
                        black_in_tower += 1;
                        if piece.piece_type == PieceType::Marshal {
                            black_marshal_sq = Some(sq);
                        }
                    }
                }
            }

            if let Some((top_piece, _tier)) = top {
                let bonus = TOWER_CONTROL_BONUS[height as usize];
                match top_piece.color {
                    Color::White => white_score += bonus,
                    Color::Black => black_score += bonus,
                }
            }

            if white_in_tower > 1 {
                white_score += TOWER_HEIGHT_BONUS[white_in_tower as usize];
            }
            if black_in_tower > 1 {
                black_score += TOWER_HEIGHT_BONUS[black_in_tower as usize];
            }
        }

        white_score += marshal_safety(board, Color::White, white_marshal_sq);
        black_score += marshal_safety(board, Color::Black, black_marshal_sq);

        for hp in position.hand.iter() {
            if hp.count == 0 {
                continue;
            }
            let val = piece_value(hp.piece_type) * i32::from(hp.count);
            let hand_val = (val * 6) / 10;
            match hp.color {
                Color::White => white_score += hand_val,
                Color::Black => black_score += hand_val,
            }
        }

        Score(white_score - black_score)
    }
}

#[inline]
fn piece_value(pt: PieceType) -> i32 {
    PIECE_VALUES[pt as usize]
}

fn pst_bonus(piece: &Piece, sq: Square) -> i32 {
    let rank_from_own_side = match piece.color {
        Color::White => (9 - sq.rank) as usize,
        Color::Black => (sq.rank - 1) as usize,
    };

    let dist_from_center = ((sq.file as i32) - 5).unsigned_abs() as usize;

    match piece.piece_type {
        PieceType::Marshal => PST_MARSHAL[rank_from_own_side][dist_from_center],
        PieceType::Fortress => PST_FORTRESS[rank_from_own_side][dist_from_center],
        PieceType::Soldier => PST_ADVANCEMENT[rank_from_own_side][dist_from_center],
        _ => PST_ADVANCEMENT[rank_from_own_side][dist_from_center] / 2,
    }
}

fn marshal_safety(board: &Board, color: Color, marshal_sq: Option<Square>) -> i32 {
    let Some(sq) = marshal_sq else {
        return 0;
    };

    let mut bonus = 0i32;

    for (dr, df) in [
        (-1i8, -1i8),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ] {
        let nr = sq.rank as i8 + dr;
        let nf = sq.file as i8 + df;
        if !(1..=9).contains(&nr) || !(1..=9).contains(&nf) {
            continue;
        }
        let neighbor = Square::new_unchecked(nr as u8, nf as u8);
        if let Some((piece, _)) = board.get_top(neighbor) {
            if piece.color == color {
                bonus += MARSHAL_SHIELD_BONUS;
            }
        }
    }

    bonus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_value_table_sanity() {
        assert_eq!(piece_value(PieceType::Marshal), 0);
        assert_eq!(piece_value(PieceType::General), 900);
        assert_eq!(piece_value(PieceType::Soldier), 100);
        assert_eq!(piece_value(PieceType::Cannon), 500);
    }
}
