use std::sync::LazyLock;

use crate::board::{square_index, Board};
use crate::types::{Color, HandPiece, Piece, PieceType, Square};

const PIECE_KEYS: usize = 14 * 2 * 81 * 3;
const HAND_KEYS: usize = 14 * 2 * 10;

static ZOBRIST_KEYS: LazyLock<ZobristKeys> = LazyLock::new(ZobristKeys::new);

#[derive(Debug, Clone)]
pub struct ZobristKeys {
    piece_square_tier: [u64; PIECE_KEYS],
    hand_counts: [u64; HAND_KEYS],
    side_to_move: u64,
    drafting: [u64; 2],
}

pub fn zobrist_keys() -> &'static ZobristKeys {
    &ZOBRIST_KEYS
}

impl ZobristKeys {
    fn new() -> Self {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;

        let mut piece_square_tier = [0u64; PIECE_KEYS];
        for key in &mut piece_square_tier {
            *key = next_u64(&mut state);
        }

        let mut hand_counts = [0u64; HAND_KEYS];
        for key in &mut hand_counts {
            *key = next_u64(&mut state);
        }

        Self {
            piece_square_tier,
            hand_counts,
            side_to_move: next_u64(&mut state),
            drafting: [next_u64(&mut state), next_u64(&mut state)],
        }
    }

    pub fn piece_key(&self, piece: Piece, square: Square, tier: u8) -> Option<u64> {
        if !(1..=3).contains(&tier) {
            return None;
        }

        let (rank, file) = square_index(square)?;
        let square_idx = rank * 9 + file;
        let idx = (((piece.piece_type as usize * 2 + piece.color as usize) * 81 + square_idx) * 3)
            + usize::from(tier - 1);
        Some(self.piece_square_tier[idx])
    }

    pub fn hand_count_key(&self, piece_type: PieceType, color: Color, count: u8) -> Option<u64> {
        if count == 0 || count >= 10 {
            return None;
        }
        let idx = ((piece_type as usize * 2 + color as usize) * 10) + usize::from(count);
        Some(self.hand_counts[idx])
    }

    pub fn xor_piece(&self, hash: &mut u64, piece: Piece, square: Square, tier: u8) {
        if let Some(key) = self.piece_key(piece, square, tier) {
            *hash ^= key;
        }
    }

    pub fn update_hand_count(
        &self,
        hash: &mut u64,
        piece_type: PieceType,
        color: Color,
        old_count: u8,
        new_count: u8,
    ) {
        if let Some(key) = self.hand_count_key(piece_type, color, old_count) {
            *hash ^= key;
        }
        if let Some(key) = self.hand_count_key(piece_type, color, new_count) {
            *hash ^= key;
        }
    }

    pub fn xor_side_to_move(&self, hash: &mut u64) {
        *hash ^= self.side_to_move;
    }

    pub fn xor_drafting(&self, hash: &mut u64, color: Color) {
        *hash ^= self.drafting[color as usize];
    }

    pub fn hash_board(&self, board: &Board) -> u64 {
        let mut hash = 0u64;
        for rank in 1..=9u8 {
            for file in 1..=9u8 {
                let square = Square::new_unchecked(rank, file);
                if let Some(tower) = board.get(square) {
                    for tier in 1..=tower.height() {
                        if let Some(piece) = tower.pieces()[usize::from(tier - 1)] {
                            self.xor_piece(&mut hash, piece, square, tier);
                        }
                    }
                }
            }
        }
        hash
    }

    pub fn hash_position(
        &self,
        board: &Board,
        hand: &[HandPiece],
        turn: Color,
        drafting: [bool; 2],
    ) -> u64 {
        let mut hash = self.hash_board(board);
        for hp in hand {
            if let Some(key) = self.hand_count_key(hp.piece_type, hp.color, hp.count) {
                hash ^= key;
            }
        }
        if matches!(turn, Color::Black) {
            hash ^= self.side_to_move;
        }
        if drafting[Color::White as usize] {
            hash ^= self.drafting[Color::White as usize];
        }
        if drafting[Color::Black as usize] {
            hash ^= self.drafting[Color::Black as usize];
        }
        hash
    }
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
