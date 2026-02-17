use crate::constants::SQUARES;
use crate::movegen::{generate_all_pseudo_legal_moves_from_position, in_check_with_marshal};
use crate::position::{HistoryEntry, Position, PositionError};
use crate::types::{Color, HandPiece, Move, MoveList, MoveType, PieceType, SetupMode};

pub fn is_marshal_captured(position: &Position) -> bool {
    position.marshal_squares[0].is_none() || position.marshal_squares[1].is_none()
}

impl Position {
    pub fn in_check(&self, color: Option<Color>) -> bool {
        let side = color.unwrap_or(self.turn);
        in_check_with_marshal(&self.board, side, self.marshal_squares[side as usize])
    }

    pub fn is_checkmate(&self) -> bool {
        if self.in_draft() || is_marshal_captured(self) {
            return false;
        }
        self.in_check(None) && self.moves().is_empty()
    }

    pub fn is_stalemate(&self) -> bool {
        if self.in_draft() || is_marshal_captured(self) {
            return false;
        }
        !self.in_check(None) && self.moves().is_empty()
    }

    pub fn is_insufficient_material(&self) -> bool {
        if self.in_draft() {
            return false;
        }

        if self
            .hand
            .iter()
            .any(|piece| piece.piece_type != PieceType::Marshal && piece.count > 0)
        {
            return false;
        }

        let mut marshal_squares = Vec::with_capacity(2);
        for square in SQUARES {
            if let Some(tower) = self.board.get(square) {
                for piece in tower.iter() {
                    if piece.piece_type == PieceType::Marshal {
                        marshal_squares.push(square);
                    } else {
                        return false;
                    }
                }
            }
        }

        if marshal_squares.len() != 2 {
            return false;
        }

        let a = marshal_squares[0];
        let b = marshal_squares[1];
        let rank_diff = a.rank.abs_diff(b.rank);
        let file_diff = a.file.abs_diff(b.file);
        !(rank_diff <= 1 && file_diff <= 1)
    }

    pub fn is_fourfold_repetition(&self) -> bool {
        self.repetition_count(self.zobrist_hash) >= 4
    }

    pub fn is_draw(&self) -> bool {
        self.is_stalemate() || self.is_fourfold_repetition() || self.is_insufficient_material()
    }

    pub fn is_game_over(&self) -> bool {
        if self.in_draft() {
            return false;
        }
        if is_marshal_captured(self) {
            return true;
        }
        self.is_fourfold_repetition()
            || self.is_insufficient_material()
            || self.is_checkmate()
            || self.is_stalemate()
    }
}

#[derive(Debug, Clone)]
pub struct Gungi {
    position: Position,
}

impl Gungi {
    pub fn new(mode: SetupMode) -> Self {
        Self {
            position: Position::new(mode),
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self, PositionError> {
        Ok(Self {
            position: Position::from_fen(fen)?,
        })
    }

    pub fn load(&mut self, fen: &str) -> Result<(), PositionError> {
        self.position = Position::from_fen(fen)?;
        Ok(())
    }

    pub fn fen(&self) -> String {
        self.position.fen()
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn board(&self) -> &crate::board::Board {
        &self.position.board
    }

    pub fn hand(&self) -> &[HandPiece] {
        &self.position.hand
    }

    pub fn turn(&self) -> Color {
        self.position.turn
    }

    pub fn moves(&self) -> MoveList {
        if self.position.is_game_over() {
            MoveList::new()
        } else {
            let mut moves = generate_all_pseudo_legal_moves_from_position(&self.position);
            if self.position.in_draft() {
                moves.retain(|mv| mv.move_type == MoveType::Arata);
            }
            moves
        }
    }

    pub fn make_move(&mut self, mv: &Move) -> Result<(), PositionError> {
        self.position.make_move(mv)
    }

    pub fn undo(&mut self) -> Result<(), PositionError> {
        self.position.unmake_move()
    }

    pub fn in_check(&self) -> bool {
        self.position.in_check(None)
    }

    pub fn is_checkmate(&self) -> bool {
        self.position.is_checkmate()
    }

    pub fn is_stalemate(&self) -> bool {
        self.position.is_stalemate()
    }

    pub fn is_draw(&self) -> bool {
        self.position.is_draw()
    }

    pub fn is_insufficient_material(&self) -> bool {
        self.position.is_insufficient_material()
    }

    pub fn is_fourfold_repetition(&self) -> bool {
        self.position.is_fourfold_repetition()
    }

    pub fn is_game_over(&self) -> bool {
        self.position.is_game_over()
    }

    pub fn history(&self) -> &[HistoryEntry] {
        &self.position.history
    }

    pub fn move_number(&self) -> u32 {
        self.position.move_number
    }
}
