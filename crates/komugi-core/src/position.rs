use std::collections::HashMap;

use arrayvec::ArrayVec;
use thiserror::Error;

use crate::board::{Board, BoardError, Tower};
use crate::fen::{
    encode_fen, parse_fen, ParsedFen, ADVANCED_POSITION, BEGINNER_POSITION, INTERMEDIATE_POSITION,
    INTRO_POSITION,
};
use crate::movegen::{generate_all_legal_moves_from_position, MoveGenerator};
use crate::types::{
    Color, HandPiece, Move, MoveList, MoveType, Piece, PieceType, SetupMode, Square,
};
use crate::zobrist::zobrist_keys;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PositionError {
    #[error("{0}")]
    Fen(String),
    #[error("board error")]
    Board(#[from] BoardError),
    #[error("move color does not match turn")]
    WrongTurn,
    #[error("missing source square for board move")]
    MissingSource,
    #[error("required hand piece does not exist")]
    MissingHandPiece,
    #[error("no move to unmake")]
    EmptyHistory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryEntry {
    pub mv: Move,
    pub from_tower: Tower,
    pub to_tower: Tower,
    pub turn: Color,
    pub drafting_rights: [bool; 2],
    pub move_number: u32,
    pub zobrist_hash: u64,
    pub hand_piece_idx: Option<usize>,
    pub betrayal_hand_indices: ArrayVec<usize, 3>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    pub board: Board,
    pub hand: ArrayVec<HandPiece, 28>,
    pub turn: Color,
    pub mode: SetupMode,
    pub drafting_rights: [bool; 2],
    pub marshal_squares: [Option<Square>; 2],
    pub move_number: u32,
    pub zobrist_hash: u64,
    pub history: Vec<HistoryEntry>,
    repetition_table: HashMap<u64, u8>,
}

impl Position {
    pub fn new(mode: SetupMode) -> Self {
        let fen = match mode {
            SetupMode::Intro => INTRO_POSITION,
            SetupMode::Beginner => BEGINNER_POSITION,
            SetupMode::Intermediate => INTERMEDIATE_POSITION,
            SetupMode::Advanced => ADVANCED_POSITION,
        };
        Self::from_fen(fen).expect("mode starting FEN must be valid")
    }

    pub fn from_fen(fen: &str) -> Result<Self, PositionError> {
        let parsed = parse_fen(fen).map_err(|err| PositionError::Fen(err.to_string()))?;
        Ok(Self::from_parsed(parsed))
    }

    pub fn fen(&self) -> String {
        encode_fen(&self.to_parsed_fen())
    }

    pub fn make_move(&mut self, mv: &Move) -> Result<(), PositionError> {
        if mv.color != self.turn {
            return Err(PositionError::WrongTurn);
        }

        let from_tower = mv
            .from
            .map(|from| {
                self.board
                    .tower_copy(from.square)
                    .expect("source square from legal move must be in bounds")
            })
            .unwrap_or_default();
        let to_tower = self
            .board
            .tower_copy(mv.to.square)
            .expect("destination square from legal move must be in bounds");

        let mut entry = HistoryEntry {
            mv: mv.clone(),
            from_tower,
            to_tower,
            turn: self.turn,
            drafting_rights: self.drafting_rights,
            move_number: self.move_number,
            zobrist_hash: self.zobrist_hash,
            hand_piece_idx: None,
            betrayal_hand_indices: ArrayVec::new(),
        };

        self.apply_move(mv, &mut entry)?;
        self.history.push(entry);

        let mut next_turn = self.turn;
        if self.drafting_rights[color_idx(Color::White)]
            == self.drafting_rights[color_idx(Color::Black)]
        {
            if !(mv.draft_finished && self.turn == Color::White) {
                next_turn = opposite(self.turn);
            }
        } else if !self.drafting_rights[color_idx(Color::Black)] && self.turn == Color::Black {
            next_turn = Color::White;
        } else if !self.drafting_rights[color_idx(Color::White)] && self.turn == Color::White {
            next_turn = Color::Black;
        }

        self.turn = next_turn;
        zobrist_keys().xor_side_to_move(&mut self.zobrist_hash);
        self.move_number = self.move_number.saturating_add(1);
        self.increment_repetition(self.zobrist_hash);
        Ok(())
    }

    pub fn unmake_move(&mut self) -> Result<(), PositionError> {
        let entry = self.history.pop().ok_or(PositionError::EmptyHistory)?;
        self.decrement_repetition(self.zobrist_hash);

        if let Some(from) = entry.mv.from {
            restore_tower(&mut self.board, from.square, entry.from_tower)?;
        }
        restore_tower(&mut self.board, entry.mv.to.square, entry.to_tower)?;
        if let Some(idx) = entry.hand_piece_idx {
            self.hand[idx].count = self.hand[idx].count.saturating_add(1);
        }
        for &idx in &entry.betrayal_hand_indices {
            self.hand[idx].count = self.hand[idx].count.saturating_add(1);
        }

        self.turn = entry.turn;
        self.drafting_rights = entry.drafting_rights;
        self.move_number = entry.move_number;
        self.zobrist_hash = entry.zobrist_hash;
        self.refresh_marshal_cache_for_square(entry.mv.to.square);
        if let Some(from) = entry.mv.from {
            self.refresh_marshal_cache_for_square(from.square);
        }
        Ok(())
    }

    pub fn moves(&self) -> MoveList {
        generate_all_legal_moves_from_position(self)
    }

    pub fn in_draft(&self) -> bool {
        self.drafting_rights[0] || self.drafting_rights[1]
    }

    pub(crate) fn repetition_count(&self, hash: u64) -> u8 {
        self.repetition_table.get(&hash).copied().unwrap_or(0)
    }

    fn from_parsed(parsed: ParsedFen) -> Self {
        let hash =
            zobrist_keys().hash_position(&parsed.board, &parsed.hand, parsed.turn, parsed.drafting);
        let marshal_squares = find_marshal_squares(&parsed.board);
        let mut repetition_table = HashMap::new();
        repetition_table.insert(hash, 1);
        Self {
            board: parsed.board,
            hand: parsed.hand,
            turn: parsed.turn,
            mode: parsed.mode,
            drafting_rights: parsed.drafting,
            marshal_squares,
            move_number: parsed.move_number,
            zobrist_hash: hash,
            history: Vec::new(),
            repetition_table,
        }
    }

    fn to_parsed_fen(&self) -> ParsedFen {
        ParsedFen {
            board: self.board.clone(),
            hand: self.hand.clone(),
            turn: self.turn,
            mode: self.mode,
            drafting: self.drafting_rights,
            move_number: self.move_number,
        }
    }

    fn apply_move(&mut self, mv: &Move, entry: &mut HistoryEntry) -> Result<(), PositionError> {
        let from_square = mv.from.map(|from| from.square);

        match mv.move_type {
            MoveType::Route | MoveType::Tsuke => {
                let from = mv.from.ok_or(PositionError::MissingSource)?;
                let _ = self.board.remove_top(from.square)?;
            }
            MoveType::Capture => {
                let from = mv.from.ok_or(PositionError::MissingSource)?;
                let _ = self.board.remove_top(from.square)?;
                let _ = self.board.remove(mv.to.square, &mv.captured)?;
            }
            MoveType::Betray => {
                let from = mv.from.ok_or(PositionError::MissingSource)?;
                let _ = self.board.remove_top(from.square)?;
                let _ = self.board.convert(mv.to.square, &mv.captured)?;
                for captured in &mv.captured {
                    let (idx, old_count, new_count) =
                        self.decrement_hand_piece(mv.color, captured.piece_type)?;
                    let _ = entry.betrayal_hand_indices.try_push(idx);
                    zobrist_keys().update_hand_count(
                        &mut self.zobrist_hash,
                        captured.piece_type,
                        mv.color,
                        old_count,
                        new_count,
                    );
                }
            }
            MoveType::Arata => {
                let (idx, old_count, new_count) = self.decrement_hand_piece(mv.color, mv.piece)?;
                entry.hand_piece_idx = Some(idx);
                zobrist_keys().update_hand_count(
                    &mut self.zobrist_hash,
                    mv.piece,
                    mv.color,
                    old_count,
                    new_count,
                );
                if mv.draft_finished && self.drafting_rights[color_idx(mv.color)] {
                    zobrist_keys().xor_drafting(&mut self.zobrist_hash, mv.color);
                    self.drafting_rights[color_idx(mv.color)] = false;
                }
            }
        }

        self.board.put(
            Piece {
                piece_type: mv.piece,
                color: mv.color,
            },
            mv.to.square,
        )?;

        self.update_square_hash(mv.to.square, entry.to_tower);
        if let Some(from) = from_square {
            self.update_square_hash(from, entry.from_tower);
            self.refresh_marshal_cache_for_square(from);
        }
        self.refresh_marshal_cache_for_square(mv.to.square);

        Ok(())
    }

    fn decrement_hand_piece(
        &mut self,
        color: Color,
        piece_type: PieceType,
    ) -> Result<(usize, u8, u8), PositionError> {
        for (idx, hp) in self.hand.iter_mut().enumerate() {
            if hp.color == color && hp.piece_type == piece_type && hp.count > 0 {
                let old_count = hp.count;
                hp.count -= 1;
                return Ok((idx, old_count, hp.count));
            }
        }
        Err(PositionError::MissingHandPiece)
    }

    fn update_square_hash(&mut self, square: Square, old_tower: Tower) {
        xor_tower(&mut self.zobrist_hash, square, old_tower);
        let new_tower = self
            .board
            .tower_copy(square)
            .expect("square from legal move must be in bounds");
        xor_tower(&mut self.zobrist_hash, square, new_tower);
    }

    fn refresh_marshal_cache_for_square(&mut self, square: Square) {
        if self.marshal_squares[0] == Some(square) {
            self.marshal_squares[0] = None;
        }
        if self.marshal_squares[1] == Some(square) {
            self.marshal_squares[1] = None;
        }

        if let Some(tower) = self.board.get(square) {
            for piece in tower.iter() {
                if piece.piece_type == PieceType::Marshal {
                    self.marshal_squares[color_idx(piece.color)] = Some(square);
                }
            }
        }
    }

    fn increment_repetition(&mut self, hash: u64) {
        let entry = self.repetition_table.entry(hash).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    fn decrement_repetition(&mut self, hash: u64) {
        if let Some(entry) = self.repetition_table.get_mut(&hash) {
            *entry = entry.saturating_sub(1);
            if *entry == 0 {
                self.repetition_table.remove(&hash);
            }
        }
    }
}

impl MoveGenerator for Position {
    fn generate_moves(&self, _position: &crate::position::Position) -> MoveList {
        self.moves()
    }
}

fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

fn color_idx(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

fn find_marshal_squares(board: &Board) -> [Option<Square>; 2] {
    let mut squares = [None, None];
    for rank in 1..=9 {
        for file in 1..=9 {
            let square = Square::new_unchecked(rank, file);
            let Some(tower) = board.get(square) else {
                continue;
            };
            for piece in tower.iter() {
                if piece.piece_type == PieceType::Marshal {
                    squares[color_idx(piece.color)] = Some(square);
                }
            }
        }
    }
    squares
}

fn xor_tower(hash: &mut u64, square: Square, tower: Tower) {
    for (idx, piece) in tower.pieces()[..usize::from(tower.height())]
        .iter()
        .enumerate()
    {
        if let Some(piece) = piece {
            zobrist_keys().xor_piece(hash, *piece, square, (idx + 1) as u8);
        }
    }
}

fn restore_tower(board: &mut Board, square: Square, tower: Tower) -> Result<(), PositionError> {
    board.set_tower(square, tower)?;
    Ok(())
}
