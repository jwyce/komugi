use crate::constants::max_tier_for_mode;
use crate::types::{Color, Piece, PieceType, SetupMode, Square};
use thiserror::Error;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BoardError {
    #[error("square out of bounds")]
    OutOfBounds,
    #[error("exceeds max tier for setup mode")]
    ExceedsModeMaxTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tower {
    pieces: [Option<Piece>; 3],
    height: u8,
}

impl Tower {
    pub const fn height(&self) -> u8 {
        self.height
    }

    pub const fn pieces(&self) -> [Option<Piece>; 3] {
        self.pieces
    }

    pub fn top(&self) -> Option<(Piece, u8)> {
        if self.height == 0 {
            None
        } else {
            self.pieces[usize::from(self.height - 1)].map(|piece| (piece, self.height))
        }
    }

    pub fn get_top(&self) -> Option<(Piece, u8)> {
        self.top()
    }

    pub fn iter(&self) -> impl Iterator<Item = Piece> + '_ {
        self.pieces[..usize::from(self.height)]
            .iter()
            .copied()
            .flatten()
    }

    fn put(&mut self, piece: Piece) {
        self.pieces[usize::from(self.height)] = Some(piece);
        self.height += 1;
    }

    fn remove_top(&mut self) -> Option<Piece> {
        if self.height == 0 {
            return None;
        }
        let idx = usize::from(self.height - 1);
        let piece = self.pieces[idx].take();
        self.height -= 1;
        piece
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    mode: SetupMode,
    towers: [[Tower; 9]; 9],
}

impl Board {
    pub fn new(mode: SetupMode) -> Self {
        let mut board = Self {
            mode,
            towers: [[Tower::default(); 9]; 9],
        };
        board.setup_starting_position();
        board
    }

    pub fn empty(mode: SetupMode) -> Self {
        Self {
            mode,
            towers: [[Tower::default(); 9]; 9],
        }
    }

    pub const fn mode(&self) -> SetupMode {
        self.mode
    }

    pub fn get(&self, square: Square) -> Option<&Tower> {
        let (r, f) = square_coords(square).expect("valid square");
        let tower = &self.towers[r][f];
        if tower.height == 0 {
            None
        } else {
            Some(tower)
        }
    }

    pub fn get_top(&self, square: Square) -> Option<(Piece, u8)> {
        self.get(square).and_then(Tower::get_top)
    }

    pub(crate) fn tower_copy(&self, square: Square) -> Result<Tower, BoardError> {
        let (r, f) = square_coords(square).ok_or(BoardError::OutOfBounds)?;
        Ok(self.towers[r][f])
    }

    pub(crate) fn set_tower(&mut self, square: Square, tower: Tower) -> Result<(), BoardError> {
        let (r, f) = square_coords(square).ok_or(BoardError::OutOfBounds)?;
        self.towers[r][f] = tower;
        Ok(())
    }

    pub fn put(&mut self, piece: Piece, square: Square) -> Result<(), BoardError> {
        let max_tier = max_tier_for_mode(self.mode);
        let (r, f) = square_coords(square).ok_or(BoardError::OutOfBounds)?;
        let tower = &mut self.towers[r][f];
        if tower.height >= max_tier {
            return Err(BoardError::ExceedsModeMaxTier);
        }
        tower.put(piece);
        Ok(())
    }

    pub fn remove_top(&mut self, square: Square) -> Result<Option<Piece>, BoardError> {
        let (r, f) = square_coords(square).ok_or(BoardError::OutOfBounds)?;
        Ok(self.towers[r][f].remove_top())
    }

    pub fn remove(&mut self, square: Square, pieces: &[Piece]) -> Result<usize, BoardError> {
        let (r, f) = square_coords(square).ok_or(BoardError::OutOfBounds)?;
        let tower = self.towers[r][f];
        let mut keep = [None; 3];
        let mut keep_h = 0u8;
        let mut removed = 0usize;
        for piece in tower.pieces[..usize::from(tower.height)].iter().flatten() {
            if pieces.iter().any(|target| piece == target) {
                removed += 1;
            } else {
                keep[usize::from(keep_h)] = Some(*piece);
                keep_h += 1;
            }
        }

        self.towers[r][f] = Tower {
            pieces: keep,
            height: keep_h,
        };
        Ok(removed)
    }

    pub fn convert(&mut self, square: Square, pieces: &[Piece]) -> Result<usize, BoardError> {
        let (r, f) = square_coords(square).ok_or(BoardError::OutOfBounds)?;
        let tower = &mut self.towers[r][f];
        let mut converted = 0usize;

        for piece in tower.pieces[..usize::from(tower.height)]
            .iter_mut()
            .flatten()
        {
            if pieces.contains(piece) {
                converted += 1;
                piece.color = match piece.color {
                    Color::White => Color::Black,
                    Color::Black => Color::White,
                };
            }
        }

        Ok(converted)
    }

    fn setup_starting_position(&mut self) {
        match self.mode {
            SetupMode::Intro => {
                self.load_backrank(Color::Black, false);
                self.load_soldier_rank(Color::Black);
                self.load_soldier_rank(Color::White);
                self.load_backrank(Color::White, false);
            }
            SetupMode::Beginner => {
                self.load_backrank(Color::Black, true);
                self.load_soldier_rank(Color::Black);
                self.load_soldier_rank(Color::White);
                self.load_backrank(Color::White, true);
            }
            SetupMode::Intermediate | SetupMode::Advanced => {}
        }
    }

    fn load_backrank(&mut self, color: Color, beginner: bool) {
        let (rank_main, rank_side) = if color == Color::Black {
            (1, 2)
        } else {
            (9, 8)
        };

        let _ = self.put(
            Piece::new(PieceType::LieutenantGeneral, color),
            Square::new_unchecked(rank_main, 6),
        );
        let _ = self.put(
            Piece::new(PieceType::Marshal, color),
            Square::new_unchecked(rank_main, 5),
        );
        let _ = self.put(
            Piece::new(PieceType::General, color),
            Square::new_unchecked(rank_main, 4),
        );
        let _ = self.put(
            Piece::new(PieceType::Lancer, color),
            Square::new_unchecked(rank_side, 5),
        );

        if beginner {
            let _ = self.put(
                Piece::new(PieceType::Rider, color),
                Square::new_unchecked(rank_side, 8),
            );
            let _ = self.put(
                Piece::new(PieceType::Archer, color),
                Square::new_unchecked(rank_side, 7),
            );
            let _ = self.put(
                Piece::new(PieceType::Archer, color),
                Square::new_unchecked(rank_side, 3),
            );
            let _ = self.put(
                Piece::new(PieceType::Rider, color),
                Square::new_unchecked(rank_side, 2),
            );
        } else {
            let _ = self.put(
                Piece::new(PieceType::Spy, color),
                Square::new_unchecked(rank_side, 2),
            );
            let _ = self.put(
                Piece::new(PieceType::Spy, color),
                Square::new_unchecked(rank_side, 8),
            );
        }
    }

    fn load_soldier_rank(&mut self, color: Color) {
        let rank = if color == Color::Black { 3 } else { 7 };
        let warrior = Piece::new(PieceType::Warrior, color);
        let fortress = Piece::new(PieceType::Fortress, color);
        let soldier = Piece::new(PieceType::Soldier, color);

        let _ = self.put(soldier, Square::new_unchecked(rank, 9));
        let _ = self.put(fortress, Square::new_unchecked(rank, 7));
        let _ = self.put(warrior, Square::new_unchecked(rank, 6));
        let _ = self.put(soldier, Square::new_unchecked(rank, 5));
        let _ = self.put(warrior, Square::new_unchecked(rank, 4));
        let _ = self.put(fortress, Square::new_unchecked(rank, 3));
        let _ = self.put(soldier, Square::new_unchecked(rank, 1));
    }
}

pub fn square_index(square: Square) -> Option<(usize, usize)> {
    square_coords(square)
}

pub fn square_flat_index(square: Square) -> Option<usize> {
    let (rank, file) = square_coords(square)?;
    Some(rank * 9 + file)
}

fn square_coords(square: Square) -> Option<(usize, usize)> {
    if !(1..=9).contains(&square.rank) || !(1..=9).contains(&square.file) {
        return None;
    }
    Some((usize::from(square.rank - 1), usize::from(9 - square.file)))
}
