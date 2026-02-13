use crate::board::{Board, BoardError};
use crate::types::{Color, HandPiece, Move, MoveType, Piece, PieceType, SetupMode, Square};
use arrayvec::ArrayVec;
use thiserror::Error;

pub const INTRO_POSITION: &str =
    "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1/j2n2r2d1 w 0 - 1";
pub const BEGINNER_POSITION: &str =
    "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1";
pub const INTERMEDIATE_POSITION: &str =
    "9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1";
pub const ADVANCED_POSITION: &str =
    "9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 wb 1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedFen {
    pub board: Board,
    pub hand: ArrayVec<HandPiece, 28>,
    pub turn: Color,
    pub mode: SetupMode,
    pub drafting: [bool; 2],
    pub move_number: u32,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FenError {
    #[error("invalid fen")]
    Invalid,
    #[error("{0}")]
    Validation(String),
    #[error("invalid piece")]
    InvalidPiece,
    #[error("invalid hand")]
    InvalidHand,
    #[error("invalid mode")]
    InvalidMode,
    #[error("board error")]
    Board(#[from] BoardError),
}

pub fn validate_fen(fen: &str) -> Result<(), FenError> {
    let parts: Vec<&str> = fen.split(' ').collect();
    if parts.len() != 6 {
        return Err(FenError::Validation(format!(
            "expected 6 fields, received {}",
            parts.len()
        )));
    }

    let ranks: Vec<&str> = parts[0].split('/').collect();
    if ranks.len() != 9 {
        return Err(FenError::Validation(format!(
            "1st field (piece positions) is invalid [expected 9 ranks, received {}]",
            ranks.len()
        )));
    }

    const ALL: &str = "mgijwnrsfdcaktMGIJWNRSFDCAKT";
    for (i, rank) in ranks.iter().enumerate() {
        let mut count = 0u8;
        for part in rank.split('|') {
            if part.contains(':') {
                let tower: Vec<&str> = part.split(':').collect();
                if tower.len() > 3 {
                    return Err(FenError::Validation(format!(
                        "1st field (piece positions) is invalid [tower size] @({}-{})",
                        i + 1,
                        count + 1
                    )));
                }
                if !tower.iter().all(|p| p.len() == 1 && ALL.contains(*p)) {
                    return Err(FenError::Validation(
                        "1st field (piece positions) is invalid [invalid piece]".to_string(),
                    ));
                }
                count += 1;
            } else {
                for ch in part.chars() {
                    if ch.is_ascii_digit() {
                        count += ch.to_digit(10).unwrap_or(0) as u8;
                    } else {
                        if !ALL.contains(ch) {
                            return Err(FenError::Validation(
                                "1st field (piece positions) is invalid [invalid piece]"
                                    .to_string(),
                            ));
                        }
                        count += 1;
                    }
                }
            }
        }
        if count != 9 {
            return Err(FenError::Validation(format!(
                "1st field (piece positions) is invalid [expected 9 squares, received {}] in rank: {}",
                count,
                i + 1
            )));
        }
    }

    let hands: Vec<&str> = parts[1].split('/').collect();
    if hands.len() != 2 {
        return Err(FenError::Validation(format!(
            "2nd field (hand pieces) is invalid [expected 2 hands, received {}]",
            hands.len()
        )));
    }

    let white_hand = hands[0];
    let black_hand = hands[1];
    if (white_hand.len() == 1 && white_hand != "-") || (black_hand.len() == 1 && black_hand != "-")
    {
        return Err(FenError::Validation(
            "2nd field (hand pieces) is invalid [invalid piece]".to_string(),
        ));
    }
    if (white_hand.len() > 1 && !white_hand.len().is_multiple_of(2))
        || (black_hand.len() > 1 && !black_hand.len().is_multiple_of(2))
    {
        return Err(FenError::Validation(
            "2nd field (hand pieces) is invalid [invalid piece count]".to_string(),
        ));
    }

    for (segment, valid) in [
        (white_hand, "MGIJWNRSFDCAKT"),
        (black_hand, "mgijwnrsfdcakt"),
    ] {
        if segment.len() > 1 {
            let chars: Vec<char> = segment.chars().collect();
            for i in (0..chars.len()).step_by(2) {
                if !valid.contains(chars[i]) {
                    return Err(FenError::Validation(
                        "2nd field (hand pieces) is invalid [invalid piece]".to_string(),
                    ));
                }
                if chars[i + 1].to_digit(10).is_none() {
                    return Err(FenError::Validation(
                        "2nd field (hand pieces) is invalid [invalid count]".to_string(),
                    ));
                }
            }
        }
    }

    if parts[2] != "w" && parts[2] != "b" {
        return Err(FenError::Validation(format!(
            "3rd field (active player) is invalid [expected 'w' or 'b', received {}]",
            parts[2]
        )));
    }
    if !matches!(parts[3], "0" | "1" | "2" | "3") {
        return Err(FenError::Validation(format!(
            "4th field (setup mode) is invalid [expected '0', '1', '2', or '3', received {}]",
            parts[3]
        )));
    }
    if parts[4] != "-" && !matches!(parts[4], "w" | "b" | "wb") {
        return Err(FenError::Validation(
            "5th field (drafting availability) is invalid".to_string(),
        ));
    }
    if parts[5].parse::<u32>().is_err() {
        return Err(FenError::Validation(format!(
            "6th field (full move number) is invalid [expected an integer, received {}]",
            parts[5]
        )));
    }

    Ok(())
}

pub fn parse_fen(fen: &str) -> Result<ParsedFen, FenError> {
    validate_fen(fen)?;
    let parts: Vec<&str> = fen.split(' ').collect();
    let turn = if parts[2] == "w" {
        Color::White
    } else {
        Color::Black
    };
    let mode = SetupMode::from_code(parts[3].parse::<u8>().map_err(|_| FenError::InvalidMode)?)
        .ok_or(FenError::InvalidMode)?;

    let mut board = Board::empty(SetupMode::Advanced);
    for (ri, rank_desc) in parts[0].split('/').enumerate() {
        let rank = (ri + 1) as u8;
        let mut files: Vec<Vec<Piece>> = Vec::with_capacity(9);

        for partial in rank_desc.split('|') {
            if partial.contains(':') {
                let mut tower: Vec<Piece> = Vec::new();
                for code in partial.split(':') {
                    let code = code.trim();
                    let mut chars = code.chars();
                    let ch = chars.next().ok_or(FenError::InvalidPiece)?;
                    if chars.next().is_some() {
                        return Err(FenError::InvalidPiece);
                    }
                    tower.push(decode_piece(ch)?);
                }
                files.push(tower);
            } else {
                for ch in partial.chars() {
                    if ch.is_ascii_whitespace() {
                        continue;
                    }
                    if let Some(n) = ch.to_digit(10) {
                        for _ in 0..n {
                            files.push(Vec::new());
                        }
                    } else {
                        files.push(vec![decode_piece(ch)?]);
                    }
                }
            }
        }

        if files.len() != 9 {
            return Err(FenError::Invalid);
        }

        for (fi, tower) in files.into_iter().enumerate() {
            let file = (9 - fi) as u8;
            let square = Square::new_unchecked(rank, file);
            for piece in tower {
                board.put(piece, square)?;
            }
        }
    }

    let mut hand = ArrayVec::<HandPiece, 28>::new();
    let hp: Vec<&str> = parts[1].split('/').collect();
    parse_hand_segment(hp[0], Color::White, &mut hand)?;
    parse_hand_segment(hp[1], Color::Black, &mut hand)?;

    Ok(ParsedFen {
        board,
        hand,
        turn,
        mode,
        drafting: [parts[4].contains('w'), parts[4].contains('b')],
        move_number: parts[5].parse::<u32>().map_err(|_| FenError::Invalid)?,
    })
}

pub fn encode_fen(state: &ParsedFen) -> String {
    let mut placement = String::new();
    for rank in 1..=9u8 {
        let mut empties = 0u8;
        for file in (1..=9u8).rev() {
            let square = Square::new_unchecked(rank, file);
            if let Some(tower) = state.board.get(square) {
                if empties > 0 {
                    placement.push(char::from_digit(u32::from(empties), 10).unwrap_or('1'));
                    empties = 0;
                }
                if tower.height() > 1 {
                    placement.push('|');
                    for (i, piece) in tower.iter().enumerate() {
                        if i > 0 {
                            placement.push(':');
                        }
                        placement.push(encode_piece(piece));
                    }
                    placement.push('|');
                } else if let Some(piece) = tower.pieces()[0] {
                    placement.push(encode_piece(piece));
                }
            } else {
                empties += 1;
            }
        }
        if empties > 0 {
            placement.push(char::from_digit(u32::from(empties), 10).unwrap_or('1'));
        }
        if rank < 9 {
            placement.push('/');
        }
    }

    format!(
        "{} {}/{} {} {} {} {}",
        placement,
        encode_hand(state, Color::White),
        encode_hand(state, Color::Black),
        state.turn.to_code(),
        state.mode.to_code(),
        match (state.drafting[0], state.drafting[1]) {
            (false, false) => "-".to_string(),
            (true, false) => "w".to_string(),
            (false, true) => "b".to_string(),
            (true, true) => "wb".to_string(),
        },
        state.move_number
    )
}

pub fn apply_move_to_fen(fen: &str, mv: &Move) -> Result<String, FenError> {
    let mut parsed = parse_fen(fen)?;
    match mv.move_type {
        MoveType::Route | MoveType::Tsuke => {
            let from = mv.from.ok_or(FenError::Invalid)?;
            let _ = parsed
                .board
                .remove_top(from.square)?
                .ok_or(BoardError::OutOfBounds)?;
        }
        MoveType::Capture => {
            let from = mv.from.ok_or(FenError::Invalid)?;
            let _ = parsed
                .board
                .remove_top(from.square)?
                .ok_or(BoardError::OutOfBounds)?;
            let _ = parsed.board.remove(mv.to.square, &mv.captured);
        }
        MoveType::Betray => {
            let from = mv.from.ok_or(FenError::Invalid)?;
            let _ = parsed
                .board
                .remove_top(from.square)?
                .ok_or(BoardError::OutOfBounds)?;
            let _ = parsed.board.convert(mv.to.square, &mv.captured);
        }
        MoveType::Arata => {
            decrement_hand(
                &mut parsed.hand,
                Piece {
                    piece_type: mv.piece,
                    color: mv.color,
                },
            )?;
            if mv.draft_finished {
                parsed.drafting[color_idx(mv.color)] = false;
            }
        }
    }

    parsed.board.put(
        Piece {
            piece_type: mv.piece,
            color: mv.color,
        },
        mv.to.square,
    )?;
    parsed.turn = opposite(parsed.turn);
    Ok(encode_fen(&parsed))
}

fn parse_hand_segment(
    segment: &str,
    color: Color,
    hand: &mut ArrayVec<HandPiece, 28>,
) -> Result<(), FenError> {
    if segment == "-" {
        return Ok(());
    }
    let chars: Vec<char> = segment.chars().collect();
    if !chars.len().is_multiple_of(2) {
        return Err(FenError::InvalidHand);
    }
    let mut i = 0usize;
    while i < chars.len() {
        let piece_type =
            PieceType::from_fen_code(chars[i].to_ascii_lowercase()).ok_or(FenError::InvalidHand)?;
        let count = chars[i + 1].to_digit(10).ok_or(FenError::InvalidHand)? as u8;
        let _ = hand.try_push(HandPiece {
            piece_type,
            color,
            count,
        });
        i += 2;
    }
    Ok(())
}

fn decode_piece(ch: char) -> Result<Piece, FenError> {
    let color = if ch.is_ascii_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let piece_type =
        PieceType::from_fen_code(ch.to_ascii_lowercase()).ok_or(FenError::InvalidPiece)?;
    Ok(Piece { piece_type, color })
}

fn encode_piece(piece: Piece) -> char {
    let code = piece.piece_type.fen_code();
    if piece.color == Color::White {
        code.to_ascii_uppercase()
    } else {
        code
    }
}

fn encode_hand(state: &ParsedFen, color: Color) -> String {
    let mut out = String::new();
    for hp in state
        .hand
        .iter()
        .filter(|h| h.color == color && h.count > 0)
    {
        let code = hp.piece_type.fen_code();
        out.push(if color == Color::White {
            code.to_ascii_uppercase()
        } else {
            code
        });
        out.push(char::from_digit(u32::from(hp.count), 10).unwrap_or('1'));
    }
    if out.is_empty() {
        out.push('-');
    }
    out
}

fn decrement_hand(hand: &mut ArrayVec<HandPiece, 28>, piece: Piece) -> Result<(), FenError> {
    for hp in hand.iter_mut() {
        if hp.color == piece.color && hp.piece_type == piece.piece_type && hp.count > 0 {
            hp.count -= 1;
            return Ok(());
        }
    }
    Err(FenError::InvalidHand)
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
