use crate::constants::{SAN_ARATA, SAN_BETRAY, SAN_TAKE, SAN_TSUKE};
use crate::fen::ParsedFen;
use crate::movegen::generate_all_legal_moves_in_state;
use crate::types::{Move, MoveType};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SanError {
    #[error("illegal move")]
    Illegal,
}

pub fn move_to_san(mv: &Move) -> String {
    let from = mv
        .from
        .map(|from| format!("({}-{}-{})", from.square.rank, from.square.file, from.tier))
        .unwrap_or_default();
    let to = format!(
        "({}-{}-{})",
        mv.to.square.rank, mv.to.square.file, mv.to.tier
    );

    let mut san = String::new();
    if mv.move_type == MoveType::Arata {
        san.push(SAN_ARATA);
    }
    san.push(mv.piece.kanji());
    san.push_str(&from);
    if mv.move_type == MoveType::Capture {
        san.push(SAN_TAKE);
    }
    san.push_str(&to);

    if mv.move_type == MoveType::Betray {
        san.push(SAN_BETRAY);
        for p in &mv.captured {
            san.push(p.piece_type.kanji());
        }
    } else {
        let from_tier = mv.from.map(|f| f.tier).unwrap_or(0);
        if mv.move_type == MoveType::Tsuke || (mv.to.tier != 1 && mv.to.tier > from_tier) {
            san.push(SAN_TSUKE);
        }
    }

    if mv.draft_finished {
        san.push('終');
    }

    san
}

pub fn parse_san(san: &str, state: &ParsedFen) -> Result<Move, SanError> {
    let normalized = san.trim_end_matches(['#', '=']);
    let legal = generate_all_legal_moves_in_state(state);
    for mv in legal {
        if move_to_san(&mv) == normalized {
            return Ok(mv);
        }
    }
    Err(SanError::Illegal)
}
