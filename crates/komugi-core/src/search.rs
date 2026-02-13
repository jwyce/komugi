use crate::position::Position;
use crate::types::{Move, Score};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SearchLimits {
    pub depth: Option<u8>,
    pub nodes: Option<u64>,
    pub time_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: Score,
    pub nodes_searched: u64,
}

pub trait Searcher {
    fn search(&mut self, position: &Position, limits: SearchLimits) -> SearchResult;
}
