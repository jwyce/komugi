use std::collections::HashSet;
use std::time::{Duration, Instant};

use komugi_core::{
    is_marshal_captured, zobrist_keys, Color, Evaluator, Move, MoveType, PieceType, Position,
    Score, SearchLimits, SearchResult, Searcher,
};

use crate::classical::ClassicalEval;
use crate::tt::{Bound, Entry, TranspositionTable};

const MATE_SCORE: i32 = 30_000;
const DEFAULT_MAX_DEPTH: u8 = 4;
const CHECK_INTERVAL_NODES: u64 = 2_048;
const CHECK_EXTENSION_MAX_PLY: u8 = 32;
const PV_WALK_MAX_DEPTH: usize = 14;

#[derive(Debug, Clone, Copy)]
pub struct AlphaBetaConfig {
    pub tt_size_mb: usize,
    pub max_depth: u8,
}

impl Default for AlphaBetaConfig {
    fn default() -> Self {
        Self {
            tt_size_mb: 16,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlphaBetaResult {
    pub best_move: Option<Move>,
    pub score: Score,
    pub depth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvLine {
    pub moves: Vec<Move>,
    pub score: Score,
    pub depth: u8,
    pub nodes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiPvResult {
    pub lines: Vec<PvLine>,
    pub total_nodes: u64,
}

pub struct AlphaBetaSearcher {
    tt: TranspositionTable,
    eval: Box<dyn Evaluator>,
    max_depth: u8,
    nodes: u64,
    stop: bool,
    node_limit: Option<u64>,
    time_limit: Option<Duration>,
    started_at: Instant,
    last_completed_depth: u8,
    killers: [[Option<Move>; 2]; 64],
    history: [[i32; 81]; 28],
}

impl std::fmt::Debug for AlphaBetaSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlphaBetaSearcher")
            .field("tt", &self.tt)
            .field("eval", &"<Evaluator>")
            .field("max_depth", &self.max_depth)
            .field("nodes", &self.nodes)
            .field("stop", &self.stop)
            .field("node_limit", &self.node_limit)
            .field("time_limit", &self.time_limit)
            .field("started_at", &self.started_at)
            .field("last_completed_depth", &self.last_completed_depth)
            .field("killers", &self.killers)
            .field("history", &self.history)
            .finish()
    }
}

impl AlphaBetaSearcher {
    pub fn new(config: AlphaBetaConfig) -> Self {
        Self::with_eval(config, Box::new(ClassicalEval::new()))
    }

    pub fn with_eval(config: AlphaBetaConfig, eval: Box<dyn Evaluator>) -> Self {
        Self {
            tt: TranspositionTable::with_size_mb(config.tt_size_mb),
            eval,
            max_depth: config.max_depth,
            nodes: 0,
            stop: false,
            node_limit: None,
            time_limit: None,
            started_at: Instant::now(),
            last_completed_depth: 0,
            killers: std::array::from_fn(|_| [None, None]),
            history: [[0; 81]; 28],
        }
    }

    pub fn search_with_info(
        &mut self,
        position: &Position,
        limits: SearchLimits,
    ) -> AlphaBetaResult {
        self.search_with_info_internal(position, limits, &[])
    }

    pub fn search_with_exclusions(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        exclude_moves: &[Move],
    ) -> AlphaBetaResult {
        self.search_with_info_internal(position, limits, exclude_moves)
    }

    pub fn search_multi_pv(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        num_pv: usize,
    ) -> MultiPvResult {
        let mut exclude_moves = Vec::new();
        let mut lines = Vec::with_capacity(num_pv);
        let mut total_nodes = 0u64;

        for _ in 0..num_pv {
            let result = self.search_with_exclusions(position, limits, &exclude_moves);
            total_nodes = total_nodes.saturating_add(result.nodes);

            let Some(best_move) = result.best_move else {
                break;
            };

            let mut line_moves = result.pv;
            if line_moves.first() != Some(&best_move) {
                line_moves.insert(0, best_move.clone());
            }

            lines.push(PvLine {
                moves: line_moves,
                score: result.score,
                depth: result.depth,
                nodes: result.nodes,
            });
            exclude_moves.push(best_move);
        }

        lines.sort_by(|a, b| b.score.cmp(&a.score));

        MultiPvResult { lines, total_nodes }
    }

    fn search_with_info_internal(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        exclude_moves: &[Move],
    ) -> AlphaBetaResult {
        self.nodes = 0;
        self.stop = false;
        self.last_completed_depth = 0;
        self.node_limit = limits.nodes;
        self.time_limit = limits.time_ms.map(Duration::from_millis);
        self.started_at = Instant::now();
        self.clear_killers();
        self.age_history();

        let max_depth = limits.depth.unwrap_or(self.max_depth);
        let mut best_move = None;
        let mut best_score = self.terminal_score(position, 0);
        let mut pv = Vec::new();
        let mut working = position.clone();

        for depth in 1..=max_depth {
            if self.should_stop() {
                break;
            }

            let mut delta = 250;
            let mut alpha_window = if depth > 1 {
                best_score.saturating_sub(delta)
            } else {
                -MATE_SCORE
            };
            let mut beta_window = if depth > 1 {
                best_score.saturating_add(delta)
            } else {
                MATE_SCORE
            };
            let mut failures = 0;

            match loop {
                match self.search_root(
                    &mut working,
                    depth,
                    alpha_window,
                    beta_window,
                    exclude_moves,
                ) {
                    Ok((candidate, score))
                        if depth == 1
                            || (score > alpha_window && score < beta_window)
                            || (alpha_window == -MATE_SCORE && beta_window == MATE_SCORE) =>
                    {
                        break Ok((candidate, score));
                    }
                    Ok((_candidate, _score)) => {
                        failures += 1;
                        if failures >= 2 {
                            alpha_window = -MATE_SCORE;
                            beta_window = MATE_SCORE;
                        } else {
                            delta *= 2;
                            alpha_window = best_score.saturating_sub(delta);
                            beta_window = best_score.saturating_add(delta);
                        }
                    }
                    Err(err) => break Err(err),
                }
            } {
                Ok((candidate, score)) => {
                    best_move = candidate;
                    best_score = score;
                    self.last_completed_depth = depth;
                    let mut pv_position = position.clone();
                    pv = self.extract_pv_line(&mut pv_position);
                }
                Err(AbortSearch) => break,
            }
        }

        AlphaBetaResult {
            best_move,
            score: Score(best_score),
            depth: self.last_completed_depth,
            nodes: self.nodes,
            pv,
        }
    }

    pub fn transposition_table_len(&self) -> usize {
        self.tt.len()
    }

    fn search_root(
        &mut self,
        position: &mut Position,
        depth: u8,
        mut alpha: i32,
        beta: i32,
        exclude_moves: &[Move],
    ) -> Result<(Option<Move>, i32), AbortSearch> {
        self.bump_nodes()?;

        if is_marshal_captured(position) {
            return Ok((None, -MATE_SCORE));
        }
        if position.is_fourfold_repetition() || position.is_insufficient_material() {
            return Ok((None, 0));
        }

        let key = position.zobrist_hash;
        let tt_move = self.tt.probe(key).and_then(|entry| entry.best_move.clone());
        let in_check = position.in_check(None);
        let extension = if in_check && 1 < CHECK_EXTENSION_MAX_PLY {
            1u8
        } else {
            0u8
        };
        let next_depth = depth.saturating_sub(1).saturating_add(extension);
        let mut moves = position.moves().into_iter().collect::<Vec<_>>();
        moves.retain(|mv| !exclude_moves.contains(mv));
        self.order_moves(&mut moves, tt_move.as_ref(), 0);

        if moves.is_empty() {
            if in_check {
                return Ok((None, -MATE_SCORE));
            }
            return Ok((None, 0));
        }

        let original_alpha = alpha;
        let mut best_score = -MATE_SCORE;
        let mut best_order_score = -MATE_SCORE;
        let mut best_move = None;
        let mut searched_moves = 0usize;

        for mv in moves {
            if position.make_move(&mv).is_err() {
                continue;
            }
            let score = if searched_moves == 0 {
                match self.negamax(position, next_depth, -beta, -alpha, 1, false) {
                    Ok(score) => -score,
                    Err(err) => {
                        let _ = position.unmake_move();
                        return Err(err);
                    }
                }
            } else {
                let mut score =
                    match self.negamax(position, next_depth, -alpha - 1, -alpha, 1, false) {
                        Ok(score) => -score,
                        Err(err) => {
                            let _ = position.unmake_move();
                            return Err(err);
                        }
                    };
                if score > alpha && score < beta {
                    score = match self.negamax(position, next_depth, -beta, -alpha, 1, false) {
                        Ok(score) => -score,
                        Err(err) => {
                            let _ = position.unmake_move();
                            return Err(err);
                        }
                    };
                }
                score
            };
            let _ = position.unmake_move();
            searched_moves += 1;

            let order_score = score + if self.is_capture(&mv) { 120 } else { 0 };
            if order_score > best_order_score {
                best_score = score;
                best_order_score = order_score;
                best_move = Some(mv.clone());
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };

        self.tt.store(Entry {
            key,
            depth,
            score: best_score,
            bound,
            best_move: best_move.clone(),
        });

        Ok((best_move, best_score))
    }

    fn negamax(
        &mut self,
        position: &mut Position,
        depth: u8,
        mut alpha: i32,
        beta: i32,
        ply: u8,
        is_null_move: bool,
    ) -> Result<i32, AbortSearch> {
        self.bump_nodes()?;

        if is_marshal_captured(position) {
            return Ok(-MATE_SCORE + i32::from(ply));
        }
        if position.is_fourfold_repetition() || position.is_insufficient_material() {
            return Ok(0);
        }
        if depth == 0 {
            return self.quiescence(position, alpha, beta, ply);
        }

        let key = position.zobrist_hash;
        let mut beta = beta;
        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.best_move.clone();
            if entry.depth >= depth {
                match entry.bound {
                    Bound::Exact => return Ok(entry.score),
                    Bound::Lower => alpha = alpha.max(entry.score),
                    Bound::Upper => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return Ok(entry.score);
                }
            }
        }

        let in_check = position.in_check(None);
        let extension = if in_check && ply < CHECK_EXTENSION_MAX_PLY {
            1u8
        } else {
            0u8
        };

        if depth >= 3
            && !in_check
            && !is_null_move
            && !position.in_draft()
            && self.hand_count(position, position.turn) < 4
        {
            let r = if depth >= 7 { 3 } else { 2 };
            self.make_null_move(position);
            let null_score = self.negamax(
                position,
                depth.saturating_sub(1 + r),
                -beta,
                -beta + 1,
                ply + 1,
                true,
            );
            self.unmake_null_move(position);
            let score = -null_score?;
            if score >= beta {
                return Ok(beta);
            }
        }

        let mut moves = position.moves().into_iter().collect::<Vec<_>>();
        self.order_moves(&mut moves, tt_move.as_ref(), ply);

        if moves.is_empty() {
            if in_check {
                return Ok(-MATE_SCORE + i32::from(ply));
            }
            return Ok(0);
        }

        let original_alpha = alpha;
        let mut best_score = -MATE_SCORE;
        let mut best_move = None;
        let static_eval = if depth <= 2 && !in_check {
            Some(self.evaluate(position))
        } else {
            None
        };

        let mut searched_moves = 0usize;
        for mv in moves {
            let is_quiet = self.is_quiet(mv.move_type);

            if position.make_move(&mv).is_err() {
                continue;
            }

            if let Some(eval) = static_eval {
                let margin = if depth == 1 { 240 } else { 520 };
                if searched_moves > 0 && eval + margin < alpha && is_quiet {
                    let _ = position.unmake_move();
                    continue;
                }
            }

            let mut reduction = 0;
            if extension == 0 && depth >= 3 && searched_moves >= 3 && is_quiet && !in_check {
                reduction = 1;
                if depth >= 6 && searched_moves >= 8 {
                    reduction = 2;
                }
            }

            let next_depth = depth.saturating_sub(1).saturating_add(extension);
            let reduced_depth = next_depth.saturating_sub(reduction);

            let score = if searched_moves == 0 {
                match self.negamax(position, next_depth, -beta, -alpha, ply + 1, false) {
                    Ok(score) => -score,
                    Err(err) => {
                        let _ = position.unmake_move();
                        return Err(err);
                    }
                }
            } else {
                let mut score =
                    match self.negamax(position, reduced_depth, -alpha - 1, -alpha, ply + 1, false)
                    {
                        Ok(score) => -score,
                        Err(err) => {
                            let _ = position.unmake_move();
                            return Err(err);
                        }
                    };

                if reduction > 0 && score > alpha {
                    score = match self.negamax(
                        position,
                        next_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        false,
                    ) {
                        Ok(score) => -score,
                        Err(err) => {
                            let _ = position.unmake_move();
                            return Err(err);
                        }
                    };
                }
                if score > alpha && score < beta {
                    score = match self.negamax(position, next_depth, -beta, -alpha, ply + 1, false)
                    {
                        Ok(score) => -score,
                        Err(err) => {
                            let _ = position.unmake_move();
                            return Err(err);
                        }
                    };
                }
                score
            };
            let _ = position.unmake_move();
            searched_moves += 1;

            if score > best_score {
                best_score = score;
                best_move = Some(mv.clone());
            }

            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if is_quiet {
                    self.store_killer(&mv, ply);
                    self.bump_history(&mv, depth);
                }
                break;
            }
        }

        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };

        self.tt.store(Entry {
            key,
            depth,
            score: best_score,
            bound,
            best_move,
        });

        Ok(best_score)
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: u8,
    ) -> Result<i32, AbortSearch> {
        self.bump_nodes()?;

        if is_marshal_captured(position) {
            return Ok(-MATE_SCORE + i32::from(ply));
        }
        if position.is_fourfold_repetition() {
            return Ok(0);
        }

        let stand_pat = self.evaluate(position);
        if stand_pat >= beta {
            return Ok(beta);
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut captures = position
            .moves()
            .into_iter()
            .filter(|mv| self.is_capture(mv))
            .collect::<Vec<_>>();
        self.order_moves(&mut captures, None, ply);

        for mv in captures {
            let mut gain: i32 = mv.captured.iter().map(|p| piece_value(p.piece_type)).sum();
            if mv.move_type == MoveType::Betray {
                gain *= 2;
            }
            if stand_pat + gain + 120 < alpha {
                continue;
            }

            if position.make_move(&mv).is_err() {
                continue;
            }
            let score = match self.quiescence(position, -beta, -alpha, ply + 1) {
                Ok(score) => -score,
                Err(err) => {
                    let _ = position.unmake_move();
                    return Err(err);
                }
            };
            let _ = position.unmake_move();

            if score >= beta {
                return Ok(beta);
            }
            if score > alpha {
                alpha = score;
            }
        }

        Ok(alpha)
    }

    fn evaluate(&self, position: &Position) -> i32 {
        let score = self.eval.evaluate(position).0;
        if position.turn == Color::White {
            score
        } else {
            -score
        }
    }

    fn terminal_score(&self, position: &Position, ply: u8) -> i32 {
        if position.is_checkmate() || is_marshal_captured(position) {
            -MATE_SCORE + i32::from(ply)
        } else {
            0
        }
    }

    fn extract_pv_line(&self, position: &mut Position) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut seen = HashSet::new();
        let base_history_len = position.history.len();

        loop {
            if pv.len() >= PV_WALK_MAX_DEPTH {
                break;
            }

            let key = position.zobrist_hash;
            if !seen.insert(key) {
                break;
            }

            let Some(entry) = self.tt.probe(key) else {
                break;
            };
            let Some(best_move) = entry.best_move.clone() else {
                break;
            };

            if position.make_move(&best_move).is_err() {
                break;
            }

            pv.push(best_move);
        }

        while position.history.len() > base_history_len {
            let _ = position.unmake_move();
        }

        pv
    }

    fn order_moves(&self, moves: &mut [Move], tt_move: Option<&Move>, ply: u8) {
        moves.sort_by_key(|mv| -self.move_order_key(mv, tt_move, ply));
    }

    fn move_order_key(&self, mv: &Move, tt_move: Option<&Move>, ply: u8) -> i32 {
        if tt_move.is_some_and(|tt| tt == mv) {
            return 1_000_000;
        }

        if self.is_capture(mv) {
            return 500_000 + self.mvv_lva(mv);
        }

        if self.is_killer(mv, ply) {
            return 400_000;
        }

        if self.is_quiet(mv.move_type) {
            return self.history[self.piece_color_idx(mv.color, mv.piece)]
                [self.square_idx(mv.to.square)];
        }

        0
    }

    fn is_capture(&self, mv: &Move) -> bool {
        matches!(mv.move_type, MoveType::Capture | MoveType::Betray)
    }

    fn is_quiet(&self, move_type: MoveType) -> bool {
        matches!(
            move_type,
            MoveType::Route | MoveType::Tsuke | MoveType::Arata
        )
    }

    fn is_killer(&self, mv: &Move, ply: u8) -> bool {
        let ply_idx = usize::from(ply.min(63));
        self.killers[ply_idx][0]
            .as_ref()
            .is_some_and(|killer| killer == mv)
            || self.killers[ply_idx][1]
                .as_ref()
                .is_some_and(|killer| killer == mv)
    }

    fn store_killer(&mut self, mv: &Move, ply: u8) {
        let ply_idx = usize::from(ply.min(63));
        if self.killers[ply_idx][0]
            .as_ref()
            .is_some_and(|killer| killer == mv)
        {
            return;
        }
        self.killers[ply_idx][1] = self.killers[ply_idx][0].clone();
        self.killers[ply_idx][0] = Some(mv.clone());
    }

    fn bump_history(&mut self, mv: &Move, depth: u8) {
        let piece_idx = self.piece_color_idx(mv.color, mv.piece);
        let to_idx = self.square_idx(mv.to.square);
        let bonus = i32::from(depth) * i32::from(depth);
        self.history[piece_idx][to_idx] = self.history[piece_idx][to_idx].saturating_add(bonus);
    }

    fn clear_killers(&mut self) {
        self.killers = std::array::from_fn(|_| [None, None]);
    }

    fn age_history(&mut self) {
        for piece in &mut self.history {
            for score in piece {
                *score /= 2;
            }
        }
    }

    fn piece_color_idx(&self, color: Color, piece: PieceType) -> usize {
        color as usize * 14 + piece as usize
    }

    fn square_idx(&self, square: komugi_core::Square) -> usize {
        usize::from(square.rank - 1) * 9 + usize::from(9 - square.file)
    }

    fn hand_count(&self, position: &Position, color: Color) -> u8 {
        position
            .hand
            .iter()
            .filter(|hp| hp.color == color)
            .map(|hp| hp.count)
            .sum()
    }

    fn make_null_move(&self, position: &mut Position) {
        position.turn = opposite(position.turn);
        zobrist_keys().xor_side_to_move(&mut position.zobrist_hash);
    }

    fn unmake_null_move(&self, position: &mut Position) {
        position.turn = opposite(position.turn);
        zobrist_keys().xor_side_to_move(&mut position.zobrist_hash);
    }

    fn mvv_lva(&self, mv: &Move) -> i32 {
        let victims = mv
            .captured
            .iter()
            .map(|piece| piece_value(piece.piece_type))
            .sum::<i32>();
        victims.saturating_mul(16) - piece_value(mv.piece)
    }

    fn bump_nodes(&mut self) -> Result<(), AbortSearch> {
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes.is_multiple_of(CHECK_INTERVAL_NODES) && self.should_stop() {
            return Err(AbortSearch);
        }
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        if self.stop {
            return true;
        }

        if self.node_limit.is_some_and(|limit| self.nodes >= limit) {
            self.stop = true;
            return true;
        }

        if self
            .time_limit
            .is_some_and(|limit| self.started_at.elapsed() >= limit)
        {
            self.stop = true;
            return true;
        }

        false
    }
}

impl Default for AlphaBetaSearcher {
    fn default() -> Self {
        Self::new(AlphaBetaConfig::default())
    }
}

impl Searcher for AlphaBetaSearcher {
    fn search(&mut self, position: &Position, limits: SearchLimits) -> SearchResult {
        let result = self.search_with_info(position, limits);
        SearchResult {
            best_move: result.best_move,
            score: result.score,
            nodes_searched: result.nodes,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AbortSearch;

pub fn piece_value(piece: komugi_core::PieceType) -> i32 {
    use komugi_core::PieceType;

    match piece {
        PieceType::Marshal => 10_000,
        PieceType::General => 900,
        PieceType::LieutenantGeneral => 700,
        PieceType::MajorGeneral => 600,
        PieceType::Warrior => 450,
        PieceType::Lancer => 400,
        PieceType::Rider => 400,
        PieceType::Spy => 350,
        PieceType::Fortress => 300,
        PieceType::Soldier => 120,
        PieceType::Cannon => 300,
        PieceType::Archer => 300,
        PieceType::Musketeer => 280,
        PieceType::Tactician => 250,
    }
}

fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

#[cfg(test)]
mod tests {
    use komugi_core::SetupMode;

    use super::*;

    #[test]
    fn search_with_info_returns_pv_starting_with_best_move() {
        let position = Position::new(SetupMode::Beginner);
        let mut searcher = AlphaBetaSearcher::default();

        let result = searcher.search_with_info(
            &position,
            SearchLimits {
                depth: Some(2),
                ..SearchLimits::default()
            },
        );

        assert!(!result.pv.is_empty());
        assert_eq!(result.pv.first(), result.best_move.as_ref());
        assert!(result.pv.len() <= PV_WALK_MAX_DEPTH);
    }

    #[test]
    fn pv_extraction_stops_on_cycle_and_restores_position() {
        let mut searcher = AlphaBetaSearcher::default();
        let position = Position::new(SetupMode::Beginner);
        let (sequence, keys) =
            find_four_ply_cycle(&position).expect("expected a reversible sequence");

        for idx in 0..sequence.len() {
            searcher.tt.store(Entry {
                key: keys[idx],
                depth: 4,
                score: 0,
                bound: Bound::Exact,
                best_move: Some(sequence[idx].clone()),
            });
        }

        let mut walk_position = position.clone();
        let start_hash = walk_position.zobrist_hash;
        let start_history_len = walk_position.history.len();

        let pv = searcher.extract_pv_line(&mut walk_position);

        assert_eq!(pv, sequence);
        assert_eq!(walk_position.zobrist_hash, start_hash);
        assert_eq!(walk_position.history.len(), start_history_len);
    }

    fn find_four_ply_cycle(position: &Position) -> Option<(Vec<Move>, Vec<u64>)> {
        let root_hash = position.zobrist_hash;
        let mut current = position.clone();

        for white_1 in current.moves().into_iter().take(64) {
            current.make_move(&white_1).ok()?;
            let hash_1 = current.zobrist_hash;

            for black_1 in current.moves().into_iter().take(64) {
                current.make_move(&black_1).ok()?;
                let hash_2 = current.zobrist_hash;

                for white_2 in current.moves().into_iter().take(64) {
                    current.make_move(&white_2).ok()?;
                    let hash_3 = current.zobrist_hash;

                    for black_2 in current.moves().into_iter().take(64) {
                        current.make_move(&black_2).ok()?;

                        if current.zobrist_hash == root_hash {
                            let sequence = vec![
                                white_1.clone(),
                                black_1.clone(),
                                white_2.clone(),
                                black_2.clone(),
                            ];
                            let keys = vec![root_hash, hash_1, hash_2, hash_3];
                            let _ = current.unmake_move();
                            let _ = current.unmake_move();
                            let _ = current.unmake_move();
                            let _ = current.unmake_move();
                            return Some((sequence, keys));
                        }

                        let _ = current.unmake_move();
                    }

                    let _ = current.unmake_move();
                }

                let _ = current.unmake_move();
            }

            let _ = current.unmake_move();
        }

        None
    }

    #[test]
    fn test_exclude_best_move_returns_second_best() {
        let mut searcher = AlphaBetaSearcher::default();
        let pos = Position::new(SetupMode::Beginner);

        // Get the best move without exclusions
        let result1 = searcher.search_with_info(
            &pos,
            SearchLimits {
                depth: Some(2),
                ..SearchLimits::default()
            },
        );
        let best_move = result1.best_move.clone();

        assert!(best_move.is_some(), "Should find a best move");

        // Now exclude the best move and search again
        let mut searcher2 = AlphaBetaSearcher::default();
        let result2 = searcher2.search_with_exclusions(
            &pos,
            SearchLimits {
                depth: Some(2),
                ..SearchLimits::default()
            },
            &[best_move.clone().unwrap()],
        );
        let second_best = result2.best_move.clone();

        // The second best should be different from the best
        assert!(second_best.is_some(), "Should find a second-best move");
        assert_ne!(
            result1.best_move, result2.best_move,
            "Excluding best move should return different move"
        );
    }

    #[test]
    fn test_exclude_all_moves_returns_none() {
        let mut searcher = AlphaBetaSearcher::default();
        let pos = Position::new(SetupMode::Beginner);

        // Get all legal moves
        let all_moves = pos.moves();

        // Exclude all moves
        let result = searcher.search_with_exclusions(
            &pos,
            SearchLimits {
                depth: Some(1),
                ..SearchLimits::default()
            },
            &all_moves,
        );

        // Should return None when all moves are excluded
        assert!(
            result.best_move.is_none(),
            "Should return None when all moves are excluded"
        );
    }

    #[test]
    fn search_multi_pv_returns_unique_sorted_lines() {
        let mut searcher = AlphaBetaSearcher::default();
        let pos = Position::new(SetupMode::Beginner);

        let result = searcher.search_multi_pv(
            &pos,
            SearchLimits {
                depth: Some(2),
                ..SearchLimits::default()
            },
            3,
        );

        assert!(result.lines.len() <= 3);

        let mut seen_roots: Vec<Move> = Vec::new();
        for line in &result.lines {
            assert!(!line.moves.is_empty());
            assert!(!seen_roots.contains(&line.moves[0]));
            seen_roots.push(line.moves[0].clone());
        }

        for pair in result.lines.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }

        let summed_nodes: u64 = result.lines.iter().map(|line| line.nodes).sum();
        assert_eq!(result.total_nodes, summed_nodes);
    }

    #[test]
    fn search_multi_pv_zero_count_returns_empty() {
        let mut searcher = AlphaBetaSearcher::default();
        let pos = Position::new(SetupMode::Beginner);

        let result = searcher.search_multi_pv(
            &pos,
            SearchLimits {
                depth: Some(2),
                ..SearchLimits::default()
            },
            0,
        );

        assert!(result.lines.is_empty());
        assert_eq!(result.total_nodes, 0);
    }
}
