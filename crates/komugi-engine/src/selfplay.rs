use std::sync::Arc;

use komugi_core::{
    is_marshal_captured, move_to_san, Color, Policy, Position, SearchLimits, SetupMode,
};
use serde::{Deserialize, Serialize};

use crate::encoding::encode_position;
use crate::mcts::{HeuristicPolicy, MctsConfig, MctsSearcher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecord {
    pub fen: String,
    pub played_move: String,
    pub policy: Vec<(String, f32)>,
    pub outcome: f32,
    pub move_number: u32,
    pub encoding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    pub positions: Vec<TrainingRecord>,
    pub result: GameResult,
    pub total_moves: u32,
    pub moves: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

pub struct SelfPlayConfig {
    pub mcts_config: MctsConfig,
    pub setup_mode: SetupMode,
    pub max_moves: u32,
    pub policy: Arc<dyn Policy>,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            mcts_config: MctsConfig::default(),
            setup_mode: SetupMode::Beginner,
            max_moves: 300,
            policy: Arc::new(HeuristicPolicy),
        }
    }
}

pub fn play_game(config: &SelfPlayConfig) -> GameRecord {
    let mut position = Position::new(config.setup_mode);
    let mut searcher = MctsSearcher::new(config.mcts_config);
    let mut positions = Vec::new();
    let mut turns = Vec::new();
    let mut move_sans = Vec::new();
    let mut total_moves = 0u32;

    while total_moves < config.max_moves && !position.is_game_over() {
        let search_result =
            searcher.search_with_policy(&position, SearchLimits::default(), config.policy.as_ref());

        let policy = searcher
            .get_root_policy()
            .into_iter()
            .map(|(mv, proportion)| (move_to_san(&mv), proportion))
            .collect();

        let Some(best_move) = search_result.best_move else {
            positions.push(TrainingRecord {
                fen: position.fen(),
                played_move: String::new(),
                policy,
                outcome: 0.0,
                move_number: position.move_number,
                encoding: encode_position(&position),
            });
            turns.push(position.turn);
            break;
        };

        let san = move_to_san(&best_move);
        move_sans.push(san.clone());

        turns.push(position.turn);
        positions.push(TrainingRecord {
            fen: position.fen(),
            played_move: san,
            policy,
            outcome: 0.0,
            move_number: position.move_number,
            encoding: encode_position(&position),
        });

        if position.make_move(&best_move).is_err() {
            break;
        }

        total_moves = total_moves.saturating_add(1);
    }

    let reached_max_moves = total_moves >= config.max_moves && !position.is_game_over();
    let result = infer_result(&position, reached_max_moves);

    for (record, turn) in positions.iter_mut().zip(turns.into_iter()) {
        record.outcome = outcome_for_side(result, turn);
    }

    GameRecord {
        positions,
        result,
        total_moves,
        moves: move_sans,
    }
}

fn infer_result(position: &Position, reached_max_moves: bool) -> GameResult {
    if reached_max_moves || position.is_draw() {
        return GameResult::Draw;
    }

    if is_marshal_captured(position) || position.is_checkmate() {
        return match position.turn {
            Color::White => GameResult::BlackWin,
            Color::Black => GameResult::WhiteWin,
        };
    }

    GameResult::Draw
}

fn outcome_for_side(result: GameResult, side_to_move: Color) -> f32 {
    match result {
        GameResult::WhiteWin => {
            if side_to_move == Color::White {
                1.0
            } else {
                -1.0
            }
        }
        GameResult::BlackWin => {
            if side_to_move == Color::Black {
                1.0
            } else {
                -1.0
            }
        }
        GameResult::Draw => 0.0,
    }
}
