pub mod alphabeta;
pub mod classical;
pub mod classification;
pub mod encoding;
pub mod mcts;
#[cfg(feature = "neural")]
pub mod neural;
pub mod selfplay;
pub mod tt;

pub use alphabeta::{AlphaBetaConfig, AlphaBetaResult, AlphaBetaSearcher};
pub use classical::ClassicalEval;
pub use classification::{
    classify_move, win_percent, winning_chances, GameAnalysis, MoveAnalysis, MoveClassification,
};
pub use encoding::{
    encode_position, move_to_policy_index, square_index, BOARD_SIZE, DROP_MOVE_OFFSET,
    ENCODING_SIZE, NUM_PLANES, NUM_SQUARES, POLICY_SIZE,
};
pub use mcts::{MctsConfig, MctsSearcher};
#[cfg(feature = "neural")]
pub use neural::NeuralPolicy;
pub use selfplay::{play_game, GameRecord, GameResult, SelfPlayConfig, TrainingRecord};
pub use tt::{Bound, Entry as TTEntry, TranspositionTable};
