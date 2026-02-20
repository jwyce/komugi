use crate::position::Position;
use crate::types::{Move, Score};

pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> Score;
}

/// Prior probability distribution over legal moves.
/// Used by MCTS to guide tree exploration via PUCT.
pub trait Policy: Send + Sync {
    /// Returns a prior probability for each move in the given list.
    /// Output must be the same length as `moves` and sum to ~1.0.
    fn prior(&self, position: &Position, moves: &[Move]) -> Vec<f32>;

    /// Returns prior probabilities and optionally a value estimate from the
    /// same inference call. Neural policies return `Some(value)` (in
    /// side-to-move perspective, range `[-1, 1]`) to avoid a separate
    /// classical evaluation; heuristic policies return `None`.
    fn prior_and_value(&self, position: &Position, moves: &[Move]) -> (Vec<f32>, Option<f32>) {
        (self.prior(position, moves), None)
    }

    /// Evaluate multiple positions in a single batch. GPU policies override
    /// this to send all requests before collecting responses, enabling
    /// virtual-loss batched MCTS to amortize GPU round-trip latency.
    fn prior_and_value_batch(
        &self,
        batch: &[(&Position, &[Move])],
    ) -> Vec<(Vec<f32>, Option<f32>)> {
        batch
            .iter()
            .map(|(pos, moves)| self.prior_and_value(pos, moves))
            .collect()
    }
}

/// Uniform policy: assigns equal probability to all legal moves.
/// Baseline before neural network training.
pub struct UniformPolicy;

impl Policy for UniformPolicy {
    fn prior(&self, _position: &Position, moves: &[Move]) -> Vec<f32> {
        if moves.is_empty() {
            return Vec::new();
        }
        let p = 1.0 / moves.len() as f32;
        vec![p; moves.len()]
    }
}
