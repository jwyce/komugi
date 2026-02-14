use std::cell::UnsafeCell;
use std::path::Path;

use komugi_core::{Color, Evaluator, Move, Policy, Position, Score};
use ort::session::Session;
use ort::value::Tensor;

use crate::encoding::{
    encode_position, move_to_policy_index, BOARD_SIZE, ENCODING_SIZE, NUM_PLANES, POLICY_SIZE,
};

pub struct NeuralPolicy {
    session: UnsafeCell<Session>,
}

// Safety: each NeuralPolicy is created per-thread in selfplay. A single
// thread owns exclusive access to its session — no concurrent &mut aliases.
unsafe impl Send for NeuralPolicy {}
unsafe impl Sync for NeuralPolicy {}

impl NeuralPolicy {
    pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)?;
        Ok(Self {
            session: UnsafeCell::new(session),
        })
    }

    fn run_inference(&self, position: &Position) -> (Vec<f32>, f32) {
        let encoding = encode_position(position);
        debug_assert_eq!(encoding.len(), ENCODING_SIZE);

        let input =
            Tensor::<f32>::from_array((vec![1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE], encoding))
                .expect("encoding size must match tensor shape");

        // Safety: only one thread ever calls run_inference on this instance
        let session = unsafe { &mut *self.session.get() };
        let outputs = session
            .run(ort::inputs![input])
            .expect("ONNX inference failed");

        let (_, policy_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("policy output must be f32");
        let policy_logits = policy_slice.to_vec();
        debug_assert_eq!(policy_logits.len(), POLICY_SIZE);

        let (_, value_slice) = outputs[1]
            .try_extract_tensor::<f32>()
            .expect("value output must be f32");
        let value = value_slice[0];

        (policy_logits, value)
    }
}

impl Policy for NeuralPolicy {
    fn prior(&self, position: &Position, moves: &[Move]) -> Vec<f32> {
        if moves.is_empty() {
            return Vec::new();
        }

        let (logits, _value) = self.run_inference(position);

        let mut move_logits = Vec::with_capacity(moves.len());
        for mv in moves {
            let idx = move_to_policy_index(mv);
            let logit = if idx < logits.len() { logits[idx] } else { 0.0 };
            move_logits.push(logit);
        }

        let max_logit = move_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = Vec::with_capacity(moves.len());
        for logit in &move_logits {
            let exp = (logit - max_logit).exp();
            probs.push(exp);
            sum += exp;
        }

        if sum > f32::EPSILON {
            for prob in &mut probs {
                *prob /= sum;
            }
        } else {
            let uniform = 1.0 / moves.len() as f32;
            probs.fill(uniform);
        }

        probs
    }
}

impl Evaluator for NeuralPolicy {
    fn evaluate(&self, position: &Position) -> Score {
        let (_logits, value) = self.run_inference(position);
        let clamped = value.clamp(-0.999_999, 0.999_999) as f64;
        let cp = clamped.atanh() * 600.0;
        let white_cp = match position.turn {
            Color::White => cp,
            Color::Black => -cp,
        };
        Score(white_cp.round() as i32)
    }
}
