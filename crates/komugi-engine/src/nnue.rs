use komugi_core::{Color, Evaluator, Position, Score};

use crate::nnue_features::{extract_features, TOTAL_FEATURES};
use crate::nnue_format::{NnueError, NnueParams, QA, QB, SCALE};

const HIDDEN1_SIZE: usize = 256;
const HIDDEN2_SIZE: usize = 32;
const CONCAT_SIZE: usize = HIDDEN1_SIZE * 2;

pub struct NnueEval {
    params: NnueParams,
}

impl NnueEval {
    pub fn from_bytes(data: &[u8]) -> Result<Self, NnueError> {
        let params = NnueParams::from_bytes(data)?;
        validate_params(&params)?;
        Ok(Self { params })
    }

    pub fn evaluate_position(&self, position: &Position) -> i32 {
        let stm = position.turn;
        let opponent = opposite(stm);

        let stm_features = extract_features(position, stm);
        let opp_features = extract_features(position, opponent);

        let mut stm_acc = [0i32; HIDDEN1_SIZE];
        let mut opp_acc = [0i32; HIDDEN1_SIZE];
        self.accumulate_l0(&stm_features, &mut stm_acc);
        self.accumulate_l0(&opp_features, &mut opp_acc);

        let mut l1_input = [0i32; CONCAT_SIZE];
        let mut i = 0;
        while i < HIDDEN1_SIZE {
            l1_input[i] = screlu(stm_acc[i]);
            l1_input[i + HIDDEN1_SIZE] = screlu(opp_acc[i]);
            i += 1;
        }

        let mut l2_input = [0i32; HIDDEN2_SIZE];
        let mut out = 0;
        while out < HIDDEN2_SIZE {
            let mut acc = self.params.l1_bias[out];
            let mut inp = 0;
            while inp < CONCAT_SIZE {
                let w = i32::from(self.params.l1_weights[inp * HIDDEN2_SIZE + out]);
                acc += l1_input[inp] * w;
                inp += 1;
            }
            l2_input[out] = screlu(acc);
            out += 1;
        }

        let mut raw = self.params.l2_bias;
        let mut idx = 0;
        while idx < HIDDEN2_SIZE {
            raw += l2_input[idx] * i32::from(self.params.l2_weights[idx]);
            idx += 1;
        }

        scale_centipawns(raw)
    }

    fn accumulate_l0(&self, features: &[u16], out: &mut [i32; HIDDEN1_SIZE]) {
        let mut i = 0;
        while i < HIDDEN1_SIZE {
            out[i] = i32::from(self.params.ft_bias[i]);
            i += 1;
        }

        let total_features = self.params.total_features as usize;
        for feature in features {
            let feature_idx = usize::from(*feature);
            if feature_idx >= total_features {
                continue;
            }

            let row = feature_idx * HIDDEN1_SIZE;
            let mut hidden = 0;
            while hidden < HIDDEN1_SIZE {
                out[hidden] += i32::from(self.params.ft_weights[row + hidden]);
                hidden += 1;
            }
        }
    }
}

impl Evaluator for NnueEval {
    fn evaluate(&self, position: &Position) -> Score {
        let stm_score = self.evaluate_position(position);
        let white_score = if position.turn == Color::White {
            stm_score
        } else {
            -stm_score
        };
        Score(white_score)
    }
}

fn validate_params(params: &NnueParams) -> Result<(), NnueError> {
    let total_features = params.total_features as usize;
    if total_features != TOTAL_FEATURES {
        return Err(NnueError::InvalidHeader(format!(
            "total_features must be {}, got {}",
            TOTAL_FEATURES, total_features
        )));
    }

    if params.hidden1_size as usize != HIDDEN1_SIZE {
        return Err(NnueError::InvalidHeader(format!(
            "hidden1_size must be {}, got {}",
            HIDDEN1_SIZE, params.hidden1_size
        )));
    }

    if params.hidden2_size as usize != HIDDEN2_SIZE {
        return Err(NnueError::InvalidHeader(format!(
            "hidden2_size must be {}, got {}",
            HIDDEN2_SIZE, params.hidden2_size
        )));
    }

    if params.ft_bias.len() != HIDDEN1_SIZE {
        return Err(NnueError::InvalidHeader(format!(
            "ft_bias len must be {}, got {}",
            HIDDEN1_SIZE,
            params.ft_bias.len()
        )));
    }

    if params.ft_weights.len() != total_features * HIDDEN1_SIZE {
        return Err(NnueError::InvalidHeader(format!(
            "ft_weights len must be {}, got {}",
            total_features * HIDDEN1_SIZE,
            params.ft_weights.len()
        )));
    }

    if params.l1_bias.len() != HIDDEN2_SIZE {
        return Err(NnueError::InvalidHeader(format!(
            "l1_bias len must be {}, got {}",
            HIDDEN2_SIZE,
            params.l1_bias.len()
        )));
    }

    if params.l1_weights.len() != CONCAT_SIZE * HIDDEN2_SIZE {
        return Err(NnueError::InvalidHeader(format!(
            "l1_weights len must be {}, got {}",
            CONCAT_SIZE * HIDDEN2_SIZE,
            params.l1_weights.len()
        )));
    }

    if params.l2_weights.len() != HIDDEN2_SIZE {
        return Err(NnueError::InvalidHeader(format!(
            "l2_weights len must be {}, got {}",
            HIDDEN2_SIZE,
            params.l2_weights.len()
        )));
    }

    Ok(())
}

#[inline]
fn screlu(x: i32) -> i32 {
    let clamped = x.clamp(0, QA);
    clamped * clamped
}

#[inline]
fn scale_centipawns(raw_output: i32) -> i32 {
    let numerator = i64::from(raw_output) * i64::from(SCALE);
    let denominator = i64::from(QA) * i64::from(QB);
    let scaled = numerator / denominator;
    scaled.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[inline]
fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use komugi_core::SetupMode;

    /// E2E test: loads a real .nnue file from NNUE_TEST_PATH and evaluates positions.
    /// Skipped when NNUE_TEST_PATH is not set (normal cargo test).
    #[test]
    fn test_e2e_load_and_eval() {
        let path = match std::env::var("NNUE_TEST_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("NNUE_TEST_PATH not set, skipping E2E test");
                return;
            }
        };

        let data = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("failed to read NNUE file at {path}: {e}");
        });

        let eval = NnueEval::from_bytes(&data).unwrap_or_else(|e| {
            panic!("failed to parse NNUE file: {e:?}");
        });

        for mode in [
            SetupMode::Beginner,
            SetupMode::Intermediate,
            SetupMode::Advanced,
        ] {
            let position = Position::new(mode);
            let score = eval.evaluate(&position);
            assert!(
                score.0.abs() < 30_000,
                "score out of range for {mode:?}: {}",
                score.0
            );
            eprintln!("{mode:?} starting position: {}", score.0);
        }
    }

    #[test]
    fn nnue_evaluator_respects_side_to_move_perspective() {
        let mut params = NnueParams {
            total_features: TOTAL_FEATURES as u32,
            hidden1_size: HIDDEN1_SIZE as u32,
            hidden2_size: HIDDEN2_SIZE as u32,
            ft_bias: vec![0; HIDDEN1_SIZE],
            ft_weights: vec![0; TOTAL_FEATURES * HIDDEN1_SIZE],
            l1_bias: vec![QA; HIDDEN2_SIZE],
            l1_weights: vec![0; CONCAT_SIZE * HIDDEN2_SIZE],
            l2_bias: 0,
            l2_weights: vec![1; HIDDEN2_SIZE],
        };

        let eval = NnueEval {
            params: params.clone(),
        };
        let mut position = Position::new(SetupMode::Advanced);

        position.turn = Color::White;
        let stm_white = eval.evaluate_position(&position);
        let score_white = eval.evaluate(&position).0;
        assert_eq!(score_white, stm_white);

        position.turn = Color::Black;
        let stm_black = eval.evaluate_position(&position);
        let score_black = eval.evaluate(&position).0;
        assert_eq!(score_black, -stm_black);

        params.total_features = (TOTAL_FEATURES - 1) as u32;
        let bytes = params.to_bytes();
        let from_bytes = NnueEval::from_bytes(&bytes);
        assert!(matches!(from_bytes, Err(NnueError::InvalidHeader(_))));
    }
}
