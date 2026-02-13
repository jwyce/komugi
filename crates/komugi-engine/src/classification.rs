/// Move classification for game review (Lichess-style)
/// Based on evaluation changes and winning chances
use std::f64::consts::E;

/// Lichess winning chances formula
/// Converts centipawn evaluation to winning probability (0.0 to 1.0)
pub fn winning_chances(cp: i32) -> f64 {
    let cp_f = cp as f64;
    2.0 / (1.0 + E.powf(-0.00368208 * cp_f)) - 1.0
}

/// Convert winning chances to win percentage (0-100)
pub fn win_percent(cp: i32) -> f64 {
    (winning_chances(cp) + 1.0) / 2.0 * 100.0
}

/// Classification of a move based on evaluation change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoveClassification {
    /// Blunder: ≥30% win chance loss
    Blunder,
    /// Mistake: ≥20% win chance loss
    Mistake,
    /// Inaccuracy: ≥10% win chance loss
    Inaccuracy,
    /// Good move: played well but not best
    Good,
    /// Excellent move: very strong play
    Excellent,
    /// Best move: matches engine evaluation
    Best,
}

impl MoveClassification {
    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Blunder => "Blunder",
            Self::Mistake => "Mistake",
            Self::Inaccuracy => "Inaccuracy",
            Self::Good => "Good",
            Self::Excellent => "Excellent",
            Self::Best => "Best",
        }
    }
}

/// Classify a move based on evaluation before/after and best evaluation
///
/// # Arguments
/// * `eval_before` - Evaluation before the move (centipawns)
/// * `eval_after` - Evaluation after the move (centipawns)
/// * `best_eval` - Best evaluation available (centipawns)
///
/// # Returns
/// Classification of the move
pub fn classify_move(eval_before: i32, eval_after: i32, best_eval: i32) -> MoveClassification {
    // Calculate win chance before and after
    let chance_before = winning_chances(eval_before);
    let chance_after = winning_chances(eval_after);

    // Win chance loss from this move
    let win_loss = chance_before - chance_after;

    // If move matches best evaluation, it's best
    if eval_after == best_eval {
        return MoveClassification::Best;
    }

    // If move is better than best known, it's excellent
    if eval_after > best_eval {
        return MoveClassification::Excellent;
    }

    // Classify based on win chance loss thresholds
    if win_loss >= 0.30 {
        MoveClassification::Blunder
    } else if win_loss >= 0.20 {
        MoveClassification::Mistake
    } else if win_loss >= 0.10 {
        MoveClassification::Inaccuracy
    } else {
        MoveClassification::Good
    }
}

/// Game analysis with move classifications
#[derive(Debug, Clone)]
pub struct GameAnalysis {
    /// Moves with their classifications
    pub moves: Vec<MoveAnalysis>,
}

/// Single move analysis
#[derive(Debug, Clone)]
pub struct MoveAnalysis {
    /// Move in algebraic notation (e.g., "e2e4")
    pub move_san: String,
    /// Evaluation before move (centipawns)
    pub eval_before: i32,
    /// Evaluation after move (centipawns)
    pub eval_after: i32,
    /// Best evaluation available
    pub best_eval: i32,
    /// Classification of the move
    pub classification: MoveClassification,
}

impl GameAnalysis {
    /// Create new game analysis
    pub fn new() -> Self {
        Self { moves: Vec::new() }
    }

    /// Add a move analysis
    pub fn add_move(
        &mut self,
        move_san: String,
        eval_before: i32,
        eval_after: i32,
        best_eval: i32,
    ) {
        let classification = classify_move(eval_before, eval_after, best_eval);
        self.moves.push(MoveAnalysis {
            move_san,
            eval_before,
            eval_after,
            best_eval,
            classification,
        });
    }

    /// Count moves by classification
    pub fn count_by_classification(&self) -> std::collections::HashMap<MoveClassification, usize> {
        let mut counts = std::collections::HashMap::new();
        for move_analysis in &self.moves {
            *counts.entry(move_analysis.classification).or_insert(0) += 1;
        }
        counts
    }

    /// Get accuracy percentage (1 - average win loss)
    pub fn accuracy(&self) -> f64 {
        if self.moves.is_empty() {
            return 100.0;
        }

        let total_loss: f64 = self
            .moves
            .iter()
            .map(|m| {
                let before = winning_chances(m.eval_before);
                let after = winning_chances(m.eval_after);
                (before - after).max(0.0)
            })
            .sum();

        let avg_loss = total_loss / self.moves.len() as f64;
        ((1.0 - avg_loss) * 100.0).max(0.0)
    }
}

impl Default for GameAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winning_chances_zero() {
        let wc = winning_chances(0);
        assert!((wc - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_winning_chances_positive() {
        let wc = winning_chances(100);
        assert!(wc > 0.0);
        assert!(wc < 1.0);
    }

    #[test]
    fn test_winning_chances_negative() {
        let wc = winning_chances(-100);
        assert!(wc < 0.0);
        assert!(wc > -1.0);
    }

    #[test]
    fn test_winning_chances_large_positive() {
        let wc = winning_chances(1000);
        assert!(wc > 0.95);
    }

    #[test]
    fn test_winning_chances_large_negative() {
        let wc = winning_chances(-1000);
        assert!(wc < -0.95);
    }

    #[test]
    fn test_win_percent_zero() {
        let wp = win_percent(0);
        assert!((wp - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_win_percent_positive() {
        let wp = win_percent(100);
        assert!(wp > 50.0);
        assert!(wp < 100.0);
    }

    #[test]
    fn test_win_percent_negative() {
        let wp = win_percent(-100);
        assert!(wp < 50.0);
        assert!(wp > 0.0);
    }

    #[test]
    fn test_classify_best_move() {
        let classification = classify_move(0, 50, 50);
        assert_eq!(classification, MoveClassification::Best);
    }

    #[test]
    fn test_classify_excellent_move() {
        let classification = classify_move(0, 100, 50);
        assert_eq!(classification, MoveClassification::Excellent);
    }

    #[test]
    fn test_classify_blunder() {
        // Lose 30%+ win chance
        let classification = classify_move(0, -300, 0);
        assert_eq!(classification, MoveClassification::Blunder);
    }

    #[test]
    fn test_classify_mistake() {
        // Lose 20-30% win chance
        let classification = classify_move(0, -150, 0);
        assert_eq!(classification, MoveClassification::Mistake);
    }

    #[test]
    fn test_classify_inaccuracy() {
        // Lose 10-20% win chance
        let classification = classify_move(0, -100, 0);
        assert_eq!(classification, MoveClassification::Inaccuracy);
    }

    #[test]
    fn test_classify_good_move() {
        // Lose <10% win chance
        let classification = classify_move(0, -50, 0);
        assert_eq!(classification, MoveClassification::Good);
    }

    #[test]
    fn test_game_analysis_add_move() {
        let mut analysis = GameAnalysis::new();
        analysis.add_move("e2e4".to_string(), 0, 50, 50);
        assert_eq!(analysis.moves.len(), 1);
        assert_eq!(analysis.moves[0].classification, MoveClassification::Best);
    }

    #[test]
    fn test_game_analysis_count_by_classification() {
        let mut analysis = GameAnalysis::new();
        analysis.add_move("e2e4".to_string(), 0, 50, 50); // Best
        analysis.add_move("e7e5".to_string(), 0, -300, 0); // Blunder
        analysis.add_move("g1f3".to_string(), 0, -100, 0); // Inaccuracy

        let counts = analysis.count_by_classification();
        assert_eq!(counts.get(&MoveClassification::Best), Some(&1));
        assert_eq!(counts.get(&MoveClassification::Blunder), Some(&1));
        assert_eq!(counts.get(&MoveClassification::Inaccuracy), Some(&1));
    }

    #[test]
    fn test_game_analysis_accuracy() {
        let mut analysis = GameAnalysis::new();
        analysis.add_move("e2e4".to_string(), 0, 50, 50); // Best move
        let accuracy = analysis.accuracy();
        assert!(accuracy > 99.0); // Should be very high for best move
    }

    #[test]
    fn test_move_classification_description() {
        assert_eq!(MoveClassification::Blunder.description(), "Blunder");
        assert_eq!(MoveClassification::Mistake.description(), "Mistake");
        assert_eq!(MoveClassification::Inaccuracy.description(), "Inaccuracy");
        assert_eq!(MoveClassification::Good.description(), "Good");
        assert_eq!(MoveClassification::Excellent.description(), "Excellent");
        assert_eq!(MoveClassification::Best.description(), "Best");
    }
}
