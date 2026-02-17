/// Chess.com-style move classification for game review
/// Based on win probability changes and contextual factors
use crate::alphabeta::{piece_value, AlphaBetaConfig, AlphaBetaSearcher, MultiPvResult};
use crate::win_probability::{accuracy_for_cpl, is_garbage_time, win_percent_loss};
use komugi_core::{
    is_square_attacked, move_to_san, parse_fen, parse_san, Color, Move, MoveType, Position,
    SearchLimits,
};

/// Classification of a move using Chess.com-style categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoveClassification {
    /// Brilliant: sacrifice that's hard to find but strong
    Brilliant,
    /// Great: best move after opponent's mistake or following own brilliant
    Great,
    /// Best: engine's top choice (< 1% win loss)
    Best,
    /// Excellent: very close to best (< 2% win loss)
    Excellent,
    /// Good: solid play (< 10% win loss)
    Good,
    /// Miss: missed a big opportunity (10%+ loss, best was 200cp+ better than position)
    Miss,
    /// Inaccuracy: 10-20% win chance loss
    Inaccuracy,
    /// Mistake: 20-30% win chance loss
    Mistake,
    /// Blunder: >= 30% win chance loss
    Blunder,
}

impl MoveClassification {
    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Brilliant => "Brilliant",
            Self::Great => "Great",
            Self::Best => "Best",
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Miss => "Miss",
            Self::Inaccuracy => "Inaccuracy",
            Self::Mistake => "Mistake",
            Self::Blunder => "Blunder",
        }
    }
}

/// All inputs needed to classify a single move
#[derive(Debug, Clone)]
pub struct ClassificationContext {
    /// Evaluation before the move (centipawns, from side-to-move perspective)
    pub eval_before: i32,
    /// Evaluation after the move (centipawns, from side-to-move perspective)
    pub eval_after: i32,
    /// Evaluation of the best available move
    pub best_eval: i32,
    /// Evaluation of the second-best available move
    pub second_best_eval: i32,
    /// Whether this move was the engine's top choice
    pub is_best_move: bool,
    /// Classification of the player's previous move
    pub prev_classification: Option<MoveClassification>,
    /// Classification of the opponent's previous move
    pub opponent_prev_classification: Option<MoveClassification>,
    /// Whether this move is a sacrifice (material loss for positional gain)
    pub is_sacrifice: bool,
}

/// Classify a move using the Chess.com-style algorithm
///
/// Classification order: Brilliant → Great → Best → Excellent → Good → Miss → Inaccuracy → Mistake → Blunder
///
/// Garbage time override: if eval_before indicates overwhelming advantage (>= 700cp),
/// classification is clamped to Best/Excellent/Good only.
pub fn classify_move(ctx: &ClassificationContext) -> MoveClassification {
    let wpl = win_percent_loss(ctx.eval_before, ctx.eval_after);
    let eval_loss_cp = ctx.best_eval - ctx.eval_after;

    // Garbage time override: no harsh classifications when game is already decided
    if is_garbage_time(ctx.eval_before) {
        if wpl < 1.0 {
            return MoveClassification::Best;
        }
        if wpl < 2.0 {
            return MoveClassification::Excellent;
        }
        return MoveClassification::Good;
    }

    // 1. Brilliant: sacrifice that isn't obvious but strong
    let is_best_move_obvious = (ctx.best_eval - ctx.second_best_eval) > 150;
    if ctx.is_sacrifice && eval_loss_cp <= 50 && !is_best_move_obvious {
        return MoveClassification::Brilliant;
    }

    // 2. Great: best move after opponent blunder or following own brilliant
    if ctx.is_best_move {
        let after_brilliant = ctx.prev_classification == Some(MoveClassification::Brilliant);
        let punishing_opponent = matches!(
            ctx.opponent_prev_classification,
            Some(MoveClassification::Mistake)
                | Some(MoveClassification::Blunder)
                | Some(MoveClassification::Inaccuracy)
        );
        if after_brilliant || punishing_opponent {
            return MoveClassification::Great;
        }
    }

    // 3. Best: < 1% win probability loss
    if wpl < 1.0 {
        return MoveClassification::Best;
    }

    // 4. Excellent: < 2% win probability loss
    if wpl < 2.0 {
        return MoveClassification::Excellent;
    }

    // 5. Good: < 10% win probability loss
    if wpl < 10.0 {
        return MoveClassification::Good;
    }

    // 6. Miss: lost 10%+ but there was a big opportunity (200cp+ better move existed)
    if (ctx.best_eval - ctx.eval_before) >= 200 {
        return MoveClassification::Miss;
    }

    // 7. Inaccuracy: 10-20% win probability loss
    if wpl < 20.0 {
        return MoveClassification::Inaccuracy;
    }

    // 8. Mistake: 20-30% win probability loss
    if wpl < 30.0 {
        return MoveClassification::Mistake;
    }

    // 9. Blunder: >= 30% win probability loss
    MoveClassification::Blunder
}

pub fn classify_move_in_game(
    position: &Position,
    eval_before: i32,
    played_move: &Move,
    multi_pv_result: &MultiPvResult,
    prev_classification: Option<MoveClassification>,
    opponent_prev_classification: Option<MoveClassification>,
) -> Option<MoveClassification> {
    let best_line = multi_pv_result.lines.first()?;
    let best_eval = best_line.score.0;
    let second_best_eval = multi_pv_result
        .lines
        .get(1)
        .map(|line| line.score.0)
        .unwrap_or(best_eval);
    let played_line_score = multi_pv_result
        .lines
        .iter()
        .find(|line| line.moves.first() == Some(played_move))
        .map(|line| line.score.0)
        .unwrap_or(best_eval);
    let is_best_move = best_line.moves.first() == Some(played_move);
    let hard_to_find = (best_eval - second_best_eval) < 50;

    let ctx = ClassificationContext {
        eval_before,
        eval_after: -played_line_score,
        best_eval,
        second_best_eval,
        is_best_move,
        prev_classification,
        opponent_prev_classification,
        is_sacrifice: is_sacrifice_move(position, played_move) && hard_to_find,
    };

    Some(classify_move(&ctx))
}

pub fn analyze_full_game(
    move_sequence: &[(String, String)],
    depth: u8,
    num_pv: usize,
) -> Result<GameAnalysis, String> {
    let mut analysis = GameAnalysis::new();
    let mut white_prev = None;
    let mut black_prev = None;
    let mut searcher = AlphaBetaSearcher::new(AlphaBetaConfig::default());
    let pv_count = num_pv.max(1);

    for (fen, move_san) in move_sequence {
        let position =
            Position::from_fen(fen).map_err(|err| format!("invalid FEN '{}': {}", fen, err))?;
        let parsed_fen = parse_fen(fen).map_err(|err| format!("invalid FEN '{}': {}", fen, err))?;
        let played_move = parse_san(move_san, &parsed_fen)
            .map_err(|err| format!("illegal SAN '{}' for '{}': {}", move_san, fen, err))?;
        let multi_pv = searcher.search_multi_pv(
            &position,
            SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            },
            pv_count,
        );
        let eval_before = searcher
            .search_with_info(
                &position,
                SearchLimits {
                    depth: Some(depth),
                    ..SearchLimits::default()
                },
            )
            .score
            .0;

        let (prev_classification, opponent_prev_classification) = match position.turn {
            Color::White => (white_prev, black_prev),
            Color::Black => (black_prev, white_prev),
        };

        let classification = classify_move_in_game(
            &position,
            eval_before,
            &played_move,
            &multi_pv,
            prev_classification,
            opponent_prev_classification,
        )
        .ok_or_else(|| format!("search returned no PV lines for '{}'", fen))?;

        let best_eval = multi_pv.lines[0].score.0;
        let played_line_score = multi_pv
            .lines
            .iter()
            .find(|line| line.moves.first() == Some(&played_move))
            .map(|line| line.score.0)
            .unwrap_or(best_eval);

        analysis.add_move(
            move_san.clone(),
            eval_before,
            -played_line_score,
            multi_pv.lines[0]
                .moves
                .first()
                .map(move_to_san)
                .unwrap_or_else(|| move_san.clone()),
            best_eval,
            classification,
        );

        match position.turn {
            Color::White => white_prev = Some(classification),
            Color::Black => black_prev = Some(classification),
        }
    }

    Ok(analysis)
}

pub fn is_sacrifice_move(position: &Position, played_move: &Move) -> bool {
    if !matches!(played_move.move_type, MoveType::Capture | MoveType::Arata) {
        return false;
    }

    let mut after = position.clone();
    if after.make_move(played_move).is_err() {
        return false;
    }

    if !is_square_attacked(
        &after.board,
        played_move.to.square,
        opposite(played_move.color),
    ) {
        return false;
    }

    match played_move.move_type {
        MoveType::Capture => {
            let captured_value: i32 = played_move
                .captured
                .iter()
                .map(|piece| piece_value(piece.piece_type))
                .sum();
            piece_value(played_move.piece) > captured_value
        }
        MoveType::Arata => piece_value(played_move.piece) >= 300,
        _ => false,
    }
}

fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

/// Single move analysis
#[derive(Debug, Clone)]
pub struct MoveAnalysis {
    /// Move in algebraic notation
    pub move_san: String,
    /// Evaluation before move (centipawns)
    pub eval_before: i32,
    /// Evaluation after move (centipawns)
    pub eval_after: i32,
    pub best_san: String,
    /// Best evaluation available
    pub best_eval: i32,
    /// Classification of the move
    pub classification: MoveClassification,
}

/// Game analysis with move classifications and per-player accuracy
#[derive(Debug, Clone)]
pub struct GameAnalysis {
    /// Moves with their classifications
    pub moves: Vec<MoveAnalysis>,
    /// White player accuracy (0-100)
    pub white_accuracy: f64,
    /// Black player accuracy (0-100)
    pub black_accuracy: f64,
}

impl GameAnalysis {
    /// Create new game analysis
    pub fn new() -> Self {
        Self {
            moves: Vec::new(),
            white_accuracy: 100.0,
            black_accuracy: 100.0,
        }
    }

    /// Add a move analysis and recompute per-player accuracy
    pub fn add_move(
        &mut self,
        move_san: String,
        eval_before: i32,
        eval_after: i32,
        best_san: String,
        best_eval: i32,
        classification: MoveClassification,
    ) {
        self.moves.push(MoveAnalysis {
            move_san,
            eval_before,
            eval_after,
            best_san,
            best_eval,
            classification,
        });
        self.recompute_accuracy();
    }

    /// Count moves by classification
    pub fn count_by_classification(&self) -> std::collections::HashMap<MoveClassification, usize> {
        let mut counts = std::collections::HashMap::new();
        for m in &self.moves {
            *counts.entry(m.classification).or_insert(0) += 1;
        }
        counts
    }

    /// Recompute white and black accuracy from all moves
    fn recompute_accuracy(&mut self) {
        let mut white_sum = 0.0;
        let mut white_count = 0;
        let mut black_sum = 0.0;
        let mut black_count = 0;

        for (i, m) in self.moves.iter().enumerate() {
            let cpl = (m.best_eval - m.eval_after).max(0) as f64;
            let acc = accuracy_for_cpl(cpl);
            if i % 2 == 0 {
                // White moves (0, 2, 4, ...)
                white_sum += acc;
                white_count += 1;
            } else {
                // Black moves (1, 3, 5, ...)
                black_sum += acc;
                black_count += 1;
            }
        }

        self.white_accuracy = if white_count > 0 {
            white_sum / white_count as f64
        } else {
            100.0
        };
        self.black_accuracy = if black_count > 0 {
            black_sum / black_count as f64
        } else {
            100.0
        };
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
    use crate::alphabeta::PvLine;
    use crate::win_probability::win_percent_loss;
    use komugi_core::{move_to_san, Score, SetupMode};

    /// Helper: build a basic context with sensible defaults
    fn ctx(eval_before: i32, eval_after: i32, best_eval: i32) -> ClassificationContext {
        ClassificationContext {
            eval_before,
            eval_after,
            best_eval,
            second_best_eval: best_eval - 10,
            is_best_move: eval_after == best_eval,
            prev_classification: None,
            opponent_prev_classification: None,
            is_sacrifice: false,
        }
    }

    // ---- Threshold tests ----

    #[test]
    fn test_best_move_tiny_loss() {
        // win_percent_loss < 1.0 → Best
        let c = ctx(0, -5, 0);
        assert_eq!(classify_move(&c), MoveClassification::Best);
    }

    #[test]
    fn test_excellent_move() {
        // win_percent_loss between 1.0 and 2.0 → Excellent
        let c = ctx(0, -15, 0);
        let wpl = win_percent_loss(0, -15);
        assert!(wpl >= 1.0 && wpl < 2.0, "wpl={} should be in [1,2)", wpl);
        assert_eq!(classify_move(&c), MoveClassification::Excellent);
    }

    #[test]
    fn test_good_move() {
        // win_percent_loss between 2.0 and 10.0 → Good
        let c = ctx(0, -50, 0);
        let wpl = win_percent_loss(0, -50);
        assert!(wpl >= 2.0 && wpl < 10.0, "wpl={} should be in [2,10)", wpl);
        assert_eq!(classify_move(&c), MoveClassification::Good);
    }

    #[test]
    fn test_inaccuracy() {
        // win_percent_loss between 10.0 and 20.0 → Inaccuracy
        let c = ctx(0, -150, 0);
        let wpl = win_percent_loss(0, -150);
        assert!(
            wpl >= 10.0 && wpl < 20.0,
            "wpl={} should be in [10,20)",
            wpl
        );
        assert_eq!(classify_move(&c), MoveClassification::Inaccuracy);
    }

    #[test]
    fn test_mistake() {
        // win_percent_loss between 20.0 and 30.0 → Mistake
        let c = ctx(0, -300, 0);
        let wpl = win_percent_loss(0, -300);
        assert!(
            wpl >= 20.0 && wpl < 30.0,
            "wpl={} should be in [20,30)",
            wpl
        );
        assert_eq!(classify_move(&c), MoveClassification::Mistake);
    }

    #[test]
    fn test_blunder() {
        // win_percent_loss >= 30.0 → Blunder
        let c = ctx(0, -500, 0);
        let wpl = win_percent_loss(0, -500);
        assert!(wpl >= 30.0, "wpl={} should be >= 30", wpl);
        assert_eq!(classify_move(&c), MoveClassification::Blunder);
    }

    #[test]
    fn test_miss_big_opportunity() {
        // win_percent_loss >= 10.0 AND best_eval - eval_before >= 200 → Miss
        let c = ClassificationContext {
            eval_before: -100,
            eval_after: -250,
            best_eval: 150, // 150 - (-100) = 250 >= 200 → miss
            second_best_eval: 140,
            is_best_move: false,
            prev_classification: None,
            opponent_prev_classification: None,
            is_sacrifice: false,
        };
        let wpl = win_percent_loss(-100, -250);
        assert!(wpl >= 10.0, "wpl={} should be >= 10", wpl);
        assert_eq!(classify_move(&c), MoveClassification::Miss);
    }

    // ---- Brilliant tests ----

    #[test]
    fn test_brilliant_sacrifice() {
        // is_sacrifice=true, eval_loss_cp <= 50, best move not obvious
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: -10,
            best_eval: 20,         // eval_loss_cp = 20 - (-10) = 30 <= 50
            second_best_eval: -10, // gap = 20 - (-10) = 30 <= 150, not obvious
            is_best_move: false,
            prev_classification: None,
            opponent_prev_classification: None,
            is_sacrifice: true,
        };
        assert_eq!(classify_move(&c), MoveClassification::Brilliant);
    }

    #[test]
    fn test_not_brilliant_when_obvious() {
        // is_sacrifice=true but best move is obvious (gap > 150)
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: -10,
            best_eval: 200,      // eval_loss_cp = 210, too high anyway
            second_best_eval: 0, // gap = 200 > 150, obvious
            is_best_move: false,
            prev_classification: None,
            opponent_prev_classification: None,
            is_sacrifice: true,
        };
        assert_ne!(classify_move(&c), MoveClassification::Brilliant);
    }

    #[test]
    fn test_not_brilliant_high_eval_loss() {
        // is_sacrifice=true but eval_loss_cp > 50
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: -100,
            best_eval: 0, // eval_loss_cp = 0 - (-100) = 100 > 50
            second_best_eval: -50,
            is_best_move: false,
            prev_classification: None,
            opponent_prev_classification: None,
            is_sacrifice: true,
        };
        assert_ne!(classify_move(&c), MoveClassification::Brilliant);
    }

    // ---- Great tests ----

    #[test]
    fn test_great_after_brilliant() {
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: 50,
            best_eval: 50,
            second_best_eval: 40,
            is_best_move: true,
            prev_classification: Some(MoveClassification::Brilliant),
            opponent_prev_classification: None,
            is_sacrifice: false,
        };
        assert_eq!(classify_move(&c), MoveClassification::Great);
    }

    #[test]
    fn test_great_punishing_opponent_blunder() {
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: 50,
            best_eval: 50,
            second_best_eval: 40,
            is_best_move: true,
            prev_classification: None,
            opponent_prev_classification: Some(MoveClassification::Blunder),
            is_sacrifice: false,
        };
        assert_eq!(classify_move(&c), MoveClassification::Great);
    }

    #[test]
    fn test_great_punishing_opponent_mistake() {
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: 50,
            best_eval: 50,
            second_best_eval: 40,
            is_best_move: true,
            prev_classification: None,
            opponent_prev_classification: Some(MoveClassification::Mistake),
            is_sacrifice: false,
        };
        assert_eq!(classify_move(&c), MoveClassification::Great);
    }

    #[test]
    fn test_great_punishing_opponent_inaccuracy() {
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: 50,
            best_eval: 50,
            second_best_eval: 40,
            is_best_move: true,
            prev_classification: None,
            opponent_prev_classification: Some(MoveClassification::Inaccuracy),
            is_sacrifice: false,
        };
        assert_eq!(classify_move(&c), MoveClassification::Great);
    }

    #[test]
    fn test_not_great_when_not_best_move() {
        // Even with opponent blunder, not Great if not best move
        let c = ClassificationContext {
            eval_before: 0,
            eval_after: 40,
            best_eval: 50,
            second_best_eval: 40,
            is_best_move: false,
            prev_classification: None,
            opponent_prev_classification: Some(MoveClassification::Blunder),
            is_sacrifice: false,
        };
        assert_ne!(classify_move(&c), MoveClassification::Great);
    }

    // ---- Garbage time tests ----

    #[test]
    fn test_garbage_time_no_blunder() {
        // eval_before >= 700 → garbage time, blunder-level loss clamped to Good
        let c = ctx(800, 500, 800);
        assert!(
            matches!(
                classify_move(&c),
                MoveClassification::Best | MoveClassification::Excellent | MoveClassification::Good
            ),
            "garbage time should clamp to Best/Excellent/Good"
        );
    }

    #[test]
    fn test_garbage_time_negative() {
        // eval_before <= -700 → garbage time (losing badly)
        let c = ctx(-800, -900, -800);
        assert!(
            matches!(
                classify_move(&c),
                MoveClassification::Best | MoveClassification::Excellent | MoveClassification::Good
            ),
            "garbage time should clamp to Best/Excellent/Good"
        );
    }

    #[test]
    fn test_not_garbage_time_under_threshold() {
        // eval_before = 699 → NOT garbage time, blunder possible
        let c = ctx(699, 0, 699);
        let wpl = win_percent_loss(699, 0);
        assert!(wpl >= 30.0, "wpl={} should be >= 30", wpl);
        assert_eq!(classify_move(&c), MoveClassification::Blunder);
    }

    // ---- GameAnalysis tests ----

    #[test]
    fn test_game_analysis_add_move() {
        let mut analysis = GameAnalysis::new();
        analysis.add_move(
            "e2e4".to_string(),
            0,
            50,
            "e2e4".to_string(),
            50,
            MoveClassification::Best,
        );
        assert_eq!(analysis.moves.len(), 1);
        assert_eq!(analysis.moves[0].classification, MoveClassification::Best);
        assert_eq!(analysis.moves[0].best_san, "e2e4");
    }

    #[test]
    fn test_game_analysis_per_player_accuracy() {
        let mut analysis = GameAnalysis::new();
        // White: perfect move (cpl=0 → accuracy=100%)
        analysis.add_move(
            "e2e4".to_string(),
            0,
            50,
            "e2e4".to_string(),
            50,
            MoveClassification::Best,
        );
        // Black: moderate loss (cpl=50 → accuracy < 100%)
        analysis.add_move(
            "e7e5".to_string(),
            0,
            -50,
            "e7e5".to_string(),
            0,
            MoveClassification::Good,
        );
        // White: perfect again
        analysis.add_move(
            "g1f3".to_string(),
            50,
            80,
            "g1f3".to_string(),
            80,
            MoveClassification::Best,
        );

        assert!(
            analysis.white_accuracy > 99.0,
            "white accuracy should be ~100, got {}",
            analysis.white_accuracy
        );
        assert!(
            analysis.black_accuracy < analysis.white_accuracy,
            "black accuracy ({}) should be lower than white ({})",
            analysis.black_accuracy,
            analysis.white_accuracy
        );
    }

    #[test]
    fn test_game_analysis_empty_accuracy() {
        let analysis = GameAnalysis::new();
        assert!((analysis.white_accuracy - 100.0).abs() < 0.001);
        assert!((analysis.black_accuracy - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_game_analysis_count_by_classification() {
        let mut analysis = GameAnalysis::new();
        analysis.add_move(
            "e2e4".to_string(),
            0,
            50,
            "e2e4".to_string(),
            50,
            MoveClassification::Best,
        );
        analysis.add_move(
            "e7e5".to_string(),
            0,
            -500,
            "e7e5".to_string(),
            0,
            MoveClassification::Blunder,
        );
        analysis.add_move(
            "g1f3".to_string(),
            0,
            -150,
            "g1f3".to_string(),
            0,
            MoveClassification::Inaccuracy,
        );

        let counts = analysis.count_by_classification();
        assert_eq!(counts.get(&MoveClassification::Best), Some(&1));
        assert_eq!(counts.get(&MoveClassification::Blunder), Some(&1));
        assert_eq!(counts.get(&MoveClassification::Inaccuracy), Some(&1));
    }

    // ---- Description tests ----

    #[test]
    fn test_all_descriptions() {
        assert_eq!(MoveClassification::Brilliant.description(), "Brilliant");
        assert_eq!(MoveClassification::Great.description(), "Great");
        assert_eq!(MoveClassification::Best.description(), "Best");
        assert_eq!(MoveClassification::Excellent.description(), "Excellent");
        assert_eq!(MoveClassification::Good.description(), "Good");
        assert_eq!(MoveClassification::Miss.description(), "Miss");
        assert_eq!(MoveClassification::Inaccuracy.description(), "Inaccuracy");
        assert_eq!(MoveClassification::Mistake.description(), "Mistake");
        assert_eq!(MoveClassification::Blunder.description(), "Blunder");
    }

    fn mpv_line(mv: Move, score: i32) -> PvLine {
        PvLine {
            moves: vec![mv],
            score: Score(score),
            depth: 1,
            nodes: 1,
        }
    }

    #[test]
    fn test_capture_sacrifice_detection() {
        let fen = "8M/9/9/9/3dG4/3m5/9/9/9 -/- w 1 - 1";
        let position = Position::from_fen(fen).expect("valid position");
        let parsed = parse_fen(fen).expect("valid parsed fen");
        let mv = parse_san("大(5-5-1)取(5-6-1)", &parsed).expect("legal capture move");

        assert!(is_sacrifice_move(&position, &mv));
    }

    #[test]
    fn test_drop_sacrifice_detection() {
        let fen = "9/9/9/9/9/9/9/4m4/8M F1/- w 1 - 1";
        let position = Position::from_fen(fen).expect("valid position");
        let parsed = parse_fen(fen).expect("valid parsed fen");
        let mv = parse_san("新砦(9-4-1)", &parsed).expect("legal drop move");

        assert!(is_sacrifice_move(&position, &mv));
    }

    #[test]
    fn test_classify_move_in_game_hard_to_find_gap() {
        let fen = "9/9/9/9/9/9/9/4m4/8M F1/- w 1 - 1";
        let position = Position::from_fen(fen).expect("valid position");
        let parsed = parse_fen(fen).expect("valid parsed fen");
        let played = parse_san("新砦(9-4-1)", &parsed).expect("legal drop move");
        let alt = position
            .moves()
            .into_iter()
            .find(|mv| mv != &played)
            .expect("alternate legal move");

        let hard_to_find = MultiPvResult {
            lines: vec![mpv_line(played.clone(), 20), mpv_line(alt.clone(), -20)],
            total_nodes: 2,
        };
        assert_eq!(
            classify_move_in_game(&position, 0, &played, &hard_to_find, None, None),
            Some(MoveClassification::Brilliant)
        );

        let obvious = MultiPvResult {
            lines: vec![mpv_line(played.clone(), 20), mpv_line(alt, -40)],
            total_nodes: 2,
        };
        assert_ne!(
            classify_move_in_game(&position, 0, &played, &obvious, None, None),
            Some(MoveClassification::Brilliant)
        );
    }

    #[test]
    fn test_classify_move_in_game_uses_previous_classifications_for_great() {
        let position = Position::new(SetupMode::Beginner);
        let legal = position.moves();
        let played = legal[0].clone();
        let alternate = legal[1].clone();
        let mpv = MultiPvResult {
            lines: vec![mpv_line(played.clone(), 20), mpv_line(alternate, 0)],
            total_nodes: 2,
        };

        let classification = classify_move_in_game(
            &position,
            0,
            &played,
            &mpv,
            None,
            Some(MoveClassification::Blunder),
        );
        assert_eq!(classification, Some(MoveClassification::Great));
    }

    #[test]
    fn test_analyze_full_game_sequence_with_context() {
        let mut position = Position::new(SetupMode::Beginner);
        let mut sequence = Vec::new();

        for _ in 0..4 {
            let legal = position.moves();
            let mv = legal[0].clone();
            sequence.push((position.fen(), move_to_san(&mv)));
            position.make_move(&mv).expect("legal move");
        }

        let analysis = analyze_full_game(&sequence, 1, 2).expect("analysis should succeed");
        assert_eq!(analysis.moves.len(), 4);
        assert!((0.0..=100.0).contains(&analysis.white_accuracy));
        assert!((0.0..=100.0).contains(&analysis.black_accuracy));
        assert!(!analysis.moves[0].best_san.is_empty());
    }
}
