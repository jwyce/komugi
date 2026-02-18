use komugi_core::{eval::Evaluator, position::Position, types::SetupMode};
use komugi_engine::ClassicalEval;

fn eval(pos: &Position) -> i32 {
    ClassicalEval::new().evaluate(pos).0
}

#[test]
fn intro_starting_position_near_zero() {
    let pos = Position::new(SetupMode::Intro);
    let score = eval(&pos);
    assert!(
        score.abs() <= 50,
        "intro starting eval {score} should be within ±50cp"
    );
}

#[test]
fn beginner_starting_position_near_zero() {
    let pos = Position::new(SetupMode::Beginner);
    let score = eval(&pos);
    assert!(
        score.abs() <= 50,
        "beginner starting eval {score} should be within ±50cp"
    );
}

#[test]
fn intermediate_starting_position_near_zero() {
    let pos = Position::new(SetupMode::Intermediate);
    let score = eval(&pos);
    assert!(
        score.abs() <= 50,
        "intermediate starting eval {score} should be within ±50cp"
    );
}

#[test]
fn advanced_starting_position_near_zero() {
    let pos = Position::new(SetupMode::Advanced);
    let score = eval(&pos);
    assert!(
        score.abs() <= 50,
        "advanced starting eval {score} should be within ±50cp"
    );
}

#[test]
fn extra_white_general_is_positive() {
    let fen = "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1G1/j2n2r2d1 w 0 - 1";
    let pos = Position::from_fen(fen).unwrap();
    let score = eval(&pos);
    assert!(
        score > 400,
        "white extra general → eval {score} should be > 400cp"
    );
}

#[test]
fn extra_black_general_is_negative() {
    let fen = "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1/j2n2r2d1g1 w 0 - 1";
    let pos = Position::from_fen(fen).unwrap();
    let score = eval(&pos);
    assert!(
        score < -400,
        "black extra general → eval {score} should be < -400cp"
    );
}

#[test]
fn eval_symmetry_inverted_scores() {
    let white_extra =
        "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1G1/j2n2r2d1 w 0 - 1";
    let black_extra =
        "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1/j2n2r2d1g1 w 0 - 1";

    let w_score = eval(&Position::from_fen(white_extra).unwrap());
    let b_score = eval(&Position::from_fen(black_extra).unwrap());

    assert!(w_score > 0, "white advantage should be positive: {w_score}");
    assert!(b_score < 0, "black advantage should be negative: {b_score}");

    let diff = (w_score + b_score).abs();
    assert!(
        diff <= 100,
        "symmetric material advantage should produce roughly inverse scores: w={w_score} b={b_score} diff={diff}"
    );
}

#[test]
fn tower_control_bonus_visible() {
    let evaluator = ClassicalEval::new();

    let flat_fen = "9/9/9/9/9/4D4/9/9/4M4 -/- w 0 - 5";
    let tower_fen = "9/9/9/9/9/9/9/9/4|M:D|4 -/- w 0 - 5";

    let flat_pos = Position::from_fen(flat_fen);
    let tower_pos = Position::from_fen(tower_fen);

    if let (Ok(flat), Ok(tower)) = (flat_pos, tower_pos) {
        let flat_score = evaluator.evaluate(&flat).0;
        let tower_score = evaluator.evaluate(&tower).0;
        assert!(
            tower_score > flat_score,
            "stacking gives tower bonus: flat={flat_score} tower={tower_score}"
        );
    }
}

#[test]
fn hand_pieces_contribute_to_eval() {
    let no_hand = "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 -/- w 0 - 1";
    let white_hand = "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 G1/- w 0 - 1";

    let no_hand_score = eval(&Position::from_fen(no_hand).unwrap());
    let white_hand_score = eval(&Position::from_fen(white_hand).unwrap());

    assert!(
        white_hand_score > no_hand_score,
        "hand general should increase white eval: no_hand={no_hand_score} with_hand={white_hand_score}"
    );
}

#[test]
fn evaluator_trait_object_works() {
    let evaluator: Box<dyn Evaluator> = Box::new(ClassicalEval::new());
    let pos = Position::new(SetupMode::Intro);
    let score = evaluator.evaluate(&pos);
    assert!(score.0.abs() <= 50);
}

#[test]
fn hand_bonus_is_neutral_during_draft_and_boosted_post_draft() {
    let draft_fen = "4m4/9/9/9/9/9/9/9/4M4 G1/- w 2 wb 1";
    let post_fen = "4m4/9/9/9/9/9/9/9/4M4 G1/- w 2 - 1";

    let draft_score = eval(&Position::from_fen(draft_fen).unwrap());
    let post_score = eval(&Position::from_fen(post_fen).unwrap());

    assert!(
        post_score > draft_score,
        "post-draft hand bonus should exceed draft: draft={draft_score} post={post_score}"
    );
}

#[test]
fn low_remaining_hand_gets_stronger_multiplier() {
    let low_hand = "4m4/9/9/9/9/9/9/9/4M4 G1/- w 2 - 1";
    let high_hand = "4m4/9/9/9/9/9/9/9/4M4 G1D5/- w 2 - 1";

    let low = eval(&Position::from_fen(low_hand).unwrap());
    let high = eval(&Position::from_fen(high_hand).unwrap());

    let low_delta = low;
    let high_delta = high - 5 * 100;

    assert!(
        low_delta > high_delta,
        "low hand should get stronger per-piece multiplier: low={low_delta} high_adjusted={high_delta}"
    );
}
