use komugi_core::fen::{
    apply_move_to_fen, encode_fen, parse_fen, ADVANCED_POSITION, BEGINNER_POSITION,
    INTERMEDIATE_POSITION, INTRO_POSITION,
};
use komugi_core::movegen::generate_all_legal_moves_in_state;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PerftBaseline {
    fen: String,
    depth: u8,
    nodes: u64,
}

fn perft(fen: &str, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let state = parse_fen(fen).expect("valid fen");
    let moves = generate_all_legal_moves_in_state(&state);
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for mv in moves {
        let next = apply_move_to_fen(fen, &mv).expect("apply move");
        nodes += perft(&next, depth - 1);
    }
    nodes
}

#[test]
fn perft_validation_against_baselines() {
    let fixture_path = format!(
        "{}/tests/fixtures/perft_baselines.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture = std::fs::read_to_string(fixture_path).expect("read fixture");
    let baselines: Vec<PerftBaseline> = serde_json::from_str(&fixture).expect("parse fixture");

    for baseline in baselines.into_iter().filter(|b| b.depth <= 2) {
        let actual = perft(&baseline.fen, baseline.depth);
        assert_eq!(
            actual, baseline.nodes,
            "perft mismatch: fen={}, depth={}, expected={}, actual={}",
            baseline.fen, baseline.depth, baseline.nodes, actual
        );
    }
}

#[test]
fn fen_round_trip_intro_position() {
    let parsed = parse_fen(INTRO_POSITION).expect("parse intro");
    let encoded = encode_fen(&parsed);
    let reparsed = parse_fen(&encoded).expect("reparse encoded");

    assert_eq!(parsed.turn, reparsed.turn);
    assert_eq!(parsed.mode, reparsed.mode);
    assert_eq!(parsed.move_number, reparsed.move_number);
}

#[test]
fn fen_round_trip_beginner_position() {
    let parsed = parse_fen(BEGINNER_POSITION).expect("parse beginner");
    let encoded = encode_fen(&parsed);
    let reparsed = parse_fen(&encoded).expect("reparse encoded");

    assert_eq!(parsed.turn, reparsed.turn);
    assert_eq!(parsed.mode, reparsed.mode);
    assert_eq!(parsed.move_number, reparsed.move_number);
}

#[test]
fn fen_round_trip_intermediate_position() {
    let parsed = parse_fen(INTERMEDIATE_POSITION).expect("parse intermediate");
    let encoded = encode_fen(&parsed);
    let reparsed = parse_fen(&encoded).expect("reparse encoded");

    assert_eq!(parsed.turn, reparsed.turn);
    assert_eq!(parsed.mode, reparsed.mode);
    assert_eq!(parsed.move_number, reparsed.move_number);
}

#[test]
fn fen_round_trip_advanced_position() {
    let parsed = parse_fen(ADVANCED_POSITION).expect("parse advanced");
    let encoded = encode_fen(&parsed);
    let reparsed = parse_fen(&encoded).expect("reparse encoded");

    assert_eq!(parsed.turn, reparsed.turn);
    assert_eq!(parsed.mode, reparsed.mode);
    assert_eq!(parsed.move_number, reparsed.move_number);
}

#[test]
fn full_game_replay_from_starting_position() {
    let mut current_fen = BEGINNER_POSITION.to_string();

    // Generate moves from starting position
    let state = parse_fen(&current_fen).expect("parse starting");
    let moves = generate_all_legal_moves_in_state(&state);

    assert!(
        !moves.is_empty(),
        "should have legal moves from starting position"
    );

    // Apply first move
    let first_move = &moves[0];
    current_fen = apply_move_to_fen(&current_fen, first_move).expect("apply first move");

    // Verify state is valid after move
    let state_after = parse_fen(&current_fen).expect("parse after first move");
    assert_ne!(
        state_after.turn, state.turn,
        "turn should change after move"
    );

    // Apply second move
    let moves_after = generate_all_legal_moves_in_state(&state_after);
    assert!(
        !moves_after.is_empty(),
        "should have legal moves after first move"
    );

    let second_move = &moves_after[0];
    current_fen = apply_move_to_fen(&current_fen, second_move).expect("apply second move");

    // Verify state is valid after second move
    let state_after_second = parse_fen(&current_fen).expect("parse after second move");
    assert_eq!(
        state_after_second.turn, state.turn,
        "turn should cycle back after two moves"
    );
}

#[test]
fn game_replay_maintains_valid_state() {
    let mut current_fen = BEGINNER_POSITION.to_string();
    let mut parsed = parse_fen(&current_fen).expect("parse starting");

    for _ in 0..4 {
        let moves = generate_all_legal_moves_in_state(&parsed);
        if moves.is_empty() {
            break;
        }

        current_fen = apply_move_to_fen(&current_fen, &moves[0]).expect("apply move");
        parsed = parse_fen(&current_fen).expect("parse after move");

        assert!(parsed.move_number > 0, "move number should be positive");
    }
}

#[test]
fn perft_depth_2_completes_quickly() {
    let start = std::time::Instant::now();
    let result = perft(BEGINNER_POSITION, 2);
    let elapsed = start.elapsed();

    assert!(result > 0, "perft should return non-zero nodes");
    assert!(
        elapsed.as_secs() < 5,
        "perft depth 2 took too long: {:?}",
        elapsed
    );
}
