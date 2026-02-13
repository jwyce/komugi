use komugi_core::fen::{apply_move_to_fen, parse_fen, ADVANCED_POSITION, BEGINNER_POSITION};
use komugi_core::movegen::{
    generate_all_legal_moves_in_state, generate_arata, generate_moves_for_square, Probe, DIRS,
    PIECE_PROBES,
};
use komugi_core::san::{move_to_san, parse_san};
use komugi_core::types::{Color, HandPiece, MoveType, PieceType};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PerftBaseline {
    fen: String,
    depth: u8,
    nodes: u64,
}

#[test]
fn dirs_and_piece_probes_match_gungi_js() {
    assert_eq!(
        DIRS,
        [
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, 0),
            (1, -1)
        ]
    );

    assert_eq!(
        PIECE_PROBES[PieceType::Marshal as usize],
        [
            Probe::Finite { start: 1, carry: 1 },
            Probe::Finite { start: 1, carry: 1 },
            Probe::Finite { start: 1, carry: 1 },
            Probe::Finite { start: 1, carry: 1 },
            Probe::Finite { start: 1, carry: 1 },
            Probe::Finite { start: 1, carry: 1 },
            Probe::Finite { start: 1, carry: 1 },
            Probe::Finite { start: 1, carry: 1 },
        ]
    );

    assert_eq!(
        PIECE_PROBES[PieceType::General as usize],
        [
            Probe::Finite { start: 1, carry: 1 },
            Probe::Infinite,
            Probe::Finite { start: 1, carry: 1 },
            Probe::Infinite,
            Probe::Infinite,
            Probe::Finite { start: 1, carry: 1 },
            Probe::Infinite,
            Probe::Finite { start: 1, carry: 1 },
        ]
    );
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
fn piece_move_generation_covers_multiple_piece_types() {
    let state = parse_fen(BEGINNER_POSITION).expect("parse beginner");
    let mut seen = [0usize; 14];

    for sq in komugi_core::SQUARES {
        if let Some((piece, _)) = state.board.get_top(sq) {
            if piece.color != state.turn {
                continue;
            }
            let moves = generate_moves_for_square(&state, sq);
            if !moves.is_empty() {
                seen[piece.piece_type as usize] += moves.len();
            }
        }
    }

    let count = seen.iter().filter(|n| **n > 0).count();
    assert!(count >= 5, "expected broad piece coverage");
    assert!(seen[PieceType::Marshal as usize] > 0);
    assert!(seen[PieceType::General as usize] > 0);
    assert!(seen[PieceType::Lancer as usize] > 0);
}

#[test]
fn arata_generation_and_san_round_trip_work() {
    let state = parse_fen(ADVANCED_POSITION).expect("parse advanced");
    let hand_piece = HandPiece {
        piece_type: PieceType::Marshal,
        color: Color::White,
        count: 1,
    };

    let arata = generate_arata(&state, hand_piece);
    assert!(!arata.is_empty());
    assert!(arata.iter().all(|mv| mv.move_type == MoveType::Arata));

    let san = move_to_san(&arata[0]);
    let parsed = parse_san(&san, &state).expect("parse generated san");
    assert_eq!(parsed, arata[0]);
}

#[test]
fn move_types_include_route_tsuke_and_capture() {
    let state = parse_fen(BEGINNER_POSITION).expect("parse beginner");
    let all = generate_all_legal_moves_in_state(&state);

    assert!(all.iter().any(|m| m.move_type == MoveType::Route));
    assert!(all.iter().any(|m| m.move_type == MoveType::Tsuke));

    let capture_fen = "9/9/9/9/4d4/4W4/9/9/9 -/- w 3 - 1";
    let capture_state = parse_fen(capture_fen).expect("parse capture state");
    let capture_moves = generate_all_legal_moves_in_state(&capture_state);
    assert!(capture_moves
        .iter()
        .any(|m| m.move_type == MoveType::Capture));
}

#[test]
fn perft_matches_baselines_for_depth_1_2() {
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
            "fen={}, depth={}",
            baseline.fen, baseline.depth
        );
    }
}
