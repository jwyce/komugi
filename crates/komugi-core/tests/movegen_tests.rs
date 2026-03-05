use komugi_core::fen::{apply_move_to_fen, parse_fen, ADVANCED_POSITION, BEGINNER_POSITION};
use komugi_core::movegen::{
    generate_all_legal_moves_in_state, generate_arata, generate_moves_for_square, Probe, DIRS,
    PIECE_PROBES,
};
use komugi_core::san::{move_to_san, parse_san};
use komugi_core::types::{Color, HandPiece, MoveType, PieceType, Square};
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
fn archer_diagonal_probe_blocked_by_taller_wing_stack() {
    let fen =
        "4mg3/1r|a:r|in1a|s:f|1/d1fwd3d/2s1w2j1/9/9/D|S:F|1W|D:R|WF1D/1SA1N1ARN/3|G:D|M1I2 J2N1/j1n2d1 w 1 - 7";
    let state = parse_fen(fen).expect("parse archer blocker fixture");
    let moves = generate_moves_for_square(&state, Square::new_unchecked(8, 7));
    let sans: Vec<String> = moves.iter().map(move_to_san).collect();

    assert!(
        !sans.iter().any(|san| san == "弓(8-7-1)(6-8-1)"),
        "archer diagonal side probe should be blocked by taller adjacent wing stack"
    );
}

#[test]
fn leap_pieces_cannot_jump_over_side_or_back_blockers() {
    let archer_fen =
        "4f1|d:s|2/j1asnm2j/dr2|w:d|1i1|d:f|/1WA1D1|g:a|2/5G3/D1Jn2n1|I:r|/1S2D|W:J|MR|D:F|/1|S:N|4AR1/2F1N4 -/- b 1 - 48";
    let archer_state = parse_fen(archer_fen).expect("parse archer leap fixture");
    let archer_sans: Vec<String> = generate_all_legal_moves_in_state(&archer_state)
        .iter()
        .map(move_to_san)
        .collect();
    assert!(
        !archer_sans.iter().any(|san| san == "弓(4-3-2)(2-3-1)"),
        "archer must not leap over backward blocker"
    );

    let cannon_fen =
        "|n:t|an|r:d:c|d1|f:d:s|f|w:g|/5m|k:n||r:d|1/w2j3C1/1|I:J|3|j:N|1ag/1T2s|W:D:S|3/2S3R2/3ANi2|J:D|/2|F:D|2M1D1/|A:F|G2K1N2 -/- b 3 - 48";
    let cannon_state = parse_fen(cannon_fen).expect("parse cannon leap fixture");
    let cannon_sans: Vec<String> = generate_all_legal_moves_in_state(&cannon_state)
        .iter()
        .map(move_to_san)
        .collect();
    assert!(
        !cannon_sans.iter().any(|san| san == "砲(1-6-3)(1-4-1)"),
        "cannon must not leap over side blocker"
    );

    let musketeer_fen =
        "2w2|r:K||n:k:d||r:f:c|j/5dt2/djiJ1N2s/|s:A|8/3RD4/1g1T1n1m1/DNS3M1N/G1WS5/1F|A:D|1R1F2 -/- b 3 - 81";
    let musketeer_state = parse_fen(musketeer_fen).expect("parse musketeer leap fixture");
    let musketeer_sans: Vec<String> = generate_all_legal_moves_in_state(&musketeer_state)
        .iter()
        .map(move_to_san)
        .collect();
    assert!(
        !musketeer_sans.iter().any(|san| san == "筒(1-4-2)(3-2-1)"),
        "musketeer must not leap over non-forward blocker"
    );
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
        let drafting_field = baseline.fen.split_whitespace().nth(4).unwrap_or("-");
        if drafting_field.contains('w') || drafting_field.contains('b') {
            continue;
        }
        let actual = perft(&baseline.fen, baseline.depth);
        assert_eq!(
            actual, baseline.nodes,
            "fen={}, depth={}",
            baseline.fen, baseline.depth
        );
    }
}
