use komugi_core::fen::parse_fen;
use komugi_core::san::parse_san;
use komugi_core::{
    is_marshal_captured, Gungi, Move, MoveType, PieceType, Position, PositionError, SetupMode,
    Square,
};

fn find_move(
    moves: &[Move],
    piece: PieceType,
    from: Option<(u8, u8)>,
    to: (u8, u8),
) -> Result<Move, PositionError> {
    moves
        .iter()
        .find(|mv| {
            mv.piece == piece
                && mv
                    .from
                    .map(|from_sq| (from_sq.square.rank, from_sq.square.file))
                    == from
                && (mv.to.square.rank, mv.to.square.file) == to
        })
        .cloned()
        .ok_or(PositionError::Fen("expected move not found".to_string()))
}

#[test]
fn make_unmake_round_trip_restores_position() {
    let mut position = Position::new(SetupMode::Beginner);
    let start_fen = position.fen();
    let start_hash = position.zobrist_hash;
    let start_turn = position.turn;
    let start_number = position.move_number;

    let mv = position.moves().first().cloned().expect("must have move");
    position.make_move(&mv).unwrap();
    position.unmake_move().unwrap();

    assert_eq!(position.fen(), start_fen);
    assert_eq!(position.zobrist_hash, start_hash);
    assert_eq!(position.turn, start_turn);
    assert_eq!(position.move_number, start_number);
    assert_eq!(position.history.len(), 0);
}

#[test]
fn detects_checkmate() {
    let gungi = Gungi::from_fen(
        "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3 J2N2S1R1D1/j2n2s1r1d1 b 1 - 3",
    )
    .unwrap();
    assert!(gungi.is_checkmate());
    assert!(gungi.in_check());
    assert!(gungi.is_game_over());
}

#[test]
fn not_checkmate_with_escape() {
    let gungi = Gungi::from_fen(
        "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 b 1 - 3",
    )
    .unwrap();
    assert!(!gungi.is_checkmate());
    assert!(gungi.in_check());
    assert!(!gungi.moves().is_empty());
}

#[test]
fn detects_stalemate() {
    let gungi = Gungi::from_fen("8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1").unwrap();
    assert!(!gungi.in_check());
    assert!(gungi.is_stalemate());
    assert!(gungi.is_draw());
    assert!(gungi.is_game_over());
    assert_eq!(gungi.moves().len(), 0);
}

#[test]
fn insufficient_material_rules_match_reference() {
    assert!(Gungi::from_fen("m8/9/9/9/9/9/9/9/8M -/- w 3 - 1")
        .unwrap()
        .is_insufficient_material());
    assert!(!Gungi::from_fen("mM7/9/9/9/9/9/9/9/9 -/- w 3 - 1")
        .unwrap()
        .is_insufficient_material());
    assert!(!Gungi::from_fen("m8/1M7/9/9/9/9/9/9/9 -/- w 3 - 1")
        .unwrap()
        .is_insufficient_material());
    assert!(!Gungi::from_fen("m8/9/9/9/9/9/9/9/8M D1/d1 w 3 - 1")
        .unwrap()
        .is_insufficient_material());
    assert!(!Gungi::from_fen("m8/9/9/9/9/9/9/9/9 -/- w 3 - 1")
        .unwrap()
        .is_insufficient_material());
    assert!(!Gungi::from_fen("mMm6/9/9/9/9/9/9/9/9 -/- w 3 - 1")
        .unwrap()
        .is_insufficient_material());
}

#[test]
fn draft_phase_disables_endgame_flags() {
    let gungi = Gungi::from_fen(
        "9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1",
    )
    .unwrap();
    assert!(!gungi.is_checkmate());
    assert!(!gungi.is_stalemate());
    assert!(!gungi.is_insufficient_material());
    assert!(!gungi.is_game_over());
}

#[test]
fn marshal_capture_ends_game() {
    let gungi = Gungi::from_fen(
        "1|g:N|2|W:N|Ad1f/7r1/1nd2Adfr/2|c:G|j2K2/6s1D/1w|W:T|6/2F4J|F:D|/i8/2|S:w||R:M|3C1 -/- b 3 - 164",
    )
    .unwrap();
    assert!(gungi.is_game_over());
    assert!(!gungi.is_checkmate());
    assert!(!gungi.is_stalemate());
}

#[test]
fn draw_and_game_over_integration() {
    let normal = Gungi::from_fen(
        "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1",
    )
    .unwrap();
    assert!(!normal.is_draw());
    assert!(!normal.is_game_over());

    let draw = Gungi::from_fen("m8/9/9/9/9/9/9/9/8M -/- w 3 - 1").unwrap();
    assert!(draw.is_draw());
    assert!(draw.is_game_over());

    let mate = Gungi::from_fen(
        "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3 J2N2S1R1D1/j2n2s1r1d1 b 1 - 3",
    )
    .unwrap();
    assert!(!mate.is_draw());
    assert!(mate.is_game_over());
}

#[test]
fn draft_move_generation_marks_end_moves() {
    let one_left =
        Gungi::from_fen("9/9/9/9/9/9/9/9/4M4 D1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 w 1").unwrap();
    let one_left_moves = one_left.moves();
    assert!(!one_left_moves.is_empty());
    assert!(one_left_moves.iter().all(|mv| mv.draft_finished));

    let many_left =
        Gungi::from_fen("9/9/9/9/9/9/9/9/4M4 D2/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 w 1").unwrap();
    let many_left_moves = many_left.moves();
    assert!(many_left_moves.iter().any(|mv| mv.draft_finished));
    assert!(many_left_moves.iter().any(|mv| !mv.draft_finished));
}

#[test]
fn fourfold_repetition_detected_via_zobrist() {
    let mut position = Position::from_fen("m8/9/9/9/9/9/9/1G7/8M -/- w 3 - 1").unwrap();

    for _ in 0..3 {
        let w1 = find_move(&position.moves(), PieceType::General, Some((8, 8)), (8, 7)).unwrap();
        position.make_move(&w1).unwrap();

        let b1 = find_move(&position.moves(), PieceType::Marshal, Some((1, 9)), (1, 8)).unwrap();
        position.make_move(&b1).unwrap();

        let w2 = find_move(&position.moves(), PieceType::General, Some((8, 7)), (8, 8)).unwrap();
        position.make_move(&w2).unwrap();

        let b2 = find_move(&position.moves(), PieceType::Marshal, Some((1, 8)), (1, 9)).unwrap();
        position.make_move(&b2).unwrap();
    }

    assert!(position.is_fourfold_repetition());
}

#[test]
fn gungi_wrapper_load_make_undo_and_accessors_work() {
    let mut gungi = Gungi::new(SetupMode::Beginner);
    let original = gungi.fen();
    let initial_turn = gungi.turn();

    let mv = gungi.moves().first().cloned().expect("must have move");
    gungi.make_move(&mv).unwrap();
    assert_ne!(gungi.turn(), initial_turn);
    assert_eq!(gungi.history().len(), 1);

    gungi.undo().unwrap();
    assert_eq!(gungi.fen(), original);
    assert_eq!(gungi.history().len(), 0);

    gungi
        .load("9/9/9/9/9/9/9/9/9 -/- b 3 - 12")
        .expect("load works");
    assert_eq!(gungi.move_number(), 12);
    assert_eq!(gungi.turn(), komugi_core::Color::Black);
    assert!(gungi.board().get(Square::new_unchecked(1, 1)).is_none());
    assert!(gungi.hand().is_empty());
}

#[test]
fn game113_should_not_terminate_at_move_104() {
    let moves_str = "新小(8-3-2)付 新忍(2-5-2)付 新馬(9-9-1) 新小(1-3-1) 新槍(7-9-2)付 新兵(1-3-2)付 新忍(8-7-2)付 新馬(1-6-2)付 新小(8-8-2)付 新槍(3-7-2)付 新槍(9-4-2)付 新槍(2-9-1) 新兵(7-1-2)付 新小(1-1-1) 大(9-6-1)(9-9-2)付 大(1-4-1)(3-4-2)付 侍(7-4-1)(8-4-1) 帥(1-5-1)(1-4-1) 大(9-9-2)(9-6-1) 小(1-1-1)(2-2-2)付 馬(8-2-1)(6-2-1) 砦(3-3-1)(2-4-1) 小(8-8-2)(6-8-1) 馬(2-8-1)(4-8-1) 槍(7-9-2)(6-8-2)付 槍(2-9-1)(3-9-2)付 大(9-6-1)(7-6-2)付 兵(1-3-2)(2-3-2)付 砦(7-3-1)(8-4-2)付 小(1-3-1)(2-4-2)付 馬(9-9-1)(7-9-2)付 帥(1-4-1)(1-5-1) 槍(9-4-2)(8-5-2)付 兵(3-1-1)(2-1-1) 忍(8-8-1)(7-7-2)付 馬(4-8-1)(4-7-1) 兵(7-5-1)(6-5-1) 兵(2-1-1)(1-1-1) 馬(6-2-1)(6-3-1) 小(2-4-2)(1-4-1) 槍(8-5-2)(6-5-2)付 弓(2-7-1)(1-7-1) 中(9-4-1)(8-5-2)付 馬(4-7-1)(4-8-1) 中(8-5-2)(6-3-2)付 槍(3-7-2)(1-7-2)付 砦(8-4-2)(8-5-2)付 小(1-4-1)(2-4-2)付 槍(6-8-2)取(4-8-1) 槍(3-9-2)取(4-8-1) 忍(7-7-2)(6-8-2)付 帥(1-5-1)(2-6-1) 砦(7-7-1)(8-6-1) 兵(3-5-1)(4-5-1) 砦(8-5-2)(8-6-2)付 槍(1-7-2)(3-7-2)付 帥(9-5-1)(9-4-1) 槍(3-7-2)(4-8-2)付 忍(6-8-2)(5-9-1) 兵(3-9-1)(4-9-1) 中(6-3-2)(6-2-1) 兵(4-5-1)(3-5-1) 小(6-8-1)(5-9-2)付 槍(4-8-2)取(5-9-1) 帥(9-4-1)(9-3-1) 槍(5-9-1)(6-8-1) 馬(7-9-2)取(4-9-1) 槍(4-8-1)(6-8-2)付 砦(8-6-2)(8-5-2)付 兵(1-1-1)(2-1-1) 大(7-6-2)(6-6-1) 弓(1-7-1)(3-6-2)付 大(6-6-1)(7-6-2)付 槍(6-8-2)取(7-9-1) 中(6-2-1)(6-3-2)付 兵(3-5-1)(4-5-1) 帥(9-3-1)(9-2-1) 弓(3-6-2)(6-8-2)付 大(7-6-2)取(7-9-1) 弓(6-8-2)取(8-7-1) 大(7-9-1)取(6-8-1) 大(3-4-2)(3-6-2)付 砦(8-6-1)取(8-7-1) 大(3-6-2)取(7-6-1) 砦(8-5-2)(8-6-1) 大(7-6-1)(4-6-1) 馬(4-9-1)(6-9-1) 小(2-2-2)(2-1-2)付 大(6-8-1)(6-9-2)付 砦(3-7-1)(3-6-2)付 侍(8-4-1)(7-5-1) 忍(2-2-1)(3-3-1) 砦(8-7-1)(8-6-2)付 大(4-6-1)(3-5-1) 槍(8-5-1)(7-5-2)付 大(3-5-1)(3-4-2)付 帥(9-2-1)(9-3-1) 小(2-1-2)(2-2-1) 槍(7-5-2)(5-7-1) 小(2-2-1)(3-3-2)付 侍(7-5-1)(8-5-1) 帥(2-6-1)(3-5-1) 中(6-3-2)(8-5-2)付 兵(2-1-1)(3-1-1) 小(8-3-2)(6-3-2)付 小(3-3-2)(3-1-2)付 槍(6-5-2)(5-4-1) 兵(2-3-2)(3-3-2)付 中(8-5-2)(6-5-2)付 弓(2-3-1)(1-3-1) 大(6-9-2)(9-9-1) 大(3-4-2)(1-2-1) 槍(5-4-1)取(3-4-1) 忍(2-5-2)取(3-4-1) 中(6-5-2)(7-5-1) 大(1-2-1)(2-2-1) 砦(8-6-2)(8-8-1) 忍(3-4-1)(4-5-2)付 小(6-3-2)取(4-5-1) 帥(3-5-1)取(4-5-1) 弓(8-3-1)(6-3-2)付 帥(4-5-1)(5-6-1) 弓(6-3-2)取(3-1-1) 大(2-2-1)取(3-1-1) 中(7-5-1)(5-7-2)付 帥(5-6-1)取(6-5-1) 中(5-7-2)取(2-4-1) 帥(6-5-1)(5-4-1) 馬(6-3-1)(8-3-1) 大(3-1-1)(2-1-1) 中(2-4-1)(5-7-2)付 帥(5-4-1)(4-5-1) 大(9-9-1)(8-8-2)付 弓(1-3-1)(3-2-1) 砦(8-6-1)(8-5-2)付 大(2-1-1)(3-2-2)付 馬(8-3-1)(6-3-1) 槍(2-5-1)(3-5-1) 馬(6-9-1)(5-9-1) 帥(4-5-1)(3-4-1) 馬(6-3-1)(8-3-1) 大(3-2-2)(5-4-1) 馬(5-9-1)(6-9-1) 帥(3-4-1)(4-3-1) 大(8-8-2)(6-6-1) 槍(3-5-1)(4-5-1) 馬(8-3-1)(6-3-1) 弓(3-2-1)(5-3-1) 大(6-6-1)(6-3-2)付 帥(4-3-1)(4-2-1) 兵(7-1-2)(6-1-1) 帥(4-2-1)(3-2-1) 帥(9-3-1)(8-3-1) 帥(3-2-1)(2-3-1) 帥(8-3-1)(9-3-1) 帥(2-3-1)(1-2-1) 大(6-3-2)取(5-4-1) 槍(4-5-1)取(5-4-1) 馬(6-3-1)(6-2-1) 弓(5-3-1)(7-4-1) 帥(9-3-1)(8-2-1) 槍(5-4-1)(7-4-2)付 砦(8-5-2)(7-5-1) 帥(1-2-1)(2-3-1) 侍(8-5-1)(7-5-2)付 帥(2-3-1)(1-4-1) 砦(8-8-1)(8-7-1) 帥(1-4-1)(1-5-1) 兵(7-1-1)(6-1-2)付 帥(1-5-1)(2-6-1) 馬(6-9-1)(4-9-1) 帥(2-6-1)(1-5-1) 馬(6-2-1)(6-3-1) 帥(1-5-1)(1-4-1) 中(5-7-2)(7-9-1) 帥(1-4-1)(1-5-1) 槍(5-7-1)(4-7-1) 帥(1-5-1)(1-4-1) 馬(6-3-1)(8-3-1) 帥(1-4-1)(2-5-1) 馬(8-3-1)(7-3-1) 砦(3-6-2)(3-4-1) 槍(4-7-1)(5-7-1) 馬(1-6-2)(3-6-2)付 馬(4-9-1)(6-9-1) 中(1-6-1)(2-6-1) 中(7-9-1)(6-9-2)付 馬(3-6-2)(3-4-2)付 侍(7-5-2)(6-6-1) 中(2-6-1)(3-5-1) 砦(7-5-1)(6-5-1) 帥(2-5-1)(2-4-1) 侍(6-6-1)(5-7-2)付 中(3-5-1)(6-2-1) 砦(8-7-1)(8-6-1) 侍(3-6-1)(4-5-1) 砦(8-6-1)(7-6-1) 侍(4-5-1)(3-5-1) 砦(6-5-1)(7-6-2)付 槍(7-4-2)(9-2-1) 帥(8-2-1)(8-3-1) 弓(7-4-1)(6-4-1) 帥(8-3-1)取(9-2-1) 中(6-2-1)取(7-3-1) 侍(5-7-2)取(3-5-1) 帥(2-4-1)取(3-5-1) 兵(6-1-2)(5-1-1) 中(7-3-1)(8-2-1)";

    let sans: Vec<&str> = moves_str.split_whitespace().collect();

    let mut gungi = Gungi::new(SetupMode::Beginner);

    // Replay all 208 plies
    for (i, san) in sans.iter().enumerate() {
        let fen = gungi.fen();
        let parsed = parse_fen(&fen).expect(&format!("parse fen at ply {i}"));
        let mv = parse_san(san, &parsed).expect(&format!("parse san '{san}' at ply {i}"));
        gungi
            .make_move(&mv)
            .expect(&format!("make move '{san}' at ply {i}"));
    }

    // After Black's move 104 (中(7-3-1)(8-2-1)), it's White's turn
    let fen = gungi.fen();
    let position = &gungi;

    eprintln!("Final FEN: {fen}");
    eprintln!("Turn: {:?}", position.turn());
    eprintln!("Is game over: {}", position.is_game_over());
    eprintln!("Is checkmate: {}", position.is_checkmate());
    eprintln!("Is stalemate: {}", position.is_stalemate());
    eprintln!("Is draw: {}", position.is_draw());

    let pos = Position::from_fen(&fen).unwrap();
    eprintln!("Is marshal captured: {}", is_marshal_captured(&pos));
    eprintln!(
        "White in check: {}",
        pos.in_check(Some(komugi_core::Color::White))
    );

    let legal_moves = gungi.moves();
    eprintln!("Legal moves count: {}", legal_moves.len());
    for mv in legal_moves.iter().take(10) {
        eprintln!("  {}", komugi_core::move_to_san(mv));
    }

    assert!(
        !position.is_game_over(),
        "Game should not be over after move 104 — White Marshal can capture LtGeneral at (8,2)"
    );
}

#[test]
fn game158_should_not_terminate_early() {
    let moves_str = "新小(8-7-2)付 新槍(2-4-1) 新忍(8-9-1) 新馬(1-9-1) 新馬(9-1-1) 新小(2-6-1) 新小(8-8-2)付 新槍(3-6-2)付 新槍(9-1-2)付 新忍(2-3-2)付 大(9-6-1)(8-5-2)付 兵(3-1-1)(2-1-1) 新槍(8-3-2)付 新小(3-3-2)付 兵(7-9-1)(8-9-2)付 新兵(2-2-2)付 小(8-7-2)(7-7-2)付 兵(3-9-1)(4-9-1) 新兵(7-4-2)付 兵(2-1-1)(3-1-1) 中(9-4-1)(9-3-1) 砦(3-7-1)(2-6-2)付 槍(9-1-2)(8-2-2)付 中(1-6-1)(1-7-1) 馬(9-1-1)(7-1-2)付 馬(1-9-1)(3-9-1) 中(9-3-1)(7-5-2)付 馬(2-8-1)(3-8-1) 帥(9-5-1)(8-6-1) 大(1-4-1)(2-4-2)付 槍(8-2-2)(7-3-2)付 馬(3-9-1)(3-8-2)付 小(7-7-2)(7-6-2)付 中(1-7-1)(2-7-2)付 小(8-8-2)(7-9-1) 大(2-4-2)(3-5-2)付 忍(8-8-1)(7-7-2)付 侍(3-4-1)(2-4-2)付 中(7-5-2)(6-4-1) 兵(3-1-1)(4-1-1) 馬(8-2-1)(9-2-1) 兵(4-1-1)(3-1-1) 槍(7-3-2)(6-4-2)付 兵(3-1-1)(4-1-1) 馬(9-2-1)(7-2-1) 砦(2-6-2)(2-5-2)付 馬(7-2-1)(7-3-2)付 兵(4-9-1)(3-9-1) 兵(7-5-1)(6-5-1) 兵(4-1-1)(3-1-1) 帥(8-6-1)(7-5-1) 兵(3-1-1)(4-1-1) 小(7-9-1)(7-8-1) 砦(2-5-2)(1-6-1) 弓(8-7-1)(9-7-1) 小(2-6-1)(2-5-2)付 弓(9-7-1)(7-8-2)付 兵(4-1-1)(3-1-1) 兵(6-5-1)(5-5-1) 馬(3-8-2)(2-8-1) 馬(7-3-2)(6-3-1) 中(2-7-2)(2-8-2)付 砦(7-3-1)(6-3-2)付 馬(3-8-1)(3-9-2)付 槍(6-4-2)(5-4-1) 弓(2-7-1)(4-7-1) 槍(5-4-1)(6-4-2)付 兵(3-1-1)(4-1-1) 弓(7-8-2)(5-8-1) 槍(3-6-2)(4-7-2)付 忍(7-7-2)(8-6-1) 中(2-8-2)(4-8-1) 小(7-6-2)(6-6-1) 槍(4-7-2)取(5-8-1) 侍(7-6-1)(6-6-2)付 中(4-8-1)(5-8-2)付 大(8-5-2)(8-6-2)付 侍(3-6-1)(4-7-2)付 小(7-8-1)(7-7-2)付 帥(1-5-1)(1-4-1) 槍(8-5-1)(9-5-1) 馬(2-8-1)(2-7-1) 槍(9-5-1)(8-4-1) 砦(1-6-1)(1-5-1) 兵(7-4-2)(8-4-2)付 馬(2-7-1)(2-8-1) 槍(6-4-2)(5-5-2)付 小(3-3-2)取(5-5-1) 侍(6-6-2)取(5-5-1) 侍(2-4-2)(3-3-2)付 中(6-4-1)(7-4-2)付 侍(4-7-2)(3-7-1) 小(6-6-1)(6-5-1) 弓(4-7-1)(3-7-2)付 小(6-5-1)(6-4-1) 侍(3-3-2)取(5-5-1) 小(6-4-1)取(5-5-1) 槍(2-4-1)(3-3-2)付 砦(6-3-2)(6-5-1) 槍(3-3-2)取(5-5-1) 砦(6-5-1)取(5-5-1) 大(3-5-2)(3-3-2)付 砦(5-5-1)(5-4-1) 兵(3-5-1)(4-5-1) 馬(6-3-1)(7-3-1) 馬(2-8-1)(1-8-1) 馬(7-1-2)取(4-1-1) 忍(2-3-2)取(4-1-1) 中(7-4-2)取(4-1-1) 帥(1-4-1)(2-4-1) 中(4-1-1)(7-4-2)付 大(3-3-2)(2-3-2)付 馬(7-3-1)(5-3-1) 馬(1-8-1)(3-8-1) 砦(5-4-1)(5-3-2)付 大(2-3-2)(4-5-2)付 帥(7-5-1)(6-6-1) 弓(2-3-1)(4-4-1)";

    let sans: Vec<&str> = moves_str.split_whitespace().collect();
    let mut gungi = Gungi::new(SetupMode::Beginner);

    for (i, san) in sans.iter().enumerate() {
        let fen = gungi.fen();
        let parsed = parse_fen(&fen).expect(&format!("parse fen at ply {i}"));
        let mv = parse_san(san, &parsed).expect(&format!("parse san '{san}' at ply {i}"));
        gungi
            .make_move(&mv)
            .expect(&format!("make move '{san}' at ply {i}"));
    }

    assert!(
        !gungi.is_game_over(),
        "Game 158 should not end at move 59 — Archer at (4,4) does NOT attack Marshal at (6,6)"
    );
}

#[test]
fn arata_checkmate_via_soldier_drop() {
    let mut gungi = Gungi::from_fen("8m/7R1/6W2/9/9/9/9/9/4M4 D1/- w 1 - 1").unwrap();

    assert!(!gungi.is_game_over());
    assert!(!gungi.in_check());

    let moves = gungi.moves();
    let drop = moves
        .iter()
        .find(|mv| {
            mv.piece == PieceType::Soldier
                && mv.move_type == MoveType::Arata
                && mv.to.square == Square::new_unchecked(2, 1)
        })
        .cloned()
        .expect("Soldier drop at (2,1) must be a legal move");

    gungi.make_move(&drop).unwrap();

    let fen_after = gungi.fen();
    eprintln!("FEN after drop: {fen_after}");
    eprintln!("Turn: {:?}", gungi.turn());
    eprintln!("in_check: {}", gungi.in_check());
    eprintln!("is_checkmate: {}", gungi.is_checkmate());
    eprintln!("is_game_over: {}", gungi.is_game_over());
    eprintln!("legal moves: {}", gungi.moves().len());

    assert!(gungi.in_check());
    assert!(gungi.is_checkmate());
    assert!(gungi.is_game_over());
}

#[test]
fn arata_drop_blocks_check_prevents_checkmate() {
    let gungi = Gungi::from_fen("4m4/9/8d/9/4G4/9/9/9/4M4 -/f1 b 1 - 1").unwrap();

    assert!(gungi.in_check());
    assert!(
        !gungi.is_checkmate(),
        "Black can drop Fortress at (2,5) to block check"
    );
    assert!(!gungi.is_game_over());
}

#[test]
fn betrayal_generated_when_hand_has_matching_piece() {
    let gungi = Gungi::from_fen("4m4/9/9/9/9/9/3d5/4T4/4M4 D1/- w 3 - 1").unwrap();

    let moves = gungi.moves();
    let sans: Vec<String> = moves.iter().map(|m| komugi_core::move_to_san(m)).collect();

    assert!(sans.contains(&"謀(8-5-1)(7-6-2)返兵".to_string()));
    assert!(sans.contains(&"謀(8-5-1)(7-6-2)付".to_string()));
    assert!(sans.contains(&"謀(8-5-1)取(7-6-1)".to_string()));
    assert!(sans.contains(&"謀(8-5-1)(7-4-1)".to_string()));
}

#[test]
fn betrayal_blocked_when_hand_empty() {
    let gungi = Gungi::from_fen("4m4/9/9/9/9/9/3d5/4T4/4M4 -/- w 3 - 1").unwrap();

    let moves = gungi.moves();
    let sans: Vec<String> = moves.iter().map(|m| komugi_core::move_to_san(m)).collect();

    assert!(!sans.iter().any(|s| s.contains('返')));
}

#[test]
fn betrayal_converts_enemy_piece() {
    let mut gungi = Gungi::from_fen("4m4/9/9/9/9/9/3d5/4T4/4M4 D1/- w 3 - 1").unwrap();

    let moves = gungi.moves();
    let betray = moves
        .iter()
        .find(|mv| mv.move_type == MoveType::Betray)
        .cloned()
        .expect("betrayal move exists");

    gungi.make_move(&betray).unwrap();

    let fen = gungi.fen();
    assert!(
        fen.starts_with("4m4/9/9/9/9/9/3|D:T|5/9/4M4 -/- b 3 -"),
        "Board and hand should match after betrayal: {fen}"
    );
}

#[test]
fn betrayal_generates_partial_and_full_options_on_same_stack() {
    let mut gungi = Gungi::from_fen("4m4/9/9/9/9/9/3|d:f|5/4|D:T|4/4M4 D1F1/- w 3 - 1").unwrap();

    let moves = gungi.moves();
    let partial_soldier = moves
        .iter()
        .find(|mv| {
            mv.move_type == MoveType::Betray
                && mv.captured.len() == 1
                && mv.captured[0].piece_type == PieceType::Soldier
        })
        .cloned()
        .expect("partial betrayal (soldier) should exist");
    let partial_fortress = moves
        .iter()
        .find(|mv| {
            mv.move_type == MoveType::Betray
                && mv.captured.len() == 1
                && mv.captured[0].piece_type == PieceType::Fortress
        })
        .cloned()
        .expect("partial betrayal (fortress) should exist");
    let full = moves
        .iter()
        .find(|mv| mv.move_type == MoveType::Betray && mv.captured.len() == 2)
        .cloned()
        .expect("full betrayal should exist");

    assert_eq!(
        komugi_core::move_to_san(&partial_soldier),
        "謀(8-5-2)(7-6-3)返兵"
    );
    assert_eq!(
        komugi_core::move_to_san(&partial_fortress),
        "謀(8-5-2)(7-6-3)返砦"
    );
    assert!(
        komugi_core::move_to_san(&full) == "謀(8-5-2)(7-6-3)返兵砦"
            || komugi_core::move_to_san(&full) == "謀(8-5-2)(7-6-3)返砦兵"
    );

    gungi.make_move(&full).unwrap();
    let fen = gungi.fen();
    assert!(
        fen.starts_with("4m4/9/9/9/9/9/3|D:F:T|5/4D4/4M4 -/- b 3 -"),
        "full betrayal should convert both enemy pieces and consume matching hand pieces: {fen}"
    );
}

#[test]
fn arata_checkmate_general_drops_on_file() {
    // White drops General on top of Soldier at (8,5) → tier 2 tsuke
    // General's infinite rank probe attacks all the way to (1,5) where black Marshal is trapped
    let mut gungi =
        Gungi::from_fen("3|w:n:d|m|r:s:d|3/3|w:n:d|1|r:s:d|3/9/9/9/9/9/4D4/4M4 G1/- w 3 - 1")
            .unwrap();

    assert!(!gungi.is_game_over());
    assert!(!gungi.in_check());

    let moves = gungi.moves();
    let sans: Vec<String> = moves.iter().map(|m| komugi_core::move_to_san(m)).collect();
    eprintln!("Legal moves: {:?}", sans);

    let drop = moves
        .iter()
        .find(|mv| {
            mv.piece == PieceType::General
                && mv.move_type == MoveType::Arata
                && mv.to.square == Square::new_unchecked(8, 5)
                && mv.to.tier == 2
        })
        .cloned()
        .expect("General drop at (8,5,2) must be a legal move");

    gungi.make_move(&drop).unwrap();

    eprintln!("FEN after drop: {}", gungi.fen());
    eprintln!("in_check: {}", gungi.in_check());
    eprintln!("is_checkmate: {}", gungi.is_checkmate());
    eprintln!("black moves: {}", gungi.moves().len());

    assert!(
        gungi.in_check(),
        "Black Marshal should be in check from General on file 5"
    );
    assert!(
        gungi.is_checkmate(),
        "Should be checkmate — Marshal trapped by own 3-tier stacks"
    );
    assert!(gungi.is_game_over());
}

#[test]
fn betrayal_not_generated_on_marshal_top() {
    let gungi = Gungi::from_fen("9/9/9/9/9/9/3m5/4T4/4M4 D1/- w 3 - 1").unwrap();

    let moves = gungi.moves();
    let sans: Vec<String> = moves.iter().map(|m| komugi_core::move_to_san(m)).collect();

    assert!(!sans.iter().any(|s| s.contains('返')));
    assert!(sans.contains(&"謀(8-5-1)取(7-6-1)".to_string()));
}
