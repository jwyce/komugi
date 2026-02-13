use komugi_core::{Gungi, Move, PieceType, Position, PositionError, SetupMode, Square};

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
