use komugi_core::{
    encode_fen, parse_fen, validate_fen, ADVANCED_POSITION, BEGINNER_POSITION,
    INTERMEDIATE_POSITION, INTRO_POSITION,
};
use komugi_core::{Color, HandPiece, ParsedFen, Piece, PieceType, SetupMode, Square};

fn sq_from_idx(rank_idx: usize, file_idx: usize) -> Square {
    Square::new((rank_idx + 1) as u8, (9 - file_idx) as u8).unwrap()
}

fn top_piece(parsed: &ParsedFen, rank_idx: usize, file_idx: usize) -> Option<Piece> {
    parsed
        .board
        .get_top(sq_from_idx(rank_idx, file_idx))
        .map(|x| x.0)
}

fn tower_pieces(parsed: &ParsedFen, rank_idx: usize, file_idx: usize) -> Vec<Piece> {
    parsed
        .board
        .get(sq_from_idx(rank_idx, file_idx))
        .map(|tower| tower.iter().collect())
        .unwrap_or_default()
}

#[test]
fn parse_intro_position() {
    let result = parse_fen(INTRO_POSITION).unwrap();

    assert_eq!(result.turn, Color::White);
    assert_eq!(result.mode, SetupMode::Intro);
    assert_eq!(result.drafting, [false, false]);
    assert_eq!(result.move_number, 1);

    assert_eq!(
        top_piece(&result, 0, 4),
        Some(Piece::new(PieceType::Marshal, Color::Black))
    );
    assert_eq!(
        top_piece(&result, 8, 3),
        Some(Piece::new(PieceType::General, Color::White))
    );
}

#[test]
fn parse_beginner_position() {
    let result = parse_fen(BEGINNER_POSITION).unwrap();

    assert_eq!(result.turn, Color::White);
    assert_eq!(result.mode, SetupMode::Beginner);
    assert_eq!(result.drafting, [false, false]);
    assert_eq!(result.move_number, 1);

    assert_eq!(
        top_piece(&result, 1, 2),
        Some(Piece::new(PieceType::Archer, Color::Black))
    );
    assert_eq!(
        top_piece(&result, 8, 4),
        Some(Piece::new(PieceType::Marshal, Color::White))
    );
}

#[test]
fn parse_intermediate_position() {
    let result = parse_fen(INTERMEDIATE_POSITION).unwrap();

    assert_eq!(result.turn, Color::White);
    assert_eq!(result.mode, SetupMode::Intermediate);
    assert_eq!(result.drafting, [true, true]);
    assert_eq!(result.move_number, 1);

    for r in 0..9 {
        for f in 0..9 {
            assert!(tower_pieces(&result, r, f).is_empty());
        }
    }
}

#[test]
fn parse_advanced_position() {
    let result = parse_fen(ADVANCED_POSITION).unwrap();

    assert_eq!(result.turn, Color::White);
    assert_eq!(result.mode, SetupMode::Advanced);
    assert_eq!(result.drafting, [true, true]);
    assert_eq!(result.move_number, 1);

    for r in 0..9 {
        for f in 0..9 {
            assert!(tower_pieces(&result, r, f).is_empty());
        }
    }
}

#[test]
fn parse_towers_and_empty_squares() {
    let fen = "3img3/1sa|a:g:s|1s3/9/9/9/9/9/9/9 M1N3/d2 b 3 b 2";
    let result = parse_fen(fen).unwrap();

    assert_eq!(result.turn, Color::Black);
    assert_eq!(result.mode, SetupMode::Advanced);
    assert_eq!(result.drafting, [false, true]);
    assert_eq!(result.move_number, 2);

    let tower = tower_pieces(&result, 1, 3);
    assert_eq!(tower.len(), 3);
    assert_eq!(tower[0], Piece::new(PieceType::Archer, Color::Black));
    assert_eq!(tower[1], Piece::new(PieceType::General, Color::Black));
    assert_eq!(tower[2], Piece::new(PieceType::Spy, Color::Black));

    assert!(tower_pieces(&result, 8, 8).is_empty());
}

#[test]
fn parse_adjacent_towers() {
    let fen = "3img3/1sa|a:g:s|d:w|k:n|r:w|2/9/9/9/9/9/9/9 M1N3/d2 w 1 w 3";
    let result = parse_fen(fen).unwrap();

    assert_eq!(result.turn, Color::White);
    assert_eq!(result.mode, SetupMode::Beginner);
    assert_eq!(result.drafting, [true, false]);
    assert_eq!(result.move_number, 3);

    let tower1 = tower_pieces(&result, 1, 3);
    assert_eq!(tower1.len(), 3);
    assert_eq!(tower1[0], Piece::new(PieceType::Archer, Color::Black));
    assert_eq!(tower1[1], Piece::new(PieceType::General, Color::Black));
    assert_eq!(tower1[2], Piece::new(PieceType::Spy, Color::Black));

    let tower2 = tower_pieces(&result, 1, 4);
    assert_eq!(tower2.len(), 2);
    assert_eq!(tower2[0], Piece::new(PieceType::Soldier, Color::Black));
    assert_eq!(tower2[1], Piece::new(PieceType::Warrior, Color::Black));

    let tower3 = tower_pieces(&result, 1, 5);
    assert_eq!(tower3.len(), 2);
    assert_eq!(tower3[0], Piece::new(PieceType::Musketeer, Color::Black));
    assert_eq!(tower3[1], Piece::new(PieceType::Lancer, Color::Black));

    let tower4 = tower_pieces(&result, 1, 6);
    assert_eq!(tower4.len(), 2);
    assert_eq!(tower4[0], Piece::new(PieceType::Rider, Color::Black));
    assert_eq!(tower4[1], Piece::new(PieceType::Warrior, Color::Black));

    assert!(tower_pieces(&result, 0, 0).is_empty());
    assert!(tower_pieces(&result, 8, 8).is_empty());
}

#[test]
fn parse_hand_pieces() {
    let fen =
        "3img3/1sa1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1";
    let result = parse_fen(fen).unwrap();

    assert_eq!(result.hand.len(), 10);
    assert_eq!(
        result.hand[0],
        HandPiece {
            piece_type: PieceType::MajorGeneral,
            color: Color::White,
            count: 2,
        }
    );
    assert_eq!(
        result.hand[1],
        HandPiece {
            piece_type: PieceType::Lancer,
            color: Color::White,
            count: 2,
        }
    );
    assert_eq!(
        result.hand[5],
        HandPiece {
            piece_type: PieceType::MajorGeneral,
            color: Color::Black,
            count: 2,
        }
    );
    assert_eq!(
        result.hand[9],
        HandPiece {
            piece_type: PieceType::Soldier,
            color: Color::Black,
            count: 1,
        }
    );
}

#[test]
fn parse_empty_hands() {
    let fen = "9/9/9/9/9/9/9/9/9 -/- w 0 - 1";
    let result = parse_fen(fen).unwrap();

    assert_eq!(result.hand.len(), 0);
    assert_eq!(result.turn, Color::White);
    assert_eq!(result.mode, SetupMode::Intro);
    assert_eq!(result.drafting, [false, false]);
    assert_eq!(result.move_number, 1);
}

#[test]
fn parse_only_white_empty_hand() {
    let fen = "3img3/9/9/9/9/9/9/9/9 -/m1r3 b 1 - 1";
    let result = parse_fen(fen).unwrap();

    assert_eq!(result.turn, Color::Black);
    assert_eq!(result.mode, SetupMode::Beginner);
    assert_eq!(result.drafting, [false, false]);
    assert_eq!(result.move_number, 1);

    assert_eq!(result.hand.len(), 2);
    assert!(result.hand.contains(&HandPiece {
        piece_type: PieceType::Marshal,
        color: Color::Black,
        count: 1,
    }));
    assert!(result.hand.contains(&HandPiece {
        piece_type: PieceType::Rider,
        color: Color::Black,
        count: 3,
    }));
}

#[test]
fn parse_only_black_empty_hand() {
    let fen = "3img3/9/9/9/9/9/9/9/9 M1D3/- w 1 - 1";
    let result = parse_fen(fen).unwrap();

    assert_eq!(result.turn, Color::White);
    assert_eq!(result.mode, SetupMode::Beginner);
    assert_eq!(result.drafting, [false, false]);
    assert_eq!(result.move_number, 1);

    assert_eq!(result.hand.len(), 2);
    assert!(result.hand.contains(&HandPiece {
        piece_type: PieceType::Marshal,
        color: Color::White,
        count: 1,
    }));
    assert!(result.hand.contains(&HandPiece {
        piece_type: PieceType::Soldier,
        color: Color::White,
        count: 3,
    }));
}

#[test]
fn parse_and_encode_round_trip() {
    let fen =
        "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1";
    let parsed = parse_fen(fen).unwrap();
    let encoded = encode_fen(&parsed);
    assert_eq!(encoded, fen);
}

#[test]
fn validate_correct_fen() {
    assert!(validate_fen(BEGINNER_POSITION).is_ok());
}

#[test]
fn validate_incorrect_number_of_fields() {
    let fen = "3img3/1ra1n1as1/d1fwdwf1d w 1 -";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("expected 6 fields"));
}

#[test]
fn validate_invalid_piece_positions() {
    let fen = "3img3/1ra1n1xas1/d1fwdwf1d/9/9/9/9/9/9 J2N2R1D1/j2n2r2d1 w 1 - 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("invalid piece"));
}

#[test]
fn validate_invalid_rank_count() {
    let fen = "3img3/1ra1n1xas1/d1fwdwf1d/9/9/9/9/9 J2N2R1D1/j2n2r2d1 w 1 - 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("expected 9 ranks"));
}

#[test]
fn validate_invalid_file_count() {
    let fen = "3img3/1ra1n1a1/d1fwdwf1d/9/9/9/9/9/9 J2N2R1D1/j2n2r2d1 w 1 - 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("expected 9 squares"));
}

#[test]
fn validate_invalid_tower_size() {
    let fen = "3img3/1ra1n1as1/d1fw|d:f:g:c|wf1d/9/9/9/9/9/9 J2N2R1D1/j2n2r2d1 w 1 - 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("tower size"));
}

#[test]
fn validate_invalid_hand_pieces() {
    let fen = "9/9/9/9/9/9/9/9/9 J2X2R1D1/j2n2r2d1 w 1 - 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("invalid piece"));
}

#[test]
fn validate_invalid_active_player() {
    let fen = "9/9/9/9/9/9/9/9/9 -/- x 0 - 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("expected 'w' or 'b'"));
}

#[test]
fn validate_invalid_setup_mode() {
    let fen = "9/9/9/9/9/9/9/9/9 -/- w 4 - 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("expected '0', '1', '2', or '3'"));
}

#[test]
fn validate_invalid_drafting_field() {
    let fen = "9/9/9/9/9/9/9/9/9 -/- w 0 x 1";
    let err = validate_fen(fen).unwrap_err();
    assert!(err
        .to_string()
        .contains("(drafting availability) is invalid"));
}

#[test]
fn validate_invalid_move_number() {
    let fen = "9/9/9/9/9/9/9/9/9 -/- w 0 - x";
    let err = validate_fen(fen).unwrap_err();
    assert!(err.to_string().contains("expected an integer"));
}
