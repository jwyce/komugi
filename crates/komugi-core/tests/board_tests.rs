use std::mem::size_of;

use komugi_core::board::{Board, BoardError, Tower};
use komugi_core::types::{Color, Piece, PieceType, SetupMode, Square};
use komugi_core::zobrist::zobrist_keys;

fn sq(rank: u8, file: u8) -> Square {
    Square::new(rank, file).expect("valid square")
}

#[test]
fn tower_and_board_size_are_cache_friendly() {
    println!("Tower size: {} bytes", size_of::<Tower>());
    println!("Board size: {} bytes", size_of::<Board>());
    assert!(
        size_of::<Tower>() <= 8,
        "tower too large: {}",
        size_of::<Tower>()
    );
    assert!(
        size_of::<Board>() <= 700,
        "board too large: {}",
        size_of::<Board>()
    );
}

#[test]
fn board_new_builds_expected_setup_positions() {
    let intro = Board::new(SetupMode::Intro);
    let beginner = Board::new(SetupMode::Beginner);
    let intermediate = Board::new(SetupMode::Intermediate);
    let advanced = Board::new(SetupMode::Advanced);

    assert_eq!(
        intro.get_top(sq(1, 5)),
        Some((Piece::new(PieceType::Marshal, Color::Black), 1))
    );
    assert_eq!(
        intro.get_top(sq(2, 8)),
        Some((Piece::new(PieceType::Spy, Color::Black), 1))
    );
    assert_eq!(
        intro.get_top(sq(9, 5)),
        Some((Piece::new(PieceType::Marshal, Color::White), 1))
    );

    assert_eq!(
        beginner.get_top(sq(2, 8)),
        Some((Piece::new(PieceType::Rider, Color::Black), 1))
    );
    assert_eq!(
        beginner.get_top(sq(8, 2)),
        Some((Piece::new(PieceType::Rider, Color::White), 1))
    );
    assert_eq!(
        beginner.get_top(sq(2, 7)),
        Some((Piece::new(PieceType::Archer, Color::Black), 1))
    );

    assert_eq!(intermediate.get_top(sq(1, 1)), None);
    assert_eq!(advanced.get_top(sq(9, 9)), None);
}

#[test]
fn put_and_remove_top_are_incremental_and_validate_height() {
    let mut board = Board::new(SetupMode::Advanced);
    let square = sq(5, 5);
    let p1 = Piece::new(PieceType::Soldier, Color::White);
    let p2 = Piece::new(PieceType::Spy, Color::White);
    let p3 = Piece::new(PieceType::General, Color::White);

    board.put(p1, square).unwrap();
    board.put(p2, square).unwrap();
    board.put(p3, square).unwrap();
    assert_eq!(board.get(square).expect("tower").height(), 3);

    assert_eq!(board.remove_top(square).unwrap(), Some(p3));
    assert_eq!(board.remove_top(square).unwrap(), Some(p2));
    assert_eq!(board.remove_top(square).unwrap(), Some(p1));
    assert_eq!(board.remove_top(square).unwrap(), None);

    let mut intro = Board::new(SetupMode::Intro);
    let intro_square = sq(4, 4);
    intro
        .put(Piece::new(PieceType::Soldier, Color::Black), intro_square)
        .unwrap();
    intro
        .put(Piece::new(PieceType::Fortress, Color::Black), intro_square)
        .unwrap();
    assert_eq!(
        intro.put(Piece::new(PieceType::Spy, Color::Black), intro_square),
        Err(BoardError::ExceedsModeMaxTier)
    );
}

#[test]
fn remove_and_convert_operate_on_matching_pieces() {
    let mut board = Board::new(SetupMode::Advanced);
    let square = sq(6, 6);
    let soldier = Piece::new(PieceType::Soldier, Color::Black);
    let spy = Piece::new(PieceType::Spy, Color::Black);
    let warrior = Piece::new(PieceType::Warrior, Color::Black);

    board.put(soldier, square).unwrap();
    board.put(spy, square).unwrap();
    board.put(warrior, square).unwrap();

    let converted = board.convert(square, &[spy, warrior]).unwrap();
    assert_eq!(converted, 2);

    let tower = board.get(square).expect("tower");
    assert_eq!(tower.pieces()[0], Some(soldier));
    assert_eq!(
        tower.pieces()[1],
        Some(Piece::new(PieceType::Spy, Color::White))
    );
    assert_eq!(
        tower.pieces()[2],
        Some(Piece::new(PieceType::Warrior, Color::White))
    );

    let removed = board.remove(square, &[soldier]).unwrap();
    assert_eq!(removed, 1);
    assert_eq!(board.get(square).expect("tower").height(), 2);
}

#[test]
fn zobrist_is_deterministic_for_same_position() {
    let keys = zobrist_keys();
    let board_a = Board::new(SetupMode::Beginner);
    let board_b = Board::new(SetupMode::Beginner);

    let hash_a = keys.hash_position(&board_a, &[], Color::White, [false, false]);
    let hash_b = keys.hash_position(&board_b, &[], Color::White, [false, false]);
    assert_eq!(hash_a, hash_b);
}

#[test]
fn zobrist_put_then_remove_restores_hash() {
    let keys = zobrist_keys();
    let mut board = Board::new(SetupMode::Advanced);
    let square = sq(4, 5);
    let piece = Piece::new(PieceType::Musketeer, Color::Black);

    let mut hash = keys.hash_board(&board);
    let baseline = hash;

    board.put(piece, square).unwrap();
    keys.xor_piece(&mut hash, piece, square, 1);
    assert_eq!(hash, keys.hash_board(&board));

    let popped = board.remove_top(square).unwrap();
    assert_eq!(popped, Some(piece));
    keys.xor_piece(&mut hash, piece, square, 1);

    assert_eq!(hash, baseline);
    assert_eq!(hash, keys.hash_board(&board));
}
