use crate::types::{MoveType, PieceType, SetupMode, Square};

pub const WHITE: char = 'w';
pub const BLACK: char = 'b';

pub const SAN_TSUKE: char = '付';
pub const SAN_TAKE: char = '取';
pub const SAN_BETRAY: char = '返';
pub const SAN_ARATA: char = '新';

pub const PIECE_CODES: [char; 14] = [
    'm', 'g', 'i', 'j', 'w', 'n', 'r', 's', 'f', 'd', 'c', 'a', 'k', 't',
];

pub const PIECE_KANJI: [char; 14] = [
    '帥', '大', '中', '小', '侍', '槍', '馬', '忍', '砦', '兵', '砲', '弓', '筒', '謀',
];

pub const PIECE_CODE_TO_KANJI: [(char, char); 14] = [
    ('m', '帥'),
    ('g', '大'),
    ('i', '中'),
    ('j', '小'),
    ('w', '侍'),
    ('n', '槍'),
    ('r', '馬'),
    ('s', '忍'),
    ('f', '砦'),
    ('d', '兵'),
    ('c', '砲'),
    ('a', '弓'),
    ('k', '筒'),
    ('t', '謀'),
];

pub const PIECE_TYPE_TO_CODE: [(PieceType, char); 14] = [
    (PieceType::Marshal, 'm'),
    (PieceType::General, 'g'),
    (PieceType::LieutenantGeneral, 'i'),
    (PieceType::MajorGeneral, 'j'),
    (PieceType::Warrior, 'w'),
    (PieceType::Lancer, 'n'),
    (PieceType::Rider, 'r'),
    (PieceType::Spy, 's'),
    (PieceType::Fortress, 'f'),
    (PieceType::Soldier, 'd'),
    (PieceType::Cannon, 'c'),
    (PieceType::Archer, 'a'),
    (PieceType::Musketeer, 'k'),
    (PieceType::Tactician, 't'),
];

pub const PIECE_TYPE_TO_KANJI: [(PieceType, char); 14] = [
    (PieceType::Marshal, '帥'),
    (PieceType::General, '大'),
    (PieceType::LieutenantGeneral, '中'),
    (PieceType::MajorGeneral, '小'),
    (PieceType::Warrior, '侍'),
    (PieceType::Lancer, '槍'),
    (PieceType::Rider, '馬'),
    (PieceType::Spy, '忍'),
    (PieceType::Fortress, '砦'),
    (PieceType::Soldier, '兵'),
    (PieceType::Cannon, '砲'),
    (PieceType::Archer, '弓'),
    (PieceType::Musketeer, '筒'),
    (PieceType::Tactician, '謀'),
];

pub const MAX_TIER_BY_SETUP_MODE: [u8; 4] = [2, 2, 2, 3];

pub const fn max_tier_for_mode(mode: SetupMode) -> u8 {
    MAX_TIER_BY_SETUP_MODE[mode as usize]
}

pub const SAN_SYMBOLS_BY_MOVE_TYPE: [Option<char>; 5] = [
    None,
    Some(SAN_TAKE),
    Some(SAN_TSUKE),
    Some(SAN_BETRAY),
    Some(SAN_ARATA),
];

pub const fn san_symbol_for_move_type(move_type: MoveType) -> Option<char> {
    SAN_SYMBOLS_BY_MOVE_TYPE[move_type as usize]
}

pub const SQUARES: [Square; 81] = [
    Square::new_unchecked(1, 9),
    Square::new_unchecked(1, 8),
    Square::new_unchecked(1, 7),
    Square::new_unchecked(1, 6),
    Square::new_unchecked(1, 5),
    Square::new_unchecked(1, 4),
    Square::new_unchecked(1, 3),
    Square::new_unchecked(1, 2),
    Square::new_unchecked(1, 1),
    Square::new_unchecked(2, 9),
    Square::new_unchecked(2, 8),
    Square::new_unchecked(2, 7),
    Square::new_unchecked(2, 6),
    Square::new_unchecked(2, 5),
    Square::new_unchecked(2, 4),
    Square::new_unchecked(2, 3),
    Square::new_unchecked(2, 2),
    Square::new_unchecked(2, 1),
    Square::new_unchecked(3, 9),
    Square::new_unchecked(3, 8),
    Square::new_unchecked(3, 7),
    Square::new_unchecked(3, 6),
    Square::new_unchecked(3, 5),
    Square::new_unchecked(3, 4),
    Square::new_unchecked(3, 3),
    Square::new_unchecked(3, 2),
    Square::new_unchecked(3, 1),
    Square::new_unchecked(4, 9),
    Square::new_unchecked(4, 8),
    Square::new_unchecked(4, 7),
    Square::new_unchecked(4, 6),
    Square::new_unchecked(4, 5),
    Square::new_unchecked(4, 4),
    Square::new_unchecked(4, 3),
    Square::new_unchecked(4, 2),
    Square::new_unchecked(4, 1),
    Square::new_unchecked(5, 9),
    Square::new_unchecked(5, 8),
    Square::new_unchecked(5, 7),
    Square::new_unchecked(5, 6),
    Square::new_unchecked(5, 5),
    Square::new_unchecked(5, 4),
    Square::new_unchecked(5, 3),
    Square::new_unchecked(5, 2),
    Square::new_unchecked(5, 1),
    Square::new_unchecked(6, 9),
    Square::new_unchecked(6, 8),
    Square::new_unchecked(6, 7),
    Square::new_unchecked(6, 6),
    Square::new_unchecked(6, 5),
    Square::new_unchecked(6, 4),
    Square::new_unchecked(6, 3),
    Square::new_unchecked(6, 2),
    Square::new_unchecked(6, 1),
    Square::new_unchecked(7, 9),
    Square::new_unchecked(7, 8),
    Square::new_unchecked(7, 7),
    Square::new_unchecked(7, 6),
    Square::new_unchecked(7, 5),
    Square::new_unchecked(7, 4),
    Square::new_unchecked(7, 3),
    Square::new_unchecked(7, 2),
    Square::new_unchecked(7, 1),
    Square::new_unchecked(8, 9),
    Square::new_unchecked(8, 8),
    Square::new_unchecked(8, 7),
    Square::new_unchecked(8, 6),
    Square::new_unchecked(8, 5),
    Square::new_unchecked(8, 4),
    Square::new_unchecked(8, 3),
    Square::new_unchecked(8, 2),
    Square::new_unchecked(8, 1),
    Square::new_unchecked(9, 9),
    Square::new_unchecked(9, 8),
    Square::new_unchecked(9, 7),
    Square::new_unchecked(9, 6),
    Square::new_unchecked(9, 5),
    Square::new_unchecked(9, 4),
    Square::new_unchecked(9, 3),
    Square::new_unchecked(9, 2),
    Square::new_unchecked(9, 1),
];
