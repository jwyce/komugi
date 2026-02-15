use arrayvec::ArrayVec;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const fn to_code(self) -> char {
        match self {
            Self::White => 'w',
            Self::Black => 'b',
        }
    }

    pub const fn from_code(code: char) -> Option<Self> {
        match code {
            'w' => Some(Self::White),
            'b' => Some(Self::Black),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Marshal = 0,
    General = 1,
    LieutenantGeneral = 2,
    MajorGeneral = 3,
    Warrior = 4,
    Lancer = 5,
    Rider = 6,
    Spy = 7,
    Fortress = 8,
    Soldier = 9,
    Cannon = 10,
    Archer = 11,
    Musketeer = 12,
    Tactician = 13,
}

impl PieceType {
    pub const ALL: [Self; 14] = [
        Self::Marshal,
        Self::General,
        Self::LieutenantGeneral,
        Self::MajorGeneral,
        Self::Warrior,
        Self::Lancer,
        Self::Rider,
        Self::Spy,
        Self::Fortress,
        Self::Soldier,
        Self::Cannon,
        Self::Archer,
        Self::Musketeer,
        Self::Tactician,
    ];

    pub const fn kanji(self) -> char {
        match self {
            Self::Marshal => '帥',
            Self::General => '大',
            Self::LieutenantGeneral => '中',
            Self::MajorGeneral => '小',
            Self::Warrior => '侍',
            Self::Lancer => '槍',
            Self::Rider => '馬',
            Self::Spy => '忍',
            Self::Fortress => '砦',
            Self::Soldier => '兵',
            Self::Cannon => '砲',
            Self::Archer => '弓',
            Self::Musketeer => '筒',
            Self::Tactician => '謀',
        }
    }

    pub const fn from_kanji(kanji: char) -> Option<Self> {
        match kanji {
            '帥' => Some(Self::Marshal),
            '大' => Some(Self::General),
            '中' => Some(Self::LieutenantGeneral),
            '小' => Some(Self::MajorGeneral),
            '侍' => Some(Self::Warrior),
            '槍' => Some(Self::Lancer),
            '馬' => Some(Self::Rider),
            '忍' => Some(Self::Spy),
            '砦' => Some(Self::Fortress),
            '兵' => Some(Self::Soldier),
            '砲' => Some(Self::Cannon),
            '弓' => Some(Self::Archer),
            '筒' => Some(Self::Musketeer),
            '謀' => Some(Self::Tactician),
            _ => None,
        }
    }

    pub const fn fen_code(self) -> char {
        match self {
            Self::Marshal => 'm',
            Self::General => 'g',
            Self::LieutenantGeneral => 'i',
            Self::MajorGeneral => 'j',
            Self::Warrior => 'w',
            Self::Lancer => 'n',
            Self::Rider => 'r',
            Self::Spy => 's',
            Self::Fortress => 'f',
            Self::Soldier => 'd',
            Self::Cannon => 'c',
            Self::Archer => 'a',
            Self::Musketeer => 'k',
            Self::Tactician => 't',
        }
    }

    pub const fn from_fen_code(code: char) -> Option<Self> {
        match code {
            'm' => Some(Self::Marshal),
            'g' => Some(Self::General),
            'i' => Some(Self::LieutenantGeneral),
            'j' => Some(Self::MajorGeneral),
            'w' => Some(Self::Warrior),
            'n' => Some(Self::Lancer),
            'r' => Some(Self::Rider),
            's' => Some(Self::Spy),
            'f' => Some(Self::Fortress),
            'd' => Some(Self::Soldier),
            'c' => Some(Self::Cannon),
            'a' => Some(Self::Archer),
            'k' => Some(Self::Musketeer),
            't' => Some(Self::Tactician),
            _ => None,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

impl Piece {
    pub const fn new(piece_type: PieceType, color: Color) -> Self {
        Self { piece_type, color }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square {
    pub rank: u8,
    pub file: u8,
}

impl Square {
    pub const fn new(rank: u8, file: u8) -> Option<Self> {
        if rank >= 1 && rank <= 9 && file >= 1 && file <= 9 {
            Some(Self { rank, file })
        } else {
            None
        }
    }

    pub const fn new_unchecked(rank: u8, file: u8) -> Self {
        Self { rank, file }
    }

    pub fn parse(input: &str) -> Option<Self> {
        let (rank, file) = input.split_once('-')?;
        let rank = rank.parse::<u8>().ok()?;
        let file = file.parse::<u8>().ok()?;
        Self::new(rank, file)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TieredSquare {
    pub square: Square,
    pub tier: u8,
}

impl TieredSquare {
    pub const fn new(square: Square, tier: u8) -> Option<Self> {
        if tier >= 1 && tier <= 3 {
            Some(Self { square, tier })
        } else {
            None
        }
    }

    pub const fn new_unchecked(square: Square, tier: u8) -> Self {
        Self { square, tier }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoveType {
    Route = 0,
    Capture = 1,
    Tsuke = 2,
    Betray = 3,
    Arata = 4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Move {
    pub color: Color,
    pub piece: PieceType,
    pub from: Option<TieredSquare>,
    pub to: TieredSquare,
    pub move_type: MoveType,
    pub draft_finished: bool,
    pub captured: ArrayVec<Piece, 3>,
}

impl Move {
    pub fn new(
        color: Color,
        piece: PieceType,
        from: Option<TieredSquare>,
        to: TieredSquare,
        move_type: MoveType,
    ) -> Self {
        Self {
            color,
            piece,
            from,
            to,
            move_type,
            draft_finished: false,
            captured: ArrayVec::new(),
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetupMode {
    Intro = 0,
    Beginner = 1,
    Intermediate = 2,
    Advanced = 3,
}

impl SetupMode {
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Intro),
            1 => Some(Self::Beginner),
            2 => Some(Self::Intermediate),
            3 => Some(Self::Advanced),
            _ => None,
        }
    }

    pub const fn to_code(self) -> u8 {
        self as u8
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandPiece {
    pub piece_type: PieceType,
    pub color: Color,
    pub count: u8,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Score(pub i32);

pub type MoveList = ArrayVec<Move, 1024>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_code_kanji_round_trip() {
        for piece in PieceType::ALL {
            let code = piece.fen_code();
            let kanji = piece.kanji();

            assert_eq!(PieceType::from_fen_code(code), Some(piece));
            assert_eq!(PieceType::from_kanji(kanji), Some(piece));
            assert_eq!(
                PieceType::from_fen_code(code).map(PieceType::kanji),
                Some(kanji)
            );
            assert_eq!(
                PieceType::from_kanji(kanji).map(PieceType::fen_code),
                Some(code)
            );
        }
    }

    #[test]
    fn parse_square() {
        assert_eq!(Square::parse("1-9"), Some(Square::new_unchecked(1, 9)));
        assert_eq!(Square::parse("9-1"), Some(Square::new_unchecked(9, 1)));
        assert_eq!(Square::parse("0-1"), None);
        assert_eq!(Square::parse("1-10"), None);
        assert_eq!(Square::parse("bad"), None);
    }

    #[test]
    fn setup_mode_code_conversion() {
        assert_eq!(SetupMode::from_code(0), Some(SetupMode::Intro));
        assert_eq!(SetupMode::from_code(1), Some(SetupMode::Beginner));
        assert_eq!(SetupMode::from_code(2), Some(SetupMode::Intermediate));
        assert_eq!(SetupMode::from_code(3), Some(SetupMode::Advanced));
        assert_eq!(SetupMode::from_code(4), None);
        assert_eq!(SetupMode::Advanced.to_code(), 3);
    }

    #[test]
    fn piece_is_two_bytes() {
        assert_eq!(core::mem::size_of::<Piece>(), 2);
    }
}
