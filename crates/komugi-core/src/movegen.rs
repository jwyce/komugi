use crate::board::{Board, Tower};
use crate::constants::{max_tier_for_mode, SQUARES};
use crate::fen::ParsedFen;
use crate::position::Position;
use crate::types::{
    Color, HandPiece, Move, MoveList, MoveType, Piece, PieceType, SetupMode, Square, TieredSquare,
};

pub const DIRS: [(i8, i8); 8] = [
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, 0),
    (1, -1),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    None,
    Finite { start: u8, carry: u8 },
    Infinite,
}

pub const PIECE_PROBES: [[Probe; 8]; 14] = [
    [
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
    ],
    [
        Probe::Finite { start: 1, carry: 1 },
        Probe::Infinite,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Infinite,
        Probe::Infinite,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Infinite,
        Probe::Finite { start: 1, carry: 1 },
    ],
    [
        Probe::Infinite,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Infinite,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Infinite,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Infinite,
    ],
    [
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
    ],
    [
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::None,
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
    ],
    [
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 2 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::None,
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
    ],
    [
        Probe::None,
        Probe::Finite { start: 1, carry: 2 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 2 },
        Probe::None,
    ],
    [
        Probe::Finite { start: 1, carry: 2 },
        Probe::None,
        Probe::Finite { start: 1, carry: 2 },
        Probe::None,
        Probe::None,
        Probe::Finite { start: 1, carry: 2 },
        Probe::None,
        Probe::Finite { start: 1, carry: 2 },
    ],
    [
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
    ],
    [
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::None,
        Probe::None,
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
    ],
    [
        Probe::None,
        Probe::Finite { start: 3, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
    ],
    [
        Probe::Finite { start: 2, carry: 1 },
        Probe::Finite { start: 2, carry: 1 },
        Probe::Finite { start: 2, carry: 1 },
        Probe::None,
        Probe::None,
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
    ],
    [
        Probe::None,
        Probe::Finite { start: 2, carry: 1 },
        Probe::None,
        Probe::None,
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
    ],
    [
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
        Probe::None,
        Probe::None,
        Probe::Finite { start: 1, carry: 1 },
        Probe::None,
    ],
];

pub trait MoveGenerator {
    fn generate_moves(&self, _position: &Position) -> MoveList {
        MoveList::new()
    }
}

pub fn generate_all_legal_moves(board: &Board, turn: Color) -> MoveList {
    let mut cloned = board.clone();
    generate_all_moves_with_mode(&mut cloned, turn, board.mode(), None, true)
}

pub fn generate_all_legal_moves_in_state(state: &ParsedFen) -> MoveList {
    let mut board = state.board.clone();
    let marshal_square = find_marshal_square(&board, state.turn);
    let drafting = state.drafting[state.turn as usize];

    let mut moves = if drafting {
        MoveList::new()
    } else {
        generate_all_moves_with_mode(&mut board, state.turn, state.mode, Some(&state.hand), true)
    };

    for hand_piece in state
        .hand
        .iter()
        .copied()
        .filter(|hp| hp.color == state.turn && hp.count > 0)
    {
        let ctx = ArataContext {
            hand: &state.hand,
            turn: state.turn,
            mode: state.mode,
            drafting_rights: state.drafting,
            marshal_square,
            enforce_legality: !drafting,
        };
        append_arata_moves(&mut moves, &mut board, &ctx, hand_piece);
    }
    moves
}

pub fn generate_all_pseudo_legal_moves_in_state(state: &ParsedFen) -> MoveList {
    let mut board = state.board.clone();
    let marshal_square = find_marshal_square(&board, state.turn);
    let mut moves =
        generate_all_moves_with_mode(&mut board, state.turn, state.mode, Some(&state.hand), false);
    if state.drafting[state.turn as usize] {
        moves.clear();
    }
    for hand_piece in state
        .hand
        .iter()
        .copied()
        .filter(|hp| hp.color == state.turn && hp.count > 0)
    {
        let ctx = ArataContext {
            hand: &state.hand,
            turn: state.turn,
            mode: state.mode,
            drafting_rights: state.drafting,
            marshal_square,
            enforce_legality: false,
        };
        append_arata_moves(&mut moves, &mut board, &ctx, hand_piece);
    }
    moves
}

pub fn generate_all_legal_moves_from_position(position: &Position) -> MoveList {
    let mut board = position.board.clone();
    let marshal_square = position.marshal_squares[position.turn as usize];
    let drafting = position.drafting_rights[position.turn as usize];

    let mut moves = if drafting {
        MoveList::new()
    } else {
        generate_all_moves_with_mode(
            &mut board,
            position.turn,
            position.mode,
            Some(&position.hand),
            true,
        )
    };

    for hand_piece in position
        .hand
        .iter()
        .copied()
        .filter(|hp| hp.color == position.turn && hp.count > 0)
    {
        let ctx = ArataContext {
            hand: &position.hand,
            turn: position.turn,
            mode: position.mode,
            drafting_rights: position.drafting_rights,
            marshal_square,
            enforce_legality: !drafting,
        };
        append_arata_moves(&mut moves, &mut board, &ctx, hand_piece);
    }
    moves
}

pub fn generate_all_pseudo_legal_moves_from_position(position: &Position) -> MoveList {
    let mut board = position.board.clone();
    let marshal_square = position.marshal_squares[position.turn as usize];
    let mut moves = generate_all_moves_with_mode(
        &mut board,
        position.turn,
        position.mode,
        Some(&position.hand),
        false,
    );
    if position.drafting_rights[position.turn as usize] {
        moves.clear();
    }
    for hand_piece in position
        .hand
        .iter()
        .copied()
        .filter(|hp| hp.color == position.turn && hp.count > 0)
    {
        let ctx = ArataContext {
            hand: &position.hand,
            turn: position.turn,
            mode: position.mode,
            drafting_rights: position.drafting_rights,
            marshal_square,
            enforce_legality: false,
        };
        append_arata_moves(&mut moves, &mut board, &ctx, hand_piece);
    }
    moves
}

pub fn is_square_attacked(board: &Board, square: Square, by_color: Color) -> bool {
    for src in SQUARES {
        let Some((piece, _)) = board.get_top(src) else {
            continue;
        };
        if piece.color != by_color {
            continue;
        }
        let attacks = attacked_squares_for_piece(board, src, piece);
        if attacks.iter().any(|&sq| sq == square) {
            return true;
        }
    }
    false
}

pub fn in_check(board: &Board, turn: Color) -> bool {
    in_check_with_marshal(board, turn, find_marshal_square(board, turn))
}

pub fn in_check_with_marshal(board: &Board, turn: Color, marshal_square: Option<Square>) -> bool {
    let by_color = opposite(turn);
    marshal_square
        .map(|sq| is_square_attacked(board, sq, by_color))
        .unwrap_or(false)
}

pub fn generate_moves_for_square(state: &ParsedFen, square: Square) -> MoveList {
    let mut all = generate_all_legal_moves_in_state(state);
    let mut out = MoveList::new();
    while let Some(mv) = all.pop() {
        if mv.from.is_some_and(|from| from.square == square) {
            let _ = out.try_push(mv);
        }
    }
    out
}

pub fn generate_arata(state: &ParsedFen, hand_piece: HandPiece) -> MoveList {
    let mut out = MoveList::new();
    let mut board = state.board.clone();
    let marshal_square = find_marshal_square(&board, state.turn);
    append_arata_moves(
        &mut out,
        &mut board,
        &ArataContext {
            hand: &state.hand,
            turn: state.turn,
            mode: state.mode,
            drafting_rights: state.drafting,
            marshal_square,
            enforce_legality: true,
        },
        hand_piece,
    );
    out
}

fn generate_all_moves_with_mode(
    board: &mut Board,
    turn: Color,
    mode: SetupMode,
    hand: Option<&[HandPiece]>,
    enforce_legality: bool,
) -> MoveList {
    let max_tier = max_tier_for_mode(mode);
    let marshal_can_stack = matches!(mode, SetupMode::Advanced | SetupMode::Intermediate);
    let marshal_square = find_marshal_square(board, turn);

    let mut legal = MoveList::new();
    for from_square in SQUARES {
        let Some((piece, tier)) = board.get_top(from_square) else {
            continue;
        };
        if piece.color != turn {
            continue;
        }

        let from = TieredSquare::new_unchecked(from_square, tier);
        let targets = attacked_squares_for_piece(board, from_square, piece);
        for target_square in targets {
            let maybe_top = board.get(target_square).and_then(|tower| tower.get_top());

            if maybe_top.is_none() {
                let mv = Move::new(
                    turn,
                    piece.piece_type,
                    Some(from),
                    TieredSquare::new_unchecked(target_square, 1),
                    MoveType::Route,
                );
                if !enforce_legality || is_legal_after_move(board, turn, &mv, marshal_square) {
                    let _ = legal.try_push(mv);
                }
                continue;
            }

            let (top_piece, top_tier) = maybe_top.expect("checked is_some");

            // Tsuke — generated for ANY top piece (friendly or enemy),
            // as long as tier < maxTier and top is not Marshal.
            // Matches gungi.js move_gen.ts line 141.
            if top_tier < max_tier
                && top_piece.piece_type != PieceType::Marshal
                && (piece.piece_type != PieceType::Marshal || marshal_can_stack)
            {
                let mv = Move::new(
                    turn,
                    piece.piece_type,
                    Some(from),
                    TieredSquare::new_unchecked(target_square, top_tier + 1),
                    MoveType::Tsuke,
                );
                if !enforce_legality || is_legal_after_move(board, turn, &mv, marshal_square) {
                    let _ = legal.try_push(mv);
                }

                // Betrayal — only for enemy tops, inside tsuke block so it
                // inherits the "not Marshal" check (BUG #3 fix).
                // Matches gungi.js move_gen.ts line 147.
                if piece.piece_type == PieceType::Tactician && top_piece.color != turn {
                    if let Some(tower) = board.get(target_square) {
                        let mut enemies = arrayvec::ArrayVec::<Piece, 3>::new();
                        for p in tower.iter().filter(|p| p.color != turn) {
                            let _ = enemies.try_push(p);
                        }

                        if !enemies.is_empty() {
                            let mut enemy_count = [0u8; 14];
                            for e in &enemies {
                                enemy_count[e.piece_type as usize] += 1;
                            }

                            let mut hand_count = [0u8; 14];
                            if let Some(hand) = hand {
                                for hp in hand.iter().filter(|h| h.color == turn) {
                                    hand_count[hp.piece_type as usize] = hp.count;
                                }
                            }

                            let mut betrayal_options = arrayvec::ArrayVec::<Piece, 3>::new();
                            for e in &enemies {
                                let needed = enemy_count[e.piece_type as usize];
                                let have = hand_count[e.piece_type as usize];
                                if have >= needed {
                                    let _ = betrayal_options.try_push(*e);
                                }
                            }

                            for combo in combinations(&betrayal_options) {
                                let mut mv = Move::new(
                                    turn,
                                    piece.piece_type,
                                    Some(from),
                                    TieredSquare::new_unchecked(target_square, top_tier + 1),
                                    MoveType::Betray,
                                );
                                for p in combo {
                                    let _ = mv.captured.try_push(p);
                                }
                                if !enforce_legality
                                    || is_legal_after_move(board, turn, &mv, marshal_square)
                                {
                                    let _ = legal.try_push(mv);
                                }
                            }
                        }
                    }
                }
            }

            // Capture — only for enemy tops.
            if top_piece.color != turn {
                let mut mv = Move::new(
                    turn,
                    piece.piece_type,
                    Some(from),
                    TieredSquare::new_unchecked(
                        target_square,
                        capture_tier(board, target_square, turn),
                    ),
                    MoveType::Capture,
                );
                if let Some(tower) = board.get(target_square) {
                    for captured in tower.iter().filter(|p| p.color != turn) {
                        let _ = mv.captured.try_push(captured);
                    }
                }
                if !enforce_legality || is_legal_after_move(board, turn, &mv, marshal_square) {
                    let _ = legal.try_push(mv);
                }
            }
        }
    }

    legal
}

fn append_arata_moves(
    all: &mut MoveList,
    board: &mut Board,
    ctx: &ArataContext<'_>,
    hand_piece: HandPiece,
) {
    if hand_piece.color != ctx.turn || hand_piece.count == 0 {
        return;
    }

    let marshal_is_placed = !ctx
        .hand
        .iter()
        .any(|h| h.color == hand_piece.color && h.piece_type == PieceType::Marshal && h.count > 0);
    if !marshal_is_placed && hand_piece.piece_type != PieceType::Marshal {
        return;
    }

    let max_tier = max_tier_for_mode(ctx.mode);
    let mut ranks = ArrayVecRanks::new();
    if ctx.drafting_rights[0] || ctx.drafting_rights[1] {
        if hand_piece.color == Color::White {
            ranks.extend([7, 8, 9]);
        } else {
            ranks.extend([1, 2, 3]);
        }
    } else {
        collect_deepest_ranks(board, hand_piece.color, &mut ranks);
    }

    let player_hand_count: u32 = ctx
        .hand
        .iter()
        .filter(|h| h.color == hand_piece.color)
        .map(|h| u32::from(h.count))
        .sum();
    let is_last_piece = player_hand_count == 1;
    let color_drafting = ctx.drafting_rights[usize::from(hand_piece.color as u8)];

    for rank in ranks.iter().copied() {
        for file in 1..=9 {
            let target = Square::new_unchecked(rank, file);
            let top = board.get_top(target);
            if let Some((piece, tier)) = top {
                if piece.color != hand_piece.color
                    || tier >= max_tier
                    || piece.piece_type == PieceType::Marshal
                {
                    continue;
                }
            }

            let next_tier = top.map(|(_, tier)| tier + 1).unwrap_or(1);

            if color_drafting {
                if !is_last_piece {
                    let mv = Move::new(
                        hand_piece.color,
                        hand_piece.piece_type,
                        None,
                        TieredSquare::new_unchecked(target, next_tier),
                        MoveType::Arata,
                    );
                    if !ctx.enforce_legality
                        || is_legal_after_move(board, hand_piece.color, &mv, ctx.marshal_square)
                    {
                        let _ = all.try_push(mv);
                    }
                }

                let mut mv = Move::new(
                    hand_piece.color,
                    hand_piece.piece_type,
                    None,
                    TieredSquare::new_unchecked(target, next_tier),
                    MoveType::Arata,
                );
                mv.draft_finished = true;
                if !ctx.enforce_legality
                    || is_legal_after_move(board, hand_piece.color, &mv, ctx.marshal_square)
                {
                    let _ = all.try_push(mv);
                }
            } else {
                let mv = Move::new(
                    hand_piece.color,
                    hand_piece.piece_type,
                    None,
                    TieredSquare::new_unchecked(target, next_tier),
                    MoveType::Arata,
                );
                if !ctx.enforce_legality
                    || is_legal_after_move(board, hand_piece.color, &mv, ctx.marshal_square)
                {
                    let _ = all.try_push(mv);
                }
            }
        }
    }
}

fn collect_deepest_ranks(board: &Board, color: Color, out: &mut ArrayVecRanks) {
    let mut maybe = ArrayVecRanks::new();
    let (start, end, step): (i8, i8, i8) = if color == Color::Black {
        (1, 9, 1)
    } else {
        (9, 1, -1)
    };
    let mut rank = start;
    while if color == Color::Black {
        rank <= end
    } else {
        rank >= end
    } {
        let mut has_piece = false;
        for file in 1..=9 {
            let square = Square::new_unchecked(rank as u8, file);
            if let Some((piece, _)) = board.get_top(square) {
                if piece.color == color {
                    has_piece = true;
                    break;
                }
            }
        }

        if has_piece {
            for candidate in maybe.iter().copied() {
                out.push(candidate);
            }
            maybe.clear();
            out.push(rank as u8);
        } else {
            maybe.push(rank as u8);
        }

        rank += step;
    }
}

fn attacked_squares_for_piece(board: &Board, origin: Square, piece: Piece) -> ArrayVecSquares {
    let mut result = ArrayVecSquares::new();
    let origin_tier = board.get_top(origin).map(|(_, tier)| tier).unwrap_or(1);

    let probes = PIECE_PROBES[piece.piece_type as usize];
    for (idx, probe) in probes.iter().copied().enumerate() {
        let (start, length) = match probe {
            Probe::None => continue,
            Probe::Infinite => (
                (origin.rank as i8 + adjusted_dir(piece.color, DIRS[idx]).0),
                ProbeLength::Infinite,
            ),
            Probe::Finite {
                start: probe_start,
                carry,
            } => (
                origin.rank as i8 + adjusted_dir(piece.color, DIRS[idx]).0 * probe_start as i8,
                ProbeLength::Finite(origin_tier + carry - 1),
            ),
        };

        let (dy, dx) = adjusted_dir(piece.color, DIRS[idx]);
        let start_file = origin.file as i8 + dx;

        let mut squares = get_available_squares(
            (dy, dx),
            (start, start_file),
            origin,
            piece,
            origin_tier,
            length,
            board,
        );
        for sq in squares.drain(..) {
            result.push(sq);
        }
    }

    result
}

fn get_available_squares(
    dir: (i8, i8),
    start: (i8, i8),
    origin: Square,
    origin_piece: Piece,
    origin_tier: u8,
    length: ProbeLength,
    board: &Board,
) -> ArrayVecSquares {
    let (dy, dx) = dir;
    let (origin_rank, origin_file) = (origin.rank as i8, origin.file as i8);
    let mut available = ArrayVecSquares::new();

    let mut reverse_rank = start.0;
    let mut reverse_file = start.1;
    while reverse_rank != origin_rank || reverse_file != origin_file {
        reverse_rank -= dy;
        reverse_file -= dx;

        if let Some((piece, tier)) = get_top_coords(board, reverse_rank, reverse_file) {
            if piece.color != origin_piece.color && tier > origin_tier {
                return available;
            }
            if piece.color == origin_piece.color && tier > origin_tier {
                return available;
            }
        }

        let side = if origin_piece.color == Color::Black {
            -1
        } else {
            1
        };
        let below_rank = reverse_rank + side;
        if below_rank == origin_rank
            && reverse_file == origin_file
            && get_top_coords(board, below_rank, reverse_file).is_some()
        {
            break;
        }
    }

    let mut forward_rank = start.0;
    let mut forward_file = start.1;
    let mut step = 0u8;
    loop {
        if !(1..=9).contains(&forward_rank) || !(1..=9).contains(&forward_file) {
            break;
        }

        if let Some((_, tier)) = get_top_coords(board, forward_rank, forward_file) {
            if tier > origin_tier {
                break;
            }
        }

        available.push(Square::new_unchecked(
            forward_rank as u8,
            forward_file as u8,
        ));

        if let Some((piece, _)) = get_top_coords(board, forward_rank, forward_file) {
            if !is_leap_piece(origin_piece.piece_type) {
                let _ = piece;
                break;
            }
        }

        forward_rank += dy;
        forward_file += dx;
        step = step.saturating_add(1);

        if matches!(length, ProbeLength::Finite(max) if step >= max) {
            break;
        }
    }

    available
}

fn combinations(items: &[Piece]) -> arrayvec::ArrayVec<arrayvec::ArrayVec<Piece, 3>, 7> {
    let mut out = arrayvec::ArrayVec::<arrayvec::ArrayVec<Piece, 3>, 7>::new();
    let n = items.len();
    for mask in 1..(1usize << n) {
        let mut subset = arrayvec::ArrayVec::<Piece, 3>::new();
        for (i, item) in items.iter().enumerate() {
            if (mask & (1usize << i)) != 0 {
                let _ = subset.try_push(*item);
            }
        }
        let _ = out.try_push(subset);
    }
    out
}

fn is_legal_after_move(
    board: &mut Board,
    turn: Color,
    mv: &Move,
    marshal_square: Option<Square>,
) -> bool {
    let undo = apply_move_on_board(board, mv);
    let next_marshal_square = if mv.piece == PieceType::Marshal && mv.color == turn {
        Some(mv.to.square)
    } else {
        marshal_square
    };
    let legal = !in_check_with_marshal(board, turn, next_marshal_square);
    undo_move_on_board(board, undo);
    legal
}

fn apply_move_on_board(board: &mut Board, mv: &Move) -> MoveUndo {
    let from_tower = mv
        .from
        .map(|from| (from.square, snapshot_tower(board, from.square)));
    let to_tower = (mv.to.square, snapshot_tower(board, mv.to.square));

    match mv.move_type {
        MoveType::Route | MoveType::Tsuke => {
            if let Some(from) = mv.from {
                let _ = board.remove_top(from.square);
            }
        }
        MoveType::Capture => {
            if let Some(from) = mv.from {
                let _ = board.remove_top(from.square);
            }
            let _ = board.remove(mv.to.square, &mv.captured);
        }
        MoveType::Betray => {
            if let Some(from) = mv.from {
                let _ = board.remove_top(from.square);
            }
            let _ = board.convert(mv.to.square, &mv.captured);
        }
        MoveType::Arata => {}
    }

    let _ = board.put(
        Piece {
            piece_type: mv.piece,
            color: mv.color,
        },
        mv.to.square,
    );

    MoveUndo {
        from_tower,
        to_tower,
    }
}

fn undo_move_on_board(board: &mut Board, undo: MoveUndo) {
    restore_tower(board, undo.to_tower.0, undo.to_tower.1);
    if let Some((square, tower)) = undo.from_tower {
        if square != undo.to_tower.0 {
            restore_tower(board, square, tower);
        }
    }
}

fn snapshot_tower(board: &Board, square: Square) -> Tower {
    board
        .tower_copy(square)
        .expect("generated move square must be in bounds")
}

fn restore_tower(board: &mut Board, square: Square, tower: Tower) {
    board
        .set_tower(square, tower)
        .expect("generated move square must be in bounds");
}

fn find_marshal_square(board: &Board, color: Color) -> Option<Square> {
    for sq in SQUARES {
        let Some((piece, _)) = board.get_top(sq) else {
            continue;
        };
        if piece.piece_type == PieceType::Marshal && piece.color == color {
            return Some(sq);
        }
    }
    None
}

fn capture_tier(board: &Board, target: Square, color: Color) -> u8 {
    let Some(tower) = board.get(target) else {
        return 1;
    };
    let friendly = tower.iter().filter(|piece| piece.color == color).count() as u8;
    friendly + 1
}

fn adjusted_dir(color: Color, dir: (i8, i8)) -> (i8, i8) {
    if color == Color::Black {
        (-dir.0, -dir.1)
    } else {
        dir
    }
}

fn get_top_coords(board: &Board, rank: i8, file: i8) -> Option<(Piece, u8)> {
    if !(1..=9).contains(&rank) || !(1..=9).contains(&file) {
        return None;
    }
    board.get_top(Square::new_unchecked(rank as u8, file as u8))
}

fn is_leap_piece(piece_type: PieceType) -> bool {
    matches!(
        piece_type,
        PieceType::Cannon | PieceType::Archer | PieceType::Musketeer
    )
}

fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeLength {
    Finite(u8),
    Infinite,
}

type ArrayVecSquares = arrayvec::ArrayVec<Square, 81>;
type ArrayVecRanks = arrayvec::ArrayVec<u8, 9>;

#[derive(Debug, Clone, Copy)]
struct MoveUndo {
    from_tower: Option<(Square, Tower)>,
    to_tower: (Square, Tower),
}

struct ArataContext<'a> {
    hand: &'a [HandPiece],
    turn: Color,
    mode: SetupMode,
    drafting_rights: [bool; 2],
    marshal_square: Option<Square>,
    enforce_legality: bool,
}
