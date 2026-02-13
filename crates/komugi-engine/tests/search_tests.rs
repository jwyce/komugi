use komugi_core::{Color, MoveType, Position, SearchLimits, Searcher, SetupMode, SQUARES};
use komugi_engine::AlphaBetaSearcher;

#[test]
fn search_returns_legal_move() {
    let position = Position::new(SetupMode::Beginner);
    let mut searcher = AlphaBetaSearcher::default();

    let result = searcher.search_with_info(
        &position,
        SearchLimits {
            depth: Some(2),
            ..SearchLimits::default()
        },
    );

    let best_move = result.best_move.expect("search should return move");
    assert!(position.moves().iter().any(|mv| mv == &best_move));
    assert!(result.depth >= 1);
    assert!(result.nodes > 0);
}

#[test]
fn search_finds_winning_capture() {
    let mut position = Position::from_fen("8m/9/9/9/4Gd3/9/9/9/M8 -/- w 3 - 1").unwrap();
    let mut searcher = AlphaBetaSearcher::default();
    let before_black_pieces = count_pieces(&position, Color::Black);

    let result = searcher.search_with_info(
        &position,
        SearchLimits {
            depth: Some(2),
            ..SearchLimits::default()
        },
    );

    let best_move = result.best_move.expect("search should find winning move");
    assert!(matches!(
        best_move.move_type,
        MoveType::Capture | MoveType::Betray
    ));

    position.make_move(&best_move).unwrap();
    let after_black_pieces = count_pieces(&position, Color::Black);
    assert!(after_black_pieces < before_black_pieces);
}

#[test]
fn search_recognizes_checkmated_side() {
    let position = Position::from_fen(
        "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3 J2N2S1R1D1/j2n2s1r1d1 b 1 - 3",
    )
    .unwrap();
    let mut searcher = AlphaBetaSearcher::default();

    let result = searcher.search_with_info(
        &position,
        SearchLimits {
            depth: Some(3),
            ..SearchLimits::default()
        },
    );

    assert!(result.best_move.is_none());
    assert!(result.score.0 <= -29_000);
}

#[test]
fn searcher_trait_returns_result() {
    let position = Position::new(SetupMode::Beginner);
    let mut searcher = AlphaBetaSearcher::default();

    let result = Searcher::search(
        &mut searcher,
        &position,
        SearchLimits {
            depth: Some(1),
            ..SearchLimits::default()
        },
    );

    assert!(result.best_move.is_some());
    assert!(result.nodes_searched > 0);
}

fn count_pieces(position: &Position, color: Color) -> usize {
    let mut pieces = 0usize;
    for square in SQUARES {
        if let Some(tower) = position.board.get(square) {
            pieces += tower.iter().filter(|piece| piece.color == color).count();
        }
    }
    pieces
}

#[test]
fn baseline_depth_and_nps() {
    use std::time::Instant;

    let position = Position::new(SetupMode::Beginner);

    // Measure move count
    let moves = position.moves();
    eprintln!("Beginner opening: {} legal moves", moves.len());

    // Depth 1
    let mut searcher = AlphaBetaSearcher::default();
    let start = Instant::now();
    let r1 = searcher.search_with_info(
        &position,
        SearchLimits {
            depth: Some(1),
            ..Default::default()
        },
    );
    let t1 = start.elapsed();
    eprintln!(
        "Depth 1: {} nodes in {:.3}s ({:.0} NPS)",
        r1.nodes,
        t1.as_secs_f64(),
        r1.nodes as f64 / t1.as_secs_f64()
    );

    // Depth 2
    let mut searcher = AlphaBetaSearcher::default();
    let start = Instant::now();
    let r2 = searcher.search_with_info(
        &position,
        SearchLimits {
            depth: Some(2),
            ..Default::default()
        },
    );
    let t2 = start.elapsed();
    eprintln!(
        "Depth 2: {} nodes in {:.3}s ({:.0} NPS)",
        r2.nodes,
        t2.as_secs_f64(),
        r2.nodes as f64 / t2.as_secs_f64()
    );

    // Depth 3
    let mut searcher = AlphaBetaSearcher::default();
    let start = Instant::now();
    let r3 = searcher.search_with_info(
        &position,
        SearchLimits {
            depth: Some(3),
            ..Default::default()
        },
    );
    let t3 = start.elapsed();
    eprintln!(
        "Depth 3: {} nodes in {:.3}s ({:.0} NPS)",
        r3.nodes,
        t3.as_secs_f64(),
        r3.nodes as f64 / t3.as_secs_f64()
    );

    // Depth 5 with 30s time limit
    let mut searcher = AlphaBetaSearcher::default();
    let start = Instant::now();
    let r5 = searcher.search_with_info(
        &position,
        SearchLimits {
            depth: Some(5),
            time_ms: Some(30_000),
            ..Default::default()
        },
    );
    let t5 = start.elapsed();
    eprintln!(
        "Depth 5 (30s limit): depth={} {} nodes in {:.3}s ({:.0} NPS)",
        r5.depth,
        r5.nodes,
        t5.as_secs_f64(),
        r5.nodes as f64 / t5.as_secs_f64()
    );

    // Also test a midgame FEN with fewer pieces
    let midgame = Position::from_fen("8m/9/9/9/4Gd3/9/9/9/M8 -/- w 3 - 1").unwrap();
    let mid_moves = midgame.moves();
    eprintln!("\nMidgame (sparse): {} legal moves", mid_moves.len());

    let mut searcher = AlphaBetaSearcher::default();
    let start = Instant::now();
    let rm = searcher.search_with_info(
        &midgame,
        SearchLimits {
            depth: Some(20),
            time_ms: Some(10_000),
            ..Default::default()
        },
    );
    let tm = start.elapsed();
    eprintln!(
        "Endgame depth 20 (10s): depth={} {} nodes in {:.3}s ({:.0} NPS)",
        rm.depth,
        rm.nodes,
        tm.as_secs_f64(),
        rm.nodes as f64 / tm.as_secs_f64()
    );

    let mut dense_mid = Position::new(SetupMode::Beginner);
    let opening_moves = dense_mid.moves();
    if let Some(mv) = opening_moves.first() {
        let _ = dense_mid.make_move(mv);
        let resp = dense_mid.moves();
        if let Some(mv2) = resp.first() {
            let _ = dense_mid.make_move(mv2);
        }
    }
    let dense_moves = dense_mid.moves();
    eprintln!("\nDense midgame: {} legal moves", dense_moves.len());

    for depth in [4, 5, 6, 7, 8] {
        let mut searcher = AlphaBetaSearcher::default();
        let start = Instant::now();
        let rd = searcher.search_with_info(
            &dense_mid,
            SearchLimits {
                depth: Some(depth),
                time_ms: Some(30_000),
                ..Default::default()
            },
        );
        let td = start.elapsed();
        eprintln!(
            "Dense midgame depth {} (30s): depth={} {} nodes in {:.3}s ({:.0} NPS)",
            depth,
            rd.depth,
            rd.nodes,
            td.as_secs_f64(),
            rd.nodes as f64 / td.as_secs_f64()
        );
        if rd.depth < depth {
            break;
        }
    }

    assert!(r1.nodes > 0);
}
