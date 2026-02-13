use std::sync::{Arc, Mutex};
use std::thread;

use komugi_core::{is_marshal_captured, Color, Position, SearchLimits, Searcher, SetupMode};
use komugi_engine::mcts::{MctsConfig, MctsSearcher};
use komugi_engine::{AlphaBetaConfig, AlphaBetaSearcher, ClassicalEval};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchResult {
    MctsWin,
    AlphaBetaWin,
    Draw,
}

fn parse_mode(s: &str) -> SetupMode {
    match s.to_lowercase().as_str() {
        "intro" | "0" => SetupMode::Intro,
        "beginner" | "1" => SetupMode::Beginner,
        "intermediate" | "2" => SetupMode::Intermediate,
        "advanced" | "3" => SetupMode::Advanced,
        _ => {
            eprintln!("Unknown mode '{s}', defaulting to beginner");
            SetupMode::Beginner
        }
    }
}

fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "White",
        Color::Black => "Black",
    }
}

fn result_name(result: MatchResult) -> &'static str {
    match result {
        MatchResult::MctsWin => "MCTS win",
        MatchResult::AlphaBetaWin => "AlphaBeta win",
        MatchResult::Draw => "Draw",
    }
}

fn infer_result(position: &Position, mcts_color: Color, reached_move_limit: bool) -> MatchResult {
    if reached_move_limit || position.is_draw() {
        return MatchResult::Draw;
    }

    if is_marshal_captured(position) {
        let white_alive = position.marshal_squares[Color::White as usize].is_some();
        let black_alive = position.marshal_squares[Color::Black as usize].is_some();
        let winner = if white_alive && !black_alive {
            Color::White
        } else if black_alive && !white_alive {
            Color::Black
        } else {
            opposite(position.turn)
        };
        return if winner == mcts_color {
            MatchResult::MctsWin
        } else {
            MatchResult::AlphaBetaWin
        };
    }

    if position.is_checkmate() {
        let winner = opposite(position.turn);
        return if winner == mcts_color {
            MatchResult::MctsWin
        } else {
            MatchResult::AlphaBetaWin
        };
    }

    MatchResult::Draw
}

fn play_game(
    mode: SetupMode,
    mcts_sims: u32,
    ab_depth: u8,
    mcts_is_white: bool,
) -> (u32, MatchResult) {
    let mut position = Position::new(mode);
    let mut mcts = MctsSearcher::new(MctsConfig {
        max_simulations: mcts_sims,
        ..Default::default()
    });
    let mut alphabeta = AlphaBetaSearcher::with_eval(
        AlphaBetaConfig {
            max_depth: ab_depth,
            ..Default::default()
        },
        Box::new(ClassicalEval::new()),
    );

    let mcts_color = if mcts_is_white {
        Color::White
    } else {
        Color::Black
    };
    let mcts_limits = SearchLimits {
        nodes: Some(u64::from(mcts_sims)),
        ..Default::default()
    };
    let ab_limits = SearchLimits {
        depth: Some(ab_depth),
        ..Default::default()
    };

    let mut moves_played = 0u32;
    while !position.is_game_over() && position.move_number <= 300 {
        let result = if position.turn == mcts_color {
            mcts.search(&position, mcts_limits)
        } else {
            alphabeta.search(&position, ab_limits)
        };

        let Some(mv) = result.best_move else {
            break;
        };

        position.make_move(&mv).unwrap();
        moves_played = moves_played.saturating_add(1);
    }

    let reached_move_limit = position.move_number > 300 && !position.is_game_over();
    let outcome = infer_result(&position, mcts_color, reached_move_limit);
    (moves_played, outcome)
}

fn elo_diff(mcts_wins: u32, draws: u32, total_games: u32) -> f64 {
    if total_games == 0 {
        return 0.0;
    }
    let win_rate = (mcts_wins as f64 + draws as f64 * 0.5) / total_games as f64;
    if win_rate <= 0.0 {
        f64::NEG_INFINITY
    } else if win_rate >= 1.0 {
        f64::INFINITY
    } else {
        -400.0 * (1.0 / win_rate - 1.0).log10()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let mode = args
        .get(2)
        .map(|s| parse_mode(s))
        .unwrap_or(SetupMode::Beginner);
    let mcts_sims: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
    let ab_depth: u8 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let num_threads: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    let completed = Arc::new(Mutex::new(0u32));
    let results: Arc<Mutex<Vec<(u32, bool, u32, MatchResult)>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_games as usize)));

    eprintln!(
        "Running {num_games} arena games ({mode:?}) with MCTS nodes={mcts_sims}, AB depth={ab_depth} on {num_threads} threads..."
    );

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let completed = Arc::clone(&completed);
            let results = Arc::clone(&results);

            thread::spawn(move || loop {
                let game_num = {
                    let mut c = completed.lock().unwrap();
                    if *c >= num_games {
                        break;
                    }
                    *c += 1;
                    *c
                };

                let mcts_is_white = game_num % 2 == 1;
                let (moves, result) = play_game(mode, mcts_sims, ab_depth, mcts_is_white);

                eprintln!(
                    "[t{thread_id}] Game {game_num}/{num_games}: {moves} moves, {} (MCTS={})",
                    result_name(result),
                    color_name(if mcts_is_white {
                        Color::White
                    } else {
                        Color::Black
                    })
                );

                results
                    .lock()
                    .unwrap()
                    .push((game_num, mcts_is_white, moves, result));
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    let mut all_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    all_results.sort_by_key(|(game_num, _, _, _)| *game_num);

    let mut mcts_wins = 0u32;
    let mut ab_wins = 0u32;
    let mut draws = 0u32;
    for (_, _, _, result) in all_results {
        match result {
            MatchResult::MctsWin => mcts_wins = mcts_wins.saturating_add(1),
            MatchResult::AlphaBetaWin => ab_wins = ab_wins.saturating_add(1),
            MatchResult::Draw => draws = draws.saturating_add(1),
        }
    }

    let elo = elo_diff(mcts_wins, draws, num_games);
    eprintln!(
        "MCTS: {mcts_wins}W/{ab_wins}L/{draws}D | AlphaBeta: {ab_wins}W/{mcts_wins}L/{draws}D | Elo diff: {elo:+.1}"
    );
}
