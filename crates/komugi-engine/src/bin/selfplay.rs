use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;

use komugi_core::{parse_fen, Color, Policy, SetupMode};
use komugi_engine::mcts::{HeuristicPolicy, MctsConfig};
#[cfg(feature = "neural")]
use komugi_engine::neural::GpuInferencePool;
#[cfg(feature = "neural")]
use komugi_engine::NeuralPolicy;
use komugi_engine::{play_game, GameRecord, SelfPlayConfig};

fn detect_gpu_count() -> usize {
    let output = std::process::Command::new("nvidia-smi").arg("-L").output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter(|l| l.starts_with("GPU "))
            .count(),
        _ => 0,
    }
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

fn format_pgn(game_num: u32, record: &GameRecord, mode: SetupMode) -> String {
    let mode_name = match mode {
        SetupMode::Intro => "Intro",
        SetupMode::Beginner => "Beginner",
        SetupMode::Intermediate => "Intermediate",
        SetupMode::Advanced => "Advanced",
    };
    let result_str = match record.result {
        komugi_engine::GameResult::WhiteWin => "1-0",
        komugi_engine::GameResult::BlackWin => "0-1",
        komugi_engine::GameResult::Draw => "1/2-1/2",
    };

    let mut pgn = String::new();
    pgn.push_str(&format!("[Game \"{game_num}\"]\n"));
    pgn.push_str(&format!("[Mode \"{mode_name}\"]\n"));
    pgn.push_str(&format!("[Result \"{result_str}\"]\n"));
    pgn.push_str(&format!("[PlyCount \"{}\"]\n\n", record.total_moves));

    let mut next_move_number = 1u32;
    let mut prev_turn: Option<Color> = None;
    for (i, san) in record.moves.iter().enumerate() {
        let current_turn = record
            .positions
            .get(i)
            .and_then(|position| parse_fen(&position.fen).ok())
            .map(|parsed| parsed.turn)
            .unwrap_or(if i % 2 == 0 {
                Color::White
            } else {
                Color::Black
            });

        let should_print_number = current_turn == Color::White && prev_turn != Some(Color::White);

        if should_print_number {
            pgn.push_str(&format!("{}. ", next_move_number));
            next_move_number = next_move_number.saturating_add(1);
        }
        pgn.push_str(san);
        pgn.push(' ');
        prev_turn = Some(current_turn);
    }
    pgn.push_str(result_str);
    pgn.push('\n');
    pgn
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let output_file = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("selfplay_data.jsonl");
    let simulations: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(800);
    let model_path = args.get(4).map(String::as_str).filter(|s| *s != "-");
    let mode = args
        .get(5)
        .map(|s| parse_mode(s))
        .unwrap_or(SetupMode::Beginner);
    let num_threads: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    let mcts_config = MctsConfig {
        max_simulations: simulations,
        ..Default::default()
    };

    #[cfg(feature = "neural")]
    let gpu_pool: Option<Arc<GpuInferencePool>> = model_path.and_then(|path| {
        let num_gpus = detect_gpu_count();
        if num_gpus == 0 {
            eprintln!("No GPUs detected, falling back to CPU inference");
            return None;
        }
        match GpuInferencePool::new(path, num_gpus) {
            Ok(pool) => {
                eprintln!("Using GPU batch inference on {num_gpus} GPUs");
                Some(Arc::new(pool))
            }
            Err(e) => {
                eprintln!("GPU pool init failed ({e}), falling back to CPU inference");
                None
            }
        }
    });

    #[cfg(not(feature = "neural"))]
    let gpu_pool: Option<Arc<()>> = None;

    let model_path_owned = model_path.map(String::from);
    let completed = Arc::new(Mutex::new(0u32));
    let results: Arc<Mutex<Vec<(u32, GameRecord)>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_games as usize)));

    eprintln!(
        "Generating {num_games} games ({mode:?}) with {simulations} sims on {num_threads} threads..."
    );

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let model_path = model_path_owned.clone();
            let completed = Arc::clone(&completed);
            let results = Arc::clone(&results);
            #[cfg(feature = "neural")]
            let gpu_pool = gpu_pool.clone();

            thread::spawn(move || {
                let policy: Arc<dyn Policy> = {
                    #[cfg(feature = "neural")]
                    {
                        if let Some(ref pool) = gpu_pool {
                            Arc::new(pool.policy(thread_id))
                        } else if let Some(ref path) = model_path {
                            Arc::new(NeuralPolicy::from_file(path).expect("failed to load model"))
                        } else {
                            Arc::new(HeuristicPolicy)
                        }
                    }
                    #[cfg(not(feature = "neural"))]
                    {
                        let _ = &model_path;
                        Arc::new(HeuristicPolicy)
                    }
                };

                loop {
                    let game_num = {
                        let mut c = completed.lock().unwrap();
                        if *c >= num_games {
                            break;
                        }
                        *c += 1;
                        *c
                    };

                    let config = SelfPlayConfig {
                        mcts_config,
                        setup_mode: mode,
                        max_moves: 300,
                        policy: Arc::clone(&policy),
                    };

                    let record = play_game(&config);
                    eprintln!(
                        "[t{thread_id}] Game {game_num}/{num_games}: {} moves, {:?}",
                        record.total_moves, record.result
                    );

                    results.lock().unwrap().push((game_num, record));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    let mut all_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    all_results.sort_by_key(|(num, _)| *num);

    let mut file = std::fs::File::create(output_file).expect("failed to create output file");
    for (_, record) in &all_results {
        for position_record in &record.positions {
            let json = serde_json::to_string(position_record).expect("serialization failed");
            writeln!(file, "{json}").expect("write failed");
        }
    }

    let pgn_path = output_file.replace(".jsonl", ".pgn");
    let mut pgn_file = std::fs::File::create(&pgn_path).expect("failed to create PGN file");
    for (game_num, record) in &all_results {
        write!(pgn_file, "{}", format_pgn(*game_num, record, mode)).expect("PGN write failed");
        writeln!(pgn_file).expect("PGN write failed");
    }

    eprintln!("Done. Wrote {num_games} games to {output_file}");
    eprintln!("PGN: {pgn_path}");
}

#[cfg(test)]
mod tests {
    use super::format_pgn;
    use komugi_core::SetupMode;
    use komugi_engine::{GameRecord, GameResult, TrainingRecord};

    fn rec(fen: &str) -> TrainingRecord {
        TrainingRecord {
            fen: fen.to_string(),
            played_move: String::new(),
            policy: Vec::new(),
            outcome: 0.0,
            move_number: 1,
            encoding: Vec::new(),
        }
    }

    fn rec_turn(turn: char, drafting: &str) -> TrainingRecord {
        rec(&format!("9/9/9/9/9/9/9/9/9 -/- {turn} 2 {drafting} 1"))
    }

    #[test]
    fn format_pgn_numbers_by_white_turn_segments() {
        let record = GameRecord {
            positions: vec![
                rec_turn('w', "wb"),
                rec_turn('b', "wb"),
                rec_turn('w', "b"),
                rec_turn('b', "b"),
                rec_turn('b', "b"),
                rec_turn('b', "b"),
                rec_turn('w', "-"),
                rec_turn('b', "-"),
            ],
            result: GameResult::Draw,
            total_moves: 8,
            moves: vec![
                "w1".to_string(),
                "b1".to_string(),
                "w2".to_string(),
                "b2".to_string(),
                "b3".to_string(),
                "b4".to_string(),
                "w3".to_string(),
                "b5".to_string(),
            ],
        };

        let pgn = format_pgn(1, &record, SetupMode::Intermediate);
        assert!(pgn.contains("1. w1 b1 2. w2 b2 b3 b4 3. w3 b5"));
    }
}
