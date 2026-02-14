use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;

use komugi_core::{Policy, SetupMode};
use komugi_engine::mcts::{HeuristicPolicy, MctsConfig};
#[cfg(feature = "neural")]
use komugi_engine::NeuralPolicy;
use komugi_engine::{play_game, GameRecord, SelfPlayConfig};

fn create_policy(model_path: Option<&str>) -> Arc<dyn Policy> {
    #[cfg(feature = "neural")]
    if let Some(path) = model_path {
        return Arc::new(NeuralPolicy::from_file(path).expect("failed to load model"));
    }
    let _ = model_path;
    Arc::new(HeuristicPolicy)
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

    for (i, san) in record.moves.iter().enumerate() {
        if i % 2 == 0 {
            pgn.push_str(&format!("{}. ", i / 2 + 1));
        }
        pgn.push_str(san);
        pgn.push(' ');
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

            thread::spawn(move || {
                let policy = create_policy(model_path.as_deref());

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
