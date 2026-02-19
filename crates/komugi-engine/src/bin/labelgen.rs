use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam::channel;
use komugi_core::{move_to_san, Policy, Position, SearchLimits, SetupMode};
use komugi_engine::mcts::{HeuristicPolicy, MctsConfig, MctsSearcher};
#[cfg(feature = "neural")]
use komugi_engine::neural::GpuInferencePool;
#[cfg(feature = "neural")]
use komugi_engine::NeuralPolicy;
use komugi_engine::{play_game, SelfPlayConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Phase1Record {
    fen: String,
    outcome: f32,
    move_number: u32,
}

#[derive(Debug, Clone, Serialize)]
struct LabelRecord {
    fen: String,
    outcome: f32,
    search_value: i32,
    policy: Vec<(String, f32)>,
    move_number: u32,
}

fn detect_gpu_count() -> usize {
    if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        let trimmed = visible.trim();
        if trimmed == "-1" {
            return 0;
        }
        if !trimmed.is_empty() {
            let count = trimmed
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty() && *s != "NoDevFiles")
                .count();
            if count > 0 {
                return count;
            }
        }
    }

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

fn parse_model_arg(arg: Option<&String>) -> Option<String> {
    arg.and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed == "-" {
            None
        } else {
            Some(trimmed.to_owned())
        }
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let num_games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let output_file = args.get(2).map(String::as_str).unwrap_or("labels.jsonl");
    let play_sims: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
    let eval_sims: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1600);
    let model_path = parse_model_arg(args.get(5));
    let mode = args
        .get(6)
        .map(|s| parse_mode(s))
        .unwrap_or(SetupMode::Beginner);
    let num_threads: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    let play_config = MctsConfig {
        max_simulations: play_sims,
        ..Default::default()
    };
    let eval_config = MctsConfig {
        max_simulations: eval_sims,
        ..Default::default()
    };

    #[cfg(feature = "neural")]
    let gpu_pool: Option<Arc<GpuInferencePool>> = model_path.as_deref().and_then(|path| {
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
    if model_path.is_some() {
        eprintln!("Model path provided but neural feature is disabled; using heuristic policy");
    }

    eprintln!(
        "Phase 1: generating {num_games} games ({mode:?}) with {play_sims} sims on {num_threads} threads"
    );

    let phase1_path = format!("{output_file}.phase1.tmp.jsonl");
    let phase1_file = File::create(&phase1_path)?;
    let (phase1_tx, phase1_rx) = channel::bounded::<Phase1Record>(4096);

    let phase1_writer = thread::spawn(move || -> Result<u64, String> {
        let mut writer = BufWriter::new(phase1_file);
        let mut count = 0u64;
        for record in phase1_rx {
            serde_json::to_writer(&mut writer, &record).map_err(|e| e.to_string())?;
            writer.write_all(b"\n").map_err(|e| e.to_string())?;
            count = count.saturating_add(1);
            if count % 10_000 == 0 {
                eprintln!("Phase 1 positions: {count}");
            }
        }
        writer.flush().map_err(|e| e.to_string())?;
        Ok(count)
    });

    let games_done = Arc::new(Mutex::new(0u32));
    let model_path_owned = model_path.clone();

    let phase1_handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let games_done = Arc::clone(&games_done);
            let phase1_tx = phase1_tx.clone();
            let model_path = model_path_owned.clone();
            #[cfg(feature = "neural")]
            let gpu_pool = gpu_pool.clone();

            thread::spawn(move || {
                let policy: Arc<dyn Policy> = {
                    #[cfg(feature = "neural")]
                    {
                        if let Some(ref pool) = gpu_pool {
                            Arc::new(pool.policy(thread_id))
                        } else if let Some(ref path) = model_path {
                            Arc::new(
                                NeuralPolicy::from_file(path)
                                    .expect("failed to load model for phase 1"),
                            )
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
                        let mut g = games_done.lock().expect("failed to lock game counter");
                        if *g >= num_games {
                            break;
                        }
                        *g += 1;
                        *g
                    };

                    let config = SelfPlayConfig {
                        mcts_config: play_config,
                        setup_mode: mode,
                        max_moves: 300,
                        policy: Arc::clone(&policy),
                    };

                    let record = play_game(&config);
                    for pos in record.positions {
                        if phase1_tx
                            .send(Phase1Record {
                                fen: pos.fen,
                                outcome: pos.outcome,
                                move_number: pos.move_number,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }

                    eprintln!(
                        "[p1 t{thread_id}] game {game_num}/{num_games}: {} moves, {:?}",
                        record.total_moves, record.result
                    );
                }
            })
        })
        .collect();

    drop(phase1_tx);
    for handle in phase1_handles {
        handle
            .join()
            .map_err(|_| "phase 1 worker thread panicked")?;
    }

    let total_positions = phase1_writer
        .join()
        .map_err(|_| "phase 1 writer thread panicked")?
        .map_err(|e| format!("phase 1 writer failed: {e}"))?;

    eprintln!("Phase 1 complete: {total_positions} positions in {phase1_path}");

    eprintln!(
        "Phase 2: re-evaluating {total_positions} positions with {eval_sims} sims on {num_threads} threads"
    );

    let input_file = File::open(&phase1_path)?;
    let output = File::create(output_file)?;
    let (jobs_tx, jobs_rx) = channel::bounded::<Phase1Record>(4096);
    let (out_tx, out_rx) = channel::bounded::<LabelRecord>(4096);

    let output_writer = thread::spawn(move || -> Result<u64, String> {
        let mut writer = BufWriter::new(output);
        let mut written = 0u64;
        for rec in out_rx {
            serde_json::to_writer(&mut writer, &rec).map_err(|e| e.to_string())?;
            writer.write_all(b"\n").map_err(|e| e.to_string())?;
            written = written.saturating_add(1);
            if written % 10_000 == 0 {
                eprintln!("Phase 2 written: {written}");
            }
        }
        writer.flush().map_err(|e| e.to_string())?;
        Ok(written)
    });

    let processed = Arc::new(AtomicU32::new(0));
    let mut worker_handles = Vec::with_capacity(num_threads);
    for thread_id in 0..num_threads {
        let jobs_rx = jobs_rx.clone();
        let out_tx = out_tx.clone();
        let model_path = model_path.clone();
        let processed = Arc::clone(&processed);
        #[cfg(feature = "neural")]
        let gpu_pool = gpu_pool.clone();

        let handle = thread::spawn(move || {
            let policy: Arc<dyn Policy> = {
                #[cfg(feature = "neural")]
                {
                    if let Some(ref pool) = gpu_pool {
                        Arc::new(pool.policy(thread_id + num_threads))
                    } else if let Some(ref path) = model_path {
                        Arc::new(
                            NeuralPolicy::from_file(path)
                                .expect("failed to load model for phase 2"),
                        )
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

            let mut searcher = MctsSearcher::new(eval_config);

            for job in jobs_rx {
                let position = match Position::from_fen(&job.fen) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("[p2 t{thread_id}] skipped invalid FEN: {e}");
                        continue;
                    }
                };

                let search_result = searcher.search_with_policy(
                    &position,
                    SearchLimits::default(),
                    policy.as_ref(),
                );
                let policy_dist = searcher
                    .get_root_policy()
                    .into_iter()
                    .map(|(mv, proportion)| (move_to_san(&mv), proportion))
                    .collect();

                if out_tx
                    .send(LabelRecord {
                        fen: job.fen,
                        outcome: job.outcome,
                        search_value: search_result.score.0,
                        policy: policy_dist,
                        move_number: job.move_number,
                    })
                    .is_err()
                {
                    return;
                }

                let done = processed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 5_000 == 0 {
                    eprintln!("Phase 2 progress: {done}/{total_positions}");
                }
            }
        });
        worker_handles.push(handle);
    }

    drop(out_tx);

    for line in BufReader::new(input_file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Phase1Record = serde_json::from_str(&line)?;
        jobs_tx.send(record)?;
    }
    drop(jobs_tx);

    for handle in worker_handles {
        handle
            .join()
            .map_err(|_| "phase 2 worker thread panicked")?;
    }

    let written = output_writer
        .join()
        .map_err(|_| "phase 2 writer thread panicked")?
        .map_err(|e| format!("phase 2 writer failed: {e}"))?;

    let _ = std::fs::remove_file(&phase1_path);
    eprintln!("Done. Wrote {written} labels to {output_file}");

    Ok(())
}
