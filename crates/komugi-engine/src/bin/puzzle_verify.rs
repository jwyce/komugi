use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use komugi_core::{move_to_san, Color, Position, SearchLimits};
use komugi_engine::{AlphaBetaConfig, AlphaBetaSearcher};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
struct Config {
    input: String,
    output_prefix: String,
    depth: u8,
    num_pv: usize,
    max_candidates: Option<usize>,
    easy_min_ply: usize,
    easy_max_ply: usize,
    medium_min_ply: usize,
    medium_max_ply: usize,
    hard_min_ply: usize,
    hard_max_ply: usize,
    easy_attacker_gap: i32,
    medium_attacker_gap: i32,
    hard_attacker_gap: i32,
    easy_min_final_eval: i32,
    medium_min_final_eval: i32,
    hard_min_final_eval: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct Stage1Puzzle {
    fen: String,
    solution: String,
    #[serde(flatten)]
    extra: Value,
}

#[derive(Debug, Clone, Serialize)]
struct VerifiedPuzzle {
    fen: String,
    solution: String,
    stage2_difficulty: String,
    verified_line: Vec<String>,
    verified_ply: usize,
    depth: u8,
    num_pv: usize,
    root_eval_cp: i32,
    final_eval_cp: i32,
    attacker_min_gap_cp: i32,
    #[serde(flatten)]
    extra: Value,
}

#[derive(Debug, Clone)]
struct TierParams {
    name: &'static str,
    min_ply: usize,
    max_ply: usize,
    attacker_min_gap_cp: i32,
    min_final_eval_cp: i32,
}

#[derive(Debug, Clone)]
struct TierResult {
    line: Vec<String>,
    root_eval_cp: i32,
    final_eval_cp: i32,
}

fn parse_arg<T: std::str::FromStr>(args: &[String], i: &mut usize, key: &str) -> Result<T, String> {
    if *i + 1 >= args.len() {
        return Err(format!("missing value for {key}"));
    }
    let raw = &args[*i + 1];
    *i += 2;
    raw.parse::<T>()
        .map_err(|_| format!("invalid value for {key}: {raw}"))
}

fn parse_string_arg(args: &[String], i: &mut usize, key: &str) -> Result<String, String> {
    if *i + 1 >= args.len() {
        return Err(format!("missing value for {key}"));
    }
    let value = args[*i + 1].clone();
    *i += 2;
    Ok(value)
}

fn parse_config_from_args(args: &[String]) -> Result<Config, String> {
    let mut cfg = Config {
        input: String::new(),
        output_prefix: String::from("puzzles_stage2"),
        depth: 4,
        num_pv: 3,
        max_candidates: None,
        easy_min_ply: 1,
        easy_max_ply: 3,
        medium_min_ply: 5,
        medium_max_ply: 11,
        hard_min_ply: 13,
        hard_max_ply: 19,
        easy_attacker_gap: 120,
        medium_attacker_gap: 80,
        hard_attacker_gap: 45,
        easy_min_final_eval: 350,
        medium_min_final_eval: 500,
        hard_min_final_eval: 700,
    };

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => cfg.input = parse_string_arg(args, &mut i, "--input")?,
            "--output-prefix" => {
                cfg.output_prefix = parse_string_arg(args, &mut i, "--output-prefix")?
            }
            "--depth" => cfg.depth = parse_arg(args, &mut i, "--depth")?,
            "--num-pv" => cfg.num_pv = parse_arg(args, &mut i, "--num-pv")?,
            "--max-candidates" => {
                cfg.max_candidates = Some(parse_arg(args, &mut i, "--max-candidates")?)
            }
            "--easy-ply" => {
                let ply: usize = parse_arg(args, &mut i, "--easy-ply")?;
                cfg.easy_min_ply = ply;
                cfg.easy_max_ply = ply;
            }
            "--medium-ply" => {
                let ply: usize = parse_arg(args, &mut i, "--medium-ply")?;
                cfg.medium_min_ply = ply;
                cfg.medium_max_ply = ply;
            }
            "--hard-ply" => {
                let ply: usize = parse_arg(args, &mut i, "--hard-ply")?;
                cfg.hard_min_ply = ply;
                cfg.hard_max_ply = ply;
            }
            "--easy-min-ply" => cfg.easy_min_ply = parse_arg(args, &mut i, "--easy-min-ply")?,
            "--easy-max-ply" => cfg.easy_max_ply = parse_arg(args, &mut i, "--easy-max-ply")?,
            "--medium-min-ply" => cfg.medium_min_ply = parse_arg(args, &mut i, "--medium-min-ply")?,
            "--medium-max-ply" => cfg.medium_max_ply = parse_arg(args, &mut i, "--medium-max-ply")?,
            "--hard-min-ply" => cfg.hard_min_ply = parse_arg(args, &mut i, "--hard-min-ply")?,
            "--hard-max-ply" => cfg.hard_max_ply = parse_arg(args, &mut i, "--hard-max-ply")?,
            "--easy-attacker-gap" => {
                cfg.easy_attacker_gap = parse_arg(args, &mut i, "--easy-attacker-gap")?
            }
            "--medium-attacker-gap" => {
                cfg.medium_attacker_gap = parse_arg(args, &mut i, "--medium-attacker-gap")?
            }
            "--hard-attacker-gap" => {
                cfg.hard_attacker_gap = parse_arg(args, &mut i, "--hard-attacker-gap")?
            }
            "--easy-min-final-eval" => {
                cfg.easy_min_final_eval = parse_arg(args, &mut i, "--easy-min-final-eval")?
            }
            "--medium-min-final-eval" => {
                cfg.medium_min_final_eval = parse_arg(args, &mut i, "--medium-min-final-eval")?
            }
            "--hard-min-final-eval" => {
                cfg.hard_min_final_eval = parse_arg(args, &mut i, "--hard-min-final-eval")?
            }
            "--help" | "-h" => {
                println!(
                    "Usage: puzzle_verify --input <stage1.jsonl> [options]\n\n\
Options:\n\
  --output-prefix <path>         Output prefix (default: puzzles_stage2)\n\
  --depth <n>                    Search depth (default: 4)\n\
  --num-pv <n>                   Multi-PV count (default: 3)\n\
  --max-candidates <n>           Stop after N input records\n\
  --easy-min-ply <n>             Min verified line ply for easy (default: 1)\n\
  --easy-max-ply <n>             Max verified line ply for easy (default: 3)\n\
  --medium-min-ply <n>           Min verified line ply for medium (default: 5)\n\
  --medium-max-ply <n>           Max verified line ply for medium (default: 11)\n\
  --hard-min-ply <n>             Min verified line ply for hard (default: 13)\n\
  --hard-max-ply <n>             Max verified line ply for hard (default: 19)\n\
  --easy-ply <n>                 Legacy: set easy min=max=<n>\n\
  --medium-ply <n>               Legacy: set medium min=max=<n>\n\
  --hard-ply <n>                 Legacy: set hard min=max=<n>\n\
  --easy-attacker-gap <cp>       Min attacker move gap for easy (default: 120)\n\
  --medium-attacker-gap <cp>     Min attacker move gap for medium (default: 80)\n\
  --hard-attacker-gap <cp>       Min attacker move gap for hard (default: 45)\n\
  --easy-min-final-eval <cp>     Min final eval from root side for easy (default: 350)\n\
  --medium-min-final-eval <cp>   Min final eval from root side for medium (default: 500)\n\
  --hard-min-final-eval <cp>     Min final eval from root side for hard (default: 700)"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    if cfg.input.is_empty() {
        return Err(String::from("--input is required"));
    }
    if cfg.num_pv < 2 {
        return Err(String::from("--num-pv must be >= 2"));
    }
    if cfg.easy_min_ply > cfg.easy_max_ply {
        return Err(String::from("expected easy-min-ply <= easy-max-ply"));
    }
    if cfg.medium_min_ply > cfg.medium_max_ply {
        return Err(String::from("expected medium-min-ply <= medium-max-ply"));
    }
    if cfg.hard_min_ply > cfg.hard_max_ply {
        return Err(String::from("expected hard-min-ply <= hard-max-ply"));
    }
    if cfg.easy_max_ply >= cfg.medium_min_ply {
        return Err(String::from(
            "expected easy-max-ply < medium-min-ply to avoid overlap",
        ));
    }
    if cfg.medium_max_ply >= cfg.hard_min_ply {
        return Err(String::from(
            "expected medium-max-ply < hard-min-ply to avoid overlap",
        ));
    }

    Ok(cfg)
}

fn parse_config() -> Result<Config, String> {
    let args: Vec<String> = std::env::args().collect();
    parse_config_from_args(&args)
}

fn to_root_score(score_stm: i32, turn: Color, root_turn: Color) -> i32 {
    if turn == root_turn {
        score_stm
    } else {
        -score_stm
    }
}

fn verify_exact_ply(
    stage1: &Stage1Puzzle,
    searcher: &mut AlphaBetaSearcher,
    depth: u8,
    num_pv: usize,
    target_ply: usize,
    tier: &TierParams,
) -> Option<TierResult> {
    let mut position = Position::from_fen(&stage1.fen).ok()?;
    let root_turn = position.turn;
    let limits = SearchLimits {
        depth: Some(depth),
        ..SearchLimits::default()
    };

    let mut line = Vec::with_capacity(target_ply);
    let mut root_eval_cp = None;

    for ply_idx in 0..target_ply {
        if position.is_game_over() {
            return None;
        }

        let multi_pv = searcher.search_multi_pv(&position, limits, num_pv);
        if multi_pv.lines.is_empty() {
            return None;
        }

        let best_line = &multi_pv.lines[0];
        let best_move = best_line.moves.first()?.clone();
        let best_san = move_to_san(&best_move);

        if ply_idx == 0 && best_san != stage1.solution {
            return None;
        }

        let best_score_stm = best_line.score.0;
        let second_score_stm = multi_pv
            .lines
            .get(1)
            .map(|line| line.score.0)
            .unwrap_or(best_score_stm);

        if ply_idx == 0 {
            root_eval_cp = Some(to_root_score(best_score_stm, position.turn, root_turn));
        }

        if position.turn == root_turn {
            let gap = best_score_stm - second_score_stm;
            if gap < tier.attacker_min_gap_cp {
                return None;
            }
        }

        line.push(best_san);
        if position.make_move(&best_move).is_err() {
            return None;
        }
    }

    let final_eval_stm = searcher.search_with_info(&position, limits).score.0;
    let final_eval_cp = to_root_score(final_eval_stm, position.turn, root_turn);

    if final_eval_cp < tier.min_final_eval_cp {
        return None;
    }

    Some(TierResult {
        line,
        root_eval_cp: root_eval_cp.unwrap_or(0),
        final_eval_cp,
    })
}

fn verify_tier(
    stage1: &Stage1Puzzle,
    searcher: &mut AlphaBetaSearcher,
    depth: u8,
    num_pv: usize,
    tier: &TierParams,
) -> Option<TierResult> {
    for target_ply in (tier.min_ply..=tier.max_ply).rev() {
        if let Some(result) = verify_exact_ply(stage1, searcher, depth, num_pv, target_ply, tier) {
            return Some(result);
        }
    }
    None
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = parse_config().map_err(|e| format!("argument error: {e}"))?;

    let input = File::open(&cfg.input)?;
    let mut all_out = BufWriter::new(File::create(format!("{}_all.jsonl", cfg.output_prefix))?);
    let mut easy_out = BufWriter::new(File::create(format!("{}_easy.jsonl", cfg.output_prefix))?);
    let mut medium_out =
        BufWriter::new(File::create(format!("{}_medium.jsonl", cfg.output_prefix))?);
    let mut hard_out = BufWriter::new(File::create(format!("{}_hard.jsonl", cfg.output_prefix))?);

    let tiers = vec![
        TierParams {
            name: "hard",
            min_ply: cfg.hard_min_ply,
            max_ply: cfg.hard_max_ply,
            attacker_min_gap_cp: cfg.hard_attacker_gap,
            min_final_eval_cp: cfg.hard_min_final_eval,
        },
        TierParams {
            name: "medium",
            min_ply: cfg.medium_min_ply,
            max_ply: cfg.medium_max_ply,
            attacker_min_gap_cp: cfg.medium_attacker_gap,
            min_final_eval_cp: cfg.medium_min_final_eval,
        },
        TierParams {
            name: "easy",
            min_ply: cfg.easy_min_ply,
            max_ply: cfg.easy_max_ply,
            attacker_min_gap_cp: cfg.easy_attacker_gap,
            min_final_eval_cp: cfg.easy_min_final_eval,
        },
    ];

    let mut searcher = AlphaBetaSearcher::new(AlphaBetaConfig::default());

    let mut scanned = 0usize;
    let mut parse_errors = 0usize;
    let mut kept_easy = 0usize;
    let mut kept_medium = 0usize;
    let mut kept_hard = 0usize;

    for line in BufReader::new(input).lines() {
        if let Some(max) = cfg.max_candidates {
            if scanned >= max {
                break;
            }
        }

        scanned = scanned.saturating_add(1);
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let stage1: Stage1Puzzle = match serde_json::from_str(&line) {
            Ok(rec) => rec,
            Err(_) => {
                parse_errors = parse_errors.saturating_add(1);
                continue;
            }
        };

        let mut chosen: Option<(String, TierParams, TierResult)> = None;
        for tier in &tiers {
            if let Some(result) = verify_tier(&stage1, &mut searcher, cfg.depth, cfg.num_pv, tier) {
                chosen = Some((tier.name.to_owned(), tier.clone(), result));
                break;
            }
        }

        let Some((difficulty, tier, result)) = chosen else {
            if scanned % 1000 == 0 {
                eprintln!("Scanned {scanned} candidates...");
            }
            continue;
        };

        let verified_ply = result.line.len();
        let verified = VerifiedPuzzle {
            fen: stage1.fen,
            solution: stage1.solution,
            stage2_difficulty: difficulty.clone(),
            verified_line: result.line,
            verified_ply,
            depth: cfg.depth,
            num_pv: cfg.num_pv,
            root_eval_cp: result.root_eval_cp,
            final_eval_cp: result.final_eval_cp,
            attacker_min_gap_cp: tier.attacker_min_gap_cp,
            extra: stage1.extra,
        };

        serde_json::to_writer(&mut all_out, &verified)?;
        all_out.write_all(b"\n")?;

        match difficulty.as_str() {
            "easy" => {
                kept_easy = kept_easy.saturating_add(1);
                serde_json::to_writer(&mut easy_out, &verified)?;
                easy_out.write_all(b"\n")?;
            }
            "medium" => {
                kept_medium = kept_medium.saturating_add(1);
                serde_json::to_writer(&mut medium_out, &verified)?;
                medium_out.write_all(b"\n")?;
            }
            "hard" => {
                kept_hard = kept_hard.saturating_add(1);
                serde_json::to_writer(&mut hard_out, &verified)?;
                hard_out.write_all(b"\n")?;
            }
            _ => {}
        }

        if scanned % 1000 == 0 {
            eprintln!(
                "Scanned {scanned} candidates... kept easy={kept_easy}, medium={kept_medium}, hard={kept_hard}"
            );
        }
    }

    all_out.flush()?;
    easy_out.flush()?;
    medium_out.flush()?;
    hard_out.flush()?;

    eprintln!(
        "Done. scanned={scanned} parse_errors={parse_errors} kept easy={kept_easy}, medium={kept_medium}, hard={kept_hard}"
    );
    eprintln!(
        "Outputs: {}_all.jsonl, {}_easy.jsonl, {}_medium.jsonl, {}_hard.jsonl",
        cfg.output_prefix, cfg.output_prefix, cfg.output_prefix, cfg.output_prefix
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_args(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_defaults_with_required_input() {
        let args = vec_args(&["puzzle_verify", "--input", "seed.jsonl"]);
        let cfg = parse_config_from_args(&args).expect("config should parse");
        assert_eq!(cfg.input, "seed.jsonl");
        assert_eq!(cfg.depth, 4);
        assert_eq!(cfg.num_pv, 3);
        assert_eq!(cfg.easy_min_ply, 1);
        assert_eq!(cfg.easy_max_ply, 3);
        assert_eq!(cfg.medium_min_ply, 5);
        assert_eq!(cfg.medium_max_ply, 11);
        assert_eq!(cfg.hard_min_ply, 13);
        assert_eq!(cfg.hard_max_ply, 19);
    }

    #[test]
    fn parse_legacy_single_ply_flags() {
        let args = vec_args(&[
            "puzzle_verify",
            "--input",
            "seed.jsonl",
            "--easy-ply",
            "3",
            "--medium-ply",
            "7",
            "--hard-ply",
            "15",
        ]);
        let cfg = parse_config_from_args(&args).expect("config should parse");
        assert_eq!(cfg.easy_min_ply, 3);
        assert_eq!(cfg.easy_max_ply, 3);
        assert_eq!(cfg.medium_min_ply, 7);
        assert_eq!(cfg.medium_max_ply, 7);
        assert_eq!(cfg.hard_min_ply, 15);
        assert_eq!(cfg.hard_max_ply, 15);
    }

    #[test]
    fn rejects_num_pv_below_two() {
        let args = vec_args(&["puzzle_verify", "--input", "seed.jsonl", "--num-pv", "1"]);
        let err = parse_config_from_args(&args).expect_err("should reject num-pv < 2");
        assert!(err.contains("--num-pv must be >= 2"));
    }

    #[test]
    fn rejects_overlapping_ply_ranges() {
        let args = vec_args(&[
            "puzzle_verify",
            "--input",
            "seed.jsonl",
            "--easy-max-ply",
            "6",
            "--medium-min-ply",
            "5",
        ]);
        let err = parse_config_from_args(&args).expect_err("should reject overlap");
        assert!(err.contains("easy-max-ply < medium-min-ply"));
    }
}
