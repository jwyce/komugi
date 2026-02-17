use komugi_core::{Color, Gungi};
use komugi_engine::nnue_features;
use serde_json::json;
use std::env;
use std::fs;
use std::io::{self, BufRead};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <fen_file>", args[0]);
        std::process::exit(1);
    }

    let fen_file = &args[1];
    let file = fs::File::open(fen_file)?;
    let reader = io::BufReader::new(file);

    let mut results = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse FEN and extract features
        match Gungi::from_fen(line) {
            Ok(gungi) => {
                let white_features: Vec<u16> =
                    nnue_features::extract_features(gungi.position(), Color::White);
                let black_features: Vec<u16> =
                    nnue_features::extract_features(gungi.position(), Color::Black);

                results.push(json!({
                    "fen": line,
                    "white": white_features,
                    "black": black_features,
                }));
            }
            Err(e) => {
                eprintln!("Failed to parse FEN '{}': {}", line, e);
                std::process::exit(1);
            }
        }
    }

    // Output as JSON array
    let output = serde_json::to_string_pretty(&results).unwrap();
    println!("{}", output);

    Ok(())
}
