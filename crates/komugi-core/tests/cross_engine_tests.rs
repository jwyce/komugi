use komugi_core::{move_to_san, Gungi};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TestVector {
    fen: String,
    moves: Vec<String>,
    move_count: usize,
    in_check: bool,
    is_checkmate: bool,
    is_stalemate: bool,
    is_draw: bool,
    is_game_over: bool,
}

fn strip_game_over_suffix(san: &str) -> &str {
    san.trim_end_matches(['#', '='])
}

#[test]
fn cross_engine_move_generation_matches_gungi_js() {
    let fixture_path = format!(
        "{}/tests/fixtures/cross_engine_vectors.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture = std::fs::read_to_string(&fixture_path).expect("read fixture");
    let vectors: Vec<TestVector> = serde_json::from_str(&fixture).expect("parse fixture");

    let mut passed = 0;
    let mut failed = 0;

    for (i, vector) in vectors.iter().enumerate() {
        let gungi = match Gungi::from_fen(&vector.fen) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[{i}] FEN parse error: {e} — FEN: {}", vector.fen);
                failed += 1;
                continue;
            }
        };

        let moves = gungi.moves();
        let mut rust_sans: Vec<String> = moves
            .iter()
            .map(|m| strip_game_over_suffix(&move_to_san(m)).to_string())
            .collect();
        rust_sans.sort();
        rust_sans.dedup();

        if rust_sans != vector.moves {
            let rust_set: std::collections::BTreeSet<&str> =
                rust_sans.iter().map(|s| s.as_str()).collect();
            let js_set: std::collections::BTreeSet<&str> =
                vector.moves.iter().map(|s| s.as_str()).collect();

            let only_rust: Vec<&&str> = rust_set.difference(&js_set).collect();
            let only_js: Vec<&&str> = js_set.difference(&rust_set).collect();

            eprintln!(
                "[{i}] MOVE MISMATCH — FEN: {} | rust={} js={}",
                vector.fen,
                rust_sans.len(),
                vector.moves.len()
            );
            if !only_rust.is_empty() {
                eprintln!("  only in komugi: {:?}", only_rust);
            }
            if !only_js.is_empty() {
                eprintln!("  only in gungi.js: {:?}", only_js);
            }
            failed += 1;
            continue;
        }

        passed += 1;
    }

    eprintln!(
        "\nCross-engine: {passed} passed, {failed} failed out of {} vectors",
        vectors.len()
    );
    assert_eq!(failed, 0, "{failed} vectors had move generation mismatches");
}

#[test]
fn cross_engine_game_state_matches_gungi_js() {
    let fixture_path = format!(
        "{}/tests/fixtures/cross_engine_vectors.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture = std::fs::read_to_string(&fixture_path).expect("read fixture");
    let vectors: Vec<TestVector> = serde_json::from_str(&fixture).expect("parse fixture");

    let mut passed = 0;
    let mut failed = 0;

    for (i, vector) in vectors.iter().enumerate() {
        let gungi = match Gungi::from_fen(&vector.fen) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[{i}] FEN parse error: {e} — FEN: {}", vector.fen);
                failed += 1;
                continue;
            }
        };

        let mut errors = Vec::new();

        if gungi.in_check() != vector.in_check {
            errors.push(format!(
                "in_check: rust={} js={}",
                gungi.in_check(),
                vector.in_check
            ));
        }
        if gungi.is_checkmate() != vector.is_checkmate {
            errors.push(format!(
                "is_checkmate: rust={} js={}",
                gungi.is_checkmate(),
                vector.is_checkmate
            ));
        }
        if gungi.is_stalemate() != vector.is_stalemate {
            errors.push(format!(
                "is_stalemate: rust={} js={}",
                gungi.is_stalemate(),
                vector.is_stalemate
            ));
        }
        if gungi.is_draw() != vector.is_draw {
            errors.push(format!(
                "is_draw: rust={} js={}",
                gungi.is_draw(),
                vector.is_draw
            ));
        }
        if gungi.is_game_over() != vector.is_game_over {
            errors.push(format!(
                "is_game_over: rust={} js={}",
                gungi.is_game_over(),
                vector.is_game_over
            ));
        }

        if !errors.is_empty() {
            eprintln!("[{i}] STATE MISMATCH — FEN: {}", vector.fen);
            for e in &errors {
                eprintln!("  {e}");
            }
            failed += 1;
        } else {
            passed += 1;
        }
    }

    eprintln!(
        "\nCross-engine state: {passed} passed, {failed} failed out of {} vectors",
        vectors.len()
    );
    assert_eq!(failed, 0, "{failed} vectors had game state mismatches");
}
