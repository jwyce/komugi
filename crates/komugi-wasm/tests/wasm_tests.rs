#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use komugi_wasm::KomugiEngine;

#[wasm_bindgen_test]
fn create_intro_game() {
    let engine = KomugiEngine::new(0).unwrap();
    assert_eq!(engine.turn(), "w");
    assert!(!engine.is_game_over());
    assert!(!engine.is_checkmate());
}

#[wasm_bindgen_test]
fn create_all_modes() {
    for mode in 0..=3u8 {
        let engine = KomugiEngine::new(mode).unwrap();
        assert!(!engine.fen().is_empty());
    }
}

#[wasm_bindgen_test]
fn invalid_mode_errors() {
    assert!(KomugiEngine::new(4).is_err());
    assert!(KomugiEngine::new(255).is_err());
}

#[wasm_bindgen_test]
fn fen_round_trip() {
    let engine = KomugiEngine::new(0).unwrap();
    let fen = engine.fen();
    let engine2 = KomugiEngine::load_fen(&fen).unwrap();
    assert_eq!(engine.fen(), engine2.fen());
}

#[wasm_bindgen_test]
fn invalid_fen_errors() {
    assert!(KomugiEngine::load_fen("garbage").is_err());
}

#[wasm_bindgen_test]
fn legal_moves_returns_array() {
    let engine = KomugiEngine::new(0).unwrap();
    let moves = engine.legal_moves().unwrap();
    assert!(js_sys::Array::is_array(&moves));
    let arr = js_sys::Array::from(&moves);
    assert!(arr.length() > 0);
}

#[wasm_bindgen_test]
fn make_move_and_undo() {
    let mut engine = KomugiEngine::new(0).unwrap();
    let original_fen = engine.fen();

    let moves = engine.legal_moves().unwrap();
    let arr = js_sys::Array::from(&moves);
    let first = arr.get(0);
    let san: String = js_sys::Reflect::get(&first, &"san".into())
        .unwrap()
        .as_string()
        .unwrap();

    engine.make_move(&san).unwrap();
    assert_ne!(engine.fen(), original_fen);
    assert_eq!(engine.turn(), "b");

    assert!(engine.undo());
    assert_eq!(engine.fen(), original_fen);
    assert_eq!(engine.turn(), "w");
}

#[wasm_bindgen_test]
fn undo_on_empty_history_returns_false() {
    let mut engine = KomugiEngine::new(0).unwrap();
    assert!(!engine.undo());
}

#[wasm_bindgen_test]
fn illegal_move_errors() {
    let mut engine = KomugiEngine::new(0).unwrap();
    assert!(engine.make_move("definitely_not_a_move").is_err());
}

#[wasm_bindgen_test]
fn evaluate_returns_score() {
    let engine = KomugiEngine::new(0).unwrap();
    let _score = engine.evaluate();
}

#[wasm_bindgen_test]
fn best_move_returns_san() {
    let mut engine = KomugiEngine::new(0).unwrap();
    let san = engine.best_move(1).unwrap();
    assert!(!san.is_empty());
}

#[wasm_bindgen_test]
fn set_tt_size_does_not_panic() {
    let mut engine = KomugiEngine::new(0).unwrap();
    engine.set_tt_size(1);
    engine.set_tt_size(32);
}

#[wasm_bindgen_test]
fn move_number_starts_at_one() {
    let engine = KomugiEngine::new(0).unwrap();
    assert_eq!(engine.move_number(), 1);
}

#[wasm_bindgen_test]
fn game_state_queries() {
    let engine = KomugiEngine::new(0).unwrap();
    assert!(!engine.is_draw());
    assert!(!engine.is_stalemate());
    assert!(!engine.is_insufficient_material());
    assert!(!engine.is_fourfold_repetition());
}
