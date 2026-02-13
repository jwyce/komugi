use wasm_bindgen::prelude::*;

use komugi_core::{
    eval::Evaluator,
    fen::ParsedFen,
    game::Gungi,
    position::Position,
    san::{move_to_san, parse_san},
    search::SearchLimits,
    types::{Color, MoveType, SetupMode},
};
use komugi_engine::{AlphaBetaConfig, AlphaBetaSearcher, ClassicalEval};
use serde::Serialize;

/// Initialize panic hook for readable error messages in browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Serializable move representation for JS consumers.
#[derive(Serialize)]
struct JsMove {
    san: String,
    piece: String,
    from: Option<String>,
    to: String,
    move_type: String,
    draft_finished: bool,
}

fn move_type_str(mt: MoveType) -> &'static str {
    match mt {
        MoveType::Route => "route",
        MoveType::Capture => "capture",
        MoveType::Tsuke => "tsuke",
        MoveType::Betray => "betray",
        MoveType::Arata => "arata",
    }
}

fn tiered_square_str(ts: &komugi_core::types::TieredSquare) -> String {
    format!("{}-{}-{}", ts.square.rank, ts.square.file, ts.tier)
}

/// Main WASM-exported Gungi engine.
#[wasm_bindgen]
pub struct KomugiEngine {
    game: Gungi,
    searcher: AlphaBetaSearcher,
    evaluator: ClassicalEval,
}

#[wasm_bindgen]
impl KomugiEngine {
    /// Create a new game with the given setup mode (0=Intro, 1=Beginner, 2=Intermediate, 3=Advanced).
    #[wasm_bindgen(constructor)]
    pub fn new(mode: u8) -> Result<KomugiEngine, JsError> {
        let setup =
            SetupMode::from_code(mode).ok_or_else(|| JsError::new("invalid mode: expected 0-3"))?;
        Ok(Self {
            game: Gungi::new(setup),
            searcher: AlphaBetaSearcher::new(AlphaBetaConfig::default()),
            evaluator: ClassicalEval::new(),
        })
    }

    /// Load a position from a FEN string.
    #[wasm_bindgen(js_name = "loadFen")]
    pub fn load_fen(fen: &str) -> Result<KomugiEngine, JsError> {
        let game = Gungi::from_fen(fen).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            game,
            searcher: AlphaBetaSearcher::new(AlphaBetaConfig::default()),
            evaluator: ClassicalEval::new(),
        })
    }

    /// Get the current position as a FEN string.
    pub fn fen(&self) -> String {
        self.game.fen()
    }

    /// Get all legal moves as a JSON array.
    #[wasm_bindgen(js_name = "legalMoves")]
    pub fn legal_moves(&self) -> Result<JsValue, JsError> {
        let moves = self.game.moves();
        let js_moves: Vec<JsMove> = moves
            .iter()
            .map(|mv| JsMove {
                san: move_to_san(mv),
                piece: mv.piece.kanji().to_string(),
                from: mv.from.as_ref().map(tiered_square_str),
                to: tiered_square_str(&mv.to),
                move_type: move_type_str(mv.move_type).to_string(),
                draft_finished: mv.draft_finished,
            })
            .collect();

        serde_wasm_bindgen::to_value(&js_moves).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Make a move by SAN notation.
    #[wasm_bindgen(js_name = "makeMove")]
    pub fn make_move(&mut self, san: &str) -> Result<(), JsError> {
        let position =
            Position::from_fen(&self.game.fen()).map_err(|e| JsError::new(&e.to_string()))?;
        let parsed_fen = ParsedFen {
            board: position.board.clone(),
            hand: position.hand.clone(),
            turn: position.turn,
            mode: position.mode,
            drafting: position.drafting_rights,
            move_number: position.move_number,
        };

        let mv = parse_san(san, &parsed_fen).map_err(|e| JsError::new(&e.to_string()))?;
        self.game
            .make_move(&mv)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Undo the last move. Returns true if successful.
    pub fn undo(&mut self) -> bool {
        self.game.undo().is_ok()
    }

    /// Static evaluation of the current position in centipawns (positive = white advantage).
    pub fn evaluate(&self) -> i32 {
        let position = match Position::from_fen(&self.game.fen()) {
            Ok(p) => p,
            Err(_) => return 0,
        };
        self.evaluator.evaluate(&position).0
    }

    /// Search for the best move at the given depth. Returns SAN notation.
    #[wasm_bindgen(js_name = "bestMove")]
    pub fn best_move(&mut self, depth: u8) -> Result<String, JsError> {
        let position =
            Position::from_fen(&self.game.fen()).map_err(|e| JsError::new(&e.to_string()))?;

        let limits = SearchLimits {
            depth: Some(depth),
            nodes: None,
            time_ms: None,
        };

        let result = self.searcher.search_with_info(&position, limits);
        match result.best_move {
            Some(mv) => Ok(move_to_san(&mv)),
            None => Err(JsError::new("no legal moves")),
        }
    }

    /// Returns true if the game is over.
    #[wasm_bindgen(js_name = "isGameOver")]
    pub fn is_game_over(&self) -> bool {
        self.game.is_game_over()
    }

    /// Returns true if the current side is in checkmate.
    #[wasm_bindgen(js_name = "isCheckmate")]
    pub fn is_checkmate(&self) -> bool {
        self.game.is_checkmate()
    }

    /// Returns true if the position is a stalemate.
    #[wasm_bindgen(js_name = "isStalemate")]
    pub fn is_stalemate(&self) -> bool {
        self.game.is_stalemate()
    }

    /// Returns true if the position is a draw.
    #[wasm_bindgen(js_name = "isDraw")]
    pub fn is_draw(&self) -> bool {
        self.game.is_draw()
    }

    /// Returns true if the current side is in check.
    #[wasm_bindgen(js_name = "inCheck")]
    pub fn in_check(&self) -> bool {
        self.game.in_check()
    }

    /// Returns "w" or "b" for the side to move.
    pub fn turn(&self) -> String {
        match self.game.turn() {
            Color::White => "w".to_string(),
            Color::Black => "b".to_string(),
        }
    }

    /// Configure the transposition table size in megabytes.
    #[wasm_bindgen(js_name = "setTtSize")]
    pub fn set_tt_size(&mut self, mb: u32) {
        self.searcher = AlphaBetaSearcher::new(AlphaBetaConfig {
            tt_size_mb: mb as usize,
            ..AlphaBetaConfig::default()
        });
    }

    /// Get the current full move number.
    #[wasm_bindgen(js_name = "moveNumber")]
    pub fn move_number(&self) -> u32 {
        self.game.move_number()
    }

    /// Returns true if the position has insufficient material for checkmate.
    #[wasm_bindgen(js_name = "isInsufficientMaterial")]
    pub fn is_insufficient_material(&self) -> bool {
        self.game.is_insufficient_material()
    }

    /// Returns true if fourfold repetition has occurred.
    #[wasm_bindgen(js_name = "isFourfoldRepetition")]
    pub fn is_fourfold_repetition(&self) -> bool {
        self.game.is_fourfold_repetition()
    }
}
