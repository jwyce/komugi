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
use komugi_engine::{
    accuracy_for_cpl, classify_move_in_game, format_score, is_sacrifice_move, mate_in_n,
    win_percent, win_percent_loss, AlphaBetaConfig, AlphaBetaSearcher, ClassicalEval,
    MoveClassification, NnueEval,
};
use serde::Serialize;

/// Embedded NNUE model (3.5MB placeholder)
static NNUE_BYTES: &[u8] = include_bytes!("../../../models/gungi.nnue");

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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisLine {
    moves: Vec<String>,
    score: i32,
    mate: Option<i32>,
    score_display: String,
    win_percent: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisResult {
    depth: u8,
    lines: Vec<AnalysisLine>,
    nodes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsEvaluation {
    score: i32,
    score_display: String,
    mate: Option<i32>,
    win_percent: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsClassification {
    classification: String,
    eval_before: i32,
    eval_after: i32,
    best_move: Option<String>,
    best_eval: i32,
    win_percent_loss: f64,
    is_sacrifice: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsMoveAnalysis {
    san: String,
    classification: String,
    eval_before: i32,
    eval_after: i32,
    best_move: Option<String>,
    best_eval: i32,
    score_display: String,
    win_percent: f64,
    top_lines: Vec<AnalysisLine>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsGameAnalysis {
    moves: Vec<JsMoveAnalysis>,
    white_accuracy: f64,
    black_accuracy: f64,
}

/// Main WASM-exported Gungi engine.
#[wasm_bindgen]
pub struct KomugiEngine {
    game: Gungi,
    searcher: AlphaBetaSearcher,
    evaluator: Box<dyn Evaluator>,
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
            evaluator: Box::new(ClassicalEval::new()),
        })
    }

    /// Create a new game with NNUE evaluator.
    #[wasm_bindgen(js_name = "newWithNnue")]
    pub fn new_with_nnue(mode: u8) -> Result<KomugiEngine, JsError> {
        let setup =
            SetupMode::from_code(mode).ok_or_else(|| JsError::new("invalid mode: expected 0-3"))?;
        let nnue = NnueEval::from_bytes(NNUE_BYTES)
            .map_err(|e| JsError::new(&format!("NNUE load failed: {:?}", e)))?;
        Ok(Self {
            game: Gungi::new(setup),
            searcher: AlphaBetaSearcher::new(AlphaBetaConfig::default()),
            evaluator: Box::new(nnue),
        })
    }

    /// Load a position from a FEN string.
    #[wasm_bindgen(js_name = "loadFen")]
    pub fn load_fen(fen: &str) -> Result<KomugiEngine, JsError> {
        let game = Gungi::from_fen(fen).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            game,
            searcher: AlphaBetaSearcher::new(AlphaBetaConfig::default()),
            evaluator: Box::new(ClassicalEval::new()),
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

    /// Evaluate the current position. Returns JSON: { score, scoreDisplay, mate, winPercent }.
    pub fn evaluate(&self) -> Result<JsValue, JsError> {
        let position =
            Position::from_fen(&self.game.fen()).map_err(|e| JsError::new(&e.to_string()))?;
        let score = self.evaluator.evaluate(&position).0;
        let result = JsEvaluation {
            score,
            score_display: format_score(score),
            mate: mate_in_n(score),
            win_percent: win_percent(score),
        };
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
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

    #[wasm_bindgen(js_name = "classifyMove")]
    pub fn classify_move(&mut self, san: &str, depth: u8, num_pv: u8) -> Result<JsValue, JsError> {
        let fen = self.game.fen();
        let position = Position::from_fen(&fen).map_err(|e| JsError::new(&e.to_string()))?;
        let parsed = komugi_core::fen::parse_fen(&fen).map_err(|e| JsError::new(&e.to_string()))?;
        let played = parse_san(san, &parsed).map_err(|e| JsError::new(&e.to_string()))?;

        let multi_pv = self.searcher.search_multi_pv(
            &position,
            SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            },
            num_pv.max(1) as usize,
        );
        let eval_before = self
            .searcher
            .search_with_info(
                &position,
                SearchLimits {
                    depth: Some(depth),
                    ..SearchLimits::default()
                },
            )
            .score
            .0;

        let classification =
            classify_move_in_game(&position, eval_before, &played, &multi_pv, None, None);

        let best_line = multi_pv.lines.first();
        let best_eval = best_line.map(|l| l.score.0).unwrap_or(0);
        let played_eval = multi_pv
            .lines
            .iter()
            .find(|l| l.moves.first() == Some(&played))
            .map(|l| l.score.0)
            .unwrap_or(best_eval);
        let eval_after = -played_eval;

        let result = JsClassification {
            classification: classification
                .map(|c| c.description().to_string())
                .unwrap_or_else(|| "Unknown".to_string()),
            eval_before,
            eval_after,
            best_move: best_line.and_then(|l| l.moves.first()).map(move_to_san),
            best_eval,
            win_percent_loss: win_percent_loss(eval_before, eval_after),
            is_sacrifice: is_sacrifice_move(&position, &played),
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "analyzeGame")]
    pub fn analyze_game(
        &mut self,
        moves_json: &str,
        depth: u8,
        num_pv: u8,
    ) -> Result<JsValue, JsError> {
        let sans: Vec<String> = serde_json::from_str(moves_json)
            .map_err(|e| JsError::new(&format!("invalid JSON: {}", e)))?;

        let position =
            Position::from_fen(&self.game.fen()).map_err(|e| JsError::new(&e.to_string()))?;
        let mut replay = Gungi::new(position.mode);
        let mut searcher = AlphaBetaSearcher::new(AlphaBetaConfig::default());
        let pv_count = num_pv.max(1) as usize;

        let mut moves_result = Vec::new();
        let mut white_prev: Option<MoveClassification> = None;
        let mut black_prev: Option<MoveClassification> = None;
        let mut white_acc_sum = 0.0;
        let mut white_count = 0u32;
        let mut black_acc_sum = 0.0;
        let mut black_count = 0u32;

        for san in &sans {
            let fen = replay.fen();
            let pos = Position::from_fen(&fen).map_err(|e| JsError::new(&e.to_string()))?;
            let parsed =
                komugi_core::fen::parse_fen(&fen).map_err(|e| JsError::new(&e.to_string()))?;
            let played = parse_san(san, &parsed).map_err(|e| JsError::new(&e.to_string()))?;

            let multi_pv = searcher.search_multi_pv(
                &pos,
                SearchLimits {
                    depth: Some(depth),
                    ..SearchLimits::default()
                },
                pv_count,
            );
            let eval_before = searcher
                .search_with_info(
                    &pos,
                    SearchLimits {
                        depth: Some(depth),
                        ..SearchLimits::default()
                    },
                )
                .score
                .0;

            let (prev, opp_prev) = match pos.turn {
                Color::White => (white_prev, black_prev),
                Color::Black => (black_prev, white_prev),
            };

            let classification =
                classify_move_in_game(&pos, eval_before, &played, &multi_pv, prev, opp_prev);

            let best_line = multi_pv.lines.first();
            let best_eval = best_line.map(|l| l.score.0).unwrap_or(0);
            let played_eval = multi_pv
                .lines
                .iter()
                .find(|l| l.moves.first() == Some(&played))
                .map(|l| l.score.0)
                .unwrap_or(best_eval);
            let eval_after = -played_eval;

            let cpl = (best_eval - eval_after).max(0) as f64;
            match pos.turn {
                Color::White => {
                    white_acc_sum += accuracy_for_cpl(cpl);
                    white_count += 1;
                }
                Color::Black => {
                    black_acc_sum += accuracy_for_cpl(cpl);
                    black_count += 1;
                }
            }

            let top_lines: Vec<AnalysisLine> = multi_pv
                .lines
                .iter()
                .map(|line| AnalysisLine {
                    moves: line.moves.iter().map(move_to_san).collect(),
                    score: line.score.0,
                    mate: mate_in_n(line.score.0),
                    score_display: format_score(line.score.0),
                    win_percent: win_percent(line.score.0),
                })
                .collect();

            moves_result.push(JsMoveAnalysis {
                san: san.clone(),
                classification: classification
                    .map(|c| c.description().to_string())
                    .unwrap_or_else(|| "Unknown".to_string()),
                eval_before,
                eval_after,
                best_move: best_line.and_then(|l| l.moves.first()).map(move_to_san),
                best_eval,
                score_display: format_score(eval_after),
                win_percent: win_percent(eval_after),
                top_lines,
            });

            if let Some(cls) = classification {
                match pos.turn {
                    Color::White => white_prev = Some(cls),
                    Color::Black => black_prev = Some(cls),
                }
            }

            replay
                .make_move(&played)
                .map_err(|e| JsError::new(&e.to_string()))?;
        }

        let result = JsGameAnalysis {
            moves: moves_result,
            white_accuracy: if white_count > 0 {
                white_acc_sum / white_count as f64
            } else {
                100.0
            },
            black_accuracy: if black_count > 0 {
                black_acc_sum / black_count as f64
            } else {
                100.0
            },
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "loadFenWithNnue")]
    pub fn load_fen_with_nnue(fen: &str) -> Result<KomugiEngine, JsError> {
        let game = Gungi::from_fen(fen).map_err(|e| JsError::new(&e.to_string()))?;
        let nnue_search = NnueEval::from_bytes(NNUE_BYTES)
            .map_err(|e| JsError::new(&format!("NNUE load failed: {:?}", e)))?;
        let nnue_eval = NnueEval::from_bytes(NNUE_BYTES)
            .map_err(|e| JsError::new(&format!("NNUE load failed: {:?}", e)))?;
        Ok(Self {
            game,
            searcher: AlphaBetaSearcher::with_eval(
                AlphaBetaConfig::default(),
                Box::new(nnue_search),
            ),
            evaluator: Box::new(nnue_eval),
        })
    }

    /// Multi-PV position analysis. TT persists between calls for progressive deepening.
    #[wasm_bindgen(js_name = "analyzePosition")]
    pub fn analyze_position(&mut self, depth: u8, num_pv: u8) -> Result<JsValue, JsError> {
        let position =
            Position::from_fen(&self.game.fen()).map_err(|e| JsError::new(&e.to_string()))?;

        let limits = SearchLimits {
            depth: Some(depth),
            nodes: None,
            time_ms: None,
        };

        let result = self
            .searcher
            .search_multi_pv(&position, limits, num_pv as usize);
        let turn = position.turn;

        let lines: Vec<AnalysisLine> = result
            .lines
            .iter()
            .map(|pv_line| {
                let san_moves: Vec<String> = pv_line.moves.iter().map(move_to_san).collect();

                let white_score = match turn {
                    Color::White => pv_line.score.0,
                    Color::Black => -pv_line.score.0,
                };

                AnalysisLine {
                    moves: san_moves,
                    score: white_score,
                    mate: mate_in_n(white_score),
                    score_display: format_score(white_score),
                    win_percent: win_percent(white_score),
                }
            })
            .collect();

        let depth_reached = result.lines.first().map(|l| l.depth).unwrap_or(depth);

        serde_wasm_bindgen::to_value(&AnalysisResult {
            depth: depth_reached,
            lines,
            nodes: result.total_nodes,
        })
        .map_err(|e| JsError::new(&e.to_string()))
    }
}
