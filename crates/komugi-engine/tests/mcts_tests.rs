use std::sync::{Arc, Mutex};

use komugi_core::{Color, Move, Policy, Position, SearchLimits, SetupMode, SQUARES};
use komugi_engine::{
    mcts::HeuristicPolicy, play_game, play_games_with_broker, MctsConfig, MctsSearcher,
    SelfPlayConfig, ENCODING_SIZE,
};

#[test]
fn mcts_returns_legal_move() {
    let position = Position::new(SetupMode::Beginner);
    let mut searcher = MctsSearcher::new(MctsConfig {
        max_simulations: 100,
        ..MctsConfig::default()
    });

    let result = searcher.search_with_policy(&position, SearchLimits::default(), &HeuristicPolicy);

    let best_move = result.best_move.expect("MCTS should return a move");
    assert!(position.moves().iter().any(|mv| mv == &best_move));
    assert!(result.nodes_searched > 0);
}

#[test]
fn mcts_returns_legal_move_in_intermediate_draft() {
    let position = Position::new(SetupMode::Intermediate);
    let mut searcher = MctsSearcher::new(MctsConfig {
        max_simulations: 100,
        ..MctsConfig::default()
    });

    let result = searcher.search_with_policy(&position, SearchLimits::default(), &HeuristicPolicy);

    let best_move = result.best_move.expect("MCTS should return a draft move");
    assert!(position.moves().iter().any(|mv| mv == &best_move));
    assert!(result.nodes_searched > 0);
}

#[test]
fn mcts_finds_winning_capture() {
    let position = Position::from_fen("8m/9/9/9/4Gd3/9/9/9/M8 -/- w 3 - 1").unwrap();
    let legal_moves = position.moves();
    let before_black_pieces = count_pieces(&position, Color::Black);

    let mut searcher = MctsSearcher::new(MctsConfig {
        max_simulations: 1000,
        dirichlet_epsilon: 0.0,
        temperature: 0.0,
        ..MctsConfig::default()
    });

    let result = searcher.search_with_policy(&position, SearchLimits::default(), &HeuristicPolicy);

    let best_move = result.best_move.expect("MCTS should find a move");
    assert!(legal_moves.iter().any(|mv| mv == &best_move));

    let mut test_pos = position.clone();
    test_pos.make_move(&best_move).unwrap();
    let after_black_pieces = count_pieces(&test_pos, Color::Black);

    // With enough simulations, MCTS should find the winning capture
    // The position has a clear winning capture available
    assert!(
        after_black_pieces < before_black_pieces,
        "MCTS should find the winning capture in this sparse position"
    );
}

#[test]
fn mcts_selfplay_produces_valid_data() {
    let config = SelfPlayConfig {
        mcts_config: MctsConfig {
            max_simulations: 50,
            ..MctsConfig::default()
        },
        setup_mode: SetupMode::Beginner,
        max_moves: 10,
        policy: std::sync::Arc::new(HeuristicPolicy),
    };

    let game = play_game(&config);

    assert!(!game.positions.is_empty());
    assert!(game.total_moves <= 10);

    for record in &game.positions {
        assert!(!record.fen.is_empty());
        assert!(!record.policy.is_empty());
        assert_eq!(record.encoding.len(), ENCODING_SIZE);
    }
}

fn count_pieces(position: &Position, color: Color) -> usize {
    let mut pieces = 0usize;
    for square in SQUARES {
        if let Some(tower) = position.board.get(square) {
            pieces += tower.iter().filter(|piece| piece.color == color).count();
        }
    }
    pieces
}

#[derive(Clone, Default)]
struct RecordingHeuristicPolicy {
    batch_sizes: Arc<Mutex<Vec<usize>>>,
}

impl RecordingHeuristicPolicy {
    fn batch_sizes(&self) -> Vec<usize> {
        self.batch_sizes.lock().unwrap().clone()
    }
}

impl Policy for RecordingHeuristicPolicy {
    fn prior(&self, position: &Position, moves: &[Move]) -> Vec<f32> {
        HeuristicPolicy.prior(position, moves)
    }

    fn prior_and_value_batch(
        &self,
        batch: &[(&Position, &[Move])],
    ) -> Vec<(Vec<f32>, Option<f32>)> {
        self.batch_sizes.lock().unwrap().push(batch.len());
        batch
            .iter()
            .map(|(position, moves)| (HeuristicPolicy.prior(position, moves), None))
            .collect()
    }
}

fn draft_game396_position() -> Position {
    use komugi_core::{MoveType, PieceType, Square};

    let mut position = Position::new(SetupMode::Intermediate);
    assert!(position.in_draft());

    let moves = position.moves();
    let mv1 = moves
        .iter()
        .find(|mv| {
            mv.piece == PieceType::Marshal
                && mv.to.square == Square::new_unchecked(7, 2)
                && mv.move_type == MoveType::Arata
                && !mv.draft_finished
        })
        .expect("White should be able to drop marshal at (7,2)");
    position.make_move(mv1).unwrap();

    let moves = position.moves();
    let mv2 = moves
        .iter()
        .find(|mv| {
            mv.piece == PieceType::Marshal
                && mv.to.square == Square::new_unchecked(3, 2)
                && mv.move_type == MoveType::Arata
                && !mv.draft_finished
        })
        .expect("Black should be able to drop marshal at (3,2)");
    position.make_move(mv2).unwrap();

    let moves = position.moves();
    let mv3 = moves
        .iter()
        .find(|mv| {
            mv.piece == PieceType::LieutenantGeneral
                && mv.to.square == Square::new_unchecked(7, 6)
                && mv.move_type == MoveType::Arata
                && !mv.draft_finished
        })
        .expect("White should be able to drop lieutenant at (7,6)");
    position.make_move(mv3).unwrap();

    assert!(position.in_draft());
    assert_eq!(position.turn, Color::Black);
    position
}

#[test]
fn vl_batch_mcts_returns_move_in_draft_game396() {
    let position = draft_game396_position();
    let legal_moves = position.moves();
    assert!(
        legal_moves.len() > 100,
        "Expected many legal moves, got {}",
        legal_moves.len(),
    );

    let mut searcher = MctsSearcher::new(MctsConfig {
        max_simulations: 400,
        vl_batch_size: 8,
        ..MctsConfig::default()
    });

    let result = searcher.search_with_policy(&position, SearchLimits::default(), &HeuristicPolicy);

    assert!(
        result.best_move.is_some(),
        "VL-batched MCTS must return a move for draft position with {} legal moves (nodes_searched: {})",
        legal_moves.len(),
        result.nodes_searched,
    );
    assert!(
        result.nodes_searched >= 100,
        "Should complete many simulations"
    );
}

#[test]
fn vl_batch_mcts_retries_frontier_after_collision_start() {
    let position = draft_game396_position();
    let policy = RecordingHeuristicPolicy::default();
    let mut searcher = MctsSearcher::new(MctsConfig {
        max_simulations: 24,
        vl_batch_size: 8,
        dirichlet_epsilon: 0.0,
        temperature: 0.0,
        ..MctsConfig::default()
    });

    let result = searcher.search_with_policy(&position, SearchLimits::default(), &policy);
    let batch_sizes = policy.batch_sizes();

    assert_eq!(result.nodes_searched, 24);
    assert_eq!(batch_sizes.first().copied(), Some(1));
    assert!(
        batch_sizes.get(1).copied().unwrap_or_default() >= 3,
        "Expected retries to recover multiple pending leaves after the collision-heavy opening batch, got {:?}",
        batch_sizes,
    );
    assert!(
        batch_sizes.iter().any(|&batch_size| batch_size == 8),
        "Expected at least one full VL batch after the frontier widens, got {:?}",
        batch_sizes,
    );
}

#[test]
fn vl_batch_mcts_collision_heavy_search_still_returns_legal_move() {
    let position = draft_game396_position();
    let legal_moves = position.moves();
    let policy = RecordingHeuristicPolicy::default();
    let mut searcher = MctsSearcher::new(MctsConfig {
        max_simulations: 24,
        vl_batch_size: 8,
        dirichlet_epsilon: 0.0,
        temperature: 0.0,
        ..MctsConfig::default()
    });

    let result = searcher.search_with_policy(&position, SearchLimits::default(), &policy);

    let best_move = result
        .best_move
        .expect("VL-batched MCTS should still return a move after repeated collisions");
    assert!(legal_moves.iter().any(|mv| mv == &best_move));
    assert_eq!(result.nodes_searched, 24);
}

#[test]
fn vl_batch_selfplay_intermediate_no_draft_draws() {
    for game_idx in 0..3 {
        let config = SelfPlayConfig {
            mcts_config: MctsConfig {
                max_simulations: 50,
                vl_batch_size: 8,
                ..MctsConfig::default()
            },
            setup_mode: SetupMode::Intermediate,
            max_moves: 20,
            policy: std::sync::Arc::new(HeuristicPolicy),
        };

        let game = play_game(&config);

        if game.total_moves < 20 {
            if let Some(last) = game.positions.last() {
                let pos = Position::from_fen(&last.fen).unwrap();
                assert!(
                    !pos.in_draft() || pos.is_game_over(),
                    "Game {game_idx}: ended at ply {} during draft phase! FEN: {}",
                    game.total_moves,
                    last.fen,
                );
            }
        }
    }
}

#[test]
fn broker_runtime_advances_multiple_games_stepwise() {
    let config = SelfPlayConfig {
        mcts_config: MctsConfig {
            max_simulations: 32,
            vl_batch_size: 8,
            ..MctsConfig::default()
        },
        setup_mode: SetupMode::Beginner,
        max_moves: 12,
        policy: Arc::new(HeuristicPolicy),
    };

    let games = play_games_with_broker(&config, 4, 2);
    assert_eq!(games.len(), 4);
    assert_eq!(games[0].0, 1);
    assert_eq!(games[1].0, 2);
    assert_eq!(games[2].0, 3);
    assert_eq!(games[3].0, 4);
    assert!(games.iter().any(|(_, game)| game.total_moves > 0));
    assert!(games.iter().all(|(_, game)| game.total_moves <= 12));
}

#[test]
fn broker_runtime_preserves_full_vl_batches() {
    let policy = RecordingHeuristicPolicy::default();
    let config = SelfPlayConfig {
        mcts_config: MctsConfig {
            max_simulations: 24,
            vl_batch_size: 8,
            dirichlet_epsilon: 0.0,
            temperature: 0.0,
            ..MctsConfig::default()
        },
        setup_mode: SetupMode::Intermediate,
        max_moves: 4,
        policy: Arc::new(policy.clone()),
    };

    let games = play_games_with_broker(&config, 3, 2);
    let batch_sizes = policy.batch_sizes();

    assert_eq!(games.len(), 3);
    assert!(games.iter().all(|(_, game)| !game.positions.is_empty()));
    assert!(games.iter().all(|(_, game)| game.total_moves <= 4));
    assert!(
        batch_sizes.iter().any(|&batch_size| batch_size == 8),
        "Broker self-play should keep hitting full VL batches, got {:?}",
        batch_sizes,
    );
}
