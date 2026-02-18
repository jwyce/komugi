use komugi_core::{Color, Position, SearchLimits, SetupMode, SQUARES};
use komugi_engine::{
    mcts::HeuristicPolicy, play_game, MctsConfig, MctsSearcher, SelfPlayConfig, ENCODING_SIZE,
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
