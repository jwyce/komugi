use std::env;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

use komugi_core::{
    is_marshal_captured, Color, Evaluator, Move, MoveType, Policy, Position, Score, SearchLimits,
    SearchResult, Searcher,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::classical::ClassicalEval;
use crate::mcts_broker::{
    BrokerSearchId, BrokerSearchTask, PendingLeafTask, SearchGameIdentity, TreeOwnership,
};

const DEFAULT_MAX_SIMULATIONS: u32 = 800;
const DEFAULT_C_PUCT: f32 = 4.0;
const DEFAULT_DIRICHLET_CONCENTRATION: f32 = 10.0;
const DEFAULT_DIRICHLET_EPSILON: f32 = 0.25;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_TEMPERATURE_DROP_MOVE: u32 = 25;
const INITIAL_PROGRESSIVE_WIDTH: usize = 12;
const PROGRESSIVE_WIDTH_STEP: usize = 4;
const BROKER_DEBUG_ENV: &str = "KOMUGI_DEBUG_BROKER";
const MCTS_BATCH_DEBUG_ENV: &str = "KOMUGI_DEBUG_MCTS_BATCH";

fn env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .and_then(|value| match value.as_str() {
            "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "on" | "ON" => Some(true),
            "0" | "false" | "FALSE" | "False" | "no" | "NO" | "off" | "OFF" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn log_mcts_batch(requested: u32, pending: usize, collisions: u32, completed: u32) {
    eprintln!(
        "MCTS_BATCH requested={requested} pending={pending} collisions={collisions} completed={completed}"
    );
}

fn log_broker_pending(requested: u32, pending_leaves: usize, collisions: u32, completed: u32) {
    eprintln!(
        "BROKER_STAGE event=pending requested={requested} pending_leaves={pending_leaves} collisions={collisions} completed={completed}"
    );
}

fn log_broker_resume(resumed: usize, pending_leaves: usize, avg_us: u128, max_us: u128) {
    eprintln!(
        "BROKER_STAGE event=resume resumed={resumed} pending_leaves={pending_leaves} resume_avg_us={avg_us} resume_max_us={max_us}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use komugi_core::{PieceType, Policy, SetupMode, Square};

    fn draft_game396_position() -> Position {
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

        position
    }

    fn deterministic_broker_config(max_simulations: u32) -> MctsConfig {
        MctsConfig {
            max_simulations,
            vl_batch_size: 8,
            dirichlet_epsilon: 0.0,
            temperature: 0.0,
            ..MctsConfig::default()
        }
    }

    fn paused_batch_results(paused: &BrokerPausedSearch) -> Vec<(Vec<f32>, Option<f32>)> {
        let batch_items: Vec<(&Position, &[Move])> = paused
            .pending_leaves()
            .iter()
            .map(|leaf| (&leaf.position, leaf.moves.as_slice()))
            .collect();
        HeuristicPolicy.prior_and_value_batch(&batch_items)
    }

    #[test]
    fn broker_resume_keeps_runtime_and_tree_in_sync() {
        let position = draft_game396_position();
        let mut searcher = MctsSearcher::new(deterministic_broker_config(24));
        let mut runtime = searcher.start_broker_search(SearchLimits::default());

        let paused = match searcher.run_search_until_pause_or_finish(
            &position,
            BrokerSearchId::new(1),
            &mut runtime,
        ) {
            BrokerWorkerStep::Paused(paused) => paused,
            step => panic!("expected paused search, got {step:?}"),
        };

        assert!(paused.pending_len() > 0);

        let expected_simulations = paused.completed + paused.pending_len() as u32;
        let results = paused_batch_results(&paused);
        searcher.resume_paused_search(&mut runtime, paused, results);

        assert_eq!(runtime.simulations, expected_simulations);
        assert_eq!(searcher.arena[0].visits, expected_simulations);
        assert!(searcher.arena.iter().all(|node| node.in_flight == 0));
        assert!(!searcher.get_root_policy().is_empty());
    }

    #[test]
    fn broker_finish_drains_pending_work_before_returning_result() {
        let position = draft_game396_position();
        let legal_moves = position.moves();
        let mut searcher = MctsSearcher::new(deterministic_broker_config(24));
        let mut runtime = searcher.start_broker_search(SearchLimits::default());
        let result = loop {
            match searcher.run_search_until_pause_or_finish(
                &position,
                BrokerSearchId::new(2),
                &mut runtime,
            ) {
                BrokerWorkerStep::Paused(paused) => {
                    let results = paused_batch_results(&paused);
                    searcher.resume_paused_search(&mut runtime, paused, results);
                }
                BrokerWorkerStep::Yielded => continue,
                BrokerWorkerStep::Finished(result) => break result,
            }
        };

        let best_move = result
            .best_move
            .expect("broker search should return a legal move after draining");
        assert!(legal_moves.iter().any(|mv| mv == &best_move));
        assert_eq!(result.nodes_searched, 24);
        assert_eq!(runtime.simulations, 24);
        assert_eq!(searcher.arena[0].visits, 24);
        assert!(searcher.arena.iter().all(|node| node.in_flight == 0));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub max_simulations: u32,
    pub time_limit_ms: Option<u64>,
    pub c_puct: f32,
    pub dirichlet_concentration: f32,
    pub dirichlet_epsilon: f32,
    pub temperature: f32,
    pub temperature_drop_move: u32,
    pub vl_batch_size: u32,
}

#[derive(Debug)]
pub(crate) struct BrokerSearchRuntime {
    max_simulations: u32,
    time_limit: Option<Duration>,
    started_at: Instant,
    simulations: u32,
    rng: StdRng,
}

impl BrokerSearchRuntime {
    fn reached_limit(&self) -> bool {
        self.simulations >= self.max_simulations
            || self
                .time_limit
                .is_some_and(|time_limit| self.started_at.elapsed() >= time_limit)
    }

    fn remaining_slots(&self) -> u32 {
        self.max_simulations.saturating_sub(self.simulations)
    }
}

#[derive(Debug)]
pub(crate) struct BrokerPausedSearch {
    requested: u32,
    completed: u32,
    collisions: u32,
    task: BrokerSearchTask,
}

impl BrokerPausedSearch {
    pub(crate) fn pending_len(&self) -> usize {
        self.task.pending_len()
    }

    pub(crate) fn pending_leaves(&self) -> &[PendingLeafTask] {
        self.task.pending_leaves()
    }
}

#[derive(Debug)]
pub(crate) enum BrokerWorkerStep {
    Paused(BrokerPausedSearch),
    Finished(SearchResult),
    Yielded,
}

#[derive(Debug)]
struct CollectedBatch {
    requested: u32,
    completed: u32,
    collisions: u32,
    task: BrokerSearchTask,
}

#[derive(Debug)]
struct ResumeStats {
    requested: u32,
    collisions: u32,
    completed_before_resume: u32,
    completed: u32,
    resumed: usize,
    pending_count: usize,
    avg_us: u128,
    max_us: u128,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_simulations: DEFAULT_MAX_SIMULATIONS,
            time_limit_ms: None,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_concentration: DEFAULT_DIRICHLET_CONCENTRATION,
            dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
            temperature: DEFAULT_TEMPERATURE,
            temperature_drop_move: DEFAULT_TEMPERATURE_DROP_MOVE,
            vl_batch_size: 1,
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    parent: Option<usize>,
    children: Vec<usize>,
    all_moves: Vec<(Move, f32)>,
    active_children_count: usize,
    mv: Option<Move>,
    visits: u32,
    total_value: f64,
    in_flight: u32,
    prior: f32,
    is_expanded: bool,
    is_terminal: bool,
}

impl Node {
    fn root() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            all_moves: Vec::new(),
            active_children_count: 0,
            mv: None,
            visits: 0,
            total_value: 0.0,
            in_flight: 0,
            prior: 1.0,
            is_expanded: false,
            is_terminal: false,
        }
    }

    fn child(parent: usize, mv: Move, prior: f32) -> Self {
        Self {
            parent: Some(parent),
            children: Vec::new(),
            all_moves: Vec::new(),
            active_children_count: 0,
            mv: Some(mv),
            visits: 0,
            total_value: 0.0,
            in_flight: 0,
            prior,
            is_expanded: false,
            is_terminal: false,
        }
    }
}

pub struct HeuristicPolicy;

impl Policy for HeuristicPolicy {
    fn prior(&self, _position: &Position, moves: &[Move]) -> Vec<f32> {
        if moves.is_empty() {
            return Vec::new();
        }

        let logits = moves
            .iter()
            .map(|mv| match mv.move_type {
                MoveType::Capture => 3.0f32,
                MoveType::Betray => 4.0f32,
                MoveType::Arata => 0.5f32,
                MoveType::Route | MoveType::Tsuke => 0.0f32,
            })
            .collect::<Vec<_>>();

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = Vec::with_capacity(logits.len());
        let mut sum = 0.0f32;

        for logit in logits {
            let value = (logit - max_logit).exp();
            probs.push(value);
            sum += value;
        }

        if sum <= f32::EPSILON {
            return vec![1.0 / moves.len() as f32; moves.len()];
        }

        for prob in &mut probs {
            *prob /= sum;
        }

        probs
    }
}

#[derive(Debug)]
pub struct MctsSearcher {
    config: MctsConfig,
    arena: Vec<Node>,
    eval: ClassicalEval,
}

impl MctsSearcher {
    pub fn new(config: MctsConfig) -> Self {
        Self {
            config,
            arena: Vec::new(),
            eval: ClassicalEval::new(),
        }
    }

    pub fn search_with_policy(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        policy: &dyn Policy,
    ) -> SearchResult {
        if self.config.vl_batch_size <= 1 {
            self.arena.clear();
            self.arena.push(Node::root());

            let max_simulations = limits
                .nodes
                .map(|nodes| nodes.min(u64::from(u32::MAX)) as u32)
                .unwrap_or(self.config.max_simulations);
            let time_limit = limits
                .time_ms
                .or(self.config.time_limit_ms)
                .map(Duration::from_millis);

            let started_at = Instant::now();
            let mut simulations = 0u32;
            let mut rng = rand::thread_rng();

            while simulations < max_simulations {
                if time_limit.is_some_and(|limit| started_at.elapsed() >= limit) {
                    break;
                }
                self.run_simulation(position, policy, &mut rng);
                simulations = simulations.saturating_add(1);
            }

            return SearchResult {
                best_move: self.select_root_move(position.move_number, &mut rng),
                score: self.root_score(),
                nodes_searched: u64::from(simulations),
            };
        }

        let mut runtime = self.start_broker_search(limits);
        let legacy_search_id = BrokerSearchId::new(0);

        loop {
            match self.run_search_until_pause_or_finish(position, legacy_search_id, &mut runtime) {
                BrokerWorkerStep::Paused(paused) => {
                    let batch_items: Vec<(&Position, &[Move])> = paused
                        .pending_leaves()
                        .iter()
                        .map(|leaf| (&leaf.position, leaf.moves.as_slice()))
                        .collect();
                    let results = policy.prior_and_value_batch(&batch_items);
                    self.resume_paused_search(&mut runtime, paused, results);
                }
                BrokerWorkerStep::Yielded => continue,
                BrokerWorkerStep::Finished(result) => return result,
            }
        }
    }

    pub fn get_root_policy(&self) -> Vec<(Move, f32)> {
        let Some(root) = self.arena.first() else {
            return Vec::new();
        };
        if root.children.is_empty() {
            return Vec::new();
        }

        let total_visits: f32 = root
            .children
            .iter()
            .map(|&child_idx| self.arena[child_idx].visits as f32)
            .sum();
        if total_visits <= f32::EPSILON {
            let uniform = 1.0 / root.children.len() as f32;
            return root
                .children
                .iter()
                .filter_map(|&child_idx| self.arena[child_idx].mv.clone().map(|mv| (mv, uniform)))
                .collect();
        }

        root.children
            .iter()
            .filter_map(|&child_idx| {
                self.arena[child_idx].mv.clone().map(|mv| {
                    let proportion = self.arena[child_idx].visits as f32 / total_visits;
                    (mv, proportion)
                })
            })
            .collect()
    }

    fn run_simulation(
        &mut self,
        root_position: &Position,
        policy: &dyn Policy,
        rng: &mut impl Rng,
    ) {
        let mut working = root_position.clone();
        let mut node_idx = 0usize;

        loop {
            if self.arena[node_idx].is_terminal || !self.arena[node_idx].is_expanded {
                break;
            }

            self.progressive_widen(node_idx);
            if self.arena[node_idx].children.is_empty() {
                break;
            }

            let child_idx = self.select_child(node_idx);
            let mv = self.arena[child_idx]
                .mv
                .as_ref()
                .expect("child node must contain a move");
            working.make_move(mv).unwrap_or_else(|err| {
                panic!(
                    "tree child move must stay legal: {err:?}; move={mv:?}; fen={}",
                    working.fen()
                )
            });
            node_idx = child_idx;
        }

        let value = if self.arena[node_idx].is_terminal {
            self.terminal_value(&working)
        } else {
            self.expand_and_evaluate(node_idx, &working, policy, node_idx == 0, rng)
        };

        self.backpropagate(node_idx, value);
    }

    fn expand_and_evaluate(
        &mut self,
        node_idx: usize,
        position: &Position,
        policy: &dyn Policy,
        apply_root_noise: bool,
        rng: &mut impl Rng,
    ) -> f64 {
        // Quick terminal checks that don't require move generation.
        if !position.in_draft() {
            if is_marshal_captured(position) {
                let node = &mut self.arena[node_idx];
                node.is_terminal = true;
                node.is_expanded = true;
                node.children.clear();
                node.all_moves.clear();
                node.active_children_count = 0;
                return -1.0;
            }
            if position.is_fourfold_repetition() || position.is_insufficient_material() {
                let node = &mut self.arena[node_idx];
                node.is_terminal = true;
                node.is_expanded = true;
                node.children.clear();
                node.all_moves.clear();
                node.active_children_count = 0;
                return 0.0;
            }
        }

        // Generate moves ONCE — used for both terminal detection and expansion.
        let moves = position.moves().into_iter().collect::<Vec<_>>();
        if moves.is_empty() {
            let node = &mut self.arena[node_idx];
            node.is_terminal = true;
            node.is_expanded = true;
            node.children.clear();
            node.all_moves.clear();
            node.active_children_count = 0;
            return if position.in_check(None) { -1.0 } else { 0.0 };
        }

        let (raw_priors, neural_value) = policy.prior_and_value(position, &moves);
        let mut priors = self.sanitize_priors(raw_priors, moves.len());
        if apply_root_noise {
            let alpha = self.config.dirichlet_concentration / moves.len() as f32;
            self.add_dirichlet_noise(&mut priors, alpha, rng);
        }

        let mut move_priors = moves.into_iter().zip(priors).collect::<Vec<(Move, f32)>>();
        move_priors.sort_by(|a, b| b.1.total_cmp(&a.1));

        let initial_count = self.initial_width().min(move_priors.len());
        let initial_moves = move_priors[..initial_count].to_vec();
        let mut children = Vec::with_capacity(initial_count);

        for (mv, prior) in initial_moves {
            let child_idx = self.arena.len();
            self.arena.push(Node::child(node_idx, mv, prior));
            children.push(child_idx);
        }

        let node = &mut self.arena[node_idx];
        node.children = children;
        node.all_moves = move_priors;
        node.active_children_count = initial_count;
        node.is_expanded = true;
        node.is_terminal = false;

        neural_value.map_or_else(
            || self.evaluate_value(position),
            |v| {
                let fv = f64::from(v);
                if fv.is_finite() {
                    fv.clamp(-1.0, 1.0)
                } else {
                    self.evaluate_value(position)
                }
            },
        )
    }

    fn select_child(&self, parent_idx: usize) -> usize {
        let parent = &self.arena[parent_idx];
        let parent_eff = f64::from((parent.visits + parent.in_flight).max(1)).sqrt();

        let mut best_idx = parent.children[0];
        let mut best_score = f64::NEG_INFINITY;

        for &child_idx in &parent.children {
            let child = &self.arena[child_idx];
            let child_eff = child.visits + child.in_flight;

            let q = if child_eff == 0 {
                0.0
            } else {
                -child.total_value / f64::from(child_eff)
            };
            let u = f64::from(self.config.c_puct) * f64::from(child.prior) * parent_eff
                / (1.0 + f64::from(child_eff));
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }

        best_idx
    }

    fn progressive_widen(&mut self, node_idx: usize) {
        let (current, total, target) = {
            let node = &self.arena[node_idx];
            let total = node.all_moves.len();
            let target = self.target_width(node.visits).min(total);
            (node.active_children_count, total, target)
        };

        if current >= total || current >= target {
            return;
        }

        let additions = {
            let node = &self.arena[node_idx];
            node.all_moves[current..target].to_vec()
        };

        let mut child_indices = Vec::with_capacity(additions.len());
        for (mv, prior) in additions {
            let child_idx = self.arena.len();
            self.arena.push(Node::child(node_idx, mv, prior));
            child_indices.push(child_idx);
        }

        let node = &mut self.arena[node_idx];
        node.children.extend(child_indices);
        node.active_children_count = target;
    }

    fn initial_width(&self) -> usize {
        INITIAL_PROGRESSIVE_WIDTH
    }

    fn target_width(&self, visits: u32) -> usize {
        let growth = (f64::from(visits).sqrt().floor() as usize) * PROGRESSIVE_WIDTH_STEP;
        self.initial_width().saturating_add(growth)
    }

    fn backpropagate(&mut self, mut node_idx: usize, mut value: f64) {
        loop {
            let node = &mut self.arena[node_idx];
            node.visits = node.visits.saturating_add(1);
            node.total_value += value;

            let Some(parent_idx) = node.parent else {
                break;
            };

            node_idx = parent_idx;
            value = -value;
        }
    }

    pub(crate) fn start_broker_search(&mut self, limits: SearchLimits) -> BrokerSearchRuntime {
        self.arena.clear();
        self.arena.push(Node::root());

        let max_simulations = limits
            .nodes
            .map(|nodes| nodes.min(u64::from(u32::MAX)) as u32)
            .unwrap_or(self.config.max_simulations);
        let time_limit = limits
            .time_ms
            .or(self.config.time_limit_ms)
            .map(Duration::from_millis);

        BrokerSearchRuntime {
            max_simulations,
            time_limit,
            started_at: Instant::now(),
            simulations: 0,
            rng: StdRng::from_entropy(),
        }
    }

    pub(crate) fn run_search_until_pause_or_finish(
        &mut self,
        root_position: &Position,
        search_id: BrokerSearchId,
        runtime: &mut BrokerSearchRuntime,
    ) -> BrokerWorkerStep {
        let debug_broker = env_bool(BROKER_DEBUG_ENV, false);
        let debug_batch = env_bool(MCTS_BATCH_DEBUG_ENV, false);
        let mut stalled_batches = 0u8;

        loop {
            if runtime.reached_limit() {
                return BrokerWorkerStep::Finished(
                    self.finish_broker_search(root_position.move_number, runtime),
                );
            }

            let remaining = runtime.remaining_slots();
            if remaining == 0 {
                return BrokerWorkerStep::Finished(
                    self.finish_broker_search(root_position.move_number, runtime),
                );
            }

            let requested = self.config.vl_batch_size.min(remaining).max(1);
            let collected = self.collect_broker_batch(root_position, requested, search_id);
            runtime.simulations = runtime.simulations.saturating_add(collected.completed);

            if debug_broker {
                log_broker_pending(
                    collected.requested,
                    collected.task.pending_len(),
                    collected.collisions,
                    collected.completed,
                );
            }

            if collected.task.pending_len() > 0 {
                return BrokerWorkerStep::Paused(BrokerPausedSearch {
                    requested: collected.requested,
                    completed: collected.completed,
                    collisions: collected.collisions,
                    task: collected.task,
                });
            }

            let resumed = self.resume_collected_batch(collected, Vec::new(), &mut runtime.rng);
            runtime.simulations = runtime.simulations.saturating_add(resumed.completed);

            if debug_broker {
                log_broker_resume(
                    resumed.resumed,
                    resumed.pending_count,
                    resumed.avg_us,
                    resumed.max_us,
                );
            }
            if debug_batch {
                log_mcts_batch(
                    resumed.requested,
                    resumed.pending_count,
                    resumed.collisions,
                    resumed.completed_before_resume + resumed.completed,
                );
            }

            if resumed.completed_before_resume == 0 && resumed.completed == 0 {
                stalled_batches = stalled_batches.saturating_add(1);
                if stalled_batches >= 2 {
                    return BrokerWorkerStep::Yielded;
                }
            } else {
                stalled_batches = 0;
            }
        }
    }

    pub(crate) fn resume_paused_search(
        &mut self,
        runtime: &mut BrokerSearchRuntime,
        paused: BrokerPausedSearch,
        results: Vec<(Vec<f32>, Option<f32>)>,
    ) {
        let debug_broker = env_bool(BROKER_DEBUG_ENV, false);
        let debug_batch = env_bool(MCTS_BATCH_DEBUG_ENV, false);

        let stats = self.resume_collected_batch(
            CollectedBatch {
                requested: paused.requested,
                completed: paused.completed,
                collisions: paused.collisions,
                task: paused.task,
            },
            results,
            &mut runtime.rng,
        );
        runtime.simulations = runtime.simulations.saturating_add(stats.completed);

        if debug_broker {
            log_broker_resume(
                stats.resumed,
                stats.pending_count,
                stats.avg_us,
                stats.max_us,
            );
        }
        if debug_batch {
            log_mcts_batch(
                stats.requested,
                stats.pending_count,
                stats.collisions,
                stats.completed_before_resume + stats.completed,
            );
        }
    }

    fn finish_broker_search(
        &self,
        move_number: u32,
        runtime: &mut BrokerSearchRuntime,
    ) -> SearchResult {
        SearchResult {
            best_move: self.select_root_move(move_number, &mut runtime.rng),
            score: self.root_score(),
            nodes_searched: u64::from(runtime.simulations),
        }
    }

    fn collect_broker_batch(
        &mut self,
        root_position: &Position,
        requested: u32,
        search_id: BrokerSearchId,
    ) -> CollectedBatch {
        let mut task = BrokerSearchTask::submit(
            search_id,
            SearchGameIdentity::from_position(root_position),
            TreeOwnership { root_node_idx: 0 },
        );
        task.pause()
            .expect("search task must transition submit -> pause");
        debug_assert_eq!(task.search_id(), search_id);
        debug_assert_eq!(task.game().move_number, root_position.move_number);
        debug_assert_eq!(task.tree().root_node_idx, 0);

        let mut completed = 0u32;
        let mut collisions = 0u32;
        let target_slots = requested as usize;
        let retry_budget = target_slots.saturating_mul(4).max(1);
        let mut retries = 0usize;

        while task.pending_len() + (completed as usize) < target_slots && retries < retry_budget {
            let mut working = root_position.clone();
            let mut node_idx = 0usize;
            let mut path: Vec<usize> = Vec::new();
            let mut slot_filled = false;

            loop {
                if self.arena[node_idx].is_terminal {
                    let value = self.terminal_value(&working);
                    self.undo_virtual_loss_path(&path);
                    self.backpropagate(node_idx, value);
                    completed += 1;
                    slot_filled = true;
                    break;
                }

                if !self.arena[node_idx].is_expanded {
                    if self.arena[node_idx].in_flight > 0 {
                        collisions += 1;
                        self.undo_virtual_loss_path(&path);
                        break;
                    }

                    self.arena[node_idx].in_flight += 1;
                    path.push(node_idx);

                    if !working.in_draft() {
                        if is_marshal_captured(&working) {
                            let node = &mut self.arena[node_idx];
                            node.is_terminal = true;
                            node.is_expanded = true;
                            self.undo_virtual_loss_path(&path);
                            self.backpropagate(node_idx, -1.0);
                            completed += 1;
                            slot_filled = true;
                            break;
                        }
                        if working.is_fourfold_repetition() || working.is_insufficient_material() {
                            let node = &mut self.arena[node_idx];
                            node.is_terminal = true;
                            node.is_expanded = true;
                            self.undo_virtual_loss_path(&path);
                            self.backpropagate(node_idx, 0.0);
                            completed += 1;
                            slot_filled = true;
                            break;
                        }
                    }

                    let moves: Vec<Move> = working.moves().into_iter().collect();
                    if moves.is_empty() {
                        let value = if working.in_check(None) { -1.0 } else { 0.0 };
                        let node = &mut self.arena[node_idx];
                        node.is_terminal = true;
                        node.is_expanded = true;
                        self.undo_virtual_loss_path(&path);
                        self.backpropagate(node_idx, value);
                        completed += 1;
                        slot_filled = true;
                        break;
                    }

                    task.enqueue_leaf(PendingLeafTask {
                        node_idx,
                        position: working,
                        moves,
                        path,
                        is_root: node_idx == 0,
                        queued_at: Instant::now(),
                    })
                    .expect("search task must allow enqueue after pause");
                    slot_filled = true;
                    break;
                }

                self.arena[node_idx].in_flight += 1;
                path.push(node_idx);

                self.progressive_widen(node_idx);
                if self.arena[node_idx].children.is_empty() {
                    self.undo_virtual_loss_path(&path);
                    break;
                }

                let child_idx = self.select_child(node_idx);
                let mv = self.arena[child_idx]
                    .mv
                    .as_ref()
                    .expect("child node must contain a move");
                working.make_move(mv).unwrap_or_else(|err| {
                    panic!(
                        "tree child move must stay legal: {err:?}; move={mv:?}; fen={}",
                        working.fen()
                    )
                });
                node_idx = child_idx;
            }

            if !slot_filled {
                retries += 1;
            }
        }

        CollectedBatch {
            requested,
            completed,
            collisions,
            task,
        }
    }

    fn resume_collected_batch(
        &mut self,
        mut collected: CollectedBatch,
        results: Vec<(Vec<f32>, Option<f32>)>,
        rng: &mut impl Rng,
    ) -> ResumeStats {
        let pending_count = collected.task.pending_len();

        if pending_count == 0 {
            let resumed = collected
                .task
                .resume()
                .expect("search task must allow resume without leaves");
            debug_assert!(resumed.is_empty());
            let drained = collected
                .task
                .cancel_drain()
                .expect("search task must allow cancel/drain after resume");
            debug_assert!(drained.is_empty());
            collected
                .task
                .shutdown()
                .expect("search task must shut down after cancel/drain");

            return ResumeStats {
                requested: collected.requested,
                collisions: collected.collisions,
                completed_before_resume: collected.completed,
                completed: 0,
                resumed: 0,
                pending_count: 0,
                avg_us: 0,
                max_us: 0,
            };
        }

        let mut pending = collected
            .task
            .resume()
            .expect("search task must transition enqueue -> resume");
        debug_assert_eq!(pending.len(), results.len());

        let mut completed = 0u32;
        let mut resumed = 0usize;
        let mut resume_sum_us = 0u128;
        let mut resume_max_us = 0u128;

        for (leaf, (raw_priors, neural_value)) in pending.drain(..).zip(results) {
            let resume_us = leaf.queued_at.elapsed().as_micros();
            resume_sum_us += resume_us;
            resume_max_us = resume_max_us.max(resume_us);
            resumed += 1;

            self.undo_virtual_loss_path(&leaf.path);
            let value = self.expand_with_result(
                leaf.node_idx,
                &leaf.position,
                leaf.moves,
                raw_priors,
                neural_value,
                leaf.is_root,
                rng,
            );
            self.backpropagate(leaf.node_idx, value);
            collected
                .task
                .complete_resumed_leaf()
                .expect("resumed leaves must complete while task is in resume phase");
            completed += 1;
        }

        let drained = collected
            .task
            .cancel_drain()
            .expect("search task must allow cancel/drain after resume");
        for leaf in drained {
            self.undo_virtual_loss_path(&leaf.path);
        }
        collected
            .task
            .shutdown()
            .expect("search task must shut down after cancel/drain");

        let avg_us = if resumed == 0 {
            0
        } else {
            resume_sum_us / resumed as u128
        };

        ResumeStats {
            requested: collected.requested,
            collisions: collected.collisions,
            completed_before_resume: collected.completed,
            completed,
            resumed,
            pending_count,
            avg_us,
            max_us: resume_max_us,
        }
    }

    fn expand_with_result(
        &mut self,
        node_idx: usize,
        position: &Position,
        moves: Vec<Move>,
        raw_priors: Vec<f32>,
        neural_value: Option<f32>,
        apply_root_noise: bool,
        rng: &mut impl Rng,
    ) -> f64 {
        let mut priors = self.sanitize_priors(raw_priors, moves.len());
        if apply_root_noise {
            let alpha = self.config.dirichlet_concentration / moves.len() as f32;
            self.add_dirichlet_noise(&mut priors, alpha, rng);
        }

        let mut move_priors: Vec<(Move, f32)> = moves.into_iter().zip(priors).collect();
        move_priors.sort_by(|a, b| b.1.total_cmp(&a.1));

        let initial_count = self.initial_width().min(move_priors.len());
        let initial_moves = move_priors[..initial_count].to_vec();
        let mut children = Vec::with_capacity(initial_count);

        for (mv, prior) in initial_moves {
            let child_idx = self.arena.len();
            self.arena.push(Node::child(node_idx, mv, prior));
            children.push(child_idx);
        }

        let node = &mut self.arena[node_idx];
        node.children = children;
        node.all_moves = move_priors;
        node.active_children_count = initial_count;
        node.is_expanded = true;
        node.is_terminal = false;

        neural_value.map_or_else(
            || self.evaluate_value(position),
            |v| {
                let fv = f64::from(v);
                if fv.is_finite() {
                    fv.clamp(-1.0, 1.0)
                } else {
                    self.evaluate_value(position)
                }
            },
        )
    }

    fn undo_virtual_loss_path(&mut self, path: &[usize]) {
        for &idx in path {
            self.arena[idx].in_flight = self.arena[idx].in_flight.saturating_sub(1);
        }
    }

    fn sanitize_priors(&self, mut priors: Vec<f32>, num_moves: usize) -> Vec<f32> {
        if num_moves == 0 {
            return Vec::new();
        }

        if priors.len() != num_moves {
            return vec![1.0 / num_moves as f32; num_moves];
        }

        let mut sum = 0.0f32;
        for prior in &mut priors {
            if !prior.is_finite() || *prior < 0.0 {
                *prior = 0.0;
            }
            sum += *prior;
        }

        if sum <= f32::EPSILON {
            return vec![1.0 / num_moves as f32; num_moves];
        }

        for prior in &mut priors {
            *prior /= sum;
        }

        priors
    }

    fn add_dirichlet_noise(&self, priors: &mut [f32], alpha: f32, rng: &mut impl Rng) {
        if priors.is_empty() || alpha <= 0.0 || self.config.dirichlet_epsilon <= 0.0 {
            return;
        }

        let mut noise = Vec::with_capacity(priors.len());
        let mut noise_sum = 0.0f64;
        for _ in 0..priors.len() {
            let sample = Self::sample_gamma(rng, f64::from(alpha));
            noise.push(sample);
            noise_sum += sample;
        }

        if noise_sum <= f64::EPSILON {
            return;
        }

        let epsilon = self.config.dirichlet_epsilon.clamp(0.0, 1.0);
        let keep = 1.0 - epsilon;

        for (prior, dir_sample) in priors.iter_mut().zip(noise.into_iter()) {
            let dirichlet_value = (dir_sample / noise_sum) as f32;
            *prior = keep * *prior + epsilon * dirichlet_value;
        }
    }

    fn sample_gamma(rng: &mut impl Rng, alpha: f64) -> f64 {
        if alpha <= 0.0 {
            return 0.0;
        }

        if alpha < 1.0 {
            let u = rng
                .gen::<f64>()
                .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
            return Self::sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
        }

        let d = alpha - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();

        loop {
            let x = Self::sample_standard_normal(rng);
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }

            let v3 = v * v * v;
            let u = rng.gen::<f64>();

            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v3;
            }

            if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }

    fn sample_standard_normal(rng: &mut impl Rng) -> f64 {
        let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    fn evaluate_value(&self, position: &Position) -> f64 {
        let cp = f64::from(self.eval.evaluate(position).0);
        let white_value = (cp / 600.0).tanh();
        match position.turn {
            Color::White => white_value,
            Color::Black => -white_value,
        }
    }

    fn terminal_value(&self, position: &Position) -> f64 {
        if is_marshal_captured(position) || position.is_checkmate() {
            -1.0
        } else {
            0.0
        }
    }

    fn select_root_move(&self, move_number: u32, rng: &mut impl Rng) -> Option<Move> {
        let root = self.arena.first()?;
        if root.children.is_empty() {
            return None;
        }

        if move_number >= self.config.temperature_drop_move || self.config.temperature <= 0.0 {
            let best_idx = root
                .children
                .iter()
                .copied()
                .max_by_key(|&child_idx| self.arena[child_idx].visits)?;
            return self.arena[best_idx].mv.clone();
        }

        let inv_temperature = 1.0 / f64::from(self.config.temperature.max(1e-3));
        let mut weights = Vec::with_capacity(root.children.len());
        let mut sum = 0.0f64;

        for &child_idx in &root.children {
            let visits = f64::from(self.arena[child_idx].visits);
            let weight = visits.powf(inv_temperature);
            weights.push(weight);
            sum += weight;
        }

        let chosen_child_idx = if sum > 0.0 {
            let mut ticket = rng.gen::<f64>() * sum;
            let mut selected = root.children[0];
            for (idx, weight) in root.children.iter().copied().zip(weights.into_iter()) {
                if ticket <= weight {
                    selected = idx;
                    break;
                }
                ticket -= weight;
            }
            selected
        } else {
            let random_idx = rng.gen_range(0..root.children.len());
            root.children[random_idx]
        };

        self.arena[chosen_child_idx].mv.clone()
    }

    fn root_score(&self) -> Score {
        let Some(root) = self.arena.first() else {
            return Score(0);
        };
        if root.visits == 0 {
            return Score(0);
        }

        let value = root.total_value / f64::from(root.visits);
        Self::value_to_score(value)
    }

    fn value_to_score(value: f64) -> Score {
        let clipped = value.clamp(-0.999_999, 0.999_999);
        Score((clipped.atanh() * 600.0).round() as i32)
    }
}

impl Default for MctsSearcher {
    fn default() -> Self {
        Self::new(MctsConfig::default())
    }
}

impl Searcher for MctsSearcher {
    fn search(&mut self, position: &Position, limits: SearchLimits) -> SearchResult {
        let policy = HeuristicPolicy;
        self.search_with_policy(position, limits, &policy)
    }
}
