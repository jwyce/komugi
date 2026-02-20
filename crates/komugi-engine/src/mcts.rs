use std::f64::consts::PI;
use std::time::{Duration, Instant};

use komugi_core::{
    is_marshal_captured, Color, Evaluator, Move, MoveType, Policy, Position, Score, SearchLimits,
    SearchResult, Searcher,
};
use rand::Rng;

use crate::classical::ClassicalEval;

const DEFAULT_MAX_SIMULATIONS: u32 = 800;
const DEFAULT_C_PUCT: f32 = 4.0;
const DEFAULT_DIRICHLET_CONCENTRATION: f32 = 10.0;
const DEFAULT_DIRICHLET_EPSILON: f32 = 0.25;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_TEMPERATURE_DROP_MOVE: u32 = 25;
const INITIAL_PROGRESSIVE_WIDTH: usize = 12;
const PROGRESSIVE_WIDTH_STEP: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub max_simulations: u32,
    pub time_limit_ms: Option<u64>,
    pub c_puct: f32,
    pub dirichlet_concentration: f32,
    pub dirichlet_epsilon: f32,
    pub temperature: f32,
    pub temperature_drop_move: u32,
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

        let best_move = self.select_root_move(position.move_number, &mut rng);

        SearchResult {
            best_move,
            score: self.root_score(),
            nodes_searched: u64::from(simulations),
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
            working
                .make_move(mv)
                .expect("tree child move must stay legal");
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
        if self.is_terminal_position(position) {
            let node = &mut self.arena[node_idx];
            node.is_terminal = true;
            node.is_expanded = true;
            node.children.clear();
            node.all_moves.clear();
            node.active_children_count = 0;
            return self.terminal_value(position);
        }

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
        let parent_visits = f64::from(parent.visits.max(1)).sqrt();

        let mut best_idx = parent.children[0];
        let mut best_score = f64::NEG_INFINITY;

        for &child_idx in &parent.children {
            let child = &self.arena[child_idx];

            // PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).
            let q = if child.visits == 0 {
                0.0
            } else {
                -child.total_value / f64::from(child.visits)
            };
            let u = f64::from(self.config.c_puct) * f64::from(child.prior) * parent_visits
                / (1.0 + f64::from(child.visits));
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

    fn is_terminal_position(&self, position: &Position) -> bool {
        if position.in_draft() {
            return false;
        }
        is_marshal_captured(position) || position.is_checkmate() || position.is_draw()
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
