use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use komugi_core::{
    is_marshal_captured, move_to_san, Color, Policy, Position, SearchLimits, SearchResult,
    SetupMode,
};
use serde::{Deserialize, Serialize};

use crate::encoding::encode_position;
use crate::mcts::{
    BrokerPausedSearch, BrokerSearchRuntime, BrokerWorkerStep, HeuristicPolicy, MctsConfig,
    MctsSearcher,
};
use crate::mcts_broker::BrokerSearchId;

const BROKER_DEBUG_ENV: &str = "KOMUGI_DEBUG_BROKER";
const ACTIVE_GAME_WINDOW_MULTIPLIER: usize = 2;
const NEAR_COMPLETE_REMAINING_MOVES: u32 = 12;

static ACTIVE_SEARCHES: AtomicUsize = AtomicUsize::new(0);
static SEARCH_COUNTER: AtomicUsize = AtomicUsize::new(0);

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

fn log_broker_search(
    event: &str,
    search: usize,
    ply: u32,
    move_number: u32,
    active_searches: usize,
    elapsed_us: Option<u128>,
) {
    match elapsed_us {
        Some(elapsed_us) => eprintln!(
            "BROKER_STAGE event={event} search={search} ply={ply} move_number={move_number} active_searches={active_searches} elapsed_us={elapsed_us}"
        ),
        None => eprintln!(
            "BROKER_STAGE event={event} search={search} ply={ply} move_number={move_number} active_searches={active_searches}"
        ),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecord {
    pub fen: String,
    pub played_move: String,
    pub policy: Vec<(String, f32)>,
    pub outcome: f32,
    pub move_number: u32,
    pub encoding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    pub positions: Vec<TrainingRecord>,
    pub result: GameResult,
    pub total_moves: u32,
    pub moves: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

pub struct SelfPlayConfig {
    pub mcts_config: MctsConfig,
    pub setup_mode: SetupMode,
    pub max_moves: u32,
    pub policy: Arc<dyn Policy>,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            mcts_config: MctsConfig::default(),
            setup_mode: SetupMode::Beginner,
            max_moves: 300,
            policy: Arc::new(HeuristicPolicy),
        }
    }
}

struct GameSlot {
    game_num: u32,
    position: Position,
    searcher: MctsSearcher,
    positions: Vec<TrainingRecord>,
    turns: Vec<Color>,
    move_sans: Vec<String>,
    total_moves: u32,
    terminated: bool,
}

impl GameSlot {
    fn new(game_num: u32, config: &SelfPlayConfig) -> Self {
        Self {
            game_num,
            position: Position::new(config.setup_mode),
            searcher: MctsSearcher::new(config.mcts_config),
            positions: Vec::new(),
            turns: Vec::new(),
            move_sans: Vec::new(),
            total_moves: 0,
            terminated: false,
        }
    }

    fn needs_search(&self, max_moves: u32) -> bool {
        !self.terminated && self.total_moves < max_moves && !self.position.is_game_over()
    }

    fn apply_search_result(&mut self, search_result: SearchResult) {
        let policy = self
            .searcher
            .get_root_policy()
            .into_iter()
            .map(|(mv, proportion)| (move_to_san(&mv), proportion))
            .collect();

        let Some(best_move) = search_result.best_move else {
            self.positions.push(TrainingRecord {
                fen: self.position.fen(),
                played_move: String::new(),
                policy,
                outcome: 0.0,
                move_number: self.position.move_number,
                encoding: encode_position(&self.position),
            });
            self.turns.push(self.position.turn);
            self.terminated = true;
            return;
        };

        let san = move_to_san(&best_move);
        self.move_sans.push(san.clone());
        self.turns.push(self.position.turn);
        self.positions.push(TrainingRecord {
            fen: self.position.fen(),
            played_move: san,
            policy,
            outcome: 0.0,
            move_number: self.position.move_number,
            encoding: encode_position(&self.position),
        });

        if self.position.make_move(&best_move).is_err() {
            self.terminated = true;
            return;
        }

        self.total_moves = self.total_moves.saturating_add(1);
    }

    fn into_record(mut self, max_moves: u32) -> GameRecord {
        let reached_max_moves = self.total_moves >= max_moves && !self.position.is_game_over();
        let result = infer_result(&self.position, reached_max_moves);

        for (record, turn) in self.positions.iter_mut().zip(self.turns.into_iter()) {
            record.outcome = outcome_for_side(result, turn);
        }

        GameRecord {
            positions: self.positions,
            result,
            total_moves: self.total_moves,
            moves: self.move_sans,
        }
    }
}

struct SearchTask {
    search_id: BrokerSearchId,
    started_at: Instant,
    root_position: Position,
    runtime: BrokerSearchRuntime,
    slot: GameSlot,
}

impl SearchTask {
    fn new(search_id: BrokerSearchId, mut slot: GameSlot) -> Self {
        let root_position = slot.position.clone();
        let runtime = slot.searcher.start_broker_search(SearchLimits::default());

        Self {
            search_id,
            started_at: Instant::now(),
            root_position,
            runtime,
            slot,
        }
    }
}

struct PausedSearchTask {
    task: SearchTask,
    paused: BrokerPausedSearch,
}

enum WorkerCommand {
    Run(SearchTask),
    Shutdown,
}

enum WorkerEvent {
    Paused(PausedSearchTask),
    Finished(SearchTask, SearchResult),
    Yielded(SearchTask),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReadyTaskKind {
    Resumed,
    ContinueSearch,
    NewGame,
}

struct ReadySearchTask {
    task: SearchTask,
    kind: ReadyTaskKind,
    queued_order: u64,
}

struct QueuedPausedSearch {
    paused: PausedSearchTask,
    queued_order: u64,
}

fn prepare_search_task(slot: GameSlot, next_search_id: &mut u64) -> SearchTask {
    let search_id = BrokerSearchId::new(*next_search_id);
    *next_search_id = (*next_search_id).saturating_add(1);

    SearchTask::new(search_id, slot)
}

fn dispatch_search_task(
    mut ready_task: ReadySearchTask,
    command_tx: &mpsc::Sender<WorkerCommand>,
    debug_broker: bool,
    active_searches: &mut usize,
    running_searches: &mut usize,
) {
    if ready_task.kind != ReadyTaskKind::Resumed {
        ready_task.task.started_at = Instant::now();
        *active_searches = active_searches.saturating_add(1);

        if debug_broker {
            log_broker_search(
                "search_start",
                ready_task.task.search_id.get() as usize,
                ready_task.task.slot.total_moves,
                ready_task.task.slot.position.move_number,
                *active_searches,
                None,
            );
        }
    }

    command_tx
        .send(WorkerCommand::Run(ready_task.task))
        .expect("broker worker channel closed while dispatching search");
    *running_searches = running_searches.saturating_add(1);
}

fn broker_worker_loop(
    command_rx: Arc<Mutex<mpsc::Receiver<WorkerCommand>>>,
    event_tx: mpsc::Sender<WorkerEvent>,
) {
    loop {
        let command = match command_rx.lock().unwrap().recv() {
            Ok(command) => command,
            Err(_) => break,
        };

        match command {
            WorkerCommand::Shutdown => break,
            WorkerCommand::Run(mut task) => {
                let worker_result = task.slot.searcher.run_search_until_pause_or_finish(
                    &task.root_position,
                    task.search_id,
                    &mut task.runtime,
                );
                let event = match worker_result {
                    BrokerWorkerStep::Paused(paused) => {
                        WorkerEvent::Paused(PausedSearchTask { task, paused })
                    }
                    BrokerWorkerStep::Finished(result) => WorkerEvent::Finished(task, result),
                    BrokerWorkerStep::Yielded => WorkerEvent::Yielded(task),
                };
                if event_tx.send(event).is_err() {
                    break;
                }
            }
        }
    }
}

fn flush_paused_searches(
    paused_searches: &mut Vec<QueuedPausedSearch>,
    policy: &dyn Policy,
) -> Vec<ReadySearchTask> {
    if paused_searches.is_empty() {
        return Vec::new();
    }

    paused_searches.sort_by_key(|paused_search| paused_search.queued_order);

    let mut batch_items = Vec::new();
    for paused_search in paused_searches.iter() {
        for leaf in paused_search.paused.paused.pending_leaves() {
            batch_items.push((&leaf.position, leaf.moves.as_slice()));
        }
    }

    let mut batch_results = policy.prior_and_value_batch(&batch_items).into_iter();
    let mut resumed_tasks = Vec::with_capacity(paused_searches.len());

    for mut paused_search in paused_searches.drain(..) {
        let pending = paused_search.paused.paused.pending_len();
        let mut results = Vec::with_capacity(pending);
        for _ in 0..pending {
            let result = batch_results
                .next()
                .expect("global broker flush must return one result per pending leaf");
            results.push(result);
        }

        paused_search
            .paused
            .task
            .slot
            .searcher
            .resume_paused_search(
                &mut paused_search.paused.task.runtime,
                paused_search.paused.paused,
                results,
            );
        resumed_tasks.push(ReadySearchTask {
            task: paused_search.paused.task,
            kind: ReadyTaskKind::Resumed,
            queued_order: paused_search.queued_order,
        });
    }

    debug_assert!(batch_results.next().is_none());
    resumed_tasks
}

fn active_game_window(worker_count: usize) -> usize {
    worker_count
        .saturating_mul(ACTIVE_GAME_WINDOW_MULTIPLIER)
        .max(1)
}

fn active_game_count(running_searches: usize, ready_count: usize, paused_count: usize) -> usize {
    running_searches
        .saturating_add(ready_count)
        .saturating_add(paused_count)
}

fn available_new_game_slots(
    running_searches: usize,
    ready_count: usize,
    paused_count: usize,
    active_window: usize,
) -> usize {
    active_window.saturating_sub(active_game_count(
        running_searches,
        ready_count,
        paused_count,
    ))
}

fn ready_task_priority(
    task: &ReadySearchTask,
    max_moves: u32,
) -> (ReadyTaskKind, u8, u64, u32, u32) {
    let remaining_moves = max_moves.saturating_sub(task.task.slot.total_moves);
    let near_complete_rank = u8::from(remaining_moves > NEAR_COMPLETE_REMAINING_MOVES);

    (
        task.kind,
        near_complete_rank,
        task.queued_order,
        remaining_moves,
        task.task.slot.game_num,
    )
}

fn pop_next_ready_search(
    ready_searches: &mut Vec<ReadySearchTask>,
    max_moves: u32,
) -> Option<ReadySearchTask> {
    let next_idx = ready_searches
        .iter()
        .enumerate()
        .min_by_key(|(_, task)| ready_task_priority(task, max_moves))?
        .0;
    Some(ready_searches.swap_remove(next_idx))
}

fn next_queue_order(queue_order: &mut u64) -> u64 {
    let order = *queue_order;
    *queue_order = queue_order.saturating_add(1);
    order
}

fn enqueue_ready_task(
    ready_searches: &mut Vec<ReadySearchTask>,
    task: SearchTask,
    kind: ReadyTaskKind,
    queue_order: &mut u64,
) {
    ready_searches.push(ReadySearchTask {
        task,
        kind,
        queued_order: next_queue_order(queue_order),
    });
}

fn enqueue_paused_search(
    paused_searches: &mut Vec<QueuedPausedSearch>,
    paused: PausedSearchTask,
    queue_order: &mut u64,
) {
    paused_searches.push(QueuedPausedSearch {
        paused,
        queued_order: next_queue_order(queue_order),
    });
}

fn admit_new_games(
    config: &SelfPlayConfig,
    num_games: u32,
    next_game_num: &mut u32,
    next_search_id: &mut u64,
    ready_searches: &mut Vec<ReadySearchTask>,
    paused_searches: &[QueuedPausedSearch],
    running_searches: usize,
    active_window: usize,
    queue_order: &mut u64,
) {
    let slots = available_new_game_slots(
        running_searches,
        ready_searches.len(),
        paused_searches.len(),
        active_window,
    );

    for _ in 0..slots {
        if *next_game_num > num_games {
            break;
        }

        let task = prepare_search_task(GameSlot::new(*next_game_num, config), next_search_id);
        enqueue_ready_task(ready_searches, task, ReadyTaskKind::NewGame, queue_order);
        *next_game_num = next_game_num.saturating_add(1);
    }
}

fn schedule_broker_tasks(
    config: &SelfPlayConfig,
    num_games: u32,
    worker_count: usize,
    active_window: usize,
    command_tx: &mpsc::Sender<WorkerCommand>,
    ready_searches: &mut Vec<ReadySearchTask>,
    paused_searches: &mut Vec<QueuedPausedSearch>,
    next_game_num: &mut u32,
    next_search_id: &mut u64,
    queue_order: &mut u64,
    debug_broker: bool,
    active_searches: &mut usize,
    running_searches: &mut usize,
) {
    if !paused_searches.is_empty()
        && (*running_searches < worker_count || ready_searches.is_empty())
    {
        ready_searches.extend(flush_paused_searches(
            paused_searches,
            config.policy.as_ref(),
        ));
    }

    admit_new_games(
        config,
        num_games,
        next_game_num,
        next_search_id,
        ready_searches,
        paused_searches,
        *running_searches,
        active_window,
        queue_order,
    );

    while *running_searches < worker_count {
        if ready_searches.is_empty() && !paused_searches.is_empty() {
            ready_searches.extend(flush_paused_searches(
                paused_searches,
                config.policy.as_ref(),
            ));
        }

        let Some(task) = pop_next_ready_search(ready_searches, config.max_moves) else {
            break;
        };
        dispatch_search_task(
            task,
            command_tx,
            debug_broker,
            active_searches,
            running_searches,
        );
    }
}

pub fn play_games_with_broker(
    config: &SelfPlayConfig,
    num_games: u32,
    num_workers: usize,
) -> Vec<(u32, GameRecord)> {
    if num_games == 0 {
        return Vec::new();
    }

    let worker_count = num_workers.max(1);
    let debug_broker = env_bool(BROKER_DEBUG_ENV, false);
    let (command_tx, command_rx) = mpsc::channel::<WorkerCommand>();
    let command_rx = Arc::new(Mutex::new(command_rx));
    let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>();

    let mut worker_handles = Vec::with_capacity(worker_count);
    for worker_id in 0..worker_count {
        let worker_rx = Arc::clone(&command_rx);
        let worker_tx = event_tx.clone();
        let handle = thread::Builder::new()
            .name(format!("selfplay-broker-worker-{worker_id}"))
            .spawn(move || broker_worker_loop(worker_rx, worker_tx))
            .expect("failed to spawn selfplay broker worker");
        worker_handles.push(handle);
    }
    drop(event_tx);

    let mut completed_records = Vec::with_capacity(num_games as usize);
    let mut paused_searches = Vec::new();
    let mut ready_searches = Vec::new();
    let mut running_searches = 0usize;
    let mut next_game_num = 1u32;
    let mut next_search_id = 1u64;
    let mut queue_order = 0u64;
    let mut active_searches = 0usize;
    let active_window = active_game_window(worker_count);

    while completed_records.len() < num_games as usize {
        schedule_broker_tasks(
            config,
            num_games,
            worker_count,
            active_window,
            &command_tx,
            &mut ready_searches,
            &mut paused_searches,
            &mut next_game_num,
            &mut next_search_id,
            &mut queue_order,
            debug_broker,
            &mut active_searches,
            &mut running_searches,
        );

        if running_searches == 0 {
            panic!("broker scheduler ran out of active work before all games finished");
        }

        let event = event_rx
            .recv()
            .expect("broker worker channel closed before all games finished");
        running_searches = running_searches.saturating_sub(1);

        match event {
            WorkerEvent::Paused(paused) => {
                enqueue_paused_search(&mut paused_searches, paused, &mut queue_order);
            }
            WorkerEvent::Yielded(task) => {
                enqueue_ready_task(
                    &mut ready_searches,
                    task,
                    ReadyTaskKind::Resumed,
                    &mut queue_order,
                );
            }
            WorkerEvent::Finished(mut task, search_result) => {
                if debug_broker {
                    active_searches = active_searches.saturating_sub(1);
                    log_broker_search(
                        "search_end",
                        task.search_id.get() as usize,
                        task.slot.total_moves,
                        task.slot.position.move_number,
                        active_searches,
                        Some(task.started_at.elapsed().as_micros()),
                    );
                }

                task.slot.apply_search_result(search_result);

                if task.slot.needs_search(config.max_moves) {
                    let continued = prepare_search_task(task.slot, &mut next_search_id);
                    enqueue_ready_task(
                        &mut ready_searches,
                        continued,
                        ReadyTaskKind::ContinueSearch,
                        &mut queue_order,
                    );
                } else {
                    let game_num = task.slot.game_num;
                    let record = task.slot.into_record(config.max_moves);
                    eprintln!(
                        "[broker] Game {game_num}/{num_games}: {} moves, {:?}",
                        record.total_moves, record.result
                    );
                    completed_records.push((game_num, record));
                }
            }
        }
    }

    for _ in 0..worker_count {
        let _ = command_tx.send(WorkerCommand::Shutdown);
    }
    for handle in worker_handles {
        handle
            .join()
            .expect("selfplay broker worker thread panicked");
    }

    completed_records.sort_by_key(|(game_num, _)| *game_num);
    completed_records
}

pub fn play_game(config: &SelfPlayConfig) -> GameRecord {
    if config.mcts_config.vl_batch_size > 1 {
        let mut games = play_games_with_broker(config, 1, 1);
        return games
            .pop()
            .expect("single-game broker run must produce one game")
            .1;
    }

    play_game_blocking(config)
}

fn play_game_blocking(config: &SelfPlayConfig) -> GameRecord {
    let mut position = Position::new(config.setup_mode);
    let mut searcher = MctsSearcher::new(config.mcts_config);
    let debug_broker = env_bool(BROKER_DEBUG_ENV, false);
    let mut positions = Vec::new();
    let mut turns = Vec::new();
    let mut move_sans = Vec::new();
    let mut total_moves = 0u32;

    while total_moves < config.max_moves && !position.is_game_over() {
        let search_result = if debug_broker {
            let search = SEARCH_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
            let active_searches = ACTIVE_SEARCHES.fetch_add(1, Ordering::Relaxed) + 1;
            let started_at = Instant::now();
            log_broker_search(
                "search_start",
                search,
                total_moves,
                position.move_number,
                active_searches,
                None,
            );
            let search_result = searcher.search_with_policy(
                &position,
                SearchLimits::default(),
                config.policy.as_ref(),
            );
            let active_searches = ACTIVE_SEARCHES.fetch_sub(1, Ordering::Relaxed) - 1;
            log_broker_search(
                "search_end",
                search,
                total_moves,
                position.move_number,
                active_searches,
                Some(started_at.elapsed().as_micros()),
            );
            search_result
        } else {
            searcher.search_with_policy(&position, SearchLimits::default(), config.policy.as_ref())
        };

        let policy = searcher
            .get_root_policy()
            .into_iter()
            .map(|(mv, proportion)| (move_to_san(&mv), proportion))
            .collect();

        let Some(best_move) = search_result.best_move else {
            positions.push(TrainingRecord {
                fen: position.fen(),
                played_move: String::new(),
                policy,
                outcome: 0.0,
                move_number: position.move_number,
                encoding: encode_position(&position),
            });
            turns.push(position.turn);
            break;
        };

        let san = move_to_san(&best_move);
        move_sans.push(san.clone());

        turns.push(position.turn);
        positions.push(TrainingRecord {
            fen: position.fen(),
            played_move: san,
            policy,
            outcome: 0.0,
            move_number: position.move_number,
            encoding: encode_position(&position),
        });

        if position.make_move(&best_move).is_err() {
            break;
        }

        total_moves = total_moves.saturating_add(1);
    }

    let reached_max_moves = total_moves >= config.max_moves && !position.is_game_over();
    let result = infer_result(&position, reached_max_moves);

    for (record, turn) in positions.iter_mut().zip(turns.into_iter()) {
        record.outcome = outcome_for_side(result, turn);
    }

    GameRecord {
        positions,
        result,
        total_moves,
        moves: move_sans,
    }
}

fn infer_result(position: &Position, reached_max_moves: bool) -> GameResult {
    if position.in_draft() {
        return GameResult::Draw;
    }

    if reached_max_moves || position.is_draw() {
        return GameResult::Draw;
    }

    if is_marshal_captured(position) || position.is_checkmate() {
        return match position.turn {
            Color::White => GameResult::BlackWin,
            Color::Black => GameResult::WhiteWin,
        };
    }

    GameResult::Draw
}

fn outcome_for_side(result: GameResult, side_to_move: Color) -> f32 {
    match result {
        GameResult::WhiteWin => {
            if side_to_move == Color::White {
                1.0
            } else {
                -1.0
            }
        }
        GameResult::BlackWin => {
            if side_to_move == Color::Black {
                1.0
            } else {
                -1.0
            }
        }
        GameResult::Draw => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_broker_config() -> MctsConfig {
        MctsConfig {
            max_simulations: 24,
            vl_batch_size: 8,
            dirichlet_epsilon: 0.0,
            temperature: 0.0,
            ..MctsConfig::default()
        }
    }

    fn paused_search_task(
        game_num: u32,
        config: &SelfPlayConfig,
        next_search_id: &mut u64,
        queued_order: u64,
    ) -> QueuedPausedSearch {
        let mut task = prepare_search_task(GameSlot::new(game_num, config), next_search_id);
        let paused = match task.slot.searcher.run_search_until_pause_or_finish(
            &task.root_position,
            task.search_id,
            &mut task.runtime,
        ) {
            BrokerWorkerStep::Paused(paused) => paused,
            step => panic!("expected paused search, got {step:?}"),
        };

        QueuedPausedSearch {
            paused: PausedSearchTask { task, paused },
            queued_order,
        }
    }

    fn test_ready_task(
        game_num: u32,
        total_moves: u32,
        kind: ReadyTaskKind,
        queued_order: u64,
    ) -> ReadySearchTask {
        let config = SelfPlayConfig::default();
        let mut slot = GameSlot::new(game_num, &config);
        slot.total_moves = total_moves;

        ReadySearchTask {
            task: SearchTask::new(BrokerSearchId::new(game_num as u64), slot),
            kind,
            queued_order,
        }
    }

    #[test]
    fn fairness_priority_prefers_resumed_then_near_complete_then_new() {
        let mut ready = vec![
            test_ready_task(1, 0, ReadyTaskKind::NewGame, 0),
            test_ready_task(2, 40, ReadyTaskKind::ContinueSearch, 1),
            test_ready_task(3, 295, ReadyTaskKind::Resumed, 2),
            test_ready_task(4, 100, ReadyTaskKind::Resumed, 3),
        ];

        let first = pop_next_ready_search(&mut ready, 300).unwrap();
        let second = pop_next_ready_search(&mut ready, 300).unwrap();
        let third = pop_next_ready_search(&mut ready, 300).unwrap();
        let fourth = pop_next_ready_search(&mut ready, 300).unwrap();

        assert_eq!(first.task.slot.game_num, 3);
        assert_eq!(second.task.slot.game_num, 4);
        assert_eq!(third.task.slot.game_num, 2);
        assert_eq!(fourth.task.slot.game_num, 1);
    }

    #[test]
    fn active_game_window_stays_bounded() {
        assert_eq!(active_game_window(3), 6);
        assert_eq!(available_new_game_slots(2, 1, 3, 6), 0);
        assert_eq!(available_new_game_slots(1, 1, 1, 6), 3);
    }

    #[test]
    fn admit_new_games_counts_paused_work_toward_backpressure() {
        let config = SelfPlayConfig {
            mcts_config: deterministic_broker_config(),
            setup_mode: SetupMode::Intermediate,
            max_moves: 12,
            policy: Arc::new(HeuristicPolicy),
        };
        let active_window = active_game_window(2);
        let mut next_search_id = 1u64;
        let mut next_game_num = 4u32;
        let mut queue_order = 0u64;
        let mut ready_searches = vec![test_ready_task(1, 0, ReadyTaskKind::NewGame, 0)];
        let mut paused_searches = vec![
            paused_search_task(2, &config, &mut next_search_id, 1),
            paused_search_task(3, &config, &mut next_search_id, 2),
        ];

        admit_new_games(
            &config,
            8,
            &mut next_game_num,
            &mut next_search_id,
            &mut ready_searches,
            &paused_searches,
            1,
            active_window,
            &mut queue_order,
        );

        assert_eq!(ready_searches.len(), 1);
        assert_eq!(next_game_num, 4);

        paused_searches.pop();
        admit_new_games(
            &config,
            8,
            &mut next_game_num,
            &mut next_search_id,
            &mut ready_searches,
            &paused_searches,
            1,
            active_window,
            &mut queue_order,
        );

        assert_eq!(ready_searches.len(), 2);
        assert_eq!(next_game_num, 5);
    }
}
