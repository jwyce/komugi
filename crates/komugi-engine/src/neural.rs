use std::cell::UnsafeCell;
use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use komugi_core::{Color, Evaluator, Move, Policy, Position, Score};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;

use crate::encoding::{
    encode_position, move_to_policy_index, BOARD_SIZE, ENCODING_SIZE, NUM_PLANES, POLICY_SIZE,
};

// ---------------------------------------------------------------------------
// CPU per-thread inference (fallback for gen0 heuristic / no-GPU)
// ---------------------------------------------------------------------------

pub struct NeuralPolicy {
    session: UnsafeCell<Session>,
}

// Safety: each NeuralPolicy is created per-thread in selfplay. A single
// thread owns exclusive access to its session — no concurrent &mut aliases.
unsafe impl Send for NeuralPolicy {}
unsafe impl Sync for NeuralPolicy {}

impl NeuralPolicy {
    pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)?;
        Ok(Self {
            session: UnsafeCell::new(session),
        })
    }

    fn run_inference(&self, position: &Position) -> (Vec<f32>, f32) {
        let encoding = encode_position(position);
        debug_assert_eq!(encoding.len(), ENCODING_SIZE);

        let input =
            Tensor::<f32>::from_array((vec![1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE], encoding))
                .expect("encoding size must match tensor shape");

        // Safety: only one thread ever calls run_inference on this instance
        let session = unsafe { &mut *self.session.get() };
        let outputs = session
            .run(ort::inputs![input])
            .expect("ONNX inference failed");

        let (_, policy_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("policy output must be f32");
        let policy_logits = policy_slice.to_vec();
        debug_assert_eq!(policy_logits.len(), POLICY_SIZE);

        let (_, value_slice) = outputs[1]
            .try_extract_tensor::<f32>()
            .expect("value output must be f32");
        let value = value_slice[0];

        (policy_logits, value)
    }
}

fn logits_to_priors(logits: &[f32], moves: &[Move]) -> Vec<f32> {
    let mut move_logits = Vec::with_capacity(moves.len());
    for mv in moves {
        let idx = move_to_policy_index(mv);
        let logit = if idx < logits.len() { logits[idx] } else { 0.0 };
        move_logits.push(logit);
    }

    let max_logit = move_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = Vec::with_capacity(moves.len());
    for logit in &move_logits {
        let exp = (logit - max_logit).exp();
        probs.push(exp);
        sum += exp;
    }

    if sum > f32::EPSILON {
        for prob in &mut probs {
            *prob /= sum;
        }
    } else {
        let uniform = 1.0 / moves.len() as f32;
        probs.fill(uniform);
    }

    probs
}

impl Policy for NeuralPolicy {
    fn prior(&self, position: &Position, moves: &[Move]) -> Vec<f32> {
        if moves.is_empty() {
            return Vec::new();
        }
        let (logits, _value) = self.run_inference(position);
        logits_to_priors(&logits, moves)
    }

    fn prior_and_value(&self, position: &Position, moves: &[Move]) -> (Vec<f32>, Option<f32>) {
        if moves.is_empty() {
            return (Vec::new(), None);
        }
        let (logits, value) = self.run_inference(position);
        (logits_to_priors(&logits, moves), Some(value))
    }
}

impl Evaluator for NeuralPolicy {
    fn evaluate(&self, position: &Position) -> Score {
        let (_logits, value) = self.run_inference(position);
        let clamped = value.clamp(-0.999_999, 0.999_999) as f64;
        let cp = clamped.atanh() * 600.0;
        let white_cp = match position.turn {
            Color::White => cp,
            Color::Black => -cp,
        };
        Score(white_cp.round() as i32)
    }
}

// ---------------------------------------------------------------------------
// GPU batch inference server
// ---------------------------------------------------------------------------

const DEFAULT_MAX_BATCH_SIZE: usize = 256;
const DEFAULT_BATCH_TIMEOUT_MS: u64 = 2;
const DEFAULT_QUEUE_CAPACITY: usize = 512;
const DEFAULT_WORKERS_PER_GPU: usize = 1;
const BROKER_DEBUG_ENV: &str = "KOMUGI_DEBUG_BROKER";
const GPU_BATCH_DEBUG_ENV: &str = "KOMUGI_DEBUG_GPU_BATCH";

static BROKER_REQUEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy)]
struct GpuPoolConfig {
    max_batch_size: usize,
    batch_timeout: Duration,
    queue_capacity: usize,
    workers_per_gpu: usize,
}

impl GpuPoolConfig {
    fn from_env() -> Self {
        let max_batch_size = env_usize("KOMUGI_GPU_MAX_BATCH", DEFAULT_MAX_BATCH_SIZE).max(1);
        let batch_timeout_ms = env_u64("KOMUGI_GPU_BATCH_TIMEOUT_MS", DEFAULT_BATCH_TIMEOUT_MS);
        let queue_capacity = env_usize("KOMUGI_GPU_QUEUE_CAPACITY", DEFAULT_QUEUE_CAPACITY).max(1);
        let workers_per_gpu =
            env_usize("KOMUGI_GPU_WORKERS_PER_GPU", DEFAULT_WORKERS_PER_GPU).max(1);

        Self {
            max_batch_size,
            batch_timeout: Duration::from_millis(batch_timeout_ms),
            queue_capacity,
            workers_per_gpu,
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

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

fn log_gpu_batch(worker: &str, size: usize, max: usize, timeout_ms: u128) {
    eprintln!("GPU_BATCH worker={worker} size={size} max={max} timeout_ms={timeout_ms}");
}

fn log_broker_enqueue(request: usize, channel: usize, enqueue_wait_us: u128) {
    eprintln!(
        "BROKER_STAGE event=enqueue request={request} channel={channel} enqueue_wait_us={enqueue_wait_us}"
    );
}

fn log_broker_route(request: usize, channel: usize, observed_load: usize, reserved_load: usize) {
    eprintln!(
        "BROKER_STAGE event=route request={request} channel={channel} observed_load={observed_load} reserved_load={reserved_load}"
    );
}

fn log_broker_queue(
    worker: &str,
    queue_depth: usize,
    drained: usize,
    avg_wait_us: u128,
    max_wait_us: u128,
    canceled: usize,
) {
    eprintln!(
        "BROKER_STAGE event=queue worker={worker} queue_depth={queue_depth} drained={drained} queue_wait_avg_us={avg_wait_us} queue_wait_max_us={max_wait_us} canceled={canceled}"
    );
}

struct InferenceRequest {
    encoding: Vec<f32>,
    response_tx: mpsc::SyncSender<Vec<f32>>,
    queued_at: Instant,
    worker_load: std::sync::Arc<GpuWorkerLoad>,
}

#[derive(Debug)]
struct GpuWorkerLoad {
    channel: usize,
    pending_requests: AtomicUsize,
}

impl GpuWorkerLoad {
    fn new(channel: usize) -> Self {
        Self {
            channel,
            pending_requests: AtomicUsize::new(0),
        }
    }

    fn snapshot(&self) -> usize {
        self.pending_requests.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone)]
struct GpuWorkerHandle {
    sender: mpsc::SyncSender<InferenceRequest>,
    load: std::sync::Arc<GpuWorkerLoad>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkerRouteChoice {
    index: usize,
    observed_load: usize,
}

fn select_least_loaded_worker(loads: &[usize], tie_start: usize) -> WorkerRouteChoice {
    debug_assert!(!loads.is_empty());
    let start = tie_start % loads.len();
    let mut best = WorkerRouteChoice {
        index: start,
        observed_load: loads[start],
    };

    for offset in 1..loads.len() {
        let index = (start + offset) % loads.len();
        let load = loads[index];
        if load < best.observed_load {
            best = WorkerRouteChoice {
                index,
                observed_load: load,
            };
            if load == 0 {
                break;
            }
        }
    }

    best
}

pub struct GpuBatchPolicy {
    workers: Vec<GpuWorkerHandle>,
    route_counter: std::sync::Arc<AtomicUsize>,
    debug_broker: bool,
}

impl GpuBatchPolicy {
    fn enqueue(&self, encoding: Vec<f32>) -> mpsc::Receiver<Vec<f32>> {
        let request = BROKER_REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        let queued_at = Instant::now();
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);

        let tie_start = self.route_counter.fetch_add(1, Ordering::Relaxed);
        let loads: Vec<usize> = self
            .workers
            .iter()
            .map(|worker| worker.load.snapshot())
            .collect();
        let route = select_least_loaded_worker(&loads, tie_start);
        let worker = &self.workers[route.index];
        let reserved_load = worker.load.pending_requests.fetch_add(1, Ordering::AcqRel) + 1;

        if self.debug_broker {
            log_broker_route(
                request,
                worker.load.channel,
                route.observed_load,
                reserved_load,
            );
        }

        if worker
            .sender
            .send(InferenceRequest {
                encoding,
                response_tx: resp_tx,
                queued_at,
                worker_load: std::sync::Arc::clone(&worker.load),
            })
            .is_err()
        {
            worker.load.pending_requests.fetch_sub(1, Ordering::AcqRel);
            panic!("GPU inference server died");
        }

        if self.debug_broker {
            log_broker_enqueue(
                request,
                worker.load.channel,
                queued_at.elapsed().as_micros(),
            );
        }
        resp_rx
    }

    fn submit(&self, position: &Position) -> Vec<f32> {
        let encoding = encode_position(position);
        let resp_rx = self.enqueue(encoding);
        resp_rx.recv().expect("GPU inference server died")
    }
}

impl Policy for GpuBatchPolicy {
    fn prior(&self, position: &Position, moves: &[Move]) -> Vec<f32> {
        if moves.is_empty() {
            return Vec::new();
        }
        let logits = self.submit(position);
        logits_to_priors(&logits[..POLICY_SIZE], moves)
    }

    fn prior_and_value(&self, position: &Position, moves: &[Move]) -> (Vec<f32>, Option<f32>) {
        if moves.is_empty() {
            return (Vec::new(), None);
        }
        let result = self.submit(position);
        let priors = logits_to_priors(&result[..POLICY_SIZE], moves);
        let value = result[POLICY_SIZE];
        (priors, Some(value))
    }

    fn prior_and_value_batch(
        &self,
        batch: &[(&Position, &[Move])],
    ) -> Vec<(Vec<f32>, Option<f32>)> {
        if batch.is_empty() {
            return Vec::new();
        }
        // Send all requests first (non-blocking enqueue to GPU worker).
        // The GPU worker sees all N requests in its channel and batches
        // them into a single inference call instead of N separate calls.
        let mut resp_rxs = Vec::with_capacity(batch.len());
        for (pos, _) in batch {
            let encoding = encode_position(pos);
            let resp_rx = self.enqueue(encoding);
            resp_rxs.push(resp_rx);
        }
        // Collect all responses (GPU processed them as one batch).
        batch
            .iter()
            .zip(resp_rxs)
            .map(|((_, moves), resp_rx)| {
                let result = resp_rx.recv().expect("GPU inference server died");
                let priors = logits_to_priors(&result[..POLICY_SIZE], moves);
                let value = result[POLICY_SIZE];
                (priors, Some(value))
            })
            .collect()
    }
}

impl Evaluator for GpuBatchPolicy {
    fn evaluate(&self, position: &Position) -> Score {
        let result = self.submit(position);
        let value = result[POLICY_SIZE];
        let clamped = value.clamp(-0.999_999, 0.999_999) as f64;
        let cp = clamped.atanh() * 600.0;
        let white_cp = match position.turn {
            Color::White => cp,
            Color::Black => -cp,
        };
        Score(white_cp.round() as i32)
    }
}

fn gpu_inference_loop(
    rx: mpsc::Receiver<InferenceRequest>,
    mut session: Session,
    cfg: GpuPoolConfig,
    worker: String,
    debug_batch: bool,
    debug_broker: bool,
) {
    loop {
        let first = match rx.recv() {
            Ok(req) => req,
            Err(_) => return,
        };

        let mut batch = vec![first];
        // Wait briefly for cross-thread coalescing (recv_timeout), then
        // fast-drain remaining requests (try_recv). This lets requests from
        // multiple threads arriving microseconds apart get batched together.
        if batch.len() < cfg.max_batch_size {
            if let Ok(req) = rx.recv_timeout(cfg.batch_timeout) {
                batch.push(req);
                while batch.len() < cfg.max_batch_size {
                    match rx.try_recv() {
                        Ok(req) => batch.push(req),
                        Err(_) => break,
                    }
                }
            }
        }

        let batch_size = batch.len();
        let queue_depth = batch[0].worker_load.snapshot();
        let queue_wait_sum_us: u128 = batch
            .iter()
            .map(|req| req.queued_at.elapsed().as_micros())
            .sum();
        let queue_wait_max_us = batch
            .iter()
            .map(|req| req.queued_at.elapsed().as_micros())
            .max()
            .unwrap_or(0);
        let mut input_data = Vec::with_capacity(batch_size * ENCODING_SIZE);
        for req in &batch {
            input_data.extend_from_slice(&req.encoding);
        }

        let input = Tensor::<f32>::from_array((
            vec![batch_size, NUM_PLANES, BOARD_SIZE, BOARD_SIZE],
            input_data,
        ))
        .expect("batch tensor shape mismatch");

        let outputs = session
            .run(ort::inputs![input])
            .expect("GPU ONNX inference failed");

        let (_, policy_flat) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("policy output must be f32");
        let (_, value_flat) = outputs[1]
            .try_extract_tensor::<f32>()
            .expect("value output must be f32");

        if debug_batch {
            log_gpu_batch(
                &worker,
                batch_size,
                cfg.max_batch_size,
                cfg.batch_timeout.as_millis(),
            );
        }

        let mut canceled = 0usize;

        for (i, req) in batch.into_iter().enumerate() {
            let start = i * POLICY_SIZE;
            let mut result = Vec::with_capacity(POLICY_SIZE + 1);
            result.extend_from_slice(&policy_flat[start..start + POLICY_SIZE]);
            result.push(value_flat[i]);
            if req.response_tx.send(result).is_err() {
                canceled += 1;
            }
            req.worker_load
                .pending_requests
                .fetch_sub(1, Ordering::AcqRel);
        }

        if debug_broker {
            let queue_wait_avg_us = queue_wait_sum_us / batch_size as u128;
            let drained = batch_size;
            log_broker_queue(
                &worker,
                queue_depth,
                drained,
                queue_wait_avg_us,
                queue_wait_max_us,
                canceled,
            );
        }
    }
}

pub struct GpuInferencePool {
    workers: Vec<GpuWorkerHandle>,
    route_counter: std::sync::Arc<AtomicUsize>,
    debug_broker: bool,
}

impl GpuInferencePool {
    pub fn new(model_path: impl AsRef<Path>, num_gpus: usize) -> Result<Self, ort::Error> {
        let cfg = GpuPoolConfig::from_env();
        let debug_broker = env_bool(BROKER_DEBUG_ENV, false);
        let debug_batch = env_bool(GPU_BATCH_DEBUG_ENV, false);
        let mut workers = Vec::with_capacity(num_gpus * cfg.workers_per_gpu);
        let model_path = model_path.as_ref();

        for gpu_id in 0..num_gpus {
            for worker_idx in 0..cfg.workers_per_gpu {
                let session = Session::builder()?
                    .with_intra_threads(1)?
                    .with_inter_threads(1)?
                    .with_execution_providers([CUDAExecutionProvider::default()
                        .with_device_id(gpu_id as i32)
                        .build()])?
                    .commit_from_file(model_path)?;
                let (tx, rx) = mpsc::sync_channel(cfg.queue_capacity);
                let worker_cfg = cfg;
                let worker_name = format!("gpu-infer-{gpu_id}-{worker_idx}");
                thread::Builder::new()
                    .name(worker_name.clone())
                    .spawn(move || {
                        gpu_inference_loop(
                            rx,
                            session,
                            worker_cfg,
                            worker_name,
                            debug_batch,
                            debug_broker,
                        )
                    })
                    .expect("failed to spawn GPU inference thread");

                let channel = workers.len();
                workers.push(GpuWorkerHandle {
                    sender: tx,
                    load: std::sync::Arc::new(GpuWorkerLoad::new(channel)),
                });
            }
        }

        eprintln!(
            "GPU batch inference pool: {num_gpus} GPUs, workers_per_gpu={}, max_batch={}, timeout_ms={}, queue_capacity={} (total_workers={})",
            cfg.workers_per_gpu,
            cfg.max_batch_size,
            cfg.batch_timeout.as_millis(),
            cfg.queue_capacity,
            workers.len()
        );
        Ok(Self {
            workers,
            route_counter: std::sync::Arc::new(AtomicUsize::new(0)),
            debug_broker,
        })
    }

    pub fn policy(&self) -> GpuBatchPolicy {
        GpuBatchPolicy {
            workers: self.workers.clone(),
            route_counter: std::sync::Arc::clone(&self.route_counter),
            debug_broker: self.debug_broker,
        }
    }

    pub fn num_gpus(&self) -> usize {
        self.workers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::select_least_loaded_worker;

    #[test]
    fn least_loaded_worker_beats_tie_cursor() {
        let choice = select_least_loaded_worker(&[4, 1, 3, 2], 0);
        assert_eq!(choice.index, 1);
        assert_eq!(choice.observed_load, 1);
    }

    #[test]
    fn least_loaded_worker_rotates_equal_load_ties() {
        let first = select_least_loaded_worker(&[2, 2, 2], 1);
        let second = select_least_loaded_worker(&[2, 2, 2], 2);

        assert_eq!(first.index, 1);
        assert_eq!(second.index, 2);
    }
}
