use std::cell::UnsafeCell;
use std::env;
use std::path::Path;
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

struct InferenceRequest {
    encoding: Vec<f32>,
    response_tx: mpsc::SyncSender<Vec<f32>>,
}

pub struct GpuBatchPolicy {
    request_tx: mpsc::SyncSender<InferenceRequest>,
}

impl GpuBatchPolicy {
    fn submit(&self, position: &Position) -> Vec<f32> {
        let encoding = encode_position(position);
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.request_tx
            .send(InferenceRequest {
                encoding,
                response_tx: resp_tx,
            })
            .expect("GPU inference server died");
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
) {
    loop {
        let first = match rx.recv() {
            Ok(req) => req,
            Err(_) => return,
        };

        let mut batch = vec![first];
        let deadline = Instant::now() + cfg.batch_timeout;
        while batch.len() < cfg.max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        let batch_size = batch.len();
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

        for (i, req) in batch.into_iter().enumerate() {
            let start = i * POLICY_SIZE;
            let mut result = Vec::with_capacity(POLICY_SIZE + 1);
            result.extend_from_slice(&policy_flat[start..start + POLICY_SIZE]);
            result.push(value_flat[i]);
            let _ = req.response_tx.send(result);
        }
    }
}

pub struct GpuInferencePool {
    senders: Vec<mpsc::SyncSender<InferenceRequest>>,
}

impl GpuInferencePool {
    pub fn new(model_path: impl AsRef<Path>, num_gpus: usize) -> Result<Self, ort::Error> {
        let cfg = GpuPoolConfig::from_env();
        let mut senders = Vec::with_capacity(num_gpus * cfg.workers_per_gpu);
        let model_path = model_path.as_ref();

        for gpu_id in 0..num_gpus {
            for worker_idx in 0..cfg.workers_per_gpu {
                let session = Session::builder()?
                    .with_execution_providers([CUDAExecutionProvider::default()
                        .with_device_id(gpu_id as i32)
                        .build()])?
                    .commit_from_file(model_path)?;

                let (tx, rx) = mpsc::sync_channel(cfg.queue_capacity);
                let worker_cfg = cfg;
                thread::Builder::new()
                    .name(format!("gpu-infer-{gpu_id}-{worker_idx}"))
                    .spawn(move || gpu_inference_loop(rx, session, worker_cfg))
                    .expect("failed to spawn GPU inference thread");

                senders.push(tx);
            }
        }

        eprintln!(
            "GPU batch inference pool: {num_gpus} GPUs, workers_per_gpu={}, max_batch={}, timeout_ms={}, queue_capacity={} (total_workers={})",
            cfg.workers_per_gpu,
            cfg.max_batch_size,
            cfg.batch_timeout.as_millis(),
            cfg.queue_capacity,
            senders.len()
        );
        Ok(Self { senders })
    }

    pub fn policy(&self, gpu_idx: usize) -> GpuBatchPolicy {
        GpuBatchPolicy {
            request_tx: self.senders[gpu_idx % self.senders.len()].clone(),
        }
    }

    pub fn num_gpus(&self) -> usize {
        self.senders.len()
    }
}
