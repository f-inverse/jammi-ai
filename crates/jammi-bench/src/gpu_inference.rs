//! On-GPU embedding throughput/latency baseline — the perf-regression net the
//! GPU-inference optimizations (fp8, cuDNN/flash-attn, the memory scheduler,
//! batch coalescing) move a number against.
//!
//! ## Why this exists next to the CPU-hermetic `model_inference` tier
//!
//! `model_inference` deliberately serves on `Device::Cpu` so it runs inside the
//! hermetic `cargo test` net; it proves byte-determinism and a coarse CPU
//! code-path rate, but says nothing about the real device. This tier is its GPU
//! peer for the embedding path: it serves the real `generate_text_embeddings`
//! verb over the same tiny committed bundle on `gpu.device = 0`, and measures the
//! two quantities a GPU optimization actually moves — sustained throughput
//! (rows/s) and per-serve tail latency (p50/p99). It is gated (behind the `cuda`
//! feature and the `live-gpu-tests` lane) so default `cargo test` stays hermetic.
//!
//! ## Scope: embedding only
//!
//! The embedding forward is the primary GPU workload and the one the open
//! optimizations target (the encoder forward, its precision, its attention
//! kernel). The classification (`infer`) path is intentionally out of scope here
//! — it is not yet a validated GPU path (the gated `gpu_capability` suite covers
//! embeddings, fine-tune, and bf16, not classification), so a perf baseline is
//! premature until its correctness on-device is established.
//!
//! ## What it measures and gates
//!
//! After a warmup serve that pays the one-time model-load + PTX JIT cost, `iters`
//! measured serves are timed; from the per-serve wall-times come p50 / p99
//! (nearest-rank) and a rows/s throughput at the median. Two gates hold against a
//! committed same-device baseline (`baselines/gpu_inference.json`, captured on the
//! A100 the prove-lane runs):
//!
//! - **throughput floor** — rows/s must clear `baseline · (1 − threshold)` (the
//!   same relative-drop [`crate::rate_gate`] the CPU tier uses).
//! - **tail-latency ceiling** — p99 must stay under `baseline · (1 + threshold)`.
//!
//! Cross-repeat determinism is *recorded* (every measured serve's digest vs the
//! first) but not gated — GPU float bit-equality is not a property this codebase
//! asserts (the device correctness contract is the parity suite's tolerance).
//!
//! The threshold is generous by design (GPU wall-times vary with the pod, its
//! neighbours, and thermals); this is a coarse regression net, not a
//! micro-benchmark.

use std::error::Error;

use jammi_db::store::manifest::ComputeDevice;
use serde::{Deserialize, Serialize};

use crate::model_inference::{
    build_corpus, corpus_session_on_device, local_model_id, rows_per_s, serve_embed,
    ModelInferenceSpec, Row,
};
use crate::rate_gate::{RateGate, DEFAULT_REGRESSION_THRESHOLD};
use crate::report::{GpuInferenceTier, GpuLane, LatencyVerdict, Measurement, RateVerdict};

/// The CUDA device ordinal the tier serves on. Device 0 is the prove-lane A100.
const GPU_DEVICE: i32 = 0;

/// The committed on-GPU embedding baseline: the corpus shape and the embed
/// throughput + tail-latency baselines. On-disk at `baselines/gpu_inference.json`;
/// captured off-box by `rebuild-gpu-inference-spec` on the same device class the
/// gate runs on (A100).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInferenceSpec {
    /// Synthetic corpus row count — how many rows each serve embeds.
    pub row_count: usize,
    /// Corpus generation seed.
    pub corpus_seed: u64,
    /// Measured serves (after warmup) the percentiles fold over.
    pub iters: usize,
    /// Committed same-device embed throughput baseline, rows/s.
    pub baseline_embed_rows_per_s: f64,
    /// Committed same-device embed p99 serve latency baseline, ms.
    pub baseline_embed_p99_ms: f64,
}

impl GpuInferenceSpec {
    /// Path to the committed baseline, `baselines/gpu_inference.json`.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("gpu_inference.json")
    }

    /// Load the committed spec.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let text = std::fs::read_to_string(Self::path())?;
        Ok(serde_json::from_str(&text)?)
    }
}

/// The generation parameters a rebuild draws the baseline from.
#[derive(Debug, Clone, Copy)]
pub struct GpuInferenceParams {
    /// The synthetic corpus row count.
    pub row_count: usize,
    /// The corpus generation seed.
    pub corpus_seed: u64,
    /// Measured serves (after warmup).
    pub iters: usize,
}

/// The p50 and p99 of a set of per-serve latencies (ms), by nearest-rank on the
/// sorted samples. `p99` of a short sample is its slowest serve — the honest tail
/// for the small `iters` a coarse net runs.
fn percentiles_ms(mut latencies_ms: Vec<f64>) -> (f64, f64) {
    latencies_ms.sort_by(|a, b| a.total_cmp(b));
    let n = latencies_ms.len();
    let rank = |p: f64| -> usize {
        // Nearest-rank: ceil(p · n) clamped into [1, n], then 0-indexed.
        (((p * n as f64).ceil() as usize).clamp(1, n)) - 1
    };
    (latencies_ms[rank(0.50)], latencies_ms[rank(0.99)])
}

/// Serve the embed verb `warmup + iters` times on the GPU session, returning the
/// measured lane (throughput + p50/p99 + both gate verdicts + the determinism
/// verdict).
async fn measure_embed_lane(
    session: &std::sync::Arc<jammi_ai::session::InferenceSession>,
    model_id: &str,
    iters: usize,
    baseline_rows_per_s: f64,
    baseline_p99_ms: f64,
) -> Result<GpuLane, Box<dyn Error>> {
    // Warmup: pays the one-time model-load + PTX-JIT cost so it does not land in
    // the measured tail.
    let (first_digest, _warm_ms, rows) = serve_embed(session, model_id).await?;

    let mut latencies_ms = Vec::with_capacity(iters);
    let mut deterministic = true;
    for _ in 0..iters {
        let (digest, serve_ms, _rows) = serve_embed(session, model_id).await?;
        if digest != first_digest {
            deterministic = false;
        }
        latencies_ms.push(serve_ms);
    }

    let (p50_ms, p99_ms) = percentiles_ms(latencies_ms);
    // Throughput at the median serve — the representative steady-state rate.
    let rate = rows_per_s(rows, p50_ms);

    let rate_gate = RateGate::evaluate(rate, baseline_rows_per_s, DEFAULT_REGRESSION_THRESHOLD);
    let latency = LatencyVerdict::evaluate(p99_ms, baseline_p99_ms, DEFAULT_REGRESSION_THRESHOLD);

    Ok(GpuLane {
        rows_per_s: Measurement::measured(rate, "rows_per_s"),
        p50_ms: Measurement::measured(p50_ms, "ms"),
        p99_ms: Measurement::measured(p99_ms, "ms"),
        rate_gate: Some(RateVerdict {
            measured_pairs_per_s: rate_gate.measured,
            baseline_pairs_per_s: rate_gate.baseline,
            threshold: rate_gate.threshold,
            floor_pairs_per_s: rate_gate.floor,
            passed: rate_gate.passed,
            detail: rate_gate.detail(),
        }),
        latency_gate: Some(latency),
        deterministic,
    })
}

/// Assert the session actually resolved to a CUDA device, so a CPU-fallback
/// (a non-`cuda` build, or a machine with no GPU) fails the tier loudly instead
/// of silently baselining CPU numbers on the GPU lane.
fn require_cuda(device: ComputeDevice) -> Result<String, Box<dyn Error>> {
    match device {
        ComputeDevice::Cuda { ordinal } => Ok(format!("cuda:{ordinal}")),
        other => Err(format!(
            "gpu-inference requires a CUDA device but the session resolved to {other:?}; \
             build jammi-bench with --features cuda and run on a GPU"
        )
        .into()),
    }
}

/// Run the GPU-inference tier against the committed baseline: serve the embed
/// verb on `gpu.device = 0`, measure throughput + p50/p99, and assemble the tier
/// with the throughput-floor and tail-latency-ceiling gates.
pub async fn run(spec: &GpuInferenceSpec) -> Result<GpuInferenceTier, Box<dyn Error>> {
    let rows = build_corpus_from(spec.row_count, spec.corpus_seed);
    let (session, _dir) = corpus_session_on_device(&rows, GPU_DEVICE).await?;
    let device = require_cuda(session.compute_device())?;
    let embed_id = local_model_id(&ModelInferenceSpec::embed_model_dir())?;
    let embed = measure_embed_lane(
        &session,
        &embed_id,
        spec.iters,
        spec.baseline_embed_rows_per_s,
        spec.baseline_embed_p99_ms,
    )
    .await?;

    Ok(GpuInferenceTier {
        device,
        iters: spec.iters,
        embed,
    })
}

/// Whether every gate held: throughput cleared its floor and p99 stayed under
/// ceiling. (Cross-repeat determinism is recorded on the lane but not gated — see
/// the module docs.)
pub fn gates_passed(tier: &GpuInferenceTier) -> bool {
    tier.embed.rate_gate.as_ref().is_none_or(|v| v.passed)
        && tier.embed.latency_gate.as_ref().is_none_or(|v| v.passed)
}

/// Re-derive the committed baseline from a fresh serve on this device: measure
/// the embed verb and record its throughput + p99 baselines. The off-box one-shot
/// that writes `baselines/gpu_inference.json`, run on the target device class
/// (A100); the gate only ever loads and re-serves.
pub async fn rebuild_spec(params: GpuInferenceParams) -> Result<GpuInferenceSpec, Box<dyn Error>> {
    // Zeroed floor / infinite ceiling so the capture serve's gates are vacuously
    // wide; overwrite with the measured values below.
    let mut spec = GpuInferenceSpec {
        row_count: params.row_count,
        corpus_seed: params.corpus_seed,
        iters: params.iters,
        baseline_embed_rows_per_s: 0.0,
        baseline_embed_p99_ms: f64::INFINITY,
    };

    let tier = run(&spec).await?;
    spec.baseline_embed_rows_per_s = tier.embed.rows_per_s.value.unwrap_or(0.0);
    spec.baseline_embed_p99_ms = tier.embed.p99_ms.value.unwrap_or(0.0);
    Ok(spec)
}

/// Build the corpus for a `(row_count, seed)` without a full spec — the rebuild
/// path needs it before the baselines exist.
fn build_corpus_from(row_count: usize, corpus_seed: u64) -> Vec<Row> {
    // `build_corpus` reads only `row_count` and `corpus_seed` off the spec; the
    // baseline/digest fields are irrelevant to the corpus, so a throwaway spec is
    // the honest way to reuse the one corpus generator (no second copy of the
    // seed-rotation logic).
    let scratch = ModelInferenceSpec {
        row_count,
        corpus_seed,
        target_keys: Vec::new(),
        embed_digest: String::new(),
        infer_digest: String::new(),
        baseline_embed_rows_per_s: 0.0,
        baseline_infer_rows_per_s: 0.0,
    };
    build_corpus(&scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_pick_median_and_tail() {
        let (p50, p99) = percentiles_ms(vec![10.0, 20.0, 30.0, 40.0, 100.0]);
        assert_eq!(p50, 30.0, "median of five is the third");
        assert_eq!(p99, 100.0, "p99 of a short sample is its slowest serve");
    }

    #[test]
    fn percentiles_single_sample() {
        let (p50, p99) = percentiles_ms(vec![42.0]);
        assert_eq!(p50, 42.0);
        assert_eq!(p99, 42.0);
    }
}
