//! On-GPU throughput/latency baseline — the perf-regression net the rest of the
//! GPU-inference tiers (fp8, cuDNN/flash-attn, the memory scheduler, batch
//! coalescing) move a number against.
//!
//! ## Why this exists next to the CPU-hermetic `model_inference` tier
//!
//! `model_inference` deliberately serves on `Device::Cpu` so it runs inside the
//! hermetic `cargo test` net; it can prove byte-determinism and a coarse CPU
//! code-path rate, but it says nothing about the real device. This tier is its
//! GPU peer: it serves the same real verbs (`generate_text_embeddings`,
//! `infer`) over the same tiny committed bundles on `gpu.device = 0`, and
//! measures the two quantities a GPU optimization actually moves — sustained
//! throughput (rows/s) and per-serve tail latency (p50/p99). It is gated (behind
//! the `cuda` feature and the `live-gpu-tests` lane) so default `cargo test`
//! stays hermetic.
//!
//! ## What it measures and gates
//!
//! For each verb: after a warmup serve that pays the one-time model-load + PTX
//! JIT cost, `iters` measured serves are timed. From the per-serve wall-times
//! come p50 / p99 (nearest-rank) and a rows/s throughput taken at the median.
//! Two gates hold, both against a committed same-device baseline
//! (`baselines/gpu_inference.json`, captured on the A100 the prove-lane runs):
//!
//! - **throughput floor** — rows/s must clear `baseline · (1 − threshold)`
//!   (the same relative-drop [`crate::rate_gate`] the CPU tier uses), catching a
//!   throughput regression.
//! - **tail-latency ceiling** — p99 must stay under `baseline · (1 + threshold)`,
//!   catching a latency regression that leaves mean throughput unmoved.
//!
//! A determinism check rides along: every measured serve's digest must equal the
//! first, proving the GPU kernel path is deterministic across repeats on the box.
//!
//! The threshold is generous by design (GPU wall-times vary with the pod, its
//! neighbours, and thermals); this is a coarse 2×-regression net, not a
//! micro-benchmark.

use std::error::Error;

use jammi_db::store::manifest::ComputeDevice;
use serde::{Deserialize, Serialize};

use crate::model_inference::{
    build_corpus, corpus_session_on_device, local_model_id, rows_per_s, serve_embed, serve_infer,
    ModelInferenceSpec, Row,
};
use crate::rate_gate::{RateGate, DEFAULT_REGRESSION_THRESHOLD};
use crate::report::{GpuInferenceTier, GpuLane, LatencyVerdict, Measurement, RateVerdict};

/// The CUDA device ordinal the tier serves on. Device 0 is the prove-lane A100.
const GPU_DEVICE: i32 = 0;

/// The committed on-GPU baseline: the corpus shape, the per-verb throughput and
/// tail-latency baselines, and the fold width. On-disk at
/// `baselines/gpu_inference.json`; captured off-box by `rebuild-gpu-inference-spec`
/// on the same device class the gate runs on (A100).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInferenceSpec {
    /// Synthetic corpus row count — how many rows each serve embeds/infers.
    pub row_count: usize,
    /// Corpus generation seed.
    pub corpus_seed: u64,
    /// Measured serves per verb (after warmup) the percentiles fold over.
    pub iters: usize,
    /// Infer digest fold width — how many corpus rows the infer determinism
    /// digest walks, in committed order.
    pub target_keys: Vec<String>,
    /// Committed same-device embed throughput baseline, rows/s.
    pub baseline_embed_rows_per_s: f64,
    /// Committed same-device embed p99 serve latency baseline, ms.
    pub baseline_embed_p99_ms: f64,
    /// Committed same-device infer throughput baseline, rows/s.
    pub baseline_infer_rows_per_s: f64,
    /// Committed same-device infer p99 serve latency baseline, ms.
    pub baseline_infer_p99_ms: f64,
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
    /// Measured serves per verb (after warmup).
    pub iters: usize,
    /// How many corpus rows the infer digest folds over.
    pub target_count: usize,
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

/// Serve one verb `warmup + iters` times on the GPU session, returning the
/// measured lane (throughput + p50/p99 + both gate verdicts + the determinism
/// verdict). `serve` runs one serve and yields `(digest, serve_ms, rows)`.
async fn measure_lane<F, Fut>(
    iters: usize,
    baseline_rows_per_s: f64,
    baseline_p99_ms: f64,
    mut serve: F,
) -> Result<GpuLane, Box<dyn Error>>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<(String, f64, usize), Box<dyn Error>>>,
{
    // Warmup: pays the one-time model-load + PTX-JIT cost so it does not land in
    // the measured tail.
    let (first_digest, _warm_ms, rows) = serve().await?;

    let mut latencies_ms = Vec::with_capacity(iters);
    let mut deterministic = true;
    for _ in 0..iters {
        let (digest, serve_ms, _rows) = serve().await?;
        if digest != first_digest {
            deterministic = false;
        }
        latencies_ms.push(serve_ms);
    }

    let (p50_ms, p99_ms) = percentiles_ms(latencies_ms.clone());
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

/// Run the GPU-inference tier against the committed baseline: serve both verbs on
/// `gpu.device = 0`, measure throughput + p50/p99, and assemble the tier with the
/// throughput-floor and tail-latency-ceiling gates.
pub async fn run(spec: &GpuInferenceSpec) -> Result<GpuInferenceTier, Box<dyn Error>> {
    let rows = build_corpus_from(spec.row_count, spec.corpus_seed);

    // Embed lane.
    let (session, _dir) = corpus_session_on_device(&rows, GPU_DEVICE).await?;
    let device = require_cuda(session.compute_device())?;
    let embed_id = local_model_id(&ModelInferenceSpec::embed_model_dir())?;
    let embed = measure_lane(
        spec.iters,
        spec.baseline_embed_rows_per_s,
        spec.baseline_embed_p99_ms,
        || serve_embed(&session, &embed_id),
    )
    .await?;

    // Infer lane — a fresh session so the embed serves do not warm its cache.
    let (session, _dir) = corpus_session_on_device(&rows, GPU_DEVICE).await?;
    require_cuda(session.compute_device())?;
    let infer_id = local_model_id(&ModelInferenceSpec::classifier_model_dir())?;
    let infer = measure_lane(
        spec.iters,
        spec.baseline_infer_rows_per_s,
        spec.baseline_infer_p99_ms,
        || serve_infer(&session, &infer_id, &spec.target_keys),
    )
    .await?;

    Ok(GpuInferenceTier {
        device,
        iters: spec.iters,
        embed,
        infer,
    })
}

/// Whether every gate held: both verbs deterministic across repeats, both
/// throughputs cleared their floors, and both p99 latencies stayed under ceiling.
pub fn gates_passed(tier: &GpuInferenceTier) -> bool {
    [&tier.embed, &tier.infer].iter().all(|lane| {
        lane.deterministic
            && lane.rate_gate.as_ref().is_none_or(|v| v.passed)
            && lane.latency_gate.as_ref().is_none_or(|v| v.passed)
    })
}

/// Re-derive the committed baseline from a fresh serve on this device: measure
/// both verbs and record their throughput + p99 baselines. The off-box one-shot
/// that writes `baselines/gpu_inference.json`, run on the target device class
/// (A100); the gate only ever loads and re-serves.
pub async fn rebuild_spec(params: GpuInferenceParams) -> Result<GpuInferenceSpec, Box<dyn Error>> {
    let rows = build_corpus_from(params.row_count, params.corpus_seed);
    let target_keys: Vec<String> = rows
        .iter()
        .take(params.target_count)
        .map(|r| r.id.clone())
        .collect();

    // Zeroed baselines so the first measurement's rate gate is vacuously wide;
    // we overwrite them with the measured values below.
    let mut spec = GpuInferenceSpec {
        row_count: params.row_count,
        corpus_seed: params.corpus_seed,
        iters: params.iters,
        target_keys: target_keys.clone(),
        baseline_embed_rows_per_s: 0.0,
        baseline_embed_p99_ms: f64::INFINITY,
        baseline_infer_rows_per_s: 0.0,
        baseline_infer_p99_ms: f64::INFINITY,
    };

    let tier = run(&spec).await?;
    spec.baseline_embed_rows_per_s = tier.embed.rows_per_s.value.unwrap_or(0.0);
    spec.baseline_embed_p99_ms = tier.embed.p99_ms.value.unwrap_or(0.0);
    spec.baseline_infer_rows_per_s = tier.infer.rows_per_s.value.unwrap_or(0.0);
    spec.baseline_infer_p99_ms = tier.infer.p99_ms.value.unwrap_or(0.0);
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
