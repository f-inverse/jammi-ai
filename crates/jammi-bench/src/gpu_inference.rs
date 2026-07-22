//! On-GPU embedding throughput/latency observability — the GPU peer of the
//! CPU-hermetic `model_inference` tier, serving the real
//! `generate_text_embeddings` verb on a real CUDA device.
//!
//! ## Why this exists next to the CPU-hermetic `model_inference` tier
//!
//! `model_inference` deliberately serves on `Device::Cpu` so it runs inside the
//! hermetic `cargo test` net; it proves byte-determinism and a coarse CPU
//! code-path rate, but says nothing about the real device. This tier is its GPU
//! peer for the embedding path: it serves the real `generate_text_embeddings`
//! verb over the same tiny committed bundle on `gpu.device = 0`, and measures the
//! two quantities a GPU optimization actually moves — sustained throughput
//! (rows/s) and per-serve tail latency (p50/p99). It is gated behind the `cuda`
//! feature and the `live-gpu-tests` lane so default `cargo test` stays hermetic.
//!
//! ## Scope: embedding only
//!
//! The embedding forward is the primary GPU workload and the one open
//! optimizations target (the encoder forward, its precision, its attention
//! kernel). The classification (`infer`) path is intentionally out of scope here
//! — it is not yet a validated GPU path (the gated `gpu_capability` suite covers
//! embeddings, fine-tune, and bf16, not classification), so measuring it here is
//! premature until its correctness on-device is established.
//!
//! ## What is device-independent (hard-gated) vs device-dependent (recorded)
//!
//! An absolute throughput or tail latency is a property of `code × device ×
//! pod-conditions`, not of the code alone: the prove lane runs on an ephemeral
//! heterogeneous rented fleet (SXM4 / PCIe A100s, no pinning), so the same code
//! measures a different rate on every pod it happens to land on. Pinning an
//! absolute floor against that fleet gates pod variance, not a code regression.
//!
//! CPU↔GPU parity *is* a device-independent property of the code, and is
//! hard-gated in the separate `gpu_capability` suite, not here: the served
//! output must fall within tolerance of the CPU path. Cross-repeat
//! determinism is not a hard gate anywhere in this codebase — GPU float
//! bit-equality across repeats is not a property this codebase asserts — so
//! this tier records only the device-dependent quantities: throughput
//! (rows/s), tail latency (p50/p99), and cross-repeat determinism, all
//! *recorded* as observability
//! tagged with the concrete device that produced them, never asserted against a
//! remembered constant.
//!
//! The designated perf-regression gate for when a GPU-specific optimization
//! lands (fp8, cuDNN / flash-attention, a memory scheduler, batch coalescing) is
//! a **within-run A/B ratio** — parent-HEAD vs the PR change, measured back to
//! back on the *same* pod in the *same* run, so the device and its conditions
//! cancel by construction — not a resurrected absolute baseline.

use std::error::Error;

use jammi_db::store::manifest::ComputeDevice;

use crate::model_inference::{
    build_corpus, corpus_session_on_device, local_model_id, rows_per_s, serve_embed,
    ModelInferenceSpec, Row,
};
use crate::report::{GpuInferenceTier, GpuLane, Measurement};

/// The CUDA device ordinal the tier serves on. Device 0 is the prove-lane A100.
const GPU_DEVICE: i32 = 0;

/// The generation parameters the tier drives its corpus and measurement off of.
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
/// measured lane (throughput + p50/p99 + the determinism verdict). Recorded
/// observability only — no gate, per the module docs.
async fn measure_embed_lane(
    session: &std::sync::Arc<jammi_ai::session::InferenceSession>,
    model_id: &str,
    iters: usize,
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

    Ok(GpuLane {
        rows_per_s: Measurement::measured(rate, "rows_per_s"),
        p50_ms: Measurement::measured(p50_ms, "ms"),
        p99_ms: Measurement::measured(p99_ms, "ms"),
        deterministic,
    })
}

/// Assert the session actually resolved to a CUDA device, so a CPU-fallback
/// (a non-`cuda` build, or a machine with no GPU) fails the tier loudly instead
/// of silently recording CPU numbers on the GPU lane.
fn require_cuda(device: ComputeDevice) -> Result<u32, Box<dyn Error>> {
    match device {
        ComputeDevice::Cuda { ordinal } => Ok(ordinal),
        other => Err(format!(
            "gpu-inference requires a CUDA device but the session resolved to {other:?}; \
             build jammi-bench with --features cuda and run on a GPU"
        )
        .into()),
    }
}

/// The concrete CUDA device sub-class name (e.g. `NVIDIA A100-SXM4-80GB`) for
/// the ordinal the tier serves on — the provenance tag that makes a recorded
/// rate/latency interpretable on an ephemeral heterogeneous fleet where the same
/// code measures a different number on every pod it lands on.
///
/// Queried in-process through cudarc's device API (candle re-exports cudarc as
/// `candle_core::cuda::cudarc`) rather than shelling out: it is the same driver
/// handle the session's own CUDA backend already opened, needs no subprocess,
/// and cannot silently return another host's GPU.
#[cfg(feature = "cuda")]
fn cuda_device_name(ordinal: u32) -> Result<String, Box<dyn Error>> {
    use candle_core::cuda::cudarc::driver::result as cuda;
    cuda::init()?;
    let device = cuda::device::get(ordinal as i32)?;
    Ok(cuda::device::get_name(device)?)
}

/// Non-`cuda`-feature stub: unreachable in practice, since [`require_cuda`]
/// already fails the tier before this is ever called on a build with no CUDA
/// backend compiled in. Exists only so the tier compiles in the default
/// CPU-hermetic build.
#[cfg(not(feature = "cuda"))]
fn cuda_device_name(_ordinal: u32) -> Result<String, Box<dyn Error>> {
    Err("gpu-inference built without the cuda feature; no device to name".into())
}

/// Run the GPU-inference tier: serve the embed verb on `gpu.device = 0` and
/// record throughput, p50/p99 tail latency, and cross-repeat determinism,
/// tagged with the concrete device that served them.
pub async fn run(params: GpuInferenceParams) -> Result<GpuInferenceTier, Box<dyn Error>> {
    let rows = build_corpus_from(params.row_count, params.corpus_seed);
    let (session, _dir) = corpus_session_on_device(&rows, GPU_DEVICE).await?;
    let ordinal = require_cuda(session.compute_device())?;
    let device_name = cuda_device_name(ordinal)?;
    let embed_id = local_model_id(&ModelInferenceSpec::embed_model_dir())?;
    let embed = measure_embed_lane(&session, &embed_id, params.iters).await?;

    Ok(GpuInferenceTier {
        device: format!("cuda:{ordinal}"),
        device_name,
        iters: params.iters,
        embed,
    })
}

/// Build the corpus for a `(row_count, seed)` without a full spec — the tier
/// draws only these two fields off [`ModelInferenceSpec`], so a throwaway spec is
/// the honest way to reuse the one corpus generator (no second copy of the
/// seed-rotation logic).
fn build_corpus_from(row_count: usize, corpus_seed: u64) -> Vec<Row> {
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
