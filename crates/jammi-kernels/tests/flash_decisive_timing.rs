//! P6 Stage B decisive timing measurement (lead's directive, exclusive
//! box). Raw `crate::flash::flash_varlen_fwd`/`flash_varlen_bwd` wall
//! time at the FFI-boundary layer (no op/autograd overhead) — b8-s512
//! h16 d64 dense `cu_seqlens` and b1-s512, `deterministic` ON vs OFF,
//! at least 25 launches after 5 warmup, a device sync bracketing each launch
//! (candle's `BackendDevice::synchronize`, equivalent to a `cudaEvent`
//! pair for wall-time purposes — no async queue depth to hide behind
//! since each iteration syncs before starting the next), bf16 random
//! inputs at production amplitude (`|qkv| <= 18`, ModernBERT's typical
//! post-LN activation range).
//!
//! `#[ignore]`d: this needs an EXCLUSIVE GPU (shared-box timing is
//! invalid, `contract-P6-stage-B-v4.md` doesn't scope B1 timing at all —
//! this is the lead's own decisive-measurement request run separately).
//! Invoke with `cargo test --release -p jammi-kernels --features
//! flash-attn --test flash_decisive_timing -- --ignored --nocapture`.
//! Writes the artifact JSON itself (`crates/jammi-kernels/artifacts/
//! cuda-runs/...`) rather than relying on transcribing stdout.

#![cfg(feature = "flash-attn")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::backend::BackendDevice;
use candle_core::{CudaDevice, Device};
use half::bf16;
use jammi_kernels::flash::{
    dq_accum_splits, flash_varlen_bwd, flash_varlen_fwd, CuSeqlens, VarlenConfig, HEAD_DIM,
};

const NUM_HEADS: usize = 16;
const WARMUP: usize = 5;
const ITERS: usize = 25;

fn cuda_device() -> CudaDevice {
    Device::new_cuda(0)
        .expect("flash_decisive_timing requires an exclusive CUDA device")
        .as_cuda_device()
        .unwrap()
        .clone()
}

/// Deterministic-but-spread bf16 fill in `[-18, 18]` (production
/// amplitude) — a fixed irrational-multiplier fractional sequence, not an
/// RNG dependency; exact values are irrelevant to wall time (only
/// shape/dtype/finiteness matter for a timing harness), so this avoids
/// pulling in `rand` for a test-only concern.
fn random_bf16(n: usize) -> Vec<bf16> {
    (0..n)
        .map(|i| {
            let frac = (i as f64 * 0.618_033_988_749_895).fract();
            bf16::from_f64((frac - 0.5) * 36.0)
        })
        .collect()
}

struct Stats {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    n: usize,
}

fn stats(mut samples: Vec<f64>) -> Stats {
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    Stats {
        mean_ms: mean,
        min_ms: samples[0],
        max_ms: samples[n - 1],
        p50_ms: samples[n / 2],
        n,
    }
}

fn time_fwd(
    dev: &CudaDevice,
    qkv: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    cu: &CuSeqlens,
    cfg: &VarlenConfig,
) -> Stats {
    for _ in 0..WARMUP {
        let (o, lse) = flash_varlen_fwd(dev, qkv, cu, NUM_HEADS, cfg).unwrap();
        drop(o);
        drop(lse);
    }
    dev.synchronize().unwrap();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let (o, lse) = flash_varlen_fwd(dev, qkv, cu, NUM_HEADS, cfg).unwrap();
        dev.synchronize().unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        drop(o);
        drop(lse);
    }
    stats(samples)
}

#[allow(clippy::too_many_arguments)]
fn time_bwd(
    dev: &CudaDevice,
    qkv: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    cu: &CuSeqlens,
    o: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    lse: &candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    d_o: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    cfg: &VarlenConfig,
) -> Stats {
    for _ in 0..WARMUP {
        let d_qkv = flash_varlen_bwd(dev, qkv, cu, NUM_HEADS, o, lse, d_o, cfg).unwrap();
        drop(d_qkv);
    }
    dev.synchronize().unwrap();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let d_qkv = flash_varlen_bwd(dev, qkv, cu, NUM_HEADS, o, lse, d_o, cfg).unwrap();
        dev.synchronize().unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        drop(d_qkv);
    }
    stats(samples)
}

fn json_stats(s: &Stats) -> String {
    format!(
        "{{\"mean_ms\":{:.4},\"p50_ms\":{:.4},\"min_ms\":{:.4},\"max_ms\":{:.4},\"n\":{}}}",
        s.mean_ms, s.p50_ms, s.min_ms, s.max_ms, s.n
    )
}

#[test]
#[ignore]
fn decisive_timing_measurement() {
    let dev = cuda_device();
    let (major, minor) = dev.cuda_stream().context().compute_capability().unwrap();
    let sm_name = std::env::var("JAMMI_TIMING_BOX_NAME").unwrap_or_else(|_| "unknown".to_string());
    let driver = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string();
    let gpu_name = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string();

    let mut legs_json = Vec::new();

    for (leg_name, lengths) in [
        ("b8_s512_dense", vec![512usize; 8]),
        ("b1_s512", vec![512usize]),
    ] {
        let cu = CuSeqlens::from_lengths(&lengths, &dev).unwrap();
        let batch = lengths.len();
        let total_q: usize = lengths.iter().sum();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let host_qkv = random_bf16(total_q * 3 * NUM_HEADS * HEAD_DIM);
        let qkv = dev.clone_htod(&host_qkv).unwrap();

        // ---- forward, deterministic is a BACKWARD-only knob (module doc:
        // forward has no split-KV path reachable here regardless), so
        // fwd is timed ONCE per leg, not per deterministic setting.
        let fwd_cfg = VarlenConfig {
            softmax_scale: scale,
            window: None,
            deterministic: true,
        };
        let fwd_stats = time_fwd(&dev, &qkv, &cu, &fwd_cfg);
        eprintln!(
            "{leg_name} fwd: mean={:.4}ms p50={:.4}ms min={:.4}ms max={:.4}ms n={}",
            fwd_stats.mean_ms, fwd_stats.p50_ms, fwd_stats.min_ms, fwd_stats.max_ms, fwd_stats.n
        );

        // One real (o, lse) pair (post-warmup values, not timed) to feed backward.
        let (o, lse) = flash_varlen_fwd(&dev, &qkv, &cu, NUM_HEADS, &fwd_cfg).unwrap();
        let host_do = random_bf16(total_q * NUM_HEADS * HEAD_DIM);
        let d_o = dev.clone_htod(&host_do).unwrap();

        let mut bwd_json = Vec::new();
        for deterministic in [true, false] {
            let cfg = VarlenConfig {
                softmax_scale: scale,
                window: None,
                deterministic,
            };
            let splits = dq_accum_splits(&dev, batch, NUM_HEADS, deterministic).unwrap();
            let bwd_stats = time_bwd(&dev, &qkv, &cu, &o, &lse, &d_o, &cfg);
            eprintln!(
                "{leg_name} bwd deterministic={deterministic} splits={splits}: mean={:.4}ms p50={:.4}ms min={:.4}ms max={:.4}ms n={}",
                bwd_stats.mean_ms, bwd_stats.p50_ms, bwd_stats.min_ms, bwd_stats.max_ms, bwd_stats.n
            );
            bwd_json.push(format!(
                "{{\"deterministic\":{deterministic},\"splits\":{splits},\"stats\":{}}}",
                json_stats(&bwd_stats)
            ));
        }

        legs_json.push(format!(
            "{{\"leg\":\"{leg_name}\",\"batch\":{batch},\"total_q\":{total_q},\"num_heads\":{NUM_HEADS},\"head_dim\":{HEAD_DIM},\"fwd\":{},\"bwd\":[{}]}}",
            json_stats(&fwd_stats),
            bwd_json.join(",")
        ));
    }

    let sha = std::env::var("JAMMI_TIMING_SHA").unwrap_or_else(|_| "unknown".to_string());
    let artifact = format!(
        "{{\n  \"tip_sha\": \"{sha}\",\n  \"box\": \"{sm_name}\",\n  \"gpu\": \"{gpu_name}\",\n  \"driver\": \"{driver}\",\n  \"compute_capability\": [{major}, {minor}],\n  \"warmup\": {WARMUP},\n  \"iters\": {ITERS},\n  \"amplitude_note\": \"bf16 random fill in [-18, 18], production amplitude, deterministic irrational-multiplier fractional sequence (not an RNG dependency, values numerically irrelevant to wall time)\",\n  \"legs\": [{}]\n}}\n",
        legs_json.join(",")
    );

    let out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/cuda-runs")
        .join(format!(
            "2026-08-25-p6-b1-flash-timing-{sha}-a100-sxm4.json"
        ));
    std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();
    std::fs::write(&out_path, &artifact).unwrap();
    eprintln!("wrote {}", out_path.display());
}
