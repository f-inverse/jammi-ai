//! P6 Stage B decisive timing measurement (lead's directive, exclusive
//! box). Raw `crate::flash::flash_varlen_fwd_into`/`flash_varlen_bwd_into`
//! (KERNEL bracket, no allocation in the timed region) and
//! `flash_varlen_fwd`/`flash_varlen_bwd` (WRAPPER bracket, the public
//! convenience API that DOES allocate its outputs on every call) at
//! production geometry: H=16, D=64, b8-s512 dense and b1-s512,
//! `deterministic=true` only (see "Why deterministic-only" below), bf16
//! random inputs at production amplitude (`|qkv| <= 18`).
//!
//! # Fix round (10b1f3b audit, BLOCKING finding 3; tightened by a same-day
//! lead follow-up on a different kernel's unreproducible artifact)
//!
//! The audited version's single bracket (i) allocated `o`/`lse`/scratch/
//! `d_qkv` INSIDE the timed loop, (ii) memset `dq_accum` TWICE per bwd call
//! (once in `BwdScratch::alloc`, once again — unconditionally — inside
//! `flash_varlen_bwd_into`), (iii) used only 5 warmup iterations (mean-p50
//! gap was one max outlier over 25 samples), and (iv) reported a bimodal
//! non-deterministic leg (min 0.26ms, p50 0.376ms) with no explanation.
//! This version:
//!
//! - Separates a **KERNEL bracket** (buffers preallocated ONCE outside the
//!   loop, `_into` called on reused views — zero device allocation inside
//!   the timed region) from a **WRAPPER bracket** (the public
//!   `flash_varlen_fwd`/`flash_varlen_bwd`, which allocates fresh outputs
//!   every call — reported explicitly AS SUCH, never silently compared
//!   against nsys/kernel-only numbers elsewhere in this repo).
//! - `>= 20` warmup iterations, `>= 200` measured iterations.
//! - Reports **both min and median** (not just mean/p50); asserts
//!   steady-state (median within 5% of mean) and REFUSES rather than
//!   trusting a bimodal or outlier-dominated sample.
//! - Runs the ENTIRE measurement **twice** (`RUNS = 2`) and writes both
//!   runs into the artifact, so reproducibility is a stored fact, not an
//!   assertion made once and never checked again.
//! - Syncs the device (`BackendDevice::synchronize`) after every iteration
//!   — the "per-iteration synchronize" timing method, stated explicitly in
//!   the artifact's `method` block (no CUDA-event timer is used; candle's
//!   `cudarc` binding exposes stream sync, not a raw event-pair API at this
//!   crate's dependency version, so per-iteration sync is the method this
//!   harness actually has).
//!
//! ## Why deterministic-only (the non-det bwd leg is DROPPED, not fixed)
//!
//! `10b1f3b`'s artifact measured `deterministic=false` bwd as bimodal
//! (min 0.26ms / p50 0.376ms over 25 samples) with no root cause found.
//! `crate::ops::flash_attention_varlen` (the only production entry point,
//! see its own module doc's "Domain" section) never passes
//! `deterministic=false` — every real call site pins `true` — so the
//! non-deterministic path is not on the measured product's critical path.
//! Rather than publish an unexplained number, this harness drops that leg
//! entirely; `tests/flash_op_oracles.rs`'s
//! `poison_non_deterministic_dq_accum_is_a_dead_path_guard_not_reachable_via_the_op`
//! already covers its CORRECTNESS as a dead-path guard on the lower-level
//! primitive. A future timing pass that wants the non-det number should
//! first characterise the bimodality (e.g. `nsys` on the two clusters) —
//! that is out of scope for a fix round whose job is closing the audit's
//! BLOCKING findings, not opening a new investigation.
//!
//! `#[ignore]`d: needs an EXCLUSIVE GPU. Invoke with `cargo test --release
//! -p jammi-kernels --features flash-attn --test flash_decisive_timing --
//! --ignored --nocapture`. Requires `JAMMI_TIMING_BOX_NAME` (refused if
//! unset — see "No unknown fields" below) and a clean worktree (refused if
//! dirty, or if `JAMMI_TIMING_SHA` disagrees with `git rev-parse HEAD`).
//! Writes the artifact JSON itself.
//!
//! ## No "unknown" fields
//!
//! Every provenance field (`tip_sha`, `box`, `gpu`, `driver`,
//! `compute_capability`) is either derived from a live query that this
//! harness asserts succeeded and is non-empty, or read from a required env
//! var — never defaulted to the string `"unknown"`. `10b1f3b`'s version
//! defaulted three of these silently; a green artifact with unverifiable
//! provenance is evidence about the harness, not the kernel (`docs/
//! maintainer/cuda-kernel-guide.md` §4's "commit the artifact... carrying
//! the git_sha of the tip it measured" — the same principle extended to
//! every other provenance field).

#![cfg(feature = "flash-attn")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::backend::BackendDevice;
use candle_core::{CudaDevice, Device};
use half::bf16;
use jammi_kernels::flash::{
    flash_varlen_bwd, flash_varlen_bwd_into, flash_varlen_fwd, flash_varlen_fwd_into, BwdBuffers,
    BwdScratch, CuSeqlens, VarlenConfig, HEAD_DIM,
};

const NUM_HEADS: usize = 16;
const WARMUP: usize = 20;
const ITERS: usize = 200;
const RUNS: usize = 2;
/// Steady-state tolerance: `|median - mean| / mean` must not exceed this.
const STEADY_STATE_REL_TOL: f64 = 0.05;

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
/// shape/dtype/finiteness matter for a timing harness).
fn random_bf16(n: usize) -> Vec<bf16> {
    (0..n)
        .map(|i| {
            let frac = (i as f64 * 0.618_033_988_749_895).fract();
            bf16::from_f64((frac - 0.5) * 36.0)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    median_ms: f64,
    n: usize,
}

fn stats(mut samples: Vec<f64>) -> Stats {
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 0 {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    } else {
        samples[n / 2]
    };
    Stats {
        mean_ms: mean,
        min_ms: samples[0],
        max_ms: samples[n - 1],
        median_ms: median,
        n,
    }
}

/// Refuses (panics) rather than trusting a non-steady-state sample —
/// mirrors `assert!(x.is_finite() && ...)`'s affirmative-write discipline
/// (`docs/maintainer/cuda-kernel-guide.md` §3.7): a bimodal or
/// outlier-dominated distribution is a RED result, not a number to publish.
fn assert_steady_state(label: &str, s: &Stats) {
    let rel = (s.median_ms - s.mean_ms).abs() / s.mean_ms.max(1e-9);
    assert!(
        s.mean_ms.is_finite() && s.median_ms.is_finite() && rel <= STEADY_STATE_REL_TOL,
        "{label}: not steady-state — median {:.4}ms vs mean {:.4}ms differ by {:.1}% \
         (tolerance {:.0}%); n={} min={:.4}ms max={:.4}ms — refusing to publish a bimodal/\
         outlier-dominated sample rather than trusting it",
        s.median_ms,
        s.mean_ms,
        rel * 100.0,
        STEADY_STATE_REL_TOL * 100.0,
        s.n,
        s.min_ms,
        s.max_ms
    );
}

fn json_stats(s: &Stats) -> String {
    format!(
        "{{\"mean_ms\":{:.5},\"median_ms\":{:.5},\"min_ms\":{:.5},\"max_ms\":{:.5},\"n\":{}}}",
        s.mean_ms, s.median_ms, s.min_ms, s.max_ms, s.n
    )
}

/// `git rev-parse HEAD`, refusing on a dirty worktree or a
/// `JAMMI_TIMING_SHA` env var that disagrees — an artifact's `tip_sha` must
/// exactly identify the measured code, never a best-effort guess.
fn resolve_sha() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let rev = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(manifest_dir)
        .output()
        .expect("flash_decisive_timing: `git rev-parse HEAD` failed to launch");
    assert!(
        rev.status.success(),
        "flash_decisive_timing: git rev-parse HEAD exited non-zero: {}",
        String::from_utf8_lossy(&rev.stderr)
    );
    let sha = String::from_utf8(rev.stdout).unwrap().trim().to_string();
    assert!(
        !sha.is_empty(),
        "flash_decisive_timing: git rev-parse HEAD returned an empty SHA"
    );

    let status = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(manifest_dir)
        .output()
        .expect("flash_decisive_timing: `git status --porcelain` failed to launch");
    assert!(
        status.status.success(),
        "flash_decisive_timing: git status --porcelain exited non-zero: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let dirty = !String::from_utf8_lossy(&status.stdout).trim().is_empty();
    assert!(
        !dirty,
        "flash_decisive_timing: worktree is dirty — an artifact's tip_sha must exactly identify \
         the measured code; commit or stash before running this harness:\n{}",
        String::from_utf8_lossy(&status.stdout)
    );

    if let Ok(env_sha) = std::env::var("JAMMI_TIMING_SHA") {
        assert_eq!(
            env_sha, sha,
            "flash_decisive_timing: JAMMI_TIMING_SHA={env_sha} disagrees with \
             `git rev-parse HEAD`={sha} — refusing to write an artifact whose provenance is \
             ambiguous"
        );
    }
    sha
}

/// Required env var — refused (not defaulted to `"unknown"`) if unset.
fn require_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!(
            "flash_decisive_timing: {name} must be set — this harness refuses to write an \
             artifact with an \"unknown\" field"
        )
    })
}

/// One `nvidia-smi --query-gpu=<field>` value, refused if the command fails
/// or returns empty (never silently defaulted).
fn nvidia_smi_field(field: &str) -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            format!("--query-gpu={field}"),
            "--format=csv,noheader".into(),
        ])
        .output()
        .unwrap_or_else(|e| {
            panic!("flash_decisive_timing: nvidia-smi --query-gpu={field} failed to launch: {e}")
        });
    assert!(
        out.status.success(),
        "flash_decisive_timing: nvidia-smi --query-gpu={field} exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8(out.stdout).unwrap().trim().to_string();
    assert!(
        !s.is_empty(),
        "flash_decisive_timing: nvidia-smi --query-gpu={field} returned an empty value"
    );
    s
}

/// KERNEL bracket: `qkv`/`o`/`lse` are ALL preallocated by the caller
/// before this runs — zero device allocation inside the timed loop.
fn time_fwd_kernel(
    dev: &CudaDevice,
    qkv: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    o: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    lse: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    cu: &CuSeqlens,
    cfg: &VarlenConfig,
) -> Stats {
    for _ in 0..WARMUP {
        flash_varlen_fwd_into(
            dev,
            qkv.as_view(),
            cu,
            o.as_view_mut(),
            lse.as_view_mut(),
            NUM_HEADS,
            cfg,
        )
        .unwrap();
    }
    dev.synchronize().unwrap();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        flash_varlen_fwd_into(
            dev,
            qkv.as_view(),
            cu,
            o.as_view_mut(),
            lse.as_view_mut(),
            NUM_HEADS,
            cfg,
        )
        .unwrap();
        dev.synchronize().unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    stats(samples)
}

/// WRAPPER bracket: the public `flash_varlen_fwd`, which allocates fresh
/// `o`/`lse` on EVERY call — measures the real per-call public-API cost,
/// documented as such (never compared against a kernel-only number).
fn time_fwd_wrapper(
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

/// KERNEL bracket: `qkv`/`o`/`lse`/`d_o`/scratch/`d_qkv` are ALL
/// preallocated by the caller — zero device allocation inside the timed
/// loop. The `dq_accum` zero-fill `flash_varlen_bwd_into` performs
/// internally (unconditional when `cfg.deterministic`, `flash/mod.rs`'s own
/// doc on `BwdBuffers::dq_accum`) is NOT eliminable from this bracket
/// without bypassing the safe API with raw FFI (out of scope for a test
/// harness) — it remains INSIDE `kernel_ms`, stated here rather than
/// silently claimed away as "pure launch time".
#[allow(clippy::too_many_arguments)]
fn time_bwd_kernel(
    dev: &CudaDevice,
    qkv: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    cu: &CuSeqlens,
    o: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    lse: &candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    d_o: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    d_qkv: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    scratch: &mut BwdScratch,
    cfg: &VarlenConfig,
) -> Stats {
    let run = |dev: &CudaDevice,
               d_qkv: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
               scratch: &mut BwdScratch| {
        flash_varlen_bwd_into(
            dev,
            cu,
            NUM_HEADS,
            BwdBuffers {
                qkv: qkv.as_view(),
                o: o.as_view(),
                lse: lse.as_view(),
                d_o: d_o.as_view(),
                d_qkv: d_qkv.as_view_mut(),
                softmax_d: scratch.softmax_d.as_view_mut(),
                dq_accum: scratch.dq_accum.as_view_mut(),
                dq_accum_splits: scratch.splits,
            },
            cfg,
        )
        .unwrap();
    };
    for _ in 0..WARMUP {
        run(dev, d_qkv, scratch);
    }
    dev.synchronize().unwrap();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        run(dev, d_qkv, scratch);
        dev.synchronize().unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    stats(samples)
}

/// WRAPPER bracket: the public `flash_varlen_bwd`, which allocates fresh
/// scratch + `d_qkv` on EVERY call.
#[allow(clippy::too_many_arguments)]
fn time_bwd_wrapper(
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

/// One full measurement pass (all legs, both brackets, fwd+bwd) — run
/// `RUNS` times to prove reproducibility (the lead's follow-up clause).
fn one_full_run(dev: &CudaDevice) -> Vec<String> {
    let mut legs_json = Vec::new();
    for (leg_name, lengths) in [
        ("b8_s512_dense", vec![512usize; 8]),
        ("b1_s512", vec![512usize]),
    ] {
        let cu = CuSeqlens::from_lengths(&lengths, dev).unwrap();
        let batch = lengths.len();
        let total_q: usize = lengths.iter().sum();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let geom = cu.geometry(NUM_HEADS).unwrap();

        let host_qkv = random_bf16(total_q * 3 * NUM_HEADS * HEAD_DIM);
        let qkv = dev.clone_htod(&host_qkv).unwrap();
        let cfg = VarlenConfig {
            softmax_scale: scale,
            window: None,
            deterministic: true,
        };

        // ---- forward: kernel bracket (preallocated) then wrapper bracket.
        // SAFETY: uninitialised outputs the kernel fully overwrites (same
        // allocation `flash_varlen_fwd` itself makes).
        let mut o_k = unsafe { dev.alloc::<bf16>(geom.o_len()) }.unwrap();
        let mut lse_k = unsafe { dev.alloc::<f32>(geom.lse_len()) }.unwrap();
        let fwd_kernel = time_fwd_kernel(dev, &qkv, &mut o_k, &mut lse_k, &cu, &cfg);
        assert_steady_state(&format!("{leg_name} fwd kernel"), &fwd_kernel);
        eprintln!(
            "{leg_name} fwd KERNEL: mean={:.4}ms median={:.4}ms min={:.4}ms max={:.4}ms n={}",
            fwd_kernel.mean_ms,
            fwd_kernel.median_ms,
            fwd_kernel.min_ms,
            fwd_kernel.max_ms,
            fwd_kernel.n
        );
        let fwd_wrapper = time_fwd_wrapper(dev, &qkv, &cu, &cfg);
        assert_steady_state(&format!("{leg_name} fwd wrapper"), &fwd_wrapper);
        eprintln!(
            "{leg_name} fwd WRAPPER: mean={:.4}ms median={:.4}ms min={:.4}ms max={:.4}ms n={}",
            fwd_wrapper.mean_ms,
            fwd_wrapper.median_ms,
            fwd_wrapper.min_ms,
            fwd_wrapper.max_ms,
            fwd_wrapper.n
        );

        // One real (o, lse) pair (not timed) to feed backward.
        let (o, lse) = flash_varlen_fwd(dev, &qkv, &cu, NUM_HEADS, &cfg).unwrap();
        let host_do = random_bf16(total_q * NUM_HEADS * HEAD_DIM);
        let d_o = dev.clone_htod(&host_do).unwrap();

        // ---- backward, deterministic=true ONLY (see module doc).
        let mut scratch_k = BwdScratch::alloc(dev, &geom, cfg.deterministic).unwrap();
        // SAFETY: as `flash_varlen_bwd`'s own allocation.
        let mut d_qkv_k = unsafe { dev.alloc::<bf16>(geom.qkv_len()) }.unwrap();
        let bwd_kernel = time_bwd_kernel(
            dev,
            &qkv,
            &cu,
            &o,
            &lse,
            &d_o,
            &mut d_qkv_k,
            &mut scratch_k,
            &cfg,
        );
        assert_steady_state(&format!("{leg_name} bwd kernel"), &bwd_kernel);
        eprintln!(
            "{leg_name} bwd KERNEL (det=true): mean={:.4}ms median={:.4}ms min={:.4}ms max={:.4}ms n={}",
            bwd_kernel.mean_ms, bwd_kernel.median_ms, bwd_kernel.min_ms, bwd_kernel.max_ms, bwd_kernel.n
        );
        let bwd_wrapper = time_bwd_wrapper(dev, &qkv, &cu, &o, &lse, &d_o, &cfg);
        assert_steady_state(&format!("{leg_name} bwd wrapper"), &bwd_wrapper);
        eprintln!(
            "{leg_name} bwd WRAPPER (det=true): mean={:.4}ms median={:.4}ms min={:.4}ms max={:.4}ms n={}",
            bwd_wrapper.mean_ms, bwd_wrapper.median_ms, bwd_wrapper.min_ms, bwd_wrapper.max_ms, bwd_wrapper.n
        );

        legs_json.push(format!(
            "{{\"leg\":\"{leg_name}\",\"batch\":{batch},\"total_q\":{total_q},\"num_heads\":{NUM_HEADS},\
             \"head_dim\":{HEAD_DIM},\
             \"fwd_kernel_ms\":{},\"fwd_wrapper_ms\":{},\
             \"bwd_deterministic_kernel_ms\":{},\"bwd_deterministic_wrapper_ms\":{}}}",
            json_stats(&fwd_kernel),
            json_stats(&fwd_wrapper),
            json_stats(&bwd_kernel),
            json_stats(&bwd_wrapper),
        ));
    }
    legs_json
}

#[test]
#[ignore]
fn decisive_timing_measurement() {
    let dev = cuda_device();
    let (major, minor) = dev.cuda_stream().context().compute_capability().unwrap();
    let sha = resolve_sha();
    let sm_name = require_env("JAMMI_TIMING_BOX_NAME");
    let driver = nvidia_smi_field("driver_version");
    let gpu_name = nvidia_smi_field("name");

    let mut runs_json = Vec::with_capacity(RUNS);
    for run_idx in 0..RUNS {
        eprintln!("=== run {run_idx} of {RUNS} ===");
        let legs = one_full_run(&dev);
        runs_json.push(format!(
            "{{\"run\":{run_idx},\"legs\":[{}]}}",
            legs.join(",")
        ));
    }

    let method = "kernel_ms: flash_varlen_{fwd,bwd}_into on buffers preallocated ONCE outside \
                  the timed loop (zero device allocation in the timed region; bwd's internal \
                  dq_accum zero-fill, unconditional when deterministic, remains inside kernel_ms \
                  — not eliminable without an unsafe raw-FFI bypass). wrapper_ms: the public \
                  flash_varlen_{fwd,bwd}, which allocates fresh outputs every call (the real \
                  per-call public-API cost, never compared against kernel_ms or an external \
                  nsys kernel-only number). sync_method: per-iteration \
                  BackendDevice::synchronize() after every launch (no CUDA-event timer available \
                  at this crate's cudarc binding). warmup=20, iters=200 per bracket, run twice \
                  (RUNS=2, both included below) to demonstrate reproducibility. steady-state: \
                  |median-mean|/mean <= 5% asserted per bracket per run — the harness panics \
                  (refuses to write the artifact) rather than publish a bimodal/outlier sample. \
                  non-deterministic bwd is DROPPED (not measured): it is unreachable from the \
                  only production entry point (ops::flash_attention_varlen pins \
                  deterministic=true at every real call site) and 10b1f3b's artifact found it \
                  bimodal with no root cause — publishing an unexplained number was rejected in \
                  favour of dropping the leg; see this file's module doc.";

    let artifact = format!(
        "{{\n  \"tip_sha\": \"{sha}\",\n  \"box\": \"{sm_name}\",\n  \"gpu\": \"{gpu_name}\",\n  \
         \"driver\": \"{driver}\",\n  \"compute_capability\": [{major}, {minor}],\n  \
         \"warmup\": {WARMUP},\n  \"iters\": {ITERS},\n  \"runs\": {RUNS},\n  \
         \"amplitude_note\": \"bf16 random fill in [-18, 18], production amplitude, \
         deterministic irrational-multiplier fractional sequence (not an RNG dependency)\",\n  \
         \"method\": \"{method}\",\n  \"measurements\": [{}]\n}}\n",
        runs_json.join(",")
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
