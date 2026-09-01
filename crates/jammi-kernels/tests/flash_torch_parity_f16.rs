//! O0(a)'s fp16 twin (campaign #443 D2/D3): parity vs torch's OWN vendored
//! FA2 at fp16, mirroring `tests/flash_torch_parity.rs`'s structure EXACTLY
//! — same `LEGS` sweep, same production amplitude/spread, same
//! from-scratch-f64-TRUTH-anchored, truth-relative bound (never an absolute
//! ULP floor, per `docs/maintainer/cuda-kernel-guide.md` §3.8), same two RED
//! controls. Only three things differ from the bf16 file: (1) the fixture
//! files are `f16_`-prefixed and loaded via `generate_fixtures.py --dtype
//! float16` (this directory's own `sidecar_f16.json`, not `sidecar.json`);
//! (2) `q`/`k`/`v`/`grad_out` are stored as NATIVE numpy `float16` (`'<f2'`)
//! rather than the bf16 legs' `int16`-bit-pattern workaround — candle-core
//! 0.11.0's `npy.rs` `Header::parse` DOES map `"f2"`/`"e"` to `DType::F16`
//! (unlike bf16, which has no descr mapping at all), so `Tensor::read_npy`
//! loads these directly, no bit-reinterpretation needed; (3) the FFI calls
//! are `crate::flash::flash_varlen_{fwd,bwd}_f16` (the fp16 twins), not the
//! bf16 originals.
//!
//! Runs `crate::flash::flash_varlen_fwd_f16`/`flash_varlen_bwd_f16` DIRECTLY
//! (the FFI-boundary layer), not `ops::flash_attention_varlen` — same
//! layering rationale as the bf16 file's own module doc.
//!
//! # Truth-relative bound (see `flash_torch_parity.rs`'s "Fix round" section
//! for the full derivation this file inherits unchanged)
//!
//! For each of `o`/`lse`/`dq`/`dk`/`dv`:
//!
//! ```text
//! max|jammi - truth|  <=  slack * max|torchFA - truth|
//! mean|jammi - truth| <=  slack * mean|torchFA - truth|
//! ```
//!
//! `torchFA`'s own distance to truth is computed HERE, live, from the SAME
//! loaded fixture tensors — never trusted from a precomputed sidecar number
//! (family F). `slack` is WIDER than the bf16 file's `1.5` — see
//! [`TRUTH_RELATIVE_SLACK`]'s own doc for why fp16's narrower 11-bit
//! significand (vs bf16's wider 8-bit exponent / 8-bit mantissa split)
//! changes the SHAPE of the divergence this bound must tolerate, not just
//! its magnitude, and why that is a per-regime derived choice rather than a
//! constant carried over from the bf16 file unexamined
//! (`docs/maintainer/cuda-kernel-guide.md` §3's "f16 per-op reference-regime
//! table" doctrine, applied here to FA2 itself rather than a CPU op).
//!
//! # RED controls (must fail this oracle — proves it discriminates)
//!
//! Identical injections to the bf16 file's own two controls, at fp16:
//! 1. `softmax_scale * 1.05` on the jammi side only.
//! 2. Window radius `w +/- 1`.

#![cfg(feature = "flash-attn")]

use std::path::{Path, PathBuf};

use candle_core::{CudaDevice, DType, Device, Tensor};
use half::f16;
use jammi_kernels::flash::{CuSeqlens, VarlenConfig};

fn cuda_device() -> Option<CudaDevice> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d.as_cuda_device().unwrap().clone()),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "flash_torch_parity_f16: JAMMI_REQUIRE_CUDA is set but no CUDA device could \
                     be acquired — a silent skip here is not acceptable: {e}"
                );
            }
            eprintln!("flash_torch_parity_f16: skipping — no CUDA device available ({e})");
            None
        }
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flash_reference")
}

/// The fp16-exact inputs (`q`/`k`/`v`/`grad_out`), `f16_`-prefixed, stored
/// as NATIVE `float16` `.npy` (see the module doc's item (2)) —
/// `Tensor::read_npy` loads these directly as `DType::F16`, no
/// bit-reinterpretation needed (unlike the bf16 file's `load_bf16_exact`).
fn load_f16_exact(leg: &str, name: &str) -> Tensor {
    let path = fixtures_dir().join(format!("f16_{leg}_{name}.npy"));
    let t = Tensor::read_npy(&path)
        .unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()));
    assert_eq!(
        t.dtype(),
        DType::F16,
        "{}: expected native F16 storage (numpy '<f2' descr), got {:?} — \
         generate_fixtures.py's --dtype float16 storage convention changed without this loader \
         following it",
        path.display(),
        t.dtype()
    );
    t
}

fn load_f32(leg: &str, name: &str) -> Tensor {
    let path = fixtures_dir().join(format!("f16_{leg}_{name}.npy"));
    Tensor::read_npy(&path).unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()))
}

/// One B0 leg: `(name, lengths, window_radius)`. Mirrors
/// `flash_torch_parity.rs`'s own `LEGS` (and `generate_fixtures.py`'s
/// `LEGS` list) exactly — the SAME sweep at fp16.
struct Leg {
    name: &'static str,
    lengths: &'static [usize],
    window: Option<u32>,
}

const LEGS: &[Leg] = &[
    Leg {
        name: "b1_s512",
        lengths: &[512],
        window: None,
    },
    Leg {
        name: "b1_s512_win64",
        lengths: &[512],
        window: Some(64),
    },
    Leg {
        name: "b8_s512",
        lengths: &[512, 500, 512, 480, 512, 505, 490, 512],
        window: None,
    },
    Leg {
        name: "b8_s512_win64",
        lengths: &[512, 500, 512, 480, 512, 505, 490, 512],
        window: Some(64),
    },
    Leg {
        name: "b8_s128",
        lengths: &[128, 120, 128, 100, 128, 115, 128, 128],
        window: None,
    },
    Leg {
        name: "b1_tile129",
        lengths: &[129],
        window: None,
    },
    Leg {
        name: "b1_tile257",
        lengths: &[257],
        window: None,
    },
    Leg {
        name: "prefix_mixed",
        lengths: &[512, 300, 129, 64],
        window: None,
    },
];

const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const SOFTMAX_SCALE: f32 = 0.125; // 1/sqrt(64)

/// The truth-relative slack factor — see the module doc's "Truth-relative
/// bound" section. WIDER than the bf16 file's `1.5`: fp16's significand is
/// narrower (10 explicit mantissa bits vs bf16's 7), so a single rounding
/// event at fp16 carries a larger relative error than the SAME event at
/// bf16 (`no-producer: derived from fp16 vs bf16's mantissa-bit-count
/// difference, IEEE 754-2019 §3.6`) — jammi's own fp16 kernel and torch's
/// cross-build fp16 FA2 reference are still two DIFFERENT fp16 kernels
/// (same FMA-fusion/fast-math divergence the bf16 file's own doc explains),
/// so this is not "fp16 is universally 2x worse than bf16", only that the
/// SLACK this truth-relative bound needs to stay non-vacuous scales with
/// the dtype's own rounding granularity — `2.0`, not `1.5`, until a live
/// pod measurement (this file's own `o_lse_dq_dk_dv_match_truth...` test,
/// which PRINTS every leg's measured ratio via `eprintln!`) shows the real
/// achieved ratio is tighter, at which point this constant should be
/// re-derived from that measurement rather than left here unexamined.
const TRUTH_RELATIVE_SLACK: f64 = 2.0;

/// Packs `q`/`k`/`v` ([total_q, H, D] each, f16) into `[total_q, 3, H, D]`
/// f16 — identical layout convention to the bf16 file's own `pack_qkv`,
/// dtype aside.
fn pack_qkv(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    Tensor::stack(&[q, k, v], 1).unwrap()
}

/// `(max, mean)` absolute difference, computed in f64 throughout.
fn max_mean_abs_diff(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let d = (a.to_dtype(DType::F64).unwrap() - b.to_dtype(DType::F64).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap();
    let max = d.max(0).unwrap().to_scalar::<f64>().unwrap();
    let mean = d.mean_all().unwrap().to_scalar::<f64>().unwrap();
    (max, mean)
}

fn assert_finite(t: &Tensor, what: &str) {
    let v = t
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap()
        .to_vec1::<f64>()
        .unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{what}: contains a non-finite value"
    );
}

/// Asserts `max|jammi - truth| <= slack * max|torchFA - truth|` AND the
/// same on the mean, finiteness-affirmative first (`docs/maintainer/
/// cuda-kernel-guide.md` §3.7: `assert!(x.is_finite() && x <= bound)`,
/// never a negated form).
fn assert_truth_relative(
    what: &str,
    jammi: &Tensor,
    torch_ref: &Tensor,
    truth: &Tensor,
    slack: f64,
) {
    let (jammi_max, jammi_mean) = max_mean_abs_diff(jammi, truth);
    let (ref_max, ref_mean) = max_mean_abs_diff(torch_ref, truth);
    let bound_max = slack * ref_max;
    let bound_mean = slack * ref_mean;
    eprintln!(
        "{what}: |jammi-truth| max={jammi_max:.6e} mean={jammi_mean:.6e} | |torchFA-truth| \
         max={ref_max:.6e} mean={ref_mean:.6e} | bounds max<={bound_max:.6e} mean<={bound_mean:.6e} \
         | ratio(max)={:.4} ratio(mean)={:.4}",
        jammi_max / ref_max.max(f64::MIN_POSITIVE),
        jammi_mean / ref_mean.max(f64::MIN_POSITIVE),
    );
    assert!(
        jammi_max.is_finite() && jammi_max <= bound_max,
        "{what}: max|jammi-truth|={jammi_max:.6e} exceeds {slack}*max|torchFA-truth|={bound_max:.6e}"
    );
    assert!(
        jammi_mean.is_finite() && jammi_mean <= bound_mean,
        "{what}: mean|jammi-truth|={jammi_mean:.6e} exceeds {slack}*mean|torchFA-truth|={bound_mean:.6e}"
    );
}

/// Runs jammi's fp16 forward+backward on one leg, returning
/// `(o, lse, dq, dk, dv)` as f64 CPU tensors — shared by the GREEN test and
/// the RED injection controls below. Identical shape to the bf16 file's own
/// `run_jammi`, calling the `_f16` FFI twins instead.
fn run_jammi(
    dev: &CudaDevice,
    leg: &Leg,
    scale: f32,
    window: Option<u32>,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let q = load_f16_exact(leg.name, "q");
    let k = load_f16_exact(leg.name, "k");
    let v = load_f16_exact(leg.name, "v");
    let grad_out = load_f16_exact(leg.name, "grad_out");

    let qkv = pack_qkv(&q, &k, &v);
    let (total_q, _, num_h, _d) = qkv.dims4().unwrap();
    assert_eq!(num_h, NUM_HEADS);
    let cu = CuSeqlens::from_lengths(leg.lengths, dev).unwrap();
    let cfg = VarlenConfig {
        softmax_scale: scale,
        window,
        deterministic: true,
    };

    let host_qkv: Vec<f16> = qkv.flatten_all().unwrap().to_vec1::<f16>().unwrap();
    let qkv_dev = dev.clone_htod(&host_qkv).unwrap();

    let (o, lse) =
        jammi_kernels::flash::flash_varlen_fwd_f16(dev, &qkv_dev, &cu, NUM_HEADS, &cfg).unwrap();
    let o_host: Vec<f16> = dev.clone_dtoh(&o).unwrap();
    let o_tensor = Tensor::from_vec(o_host, (total_q, NUM_HEADS, HEAD_DIM), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap();
    let lse_host: Vec<f32> = dev.clone_dtoh(&lse).unwrap();
    let lse_tensor = Tensor::from_vec(lse_host, (NUM_HEADS, total_q), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap();
    assert_finite(&o_tensor, &format!("{}: o", leg.name));
    assert_finite(&lse_tensor, &format!("{}: lse", leg.name));

    let do_host: Vec<f16> = grad_out.flatten_all().unwrap().to_vec1::<f16>().unwrap();
    let do_dev = dev.clone_htod(&do_host).unwrap();
    let d_qkv = jammi_kernels::flash::flash_varlen_bwd_f16(
        dev, &qkv_dev, &cu, NUM_HEADS, &o, &lse, &do_dev, &cfg,
    )
    .unwrap();
    let d_qkv_host: Vec<f16> = dev.clone_dtoh(&d_qkv).unwrap();
    let d_qkv_tensor =
        Tensor::from_vec(d_qkv_host, (total_q, 3, NUM_HEADS, HEAD_DIM), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
    let dq_t = d_qkv_tensor.narrow(1, 0, 1).unwrap().squeeze(1).unwrap();
    let dk_t = d_qkv_tensor.narrow(1, 1, 1).unwrap().squeeze(1).unwrap();
    let dv_t = d_qkv_tensor.narrow(1, 2, 1).unwrap().squeeze(1).unwrap();
    assert_finite(&dq_t, &format!("{}: dq", leg.name));
    assert_finite(&dk_t, &format!("{}: dk", leg.name));
    assert_finite(&dv_t, &format!("{}: dv", leg.name));

    (o_tensor, lse_tensor, dq_t, dk_t, dv_t)
}

/// Fixture-presence gate: the fp16 fixtures are generated on the pod
/// (`generate_fixtures.py --dtype float16`, requires torch+CUDA) and are
/// NOT expected to exist in every checkout the moment this file lands —
/// unlike the bf16 fixtures (tracked, always present). Skips (does not
/// fail) when absent, UNLESS `JAMMI_REQUIRE_CUDA` is set, matching this
/// file's own `cuda_device`'s skip-vs-fail convention: a required-CUDA run
/// that cannot find its own fixtures is a real gap, not a benign skip.
fn f16_fixtures_present_or_skip(test_name: &str) -> bool {
    let probe = fixtures_dir().join("f16_b1_s512_q.npy");
    if probe.exists() {
        return true;
    }
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!(
            "{test_name}: JAMMI_REQUIRE_CUDA is set but the fp16 fixtures are missing at {} — \
             run generate_fixtures.py --dtype float16 first",
            probe.display()
        );
    }
    eprintln!(
        "{test_name}: skipping — fp16 fixtures not found at {} (run generate_fixtures.py \
         --dtype float16 to produce them)",
        probe.display()
    );
    false
}

#[test]
fn o_lse_dq_dk_dv_match_truth_within_the_torch_relative_bound_f16() {
    let Some(dev) = cuda_device() else { return };
    if !f16_fixtures_present_or_skip(
        "o_lse_dq_dk_dv_match_truth_within_the_torch_relative_bound_f16",
    ) {
        return;
    }

    for leg in LEGS {
        let (o, lse, dq, dk, dv) = run_jammi(&dev, leg, SOFTMAX_SCALE, leg.window);

        let ref_o = load_f32(leg.name, "o").to_dtype(DType::F64).unwrap();
        let ref_lse = load_f32(leg.name, "lse").to_dtype(DType::F64).unwrap();
        let ref_dq = load_f32(leg.name, "dq").to_dtype(DType::F64).unwrap();
        let ref_dk = load_f32(leg.name, "dk").to_dtype(DType::F64).unwrap();
        let ref_dv = load_f32(leg.name, "dv").to_dtype(DType::F64).unwrap();

        let truth_o = load_f32(leg.name, "truth_o").to_dtype(DType::F64).unwrap();
        let truth_lse = load_f32(leg.name, "truth_lse")
            .to_dtype(DType::F64)
            .unwrap();
        let truth_dq = load_f32(leg.name, "truth_dq").to_dtype(DType::F64).unwrap();
        let truth_dk = load_f32(leg.name, "truth_dk").to_dtype(DType::F64).unwrap();
        let truth_dv = load_f32(leg.name, "truth_dv").to_dtype(DType::F64).unwrap();

        assert_truth_relative(
            &format!("{}: o", leg.name),
            &o,
            &ref_o,
            &truth_o,
            TRUTH_RELATIVE_SLACK,
        );
        assert_truth_relative(
            &format!("{}: lse", leg.name),
            &lse,
            &ref_lse,
            &truth_lse,
            TRUTH_RELATIVE_SLACK,
        );
        assert_truth_relative(
            &format!("{}: dq", leg.name),
            &dq,
            &ref_dq,
            &truth_dq,
            TRUTH_RELATIVE_SLACK,
        );
        assert_truth_relative(
            &format!("{}: dk", leg.name),
            &dk,
            &ref_dk,
            &truth_dk,
            TRUTH_RELATIVE_SLACK,
        );
        assert_truth_relative(
            &format!("{}: dv", leg.name),
            &dv,
            &ref_dv,
            &truth_dv,
            TRUTH_RELATIVE_SLACK,
        );
    }
}

/// RED control 1 (fp16 twin of `softmax_scale_times_1_05_injection_reds_the_parity_oracle`):
/// `softmax_scale * 1.05` on the jammi side only must RED the truth-relative
/// bound on `o`.
#[test]
fn softmax_scale_times_1_05_injection_reds_the_parity_oracle_f16() {
    let Some(dev) = cuda_device() else { return };
    if !f16_fixtures_present_or_skip(
        "softmax_scale_times_1_05_injection_reds_the_parity_oracle_f16",
    ) {
        return;
    }
    let leg = &LEGS[0]; // b1_s512
    assert_eq!(leg.name, "b1_s512");

    let (o, ..) = run_jammi(&dev, leg, SOFTMAX_SCALE * 1.05, leg.window);
    let ref_o = load_f32(leg.name, "o").to_dtype(DType::F64).unwrap();
    let truth_o = load_f32(leg.name, "truth_o").to_dtype(DType::F64).unwrap();

    let (jammi_max, _) = max_mean_abs_diff(&o, &truth_o);
    let (ref_max, _) = max_mean_abs_diff(&ref_o, &truth_o);
    let bound = TRUTH_RELATIVE_SLACK * ref_max;
    assert!(
        jammi_max > bound,
        "softmax_scale*1.05 injection did NOT red the oracle: |jammi-truth|max={jammi_max:.6e} \
         <= bound {bound:.6e} — the comparison is not discriminating"
    );
}

/// RED control 2 (fp16 twin of `window_off_by_one_injection_reds_the_parity_oracle`):
/// window radius `w +/- 1` must RED the truth-relative bound on `o`.
#[test]
fn window_off_by_one_injection_reds_the_parity_oracle_f16() {
    let Some(dev) = cuda_device() else { return };
    if !f16_fixtures_present_or_skip("window_off_by_one_injection_reds_the_parity_oracle_f16") {
        return;
    }
    let leg = LEGS.iter().find(|l| l.name == "b1_s512_win64").unwrap();
    let correct_window = leg.window.unwrap();

    let ref_o = load_f32(leg.name, "o").to_dtype(DType::F64).unwrap();
    let truth_o = load_f32(leg.name, "truth_o").to_dtype(DType::F64).unwrap();
    let (ref_max, _) = max_mean_abs_diff(&ref_o, &truth_o);
    let bound = TRUTH_RELATIVE_SLACK * ref_max;

    for (delta_name, injected_window) in [("w+1", correct_window + 1), ("w-1", correct_window - 1)]
    {
        let (o, ..) = run_jammi(&dev, leg, SOFTMAX_SCALE, Some(injected_window));
        let (jammi_max, _) = max_mean_abs_diff(&o, &truth_o);
        assert!(
            jammi_max > bound,
            "window {delta_name} injection (used {injected_window}, fixture is {correct_window}) \
             did NOT red the oracle: |jammi-truth|max={jammi_max:.6e} <= bound {bound:.6e}"
        );
    }
}

/// Boundary/finiteness leg (contract item 2's own explicit ask, beyond the
/// bf16 file's own coverage): the single-token degenerate case (`lengths =
/// [1]`, no window) at fp16 — every softmax row has exactly one key, so
/// `exp(0)/1.0 = 1.0` exactly and the output must equal `v` bit-for-bit
/// (mod fp16 rounding of the copy itself), never NaN/Inf from a
/// divide-by-zero or an empty reduction. Synthetic (family L), no fixture
/// needed — this is a domain-edge oracle, not a magnitude-parity one.
#[test]
fn single_token_sequence_is_finite_and_exactly_reproduces_v_f16() {
    let Some(dev) = cuda_device() else { return };
    let lengths = [1usize];
    let cu = CuSeqlens::from_lengths(&lengths, &dev).unwrap();
    let cfg = VarlenConfig {
        softmax_scale: SOFTMAX_SCALE,
        window: None,
        deterministic: true,
    };
    let host_qkv: Vec<f16> = (0..3 * NUM_HEADS * HEAD_DIM)
        .map(|i| f16::from_f32(((i % 5) as f32 - 2.0) * 0.25))
        .collect();
    let qkv = dev.clone_htod(&host_qkv).unwrap();
    let (o, lse) =
        jammi_kernels::flash::flash_varlen_fwd_f16(&dev, &qkv, &cu, NUM_HEADS, &cfg).unwrap();
    let o_host: Vec<f16> = dev.clone_dtoh(&o).unwrap();
    let lse_host: Vec<f32> = dev.clone_dtoh(&lse).unwrap();
    assert!(
        o_host.iter().all(|x| x.to_f32().is_finite()),
        "single-token o must be finite"
    );
    assert!(
        lse_host.iter().all(|x| x.is_finite()),
        "single-token lse must be finite"
    );
    // v is qkv's THIRD slab ([total_q=1, 3, H, D], slot index 2).
    let v_expected: Vec<f16> =
        host_qkv[2 * NUM_HEADS * HEAD_DIM..3 * NUM_HEADS * HEAD_DIM].to_vec();
    assert_eq!(
        o_host, v_expected,
        "a single-key softmax row's output must equal v bit-for-bit — the only key gets \
         probability exactly 1.0"
    );
}
