//! O0(a) (P6 Stage B contract §4): parity vs torch's OWN vendored FA2
//! (`torch.ops.aten._flash_attention_forward`/`_backward`) on the B0
//! fixtures (`tests/fixtures/flash_reference/`, tracked, generator +
//! sidecar in the same directory — see that sidecar's
//! `version_mismatch_note` for the reference's exact identity: torch pins
//! `Dao-AILab/flash-attention` at commit `6c4f74fb338e` = release **2.8.4**
//! (`flash_attn/__init__.py:6`), the SAME `csrc/flash_attn/src` kernel
//! subtree jammi vendors at `v2.8.3.post1` (one patch release apart), but
//! torch's build defines `UNFUSE_FMA` and skips `--use_fast_math` while
//! jammi's does not — a cross-build (FMA-fusion / fast-math) reference,
//! not a same-flags one.
//!
//! Runs `crate::flash::flash_varlen_fwd`/`flash_varlen_bwd` DIRECTLY (the
//! FFI-boundary layer), not `ops::flash_attention_varlen` — this is where
//! `lse` is a first-class return value (the `ops` wrapper stashes it in a
//! private `Saved` slot, see that module's own doc for why); the `ops`
//! layer's own oracle suite (`tests/flash_op_oracles.rs`) covers the
//! `Tensor`/autograd wiring on top of this, not the numerics again.
//!
//! # Fix round (`10b1f3b` audit, BLOCKING finding 1)
//!
//! The prior version's bound was `k * bf16_ulp * depth * max(|ref|, 1)` —
//! an ABSOLUTE floor (`docs/maintainer/cuda-kernel-guide.md` §3.8
//! explicitly forbids this shape: "a `k · ulp(max)` floor charges every
//! element the allowance of the largest and hides exactly the divergence
//! you are hunting"). The auditor showed it was 50-100x wider than the
//! REAL divergence (measured by hand with numpy: on `b1_s512_win64`,
//! `|f64_truth - ref_o|max = 0.0036`, `|f64_truth - ref_lse|max = 7.7e-7` — no-producer: that audit round's one-off hand probe, uncommitted),
//! and that `lse` (an `f32` TENSOR — never rounded to bf16 anywhere in this
//! op's pipeline) was being bounded by a bf16 ULP fraction, a category
//! error the guide's family-D "pin the mathematical object" principle
//! forbids. A deliberate `softmax_scale * 1.05` injection PASSED every
//! tensor under the old bound.
//!
//! **This version's bound is TRUTH-RELATIVE, computed live, per tensor:**
//! for each of `o`/`lse`/`dq`/`dk`/`dv`, `generate_fixtures.py`'s
//! `truth_{o,lse,dq,dk,dv}` (a from-scratch f64 eager attention — no
//! FlashAttention kernel of any generation — on the SAME bf16-exact
//! inputs) is the higher-precision anchor
//! (`docs/maintainer/cuda-kernel-guide.md` §3.3). This test asserts, on
//! BOTH the max and the mean absolute difference:
//!
//! ```text
//! max|jammi - truth|  <=  1.5 * max|torchFA - truth|
//! mean|jammi - truth| <=  1.5 * mean|torchFA - truth|
//! ```
//!
//! `torchFA`'s own distance to truth is computed HERE, live, from the SAME
//! loaded fixture tensors (never trusted from the sidecar's precomputed
//! JSON number) — a measured-and-asserted control (family F), not an
//! assumed one. No absolute floor is added on top; `lse` is compared in
//! its own native `f32` units, never scaled by a bf16 ULP fraction. The
//! sidecar's own same-build backward self-diff (`self_noise_max_abs_diff`)
//! is REPORTED (printed) for `dq`/`dk`/`dv`, never used as a bound (the
//! prior version's doc claimed it was "OR"ed into the tolerance; the
//! auditor found that derivation "not computed" — dropped rather than kept
//! as an uncomputed claim).
//!
//! # RED controls (must fail this oracle — proves it discriminates)
//!
//! 1. `softmax_scale * 1.05` on the jammi side only (not `* 2` — the prior
//!    version's `* 2` control passed even under the OLD vacuous bound in
//!    some regime; `* 1.05` is a tighter, more realistic scale bug and
//!    still must RED under the truth-relative bound).
//! 2. Window radius `w +/- 1` (the fixture is generated at `w`; the jammi
//!    call site is fed `w+1` and `w-1` in turn) — an off-by-one window slips
//!    one extra/one fewer key into every row's softmax.
//! 3. "a wrong-RoPE-less input" (per the fix-round dispatch): **N/A for
//!    this op** — `ops::flash_attention::FlashVarlenAttention`'s own module
//!    doc, "Domain" section, states RoPE is NOT applied inside this op (the
//!    caller rotates Q/K before packing `qkv`); there is no RoPE-application
//!    code path here to inject a wrong instance of. Recorded rather than
//!    silently dropped.

#![cfg(feature = "flash-attn")]

use std::path::{Path, PathBuf};

use candle_core::{CudaDevice, DType, Device, Tensor};
use half::bf16;
use jammi_kernels::flash::{CuSeqlens, VarlenConfig};

fn cuda_device() -> Option<CudaDevice> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d.as_cuda_device().unwrap().clone()),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "flash_torch_parity: JAMMI_REQUIRE_CUDA is set but no CUDA device could be \
                     acquired — a silent skip here is not acceptable: {e}"
                );
            }
            eprintln!("flash_torch_parity: skipping — no CUDA device available ({e})");
            None
        }
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flash_reference")
}

/// The bf16-exact inputs (`q`/`k`/`v`/`grad_out`) are stored as raw `int16`
/// `.npy` — a bit-for-bit reinterpretation of the bf16 payload
/// (`generate_fixtures.py`'s `npi16_bf16_bits`), because candle-core
/// 0.11.0's `npy.rs` `Header::parse` has no numpy `descr` mapped to
/// `DType::BF16` at all. Reads the `i16` tensor and bit-reinterprets
/// (`i16 as u16` preserves the bit pattern via 2's-complement identity;
/// `bf16::from_bits` is the exact inverse of `.view(torch.int16)` on the
/// python side) — never a re-round.
fn load_bf16_exact(leg: &str, name: &str) -> Tensor {
    let path = fixtures_dir().join(format!("{leg}_{name}.npy"));
    let raw = Tensor::read_npy(&path)
        .unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()));
    assert_eq!(
        raw.dtype(),
        DType::I16,
        "{}: expected int16 bit-pattern storage, got {:?} — generate_fixtures.py's storage \
         convention changed without this loader following it",
        path.display(),
        raw.dtype()
    );
    let dims = raw.dims().to_vec();
    let bits: Vec<i16> = raw.flatten_all().unwrap().to_vec1::<i16>().unwrap();
    let vals: Vec<bf16> = bits.iter().map(|&b| bf16::from_bits(b as u16)).collect();
    Tensor::from_vec(vals, dims, &Device::Cpu).unwrap()
}

fn load_f32(leg: &str, name: &str) -> Tensor {
    let path = fixtures_dir().join(format!("{leg}_{name}.npy"));
    Tensor::read_npy(&path).unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()))
}

/// One B0 leg: `(name, lengths, window_radius)`. Mirrors
/// `generate_fixtures.py`'s `LEGS` list exactly (H=16, D=64,
/// softmax_scale=1/8 for every leg — see that script and the sidecar).
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
/// The truth-relative slack factor — see the module doc's "Fix round"
/// section. `1.5`, not `1.0`: jammi's own bf16 kernel and torch's
/// cross-build FA2 reference are TWO DIFFERENT bf16 kernels (different
/// FMA-fusion/fast-math flags, see the sidecar's `version_mismatch_note`),
/// so exact equality of their distances to the f64 truth is not expected —
/// only that jammi is not MEANINGFULLY further from truth than torch's own
/// reference is.
const TRUTH_RELATIVE_SLACK: f64 = 1.5;

/// Packs `q`/`k`/`v` ([total_q, H, D] each, bf16) into `[total_q, 3, H, D]`
/// bf16 — the layout `crate::flash`'s FFI requires (`Tensor::stack` along a
/// NEW axis 1; matches `flash_attn_varlen_qkvpacked_func`'s own upstream
/// layout `crate::flash`'s module doc cites). Inputs are ALREADY bf16
/// (loaded via `load_bf16_exact`), so this is a pure layout op — no
/// dtype cast, no re-rounding.
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
/// never a negated form — a NaN must FAIL, not read as a fit).
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
         max={ref_max:.6e} mean={ref_mean:.6e} | bounds max<={bound_max:.6e} mean<={bound_mean:.6e}"
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

/// Runs jammi's forward+backward on one leg, returning
/// `(o, lse, dq, dk, dv)` as f64 CPU tensors — shared by the GREEN test and
/// the RED injection controls below.
fn run_jammi(
    dev: &CudaDevice,
    leg: &Leg,
    scale: f32,
    window: Option<u32>,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let q = load_bf16_exact(leg.name, "q");
    let k = load_bf16_exact(leg.name, "k");
    let v = load_bf16_exact(leg.name, "v");
    let grad_out = load_bf16_exact(leg.name, "grad_out");

    let qkv = pack_qkv(&q, &k, &v);
    let (total_q, _, num_h, _d) = qkv.dims4().unwrap();
    assert_eq!(num_h, NUM_HEADS);
    let cu = CuSeqlens::from_lengths(leg.lengths, dev).unwrap();
    let cfg = VarlenConfig {
        softmax_scale: scale,
        window,
        deterministic: true,
    };

    let host_qkv: Vec<bf16> = qkv.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
    let qkv_dev = dev.clone_htod(&host_qkv).unwrap();

    let (o, lse) =
        jammi_kernels::flash::flash_varlen_fwd(dev, &qkv_dev, &cu, NUM_HEADS, &cfg).unwrap();
    let o_host: Vec<bf16> = dev.clone_dtoh(&o).unwrap();
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

    let do_host: Vec<bf16> = grad_out.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
    let do_dev = dev.clone_htod(&do_host).unwrap();
    let d_qkv = jammi_kernels::flash::flash_varlen_bwd(
        dev, &qkv_dev, &cu, NUM_HEADS, &o, &lse, &do_dev, &cfg,
    )
    .unwrap();
    let d_qkv_host: Vec<bf16> = dev.clone_dtoh(&d_qkv).unwrap();
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

#[test]
fn o_lse_dq_dk_dv_match_truth_within_the_torch_relative_bound() {
    let Some(dev) = cuda_device() else { return };

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

        // The same-build backward self-diff (two torch FA2 runs, identical
        // inputs) is REPORTED in this leg's `sidecar.json` entry
        // (`self_noise_max_abs_diff`), never used as a bound here — see the
        // module doc's "Fix round" section for why the prior version's
        // uncomputed "OR the self-diff" derivation was dropped rather than
        // kept.
    }
}

/// RED control 1: `softmax_scale * 1.05` on the jammi side only (the
/// fixture/truth are both at the CORRECT scale) must RED the truth-relative
/// bound on `o` — proves the bound actually discriminates rather than
/// being vacuously wide. `b1_s512` is enough to prove discrimination.
#[test]
fn softmax_scale_times_1_05_injection_reds_the_parity_oracle() {
    let Some(dev) = cuda_device() else { return };
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

/// RED control 2: window radius `w +/- 1` on the jammi side only (the
/// fixture/truth are both generated at the correct `w=64`) must RED the
/// truth-relative bound on `o` — an off-by-one window slips one extra/one
/// fewer key into every row's softmax. Uses `b1_s512_win64`.
#[test]
fn window_off_by_one_injection_reds_the_parity_oracle() {
    let Some(dev) = cuda_device() else { return };
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
