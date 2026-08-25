//! FlashAttention-2 FFI smoke — the pod-run landing proof for the vendored
//! kernels + `flash_api_jammi.cu` + `jammi_kernels::flash`.
//!
//! Builds only with `--features flash-attn` (`required-features` in
//! `Cargo.toml`). Device acquisition follows `cuda_parity.rs`: a failure to
//! open CUDA device 0 is a SKIP unless `JAMMI_REQUIRE_CUDA` is set, in which
//! case it PANICS — the pod session that is this file's landing proof sets
//! it, so a silent skip can never read as green there.
//!
//! Two fixtures, H = 2, D = 64, bf16 inputs from a deterministic `sin`
//! fixture, `softmax_scale = 1/8`:
//!
//! - [`SMALL_LENS`] `= [5, 9, 7]` (`total_q = 21`, `max_seqlen = 9`, never
//!   leaves `kBlockM = kBlockN = 128` tile 0) — used by every test AND is
//!   the only fixture the central-finite-difference backward oracle runs
//!   at (intractable at multi-tile scale, see its doc).
//! - [`LARGE_LENS`] `= [5, 137, 260, 128, 129]` (`total_q = 659`,
//!   `max_seqlen = 260`, crosses several `kBlockN = 128` tiles, two lengths
//!   sit at/one-past the 128 boundary) — every OTHER oracle in this file
//!   (forward-vs-reference, bit-identity, backward-vs-analytic-reference,
//!   guard regions) also runs at this fixture, in a `_multi_tile` test
//!   (F3: the small fixture alone never exercises the multi-`n_block` loop,
//!   the backward's `Is_first`/`Is_last` split, or the sliding-window
//!   `n_block_min` skip).
//!
//! The CPU reference is the eager composition in f64 over the bf16-exact
//! inputs: per (sequence, head, row) `softmax(scale · q·kᵀ + band) · v`
//! with `band` the symmetric radius `|r − c| <= w`, and `lse = m + ln Σ e`.
//!
//! Every tolerance below states its derivation, its mutation (what change
//! it is proven to detect), and the ratio bound / max|signal| is printed
//! from the run (the pod output is pasted into the landing commit).
#![cfg(feature = "flash-attn")]

use candle_core::cuda_backend::cudarc::driver::DeviceRepr;
use candle_core::{CudaDevice, Device};
use half::bf16;
use jammi_kernels::flash::{
    self, dq_accum_splits, flash_varlen_bwd, flash_varlen_bwd_into, flash_varlen_fwd,
    flash_varlen_fwd_into, raw, BwdBuffers, BwdScratch, CuSeqlens, FlashError, FlashStatus,
    VarlenConfig, VarlenGeometry, HEAD_DIM,
};

fn cuda_device() -> Option<CudaDevice> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d.as_cuda_device().unwrap().clone()),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "flash_smoke: JAMMI_REQUIRE_CUDA is set but no CUDA device could be \
                     acquired — this is the landing proof, a silent skip here is not \
                     acceptable: {e}"
                );
            }
            eprintln!("flash_smoke: skipping — no CUDA device available ({e})");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture + CPU reference
// ---------------------------------------------------------------------------

/// The ORIGINAL fixture: three sequences, none longer than `kBlockM =
/// kBlockN = 128` (`flash_fwd_launch_template.h:188`,
/// `flash_bwd_launch_template.h:178-180`) — `max_seqlen = 9` never leaves
/// tile 0. Kept for the central-finite-difference backward oracle, which is
/// O(total · H · max_seqlen · D) PER ELEMENT differentiated (`2 · qkv.len()`
/// loss evaluations) — intractable at multi-tile scale (a `[5, 137, 260,
/// 128, 129]` fixture would be a ~1e13-operation test). Every OTHER oracle
/// in this file also runs at [`LARGE_LENS`] (the `_multi_tile` variant of
/// each test), which crosses several tiles instead.
const SMALL_LENS: [usize; 3] = [5, 9, 7];
/// Multi-tile, ragged fixture (F3): five sequences, none a multiple of the
/// other, `max_seqlen = 260` crosses TWO `kBlockN = 128` tiles (the
/// backward's `n_block` loop, `Is_first`/`Is_last` split at
/// `flash_bwd_kernel.h:811-818`, and the forward's multi-block loop all run
/// more than once), and two lengths (128, 129) sit AT and ONE PAST the tile
/// boundary (`round_multiple(seqlen, 128)` buckets 128 and 256 both get
/// exercised). Every oracle below runs `Some(2)` as one of its windows —
/// `2 <= kBlockN`, narrower than one tile, so the sliding-window
/// `n_block_min` skip (`flash_fwd_kernel.h:83`) is exercised at every
/// `n_block` past the first, not just the last.
const LARGE_LENS: [usize; 5] = [5, 137, 260, 128, 129];
const H: usize = 2;
const D: usize = HEAD_DIM;
const SCALE: f32 = 0.125;

struct Fixture {
    lens: Vec<usize>,
    total: usize,
    /// bf16-exact values, `[total, 3, H, D]`.
    qkv: Vec<f64>,
    qkv_bf16: Vec<bf16>,
    /// bf16-exact values, `[total, H, D]`, non-uniform.
    d_o: Vec<f64>,
    d_o_bf16: Vec<bf16>,
}

fn geom(cu: &CuSeqlens) -> VarlenGeometry {
    cu.geometry(H).unwrap()
}

fn fixture(lens: &[usize]) -> Fixture {
    let total: usize = lens.iter().sum();
    let n_qkv = total * 3 * H * D;
    // Values in (-1, 1), rounded to bf16 once; the f64 copy holds the
    // rounded values exactly so reference and kernel see the SAME inputs.
    let qkv_bf16: Vec<bf16> = (0..n_qkv)
        .map(|i| bf16::from_f32(((i as f32) * 0.37 + 0.5).sin()))
        .collect();
    let qkv: Vec<f64> = qkv_bf16.iter().map(|v| v.to_f64()).collect();
    let n_o = total * H * D;
    // Non-uniform d_o: a product of two incommensurate frequencies so no
    // two rows/columns share a pattern (a uniform d_o would make the
    // backward's dP − δ term vanish per row, hiding a wrong dS).
    let d_o_bf16: Vec<bf16> = (0..n_o)
        .map(|i| bf16::from_f32(0.7 * ((i as f32) * 0.61).cos() * ((i as f32) * 0.053 + 1.0).sin()))
        .collect();
    let d_o: Vec<f64> = d_o_bf16.iter().map(|v| v.to_f64()).collect();
    Fixture {
        lens: lens.to_vec(),
        total,
        qkv,
        qkv_bf16,
        d_o,
        d_o_bf16,
    }
}

#[inline]
fn iqkv(r: usize, slot: usize, h: usize, d: usize) -> usize {
    ((r * 3 + slot) * H + h) * D + d
}
#[inline]
fn io(r: usize, h: usize, d: usize) -> usize {
    (r * H + h) * D + d
}

/// Eager reference in f64: returns `(o [total, H, D], lse [H, total])`.
fn attn_ref(
    qkv: &[f64],
    lens: &[usize],
    scale: f64,
    window: Option<usize>,
) -> (Vec<f64>, Vec<f64>) {
    let total: usize = lens.iter().sum();
    let mut o = vec![0.0; total * H * D];
    let mut lse = vec![0.0; H * total];
    let mut start = 0;
    for &n in lens {
        for h in 0..H {
            for r in 0..n {
                let rq = start + r;
                let mut scores = Vec::with_capacity(n);
                for c in 0..n {
                    let in_band = match window {
                        None => true,
                        Some(w) => (r as i64 - c as i64).abs() <= w as i64,
                    };
                    if !in_band {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }
                    let rk = start + c;
                    let mut s = 0.0;
                    for d in 0..D {
                        s += qkv[iqkv(rq, 0, h, d)] * qkv[iqkv(rk, 1, h, d)];
                    }
                    scores.push(s * scale);
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut l = 0.0;
                let mut acc = vec![0.0; D];
                for (c, &s) in scores.iter().enumerate() {
                    if s == f64::NEG_INFINITY {
                        continue;
                    }
                    let p = (s - m).exp();
                    l += p;
                    let rv = start + c;
                    for (d, a) in acc.iter_mut().enumerate() {
                        *a += p * qkv[iqkv(rv, 2, h, d)];
                    }
                }
                for (d, a) in acc.iter().enumerate() {
                    o[io(rq, h, d)] = a / l;
                }
                lse[h * total + rq] = m + l.ln();
            }
        }
        start += n;
    }
    (o, lse)
}

/// `L(qkv) = Σ d_o ⊙ o_ref(qkv)`, the scalar the finite differences
/// differentiate.
fn loss_ref(qkv: &[f64], d_o: &[f64], lens: &[usize], scale: f64, window: Option<usize>) -> f64 {
    let (o, _) = attn_ref(qkv, lens, scale, window);
    o.iter().zip(d_o).map(|(a, b)| a * b).sum()
}

/// Central finite differences of `loss_ref` w.r.t. every qkv entry, in
/// f64 with `eps = 1e-3`: truncation error `O(eps² · L''')` ≈ 1e-6 ·
/// (third derivative, O(1) here), round-off `O(1e-16 / eps)` ≈ 1e-13 —
/// both far below the bf16-scale tolerances the kernel is held to.
fn grad_fd(
    qkv: &[f64],
    d_o: &[f64],
    lens: &[usize],
    scale: f64,
    window: Option<usize>,
) -> Vec<f64> {
    let eps = 1e-3;
    let mut g = vec![0.0; qkv.len()];
    let mut x = qkv.to_vec();
    for i in 0..qkv.len() {
        let x0 = x[i];
        x[i] = x0 + eps;
        let lp = loss_ref(&x, d_o, lens, scale, window);
        x[i] = x0 - eps;
        let lm = loss_ref(&x, d_o, lens, scale, window);
        x[i] = x0;
        g[i] = (lp - lm) / (2.0 * eps);
    }
    g
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn dtoh<T: DeviceRepr + Clone>(
    dev: &CudaDevice,
    s: &candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
) -> Vec<T> {
    dev.clone_dtoh(s).unwrap()
}

fn bf16_to_f64(v: &[bf16]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

fn cfg(window: Option<u32>, deterministic: bool) -> VarlenConfig {
    VarlenConfig {
        softmax_scale: SCALE,
        window,
        deterministic,
    }
}

struct DeviceFixture {
    qkv: candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    /// Built via [`CuSeqlens::from_lengths`] — the only sanctioned way to
    /// reach a launch (F1). The `raw`-escape-hatch tests (`c_wrapper_refusal_cells`
    /// etc.) still reach the device array through this `CuSeqlens` (via
    /// [`dptr_cu`]), not a second independent buffer — there is no other
    /// legitimate way to get one.
    cu: CuSeqlens,
    d_o: candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
}

fn upload(dev: &CudaDevice, fx: &Fixture) -> DeviceFixture {
    DeviceFixture {
        qkv: dev.clone_htod(&fx.qkv_bf16).unwrap(),
        cu: CuSeqlens::from_lengths(&fx.lens, dev).unwrap(),
        d_o: dev.clone_htod(&fx.d_o_bf16).unwrap(),
    }
}

// ---------------------------------------------------------------------------
// (a) forward vs the CPU reference, window None and Some(2)
// ---------------------------------------------------------------------------

/// Forward output bound, ABSOLUTE, as a fraction of `max|v|` (= 1 for the
/// fixture, every entry of v is in (−1, 1)):
///
/// `o = Σ_c p_c · v_c` is a convex combination of v rows. FA2 rounds the
/// probabilities to bf16 before the P·V tensor-core GEMM (`flash_fwd_kernel.h`,
/// `convert_type<Element>(acc_s)`): bf16 has 7 explicit mantissa bits, so
/// round-to-nearest carries relative error `<= 2^-8` (half an ulp at the
/// bottom of a binade), and `|Δo| <= 2^-8 · Σ p_c |v_c| <= 2^-8 · max|v|`.
/// The bf16 store of `o` adds `<= 2^-8 · |o| <= 2^-8 · max|v|`. The
/// fast-math `exp2` (`ex2.approx`, `<= 2 ulp` f32) and the f32 row-sum
/// contribute `< 2^-16` relative together — FA2's online softmax never
/// reduces more than `kBlockN = 128` terms in one register-level sum (it
/// processes keys in blocks of 128, rescaling a running max/sum between
/// blocks, `flash_fwd_kernel.h`'s main loop), so this bound does not grow
/// with `max_seqlen`: the [`LARGE_LENS`] fixture (`max_seqlen = 260`, 3
/// blocks) adds only `O(log₂ 3)` block-merge rescalings on top of the same
/// per-block sum, negligible next to the `2^-7` bf16 term. The bound is that
/// SUP itself, `(2^-7 + 2^-16) · max|v|` — no headroom factor, because the
/// derivation is an upper bound over every rounding alignment (no
/// correct kernel can exceed it) and a looser bound stops discriminating
/// the one-key window mutation below (measured on the A100: 3.1e-3 = 0.39
/// of the sup; a `2^-6` bound let the `Some(3)` mutation, 1.46e-2, pass).
///
/// Mutations it is proven to detect (asserted below): the reference with
/// the WRONG window (`None` vs `Some(2)`, and the adjacent radii 1 and 3),
/// and the reference with a 1.5× scale — each exceeds the bound.
const O_TOL_FRAC: f64 = 1.0 / 128.0 + 1.0 / 65536.0;

/// LSE bound, ABSOLUTE: FA2 forms `q·k` on the bf16 tensor cores with f32
/// accumulation over 64 exact bf16×bf16 products (`<= 64 · 2^-24 · |s|`,
/// `|s| <= 64 · 0.125 = 8` ⇒ `< 4e-5`), scales by `scale · log2(e)`, takes
/// `exp2` (`<= 2 ulp`), sums in f32 and applies `__logf` (`<= 2 ulp`):
/// total `< 1e-4` absolute. Bound `1e-3`, 10× headroom. Mutation: the
/// 1.5× scale reference shifts lse by O(0.1–1) and fails.
const LSE_TOL: f64 = 1e-3;

fn run_fwd(dev: &CudaDevice, dfx: &DeviceFixture, window: Option<u32>) -> (Vec<bf16>, Vec<f32>) {
    let (o, lse) = flash_varlen_fwd(dev, &dfx.qkv, &dfx.cu, H, &cfg(window, true)).unwrap();
    dev.cuda_stream().synchronize().unwrap();
    (dtoh(dev, &o), dtoh(dev, &lse))
}

fn fwd_matches_cpu_reference_window_none_and_some2_body(dev: &CudaDevice, lens: &[usize]) {
    let fx = fixture(lens);
    let dfx = upload(dev, &fx);
    let vmax = (0..fx.total)
        .flat_map(|r| (0..H).flat_map(move |h| (0..D).map(move |d| iqkv(r, 2, h, d))))
        .map(|i| fx.qkv[i].abs())
        .fold(0.0, f64::max);
    let o_tol = O_TOL_FRAC * vmax;

    for window in [None, Some(2u32)] {
        let (o_fa, lse_fa) = run_fwd(dev, &dfx, window);
        let o_fa = bf16_to_f64(&o_fa);
        let lse_fa: Vec<f64> = lse_fa.iter().map(|x| *x as f64).collect();
        let w = window.map(|w| w as usize);
        let (o_ref, lse_ref) = attn_ref(&fx.qkv, lens, SCALE as f64, w);

        let err_o = max_abs_diff(&o_fa, &o_ref);
        let err_lse = max_abs_diff(&lse_fa, &lse_ref);
        println!(
            "lens={lens:?} fwd window={window:?}: max|o_fa-o_ref|={err_o:.3e} (bound {o_tol:.3e}, max|o_ref|={:.3e}, bound/max|signal|={:.3e}); max|lse_fa-lse_ref|={err_lse:.3e} (bound {LSE_TOL:.1e}, max|lse_ref|={:.3e})",
            max_abs(&o_ref),
            o_tol / max_abs(&o_ref),
            max_abs(&lse_ref)
        );
        assert!(
            err_o <= o_tol,
            "lens={lens:?} window {window:?}: forward output off by {err_o:.3e} > {o_tol:.3e}"
        );
        assert!(
            err_lse <= LSE_TOL,
            "lens={lens:?} window {window:?}: lse off by {err_lse:.3e} > {LSE_TOL:.1e}"
        );
        assert!(
            o_fa.iter().all(|x| x.is_finite()) && lse_fa.iter().all(|x| x.is_finite()),
            "non-finite output"
        );

        // Discrimination: the SAME assertion fails against each mutated
        // reference — the wrong window (every adjacent radius and the
        // opposite of the tested setting) and the wrong scale.
        let wrong_windows: Vec<Option<usize>> = match w {
            None => vec![Some(2), Some(3)],
            Some(_) => vec![None, Some(1), Some(3)],
        };
        for ww in wrong_windows {
            let (o_wrong, _) = attn_ref(&fx.qkv, lens, SCALE as f64, ww);
            let e = max_abs_diff(&o_fa, &o_wrong);
            assert!(
                e > o_tol,
                "lens={lens:?} window {window:?}: the bound cannot distinguish the window from {ww:?} (diff {e:.3e} <= {o_tol:.3e})"
            );
        }
        let (o_wrong_scale, lse_wrong_scale) = attn_ref(&fx.qkv, lens, 1.5 * SCALE as f64, w);
        let e = max_abs_diff(&o_fa, &o_wrong_scale);
        assert!(e > o_tol, "scale mutation undetected in o: {e:.3e}");
        let e = max_abs_diff(&lse_fa, &lse_wrong_scale);
        assert!(e > LSE_TOL, "scale mutation undetected in lse: {e:.3e}");
    }
}

#[test]
fn fwd_matches_cpu_reference_window_none_and_some2() {
    let Some(dev) = cuda_device() else { return };
    fwd_matches_cpu_reference_window_none_and_some2_body(&dev, &SMALL_LENS);
}

#[test]
fn fwd_matches_cpu_reference_window_none_and_some2_multi_tile() {
    let Some(dev) = cuda_device() else { return };
    fwd_matches_cpu_reference_window_none_and_some2_body(&dev, &LARGE_LENS);
}

// ---------------------------------------------------------------------------
// (b) recompute soundness: two forwards are bit-identical
// ---------------------------------------------------------------------------

fn fwd_twice_is_bit_identical_in_o_and_lse_body(dev: &CudaDevice, lens: &[usize]) {
    let fx = fixture(lens);
    let dfx = upload(dev, &fx);
    for window in [None, Some(2u32)] {
        let (o1, lse1) = run_fwd(dev, &dfx, window);
        let (o2, lse2) = run_fwd(dev, &dfx, window);
        let o1: Vec<u16> = o1.iter().map(|x| x.to_bits()).collect();
        let o2: Vec<u16> = o2.iter().map(|x| x.to_bits()).collect();
        let lse1: Vec<u32> = lse1.iter().map(|x| x.to_bits()).collect();
        let lse2: Vec<u32> = lse2.iter().map(|x| x.to_bits()).collect();
        let g = geom(&dfx.cu);
        assert_eq!(o1.len(), g.o_len());
        assert_eq!(lse1.len(), g.lse_len());
        assert_eq!(
            o1, o2,
            "lens={lens:?} window {window:?}: o differs between two forwards"
        );
        assert_eq!(
            lse1, lse2,
            "lens={lens:?} window {window:?}: lse differs between two forwards"
        );
        // Non-vacuous: the outputs are not all one value (a kernel that
        // never wrote would also be "bit-identical").
        assert!(
            o1.iter().any(|b| *b != o1[0]),
            "o is constant — nothing was computed"
        );
        assert!(
            lse1.iter().any(|b| *b != lse1[0]),
            "lse is constant — nothing was computed"
        );
    }
}

#[test]
fn fwd_twice_is_bit_identical_in_o_and_lse() {
    let Some(dev) = cuda_device() else { return };
    fwd_twice_is_bit_identical_in_o_and_lse_body(&dev, &SMALL_LENS);
}

#[test]
fn fwd_twice_is_bit_identical_in_o_and_lse_multi_tile() {
    let Some(dev) = cuda_device() else { return };
    fwd_twice_is_bit_identical_in_o_and_lse_body(&dev, &LARGE_LENS);
}

// ---------------------------------------------------------------------------
// (c) backward vs central finite differences
// ---------------------------------------------------------------------------

/// Analytic backward reference (f64) and, per gradient element, the
/// ERROR MASS the kernel's bf16 rounding points can move it by, in units
/// of one bf16 half-ulp (`2^-8` relative — 7 explicit mantissa bits, so
/// round-to-nearest is within `2^-8` of the value at the bottom of a
/// binade; the same unit as [`O_TOL_FRAC`]'s derivation).
///
/// Rounding points in FA2's backward (`flash_bwd_kernel.h`): `P` is
/// converted to bf16 before `dV = Pᵀ·dO`; `δ_r = Σ_d o_rd·do_rd` is formed
/// from the bf16-stored `o` (`flash_bwd_preprocess_kernel.h`, `dot_do_o`);
/// `dS = P ⊙ (dP − δ)` is formed in f32 then converted to bf16 before
/// `dQ = dS·K` and `dK = dSᵀ·Q`; every output is stored once in bf16. The
/// tensor-core products of bf16 operands are exact in f32 and the sums
/// (≤ 9 terms here) accumulate in f32 — negligible next to `2^-8`. So:
///
/// - `mass(dv[c,d]) = Σ_r P_rc·|do_rd| + |dv[c,d]|`
/// - `mass(dq[r,d]) = scale·Σ_c (|dS_rc| + P_rc·m_r)·|k_cd| + |dq[r,d]|`
/// - `mass(dk[c,d]) = scale·Σ_r (|dS_rc| + P_rc·m_r)·|q_rd| + |dk[c,d]|`
///
/// with `m_r = Σ_d |o_rd·do_rd|` (the δ-from-bf16-`o` term, `Δδ_r <= 2^-8·m_r`,
/// which enters `dS` scaled by `P_rc`). The signed sums CANCEL (dS has
/// both signs), which is exactly why a bound relative to `max|g|` is the
/// wrong shape: the error scales with the L1 mass of the terms, not with
/// the result.
struct BwdRef {
    /// `[total, 3, H, D]` analytic gradient.
    g: Vec<f64>,
    /// `[total, 3, H, D]` error mass (see above).
    mass: Vec<f64>,
}

fn attn_bwd_ref(
    qkv: &[f64],
    d_o: &[f64],
    lens: &[usize],
    scale: f64,
    window: Option<usize>,
) -> BwdRef {
    let total: usize = lens.iter().sum();
    let mut g = vec![0.0; total * 3 * H * D];
    let mut mass = vec![0.0; total * 3 * H * D];
    let (o_all, _) = attn_ref(qkv, lens, scale, window);
    let mut start = 0;
    for &n in lens {
        for h in 0..H {
            // P [n, n], dS [n, n], m [n]
            let mut p = vec![0.0; n * n];
            let mut ds = vec![0.0; n * n];
            let mut m = vec![0.0; n];
            for r in 0..n {
                let rq = start + r;
                let mut scores = vec![f64::NEG_INFINITY; n];
                for (c, sc) in scores.iter_mut().enumerate() {
                    let in_band = window.is_none_or(|w| (r as i64 - c as i64).abs() <= w as i64);
                    if in_band {
                        let rk = start + c;
                        *sc = (0..D)
                            .map(|d| qkv[iqkv(rq, 0, h, d)] * qkv[iqkv(rk, 1, h, d)])
                            .sum::<f64>()
                            * scale;
                    }
                }
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let l: f64 = scores.iter().map(|s| (s - mx).exp()).sum();
                for c in 0..n {
                    p[r * n + c] = (scores[c] - mx).exp() / l;
                }
                let delta: f64 = (0..D)
                    .map(|d| o_all[io(rq, h, d)] * d_o[io(rq, h, d)])
                    .sum();
                m[r] = (0..D)
                    .map(|d| (o_all[io(rq, h, d)] * d_o[io(rq, h, d)]).abs())
                    .sum();
                for c in 0..n {
                    let rv = start + c;
                    let dp: f64 = (0..D)
                        .map(|d| d_o[io(rq, h, d)] * qkv[iqkv(rv, 2, h, d)])
                        .sum();
                    ds[r * n + c] = p[r * n + c] * (dp - delta);
                }
            }
            // dv_c = Σ_r P_rc do_r ; dq_r = scale Σ_c dS_rc k_c ; dk_c = scale Σ_r dS_rc q_r
            for c in 0..n {
                let rc = start + c;
                for d in 0..D {
                    let mut dv = 0.0;
                    let mut dv_mass = 0.0;
                    let mut dk = 0.0;
                    let mut dk_mass = 0.0;
                    for r in 0..n {
                        let rq = start + r;
                        let prc = p[r * n + c];
                        dv += prc * d_o[io(rq, h, d)];
                        dv_mass += prc * d_o[io(rq, h, d)].abs();
                        let dsrc = ds[r * n + c];
                        dk += dsrc * qkv[iqkv(rq, 0, h, d)];
                        dk_mass += (dsrc.abs() + prc * m[r]) * qkv[iqkv(rq, 0, h, d)].abs();
                    }
                    g[iqkv(rc, 2, h, d)] = dv;
                    mass[iqkv(rc, 2, h, d)] = dv_mass + dv.abs();
                    g[iqkv(rc, 1, h, d)] = scale * dk;
                    mass[iqkv(rc, 1, h, d)] = scale * dk_mass + (scale * dk).abs();
                }
            }
            for r in 0..n {
                let rq = start + r;
                for d in 0..D {
                    let mut dq = 0.0;
                    let mut dq_mass = 0.0;
                    for c in 0..n {
                        let rk = start + c;
                        let dsrc = ds[r * n + c];
                        dq += dsrc * qkv[iqkv(rk, 1, h, d)];
                        dq_mass +=
                            (dsrc.abs() + p[r * n + c] * m[r]) * qkv[iqkv(rk, 1, h, d)].abs();
                    }
                    g[iqkv(rq, 0, h, d)] = scale * dq;
                    mass[iqkv(rq, 0, h, d)] = scale * dq_mass + (scale * dq).abs();
                }
            }
        }
        start += n;
    }
    BwdRef { g, mass }
}

/// Per-slot gradient bound: `2 · 2^-8 · max_i mass_i` over the slot — the
/// worst-case rounding mass (derivation on [`attn_bwd_ref`]) with a 2×
/// headroom factor, compared against the slot's max error (measured on
/// the A100: 0.06–0.60 of the worst case per slot, dv the closest).
/// Mutations it
/// is proven to detect (asserted below): the finite-difference gradient of
/// the loss with a UNIFORM d_o (every entry the mean of the fixture's),
/// and with a 1.5× scale — both exceed the bound in at least one slot.
const G_HEADROOM: f64 = 2.0;
const BF16_HALF_ULP: f64 = 1.0 / 256.0;

/// The analytic reference must agree with the finite differences to far
/// below any bf16 scale — this pins the derivation the masses rest on
/// (FD truncation ≈ 1e-6·scale of the third derivative, see [`grad_fd`]).
const FD_VS_ANALYTIC_TOL: f64 = 1e-6;

fn run_bwd(
    dev: &CudaDevice,
    dfx: &DeviceFixture,
    window: Option<u32>,
    deterministic: bool,
) -> Vec<bf16> {
    let c = cfg(window, deterministic);
    let (o, lse) = flash_varlen_fwd(dev, &dfx.qkv, &dfx.cu, H, &c).unwrap();
    let d_qkv = flash_varlen_bwd(dev, &dfx.qkv, &dfx.cu, H, &o, &lse, &dfx.d_o, &c).unwrap();
    dev.cuda_stream().synchronize().unwrap();
    dtoh(dev, &d_qkv)
}

/// `(max |a − b|, max |b|, max mass)` over one slot (0 = dq, 1 = dk, 2 = dv).
fn slot_stats(a: &[f64], b: &[f64], mass: &[f64], lens: &[usize], slot: usize) -> (f64, f64, f64) {
    let total: usize = lens.iter().sum();
    let mut err: f64 = 0.0;
    let mut bmax: f64 = 0.0;
    let mut mmax: f64 = 0.0;
    for r in 0..total {
        for h in 0..H {
            for d in 0..D {
                let i = iqkv(r, slot, h, d);
                err = err.max((a[i] - b[i]).abs());
                bmax = bmax.max(b[i].abs());
                mmax = mmax.max(mass[i].abs());
            }
        }
    }
    (err, bmax, mmax)
}

/// Full central-FD gradcheck (the `_an`/uniform-do/wrong-scale discrimination
/// suite) — SMALL_LENS only, see [`SMALL_LENS`]'s doc for why (intractable
/// at multi-tile scale).
#[test]
fn bwd_matches_central_finite_differences() {
    let Some(dev) = cuda_device() else { return };
    let lens = &SMALL_LENS;
    let fx = fixture(lens);
    let dfx = upload(&dev, &fx);
    let scale = SCALE as f64;

    for window in [None, Some(2u32)] {
        let w = window.map(|w| w as usize);
        let g_fd = grad_fd(&fx.qkv, &fx.d_o, lens, scale, w);
        let bwd_ref = attn_bwd_ref(&fx.qkv, &fx.d_o, lens, scale, w);
        // The derivation behind the masses is itself checked: analytic == FD.
        let e_an = max_abs_diff(&bwd_ref.g, &g_fd);
        let g_scale = 1.0 + max_abs(&g_fd);
        println!(
            "bwd window={window:?}: max|g_analytic-g_fd|={e_an:.3e} (tol {:.1e})",
            FD_VS_ANALYTIC_TOL * g_scale
        );
        assert!(
            e_an <= FD_VS_ANALYTIC_TOL * g_scale,
            "analytic reference disagrees with FD"
        );
        // Mutated references for the discrimination proof.
        let mean_do = fx.d_o.iter().sum::<f64>() / fx.d_o.len() as f64;
        let d_o_uniform = vec![mean_do; fx.d_o.len()];
        let g_fd_uniform_do = grad_fd(&fx.qkv, &d_o_uniform, lens, scale, w);
        let g_fd_wrong_scale = grad_fd(&fx.qkv, &fx.d_o, lens, 1.5 * scale, w);

        for deterministic in [true, false] {
            let g_fa = bf16_to_f64(&run_bwd(&dev, &dfx, window, deterministic));
            assert_eq!(g_fa.len(), geom(&dfx.cu).qkv_len());
            assert!(g_fa.iter().all(|x| x.is_finite()), "non-finite gradient");
            let mut any_uniform_detected = false;
            let mut any_scale_detected = false;
            for (slot, name) in [(0, "dq"), (1, "dk"), (2, "dv")] {
                let (err, gmax, mmax) = slot_stats(&g_fa, &g_fd, &bwd_ref.mass, lens, slot);
                let tol = G_HEADROOM * BF16_HALF_ULP * mmax;
                println!(
                    "bwd window={window:?} deterministic={deterministic}: {name} max|g_fa-g_fd|={err:.3e} (bound {tol:.3e} = 2·2^-8·max mass {mmax:.3e}; max|g_fd|={gmax:.3e}; bound/max|signal|={:.3e}; err/derived-worst-case={:.2})",
                    tol / gmax,
                    err / (BF16_HALF_ULP * mmax)
                );
                assert!(gmax > 0.0, "{name}: reference gradient is identically zero");
                assert!(
                    err <= tol,
                    "window {window:?} deterministic={deterministic}: {name} off by {err:.3e} > {tol:.3e}"
                );
                let (e_u, _, _) = slot_stats(&g_fa, &g_fd_uniform_do, &bwd_ref.mass, lens, slot);
                let (e_s, _, _) = slot_stats(&g_fa, &g_fd_wrong_scale, &bwd_ref.mass, lens, slot);
                any_uniform_detected |= e_u > tol;
                any_scale_detected |= e_s > tol;
            }
            assert!(
                any_uniform_detected,
                "the bound cannot distinguish the fixture's d_o from a uniform one"
            );
            assert!(
                any_scale_detected,
                "the bound cannot distinguish the scale from 1.5× the scale"
            );
        }
    }
}

/// Multi-tile leg of the backward oracle (F3): the analytic reference
/// (`attn_bwd_ref`) ONLY — no finite differences (intractable at this
/// scale, see [`SMALL_LENS`]'s doc) — but `bwd_matches_central_finite_differences`
/// already proves the analytic reference agrees with FD to `1e-6·scale`
/// relative, so the analytic reference alone is a validated oracle here.
/// Discrimination mirrors the forward test: the SAME bound must reject the
/// wrong window and the wrong scale.
#[test]
fn bwd_matches_analytic_reference_multi_tile() {
    let Some(dev) = cuda_device() else { return };
    let lens = &LARGE_LENS;
    let fx = fixture(lens);
    let dfx = upload(&dev, &fx);
    let scale = SCALE as f64;

    for window in [None, Some(2u32)] {
        let w = window.map(|w| w as usize);
        let bwd_ref = attn_bwd_ref(&fx.qkv, &fx.d_o, lens, scale, w);
        let wrong_window = match w {
            None => Some(2usize),
            Some(_) => None,
        };
        let bwd_ref_wrong_window = attn_bwd_ref(&fx.qkv, &fx.d_o, lens, scale, wrong_window);
        let bwd_ref_wrong_scale = attn_bwd_ref(&fx.qkv, &fx.d_o, lens, 1.5 * scale, w);

        for deterministic in [true, false] {
            let g_fa = bf16_to_f64(&run_bwd(&dev, &dfx, window, deterministic));
            assert_eq!(g_fa.len(), geom(&dfx.cu).qkv_len());
            assert!(g_fa.iter().all(|x| x.is_finite()), "non-finite gradient");
            let mut any_window_detected = false;
            let mut any_scale_detected = false;
            for (slot, name) in [(0, "dq"), (1, "dk"), (2, "dv")] {
                let (err, gmax, mmax) = slot_stats(&g_fa, &bwd_ref.g, &bwd_ref.mass, lens, slot);
                let tol = G_HEADROOM * BF16_HALF_ULP * mmax;
                println!(
                    "multi_tile bwd window={window:?} deterministic={deterministic}: {name} max|g_fa-g_ref|={err:.3e} (bound {tol:.3e}; max|g_ref|={gmax:.3e})"
                );
                assert!(gmax > 0.0, "{name}: reference gradient is identically zero");
                assert!(
                    err <= tol,
                    "lens={lens:?} window {window:?} deterministic={deterministic}: {name} off by {err:.3e} > {tol:.3e}"
                );
                let (e_w, _, _) =
                    slot_stats(&g_fa, &bwd_ref_wrong_window.g, &bwd_ref.mass, lens, slot);
                let (e_s, _, _) =
                    slot_stats(&g_fa, &bwd_ref_wrong_scale.g, &bwd_ref.mass, lens, slot);
                any_window_detected |= e_w > tol;
                any_scale_detected |= e_s > tol;
            }
            assert!(
                any_window_detected,
                "the bound cannot distinguish the window from {wrong_window:?}"
            );
            assert!(
                any_scale_detected,
                "the bound cannot distinguish the scale from 1.5× the scale"
            );
        }
    }
}

fn bwd_deterministic_rerun_is_bit_identical_body(dev: &CudaDevice, lens: &[usize]) {
    let fx = fixture(lens);
    let dfx = upload(dev, &fx);
    let bits = |v: Vec<bf16>| -> Vec<u16> { v.iter().map(|x| x.to_bits()).collect() };
    let a = bits(run_bwd(dev, &dfx, Some(2), true));
    let b = bits(run_bwd(dev, &dfx, Some(2), true));
    assert_eq!(a, b, "deterministic backward differs between two runs");
    assert!(
        a.iter().any(|x| *x != a[0]),
        "d_qkv is constant — nothing was computed"
    );
    // Observation only (not a gate): whether deterministic=false differs
    // at this geometry. At B·H = 6 the non-deterministic path still uses
    // atomics across the 1–2 key blocks per sequence, whose order may or
    // may not vary between runs.
    let c = bits(run_bwd(dev, &dfx, Some(2), false));
    let d = bits(run_bwd(dev, &dfx, Some(2), false));
    println!(
        "observation: deterministic=false reruns bit-identical: {}; equals deterministic=true: {}",
        c == d,
        c == a
    );
}

#[test]
fn bwd_deterministic_rerun_is_bit_identical() {
    let Some(dev) = cuda_device() else { return };
    bwd_deterministic_rerun_is_bit_identical_body(&dev, &SMALL_LENS);
}

#[test]
fn bwd_deterministic_rerun_is_bit_identical_multi_tile() {
    let Some(dev) = cuda_device() else { return };
    bwd_deterministic_rerun_is_bit_identical_body(&dev, &LARGE_LENS);
}

// ---------------------------------------------------------------------------
// (d) the C wrapper's refusal cells, driven through `raw`
// ---------------------------------------------------------------------------

struct RawBuffers {
    dfx: DeviceFixture,
    o: candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    lse: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    d_qkv: candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    scratch: BwdScratch,
}

fn raw_buffers(dev: &CudaDevice, fx: &Fixture) -> RawBuffers {
    let dfx = upload(dev, fx);
    let g = geom(&dfx.cu);
    RawBuffers {
        o: dev.alloc_zeros::<bf16>(g.o_len()).unwrap(),
        lse: dev.alloc_zeros::<f32>(g.lse_len()).unwrap(),
        d_qkv: dev.alloc_zeros::<bf16>(g.qkv_len()).unwrap(),
        scratch: BwdScratch::alloc(dev, &g, true).unwrap(),
        dfx,
    }
}

fn dptr<T>(
    dev: &CudaDevice,
    s: &candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
) -> *mut std::ffi::c_void {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let stream = dev.cuda_stream();
    let (p, _g) = s.device_ptr(&stream);
    p as usize as *mut std::ffi::c_void
}

/// Like [`dptr`], but for the `CuSeqlens`' device array — the `raw`
/// escape hatch is the one place this module deliberately reaches past
/// `CuSeqlens` to drive the C wrapper's OWN refusal cells directly (F1's
/// safe API is not involved in these tests at all).
fn dptr_cu(dev: &CudaDevice, cu: &CuSeqlens) -> *const i32 {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let stream = dev.cuda_stream();
    let view = cu.as_view();
    let (p, _g) = view.device_ptr(&stream);
    p as usize as *const i32
}

fn valid_fwd_args(dev: &CudaDevice, b: &RawBuffers) -> raw::FwdArgs {
    let g = geom(&b.dfx.cu);
    raw::FwdArgs {
        qkv: dptr(dev, &b.dfx.qkv),
        o: dptr(dev, &b.o),
        softmax_lse: dptr(dev, &b.lse) as *mut f32,
        cu_seqlens: dptr_cu(dev, &b.dfx.cu),
        stream: dev.cuda_stream().cu_stream() as *mut std::ffi::c_void,
        qkv_len: g.qkv_len() as i64,
        o_len: g.o_len() as i64,
        softmax_lse_len: g.lse_len() as i64,
        cu_seqlens_len: g.cu_seqlens_len() as i64,
        struct_size: std::mem::size_of::<raw::FwdArgs>() as i32,
        total_q: g.total_q() as i32,
        batch: g.batch() as i32,
        num_heads: g.num_heads() as i32,
        head_dim: HEAD_DIM as i32,
        max_seqlen: g.max_seqlen() as i32,
        window_size_left: 2,
        window_size_right: 2,
        softmax_scale: SCALE,
        p_dropout: 0.0,
    }
}

fn valid_bwd_args(dev: &CudaDevice, b: &RawBuffers) -> raw::BwdArgs {
    let g = geom(&b.dfx.cu);
    raw::BwdArgs {
        qkv: dptr(dev, &b.dfx.qkv),
        o: dptr(dev, &b.o),
        softmax_lse: dptr(dev, &b.lse) as *const f32,
        d_o: dptr(dev, &b.dfx.d_o),
        d_qkv: dptr(dev, &b.d_qkv),
        softmax_d: dptr(dev, &b.scratch.softmax_d) as *mut f32,
        dq_accum: dptr(dev, &b.scratch.dq_accum) as *mut f32,
        cu_seqlens: dptr_cu(dev, &b.dfx.cu),
        stream: dev.cuda_stream().cu_stream() as *mut std::ffi::c_void,
        qkv_len: g.qkv_len() as i64,
        o_len: g.o_len() as i64,
        softmax_lse_len: g.lse_len() as i64,
        d_o_len: g.o_len() as i64,
        d_qkv_len: g.qkv_len() as i64,
        softmax_d_len: g.softmax_d_len() as i64,
        dq_accum_len: g.dq_accum_len(b.scratch.splits) as i64,
        cu_seqlens_len: g.cu_seqlens_len() as i64,
        struct_size: std::mem::size_of::<raw::BwdArgs>() as i32,
        total_q: g.total_q() as i32,
        batch: g.batch() as i32,
        num_heads: g.num_heads() as i32,
        head_dim: HEAD_DIM as i32,
        max_seqlen: g.max_seqlen() as i32,
        window_size_left: 2,
        window_size_right: 2,
        softmax_scale: SCALE,
        p_dropout: 0.0,
        deterministic: 1,
        dq_accum_splits: b.scratch.splits as i32,
    }
}

fn status_of(code: i32) -> Option<FlashStatus> {
    match flash::check_status(code) {
        Ok(()) => None,
        Err(FlashError::Refused { status, .. }) => Some(status),
        Err(e) => panic!("unexpected error class: {e}"),
    }
}

#[test]
fn c_wrapper_refuses_dropout_with_the_typed_status() {
    let Some(dev) = cuda_device() else { return };
    let fx = fixture(&SMALL_LENS);
    let b = raw_buffers(&dev, &fx);
    // Positive control first: the unmutated args run (the cell table
    // below is only meaningful if the baseline is accepted).
    let ok = valid_fwd_args(&dev, &b);
    assert_eq!(status_of(unsafe { raw::jammi_flash_varlen_fwd(&ok) }), None);
    let ok_b = valid_bwd_args(&dev, &b);
    assert_eq!(
        status_of(unsafe { raw::jammi_flash_varlen_bwd(&ok_b) }),
        None
    );
    dev.cuda_stream().synchronize().unwrap();

    let mut a = ok;
    a.p_dropout = 0.1;
    let code = unsafe { raw::jammi_flash_varlen_fwd(&a) };
    assert_eq!(status_of(code), Some(FlashStatus::DropoutUnsupported));
    let err = flash::check_status(code).unwrap_err();
    assert_eq!(err.status(), Some(FlashStatus::DropoutUnsupported));
    assert!(err.to_string().contains("p_dropout must be 0.0"), "{err}");

    let mut a = ok_b;
    a.p_dropout = 0.1;
    assert_eq!(
        status_of(unsafe { raw::jammi_flash_varlen_bwd(&a) }),
        Some(FlashStatus::DropoutUnsupported)
    );
    // NaN is "not zero" too.
    a.p_dropout = f32::NAN;
    assert_eq!(
        status_of(unsafe { raw::jammi_flash_varlen_bwd(&a) }),
        Some(FlashStatus::DropoutUnsupported)
    );
}

#[test]
fn c_wrapper_refusal_cells() {
    let Some(dev) = cuda_device() else { return };
    let fx = fixture(&SMALL_LENS);
    let b = raw_buffers(&dev, &fx);
    let ok = valid_fwd_args(&dev, &b);
    let ok_b = valid_bwd_args(&dev, &b);

    type FwdMut = fn(&mut raw::FwdArgs);
    let fwd_cells: Vec<(&str, FwdMut, FlashStatus)> = vec![
        (
            "null qkv",
            |a| a.qkv = std::ptr::null(),
            FlashStatus::NullPointer,
        ),
        (
            "null cu_seqlens",
            |a| a.cu_seqlens = std::ptr::null(),
            FlashStatus::NullPointer,
        ),
        ("head_dim 32", |a| a.head_dim = 32, FlashStatus::HeadDim),
        ("head_dim 128", |a| a.head_dim = 128, FlashStatus::HeadDim),
        ("batch 0", |a| a.batch = 0, FlashStatus::Dims),
        ("num_heads 0", |a| a.num_heads = 0, FlashStatus::Dims),
        ("total_q 0", |a| a.total_q = 0, FlashStatus::Dims),
        ("max_seqlen 0", |a| a.max_seqlen = 0, FlashStatus::Dims),
        (
            "max_seqlen > total_q",
            |a| a.max_seqlen = a.total_q + 1,
            FlashStatus::Dims,
        ),
        (
            "window (-1, 0) = causal",
            |a| {
                a.window_size_left = -1;
                a.window_size_right = 0;
            },
            FlashStatus::CausalUnsupported,
        ),
        (
            "window -2",
            |a| a.window_size_left = -2,
            FlashStatus::Window,
        ),
        ("scale 0", |a| a.softmax_scale = 0.0, FlashStatus::Scale),
        (
            "scale NaN",
            |a| a.softmax_scale = f32::NAN,
            FlashStatus::Scale,
        ),
        ("qkv_len short", |a| a.qkv_len -= 1, FlashStatus::BufferLen),
        ("o_len long", |a| a.o_len += 1, FlashStatus::BufferLen),
        (
            "lse_len short",
            |a| a.softmax_lse_len -= 1,
            FlashStatus::BufferLen,
        ),
        (
            "cu_seqlens_len long",
            |a| a.cu_seqlens_len += 1,
            FlashStatus::BufferLen,
        ),
        (
            "struct_size wrong",
            |a| a.struct_size += 8,
            FlashStatus::Abi,
        ),
    ];
    for (name, mutate, expected) in fwd_cells {
        let mut a = ok;
        mutate(&mut a);
        let got = status_of(unsafe { raw::jammi_flash_varlen_fwd(&a) });
        assert_eq!(got, Some(expected), "fwd cell `{name}`");
    }
    // The window >= max_seqlen normalisation (flash_api.cpp:608-609) is
    // NOT a refusal: (9, 9) at max_seqlen 9 becomes (-1, -1) and runs.
    let mut a = ok;
    a.window_size_left = 9;
    a.window_size_right = 9;
    assert_eq!(status_of(unsafe { raw::jammi_flash_varlen_fwd(&a) }), None);
    // A one-sided window (left >= 0, right < 0) is legal upstream
    // (right := seqlen_k, flash_api.cpp:142) and runs.
    let mut a = ok;
    a.window_size_right = -1;
    assert_eq!(status_of(unsafe { raw::jammi_flash_varlen_fwd(&a) }), None);

    type BwdMut = fn(&mut raw::BwdArgs);
    let bwd_cells: Vec<(&str, BwdMut, FlashStatus)> = vec![
        (
            "null d_qkv",
            |a| a.d_qkv = std::ptr::null_mut(),
            FlashStatus::NullPointer,
        ),
        (
            "null dq_accum",
            |a| a.dq_accum = std::ptr::null_mut(),
            FlashStatus::NullPointer,
        ),
        ("head_dim 96", |a| a.head_dim = 96, FlashStatus::HeadDim),
        ("batch 0", |a| a.batch = 0, FlashStatus::Dims),
        (
            "window (-1, 0) = causal",
            |a| {
                a.window_size_left = -1;
                a.window_size_right = 0;
            },
            FlashStatus::CausalUnsupported,
        ),
        (
            "window right -3",
            |a| a.window_size_right = -3,
            FlashStatus::Window,
        ),
        ("scale -1", |a| a.softmax_scale = -1.0, FlashStatus::Scale),
        (
            "dq_accum_splits + 1",
            |a| a.dq_accum_splits += 1,
            FlashStatus::DqAccumSplits,
        ),
        (
            "dq_accum_splits 0",
            |a| a.dq_accum_splits = 0,
            FlashStatus::DqAccumSplits,
        ),
        (
            "softmax_d_len short",
            |a| a.softmax_d_len -= 1,
            FlashStatus::BufferLen,
        ),
        (
            "dq_accum_len short",
            |a| a.dq_accum_len -= 1,
            FlashStatus::BufferLen,
        ),
        ("d_o_len long", |a| a.d_o_len += 1, FlashStatus::BufferLen),
        (
            "d_qkv_len short",
            |a| a.d_qkv_len -= 1,
            FlashStatus::BufferLen,
        ),
        (
            "struct_size wrong",
            |a| a.struct_size -= 4,
            FlashStatus::Abi,
        ),
    ];
    for (name, mutate, expected) in bwd_cells {
        let mut a = ok_b;
        mutate(&mut a);
        let got = status_of(unsafe { raw::jammi_flash_varlen_bwd(&a) });
        assert_eq!(got, Some(expected), "bwd cell `{name}`");
    }
    // deterministic=0 with a splits=1 buffer is the non-deterministic
    // cell: the splits check passes (1 == 1) and the buffer length must
    // then be the single-split size.
    let mut a = ok_b;
    a.deterministic = 0;
    a.dq_accum_splits = 1;
    a.dq_accum_len = geom(&b.dfx.cu).dq_accum_len(1) as i64;
    assert_eq!(status_of(unsafe { raw::jammi_flash_varlen_bwd(&a) }), None);
    dev.cuda_stream().synchronize().unwrap();

    // Null args pointer.
    assert_eq!(
        status_of(unsafe { raw::jammi_flash_varlen_fwd(std::ptr::null()) }),
        Some(FlashStatus::NullPointer)
    );
    // strerror never returns NULL, even for garbage.
    for code in [-5, 0, 1, 13, 14, 1000] {
        let p = unsafe { raw::jammi_flash_strerror(code) };
        assert!(!p.is_null());
    }
}

#[test]
fn rust_boundary_refuses_wrong_buffer_lengths_and_splits() {
    let Some(dev) = cuda_device() else { return };
    let fx = fixture(&SMALL_LENS);
    let dfx = upload(&dev, &fx);
    let g = geom(&dfx.cu);
    let c = cfg(Some(2), true);
    let mut o = dev.alloc_zeros::<bf16>(g.o_len()).unwrap();
    let mut lse = dev.alloc_zeros::<f32>(g.lse_len()).unwrap();
    // A short qkv view.
    let e = flash_varlen_fwd_into(
        &dev,
        dfx.qkv.slice(0..g.qkv_len() - 64),
        &dfx.cu,
        o.as_view_mut(),
        lse.as_view_mut(),
        H,
        &c,
    )
    .unwrap_err();
    assert!(matches!(e, FlashError::Geometry(_)), "{e}");
    assert!(e.to_string().contains("qkv"), "{e}");
    // A short lse view.
    let e = flash_varlen_fwd_into(
        &dev,
        dfx.qkv.as_view(),
        &dfx.cu,
        o.as_view_mut(),
        lse.slice_mut(0..g.lse_len() - 1),
        H,
        &c,
    )
    .unwrap_err();
    assert!(
        matches!(e, FlashError::Geometry(_)) && e.to_string().contains("lse"),
        "{e}"
    );
    // The unmutated call runs (positive control).
    flash_varlen_fwd_into(
        &dev,
        dfx.qkv.as_view(),
        &dfx.cu,
        o.as_view_mut(),
        lse.as_view_mut(),
        H,
        &c,
    )
    .unwrap();

    // Backward: scratch allocated for deterministic=true carries
    // `splits` >= 1 = ceil(num_SM / 6) on any GPU with > 6 SMs; passing
    // it to a deterministic=false call (which uses 1) is the splits cell.
    let mut scratch = BwdScratch::alloc(&dev, &g, true).unwrap();
    let expected = dq_accum_splits(&dev, g.batch(), g.num_heads(), true).unwrap();
    assert_eq!(scratch.splits, expected);
    println!("dq_accum splits at B·H=6 on this device: {expected}");
    let mut d_qkv = dev.alloc_zeros::<bf16>(g.qkv_len()).unwrap();
    if expected != 1 {
        let e = flash_varlen_bwd_into(
            &dev,
            &dfx.cu,
            H,
            BwdBuffers {
                qkv: dfx.qkv.as_view(),
                o: o.as_view(),
                lse: lse.as_view(),
                d_o: dfx.d_o.as_view(),
                d_qkv: d_qkv.as_view_mut(),
                softmax_d: scratch.softmax_d.as_view_mut(),
                dq_accum: scratch.dq_accum.as_view_mut(),
                dq_accum_splits: scratch.splits,
            },
            &cfg(Some(2), false),
        )
        .unwrap_err();
        assert!(
            matches!(e, FlashError::Geometry(_)) && e.to_string().contains("split"),
            "{e}"
        );
    }
    // A short dq_accum view with the right splits count.
    let e = flash_varlen_bwd_into(
        &dev,
        &dfx.cu,
        H,
        BwdBuffers {
            qkv: dfx.qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: dfx.d_o.as_view(),
            d_qkv: d_qkv.as_view_mut(),
            softmax_d: scratch.softmax_d.as_view_mut(),
            dq_accum: scratch.dq_accum.slice_mut(0..g.dq_accum_len(expected) - 1),
            dq_accum_splits: scratch.splits,
        },
        &c,
    )
    .unwrap_err();
    assert!(
        matches!(e, FlashError::Geometry(_)) && e.to_string().contains("dq_accum"),
        "{e}"
    );
    // The unmutated backward runs (positive control).
    flash_varlen_bwd_into(
        &dev,
        &dfx.cu,
        H,
        BwdBuffers {
            qkv: dfx.qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: dfx.d_o.as_view(),
            d_qkv: d_qkv.as_view_mut(),
            softmax_d: scratch.softmax_d.as_view_mut(),
            dq_accum: scratch.dq_accum.as_view_mut(),
            dq_accum_splits: scratch.splits,
        },
        &c,
    )
    .unwrap();
    dev.cuda_stream().synchronize().unwrap();
    // Geometry refusals reach the safe API before any FFI call: num_heads
    // == 0 is the analogue of the old "independently-supplied bad
    // VarlenGeometry" cell — driven through the REAL entry point now that
    // `VarlenGeometry` can no longer be constructed independently of a
    // `CuSeqlens` (F1).
    let e = flash_varlen_fwd(&dev, &dfx.qkv, &dfx.cu, 0, &c).unwrap_err();
    assert!(matches!(e, FlashError::Geometry(_)), "{e}");
}

// ---------------------------------------------------------------------------
// (e) guard-region poison: 4 KB before AND after every output/scratch buffer
// ---------------------------------------------------------------------------

/// A poisoned buffer: `[pad | payload | pad]` with `pad` = 4096 bytes of a
/// recognisable pattern on each side; `payload` is the view the kernel
/// gets. After the run both pads must be BIT-identical to the pattern
/// (the poison is a NaN pattern, so value comparison would report every
/// element as changed — `NaN != NaN` — and the test would be vacuous in
/// the wrong direction; bits are what a stray write changes).
struct Poisoned<T> {
    buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    pad: usize,
    n: usize,
}

/// Bit view + zero for the two element types the buffers use.
trait Bits: Copy {
    fn bits(self) -> u32;
    fn zero() -> Self;
}
impl Bits for f32 {
    fn bits(self) -> u32 {
        self.to_bits()
    }
    fn zero() -> Self {
        0.0
    }
}
impl Bits for bf16 {
    fn bits(self) -> u32 {
        self.to_bits() as u32
    }
    fn zero() -> Self {
        bf16::ZERO
    }
}

impl<T: DeviceRepr + Bits + std::fmt::Debug + 'static> Poisoned<T> {
    fn new(dev: &CudaDevice, n: usize, poison: T) -> Self {
        let pad = 4096 / std::mem::size_of::<T>();
        let host = vec![poison; n + 2 * pad];
        let buf = dev.clone_htod(&host).unwrap();
        Self { buf, pad, n }
    }
    fn view(&self) -> candle_core::cuda_backend::cudarc::driver::CudaView<'_, T> {
        self.buf.slice(self.pad..self.pad + self.n)
    }
    fn view_mut(&mut self) -> candle_core::cuda_backend::cudarc::driver::CudaViewMut<'_, T> {
        self.buf.slice_mut(self.pad..self.pad + self.n)
    }
    /// Zero the payload (the deterministic dq_accum precondition) without
    /// touching the pads.
    fn zero_payload(&mut self, dev: &CudaDevice) {
        let zeros = vec![T::zero(); self.n];
        let mut v = self.view_mut();
        dev.memcpy_htod(&zeros, &mut v).unwrap();
    }
    fn assert_pads_intact(&self, dev: &CudaDevice, poison: T, name: &str) {
        let host = dtoh(dev, &self.buf);
        let pb = poison.bits();
        let before = &host[..self.pad];
        let after = &host[self.pad + self.n..];
        let bad_before = before.iter().position(|x| x.bits() != pb);
        let bad_after = after.iter().position(|x| x.bits() != pb);
        assert!(
            bad_before.is_none(),
            "{name}: guard region BEFORE the buffer was written at element {bad_before:?} (of {} pad elements)",
            self.pad
        );
        assert!(
            bad_after.is_none(),
            "{name}: guard region AFTER the buffer was written at element {bad_after:?} (of {} pad elements)",
            self.pad
        );
        // Non-vacuous: the payload was actually written (not all poison).
        let payload = &host[self.pad..self.pad + self.n];
        assert!(
            payload.iter().any(|x| x.bits() != pb),
            "{name}: payload still entirely poison — the kernel wrote nothing"
        );
    }
}

fn guard_regions_around_every_output_and_scratch_buffer_are_untouched_body(
    dev: &CudaDevice,
    lens: &[usize],
) {
    let fx = fixture(lens);
    let dfx = upload(dev, &fx);
    let g = geom(&dfx.cu);
    // Distinct, non-zero, non-finite-looking patterns: a bf16 NaN payload
    // and an f32 NaN payload with a recognisable mantissa.
    let pb = bf16::from_bits(0xFFAD);
    let pf = f32::from_bits(0xFFC0_DEAD);

    for deterministic in [true, false] {
        let c = cfg(Some(2), deterministic);
        let mut o = Poisoned::new(dev, g.o_len(), pb);
        let mut lse = Poisoned::new(dev, g.lse_len(), pf);
        flash_varlen_fwd_into(
            dev,
            dfx.qkv.as_view(),
            &dfx.cu,
            o.view_mut(),
            lse.view_mut(),
            H,
            &c,
        )
        .unwrap();
        dev.cuda_stream().synchronize().unwrap();
        o.assert_pads_intact(dev, pb, "o");
        lse.assert_pads_intact(dev, pf, "lse");

        let splits = dq_accum_splits(dev, g.batch(), g.num_heads(), deterministic).unwrap();
        let mut d_qkv = Poisoned::new(dev, g.qkv_len(), pb);
        let mut softmax_d = Poisoned::new(dev, g.softmax_d_len(), pf);
        let mut dq_accum = Poisoned::new(dev, g.dq_accum_len(splits), pf);
        if deterministic {
            dq_accum.zero_payload(dev);
        }
        flash_varlen_bwd_into(
            dev,
            &dfx.cu,
            H,
            BwdBuffers {
                qkv: dfx.qkv.as_view(),
                o: o.view(),
                lse: lse.view(),
                d_o: dfx.d_o.as_view(),
                d_qkv: d_qkv.view_mut(),
                softmax_d: softmax_d.view_mut(),
                dq_accum: dq_accum.view_mut(),
                dq_accum_splits: splits,
            },
            &c,
        )
        .unwrap();
        dev.cuda_stream().synchronize().unwrap();
        d_qkv.assert_pads_intact(dev, pb, "d_qkv");
        softmax_d.assert_pads_intact(dev, pf, "softmax_d");
        dq_accum.assert_pads_intact(dev, pf, "dq_accum");
        // And the poisoned-then-overwritten outputs are still correct:
        // the poison in the payload did not leak into the result (every
        // written element is finite), and d_qkv matches the clean run.
        let d_qkv_host = dtoh(dev, &d_qkv.buf);
        let payload = &d_qkv_host[d_qkv.pad..d_qkv.pad + d_qkv.n];
        assert!(payload.iter().all(|x| x.to_f32().is_finite()));
        let clean = run_bwd(dev, &dfx, Some(2), deterministic);
        if deterministic {
            let a: Vec<u16> = payload.iter().map(|x| x.to_bits()).collect();
            let b: Vec<u16> = clean.iter().map(|x| x.to_bits()).collect();
            assert_eq!(
                a, b,
                "poisoned-buffer deterministic backward differs from the clean run"
            );
        }
        println!(
            "lens={lens:?} guard regions intact (deterministic={deterministic}, splits={splits})"
        );
    }
}

#[test]
fn guard_regions_around_every_output_and_scratch_buffer_are_untouched() {
    let Some(dev) = cuda_device() else { return };
    guard_regions_around_every_output_and_scratch_buffer_are_untouched_body(&dev, &SMALL_LENS);
}

#[test]
fn guard_regions_around_every_output_and_scratch_buffer_are_untouched_multi_tile() {
    let Some(dev) = cuda_device() else { return };
    guard_regions_around_every_output_and_scratch_buffer_are_untouched_body(&dev, &LARGE_LENS);
}

/// F4: `flash_varlen_bwd_into` re-zeroes `dq_accum` itself when
/// `cfg.deterministic`, rather than trusting the caller's `BwdBuffers` to
/// arrive zeroed (the deterministic launch accumulates without clearing,
/// `flash_bwd_launch_template.h:84-88` — a non-zero-on-entry `dq_accum`
/// silently produces a WRONG `dQ`, not an error). This is RED under the
/// pre-fix behaviour: poison `dq_accum` with a large non-zero pattern
/// (deliberately NOT zeroing it, unlike the guard-region test above) and
/// assert the deterministic backward still matches a clean run — that can
/// only hold if the function zeroes the buffer itself.
#[test]
fn bwd_deterministic_ignores_a_poisoned_dq_accum_on_entry() {
    let Some(dev) = cuda_device() else { return };
    let fx = fixture(&SMALL_LENS);
    let dfx = upload(&dev, &fx);
    let g = geom(&dfx.cu);
    let c = cfg(Some(2), true);
    let splits = dq_accum_splits(&dev, g.batch(), g.num_heads(), true).unwrap();

    let mut o = dev.alloc_zeros::<bf16>(g.o_len()).unwrap();
    let mut lse = dev.alloc_zeros::<f32>(g.lse_len()).unwrap();
    flash_varlen_fwd_into(
        &dev,
        dfx.qkv.as_view(),
        &dfx.cu,
        o.as_view_mut(),
        lse.as_view_mut(),
        H,
        &c,
    )
    .unwrap();
    dev.cuda_stream().synchronize().unwrap();

    // A large-magnitude, non-zero pattern — if the kernel's atomics simply
    // ADD to it (the bug this closes), the result is not just "off by a
    // rounding error", it is dominated by this poison and would trip the
    // bwd oracle's own bound by many orders of magnitude.
    let poison = vec![1.0e6f32; g.dq_accum_len(splits)];
    let mut dq_accum: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32> =
        dev.clone_htod(&poison).unwrap();
    let mut softmax_d = dev.alloc_zeros::<f32>(g.softmax_d_len()).unwrap();
    let mut d_qkv = dev.alloc_zeros::<bf16>(g.qkv_len()).unwrap();
    flash_varlen_bwd_into(
        &dev,
        &dfx.cu,
        H,
        BwdBuffers {
            qkv: dfx.qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: dfx.d_o.as_view(),
            d_qkv: d_qkv.as_view_mut(),
            softmax_d: softmax_d.as_view_mut(),
            dq_accum: dq_accum.as_view_mut(),
            dq_accum_splits: splits,
        },
        &c,
    )
    .unwrap();
    dev.cuda_stream().synchronize().unwrap();

    let poisoned_result = dtoh(&dev, &d_qkv);
    let clean_result = run_bwd(&dev, &dfx, Some(2), true);
    let a: Vec<u16> = poisoned_result.iter().map(|x| x.to_bits()).collect();
    let b: Vec<u16> = clean_result.iter().map(|x| x.to_bits()).collect();
    assert_eq!(
        a, b,
        "a poisoned (non-zero) dq_accum on entry changed the deterministic \
         backward's dQ — the function is not re-zeroing it"
    );
}

#[test]
fn num_sms_and_dq_accum_splits_agree() {
    let Some(dev) = cuda_device() else { return };
    let n = flash::num_sms(&dev).unwrap();
    // Any GPU this feature targets has many SMs (A100: 108); a stub
    // answer of 0 or 1 is not a device.
    assert!(n >= 2, "num_sms = {n}");
    // ceil(num_SM / (1·1)) = num_SM: the two helpers share one source.
    assert_eq!(dq_accum_splits(&dev, 1, 1, true).unwrap(), n);
    assert_eq!(dq_accum_splits(&dev, 1, 1, false).unwrap(), 1);
    // ceil(n / 6) for the fixture's B·H = 6.
    assert_eq!(dq_accum_splits(&dev, 3, 2, true).unwrap(), n.div_ceil(6));
    println!("num_sms = {n}");
}

#[test]
fn abi_sizes_match_between_rust_and_c() {
    // No device needed, but the library must be linked: the C side
    // reports its sizeof, Rust its own.
    let fwd = unsafe { raw::jammi_flash_sizeof_fwd_args() };
    let bwd = unsafe { raw::jammi_flash_sizeof_bwd_args() };
    assert_eq!(fwd, std::mem::size_of::<raw::FwdArgs>());
    assert_eq!(bwd, std::mem::size_of::<raw::BwdArgs>());
    assert_eq!(fwd, 112);
    assert_eq!(bwd, 184);
}
