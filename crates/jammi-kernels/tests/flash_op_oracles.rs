//! O0(b)/(c)/(d) (P6 Stage B contract §4) — the `ops::flash_attention_varlen`
//! `Tensor`/autograd-level oracles. O0(a) (numeric parity vs torch) is
//! `tests/flash_torch_parity.rs`, at the `crate::flash` FFI-boundary layer;
//! this file is the layer ABOVE it: the `Saved<T>`/`StatefulKernelOp`
//! wiring specifically.

#![cfg(feature = "flash-attn")]

use candle_core::{CudaDevice, DType, Device, Tensor};
use jammi_kernels::flash::{CuSeqlens, VarlenConfig};
use jammi_kernels::ops::flash_attention_varlen;

fn cuda_device() -> Option<CudaDevice> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d.as_cuda_device().unwrap().clone()),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "flash_op_oracles: JAMMI_REQUIRE_CUDA is set but no CUDA device could be \
                     acquired — a silent skip here is not acceptable: {e}"
                );
            }
            eprintln!("flash_op_oracles: skipping — no CUDA device available ({e})");
            None
        }
    }
}

const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = 64;

/// A synthetic bf16 `[total_q, 3, H, HEAD_DIM]` fixture — these tests are
/// STRUCTURAL (autograd wiring, `Saved` lifetime), not numeric-parity
/// oracles (that is `flash_torch_parity.rs`), so an exact reference value
/// is not needed — only that each call's tensor VALUES differ (so the
/// interleaving test's two batches are numerically distinguishable, and a
/// non-trivial gradient is a meaningful signal). `Tensor::randn` has no
/// per-call seed parameter reachable here without a `Device`-level RNG
/// handle, so a `seed_offset`-scaled affine on top of a fresh draw is the
/// simplest way to vary values across calls deterministically enough for
/// these assertions.
fn random_qkv(dev: &Device, total_q: usize, seed_offset: u64) -> Tensor {
    let base = Tensor::randn(0f32, 1.0, (total_q, 3, NUM_HEADS, HEAD_DIM), dev).unwrap();
    (base * (1.0 + 0.01 * seed_offset as f64))
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn cfg() -> VarlenConfig {
    VarlenConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        window: None,
        deterministic: true,
    }
}

/// O0(d): bwd lattice. `FlashVarlenAttention` is a `CustomOp1` — there is
/// exactly ONE slot (`qkv`), never a "provably constant leaf" (RoPE'd
/// Q/K/V always need a gradient in training) — so the lattice has one
/// cell: `Some(dqkv)`, always. Verified via a real `.backward()` on the
/// real `Var`-rooted graph (not `bwd()` called directly — the sanctioned
/// entry is `.backward()`).
#[test]
fn bwd_always_returns_some_dqkv_for_the_single_qkv_slot() {
    let Some(dev) = cuda_device() else { return };
    let device = Device::Cuda(dev.clone());
    let lengths = [64usize];
    let cu = CuSeqlens::from_lengths(&lengths, &dev).unwrap();

    let qkv_var = candle_core::Var::from_tensor(&random_qkv(&device, 64, 0)).unwrap();
    let o = flash_attention_varlen(qkv_var.as_tensor(), &cu, &cfg()).unwrap();
    let loss = o.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let d_qkv = grads.get(qkv_var.as_tensor());
    assert!(
        d_qkv.is_some(),
        "the single qkv slot must always receive Some(dqkv) — it is never a provably constant \
         leaf in this op"
    );
    let d_qkv = d_qkv.unwrap();
    assert_eq!(d_qkv.dims(), qkv_var.as_tensor().dims());
    assert_eq!(d_qkv.dtype(), qkv_var.as_tensor().dtype());
}

/// O0(c), GREEN half: two `flash_attention_varlen` calls on distinct
/// batches, losses concatenated, ONE `.backward()` — each call's own
/// `bwd` must read its OWN `lse` (a fresh op instance per call, per
/// `apply_stateful1`'s own contract). If the two calls somehow shared
/// state, this would either panic (a `SavedError`, surfaced as a candle
/// `Error::Msg` from `bwd`, which `.backward()` propagates as an `Err`) or
/// silently produce a wrong gradient — this test's positive form is simply
/// that it SUCCEEDS and produces two independently-finite, non-trivial
/// gradients.
#[test]
fn interleaved_calls_on_distinct_batches_each_read_their_own_lse() {
    let Some(dev) = cuda_device() else { return };
    let device = Device::Cuda(dev.clone());
    let cu_a = CuSeqlens::from_lengths(&[64usize], &dev).unwrap();
    let cu_b = CuSeqlens::from_lengths(&[48usize, 16], &dev).unwrap();

    let qkv_a = candle_core::Var::from_tensor(&random_qkv(&device, 64, 1)).unwrap();
    let qkv_b = candle_core::Var::from_tensor(&random_qkv(&device, 64, 2)).unwrap();

    // Two independent FORWARD calls before either backward — exactly the
    // shape a hoisted/shared op instance would corrupt.
    let o_a = flash_attention_varlen(qkv_a.as_tensor(), &cu_a, &cfg()).unwrap();
    let o_b = flash_attention_varlen(qkv_b.as_tensor(), &cu_b, &cfg()).unwrap();

    let loss = (o_a.sum_all().unwrap() + o_b.sum_all().unwrap()).unwrap();
    let grads = loss.backward().unwrap();

    let d_a = grads
        .get(qkv_a.as_tensor())
        .expect("qkv_a must have a gradient");
    let d_b = grads
        .get(qkv_b.as_tensor())
        .expect("qkv_b must have a gradient");

    for (name, t) in [("d_a", d_a), ("d_b", d_b)] {
        let v = t
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{name}: non-finite gradient"
        );
        assert!(
            v.iter().any(|&x| x != 0.0),
            "{name}: gradient is all-zero — the interleaving likely corrupted state"
        );
    }
}

/// O0(c), GradCache-shaped GREEN: forward, drop WITHOUT backward (pass 1,
/// detached), forward again, backward (pass 2) — `crates/jammi-ai/src/
/// fine_tune/gradcache.rs:78-81`'s exact sequence, through the real `pub
/// fn`. The FIRST forward's `Saved` slot is left `Some` when its op
/// instance is dropped (never taken) — `saved::tests::
/// set_without_take_drops_cleanly_forward_only_gradcache_shape` proves
/// that alone drops cleanly; this test proves the SECOND, independent
/// forward+backward on a FRESH instance still succeeds afterward (the
/// first call's abandoned state cannot leak into or block the second).
#[test]
fn gradcache_detached_pass_one_then_a_real_forward_backward_is_green() {
    let Some(dev) = cuda_device() else { return };
    let device = Device::Cuda(dev.clone());
    let cu = CuSeqlens::from_lengths(&[64usize], &dev).unwrap();

    // Pass 1: forward only, dropped without `.backward()`.
    {
        let qkv_detached = random_qkv(&device, 64, 3);
        let o = flash_attention_varlen(&qkv_detached, &cu, &cfg()).unwrap();
        drop(o); // no backward — the op instance's Saved<lse> is abandoned Some.
    }

    // Pass 2: a FRESH forward + a real backward must still succeed.
    let qkv_var = candle_core::Var::from_tensor(&random_qkv(&device, 64, 4)).unwrap();
    let o2 = flash_attention_varlen(qkv_var.as_tensor(), &cu, &cfg()).unwrap();
    let loss = o2.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let d_qkv = grads
        .get(qkv_var.as_tensor())
        .expect("pass 2 must produce a gradient after pass 1's abandoned state");
    assert!(
        d_qkv
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x: &f32| x.is_finite()),
        "pass 2 gradient must be finite"
    );
}

/// O0(b): poison-then-verify on `softmax_d` — always live (read by every
/// backward regardless of `deterministic`, `flash/mod.rs:822`'s
/// uninitialised-inter-tile-padding-rows doc). Fills the RAW scratch
/// buffer with NaN before the backward launch (via `crate::flash`'s own
/// `BwdScratch`/`flash_varlen_bwd_into` at the FFI-boundary layer — the op
/// layer allocates its own scratch internally and does not expose a hook
/// to poison it, so this drives the lower-level API directly, matching
/// `flash_torch_parity.rs`'s own choice of layer for FFI-level concerns).
/// If any pad row's uninitialised NaN were read, the launch's own outputs
/// (`dq`/`dk`/`dv`) would go non-finite (`NaN` propagates through every
/// arithmetic op that touches it) — this test's PASS means every output
/// bit was independent of the poison, i.e. those rows are truly never read.
#[test]
fn poison_softmax_d_before_backward_does_not_change_any_output_bit() {
    use half::bf16;
    use jammi_kernels::flash::{
        flash_varlen_bwd_into, flash_varlen_fwd, BwdBuffers, BwdScratch, HEAD_DIM as FD,
    };

    let Some(dev) = cuda_device() else { return };
    let lengths = [5usize, 137, 260, 128, 129]; // multi-tile, matches flash_smoke.rs's LARGE_LENS
    let cu = CuSeqlens::from_lengths(&lengths, &dev).unwrap();
    let total_q: usize = lengths.iter().sum();
    let cfg = cfg();

    let host_qkv: Vec<bf16> = (0..total_q * 3 * NUM_HEADS * FD)
        .map(|i| bf16::from_f32(((i % 13) as f32 - 6.0) * 0.05))
        .collect();
    let qkv = dev.htod_copy(host_qkv).unwrap();
    let (o, lse) = flash_varlen_fwd(&dev, &qkv, &cu, NUM_HEADS, &cfg).unwrap();
    let host_do: Vec<bf16> = (0..total_q * NUM_HEADS * FD)
        .map(|i| bf16::from_f32(((i % 7) as f32 - 3.0) * 0.03))
        .collect();
    let d_o = dev.htod_copy(host_do).unwrap();

    let geom = cu.geometry(NUM_HEADS).unwrap();

    // Run 1: clean scratch (BwdScratch::alloc's own uninitialised alloc).
    let mut scratch1 = BwdScratch::alloc(&dev, &geom, cfg.deterministic).unwrap();
    let mut d_qkv1 = unsafe { dev.alloc::<bf16>(geom.qkv_len()) }.unwrap();
    flash_varlen_bwd_into(
        &dev,
        &cu,
        NUM_HEADS,
        BwdBuffers {
            qkv: qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: d_o.as_view(),
            d_qkv: d_qkv1.as_view_mut(),
            softmax_d: scratch1.softmax_d.as_view_mut(),
            dq_accum: scratch1.dq_accum.as_view_mut(),
            dq_accum_splits: scratch1.splits,
        },
        &cfg,
    )
    .unwrap();

    // Run 2: `softmax_d` poisoned with NaN before the launch.
    let mut scratch2 = BwdScratch::alloc(&dev, &geom, cfg.deterministic).unwrap();
    let nan_host = vec![f32::NAN; geom.softmax_d_len()];
    scratch2.softmax_d = dev.htod_copy(nan_host).unwrap();
    let mut d_qkv2 = unsafe { dev.alloc::<bf16>(geom.qkv_len()) }.unwrap();
    flash_varlen_bwd_into(
        &dev,
        &cu,
        NUM_HEADS,
        BwdBuffers {
            qkv: qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: d_o.as_view(),
            d_qkv: d_qkv2.as_view_mut(),
            softmax_d: scratch2.softmax_d.as_view_mut(),
            dq_accum: scratch2.dq_accum.as_view_mut(),
            dq_accum_splits: scratch2.splits,
        },
        &cfg,
    )
    .unwrap();

    let h1: Vec<bf16> = dev.dtoh_sync_copy(&d_qkv1).unwrap();
    let h2: Vec<bf16> = dev.dtoh_sync_copy(&d_qkv2).unwrap();
    assert_eq!(h1.len(), h2.len());
    assert!(
        h1.iter()
            .zip(h2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "poisoning softmax_d's scratch buffer changed at least one output bit of dq/dk/dv — a \
         pad row that should never be read WAS read"
    );
    assert!(
        h2.iter().all(|x| x.is_finite()),
        "poisoned-run output contains a non-finite value — NaN leaked from the poisoned scratch"
    );
}

/// O0(b), labelled dead-path guard (per the lead's v5 correction): the
/// NON-deterministic `dq_accum` scratch's uninitialised inter-tile rows
/// are documented as never-read too (`flash/mod.rs:829`), but
/// `ops::flash_attention_varlen`'s own `cfg.deterministic` is pinned
/// `true` by every real call site (Stage B2) — this cell is NEVER reached
/// through the op's public surface. Kept as defense-in-depth on the
/// LOWER-level `crate::flash` primitive itself (which a future,
/// non-deterministic caller COULD reach), not as coverage of anything
/// `flash_attention_varlen` itself exercises.
#[test]
fn poison_non_deterministic_dq_accum_is_a_dead_path_guard_not_reachable_via_the_op() {
    use half::bf16;
    use jammi_kernels::flash::{
        flash_varlen_bwd_into, flash_varlen_fwd, BwdBuffers, BwdScratch, HEAD_DIM as FD,
    };

    let Some(dev) = cuda_device() else { return };
    let lengths = [5usize, 137, 260, 128, 129];
    let cu = CuSeqlens::from_lengths(&lengths, &dev).unwrap();
    let total_q: usize = lengths.iter().sum();
    let cfg_nondet = VarlenConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        window: None,
        deterministic: false, // NEVER what the op passes — see this test's own doc.
    };

    let host_qkv: Vec<bf16> = (0..total_q * 3 * NUM_HEADS * FD)
        .map(|i| bf16::from_f32(((i % 11) as f32 - 5.0) * 0.04))
        .collect();
    let qkv = dev.htod_copy(host_qkv).unwrap();
    let (o, lse) = flash_varlen_fwd(&dev, &qkv, &cu, NUM_HEADS, &cfg_nondet).unwrap();
    let host_do: Vec<bf16> = (0..total_q * NUM_HEADS * FD)
        .map(|i| bf16::from_f32(((i % 9) as f32 - 4.0) * 0.02))
        .collect();
    let d_o = dev.htod_copy(host_do).unwrap();
    let geom = cu.geometry(NUM_HEADS).unwrap();

    let mut scratch1 = BwdScratch::alloc(&dev, &geom, cfg_nondet.deterministic).unwrap();
    let mut d_qkv1 = unsafe { dev.alloc::<bf16>(geom.qkv_len()) }.unwrap();
    flash_varlen_bwd_into(
        &dev,
        &cu,
        NUM_HEADS,
        BwdBuffers {
            qkv: qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: d_o.as_view(),
            d_qkv: d_qkv1.as_view_mut(),
            softmax_d: scratch1.softmax_d.as_view_mut(),
            dq_accum: scratch1.dq_accum.as_view_mut(),
            dq_accum_splits: scratch1.splits,
        },
        &cfg_nondet,
    )
    .unwrap();

    let mut scratch2 = BwdScratch::alloc(&dev, &geom, cfg_nondet.deterministic).unwrap();
    let nan_host = vec![f32::NAN; geom.dq_accum_len(scratch2.splits)];
    scratch2.dq_accum = dev.htod_copy(nan_host).unwrap();
    let mut d_qkv2 = unsafe { dev.alloc::<bf16>(geom.qkv_len()) }.unwrap();
    flash_varlen_bwd_into(
        &dev,
        &cu,
        NUM_HEADS,
        BwdBuffers {
            qkv: qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: d_o.as_view(),
            d_qkv: d_qkv2.as_view_mut(),
            softmax_d: scratch2.softmax_d.as_view_mut(),
            dq_accum: scratch2.dq_accum.as_view_mut(),
            dq_accum_splits: scratch2.splits,
        },
        &cfg_nondet,
    )
    .unwrap();

    let h1: Vec<bf16> = dev.dtoh_sync_copy(&d_qkv1).unwrap();
    let h2: Vec<bf16> = dev.dtoh_sync_copy(&d_qkv2).unwrap();
    assert!(
        h1.iter()
            .zip(h2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "poisoning the non-deterministic dq_accum's inter-tile padding changed an output bit"
    );
}
