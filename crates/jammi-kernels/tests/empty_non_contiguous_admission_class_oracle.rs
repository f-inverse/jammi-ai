//! Class oracle: across this crate's `crate::cuda::*` glue, the arm's own
//! `contiguous_offsets()` layout check runs BEFORE any zero-element fast
//! path (`if hidden == 0 || n == 0 { .. }` and its siblings). In the other
//! order a zero-element NON-CONTIGUOUS layout (e.g. a `(0, 3)` tensor
//! transposed to `(3, 0)`: `candle_core::Shape::is_contiguous` resets its
//! running stride accumulator to `0` at a zero-sized dim, so a LATER dim
//! with a real stride correctly reads as non-contiguous even though the
//! tensor holds no elements — reproduced directly below) is silently
//! ADMITTED on CUDA while `cpu_fwd` refuses it outright.
//! `ops::dropout::DropoutFused::metal_fwd` has the identical shape; every
//! `crate::cuda::*` site checks contiguity FIRST, matching each op's own
//! `cpu_fwd` domain exactly (see each site's own comment in
//! `crate::cuda::{dropout,layer_norm,softmax,rope,rope_positions,
//! attention_block}` for the specific split — an axis that `cpu_fwd`
//! itself EXEMPTS from contiguity, e.g. `LayerNormFused`'s `hidden == 0`,
//! is exempt here too; only the broader `n == 0` sub-case is not).
//!
//! This file is ONE shared class oracle, not one copy per op. [`assert_class_refuses_empty_non_contiguous_admission`]
//! is the single assertion helper, run against `Device::Cpu`
//! unconditionally (so a plain `cargo test -p jammi-kernels` — no `cuda`
//! feature — still proves the CPU side, which is every one of these
//! representative ops' actual documented domain) and, under the `cuda`
//! `live-gpu-tests` feature, against a real CUDA device too — the SAME
//! fixtures, the SAME assertions, one code path for both devices.
//!
//! ## Representative op set — chosen for domain-SHAPE coverage, not for size
//!
//! Three distinct `cpu_fwd` domain shapes exist across the class (module
//! docs of each op cited below); one representative per shape is enough to
//! prove the CUDA glue's admission matches ALL of them:
//!
//! 1. **No empty fast path at all** — `ops::dropout::DropoutFused::cpu_fwd`
//!    calls `contiguous_offsets()` UNCONDITIONALLY; an empty tensor is a
//!    no-op only if it is ALSO contiguous. [`DropoutFused`] is this leg.
//! 2. **A single axis exempts contiguity, not `n == 0` generally** —
//!    `ops::layer_norm::LayerNormFused::cpu_fwd` (and `ops::softmax`'s
//!    `SoftmaxLastDimFused`/`ops::rope`'s `RopeFused`/`ops::rope_positions`'s
//!    `RopePositionsFused`, the same shape) skip contiguity ONLY when
//!    their own reduction axis (`hidden`/`last`/`hidden`/`d`) is zero —
//!    NOT for a `rows == 0`-but-that-axis-nonzero empty-non-contiguous
//!    layout, which still hits `contiguous_offsets()`. [`LayerNormFused`]
//!    and [`SoftmaxLastDimFused`] are this leg (two, since each fused
//!    op's CUDA glue decides its axis check independently — worth pinning
//!    on more than one sibling).
//! 3. **No empty fast path at ALL, for a THIRD reason (a composed op)** —
//!    `ops::attention_block::AttentionBlockFused::cpu_fwd`'s own comment:
//!    "No empty fast path on this arm" — its mask/dtype/contiguity checks
//!    run even when `b`/`s`/`h` is 0; only the actual GEMM/gather compute
//!    is skipped for an empty `qkv`. [`AttentionBlockFused`] is this leg.
//!
//! `crate::cuda::{scaled_cast_add,cast_scale,adamw_step}` (the SAME
//! ordering, `crate::cuda`'s own module doc) are a DIFFERENT shape not
//! exercised by THIS file: those ops' `cpu_fwd` arms
//! never call `contiguous_offsets()` at all (they walk `StridedOffsets`,
//! tolerating any stride), so there is no "same refusal on CPU" to prove
//! for them — the property there is the CUDA arm's OWN internal
//! self-consistency (an empty tensor is refused the same way a non-empty
//! one is, matching that arm's own documented domain), not a CPU/CUDA
//! parity fact this file's shared-assertion shape can state.
//! `ops::geglu::GegluFused`'s CUDA glue was read against the SAME method
//! and is aligned (its `intermediate == 0` fast path is the
//! IDENTICAL condition `cpu_fwd` itself gates on, in the same relative
//! position) — no divergence, so no fixture is needed there either.

use candle_core::{DType, Device, Error, Tensor, Var};
use jammi_kernels::ops::{
    apply2, apply3, AttentionBlockFused, DropoutFused, FullyMaskedPolicy, LayerNormFused,
    SoftmaxLastDimFused,
};

/// Runs every representative leg (module doc above) on `device`, each with
/// a hand-built EMPTY, NON-CONTIGUOUS admission-edge input, asserting the
/// SAME typed `Error::RequiresContiguous` refusal every leg's own
/// `cpu_fwd` already gives — panicking (via `expect`/`assert!`, not a
/// silently-continuing bool return) naming exactly which leg and which
/// wrong outcome on any regression, so a future partial revert of one
/// site is caught precisely rather than folded into one opaque failure.
fn assert_class_refuses_empty_non_contiguous_admission(device: &Device) {
    // Leg 1 (shape 1 — no empty fast path at all): `DropoutFused`. A
    // `(0, 5)` tensor transposed to `(5, 0)`: zero elements, genuinely
    // non-contiguous (candle's own `Shape::is_contiguous` — see this
    // file's module doc).
    {
        let x = Tensor::zeros((0usize, 5usize), DType::F32, device)
            .unwrap()
            .t()
            .unwrap();
        assert_eq!(x.dims(), &[5, 0], "dropout: fixture shape");
        assert_eq!(x.elem_count(), 0, "dropout: fixture must be empty");
        assert!(
            !x.is_contiguous(),
            "dropout: fixture must be non-contiguous"
        );
        let op = DropoutFused::new(1, 0, 0, 0.3).unwrap();
        let err = jammi_kernels::ops::apply1(&x, op)
            .expect_err("dropout: an empty, non-contiguous input must be refused");
        assert!(
            matches!(err, Error::RequiresContiguous { .. }),
            "dropout: expected RequiresContiguous, got {err:?}"
        );
    }

    // Leg 2a (shape 2 — axis-exempt, `hidden`): `LayerNormFused`. `x`'s
    // OWN reduction axis (last dim, `hidden`) stays a NONZERO `5`; the
    // ZERO axis sits earlier (`rows == 0` via a DIFFERENT dim), pinning
    // the exact split between `hidden == 0` (`cpu_fwd` exempts contiguity
    // there) and this `n == 0`-with-`hidden != 0` case (`cpu_fwd` does NOT
    // exempt it).
    {
        let x = Tensor::zeros((0usize, 3usize, 5usize), DType::F32, device)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert_eq!(x.dims(), &[3, 0, 5], "layer_norm: fixture shape");
        assert_eq!(x.elem_count(), 0, "layer_norm: fixture must be empty");
        assert!(
            !x.is_contiguous(),
            "layer_norm: fixture must be non-contiguous"
        );
        let gamma = Tensor::zeros((5usize,), DType::F32, device).unwrap();
        let err = apply2(&x, &gamma, LayerNormFused::new(1e-5, false)).expect_err(
            "layer_norm: an empty, non-contiguous input with hidden != 0 must be refused",
        );
        assert!(
            matches!(err, Error::RequiresContiguous { .. }),
            "layer_norm: expected RequiresContiguous, got {err:?}"
        );
    }

    // Leg 2b (shape 2 — axis-exempt, `last`): `SoftmaxLastDimFused`. The
    // identical construction as leg 2a, over `scores`/`mask` instead of
    // `x`/`gamma` — a SIBLING fused op whose CUDA glue decides its
    // `last == 0` exemption independently, so it is pinned separately
    // rather than assumed to share `LayerNormFused`'s behavior.
    {
        let scores = Tensor::zeros((0usize, 3usize, 5usize), DType::F32, device)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert_eq!(scores.dims(), &[3, 0, 5], "softmax: fixture shape");
        assert_eq!(scores.elem_count(), 0, "softmax: fixture must be empty");
        assert!(
            !scores.is_contiguous(),
            "softmax: fixture must be non-contiguous"
        );
        let mask = Tensor::zeros((1usize, 1usize, 5usize), DType::F32, device).unwrap();
        let op = SoftmaxLastDimFused::new(FullyMaskedPolicy::Propagate);
        let err = apply2(&scores, &mask, op)
            .expect_err("softmax: an empty, non-contiguous input with last != 0 must be refused");
        assert!(
            matches!(err, Error::RequiresContiguous { .. }),
            "softmax: expected RequiresContiguous, got {err:?}"
        );
    }

    // Leg 3 (shape 3 — composed op, no empty fast path at all):
    // `AttentionBlockFused`. `qkv`'s OWN `seq` axis is empty (`s == 0`,
    // one of the three axes of the CUDA arm's `b == 0 || s == 0 || h == 0`
    // fast path, which runs AFTER contiguity); `batch`/`heads` stay
    // nonzero. `rope: false`, so `rope_pack`'s own shape/dtype is
    // irrelevant (never read on this path — module doc); `mask` is a
    // small, VALID, contiguous fixture (its own domain is not what this
    // leg tests) shaped to `check_mask`'s `[batch|1, 1, seq|1, seq]` rule
    // with `seq == 0`.
    {
        let qkv = Tensor::zeros(
            (0usize, 2usize, 3usize, 1usize, 64usize),
            DType::F32,
            device,
        )
        .unwrap()
        .transpose(0, 1)
        .unwrap();
        assert_eq!(
            qkv.dims(),
            &[2, 0, 3, 1, 64],
            "attention_block: fixture shape"
        );
        assert_eq!(
            qkv.elem_count(),
            0,
            "attention_block: fixture must be empty"
        );
        assert!(
            !qkv.is_contiguous(),
            "attention_block: fixture must be non-contiguous"
        );
        let rope_pack = Tensor::zeros((1usize,), DType::F32, device).unwrap();
        let mask = Tensor::zeros((1usize, 1usize, 1usize, 0usize), DType::F32, device).unwrap();
        let op = AttentionBlockFused::new(1.0, FullyMaskedPolicy::Propagate, false).unwrap();
        let err = apply3(&qkv, &rope_pack, &mask, op)
            .expect_err("attention_block: an empty, non-contiguous qkv must be refused");
        assert!(
            matches!(err, Error::RequiresContiguous { .. }),
            "attention_block: expected RequiresContiguous, got {err:?}"
        );
    }
}

/// The CPU leg — unconditional (no `cuda` feature required), so this is
/// what `cargo test -p jammi-kernels`'s ordinary run actually proves: the
/// class oracle's fixtures ARE genuinely refused, matching each op's own
/// documented `cpu_fwd` domain.
#[test]
fn cpu_refuses_empty_non_contiguous_admission_across_representative_ops() {
    assert_class_refuses_empty_non_contiguous_admission(&Device::Cpu);
}

/// A DIFFERENT class from the admission order above (shape, not cause):
/// `crate::cuda::layer_norm::cuda_bwd_dgamma`'s `rows == 0, hidden != 0`
/// fast path must return a `[hidden]`-shaped, all-zero buffer, as
/// `cpu_fwd`'s OWN `rows == 0` path (`ln_bwd_dgamma_f32`'s
/// `vec![0f32; hidden]`) does — not `alloc_empty`'s `[0]`-shaped buffer, a
/// WRONG-SHAPED `dgamma` for a zero-row batch. Pinned via the only public surface that reaches the
/// (private) `LayerNormBwdDgamma` op: `LayerNormFused` with
/// `dgamma_needed: true`, differentiated through `Tensor::backward()` —
/// matching `ops::layer_norm::tests::bwd_dgamma_zero_rows_hidden_nonzero_
/// is_hidden_shaped_all_zero_not_zero_length`'s CPU fixture exactly, one
/// device parameter apart.
fn assert_dgamma_zero_rows_hidden_nonzero_is_hidden_shaped_all_zero(device: &Device) {
    let x = Tensor::zeros((0usize, 4usize), DType::F32, device).unwrap();
    let gamma = Var::from_tensor(&Tensor::zeros((4usize,), DType::F32, device).unwrap()).unwrap();
    let out = apply2(&x, gamma.as_tensor(), LayerNormFused::new(1e-5, true)).unwrap();
    assert_eq!(
        out.dims(),
        &[0, 4],
        "dgamma_class_oracle: fwd output shape sanity check"
    );

    let grads = out
        .sum_all()
        .unwrap()
        .backward()
        .expect("dgamma_class_oracle: backward through a zero-row batch must not error");
    let dgamma = grads
        .get(gamma.as_tensor())
        .expect("dgamma_class_oracle: gamma must have a recorded gradient");
    assert_eq!(
        dgamma.dims(),
        &[4],
        "dgamma_class_oracle: rows == 0, hidden != 0 dgamma must be [hidden]-shaped, \
         not [0]-shaped"
    );
    let values: Vec<f32> = dgamma.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        values.iter().all(|&v| v == 0.0),
        "dgamma_class_oracle: summing zero rows must give exactly zero at every \
         position, got {values:?}"
    );
}

/// The CPU leg — unconditional, proving the fixture itself (and
/// `LayerNormFused`'s CPU path) behaves as this file's module doc for the
/// dgamma-shape class describes, independent of CUDA availability.
#[test]
fn cpu_dgamma_zero_rows_hidden_nonzero_is_hidden_shaped_all_zero() {
    assert_dgamma_zero_rows_hidden_nonzero_is_hidden_shaped_all_zero(&Device::Cpu);
}

#[cfg(feature = "live-gpu-tests")]
mod gpu {
    use super::*;

    /// The CUDA leg — compiled only under `live-gpu-tests`: proves
    /// `crate::cuda::layer_norm::cuda_bwd_dgamma`'s `rows == 0, hidden != 0`
    /// fast path returns a `[hidden]`-shaped, all-zero `dgamma`, matching
    /// `cpu_fwd` exactly.
    #[test]
    fn cuda_dgamma_zero_rows_hidden_nonzero_is_hidden_shaped_all_zero() {
        let device = jammi_test_resources::cuda_device(0);
        assert_dgamma_zero_rows_hidden_nonzero_is_hidden_shaped_all_zero(&device);
    }

    /// The CUDA leg — compiled only under `live-gpu-tests`: proves
    /// `crate::cuda::{dropout,layer_norm,softmax,attention_block}`'s glue
    /// refuses the SAME fixtures the CPU leg above does, rather than silently
    /// admitting them through a zero-element fast path.
    #[test]
    fn cuda_refuses_empty_non_contiguous_admission_across_representative_ops() {
        let device = jammi_test_resources::cuda_device(0);
        assert_class_refuses_empty_non_contiguous_admission(&device);
    }
}
