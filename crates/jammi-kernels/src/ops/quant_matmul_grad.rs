//! `QuantMatMulGrad`: a `CustomOp1` wrapping an `Arc<candle_core::quantized::
//! QTensor>` that makes GGUF-quantized weight matmul ALWAYS-DIFFERENTIABLE
//! — the bitsandbytes `MatMul4Bit` contract (gradient wrt the ACTIVATION
//! only; a quantized weight is frozen, never trained through directly).
//!
//! ## Why wrap `QTensor` at all, rather than call it directly
//!
//! `candle_core::quantized::QTensor` already implements `CustomOp1` itself
//! (`candle-core` 0.11.0 `src/quantized/mod.rs:899-1018`, `name() ==
//! "qmatmul"`) — its `cpu_fwd`/`cuda_fwd`/`metal_fwd` ARE the fast quantized
//! matmul kernels (AVX2/NEON dot-product paths on CPU, cuBLAS-adjacent
//! MMVQ/MMQ on CUDA, a dedicated Metal kernel). But `CustomOp1`'s own
//! DEFAULT `bwd` (`custom_op.rs:38-40`) returns `Err(BackwardNotSupported)`
//! — `QTensor` itself never overrides it. Reaching `QTensor`'s `cpu_fwd`
//! through candle's own `Tensor::apply_op1_no_bwd`
//! (`custom_op.rs:159-176`) — the natural "I know there's no bwd, skip it"
//! entry point — is FORBIDDEN EVERYWHERE in this crate (see
//! `tests/stateful_op_discipline.rs`'s `FORBIDDEN_EVERYWHERE` list): that
//! entry point builds an op node with `BackpropOp::none()` UNCONDITIONALLY,
//! so a caller who forgets it is inside a `no_bwd` call gets a live forward
//! value and a SILENTLY MISSING gradient rather than a loud
//! `BackwardNotSupported` — exactly the esc-037 hazard class (a no-bwd op
//! reachable under grad, silently). This op exists SPECIFICALLY so nothing
//! in this workspace ever needs that entry point for a quantized weight:
//! `bwd` below is a REAL implementation (module doc section below), so the
//! ordinary, always-tracked `apply_op1`-style path (via
//! [`super::apply_stateful1`], not `apply_op1_no_bwd`) is correct and safe
//! to use unconditionally.
//!
//! ## Entry point: `apply_stateful1`, not `apply1`
//!
//! `QuantMatMulGrad` holds `Arc<QTensor>` — `Arc` is not `Copy`, so this
//! type cannot implement [`super::KernelOp`] (which requires `Copy` — see
//! `ops/mod.rs`'s own doc for what that bound proves) and therefore cannot
//! run through [`super::apply1`]. It runs through
//! [`super::apply_stateful1`] instead: `StatefulKernelOp`'s bound (`Send +
//! Sync + 'static + Sealed`, no `Copy`/`Clone` requirement) is satisfied
//! trivially, the same way every non-`Saved`-bearing `Sealed` type in this
//! crate satisfies it (see `MemEfficientAttention`'s own module doc, "This
//! is NOT mutual exclusion") — `QuantMatMulGrad` carries no [`super::Saved`]
//! field, but it is `!Copy` for a DIFFERENT reason (the `Arc`), which is
//! equally sufficient to rule out [`super::KernelOp`].
//!
//! ## The public helper is unconditionally differentiable — no train/eval branch
//!
//! [`quant_matmul_grad`] always calls `x.contiguous()?` then
//! [`super::apply_stateful1`]. `Tensor::apply_op1` builds its returned
//! tensor's op via `candle_core::op::BackpropOp::new1` (`op.rs:1100-1107`),
//! which itself self-prunes: `if arg.track_op() { Some(..) } else { None }`
//! — an UNTRACKED input (eval, no `Var` upstream) produces an output with
//! no recorded op at all (`Tensor::track_op()` returns `false` on it, same
//! as any ordinary candle op), at ZERO extra branching in this file. A
//! TRACKED input produces a real graph node whose `bwd` this op supplies.
//! One code path serves both regimes; adding an eval/train `if` here would
//! only reintroduce, by hand, exactly the pruning candle's own
//! `BackpropOp::new1` already performs (see the module-level
//! `eval_path_output_carries_no_grad_op_and_matches_tracked_value` test,
//! which proves the self-pruning AND the bit-identical value on the SAME
//! code path).
//!
//! ## `bwd`: the bitsandbytes `MatMul4Bit` contract (grad wrt input ONLY)
//!
//! Forward computes `y = x @ Wᵀ` (`W`: `[out, in]`, `QTensor`'s own
//! `cpu_fwd` convention — "self is transposed so n is first then k",
//! `src/quantized/mod.rs:913`). `bwd` therefore computes `dx = dy @ W`
//! (`dy`: `[.., out]`, `W`: `[out, in]` — no transpose needed, since `W` is
//! ALREADY stored `[out, in]` and `dx = dy @ W` is exactly `[.., out] @
//! [out, in] -> [.., in]`). There is no gradient wrt `W` at all: `W` is not
//! a differentiable `Tensor` argument to this op (it is `Arc`-held
//! construction data, matching how a frozen/quantized base weight is
//! supposed to behave — LoRA trains only the low-rank adapters, never the
//! frozen base, quantized or not) — `CustomOp1::bwd`'s signature has
//! exactly one argument slot to fill (`arg`), and this op fills it, always.
//!
//! `bwd` NEVER returns `Ok(None)` for that one gradient slot: candle's own
//! `Tensor::backward()` (`backprop.rs:663`, cited by this workspace's
//! esc-037 finding) drops a `None` gradient SILENTLY rather than erroring —
//! so a `CustomOp1::bwd` that decided "no meaningful gradient here" and
//! returned `Ok(None)` would look, from the caller's side, EXACTLY like a
//! correctly-computed all-zero gradient right up until a real training run
//! quietly stops learning through this op. `bwd` below always computes and
//! returns `Ok(Some(..))` — see
//! `bwd_never_returns_none_for_the_input_gradient` for the literal
//! assertion.
//!
//! Steps, in order (module doc's own summary, mirrored exactly in the
//! implementation below — no step reordered, no step skipped):
//! 1. Dequantize `W` via `QTensor::dequantize(&device)` (`src/quantized/
//!    mod.rs:689`) — ALWAYS returns `f32`, regardless of `W`'s own
//!    quantization format (`src/quantized/mod.rs:497`, every `QuantizedType::
//!    dequantize` impl returns `CpuStorage::F32`). Dequantize-and-DISCARD:
//!    nothing here caches the dequantized `W` across calls — a fresh `f32`
//!    tensor is materialized and dropped every `bwd`, trading recompute for
//!    zero persistent state (matching this op's own `!Copy`/no-`Saved`
//!    status: there is nothing here FOR a cache to live in even if one were
//!    wanted).
//! 2. Reshape `dy` (`grad_res`) to rank 2, `[m, out]`, where `m` is the
//!    product of every leading dimension (`dy` may be rank 2 `[batch, out]`
//!    OR rank 3 `[batch, seq, out]` — both collapse to the same `[m, out]`
//!    shape this step produces) and cast to `f32` (matching `W`'s
//!    dequantized dtype — candle's CPU backend has no non-`f32` `matmul`
//!    for this class of op, the same pre-existing limitation this crate's
//!    other attention ops already document).
//! 3. ONE rank-2 matmul: `dy2 [m, out] @ W_deq [out, in] -> [m, in]`. NEVER
//!    `Tensor::broadcast_matmul`: its `(false, true)` broadcast arm
//!    materializes one FULL COPY of `W_deq` per batch element it broadcasts
//!    over — for a `[batch, seq, out]`-shaped `dy` that is `batch` redundant
//!    copies of an already-dequantized `[out, in]` tensor, a real, avoidable
//!    memory blowup this op's whole design (recompute, don't retain) exists
//!    to avoid. Flattening to rank 2 first makes the GEMM a single,
//!    unbroadcast call.
//! 4. Reshape the `[m, in]` result back to `dy`'s own leading dimensions
//!    with the last axis replaced by `in` (produces exactly `arg`'s own
//!    shape, `[batch, in]` or `[batch, seq, in]`, matching a gradient's
//!    required shape-equals-input-shape contract) and cast to `dy`'s
//!    ORIGINAL dtype (before step 2's `f32` upcast) — the one round-back
//!    point, mirroring every other fused op in this crate's "one round
//!    point" rounding doctrine (see e.g. `ops::mem_efficient_attention`'s
//!    module doc).
//!
//! ## Domain (family D)
//!
//! `W` (`self.w`): rank 2, `[out, in]` — enforced by `QTensor::cpu_fwd`'s
//! own `self.shape.dims2()?` (an `Err`, not a silent reinterpretation, on
//! any other rank), so this op does not duplicate that check. `x` (`arg` at
//! the public helper): made contiguous by [`quant_matmul_grad`] before
//! dispatch (`QTensor::cpu_fwd` itself refuses a non-contiguous layout —
//! candle 0.11 does `crate::bail!("input tensor is not contiguous
//! {layout:?}")` (`quantized/mod.rs:909-911`), an untyped `Error::Msg`, not
//! a typed variant — module doc's forward-delegation section); rank `>= 2`
//! (`QTensor::cpu_fwd`'s own check: "input tensor has
//! only one dimension" is refused). `dy` (`grad_res`, `bwd`'s own input):
//! rank 2 or 3 in every production call shape this op targets (a rank-2 `x`
//! produces a rank-2 `y`/`dy`; a rank-3 `x`, rank-3) — `bwd`'s own reshape
//! step generalizes to ANY rank `>= 1` (`m` is simply the product of every
//! leading dim, `[]` for a rank-1 `dy` collapsing to `m == 1`), so no
//! separate rank-2-vs-rank-3 branch exists in the code; the module doc
//! states rank 2/3 because that is what this op's own callers ever
//! construct, not because the implementation itself is narrower.
//!
//! ## `dtype` (family D: CPU accepts `f32`/`f16` only; `jammi-lora`'s own
//! uniform rule keeps this op's dtype surface small)
//!
//! `QTensor::cpu_fwd` accepts `x` in `f32` or `f16` and refuses anything
//! else with `"Expected f32/f16"` (`src/quantized/mod.rs:991`) — this op
//! does not re-check that; `QTensor`'s own typed refusal is the domain
//! boundary. `jammi_lora::QuantizedLinear` (the sole production caller,
//! wave 3) additionally imposes a UNIFORM rule on top — cast `x` to `f32`
//! before calling [`quant_matmul_grad`], regardless of device — so in
//! practice this op only ever receives `f32` `x` in this workspace; the
//! wider `f16`-on-CPU acceptance is `QTensor`'s own, inherited, not
//! narrowed or widened here.
//!
//! ## `repacked_qs`: `QTensor`'s own interior-mutable state (argued, not denied)
//!
//! `QTensor` carries `repacked_qs: OnceLock<Option<Vec<u8>>>`
//! (`src/quantized/mod.rs:65`) — a lazily-built, ALTERNATE packing of the
//! SAME quantized bytes (`aarch64`+`dotprod`'s `BlockQ4Kx8` repack path,
//! `cpu_fwd`'s own `self.repacked_qs.get_or_init(..)` at
//! `src/quantized/mod.rs:944`), consulted only for `Q4K`-dtype, `n %
//! 8 == 0` CPU matmuls on that target. This is exactly the "interior-
//! mutable/Arc-carried field" `tests/stateful_op_discipline.rs`'s widened
//! scope (see that file's own doc) now sweeps this module into: `self.w`
//! is `Arc<QTensor>`, and `QTensor` itself (behind that `Arc`) owns a
//! `OnceLock`. Argued, not denied, safe for THREE independent reasons:
//! 1. **Content-derived**: the closure `get_or_init` runs reads only
//!    `self_storage`'s own already-quantized, immutable bytes (never `x`,
//!    `layout`, or anything call-specific) — every thread that ever races
//!    to initialize it computes the BYTE-IDENTICAL repacked buffer, so a
//!    race decides only which thread's (identical) computation "wins", not
//!    which VALUE wins.
//! 2. **Idempotent + write-once**: `OnceLock::get_or_init` itself guarantees
//!    the closure runs to completion for at most one caller (any concurrent
//!    caller BLOCKS on the in-progress initialization rather than racing a
//!    second write) and every subsequent call reads the same cached
//!    `Some(..)` — this is the standard library's own safety contract for
//!    `OnceLock`, not something this op adds.
//! 3. **Never affects this op's OWN output values**: whichever branch
//!    `cpu_fwd` takes (the repacked `BlockQ4Kx8` fast path or the
//!    unrepacked `matmul_t` fallback) computes the SAME mathematical
//!    quantized matmul — the repack is a LAYOUT optimization for
//!    `dotprod`-capable hardware, not an alternate numerical algorithm — so
//!    even a hypothetically-observable initialization-order difference
//!    across two runs could never change a `y`/`bwd` VALUE this op returns,
//!    only (invisibly) which run paid the one-time repack cost. Family J's
//!    "same inputs -> same values" determinism promise is about VALUES; it
//!    is preserved.
//!
//! `QuantMatMulGrad` itself derives neither `Clone` nor `Copy` (mirroring
//! every `Saved`-bearing op's own convention, `ops/mod.rs`'s
//! `StatefulKernelOp` doc "Clone is actively refused AT THEIR DEFINITION
//! SITE" section) — not because cloning an `Arc<QTensor>` would corrupt
//! anything (points 1-3 above show it would not), but because the crate-
//! wide discipline this op is now inside (`tests/stateful_op_discipline.rs`)
//! is "construct fresh, pass by value, never reuse an op instance across
//! calls", and following it uniformly costs nothing here (every real call
//! site already holds its own `Arc<QTensor>` clone to construct a fresh
//! [`QuantMatMulGrad`] from, via [`QuantMatMulGrad::new`]).
//!
//! Generic primitive (family L): this crate names no consumer. Module-doc
//! shapes/dtypes exist only to explain numeric choices.

use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, DType, Error, Layout, MetalStorage, Result, Shape, Tensor,
};

use super::apply_stateful1;

/// See the module doc. Constructed only through [`QuantMatMulGrad::new`].
pub struct QuantMatMulGrad {
    w: Arc<QTensor>,
}

impl QuantMatMulGrad {
    /// `w` must be rank 2, `[out, in]` — enforced by candle's own
    /// `QTensor::cpu_fwd`/`cuda_fwd`/`metal_fwd` at dispatch time (module
    /// doc's "Domain" section); not re-checked here so there is exactly one
    /// place (candle's own quantized matmul) this shape requirement is
    /// enforced.
    pub fn new(w: Arc<QTensor>) -> Self {
        Self { w }
    }
}

impl super::sealed::Sealed for QuantMatMulGrad {}

impl CustomOp1 for QuantMatMulGrad {
    fn name(&self) -> &'static str {
        "quant_matmul_grad"
    }

    /// Delegates directly to `QTensor::cpu_fwd` (module doc: `QTensor`
    /// itself implements `CustomOp1`; this is its own AVX2/NEON-accelerated
    /// quantized matmul kernel, reused rather than reimplemented).
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        self.w.cpu_fwd(storage, layout)
    }

    /// Delegates directly to `QTensor::cuda_fwd` (module doc).
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        self.w.cuda_fwd(storage, layout)
    }

    /// Delegates directly to `QTensor::metal_fwd` (module doc). `QTensor`'s
    /// own Metal arm `assert_eq!(storage.dtype(), DType::F32)`s internally
    /// (`src/quantized/metal.rs:390`) — a PANIC, not a typed `Result::Err`,
    /// on any other input dtype; this op does not add a typed guard in
    /// front of it (doing so would only turn one crate's panic into a
    /// different crate's error for the exact same caller mistake).
    /// `jammi_lora::QuantizedLinear`'s uniform "cast to f32 first" rule
    /// (module doc) is what keeps a non-`f32` input from ever reaching this
    /// arm in production.
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        self.w.metal_fwd(storage, layout)
    }

    /// The bitsandbytes `MatMul4Bit` contract: gradient wrt the input `x`
    /// only. See the module doc's numbered step list — this implementation
    /// mirrors it exactly, step for step.
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let op = self.name();
        let device = grad_res.device();

        // Step 1: dequantize W. Always f32 (module doc). Dequantize-and-
        // discard: `w_deq` is dropped at the end of this call, nothing
        // caches it.
        let w_deq = self.w.dequantize(device)?;
        let (out_features, in_features) = w_deq.dims2()?;

        // Step 2: flatten dy to [m, out], cast to f32.
        let dy_dtype = grad_res.dtype();
        let dy_dims = grad_res.dims().to_vec();
        let Some((&last, leading)) = dy_dims.split_last() else {
            return Err(Error::Msg(format!(
                "{op}: grad_res must have rank >= 1, got shape {dy_dims:?}"
            )));
        };
        if last != out_features {
            return Err(Error::Msg(format!(
                "{op}: grad_res's last dim ({last}) must equal W's out_features \
                 ({out_features}); grad_res shape {dy_dims:?}, W shape [{out_features}, \
                 {in_features}]"
            )));
        }
        let m: usize = leading.iter().product();
        let dy2 = grad_res.reshape((m, last))?.to_dtype(DType::F32)?;

        // Step 3: ONE rank-2 matmul, never broadcast_matmul (module doc).
        let dx2 = dy2.matmul(&w_deq)?; // [m, out] @ [out, in] -> [m, in]

        // Step 4: reshape back to arg's own shape (dy's leading dims, last
        // axis replaced by in_features), cast back to dy's original dtype.
        let mut dx_shape = dy_dims;
        *dx_shape
            .last_mut()
            .expect("dy_dims is non-empty: split_last succeeded above") = in_features;
        let dx = dx2.reshape(dx_shape)?.to_dtype(dy_dtype)?;

        // Never Ok(None) for the input gradient (module doc: candle's own
        // backprop.rs:663 drops a None gradient silently).
        Ok(Some(dx))
    }
}

/// The ONLY public entry point besides [`QuantMatMulGrad::new`] itself —
/// always-differentiable, one code path for train and eval (module doc).
pub fn quant_matmul_grad(x: &Tensor, w: Arc<QTensor>) -> Result<Tensor> {
    let x = x.contiguous()?;
    apply_stateful1(&x, QuantMatMulGrad::new(w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::{Device, Var};

    fn make_weight(
        out_f: usize,
        in_f: usize,
        dtype: GgmlDType,
        seed: f64,
    ) -> (Tensor, Arc<QTensor>) {
        let device = Device::Cpu;
        let w_f32: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f64) * 0.037 + seed).sin() as f32)
            .collect();
        let w = Tensor::from_vec(w_f32, (out_f, in_f), &device).unwrap();
        let q = QTensor::quantize(&w, dtype).unwrap();
        (w, Arc::new(q))
    }

    fn make_x(rows: usize, in_f: usize, seed: f64) -> Tensor {
        let device = Device::Cpu;
        let x_v: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i as f64) * 0.091 + seed).cos() as f32)
            .collect();
        Tensor::from_vec(x_v, (rows, in_f), &device).unwrap()
    }

    /// Oracle (a): forward parity against the dense dequantized reference,
    /// for q8_0, q4_0 (last dim % 32 == 0) and one q4k case (last dim ==
    /// 256, QK_K's own block size).
    #[test]
    fn forward_parity_against_dense_dequantized_reference_q8_0_q4_0_q4k() {
        // Tolerances are MEASURED, not asserted-by-hope (family F): the op's
        // own quantized dot-product kernel (candle's `matmul_t`, block-wise
        // int accumulation) and the dense `dequantize -> matmul` reference
        // sum the SAME underlying quantized values in a DIFFERENT order —
        // this is a genuine reduction-order divergence (family J: no
        // cross-order bit-identity is claimed), not a correctness bug.
        // Measured `max_abs_diff` at this fixture's shape/seed — see
        // `forward_parity_against_dense_dequantized_reference_q8_0_q4_0_q4k`
        // (this very test, re-run to reproduce): `Q8_0 0.0288` (max |value|
        // `18.40`, ~0.16% relative), `Q4_0 0.0285` (max |value| `18.31`,
        // ~0.16% relative), `Q4K 0.0527` (max |value| `9.17`, ~0.57%
        // relative, wider block scan accumulating more reduction-order
        // divergence at `QK_K=256`). Tolerances below carry a real margin
        // over each measured figure, not a vacuously wide bound.
        for (dtype, out_f, in_f, tol) in [
            (GgmlDType::Q8_0, 4usize, 64usize, 0.05f32),
            (GgmlDType::Q4_0, 4usize, 64usize, 0.05f32),
            (GgmlDType::Q4K, 2usize, 256usize, 0.08f32),
        ] {
            let rows = 3;
            let (_w, wq) = make_weight(out_f, in_f, dtype, 0.5);
            let x = make_x(rows, in_f, 1.0);

            let got = quant_matmul_grad(&x, wq.clone()).unwrap();
            assert_eq!(got.dims(), &[rows, out_f]);

            let w_deq = wq.dequantize(&Device::Cpu).unwrap();
            let expected = x.matmul(&w_deq.t().unwrap()).unwrap();

            let got_v = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let expected_v = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut max_abs_diff = 0f32;
            for (g, e) in got_v.iter().zip(expected_v.iter()) {
                max_abs_diff = max_abs_diff.max((g - e).abs());
            }
            assert!(
                max_abs_diff < tol,
                "{dtype:?}: max abs diff {max_abs_diff} >= tol {tol}"
            );
        }
    }

    /// Oracle (b): bwd parity — the op's own input gradient equals the
    /// gradient candle's own autograd computes through the plain dense
    /// `x @ dequantize(W)^T` reference, exact same math (dequantized once
    /// for both paths), tight tolerance.
    #[test]
    fn bwd_parity_against_dense_dequantized_reference_autograd() {
        let (out_f, in_f, rows) = (5usize, 64usize, 3usize);
        let (_w, wq) = make_weight(out_f, in_f, GgmlDType::Q8_0, 0.25);
        let device = Device::Cpu;
        let x_v: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i as f64) * 0.061 + 2.0).sin() as f32)
            .collect();

        let x_op = Var::from_tensor(&Tensor::from_vec(x_v.clone(), (rows, in_f), &device).unwrap())
            .unwrap();
        let y_op = quant_matmul_grad(x_op.as_tensor(), wq.clone()).unwrap();
        let loss_op = y_op.sum_all().unwrap();
        let grads_op = loss_op.backward().unwrap();
        let grad_op = grads_op.get(x_op.as_tensor()).unwrap();

        let w_deq = wq.dequantize(&device).unwrap();
        let x_dense =
            Var::from_tensor(&Tensor::from_vec(x_v, (rows, in_f), &device).unwrap()).unwrap();
        let y_dense = x_dense.as_tensor().matmul(&w_deq.t().unwrap()).unwrap();
        let loss_dense = y_dense.sum_all().unwrap();
        let grads_dense = loss_dense.backward().unwrap();
        let grad_dense = grads_dense.get(x_dense.as_tensor()).unwrap();

        let got_v = grad_op.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_v = grad_dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_abs_diff = 0f32;
        for (g, e) in got_v.iter().zip(expected_v.iter()) {
            max_abs_diff = max_abs_diff.max((g - e).abs());
        }
        assert!(max_abs_diff < 1e-3, "max abs diff {max_abs_diff}");
    }

    /// Oracle (c): gradient reachability, no skip route — a single
    /// quantized-linear application, a trainable `Var` exclusively
    /// upstream, no residual: grad must be `Some` and non-zero (the
    /// esc-037-faithful shape — a residual-skip fixture would let a
    /// silently-`None`-dropped gradient hide behind the residual's own
    /// contribution and would be vacuous).
    #[test]
    fn gradient_is_reachable_with_no_skip_route() {
        let (out_f, in_f, rows) = (3usize, 32usize, 2usize);
        let (_w, wq) = make_weight(out_f, in_f, GgmlDType::Q4_0, 0.75);
        let device = Device::Cpu;
        let x_v: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i as f64) * 0.043 + 0.9).sin() as f32)
            .collect();
        let x = Var::from_tensor(&Tensor::from_vec(x_v, (rows, in_f), &device).unwrap()).unwrap();

        // No residual/skip anywhere: y is the op's own output directly.
        let y = quant_matmul_grad(x.as_tensor(), wq).unwrap();
        let loss = y.sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let grad = grads.get(x.as_tensor());
        assert!(grad.is_some(), "input gradient must be Some, never dropped");
        let grad = grad.unwrap();
        let grad_abs_sum: f32 = grad
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.abs())
            .sum();
        assert!(
            grad_abs_sum > 0.0,
            "gradient must be non-zero, got sum {grad_abs_sum}"
        );
    }

    /// `bwd` never returns `Ok(None)` — a literal mechanism pin for the
    /// module doc's own claim (candle's `backprop.rs:663` drops `None`
    /// silently, the esc-037 hazard this op exists to avoid).
    #[test]
    fn bwd_never_returns_none_for_the_input_gradient() {
        let (out_f, in_f, rows) = (2usize, 32usize, 1usize);
        let (_w, wq) = make_weight(out_f, in_f, GgmlDType::Q4_0, 1.5);
        let device = Device::Cpu;
        let x = Tensor::zeros((rows, in_f), DType::F32, &device).unwrap();
        let dy = Tensor::ones((rows, out_f), DType::F32, &device).unwrap();
        let op = QuantMatMulGrad::new(wq);
        let y = quant_matmul_grad(&x, op.w.clone()).unwrap();
        let grad = op.bwd(&x, &y, &dy).unwrap();
        assert!(grad.is_some(), "bwd must never return Ok(None)");
    }

    /// Oracle (d): eval-path pruning — the op's output on a NON-tracked
    /// input carries no recorded grad-op (candle's own `BackpropOp::new1`
    /// self-pruning, module doc) and is bit-identical to the tracked
    /// forward's own value — one code path, not a train/eval branch.
    #[test]
    fn eval_path_output_carries_no_grad_op_and_matches_tracked_value() {
        let (out_f, in_f, rows) = (3usize, 32usize, 2usize);
        let (_w, wq) = make_weight(out_f, in_f, GgmlDType::Q4_0, 3.25);
        let device = Device::Cpu;
        let x_v: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i as f64) * 0.021 + 1.1).cos() as f32)
            .collect();

        let x_plain = Tensor::from_vec(x_v.clone(), (rows, in_f), &device).unwrap();
        assert!(
            !x_plain.track_op(),
            "sanity: a plain (non-Var) Tensor must not track ops"
        );
        let y_plain = quant_matmul_grad(&x_plain, wq.clone()).unwrap();
        assert!(
            !y_plain.track_op(),
            "an untracked input must produce an untracked output (self-pruned BackpropOp)"
        );

        let x_tracked =
            Var::from_tensor(&Tensor::from_vec(x_v, (rows, in_f), &device).unwrap()).unwrap();
        let y_tracked = quant_matmul_grad(x_tracked.as_tensor(), wq).unwrap();
        assert!(
            y_tracked.track_op(),
            "sanity: a Var-derived input must track ops"
        );

        let plain_v = y_plain.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let tracked_v = y_tracked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            plain_v, tracked_v,
            "eval-path (untracked) output must be bit-identical to the tracked forward's value"
        );
    }
}
