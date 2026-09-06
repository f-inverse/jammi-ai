//! Single LoRA-augmented linear layer: frozen base + trainable A and B matrices.

use candle_core::{DType, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
    DispatchSnapshot,
};
use jammi_kernels::ops::{apply1, apply3, DropoutFused, DropoutKey, LowRankResidualLinear};

use crate::error::LoraError;
use crate::frozen_base::FrozenBase;
use crate::init::LoraInitMode;
use crate::seeded::{
    gaussian_fill, kaiming_uniform_fill, seed_for_param, DropoutMasks, SplitMix64,
};

/// Per-op fused/eager dispatch counts for the device-side dropout op
/// (`jammi_kernels::ops::DropoutFused`), read from the same op-keyed
/// registry `lora_epilogue_counters` (below) uses.
///
/// **Permanently `{fused: 0, eager: 0}` today.** `forward`'s training arm
/// no longer calls `DropoutMasks::apply` (which recorded here via
/// `admit`) — dropout is now reserved via `DropoutMasks::next_key` and
/// consumed DIRECTLY by [`LowRankResidualLinear`]/[`DropoutFused::new`] on
/// EITHER arm (fused-site or eager-fallback), bypassing this counter's
/// own `admit` call entirely (see `forward`'s doc). The eager-fallback
/// arm's own dropout call (`apply1`) is a PLAIN function call too, not
/// routed through `admit`, so this counter stays zero on that arm as
/// well — not just the fused one. The function is kept, unchanged, for
/// source and snapshot-schema compatibility with any existing
/// durable-job-record reader of this name; a NEW consumer wanting
/// dropout's fused/eager split should read
/// [`lora_linear_fused_dispatch_snapshot`] instead — dropout is now
/// folded into that ONE counter, the same way `lora_epilogue`'s is.
fn lora_dropout_counters() -> &'static DispatchCounters {
    counters_for("lora_dropout")
}

/// A snapshot of the fused/eager dispatch counts for the LoRA dropout op —
/// mirrors [`lora_epilogue_dispatch_snapshot`]. See `lora_dropout_counters`'s
/// doc: permanently zero today, on both arms.
pub fn lora_dropout_dispatch_snapshot() -> DispatchSnapshot {
    lora_dropout_counters().snapshot()
}

/// Per-op fused/eager dispatch counts for the LoRA-site epilogue
/// (`base_out + cast(lora_out * scaling)`), read from `jammi-kernels`' new
/// op-keyed registry (`counters_for`) rather than a hand-declared
/// `static DispatchCounters` — this crate is the first to use the
/// generalized form C6 adds (see `jammi_kernels::admission`'s module doc);
/// C2-C5's four ops in `jammi-encoders` keep their own pre-existing
/// statics unchanged.
///
/// **Permanently `{fused: 0, eager: 0}` today**, for the same reason
/// [`lora_dropout_counters`] is: the standalone epilogue call `forward`
/// used to make
/// (`apply2(base_out, lora_out, ScaledCastAdd::new(..))`) is superseded by
/// [`LowRankResidualLinear`], which reuses `ScaledCastAdd`'s `cpu_fwd`/`cuda_fwd`
/// DIRECTLY (a plain function call, not through `admit`) as its own
/// internal epilogue step — see `jammi_kernels::ops::low_rank_residual_linear`'s module
/// doc. Kept, unchanged, for source/snapshot-schema compatibility; see
/// [`lora_linear_fused_dispatch_snapshot`] for the counter that now
/// reflects this call site's real dispatch split.
fn lora_epilogue_counters() -> &'static DispatchCounters {
    counters_for("lora_epilogue")
}

/// A snapshot of the fused/eager dispatch counts for the LoRA-site
/// epilogue, mirroring `jammi_encoders::ln_dispatch_snapshot` /
/// `rope_dispatch_snapshot` / `softmax_dispatch_snapshot` /
/// `geglu_dispatch_snapshot` — the read API a durable job record or a
/// bench report uses to state which kernel path actually ran. See
/// `lora_epilogue_counters`'s doc: permanently zero today.
pub fn lora_epilogue_dispatch_snapshot() -> DispatchSnapshot {
    lora_epilogue_counters().snapshot()
}

/// The eager `[mul, add, cast]` composition the fused epilogue replaces:
/// widen `base_out` and the scaled `lora_out` delta to the WIDER of the
/// two dtypes (torch's own promotion rule — never narrow the wider
/// operand toward the narrower one), add in that dtype, and cast the SUM
/// to `base_out`'s ORIGINAL dtype ONCE — esc-046 fix (GH#374), matching
/// PEFT's `Linear.forward` (`peft/tuners/lora/layer.py` 1044-1069,
/// `v0.20.0`, re-read at source on pod a100e 2026-08-26): torch's `+`
/// promotes to the WIDER dtype of its two operands (never toward the
/// narrower one), adds once, and only THEN casts back to
/// `torch_result_dtype` (`result`'s OWN original dtype) once via
/// `.to(torch_result_dtype)`.
///
/// **Round 2 fix (esc-046 audit finding 1, `lora_linear.rs:103`):** an
/// earlier revision of this function promoted `base_out` to `scaled`'s
/// OWN dtype unconditionally (`base_out.to_dtype(scaled.dtype())`) rather
/// than to the wider of the two — correct for the (`BF16` base, `F32`
/// lora) pair this fix was written against (the wider dtype IS `scaled`'s
/// there), but for the reachable inverse pair (`F32` base, `BF16` lora —
/// reachable via [`LoraLinear::from_loaded`], whose `lora_a`/`lora_b` are
/// raw, dtype-unconstrained `Tensor`s, and via eval-mode `forward`, which
/// calls this function unconditionally) it NARROWED the `f32` base down
/// to `bf16` before the add, silently destroying the base signal's own
/// precision — measured 4095/4096 elements diverging from torch's actual
/// promotion by up to `7.23e-1` (vs torch's own `3.81e-6`) on a
/// `n=4096`, `|base|~100` fixture; the same divergence class is
/// re-measured on every run — see
/// `eager_epilogue_f32_base_bf16_lora_would_diverge_under_the_narrow_first_regression`. [`wider_float_dtype`] below is the
/// single source of truth for "which dtype must the add happen in",
/// shared by every cell of the dtype lattice this function's own tests
/// (`eager_epilogue_tests`) exercise.
///
/// **The `(BF16, BF16)` cell is NOT computed as a native `bf16` add**,
/// even though `bf16` is trivially "the wider of two equal dtypes" —
/// [`wider_float_dtype`] floors the compute dtype at `F32` for any
/// half-precision input, matching torch's own bf16-arithmetic convention
/// (bf16 ops compute in `f32` internally, never natively) AND sidestepping
/// a measured candle anomaly: candle-core 0.11.0's CPU `BF16` `Tensor`
/// `Add` is SIZE-DEPENDENT (the NEON-vectorized path narrows via a bare
/// truncation, not round-to-nearest-even, for `n` past its `STEP = 32`
/// lane width — the same anomaly `cast_scale.rs`'s own module doc
/// documents for `CastAddBf16`'s CPU arm) — `bf16(8.625) + bf16(2.859375)`
/// measured `11.5` at `n = 8` but `11.4375` at `n = 4096` on this
/// codebase's own dev/CI arm64 hosts. Routing every add through `F32`
/// unconditionally means this function never reaches that op at all.
///
/// Kept as its own function so both the eval-mode path (which always uses
/// it — see `forward`'s doc) and the training-mode fallback (when the
/// fused kernel's domain does not hold) share exactly one implementation.
fn eager_epilogue(base_out: &Tensor, lora_out: &Tensor, scaling: f64) -> Result<Tensor, LoraError> {
    let scaled = (lora_out * scaling)?;
    let base_dtype = base_out.dtype();
    let compute_dtype = wider_float_dtype(base_dtype, scaled.dtype())?;
    let base_wide = if base_dtype == compute_dtype {
        base_out.clone()
    } else {
        base_out.to_dtype(compute_dtype)?
    };
    let scaled_wide = if scaled.dtype() == compute_dtype {
        scaled
    } else {
        scaled.to_dtype(compute_dtype)?
    };
    let sum = (&base_wide + &scaled_wide)?;
    Ok(if sum.dtype() == base_dtype {
        sum
    } else {
        sum.to_dtype(base_dtype)?
    })
}

/// The single source of truth for "which dtype must an `eager_epilogue`
/// add happen in", torch's own floating-point promotion rule (family D:
/// the domain is floating dtypes only — an integer dtype here is a typed
/// refusal, never a silent reinterpretation): `F64` if either operand is
/// `F64`, else `F32` unconditionally — `F16`/`BF16` NEVER win (both are
/// floored at `F32`, matching torch's own "half-precision ops compute in
/// f32" convention, not merely "wider of the two REPRESENTED widths");
/// `F32` vs `F32` (or `F16`/`BF16` vs `F16`/`BF16`) stays `F32`. See
/// `eager_epilogue`'s own doc for the measured regression this closes and
/// the candle CPU `BF16` `Tensor::add` anomaly this floor sidesteps.
fn wider_float_dtype(a: DType, b: DType) -> Result<DType, LoraError> {
    let is_float = |d: DType| matches!(d, DType::F64 | DType::F32 | DType::F16 | DType::BF16);
    if !is_float(a) || !is_float(b) {
        return Err(LoraError::Config(format!(
            "eager_epilogue: unsupported dtype pair ({a:?}, {b:?}) — both operands must be a \
             floating dtype (F64/F32/F16/BF16)"
        )));
    }
    Ok(if a == DType::F64 || b == DType::F64 {
        DType::F64
    } else {
        DType::F32
    })
}

/// Per-op fused/eager dispatch counts for the fused LoRA SITE
/// (`jammi_kernels::ops::LowRankResidualLinear`, the whole-site fusion),
/// read from the same op-keyed registry `lora_epilogue_counters` uses.
/// MEASURED (not estimated) at the production `LoraLinear::forward` path,
/// on the BIAS-FREE harness (`rank`-3 `x`, `F32`, `dropout = 0.3`, a
/// frozen, bias-free `w`) via `Tensor::sorted_nodes().len()`: the fused
/// arm retains 5 tape nodes end to end (3 OP-CARRYING — `A.t()`, the `ab`
/// pack `Op::Cat`, and this op's own `CustomOp3` call — plus the 2 `Var`
/// leaves, `A`/`B`) versus 9 op-carrying nodes (11 total) for the eager
/// composition `eager_epilogue` and its own `A`/`B`/dropout sub-linears
/// build — see `crates/jammi-lora/tests/fused_epilogue.rs`'s
/// `production_path_retains_fewer_tape_nodes_fused_vs_eager_fallback` for
/// the harness these numbers come from. **The SAME bias-free harness with
/// a real, untracked-leaf bias added to `w` (#428 P2b) measures the
/// IDENTICAL pair, 5 fused / 11 eager** — see that test's own sibling,
/// `production_path_retains_fewer_tape_nodes_fused_vs_eager_fallback_with_bias`:
/// the frozen bias contributes no tape node of its own on either arm at
/// this measurement's granularity (the fused arm's `ab` pack merely gains
/// a third, still-single-node `Tensor::cat` argument). The eager arm's own
/// bias add is NOT absorbed into an existing node — `candle-nn` 0.11.0's
/// `Linear::forward` (`linear.rs:73-76`) ends with a separate
/// `x.broadcast_add(bias)`, which IS its own tracked node whenever its
/// inputs track (the op-level oracle in
/// `jammi_kernels::ops::low_rank_residual_linear` measures exactly that
/// case directly: 10 eager tape nodes bias-free vs 11 with a bias, see
/// `fused_site_with_bias_retains_fewer_tape_nodes_than_the_eager_biased_composition`).
/// The reason THIS harness measures the identical 5/11 pair with and
/// without a bias is that here `x` (a plain `Tensor::randn`,
/// `crates/jammi-lora/tests/fused_epilogue.rs`'s `rand_input`, used by both
/// the bias-free and bias-carrying production harnesses) and this
/// harness's `w`/bias are all untracked leaves, so the entire base branch —
/// the matmul AND the bias add — is off the tape on both arms; only the
/// `A`/`B` LoRA branch is tracked at all, and it is unaffected by the
/// bias. Stated as the MEASURED fact, not assumed equal to the bias-free
/// pair without checking it. Every op-carrying node is one
/// `GradStore::or_insert` (`backprop.rs`) full-size `zeros_like` + `add`
/// at backward time — `A.t()`/the `ab` pack are the two this op's own
/// `CustomOp3` collapse does NOT eliminate (they still cost their own
/// node each), disclosed here rather than folded silently into "one
/// node".
///
/// **`lora_epilogue`/`lora_dropout` legitimately read `0` for every
/// forward that dispatches through this counter instead.** Once the
/// training arm routes through [`LowRankResidualLinear`], neither
/// [`ScaledCastAdd`] nor a standalone [`DropoutFused`] call is EVER made
/// for that forward — both are reused INSIDE the fused kernel's own
/// `cpu_fwd`/`cuda_fwd` (see `jammi_kernels::ops::low_rank_residual_linear`'s module
/// doc), called directly as plain functions, never through
/// `jammi_kernels::ops::apply1`/`apply2`'s own dispatch-counted path. This
/// is not a regression in observability: the fused/eager split IS
/// observable, just under this counter's name instead of the two it
/// superseded for the training arm.
fn lora_linear_fused_counters() -> &'static DispatchCounters {
    counters_for("lora_linear_fused")
}

/// A snapshot of the fused/eager dispatch counts for the fused LoRA site —
/// mirrors [`lora_epilogue_dispatch_snapshot`]/[`lora_dropout_dispatch_snapshot`].
pub fn lora_linear_fused_dispatch_snapshot() -> DispatchSnapshot {
    lora_linear_fused_counters().snapshot()
}

/// The fused LoRA-SITE kernel's domain, checked at the call site (family D
/// / K2): [`device_is_supported`]; `x` rank 2 (a pooled head) or 3
/// (`[batch, seq, in]`); `x`/`w` share a dtype
/// that is `F32`, `BF16`, or `F16` (all three matched, never mixed — the
/// three combinations [`jammi_kernels::ops::LowRankResidualLinear`]
/// actually implements, `F16` widened in campaign #443 D1: that op's own
/// `cpu_fwd`/`cuda_fwd` dtype gate, and `ScaledCastAdd`'s CPU epilogue,
/// both admit `F16` end to end — this call-site predicate had NOT been
/// widened to match until this fix, so an `F16` backbone fell back to
/// [`LoraLinear::forward_composed`]'s eager `[mul, cast, add]` composition
/// for every training forward despite the fused kernel's own domain
/// already covering it (esc-075/esc-076's own triage row: an `F16`
/// backbone was silently, permanently eager at this site — the fused/0
/// eager dispatch-count proof below, and the pod's own 17360-fused/0-eager
/// trace, is what this widening closes). This widening is NOT a fix for
/// the separate `s512` held-out-eval OOM esc-076 also tracked: that OOM's
/// mechanism was pinned by pod trace evidence to `evaluate_held_out`
/// rounding its first eval batch UP to the `512` bucket rung (a bucketing
/// call-site bug, `be1450ae`) — it reproduced on BOTH `bf16` and `f16`
/// `alloff` legs alike, not `F16` alone, and was fixed independently, at
/// that call site, by routing eval through natural-width tokenization
/// instead of bucket-up padding. No causal claim is made here about
/// eager-arm allocator fragmentation contributing to (or explaining) that
/// OOM — this predicate's own domain-coverage gap is the only thing
/// measured and fixed by this change; `w`
/// contiguous (`x` is NOT required to be — the op materializes a
/// non-contiguous `x` internally; see the op's own domain doc). A base
/// bias is no longer a domain refusal (#428 P2b): a bias-carrying base
/// FUSES whenever [`bias_gate`] produced a `bias_pack` for it (an
/// untracked leaf, the overwhelmingly common case) — the ONLY bias-shaped
/// refusal left is a trainable `Var` bias, `bias_is_frozen_leaf`, a
/// COUNTED miss checked FIRST, below. `out_features >= 1`
/// and `rank >= 1` are guaranteed by construction (`LoraLinear::new`
/// refuses `rank == 0`, and a real `Linear`'s weight always has
/// `out_features >= 1`) — not re-checked here, but re-validated
/// independently by [`LowRankResidualLinear::new`] regardless (family D: an op
/// trusts no caller for its own domain). `lora_a`/`lora_b` (packed into
/// `ab`) must be `F32` — the op's own domain requires it — checked here
/// too as a COUNTED domain miss, not only as the op's own hard refusal.
fn lora_linear_admission_predicate(
    x: &Tensor,
    w: &Tensor,
    lora_dtype: DType,
    base_has_bias: bool,
    bias_pack_is_some: bool,
) -> (bool, &'static str) {
    // #428 P2b: a bias-carrying base is no longer an unconditional domain
    // miss (`base_has_no_bias` is DELETED) — a bias packs into `ab`'s
    // trailing rows whenever [`bias_gate`] produced one. The ONLY
    // remaining bias-shaped refusal is the case `bias_gate` deliberately
    // returns `None` for despite a bias being PRESENT on the base: a
    // trainable `Var` bias, which the eager composition already tracks
    // correctly and the fused kernel has no slot for (a frozen-only pack).
    // COUNTED (not a silent "no bias" misread) — see `bias_gate`'s own doc
    // for why this state needs its own named reason.
    if base_has_bias && !bias_pack_is_some {
        return (false, "bias_is_frozen_leaf");
    }
    // [`jammi_kernels::ops::LowRankResidualLinear`]'s own domain requires `ab`
    // to be `F32` (checked again by the op itself, family D — this is a
    // COUNTED domain miss at the call site, not a substitute for that
    // check). Today's workspace fact is that `lora_a`/`lora_b` are always
    // built `F32` (this predicate's own doc), so this branch is not
    // expected to ever fire in this workspace — it exists so a FUTURE
    // non-`F32` adapter falls back loudly-counted rather than reaching the
    // op's own hard `UnsupportedDTypeForOp` refusal via a code path that
    // looks like ordinary fused dispatch.
    if lora_dtype != DType::F32 {
        return (false, "lora_ab_dtype_f32");
    }
    if !device_is_supported(x.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    let x_rank = x.dims().len();
    if x_rank != 2 && x_rank != 3 {
        return (false, "x_rank_2_or_3");
    }
    // `x` is deliberately NOT required to be contiguous here (round-2
    // audit finding A2): `jammi_kernels::ops::LowRankResidualLinear`
    // materializes a non-contiguous `x` internally, at the storage level,
    // rather than refusing it (see its own
    // `materialize_contiguous_if_needed` doc) — the ONE argument a real
    // call site does not fully control the layout of (an upstream
    // reshape/transpose can hand this a strided view). Refusing it here
    // too would leave that op-level handling permanently unreachable from
    // production and push a numerically harmless shape to the eager
    // fallback for no reason.
    if !w.is_contiguous() {
        return (false, "w_contiguous");
    }
    // `w.dims()[0]` is `out_features` — always `>= 1` for a real `Linear`
    // weight, but checked explicitly rather than assumed (family D): a
    // degenerate zero-row weight is exactly the kind of edge this
    // predicate exists to catch, not silently pass through to a GEMM with
    // an illegal dimension.
    if w.dims().first().copied().unwrap_or(0) == 0 {
        return (false, "out_features_ge_1");
    }
    match (x.dtype(), w.dtype()) {
        (DType::F32, DType::F32) | (DType::BF16, DType::BF16) | (DType::F16, DType::F16) => {}
        _ => return (false, "base_dtype_f32_bf16_or_f16_matched"),
    }
    (true, "domain_ok")
}

/// The frozen-base-weight gate (rule 6, refined): a `LoraLinear`'s
/// contract is a FROZEN base, so its weight must be either a true leaf
/// (`!w.track_op()`, `dweight_needed = false` — the ordinary case, a
/// weight loaded straight from a `VarBuilder`) or itself a trainable
/// `Var` (`w.is_variable()`, `dweight_needed = true` — an unusual but
/// legitimate "also fine-tune the base" configuration). A base weight
/// that is TRACKED (carries an `Op`, e.g. a `Tensor::to_dtype` cast
/// applied after loading) but is NOT a `Var` is refused with a typed
/// error: `w.is_variable()` alone would silently miss this case (a
/// tracked non-`Var` intermediate is neither "definitely frozen" nor
/// "definitely trainable"), and `!w.track_op()` alone would OVER-refuse a
/// legitimate trainable `Var` — candle-core 0.11's `Tensor::track_op` is
/// `is_variable() || op.is_some()` (`tensor.rs:592-594`), so a `Var`
/// itself DOES report `track_op() == true` (unlike a true frozen leaf).
/// The `is_variable()` branch must therefore be tried FIRST: it is the
/// only test that can tell a trainable `Var` apart from a tracked
/// non-`Var` intermediate, both of which have `track_op() == true`. See
/// `jammi_kernels::ops::low_rank_residual_linear`'s module doc for what `dweight_needed`
/// controls in the fused kernel's own `bwd`.
pub(crate) fn frozen_weight_gate(w: &Tensor) -> Result<bool, LoraError> {
    if w.is_variable() {
        Ok(true)
    } else if !w.track_op() {
        Ok(false)
    } else {
        Err(LoraError::Config(
            "LoraLinear: base weight is a TRACKED tensor (carries an Op) but is not a Var — \
             a LoRA base must be either a true frozen leaf or an explicitly trainable Var; \
             a tracked non-Var base would silently lose its own gradient contribution"
                .into(),
        ))
    }
}

/// The bias three-way gate (#428 P2b), mirroring [`frozen_weight_gate`]'s
/// own shape: `bias` is the base weight's OWN bias (`None` for a bias-free
/// base — the ordinary `linear_no_bias` case). `rank`/`out_features` are
/// the LoRA site's own construction data (needed to compute `bias_rows =
/// out_features.div_ceil(rank)`, matching
/// [`jammi_kernels::ops::LowRankResidualLinear::bias_rows`]'s own
/// derivation bit-for-bit — this function and that private method must
/// never independently drift).
///
/// - **No bias** (`bias.is_none()`): `Ok(None)` — the ordinary,
///   overwhelmingly common case.
/// - **Untracked leaf** (`!bias.track_op()`, a bias loaded straight from a
///   `VarBuilder`): `Ok(Some(pack))`, the padded `[bias_rows, rank]` block
///   — widened to `F32` (lossless from `BF16`/`F16`, exact identity from
///   `F32`), zero-padded past `out_features` via `Tensor::pad_with_zeros`
///   (dim 0), then reshaped. Built on an untracked leaf: `pad_with_zeros`'s
///   own `Tensor::zeros`/`Tensor::cat` never attach a tracked `Op` unless
///   an argument already `track_op()`s (candle-core 0.11.0's
///   `BackpropOp::new` family), so the returned pack is itself untracked —
///   packing it into `ab` costs no MORE tape nodes than the bias-free
///   `ab` pack already does.
/// - **A trainable `Var`** (`bias.is_variable()`): `Ok(None)` — the eager
///   composition already tracks a `Var` bias correctly (it is a genuine
///   `candle_nn::Linear` argument there), so no pack is built; unlike the
///   "no bias" case above, THIS `None` must still surface as a bias
///   PRESENT on the base to `lora_linear_admission_predicate` (that
///   predicate reads `base_linear.bias().is_some()` independently — see
///   its own doc), which turns it into a COUNTED `bias_is_frozen_leaf`
///   refusal rather than silently reading identically to "no bias at
///   all".
/// - **Tracked, non-`Var`** (`bias.track_op() && !bias.is_variable()`):
///   a typed [`LoraError::Config`] refusal — the SAME ambiguous-tracked-
///   state policy `frozen_weight_gate` applies to `w` (a tracked
///   intermediate is neither definitely frozen nor definitely trainable;
///   silently choosing either would risk losing its gradient or
///   miscounting a "frozen" site as a Var).
///
/// Also validates (family D: an op trusts no caller for its own domain,
/// checked HERE rather than only inside
/// `LowRankResidualLinear::check_w_and_ab`) that `bias` is exactly
/// `[out_features]` and shares `w`'s dtype — both guaranteed by
/// `candle_nn::Linear`'s own construction in every path this workspace
/// exercises today, but re-checked rather than assumed.
fn bias_gate(
    bias: Option<&Tensor>,
    w_dtype: DType,
    out_features: usize,
    rank: usize,
) -> Result<Option<Tensor>, LoraError> {
    let Some(bias) = bias else {
        return Ok(None);
    };
    if bias.dims() != [out_features] {
        return Err(LoraError::Config(format!(
            "LoraLinear: base bias must be [{out_features}], got {:?}",
            bias.dims()
        )));
    }
    if bias.dtype() != w_dtype {
        return Err(LoraError::Config(format!(
            "LoraLinear: base bias dtype {:?} must match the base weight's dtype {w_dtype:?}",
            bias.dtype()
        )));
    }
    if bias.is_variable() {
        return Ok(None);
    }
    if bias.track_op() {
        return Err(LoraError::Config(
            "LoraLinear: base bias is a TRACKED tensor (carries an Op) but is not a Var — a \
             LoRA base bias must be either a true frozen leaf or an explicitly trainable Var; \
             a tracked non-Var bias would silently lose its own gradient contribution"
                .into(),
        ));
    }
    let bias_rows = out_features.div_ceil(rank);
    let bias_f32 = if bias.dtype() == DType::F32 {
        bias.clone()
    } else {
        bias.to_dtype(DType::F32)?
    };
    let pad = bias_rows * rank - out_features;
    let padded = bias_f32.pad_with_zeros(0, 0, pad)?;
    Ok(Some(padded.reshape((bias_rows, rank))?))
}

/// The effective LoRA scaling factor `γ_r` applied to `B @ A @ x` before it is
/// added to the frozen base output.
///
/// Vanilla LoRA (Hu et al. 2021) uses `γ_r = alpha / rank`. rsLoRA
/// (Kalajdzievski 2023, arXiv:2312.03732, §3, eq. "γ_r = α/√r") instead uses
/// `γ_r = alpha / sqrt(rank)` — chosen so the per-update variance stays
/// bounded as `rank` grows, which vanilla scaling does not. This matches
/// PEFT's own reference implementation
/// (`src/peft/tuners/lora/layer.py`, `LoraLayer.update_layer`):
/// `self.scaling[adapter_name] = lora_alpha / math.sqrt(r) if use_rslora
/// else lora_alpha / r`.
///
/// Scaling is a pure function of `(alpha, rank, use_rslora)` — nothing else
/// determines it, and every constructor of [`LoraLinear`] (both a fresh
/// `new` and a `from_loaded` reconstruction from a saved adapter) MUST route
/// through this one function so the two can never silently disagree.
///
/// Domain (family D / K2): `rank` must be `>= 1`. At `rank == 0` both
/// branches divide by zero (`alpha / 0` is `+inf`/`-inf`/`NaN` in f64, never
/// a `DivisionByZero` panic) — a confident wrong number, not a loud failure
/// — so this is refused with a typed [`LoraError::Config`] rather than
/// silently returned.
pub fn lora_scaling(alpha: f64, rank: usize, use_rslora: bool) -> Result<f64, LoraError> {
    if rank == 0 {
        return Err(LoraError::Config("LoRA rank must be > 0".into()));
    }
    Ok(if use_rslora {
        alpha / (rank as f64).sqrt()
    } else {
        alpha / rank as f64
    })
}

/// Overwrite the storage of the `Var` already registered at `name` in `varmap`
/// with `value`, reaching it through the shared `&VarMap` (no `&mut` needed
/// because `Var::set` is `&self`). Fails if no such `Var` exists — the caller
/// must have registered it (via `get_with_hints`) first.
fn set_var(varmap: &VarMap, name: &str, value: &Tensor) -> Result<(), LoraError> {
    let data = varmap.data().lock().map_err(|_| {
        LoraError::Config(format!("seeded init: VarMap mutex poisoned setting {name}"))
    })?;
    let var = data.get(name).ok_or_else(|| {
        LoraError::Config(format!("seeded init: {name} not registered in VarMap"))
    })?;
    var.set(value)
        .map_err(|e| LoraError::Config(format!("seeded init set {name}: {e}")))
}

/// A linear layer wrapped with a LoRA adapter.
///
/// The base weight is treated as frozen. The output is
/// `base(x) + scaling * dropout(x @ A^T @ B^T)`.
pub struct LoraLinear {
    base: FrozenBase,
    /// LoRA A matrix with shape `(rank, in_features)`.
    pub lora_a: Tensor,
    /// LoRA B matrix with shape `(out_features, rank)`.
    pub lora_b: Tensor,
    /// Pre-computed scaling factor (`alpha / rank` or `alpha / sqrt(rank)`).
    scaling: f64,
    /// Optional dropout probability applied to the LoRA path while training.
    /// Validated to `[0.0, 1.0)` in `new` (a typed `LoraError::Config`, not
    /// silently accepted — `lora_dropout` was UNVALIDATED before this
    /// commit).
    dropout: Option<f32>,
    /// Run-owned, counter-keyed dropout mask source. `Some` exactly when
    /// `dropout > 0`. Interior-mutable because the `Module`-style
    /// `forward(&self, …)` advances the forward counter; no `Mutex` is
    /// needed (unlike the design this replaces) because `DropoutMasks`
    /// itself holds only an `AtomicU64`, which is natively `Sync` — the
    /// wip branch's "atomic counter replacing the per-layer `Mutex`" shape,
    /// adopted here.
    dropout_masks: Option<DropoutMasks>,
    /// Whether the layer is currently in training mode.
    training: bool,
    /// `in_features`/`out_features` of the base weight, `rank` of the LoRA
    /// adapter — cached at construction (`base.weight().dim(1)`/`dim(0)`,
    /// `lora_a.dim(0)`) so the fused-site call site never re-derives them
    /// (and never re-propagates a `dim()` `Result`) on every forward.
    in_features: usize,
    out_features: usize,
    rank: usize,
    /// Whether the fused LoRA site's `bwd` must compute and return
    /// `Some(dW)` for the base weight — see [`frozen_weight_gate`]'s doc
    /// for the three-way base-weight classification this is derived from,
    /// evaluated ONCE at construction (the base weight's tracked/`Var`
    /// status cannot change over a `LoraLinear`'s lifetime — nothing in
    /// this crate ever swaps out `self.base`).
    dweight_needed: bool,
    /// The frozen base bias, pre-packed into
    /// [`jammi_kernels::ops::LowRankResidualLinear`]'s `[bias_rows,
    /// rank]` block (`F32`, zero-padded past `out_features` — see that
    /// op's own module doc, "Bias: packed into `ab`'s trailing rows"),
    /// computed ONCE by [`bias_gate`] at construction — never re-derived
    /// per forward. `None` in THREE distinct cases, all handled by
    /// `forward`'s own admission logic: the base carries no bias at all;
    /// the base bias is itself a trainable `Var` (the eager composition
    /// already tracks it correctly, so no pack is needed — but the fused
    /// site must then be a COUNTED refusal, `bias_is_frozen_leaf`, not a
    /// silent eager fallback that looks the same as "no bias"); or the
    /// base is `FrozenBase::Quantized` (module doc consumer 6: the fused
    /// kernel is Dense-only, so a Quantized base's bias — if any — never
    /// gets a pack at all, by construction, not by a failed gate check).
    bias_pack: Option<Tensor>,
}

impl LoraLinear {
    /// Wrap a frozen `Linear` layer with a LoRA adapter.
    ///
    /// `rank` is the low-rank dimension. `alpha` scales the LoRA contribution.
    /// With `use_rslora`, the scaling becomes `alpha / sqrt(rank)` instead of
    /// `alpha / rank`. `init_mode` selects how the A and B tensors are seeded.
    ///
    /// `seed` makes the A/B init and the dropout mask a pure function of the run
    /// seed and the parameter's fully-qualified name (`{vb.prefix()}.lora_a` /
    /// `…lora_b`): the draws come from a jammi-owned `SplitMix64`, **not**
    /// candle's unseedable global RNG. Each parameter's stream is keyed by name
    /// (not by `VarMap`/construction order), so the same seed yields
    /// byte-identical adapters run-to-run and across processes. The seeded
    /// tensors are still registered as trainable `Var`s in `varmap` — candle's
    /// own `Init` first allocates the `Var` (with a deterministic placeholder so
    /// no RNG is touched), then the seeded values are written into it in place.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: Linear,
        rank: usize,
        alpha: f64,
        use_rslora: bool,
        init_mode: LoraInitMode,
        dropout: Option<f32>,
        seed: u64,
        varmap: &VarMap,
        vb: &VarBuilder,
    ) -> Result<Self, LoraError> {
        Self::new_with_base(
            FrozenBase::Dense(base),
            rank,
            alpha,
            use_rslora,
            init_mode,
            dropout,
            seed,
            varmap,
            vb,
        )
    }

    /// Wrap ANY [`FrozenBase`] — dense OR GGUF-quantized — with a LoRA
    /// adapter. [`Self::new`] is the Dense-only convenience wrapper every
    /// EXISTING construction path uses unchanged (`FrozenBase::Dense(base)`,
    /// then this function); a quantized base (wave-3 GGUF loading) calls
    /// this directly with `FrozenBase::Quantized(..)`. See [`Self::new`]'s
    /// own doc for the rank/init/seed/dropout semantics, identical here.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_base(
        base: FrozenBase,
        rank: usize,
        alpha: f64,
        use_rslora: bool,
        init_mode: LoraInitMode,
        dropout: Option<f32>,
        seed: u64,
        varmap: &VarMap,
        vb: &VarBuilder,
    ) -> Result<Self, LoraError> {
        if rank == 0 {
            return Err(LoraError::Config("LoRA rank must be > 0".into()));
        }
        let in_features = base.in_features()?;
        let out_features = base.out_features()?;
        let device = vb.device().clone();

        // Fetch (or, in the training path, allocate + register) the A/B tensors.
        // `Init::Const(0.0)` is deterministic so candle's RNG is never invoked.
        // In the TRAINING path the `VarBuilder` is VarMap-backed: this registers
        // fresh trainable `Var`s, which we then overwrite with the seeded draw.
        // In the INFERENCE path the `VarBuilder` is mmaped-safetensors-backed:
        // `get_with_hints` returns the SAVED adapter tensors and nothing is in
        // the (dummy) VarMap — so the seeded fill is correctly skipped and the
        // loaded weights stand.
        let lora_a = vb.get_with_hints((rank, in_features), "lora_a", Init::Const(0.0))?;
        let lora_b = vb.get_with_hints((out_features, rank), "lora_b", Init::Const(0.0))?;

        // Fully-qualified parameter names — the stable per-parameter draw key.
        // Built exactly as candle's `VarBuilder::path` joins them (no leading
        // dot when the prefix is empty) so they match the registered `Var` keys.
        let prefix = vb.prefix();
        let qualify = |leaf: &str| {
            if prefix.is_empty() {
                leaf.to_string()
            } else {
                format!("{prefix}.{leaf}")
            }
        };
        let a_name = qualify("lora_a");
        let b_name = qualify("lora_b");

        // Only seed-init the parameters that were just registered as trainable
        // `Var`s in `varmap`. If they are absent, this is the load-from-adapter
        // inference path and the values `get_with_hints` returned are the saved
        // weights, which must not be perturbed.
        let registered = {
            let data = varmap
                .data()
                .lock()
                .map_err(|_| LoraError::Config("seeded init: VarMap mutex poisoned".into()))?;
            data.contains_key(&a_name) && data.contains_key(&b_name)
        };

        if registered {
            let (a_values, b_values): (Vec<f32>, Vec<f32>) = match init_mode {
                LoraInitMode::ZerosB => {
                    // A: Kaiming-uniform over fan_in = in_features. B: zeros.
                    let mut rng = SplitMix64::new(seed_for_param(seed, &a_name));
                    let a = kaiming_uniform_fill(&mut rng, rank * in_features, in_features);
                    let b = vec![0.0_f32; out_features * rank];
                    (a, b)
                }
                LoraInitMode::Gaussian => {
                    // Both A and B ~ Normal(0, 0.02), independent name-keyed streams.
                    let mut rng_a = SplitMix64::new(seed_for_param(seed, &a_name));
                    let mut rng_b = SplitMix64::new(seed_for_param(seed, &b_name));
                    let a = gaussian_fill(&mut rng_a, rank * in_features, 0.02);
                    let b = gaussian_fill(&mut rng_b, out_features * rank, 0.02);
                    (a, b)
                }
            };

            let a_tensor = Tensor::from_vec(a_values, (rank, in_features), &device)?;
            let b_tensor = Tensor::from_vec(b_values, (out_features, rank), &device)?;
            // Overwrite the just-registered `Var`s' storage in place. `Var::set`
            // takes `&self`, so we reach it through the shared `VarMap` (which the
            // `VarBuilder` registered into) without needing `&mut` — the `Var`
            // identity the optimiser collects via `all_vars()` is preserved.
            set_var(varmap, &a_name, &a_tensor)?;
            set_var(varmap, &b_name, &b_tensor)?;
        }

        // `lora_dropout` was UNVALIDATED before this commit (any `f32` was
        // accepted, config.rs:26) — validate at the input edge (family D).
        // `p == 1.0` would drop every element and make the inverted-dropout
        // scale infinite; `(0.0..1.0).contains` is `false` for `NaN` too
        // (every comparison with `NaN` is `false`), so this also refuses a
        // non-finite probability rather than silently propagating it.
        if let Some(p) = dropout {
            if !(0.0..1.0).contains(&p) {
                return Err(LoraError::Config(format!(
                    "lora_dropout must be in [0.0, 1.0), got {p}"
                )));
            }
        }

        // `rank == 0` was already refused above, but `lora_scaling` is the
        // single source of truth for the scaling math (`from_loaded` below
        // routes through the same function) — see its own doc.
        let scaling = lora_scaling(alpha, rank, use_rslora)?;

        let dropout_masks = dropout
            .filter(|p| *p > 0.0)
            .map(|_| DropoutMasks::new(seed, &vb.prefix()));

        let dweight_needed = base.dweight_needed()?;
        // #428 P2b: only a `Dense` base ever gets a `bias_pack` — the
        // fused kernel is Dense-only (module doc consumer 6), so a
        // `Quantized` base's bias (if any) is left entirely to its own
        // `QuantizedLinear::forward` composition.
        let bias_pack = match &base {
            FrozenBase::Dense(l) => bias_gate(l.bias(), l.weight().dtype(), out_features, rank)?,
            FrozenBase::Quantized(_) => None,
        };

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling,
            dropout,
            dropout_masks,
            training: true,
            in_features,
            out_features,
            rank,
            dweight_needed,
            bias_pack,
        })
    }

    /// Convenience constructor: `ZerosB` init, no dropout, vanilla `alpha/rank`
    /// scaling, seeded from `seed`.
    pub fn new_simple(
        base: Linear,
        rank: usize,
        alpha: f64,
        seed: u64,
        varmap: &VarMap,
        vb: &VarBuilder,
    ) -> Result<Self, LoraError> {
        Self::new(
            base,
            rank,
            alpha,
            false,
            LoraInitMode::ZerosB,
            None,
            seed,
            varmap,
            vb,
        )
    }

    /// Reconstruct a `LoraLinear` from tensors already loaded from disk.
    ///
    /// `rank` is inferred from `lora_a.dims()[0]`; scaling is `lora_scaling`
    /// of `(alpha, rank, use_rslora)` — the SAME pure function `new` calls,
    /// so a reload can never silently disagree with the run that trained the
    /// adapter. The invariant is that scaling is entirely determined by the
    /// persisted `(alpha, rank, use_rslora)` triple: the caller must pass
    /// the `use_rslora` the adapter was actually trained with (typically
    /// read back from the adapter's own [`crate::AdapterConfig::use_rslora`]),
    /// not assume vanilla scaling.
    ///
    /// Refuses (typed, [`LoraError::Config`]) when `lora_a.dims()[0] == 0` —
    /// see `lora_scaling`'s domain doc.
    pub fn from_loaded(
        base: Linear,
        lora_a: Tensor,
        lora_b: Tensor,
        alpha: f64,
        use_rslora: bool,
    ) -> Result<Self, LoraError> {
        Self::from_loaded_with_base(FrozenBase::Dense(base), lora_a, lora_b, alpha, use_rslora)
    }

    /// Reconstruct a `LoraLinear` from tensors already loaded from disk,
    /// over ANY [`FrozenBase`] — dense OR GGUF-quantized. [`Self::from_loaded`]
    /// is the Dense-only convenience wrapper every EXISTING path uses
    /// unchanged; see its own doc for the rank/scaling semantics, identical
    /// here.
    pub fn from_loaded_with_base(
        base: FrozenBase,
        lora_a: Tensor,
        lora_b: Tensor,
        alpha: f64,
        use_rslora: bool,
    ) -> Result<Self, LoraError> {
        let rank = lora_a.dims()[0];
        let scaling = lora_scaling(alpha, rank, use_rslora)?;
        let in_features = base.in_features()?;
        let out_features = base.out_features()?;
        let dweight_needed = base.dweight_needed()?;
        let bias_pack = match &base {
            FrozenBase::Dense(l) => bias_gate(l.bias(), l.weight().dtype(), out_features, rank)?,
            FrozenBase::Quantized(_) => None,
        };
        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling,
            dropout: None,
            dropout_masks: None,
            training: false,
            in_features,
            out_features,
            rank,
            dweight_needed,
            bias_pack,
        })
    }

    /// The effective LoRA scaling factor this layer applies — see
    /// `lora_scaling`'s doc for what determines it. A pure read of the
    /// value computed at construction (`new` or `from_loaded`), useful for
    /// tests and diagnostics that need to confirm the two constructors agree
    /// bit-for-bit without re-deriving it from a forward pass.
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// Toggle training mode. When `false`, dropout in the LoRA path is skipped
    /// so validation loss and inference outputs are deterministic.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward: `base(x) + scaling * dropout(x @ A^T @ B^T)`.
    ///
    /// ## Eval/serving: the eager composition, unconditionally
    ///
    /// `!self.training` (a `LoraLinear` also SERVES inference —
    /// `from_loaded`, `training: false`) ALWAYS runs the eager `[reshape,
    /// matmul, reshape]`-per-sub-linear composition, byte-for-byte
    /// unchanged from every prior release, regardless of the fused
    /// LoRA-site kernel's existence or domain — no dropout, no admission
    /// check, no dispatch counter touched. This is checked FIRST and
    /// returns immediately, so eval and a training-arm domain miss are
    /// NOT the same code path here (unlike the single-op epilogue this
    /// replaced): eval never even evaluates the fused kernel's domain
    /// predicate.
    ///
    /// ## Training: `jammi_kernels::ops::LowRankResidualLinear`, 9→3 op-nodes
    ///
    /// The training arm routes the ENTIRE site — `base = x @ w^T`, the
    /// dropout draw, both LoRA GEMMs, and the epilogue — through ONE
    /// `CustomOp3` call ([`LowRankResidualLinear`]) when the kernel's own domain
    /// holds (`lora_linear_admission_predicate`): this collapses the
    /// eager composition's 9 op-carrying tape nodes (11 total with the 2
    /// `A`/`B` `Var` leaves; each op-carrying node its own `zeros_like`+
    /// `add` in candle's backward, `GradStore::or_insert`) down to 3
    /// op-carrying nodes (5 total) — this op's own single `CustomOp3`
    /// call, PLUS the `A.t()` view and the `ab`-packing `Tensor::cat`
    /// this collapse does NOT eliminate (both still cost their own node;
    /// disclosed, not folded silently into "one node"). MEASURED (not
    /// estimated) — see [`lora_linear_fused_dispatch_snapshot`]'s own doc
    /// for the exact harness. A bias-carrying base FUSES (#428 P2b) via
    /// `bias_pack`'s trailing block in `ab`, unless the base bias is
    /// itself a trainable `Var` (`bias_is_frozen_leaf`, a COUNTED refusal
    /// — see `bias_gate`'s doc). Outside the fused kernel's domain (that
    /// one bias case, an unsupported dtype/device, a non-contiguous `w`,
    /// an unsupported rank), the training arm falls back to the SAME
    /// `[base matmul, dropout, A-matmul, B-matmul, epilogue]` eager
    /// composition eval uses — see `eager_epilogue` — so a domain miss
    /// reproduces eval's own math exactly, just still gated to `training
    /// == true` (dropout still applies on this fallback, which eval's own
    /// path never runs).
    ///
    /// **Dropout key reservation.** `DropoutMasks::next_key` is called
    /// EXACTLY ONCE per training forward, BEFORE the admission decision —
    /// both the fused arm (passed into [`LowRankResidualLinear`]'s construction
    /// data) and the eager-fallback arm (passed directly to
    /// `DropoutFused::new`) consume the SAME reserved key. Neither arm
    /// calls `DropoutMasks::apply` (which would reserve a SECOND,
    /// different `forward_idx` for the same logical forward) — this is
    /// what keeps esc-033's O(1) resume invariant intact regardless of
    /// which arm a given forward takes: the counter always advances by
    /// exactly one per training forward, never zero (fallback skipping
    /// dropout entirely) and never two (both arms drawing their own key).
    ///
    /// The LoRA-arm dtype `lora_a`/`lora_b` run at (`self.lora_a.dtype()`)
    /// is `F32` in every training-mode call site in this workspace TODAY —
    /// a WORKSPACE FACT about today's call sites
    /// (`ModernBertBuilder::build`'s `lora_vb` construction,
    /// `crates/jammi-encoders/src/modernbert.rs`), not a
    /// `candle_nn::VarBuilder::from_varmap` API guarantee — see
    /// `lora_linear_admission_predicate`'s doc for what this bounds.
    ///
    /// ## `FrozenBase::Quantized` ALWAYS composes — the fused site is Dense-ONLY
    ///
    /// `jammi_kernels::ops::LowRankResidualLinear`'s own domain requires a
    /// dense `Tensor` weight argument (`apply3(x, w, &ab, op)` — `w` is a
    /// plain matmul operand the op's `cpu_fwd`/`cuda_fwd` read directly);
    /// there is no quantized-weight arm of that kernel. A `FrozenBase::
    /// Quantized` base is therefore NEVER even offered to
    /// `lora_linear_admission_predicate` — the training arm branches on
    /// `self.base` FIRST, and only the `Dense` arm ever reaches the
    /// admission/dispatch-counter machinery below. This is a STRUCTURAL
    /// absence, not a domain-check failure: `lora_linear_fused_counters()`
    /// is never touched by a quantized-base forward at all (neither a
    /// `Fused` nor an `Eager` count) — the alternative (routing a
    /// quantized base through `admit()` with a permanently-failing
    /// predicate) would misrepresent "there is no fused kernel for this
    /// storage format" as "the fused kernel's domain check declined this
    /// call", which is a different, weaker claim.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        if !self.training {
            // Eval/serving: always the eager composition, unconditionally
            // — see `forward`'s doc for why this must stay bit-identical
            // regardless of the fused kernel's existence.
            return self.forward_composed(x, None);
        }

        // Training: reserve the dropout key ONCE, before the Dense/Quantized
        // branch and before admission — see `forward`'s doc's "Dropout key
        // reservation" section. Reserved uniformly regardless of base
        // storage format, so esc-033's O(1) resume invariant holds for a
        // quantized base exactly as it does for a dense one.
        let dropout_key: Option<DropoutKey> = match (self.dropout, &self.dropout_masks) {
            (Some(p), Some(masks)) if p > 0.0 => Some(masks.next_key(p)?),
            _ => None,
        };

        // `FrozenBase::Quantized` ALWAYS composes — see this method's own
        // doc, "the fused site is Dense-ONLY".
        let FrozenBase::Dense(base_linear) = &self.base else {
            return self.forward_composed(x, dropout_key);
        };

        let base_has_bias = base_linear.bias().is_some();
        let (holds, predicate) = lora_linear_admission_predicate(
            x,
            base_linear.weight(),
            self.lora_a.dtype(),
            base_has_bias,
            self.bias_pack.is_some(),
        );
        let outcome = admit(
            admission_mode(),
            "lora_linear_fused",
            predicate,
            holds,
            lora_linear_fused_counters(),
        )?;

        match outcome {
            DispatchOutcome::Fused => {
                // Row-packed layout (`jammi_kernels::ops::low_rank_residual_linear`'s
                // module doc, "the packed-`ab` GEMM eligibility problem"):
                // `A^T` (`self.lora_a.t()`) stacked over `B`
                // (`self.lora_b`, no pre-transpose needed) along dim 0,
                // followed — only when `self.bias_pack` is `Some` (#428
                // P2b) — by the pre-packed, zero-padded `[bias_rows,
                // rank]` bias block [`bias_gate`] built ONCE at
                // construction: `[in + out(+bias_rows), rank]`.
                // `self.lora_a.t()` is a non-contiguous VIEW;
                // `Tensor::cat`'s dim-0 path (`cat0`) copies via each arg's
                // own `Layout` regardless (`copy_strided_src`), so no
                // `.contiguous()` call is needed before packing (unlike the
                // column-packed layout this replaced, which needed one for
                // `B^T`).
                let lora_a_t = self.lora_a.t()?;
                let ab = match &self.bias_pack {
                    Some(pack) => Tensor::cat(&[&lora_a_t, &self.lora_b, pack], 0)?,
                    None => Tensor::cat(&[&lora_a_t, &self.lora_b], 0)?,
                };
                let op = LowRankResidualLinear::new(
                    self.scaling as f32,
                    self.in_features,
                    self.out_features,
                    self.rank,
                    dropout_key,
                    self.dweight_needed,
                )?
                .with_bias(self.bias_pack.is_some());
                Ok(apply3(x, base_linear.weight(), &ab, op)?)
            }
            DispatchOutcome::Eager => self.forward_composed(x, dropout_key),
        }
    }

    /// The shared `[base, dropout, A-matmul, B-matmul, epilogue]`
    /// composition — used by eval (`dropout_key == None`, always), the
    /// Dense eager-fallback arm, and EVERY `Quantized`-base training
    /// forward (see `forward`'s own doc). `self.base.forward(x)` is
    /// [`FrozenBase::forward`] — Dense's cast-to-weight-dtype-then-forward,
    /// preserved byte-for-byte from every prior release; Quantized's
    /// uniform F32 rule.
    fn forward_composed(
        &self,
        x: &Tensor,
        dropout_key: Option<DropoutKey>,
    ) -> Result<Tensor, LoraError> {
        let base_out = self.base.forward(x)?;

        let lora_dtype = self.lora_a.dtype();
        let x_lora = if x.dtype() != lora_dtype {
            x.to_dtype(lora_dtype)?
        } else {
            x.clone()
        };
        let lora_in = match dropout_key {
            Some(key) => {
                let op = DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p)?;
                apply1(&x_lora, op)?
            }
            None => x_lora,
        };

        let a_lin = Linear::new(self.lora_a.clone(), None);
        let after_a = a_lin.forward(&lora_in)?;
        let b_lin = Linear::new(self.lora_b.clone(), None);
        let lora_out = b_lin.forward(&after_a)?;
        eager_epilogue(&base_out, &lora_out, self.scaling)
    }

    /// References to the two trainable LoRA parameter tensors.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// The FROZEN base weight this layer wraps — read access only (a
    /// `LoraLinear`'s base is immutable after construction, matching
    /// `candle_nn::Linear`'s own convention). Reached from outside this
    /// crate through [`crate::MaybeLoraLinear::base`], which is what a
    /// consumer that must inspect a site's base tensor regardless of
    /// whether an adapter is installed on it actually calls.
    pub(crate) fn base(&self) -> &FrozenBase {
        &self.base
    }

    /// This layer's dropout forward counter — the number of TRAINING
    /// FORWARDS taken through it (NOT a draw count; see
    /// `jammi-ai/src/fine_tune/resume.rs`'s schema-version doc for the unit
    /// change this commit makes) — or `None` when the layer has no dropout
    /// mask source (`lora_dropout == 0`). It is the resume state for the
    /// layer's dropout: a resumed run sets its counter to this position
    /// (O(1) — an assignment, not a replay; closes esc-033) so its next
    /// masks byte-match the uninterrupted run.
    pub fn dropout_position(&self) -> Result<Option<u64>, LoraError> {
        Ok(self.dropout_masks.as_ref().map(DropoutMasks::position))
    }

    /// Restore this layer's dropout forward counter to `position` — O(1),
    /// an assignment (see [`Self::dropout_position`]'s doc) — so the next
    /// training forwards draw the same masks the uninterrupted run drew. A
    /// no-op when the layer has no dropout mask source.
    pub fn restore_dropout_position(&self, position: u64) -> Result<(), LoraError> {
        if let Some(masks) = &self.dropout_masks {
            masks.restore_position(position);
        }
        Ok(())
    }
}

#[cfg(test)]
mod frozen_weight_gate_tests {
    use super::frozen_weight_gate;
    use candle_core::{Device, Tensor, Var};

    /// The ordinary case: a weight loaded straight from a `VarBuilder`
    /// (a true untracked leaf) — `dweight_needed` must be `false`.
    #[test]
    fn untracked_leaf_is_false() {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (3, 4), &device).unwrap();
        assert!(!w.is_variable() && !w.track_op());
        assert!(!frozen_weight_gate(&w).unwrap());
    }

    /// The "also fine-tune the base" case: `w` is itself a trainable
    /// `Var` — `dweight_needed` must be `true`. Per candle-core 0.11's
    /// `Tensor::track_op` (`is_variable() || op.is_some()`), a `Var`
    /// DOES report `track_op() == true` — this is exactly why
    /// `is_variable()` must be checked FIRST (see `frozen_weight_gate`'s
    /// own doc).
    #[test]
    fn trainable_var_is_true() {
        let device = Device::Cpu;
        let w = Var::from_tensor(&Tensor::randn(0f32, 1.0, (3, 4), &device).unwrap()).unwrap();
        assert!(w.as_tensor().is_variable());
        assert!(w.as_tensor().track_op());
        assert!(frozen_weight_gate(w.as_tensor()).unwrap());
    }

    /// The refused, ambiguous case: `w` carries an `Op` (e.g. a
    /// `Tensor::to_dtype` cast applied after loading) but is NOT a `Var`
    /// — neither "definitely frozen" nor "definitely trainable"; a typed
    /// refusal, not a silent guess.
    #[test]
    fn tracked_non_var_is_a_typed_refusal() {
        let device = Device::Cpu;
        // `BackpropOp::new1` (candle-core 0.11's `op.rs:1100-1107`) only
        // attaches an `Op` when its OWN argument already `track_op()`s —
        // a plain leaf `Tensor::randn(..)` does not, so casting IT would
        // produce another untracked leaf, not the ambiguous state this
        // test targets. Starting from a `Var` (which DOES `track_op()`)
        // and casting to a DIFFERENT dtype (a same-dtype cast
        // short-circuits to `self.clone()`, `tensor.rs:2453-2461`, and
        // would just return the `Var` itself) produces a tensor that is
        // TRACKED (inherits `track_op()` from its `Var` input) but is
        // itself NOT a `Var` — exactly the case `frozen_weight_gate`
        // must refuse.
        let w = Var::from_tensor(&Tensor::randn(0f32, 1.0, (3, 4), &device).unwrap()).unwrap();
        let tracked = w.as_tensor().to_dtype(candle_core::DType::F64).unwrap();
        assert!(!tracked.is_variable());
        assert!(tracked.track_op(), "fixture must actually be tracked");
        let err = frozen_weight_gate(&tracked).unwrap_err();
        assert!(matches!(err, crate::error::LoraError::Config(_)));
    }
}

/// #428 P2b: [`bias_gate`]'s own three-way (plus "no bias") lattice —
/// mirrors `frozen_weight_gate_tests`'s shape, one test per cell.
#[cfg(test)]
mod bias_gate_tests {
    use super::bias_gate;
    use candle_core::{DType, Device, Tensor, Var};

    #[test]
    fn no_bias_is_none() {
        assert!(bias_gate(None, DType::F32, 4, 2).unwrap().is_none());
    }

    /// An untracked leaf (loaded straight from a `VarBuilder`) produces
    /// `Some(pack)`, `[bias_rows, rank]`, zero-padded past `out_features`.
    #[test]
    fn untracked_leaf_produces_a_zero_padded_pack() {
        let device = Device::Cpu;
        let (out_features, rank) = (5usize, 2usize); // bias_rows = 3.
        let bias =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], out_features, &device).unwrap();
        assert!(!bias.is_variable() && !bias.track_op());
        let pack = bias_gate(Some(&bias), DType::F32, out_features, rank)
            .unwrap()
            .expect("an untracked leaf must produce a pack");
        assert_eq!(pack.dims(), &[3, rank]);
        let flat: Vec<f32> = pack.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
    }

    /// A trainable `Var` bias produces `None` — NOT because there is no
    /// bias (the caller must distinguish these two `None`s via
    /// `base_linear.bias().is_some()`, per `lora_linear_admission_predicate`'s
    /// own doc), but because the eager composition already tracks it
    /// correctly and the fused kernel has no trainable-bias slot.
    #[test]
    fn trainable_var_is_none() {
        let device = Device::Cpu;
        let bias = Var::from_tensor(&Tensor::from_slice(&[1.0f32, 2.0, 3.0], 3, &device).unwrap())
            .unwrap();
        assert!(bias_gate(Some(bias.as_tensor()), DType::F32, 3, 2)
            .unwrap()
            .is_none());
    }

    /// Tracked-but-not-`Var` (the same ambiguous state
    /// `frozen_weight_gate` refuses for `w`) is a typed refusal.
    #[test]
    fn tracked_non_var_is_a_typed_refusal() {
        let device = Device::Cpu;
        let bias_var =
            Var::from_tensor(&Tensor::from_slice(&[1.0f32, 2.0, 3.0], 3, &device).unwrap())
                .unwrap();
        let tracked = bias_var.as_tensor().to_dtype(DType::F64).unwrap();
        assert!(!tracked.is_variable() && tracked.track_op());
        let err = bias_gate(Some(&tracked), DType::F64, 3, 2).unwrap_err();
        assert!(matches!(err, crate::error::LoraError::Config(_)));
    }

    #[test]
    fn wrong_shape_is_a_typed_refusal() {
        let device = Device::Cpu;
        let bias = Tensor::from_slice(&[1.0f32, 2.0], 2, &device).unwrap();
        let err = bias_gate(Some(&bias), DType::F32, 3, 2).unwrap_err();
        assert!(matches!(err, crate::error::LoraError::Config(_)));
    }

    #[test]
    fn mismatched_dtype_is_a_typed_refusal() {
        let device = Device::Cpu;
        let bias = Tensor::from_slice(&[1.0f32, 2.0, 3.0], 3, &device).unwrap();
        let err = bias_gate(Some(&bias), DType::F64, 3, 2).unwrap_err();
        assert!(matches!(err, crate::error::LoraError::Config(_)));
    }

    /// `out_features` an exact multiple of `rank` (`bias_rows * rank ==
    /// out_features`, `pad == 0`) is a normal, covered case —
    /// `Tensor::pad_with_zeros`'s own `left == 0 && right == 0` fast path.
    #[test]
    fn exact_multiple_of_rank_needs_no_padding() {
        let device = Device::Cpu;
        let (out_features, rank) = (4usize, 2usize); // bias_rows = 2, no padding.
        let bias = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], out_features, &device).unwrap();
        let pack = bias_gate(Some(&bias), DType::F32, out_features, rank)
            .unwrap()
            .unwrap();
        assert_eq!(pack.dims(), &[2, rank]);
        let flat: Vec<f32> = pack.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }
}

/// `LoraLinear::forward`'s `FrozenBase::Quantized` branch — see that
/// method's own doc, "`FrozenBase::Quantized` ALWAYS composes". Isolated
/// from `tests/fused_epilogue.rs`'s own dispatch-counter tests
/// deliberately: this crate's LIB test binary (`jammi_lora-*`, what
/// `cargo test -p jammi-lora` runs `#[cfg(test)]` code as) has NO OTHER
/// test that calls `LoraLinear::forward` at all (`frozen_weight_gate_tests`/
/// `lora_scaling_tests`/`eager_epilogue_tests` all exercise narrower
/// functions directly), so a before/after `lora_linear_fused_counters()`
/// snapshot EQUALITY assertion here is race-free under `cargo test`'s
/// default concurrent-test-thread execution — an integration-test-level
/// version of this same claim would NOT be safe (see
/// `tests/fused_epilogue.rs`'s `esc_031_quantized_twin` module's own doc
/// for why: sibling tests IN THAT FILE deliberately increment the same
/// process-global counter for their own Dense-base assertions).
#[cfg(test)]
mod quantized_base_forward_tests {
    use super::{lora_linear_fused_counters, LoraLinear};
    use crate::frozen_base::{FrozenBase, QuantizedLinear};
    use crate::init::LoraInitMode;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_nn::VarMap;
    use std::sync::Arc;

    fn quantized_base(out_f: usize, in_f: usize) -> FrozenBase {
        let device = Device::Cpu;
        let w_v: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f64) * 0.029 + 0.7).sin() as f32)
            .collect();
        let w = Tensor::from_vec(w_v, (out_f, in_f), &device).unwrap();
        let q = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
        FrozenBase::Quantized(QuantizedLinear::new(Arc::new(q), None).unwrap())
    }

    #[test]
    fn quantized_base_forward_never_touches_the_fused_dispatch_counters() {
        let device = Device::Cpu;
        let (out_f, in_f, rows) = (4usize, 32usize, 2usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let lora = LoraLinear::new_with_base(
            quantized_base(out_f, in_f),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            11,
            &varmap,
            &vb,
        )
        .unwrap();
        let x_v: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i as f64) * 0.013 + 0.2).cos() as f32)
            .collect();
        let x = Tensor::from_vec(x_v, (rows, in_f), &device).unwrap();

        let before = lora_linear_fused_counters().snapshot();
        let _ = lora.forward(&x).unwrap();
        let after = lora_linear_fused_counters().snapshot();
        assert_eq!(
            (before.fused, before.eager),
            (after.fused, after.eager),
            "a Quantized base must never touch the (Dense-only) fused-site dispatch \
             counters — before={before:?}, after={after:?}"
        );
    }

    /// `dweight_needed` is `false` for a `LoraLinear` constructed over a
    /// `Quantized` base — mirrors `FrozenBase::dweight_needed`'s own unit
    /// test (`frozen_base.rs`), pinned again here through the full
    /// `LoraLinear::new_with_base` construction path.
    #[test]
    fn dweight_needed_is_false_through_new_with_base_over_a_quantized_base() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let lora = LoraLinear::new_with_base(
            quantized_base(4, 32),
            4,
            8.0,
            false,
            LoraInitMode::ZerosB,
            None,
            13,
            &varmap,
            &vb,
        )
        .unwrap();
        assert!(!lora.dweight_needed);
    }
}

#[cfg(test)]
mod lora_scaling_tests {
    use super::lora_scaling;

    /// Table-driven, bit-exact pin of [`lora_scaling`] against its two
    /// documented formulas (vanilla `alpha/rank`, rsLoRA `alpha/sqrt(rank)`)
    /// — the numpy-first oracle here is the closed-form arithmetic itself,
    /// computed independently of the function under test on each row.
    #[test]
    fn table_pins_vanilla_and_rslora_formulas_bit_exact() {
        let cases: &[(f64, usize, bool)] = &[
            (8.0, 4, false),
            (8.0, 4, true),
            (16.0, 8, false),
            (16.0, 8, true),
            (1.0, 1, false),
            (1.0, 1, true),
            (32.0, 16, false),
            (32.0, 16, true),
            (0.0, 4, false), // alpha == 0 is a valid, if inert, scaling.
            (0.0, 4, true),
        ];
        for &(alpha, rank, use_rslora) in cases {
            let expected = if use_rslora {
                alpha / (rank as f64).sqrt()
            } else {
                alpha / rank as f64
            };
            let got = lora_scaling(alpha, rank, use_rslora).unwrap();
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "alpha={alpha} rank={rank} use_rslora={use_rslora}: got {got}, expected {expected}"
            );
        }
    }

    /// Domain boundary (family D / K2): `rank == 0` must be a typed refusal,
    /// not `alpha / 0` silently propagating as `inf`/`NaN`. Both `use_rslora`
    /// arms are checked so neither the vanilla nor the rsLoRA branch is
    /// reachable with a zero rank.
    #[test]
    fn rank_zero_is_a_typed_refusal_both_arms() {
        for use_rslora in [false, true] {
            let err = lora_scaling(8.0, 0, use_rslora).unwrap_err();
            assert!(
                matches!(err, crate::error::LoraError::Config(_)),
                "expected a Config refusal for rank=0, got {err:?}"
            );
        }
    }

    /// Non-finite `alpha` (family F non-vacuity: a naive `> c` bound is
    /// `false` for `NaN`, so this asserts on the actual bit pattern instead
    /// of a comparison a `NaN` could vacuously dodge) propagates as a
    /// `NaN`/`inf` scaling rather than being silently coerced — `rank`, not
    /// `alpha`, is this function's validated domain edge; a caller passing a
    /// non-finite `alpha` gets a non-finite scaling back, visibly.
    #[test]
    fn non_finite_alpha_propagates_visibly_not_silently() {
        let got = lora_scaling(f64::NAN, 4, false).unwrap();
        assert!(got.is_nan(), "NaN alpha must yield a visible NaN scaling");
    }
}

/// esc-046 (GH#374) — `eager_epilogue` itself, CPU-hermetic, exercised
/// directly (not through `LoraLinear::forward`'s dispatch — the
/// production-width, real-dispatch biting oracle with `DispatchCounters`
/// live in `crates/jammi-lora/tests/esc046_epilogue_biting_oracle.rs`,
/// this crate's own integration-test tier). Both tests here compare
/// against a truth built from candle's own (trusted, generic) `Tensor`
/// arithmetic and `to_dtype` cast — NEVER a re-implementation of
/// `eager_epilogue`'s own logic — so a regression to the pre-fix
/// round-before-add model would fail these, not silently agree with a
/// copy of itself.
#[cfg(test)]
mod eager_epilogue_tests {
    use super::eager_epilogue;
    use candle_core::{DType, Device, Tensor};

    /// Widening a `BF16` tensor to `F32` (`Tensor::to_dtype`) is EXACT —
    /// `bf16` has strictly fewer significand bits than `f32`, so every
    /// `bf16` value is exactly representable in `f32` and the widen
    /// introduces no rounding at all (only narrowing, the other
    /// direction, rounds). Comparing the widened `f32` values for
    /// equality is therefore equivalent to comparing the underlying
    /// `bf16` bit patterns directly, without this crate needing its own
    /// dependency on the `half` crate (a candle-free workaround; `Cargo.toml`
    /// is the lead/docs-ci shared-declaration class, not freely editable
    /// here).
    fn widen_to_f32(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32).unwrap().to_vec1().unwrap()
    }

    /// Hand-verified small fixture, the same "opposite sides of a bf16
    /// rounding boundary" discipline
    /// `scaled_cast_add_oracles.rs`'s `fused_vs_eager_bf16_base_f32_lora_fwd_and_bwd_are_bit_exact_on_a_divergent_fixture`
    /// uses: `base = 1.0078125` (exactly bf16-representable), `delta_f32
    /// = 2.2508249282836914` (`22.508249282836914 * 0.1`, the SAME f32
    /// bit pattern that test hand-verifies). Round-before-add would round
    /// `delta_f32` to bf16 (`2.25`) FIRST, landing the sum EXACTLY halfway
    /// between two bf16 grid points (`3.2578125`, resolved by
    /// round-to-even to `3.25`); f32-accumulate sums first
    /// (`3.2586374282836914`) and rounds once to `3.265625` — the two
    /// models disagree by exactly one bf16 ULP on this element.
    // `22.508249282836914` is the exact decimal expansion of one specific
    // f32 bit pattern, verified against `scaled_cast_add_oracles.rs`'s own
    // hand computation — kept at full precision rather than clippy's own
    // suggested truncation (`22.508_25`) so nothing here risks silently
    // landing on a DIFFERENT f32 value than the one this test's documented
    // hand computation is actually about.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn eager_epilogue_matches_hand_computed_peft_rounding_on_a_divergent_fixture() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[1.0078125_f32], (1,), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        // `lora_out = 22.508249282836914`, `scaling = 0.1` — matches
        // `eager_epilogue`'s own `lora_out * scaling` step, isolating the
        // multiply from the add-then-round the fixture targets.
        let lora_out = Tensor::from_slice(&[22.508249282836914_f32], (1,), &device).unwrap();

        let got = eager_epilogue(&base, &lora_out, 0.1).unwrap();
        assert_eq!(got.dtype(), DType::BF16);
        let got_v = widen_to_f32(&got);
        assert_eq!(
            got_v[0], 3.265625,
            "eager_epilogue must match the once-rounded (PEFT) model's hand-computed value"
        );
        assert_ne!(
            got_v[0], 3.25,
            "eager_epilogue must NOT match the rejected round-before-add model's value \
             (a regression back to it would silently pass without this assertion)"
        );
    }

    /// Production-width (`n = 4096`) sweep — the same amplitude regime
    /// esc-046's own lead-measured reproduction and
    /// `tests/scaled_cast_add_peft_rounding.rs` use (`base` amplitude
    /// ~100 — a bf16-rounded GEMM-output scale — `delta` amplitude ~3, a
    /// scaled LoRA contribution). Deterministic trig fixture (the same
    /// idiom `cast_scale.rs`'s own production-amplitude tests use, e.g.
    /// `cast_add_bit_identical_to_the_eager_two_kernel_chain_at_production_amplitude`)
    /// rather than a from-scratch PRNG — family L: no untracked external
    /// generator, and no need to re-derive Box-Muller when a closed form
    /// already covers "wide, non-tidy" values.
    #[test]
    fn eager_epilogue_matches_peft_rounding_at_production_width() {
        const N: usize = 4096;
        let device = Device::Cpu;
        let base_v: Vec<f32> = (0..N)
            .map(|i| ((i as f32 * 0.0173).sin()) * 100.0)
            .collect();
        let delta_v: Vec<f32> = (0..N).map(|i| ((i as f32 * 0.0611).cos()) * 3.0).collect();

        let base_f32 = Tensor::from_slice(&base_v, (N,), &device).unwrap();
        let base_bf16 = base_f32.to_dtype(DType::BF16).unwrap();
        let delta_f32 = Tensor::from_slice(&delta_v, (N,), &device).unwrap();

        // The real function under test. `scaling = 1.0` so `lora_out *
        // scaling == delta_f32` unchanged, isolating the epilogue's own
        // add-then-round from the (separately tested) multiply.
        let got = eager_epilogue(&base_bf16, &delta_f32, 1.0).unwrap();

        // PEFT-ordered truth, built from candle's own `Tensor` arithmetic
        // and `to_dtype` cast directly (never `eager_epilogue`'s own
        // code): widen `base` to `f32` (lossless), add the `f32` delta,
        // round to `bf16` ONCE.
        let peft_truth = (base_bf16.to_dtype(DType::F32).unwrap() + &delta_f32)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // The REJECTED, pre-esc-046 formula: round the delta to `bf16`
        // FIRST, then add-and-round again.
        let delta_rounded_first = delta_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let mis_ordered = (base_bf16.to_dtype(DType::F32).unwrap() + &delta_rounded_first)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // `base_bf16`'s OWN (already-rounded) value — widening it back to
        // `f32` is exact — is the real additive target `eager_epilogue`
        // rounds against, NOT the pre-bf16-rounding synthetic `base_v[i]`
        // (which is not itself bf16-representable in general): control
        // (b)'s `truth_f64` must be built from what the function under
        // test actually received, or the comparison is measuring the
        // wrong quantity.
        let base_bf16_v = widen_to_f32(&base_bf16);
        let got_v = widen_to_f32(&got);
        let truth_v = widen_to_f32(&peft_truth);
        let mis_v = widen_to_f32(&mis_ordered);

        // Non-finite counts as a mismatch, written affirmatively, BEFORE
        // any comparison (family F).
        for i in 0..N {
            assert!(
                base_v[i].is_finite()
                    && delta_v[i].is_finite()
                    && got_v[i].is_finite()
                    && truth_v[i].is_finite()
                    && mis_v[i].is_finite(),
                "index {i}: a non-finite value slipped through"
            );
        }

        // NON-VACUITY: the fixture must actually separate the two
        // candidate formulas — computed from the two hand/candle-derived
        // reference formulas alone, independent of what `eager_epilogue`
        // itself returns.
        let discriminating = (0..N).filter(|&i| truth_v[i] != mis_v[i]).count();
        assert!(
            discriminating >= 20,
            "fixture is not discriminating: only {discriminating}/{N} elements separate the \
             once-rounded formula from the round-then-add one — this fixture would read GREEN \
             on a broken build regardless of eager_epilogue's own logic"
        );

        // DEFECT (post-fix: GREEN). Raw value equality on the lossless
        // bf16-widened-to-f32 representation, never a tolerance.
        let mismatches: Vec<usize> = (0..N).filter(|&i| got_v[i] != truth_v[i]).collect();
        assert!(
            mismatches.is_empty(),
            "eager_epilogue does NOT match PEFT's rounding order on {}/{N} elements (esc-046) \
             — first mismatch idx={} base={} delta={} got={:?} peft_truth={:?}",
            mismatches.len(),
            mismatches[0],
            base_v[mismatches[0]],
            delta_v[mismatches[0]],
            got_v[mismatches[0]],
            truth_v[mismatches[0]],
        );

        // Control (a) POWER OF THE COMPARISON: the rejected model must
        // itself genuinely diverge from the real function's output on the
        // differing elements (re-derived here from `got`, not merely from
        // the two reference formulas above) — otherwise a RED reading
        // pre-fix could be an artifact of fold-order noise, not rounding
        // placement.
        let got_vs_mis = (0..N).filter(|&i| got_v[i] != mis_v[i]).count();
        assert!(
            got_vs_mis >= 20,
            "control (a) void: eager_epilogue's real output and the rejected round-before-add \
             model must diverge on >= 20 elements for a RED-on-old-code reading to mean \
             anything; measured {got_vs_mis}"
        );

        // Control (b) F32-TRUTH DIRECTION: on exactly the elements where
        // the two candidate formulas disagree, the once-rounded
        // (produced) value must be no farther from f64 truth than the
        // round-then-add (rejected) value is, strict on at least one.
        let mut strict_improvements = 0usize;
        let mut violations = 0usize;
        for i in 0..N {
            if truth_v[i] == mis_v[i] {
                continue;
            }
            let truth_f64 = f64::from(base_bf16_v[i]) + f64::from(delta_v[i]);
            let once_err = (f64::from(got_v[i]) - truth_f64).abs();
            let old_err = (f64::from(mis_v[i]) - truth_f64).abs();
            if once_err > old_err + 1e-9 {
                violations += 1;
            }
            if once_err + 1e-9 < old_err {
                strict_improvements += 1;
            }
        }
        assert_eq!(
            violations, 0,
            "control (b) violated: on {violations} differing elements the once-rounded value \
             is FARTHER from f64 truth than the round-then-add value"
        );
        assert!(
            strict_improvements >= 1,
            "control (b) is vacuous: the once-rounded value must be STRICTLY closer to f64 \
             truth than the round-then-add value on at least one differing element"
        );
    }

    /// Independent reference for `eager_epilogue`'s dtype-promotion rule —
    /// built directly from candle `Tensor` ops, ALWAYS explicitly widening
    /// to `f32` (hardcoded here, never derived from
    /// `wider_float_dtype`/`eager_epilogue` itself, and never candle's
    /// native `bf16` `Tensor::add` — see `eager_epilogue`'s own doc for
    /// why). `scaled = lora_out * scaling` is left in `lora_out`'s OWN
    /// dtype (torch's "weak Python-scalar" tensor-scalar promotion rule:
    /// `bf16_tensor * python_float` stays `bf16`) before being widened for
    /// the add, matching what `eager_epilogue` itself does.
    fn reference_eager_epilogue(
        base: &Tensor,
        lora_out: &Tensor,
        scaling: f64,
        base_dtype: DType,
    ) -> Vec<f32> {
        let scaled = (lora_out * scaling).unwrap();
        let base_f32 = base.to_dtype(DType::F32).unwrap();
        let scaled_f32 = scaled.to_dtype(DType::F32).unwrap();
        let sum_f32 = (&base_f32 + &scaled_f32).unwrap();
        sum_f32
            .to_dtype(base_dtype)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// esc-046 audit round 2, finding 1 (`lora_linear.rs:103`): every cell
    /// of the base/lora dtype lattice, each checked bit-for-bit against
    /// [`reference_eager_epilogue`] at production width (`n = 4096`). The
    /// `(F32 base, BF16 lora)` cell is the one the audit found NARROWED
    /// the base to `bf16` before the add (a regression this test would
    /// catch — see the companion non-vacuity test below for the RED
    /// proof). The `(BF16, BF16)` cell is included to confirm this
    /// function never reaches candle's native (size-dependent) `bf16`
    /// `Tensor::add`.
    #[test]
    fn eager_epilogue_matches_the_wider_dtype_reference_across_the_full_dtype_lattice() {
        const N: usize = 4096;
        let device = Device::Cpu;
        let base_v: Vec<f32> = (0..N)
            .map(|i| ((i as f32 * 0.0173).sin()) * 100.0)
            .collect();
        let delta_v: Vec<f32> = (0..N).map(|i| ((i as f32 * 0.0611).cos()) * 3.0).collect();

        for &(base_dtype, lora_dtype) in &[
            (DType::BF16, DType::F32), // the esc-046 pair
            (DType::F32, DType::BF16), // audit finding 1's narrowing bug
            (DType::F32, DType::F32),
            (DType::BF16, DType::BF16),
        ] {
            let base_f32 = Tensor::from_slice(&base_v, (N,), &device).unwrap();
            let base = base_f32.to_dtype(base_dtype).unwrap();
            let lora_f32 = Tensor::from_slice(&delta_v, (N,), &device).unwrap();
            let lora_out = lora_f32.to_dtype(lora_dtype).unwrap();

            let got = eager_epilogue(&base, &lora_out, 1.0).unwrap();
            assert_eq!(
                got.dtype(),
                base_dtype,
                "{base_dtype:?}/{lora_dtype:?}: output must be base's own dtype"
            );
            let got_v: Vec<f32> = got.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
            let expected_v = reference_eager_epilogue(&base, &lora_out, 1.0, base_dtype);

            for i in 0..N {
                assert!(
                    got_v[i].is_finite() && expected_v[i].is_finite(),
                    "{base_dtype:?}/{lora_dtype:?} index {i}: a non-finite value slipped through \
                     (got={} expected={})",
                    got_v[i],
                    expected_v[i]
                );
            }
            assert_eq!(
                got_v, expected_v,
                "{base_dtype:?}/{lora_dtype:?}: eager_epilogue must match the wider-dtype \
                 reference bit-for-bit"
            );
        }
    }

    /// Non-vacuity companion to the lattice test above, for the specific
    /// `(F32 base, BF16 lora)` pair the audit's finding 1 targeted: the
    /// REJECTED, narrow-first formula (`base.to_dtype(scaled.dtype())`
    /// before the add — the audit's own `lora_linear.rs:103`, the exact
    /// pre-round-2 production code) must diverge substantially from the
    /// wider-dtype reference on this fixture, proving the lattice test
    /// above is non-vacuous for this cell: a regression back to
    /// narrowing-first would fail it, not silently pass.
    #[test]
    fn eager_epilogue_f32_base_bf16_lora_would_diverge_under_the_narrow_first_regression() {
        const N: usize = 4096;
        let device = Device::Cpu;
        let base_v: Vec<f32> = (0..N)
            .map(|i| ((i as f32 * 0.0173).sin()) * 100.0)
            .collect();
        let delta_v: Vec<f32> = (0..N).map(|i| ((i as f32 * 0.0611).cos()) * 3.0).collect();

        let base = Tensor::from_slice(&base_v, (N,), &device).unwrap(); // F32
        let lora_f32 = Tensor::from_slice(&delta_v, (N,), &device).unwrap();
        let lora_out = lora_f32.to_dtype(DType::BF16).unwrap();

        let scaled = (&lora_out * 1.0).unwrap();
        // The REJECTED, audit-flagged formula: promote base to `scaled`'s
        // OWN dtype (narrowing f32 -> bf16) instead of the wider of the
        // two.
        let narrowed_sum = (&base.to_dtype(scaled.dtype()).unwrap() + &scaled).unwrap();
        let narrow_first_v: Vec<f32> = narrowed_sum
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        let expected_v = reference_eager_epilogue(&base, &lora_out, 1.0, DType::F32);

        let discriminating = (0..N)
            .filter(|&i| narrow_first_v[i] != expected_v[i])
            .count();
        assert!(
            discriminating >= N * 9 / 10,
            "fixture is not discriminating enough for the narrow-first regression: only \
             {discriminating}/{N} elements would separate it from the wider-dtype reference"
        );
        let max_err = (0..N)
            .map(|i| (f64::from(narrow_first_v[i]) - f64::from(expected_v[i])).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err > 0.1,
            "fixture's narrow-first divergence is too small to be a meaningful regression \
             proof: {max_err}"
        );

        // GREEN: the REAL eager_epilogue must NOT reproduce the
        // narrow-first formula — it must match the reference instead
        // (re-asserted here, self-contained, not just via the lattice
        // test above).
        let got = eager_epilogue(&base, &lora_out, 1.0).unwrap();
        let got_v: Vec<f32> = got.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        assert_eq!(got_v, expected_v);
        let mismatches_vs_narrow = (0..N).filter(|&i| got_v[i] != narrow_first_v[i]).count();
        assert!(
            mismatches_vs_narrow >= N * 9 / 10,
            "the real eager_epilogue must diverge from the narrow-first (rejected) formula on \
             most elements; measured {mismatches_vs_narrow}/{N}"
        );
    }
}
