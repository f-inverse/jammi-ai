//! Single LoRA-augmented linear layer: frozen base + trainable A and B matrices.

use candle_core::{DType, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
    DispatchSnapshot,
};
use jammi_kernels::ops::{apply1, apply3, DropoutFused, DropoutKey, LowRankResidualLinear};

use crate::error::LoraError;
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

/// The eager `[mul, cast, add]` composition the fused epilogue replaces:
/// `base_out + cast_to(base_out.dtype())(lora_out * scaling)`. Kept as its
/// own function so both the eval-mode path (which always uses it — see
/// `forward`'s doc) and the training-mode fallback (when the fused
/// kernel's domain does not hold) share exactly one implementation.
fn eager_epilogue(base_out: &Tensor, lora_out: &Tensor, scaling: f64) -> Result<Tensor, LoraError> {
    let scaled = (lora_out * scaling)?;
    let scaled_cast = if scaled.dtype() != base_out.dtype() {
        scaled.to_dtype(base_out.dtype())?
    } else {
        scaled
    };
    Ok((base_out + &scaled_cast)?)
}

/// Per-op fused/eager dispatch counts for the fused LoRA SITE
/// (`jammi_kernels::ops::LowRankResidualLinear`, the whole-site fusion),
/// read from the same op-keyed registry `lora_epilogue_counters` uses.
/// MEASURED (not estimated) at the production `LoraLinear::forward` path
/// (`rank`-3 `x`, `F32`, `dropout = 0.3`, a frozen `w`) via
/// `Tensor::sorted_nodes().len()`: the fused arm retains 5 tape nodes end
/// to end (3 OP-CARRYING — `A.t()`, the `ab` pack `Op::Cat`, and this
/// op's own `CustomOp3` call — plus the 2 `Var` leaves, `A`/`B`) versus
/// 9 op-carrying nodes (11 total) for the eager composition
/// `eager_epilogue` and its own `A`/`B`/dropout sub-linears build —
/// see `crates/jammi-lora/tests/fused_epilogue.rs`'s
/// `production_path_retains_fewer_tape_nodes_fused_vs_eager_fallback` for
/// the harness these numbers come from. Every op-carrying node is one
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
/// that is EITHER both `F32` or both `BF16` (the two combinations
/// [`jammi_kernels::ops::LowRankResidualLinear`] actually implements); `w`
/// contiguous (`x` is NOT required to be — the op materializes a
/// non-contiguous `x` internally; see the op's own domain doc); the base
/// weight carries no bias (see
/// [`jammi_kernels::ops::low_rank_residual_linear`]'s module doc for why a bias is a
/// domain refusal here rather than packed into `ab`). `out_features >= 1`
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
    has_bias: bool,
) -> (bool, &'static str) {
    if has_bias {
        return (false, "base_has_no_bias");
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
        (DType::F32, DType::F32) | (DType::BF16, DType::BF16) => {}
        _ => return (false, "base_dtype_f32_or_bf16_matched"),
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
fn frozen_weight_gate(w: &Tensor) -> Result<bool, LoraError> {
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
    base: Linear,
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
        if rank == 0 {
            return Err(LoraError::Config("LoRA rank must be > 0".into()));
        }
        let in_features = base.weight().dim(1)?;
        let out_features = base.weight().dim(0)?;
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

        let dweight_needed = frozen_weight_gate(base.weight())?;

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
        let rank = lora_a.dims()[0];
        let scaling = lora_scaling(alpha, rank, use_rslora)?;
        let in_features = base.weight().dim(1)?;
        let out_features = base.weight().dim(0)?;
        let dweight_needed = frozen_weight_gate(base.weight())?;
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
    /// for the exact harness. Outside the fused kernel's domain (a bias-carrying
    /// base, an unsupported dtype/device, a non-contiguous view, an
    /// unsupported rank), the training arm falls back to the SAME `[base
    /// matmul, dropout, A-matmul, B-matmul, epilogue]` eager composition
    /// eval uses — see `eager_epilogue` — so a domain miss reproduces
    /// eval's own math exactly, just still gated to `training == true`
    /// (dropout still applies on this fallback, which eval's own path
    /// never runs).
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
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        if !self.training {
            // Eval/serving: always the eager composition, unconditionally
            // — see `forward`'s doc for why this must stay bit-identical
            // regardless of the fused kernel's existence.
            let base_dtype = self.base.weight().dtype();
            let x_base = if x.dtype() == base_dtype {
                x.clone()
            } else {
                x.to_dtype(base_dtype)?
            };
            let base_out = self.base.forward(&x_base)?;
            let lora_dtype = self.lora_a.dtype();
            let x_lora = if x.dtype() != lora_dtype {
                x.to_dtype(lora_dtype)?
            } else {
                x.clone()
            };
            let a_lin = Linear::new(self.lora_a.clone(), None);
            let after_a = a_lin.forward(&x_lora)?;
            let b_lin = Linear::new(self.lora_b.clone(), None);
            let lora_out = b_lin.forward(&after_a)?;
            return eager_epilogue(&base_out, &lora_out, self.scaling);
        }

        // Training: reserve the dropout key ONCE, before admission — see
        // `forward`'s doc's "Dropout key reservation" section.
        let dropout_key: Option<DropoutKey> = match (self.dropout, &self.dropout_masks) {
            (Some(p), Some(masks)) if p > 0.0 => Some(masks.next_key(p)?),
            _ => None,
        };

        let has_bias = self.base.bias().is_some();
        let (holds, predicate) =
            lora_linear_admission_predicate(x, self.base.weight(), self.lora_a.dtype(), has_bias);
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
                // (`self.lora_b`, no pre-transpose needed) along dim 0 —
                // `[in + out, rank]`. `self.lora_a.t()` is a non-contiguous
                // VIEW; `Tensor::cat`'s dim-0 path (`cat0`) copies via each
                // arg's own `Layout` regardless (`copy_strided_src`), so no
                // `.contiguous()` call is needed before packing (unlike the
                // column-packed layout this replaced, which needed one for
                // `B^T`).
                let ab = Tensor::cat(&[&self.lora_a.t()?, &self.lora_b], 0)?;
                let op = LowRankResidualLinear::new(
                    self.scaling as f32,
                    self.in_features,
                    self.out_features,
                    self.rank,
                    dropout_key,
                    self.dweight_needed,
                )?;
                Ok(apply3(x, self.base.weight(), &ab, op)?)
            }
            DispatchOutcome::Eager => self.forward_eager_training_composition(x, dropout_key),
        }
    }

    /// TODAY'S exact training-arm eager composition, extracted verbatim
    /// (not rewritten) from `forward`'s own `DispatchOutcome::Eager` arm —
    /// same pattern as `jammi_encoders::modernbert`'s
    /// `forward_eager_training_attention_composition` (extracted so the
    /// production fallback is directly callable, bypassing admission,
    /// for a same-process fused-vs-eager A/B without a second
    /// `JAMMI_KERNELS_DISABLE`-configured build). `forward`'s own
    /// `DispatchOutcome::Eager` arm calls this SAME function — there is
    /// exactly one definition of the training-arm eager composition, not
    /// one per caller.
    fn forward_eager_training_composition(
        &self,
        x: &Tensor,
        dropout_key: Option<DropoutKey>,
    ) -> Result<Tensor, LoraError> {
        let base_dtype = self.base.weight().dtype();
        let x_base = if x.dtype() == base_dtype {
            x.clone()
        } else {
            x.to_dtype(base_dtype)?
        };
        let base_out = self.base.forward(&x_base)?;

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

/// BF16/CUDA production-width oracles for the fused LoRA site
/// (`jammi_kernels::ops::LowRankResidualLinear`, dispatched here as
/// `"lora_linear_fused"`), against the REAL production eager composition
/// (`LoraLinear::forward_eager_training_composition` — the exact function
/// `forward`'s own `DispatchOutcome::Eager` arm calls, extracted so this
/// module can call it directly, bypassing admission, for a same-process
/// A/B — never a re-implementation of its math). Pod-run only (CUDA
/// feature + a physical device): `cuda_device` skips (not fails) on a
/// CUDA-feature build with no physical GPU, per `docs/maintainer/
/// cuda-kernel-guide.md`'s own `cuda_parity.rs`/`modernbert.rs` convention.
///
/// `jammi-kernels` is a declared leaf crate ("no jammi-* dependencies") —
/// this suite therefore lives HERE, in `jammi-lora` (which already depends
/// on `jammi-kernels`), rather than as a `jammi-kernels` dev-dependency on
/// `jammi-lora` (which would need a dev-dependency cycle jammi-kernels'
/// own `Cargo.toml` disclaims).
///
/// Three deliverables, each a `#[test]` below:
/// 1. `lora_linear_bwd_bf16_production_oracle_sweep_cuda` — fwd + all
///    three backward outputs (`dx`, `dA`, `dB`) at production geometry
///    (`in=1024`, `out ∈ {1024, 3072, 4096}`, `rank=16`, `alpha=32`,
///    `rows ∈ {128, 512, 4096}`, 3 seeds), Gaussian `x` at production
///    amplitude and a Gaussian `N(0, 1e-3)` cotangent (the live-gradient
///    regime — an upstream loss gradient is small relative to the
///    activations that produced it, not unit-amplitude), compared via the
///    FA2-upstream form: `max|fused−ref_f32| <= 2×max|eager_bf16−ref_f32|`
///    (fwd) / `<= 3×` (each gradient), NO absolute floor (guide §3.8).
/// 2. `lora_linear_bwd_bf16_wrong_alpha_scale_red_control_trips_the_bound_cuda`
///    — a committed RED control: the REAL fused kernel, called directly
///    with a deliberately wrong `scale` (a 50% alpha error), must FAIL
///    every one of the four bounds above — proving this file's own
///    assertion machinery is non-vacuous (fix-verifier's red-green
///    discipline, permanently embedded rather than a one-off pod finding).
/// 3. `lora_linear_fused_vs_eager_error_ratio_does_not_grow_with_depth_bf16_cuda`
///    — the growth oracle (guide §3.2/esc-044's signature): 8 fused sites
///    chained back-to-back (a self-loop, `in == out == 1024`, so one
///    site's output feeds the next site's input with no extra
///    projection — family L, a weight-free shape-correct bridge), `r(L) =
///    Σ|dx_fused − dx_eager| / Σ|dx_eager|` reported at every depth `L =
///    1..=8`, gated on `r(L) <= C·max(r(1), EPS)` — a budget derived from
///    THIS RUN'S OWN `r(1)`, never an absolute ULP constant.
///
/// Every leg is dispatch-counted (guide §3.5: zero dispatch is RED, never
/// green) using `>`/`>=` deltas, not `==` — the process-wide dispatch
/// counters are shared with every OTHER test in this binary running
/// concurrently (`cargo test`'s default thread-per-test model), the same
/// raciness `crates/jammi-lora/tests/fused_epilogue.rs`'s own
/// `eval_mode_never_dispatches_the_fused_kernel` documents.
#[cfg(all(test, feature = "cuda"))]
mod cuda_bf16_production_oracles {
    use super::*;
    use candle_core::{Device, Var};

    /// Production ModernBERT-large hidden width — shared by every geometry
    /// this suite exercises (`out_features` varies; `in_features` does
    /// not, so a single production hidden-state activation feeds every
    /// leg identically, matching how one hidden state feeds Wqkv/Wo/Wi in
    /// production).
    const IN_FEATURES: usize = 1024;
    const RANK: usize = 16;
    const ALPHA: f64 = 32.0;
    /// A production-representative activation amplitude: this crate has
    /// no single canonical "hidden state std" constant, but
    /// `jammi_encoders::modernbert`'s own module doc measures
    /// `max|qkv| ≈ 9–18` on the REAL ModernBERT-large checkpoint (`qkv =
    /// Wqkv(hidden)`, the same hidden state a LoRA-augmented `Wqkv`/`Wo`/
    /// `Wi` consumes as `x` here) — `X_STD` is chosen so a Gaussian draw
    /// at this row count lands `max|x|` in that same order of magnitude
    /// (reported, not hardcoded — see each test's own `eprintln!`).
    /// Generic/synthetic (family L: this crate names no consumer), not a
    /// literal replay of that measurement.
    const X_STD: f32 = 3.5;
    /// A small base-weight init amplitude (matches a typical
    /// Kaiming/Xavier-scale frozen `Linear` weight — NOT LoRA's own
    /// `A`/`B` amplitude, which `LoraInitMode::Gaussian` already fixes at
    /// `0.02`, `crate::seeded::gaussian_fill`'s call sites in `LoraLinear
    /// ::new`).
    const W_STD: f32 = 0.04;
    /// The live-gradient regime (task contract): an upstream loss
    /// gradient is small relative to the activations/weights that
    /// produced it, not unit-amplitude — never `X_STD`/`W_STD`-scale.
    const DY_STD: f32 = 1e-3;

    /// Mirrors `tests/cuda_parity.rs`'s own `cuda_device` (`jammi-kernels`)
    /// and `jammi_encoders::modernbert`'s `growth_oracle_cuda_device`: a
    /// machine that compiled with the `cuda` feature but has no physical
    /// GPU is "skip", not "fail", UNLESS `JAMMI_REQUIRE_CUDA` is set, in
    /// which case device-acquisition failure panics rather than silently
    /// reading as a skip.
    fn cuda_device() -> Option<Device> {
        match Device::new_cuda(0) {
            Ok(d) => Some(d),
            Err(e) => {
                if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                    panic!(
                        "lora_linear BF16/CUDA production oracle: JAMMI_REQUIRE_CUDA is set but \
                         no CUDA device could be acquired: {e}"
                    );
                }
                eprintln!(
                    "lora_linear BF16/CUDA production oracle: skipping — no CUDA device \
                     available ({e})"
                );
                None
            }
        }
    }

    /// A deterministic Gaussian `f32` host buffer via this crate's own
    /// `SplitMix64`/`gaussian_fill` (no `rand` dependency — the same
    /// seeded-init machinery `LoraLinear::new`'s own `LoraInitMode::
    /// Gaussian` branch uses), never candle's unseedable global RNG.
    fn gaussian_vec(seed: u64, n: usize, stdev: f32) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        gaussian_fill(&mut rng, n, stdev)
    }

    /// A matched pair of `LoraLinear`s sharing ONE seed: `bf16` runs the
    /// production `BF16` base / `F32` LoRA-adapter combination this op's
    /// own domain doc calls out (`(BF16, BF16, F32)`); `f32_ref` runs the
    /// SAME recipe with NOTHING rounded to `bf16` anywhere — the base
    /// weight is the pre-rounding `F32` values, not `w_bf16` cast back up
    /// (which would already carry `bf16`'s rounding). Both layers' `lora_a`/
    /// `lora_b` are bit-identical: `LoraLinear::new`'s seeded draw is a
    /// pure function of `(seed, fully-qualified parameter name)`
    /// (`crate::seeded::seed_for_param`'s own doc), independent of the
    /// base weight's dtype, so two fresh `VarMap`/`VarBuilder` pairs
    /// constructed with the same `seed` and the same (default, empty)
    /// prefix draw byte-identical `F32` `lora_a`/`lora_b` regardless of
    /// which `base` they are paired with.
    struct ProdLayers {
        bf16: LoraLinear,
        f32_ref: LoraLinear,
    }

    fn build_prod_layers(out_features: usize, seed: u64, device: &Device) -> ProdLayers {
        let w_v = gaussian_vec(seed ^ 0x5741_1E17, out_features * IN_FEATURES, W_STD);
        let w_f32 = Tensor::from_vec(w_v, (out_features, IN_FEATURES), device).unwrap();
        let w_bf16 = w_f32.to_dtype(DType::BF16).unwrap();
        let base_bf16 = Linear::new(w_bf16, None);
        let base_f32 = Linear::new(w_f32.clone(), None);

        let varmap_bf16 = VarMap::new();
        let vb_bf16 = VarBuilder::from_varmap(&varmap_bf16, DType::F32, device);
        let bf16_layer = LoraLinear::new(
            base_bf16,
            RANK,
            ALPHA,
            false,
            LoraInitMode::Gaussian,
            None,
            seed,
            &varmap_bf16,
            &vb_bf16,
        )
        .unwrap();

        let varmap_f32 = VarMap::new();
        let vb_f32 = VarBuilder::from_varmap(&varmap_f32, DType::F32, device);
        let f32_layer = LoraLinear::new(
            base_f32,
            RANK,
            ALPHA,
            false,
            LoraInitMode::Gaussian,
            None,
            seed,
            &varmap_f32,
            &vb_f32,
        )
        .unwrap();

        ProdLayers {
            bf16: bf16_layer,
            f32_ref: f32_layer,
        }
    }

    fn to_f32_vec(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn assert_all_finite(v: &[f32], label: &str) {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{label}: non-finite element present"
        );
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
    }

    /// The FA2-upstream comparison form: `max|fused-ref| <=
    /// multiplier*max|eager-ref|`, NO absolute floor (guide §3.8 — a
    /// `k*ulp(max)` floor would charge every element the allowance of the
    /// largest element and hide exactly the divergence an oracle exists
    /// to catch). Non-finite checked BEFORE any comparison (guide §3.7/
    /// family F), written affirmatively. Returns `(within_bound, e_fused,
    /// e_eager)` rather than asserting directly, so the RED control below
    /// can assert the NEGATION without a `catch_unwind`.
    fn fa2_bound(
        label: &str,
        fused: &[f32],
        eager: &[f32],
        reference: &[f32],
        multiplier: f32,
    ) -> (bool, f32, f32) {
        assert_all_finite(fused, &format!("{label} fused"));
        assert_all_finite(eager, &format!("{label} eager"));
        assert_all_finite(reference, &format!("{label} reference"));
        assert_eq!(
            fused.len(),
            reference.len(),
            "{label}: fused/reference length mismatch"
        );
        assert_eq!(
            eager.len(),
            reference.len(),
            "{label}: eager/reference length mismatch"
        );
        let e_fused = max_abs_diff(fused, reference);
        let e_eager = max_abs_diff(eager, reference);
        let bound = multiplier * e_eager;
        let ok = e_fused.is_finite() && e_fused <= bound;
        (ok, e_fused, e_eager)
    }

    fn assert_fa2_bound(
        label: &str,
        fused: &[f32],
        eager: &[f32],
        reference: &[f32],
        multiplier: f32,
        context: &str,
    ) {
        let (ok, e_fused, e_eager) = fa2_bound(label, fused, eager, reference, multiplier);
        assert!(
            ok,
            "{label} ({context}): max|fused-ref_f32|={e_fused:e} exceeds {multiplier}x \
             max|eager_bf16-ref_f32|={e_eager:e} (bound={:e}, NO floor)",
            multiplier * e_eager
        );
    }

    struct OracleOutputs {
        out: Vec<f32>,
        dx: Vec<f32>,
        da: Vec<f32>,
        db: Vec<f32>,
    }

    /// Runs ONE arm of `layer` on `(x, dy)` and returns its forward output
    /// plus all three backward outputs (`dx`, `dA`, `dB`), together with
    /// the `lora_linear_fused` dispatch-counter delta observed around the
    /// call (`(fused_delta, eager_delta)`). `force_eager == false` calls
    /// the REAL public dispatch site (`LoraLinear::forward`, which admits
    /// the fused kernel on this domain); `force_eager == true` calls
    /// `forward_eager_training_composition` DIRECTLY — the exact function
    /// `forward`'s own `Eager` arm calls, bypassing `admit` entirely (so
    /// its own dispatch counters never move) — never a re-implementation.
    fn run_arm(
        layer: &LoraLinear,
        x: &Tensor,
        dy: &Tensor,
        force_eager: bool,
    ) -> (OracleOutputs, u64, u64) {
        let before = lora_linear_fused_dispatch_snapshot();
        let x_var = Var::from_tensor(x).unwrap();
        let out = if force_eager {
            layer
                .forward_eager_training_composition(x_var.as_tensor(), None)
                .unwrap()
        } else {
            layer.forward(x_var.as_tensor()).unwrap()
        };
        let loss = (&out * dy).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let dx = grads
            .get(&x_var)
            .unwrap_or_else(|| panic!("run_arm: no dx (force_eager={force_eager})"));
        let da = grads
            .get(&layer.lora_a)
            .unwrap_or_else(|| panic!("run_arm: no dA (force_eager={force_eager})"));
        let db = grads
            .get(&layer.lora_b)
            .unwrap_or_else(|| panic!("run_arm: no dB (force_eager={force_eager})"));
        let outputs = OracleOutputs {
            out: to_f32_vec(&out),
            dx: to_f32_vec(dx),
            da: to_f32_vec(da),
            db: to_f32_vec(db),
        };
        let after = lora_linear_fused_dispatch_snapshot();
        (
            outputs,
            after.fused - before.fused,
            after.eager - before.eager,
        )
    }

    /// Deliverable 1: one production geometry cell — builds the matched
    /// `bf16`/`f32_ref` pair, draws production-amplitude `x` and a
    /// live-gradient-regime `dy`, runs all three arms (fused-bf16,
    /// eager-bf16, eager-f32-reference), and asserts the FA2-upstream
    /// bound on `fwd`/`dx`/`dA`/`dB`.
    fn compare_production_geometry(
        layers: &ProdLayers,
        device: &Device,
        out_features: usize,
        rows: usize,
        seed: u64,
    ) {
        let context = format!("out_features={out_features} rows={rows} seed={seed}");

        let x_v = gaussian_vec(
            seed ^ 0x5EED_1234 ^ (rows as u64),
            rows * IN_FEATURES,
            X_STD,
        );
        let x_f32 = Tensor::from_vec(x_v, (rows, IN_FEATURES), device).unwrap();
        let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();

        let dy_v = gaussian_vec(
            seed ^ 0xD137_1234 ^ (rows as u64) ^ ((out_features as u64) << 32),
            rows * out_features,
            DY_STD,
        );
        let dy_f32 = Tensor::from_vec(dy_v, (rows, out_features), device).unwrap();
        let dy_bf16 = dy_f32.to_dtype(DType::BF16).unwrap();

        let (fused, fused_ctr, fused_eager_ctr) = run_arm(&layers.bf16, &x_bf16, &dy_bf16, false);
        assert!(
            fused_ctr > 0,
            "{context}: fused leg must dispatch the fused kernel at least once (zero dispatch \
             is RED, never green — guide §3.5)"
        );
        assert_eq!(
            fused_eager_ctr, 0,
            "{context}: fused leg must never silently fall back to eager"
        );

        let (eager, _, _) = run_arm(&layers.bf16, &x_bf16, &dy_bf16, true);
        let (reference, _, _) = run_arm(&layers.f32_ref, &x_f32, &dy_f32, true);

        assert_fa2_bound("fwd", &fused.out, &eager.out, &reference.out, 2.0, &context);
        assert_fa2_bound("dx", &fused.dx, &eager.dx, &reference.dx, 3.0, &context);
        assert_fa2_bound("dA", &fused.da, &eager.da, &reference.da, 3.0, &context);
        assert_fa2_bound("dB", &fused.db, &eager.db, &reference.db, 3.0, &context);
    }

    /// Deliverable 1. Production geometry: `in=1024`, `out ∈ {1024,
    /// 3072, 4096}`, `rank=16`, `alpha=32` (scaling = 2.0, vanilla), rows
    /// `∈ {128, 512, 4096}`, 3 seeds — 27 cells, each independently
    /// asserted (the panic message names the failing cell). `layers`
    /// (and hence the base weight / `lora_a` / `lora_b`) is rebuilt once
    /// per `(out_features, seed)` pair and reused across the 3 row
    /// counts, so the same weights are exercised at every row count
    /// (matching how one adapter serves every batch shape in production).
    #[test]
    fn lora_linear_bwd_bf16_production_oracle_sweep_cuda() {
        let Some(device) = cuda_device() else {
            return;
        };
        const OUT_FEATURES: [usize; 3] = [1024, 3072, 4096];
        const ROWS: [usize; 3] = [128, 512, 4096];
        const SEEDS: [u64; 3] = [41, 42, 43];

        for &seed in &SEEDS {
            for &out_features in &OUT_FEATURES {
                let layers = build_prod_layers(out_features, seed, &device);
                for &rows in &ROWS {
                    compare_production_geometry(&layers, &device, out_features, rows, seed);
                }
            }
        }
    }

    /// Deliverable 2: a committed RED control. The REAL fused kernel
    /// (`LowRankResidualLinear`, called directly through `apply3` — the
    /// SAME function `LoraLinear::forward`'s `Fused` arm calls, same GEMM
    /// sequence, same CUDA kernel), given a deliberately WRONG `scale`
    /// (`1.5x` the layer's real `alpha/rank` — a 50% alpha-scale error),
    /// must FAIL the FA2-upstream bound on every one of `fwd`/`dx`/`dA`/
    /// `dB` against the honest eager-bf16/f32-reference pair. If this
    /// does NOT trip, the oracle above is vacuous — it would pass on a
    /// broken kernel just as readily as on a correct one (fix-verifier's
    /// red-green discipline, permanently embedded rather than a one-off
    /// pod finding).
    #[test]
    fn lora_linear_bwd_bf16_wrong_alpha_scale_red_control_trips_the_bound_cuda() {
        let Some(device) = cuda_device() else {
            return;
        };
        let out_features = 1024usize;
        let rows = 512usize;
        let seed = 42u64;
        let layers = build_prod_layers(out_features, seed, &device);

        let x_v = gaussian_vec(
            seed ^ 0x5EED_1234 ^ (rows as u64),
            rows * IN_FEATURES,
            X_STD,
        );
        let x_f32 = Tensor::from_vec(x_v, (rows, IN_FEATURES), &device).unwrap();
        let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();
        let dy_v = gaussian_vec(
            seed ^ 0xD137_1234 ^ (rows as u64) ^ ((out_features as u64) << 32),
            rows * out_features,
            DY_STD,
        );
        let dy_f32 = Tensor::from_vec(dy_v, (rows, out_features), &device).unwrap();
        let dy_bf16 = dy_f32.to_dtype(DType::BF16).unwrap();

        // Honest legs, unchanged.
        let (eager, _, _) = run_arm(&layers.bf16, &x_bf16, &dy_bf16, true);
        let (reference, _, _) = run_arm(&layers.f32_ref, &x_f32, &dy_f32, true);

        // BROKEN "fused": the REAL op, the REAL `apply3` call site, a
        // deliberately wrong `scale` — `dweight_needed = false` matches
        // `layers.bf16`'s own base (an untracked leaf; see
        // `frozen_weight_gate`).
        let wrong_scale = (layers.bf16.scaling() * 1.5) as f32;
        let ab = Tensor::cat(&[&layers.bf16.lora_a.t().unwrap(), &layers.bf16.lora_b], 0).unwrap();
        let op =
            LowRankResidualLinear::new(wrong_scale, IN_FEATURES, out_features, RANK, None, false)
                .unwrap();
        let x_var = Var::from_tensor(&x_bf16).unwrap();
        let out = apply3(x_var.as_tensor(), layers.bf16.base.weight(), &ab, op).unwrap();
        let loss = (&out * &dy_bf16).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let broken = OracleOutputs {
            out: to_f32_vec(&out),
            dx: to_f32_vec(grads.get(&x_var).unwrap()),
            da: to_f32_vec(grads.get(&layers.bf16.lora_a).unwrap()),
            db: to_f32_vec(grads.get(&layers.bf16.lora_b).unwrap()),
        };

        let (fwd_ok, e_fwd, ee_fwd) =
            fa2_bound("fwd", &broken.out, &eager.out, &reference.out, 2.0);
        let (dx_ok, e_dx, ee_dx) = fa2_bound("dx", &broken.dx, &eager.dx, &reference.dx, 3.0);
        let (da_ok, e_da, ee_da) = fa2_bound("dA", &broken.da, &eager.da, &reference.da, 3.0);
        let (db_ok, e_db, ee_db) = fa2_bound("dB", &broken.db, &eager.db, &reference.db, 3.0);

        assert!(
            !fwd_ok,
            "RED CONTROL FAILED TO TRIP (fwd): a 50% wrong alpha scale must exceed the FA2 \
             bound, but e_fused={e_fwd:e} <= 2*e_eager={:e} — this oracle's own assertion \
             machinery is vacuous",
            2.0 * ee_fwd
        );
        assert!(
            !dx_ok,
            "RED CONTROL FAILED TO TRIP (dx): e_fused={e_dx:e} <= 3*e_eager={:e}",
            3.0 * ee_dx
        );
        assert!(
            !da_ok,
            "RED CONTROL FAILED TO TRIP (dA): e_fused={e_da:e} <= 3*e_eager={:e}",
            3.0 * ee_da
        );
        assert!(
            !db_ok,
            "RED CONTROL FAILED TO TRIP (dB): e_fused={e_db:e} <= 3*e_eager={:e}",
            3.0 * ee_db
        );
    }

    /// Deliverable 3: the growth oracle (guide §3.2, esc-044's
    /// signature). 8 fused sites chained back-to-back from ONE tracked
    /// `x` `Var` (`in == out == 1024`: a self-loop, weight-free
    /// shape-correct bridge — the SAME `LoraLinear` instance is reused at
    /// every depth, exactly as `jammi_encoders::modernbert`'s own
    /// `attention_block_fused_vs_eager_dqkv_divergence_grows_with_depth_
    /// bf16_cuda` reuses one `attn`), a plain max-abs rescale between
    /// calls (no residual/LayerNorm in this synthetic chain) to stay
    /// inside this op's own validated `bf16` domain across 8 layers.
    /// `r(L) = Σ|dx_fused − dx_eager| / Σ|dx_eager|` is computed
    /// INDEPENDENTLY at every depth `L = 1..=8` (not just `L=1` and
    /// `L=8`) via 8 separate forward+backward chains per arm, reported in
    /// full via `eprintln!` and in every failing assertion's own message.
    #[test]
    fn lora_linear_fused_vs_eager_error_ratio_does_not_grow_with_depth_bf16_cuda() {
        let Some(device) = cuda_device() else {
            return;
        };
        const L_MAX: usize = 8;
        let out_features = IN_FEATURES; // self-loop chain.
        let rows = 512usize;
        let seed = 44u64;
        let layers = build_prod_layers(out_features, seed, &device);

        let x0_v = gaussian_vec(seed ^ 0x6060_A11A, rows * IN_FEATURES, X_STD);
        let x0 = Tensor::from_vec(x0_v, (rows, IN_FEATURES), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let dy_v = gaussian_vec(seed ^ 0x7070_B22B, rows * out_features, DY_STD);
        let dy = Tensor::from_vec(dy_v, (rows, out_features), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // Returns `(dx, fused_dispatch_delta)`.
        let run = |force_eager: bool, l: usize| -> (Vec<f32>, u64) {
            assert!(l > 0, "run: l must be >= 1");
            let before = lora_linear_fused_dispatch_snapshot();
            let x_var = Var::from_tensor(&x0).unwrap();
            let mut cur = x_var.as_tensor().clone();
            let mut last_out = None;
            for _ in 0..l {
                let out = if force_eager {
                    layers
                        .bf16
                        .forward_eager_training_composition(&cur, None)
                        .unwrap()
                } else {
                    layers.bf16.forward(&cur).unwrap()
                };
                let out_max = out
                    .abs()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    .max(1e-6);
                cur = (&out / f64::from(out_max)).unwrap();
                last_out = Some(out);
            }
            let loss = (last_out.unwrap() * &dy).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let dx: Vec<f32> = grads
                .get(&x_var)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let after = lora_linear_fused_dispatch_snapshot();
            (dx, after.fused - before.fused)
        };

        let mut r_table: Vec<(usize, f64)> = Vec::with_capacity(L_MAX);
        for l in 1..=L_MAX {
            let (dx_fused, fused_ctr) = run(false, l);
            let (dx_eager, _) = run(true, l);
            assert!(
                fused_ctr as usize >= l,
                "growth oracle depth {l}: fused arm must dispatch fused at every one of the {l} \
                 layers (zero dispatch is RED, never green) — got {fused_ctr}"
            );
            assert_all_finite(&dx_fused, &format!("growth L={l} fused dx"));
            assert_all_finite(&dx_eager, &format!("growth L={l} eager dx"));
            let mut num = 0f64;
            let mut den = 0f64;
            for (&f, &e) in dx_fused.iter().zip(dx_eager.iter()) {
                num += f64::from((f - e).abs());
                den += f64::from(e.abs());
            }
            assert!(
                den.is_finite() && den > 0.0,
                "growth oracle depth {l}: Σ|dx_eager| must be nonzero (signal check)"
            );
            r_table.push((l, num / den));
        }

        let report: String = r_table
            .iter()
            .map(|(l, r)| format!("r({l})={r:.3e}"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("lora_linear fused-vs-eager depth-growth (dx): {report}");

        // The gate: GROWTH, not magnitude — the same budget shape
        // `jammi_encoders::modernbert`'s own esc-044 growth oracle uses.
        // `EPS` guards only a pathological exact `r(1) == 0` tie; it is
        // two orders below any bf16-noise-scale `r(1)` this suite could
        // plausibly measure, so it never competes with a real
        // measurement.
        const C: f64 = 4.0;
        const EPS: f64 = 1e-9;
        let r1 = r_table[0].1;
        let bound = C * r1.max(EPS);
        for &(l, r) in &r_table {
            assert!(
                r.is_finite() && r <= bound,
                "lora_linear depth-growth oracle: r({l})={r:e} exceeds {C}*max(r(1),{EPS:e})=\
                 {bound:e} — table: {report} — the fused/eager divergence is growing \
                 SYSTEMATICALLY with depth, not staying at the L=1 bf16-noise scale"
            );
        }
    }
}
