//! Single LoRA-augmented linear layer: frozen base + trainable A and B matrices.

use candle_core::{DType, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
    DispatchSnapshot,
};
use jammi_kernels::ops::{apply1, apply3, DropoutFused, DropoutKey, LoraLinearFused};

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
/// consumed DIRECTLY by [`LoraLinearFused`]/[`DropoutFused::new`] on
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
/// [`LoraLinearFused`], which reuses `ScaledCastAdd`'s `cpu_fwd`/`cuda_fwd`
/// DIRECTLY (a plain function call, not through `admit`) as its own
/// internal epilogue step — see `jammi_kernels::ops::lora_linear`'s module
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
/// (`jammi_kernels::ops::LoraLinearFused`, the whole-site fusion — one
/// tape node in place of the ~11-node eager composition
/// `eager_epilogue` and its own `A`/`B`/dropout sub-linears build), read
/// from the same op-keyed registry `lora_epilogue_counters` uses.
///
/// **`lora_epilogue`/`lora_dropout` legitimately read `0` for every
/// forward that dispatches through this counter instead.** Once the
/// training arm routes through [`LoraLinearFused`], neither
/// [`ScaledCastAdd`] nor a standalone [`DropoutFused`] call is EVER made
/// for that forward — both are reused INSIDE the fused kernel's own
/// `cpu_fwd`/`cuda_fwd` (see `jammi_kernels::ops::lora_linear`'s module
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
/// / K2): [`device_is_supported`]; `x` rank 2 (a pooled head,
/// `fine_tune/lora.rs`) or 3 (`[batch, seq, in]`); `x`/`w` share a dtype
/// that is EITHER both `F32` or both `BF16` (the two combinations
/// [`jammi_kernels::ops::LoraLinearFused`] actually implements); both
/// contiguous; the base weight carries no bias (see
/// [`jammi_kernels::ops::lora_linear`]'s module doc for why a bias is a
/// domain refusal here rather than packed into `ab`). `out_features >= 1`
/// and `rank >= 1` are guaranteed by construction (`LoraLinear::new`
/// refuses `rank == 0`, and a real `Linear`'s weight always has
/// `out_features >= 1`) — not re-checked here, but re-validated
/// independently by [`LoraLinearFused::new`] regardless (family D: an op
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
    // [`jammi_kernels::ops::LoraLinearFused`]'s own domain requires `ab`
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
    if !x.is_contiguous() {
        return (false, "x_contiguous");
    }
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
/// `jammi_kernels::ops::lora_linear`'s module doc for what `dweight_needed`
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
    /// ## Training: `jammi_kernels::ops::LoraLinearFused`, one tape node
    ///
    /// The training arm routes the ENTIRE site — `base = x @ w^T`, the
    /// dropout draw, both LoRA GEMMs, and the epilogue — through ONE
    /// `CustomOp3` call ([`LoraLinearFused`]) when the kernel's own domain
    /// holds (`lora_linear_admission_predicate`): this collapses the
    /// ~11-node eager composition (each node its own `zeros_like`+`add` in
    /// candle's backward) into one node (plus the tiny `ab`-packing
    /// `Tensor::cat`). Outside the fused kernel's domain (a bias-carrying
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
    /// both the fused arm (passed into [`LoraLinearFused`]'s construction
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
                // Row-packed layout (`jammi_kernels::ops::lora_linear`'s
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
                let op = LoraLinearFused::new(
                    self.scaling as f32,
                    self.in_features,
                    self.out_features,
                    self.rank,
                    dropout_key,
                    self.dweight_needed,
                )?;
                Ok(apply3(x, self.base.weight(), &ab, op)?)
            }
            DispatchOutcome::Eager => {
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
        }
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
