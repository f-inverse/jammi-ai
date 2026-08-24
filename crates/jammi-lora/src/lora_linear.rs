//! Single LoRA-augmented linear layer: frozen base + trainable A and B matrices.

use candle_core::{DType, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
    DispatchSnapshot,
};
use jammi_kernels::ops::{apply2, ScaledCastAdd};

use crate::error::LoraError;
use crate::init::LoraInitMode;
use crate::seeded::{
    gaussian_fill, kaiming_uniform_fill, seed_for_param, DropoutMasks, SplitMix64,
};

/// Per-op fused/eager dispatch counts for the device-side dropout op
/// (`jammi_kernels::ops::DropoutFused`), read from the same op-keyed
/// registry `lora_epilogue_counters` (below) uses — see that function's doc.
fn lora_dropout_counters() -> &'static DispatchCounters {
    counters_for("lora_dropout")
}

/// A snapshot of the fused/eager dispatch counts for the LoRA dropout op —
/// mirrors [`lora_epilogue_dispatch_snapshot`].
pub fn lora_dropout_dispatch_snapshot() -> DispatchSnapshot {
    lora_dropout_counters().snapshot()
}

/// The dropout op's domain, checked at the call site (family D / K2):
/// [`device_is_supported`] (CPU always; CUDA only when this build's
/// `cuda` feature is on) and the activation dtype is `F32` or `BF16` (this
/// crate's two production dtypes — `x_lora`, below, is `F32` in every
/// training-mode call site today per `forward`'s own doc on
/// `self.lora_a.dtype()`, but the predicate is stated generically rather
/// than assuming that workspace fact holds forever).
///
/// The "eager" fallback for an out-of-domain case is candle's own
/// `candle_nn::ops::dropout` — UNSEEDABLE, and therefore NOT determinism-
/// preserving. This is a disclosed, deliberate reduction in guarantees for
/// what is, today, an UNREACHABLE branch (every real call site is CPU or
/// CUDA, `F32`): the same "unreachable-today, disclosed rather than
/// silently assumed" shape `ScaledCastAdd`'s own module doc uses for its
/// two unreachable dtype combinations. Because the eager arm never calls
/// `DropoutMasks::apply`, the layer's forward counter (and therefore its
/// resume position) does NOT advance on that arm — a live discrepancy only
/// if this predicate ever actually failed in production, which it cannot
/// today; a deployment that widens the reachable dtype/device set should
/// run `JAMMI_KERNELS_STRICT` (this op's `admit` call already honors it)
/// rather than rely on this fallback silently preserving resume-safety it
/// was never designed to.
fn dropout_admission_predicate(x: &Tensor) -> (bool, &'static str) {
    if !device_is_supported(x.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16) {
        return (false, "dtype_f32_or_bf16");
    }
    (true, "domain_ok")
}

/// Per-op fused/eager dispatch counts for the LoRA-site epilogue
/// (`base_out + cast(lora_out * scaling)`), read from `jammi-kernels`' new
/// op-keyed registry (`counters_for`) rather than a hand-declared
/// `static DispatchCounters` — this crate is the first to use the
/// generalized form C6 adds (see `jammi_kernels::admission`'s module doc);
/// C2-C5's four ops in `jammi-encoders` keep their own pre-existing
/// statics unchanged.
fn lora_epilogue_counters() -> &'static DispatchCounters {
    counters_for("lora_epilogue")
}

/// A snapshot of the fused/eager dispatch counts for the LoRA-site
/// epilogue, mirroring `jammi_encoders::ln_dispatch_snapshot` /
/// `rope_dispatch_snapshot` / `softmax_dispatch_snapshot` /
/// `geglu_dispatch_snapshot` — the read API a durable job record or a
/// bench report uses to state which kernel path actually ran.
pub fn lora_epilogue_dispatch_snapshot() -> DispatchSnapshot {
    lora_epilogue_counters().snapshot()
}

/// The fused epilogue kernel's domain, checked at the call site (family D
/// / K2): `base_out` lives on a device [`device_is_supported`] accepts,
/// both `base_out` and `lora_out` are dtype `F32` or `BF16` (independently
/// — `ScaledCastAdd` supports all four combinations, though only
/// (`F32`,`F32`) and (`BF16`,`F32`) are reachable today: `lora_a`/`lora_b`
/// are always `F32` in this workspace, since the two call sites that
/// construct a LoRA adapter's `VarBuilder` both pass `DType::F32` — see
/// this module's `forward` doc for the exact citation), both are contiguous
/// (`ScaledCastAdd`'s CUDA arm has no strided-view support — see its own
/// module doc), and the two tensors' shapes match exactly (the op is not a
/// broadcasting op). Returns the aggregate predicate and the name of
/// whichever check is the reason (the first one evaluated, or a fixed
/// "domain_ok" name when everything holds).
fn epilogue_admission_predicate(base_out: &Tensor, lora_out: &Tensor) -> (bool, &'static str) {
    if !device_is_supported(base_out.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if !matches!(base_out.dtype(), DType::F32 | DType::BF16) {
        return (false, "base_dtype_f32_or_bf16");
    }
    if !matches!(lora_out.dtype(), DType::F32 | DType::BF16) {
        return (false, "lora_dtype_f32_or_bf16");
    }
    if !base_out.is_contiguous() {
        return (false, "base_contiguous");
    }
    if !lora_out.is_contiguous() {
        return (false, "lora_contiguous");
    }
    if base_out.dims() != lora_out.dims() {
        return (false, "base_and_lora_shape_equal");
    }
    (true, "domain_ok")
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

        let scaling = if use_rslora {
            alpha / (rank as f64).sqrt()
        } else {
            alpha / rank as f64
        };

        let dropout_masks = dropout
            .filter(|p| *p > 0.0)
            .map(|_| DropoutMasks::new(seed, &vb.prefix()));

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling,
            dropout,
            dropout_masks,
            training: true,
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
    /// Scaling is derived as `alpha / rank` where rank is inferred from
    /// `lora_a.dims()[0]`. RSLoRA scaling is intentionally not represented
    /// here because callers reconstructing from disk always know the
    /// effective scaling implied by the saved adapter.
    pub fn from_loaded(base: Linear, lora_a: Tensor, lora_b: Tensor, alpha: f64) -> Self {
        let rank = lora_a.dims()[0];
        let scaling = alpha / rank as f64;
        Self {
            base,
            lora_a,
            lora_b,
            scaling,
            dropout: None,
            dropout_masks: None,
            training: false,
        }
    }

    /// Toggle training mode. When `false`, dropout in the LoRA path is skipped
    /// so validation loss and inference outputs are deterministic.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward: `base(x) + scaling * dropout(x @ A^T @ B^T)`.
    ///
    /// The frozen base path runs in F32 for device-agnostic matmul support;
    /// the result is cast back to the backbone dtype before the LoRA delta is
    /// added so downstream layers stay in their expected precision.
    ///
    /// ## The fused epilogue: `jammi_kernels::ops::ScaledCastAdd`
    ///
    /// The final `[mul, cast, add]` — `base_out + cast(lora_out * scaling)`
    /// — is replaced by one fused `CustomOp2` call when `self.training` is
    /// `true` AND the kernel's own domain holds
    /// (`epilogue_admission_predicate`): this collapses three tape nodes
    /// (each with its own `zeros_like`+`add` in backward) into one.
    /// `self.training` gates it because a `LoraLinear` also SERVES
    /// inference (`from_loaded`, `training: false`) — eval/serving keeps
    /// today's eager composition bit-for-bit, unconditionally, exactly as
    /// C2's fused LayerNorm gates on its own crate's `training` flag (see
    /// `jammi_encoders::layer_norm`'s module doc for the same argument).
    /// Outside the fused kernel's domain (an unsupported dtype/device, a
    /// non-contiguous view), the training-mode arm falls back to
    /// `eager_epilogue` too — the SAME function eval always uses — so a
    /// domain miss and eval-mode are byte-identical code paths, not two
    /// independently-maintained ones.
    ///
    /// The LoRA-arm dtype `lora_a`/`lora_b` run at (`self.lora_a.dtype()`,
    /// read below) is `F32` in every training-mode call site in this
    /// workspace TODAY — but this is a WORKSPACE FACT about today's call
    /// sites, not a `candle_nn::VarBuilder::from_varmap` API guarantee:
    /// its dtype is an ordinary caller-supplied parameter, not hardcoded
    /// by the function itself, and nothing stops a future caller from
    /// passing something else. What actually bounds it today is
    /// `ModernBertBuilder::build`'s `lora_vb` construction
    /// (`crates/jammi-encoders/src/modernbert.rs`) — the SINGLE place a
    /// LoRA adapter's `VarBuilder` is built for every ModernBERT LoRA site
    /// (Wqkv/Wo/Wi across every layer, the #352 profile's 112 sites) —
    /// whose both branches pass `DType::F32` explicitly:
    /// `VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32,
    /// device)` when reloading a saved adapter, and
    /// `VarBuilder::from_varmap(varmap, DType::F32, device)` when
    /// training. The `x.to_dtype(lora_dtype)` up-cast below is therefore a
    /// REAL cast (not a no-op) whenever `backbone_dtype` is reduced
    /// (`BF16`/`F16`) — changing which dtype that `VarBuilder` passes is a
    /// distinct precision decision (raising or lowering the adapter's own
    /// numeric precision), out of this commit's scope; this comment states
    /// which case holds today, and WHY it holds (a call-site choice, not
    /// an API guarantee), rather than leaving it silently assumed.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        // The frozen base runs at the backbone dtype, exactly as
        // `MaybeLoraLinear::Frozen` does: cast the input down to the weight and
        // matmul there. Re-materialising the weight in F32 on every forward
        // would make `backbone_dtype` inert on precisely the linears a LoRA run
        // targets — the reduced dtype would cost an extra full-size allocation
        // per forward and buy neither memory nor tensor-core throughput — and
        // would leave a targeted linear computing a different function from an
        // untargeted one over the same weights.
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

        let lora_in = if self.training {
            match (self.dropout, &self.dropout_masks) {
                (Some(p), Some(masks)) if p > 0.0 => {
                    // Device-side, counter-based dropout (NOT candle's
                    // unseedable `ops::dropout`, and NOT a host-materialized
                    // mask): `DropoutMasks::apply` runs the fused
                    // `jammi_kernels::ops::DropoutFused` Philox `CustomOp1`
                    // when its domain holds, advancing the forward counter
                    // by exactly one — so the mask is a pure function of the
                    // seed, this layer's `layer_id`, and the forward's
                    // position in the deterministic training order.
                    let (holds, predicate) = dropout_admission_predicate(&x_lora);
                    let outcome = admit(
                        admission_mode(),
                        "lora_dropout",
                        predicate,
                        holds,
                        lora_dropout_counters(),
                    )?;
                    match outcome {
                        DispatchOutcome::Fused => masks.apply(&x_lora, p)?,
                        DispatchOutcome::Eager => {
                            // See `dropout_admission_predicate`'s doc:
                            // unreachable today (every real call site is
                            // CPU/CUDA, F32) — disclosed, not silently
                            // assumed to preserve the fused path's
                            // determinism guarantee.
                            candle_nn::ops::dropout(&x_lora, p)?
                        }
                    }
                }
                _ => x_lora,
            }
        } else {
            x_lora
        };

        let a_lin = Linear::new(self.lora_a.clone(), None);
        let after_a = a_lin.forward(&lora_in)?;
        let b_lin = Linear::new(self.lora_b.clone(), None);
        let lora_out = b_lin.forward(&after_a)?;

        if !self.training {
            // Eval/serving: always the eager composition, unconditionally
            // — see `forward`'s doc for why this must stay bit-identical
            // regardless of the fused kernel's existence.
            return eager_epilogue(&base_out, &lora_out, self.scaling);
        }

        let (holds, predicate) = epilogue_admission_predicate(&base_out, &lora_out);
        let outcome = admit(
            admission_mode(),
            "lora_epilogue",
            predicate,
            holds,
            lora_epilogue_counters(),
        )?;
        match outcome {
            DispatchOutcome::Fused => Ok(apply2(
                &base_out,
                &lora_out,
                ScaledCastAdd::new(self.scaling),
            )?),
            DispatchOutcome::Eager => eager_epilogue(&base_out, &lora_out, self.scaling),
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
