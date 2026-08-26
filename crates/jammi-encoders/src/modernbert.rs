//! ModernBERT encoder.
//!
//! ModernBERT differences from classic BERT:
//! - Fused QKV projection `Wqkv` (`hidden * 3` by `hidden`).
//! - Output projection `Wo` (`hidden` by `hidden`).
//! - Rotary Position Embeddings (RoPE) applied to Q and K — no learned
//!   position-embedding table.
//! - GeGLU feed-forward: `Wi` packs gate+up (`intermediate * 2` by `hidden`),
//!   `mlp.Wo` projects back (`hidden` by `intermediate`).
//! - Pre-norm attention via `attn_norm`, except layer 0 where the embedding
//!   `norm` is the pre-norm (`attn_norm = None`).
//! - LayerNorm without a learned bias (matches the upstream
//!   `layer_norm_no_bias` configuration: mean-removing, weight-only affine).
//! - No token-type IDs.
//!
//! HuggingFace weight-key convention (prefix `model.`):
//! ```text
//! model.embeddings.tok_embeddings.weight
//! model.embeddings.norm.weight
//! model.layers.{n}.attn.Wqkv.weight
//! model.layers.{n}.attn.Wo.weight
//! model.layers.{n}.attn_norm.weight        // absent for layer 0
//! model.layers.{n}.mlp.Wi.weight
//! model.layers.{n}.mlp.Wo.weight
//! model.layers.{n}.mlp_norm.weight
//! model.final_norm.weight
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, VarBuilder, VarMap};
use jammi_kernels::admission::{
    admission_mode, admit, admit_cascade, cascade_counters_for, counters_for, device_is_supported,
    probe_cuda_compute_capability, CascadeOutcome, ComputeCapability, DispatchCounters,
    DispatchOutcome, PredicateOutcome,
};
use jammi_kernels::ops::{
    apply1, apply2, apply3, AttentionBlockFused, FullyMaskedPolicy, RopeFused, SoftmaxLastDimFused,
    ATTENTION_BLOCK_HEAD_DIM, ATTENTION_BLOCK_MAX_SEQ, MAX_HEAD_DIM, MAX_LAST_DIM, MAX_RANK,
};
use jammi_lora::{effective_rank, should_apply_lora, LoraBuildConfig, LoraLinear, MaybeLoraLinear};

use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::mask::{extended_attention_mask, sliding_window_mask};
use crate::pooling::{pool_and_normalize, Pooling};

const DEFAULT_LAYER_NORM_EPS: f64 = 1e-5;
const DEFAULT_GLOBAL_ROPE_THETA: f64 = 160_000.0;
const DEFAULT_LOCAL_ROPE_THETA: f64 = 10_000.0;
const DEFAULT_LOCAL_ATTENTION: usize = 128;
const DEFAULT_GLOBAL_ATTN_EVERY_N_LAYERS: usize = 3;

fn default_layer_norm_eps() -> f64 {
    DEFAULT_LAYER_NORM_EPS
}
fn default_global_rope_theta() -> f64 {
    DEFAULT_GLOBAL_ROPE_THETA
}
fn default_local_rope_theta() -> f64 {
    DEFAULT_LOCAL_ROPE_THETA
}
fn default_local_attention() -> usize {
    DEFAULT_LOCAL_ATTENTION
}
fn default_global_attn_every_n_layers() -> usize {
    DEFAULT_GLOBAL_ATTN_EVERY_N_LAYERS
}
fn default_attention_dropout() -> f64 {
    0.0
}

/// ModernBERT architecture configuration parsed from `config.json`.
///
/// Fields mirror the HuggingFace ModernBERT config schema, including the
/// sliding-window-local-attention set, which the forward pass honours:
/// `global_attn_every_n_layers` selects which layers are global, and a local
/// layer attends within `local_attention / 2` positions either side using
/// `local_rope_theta` as its RoPE base. See [`ModernBertConfig::is_local_layer`].
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModernBertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_global_rope_theta")]
    pub global_rope_theta: f64,
    #[serde(default = "default_local_rope_theta")]
    pub local_rope_theta: f64,
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    #[serde(default = "default_global_attn_every_n_layers")]
    pub global_attn_every_n_layers: usize,
    /// HuggingFace's `attention_dropout` — parsed (previously silently
    /// dropped: this port's `ModernBertAttention::forward` never read it at
    /// all, an escape row filed alongside the fused attention block that
    /// introduced this field) but refused if nonzero, at
    /// [`ModernBertBuilder::build`] — see [`AttentionBlockFused`]'s own
    /// domain, which has no dropout slot. `0.0` (the default) is ModernBERT's
    /// own upstream default and the only value this port supports.
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,
}

impl ModernBertConfig {
    /// Whether layer `idx` uses sliding-window local attention.
    ///
    /// Panics if `global_attn_every_n_layers` is 0; [`ModernBertBuilder::build`]
    /// refuses such a config before any layer is constructed, so this is
    /// unreachable from a loaded model.
    ///
    /// ModernBERT's rule, matching upstream
    /// (`layer_types[i] = "sliding_attention" if i % global_attn_every_n_layers
    /// else "full_attention"`): layer 0 and every `global_attn_every_n_layers`-th
    /// layer thereafter are global, and the rest are local. A checkpoint with
    /// `global_attn_every_n_layers == 1` is therefore all-global — which is why
    /// a single-layer fixture cannot distinguish an implementation that honours
    /// the window from one that ignores it.
    pub fn is_local_layer(&self, idx: usize) -> bool {
        !idx.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// Half-width of the sliding window: a local layer's query at position `i`
    /// attends to keys `j` with `|i - j| <= half_window`. Upstream stores the
    /// full width and halves it (`sliding_window = local_attention // 2`).
    pub fn half_window(&self) -> usize {
        self.local_attention / 2
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE
// ─────────────────────────────────────────────────────────────────────────────

/// Fused/eager dispatch counters for the ModernBERT RoPE application,
/// read from `jammi_kernels::admission`'s op-keyed registry — see
/// `crate::layer_norm::LN_DISPATCH_COUNTERS`'s doc for why this is a
/// `LazyLock` over `counters_for`, not a directly-owned
/// `static DispatchCounters` (this migration's rationale in full), and
/// this module's "RoPE: table hoisting + fused rotate-half" doc section
/// for the training-only gate this counts. `pub(crate)` (not `pub`) — read
/// via [`crate::rope_dispatch_snapshot`], the same shape
/// [`crate::ln_dispatch_snapshot`] uses.
pub(crate) static ROPE_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("rope_fused"));

// ─────────────────────────────────────────────────────────────────────────────
// Fused masked softmax (C4)
// ─────────────────────────────────────────────────────────────────────────────

/// Fused/eager dispatch counters for the ModernBERT attention softmax,
/// read from the registry — mirroring `ROPE_DISPATCH_COUNTERS` /
/// `crate::layer_norm::LN_DISPATCH_COUNTERS` — see
/// [`softmax_apply_training`]'s doc for the training-only gate
/// this counts. `pub(crate)` (not `pub`) — read via
/// [`crate::softmax_dispatch_snapshot`], the same shape
/// [`crate::rope_dispatch_snapshot`] / [`crate::ln_dispatch_snapshot`] use.
pub(crate) static SOFTMAX_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("softmax_last_dim_fused"));

/// The fused masked-softmax kernel's domain, checked at the call site
/// (family D / K2): `scores`'s device is one
/// [`jammi_kernels::admission::device_is_supported`] accepts, `scores`/`mask`
/// share a dtype the kernel implements (F32 or BF16), BOTH `scores` and
/// `mask` are contiguous (`SoftmaxLastDimFused` refuses a strided view for
/// EITHER argument — see its module doc; an earlier version of this
/// predicate checked only `mask`, asymmetrically, an audit finding
/// corrected here), `scores`'s rank is within [`MAX_RANK`] (the CUDA arm's
/// fixed-arity kernel signature) and last dimension within
/// [`MAX_LAST_DIM`] (a conservative validated ceiling, not a hardware
/// limit — see that constant's own doc), and `mask` is within `scores`'s
/// supported broadcast class
/// ([`jammi_kernels::ops::mask_broadcast_class_holds`] — the SAME check
/// the op applies internally, called directly rather than re-derived here
/// to avoid a second, independently-maintained copy of that logic).
///
/// CORRECTED (an audit finding): an earlier version of this predicate
/// deliberately did NOT check the broadcast class, reasoning that a
/// mismatched mask shape reaching this call site would be "a bug in the
/// caller, not an admission question" — that reasoning does not hold: this
/// function's whole job (K2's "validate, don't silently degrade" doctrine)
/// is to make EVERY domain failure a counted, observable eager fallback
/// rather than an error surfacing from inside the op. Checking the
/// broadcast class here means a mismatched mask shape on the training arm
/// now falls back to eager (counted in [`SOFTMAX_DISPATCH_COUNTERS`]) —
/// the SAME outcome device/dtype/rank/last-dim failures already got —
/// instead of propagating `SoftmaxLastDimFused`'s own internal
/// `candle_core::Error::ShapeMismatchBinaryOp`. The op's own internal
/// check is unchanged and still the correct defense for any direct
/// `apply2` caller that bypasses this predicate entirely.
///
/// `scores_divisor` is `softmax_apply_training`'s OWN `scores_divisor`
/// argument (`sqrt(head_dim)` in production — see that function's doc), a
/// DIVISOR — named to say so explicitly, since `SoftmaxLastDimFused::scale`
/// (a field on a DIFFERENT type, in `jammi-kernels`) is a MULTIPLIER and
/// the two are easy to conflate by name alone. The fused branch folds `1.0
/// / scores_divisor` into `SoftmaxLastDimFused::scale`, and
/// `SoftmaxLastDimFused::with_scale` has a real domain of its own (family
/// D — finite and strictly positive, see its doc). The `scale_finite_positive`
/// clause below checks THAT quantity (`1.0 / scores_divisor` cast to
/// `f32`, the EXACT value the fused branch would pass to `with_scale`),
/// not `scores_divisor` itself directly, so a `scores_divisor` that is
/// finite and positive but produces a non-finite or non-positive
/// reciprocal (e.g. `scores_divisor` so large `1.0 / scores_divisor`
/// underflows to `0.0`, or `scores_divisor == 0.0` itself) is caught here
/// — a counted eager fallback (the SAME `scores / scores_divisor`
/// division the eager branch already performs, so this is never a numeric
/// domain the eager branch could not also handle) rather than
/// `with_scale` refusing deeper in the call stack.
fn softmax_admission_predicate(
    scores: &Tensor,
    mask: &Tensor,
    scores_divisor: f64,
) -> (bool, &'static str) {
    if !device_is_supported(scores.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if scores.dtype() != mask.dtype() || !matches!(scores.dtype(), DType::F32 | DType::BF16) {
        return (false, "dtype_f32_or_bf16_matching_between_scores_and_mask");
    }
    if !scores.is_contiguous() {
        return (false, "scores_contiguous");
    }
    if !mask.is_contiguous() {
        return (false, "mask_contiguous");
    }
    let rank = scores.dims().len();
    if rank == 0 || rank > MAX_RANK {
        return (false, "rank_within_kernel_max_rank");
    }
    let last = *scores.dims().last().unwrap_or(&0);
    if last == 0 || last > MAX_LAST_DIM {
        return (false, "last_dim_within_kernel_max_last_dim");
    }
    if !jammi_kernels::ops::mask_broadcast_class_holds(scores, mask) {
        return (false, "mask_broadcast_class");
    }
    let scale_mul = (1.0 / scores_divisor) as f32;
    if !(scale_mul.is_finite() && scale_mul > 0.0) {
        return (false, "scale_finite_positive");
    }
    (true, "domain_ok")
}

/// Cached, dtype-cast AND pre-broadcast-shaped (`[1, 1, max_seq_len,
/// head_dim]`) RoPE tables — see [`RotaryEmbedding::cached_tables`].
struct CastCache {
    dtype: DType,
    cos: Tensor,
    sin: Tensor,
}

/// Precomputed RoPE cos/sin tables of shape `[max_seq_len, head_dim]`.
///
/// We duplicate the `half_dim` frequencies so the tables are usable with the
/// `rotate_half(x) = cat(-x[..,half:], x[..,:half])` formulation, which is
/// the variant the upstream ModernBERT implementation uses.
///
/// ## Table hoisting: VALUES bit-neutral in eval and training-eager
/// alike — disclosed honestly, not "eval untouched"
///
/// The model has exactly TWO `RotaryEmbedding`s (global/local theta)
/// however many layers it has, but before this change every SINGLE Q/K
/// application re-cast the `f32` source table to the backbone dtype and
/// re-`unsqueeze`d it to broadcast shape — the same two ops, recomputed
/// identically on every one of `2 * num_layers` calls per forward.
/// [`Self::cached_tables`] computes that cast+unsqueeze ONCE per dtype
/// (memoised in `cast_cache`, a `Mutex` for the same reason
/// `ModernBert::band_cache` is one — the model is held across threads)
/// and every call — [`Self::apply`] (used in eval and as the training
/// fallback) AND [`Self::apply_training`] (the fused path) alike —
/// reuses it. The OUTPUT VALUES are BIT-NEUTRAL: the cached tensor holds
/// the exact same bytes a fresh `to_dtype`/`unsqueeze` pair would have
/// produced from the same source table (no rounding is introduced by
/// caching, only redundant recomputation is removed) — asserted by
/// `tests::table_hoisting_is_bit_neutral_with_the_uncached_computation`.
/// This is NOT the same claim as "eval's call sequence is unchanged":
/// `apply` now acquires `cast_cache`'s lock once per call (`2 *
/// num_layers` times per forward, uncontended in the single-model-per-
/// thread shape every caller in this repository uses today — a `Mutex`
/// lock/unlock pair on an uncontended lock is on the order of tens of
/// nanoseconds, negligible next to a single matmul, but a real,
/// previously-absent synchronization point this doc does not hide).
///
/// ## The fused rotate-half kernel: training-only
///
/// [`Self::apply_training`] is the ONLY call site that may dispatch to
/// [`jammi_kernels::ops::RopeFused`]; [`Self::apply`] never does, so
/// eval's OUTPUT VALUES (which always come from `apply`, see
/// `ModernBertAttention::forward`) are bit-identical before and after
/// this commit — table hoisting's lock is the only thing eval's call
/// shape gains. `apply_training` itself still falls back to `apply`
/// whenever the fused kernel's domain check fails (K2 admission) — the
/// training path is therefore "fused when possible, otherwise identical
/// to eval's own path", never a third distinct numeric path.
///
/// This "never a third path" property is specific to `RotaryEmbedding`'s
/// OWN training arm, not a doctrine every fused kernel in this file
/// shares: `softmax_apply_training`'s `Zeros` fully-masked-row behavior IS
/// a genuine third numeric path (matching neither eval's `candle_nn::ops::softmax`
/// output NOR that same function's own eager-fallback branch within
/// training) — see that function's doc for the full disclosure.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    cast_cache: Mutex<Option<CastCache>>,
    /// [`Self::cached_rope_pack`]'s memo, keyed by dtype exactly like
    /// `cast_cache` (single-entry: one backbone dtype per model
    /// lifetime). Separate from `cast_cache` so eval — which calls
    /// [`Self::cached_tables`] but never the pack — never pays the stack.
    rope_pack_cache: Mutex<Option<(DType, Tensor)>>,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_base: f64,
        device: &Device,
    ) -> Result<Self, EncoderError> {
        let half = head_dim / 2;
        let mut cos_vec = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_vec = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for _half_pass in 0..2 {
                for i in 0..half {
                    let theta = (pos as f64) * (rope_base.powf(-2.0 * i as f64 / head_dim as f64));
                    cos_vec.push(theta.cos() as f32);
                    sin_vec.push(theta.sin() as f32);
                }
            }
        }

        let cos = Tensor::from_vec(cos_vec, (max_seq_len, head_dim), device)?;
        let sin = Tensor::from_vec(sin_vec, (max_seq_len, head_dim), device)?;

        Ok(Self {
            cos,
            sin,
            cast_cache: Mutex::new(None),
            rope_pack_cache: Mutex::new(None),
        })
    }

    /// Returns the `[1, 1, max_seq_len, head_dim]` cos/sin tables cast to
    /// `dtype`, computing and memoising them on the first call for that
    /// dtype (see the struct doc's "table hoisting" section). A model
    /// instance uses exactly one backbone dtype for its lifetime, so this
    /// is a single-entry cache in practice; a later call with a DIFFERENT
    /// dtype still computes the right answer (it just recomputes and
    /// overwrites the single cached entry rather than growing a map — a
    /// case that never arises for one model instance).
    fn cached_tables(&self, dtype: DType) -> Result<(Tensor, Tensor), EncoderError> {
        {
            let cache = self
                .cast_cache
                .lock()
                .map_err(|_| EncoderError::Config("RoPE table cache poisoned".into()))?;
            if let Some(c) = cache.as_ref() {
                if c.dtype == dtype {
                    return Ok((c.cos.clone(), c.sin.clone()));
                }
            }
        }
        let cos = self.cos.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)?;
        let mut cache = self
            .cast_cache
            .lock()
            .map_err(|_| EncoderError::Config("RoPE table cache poisoned".into()))?;
        *cache = Some(CastCache {
            dtype,
            cos: cos.clone(),
            sin: sin.clone(),
        });
        Ok((cos, sin))
    }

    /// Apply RoPE to a `[batch, num_heads, seq, head_dim]` tensor — the
    /// eager composition, whose OUTPUT VALUES are unchanged from before
    /// table hoisting existed (see the struct doc's disclosure: this call
    /// now also takes `cast_cache`'s lock once, which is new). Used
    /// directly in eval and as [`Self::apply_training`]'s fallback.
    fn apply(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (_batch, _heads, seq, head_dim) = x.dims4()?;
        let half = head_dim / 2;
        let x_dtype = x.dtype();

        let (cos_full, sin_full) = self.cached_tables(x_dtype)?;
        let cos = cos_full.narrow(2, 0, seq)?;
        let sin = sin_full.narrow(2, 0, seq)?;

        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let neg_x2 = (x2 * -1.0f64)?;
        let rot_half = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;

        let cos_part = x.broadcast_mul(&cos)?;
        let sin_part = rot_half.broadcast_mul(&sin)?;
        Ok((cos_part + sin_part)?)
    }

    /// `[2, 1, 1, max_seq_len, head_dim]` — [`Self::cached_tables`]'s own
    /// `(cos, sin)` pair `Tensor::stack`-ed along a new leading axis, the
    /// exact packing [`jammi_kernels::ops::AttentionBlockFused`]'s
    /// `rope_pack` argument requires (see that op's module doc's "The
    /// `rope_pack` argument" section for why: `CustomOp3`'s 3-tensor arity
    /// has no room for `cos`/`sin` as separate arguments alongside `qkv`
    /// and `mask`). A pure memory copy of the SAME cached bytes
    /// `Self::cached_tables` already produces — no new rounding.
    ///
    /// MEMOISED per dtype (`rope_pack_cache`), built on the first call.
    /// An earlier revision stacked it fresh on every call: one
    /// `[2, 1, 1, max_seq_len, head_dim]` device copy per training-arm
    /// forward per layer, AND — because each layer's `AttentionBlockFused`
    /// node clones its `rope_pack` argument into its own `Op` until the
    /// backward pass releases it — one such tensor RETAINED per layer for
    /// the whole step: at `max_position_embeddings = 8192`, `head_dim =
    /// 64`, `BF16`, `2 * 8192 * 64 * 2 = 2_097_152` bytes (2 MB) per
    /// layer, `28 * 2 MB = 56 MB` retained per step plus 28 device-to-
    /// device copies on a 28-layer model. Now: the SAME storage (one
    /// `Arc`) is handed to every layer, so per forward per layer the pack
    /// costs 0 bytes and 0 copies; what is retained is one pack per
    /// `RotaryEmbedding` per dtype for the model's lifetime — 2 MB per
    /// table at the shape above, 4 MB for the global+local pair.
    ///
    /// Not NARROWED to `seq`: a `narrow` along the pack's position axis
    /// is non-contiguous (the cos block and the sin block sit
    /// `max_seq_len * head_dim` elements apart), the op refuses a
    /// non-contiguous `rope_pack` (its `check_rope_pack` +
    /// `contiguous_offsets`), and a contiguous narrow would be a fresh
    /// per-call copy — exactly the cost this memo removes. The op reads
    /// rows `[0, seq)` of each block and nothing else, so the full table
    /// costs no extra reads.
    fn cached_rope_pack(&self, dtype: DType) -> Result<Tensor, EncoderError> {
        {
            let cache = self
                .rope_pack_cache
                .lock()
                .map_err(|_| EncoderError::Config("RoPE pack cache poisoned".into()))?;
            if let Some((cached_dtype, pack)) = cache.as_ref() {
                if *cached_dtype == dtype {
                    return Ok(pack.clone());
                }
            }
        }
        let (cos, sin) = self.cached_tables(dtype)?;
        let pack = Tensor::stack(&[&cos, &sin], 0)?;
        let mut cache = self
            .rope_pack_cache
            .lock()
            .map_err(|_| EncoderError::Config("RoPE pack cache poisoned".into()))?;
        *cache = Some((dtype, pack.clone()));
        Ok(pack)
    }

    /// The training-mode arm: dispatches to
    /// [`jammi_kernels::ops::RopeFused`] when its domain holds, else
    /// falls back to [`Self::apply`] (recording which happened either
    /// way, mirroring `crate::layer_norm`'s LN admission mechanism). Only
    /// ever called when the caller's `training` flag is `true` — see
    /// `ModernBertAttention::forward`.
    fn apply_training(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (_batch, _heads, seq, head_dim) = x.dims4()?;
        let x_dtype = x.dtype();
        let (cos_full, sin_full) = self.cached_tables(x_dtype)?;
        let cos = cos_full.narrow(2, 0, seq)?;
        let sin = sin_full.narrow(2, 0, seq)?;

        let (holds, predicate) =
            rope_admission_predicate(x_dtype, x.device(), &cos, &sin, head_dim);
        let outcome = admit(
            admission_mode(),
            "rope_fused",
            predicate,
            holds,
            *ROPE_DISPATCH_COUNTERS,
        )?;
        match outcome {
            DispatchOutcome::Fused => {
                // Paid ONLY on the admitted-fused branch: `x` is a
                // `transpose(1, 2)` view (see `ModernBertAttention::forward`)
                // and therefore not contiguous, which the fused kernel's
                // domain requires (see `RopeFused`'s module doc).
                //
                // Honest accounting (do not overclaim this is free): in the
                // EAGER path, `x` is never separately materialised before
                // RoPE — `Tensor::cat` (building `rotate_half(x)`) already
                // produces a contiguous OUTPUT as an intrinsic side effect
                // of computing the rotation itself, not via a distinct copy
                // step, so `crate::contiguous_matmul`'s own `.contiguous()`
                // call right after was ALREADY a no-op before this commit
                // (its argument was already contiguous). The fused path's
                // `x.contiguous()` here is therefore a GENUINE additional
                // memory copy the eager path never isolated as a separate
                // cost — one bandwidth-bound elementwise pass, materially
                // cheaper than the ~12-op chain (2 narrows, a `neg`, the
                // `cat`, 2 broadcast-muls, an add, plus the per-call
                // `to_dtype`/`unsqueeze` table hoisting already removes) it
                // replaces, but a real cost, not a wash. Downstream
                // `contiguous_matmul` sees an already-contiguous fused
                // output either way (every `CustomOp` allocates a fresh
                // contiguous buffer), so it stays a no-op there in both
                // paths.
                let x_c = x.contiguous()?;
                Ok(apply3(&x_c, &cos, &sin, RopeFused::new(false))?)
            }
            DispatchOutcome::Eager => self.apply(x),
        }
    }
}

/// The fused RoPE kernel's domain, checked at the call site (family D /
/// K2): `x`'s device is one [`jammi_kernels::admission::device_is_supported`]
/// accepts (CPU, or CUDA when this build compiled `jammi-kernels`' `cuda`
/// arm), `x`/`cos`/`sin` share a dtype the kernel implements (F32 or
/// BF16), `cos`/`sin` are contiguous (guaranteed by construction —
/// [`RotaryEmbedding::cached_tables`] always produces a genuinely
/// contiguous tensor and `narrow` on its leading dim preserves that — this
/// check is a defensive re-verification, not a load-bearing "maybe fails"
/// branch: family D's "validate at every numeric edge", not an assumption
/// left silently trusted), `head_dim` is nonzero and even (rotate-half's
/// domain), and within [`MAX_HEAD_DIM`] (a conservative validated ceiling,
/// not a hardware limit — see that constant's own doc). Returns the
/// aggregate predicate and the name of whichever check is the reason (or
/// a fixed "domain_ok" name when everything holds).
fn rope_admission_predicate(
    x_dtype: DType,
    x_device: &Device,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
) -> (bool, &'static str) {
    if !device_is_supported(x_device) {
        return (false, "device_is_cpu_or_cuda");
    }
    if x_dtype != cos.dtype()
        || x_dtype != sin.dtype()
        || !matches!(x_dtype, DType::F32 | DType::BF16)
    {
        return (false, "dtype_f32_or_bf16_matching_between_x_cos_sin");
    }
    if !cos.is_contiguous() || !sin.is_contiguous() {
        return (false, "cos_sin_contiguous");
    }
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return (false, "head_dim_even_and_nonzero");
    }
    if head_dim > MAX_HEAD_DIM {
        return (false, "head_dim_within_kernel_max_head_dim");
    }
    (true, "domain_ok")
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Fused whole-attention-block (P3, Tier 0)
// ─────────────────────────────────────────────────────────────────────────────

/// Fused/eager dispatch counters for ModernBERT's training-mode fused
/// whole-attention-block, read from `jammi_kernels::admission`'s op-keyed
/// registry — mirroring `ROPE_DISPATCH_COUNTERS`/`SOFTMAX_DISPATCH_COUNTERS`
/// (a `LazyLock` over `counters_for`, not a directly-owned
/// `static DispatchCounters`; this op is new enough to start on the
/// registry directly rather than needing its own migration) — see
/// `ModernBertAttention::forward_training_attention`'s doc for the
/// training-only gate this counts. `pub(crate)` (not `pub`) — read via
/// [`crate::attention_block_dispatch_snapshot`], the same shape the other
/// three snapshot functions use.
pub(crate) static ATTENTION_BLOCK_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("attention_block_fused"));

/// The fused whole-attention-block kernel's domain, checked at the call
/// site (family D / K2): `qkv`'s device is one
/// [`jammi_kernels::admission::device_is_supported`] accepts, `qkv`/`extended_mask`
/// share a dtype the kernel implements (F32 or BF16), `qkv` is contiguous
/// (the free reshape from `Wqkv`'s own output — `AttentionBlockFused`
/// refuses a strided `qkv`, same idiom as every other op in this crate),
/// `head_dim` is exactly [`ATTENTION_BLOCK_HEAD_DIM`] (`AttentionBlockFused`'s
/// own fixed domain — see that op's module doc's "Fixed domain" section),
/// `seq` is nonzero and within [`ATTENTION_BLOCK_MAX_SEQ`], `extended_mask`
/// (the padding mask ALONE, before any band is folded in — this op has no
/// `window` construction data of its own; the caller combines padding and
/// band into ONE tensor before calling it, see
/// [`ModernBertAttention::forward_training_attention`]'s doc) is contiguous
/// and shaped `[batch|1, 1, 1, seq]`, and — on a local layer only —
/// `local_mask` (the per-forward padding-plus-band sum,
/// [`FusedAttentionMasks::local`]) is present, contiguous, and shaped
/// `[batch|1, 1, seq, seq]` — the op's own `check_mask` domain for the
/// padding-plus-band class.
///
/// | `is_local` | `local_mask` | contiguous | shape | outcome |
/// |---|---|---|---|---|
/// | `false` | any | — | — | not consulted |
/// | `true` | `None` | — | — | `local_mask_present` |
/// | `true` | `Some` | no | — | `local_mask_contiguous` |
/// | `true` | `Some` | yes | ≠ `[batch\|1, 1, seq, seq]` | `local_mask_shape_batch_or_one_1_seq_seq` |
/// | `true` | `Some` | yes | `[batch\|1, 1, seq, seq]` | `domain_ok` |
fn attention_block_admission_predicate(
    qkv: &Tensor,
    seq: usize,
    _h: usize,
    d: usize,
    extended_mask: &Tensor,
    is_local: bool,
    local_mask: Option<&Tensor>,
) -> (bool, &'static str) {
    if !device_is_supported(qkv.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if qkv.dtype() != extended_mask.dtype() || !matches!(qkv.dtype(), DType::F32 | DType::BF16) {
        return (false, "dtype_f32_or_bf16_matching_between_qkv_and_mask");
    }
    if !qkv.is_contiguous() {
        return (false, "qkv_contiguous");
    }
    if d != ATTENTION_BLOCK_HEAD_DIM {
        return (false, "head_dim_is_attention_block_fixed_head_dim");
    }
    if seq == 0 || seq > ATTENTION_BLOCK_MAX_SEQ {
        return (false, "seq_within_attention_block_max_seq");
    }
    if !extended_mask.is_contiguous() {
        return (false, "mask_contiguous");
    }
    let m_dims = extended_mask.dims();
    if m_dims.len() != 4 || m_dims[1] != 1 || m_dims[2] != 1 || m_dims[3] != seq {
        return (false, "mask_shape_batch_or_one_1_1_seq");
    }
    if is_local {
        let Some(local) = local_mask else {
            return (false, "local_mask_present");
        };
        if !local.is_contiguous() {
            return (false, "local_mask_contiguous");
        }
        let l_dims = local.dims();
        if l_dims.len() != 4
            || (l_dims[0] != 1 && l_dims[0] != m_dims[0])
            || l_dims[1] != 1
            || l_dims[2] != seq
            || l_dims[3] != seq
        {
            return (false, "local_mask_shape_batch_or_one_1_seq_seq");
        }
    }
    (true, "domain_ok")
}

/// The additive masks the FUSED whole-attention-block arm consumes,
/// built ONCE per training forward by [`ModernBert::forward_hidden`] —
/// never per layer. An earlier revision rebuilt them inside every local
/// layer's fused arm (`extended.to_dtype` + `band.to_dtype` +
/// `broadcast_add`, 3 launches) and every global layer's (`to_dtype`, 1
/// launch): on a 28-layer / 10-global / 18-local ModernBERT-large
/// forward that is `10 + 18 * 3 = 64` mask launches per training step;
/// this struct's [`Self::build`] issues at most 3 (`global` cast, band
/// add, `local` cast — and the casts are storage-sharing clones, not
/// launches, when the backbone dtype is already `F32`).
///
/// ## Rounding: add-in-F32-then-cast is bit-identical to cast-then-add
///
/// The per-layer revision cast each term to the backbone dtype and added
/// in that dtype; this one adds in `F32` and casts the SUM. On the only
/// values either term takes — `0.0` and [`crate::mask::MASKED_LOGIT`]
/// (`-10_000.0`) — the two orders agree byte-for-byte at every backbone
/// dtype the fused arm admits: at `F32` both are exact; at `BF16`
/// (8 significand bits, spacing `64` at `2^13`) `-10_000` rounds to
/// `-9_984` either way, and the doubly-masked `-20_000` rounds to
/// `-19_968`, which is exactly `-9_984 + -9_984` (spacing `128` at
/// `2^14`: `20_000` sits `32` above `19_968` and `96` below `20_096`).
/// `tests::fused_masks_add_then_cast_is_bit_identical_to_cast_then_add_on_the_masked_logit_lattice`
/// sweeps all four `(padding, band)` cells at both dtypes — the sweep is
/// the claim, not this arithmetic.
struct FusedAttentionMasks {
    /// `[batch, 1, 1, seq]` in the backbone dtype — the padding mask
    /// alone, what a GLOBAL layer's fused arm passes as `mask`.
    global: Tensor,
    /// `[batch, 1, seq, seq]` in the backbone dtype — padding plus the
    /// sliding-window band, what a LOCAL layer's fused arm passes.
    /// `None` iff the model has no local layer.
    local: Option<Tensor>,
}

impl FusedAttentionMasks {
    /// `extended_f32` is [`extended_attention_mask`]'s `[batch, 1, 1,
    /// seq]` output, `local_band_f32` is [`sliding_window_mask`]'s
    /// `[1, 1, seq, seq]` band (or `None` for an all-global model), and
    /// `dtype` is the backbone dtype the attention `qkv` will carry.
    fn build(
        extended_f32: &Tensor,
        local_band_f32: Option<&Tensor>,
        dtype: DType,
    ) -> Result<Self, EncoderError> {
        let global = extended_f32.to_dtype(dtype)?;
        let local = match local_band_f32 {
            Some(band) => Some(extended_f32.broadcast_add(band)?.to_dtype(dtype)?),
            None => None,
        };
        Ok(Self { global, local })
    }
}

/// The three mask inputs [`ModernBertAttention::forward_training_attention`]
/// takes, bundled: the `F32` padding mask and band the eager FALLBACK
/// composition consumes verbatim (unchanged from before the fused arm
/// existed), and the per-forward [`FusedAttentionMasks`] the fused arm
/// consumes.
struct TrainingMaskInputs<'a> {
    extended: &'a Tensor,
    local_band: Option<&'a Tensor>,
    fused: &'a FusedAttentionMasks,
}

struct ModernBertAttention {
    wqkv: MaybeLoraLinear,
    wo: MaybeLoraLinear,
    /// `None` for layer 0 — the embedding `norm` already pre-normalises the
    /// input there, so the layer holds an identity pre-norm.
    attn_norm: Option<LayerNorm>,
    /// The RoPE table for this layer's attention type. Shared, because a model
    /// has exactly two tables (global and local) however many layers it has.
    rope: Arc<RotaryEmbedding>,
    /// `true` when this layer attends within a sliding window rather than over
    /// the whole sequence. The band itself is built once per forward and passed
    /// in, since it depends only on the sequence length.
    is_local: bool,
    num_heads: usize,
    head_dim: usize,
    /// `Some(local_attention / 2)` for a local layer, `None` for a global
    /// one — the SAME quantity [`ModernBertConfig::half_window`] derives,
    /// stored per-layer so [`Self::forward_flash_dense_attention`] can
    /// build a `VarlenConfig::window` without threading the whole model
    /// config down to a single attention layer just for this one field
    /// (contract v5 §3.6: "`half_window = local_attention/2` ->
    /// `VarlenConfig::window`"). Read only under the `flash-attn` feature
    /// (`VarlenConfig`/`CuSeqlens` do not exist otherwise).
    #[allow(dead_code)]
    half_window: Option<usize>,
    /// Whether the fused RoPE kernel may be attempted on Q/K (still gated
    /// by its own domain check — see [`RotaryEmbedding::apply_training`]).
    /// `false` (eval/serving) always calls [`RotaryEmbedding::apply`]
    /// directly, with output VALUES bit-identical to before this field
    /// existed (see `RotaryEmbedding`'s struct doc for the one disclosed,
    /// non-numeric change — a table-cache lock — `apply` itself gained);
    /// `true` (training) is the ONLY state that ever reaches
    /// `apply_training`. Propagated by [`ModernBert::set_training`], the
    /// same mechanism `LayerNorm::set_training` uses.
    training: bool,
}

impl ModernBertAttention {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// `local_band` is the `[1, 1, seq, seq]` sliding-window mask, supplied
    /// whenever the model has any local layer. A global layer ignores it.
    /// `fused_masks` is the per-forward [`FusedAttentionMasks`] bundle:
    /// REQUIRED in training mode (a typed `Config` refusal otherwise — the
    /// fused arm never rebuilds it per layer), ignored in eval (eval never
    /// reads it; passing `Some` or `None` there is byte-identical — see
    /// `tests::attention_block_eval_output_is_bit_identical_regardless_of_fused_eligibility`).
    ///
    /// | `self.training` | `fused_masks` | outcome |
    /// |---|---|---|
    /// | `false` | `None` | eval path |
    /// | `false` | `Some` | eval path (identical output; the bundle is unread) |
    /// | `true` | `Some` | training path |
    /// | `true` | `None` | `EncoderError::Config` (`tests::training_attention_forward_without_fused_masks_is_a_typed_refusal`) |
    fn forward(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        local_band: Option<&Tensor>,
        fused_masks: Option<&FusedAttentionMasks>,
        flash: Option<&FlashDecision>,
    ) -> Result<Tensor, EncoderError> {
        let normed = match &self.attn_norm {
            Some(ln) => ln.forward(hidden)?,
            None => hidden.clone(),
        };
        let (batch, seq, _) = normed.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;

        let qkv = self.wqkv.forward(&normed)?;

        let ctx = if self.training {
            let Some(fused) = fused_masks else {
                return Err(EncoderError::Config(
                    "training-mode attention reached without the per-forward fused masks — \
                     ModernBert::forward_hidden builds them once per forward; a direct caller \
                     in training mode must supply them too"
                        .into(),
                ));
            };
            let Some(flash) = flash else {
                return Err(EncoderError::Config(
                    "training-mode attention reached without the per-forward flash-cascade \
                     decision — ModernBert::forward_hidden decides it once per forward \
                     (contract v4 §3.2); a direct caller in training mode must supply it too"
                        .into(),
                ));
            };
            self.forward_training_attention(
                &qkv,
                batch,
                seq,
                h,
                d,
                TrainingMaskInputs {
                    extended: extended_mask,
                    local_band,
                    fused,
                },
                flash,
            )?
        } else {
            self.forward_eval_attention(&qkv, batch, seq, h, d, extended_mask, local_band)?
        };

        let out = self.wo.forward(&ctx)?;
        Ok((out + hidden)?)
    }

    /// Eval's UNCHANGED code path, extracted verbatim (not rewritten) from
    /// what `forward` inlined before this commit — bit-identical to before
    /// the fused whole-attention-block op existed: two SEQUENTIAL
    /// broadcast-adds, each from its own smaller shape, never combined into
    /// one tensor (see `crate::mask::sliding_window_mask`'s doc for why —
    /// neither mask is ever materialised at `[batch, heads, seq, seq]`
    /// either way, but combining them first would round differently than
    /// adding them in this order, which eval must never do — see
    /// `tests::eval_mode_attention_softmax_is_bit_identical_regardless_of_fused_eligibility`).
    /// `forward`'s `!self.training` branch is the ONLY caller, so this
    /// function's own admission machinery has no bearing on eval at all —
    /// the "deletion test" property the op contract asks for (eval never
    /// even sees `AttentionBlockFused` exist).
    #[allow(clippy::too_many_arguments)]
    fn forward_eval_attention(
        &self,
        qkv: &Tensor,
        batch: usize,
        seq: usize,
        h: usize,
        d: usize,
        extended_mask: &Tensor,
        local_band: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let q = qkv
            .narrow(D::Minus1, 0, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;
        let k = qkv
            .narrow(D::Minus1, h * d, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;
        let v = qkv
            .narrow(D::Minus1, 2 * h * d, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;

        let q = self.rope.apply(&q)?;
        let k = self.rope.apply(&k)?;

        let scale = (d as f64).sqrt();
        let scores = crate::contiguous_matmul(&q, &k.transpose(D::Minus1, D::Minus2)?)?;
        let scores = (scores / scale)?;
        let extended_mask = extended_mask.to_dtype(scores.dtype())?;

        let scores = scores.broadcast_add(&extended_mask)?;
        let scores = match (self.is_local, local_band) {
            (true, Some(band)) => scores.broadcast_add(&band.to_dtype(scores.dtype())?)?,
            (true, None) => {
                return Err(EncoderError::Config(
                    "local-attention layer reached without a sliding-window band".into(),
                ))
            }
            (false, _) => scores,
        };
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        Ok(crate::contiguous_matmul(&attn, &v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, h * d))?)
    }

    /// Training's arm: attempts
    /// [`jammi_kernels::ops::AttentionBlockFused`] — the WHOLE
    /// RoPE+`QKᵀ`+mask+softmax+`PV` chain as ONE tape node — when its
    /// domain holds (see [`attention_block_admission_predicate`]), else
    /// falls back to [`Self::forward_eager_training_attention_composition`],
    /// which is TODAY'S exact training-arm composition (the same partial
    /// fusion — RoPE and softmax each fused independently, everything else
    /// eager — this crate shipped before this commit), UNCHANGED. Recording
    /// which happened either way, mirroring every other admission-gated
    /// call site in this file.
    ///
    /// The fused arm's `mask` argument is read STRAIGHT out of
    /// `masks.fused` (`global` for a global layer, `local` for a local
    /// one) — this method builds, casts, or combines NO mask of its own;
    /// [`FusedAttentionMasks::build`] in `ModernBert::forward_hidden` did
    /// that once for the whole forward (see that struct's doc for the
    /// launch count). The eager fallback keeps consuming the `F32`
    /// `masks.extended`/`masks.local_band` pair verbatim.
    ///
    /// This is the call site `AttentionBlockFused`'s own module doc's
    /// "`BF16` validated-coverage ceiling" section cites: the `qkv`
    /// reaching the fused arm here, on the REAL ModernBERT-large
    /// checkpoint (`batch=8, seq=512`, seed 42, sha
    /// `8922094aa35d381d108420fefe82cba122bf6ebb`), measures
    /// `max|qkv| ≈ 9`–`18` — an order of magnitude past that op's own
    /// `BF16`-derived-bound validated range (`|qkv| <= 1`). The op still
    /// computes correctly there; see that section for what IS and is NOT
    /// claimed at this amplitude.
    #[allow(clippy::too_many_arguments)]
    fn forward_training_attention(
        &self,
        qkv: &Tensor,
        batch: usize,
        seq: usize,
        h: usize,
        d: usize,
        masks: TrainingMaskInputs<'_>,
        flash: &FlashDecision,
    ) -> Result<Tensor, EncoderError> {
        if self.is_local && masks.local_band.is_none() {
            return Err(EncoderError::Config(
                "local-attention layer reached without a sliding-window band".into(),
            ));
        }

        // Flash cascade (contract v4 §3.2/§3.3, wired for DENSE by P6 Stage
        // B B3): consulted PER LAYER so the counters are per-dispatch, not
        // per-forward, even though the eligibility decision itself was made
        // once, above, in `ModernBert::forward_hidden`. `true`: the block
        // arm (this function's own fallback) always can run, so Strict mode
        // never errors on the flash arm's decline either.
        let flash_dispatch = admit_cascade(
            admission_mode(),
            "attention_block_flash",
            flash.reason(),
            flash.outcome(),
            true,
            cascade_counters_for("attention_block_flash"),
        )?;
        if flash_dispatch == CascadeOutcome::Fused {
            // `FlashDecision::Fused` ALWAYS carries a `CompactedBatch` — by
            // construction, not by a runtime `Option` check (see that
            // type's own doc): `build_flash_forward_decision` has no path
            // that produces `Fused` without one. The `Declined` arm below
            // is therefore reachable only if `admit_cascade` and
            // `flash.outcome()` ever disagreed about what `flash` itself
            // is, which would be an admission-layer bug, not a missing
            // field — a typed refusal either way, never a silent wrong
            // dispatch.
            let admission = match flash {
                FlashDecision::Fused(batch) => batch,
                FlashDecision::Declined { outcome, reason } => {
                    return Err(EncoderError::Config(format!(
                        "attention_block_flash dispatched Fused but flash itself is \
                         Declined(outcome={outcome:?}, reason={reason}) -- admit_cascade and \
                         FlashDecision disagree"
                    )))
                }
            };
            return self.forward_flash_dense_attention(qkv, batch, seq, h, d, admission);
        }

        let (holds, predicate) = attention_block_admission_predicate(
            qkv,
            seq,
            h,
            d,
            &masks.fused.global,
            self.is_local,
            masks.fused.local.as_ref(),
        );
        let outcome = admit(
            admission_mode(),
            "attention_block_fused",
            predicate,
            holds,
            *ATTENTION_BLOCK_DISPATCH_COUNTERS,
        )?;
        match outcome {
            DispatchOutcome::Fused => {
                let qkv5 = qkv.reshape((batch, seq, 3, h, d))?;
                let rope_pack = self.rope.cached_rope_pack(qkv.dtype())?;
                // `AttentionBlockFused` has no `window` construction data
                // (its module doc's "window is construction data at the
                // call site" section): a local layer hands it the
                // per-forward padding-plus-band sum, a global layer the
                // padding mask alone. The admission predicate above
                // already refused a local layer whose `local` bundle is
                // missing, so the `(true, None)` arm below is a typed
                // belt-and-braces refusal, not a reachable path.
                let mask = match (self.is_local, masks.fused.local.as_ref()) {
                    (true, Some(local)) => local,
                    (true, None) => {
                        return Err(EncoderError::Config(
                            "local-attention layer reached without a combined fused mask".into(),
                        ))
                    }
                    (false, _) => &masks.fused.global,
                };
                let op = AttentionBlockFused::new(
                    1.0 / (d as f32).sqrt(),
                    FullyMaskedPolicy::Zeros,
                    true,
                )?;
                Ok(apply3(&qkv5, &rope_pack, mask, op)?)
            }
            DispatchOutcome::Eager => self.forward_eager_training_attention_composition(
                qkv,
                batch,
                seq,
                h,
                d,
                masks.extended,
                masks.local_band,
            ),
        }
    }

    /// The DENSE FlashAttention-2 arm (P6 Stage B B3-dense, contract v5
    /// §3.6): `admission.lengths` is uniform (`decide_flash_admission`
    /// only reaches here when every row's length `== seq` — see that
    /// function's dense/padded split) so NO gather/scatter is needed —
    /// `total == batch * seq` and `qkv.reshape((total, 3, h, d))` is the
    /// SAME free view [`ModernBertAttention::forward_training_attention`]'s
    /// block arm already takes (R1/R2). RoPE: [`RopePositionsFused`] on
    /// the packed buffer (ONE launch, q and k; see that op's module doc),
    /// sharing the SAME `cos`/`sin` tables `RopeFused` consumes. Backward
    /// flows through the op's own `Saved` LSE (`flash_attention_varlen`'s
    /// `CustomOp1::bwd`) and `RopePositionsFused::bwd`'s sign-flip reuse —
    /// candle's ordinary autograd composition, no manual backward call
    /// here.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "flash-attn")]
    fn forward_flash_dense_attention(
        &self,
        qkv: &Tensor,
        batch: usize,
        seq: usize,
        h: usize,
        d: usize,
        admission: &CompactedBatch,
    ) -> Result<Tensor, EncoderError> {
        use jammi_kernels::flash::{CuSeqlens, VarlenConfig};
        use jammi_kernels::ops::flash_attention_varlen_with_rope;

        let total = batch * seq;
        let qkv5 = qkv.reshape((total, 3, h, d))?;
        let (cos_full, sin_full) = self.rope.cached_tables(qkv5.dtype())?;
        let cos = cos_full.narrow(2, 0, seq)?;
        let sin = sin_full.narrow(2, 0, seq)?;
        // `flash_attention_varlen_with_rope` fuses the RoPE rotation INSIDE
        // its own `CustomOp3` node rather than rotating `qkv5` here first
        // (contrast the pre-fix composition: `apply3(&qkv5, &cos, &sin,
        // RopePositionsFused::new(seq, false))` then
        // `flash_attention_varlen(&qkv_rot, ...)`) — that two-op shape left
        // a SECOND `[total, 3, h, d]` bf16 buffer (`qkv_rot`) retained by
        // candle's tape for the whole backward pass, on top of the
        // pre-rotation `qkv` the `Wqkv` linear's own backward already needs
        // alive; see `flash_attention_varlen_with_rope`'s own doc for the
        // measured per-layer cost this closes.

        let cuda_device = match qkv.device() {
            Device::Cuda(dev) => dev,
            _ => {
                return Err(EncoderError::Config(
                    "attention_block_flash dispatched Fused on a non-CUDA device -- \
                     unreachable: decide_flash_admission gates on device.is_cuda() before \
                     returning Holds"
                        .into(),
                ))
            }
        };
        let cu_seqlens = CuSeqlens::from_lengths(&admission.lengths, cuda_device)
            .map_err(|e| EncoderError::Config(format!("attention_block_flash: {e}")))?;
        let cfg = VarlenConfig {
            softmax_scale: 1.0 / (d as f32).sqrt(),
            window: self.half_window.map(|w| w as u32),
            deterministic: true,
        };
        let o = flash_attention_varlen_with_rope(&qkv5, &cos, &sin, seq, &cu_seqlens, &cfg)
            .map_err(|e| EncoderError::Config(format!("attention_block_flash: {e}")))?;
        Ok(o.reshape((batch, seq, h * d))?)
    }

    /// Non-`flash-attn`-build stub: structurally unreachable at runtime
    /// (`decide_flash_admission` gates `PredicateOutcome::Holds` on
    /// `jammi_kernels::admission::FLASH_COMPILED`, which is `false` on
    /// this build), but the CALL SITE above must still compile without
    /// the `flash-attn` feature — this crate's own `flash-attn` feature
    /// is what gates whether `jammi_kernels::flash`/`ops::
    /// flash_attention_varlen` even exist to name (see this crate's
    /// `Cargo.toml`'s `flash-attn` stanza doc); a bare `cfg!()` runtime
    /// check around a reference to those types would fail to COMPILE
    /// with the feature off, not merely fail at runtime.
    #[allow(clippy::too_many_arguments)]
    #[cfg(not(feature = "flash-attn"))]
    fn forward_flash_dense_attention(
        &self,
        _qkv: &Tensor,
        _batch: usize,
        _seq: usize,
        _h: usize,
        _d: usize,
        _admission: &CompactedBatch,
    ) -> Result<Tensor, EncoderError> {
        Err(EncoderError::Config(
            "attention_block_flash dispatched Fused on a build without the flash-attn feature \
             -- unreachable: decide_flash_admission gates on FLASH_COMPILED before returning \
             Holds"
                .into(),
        ))
    }

    /// TODAY'S exact training-arm composition, extracted verbatim (not
    /// rewritten) so [`Self::forward_training_attention`]'s eager fallback
    /// is provably identical to what this crate shipped before
    /// `AttentionBlockFused` existed — the op contract's "eager fallback ==
    /// today's exact composition" requirement.
    #[allow(clippy::too_many_arguments)]
    fn forward_eager_training_attention_composition(
        &self,
        qkv: &Tensor,
        batch: usize,
        seq: usize,
        h: usize,
        d: usize,
        extended_mask: &Tensor,
        local_band: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let q = qkv
            .narrow(D::Minus1, 0, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;
        let k = qkv
            .narrow(D::Minus1, h * d, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;
        let v = qkv
            .narrow(D::Minus1, 2 * h * d, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;

        let q = self.rope_apply(&q)?;
        let k = self.rope_apply(&k)?;

        let scale = (d as f64).sqrt();
        // UNSCALED — the training arm folds `1/scale` into
        // `SoftmaxLastDimFused` itself (see `softmax_apply_training`'s
        // doc) rather than dividing here; an eager `scores / scale`
        // division at this point would retain a full `[batch, heads, seq,
        // seq]` `Op::Affine` tape tensor per layer, present for the
        // training arm alone. The eval arm below still divides explicitly.
        let raw_scores = crate::contiguous_matmul(&q, &k.transpose(D::Minus1, D::Minus2)?)?;
        // The additive mask is always built in F32 (see `extended_attention_mask`);
        // cast to the scores' dtype so a F16/BF16 backbone can add it (a no-op
        // when scores are already F32).
        let extended_mask = extended_mask.to_dtype(raw_scores.dtype())?;

        let attn = if self.training {
            // Combine the (up to two) additive masks into ONE small tensor
            // BEFORE calling the fused kernel — `SoftmaxLastDimFused` is a
            // `CustomOp2` (`scores`, one `mask`), and this sum is at most
            // `[batch, 1, seq, seq]` (never `[batch, heads, seq, seq]`,
            // since neither mask carries a `heads` axis) — see
            // `jammi_kernels::ops::softmax`'s module doc's "why the mask
            // is folded in BEFORE this op runs" section. This is a
            // DIFFERENT (though algebraically equivalent) computation from
            // eval's sequential adds below: floating-point addition is not
            // associative, so this arm's own fused-vs-eager oracle states
            // a tolerance rather than bit-exactness, exactly like every
            // other fused op's training arm in this crate.
            let mask = match (self.is_local, local_band) {
                (true, Some(band)) => {
                    extended_mask.broadcast_add(&band.to_dtype(raw_scores.dtype())?)?
                }
                (true, None) => {
                    return Err(EncoderError::Config(
                        "local-attention layer reached without a sliding-window band".into(),
                    ))
                }
                (false, _) => extended_mask,
            };
            softmax_apply_training(&raw_scores, &mask, scale)?
        } else {
            // Eval's UNCHANGED code path, bit-identical to before this
            // commit: divides by `scale` exactly as before (the ONE `Op::Affine`
            // node eval always retained and still does), then two
            // SEQUENTIAL broadcast-adds, each from its own smaller shape,
            // never combined into one tensor (see
            // `crate::mask::sliding_window_mask`'s doc for why — neither
            // mask is ever materialised at `[batch, heads, seq, seq]`
            // either way, but combining them first would round
            // differently than adding them in this order, which eval must
            // never do — see `tests::eval_mode_attention_softmax_is_bit_identical_regardless_of_fused_eligibility`).
            let scores = (&raw_scores / scale)?;
            let scores = scores.broadcast_add(&extended_mask)?;
            let scores = match (self.is_local, local_band) {
                (true, Some(band)) => scores.broadcast_add(&band.to_dtype(scores.dtype())?)?,
                (true, None) => {
                    return Err(EncoderError::Config(
                        "local-attention layer reached without a sliding-window band".into(),
                    ))
                }
                (false, _) => scores,
            };
            candle_nn::ops::softmax(&scores, D::Minus1)?
        };

        Ok(crate::contiguous_matmul(&attn, &v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, h * d))?)
    }

    /// Dispatches to [`RotaryEmbedding::apply_training`] (fused-when-
    /// possible) in training mode, else [`RotaryEmbedding::apply`]
    /// directly — eval never even calls the training-mode method, so
    /// its OUTPUT VALUES are bit-identical to before the fused kernel
    /// existed regardless of that method's own admission logic (`apply`
    /// itself gained a table-cache lock from table hoisting — see
    /// `RotaryEmbedding`'s struct doc — which is a real, disclosed
    /// change to what eval does, not a numeric one).
    fn rope_apply(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        if self.training {
            self.rope.apply_training(x)
        } else {
            self.rope.apply(x)
        }
    }
}

/// Only ever called from `ModernBertAttention::forward`'s `self.training`
/// arm — dispatches to [`jammi_kernels::ops::SoftmaxLastDimFused`] when its
/// domain holds, else falls back to the eager
/// `(scores / scale).broadcast_add(mask)` plus `candle_nn::ops::softmax`
/// composition (`scores` here is UNSCALED — see `softmax_apply_training`'s
/// own doc for why the division is restored explicitly in this branch),
/// recording which happened either way, mirroring `crate::layer_norm`'s
/// and `RotaryEmbedding`'s identical admission mechanism. Eval never
/// calls this function at all (see `forward`'s `match`), so it has no
/// bearing on eval's bit-identity. A free function (not a method) — it
/// needs no `ModernBertAttention` field, and keeping it free makes it
/// directly unit-testable the same way [`rope_admission_predicate`] is,
/// without constructing a full attention/linear-layer struct just to
/// exercise the dispatch decision.
///
/// ## The three-way split on a fully-masked row (a genuine third numeric
/// path, unlike `RotaryEmbedding`'s doctrine above)
///
/// This function constructs `SoftmaxLastDimFused` with
/// [`jammi_kernels::ops::FullyMaskedPolicy::Zeros`] — the ONE call site in
/// this crate that opts into it. That choice, combined with the
/// fused-vs-eager-fallback split every admission-gated call site here has,
/// produces THREE distinct numeric outcomes on a fully-masked row (a
/// query that is itself padding, in a local-attention layer — see
/// `crate::mask::sliding_window_mask`'s corrected doc), not two:
/// - Eval (`training == false`): ALWAYS `candle_nn::ops::softmax`'s own
///   output there — `NaN` for a synthetic `-inf` mask, or a finite
///   dtype-dependent (annihilated-uniform in BF16, near-normal in F32)
///   result for the real `MASKED_LOGIT` convention.
/// - Training, admitted into the fused kernel: ALL ZEROS
///   (`FullyMaskedPolicy::Zeros`'s production-attention-kernel behavior —
///   see `jammi_kernels::ops::softmax`'s module doc).
/// - Training, falling back to eager (the domain check failed — wrong
///   dtype, non-contiguous, broadcast-class violation, etc.): the SAME
///   output eval would have produced, since this branch calls the
///   identical `candle_nn::ops::softmax` composition.
///
/// So training's OWN two branches already disagree with each other on
/// this one input class — genuinely a third distinct numeric path, not
/// "fused when possible, otherwise identical to eval" the way
/// `RotaryEmbedding`'s doctrine holds for RoPE. This is inert in both
/// directions for the ONLY row class it can ever apply to (a pad-query
/// row in a local-attention layer): FORWARD, every pooling reducer this
/// crate ships discards that row's hidden state regardless of its value —
/// `mean_pool`/`weighted_mean_pool` multiply by the real attention mask
/// (`pooling.rs`'s `hidden.broadcast_mul(&mask...)`), `max_pool`
/// substitutes a sentinel there via `where_cond`, and `cls_pool` reads
/// only position `0` (never itself a pad token in a real batch) and so
/// never reads this row at all. BACKWARD, because pooling ZEROES that
/// row's contribution to the loss, the gradient flowing back into it
/// (`dy`) is exactly `0.0` — under `Zeros`, `dscores = (0 - 0) * 0 == 0`;
/// under the real finite-`MASKED_LOGIT` eager fallback, `y` is finite
/// (never `NaN`), so `dscores = (0 - sum(0*y)) * y == 0 * y == 0`
/// identically. Both training branches therefore yield an EXACTLY zero
/// gradient here — the training dynamics through a pad-query row are
/// identical whichever branch admission chose, not merely close (see
/// `jammi_kernels::ops::softmax`'s module doc and
/// `ops::softmax::tests::fully_masked_row_backward_is_zero_under_both_policies_given_pooling_style_zero_dy`
/// for the verified claim this restates).
///
/// ## `scores` is UNSCALED here — `scores_divisor` folds `sqrt(head_dim)` in
///
/// `scores` is `forward`'s `raw_scores` — the RAW `q @ k^T` matmul output,
/// never divided by `scores_divisor` (`sqrt(head_dim)`) before reaching
/// this function, unlike eval's own `scores` (see `forward`'s doc comment
/// at its call site). This removes the `Op::Affine` node (`scores /
/// scores_divisor`) the training arm used to retain at `[batch, heads,
/// seq, seq]` — the single largest per-layer tape tensor this fused
/// softmax op's own memory win already targets (see
/// `jammi_kernels::ops::softmax`'s module doc). The FUSED branch folds
/// `1 / scores_divisor` into `SoftmaxLastDimFused::scale` (`with_scale`)
/// — a MULTIPLIER, the opposite convention from this function's own
/// `scores_divisor` parameter, named explicitly to keep the two apart —
/// so the op itself computes `softmax(scale_mul * scores + mask)` in one
/// pass — see that op's module doc's "scale semantics" section for the
/// exact per-dtype rounding this reproduces. The EAGER FALLBACK branch
/// restores the `scores / scores_divisor` division EXPLICITLY, right
/// here, so a domain miss (wrong dtype, non-contiguous, broadcast-class
/// violation, …) is numerically IDENTICAL to what this function computed
/// before `scores_divisor` existed as a named parameter — the fallback
/// is never a fourth numeric path, only the pre-existing training-eager
/// composition. This is a MEASURED claim, not an assertion: EVERY other
/// test exercising this function is fused-eligible (none reaches this
/// branch), so `tests::eager_fallback_softmax_matches_inline_reference_fwd_and_bwd`
/// is the ONLY oracle that forces this arm and checks it fwd+bwd
/// bit-for-bit against the inline composition this doc describes —
/// mutation-verified (see that test's own doc).
fn softmax_apply_training(
    scores: &Tensor,
    mask: &Tensor,
    scores_divisor: f64,
) -> Result<Tensor, EncoderError> {
    let (holds, predicate) = softmax_admission_predicate(scores, mask, scores_divisor);
    let outcome = admit(
        admission_mode(),
        "softmax_last_dim_fused",
        predicate,
        holds,
        *SOFTMAX_DISPATCH_COUNTERS,
    )?;
    match outcome {
        DispatchOutcome::Fused => Ok(apply2(
            scores,
            mask,
            SoftmaxLastDimFused::new(jammi_kernels::ops::FullyMaskedPolicy::Zeros)
                .with_scale((1.0 / scores_divisor) as f32)?,
        )?),
        DispatchOutcome::Eager => Ok(candle_nn::ops::softmax(
            &(scores / scores_divisor)?.broadcast_add(mask)?,
            D::Minus1,
        )?),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused GeGLU (C5)
// ─────────────────────────────────────────────────────────────────────────────

/// Fused/eager dispatch counters for the ModernBERT MLP's GeGLU
/// activation, read from the registry — mirroring `ROPE_DISPATCH_COUNTERS`
/// / `SOFTMAX_DISPATCH_COUNTERS` / `crate::layer_norm::LN_DISPATCH_COUNTERS`
/// — see `ModernBertMlp::forward`'s doc for the training-only gate this
/// counts. `pub(crate)` (not `pub`) — read via
/// [`crate::geglu_dispatch_snapshot`], the same shape the other three
/// snapshot functions use.
pub(crate) static GEGLU_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("geglu_fused"));

/// The fused GeGLU kernel's domain, checked at the call site (family D /
/// K2): `wi_out`'s device is one [`jammi_kernels::admission::device_is_supported`]
/// accepts, its dtype is one the kernel implements (F32 or BF16),
/// `wi_out` is contiguous ([`jammi_kernels::ops::GegluFused`] refuses a
/// strided view — see its module doc), and its last dimension is nonzero
/// and even (the op splits it into two equal `gate`/`up` halves; an odd
/// width is a structural domain violation the op itself also refuses, but
/// checking it here means it becomes a counted eager fallback instead of
/// a `candle_core::Error` surfacing from inside the op).
fn geglu_admission_predicate(wi_out: &Tensor) -> (bool, &'static str) {
    if !device_is_supported(wi_out.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if !matches!(wi_out.dtype(), DType::F32 | DType::BF16) {
        return (false, "dtype_f32_or_bf16");
    }
    if !wi_out.is_contiguous() {
        return (false, "wi_out_contiguous");
    }
    let last = *wi_out.dims().last().unwrap_or(&0);
    if last == 0 || !last.is_multiple_of(2) {
        return (false, "last_dim_nonzero_and_even");
    }
    (true, "domain_ok")
}

/// Only ever called from `ModernBertMlp::forward`'s `self.training` arm —
/// dispatches to [`jammi_kernels::ops::GegluFused`] (erf variant — the
/// ONLY variant ModernBERT's MLP call site uses, see that op's own
/// `GeluVariant` doc) when its domain holds, else falls back to the
/// eager `narrow`+`narrow`+`gelu_erf`+`mul` composition (recording which
/// happened either way, mirroring `softmax_apply_training`'s / RoPE's
/// identical admission mechanism). Eval never calls this function at all
/// (see `forward`'s `match`), so it has no bearing on eval's bit-identity.
fn geglu_apply_training(wi_out: &Tensor) -> Result<Tensor, EncoderError> {
    let (holds, predicate) = geglu_admission_predicate(wi_out);
    let outcome = admit(
        admission_mode(),
        "geglu_fused",
        predicate,
        holds,
        *GEGLU_DISPATCH_COUNTERS,
    )?;
    match outcome {
        DispatchOutcome::Fused => Ok(apply1(
            wi_out,
            jammi_kernels::ops::GegluFused::new(jammi_kernels::ops::GeluVariant::Erf),
        )?),
        DispatchOutcome::Eager => {
            let intermediate = wi_out.dim(D::Minus1)? / 2;
            let gate = wi_out.narrow(D::Minus1, 0, intermediate)?;
            let up = wi_out.narrow(D::Minus1, intermediate, intermediate)?;
            Ok((gate.gelu_erf()? * up)?)
        }
    }
}

struct ModernBertMlp {
    /// Packed gate+up projection. LoRA target name: `"Wi"`.
    wi: MaybeLoraLinear,
    /// Down projection. LoRA target name: `"mlp.Wo"` (kept namespaced so
    /// `ends_with("Wo")` targeting can distinguish it from the attention
    /// output projection when callers want both).
    wo: MaybeLoraLinear,
    mlp_norm: LayerNorm,
    /// Whether the fused GeGLU kernel may be attempted (still gated by
    /// its own domain check — see [`geglu_apply_training`]). `false`
    /// (eval/serving) always runs the eager `narrow`+`narrow`+`gelu_erf`+
    /// `mul` composition below, unconditionally — the SAME code this file
    /// had before the fused kernel existed, so eval's output values are
    /// bit-identical before/after this change (see
    /// `tests::eval_mode_mlp_is_bit_identical_regardless_of_fused_eligibility`).
    /// `true` (training) is the ONLY state that ever reaches
    /// [`geglu_apply_training`]. Propagated by [`ModernBert::set_training`].
    training: bool,
}

impl ModernBertMlp {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let normed = self.mlp_norm.forward(x)?;

        let up_gate = self.wi.forward(&normed)?;

        let act = if self.training {
            geglu_apply_training(&up_gate)?
        } else {
            // Eval's UNCHANGED code path, bit-identical to before this
            // commit.
            let intermediate = up_gate.dim(D::Minus1)? / 2;
            let gate = up_gate.narrow(D::Minus1, 0, intermediate)?;
            let up = up_gate.narrow(D::Minus1, intermediate, intermediate)?;
            (gate.gelu_erf()? * up)?
        };
        let out = self.wo.forward(&act)?;

        Ok((out + x)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Flash-attention cascade admission seam (P6 Stage B B2)
//
// This section builds the SEAM `attention_block_flash` (B1, a sibling
// branch) will plug into — the op does not exist yet, so every decision
// below always declines and the training arm's dispatch (and every
// output byte) is UNCHANGED. See `decide_flash_admission`'s doc for the
// exact, temporary override that pins this, and
// `tests::flash_cascade_never_changes_the_block_arm_dispatch_or_output`
// for the oracle proving it.
// ─────────────────────────────────────────────────────────────────────────────

/// The head dim the flash cascade admits — mirrors [`ATTENTION_BLOCK_HEAD_DIM`]
/// (both `64`) and, once B1 lands, `jammi_kernels::flash::HEAD_DIM`.
/// Duplicated here as a plain constant rather than depending on
/// `jammi-kernels`'s `flash-attn` feature: this crate does not forward
/// that feature (no consumer needs the real `jammi_kernels::flash` types
/// until B1's `flash_attention_varlen` exists — see [`CompactedBatch`]'s
/// doc for why `lengths`, not a constructed `CuSeqlens`, is what this
/// commit threads).
const FLASH_HEAD_DIM: usize = 64;

/// The exact CUDA compute capability the flash cascade admits — sm_80
/// ONLY (contract v4 §3.4/F6: `>=` would admit sm_86/89/90 and crash at
/// launch, since the vendored kernel is compiled for sm_80 alone). A stub
/// this crate OWNS until B1 lands its own build-time `FLASH_COMPUTE_CAP`
/// constant (`crates/jammi-kernels/build.rs`'s planned
/// `JAMMI_FLASH_GENCODE_SM`, contract §3.4); this function is the seam
/// B1's real `check_arch`-equivalent plugs into. Uses the SAME
/// [`probe_cuda_compute_capability`] every other admission predicate in
/// this crate/`jammi-kernels` reads (`None` — non-CUDA, or `cuda` feature
/// off, or a query failure — degrades to `false`, never a panic or a
/// silently-wrong "yes").
fn flash_arch_ok(device: &Device) -> bool {
    probe_cuda_compute_capability(device) == Some(ComputeCapability::new(8, 0))
}

/// How many times this process has paid the flash cascade's device-side
/// mask reduction + host sync (contract v4 §3.7, path F, as corrected by
/// the lead: row LENGTHS alone cannot decide "prefix" — a mask
/// `[1,0,1,0]` sums to `2`, the same length a genuine prefix `[1,1,0,0]`
/// reports — so lengths AND the whole-batch prefix predicate are computed
/// on the device in ONE reduction and read back in ONE sync, not `b`
/// separate transfers). See [`compute_lengths_and_prefix`]. A durable run
/// record is expected to read this exactly `steps` (or
/// `forwards_per_step * steps` — batched vs unbatched vs GradCache-chunked,
/// contract v4's item-2 correction) on an FA2-ELIGIBLE leg (the sync only
/// runs once the cheap, sync-free gates below already passed) and `0` on
/// every block leg, including every CPU / non-`flash-attn`-feature build
/// this crate's own test suite runs under today.
static FLASH_D2H_SYNCS: AtomicU64 = AtomicU64::new(0);

/// Read API for [`FLASH_D2H_SYNCS`] — `pub(crate)` today (no external
/// consumer yet); a durable job record reads it the same way it reads
/// `crate::attention_block_dispatch_snapshot()`. `#[allow(dead_code)]`:
/// exercised directly by this module's own tests (`#[cfg(test)]`, a
/// SEPARATE compilation from the plain lib target `clippy --all-targets`
/// also checks) but not yet called from non-test production code — no
/// durable-artifact caller has been wired up to read it yet.
#[allow(dead_code)]
pub(crate) fn flash_d2h_syncs() -> u64 {
    FLASH_D2H_SYNCS.load(Ordering::Relaxed)
}

/// The once-per-forward flash-cascade decision (contract v4 §3.2), decided
/// ONCE in [`ModernBert::forward_hidden`] (mirroring [`FusedAttentionMasks`])
/// and threaded per layer. Owns the compacted batch's row `lengths` and the
/// `[total]` unpad gather indices — see [`unpad_gather_indices`] — but
/// deliberately NOT a constructed `jammi_kernels::flash::CuSeqlens`: that
/// type is feature-gated behind `jammi-kernels`'s `flash-attn` (not
/// forwarded by this crate's `Cargo.toml` yet), and `CuSeqlens::from_lengths`
/// is cheap enough to construct on demand, once, at the real flash call
/// site B1 adds — holding a `CuSeqlens` across a whole forward buys
/// nothing today and would force a premature feature dependency.
struct CompactedBatch {
    /// One length per batch element, `lengths[b] <= seq`. Consumed by
    /// [`ModernBertAttention::forward_flash_dense_attention`]
    /// (`CuSeqlens::from_lengths`) under the `flash-attn` feature; on a
    /// plain build the field is only read by this module's own tests, so
    /// `#[allow(dead_code)]` stays even though it is no longer
    /// unconditionally dead.
    #[allow(dead_code)]
    lengths: Vec<usize>,
    /// `[total]` gather indices into the flattened `[batch * seq]` row
    /// axis — every REAL (non-pad) row, batch-then-seq order. NOT
    /// consumed by the dense arm (dense skips compaction entirely, see
    /// `decide_flash_admission`'s doc) — this is the padded regime's own
    /// future consumer (B3-padded, out of this commit's scope).
    #[allow(dead_code)]
    gather_indices: Tensor,
    /// Same status as `gather_indices`: the padded arm's future consumer.
    #[allow(dead_code)]
    total: usize,
}

/// The full once-per-forward flash-cascade decision, decided ONCE in
/// [`ModernBert::forward_hidden`] (mirroring [`FusedAttentionMasks`]) and
/// threaded per layer — every LAYER's own [`admit_cascade`] call reports
/// against it (contract v4 §3.2: "the counters are per-dispatch, not
/// per-forward" — this type is what makes that per-layer call cheap: no
/// layer re-derives the outcome/reason).
///
/// Two variants, not a `CompactedBatch`/`outcome`/`reason` struct with the
/// first field optional: the prior shape let `outcome == Holds` and
/// `admission == None` be constructed simultaneously — an invalid state
/// [`ModernBertAttention::forward_training_attention`]'s own dispatch code
/// had to guard with a RUNTIME `ok_or_else` (a string-message fallback for
/// a state the type itself should have refused to represent). This enum
/// makes that state a COMPILE ERROR instead: [`Self::Fused`] always
/// carries its [`CompactedBatch`], [`Self::Declined`] never does — no
/// runtime check stands between "the cascade decided Fused" and "a
/// `CompactedBatch` exists".
///
/// [`Self::outcome`]/[`Self::reason`] recover the [`PredicateOutcome`] /
/// reason string every `admit_cascade` call site still needs (`Fused`
/// always reports `(Holds, "domain_ok_dense")` — [`build_flash_forward_decision`]'s
/// only path to that variant), so callers built around the old struct's
/// two bare fields keep exactly the same call shape.
enum FlashDecision {
    /// The batch is DENSE and flash-eligible — `attention_block_flash`
    /// dispatches `Fused` and
    /// [`ModernBertAttention::forward_flash_dense_attention`] runs on this
    /// `CompactedBatch`.
    Fused(CompactedBatch),
    /// The cascade declines — `outcome`/`reason` are whatever
    /// [`flash_admission_predicate`] (or the dense/padded split in
    /// [`build_flash_forward_decision`]) determined. NEVER carries a
    /// `CompactedBatch`, even in the "was `Holds`, downgraded for being
    /// padded" case (`reason == "flash_padded_not_yet_wired"`): that batch
    /// is out of THIS decision's scope once declined — a future
    /// B3-padded consumer building its own compacted batch does so from
    /// `mask`/`lengths` directly, not by fishing one out of a declined
    /// `FlashDecision`.
    Declined {
        outcome: PredicateOutcome,
        reason: &'static str,
    },
}

impl FlashDecision {
    /// The [`PredicateOutcome`] every `admit_cascade` call site reports —
    /// `Holds` for [`Self::Fused`] (the only outcome that variant can mean),
    /// whatever [`Self::Declined`] itself carries otherwise.
    fn outcome(&self) -> PredicateOutcome {
        match self {
            FlashDecision::Fused(_) => PredicateOutcome::Holds,
            FlashDecision::Declined { outcome, .. } => *outcome,
        }
    }

    /// The reason string every `admit_cascade` call site reports —
    /// `"domain_ok_dense"` for [`Self::Fused`] ([`build_flash_forward_decision`]'s
    /// only path to that variant), whatever [`Self::Declined`] itself
    /// carries otherwise.
    fn reason(&self) -> &'static str {
        match self {
            FlashDecision::Fused(_) => "domain_ok_dense",
            FlashDecision::Declined { reason, .. } => reason,
        }
    }
}

/// Row indices of every REAL (non-pad) row of a `[batch, seq, ..]` tensor,
/// flattened to `[total]`, in batch-then-seq order — `padded[b, s]`'s flat
/// row index `b * seq + s` for every `s < lengths[b]` (RIGHT padding: every
/// real row precedes every pad row within a batch element — contract v4 §3
/// D1's "row%period + HF absolute-padded arange" premise, `trainer.rs`'s
/// `BatchLongest`). `total = gather_indices.dim(0)? = lengths.iter().sum()`.
fn unpad_gather_indices(
    lengths: &[usize],
    seq: usize,
    device: &Device,
) -> Result<Tensor, EncoderError> {
    let total: usize = lengths.iter().map(|&l| l.min(seq)).sum();
    let mut idx: Vec<u32> = Vec::with_capacity(total);
    for (b, &len) in lengths.iter().enumerate() {
        for s in 0..len.min(seq) {
            idx.push(u32::try_from(b * seq + s).map_err(|_| {
                EncoderError::Config("unpad_gather_indices: row index overflows u32".into())
            })?);
        }
    }
    Ok(Tensor::from_vec(idx, (total,), device)?)
}

/// Unpad: `[batch, seq, hidden] -> [total, hidden]`, gathering every real
/// row via `gather_indices` (see [`unpad_gather_indices`]). A pure
/// function — no admission, no counters — the encoder-boundary half of
/// contract v4 §3.5's "unpad/repad at the ENCODER boundary" (one gather
/// before layer 0, one scatter after the last layer). `#[allow(dead_code)]`:
/// exercised by this module's own tests; not yet wired into the real
/// forward path (B1's flash call site is the eventual production caller).
#[allow(dead_code)]
fn unpad_rows(x: &Tensor, gather_indices: &Tensor) -> Result<Tensor, EncoderError> {
    let (b, s, h) = x.dims3()?;
    let flat = x.reshape((b * s, h))?;
    Ok(flat.index_select(gather_indices, 0)?)
}

/// Repad: `[total, hidden] -> [batch, seq, hidden]`, the inverse of
/// [`unpad_rows`]. The destination is `Tensor::zeros` (contract v4 §3.5's
/// `alloc_zeros`, this crate's host-composable equivalent of
/// `flash/mod.rs`'s CUDA-scratch `alloc_zeros` under `deterministic`) —
/// NEVER an uninitialised buffer multiplied by a 0/1 indicator, which is
/// exactly the `0.0 * NaN = NaN` failure `crate::pooling.rs:62`'s own doc
/// already names: `index_add` on a genuinely zeroed destination can never
/// read a pad row's stale bytes, because it never reads the destination at
/// the indices it does not touch. `gather_indices` entries are unique (one
/// real row maps to exactly one compacted row), so `index_add` — an
/// accumulate — is exactly a scatter/copy here, never a collision.
/// `#[allow(dead_code)]`: same status as [`unpad_rows`].
#[allow(dead_code)]
fn repad_rows(
    compacted: &Tensor,
    gather_indices: &Tensor,
    batch: usize,
    seq: usize,
) -> Result<Tensor, EncoderError> {
    let hidden = compacted.dim(1)?;
    let dest = Tensor::zeros((batch * seq, hidden), compacted.dtype(), compacted.device())?;
    let repadded = dest.index_add(gather_indices, compacted, 0)?;
    Ok(repadded.reshape((batch, seq, hidden))?)
}

/// Contract v4 §3.7 (path F), corrected mid-round by the lead: row
/// LENGTHS alone cannot decide "prefix" (`[1,0,1,0]` sums to `2`, same as
/// the genuine prefix `[1,1,0,0]`) — this computes BOTH `lengths[b] =
/// mask.sum(1)` AND the whole-batch prefix predicate `is_prefix = ALL_b,s
/// (mask[b,s] == (s < lengths[b]))` on the DEVICE in one pass, and reads
/// them back together in ONE sync (a single `Tensor::cat`ed `to_vec1`
/// call — `lengths` and the prefix flag share the SAME transfer, not two).
/// `mask` is `[batch, seq]`, `0.0`/`1.0`-valued padding mask (the same one
/// [`extended_attention_mask`] consumes).
fn compute_lengths_and_prefix(mask: &Tensor) -> Result<(Vec<usize>, bool), EncoderError> {
    let (batch, seq) = mask.dims2()?;
    let mask_f32 = mask.to_dtype(DType::F32)?;
    let lengths_col = mask_f32.sum_keepdim(1)?; // [batch, 1]

    // prefix_mask(lengths)[b, s] = 1.0 iff s < lengths[b] — an iota
    // compared against the broadcast length, still ON THE DEVICE.
    let iota = Tensor::arange(0u32, seq as u32, mask.device())?
        .to_dtype(DType::F32)?
        .reshape((1, seq))?;
    let reconstructed = iota.broadcast_lt(&lengths_col)?.to_dtype(DType::F32)?;
    let row_matches = mask_f32.eq(&reconstructed)?.to_dtype(DType::F32)?;
    // 1.0 iff EVERY (b, s) matches its own prefix reconstruction.
    let is_prefix_scalar = row_matches.min_all()?.reshape(1)?;

    // ONE sync: `lengths` (b floats) and the prefix flag concatenated into
    // a single `[batch + 1]` tensor and read back with ONE `to_vec1` call
    // — the transfer `FLASH_D2H_SYNCS` counts, not `b + 1` separate ones.
    let combined = Tensor::cat(&[&lengths_col.reshape(batch)?, &is_prefix_scalar], 0)?;
    let combined_host: Vec<f32> = combined.to_vec1()?;
    FLASH_D2H_SYNCS.fetch_add(1, Ordering::Relaxed);

    let lengths: Vec<usize> = combined_host[..batch]
        .iter()
        .map(|&l| l.round() as usize)
        .collect();
    let is_prefix = combined_host[batch] != 0.0;
    Ok((lengths, is_prefix))
}

/// The flash cascade's own admission predicate (contract v4 §3.2's
/// consulted terms): device is CUDA and arch EXACTLY `(8, 0)`
/// ([`flash_arch_ok`]), backbone dtype `BF16`, `head_dim ==
/// `[`FLASH_HEAD_DIM`]``, `flash-attn` compiled (`cfg!` TERM — L10, a
/// [`PredicateOutcome::CapabilityMiss`], never `#[cfg]` on the call site),
/// and the batch's mask is a prefix mask with every row length `>= 1`
/// ([`compute_lengths_and_prefix`] — a [`PredicateOutcome::DomainMiss`]
/// when it is not: a mixed/interior-zero batch legitimately does not fit
/// this arm's domain, contract v4 §3.3's L3/L4/L5).
///
/// Ordered so the EXPENSIVE, D2H-paying check (mask prefix) runs LAST,
/// after every cheap, sync-free gate — device/arch/dtype/head_dim/feature —
/// already passed. This is why [`flash_d2h_syncs`] reads `0` on every
/// block leg this crate's test suite exercises today (no CUDA, or the
/// `flash-attn` feature off): the cheap gates fail first and the mask
/// reduction is never reached.
/// `(outcome, reason, eligible)` — `eligible` is `Some((lengths, seq))`
/// ONLY on `PredicateOutcome::Holds` (the whole-batch row lengths and the
/// sequence length [`build_flash_forward_decision`] needs to build
/// [`CompactedBatch`]). Named to keep [`flash_admission_predicate`]'s and
/// [`decide_flash_admission`]'s signatures under clippy's `type_complexity`
/// ceiling.
type FlashPredicateResult = Result<FlashPredicateTriple, EncoderError>;

/// The `(outcome, reason, eligible)` triple itself, named separately from
/// [`FlashPredicateResult`] so [`flash_capability_gates`] — which never
/// fails (`Option`, not `Result`) — can reuse the same shape.
type FlashPredicateTriple = (PredicateOutcome, &'static str, Option<(Vec<usize>, usize)>);

fn flash_admission_predicate(
    device: &Device,
    dtype: DType,
    head_dim: usize,
    mask: &Tensor,
    trusted_lengths: Option<&[usize]>,
) -> FlashPredicateResult {
    // Reads `jammi_kernels::admission::FLASH_COMPILED` (contract v5 item 2)
    // rather than a LOCAL `cfg!(feature = "flash-attn")`: this crate's own
    // `flash-attn` feature is not yet forwarded any further up the stack
    // (`jammi-ai`/`jammi-bench` do not request it), and — because
    // `jammi_kernels::flash` is itself `#[cfg(feature = "flash-attn")]`-gated
    // — a call site can never "stay compiled" behind a bare `cfg!()` bool if
    // reaching the `true` branch would need to NAME a type from that module.
    // `FLASH_COMPILED` is exactly the escape hatch: a plain, unconditionally-
    // compiled `bool` reflecting how `jammi-kernels` itself was actually
    // built, readable regardless of whether THIS crate's own feature flag is
    // forwarded — see that constant's own doc.
    if let Some(miss) = flash_capability_gates(
        jammi_kernels::admission::FLASH_COMPILED,
        device,
        dtype,
        head_dim,
    ) {
        return Ok(miss);
    }
    resolve_lengths_and_prefix(mask, trusted_lengths)
}

/// The cheap, sync-free capability/domain gates — split out (and
/// `feature_compiled` taken as a PARAMETER, not read from
/// `jammi_kernels::admission::FLASH_COMPILED` directly) so every branch is
/// directly unit-testable with a literal input, independent of the REAL
/// build's `FLASH_COMPILED` value (this crate's own test suite always
/// builds with the `flash-attn` feature off, so the real constant is always
/// `false` here — a test that only ever calls
/// [`flash_admission_predicate`] itself can never observe the `device`/
/// `arch`/`dtype`/`head_dim` gates below it, since the feature gate always
/// short-circuits FIRST). `Some((outcome, reason, None))` on a miss;
/// `None` when every gate passes (the caller proceeds to
/// [`resolve_lengths_and_prefix`]).
fn flash_capability_gates(
    feature_compiled: bool,
    device: &Device,
    dtype: DType,
    head_dim: usize,
) -> Option<FlashPredicateTriple> {
    if !feature_compiled {
        return Some((
            PredicateOutcome::CapabilityMiss,
            "flash_attn_feature_compiled",
            None,
        ));
    }
    if !device.is_cuda() {
        return Some((PredicateOutcome::CapabilityMiss, "device_is_cuda", None));
    }
    if !flash_arch_ok(device) {
        return Some((PredicateOutcome::CapabilityMiss, "arch_is_sm80_exact", None));
    }
    if dtype != DType::BF16 {
        return Some((PredicateOutcome::DomainMiss, "dtype_is_bf16", None));
    }
    if head_dim != FLASH_HEAD_DIM {
        return Some((
            PredicateOutcome::DomainMiss,
            "head_dim_is_flash_head_dim",
            None,
        ));
    }
    None
}

/// The tail of [`flash_admission_predicate`] — everything AFTER the cheap,
/// device/build capability gates (feature compiled, CUDA, arch) — split out
/// so path P's `trusted_lengths` branch and path F's device-reduction
/// branch are BOTH directly unit-testable on `Device::Cpu` without needing
/// to fake past `flash_admission_predicate`'s own CUDA-only gates (this
/// crate's test suite has no CUDA device at all).
fn resolve_lengths_and_prefix(
    mask: &Tensor,
    trusted_lengths: Option<&[usize]>,
) -> FlashPredicateResult {
    let (lengths, is_prefix) = match trusted_lengths {
        Some(trusted) => {
            // Path P (contract v4 §3.7, v5 item 3): the caller ALREADY
            // knows the row lengths (its own tokenizer's output, e.g.
            // `encode_texts`) and asserts `mask` was built by a
            // construction that is a right-padded prefix by CONSTRUCTION
            // (the same `BatchLongest` premise `trainer.rs` relies on) —
            // trusted, not re-derived from `mask` on the device, which is
            // the whole point: this path pays ZERO `flash_d2h_syncs`. Only
            // a cheap, host-only shape check remains (family D: even a
            // trusted input gets ITS OWN domain check, not blind faith).
            let (batch, seq) = mask.dims2()?;
            if trusted.len() != batch {
                return Ok((
                    PredicateOutcome::DomainMiss,
                    "trusted_lengths_len_matches_batch",
                    None,
                ));
            }
            if trusted.iter().any(|&l| l > seq) {
                return Ok((
                    PredicateOutcome::DomainMiss,
                    "trusted_lengths_within_seq",
                    None,
                ));
            }
            (trusted.to_vec(), true)
        }
        None => compute_lengths_and_prefix(mask)?,
    };
    if !is_prefix {
        // Mixed batch (some rows non-prefix, contract v4 L3/L5) -> DomainMiss
        // for the WHOLE forward, at this encoder-level predicate, BEFORE any
        // `CuSeqlens` construction is attempted.
        return Ok((
            PredicateOutcome::DomainMiss,
            "mask_is_prefix_every_row",
            None,
        ));
    }
    if lengths.contains(&0) {
        // Redundant with `is_prefix` today (a length-0 row IS a valid
        // all-zero prefix and would pass the check above), kept as an
        // explicit, separately-named domain miss (L4) so a FUTURE relaxation
        // of the prefix check alone cannot silently admit a zero-length row.
        return Ok((PredicateOutcome::DomainMiss, "every_row_length_ge_1", None));
    }
    let seq = mask.dim(1)?;
    Ok((PredicateOutcome::Holds, "domain_ok", Some((lengths, seq))))
}

/// Decides the flash cascade ONCE per forward (contract v4 §3.2), building
/// [`CompactedBatch`] when eligible.
///
/// **Dense-only scope (P6 Stage B B3-dense):** when
/// [`flash_admission_predicate`] returns [`PredicateOutcome::Holds`] AND
/// the batch is DENSE (every row's length `== seq`, `cu_seqlens` would be
/// uniform), the real `Holds` is kept — `attention_block_flash` dispatches
/// `Fused` and [`ModernBertAttention::forward_flash_dense_attention`]
/// runs. A batch with REAL padding (some row length `< seq`) still
/// downgrades to `PredicateOutcome::CapabilityMiss("flash_padded_not_yet_wired")`
/// — the unpad/repad transport this arm would need is explicitly the
/// PADDED regime, out of this commit's scope (a separate B3-padded unit);
/// `tests::flash_cascade_never_changes_the_block_arm_dispatch_or_output`
/// still holds for every PADDED fixture that test exercises.
fn decide_flash_admission(
    device: &Device,
    dtype: DType,
    head_dim: usize,
    mask: &Tensor,
    trusted_lengths: Option<&[usize]>,
) -> Result<FlashDecision, EncoderError> {
    let (outcome, reason, eligible) =
        flash_admission_predicate(device, dtype, head_dim, mask, trusted_lengths)?;
    build_flash_forward_decision(outcome, reason, eligible, device)
}

/// [`decide_flash_admission`]'s device-free second half, split out so the
/// dense/padded split — and [`CompactedBatch`]'s construction from
/// `(lengths, seq)` — are directly unit-testable with a literal
/// `PredicateOutcome::Holds` input, without needing an actual CUDA device
/// to reach [`flash_admission_predicate`]'s `Holds` branch (this crate's
/// own test suite has no CUDA device to run on at all).
fn build_flash_forward_decision(
    outcome: PredicateOutcome,
    reason: &'static str,
    eligible: Option<(Vec<usize>, usize)>,
    device: &Device,
) -> Result<FlashDecision, EncoderError> {
    match (outcome, eligible) {
        (PredicateOutcome::Holds, Some((lengths, seq))) => {
            let gather_indices = unpad_gather_indices(&lengths, seq, device)?;
            let total = gather_indices.dim(0)?;
            let is_dense = lengths.iter().all(|&l| l == seq);
            let compacted = CompactedBatch {
                lengths,
                gather_indices,
                total,
            };
            if is_dense {
                Ok(FlashDecision::Fused(compacted))
            } else {
                // Real padding: the unpad/repad transport is out of this
                // commit's scope (B3-padded) -- decline. `compacted` is
                // built (proving the construction succeeds on a padded
                // batch too — `unpad_gather_indices` still runs its own
                // domain checks above) and then DROPPED: `FlashDecision::
                // Declined` never carries a `CompactedBatch` (see that
                // type's own doc) — a future B3-padded consumer builds its
                // own from `mask`/`lengths` directly.
                let _ = compacted;
                Ok(FlashDecision::Declined {
                    outcome: PredicateOutcome::CapabilityMiss,
                    reason: "flash_padded_not_yet_wired",
                })
            }
        }
        _ => Ok(FlashDecision::Declined { outcome, reason }),
    }
}

struct ModernBertLayer {
    attention: ModernBertAttention,
    mlp: ModernBertMlp,
}

impl ModernBertLayer {
    /// Passes `local_band` through to attention; whether it is consulted is the
    /// attention's own per-layer property.
    fn forward(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        local_band: Option<&Tensor>,
        fused_masks: Option<&FusedAttentionMasks>,
        flash: Option<&FlashDecision>,
    ) -> Result<Tensor, EncoderError> {
        let after_attn =
            self.attention
                .forward(hidden, extended_mask, local_band, fused_masks, flash)?;
        self.mlp.forward(&after_attn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoder
// ─────────────────────────────────────────────────────────────────────────────

/// ModernBERT encoder with selectable LoRA adapters on attention and FFN
/// linears.
///
/// Construct via [`ModernBert::builder`]; see [`ModernBertBuilder`] for the
/// configurable surface.
pub struct ModernBert {
    word_embeddings: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    pooling: Pooling,
    hidden_size: usize,
    max_position_embeddings: usize,
    /// Half-width of the sliding window, `Some` only when the model actually
    /// has a local layer. `None` means every layer is global and no band is
    /// built.
    local_half_window: Option<usize>,
    /// Sliding-window bands, keyed by sequence length.
    ///
    /// The band is a pure function of `(seq, half_window, device)` and constant
    /// for the life of the model, but the sequence length varies per batch
    /// (padding is batch-longest), so it is memoised per length rather than
    /// built once. Without this, every forward allocated and uploaded a
    /// `seq * seq` host buffer — 268 MB per forward at this family's
    /// `max_position_embeddings` of 8192 — which is the same host-generated
    /// per-forward mask cost recorded as esc-032 for LoRA dropout.
    ///
    /// A `Mutex` rather than a `RefCell`: the model is held across threads.
    band_cache: Mutex<HashMap<usize, Tensor>>,
    /// Mirrors the flag [`Self::set_training`] propagates to every layer,
    /// so [`Self::forward_hidden`] knows whether to build the per-forward
    /// [`FusedAttentionMasks`] at all (eval never reads them, so eval's
    /// call sequence stays exactly what it was).
    training: bool,
}

impl ModernBert {
    /// Start configuring a `ModernBert` instance.
    pub fn builder() -> ModernBertBuilder<'static> {
        ModernBertBuilder {
            pooling: Pooling::default(),
            lora: LoraBuildConfig::frozen(),
            backbone_dtype: DType::F32,
            adapter_file: None,
        }
    }

    /// Output dimensionality of the encoder.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Maximum sequence length the model supports (`max_position_embeddings`).
    pub fn max_seq_length(&self) -> usize {
        self.max_position_embeddings
    }

    /// Run the encoder and pool + L2-normalise the output, returning
    /// `[batch, hidden]`.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.forward_hidden(input_ids, mask)?;
        pool_and_normalize(&hidden, mask, self.pooling)
    }

    /// [`Self::forward`] with row lengths ALREADY known host-side (P6 Stage
    /// B, contract v4 §3.7 "path P", v5 item 3) — see
    /// [`Self::forward_hidden_with_lengths`]'s doc for the trust contract
    /// `lengths` carries and why this skips a device sync. The additive
    /// entry point `AnyEncoder::forward_with_lengths` (`jammi-ai`, NOT this
    /// crate) is expected to call this; this crate does not depend on
    /// `jammi-ai` and cannot wire that call site itself.
    pub fn forward_with_lengths(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
        lengths: Option<&[usize]>,
    ) -> Result<Tensor, EncoderError> {
        let hidden = self.forward_hidden_with_lengths(input_ids, mask, lengths)?;
        pool_and_normalize(&hidden, mask, self.pooling)
    }

    /// Run the encoder and return the raw last-layer hidden states
    /// `[batch, seq, hidden]`.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        self.forward_hidden_with_lengths(input_ids, mask, None)
    }

    /// [`Self::forward_hidden`], but with the batch's row `lengths`
    /// ALREADY known host-side — e.g. from the SAME tokenizer call that
    /// produced `input_ids`/`mask` (`encode_texts`, `trainer.rs`'s
    /// `BatchLongest` right-padding). `lengths` is a TRUST boundary
    /// (family D still applies at the edge, but the edge moves to the
    /// CALLER): this function does NOT re-derive lengths or re-verify the
    /// prefix structure from `mask` on the device — that is exactly the
    /// `flash_d2h_syncs` sync path P exists to avoid paying — it only
    /// checks the cheap, host-only shape facts `flash_admission_predicate`'s
    /// `trusted_lengths` branch documents (length count matches batch, each
    /// length `<= seq`). A caller whose `lengths` do NOT actually match
    /// `mask`'s real padding structure gets a WRONG flash-eligibility
    /// decision, not a caught error — this is the documented cost of
    /// skipping the sync, not a silently-tolerated bug. `lengths: None` is
    /// [`Self::forward_hidden`]'s exact prior behaviour (path F, unchanged).
    pub fn forward_hidden_with_lengths(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
        lengths: Option<&[usize]>,
    ) -> Result<Tensor, EncoderError> {
        let (_batch, seq) = input_ids.dims2()?;
        if seq > self.max_position_embeddings {
            return Err(EncoderError::SequenceTooLong {
                seq,
                max: self.max_position_embeddings,
            });
        }

        let word_emb = self.word_embeddings.forward(input_ids)?;
        let mut hidden = self.emb_norm.forward(&word_emb)?;

        let extended = extended_attention_mask(mask)?;
        // Built once per forward, not per layer: the band depends only on the
        // sequence length and the window, so every local layer shares it.
        let local_band = match self.local_half_window {
            None => None,
            Some(half) => Some(self.sliding_band(seq, half, input_ids.device())?),
        };
        // The FUSED training arm's masks, built ONCE per forward (at most
        // 3 launches — see `FusedAttentionMasks`'s doc for the count the
        // per-layer alternative paid) and shared by every layer; eval
        // never reads them, so they are not built there.
        //
        // KNOWN GAP, NOT FIXED THIS ROUND (P3 fix round 4, B4 — deferred a
        // THIRD time, honestly, rather than risk a hard regression this
        // late in the round): this bundle is built whenever `self.training`
        // is true, regardless of `head_dim`, even though
        // `AttentionBlockFused` admits ONLY `head_dim ==
        // ATTENTION_BLOCK_HEAD_DIM` (module doc's "Fixed domain" section)
        // — a head_dim-16 checkpoint (the cookbook's own
        // `tiny_modernbert_local` fixture) never dispatches fused and pays
        // a `batch·seq²` allocate-add-cast every training forward it
        // cannot use. `ModernBertAttention::forward`'s OWN contract
        // (this file, `fused_masks: Option<&FusedAttentionMasks>`'s doc
        // table) makes `fused_masks` REQUIRED whenever `self.training` —
        // `None` is a typed `Config` refusal, unconditionally, at EVERY
        // layer, not just a fusable one — so skipping construction here
        // for a non-fusable model requires first relaxing that contract to
        // "required only at a layer that can actually dispatch fused",
        // which is a real (if small) change to error-path behaviour this
        // round did not have time to make and verify safely.
        let fused_masks = if self.training {
            Some(FusedAttentionMasks::build(
                &extended,
                local_band.as_ref(),
                hidden.dtype(),
            )?)
        } else {
            None
        };
        // The flash-cascade decision (contract v4 §3.2), decided ONCE per
        // forward exactly like `fused_masks` above — `None` in eval (the
        // flash arm is training-only, contract v4 §2 scope: "eval/serving
        // stays eager"). `mask` (not `extended`, which is already additive
        // `0`/`MASKED_LOGIT`-valued) is the raw `0.0`/`1.0` padding mask
        // `compute_lengths_and_prefix` needs.
        let flash_admission = if self.training {
            let head_dim = self
                .layers
                .first()
                .map(|l| l.attention.head_dim)
                .unwrap_or(0);
            Some(decide_flash_admission(
                input_ids.device(),
                hidden.dtype(),
                head_dim,
                mask,
                lengths,
            )?)
        } else {
            None
        };
        for layer in &self.layers {
            hidden = layer.forward(
                &hidden,
                &extended,
                local_band.as_ref(),
                fused_masks.as_ref(),
                flash_admission.as_ref(),
            )?;
        }

        self.final_norm.forward(&hidden)
    }

    /// The sliding-window band for `seq`, built once per length and reused.
    fn sliding_band(
        &self,
        seq: usize,
        half: usize,
        device: &Device,
    ) -> Result<Tensor, EncoderError> {
        let mut cache = self
            .band_cache
            .lock()
            .map_err(|_| EncoderError::Config("sliding-window band cache poisoned".into()))?;
        if let Some(band) = cache.get(&seq) {
            return Ok(band.clone());
        }
        let band = sliding_window_mask(seq, half, device)?;
        cache.insert(seq, band.clone());
        Ok(band)
    }

    /// Borrowed references to every trainable LoRA tensor in the encoder.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.attention.wqkv.trainable_params());
            params.extend(layer.attention.wo.trainable_params());
            params.extend(layer.mlp.wi.trainable_params());
            params.extend(layer.mlp.wo.trainable_params());
        }
        params
    }

    /// CPU-side export of every LoRA `A` and `B` tensor, keyed by
    /// `layer.{n}.{site}.lora_{a|b}`.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            out.extend(
                layer
                    .attention
                    .wqkv
                    .named_weights(&format!("layer.{n}.Wqkv"))?,
            );
            out.extend(layer.attention.wo.named_weights(&format!("layer.{n}.Wo"))?);
            out.extend(layer.mlp.wi.named_weights(&format!("layer.{n}.Wi"))?);
            out.extend(layer.mlp.wo.named_weights(&format!("layer.{n}.mlp.Wo"))?);
        }
        Ok(out)
    }

    /// Toggle training mode on every LoRA-augmented linear, every
    /// LayerNorm, AND every attention layer's RoPE application. ModernBERT's
    /// LayerNorms use the bias-free variant: in EVAL mode (`training =
    /// false`) the forward stays on the slow primitive-op path exactly as
    /// before, unconditionally; in TRAINING mode (`training = true`) it
    /// dispatches to the fused CUDA/CPU LayerNorm kernel when that
    /// kernel's own domain holds (dtype, contiguity, device, hidden
    /// size), falling back to the slow path otherwise — see
    /// `crate::layer_norm`'s module doc. RoPE follows the SAME doctrine:
    /// eval always calls `RotaryEmbedding::apply` directly (OUTPUT VALUES
    /// bit-identical before/after the fused kernel; `apply` itself now
    /// also takes a table-cache lock from table hoisting, an uncontended,
    /// non-numeric change disclosed on `RotaryEmbedding`'s own doc),
    /// training calls `RotaryEmbedding::apply_training` (fused kernel
    /// when its own domain holds, else the identical eager `apply`) — see
    /// `ModernBertAttention::rope_apply`. Propagating the flag keeps the
    /// surface consistent with [`crate::Bert`] and [`crate::DistilBert`].
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.emb_norm.set_training(training);
        for layer in &mut self.layers {
            layer.attention.wqkv.set_training(training);
            layer.attention.wo.set_training(training);
            layer.attention.set_training(training);
            if let Some(attn_norm) = layer.attention.attn_norm.as_mut() {
                attn_norm.set_training(training);
            }
            layer.mlp.wi.set_training(training);
            layer.mlp.wo.set_training(training);
            layer.mlp.mlp_norm.set_training(training);
            layer.mlp.set_training(training);
        }
        self.final_norm.set_training(training);
    }

    /// Restore LoRA `A`/`B` tensors from a `named_trainable_weights`-shaped map.
    /// Missing keys are silently skipped — see
    /// [`MaybeLoraLinear::load_weights`].
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        for (n, layer) in self.layers.iter_mut().enumerate() {
            layer
                .attention
                .wqkv
                .load_weights(weights, &format!("layer.{n}.Wqkv"));
            layer
                .attention
                .wo
                .load_weights(weights, &format!("layer.{n}.Wo"));
            layer.mlp.wi.load_weights(weights, &format!("layer.{n}.Wi"));
            layer
                .mlp
                .wo
                .load_weights(weights, &format!("layer.{n}.mlp.Wo"));
        }
        Ok(())
    }

    /// Per-site dropout-stream positions keyed `{site}.dropout`, over the same
    /// site names [`Self::named_trainable_weights`] uses — the resume state for
    /// the adapter's dropout.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            for (site, lin) in modern_lora_sites(layer) {
                lin.collect_dropout_position(&format!("layer.{n}.{site}"), &mut out)?;
            }
        }
        Ok(out)
    }

    /// Restore each LoRA site's dropout-stream position from a
    /// [`Self::dropout_positions`]-shaped map. Missing keys are no-ops.
    pub fn restore_dropout_positions(
        &self,
        positions: &HashMap<String, u64>,
    ) -> Result<(), EncoderError> {
        for (n, layer) in self.layers.iter().enumerate() {
            for (site, lin) in modern_lora_sites(layer) {
                lin.restore_dropout_position(&format!("layer.{n}.{site}"), positions)?;
            }
        }
        Ok(())
    }
}

/// The four LoRA-wrappable linear sites of one ModernBERT layer paired with their
/// `named_trainable_weights` site names.
fn modern_lora_sites(layer: &ModernBertLayer) -> [(&'static str, &MaybeLoraLinear); 4] {
    [
        ("Wqkv", &layer.attention.wqkv),
        ("Wo", &layer.attention.wo),
        ("Wi", &layer.mlp.wi),
        ("mlp.Wo", &layer.mlp.wo),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for [`ModernBert`]. Mirrors `BertBuilder` so callers can swap
/// encoder families without touching their builder pipeline.
pub struct ModernBertBuilder<'a> {
    pooling: Pooling,
    lora: LoraBuildConfig<'a>,
    backbone_dtype: DType,
    adapter_file: Option<&'a Path>,
}

impl<'a> ModernBertBuilder<'a> {
    /// Select the sentence-embedding pooling strategy used by
    /// [`ModernBert::forward`].
    pub fn pooling(mut self, p: Pooling) -> Self {
        self.pooling = p;
        self
    }

    /// Provide a LoRA build configuration; defaults to
    /// [`LoraBuildConfig::frozen`].
    pub fn lora(mut self, l: LoraBuildConfig<'a>) -> Self {
        self.lora = l;
        self
    }

    /// Override the backbone dtype (default `F32`).
    pub fn backbone_dtype(mut self, d: DType) -> Self {
        self.backbone_dtype = d;
        self
    }

    /// Provide an optional path to a pre-trained LoRA adapter safetensors
    /// file. When `None`, LoRA tensors are initialised via the supplied
    /// [`VarMap`] at build time.
    pub fn adapter(mut self, p: Option<&'a Path>) -> Self {
        self.adapter_file = p;
        self
    }

    /// Load the backbone (and optional adapter) and assemble a [`ModernBert`].
    pub fn build(
        self,
        weights_paths: &[&Path],
        config: &ModernBertConfig,
        device: &Device,
        varmap: &VarMap,
    ) -> Result<ModernBert, EncoderError> {
        // Refuse a config this port cannot honour rather than reinterpreting it.
        // Upstream raises on `i % 0`; silently treating it as all-global would
        // be the same silent-wrong-function class this sliding-window support
        // exists to remove.
        if config.global_attn_every_n_layers == 0 {
            return Err(EncoderError::Config(
                "global_attn_every_n_layers must be > 0 (1 = every layer global)".into(),
            ));
        }
        if config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
        {
            return Err(EncoderError::Config(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                config.hidden_size, config.num_attention_heads
            )));
        }
        // `AttentionBlockFused` has no dropout slot (its own module doc's
        // "Domain" section) and ModernBertAttention::forward never applied
        // `attention_dropout` even before this commit — a loud, typed
        // refusal here converts that pre-existing silent-drop into a
        // visible error instead of a confidently-wrong forward.
        if config.attention_dropout != 0.0 {
            return Err(EncoderError::Config(format!(
                "attention_dropout must be 0.0 (this port implements no attention dropout), got \
                 {}",
                config.attention_dropout
            )));
        }

        let frozen_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weights_paths, self.backbone_dtype, device)?
        };
        let lora_vb = if let Some(adapter) = self.adapter_file {
            unsafe { VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device)? }
        } else {
            VarBuilder::from_varmap(varmap, DType::F32, device)
        };

        let head_dim = config.hidden_size / config.num_attention_heads;

        // Exactly two RoPE tables per model, shared by every layer of the
        // matching attention type. Building one per layer would allocate
        // `num_hidden_layers` identical tables.
        let global_rope = Arc::new(RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.global_rope_theta,
            device,
        )?);
        let local_rope = Arc::new(RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.local_rope_theta,
            device,
        )?);

        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            frozen_vb.pp("model.embeddings.tok_embeddings"),
        )?;
        let emb_norm = LayerNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            false,
            frozen_vb.pp("model.embeddings.norm"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for n in 0..config.num_hidden_layers {
            let layer_vb = frozen_vb.pp(format!("model.layers.{n}"));
            let lora_layer_vb = lora_vb.pp(format!("layer.{n}"));
            let site = LoraSite {
                layer_vb: &layer_vb,
                lora_layer_vb: &lora_layer_vb,
                layer_idx: n,
                lora: self.lora,
                varmap,
            };

            let wqkv = site.build(
                "Wqkv",
                "attn.Wqkv",
                config.hidden_size,
                config.hidden_size * 3,
            )?;
            let wo = site.build("Wo", "attn.Wo", config.hidden_size, config.hidden_size)?;

            let attn_norm = if n == 0 {
                None
            } else {
                Some(LayerNorm::new(
                    config.hidden_size,
                    config.layer_norm_eps,
                    false,
                    layer_vb.pp("attn_norm"),
                )?)
            };

            let is_local = config.is_local_layer(n);
            let rope = if is_local {
                Arc::clone(&local_rope)
            } else {
                Arc::clone(&global_rope)
            };

            let wi = site.build(
                "Wi",
                "mlp.Wi",
                config.hidden_size,
                config.intermediate_size * 2,
            )?;
            let mlp_wo = site.build(
                "mlp.Wo",
                "mlp.Wo",
                config.intermediate_size,
                config.hidden_size,
            )?;
            let mlp_norm = LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
                false,
                layer_vb.pp("mlp_norm"),
            )?;

            layers.push(ModernBertLayer {
                attention: ModernBertAttention {
                    wqkv,
                    wo,
                    attn_norm,
                    rope,
                    is_local,
                    num_heads: config.num_attention_heads,
                    head_dim,
                    half_window: is_local.then(|| config.half_window()),
                    training: false,
                },
                mlp: ModernBertMlp {
                    wi,
                    wo: mlp_wo,
                    mlp_norm,
                    training: false,
                },
            });
        }

        let final_norm = LayerNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            false,
            frozen_vb.pp("model.final_norm"),
        )?;

        Ok(ModernBert {
            word_embeddings,
            emb_norm,
            layers,
            final_norm,
            pooling: self.pooling,
            hidden_size: config.hidden_size,
            max_position_embeddings: config.max_position_embeddings,
            local_half_window: (0..config.num_hidden_layers)
                .any(|n| config.is_local_layer(n))
                .then(|| config.half_window()),
            band_cache: Mutex::new(HashMap::new()),
            training: false,
        })
    }
}

/// Per-layer scratchpad that captures the shared inputs of every LoRA-site
/// load — the frozen and adapter VarBuilders, the layer index, and the
/// caller's `LoraBuildConfig` — so the per-site call only varies in the four
/// values that actually differ between sites.
struct LoraSite<'a, 'b> {
    layer_vb: &'a VarBuilder<'b>,
    lora_layer_vb: &'a VarBuilder<'b>,
    layer_idx: usize,
    lora: LoraBuildConfig<'b>,
    /// The trainable `VarMap` the seeded LoRA A/B tensors are registered into.
    varmap: &'a VarMap,
}

impl<'a, 'b> LoraSite<'a, 'b> {
    fn build(
        &self,
        target_name: &str,
        safetensors_sub: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<MaybeLoraLinear, EncoderError> {
        let frozen = linear_no_bias(in_features, out_features, self.layer_vb.pp(safetensors_sub))?;
        if should_apply_lora(
            target_name,
            self.lora.target_modules,
            self.layer_idx,
            self.lora.layers_to_transform,
        ) {
            let rank = effective_rank(target_name, self.lora.lora_rank, self.lora.rank_pattern);
            let lora_linear = LoraLinear::new(
                frozen,
                rank,
                self.lora.lora_alpha,
                self.lora.use_rslora,
                self.lora.init_mode,
                self.lora.lora_dropout,
                self.lora.seed,
                self.varmap,
                &self.lora_layer_vb.pp(target_name),
            )?;
            Ok(MaybeLoraLinear::Lora(lora_linear))
        } else {
            Ok(MaybeLoraLinear::Frozen(frozen))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;
    use half::bf16;

    /// Serializes every test in THIS module that reads
    /// `ATTENTION_BLOCK_DISPATCH_COUNTERS` (a process-wide, `#[cfg(test)]`-
    /// shared static): `cargo test`'s default per-binary thread pool runs
    /// `#[test]` fns in parallel, so an exact-equality "eval must not
    /// advance the counter" assertion in one such test would be flaky if
    /// another concurrently ran a training-mode `AttentionBlockFused`
    /// dispatch — mirrors `tests/it/modernbert.rs`'s
    /// `DISPATCH_COUNTER_TEST_LOCK` (a SEPARATE lock: that one guards the
    /// integration-test binary's own tests reading the SAME process-wide
    /// counters through the crate's public `attention_block_dispatch_
    /// snapshot` API — a different binary, so a different `Mutex`).
    static ATTENTION_BLOCK_COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A flash-cascade decision that always declines — the stub every
    /// pre-existing (pre-flash) test in this module passes so its call to
    /// `ModernBertAttention::forward`/`forward_training_attention` keeps
    /// compiling against the widened signature. Matches EXACTLY what
    /// `decide_flash_admission` itself produces today (see that function's
    /// doc): `attention_block_flash` has no real op to dispatch to yet
    /// (B1), so no test-constructed decision in this module is ever
    /// `PredicateOutcome::Holds`.
    fn declined_flash() -> FlashDecision {
        FlashDecision::Declined {
            outcome: PredicateOutcome::CapabilityMiss,
            reason: "test_stub_flash_declined",
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Flash-cascade admission seam (P6 Stage B B2)
    // ─────────────────────────────────────────────────────────────────

    /// Serializes every test in this module that reads [`FLASH_D2H_SYNCS`]
    /// — the SAME process-wide-static hazard [`ATTENTION_BLOCK_COUNTER_TEST_LOCK`]
    /// already documents for `ATTENTION_BLOCK_DISPATCH_COUNTERS`, for the
    /// same reason: `cargo test`'s default parallel thread pool would
    /// otherwise let two tests' "exactly `+1`" assertions race.
    static FLASH_D2H_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn flash_arch_ok_rejects_cpu() {
        assert!(!flash_arch_ok(&Device::Cpu));
    }

    /// L7-equivalent + L10: on this build (no `cuda`/`flash-attn` feature,
    /// and CPU regardless), the CHEAPEST gate declines first —
    /// `flash_admission_predicate` never reaches `compute_lengths_and_prefix`,
    /// so it must pay ZERO device syncs. Also proves the ordering
    /// (feature/device/arch BEFORE the mask reduction) this module's doc
    /// claims is why every block leg in this crate's test suite reads
    /// `flash_d2h_syncs() == 0`.
    #[test]
    fn flash_admission_predicate_declines_on_cpu_without_paying_the_d2h_sync() {
        let _guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let mask = Tensor::from_slice(&[1f32, 1.0, 1.0, 1.0], (1, 4), &device).unwrap();
        let before = flash_d2h_syncs();
        let (outcome, reason, eligible) =
            flash_admission_predicate(&device, DType::BF16, FLASH_HEAD_DIM, &mask, None).unwrap();
        let after = flash_d2h_syncs();
        assert_eq!(outcome, PredicateOutcome::CapabilityMiss);
        assert!(
            reason == "flash_attn_feature_compiled" || reason == "device_is_cuda",
            "unexpected reason: {reason}"
        );
        assert!(eligible.is_none());
        assert_eq!(
            after, before,
            "a cheap capability gate must decline BEFORE the mask reduction ever runs"
        );
    }

    /// [`flash_capability_gates`]'s `feature_compiled` gate, tested
    /// directly with a literal `bool` — independent of the REAL build's
    /// `jammi_kernels::admission::FLASH_COMPILED` (always `false` in this
    /// crate's own test suite, which is exactly why
    /// [`flash_admission_predicate`] itself can never exercise the gates
    /// AFTER this one — see this function's own doc).
    #[test]
    fn flash_capability_gates_feature_off_is_capability_miss() {
        let miss = flash_capability_gates(false, &Device::Cpu, DType::BF16, FLASH_HEAD_DIM);
        assert_eq!(
            miss,
            Some((
                PredicateOutcome::CapabilityMiss,
                "flash_attn_feature_compiled",
                None
            ))
        );
    }

    /// The device gate, reachable ONLY with `feature_compiled = true`
    /// (forced here as a literal — the real build never sets it) — this is
    /// the one cheap gate past the feature check this crate's test suite
    /// CAN exercise without a real CUDA device: `Device::Cpu` always fails
    /// `device.is_cuda()` regardless of `feature_compiled`.
    #[test]
    fn flash_capability_gates_feature_on_cpu_device_is_capability_miss() {
        let miss = flash_capability_gates(true, &Device::Cpu, DType::BF16, FLASH_HEAD_DIM);
        assert_eq!(
            miss,
            Some((PredicateOutcome::CapabilityMiss, "device_is_cuda", None))
        );
    }

    /// Every gate passes: `None` (proceed to `resolve_lengths_and_prefix`).
    /// UNREACHABLE on `Device::Cpu` (this crate's test suite has no CUDA
    /// device) — this test is a hermetic placeholder honestly documenting
    /// that gap, not a real coverage claim: the arch/dtype/head_dim gates
    /// (`flash_arch_ok`'s call site, `dtype != DType::BF16`, `head_dim !=
    /// FLASH_HEAD_DIM`) can only be exercised with `device.is_cuda() ==
    /// true`, which requires an actual CUDA device this environment does
    /// not have. `flash_arch_ok`'s OWN internal `==` comparison IS tested
    /// directly (`flash_arch_ok_rejects_cpu`) — what remains untestable
    /// here is only the CALL SITE inside `flash_capability_gates`, and the
    /// two gates after it. A pod run (`JAMMI_REQUIRE_CUDA=1`) closing this
    /// residual class is listed explicitly in this agent's hand-off.
    #[test]
    fn flash_capability_gates_arch_dtype_head_dim_gates_are_untestable_without_cuda() {
        // Documents the gap; asserts nothing about the untestable branches.
        assert!(!Device::Cpu.is_cuda());
    }

    /// Path P (contract v4 §3.7, v5 item 3): trusted host-side lengths pay
    /// ZERO `flash_d2h_syncs` — the whole point of skipping the device
    /// reduction — and still reach `Holds` with the right `CompactedBatch`
    /// inputs.
    #[test]
    fn resolve_lengths_and_prefix_trusted_lengths_pays_zero_d2h_syncs() {
        let _guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        // The mask's OWN bytes are irrelevant on the trusted-lengths path
        // (never read for lengths/prefix) — deliberately NOT matching the
        // trusted lengths here, to prove that.
        let mask = Tensor::zeros((2, 5), DType::F32, &device).unwrap();
        let before = flash_d2h_syncs();
        let (outcome, reason, eligible) =
            resolve_lengths_and_prefix(&mask, Some(&[3usize, 5usize])).unwrap();
        let after = flash_d2h_syncs();
        assert_eq!(after, before, "path P must pay ZERO device syncs");
        assert_eq!(outcome, PredicateOutcome::Holds, "reason={reason}");
        let (lengths, seq) = eligible.unwrap();
        assert_eq!(lengths, vec![3, 5]);
        assert_eq!(seq, 5);
    }

    #[test]
    fn resolve_lengths_and_prefix_trusted_lengths_wrong_batch_count_is_domain_miss() {
        let device = Device::Cpu;
        let mask = Tensor::zeros((2, 5), DType::F32, &device).unwrap();
        let (outcome, reason, eligible) =
            resolve_lengths_and_prefix(&mask, Some(&[3usize])).unwrap();
        assert_eq!(outcome, PredicateOutcome::DomainMiss);
        assert_eq!(reason, "trusted_lengths_len_matches_batch");
        assert!(eligible.is_none());
    }

    #[test]
    fn resolve_lengths_and_prefix_trusted_lengths_exceeding_seq_is_domain_miss() {
        let device = Device::Cpu;
        let mask = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let (outcome, reason, eligible) =
            resolve_lengths_and_prefix(&mask, Some(&[5usize])).unwrap();
        assert_eq!(outcome, PredicateOutcome::DomainMiss);
        assert_eq!(reason, "trusted_lengths_within_seq");
        assert!(eligible.is_none());
    }

    /// `trusted_lengths = None` reduces `resolve_lengths_and_prefix` to
    /// EXACTLY path F (`compute_lengths_and_prefix`) — the same
    /// interior-zero-vs-prefix distinction proven directly against
    /// `compute_lengths_and_prefix` elsewhere in this module holds through
    /// this wrapper too.
    #[test]
    fn resolve_lengths_and_prefix_none_falls_back_to_path_f() {
        let _guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let mask = Tensor::from_slice(&[1f32, 0.0, 1.0, 0.0], (1, 4), &device).unwrap();
        let before = flash_d2h_syncs();
        let (outcome, reason, eligible) = resolve_lengths_and_prefix(&mask, None).unwrap();
        let after = flash_d2h_syncs();
        assert_eq!(after, before + 1, "path F pays exactly one sync");
        assert_eq!(outcome, PredicateOutcome::DomainMiss);
        assert_eq!(reason, "mask_is_prefix_every_row");
        assert!(eligible.is_none());
    }

    /// The lead's exact correction: row LENGTHS alone cannot decide
    /// "prefix" — `[1, 0, 1, 0]` sums to `2`, the SAME length a genuine
    /// prefix `[1, 1, 0, 0]` reports, but is not one (an interior zero, L3).
    #[test]
    fn compute_lengths_and_prefix_distinguishes_interior_zero_from_a_true_prefix_of_equal_length() {
        // Doesn't read `flash_d2h_syncs()` itself, but DOES increment the
        // same process-wide counter another test measures exactly — must
        // still take the lock so it cannot interleave with that one.
        let _guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let genuine_prefix = Tensor::from_slice(&[1f32, 1.0, 0.0, 0.0], (1, 4), &device).unwrap();
        let interior_zero = Tensor::from_slice(&[1f32, 0.0, 1.0, 0.0], (1, 4), &device).unwrap();

        let (lengths_a, is_prefix_a) = compute_lengths_and_prefix(&genuine_prefix).unwrap();
        assert_eq!(lengths_a, vec![2]);
        assert!(is_prefix_a);

        let (lengths_b, is_prefix_b) = compute_lengths_and_prefix(&interior_zero).unwrap();
        assert_eq!(
            lengths_b,
            vec![2],
            "same SUM as the genuine prefix — this is exactly why lengths alone cannot decide it"
        );
        assert!(
            !is_prefix_b,
            "an interior zero must NOT read as a prefix mask even though its length matches one"
        );
    }

    #[test]
    fn compute_lengths_and_prefix_all_ones_and_mixed_batch() {
        // See the sibling test's identical lock rationale.
        let _guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let all_ones = Tensor::from_slice(&[1f32, 1.0, 1.0, 1.0], (1, 4), &device).unwrap();
        let (lengths, is_prefix) = compute_lengths_and_prefix(&all_ones).unwrap();
        assert_eq!(lengths, vec![4]);
        assert!(
            is_prefix,
            "an all-ones row is a trivial (full-length) prefix"
        );

        // A mixed batch: row 0 a genuine prefix, row 1 all-ones, row 2
        // interior-zero — the WHOLE-BATCH predicate must be false the
        // moment ANY row is non-prefix (contract v4 L3/L5: mixed batch ->
        // DomainMiss for the whole forward).
        let mixed = Tensor::from_slice(
            &[
                1f32, 1.0, 0.0, 0.0, // row 0: prefix, length 2
                1.0, 1.0, 1.0, 1.0, // row 1: prefix (full), length 4
                1.0, 0.0, 1.0, 0.0, // row 2: interior zero, NOT a prefix
            ],
            (3, 4),
            &device,
        )
        .unwrap();
        let (lengths_mixed, is_prefix_mixed) = compute_lengths_and_prefix(&mixed).unwrap();
        assert_eq!(lengths_mixed, vec![2, 4, 2]);
        assert!(!is_prefix_mixed);
    }

    /// Exactly ONE sync per call — the transfer `FLASH_D2H_SYNCS` counts,
    /// not `b` separate ones (this test's own batch has `b = 5` rows).
    #[test]
    fn compute_lengths_and_prefix_counts_exactly_one_d2h_sync_per_call_regardless_of_batch_size() {
        let _guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let mask = Tensor::ones((5, 4), DType::F32, &device).unwrap();
        let before = flash_d2h_syncs();
        let _ = compute_lengths_and_prefix(&mask).unwrap();
        let after = flash_d2h_syncs();
        assert_eq!(after, before + 1);
    }

    #[test]
    fn unpad_gather_indices_matches_hand_computed_flat_row_offsets() {
        let device = Device::Cpu;
        // lengths [3, 0, 2] over seq=4: batch 1 is entirely padding (a
        // zero-length row — L4's own domain edge), batches 0 and 2 are
        // real prefixes.
        let idx = unpad_gather_indices(&[3, 0, 2], 4, &device).unwrap();
        let got: Vec<u32> = idx.to_vec1().unwrap();
        // batch 0: rows 0,1,2 (flat 0,1,2); batch 1: none; batch 2: rows
        // 0,1 at flat offset 2*4=8 -> 8,9.
        assert_eq!(got, vec![0, 1, 2, 8, 9]);
    }

    /// Contract v4 §3.5's own failure mode, disclosed directly: repad's
    /// destination must be freshly zeroed, never a buffer that could carry
    /// stale/NaN bytes into a pad row via `0.0 * NaN = NaN`
    /// (`crate::pooling`'s own doc). Round-trips `unpad_rows` then
    /// `repad_rows` and asserts pad rows are EXACTLY `0.0` (never `NaN`)
    /// and every real row is bit-identical to the original padded input.
    #[test]
    #[allow(clippy::needless_range_loop)] // `b` indexes THREE independent arrays via a computed flat offset, not one
    fn unpad_repad_round_trip_zeroes_pad_rows_exactly_and_preserves_real_rows_bit_identical() {
        let device = Device::Cpu;
        let (batch, seq, hidden) = (2usize, 4usize, 3usize);
        // batch 0: fully real (length 4); batch 1: real length 2, padded
        // rows 2,3 — deliberately LARGE, non-round values so a bit-identity
        // check on the real rows is meaningful.
        let lengths = vec![4usize, 2usize];
        let mut data = Vec::with_capacity(batch * seq * hidden);
        for i in 0..(batch * seq * hidden) {
            data.push((i as f32) * 0.371 - 5.0);
        }
        let padded = Tensor::from_vec(data.clone(), (batch, seq, hidden), &device).unwrap();

        let gather_indices = unpad_gather_indices(&lengths, seq, &device).unwrap();
        let compacted = unpad_rows(&padded, &gather_indices).unwrap();
        assert_eq!(compacted.dims(), &[6, hidden]); // total = 4 + 2

        let repadded = repad_rows(&compacted, &gather_indices, batch, seq).unwrap();
        assert_eq!(repadded.dims(), &[batch, seq, hidden]);
        let got: Vec<f32> = repadded.flatten_all().unwrap().to_vec1().unwrap();

        for b in 0..batch {
            for s in 0..seq {
                let flat = (b * seq + s) * hidden;
                if s < lengths[b] {
                    for h in 0..hidden {
                        assert_eq!(
                            got[flat + h],
                            data[flat + h],
                            "real row (b={b}, s={s}) must be bit-identical"
                        );
                    }
                } else {
                    for h in 0..hidden {
                        assert_eq!(
                            got[flat + h],
                            0.0,
                            "pad row (b={b}, s={s}) must be EXACTLY 0.0, never a stale/NaN byte"
                        );
                        assert!(!got[flat + h].is_nan());
                    }
                }
            }
        }
    }

    /// bf16 leg of the same round trip (CUDA-when-available is deferred —
    /// this crate's test suite has no CUDA device; the CPU/bf16 leg is the
    /// dtype this arm actually runs at in production).
    #[test]
    #[allow(clippy::needless_range_loop)] // `b` indexes both `lengths` and a computed flat offset into `data`/`got`
    fn unpad_repad_round_trip_bf16() {
        let device = Device::Cpu;
        let (batch, seq, hidden) = (2usize, 3usize, 2usize);
        let lengths = vec![2usize, 1usize];
        let data: Vec<bf16> = (0..(batch * seq * hidden))
            .map(|i| bf16::from_f32((i as f32) * 0.5 - 1.0))
            .collect();
        let padded = Tensor::from_vec(data.clone(), (batch, seq, hidden), &device).unwrap();
        let gather_indices = unpad_gather_indices(&lengths, seq, &device).unwrap();
        let compacted = unpad_rows(&padded, &gather_indices).unwrap();
        assert_eq!(compacted.dims(), &[3, hidden]);
        let repadded = repad_rows(&compacted, &gather_indices, batch, seq).unwrap();
        let got: Vec<bf16> = repadded.flatten_all().unwrap().to_vec1().unwrap();
        for b in 0..batch {
            for s in 0..seq {
                let flat = (b * seq + s) * hidden;
                for h in 0..hidden {
                    let expected = if s < lengths[b] {
                        data[flat + h]
                    } else {
                        bf16::from_f32(0.0)
                    };
                    assert_eq!(got[flat + h], expected, "(b={b}, s={s}, h={h})");
                }
            }
        }
    }

    /// [`build_flash_forward_decision`]'s PADDED downgrade rule, tested
    /// directly (no CUDA device needed to reach `Holds`): a `Holds`
    /// outcome on a batch with REAL padding (a length `< seq`) is DECLINED
    /// (`CapabilityMiss("flash_padded_not_yet_wired")`) — the padded
    /// regime is out of P6 Stage B B3-dense's scope, see this module's
    /// `decide_flash_admission` doc — and [`FlashDecision::Declined`]
    /// never carries a `CompactedBatch` (that type's own doc), so this
    /// test checks `CompactedBatch` CONSTRUCTION separately, via
    /// `unpad_gather_indices` directly on the SAME `lengths`, rather than
    /// through the (declined) decision. NOTE: `lengths=[3,0,2]` mixes a
    /// zero-length row with real padding purely to exercise
    /// `unpad_gather_indices`' own zero-length handling in one fixture —
    /// `flash_admission_predicate` would separately DomainMiss a
    /// zero-length row before ever reaching this function in the real
    /// call path (L4); this unit test bypasses that predicate on purpose
    /// to test `build_flash_forward_decision` in isolation.
    #[test]
    fn build_flash_forward_decision_downgrades_holds_padded_to_declined_without_a_compacted_batch()
    {
        let device = Device::Cpu;
        let lengths = vec![3usize, 0usize, 2usize];
        let decision = build_flash_forward_decision(
            PredicateOutcome::Holds,
            "domain_ok",
            Some((lengths.clone(), 4)),
            &device,
        )
        .unwrap();
        match decision {
            FlashDecision::Declined { outcome, reason } => {
                assert_eq!(outcome, PredicateOutcome::CapabilityMiss);
                assert_eq!(reason, "flash_padded_not_yet_wired");
            }
            FlashDecision::Fused(_) => panic!("a padded batch must decline, never fuse"),
        }
        // CompactedBatch construction (lengths/gather-indices/total) is a
        // property of `unpad_gather_indices` itself, checked directly —
        // this decision no longer carries one to inspect once declined.
        let total: usize = lengths.iter().sum();
        let idx: Vec<u32> = unpad_gather_indices(&lengths, 4, &device)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(idx.len(), total);
    }

    /// The DENSE counterpart (P6 Stage B B3-dense): a `Holds` outcome
    /// whose lengths are ALL `== seq` (uniform, no padding) keeps the
    /// REAL `Holds` outcome (`reason = "domain_ok_dense"`) — this is what
    /// makes `attention_block_flash` actually dispatch `Fused` for a
    /// dense batch, and [`FlashDecision::Fused`] carries the
    /// `CompactedBatch` by construction (no separate `Option` to unwrap).
    #[test]
    fn build_flash_forward_decision_keeps_holds_when_dense() {
        let device = Device::Cpu;
        let lengths = vec![4usize, 4usize, 4usize];
        let decision = build_flash_forward_decision(
            PredicateOutcome::Holds,
            "domain_ok",
            Some((lengths.clone(), 4)),
            &device,
        )
        .unwrap();
        assert_eq!(decision.outcome(), PredicateOutcome::Holds);
        assert_eq!(decision.reason(), "domain_ok_dense");
        let batch = match decision {
            FlashDecision::Fused(batch) => batch,
            FlashDecision::Declined { outcome, reason } => {
                panic!("a dense Holds predicate must fuse, got Declined({outcome:?}, {reason})")
            }
        };
        assert_eq!(batch.lengths, lengths);
        assert_eq!(batch.total, 12);
    }

    #[test]
    fn build_flash_forward_decision_domain_miss_never_builds_a_compacted_batch() {
        let device = Device::Cpu;
        let decision = build_flash_forward_decision(
            PredicateOutcome::DomainMiss,
            "mask_is_prefix_every_row",
            None,
            &device,
        )
        .unwrap();
        match decision {
            FlashDecision::Declined { outcome, reason } => {
                assert_eq!(outcome, PredicateOutcome::DomainMiss);
                assert_eq!(reason, "mask_is_prefix_every_row");
            }
            FlashDecision::Fused(_) => panic!("a DomainMiss predicate must never fuse"),
        }
    }

    /// The end-to-end proof: on the SAME `tests/fixtures/tiny_modernbert_head64`
    /// fixture `forward_hidden_reaches_the_fused_attention_block_on_a_head_dim_64_checkpoint`
    /// already drives, a training forward's flash-cascade counters read
    /// EXACTLY `declined == num_hidden_layers`, `fused == 0`, `eager == 0`
    /// (contract v4's item-2 correction states the general identity as
    /// `declined == 28 x forwards_per_step x steps`; here ONE forward,
    /// `num_hidden_layers` layers). AND the block arm's own dispatch/output
    /// is UNCHANGED — re-asserting the SAME properties that sibling test
    /// proves — which is the "nothing changes numerically" claim this
    /// seam's commit message makes.
    #[test]
    fn flash_cascade_never_changes_the_block_arm_dispatch_or_output() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _d2h_guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_modernbert_head64");
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");
        let varmap = candle_nn::VarMap::new();
        let mut model = ModernBert::builder()
            .build(&[weights.as_path()], &config, &device, &varmap)
            .unwrap();
        // Row 1 is right-padded (contract v4's premise) so the flash
        // predicate's own mask-prefix machinery is exercised end to end —
        // it must still decline (no CUDA / no flash-attn feature) and the
        // block arm's dispatch/output stay exactly what they were before
        // this seam existed.
        let input_ids =
            Tensor::new(&[[2u32, 5, 10, 3, 7, 9], [4u32, 8, 1, 6, 0, 0]], &device).unwrap();
        let mask = Tensor::new(&[[1u32, 1, 1, 1, 1, 1], [1u32, 1, 1, 1, 0, 0]], &device).unwrap();

        model.set_training(true);
        let block_before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let flash_before = cascade_counters_for("attention_block_flash").snapshot();
        let d2h_before = flash_d2h_syncs();
        let out = model.forward_hidden(&input_ids, &mask).unwrap();
        let block_after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let flash_after = cascade_counters_for("attention_block_flash").snapshot();
        let d2h_after = flash_d2h_syncs();

        assert_eq!(
            block_after.fused - block_before.fused,
            config.num_hidden_layers as u64,
            "the block arm's OWN dispatch is unchanged by the flash seam"
        );
        assert_eq!(block_after.eager, block_before.eager);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|x: &f32| x.is_finite()),
            "training output must be finite"
        );

        assert_eq!(
            flash_after.fused, flash_before.fused,
            "attention_block_flash cannot dispatch Fused — the op does not exist yet (B1)"
        );
        assert_eq!(
            flash_after.eager, flash_before.eager,
            "always 0 — see CascadeDispatchCounters's doc"
        );
        assert_eq!(
            flash_after.declined - flash_before.declined,
            config.num_hidden_layers as u64,
            "one admit_cascade call per layer per forward — contract v4 item 2's identity, \
             specialised to ONE forward here"
        );
        assert_eq!(
            d2h_after, d2h_before,
            "no CUDA / no flash-attn feature on this build -> the cheap gate declines before \
             ANY device sync — 0 on this block leg"
        );

        // Eval never even builds the flash-cascade decision at all.
        model.set_training(false);
        let flash_before_eval = cascade_counters_for("attention_block_flash").snapshot();
        let _ = model.forward_hidden(&input_ids, &mask).unwrap();
        let flash_after_eval = cascade_counters_for("attention_block_flash").snapshot();
        assert_eq!(
            flash_before_eval, flash_after_eval,
            "eval never consults the flash cascade"
        );
    }

    /// THE end-to-end proof for P6 Stage B B3-dense: on the SAME
    /// `tests/fixtures/tiny_modernbert_head64` checkpoint, a DENSE
    /// (all-ones, no padding) mask, CUDA, bf16 — `attention_block_flash`
    /// actually dispatches `Fused` (count == `num_hidden_layers`), the
    /// BLOCK arm's own counter does NOT move for those layers (flash took
    /// over, not a second arm racing it), `attention_block_flash.declined
    /// == 0`, and the output is finite. On a build WITHOUT the
    /// `flash-attn` feature (`cuda` alone), `FLASH_COMPILED` is `false`,
    /// so the SAME assertions invert: the cascade declines and the block
    /// arm fires instead — this test is meaningful (and green) under
    /// EITHER feature combination, not just the real one.
    #[test]
    #[cfg(feature = "cuda")]
    fn forward_hidden_dispatches_attention_block_flash_fused_on_a_dense_cuda_bf16_checkpoint() {
        let Some(device) = growth_oracle_cuda_device() else {
            return;
        };
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _d2h_guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_modernbert_head64");
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");
        let varmap = candle_nn::VarMap::new();
        let mut model = ModernBert::builder()
            .backbone_dtype(DType::BF16)
            .build(&[weights.as_path()], &config, &device, &varmap)
            .unwrap();
        // Dense: every row real, no padding -- `decide_flash_admission`'s
        // dense split (`build_flash_forward_decision`) keeps the real
        // `Holds`.
        let input_ids =
            Tensor::new(&[[2u32, 5, 10, 3, 7, 9], [4u32, 8, 1, 6, 9, 2]], &device).unwrap();
        let mask = Tensor::new(&[[1u32, 1, 1, 1, 1, 1], [1u32, 1, 1, 1, 1, 1]], &device).unwrap();

        model.set_training(true);
        let block_before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let flash_before = cascade_counters_for("attention_block_flash").snapshot();
        let out = model.forward_hidden(&input_ids, &mask).unwrap();
        let block_after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let flash_after = cascade_counters_for("attention_block_flash").snapshot();

        let out_f32: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            out_f32.iter().all(|x| x.is_finite()),
            "training output must be finite"
        );

        if jammi_kernels::admission::FLASH_COMPILED {
            assert_eq!(
                flash_after.fused - flash_before.fused,
                config.num_hidden_layers as u64,
                "attention_block_flash must dispatch Fused on every layer of a dense bf16 \
                 batch on this build: {flash_before:?} -> {flash_after:?}"
            );
            assert_eq!(
                flash_after.declined, flash_before.declined,
                "a dense batch must never decline on this build"
            );
            assert_eq!(
                flash_after.eager, flash_before.eager,
                "always 0 -- see CascadeDispatchCounters's doc"
            );
            assert_eq!(
                block_after.fused, block_before.fused,
                "the block arm must NOT ALSO fire for layers the flash arm already handled"
            );
        } else {
            assert_eq!(
                flash_after.declined - flash_before.declined,
                config.num_hidden_layers as u64,
                "without the flash-attn feature, FLASH_COMPILED is false and every layer \
                 declines at the cheap capability gate"
            );
            assert_eq!(
                flash_after.fused, flash_before.fused,
                "attention_block_flash cannot dispatch Fused without the flash-attn feature"
            );
            assert_eq!(
                block_after.fused - block_before.fused,
                config.num_hidden_layers as u64,
                "the block arm must fire for every layer instead"
            );
        }
    }

    // =====================================================================
    // BLOCK 1 (adversarial audit, `perf/p6-fa2-dense`): the flash arm had
    // NO encoder-level numeric oracle -- the sibling test above only
    // asserts `is_finite` + dispatch counters, which two auditor-built
    // mutants (K-slot never rotated; a local layer's sliding window
    // dropped) survive completely. This ships the cuda-kernel-guide's
    // §3.3 three-way oracle (flash-bf16 vs block-bf16 vs an f32
    // reference) on the REAL ModernBERT-large checkpoint, at production
    // shape, on BOTH the pooled embedding and a step-0 LoRA gradient.
    //
    // ## Phase-4 re-audit close-out (ledger rows 207, 214, 215) -- three
    // defects the auditor proved with numbers, all fixed in this round:
    //
    // 1. THE GRADIENT LEG WAS VACUOUS. The prior loss,
    //    `loss = sum(l2_normalize(mean_pool(hidden))^2)`, is IDENTICALLY
    //    `batch` (`pool_and_normalize` returns unit-norm rows), so
    //    `dL/d(theta) == 0` at EVERY dtype, for EVERY arm, always. The
    //    "grad err" this section used to print was a ratio over a ZERO
    //    denominator; the auditor showed a real K-unrotated kernel mutant
    //    IMPROVED that number and PASSED the grad leg. Fixed:
    //    [`flash_oracle_pooled_and_grad`] now takes a FIXED, seed-keyed
    //    random cotangent `dy` ([`flash_oracle_seeded_dy`]) and computes
    //    `loss = (pooled * dy).sum()` -- this file's own established
    //    pattern (`fused_attention_block_matches_eager_lora_gradients_at_production_seq_on_head64`'s
    //    own `dy`, same non-uniform-cotangent discipline the fully-masked
    //    softmax oracle documents the failure mode of). Since `pooled` is
    //    unit-norm and `dy` is essentially never parallel to it, the
    //    L2-normalize Jacobian's projection of `dy` onto `pooled`'s
    //    tangent space is generically nonzero, so this loss's gradient is
    //    a real, non-degenerate signal. [`relative_l1_error`] now asserts
    //    its denominator (`sum|reference|`) is POSITIVE before dividing --
    //    guide §3.7's "a NaN must fail, not read as a pass" extended to "a
    //    zero-signal reference must fail loudly, not silently divide by
    //    zero into a number that happens to look like a ratio".
    //
    // 2. THE 1.5x BOUND WAS NOT SEED-STABLE. A single seed's draw is not a
    //    distribution: the auditor's own 8-seed sweep at b8-s512 found the
    //    healthy pooled flash/block ratio ranging 0.80-1.48 (one seed at
    //    1.4789, razor-thin under the old K=1.5) while the K-unrotated /
    //    window-dropped mutants cleared that same bound by only 1.38-1.55x
    //    on some seeds -- nothing like the "8-11x" margin the single lucky
    //    seed this section used to cite suggested. Fixed:
    //    [`FLASH_ORACLE_SWEEP_SEEDS`] fixes EIGHT seeds, reused IDENTICALLY
    //    across the healthy oracle and every RED control below, at every
    //    shape; every oracle/control below asserts both the MEAN ratio
    //    over those 8 seeds and the MAX per-seed ratio against
    //    [`FLASH_ORACLE_K_MEAN_POOLED`] / [`FLASH_ORACLE_K_MAX_POOLED`] /
    //    [`FLASH_ORACLE_K_MEAN_GRAD`] / [`FLASH_ORACLE_K_MAX_GRAD`] -- see
    //    those constants' own doc comments for the measured healthy and
    //    mutant distributions this round derived them from (per-seed
    //    tables also live in the committed
    //    `2026-08-25-flash-arm-encoder-oracle-*.json` artifact).
    //
    // 3. THE STALE SIGN CLAIM. A previous revision of this comment said
    //    "the auditor's hand-run oracle measured err(flash,f32)=5.71e-2 vs
    //    err(block,f32)=9.11e-2 -- flash already closer to truth". That
    //    sentence is WRONG and is deleted here: the committed GREEN run
    //    under the OLD (single-seed) code already showed the opposite sign
    //    on the pooled leg (`0.18823 > 0.17482` at b8_s512, `0.16754 >
    //    0.15673` at b1_s128 -- flash FURTHER from the f32 reference, not
    //    closer), and this round's 8-seed sweep confirms it quantitatively
    //    (mean pooled ratio and per-seed sign reported by
    //    [`FLASH_ORACLE_K_MEAN_POOLED`]'s own doc). Guide §3.3's own
    //    acceptance line is "accept only if the fused arm is no further
    //    than eager" -- read literally, this arm does not clear that bar
    //    on the pooled leg. What [`FLASH_ORACLE_K_MEAN_POOLED`] /
    //    [`FLASH_ORACLE_K_MAX_POOLED`] actually gate on is narrower and
    //    stated honestly: flash's distance from f32 stays within a modest,
    //    BOUNDED multiple of the block arm's own distance (ordinary
    //    fused-RoPE-kernel bf16 rounding, ordinary run-to-run noise), never
    //    "closer than". Every injected wiring fault below clears that same
    //    bound many-fold on its own mean -- that separation, not a false
    //    "flash wins" claim, is what makes the bound discriminating.
    //
    // ## RED controls: fault injections, not source mutations
    //
    // None of the three edits `cuda/rope_positions.cu`, `ops/rope_positions.rs`,
    // `ops/flash_attention.rs`, or this file's own committed dispatch logic:
    //   - K-unrotated (`FlashFault::KUnrotated`): feeds
    //     [`jammi_kernels::ops::flash_attention_varlen_with_rope`] -- the
    //     REAL production op ([`FlashVarlenAttentionFusedRope`], proven
    //     bit-identical to the two-op composition by that op's own
    //     `fused_rope_matches_two_op_composition` test) -- a `qkv` whose K
    //     slot has been PRE-INVERSE-rotated (`RopePositionsFused::new(seq,
    //     true)`, the exact mechanism that op's own `bwd` un-rotation uses)
    //     so that when the op applies its OWN forward rotation to every
    //     slot, K comes out exactly as if it had never been rotated at
    //     all -- observably reproducing what a `slot == 2` -> `slot >= 1`
    //     kernel mutant would produce, without touching the op or the
    //     kernel it calls.
    //   - Window dropped: mutates the ALREADY-BUILT model's own
    //     `ModernBertAttention::half_window` field to `None` on every
    //     layer (a private, same-module field -- not a source edit)
    //     before running the SAME production `forward_flash_dense_attention`
    //     composition.
    //   - `softmax_scale` gets its own cheap fault (`FlashFault::
    //     BadSoftmaxScale`), reusing the K-unrotated harness's production
    //     op call with only `cfg.softmax_scale` replaced.
    //
    // ## Class sweep (other flash wiring quantities)
    //
    // The `[b,s,3hd]->[b*s,3,h,d]` reshape and the output unpack
    // (`o.reshape((batch, seq, h*d))`) are NOT separately injected: either
    // one scrambles EVERY element of EVERY token (not a narrow-band or
    // single-slot defect), an unmissably larger distortion than the
    // committed controls that would fail this same oracle by a much wider
    // margin -- the K-unrotated, window-dropped, and bad-softmax-scale
    // controls are the NARROW, hard-to-catch defects this oracle exists to
    // prove it catches; a reshape defect is not in that class. A NARROWER
    // `softmax_scale * 1.02` (0.1275 vs the production 0.125) class-sweep
    // probe was run this round as a diagnostic (not committed, to avoid a
    // flaky assertion on a near-bound perturbation): over the SAME 8
    // seeds, mean pooled ratio = 1.3364 (stays BELOW
    // `FLASH_ORACLE_K_MEAN_POOLED` = 1.6) and mean grad ratio = 1.7335
    // (stays BELOW `FLASH_ORACLE_K_MEAN_GRAD` = 4.5) -- NEITHER leg catches
    // this narrow a perturbation at this bound; stated explicitly rather
    // than silently dropped. The op-level `flash_torch_parity`
    // (`jammi-kernels`) already covers `* 1.05` at the kernel level, which
    // this encoder-level oracle does not duplicate.
    //
    // ## The two hand-synced mirrors, and what production each copies
    //
    // - [`forward_hidden_forcing_flash`] mirrors
    //   [`ModernBert::forward_hidden_with_lengths`]'s body, with the
    //   flash-cascade decision forced rather than derived fresh per layer.
    // - [`forward_hidden_flash_with_fault`] mirrors
    //   [`ModernBertAttention::forward_flash_dense_attention`] (called per
    //   layer, replacing `ModernBertLayer::forward`'s whole body), calling
    //   the SAME production op ([`jammi_kernels::ops::
    //   flash_attention_varlen_with_rope`]) production calls, with one of
    //   [`FlashFault`] optionally injected at the `qkv`/`cfg` boundary.
    //   [`FlashFault::NoFault`] performs NO injection at all -- it is
    //   PROVEN bit-identical to calling [`forward_hidden_forcing_flash`]
    //   (force_decline=false, i.e. the real cascade) by
    //   `flash_arm_fault_harness_nofault_matches_production_bit_identical`
    //   below, so drift between this hand-synced mirror and the real
    //   `ModernBertLayer::forward`/`decide_flash_admission` dispatch chain
    //   cannot go unnoticed.
    // =====================================================================

    /// Eight fixed seeds, reused IDENTICALLY across the healthy oracle and
    /// every RED control below, at every shape (defect 2 above) -- a
    /// single seed's draw is not a distribution.
    #[cfg(feature = "cuda")]
    const FLASH_ORACLE_SWEEP_SEEDS: [u64; 8] = [201, 202, 203, 204, 205, 206, 207, 208];

    /// Mean-ratio bound, pooled-embedding leg (`err(other,f32) /
    /// err(block,f32)`, [`relative_l1_error`], averaged over
    /// [`FLASH_ORACLE_SWEEP_SEEDS`]). Measured healthy (this round,
    /// `perf/p6-fa2-dense` @ `0f1a31a`, pod `a100c`, full per-seed table in
    /// the committed `2026-08-25-flash-arm-encoder-oracle-*.json`
    /// artifact): mean ratio = 1.0798 (b8_s512) / 1.0250 (b1_s128), i.e.
    /// flash is on average ~4-8% FURTHER from the f32 reference than the
    /// block arm is on THIS leg -- not closer (a stale earlier revision of
    /// this section's block comment claimed the opposite; deleted, see
    /// defect 3 above). `1.6` gives ~1.5x margin over the worse of those
    /// two means while sitting more than 3x below the WEAKEST measured
    /// mutant mean on this same leg (window-dropped, 5.2189; K-unrotated
    /// 9.9498; bad-softmax-scale 19.4144).
    #[cfg(feature = "cuda")]
    const FLASH_ORACLE_K_MEAN_POOLED: f64 = 1.6;

    /// Max-per-seed-ratio bound, pooled-embedding leg. Measured healthy
    /// per-seed maximum this round: 1.8675 (b8_s512, seed 206) / 1.5193
    /// (b1_s128, seed 206) -- see the artifact cited on
    /// [`FLASH_ORACLE_K_MEAN_POOLED`]. `2.3` gives ~1.23x margin over the
    /// worse of those two, while sitting below the WEAKEST individual
    /// mutant seed measured this round (window-dropped seed 202, 2.9691).
    #[cfg(feature = "cuda")]
    const FLASH_ORACLE_K_MAX_POOLED: f64 = 2.3;

    /// Mean-ratio bound, LoRA-gradient leg (last layer `Wqkv` LoRA `B`,
    /// step-0, fixed-cotangent loss -- defect 1 above -- [`cosine_distance`]
    /// ratio, NOT [`relative_l1_error`] -- see that function's own doc for
    /// why). Measured healthy mean this round: 1.4604 (b8_s512) / 1.0412
    /// (b1_s128) -- see the artifact cited on [`FLASH_ORACLE_K_MEAN_POOLED`].
    /// `4.5` gives ~3.1x margin over the worse of those two while sitting
    /// ~3x below the WEAKEST measured mutant mean on this leg
    /// (window-dropped, 13.6483; K-unrotated 26.6004; bad-softmax-scale
    /// 125.3883).
    #[cfg(feature = "cuda")]
    const FLASH_ORACLE_K_MEAN_GRAD: f64 = 4.5;

    /// Max-per-seed-ratio bound, LoRA-gradient leg. Measured healthy
    /// per-seed maximum this round: 4.8987 (b8_s512, seed 206) / 1.5425
    /// (b1_s128, seed 204). `7.0` gives ~1.43x margin over the worse of
    /// those two.
    #[cfg(feature = "cuda")]
    const FLASH_ORACLE_K_MAX_GRAD: f64 = 7.0;

    /// Panics instead of skipping when `JAMMI_REQUIRE_FLASH_ORACLE` is set
    /// -- mirrors [`growth_oracle_cuda_device`]'s own `JAMMI_REQUIRE_CUDA`
    /// gate: a machine that should be running this real-checkpoint-gated
    /// suite (the pod lane) must not silently read a missing
    /// `JAMMI_FLASH_ORACLE_MODEL_DIR` as green.
    #[cfg(feature = "cuda")]
    fn flash_oracle_require_gate(test_name: &str) {
        if std::env::var_os("JAMMI_REQUIRE_FLASH_ORACLE").is_some() {
            panic!(
                "{test_name}: JAMMI_REQUIRE_FLASH_ORACLE is set but \
                 JAMMI_FLASH_ORACLE_MODEL_DIR is not -- this lane must run the real-checkpoint \
                 flash-arm oracle, not skip it"
            );
        }
        eprintln!("{test_name}: skipping — JAMMI_FLASH_ORACLE_MODEL_DIR not set");
    }

    /// A deterministic (SplitMix64-derived) token-id batch, `vocab`-bounded
    /// and `seed`-keyed -- every arm below is driven by the exact SAME
    /// `input_ids` for a given `(batch, seq, seed)`.
    #[cfg(feature = "cuda")]
    fn flash_oracle_synthetic_ids(
        batch: usize,
        seq: usize,
        vocab: usize,
        seed: u64,
        device: &Device,
    ) -> Tensor {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut ids = Vec::with_capacity(batch * seq);
        for _ in 0..batch * seq {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ids.push((z % vocab as u64) as u32);
        }
        Tensor::from_vec(ids, (batch, seq), device).expect("build synthetic token-id batch")
    }

    /// A SplitMix64-derived, `seed`-keyed cotangent for the pooled
    /// embedding (`(batch, hidden)`, F32) -- deliberately NON-uniform (an
    /// all-equal `dy` can make a downstream softmax/normalize gradient
    /// identically zero, exactly the degenerate shape this section's own
    /// vacuous-loss defect took) and driven by a stream XOR'd with a
    /// DISTINCT odd constant from [`flash_oracle_synthetic_ids`]'s own, so
    /// the token-id draw and the cotangent draw never correlate at the
    /// same seed. Values in `[-1, 1)`.
    #[cfg(feature = "cuda")]
    fn flash_oracle_seeded_dy(batch: usize, hidden: usize, seed: u64, device: &Device) -> Tensor {
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        let mut values = Vec::with_capacity(batch * hidden);
        for _ in 0..batch * hidden {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 40) as u32 as f32) / (1u32 << 24) as f32; // [0, 1)
            values.push(u * 2.0 - 1.0);
        }
        Tensor::from_vec(values, (batch, hidden), device).expect("build seeded cotangent")
    }

    /// Builds a real ModernBERT-large checkpoint with a Gaussian-initialised
    /// (non-identity from step 0 -- unlike the default `ZerosB`, whose `dA`
    /// is trivially zero regardless of any arm's numerics) LoRA adapter on
    /// `Wqkv` only, at the given backbone `dtype`. `training` selects
    /// whether `forward_hidden` reaches the admission cascade at all
    /// (`true`) or takes the always-eager eval composition (`false` -- the
    /// F32 reference's own arm).
    #[cfg(feature = "cuda")]
    fn flash_oracle_build_model(
        config: &ModernBertConfig,
        weights: &std::path::Path,
        dtype: DType,
        seed: u64,
        device: &Device,
        training: bool,
    ) -> ModernBert {
        let varmap = VarMap::new();
        let target_modules = ["Wqkv".to_string()];
        let rank_pattern: HashMap<String, usize> = HashMap::new();
        let lora = LoraBuildConfig {
            target_modules: &target_modules,
            layers_to_transform: &None,
            lora_rank: 16,
            lora_alpha: 32.0,
            use_rslora: false,
            lora_dropout: None,
            rank_pattern: &rank_pattern,
            init_mode: jammi_lora::LoraInitMode::Gaussian,
            seed,
        };
        let mut model = ModernBert::builder()
            .pooling(Pooling::Mean)
            .backbone_dtype(dtype)
            .lora(lora)
            .build(&[weights], config, device, &varmap)
            .unwrap_or_else(|e| panic!("flash oracle: build ModernBert ({dtype:?}) failed: {e}"));
        model.set_training(training);
        model
    }

    /// Test-only mirror of [`ModernBert::forward_hidden_with_lengths`]'s
    /// body, with the flash-cascade decision **forced** rather than derived
    /// fresh per layer from `decide_flash_admission` -- lets ONE process
    /// compute the flash arm AND the block arm from the SAME loaded
    /// weights, without touching `JAMMI_KERNELS_DISABLE` (a process-wide
    /// `OnceLock` -- see `jammi_kernels::admission`'s module doc; setting
    /// it mid-test would not un-set for a later call in the SAME process).
    /// `force_decline == true` passes `declined_flash()` to every layer
    /// regardless of what the real admission would decide (the block-bf16
    /// arm); `force_decline == false` runs the REAL `decide_flash_admission`
    /// (the flash-bf16 arm) -- production's exact `forward_hidden_with_lengths`
    /// body, kept in sync by hand.
    #[cfg(feature = "cuda")]
    fn forward_hidden_forcing_flash(
        model: &ModernBert,
        input_ids: &Tensor,
        mask: &Tensor,
        force_decline: bool,
    ) -> Result<Tensor, EncoderError> {
        let (_batch, seq) = input_ids.dims2()?;
        let word_emb = model.word_embeddings.forward(input_ids)?;
        let mut hidden = model.emb_norm.forward(&word_emb)?;
        let extended = extended_attention_mask(mask)?;
        let local_band = match model.local_half_window {
            None => None,
            Some(half) => Some(model.sliding_band(seq, half, input_ids.device())?),
        };
        let fused_masks = if model.training {
            Some(FusedAttentionMasks::build(
                &extended,
                local_band.as_ref(),
                hidden.dtype(),
            )?)
        } else {
            None
        };
        let flash_admission = if model.training {
            if force_decline {
                Some(declined_flash())
            } else {
                let head_dim = model
                    .layers
                    .first()
                    .map(|l| l.attention.head_dim)
                    .unwrap_or(0);
                Some(decide_flash_admission(
                    input_ids.device(),
                    hidden.dtype(),
                    head_dim,
                    mask,
                    None,
                )?)
            }
        } else {
            None
        };
        for layer in &model.layers {
            hidden = layer.forward(
                &hidden,
                &extended,
                local_band.as_ref(),
                fused_masks.as_ref(),
                flash_admission.as_ref(),
            )?;
        }
        model.final_norm.forward(&hidden)
    }

    /// The two encoder-level flash wiring faults this oracle proves it
    /// catches (see the block comment above), plus `NoFault` -- the
    /// bit-identity anchor proving this whole harness has not drifted from
    /// production (see `flash_arm_fault_harness_nofault_matches_production_bit_identical`
    /// below).
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    enum FlashFault {
        /// No injection at all -- exactly production's
        /// `forward_flash_dense_attention` composition.
        NoFault,
        /// K (slot 1) never rotated -- the observable effect of a
        /// `slot == 2` -> `slot >= 1` kernel mutant, injected without
        /// touching the kernel (see the block comment above).
        KUnrotated,
        /// `VarlenConfig::softmax_scale` replaced with a wrong constant.
        BadSoftmaxScale(f32),
    }

    /// Test-only mirror of [`ModernBertAttention::forward_flash_dense_attention`]
    /// (called per layer, replacing `ModernBertLayer::forward`'s whole body),
    /// calling the SAME production op
    /// ([`jammi_kernels::ops::flash_attention_varlen_with_rope`], i.e.
    /// `FlashVarlenAttentionFusedRope`) that function calls, with one of
    /// [`FlashFault`] injected at the `qkv`/`cfg` BOUNDARY the op itself is
    /// handed -- never inside the op. Requires the batch to be genuinely
    /// flash-`Holds`-eligible (asserted up front) -- this harness does not
    /// implement the block-arm fallback, since its whole point is to
    /// characterize the flash arm's OWN fault surface.
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    fn forward_hidden_flash_with_fault(
        model: &ModernBert,
        input_ids: &Tensor,
        mask: &Tensor,
        fault: &FlashFault,
    ) -> Result<Tensor, EncoderError> {
        use jammi_kernels::flash::{CuSeqlens, VarlenConfig};
        use jammi_kernels::ops::{flash_attention_varlen_with_rope, RopePositionsFused};

        let (batch, seq) = input_ids.dims2()?;
        let device = input_ids.device();
        let word_emb = model.word_embeddings.forward(input_ids)?;
        let mut hidden = model.emb_norm.forward(&word_emb)?;

        let head_dim = model
            .layers
            .first()
            .map(|l| l.attention.head_dim)
            .unwrap_or(0);
        let decision = decide_flash_admission(device, hidden.dtype(), head_dim, mask, None)?;
        let admission = match &decision {
            FlashDecision::Fused(batch) => batch,
            FlashDecision::Declined { outcome, reason } => panic!(
                "flash oracle fault harness requires flash to actually be eligible on this \
                 batch (outcome={outcome:?}, reason={reason})"
            ),
        };
        let cuda_device = match device {
            Device::Cuda(dev) => dev,
            _ => panic!("flash oracle fault harness requires a CUDA device"),
        };

        for layer in &model.layers {
            let attn = &layer.attention;
            let normed = match &attn.attn_norm {
                Some(ln) => ln.forward(&hidden)?,
                None => hidden.clone(),
            };
            let h = attn.num_heads;
            let d = attn.head_dim;
            let qkv = attn.wqkv.forward(&normed)?;
            let total = batch * seq;
            let qkv5 = qkv.reshape((total, 3, h, d))?;
            let (cos_full, sin_full) = attn.rope.cached_tables(qkv5.dtype())?;
            let cos = cos_full.narrow(2, 0, seq)?;
            let sin = sin_full.narrow(2, 0, seq)?;

            let mut softmax_scale = 1.0 / (d as f32).sqrt();
            let qkv_for_op = match fault {
                FlashFault::NoFault => qkv5.clone(),
                FlashFault::KUnrotated => {
                    // The production op rotates Q and K TOGETHER, forward,
                    // from whatever it is handed (proven correct by
                    // `fused_rope_matches_two_op_composition`). Pre-apply
                    // the INVERSE rotation (`negate_sin: true`, the SAME
                    // mechanism that op's own `bwd` un-rotation uses) to
                    // K ONLY: the op's own forward rotation then cancels
                    // it exactly, leaving K exactly as it was BEFORE any
                    // rotation -- observably identical to a
                    // `slot == 2` -> `slot >= 1` kernel mutant, without
                    // editing the op or its kernel.
                    let q_orig = qkv5.narrow(1, 0, 1)?;
                    let v_orig = qkv5.narrow(1, 2, 1)?;
                    let inv = apply3(&qkv5, &cos, &sin, RopePositionsFused::new(seq, true))?;
                    let k_inv = inv.narrow(1, 1, 1)?;
                    Tensor::cat(&[&q_orig, &k_inv, &v_orig], 1)?.contiguous()?
                }
                FlashFault::BadSoftmaxScale(bad) => {
                    softmax_scale = *bad;
                    qkv5.clone()
                }
            };

            let cu_seqlens = CuSeqlens::from_lengths(&admission.lengths, cuda_device)
                .map_err(|e| EncoderError::Config(format!("flash oracle fault: {e}")))?;
            let cfg = VarlenConfig {
                softmax_scale,
                window: attn.half_window.map(|w| w as u32),
                deterministic: true,
            };
            let o =
                flash_attention_varlen_with_rope(&qkv_for_op, &cos, &sin, seq, &cu_seqlens, &cfg)
                    .map_err(|e| EncoderError::Config(format!("flash oracle fault: {e}")))?;
            let ctx = o.reshape((batch, seq, h * d))?;
            let out = attn.wo.forward(&ctx)?;
            hidden = (out + &hidden)?;
            hidden = layer.mlp.forward(&hidden)?;
        }
        model.final_norm.forward(&hidden)
    }

    /// The LAST layer's `attention.wqkv`'s LoRA `B` matrix -- `target_modules
    /// = ["Wqkv"]` with `layers_to_transform: None` guarantees EVERY layer
    /// (this one included) is the `Lora` variant. Deliberately the LAST
    /// layer, not layer 0: measured this round (`perf/p6-fa2-dense` @
    /// `0f1a31a`, pod `a100c`), layer 0's gradient is 28 backward matmuls
    /// removed from the loss, and guide §3.2's own "compounding is
    /// invisible at one call, grows with depth" phenomenon applies to
    /// ORDINARY bf16 rounding noise exactly as it does to a real defect --
    /// by layer 0 the block arm's OWN gradient (known-correct, extensively
    /// tested elsewhere) already has cosine distance from the f32
    /// reference ranging 0.23-1.04 across seeds (i.e. sometimes NEAR
    /// ORTHOGONAL to truth), a noise floor comparable to or larger than
    /// the K-unrotated/window-dropped mutants' OWN signal at that same
    /// depth -- neither `relative_l1_error` nor `cosine_distance` can
    /// discriminate a real fault from ordinary depth-compounded rounding
    /// noise at that distance. The LAST layer is one backward matmul
    /// (plus the final norm) from the loss, so its gradient carries
    /// minimal ACCUMULATED noise while still proving gradients reach a
    /// LoRA-wrapped parameter (the property this leg exists to check --
    /// nothing about that property is specific to layer 0).
    #[cfg(feature = "cuda")]
    fn flash_oracle_wqkv_lora_b(model: &ModernBert) -> &Tensor {
        let last = model
            .layers
            .last()
            .expect("flash oracle: model must have at least one layer");
        match &last.attention.wqkv {
            MaybeLoraLinear::Lora(l) => &l.lora_b,
            MaybeLoraLinear::Frozen(_) => panic!(
                "flash oracle: the last layer's Wqkv must be LoRA-wrapped -- \
                 target_modules=[\"Wqkv\"]"
            ),
        }
    }

    /// `L = (pool_and_normalize(hidden) * dy).sum()` for a FIXED, seed-keyed
    /// random cotangent `dy` (family L: a generic, consumer-free scalar) --
    /// this oracle's job is comparing arms' NUMERICS, not reproducing the
    /// production triplet-hinge objective, and NEVER `sum(pooled^2)`
    /// (identically `batch`, gradient identically zero -- see this
    /// section's own block comment, defect 1). Returns `(pooled embedding,
    /// dL/d(last layer Wqkv LoRA B))`, both `F32`, flattened.
    #[cfg(feature = "cuda")]
    fn flash_oracle_pooled_and_grad(
        model: &ModernBert,
        hidden: &Tensor,
        mask: &Tensor,
        dy: &Tensor,
    ) -> (Vec<f32>, Vec<f32>) {
        let pooled = pool_and_normalize(hidden, mask, Pooling::Mean).unwrap();
        let pooled_f32 = pooled.to_dtype(DType::F32).unwrap();
        let pooled_v: Vec<f32> = pooled_f32.flatten_all().unwrap().to_vec1().unwrap();
        let loss = (&pooled_f32 * dy).unwrap().sum_all().unwrap();
        assert!(
            loss.to_scalar::<f32>().unwrap().is_finite(),
            "flash oracle: loss must be finite before backward"
        );
        let grads = loss.backward().unwrap();
        let lora_b = flash_oracle_wqkv_lora_b(model);
        let grad_v: Vec<f32> = grads
            .get(lora_b)
            .expect("flash oracle: last layer Wqkv lora_b must have a gradient")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        (pooled_v, grad_v)
    }

    /// `Σ|arm - reference| / Σ|reference|` -- guide §3.2's own aggregate
    /// shape (`r(L) = Σ|fused-eager| / Σ|eager|`), NEVER a per-element
    /// ratio (a reference element near zero would make a per-element ratio
    /// blow up on ordinary rounding noise alone). Affirmative non-finite
    /// check FIRST (guide §3.7: never let a NaN read as a silent pass), no
    /// absolute floor (guide §3.8), and asserts the denominator carries
    /// real SIGNAL (`sum|reference| > 0`) before dividing -- the exact
    /// check that would have caught this section's own vacuous-loss defect
    /// (defect 1 above) the moment it shipped.
    #[cfg(feature = "cuda")]
    fn relative_l1_error(arm: &[f32], reference: &[f32]) -> f64 {
        assert_eq!(
            arm.len(),
            reference.len(),
            "relative_l1_error: length mismatch"
        );
        let non_finite = arm
            .iter()
            .chain(reference.iter())
            .filter(|v| !v.is_finite())
            .count();
        assert_eq!(
            non_finite, 0,
            "relative_l1_error: {non_finite} non-finite value(s)"
        );
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&a, &r) in arm.iter().zip(reference.iter()) {
            num += (a as f64 - r as f64).abs();
            den += (r as f64).abs();
        }
        assert!(
            den > 0.0,
            "relative_l1_error: reference carries no signal (sum|reference| == 0) -- dividing by \
             this would silently read as a passing ratio; the caller's loss/objective is \
             degenerate for this arm"
        );
        num / den
    }

    /// `1 - cosine_similarity(arm, reference)`, bounded `[0, 2]` (`0` =
    /// same direction, `1` = orthogonal, `2` = opposite) -- guide §3.2's
    /// "aggregate, never per-element" shape, but SCALE-INVARIANT rather
    /// than magnitude-relative. Measured this round (`perf/p6-fa2-dense`
    /// @ `0f1a31a`, pod `a100c`): the grad leg's step-0, random-`dy`-driven
    /// LoRA-B gradient has a magnitude (`Σ|grad_f32|`) that itself varies
    /// ~60x across [`FLASH_ORACLE_SWEEP_SEEDS`] (some draws of `dy` are
    /// nearly orthogonal to the local Jacobian, producing a small true
    /// gradient at that seed) -- [`relative_l1_error`]'s `Σ|Δ|/Σ|ref|`
    /// aggregate is exactly proportional to `1/Σ|ref|`, so at those
    /// low-magnitude seeds ORDINARY bf16 rounding noise alone produces a
    /// huge ratio, drowning out any real fault signal in the MEAN over
    /// seeds (the K-unrotated and window-dropped mutants' mean
    /// `relative_l1_error` ratio measured BELOW the healthy oracle's own
    /// mean under that metric). Cosine distance does not have this failure
    /// mode: it normalises by each vector's OWN norm, so a small true
    /// gradient does not inflate the metric -- it asks "does the fused
    /// arm's gradient point the same way as truth", which is what actually
    /// matters for a LoRA training step, and is what the grad leg uses
    /// below. Same affirmative-finite-first (guide §3.7) and signal-assert
    /// (both norms `> 0`) discipline as [`relative_l1_error`].
    #[cfg(feature = "cuda")]
    fn cosine_distance(arm: &[f32], reference: &[f32]) -> f64 {
        assert_eq!(
            arm.len(),
            reference.len(),
            "cosine_distance: length mismatch"
        );
        let non_finite = arm
            .iter()
            .chain(reference.iter())
            .filter(|v| !v.is_finite())
            .count();
        assert_eq!(
            non_finite, 0,
            "cosine_distance: {non_finite} non-finite value(s)"
        );
        let mut dot = 0.0f64;
        let mut norm_arm = 0.0f64;
        let mut norm_ref = 0.0f64;
        for (&a, &r) in arm.iter().zip(reference.iter()) {
            let a = a as f64;
            let r = r as f64;
            dot += a * r;
            norm_arm += a * a;
            norm_ref += r * r;
        }
        let denom = norm_arm.sqrt() * norm_ref.sqrt();
        assert!(
            denom > 0.0,
            "cosine_distance: arm or reference carries no signal (norm == 0) -- dividing by this \
             would silently read as a passing distance; the caller's loss/objective is degenerate \
             for this arm"
        );
        let cos = (dot / denom).clamp(-1.0, 1.0);
        1.0 - cos
    }

    /// Deterministic mean/max over a non-empty `f64` slice -- affirmative
    /// finiteness check first (guide §3.7) and a `total_cmp` fold (family
    /// J: float `max`/`min` combinators are NaN-blind -- `f64::max(NaN, x)
    /// == x`, silently dropping the NaN rather than failing).
    #[cfg(feature = "cuda")]
    fn mean_max(values: &[f64]) -> (f64, f64) {
        assert!(!values.is_empty(), "mean_max: empty slice");
        let non_finite = values.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(non_finite, 0, "mean_max: {non_finite} non-finite value(s)");
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;
        let max =
            values.iter().copied().fold(
                values[0],
                |a, b| if b.total_cmp(&a).is_gt() { b } else { a },
            );
        (mean, max)
    }

    /// One seed's four leg errors (both arms compared against the SAME f32
    /// reference at that seed) -- the shared unit the healthy oracle and
    /// every RED control below reduce over. Pooled uses
    /// [`relative_l1_error`]; grad uses [`cosine_distance`] (see that
    /// function's own doc for why the grad leg needs a scale-invariant
    /// metric).
    #[cfg(feature = "cuda")]
    #[derive(Clone, Copy, Debug)]
    struct FlashOracleSeedMeasurement {
        seed: u64,
        pooled_other: f64,
        pooled_block: f64,
        grad_other: f64,
        grad_block: f64,
    }

    #[cfg(feature = "cuda")]
    impl FlashOracleSeedMeasurement {
        fn pooled_ratio(&self) -> f64 {
            self.pooled_other / self.pooled_block
        }
        fn grad_ratio(&self) -> f64 {
            self.grad_other / self.grad_block
        }
    }

    /// Builds a model, runs `forward`, and reduces to `(pooled, grad)` via
    /// [`flash_oracle_pooled_and_grad`] -- the ONE per-arm measurement
    /// primitive every call site below shares (a fresh `VarMap` per call,
    /// same precedent as this section's own original `run_flash_oracle_shape`:
    /// production-scale ModernBERT-large (28 layers, hidden=1024) at
    /// forward+backward is real training-step memory, and holding more
    /// than one arm's graph alive at once OOM'd on an 80GB A100, confirmed
    /// live).
    #[cfg(feature = "cuda")]
    fn flash_oracle_measure_arm<B, F>(
        build: B,
        forward: F,
        mask: &Tensor,
        dy: &Tensor,
    ) -> (Vec<f32>, Vec<f32>)
    where
        B: FnOnce() -> ModernBert,
        F: FnOnce(&ModernBert) -> Result<Tensor, EncoderError>,
    {
        let model = build();
        let hidden = forward(&model).unwrap();
        flash_oracle_pooled_and_grad(&model, &hidden, mask, dy)
    }

    /// Sweeps `seeds`, measuring the caller-supplied "other" arm
    /// (`other_build`/`other_forward` -- the flash arm for the healthy
    /// oracle, or a fault-injected arm for a RED control) against the
    /// production block-bf16 arm and an f32 reference, ALL from the SAME
    /// `input_ids`/`dy` at each seed. Returns one
    /// [`FlashOracleSeedMeasurement`] per seed, printed as it goes
    /// (`--nocapture`) so the full per-seed table is always visible, not
    /// just the reduced statistic.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn flash_oracle_sweep<BO, FO>(
        config: &ModernBertConfig,
        weights: &std::path::Path,
        cuda: &Device,
        batch: usize,
        seq: usize,
        seeds: &[u64],
        label: &str,
        other_build: BO,
        other_forward: FO,
    ) -> Vec<FlashOracleSeedMeasurement>
    where
        BO: Fn(u64) -> ModernBert,
        FO: Fn(&ModernBert, &Tensor, &Tensor) -> Result<Tensor, EncoderError>,
    {
        let mut out = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            let ids = flash_oracle_synthetic_ids(batch, seq, config.vocab_size, seed, cuda);
            let mask = Tensor::ones((batch, seq), DType::U32, cuda).unwrap();
            let dy = flash_oracle_seeded_dy(batch, config.hidden_size, seed, cuda);

            let (pooled_other, grad_other) = flash_oracle_measure_arm(
                || other_build(seed),
                |m| other_forward(m, &ids, &mask),
                &mask,
                &dy,
            );
            let (pooled_block, grad_block) = flash_oracle_measure_arm(
                || flash_oracle_build_model(config, weights, DType::BF16, seed, cuda, true),
                |m| forward_hidden_forcing_flash(m, &ids, &mask, true),
                &mask,
                &dy,
            );
            let (pooled_f32, grad_f32) = flash_oracle_measure_arm(
                || flash_oracle_build_model(config, weights, DType::F32, seed, cuda, false),
                |m| m.forward_hidden(&ids, &mask),
                &mask,
                &dy,
            );

            let m = FlashOracleSeedMeasurement {
                seed,
                pooled_other: relative_l1_error(&pooled_other, &pooled_f32),
                pooled_block: relative_l1_error(&pooled_block, &pooled_f32),
                grad_other: cosine_distance(&grad_other, &grad_f32),
                grad_block: cosine_distance(&grad_block, &grad_f32),
            };
            eprintln!(
                "flash_oracle_sweep[{label} seed={seed}]: pooled other={:.5e} block={:.5e} \
                 ratio={:.4}; grad other={:.5e} block={:.5e} ratio={:.4}",
                m.pooled_other,
                m.pooled_block,
                m.pooled_ratio(),
                m.grad_other,
                m.grad_block,
                m.grad_ratio(),
            );
            out.push(m);
        }
        out
    }

    /// Prints the full per-seed `(pooled_ratio, grad_ratio)` table -- the
    /// artifact-grade record every `--nocapture` caller below relies on
    /// (this round's own committed `2026-08-25-flash-arm-encoder-oracle-*.json`
    /// artifact is built FROM this output), not just the reduced mean/max
    /// statistic.
    #[cfg(feature = "cuda")]
    fn print_seed_ratio_table(label: &str, measurements: &[FlashOracleSeedMeasurement]) {
        for m in measurements {
            eprintln!(
                "  seed={:>3} [{label}]: pooled_ratio={:.4} grad_ratio={:.4}",
                m.seed,
                m.pooled_ratio(),
                m.grad_ratio(),
            );
        }
    }

    /// The main oracle: flash-bf16 vs block-bf16 vs f32, on the pooled
    /// embedding AND the step-0 `dL/dWqkv-LoRA` gradient, over
    /// [`FLASH_ORACLE_SWEEP_SEEDS`], for ONE shape. Prints the per-seed
    /// table and asserts MEAN and MAX ratios against
    /// [`FLASH_ORACLE_K_MEAN_POOLED`]/[`FLASH_ORACLE_K_MAX_POOLED`]/
    /// [`FLASH_ORACLE_K_MEAN_GRAD`]/[`FLASH_ORACLE_K_MAX_GRAD`].
    #[cfg(feature = "cuda")]
    fn run_flash_oracle_shape_sweep(
        config: &ModernBertConfig,
        weights: &std::path::Path,
        cuda: &Device,
        batch: usize,
        seq: usize,
        label: &str,
    ) {
        let measurements = flash_oracle_sweep(
            config,
            weights,
            cuda,
            batch,
            seq,
            &FLASH_ORACLE_SWEEP_SEEDS,
            label,
            |seed| flash_oracle_build_model(config, weights, DType::BF16, seed, cuda, true),
            |m, ids, mask| {
                let before = cascade_counters_for("attention_block_flash").snapshot();
                let hidden = forward_hidden_forcing_flash(m, ids, mask, false)?;
                let after = cascade_counters_for("attention_block_flash").snapshot();
                assert_eq!(
                    after.fused - before.fused,
                    config.num_hidden_layers as u64,
                    "[{label}] flash arm: zero dispatch is RED (guide §3.5) -- every layer must \
                     have actually dispatched Fused on this dense batch"
                );
                Ok(hidden)
            },
        );

        print_seed_ratio_table(label, &measurements);
        let pooled_ratios: Vec<f64> = measurements.iter().map(|m| m.pooled_ratio()).collect();
        let grad_ratios: Vec<f64> = measurements.iter().map(|m| m.grad_ratio()).collect();
        let (pooled_mean, pooled_max) = mean_max(&pooled_ratios);
        let (grad_mean, grad_max) = mean_max(&grad_ratios);

        eprintln!(
            "flash_oracle[{label}] OVER {} SEEDS: pooled ratio mean={pooled_mean:.4} \
             max={pooled_max:.4} (bounds mean<={FLASH_ORACLE_K_MEAN_POOLED} \
             max<={FLASH_ORACLE_K_MAX_POOLED}); grad ratio mean={grad_mean:.4} max={grad_max:.4} \
             (bounds mean<={FLASH_ORACLE_K_MEAN_GRAD} max<={FLASH_ORACLE_K_MAX_GRAD})",
            FLASH_ORACLE_SWEEP_SEEDS.len(),
        );

        assert!(
            pooled_mean.is_finite() && pooled_mean <= FLASH_ORACLE_K_MEAN_POOLED,
            "[{label}] pooled embedding: mean ratio {pooled_mean:.4} over {} seeds exceeds \
             FLASH_ORACLE_K_MEAN_POOLED={FLASH_ORACLE_K_MEAN_POOLED}",
            FLASH_ORACLE_SWEEP_SEEDS.len(),
        );
        assert!(
            pooled_max.is_finite() && pooled_max <= FLASH_ORACLE_K_MAX_POOLED,
            "[{label}] pooled embedding: max per-seed ratio {pooled_max:.4} exceeds \
             FLASH_ORACLE_K_MAX_POOLED={FLASH_ORACLE_K_MAX_POOLED}"
        );
        assert!(
            grad_mean.is_finite() && grad_mean <= FLASH_ORACLE_K_MEAN_GRAD,
            "[{label}] LoRA gradient (last layer Wqkv B): mean ratio {grad_mean:.4} over {} seeds \
             exceeds FLASH_ORACLE_K_MEAN_GRAD={FLASH_ORACLE_K_MEAN_GRAD}",
            FLASH_ORACLE_SWEEP_SEEDS.len(),
        );
        assert!(
            grad_max.is_finite() && grad_max <= FLASH_ORACLE_K_MAX_GRAD,
            "[{label}] LoRA gradient (last layer Wqkv B): max per-seed ratio {grad_max:.4} exceeds \
             FLASH_ORACLE_K_MAX_GRAD={FLASH_ORACLE_K_MAX_GRAD}"
        );
    }

    /// `cuMemGetInfo` (via `candle_core::cuda_backend::cudarc::driver::
    /// result::mem_get_info`) after a device sync -- the SAME driver-level
    /// "free bytes right now" reading `jammi-bench`'s `peak_vram_bytes`
    /// sampler polls through `nvidia-smi`, just called in-process so it can
    /// be interleaved with individual layer forwards rather than only
    /// sampled on a background thread. Returns free memory in MiB.
    #[cfg(feature = "cuda")]
    fn cuda_free_mib(device: &Device) -> f64 {
        device
            .synchronize()
            .expect("device sync before mem_get_info");
        let (free, _total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            .expect("cuMemGetInfo_v2 failed");
        free as f64 / (1024.0 * 1024.0)
    }

    /// Per-layer VRAM attribution probe (numerics write-owner round closing
    /// the flash-arm peak-VRAM BLOCK): mirrors
    /// [`forward_hidden_forcing_flash`]'s body EXACTLY (same per-forward
    /// setup, same per-layer loop, same arm-selection mechanism), with one
    /// addition -- a [`cuda_free_mib`] reading after each layer's forward,
    /// printed as a delta against the PRIOR reading. `label` names the arm
    /// in the printed table (`"flash"` / `"block"`) purely for a human
    /// reading `--nocapture` output; this function asserts nothing -- it is
    /// a diagnostic tool, not an oracle (the calling test's own dispatch
    /// count assertion is the oracle that the intended arm actually ran).
    #[cfg(feature = "cuda")]
    fn forward_hidden_forcing_flash_vram_probe(
        model: &ModernBert,
        input_ids: &Tensor,
        mask: &Tensor,
        force_decline: bool,
        label: &str,
    ) -> Tensor {
        let device = input_ids.device().clone();
        let mut prev = cuda_free_mib(&device);
        println!("[vram-probe {label}] start free={prev:.2} MiB");
        let (_batch, seq) = input_ids.dims2().unwrap();
        let word_emb = model.word_embeddings.forward(input_ids).unwrap();
        let mut hidden = model.emb_norm.forward(&word_emb).unwrap();
        let extended = extended_attention_mask(mask).unwrap();
        let local_band = model
            .local_half_window
            .map(|half| model.sliding_band(seq, half, &device).unwrap());
        let fused_masks = if model.training {
            Some(
                FusedAttentionMasks::build(&extended, local_band.as_ref(), hidden.dtype()).unwrap(),
            )
        } else {
            None
        };
        let flash_admission = if model.training {
            if force_decline {
                Some(declined_flash())
            } else {
                let head_dim = model
                    .layers
                    .first()
                    .map(|l| l.attention.head_dim)
                    .unwrap_or(0);
                Some(decide_flash_admission(&device, hidden.dtype(), head_dim, mask, None).unwrap())
            }
        } else {
            None
        };
        let now = cuda_free_mib(&device);
        println!(
            "[vram-probe {label}] after-setup free={now:.2} MiB delta={:.2} MiB",
            prev - now
        );
        prev = now;
        for (i, layer) in model.layers.iter().enumerate() {
            hidden = layer
                .forward(
                    &hidden,
                    &extended,
                    local_band.as_ref(),
                    fused_masks.as_ref(),
                    flash_admission.as_ref(),
                )
                .unwrap();
            let now = cuda_free_mib(&device);
            println!(
                "[vram-probe {label}] layer {i:02} free={now:.2} MiB delta={:.2} MiB",
                prev - now
            );
            prev = now;
        }
        let out = model.final_norm.forward(&hidden).unwrap();
        let now = cuda_free_mib(&device);
        println!(
            "[vram-probe {label}] after-final-norm free={now:.2} MiB delta={:.2} MiB",
            prev - now
        );
        out
    }

    /// Drives [`forward_hidden_forcing_flash_vram_probe`] for BOTH arms, in
    /// ONE process, from the SAME loaded checkpoint -- the method the
    /// numerics write-owner dispatch asked for: per-layer `Device` memory
    /// queries after each layer's forward, plus one more after backward,
    /// for both arms, so the per-layer delta table can be read off
    /// `--nocapture` output directly rather than reconstructed from two
    /// separate log files. Skips (does not fail) unless
    /// `JAMMI_FLASH_ORACLE_MODEL_DIR` is set, mirroring every other
    /// real-checkpoint-gated test in this file (and `JAMMI_REQUIRE_FLASH_ORACLE`
    /// promotes that skip to a panic, same gate as every sibling test
    /// below). A fresh model (fresh `VarMap`) per arm, exactly
    /// [`run_flash_oracle_shape_sweep`]'s own precedent, so one arm's
    /// retained graph cannot skew the other's baseline.
    #[test]
    #[cfg(feature = "cuda")]
    fn flash_vs_block_per_layer_vram_attribution_probe_cuda() {
        let Ok(model_dir) = std::env::var("JAMMI_FLASH_ORACLE_MODEL_DIR") else {
            flash_oracle_require_gate("flash_vs_block_per_layer_vram_attribution_probe_cuda");
            return;
        };
        let Some(cuda) = growth_oracle_cuda_device() else {
            return;
        };
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _d2h_guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !jammi_kernels::admission::FLASH_COMPILED {
            eprintln!(
                "flash_vs_block_per_layer_vram_attribution_probe_cuda: skipping — built without \
                 the flash-attn feature (FLASH_COMPILED=false); this probe needs a real flash \
                 arm to attribute against"
            );
            return;
        }

        let dir = std::path::PathBuf::from(&model_dir);
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");
        let (batch, seq, seed) = (8usize, 512usize, 42u64);
        let ids = flash_oracle_synthetic_ids(batch, seq, config.vocab_size, seed, &cuda);
        let mask = Tensor::ones((batch, seq), DType::U32, &cuda).unwrap();

        for (force_decline, label) in [(false, "flash"), (true, "block")] {
            let bf16_model =
                flash_oracle_build_model(&config, &weights, DType::BF16, seed, &cuda, true);
            let counter_before = cascade_counters_for("attention_block_flash").snapshot();
            let hidden = forward_hidden_forcing_flash_vram_probe(
                &bf16_model,
                &ids,
                &mask,
                force_decline,
                label,
            );
            let counter_after = cascade_counters_for("attention_block_flash").snapshot();
            if force_decline {
                assert_eq!(
                    counter_after.fused - counter_before.fused,
                    0,
                    "[{label}] the block arm must not have dispatched attention_block_flash"
                );
            } else {
                assert_eq!(
                    counter_after.fused - counter_before.fused,
                    config.num_hidden_layers as u64,
                    "[vram-probe {label}] zero dispatch is RED (guide §3.5) -- every layer must \
                     have actually dispatched Fused on this dense batch"
                );
            }
            let pooled = pool_and_normalize(&hidden, &mask, Pooling::Mean).unwrap();
            let loss = pooled
                .to_dtype(DType::F32)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap();
            let before_bwd = cuda_free_mib(&cuda);
            let grads = loss.backward().unwrap();
            let after_bwd = cuda_free_mib(&cuda);
            println!(
                "[vram-probe {label}] after-backward free={after_bwd:.2} MiB delta={:.2} MiB",
                before_bwd - after_bwd
            );
            drop(grads);
            drop(hidden);
            drop(bf16_model);
            let after_drop = cuda_free_mib(&cuda);
            println!("[vram-probe {label}] after-drop free={after_drop:.2} MiB");
        }
    }

    /// Skips (does not fail) unless `JAMMI_FLASH_ORACLE_MODEL_DIR` is set --
    /// no ModernBERT-large checkpoint is committed to this repo, mirroring
    /// this file's other real-checkpoint-gated tests. `JAMMI_REQUIRE_CUDA`
    /// still turns a missing CUDA device into a hard failure
    /// (`growth_oracle_cuda_device`'s own contract), and
    /// `JAMMI_REQUIRE_FLASH_ORACLE` turns a missing `JAMMI_FLASH_ORACLE_MODEL_DIR`
    /// into a hard failure too (`flash_oracle_require_gate`'s own contract).
    #[test]
    #[cfg(feature = "cuda")]
    fn flash_arm_encoder_level_three_way_oracle_dense_cuda_bf16() {
        let Ok(model_dir) = std::env::var("JAMMI_FLASH_ORACLE_MODEL_DIR") else {
            flash_oracle_require_gate("flash_arm_encoder_level_three_way_oracle_dense_cuda_bf16");
            return;
        };
        let Some(cuda) = growth_oracle_cuda_device() else {
            return;
        };
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _d2h_guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !jammi_kernels::admission::FLASH_COMPILED {
            eprintln!(
                "flash_arm_encoder_level_three_way_oracle_dense_cuda_bf16: skipping — built \
                 without the flash-attn feature (FLASH_COMPILED=false); this oracle needs a \
                 real flash arm to compare against"
            );
            return;
        }

        let dir = std::path::PathBuf::from(&model_dir);
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");

        run_flash_oracle_shape_sweep(&config, &weights, &cuda, 8, 512, "b8_s512");
        run_flash_oracle_shape_sweep(&config, &weights, &cuda, 1, 128, "b1_s128");
    }

    /// The `NoFault` arm of [`forward_hidden_flash_with_fault`] must be
    /// BIT-IDENTICAL to [`forward_hidden_forcing_flash`]'s real
    /// (`force_decline = false`) arm -- proof that the fault-injection
    /// mirror has not drifted from what `ModernBertLayer::forward` /
    /// `ModernBertAttention::forward_flash_dense_attention` actually run
    /// (see this section's own block comment, "the two hand-synced
    /// mirrors"). If this test ever goes red, every RED control below
    /// stops being trustworthy -- they all inject faults into THIS
    /// harness, not into production directly.
    #[test]
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    fn flash_arm_fault_harness_nofault_matches_production_bit_identical() {
        let Ok(model_dir) = std::env::var("JAMMI_FLASH_ORACLE_MODEL_DIR") else {
            flash_oracle_require_gate(
                "flash_arm_fault_harness_nofault_matches_production_bit_identical",
            );
            return;
        };
        let Some(cuda) = growth_oracle_cuda_device() else {
            return;
        };
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _d2h_guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !jammi_kernels::admission::FLASH_COMPILED {
            eprintln!(
                "flash_arm_fault_harness_nofault_matches_production_bit_identical: skipping — \
                 built without the flash-attn feature"
            );
            return;
        }

        let dir = std::path::PathBuf::from(&model_dir);
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");
        let seed = 99;
        let (batch, seq) = (2usize, 64usize);
        let ids = flash_oracle_synthetic_ids(batch, seq, config.vocab_size, seed, &cuda);
        let mask = Tensor::ones((batch, seq), DType::U32, &cuda).unwrap();

        let bf16_model =
            flash_oracle_build_model(&config, &weights, DType::BF16, seed, &cuda, true);
        let production: Vec<f32> = forward_hidden_forcing_flash(&bf16_model, &ids, &mask, false)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let mirror: Vec<f32> =
            forward_hidden_flash_with_fault(&bf16_model, &ids, &mask, &FlashFault::NoFault)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
        assert_eq!(
            production, mirror,
            "the fault harness's NoFault arm must be bit-identical to production's own \
             forward_hidden_forcing_flash(force_decline=false) -- any difference means this \
             hand-synced mirror has drifted from ModernBertAttention::forward_flash_dense_attention"
        );
    }

    /// RED control: the window-dropped fault (`half_window` forced `None`
    /// on every layer, see the block comment above) must VIOLATE the same
    /// bound the real oracle asserts above, on BOTH legs, in MEAN, over
    /// the SAME [`FLASH_ORACLE_SWEEP_SEEDS`].
    #[test]
    #[cfg(feature = "cuda")]
    fn flash_arm_encoder_level_oracle_red_control_window_dropped() {
        let Ok(model_dir) = std::env::var("JAMMI_FLASH_ORACLE_MODEL_DIR") else {
            flash_oracle_require_gate("flash_arm_encoder_level_oracle_red_control_window_dropped");
            return;
        };
        let Some(cuda) = growth_oracle_cuda_device() else {
            return;
        };
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _d2h_guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !jammi_kernels::admission::FLASH_COMPILED {
            eprintln!(
                "flash_arm_encoder_level_oracle_red_control_window_dropped: skipping — built \
                 without the flash-attn feature"
            );
            return;
        }

        let dir = std::path::PathBuf::from(&model_dir);
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");
        let (batch, seq) = (8usize, 512usize);
        let label = "window_dropped_b8_s512";

        let measurements = flash_oracle_sweep(
            &config,
            &weights,
            &cuda,
            batch,
            seq,
            &FLASH_ORACLE_SWEEP_SEEDS,
            label,
            |seed| {
                // FAULT: force every layer's flash-arm sliding window off --
                // the window is construction data ONLY the flash arm reads
                // (see `ModernBertAttention::half_window`'s own doc); the
                // block arm's sliding band comes from a separate field
                // (`ModernBert::local_half_window`) and is unaffected.
                let mut m =
                    flash_oracle_build_model(&config, &weights, DType::BF16, seed, &cuda, true);
                for layer in m.layers.iter_mut() {
                    layer.attention.half_window = None;
                }
                m
            },
            |m, ids, mask| forward_hidden_forcing_flash(m, ids, mask, false),
        );

        assert_red_control_violates_bound(label, &measurements);
    }

    /// RED control: the K-unrotated fault (`FlashFault::KUnrotated`, see
    /// the block comment above) must VIOLATE the same bound the real
    /// oracle asserts above, on BOTH legs, in MEAN, over the SAME
    /// [`FLASH_ORACLE_SWEEP_SEEDS`].
    #[test]
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    fn flash_arm_encoder_level_oracle_red_control_k_unrotated() {
        run_flash_arm_fault_red_control("k_unrotated_b8_s512", &FlashFault::KUnrotated);
    }

    /// RED control: a wrong `softmax_scale` (class sweep -- see the block
    /// comment above) must VIOLATE the same bound too, on BOTH legs, in
    /// MEAN, over the SAME [`FLASH_ORACLE_SWEEP_SEEDS`].
    #[test]
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    fn flash_arm_encoder_level_oracle_red_control_bad_softmax_scale() {
        run_flash_arm_fault_red_control(
            "bad_softmax_scale_b8_s512",
            &FlashFault::BadSoftmaxScale(1.0),
        );
    }

    /// Every RED control above shares this: sweep the SAME 8 seeds, assert
    /// the fault's MEAN ratio (both legs) exceeds the SAME bound the
    /// healthy oracle asserts against -- proving the oracle actually
    /// catches the fault as a DISTRIBUTION-level effect, not merely that
    /// it "looks wrong" on one lucky draw.
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    fn run_flash_arm_fault_red_control(label: &str, fault: &FlashFault) {
        let Ok(model_dir) = std::env::var("JAMMI_FLASH_ORACLE_MODEL_DIR") else {
            flash_oracle_require_gate(label);
            return;
        };
        let Some(cuda) = growth_oracle_cuda_device() else {
            return;
        };
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _d2h_guard = FLASH_D2H_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !jammi_kernels::admission::FLASH_COMPILED {
            eprintln!("{label}: skipping — built without the flash-attn feature");
            return;
        }

        let dir = std::path::PathBuf::from(&model_dir);
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");
        let (batch, seq) = (8usize, 512usize);

        let measurements = flash_oracle_sweep(
            &config,
            &weights,
            &cuda,
            batch,
            seq,
            &FLASH_ORACLE_SWEEP_SEEDS,
            label,
            |seed| flash_oracle_build_model(&config, &weights, DType::BF16, seed, &cuda, true),
            |m, ids, mask| forward_hidden_flash_with_fault(m, ids, mask, fault),
        );

        assert_red_control_violates_bound(label, &measurements);
    }

    /// Shared assertion every RED control above ends with: the fault's MEAN
    /// pooled ratio AND MEAN grad ratio, over [`FLASH_ORACLE_SWEEP_SEEDS`],
    /// must each exceed the healthy bound -- if either does not, the real
    /// oracle above would NOT have caught this defect on that leg.
    #[cfg(feature = "cuda")]
    fn assert_red_control_violates_bound(label: &str, measurements: &[FlashOracleSeedMeasurement]) {
        print_seed_ratio_table(label, measurements);
        let pooled_ratios: Vec<f64> = measurements.iter().map(|m| m.pooled_ratio()).collect();
        let grad_ratios: Vec<f64> = measurements.iter().map(|m| m.grad_ratio()).collect();
        let (pooled_mean, _pooled_max) = mean_max(&pooled_ratios);
        let (grad_mean, _grad_max) = mean_max(&grad_ratios);

        eprintln!(
            "RED control [{label}]: pooled mean ratio={pooled_mean:.4} (bound \
             {FLASH_ORACLE_K_MEAN_POOLED}); grad mean ratio={grad_mean:.4} (bound \
             {FLASH_ORACLE_K_MEAN_GRAD})"
        );

        assert!(
            pooled_mean.is_finite() && pooled_mean > FLASH_ORACLE_K_MEAN_POOLED,
            "RED control [{label}] must VIOLATE the pooled-leg bound in mean (mean ratio \
             {pooled_mean:.4} must exceed FLASH_ORACLE_K_MEAN_POOLED={FLASH_ORACLE_K_MEAN_POOLED}) \
             -- if this assertion fails, the real oracle above would NOT have caught this defect \
             on the pooled leg"
        );
        assert!(
            grad_mean.is_finite() && grad_mean > FLASH_ORACLE_K_MEAN_GRAD,
            "RED control [{label}] must VIOLATE the grad-leg bound in mean (mean ratio \
             {grad_mean:.4} must exceed FLASH_ORACLE_K_MEAN_GRAD={FLASH_ORACLE_K_MEAN_GRAD}) -- \
             if this assertion fails, the real oracle above would NOT have caught this defect on \
             the grad leg"
        );
    }

    /// Path P's encoders-side seam (`forward_hidden_with_lengths`, contract
    /// v5 item 3): `lengths: None` is byte-identical to `forward_hidden`,
    /// and — on THIS build (no CUDA / no flash-attn feature, so the flash
    /// cascade always declines at its cheap gates before `lengths` is ever
    /// consulted) — supplying (even WRONG) `lengths` changes NOTHING about
    /// the block arm's dispatch or output either, proving the seam is
    /// dormant exactly like the path-F seam is.
    #[test]
    fn forward_hidden_with_lengths_none_is_bit_identical_to_forward_hidden() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_modernbert_head64");
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let weights = dir.join("model.safetensors");
        let varmap = candle_nn::VarMap::new();
        let mut model = ModernBert::builder()
            .build(&[weights.as_path()], &config, &device, &varmap)
            .unwrap();
        let input_ids =
            Tensor::new(&[[2u32, 5, 10, 3, 7, 9], [4u32, 8, 1, 6, 0, 0]], &device).unwrap();
        let mask = Tensor::new(&[[1u32, 1, 1, 1, 1, 1], [1u32, 1, 1, 1, 0, 0]], &device).unwrap();
        model.set_training(true);

        let a: Vec<f32> = model
            .forward_hidden(&input_ids, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = model
            .forward_hidden_with_lengths(&input_ids, &mask, None)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            a, b,
            "lengths: None must be byte-identical to forward_hidden"
        );

        // Deliberately WRONG lengths (real lengths are [6, 4]) — on this
        // CPU/no-flash-attn build the flash cascade declines at its cheap
        // gates before `lengths` is ever consulted, so the output must
        // still be unaffected.
        let c: Vec<f32> = model
            .forward_hidden_with_lengths(&input_ids, &mask, Some(&[1, 1]))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            a, c,
            "wrong lengths must not change the block arm's output on this build"
        );
    }

    fn rope(head_dim: usize, max_seq: usize, theta: f64, device: &Device) -> RotaryEmbedding {
        RotaryEmbedding::new(head_dim, max_seq, theta, device).unwrap()
    }

    /// Table hoisting (this module's doc) must be BIT-NEUTRAL: the cached,
    /// pre-cast/pre-unsqueezed table produces the exact same output
    /// `apply` would have produced computing `to_dtype`/`unsqueeze` fresh
    /// every call. Compares the CACHED path's output (first call, which
    /// populates the cache) against a SECOND call (which reads the
    /// cache) — both must be byte-identical to each other AND to a
    /// from-scratch recomputation done by hand outside the cache.
    #[test]
    fn table_hoisting_is_bit_neutral_with_the_uncached_computation() {
        let device = Device::Cpu;
        let head_dim = 8;
        let max_seq = 16;
        let r = rope(head_dim, max_seq, 10_000.0, &device);
        let xv: Vec<bf16> = (0..2 * 4 * head_dim)
            .map(|i| bf16::from_f32(i as f32 * 0.13 - 3.0))
            .collect();
        let x = Tensor::from_slice(&xv, (2, 1, 4, head_dim), &device).unwrap();

        // First call: populates the cache.
        let out_first: Vec<bf16> = r
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Second call: reads the cache.
        let out_second: Vec<bf16> = r
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            out_first, out_second,
            "cached and re-cached calls must be byte-identical"
        );

        // Hand-computed "no cache at all" reference: cast/unsqueeze fresh,
        // then the exact same rotate-half composition `apply` runs.
        let seq = 4usize;
        let half = head_dim / 2;
        let cos = r
            .cos
            .narrow(0, 0, seq)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let sin = r
            .sin
            .narrow(0, 0, seq)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let x1 = x.narrow(D::Minus1, 0, half).unwrap();
        let x2 = x.narrow(D::Minus1, half, half).unwrap();
        let neg_x2 = (x2 * -1.0f64).unwrap();
        let rot_half = Tensor::cat(&[&neg_x2, &x1], D::Minus1).unwrap();
        let uncached: Vec<bf16> = (x.broadcast_mul(&cos).unwrap()
            + rot_half.broadcast_mul(&sin).unwrap())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
        assert_eq!(
            out_first, uncached,
            "the cached path must match a from-scratch to_dtype/unsqueeze computation"
        );
    }

    /// The positive half of the RoPE device clause: CPU must satisfy it.
    #[test]
    fn rope_admission_predicate_accepts_cpu_device() {
        let device = Device::Cpu;
        let cos = Tensor::from_slice(&[1.0f32; 8], (1, 1, 1, 8), &device).unwrap();
        let sin = Tensor::from_slice(&[0.0f32; 8], (1, 1, 1, 8), &device).unwrap();
        let (holds, predicate) = rope_admission_predicate(DType::F32, &device, &cos, &sin, 8);
        assert!(holds, "CPU must satisfy the device clause: {predicate}");
    }

    /// The negative half: an odd `head_dim` is refused, not silently
    /// truncated.
    #[test]
    fn rope_admission_predicate_rejects_odd_head_dim() {
        let device = Device::Cpu;
        let cos = Tensor::from_slice(&[1.0f32; 7], (1, 1, 1, 7), &device).unwrap();
        let sin = Tensor::from_slice(&[0.0f32; 7], (1, 1, 1, 7), &device).unwrap();
        let (holds, predicate) = rope_admission_predicate(DType::F32, &device, &cos, &sin, 7);
        assert!(!holds, "odd head_dim must be refused");
        assert_eq!(predicate, "head_dim_even_and_nonzero");
    }

    /// The eval-path bit-identity requirement, mirroring
    /// `crate::layer_norm`'s identical test: a `training == false` RoPE
    /// application must be UNCHANGED by `apply_training`'s existence, on
    /// a fixture that WOULD be fused-eligible if training were true —
    /// proving eval structurally never reaches `apply_training`, not
    /// merely that this fixture happens to fail admission.
    #[test]
    fn eval_mode_rope_is_bit_identical_regardless_of_fused_eligibility() {
        let device = Device::Cpu;
        let head_dim = 8;
        let seq = 4;
        let r = rope(head_dim, 16, 10_000.0, &device);
        let xv: Vec<f32> = (0..2 * seq * head_dim)
            .map(|i| (i as f32 * 0.29 - 1.1).sin())
            .collect();
        let x = Tensor::from_slice(&xv, (1, 2, seq, head_dim), &device).unwrap();

        // Non-vacuity (mirrors `crate::layer_norm`'s identical assertion):
        // prove this fixture WOULD be admitted into the fused kernel if
        // `training` were `true`, so the test below proves eval
        // structurally never reaches `apply_training` — not merely that
        // this particular fixture happens to fail admission regardless.
        let (cos_full, sin_full) = r.cached_tables(x.dtype()).unwrap();
        let cos = cos_full.narrow(2, 0, seq).unwrap();
        let sin = sin_full.narrow(2, 0, seq).unwrap();
        let (holds, predicate) =
            rope_admission_predicate(x.dtype(), x.device(), &cos, &sin, head_dim);
        assert!(
            holds,
            "fixture must satisfy the fused RoPE domain — the test proves eval \
             skips it anyway, not that the fixture happens to be ineligible: {predicate}"
        );

        let before: Vec<f32> = r
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Exercise the fused arm (available in this binary, and PROVEN
        // eligible above) without changing the eval call itself.
        let training_before = ROPE_DISPATCH_COUNTERS.snapshot();
        let _ = r.apply_training(&x).unwrap();
        let training_after = ROPE_DISPATCH_COUNTERS.snapshot();
        assert!(
            training_after.fused > training_before.fused,
            "the eligibility check above must be load-bearing: this exercise call \
             must actually dispatch the fused kernel, not silently fall back \
             (before={training_before:?}, after={training_after:?})"
        );

        let after: Vec<f32> = r
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            before, after,
            "eval-mode RoPE forward must be byte-identical before and after \
             the fused kernel exists"
        );
    }

    /// Encoder-level oracle: `apply_training`'s actual dispatch path (fused
    /// kernel, since this fixture is fused-eligible on CPU) vs. `apply`
    /// (the eager composition), fwd AND bwd.
    #[test]
    fn fused_training_rope_matches_eager_fwd_and_bwd() {
        let device = Device::Cpu;
        let head_dim = 8;
        let seq = 4;
        let r = rope(head_dim, 16, 10_000.0, &device);
        let xv: Vec<f32> = (0..2 * seq * head_dim)
            .map(|i| (i as f32 * 0.17 - 2.0).cos() * 2.0)
            .collect();

        let x_fused =
            Var::from_tensor(&Tensor::from_slice(&xv, (1, 2, seq, head_dim), &device).unwrap())
                .unwrap();
        let before = ROPE_DISPATCH_COUNTERS.snapshot();
        let out_fused = r.apply_training(x_fused.as_tensor()).unwrap();
        let after = ROPE_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "this fixture must actually dispatch the fused kernel, not fall back \
             (before={before:?}, after={after:?})"
        );

        let x_eager =
            Var::from_tensor(&Tensor::from_slice(&xv, (1, 2, seq, head_dim), &device).unwrap())
                .unwrap();
        let out_eager = r.apply(x_eager.as_tensor()).unwrap();

        let vf: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ve: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs eager {e}");
        }

        let grads_fused = out_fused.backward().unwrap();
        let grads_eager = out_eager.backward().unwrap();
        let dxf: Vec<f32> = grads_fused
            .get(&x_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dxe: Vec<f32> = grads_eager
            .get(&x_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dx[{i}]: fused {f} vs eager {e}");
        }
    }

    // ---------------------------------------------------------------------
    // Fused masked softmax (C4)
    // ---------------------------------------------------------------------

    /// The positive half of the softmax device clause: CPU must satisfy it.
    #[test]
    fn softmax_admission_predicate_accepts_cpu_device() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 4), &device).unwrap();
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 1.0);
        assert!(holds, "CPU must satisfy the device clause: {predicate}");
    }

    /// The negative half: a `mask` dtype mismatched with `scores` is
    /// refused, not silently cast or truncated.
    #[test]
    fn softmax_admission_predicate_rejects_dtype_mismatch() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[bf16::from_f32(0.0); 4], (1, 1, 1, 4), &device).unwrap();
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 1.0);
        assert!(!holds, "dtype mismatch must be refused");
        assert_eq!(
            predicate,
            "dtype_f32_or_bf16_matching_between_scores_and_mask"
        );
    }

    /// A `last` (softmax reduction axis) size beyond `MAX_LAST_DIM` is
    /// refused, matching `LayerNormFused`'s `MAX_HIDDEN` / `RopeFused`'s
    /// `MAX_HEAD_DIM` clauses.
    #[test]
    fn softmax_admission_predicate_rejects_last_dim_above_ceiling() {
        let device = Device::Cpu;
        let last = MAX_LAST_DIM + 1;
        let scores = Tensor::from_slice(&vec![0.0f32; last], (1, 1, 1, last), &device).unwrap();
        let mask = Tensor::from_slice(&vec![0.0f32; last], (1, 1, 1, last), &device).unwrap();
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 1.0);
        assert!(!holds, "last dim above MAX_LAST_DIM must be refused");
        assert_eq!(predicate, "last_dim_within_kernel_max_last_dim");
    }

    /// A rank beyond `MAX_RANK` is refused, matching the CUDA arm's
    /// fixed-arity mask-broadcast index.
    #[test]
    fn softmax_admission_predicate_rejects_rank_above_ceiling() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 1, 4), &device).unwrap();
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 1.0);
        assert!(!holds, "rank above MAX_RANK must be refused");
        assert_eq!(predicate, "rank_within_kernel_max_rank");
    }

    /// Audit finding, corrected: `scores` (not just `mask`) must also be
    /// contiguous — an earlier version of this predicate checked only
    /// `mask`, asymmetrically.
    #[test]
    fn softmax_admission_predicate_rejects_non_contiguous_scores() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 8], (2, 4), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!scores.is_contiguous());
        let mask = Tensor::from_slice(&[0.0f32; 2], (1, 2), &device).unwrap();
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 1.0);
        assert!(!holds, "non-contiguous scores must be refused");
        assert_eq!(predicate, "scores_contiguous");
    }

    /// Advisory fix: a mask shape OUTSIDE the supported broadcast class
    /// (a leading axis neither `1` nor equal to `scores`'s) must be caught
    /// HERE — a counted eager fallback — rather than only inside the op,
    /// where it would surface as a raw `candle_core::Error` on the
    /// training arm.
    #[test]
    fn softmax_admission_predicate_rejects_broadcast_class_violation() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 3 * 2 * 4], (3, 2, 4), &device).unwrap();
        // Leading axis 0 is `2`, neither `1` nor `scores`'s `3`.
        let mask = Tensor::from_slice(&[0.0f32; 2 * 2 * 4], (2, 2, 4), &device).unwrap();
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 1.0);
        assert!(!holds, "a broadcast-class violation must be refused");
        assert_eq!(predicate, "mask_broadcast_class");
    }

    /// `scale` (the divisor `softmax_apply_training` passes through, `1.0 /
    /// scale` being the value handed to `SoftmaxLastDimFused::with_scale`)
    /// has a real domain (family D) -- `0.0`, negative, `NaN`, and `+inf`
    /// all produce a `1.0 / scale` this op's own `with_scale` would refuse,
    /// and this predicate must catch EVERY one of them here, at the call
    /// site, before ever reaching `with_scale`. In `AdmissionMode::Fallback`
    /// (the default), a bad scale becomes a counted eager fallback (K2's
    /// doctrine), not a `KernelError` propagating out of the training arm
    /// -- this is scoped to `Fallback` deliberately: in
    /// `AdmissionMode::Strict`, `admit` turns the SAME failed predicate
    /// into `KernelError::StrictModeFallback`, which DOES propagate (see
    /// `jammi_kernels::admission`'s own doc and tests), by design -- Strict
    /// mode exists precisely so a failed predicate is observable as an
    /// error rather than silently degrading. `0.125` (`1/sqrt(64)`,
    /// ModernBERT-large's real head_dim) is the positive control: it must
    /// be ACCEPTED, proving this clause does not also reject the real
    /// production value.
    #[test]
    fn softmax_admission_predicate_scale_domain() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 4), &device).unwrap();

        for bad_scale in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
            let (holds, predicate) = softmax_admission_predicate(&scores, &mask, bad_scale);
            assert!(
                !holds,
                "scale={bad_scale} must be refused (1.0/scale is not finite-and-positive)"
            );
            assert_eq!(predicate, "scale_finite_positive", "scale={bad_scale}");
        }

        // Positive control: ModernBERT-large's real `sqrt(head_dim)`
        // divisor (`1.0 / 8.0 == 0.125`) must be ACCEPTED.
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 8.0);
        assert!(holds, "scale=8.0 (1/8=0.125) must be accepted: {predicate}");
    }

    /// The eval-path bit-identity requirement, mirroring
    /// `eval_mode_rope_is_bit_identical_regardless_of_fused_eligibility`
    /// and `crate::layer_norm`'s identical test: a `training == false`
    /// attention forward must be UNCHANGED by the fused softmax kernel's
    /// existence, on a fixture that WOULD be fused-eligible if training
    /// were true — proving eval structurally never reaches
    /// `softmax_apply_training`, not merely that this fixture happens to
    /// fail admission.
    #[test]
    fn eval_mode_attention_softmax_is_bit_identical_regardless_of_fused_eligibility() {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let sv: Vec<f32> = (0..batch * heads * seq * seq)
            .map(|i| (i as f32 * 0.23 - 1.0).sin() * 2.0)
            .collect();
        let mv: Vec<f32> = (0..batch * seq)
            .map(|i| if i == 1 { -10_000.0 } else { 0.0 })
            .collect();
        let scores = Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap();
        let mask = Tensor::from_slice(&mv, (batch, 1, 1, seq), &device).unwrap();

        // Non-vacuity: this fixture WOULD be admitted into the fused
        // kernel if training were true.
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask, 1.0);
        assert!(
            holds,
            "fixture must satisfy the fused softmax domain — the test proves eval \
             skips it anyway, not that the fixture happens to be ineligible: {predicate}"
        );

        let before: Vec<f32> =
            candle_nn::ops::softmax(&scores.broadcast_add(&mask).unwrap(), D::Minus1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

        // Exercise the REAL fused-or-fallback dispatch function (proven
        // eligible above) without changing eval's own composed call above.
        // `scale = 1.0` here is deliberately a no-op: this test is about
        // EVAL never touching this function's dispatch at all, not about
        // `scale`'s own numerics (covered separately below).
        let training_before = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        let _ = softmax_apply_training(&scores, &mask, 1.0).unwrap();
        let training_after = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        assert!(
            training_after.fused > training_before.fused,
            "the eligibility check above must be load-bearing: this exercise call \
             must actually dispatch the fused kernel, not silently fall back \
             (before={training_before:?}, after={training_after:?})"
        );

        let after: Vec<f32> =
            candle_nn::ops::softmax(&scores.broadcast_add(&mask).unwrap(), D::Minus1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
        assert_eq!(
            before, after,
            "eval-mode attention softmax must be byte-identical before and after \
             the fused kernel exists"
        );
    }

    /// Fused-vs-eager attention-level oracle: `softmax_apply_training`'s
    /// actual dispatch path (fused kernel, since this fixture is
    /// fused-eligible on CPU) vs. the eager `/scale` + `broadcast_add` +
    /// `candle_nn::ops::softmax` composition (the SAME composition
    /// `forward`'s training arm used to run before `scale` was folded into
    /// this function), fwd AND bwd. `scale = 8.0` (`sqrt(64)`, ModernBERT's
    /// real `head_dim`) is a GENUINE, non-`1.0` scale — this is the
    /// oracle that actually exercises the folded-scale numerics, not just
    /// the fused/eager dispatch machinery.
    #[test]
    fn fused_training_softmax_matches_eager_fwd_and_bwd() {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let scale = 8.0f64; // sqrt(64) -- ModernBERT-large's real head_dim.
        let sv: Vec<f32> = (0..batch * heads * seq * seq)
            .map(|i| (i as f32 * 0.19 - 2.0).cos() * 2.0)
            .collect();
        let mv: Vec<f32> = (0..batch * seq)
            .map(|i| if i == 2 { -10_000.0 } else { 0.0 })
            .collect();

        let s_fused =
            Var::from_tensor(&Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap())
                .unwrap();
        let mask_fused = Tensor::from_slice(&mv, (batch, 1, 1, seq), &device).unwrap();
        let before = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        let out_fused = softmax_apply_training(&s_fused, &mask_fused, scale).unwrap();
        let after = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "this fixture must actually dispatch the fused kernel, not fall back \
             (before={before:?}, after={after:?})"
        );

        let s_eager =
            Var::from_tensor(&Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap())
                .unwrap();
        let mask_eager = Tensor::from_slice(&mv, (batch, 1, 1, seq), &device).unwrap();
        let out_eager = candle_nn::ops::softmax(
            &(s_eager.as_tensor() / scale)
                .unwrap()
                .broadcast_add(&mask_eager)
                .unwrap(),
            D::Minus1,
        )
        .unwrap();

        let vf: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ve: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vf.iter().any(|v| v.abs() > 1e-6),
            "fixture must be non-degenerate"
        );
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs eager {e}");
        }

        // A NON-UNIFORM seed, not `Tensor::backward()`'s implicit
        // all-ones `dy`: a uniform `dy` makes `dscores = (dy -
        // sum(dy*y))*y` IDENTICALLY zero for every softmax row (since
        // `sum(y) == 1`), which would make a bare `.backward()` here a
        // VACUOUS backward check (family F) — it passed before this
        // fixture had a non-vacuity assertion, silently proving nothing
        // about `bwd`'s actual scale-chain-rule multiply.
        let dy_seed_v: Vec<f32> = (0..batch * heads * seq * seq)
            .map(|i| (i as f32 * 0.31 - 1.0).sin())
            .collect();
        let dy_seed_fused =
            Tensor::from_slice(&dy_seed_v, (batch, heads, seq, seq), &device).unwrap();
        let dy_seed_eager =
            Tensor::from_slice(&dy_seed_v, (batch, heads, seq, seq), &device).unwrap();
        let loss_fused = (&out_fused * &dy_seed_fused).unwrap().sum_all().unwrap();
        let loss_eager = (&out_eager * &dy_seed_eager).unwrap().sum_all().unwrap();
        let grads_fused = loss_fused.backward().unwrap();
        let grads_eager = loss_eager.backward().unwrap();
        let dxf: Vec<f32> = grads_fused
            .get(&s_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dxe: Vec<f32> = grads_eager
            .get(&s_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            dxf.iter().any(|v| v.abs() > 1e-6),
            "gradient must be measured-nonzero"
        );
        for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dscores[{i}]: fused {f} vs eager {e}");
        }
    }

    /// Deletion-catching oracle for `softmax_apply_training`'s EAGER
    /// FALLBACK arm -- the branch every OTHER test in this file skips
    /// over, because every other fixture here is fused-eligible (three of
    /// them assert the fused counter incremented: the two tests above and
    /// `fused_training_softmax_call_site_drops_the_affine_node` below).
    /// MUTATION-VERIFIED: replacing the eager arm's `scores /
    /// scores_divisor` with `scores * scores_divisor` leaves every one of
    /// those fused-path tests, and every other encoder test, green --
    /// none of them ever reaches this branch -- while this test reddens
    /// (see the commit message that introduced this test for the actual
    /// mutate/run/revert record). This is what makes "the
    /// fallback is never a fourth numeric path, only the pre-existing
    /// training-eager composition" (`softmax_apply_training`'s own doc
    /// comment above) a MEASURED claim rather than an assertion no test
    /// actually exercises.
    ///
    /// Forces the eager arm via an admissible-by-construction domain miss
    /// -- `scores` non-contiguous (`.t()` on a `Var`, transposing the
    /// last two axes) -- rather than an env-var flip
    /// (`JAMMI_KERNELS_STRICT`), which would mutate shared process state
    /// a parallel test run could race on. `mask_broadcast_class`, dtype,
    /// rank, and scale all still hold for this fixture, so
    /// `scores_contiguous` is the ONLY predicate clause it fails --
    /// confirmed below, not assumed.
    ///
    /// Production width, not the tiny `seq = 4` toy fixtures the tests
    /// above use: `heads = 16`, `seq = 128` (one of `jammi-kernels`'s own
    /// two production `seq` classes -- see `tests/cuda_parity.rs`'s
    /// `SoftmaxLastDimFused CPU<->CUDA parity` section doc), `scale =
    /// 8.0` (`sqrt(64)`, ModernBERT-large's real `head_dim`), and the
    /// REAL `MASKED_LOGIT` mask convention
    /// ([`crate::mask::MASKED_LOGIT`], not a synthetic `-10_000.0`
    /// restated by hand) at a realistic padding density. F32, not BF16:
    /// this oracle is about a gross divide-vs-multiply operator swap (any
    /// finite `scale != 1.0` separates the two arithmetically at every
    /// dtype), not the BF16 boundary-rounding hazard `ops/softmax.rs`'s
    /// module doc discloses -- that hazard is a property of the FUSED
    /// kernel's own rounding point, which this arm never reaches.
    ///
    /// Compares fwd AND bwd bit-for-bit (`assert_eq!`, not an epsilon
    /// tolerance) against an INDEPENDENTLY-rooted inline
    /// `candle_nn::ops::softmax((scores / scale) + mask)` composition --
    /// on the same device, dtype, and deterministic CPU F32 ops, two runs
    /// of the identical composition must reproduce identical bits, so
    /// this is a measured claim ("numerically IDENTICAL to before"), not
    /// an assertion.
    #[test]
    fn eager_fallback_softmax_matches_inline_reference_fwd_and_bwd() {
        let device = Device::Cpu;
        let batch = 2;
        let heads = 16;
        let seq = 128;
        let scale = 8.0f64; // sqrt(64) -- ModernBERT-large's real head_dim.
        let sv: Vec<f32> = (0..batch * heads * seq * seq)
            .map(|i| (i as f32 * 0.017 - 5.0).sin() * 3.0)
            .collect();
        let mv: Vec<f32> = (0..batch * seq)
            .map(|i| {
                if i % 37 == 0 {
                    crate::mask::MASKED_LOGIT
                } else {
                    0.0
                }
            })
            .collect();

        let base_fn =
            Var::from_tensor(&Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap())
                .unwrap();
        let scores_fn = base_fn.as_tensor().t().unwrap();
        assert!(
            !scores_fn.is_contiguous(),
            "fixture construction bug: `.t()` must produce a non-contiguous view"
        );
        let mask_fn = Tensor::from_slice(&mv, (batch, 1, 1, seq), &device).unwrap();

        // Non-vacuity: this fixture must fail admission for EXACTLY the
        // contiguity clause -- not some other domain check silently
        // masking the one this test targets.
        let (holds, predicate) = softmax_admission_predicate(&scores_fn, &mask_fn, scale);
        assert!(!holds, "fixture must be refused admission");
        assert_eq!(
            predicate, "scores_contiguous",
            "fixture must fail admission for non-contiguity specifically, not another clause"
        );

        let base_ref =
            Var::from_tensor(&Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap())
                .unwrap();
        let scores_ref = base_ref.as_tensor().t().unwrap();
        let mask_ref = Tensor::from_slice(&mv, (batch, 1, 1, seq), &device).unwrap();

        let before = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        let out_fn = softmax_apply_training(&scores_fn, &mask_fn, scale).unwrap();
        let after = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.eager > before.eager,
            "this fixture must actually dispatch the EAGER fallback, or this oracle \
             exercises nothing (before={before:?}, after={after:?})"
        );

        let out_ref = candle_nn::ops::softmax(
            &(&scores_ref / scale)
                .unwrap()
                .broadcast_add(&mask_ref)
                .unwrap(),
            D::Minus1,
        )
        .unwrap();

        let vf: Vec<f32> = out_fn.flatten_all().unwrap().to_vec1().unwrap();
        let vr: Vec<f32> = out_ref.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vf.iter().any(|v| v.abs() > 1e-6),
            "fixture must be non-degenerate"
        );
        assert_eq!(
            vf, vr,
            "eager-fallback forward must be BIT-IDENTICAL to the inline eager reference \
             composition -- a mismatch here means the fallback arm no longer computes \
             `(scores / scores_divisor) + mask` exactly"
        );

        // A NON-UNIFORM seed -- see `fused_training_softmax_matches_eager_fwd_and_bwd`
        // above for why a uniform `dy` would make this comparison vacuous.
        let dy_seed_v: Vec<f32> = (0..batch * heads * seq * seq)
            .map(|i| (i as f32 * 0.023 - 1.0).cos())
            .collect();
        let dy_fn = Tensor::from_slice(&dy_seed_v, (batch, heads, seq, seq), &device).unwrap();
        let dy_ref = Tensor::from_slice(&dy_seed_v, (batch, heads, seq, seq), &device).unwrap();
        let loss_fn = (&out_fn * &dy_fn).unwrap().sum_all().unwrap();
        let loss_ref = (&out_ref * &dy_ref).unwrap().sum_all().unwrap();
        let grads_fn = loss_fn.backward().unwrap();
        let grads_ref = loss_ref.backward().unwrap();
        let dxf: Vec<f32> = grads_fn
            .get(&base_fn)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dxr: Vec<f32> = grads_ref
            .get(&base_ref)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            dxf.iter().any(|v| v.abs() > 1e-6),
            "gradient must be measured-nonzero"
        );
        assert_eq!(
            dxf, dxr,
            "eager-fallback backward must be BIT-IDENTICAL to the inline eager \
             reference's gradient"
        );
    }

    /// NODE-COUNT PROXY for the training arm's memory win: the call site,
    /// not just the op itself (`jammi_kernels::ops::softmax`'s own
    /// `fused_softmax_retains_fewer_tape_nodes_than_eager` already covers
    /// the op in isolation), retains FEWER tape nodes when `scale` is
    /// folded into `softmax_apply_training` directly, because the `scores
    /// / scale` `Op::Affine` node an equivalent call site without this
    /// folding would build before calling it is never constructed here.
    /// `Tensor::sorted_nodes()` is candle's own PUBLIC topological-sort-
    /// for-backward API (the exact list `Tensor::backward` walks) — a
    /// direct, honest count of what backward keeps resident, but it is
    /// still a NODE-COUNT PROXY for VRAM, not a byte measurement: one
    /// `[batch, heads, seq, seq]` node dropped from the tape is a real
    /// win in proportion to its own size, but this test does not, and
    /// cannot, measure bytes. The actual byte measurement is the committed
    /// pod A/B record,
    /// `crates/jammi-bench/baselines/p1_softmax_scale_fold_ab.json`, which
    /// discloses ALL THREE measured rows, not just the favorable one:
    /// `seq = 512` (`b8`) shows the predicted-size win (`77.46 GB -> 71.76
    /// GB`, within one allocator pool block of the retained-tensor-size
    /// arithmetic the JSON's `_comment` derives); `seq = 128` (`b16`)
    /// shows the SAME win at smaller absolute magnitude (`33.24 GB ->
    /// 32.57 GB`, also one pool block from predicted); `seq = 128` (`b8`)
    /// is the CONTRARY row — the arithmetic predicts a save, but the tip
    /// arm measures ONE allocator pool block (32 MiB) MORE than the base
    /// arm at that row, not less. See the JSON's `_comment` for the full
    /// arithmetic and the honest disclosure of why the smallest row does
    /// not follow the trend (allocator pool granularity dominates at that
    /// size, not the Affine-node removal itself).
    ///
    /// BEFORE: reconstructs the composition an equivalent call site
    /// WITHOUT `scale` folded in would run (the `Op::Affine` division,
    /// THEN `softmax_apply_training` with `scale = 1.0` — algebraically
    /// identical to calling it with the real `scale` directly, so this is
    /// a fair, apples-to-apples comparison, not a strawman). AFTER:
    /// `softmax_apply_training` called with the real `scale` directly, the
    /// call site `ModernBertAttention::forward`'s training arm actually
    /// uses. Both sides use the IDENTICAL fixture (fused-eligible on CPU,
    /// mirroring `fused_training_softmax_matches_eager_fwd_and_bwd`'s
    /// shape) so the ONLY difference between the two graphs is the
    /// presence/absence of the `Op::Affine` node.
    #[test]
    fn fused_training_softmax_call_site_drops_the_affine_node() {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq = 4;
        let scale = 8.0f64; // sqrt(64) -- ModernBERT-large's real head_dim.
        let sv: Vec<f32> = (0..batch * heads * seq * seq)
            .map(|i| (i as f32 * 0.19 - 2.0).cos() * 2.0)
            .collect();
        let mv: Vec<f32> = (0..batch * seq)
            .map(|i| if i == 2 { -10_000.0 } else { 0.0 })
            .collect();

        // BEFORE: the composition an equivalent call site WITHOUT `scale`
        // folded into `softmax_apply_training` would run.
        let s_before =
            Var::from_tensor(&Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap())
                .unwrap();
        let mask_before = Tensor::from_slice(&mv, (batch, 1, 1, seq), &device).unwrap();
        let scaled_before = (s_before.as_tensor() / scale).unwrap();
        let before_counters = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        let y_before = softmax_apply_training(&scaled_before, &mask_before, 1.0).unwrap();
        let after_before_counters = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        assert!(
            after_before_counters.fused > before_counters.fused,
            "the BEFORE fixture must actually dispatch the fused kernel too, or this \
             comparison is not apples-to-apples"
        );
        let nodes_before = y_before.sorted_nodes().len();

        // AFTER (the actual call site today): `scale` folded directly
        // into the fused op, no separate `Op::Affine` node ever built.
        let s_after =
            Var::from_tensor(&Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap())
                .unwrap();
        let mask_after = Tensor::from_slice(&mv, (batch, 1, 1, seq), &device).unwrap();
        let before_counters2 = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        let y_after = softmax_apply_training(&s_after, &mask_after, scale).unwrap();
        let after_counters2 = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        assert!(
            after_counters2.fused > before_counters2.fused,
            "the AFTER fixture must actually dispatch the fused kernel too, or this \
             comparison is not apples-to-apples"
        );
        let nodes_after = y_after.sorted_nodes().len();

        assert!(
            nodes_after < nodes_before,
            "the AFTER graph must retain FEWER tape nodes than BEFORE -- the \
             Op::Affine node must be gone: before={nodes_before} after={nodes_after}"
        );
        // Pin the MEASURED constants directly, not just "fewer than":
        // BEFORE = [s_before, scaled_before (Op::Affine), y_before] = 3
        // (mask is never a `Var` in either graph, so it never enters
        // `sorted_nodes` at all -- the same reasoning
        // `jammi_kernels::ops::softmax`'s own node-count oracle documents).
        assert_eq!(nodes_before, 3, "measured BEFORE node count");
        // AFTER = [s_after, y_after] = 2 -- the Op::Affine node is gone.
        assert_eq!(
            nodes_after, 2,
            "measured AFTER node count -- the Op::Affine node is gone"
        );
    }

    // -------------------------------------------------------------------
    // Fused GeGLU (C5)
    // -------------------------------------------------------------------

    fn eager_geglu(wi_out: &Tensor) -> Tensor {
        let intermediate = wi_out.dim(D::Minus1).unwrap() / 2;
        let gate = wi_out.narrow(D::Minus1, 0, intermediate).unwrap();
        let up = wi_out
            .narrow(D::Minus1, intermediate, intermediate)
            .unwrap();
        (gate.gelu_erf().unwrap() * up).unwrap()
    }

    #[test]
    fn geglu_admission_predicate_accepts_a_typical_modernbert_shape() {
        let device = Device::Cpu;
        let wi_out = Tensor::from_slice(&[0.0f32; 2 * 8], (1, 2, 8), &device).unwrap();
        let (holds, predicate) = geglu_admission_predicate(&wi_out);
        assert!(
            holds,
            "typical [batch, seq, 2*intermediate] must be admitted: {predicate}"
        );
    }

    /// Advisory fix mirroring `softmax_admission_predicate_rejects_
    /// broadcast_class_violation`: an odd last dimension (cannot split
    /// into equal gate/up halves) must be caught HERE — a counted eager
    /// fallback — rather than only inside the op, where it would surface
    /// as a raw `candle_core::Error` on the training arm.
    #[test]
    fn geglu_admission_predicate_rejects_odd_last_dim() {
        let device = Device::Cpu;
        let wi_out = Tensor::from_slice(&[0.0f32; 3], (1, 3), &device).unwrap();
        let (holds, predicate) = geglu_admission_predicate(&wi_out);
        assert!(!holds, "an odd last dim must be refused");
        assert_eq!(predicate, "last_dim_nonzero_and_even");
    }

    /// The eval-path bit-identity requirement, mirroring
    /// `eval_mode_attention_softmax_is_bit_identical_regardless_of_fused_
    /// eligibility`: `ModernBertMlp::forward`'s `training == false` arm
    /// always runs the eager `narrow`+`narrow`+`gelu_erf`+`mul`
    /// composition (see that method's `match`), structurally never
    /// reaching `geglu_apply_training` — this test proves that
    /// composition's OWN output is unaffected by `geglu_apply_training`
    /// existing and dispatching the fused kernel elsewhere, on a fixture
    /// that WOULD be fused-eligible.
    #[test]
    fn eval_mode_mlp_geglu_is_bit_identical_regardless_of_fused_eligibility() {
        let device = Device::Cpu;
        let intermediate = 8;
        let rows = 2;
        let wv: Vec<f32> = (0..rows * 2 * intermediate)
            .map(|i| (i as f32 * 0.23 - 1.0).sin() * 2.0)
            .collect();
        let wi_out = Tensor::from_slice(&wv, (rows, 2 * intermediate), &device).unwrap();

        // Non-vacuity: this fixture WOULD be admitted into the fused
        // kernel if training were true.
        let (holds, predicate) = geglu_admission_predicate(&wi_out);
        assert!(
            holds,
            "fixture must satisfy the fused GeGLU domain — the test proves eval \
             skips it anyway, not that the fixture happens to be ineligible: {predicate}"
        );

        let before: Vec<f32> = eager_geglu(&wi_out)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Exercise the REAL fused-or-fallback dispatch function (proven
        // eligible above) without changing eval's own composed call above.
        let training_before = GEGLU_DISPATCH_COUNTERS.snapshot();
        let _ = geglu_apply_training(&wi_out).unwrap();
        let training_after = GEGLU_DISPATCH_COUNTERS.snapshot();
        assert!(
            training_after.fused > training_before.fused,
            "the eligibility check above must be load-bearing: this exercise call \
             must actually dispatch the fused kernel, not silently fall back \
             (before={training_before:?}, after={training_after:?})"
        );

        let after: Vec<f32> = eager_geglu(&wi_out)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            before, after,
            "eval-mode MLP GeGLU must be byte-identical before and after the \
             fused kernel exists"
        );
    }

    /// Fused-vs-eager MLP-level oracle: `geglu_apply_training`'s actual
    /// dispatch path (fused kernel, since this fixture is fused-eligible
    /// on CPU) vs. the eager `narrow`+`narrow`+`gelu_erf`+`mul`
    /// composition, fwd AND bwd — mirroring
    /// `fused_training_softmax_matches_eager_fwd_and_bwd` exactly.
    #[test]
    fn fused_training_geglu_matches_eager_fwd_and_bwd() {
        let device = Device::Cpu;
        let intermediate = 8;
        let rows = 2;
        let wv: Vec<f32> = (0..rows * 2 * intermediate)
            .map(|i| (i as f32 * 0.19 - 2.0).cos() * 2.0)
            .collect();

        let wi_fused =
            Var::from_tensor(&Tensor::from_slice(&wv, (rows, 2 * intermediate), &device).unwrap())
                .unwrap();
        let before = GEGLU_DISPATCH_COUNTERS.snapshot();
        let out_fused = geglu_apply_training(&wi_fused).unwrap();
        let after = GEGLU_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "this fixture must actually dispatch the fused kernel, not fall back \
             (before={before:?}, after={after:?})"
        );

        let wi_eager =
            Var::from_tensor(&Tensor::from_slice(&wv, (rows, 2 * intermediate), &device).unwrap())
                .unwrap();
        let out_eager = eager_geglu(&wi_eager);

        let vf: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ve: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs eager {e}");
        }

        let grads_fused = out_fused.backward().unwrap();
        let grads_eager = out_eager.backward().unwrap();
        let dwf: Vec<f32> = grads_fused
            .get(&wi_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dwe: Vec<f32> = grads_eager
            .get(&wi_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (f, e)) in dwf.iter().zip(dwe.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dwi_out[{i}]: fused {f} vs eager {e}");
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Fused whole-attention-block (P3, Tier 0)
    // ─────────────────────────────────────────────────────────────────

    /// `cached_rope_pack` hands every caller the SAME tensor (same
    /// `TensorId`, therefore the same storage `Arc`) for a given dtype —
    /// the "0 bytes, 0 copies per layer" claim in its doc — and that
    /// tensor is byte-identical to a fresh `Tensor::stack` of the cached
    /// tables. A different dtype recomputes (new id, right dtype), and
    /// switching back recomputes again (single-entry memo, like
    /// `cast_cache`). Mutation this reddens under (verified, reverted):
    /// stacking fresh on every call (`TensorId`s differ).
    #[test]
    fn cached_rope_pack_is_memoised_per_dtype_and_bit_identical_to_a_fresh_stack() {
        let device = Device::Cpu;
        let (d, max_seq) = (ATTENTION_BLOCK_HEAD_DIM, 16usize);
        let rope = rope(d, max_seq, 10_000.0, &device);
        let first = rope.cached_rope_pack(DType::F32).unwrap();
        let second = rope.cached_rope_pack(DType::F32).unwrap();
        assert_eq!(
            first.id(),
            second.id(),
            "same dtype must return the memoised tensor"
        );
        assert_eq!(first.dims(), &[2, 1, 1, max_seq, d]);
        let (cos, sin) = rope.cached_tables(DType::F32).unwrap();
        let fresh = Tensor::stack(&[&cos, &sin], 0).unwrap();
        let bytes = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        assert_eq!(bytes(&first), bytes(&fresh));
        let bf16 = rope.cached_rope_pack(DType::BF16).unwrap();
        assert_ne!(bf16.id(), first.id());
        assert_eq!(bf16.dtype(), DType::BF16);
        assert_eq!(bf16.id(), rope.cached_rope_pack(DType::BF16).unwrap().id());
        let back = rope.cached_rope_pack(DType::F32).unwrap();
        assert_ne!(
            back.id(),
            first.id(),
            "single-entry memo recomputes after a dtype switch"
        );
        assert_eq!(bytes(&back), bytes(&fresh));
    }

    /// A minimal `ModernBertAttention` at `head_dim ==
    /// ATTENTION_BLOCK_HEAD_DIM`, for exercising
    /// `forward_training_attention`/`forward_eager_training_attention_composition`
    /// directly (both operate on an already-projected `qkv` supplied by
    /// the CALLER) AND `ModernBertAttention::forward` itself (which DOES
    /// run `wqkv`/`wo`) without loading a real checkpoint. `wqkv`/`wo`
    /// carry NON-DEGENERATE, deterministic seeded weights: an earlier
    /// revision used zeros, under which `qkv ≡ 0`, every score was `0`,
    /// `wo(ctx) + hidden == hidden` for ANY `ctx`, and the eval
    /// bit-identity test below passed with the mask add deleted — a
    /// vacuous oracle.
    fn attention_block_fixture(
        is_local: bool,
        h: usize,
        seq_for_table: usize,
        device: &Device,
    ) -> ModernBertAttention {
        use candle_nn::Linear;
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let wqkv_v: Vec<f32> = (0..3 * h * d * h * d)
            .map(|i| ((i as f32) * 0.0137).sin() * 0.2)
            .collect();
        let wo_v: Vec<f32> = (0..h * d * h * d)
            .map(|i| ((i as f32) * 0.0091).cos() * 0.2)
            .collect();
        let seeded_wqkv = Linear::new(
            Tensor::from_vec(wqkv_v, (3 * h * d, h * d), device).unwrap(),
            None,
        );
        let seeded_wo = Linear::new(
            Tensor::from_vec(wo_v, (h * d, h * d), device).unwrap(),
            None,
        );
        ModernBertAttention {
            wqkv: MaybeLoraLinear::Frozen(seeded_wqkv),
            wo: MaybeLoraLinear::Frozen(seeded_wo),
            attn_norm: None,
            rope: Arc::new(rope(d, seq_for_table, 10_000.0, device)),
            is_local,
            num_heads: h,
            head_dim: d,
            half_window: is_local.then_some(seq_for_table / 2),
            training: true,
        }
    }

    #[test]
    fn attention_block_admission_predicate_accepts_head_dim_64() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 4usize, 2usize, ATTENTION_BLOCK_HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, false, None);
        assert!(holds, "predicate={predicate}");
    }

    #[test]
    fn attention_block_admission_predicate_rejects_non_64_head_dim() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 4usize, 2usize, 16usize);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, false, None);
        assert!(!holds);
        assert_eq!(predicate, "head_dim_is_attention_block_fixed_head_dim");
    }

    #[test]
    fn attention_block_admission_predicate_rejects_missing_local_band() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 4usize, 2usize, ATTENTION_BLOCK_HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, true, None);
        assert!(!holds);
        assert_eq!(predicate, "local_mask_present");
    }

    /// The local-mask cells of `attention_block_admission_predicate`'s
    /// state table: `[1, 1, seq, seq]` and `[batch, 1, seq, seq]` accepted,
    /// a non-contiguous one and a wrong-shaped one refused by name.
    #[test]
    fn attention_block_admission_predicate_local_mask_cells() {
        let device = Device::Cpu;
        let (b, s, h, d) = (2usize, 4usize, 2usize, ATTENTION_BLOCK_HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let band = sliding_window_mask(s, 1, &device).unwrap();
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, true, Some(&band));
        assert!(holds, "[1,1,s,s]: predicate={predicate}");
        let combined = mask.broadcast_add(&band).unwrap();
        assert_eq!(combined.dims(), &[b, 1, s, s]);
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, true, Some(&combined));
        assert!(holds, "[b,1,s,s]: predicate={predicate}");
        let transposed = combined.transpose(2, 3).unwrap();
        assert!(!transposed.is_contiguous());
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, true, Some(&transposed));
        assert!(!holds);
        assert_eq!(predicate, "local_mask_contiguous");
        let wrong_batch = Tensor::zeros((b + 1, 1, s, s), DType::F32, &device).unwrap();
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, true, Some(&wrong_batch));
        assert!(!holds);
        assert_eq!(predicate, "local_mask_shape_batch_or_one_1_seq_seq");
        let padding_shaped = Tensor::zeros((1, 1, 1, s), DType::F32, &device).unwrap();
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, true, Some(&padding_shaped));
        assert!(!holds);
        assert_eq!(predicate, "local_mask_shape_batch_or_one_1_seq_seq");
        // A global layer never consults the local bundle.
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, false, None);
        assert!(holds, "global: predicate={predicate}");
    }

    /// `FusedAttentionMasks::build` adds the padding and band terms in
    /// `F32` and casts the SUM; the per-layer revision it replaces cast
    /// each term and added in the backbone dtype. Sweep every `(padding,
    /// band)` cell of the `{0.0, MASKED_LOGIT}` lattice at both admitted
    /// backbone dtypes and assert the two orders are byte-identical — the
    /// bit-neutrality claim the hoist rests on (see the struct's doc for
    /// the BF16 arithmetic this sweep checks rather than trusts).
    #[test]
    fn fused_masks_add_then_cast_is_bit_identical_to_cast_then_add_on_the_masked_logit_lattice() {
        let device = Device::Cpu;
        let m = crate::mask::MASKED_LOGIT;
        // One row holds every cell: padding [0, 0, M, M] against band
        // rows [0, M, 0, M] — the broadcast lays the 2x2 lattice out in
        // full, plus the real band geometry from `sliding_window_mask`.
        let s = 4usize;
        let padding = Tensor::from_slice(&[0f32, 0.0, m, m], (1, 1, 1, s), &device).unwrap();
        let band_v: Vec<f32> = (0..s * s)
            .map(|i| {
                if (i / s + i % s).is_multiple_of(2) {
                    0.0
                } else {
                    m
                }
            })
            .collect();
        let band = Tensor::from_slice(&band_v, (1, 1, s, s), &device).unwrap();
        for dtype in [DType::F32, DType::BF16] {
            let hoisted = FusedAttentionMasks::build(&padding, Some(&band), dtype).unwrap();
            let per_layer_global = padding.to_dtype(dtype).unwrap();
            let per_layer_local = padding
                .to_dtype(dtype)
                .unwrap()
                .broadcast_add(&band.to_dtype(dtype).unwrap())
                .unwrap();
            let bytes = |t: &Tensor| -> Vec<f32> {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            assert_eq!(hoisted.global.dtype(), dtype);
            assert_eq!(hoisted.local.as_ref().unwrap().dtype(), dtype);
            assert_eq!(
                bytes(&hoisted.global),
                bytes(&per_layer_global),
                "{dtype:?} global"
            );
            assert_eq!(
                bytes(hoisted.local.as_ref().unwrap()),
                bytes(&per_layer_local),
                "{dtype:?} local"
            );
            // Non-vacuity: the lattice really has all three distinct
            // values (0, one mask, two masks) in the local tensor.
            let local = bytes(hoisted.local.as_ref().unwrap());
            let mut distinct: Vec<f32> = local.clone();
            distinct.sort_by(|a, b| a.total_cmp(b));
            distinct.dedup();
            assert_eq!(distinct.len(), 3, "{dtype:?}: {distinct:?}");
        }
        let no_local = FusedAttentionMasks::build(&padding, None, DType::BF16).unwrap();
        assert!(no_local.local.is_none());
    }

    /// The `(training, fused_masks == None)` cell of
    /// `ModernBertAttention::forward`'s state table: a typed refusal, not
    /// a per-layer rebuild of the masks.
    #[test]
    fn training_attention_forward_without_fused_masks_is_a_typed_refusal() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 4usize, 2usize, ATTENTION_BLOCK_HEAD_DIM);
        let hidden = Tensor::zeros((b, s, h * d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let mut attn = attention_block_fixture(false, h, s, &device);
        attn.set_training(true);
        let err = attn
            .forward(&hidden, &mask, None, None, None)
            .expect_err("training mode without fused masks must be refused");
        assert!(matches!(err, EncoderError::Config(_)), "{err:?}");
    }

    /// Global-attention arm: the fused whole-attention-block forward must
    /// match TODAY'S eager training composition within tolerance (an
    /// algebraically-equivalent, not bit-identical, comparison — the SAME
    /// "own tolerance oracle" shape every other fused op's training arm in
    /// this crate documents; the fused KERNEL's own bit-exactness against
    /// a hand-composed reference is proven directly in
    /// `jammi_kernels::ops::attention_block`'s own test suite, not
    /// re-proven here).
    #[test]
    fn fused_training_attention_block_matches_eager_composition_within_tolerance_global() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h, d) = (2usize, 5usize, 2usize, ATTENTION_BLOCK_HEAD_DIM);
        let n = b * s * 3 * h * d;
        let qkv_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
        let qkv = Tensor::from_slice(&qkv_v, (b, s, 3 * h * d), &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();

        let attn = attention_block_fixture(false, h, s, &device);
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, false, None);
        assert!(holds, "predicate={predicate}");

        let fused = FusedAttentionMasks::build(&mask, None, DType::F32).unwrap();
        let before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let y_fused = attn
            .forward_training_attention(
                &qkv,
                b,
                s,
                h,
                d,
                TrainingMaskInputs {
                    extended: &mask,
                    local_band: None,
                    fused: &fused,
                },
                &declined_flash(),
            )
            .unwrap();
        let after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "must have actually dispatched fused"
        );

        let y_eager = attn
            .forward_eager_training_attention_composition(&qkv, b, s, h, d, &mask, None)
            .unwrap();

        let f: Vec<f32> = y_fused.flatten_all().unwrap().to_vec1().unwrap();
        let e: Vec<f32> = y_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (a, bb) in f.iter().zip(e.iter()) {
            assert!((a - bb).abs() < 1e-4, "{a} vs {bb}");
        }
    }

    /// Local-attention (window) arm: same comparison, with a real sliding
    /// window band supplied to both arms.
    #[test]
    fn fused_training_attention_block_matches_eager_composition_within_tolerance_local() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 9usize, 3usize, ATTENTION_BLOCK_HEAD_DIM);
        let half_window = 2usize;
        let n = b * s * 3 * h * d;
        let qkv_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos() * 0.4).collect();
        let qkv = Tensor::from_slice(&qkv_v, (b, s, 3 * h * d), &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let band = sliding_window_mask(s, half_window, &device).unwrap();

        let attn = attention_block_fixture(true, h, s, &device);
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, true, Some(&band));
        assert!(holds, "predicate={predicate}");

        let fused = FusedAttentionMasks::build(&mask, Some(&band), DType::F32).unwrap();
        let before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let y_fused = attn
            .forward_training_attention(
                &qkv,
                b,
                s,
                h,
                d,
                TrainingMaskInputs {
                    extended: &mask,
                    local_band: Some(&band),
                    fused: &fused,
                },
                &declined_flash(),
            )
            .unwrap();
        let after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "must have actually dispatched fused"
        );

        let y_eager = attn
            .forward_eager_training_attention_composition(&qkv, b, s, h, d, &mask, Some(&band))
            .unwrap();

        let f: Vec<f32> = y_fused.flatten_all().unwrap().to_vec1().unwrap();
        let e: Vec<f32> = y_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (a, bb) in f.iter().zip(e.iter()) {
            assert!((a - bb).abs() < 1e-4, "{a} vs {bb}");
        }
    }

    /// MEMORY ORACLE: `Tensor::sorted_nodes()` (candle's own public
    /// topological-sort-for-backward API — the exact list `Tensor::backward`
    /// walks and `GradStore::or_insert` allocates a full-size `zeros_like` +
    /// `add` for, per every other node-count oracle this crate/workspace
    /// ships) on the SAME `qkv` shape, one leg built via
    /// `forward_eager_training_attention_composition` (today's partial
    /// fusion: RoPE and softmax each their own node, everything else
    /// eager `Tensor` ops) and one via `forward_training_attention`
    /// (`AttentionBlockFused`, ONE node) — MEASURED live, not asserted a
    /// priori. Confirms the fused kernel actually dispatched (not a
    /// silent fallback, which would make "fewer nodes" a vacuous
    /// self-comparison).
    #[test]
    fn fused_attention_block_retains_fewer_tape_nodes_than_the_eager_training_composition() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 4usize, 1usize, ATTENTION_BLOCK_HEAD_DIM);
        let n = b * s * 3 * h * d;
        let qkv_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).sin()).collect();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let attn = attention_block_fixture(false, h, s, &device);

        let qkv_eager =
            Var::from_tensor(&Tensor::from_slice(&qkv_v, (b, s, 3 * h * d), &device).unwrap())
                .unwrap();
        let y_eager = attn
            .forward_eager_training_attention_composition(
                qkv_eager.as_tensor(),
                b,
                s,
                h,
                d,
                &mask,
                None,
            )
            .unwrap();
        let nodes_eager = y_eager.sorted_nodes().len();

        let qkv_fused =
            Var::from_tensor(&Tensor::from_slice(&qkv_v, (b, s, 3 * h * d), &device).unwrap())
                .unwrap();
        let fused = FusedAttentionMasks::build(&mask, None, DType::F32).unwrap();
        let before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let y_fused = attn
            .forward_training_attention(
                qkv_fused.as_tensor(),
                b,
                s,
                h,
                d,
                TrainingMaskInputs {
                    extended: &mask,
                    local_band: None,
                    fused: &fused,
                },
                &declined_flash(),
            )
            .unwrap();
        let after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "must have actually dispatched fused"
        );
        let nodes_fused = y_fused.sorted_nodes().len();

        assert!(
            nodes_fused < nodes_eager,
            "the fused whole-attention-block must retain FEWER tape nodes than today's \
             partial-fusion eager composition: eager={nodes_eager} fused={nodes_fused}"
        );
        // The EXACT fused count, not merely "fewer than": three nodes are
        // the only ones `forward_training_attention`'s fused arm can
        // build from a leaf `qkv` `Var` — the `Var` leaf itself, the
        // `qkv.reshape((batch, seq, 3, h, d))` view (`Op::Reshape`), and
        // the ONE `apply3`/`AttentionBlockFused` `CustomOp3` node — the
        // mask contributes NO node: it is a per-forward
        // `FusedAttentionMasks` tensor built from untracked inputs.
        // `Tensor::op()` (the field that would let this test inspect
        // EACH node's `Op` variant directly) is `pub(crate)` inside
        // `candle-core` — not reachable from this crate — so this
        // count IS the proof that exactly one `CustomOp3` node exists
        // here, not a substitute for inspecting it directly.
        assert_eq!(
            nodes_fused, 3,
            "fused whole-attention-block tape must be exactly {{qkv leaf, reshape, \
             AttentionBlockFused CustomOp3}} — got {nodes_fused} nodes"
        );
    }

    /// `forward_eval_attention` is exercised by NO OTHER unit test —
    /// the closest existing one (`eval_mode_attention_softmax_is_bit_
    /// identical_regardless_of_fused_eligibility`) only tests the softmax
    /// PREDICATE in isolation, never `ModernBertAttention::forward`'s own
    /// `self.training` branch. This test builds an INDEPENDENT hand
    /// composition (RoPE via `self.rope.apply` directly, `scores / scale`,
    /// one `broadcast_add`, `candle_nn::ops::softmax`, matmul, `Wo`, plus
    /// residual — the exact formula `forward_eval_attention`'s own doc
    /// describes, reconstructed here rather than calling that function, so
    /// this is not a vacuous self-comparison) and asserts `attn.forward()`
    /// at `training=false` is BYTE-IDENTICAL to it, on a fixture proven
    /// fused-eligible (so the test demonstrates eval structurally never
    /// reaches `AttentionBlockFused`, not merely that this fixture happens
    /// to fail admission) — plus that `ATTENTION_BLOCK_DISPATCH_COUNTERS`
    /// does not move at all during the eval call.
    ///
    /// The mask carries REAL padding (batch 0 pads its last key, batch 1
    /// its last two) and the fixture's weights are non-degenerate, so
    /// the mask add is load-bearing: the test asserts the padded
    /// reference differs from an unpadded one (non-vacuity), and reddens
    /// under BOTH audit-B2 mutations (verified, reverted): deleting
    /// `scores.broadcast_add(&extended_mask)` in `forward_eval_attention`
    /// (outputs differ from the hand composition), and inverting the
    /// fused-eligibility assertion's predicate (`assert!(!holds)`).
    #[test]
    fn attention_block_eval_output_is_bit_identical_regardless_of_fused_eligibility() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h, d) = (2usize, 5usize, 2usize, ATTENTION_BLOCK_HEAD_DIM);
        let n = b * s * h * d;
        let hidden_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.037).sin() * 0.5).collect();
        let hidden = Tensor::from_slice(&hidden_v, (b, s, h * d), &device).unwrap();
        // Per-batch-VARYING padding, in the exact additive form
        // `extended_attention_mask` produces (`0.0` real, `MASKED_LOGIT`
        // pad): batch 0 pads key 4, batch 1 pads keys 3 and 4.
        let mut mask_v = vec![0f32; b * s];
        mask_v[s - 1] = crate::mask::MASKED_LOGIT;
        mask_v[s + (s - 2)] = crate::mask::MASKED_LOGIT;
        mask_v[s + (s - 1)] = crate::mask::MASKED_LOGIT;
        let mask = Tensor::from_slice(&mask_v, (b, 1, 1, s), &device).unwrap();

        let mut attn = attention_block_fixture(false, h, s, &device);
        attn.set_training(false);

        // Non-vacuity: the SAME `qkv` this fixture's `wqkv` would project
        // is fused-eligible (mirrors the two `fused_training_attention_
        // block_matches_eager_composition_within_tolerance_*` tests
        // above, which already prove this admission path holds for this
        // fixture shape).
        let qkv = attn.wqkv.forward(&hidden).unwrap();
        let (holds, predicate) =
            attention_block_admission_predicate(&qkv, s, h, d, &mask, false, None);
        assert!(
            holds,
            "fixture must satisfy the fused attention-block domain — this test proves eval \
             skips it anyway, not that the fixture happens to be ineligible: {predicate}"
        );

        // Independent hand composition of `forward_eval_attention`'s own
        // documented formula.
        let q = qkv
            .narrow(D::Minus1, 0, h * d)
            .unwrap()
            .reshape((b, s, h, d))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k = qkv
            .narrow(D::Minus1, h * d, h * d)
            .unwrap()
            .reshape((b, s, h, d))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = qkv
            .narrow(D::Minus1, 2 * h * d, h * d)
            .unwrap()
            .reshape((b, s, h, d))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let q = attn.rope.apply(&q).unwrap();
        let k = attn.rope.apply(&k).unwrap();
        let scale = (d as f64).sqrt();
        let scores =
            crate::contiguous_matmul(&q, &k.transpose(D::Minus1, D::Minus2).unwrap()).unwrap();
        let scores = (scores / scale).unwrap();
        let compose = |scores: &Tensor| -> Vec<f32> {
            let p = candle_nn::ops::softmax(scores, D::Minus1).unwrap();
            let ctx = crate::contiguous_matmul(&p, &v)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((b, s, h * d))
                .unwrap();
            (attn.wo.forward(&ctx).unwrap() + &hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let before_ref = compose(&scores.broadcast_add(&mask).unwrap());
        // Non-vacuity: with real weights and real padding, the mask add
        // changes the output — otherwise the assertion below could not
        // tell a `forward_eval_attention` that dropped its mask add from
        // one that kept it.
        let unmasked_ref = compose(&scores);
        assert_ne!(
            before_ref, unmasked_ref,
            "the padded mask must change the output on this fixture, or the mask add is \
             not under test"
        );

        let counters_before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let after: Vec<f32> = attn
            .forward(&hidden, &mask, None, None, None)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // The `(eval, Some(fused))` cell: the bundle is unread in eval, so
        // supplying it changes nothing, byte for byte.
        let fused = FusedAttentionMasks::build(&mask, None, DType::F32).unwrap();
        let after_with_bundle: Vec<f32> = attn
            .forward(&hidden, &mask, None, Some(&fused), None)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            after, after_with_bundle,
            "eval must not read the fused-mask bundle"
        );
        let counters_after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();

        assert_eq!(
            counters_after.fused, counters_before.fused,
            "eval must never dispatch AttentionBlockFused (fused count moved)"
        );
        assert_eq!(
            counters_after.eager, counters_before.eager,
            "eval must never even consult this op's admission machinery (eager-fallback count \
             moved — forward_eval_attention must be a structurally separate code path, not a \
             domain-miss fallback of the training path)"
        );
        assert_eq!(
            before_ref, after,
            "eval-mode ModernBertAttention::forward output must be byte-identical to the \
             hand-composed reference regardless of AttentionBlockFused's existence"
        );
    }

    /// The counter-threading deletion test:
    /// `attn.training` gating `forward_training_attention` vs
    /// `forward_eval_attention` (`ModernBertAttention::forward`'s own
    /// `if self.training` branch) is the ONLY thing that would catch that
    /// branch being accidentally deleted or inverted. Mirrors
    /// `tests/it/modernbert.rs`'s `set_training_threading_gates_the_fused_
    /// {rope,softmax,geglu}_dispatch_counters`, at the unit level — driven
    /// through `attention_block_fixture` + `ModernBertAttention::forward`
    /// directly. (The cookbook's `tiny_modernbert_classifier` fixture has
    /// `head_dim == 16`, so it cannot reach this op; real ModernBERT-base
    /// and -large checkpoints have `head_dim == 64` — `768 / 12`, `1024 /
    /// 16` — and so does this crate's own `tests/fixtures/
    /// tiny_modernbert_head64`, which
    /// `forward_hidden_reaches_the_fused_attention_block_on_a_head_dim_64_checkpoint`
    /// drives through the full `ModernBert::forward_hidden`.)
    #[test]
    fn set_training_threading_gates_the_fused_attention_block_dispatch_counters() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 4usize, 2usize, ATTENTION_BLOCK_HEAD_DIM);
        let n = b * s * h * d;
        let hidden_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.041).cos() * 0.3).collect();
        let hidden = Tensor::from_slice(&hidden_v, (b, s, h * d), &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();

        let mut attn = attention_block_fixture(false, h, s, &device);

        let fused = FusedAttentionMasks::build(&mask, None, DType::F32).unwrap();
        attn.set_training(false);
        let before_eval = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let _ = attn.forward(&hidden, &mask, None, None, None).unwrap();
        let after_eval = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            after_eval.fused, before_eval.fused,
            "eval must never dispatch the fused attention block"
        );

        attn.set_training(true);
        let before_train = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let flash = declined_flash();
        let _ = attn
            .forward(&hidden, &mask, None, Some(&fused), Some(&flash))
            .unwrap();
        let after_train = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert!(
            after_train.fused > before_train.fused,
            "training=true must dispatch the fused attention block at least once \
             (before={before_train:?}, after={after_train:?})"
        );

        attn.set_training(false);
        let before_eval2 = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let _ = attn.forward(&hidden, &mask, None, None, None).unwrap();
        let after_eval2 = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            after_eval2.fused, before_eval2.fused,
            "set_training(false) must restore the eval-only dispatch path"
        );
    }

    /// The full-model proof: `ModernBert::forward_hidden` on a
    /// `head_dim == 64` checkpoint (this crate's own
    /// `tests/fixtures/tiny_modernbert_head64`: hidden 64, 1 head, 2
    /// layers with `global_attn_every_n_layers = 2` — layer 0 global,
    /// layer 1 local with `local_attention = 8`) reaches the fused
    /// whole-attention-block arm on BOTH layer kinds in training: the
    /// fused counter advances by exactly `num_hidden_layers` per training
    /// forward with zero eager fallbacks, and does not move in eval. This
    /// is what exercises the per-forward `FusedAttentionMasks` build in
    /// `forward_hidden` end to end (a global layer consuming `global`, a
    /// local layer consuming `local`) — the mutation it reddens under
    /// (verified, reverted): deleting `self.training = training` from
    /// `ModernBert::set_training`, which leaves `forward_hidden` building
    /// no masks and the training forward a typed `Config` refusal. The
    /// padded input (batch 1 pads its last two tokens) makes the local
    /// layer's combined mask carry all three lattice values.
    #[test]
    fn forward_hidden_reaches_the_fused_attention_block_on_a_head_dim_64_checkpoint() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_modernbert_head64");
        let config: ModernBertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(
            config.hidden_size / config.num_attention_heads,
            ATTENTION_BLOCK_HEAD_DIM
        );
        assert!(!config.is_local_layer(0) && config.is_local_layer(1));
        let weights = dir.join("model.safetensors");
        let varmap = candle_nn::VarMap::new();
        let mut model = ModernBert::builder()
            .build(&[weights.as_path()], &config, &device, &varmap)
            .unwrap();
        let input_ids =
            Tensor::new(&[[2u32, 5, 10, 3, 7, 9], [4u32, 8, 1, 6, 0, 0]], &device).unwrap();
        let mask = Tensor::new(&[[1u32, 1, 1, 1, 1, 1], [1u32, 1, 1, 1, 0, 0]], &device).unwrap();

        let before_eval = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let eval_out = model.forward_hidden(&input_ids, &mask).unwrap();
        let after_eval = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            after_eval.fused, before_eval.fused,
            "eval never dispatches fused"
        );
        assert_eq!(
            after_eval.eager, before_eval.eager,
            "eval never consults admission"
        );

        model.set_training(true);
        let before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let train_out = model.forward_hidden(&input_ids, &mask).unwrap();
        let after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            after.fused - before.fused,
            config.num_hidden_layers as u64,
            "every layer (global AND local) must take the fused arm: {before:?} -> {after:?}"
        );
        assert_eq!(
            after.eager, before.eager,
            "no eager fallback on a head_dim-64 checkpoint"
        );
        assert_eq!(train_out.dims(), eval_out.dims());
        let t: Vec<f32> = train_out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            t.iter().all(|x| x.is_finite()),
            "training output must be finite"
        );

        model.set_training(false);
        let before2 = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let _ = model.forward_hidden(&input_ids, &mask).unwrap();
        let after2 = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            after2.fused, before2.fused,
            "set_training(false) restores eval"
        );
    }

    /// A CPU/F32 fused-vs-eager comparison at PRODUCTION `heads=16`/
    /// `head_dim=64` and `seq=512`, driving the REAL `forward_training_
    /// attention` fused arm and the REAL `forward_eager_training_
    /// attention_composition` on a LOCAL layer.
    ///
    /// NOT a regression oracle for the GEMM-operand-form defect (P3 fix
    /// round 4's finding, correcting an earlier revision of this doc that
    /// called it one): this test runs on `Device::Cpu` at `F32`, and the
    /// defect is a `bf16` cuBLAS strided-batched-GEMM blocking artifact
    /// with no CPU/F32 analogue at all — this test is bit-exact-clean
    /// (`TOL = 1e-4`) with the defect present OR absent, on EITHER
    /// device, because the divergence it would need to catch does not
    /// exist at this dtype. Its real job is architectural: proving `qkv`
    /// (a tracked `Var`, never a leaf fixture — clause 3: LoRA gradients
    /// are downstream of a `Var` in production) reaches the fused arm at
    /// all at this shape, and that fused/eager agree in VALUE at F32
    /// where cuBLAS's bf16-specific operand-form sensitivity cannot
    /// apply. `tests::attention_block_fused_vs_eager_dqkv_divergence_
    /// grows_with_depth_bf16_cuda` (this file) is the actual defect
    /// oracle.
    #[test]
    fn fused_attention_block_matches_eager_lora_gradients_at_production_seq_on_head64() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let b = 2usize;
        let s = 512usize; // production ModernBERT-large seq the pod repro used
        let h = 16usize; // production ModernBERT-large head count
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let half_window = 64usize; // production local_attention=128 -> half_window=64
        let attn = attention_block_fixture(true, h, s, &device);

        let n = b * s * 3 * h * d;
        let qkv_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0197).sin() * 0.9).collect();
        let dy_v: Vec<f32> = (0..(b * s * h * d))
            .map(|i| ((i as f32) * 0.0071).cos() * 0.6 + 0.05)
            .collect();
        let dy = Tensor::from_vec(dy_v, (b, s, h * d), &device).unwrap();

        // Padding-free extended mask (matches the bench's own synthetic
        // all-ones attention_mask — the cell the round's own pod repro
        // actually exercised) combined with the production window band,
        // via the SAME `FusedAttentionMasks::build`/`sliding_window_mask`
        // the real `ModernBert::forward_hidden` call site uses.
        let extended = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let band = crate::mask::sliding_window_mask(s, half_window, &device).unwrap();
        let fused_masks = FusedAttentionMasks::build(&extended, Some(&band), DType::F32).unwrap();

        let run = |label: &str, force_eager: bool| -> (Vec<f32>, Vec<f32>) {
            let qkv = Var::from_tensor(
                &Tensor::from_vec(qkv_v.clone(), (b, s, 3 * h * d), &device).unwrap(),
            )
            .unwrap();
            let masks = TrainingMaskInputs {
                extended: &extended,
                local_band: Some(&band),
                fused: &fused_masks,
            };
            let out = if force_eager {
                attn.forward_eager_training_attention_composition(
                    qkv.as_tensor(),
                    b,
                    s,
                    h,
                    d,
                    &extended,
                    Some(&band),
                )
                .unwrap()
            } else {
                attn.forward_training_attention(
                    qkv.as_tensor(),
                    b,
                    s,
                    h,
                    d,
                    masks,
                    &declined_flash(),
                )
                .unwrap()
            };
            let loss = (&out * &dy).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let dqkv: Vec<f32> = grads
                .get(&qkv)
                .unwrap_or_else(|| panic!("{label}: no dqkv"))
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            (out_v, dqkv)
        };

        let before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        let (out_fused, dqkv_fused) = run("fused", false);
        let after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            after.fused,
            before.fused + 1,
            "the fused arm must actually admit at this (b,h,s,d) cell — a domain miss would \
             silently fall back to eager and make this test compare eager against itself"
        );
        let (out_eager, dqkv_eager) = run("eager", true);

        assert_eq!(out_fused.len(), out_eager.len());
        assert_eq!(dqkv_fused.len(), dqkv_eager.len());
        const TOL: f32 = 1e-4;
        let mut max_out_delta = 0f32;
        for (c, g) in out_fused.iter().zip(out_eager.iter()) {
            max_out_delta = max_out_delta.max((c - g).abs());
        }
        assert!(
            max_out_delta <= TOL,
            "fused vs eager forward output at production (b={b},h={h},s={s},d={d}) local layer: \
             max|Δ|={max_out_delta:e} > {TOL:e}"
        );
        let mut max_dqkv_delta = 0f32;
        for (c, g) in dqkv_fused.iter().zip(dqkv_eager.iter()) {
            max_dqkv_delta = max_dqkv_delta.max((c - g).abs());
        }
        assert!(
            max_dqkv_delta <= TOL,
            "fused vs eager dqkv at production (b={b},h={h},s={s},d={d}) local layer: \
             max|Δ|={max_dqkv_delta:e} > {TOL:e} (F32 — see this test's own doc for why this \
             leg cannot see the bf16-specific GEMM-operand-form defect)"
        );
    }

    /// Mirrors `tests/cuda_parity.rs`'s own `cuda_device` (`jammi-kernels`):
    /// a machine that compiled with the `cuda` feature but has no
    /// physical GPU is "skip", not "fail", UNLESS `JAMMI_REQUIRE_CUDA` is
    /// set, in which case device-acquisition failure panics rather than
    /// silently reading as a skip.
    #[cfg(feature = "cuda")]
    fn growth_oracle_cuda_device() -> Option<Device> {
        match Device::new_cuda(0) {
            Ok(d) => Some(d),
            Err(e) => {
                if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                    panic!(
                        "attention_block_fused_vs_eager_dqkv_divergence_grows_with_depth_bf16_cuda: \
                         JAMMI_REQUIRE_CUDA is set but no CUDA device could be acquired: {e}"
                    );
                }
                eprintln!("depth-growth oracle: skipping — no CUDA device available ({e})");
                None
            }
        }
    }

    /// P3 fix round 4, deliverable 1 (esc-044's phase-0.7 `symptom_spec`):
    /// a `.contiguous()`-restored regression is invisible to any SINGLE
    /// fused-vs-eager `bwd` call — its systematic bias is smaller than
    /// ordinary bf16 rounding noise at depth 1 (this crate's own
    /// per-element derived bounds already admit it) — and only separates
    /// from noise by COMPOUNDING through depth. This drives the REAL
    /// `forward_training_attention` (fused arm) and
    /// `forward_eager_training_attention_composition` (production eager
    /// arm) `L_MAX` times each, chaining each call's own output back into
    /// the next call's `qkv` input (`qkv_next = cat([out, out, out],
    /// last)` — a weight-free, shape-correct bridge: the mechanism under
    /// test lives entirely inside the attention call itself, not in any
    /// inter-layer projection, so no per-layer `Wqkv` is needed — family
    /// L) from a SINGLE tracked `qkv` `Var`, then compares the two arms'
    /// `dqkv` at the END of the chain, not after one call.
    ///
    /// Discriminating quantity: `r(L) = Σ|dqkv_fused - dqkv_eager| /
    /// Σ|dqkv_eager|` per `qkv` slot. Gate: `r(L_MAX) <= C * max(r(1),
    /// EPS)` — `r(1)`, MEASURED on this same run, is the noise floor, not
    /// an absolute bf16-ULP constant (floor discipline). Six non-vacuity
    /// clauses, all required for this leg to count: (1) RED-FIRST
    /// PROVENANCE — this leg must be shown RED with the three
    /// `.contiguous()` calls `jammi_kernels::ops::attention_block`'s
    /// `bwd_core` removes restored, GREEN without — both runs reported in
    /// this round's hand-off, not just asserted here. (2) DISPATCH
    /// PROVENANCE — asserted below. (3) NON-FINITE — every comparison is
    /// `assert!(x.is_finite() && x <= bound)`, never a negated `>`.
    /// (4) SIGNAL — `Σ|dqkv_eager| > 0` per slot, asserted before it is
    /// used as a denominator. (5) INDEPENDENCE — "eager" is
    /// `forward_eager_training_attention_composition` itself, called
    /// directly (the SAME method `forward_training_attention`'s own
    /// fallback calls), never a copy of its logic in this test file.
    /// (6) FLOOR DISCIPLINE — the gate above uses only measured
    /// quantities from THIS run.
    ///
    /// An `F32` run of the SAME chain (`forward_eager_training_
    /// attention_composition` at `L_MAX` deep, `F32`) is the
    /// anti-vacuity anchor: it CANNOT itself redden on this defect (both
    /// `bf16` arms' GEMM reduction orders are individually legal relative
    /// to it), so it exists only to catch the two `bf16` arms being wrong
    /// TOGETHER in some unrelated way — confirmation, never the gate.
    #[test]
    #[cfg(feature = "cuda")]
    fn attention_block_fused_vs_eager_dqkv_divergence_grows_with_depth_bf16_cuda() {
        let Some(device) = growth_oracle_cuda_device() else {
            return;
        };
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        const L_MAX: usize = 28; // the real ModernBERT-large depth this defect was found on.
        let (b, s, h): (usize, usize, usize) = (8, 512, 16);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let hd = h * d;
        let half_window = 64usize;
        let attn = attention_block_fixture(true, h, s, &device);

        let n = b * s * 3 * hd;
        let qkv0_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0137).sin() * 0.6).collect();
        let dy_v: Vec<f32> = (0..(b * s * hd))
            .map(|i| ((i as f32) * 0.0059).cos() * 0.5 + 0.05)
            .collect();
        let extended = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let band = crate::mask::sliding_window_mask(s, half_window, &device).unwrap();
        let fused_masks_bf16 =
            FusedAttentionMasks::build(&extended, Some(&band), DType::BF16).unwrap();

        // Returns `(dqkv, fused_dispatch_delta, eager_dispatch_delta)`.
        let run = |force_eager: bool, dtype: DType, l: usize| -> (Vec<f32>, u64, u64) {
            let before = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
            let qkv = Var::from_tensor(
                &Tensor::from_vec(qkv0_v.clone(), (b, s, 3 * hd), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap(),
            )
            .unwrap();
            let dy = Tensor::from_vec(dy_v.clone(), (b, s, hd), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            assert!(l > 0, "run: l must be >= 1");
            let mut cur = qkv.as_tensor().clone();
            let mut last_out = None;
            for _ in 0..l {
                let out = if force_eager {
                    attn.forward_eager_training_attention_composition(
                        &cur,
                        b,
                        s,
                        h,
                        d,
                        &extended,
                        Some(&band),
                    )
                    .unwrap()
                } else {
                    let masks = TrainingMaskInputs {
                        extended: &extended,
                        local_band: Some(&band),
                        fused: &fused_masks_bf16,
                    };
                    attn.forward_training_attention(&cur, b, s, h, d, masks, &declined_flash())
                        .unwrap()
                };
                // Amplitude control between chained calls (no residual/
                // LayerNorm in this synthetic chain — see this test's own
                // doc): a plain max-abs rescale keeps `cur` inside this
                // op's own validated bf16 domain (module doc's "BF16
                // validated-coverage ceiling" section) across `L_MAX`
                // layers.
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
                let out_n = (&out / f64::from(out_max)).unwrap();
                cur = Tensor::cat(&[&out_n, &out_n, &out_n], D::Minus1).unwrap();
                last_out = Some(out);
            }
            // `cur` (the last iteration's RE-TILED `[b, s, 3*hd]` bridge
            // for a NEXT iteration that never runs) is NOT the loss input
            // — `last_out` (`[b, s, hd]`, the last iteration's own
            // attention output, matching `dy`'s shape) is.
            let loss = (last_out.unwrap() * &dy).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let dqkv: Vec<f32> = grads
                .get(&qkv)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let after = ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot();
            (dqkv, after.fused - before.fused, after.eager - before.eager)
        };

        let (dqkv_fused_1, fused_ctr_1, eager_ctr_1) = run(false, DType::BF16, 1);
        let (dqkv_eager_1, ef_ctr_1, ee_ctr_1) = run(true, DType::BF16, 1);
        let (dqkv_fused_l, fused_ctr_l, eager_ctr_l) = run(false, DType::BF16, L_MAX);
        let (dqkv_eager_l, ef_ctr_l, ee_ctr_l) = run(true, DType::BF16, L_MAX);
        let (dqkv_ref_l, _, _) = run(true, DType::F32, L_MAX);

        // Clause (2): DISPATCH PROVENANCE — zero dispatch is RED, never
        // green: a silently-eager-fallen-back fused leg would compare
        // eager against itself.
        assert_eq!(
            fused_ctr_1, 1,
            "fused leg (L=1) must dispatch fused exactly once"
        );
        assert_eq!(
            eager_ctr_1, 0,
            "fused leg (L=1) must never fall back to eager"
        );
        assert_eq!(
            fused_ctr_l, L_MAX as u64,
            "fused leg (L={L_MAX}) must dispatch fused every layer"
        );
        assert_eq!(
            eager_ctr_l, 0,
            "fused leg (L={L_MAX}) must never fall back to eager"
        );
        assert_eq!(ef_ctr_1, 0, "the eager leg calls the eager composition directly — it must never touch the fused dispatch counter");
        assert_eq!(
            ee_ctr_1, 0,
            "the eager leg bypasses `admit` entirely — its own counter stays untouched"
        );
        assert_eq!(ef_ctr_l, 0, "the eager leg calls the eager composition directly — it must never touch the fused dispatch counter");
        assert_eq!(
            ee_ctr_l, 0,
            "the eager leg bypasses `admit` entirely — its own counter stays untouched"
        );

        // Clause (3): NON-FINITE, checked BEFORE any comparison; clause
        // (4): SIGNAL.
        let finite_sum = |v: &[f32]| -> (bool, f64) {
            let mut ok = true;
            let mut sum = 0f64;
            for &x in v {
                ok &= x.is_finite();
                sum += f64::from(x.abs());
            }
            (ok, sum)
        };
        let (fused_1_ok, _) = finite_sum(&dqkv_fused_1);
        let (eager_1_ok, eager_1_sum) = finite_sum(&dqkv_eager_1);
        let (fused_l_ok, _) = finite_sum(&dqkv_fused_l);
        let (eager_l_ok, eager_l_sum) = finite_sum(&dqkv_eager_l);
        let (ref_l_ok, _) = finite_sum(&dqkv_ref_l);
        assert!(
            fused_1_ok && eager_1_ok && fused_l_ok && eager_l_ok && ref_l_ok,
            "non-finite dqkv element(s) present before any comparison"
        );
        assert!(
            eager_1_sum.is_finite() && eager_1_sum > 0.0,
            "Σ|dqkv_eager| at L=1 must be nonzero"
        );
        assert!(
            eager_l_sum.is_finite() && eager_l_sum > 0.0,
            "Σ|dqkv_eager| at L={L_MAX} must be nonzero"
        );

        // r(L) per qkv slot (0=Q,1=K,2=V — `dqkv`'s last axis is
        // `[q_seg(hd), k_seg(hd), v_seg(hd)]`, `forward_eager_training_
        // attention_composition`'s own `narrow` layout).
        let r = |fused: &[f32], eager: &[f32]| -> [f64; 3] {
            let mut num = [0f64; 3];
            let mut den = [0f64; 3];
            for (i, (&fv, &ev)) in fused.iter().zip(eager.iter()).enumerate() {
                let slot = (i / hd) % 3;
                num[slot] += f64::from((fv - ev).abs());
                den[slot] += f64::from(ev.abs());
            }
            [
                num[0] / den[0].max(1e-30),
                num[1] / den[1].max(1e-30),
                num[2] / den[2].max(1e-30),
            ]
        };
        let r1 = r(&dqkv_fused_1, &dqkv_eager_1);
        let rl = r(&dqkv_fused_l, &dqkv_eager_l);

        // The gate: GROWTH, not magnitude. `EPS` guards only against a
        // pathological exact `r(1) == 0` tie (measured: on the FIXED
        // build `r(1)` is exactly `0.0` for every slot — this op's own
        // GEMMs are bit-identical to production's at a single call, at
        // this shape — and a floor at 0 would divide by zero below); it
        // is set two orders of magnitude BELOW the smallest genuinely-
        // measured `r(1)` this file's own pod run recorded with the
        // defect present (`~1.04e-7`, F32-epsilon scale — an ordinary
        // bf16 arm agrees with production almost exactly at ONE call even
        // WITH the defect live, which is the whole reason a single-call
        // comparison cannot see it), so it never competes with a real
        // measurement — it is not the discriminating bound itself.
        const C: f64 = 4.0;
        const EPS: f64 = 1e-9;
        for slot in 0..3 {
            let floor = r1[slot].max(EPS);
            let bound = C * floor;
            assert!(
                rl[slot].is_finite() && rl[slot] <= bound,
                "slot {slot} (0=Q,1=K,2=V): r(L={L_MAX})={:e} exceeds {C}*max(r(1),{EPS:e})={bound:e} \
                 (r(1)={:e}) — the fused/eager divergence is growing SYSTEMATICALLY with depth, not \
                 staying at the L=1 bf16-noise scale",
                rl[slot],
                r1[slot],
            );
        }

        // Anti-vacuity anchor (does NOT gate the defect — see this test's
        // own doc): both bf16 arms must each stay within a generous,
        // depth-scaled multiple of ordinary bf16 rounding of the F32
        // reference, ruling out both arms being wrong TOGETHER.
        let anchor = |bf16_v: &[f32]| -> f64 {
            let mut num = 0f64;
            let mut den = 0f64;
            for (&x, &rf) in bf16_v.iter().zip(dqkv_ref_l.iter()) {
                num += f64::from((x - rf).abs());
                den += f64::from(rf.abs());
            }
            num / den.max(1e-30)
        };
        let fused_anchor = anchor(&dqkv_fused_l);
        let eager_anchor = anchor(&dqkv_eager_l);
        assert!(
            fused_anchor.is_finite() && fused_anchor <= 0.5,
            "fused dqkv at L={L_MAX} deviates from the F32 anchor by {fused_anchor:e} — too large \
             to be ordinary bf16 rounding compounded over {L_MAX} layers"
        );
        assert!(
            eager_anchor.is_finite() && eager_anchor <= 0.5,
            "eager dqkv at L={L_MAX} deviates from the F32 anchor by {eager_anchor:e} — too large \
             to be ordinary bf16 rounding compounded over {L_MAX} layers"
        );
    }

    /// Mirrors `crate::layer_norm::tests::strict_mode_errors_instead_of_
    /// falling_back_on_a_failed_predicate`, for `"attention_block_fused"`:
    /// a domain miss under `AdmissionMode::Strict` must return
    /// `KernelError::StrictModeFallback`, never a silent eager fallback.
    /// Calls `jammi_kernels::admission::admit` directly with an explicit
    /// `Strict` mode (not the `JAMMI_KERNELS_STRICT` env var, which
    /// `admission_mode()` memoizes into a process-wide `OnceLock` the
    /// first time anything calls it — depending on env-var timing across
    /// `cargo test`'s parallel thread pool would be racy).
    #[test]
    fn attention_block_strict_mode_errors_instead_of_falling_back_on_a_failed_predicate() {
        use jammi_kernels::admission::AdmissionMode;
        let counters = jammi_kernels::admission::DispatchCounters::new();
        let err = admit(
            AdmissionMode::Strict,
            "attention_block_fused",
            "mask_shape_batch_or_one_1_1_seq",
            false,
            &counters,
        )
        .expect_err("a failed predicate in Strict mode must error");
        assert!(matches!(
            err,
            jammi_kernels::error::KernelError::StrictModeFallback {
                op: "attention_block_fused",
                predicate: "mask_shape_batch_or_one_1_1_seq"
            }
        ));
    }
}
