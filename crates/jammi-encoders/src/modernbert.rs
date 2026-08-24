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
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, VarBuilder, VarMap};
use jammi_kernels::admission::{admit, DispatchCounters, DispatchOutcome};
use jammi_kernels::ops::{
    apply1, apply2, apply3, RopeFused, SoftmaxLastDimFused, MAX_HEAD_DIM, MAX_LAST_DIM, MAX_RANK,
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
/// mirroring `crate::layer_norm::LN_DISPATCH_COUNTERS` — see this
/// module's "RoPE: table hoisting + fused rotate-half" doc section for
/// the training-only gate this counts. `pub(crate)` (not `pub`) — read via
/// [`crate::rope_dispatch_snapshot`], the same shape
/// [`crate::ln_dispatch_snapshot`] uses.
pub(crate) static ROPE_DISPATCH_COUNTERS: DispatchCounters = DispatchCounters::new();

// ─────────────────────────────────────────────────────────────────────────────
// Fused masked softmax (C4)
// ─────────────────────────────────────────────────────────────────────────────

/// Fused/eager dispatch counters for the ModernBERT attention softmax,
/// mirroring `ROPE_DISPATCH_COUNTERS` / `crate::layer_norm::LN_DISPATCH_COUNTERS`
/// — see `ModernBertAttention::softmax_apply`'s doc for the training-only
/// gate this counts. `pub(crate)` (not `pub`) — read via
/// [`crate::softmax_dispatch_snapshot`], the same shape
/// [`crate::rope_dispatch_snapshot`] / [`crate::ln_dispatch_snapshot`] use.
pub(crate) static SOFTMAX_DISPATCH_COUNTERS: DispatchCounters = DispatchCounters::new();

/// The fused masked-softmax kernel's domain, checked at the call site
/// (family D / K2): `scores`'s device is one
/// [`crate::layer_norm::device_is_supported`] accepts, `scores`/`mask`
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
fn softmax_admission_predicate(scores: &Tensor, mask: &Tensor) -> (bool, &'static str) {
    if !crate::layer_norm::device_is_supported(scores.device()) {
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
            crate::layer_norm::admission_mode(),
            "rope_fused",
            predicate,
            holds,
            &ROPE_DISPATCH_COUNTERS,
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
/// K2): `x`'s device is one [`crate::layer_norm::device_is_supported`]
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
    if !crate::layer_norm::device_is_supported(x_device) {
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
    fn forward(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        local_band: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let normed = match &self.attn_norm {
            Some(ln) => ln.forward(hidden)?,
            None => hidden.clone(),
        };
        let (batch, seq, _) = normed.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;

        let qkv = self.wqkv.forward(&normed)?;

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
        let scores = crate::contiguous_matmul(&q, &k.transpose(D::Minus1, D::Minus2)?)?;
        let scores = (scores / scale)?;
        // The additive mask is always built in F32 (see `extended_attention_mask`);
        // cast to the scores' dtype so a F16/BF16 backbone can add it (a no-op
        // when scores are already F32).
        let extended_mask = extended_mask.to_dtype(scores.dtype())?;

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
                    extended_mask.broadcast_add(&band.to_dtype(scores.dtype())?)?
                }
                (true, None) => {
                    return Err(EncoderError::Config(
                        "local-attention layer reached without a sliding-window band".into(),
                    ))
                }
                (false, _) => extended_mask,
            };
            softmax_apply_training(&scores, &mask)?
        } else {
            // Eval's UNCHANGED code path, bit-identical to before this
            // commit: two SEQUENTIAL broadcast-adds, each from its own
            // smaller shape, never combined into one tensor (see
            // `crate::mask::sliding_window_mask`'s doc for why — neither
            // mask is ever materialised at `[batch, heads, seq, seq]`
            // either way, but combining them first would round
            // differently than adding them in this order, which eval must
            // never do — see `tests::eval_mode_attention_softmax_is_bit_identical_regardless_of_fused_eligibility`).
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

        let ctx = crate::contiguous_matmul(&attn, &v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, h * d))?;

        let out = self.wo.forward(&ctx)?;
        Ok((out + hidden)?)
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
/// `scores.broadcast_add(mask)` plus `candle_nn::ops::softmax` composition
/// (recording which happened either way, mirroring `crate::layer_norm`'s
/// and `RotaryEmbedding`'s identical admission mechanism). Eval never
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
fn softmax_apply_training(scores: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
    let (holds, predicate) = softmax_admission_predicate(scores, mask);
    let outcome = admit(
        crate::layer_norm::admission_mode(),
        "softmax_last_dim_fused",
        predicate,
        holds,
        &SOFTMAX_DISPATCH_COUNTERS,
    )?;
    match outcome {
        DispatchOutcome::Fused => Ok(apply2(
            scores,
            mask,
            SoftmaxLastDimFused::new(jammi_kernels::ops::FullyMaskedPolicy::Zeros),
        )?),
        DispatchOutcome::Eager => Ok(candle_nn::ops::softmax(
            &scores.broadcast_add(mask)?,
            D::Minus1,
        )?),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused GeGLU (C5)
// ─────────────────────────────────────────────────────────────────────────────

/// Fused/eager dispatch counters for the ModernBERT MLP's GeGLU
/// activation, mirroring `ROPE_DISPATCH_COUNTERS` /
/// `SOFTMAX_DISPATCH_COUNTERS` / `crate::layer_norm::LN_DISPATCH_COUNTERS`
/// — see `ModernBertMlp::forward`'s doc for the training-only gate this
/// counts. `pub(crate)` (not `pub`) — read via
/// [`crate::geglu_dispatch_snapshot`], the same shape the other three
/// snapshot functions use.
pub(crate) static GEGLU_DISPATCH_COUNTERS: DispatchCounters = DispatchCounters::new();

/// The fused GeGLU kernel's domain, checked at the call site (family D /
/// K2): `wi_out`'s device is one [`crate::layer_norm::device_is_supported`]
/// accepts, its dtype is one the kernel implements (F32 or BF16),
/// `wi_out` is contiguous ([`jammi_kernels::ops::GegluFused`] refuses a
/// strided view — see its module doc), and its last dimension is nonzero
/// and even (the op splits it into two equal `gate`/`up` halves; an odd
/// width is a structural domain violation the op itself also refuses, but
/// checking it here means it becomes a counted eager fallback instead of
/// a `candle_core::Error` surfacing from inside the op).
fn geglu_admission_predicate(wi_out: &Tensor) -> (bool, &'static str) {
    if !crate::layer_norm::device_is_supported(wi_out.device()) {
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
        crate::layer_norm::admission_mode(),
        "geglu_fused",
        predicate,
        holds,
        &GEGLU_DISPATCH_COUNTERS,
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
    ) -> Result<Tensor, EncoderError> {
        let after_attn = self.attention.forward(hidden, extended_mask, local_band)?;
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

    /// Run the encoder and return the raw last-layer hidden states
    /// `[batch, seq, hidden]`.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
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
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &extended, local_band.as_ref())?;
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
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask);
        assert!(holds, "CPU must satisfy the device clause: {predicate}");
    }

    /// The negative half: a `mask` dtype mismatched with `scores` is
    /// refused, not silently cast or truncated.
    #[test]
    fn softmax_admission_predicate_rejects_dtype_mismatch() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 4], (1, 1, 1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[bf16::from_f32(0.0); 4], (1, 1, 1, 4), &device).unwrap();
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask);
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
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask);
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
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask);
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
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask);
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
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask);
        assert!(!holds, "a broadcast-class violation must be refused");
        assert_eq!(predicate, "mask_broadcast_class");
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
        let (holds, predicate) = softmax_admission_predicate(&scores, &mask);
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
        let training_before = SOFTMAX_DISPATCH_COUNTERS.snapshot();
        let _ = softmax_apply_training(&scores, &mask).unwrap();
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
    /// fused-eligible on CPU) vs. the eager `broadcast_add` +
    /// `candle_nn::ops::softmax` composition, fwd AND bwd.
    #[test]
    fn fused_training_softmax_matches_eager_fwd_and_bwd() {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq = 4;
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
        let out_fused = softmax_apply_training(&s_fused, &mask_fused).unwrap();
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
        let out_eager =
            candle_nn::ops::softmax(&s_eager.broadcast_add(&mask_eager).unwrap(), D::Minus1)
                .unwrap();

        let vf: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ve: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs eager {e}");
        }

        let grads_fused = out_fused.backward().unwrap();
        let grads_eager = out_eager.backward().unwrap();
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
        for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dscores[{i}]: fused {f} vs eager {e}");
        }
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
}
