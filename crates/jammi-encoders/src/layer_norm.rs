//! LayerNorm whose backward is well-defined.
//!
//! In eval mode, delegates to candle's fused `crate::ops::layer_norm` for parity
//! with `candle_nn::LayerNorm`'s fast path. In training mode, composes the same
//! math out of primitive ops whose `bwd` is implemented, so gradient propagates
//! through to upstream trainable parameters. The two paths are algebraically
//! equivalent; FP rounding differs by ~1 ULP per accumulation.
//!
//! The fast path is only entered when `bias.is_some()` and the input is
//! contiguous, matching `candle_nn::LayerNorm`'s own entry conditions.
//!
//! ## The training path: `jammi_kernels::ops::LayerNormFused` /
//! `LayerNormBiasedFused` (#460, C-LN)
//!
//! A THIRD path exists, gated on `training == true`: every ModernBERT
//! LayerNorm (`ModernBertConfig` cannot even express a biased LayerNorm —
//! no `norm_bias` field exists) dispatches to the bias-free fused
//! CUDA/CPU kernel (`LayerNormFused`), and — since #460 — every
//! BERT/DistilBERT/CLIP-text LayerNorm (all of which carry a bias)
//! dispatches to the bias-carrying sibling (`LayerNormBiasedFused`),
//! instead of the `~12`-op eager composition below, when the respective
//! fused kernel's own domain holds (`x`'s device is CPU or CUDA — neither
//! op has a `metal_fwd`, and candle's default `metal_fwd` ERRORS rather
//! than falling back, so a Metal tensor is refused by this predicate
//! rather than reaching `apply2`/`apply3` and hard-erroring; dtype
//! F32/BF16/F16 matching between `x` and `weight` — F16 widened in
//! campaign #443 W2b, exactly where `jammi_kernels::cuda::layer_norm`
//! gained a compiled F16 dispatch arm (K2's no-Hold-without-dispatch
//! rule); both contiguous; `hidden` within the kernel's ceiling; for the
//! biased case, `bias` ADDITIONALLY matching `x`'s dtype, contiguous, and
//! `[hidden]`-shaped — see [`fused_admission_predicate_biased`]). BOTH
//! variants dispatch through the SAME admission key
//! (`"layer_norm_fused"`) and the SAME [`LN_DISPATCH_COUNTERS`] pair —
//! one LayerNorm-site capability, bias presence is tensor STATE decided
//! at the call site, never a model-family branch. Outside the respective
//! domain — or on the `parity-test` path, or in eval — `slow()` runs
//! exactly as before. This is a K2 "validate, don't silently degrade"
//! admission check: the fused/eager decision is recorded and a failed
//! predicate either falls back with a log-once WARN or, in `Strict` mode
//! ([`admission_mode`]), errors instead of silently falling back.
//!
//! **Advisory (round 1 of #460's fix, pressure-test finding):** under
//! `Strict` mode, a Metal `x` tensor now surfaces as a typed
//! `EncoderError::Kernel(StrictModeFallback)` from `forward()` itself,
//! where it previously (in the only mode this crate exercised end-to-end
//! before this round) silently took `slow()` — Metal fails
//! `device_is_supported` unconditionally, so its own admission predicate
//! never holds, and `Strict` mode's whole POINT is to refuse a
//! non-holding predicate rather than degrade quietly. This was always
//! `admit()`'s documented behavior, but nothing end-to-end (through
//! `LayerNorm::forward`, not `admit()` directly) proved it until
//! `tests::layer_norm_forward_biased_strict_mode_surfaces_a_typed_error_in_a_fresh_process`
//! — a maintainer adding real Metal support to this op later should
//! expect `Strict` mode to reject it exactly like any other failed
//! predicate, not to silently fall back the way `Fallback` mode (the
//! default) does.
//!
//! Before #460, every biased LayerNorm (BERT, DistilBERT, CLIP-text)
//! trained through `slow()` unconditionally, with NO `admit()` call and NO
//! dispatch counter at all — a BERT finetune run's own `ln` counter pair
//! read `0/0` regardless of how many LayerNorms it actually ran. #460's
//! ONLY change to the eval path's own call SHAPE is none at all: eval
//! (`training == false`) still NEVER reaches the fused arm for ANY value
//! of `bias` — the `forward` match's `(Some(bias), false) if
//! x.is_contiguous()` and catch-all `_ => self.slow(x)` arms are
//! byte-for-byte the pre-#460 code. Eval/serving numerics are therefore
//! bit-identical before/after THIS ARM'S ADDITION (see this module's own
//! `tests::eval_mode_forward_is_bit_identical_regardless_of_fused_eligibility`).
//! This is NOT the same claim as "eval/serving output is unaffected by
//! every change in this file": eval structurally falls through to
//! `slow()` unchanged in call SHAPE, but `slow()`'s own internals are not
//! frozen by this doc section — the round-once and reciprocal-vs-division
//! fixes documented at [`LayerNorm::slow`]'s own doc (below) DO change
//! eval/serving's bitwise output, on reduced-precision (BF16/F16)
//! backbones and at F32 respectively, precisely BECAUSE eval reaches the
//! same (changed) `slow()` code path, not in spite of it.
//!
//! `dgamma_needed`/`dbeta_needed` are computed via [`affine_needed_gate`]
//! at every fused-path call — NOT a hardcoded `false`, and (since #460)
//! NOT a bare `is_variable()` either. `is_variable()` alone is unsound as
//! a general "does this need a gradient" predicate (see
//! `jammi_kernels::ops::layer_norm`'s module doc: it is two-state over a
//! three-state lattice, and cannot tell a true external constant apart
//! from an INTERMEDIATE on a path to a `Var`) — that hazard does not
//! apply to `weight`/`bias` here in PRACTICE (both are a `LayerNorm`'s own
//! leaf module parameters, loaded straight from a `VarBuilder` with no
//! upstream op, never an intermediate produced by composing other
//! tensors — today: never a `Var`, in this crate; only LoRA A/B are
//! trainable), but [`affine_needed_gate`] makes that a CHECKED invariant
//! (a typed refusal on the one state that WOULD be ambiguous — a tracked,
//! non-`Var` intermediate) rather than an assumption relying on
//! candle's own backward walk to panic loudly (`grad not populated`,
//! `backprop.rs:175`) if the assumption were ever wrong — the same
//! three-way policy `jammi_lora::lora_linear::frozen_weight_gate` applies
//! to a LoRA base's own weight/bias.

use std::sync::LazyLock;

use candle_core::{DType, Tensor, D};
use candle_nn::{Init, VarBuilder};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
};
use jammi_kernels::ops::{apply2, apply3, LayerNormBiasedFused, LayerNormFused, MAX_HIDDEN};

use crate::error::EncoderError;

/// Per-op fused/eager dispatch counts for the bias-free training
/// LayerNorm, read from `jammi_kernels::admission`'s op-keyed registry
/// (`counters_for`) rather than a directly-owned `static DispatchCounters`
/// — this crate's C2-C5 four ops (this one plus RoPE/softmax/GeGLU in
/// `crate::modernbert`) were the registry's pre-existing hand-declared
/// statics; migrating them here is what makes the registry the SOLE
/// source of dispatch counters crate-wide (`jammi-lora`'s LoRA-site ops
/// already used it from the start — see `jammi_kernels::admission`'s
/// module doc). A `LazyLock`, not a plain `fn`, so `LN_DISPATCH_COUNTERS`
/// stays a `static` item: `crate::ln_dispatch_snapshot` (`lib.rs`, shared
/// class, not touched by this migration) calls
/// `layer_norm::LN_DISPATCH_COUNTERS.snapshot()` — a bare path followed by
/// a method call — which keeps compiling unchanged against a `LazyLock`
/// (auto-deref resolves `.snapshot()` through it to
/// `DispatchCounters::snapshot`) but would NOT compile against a renamed
/// function (`LN_DISPATCH_COUNTERS().snapshot()` is a different call
/// shape). This static itself is `pub` but lives inside a crate-private
/// module (`mod layer_norm;` in `lib.rs`) — unnameable from outside this
/// crate; `crate::ln_dispatch_snapshot` is the actual public read API a
/// durable job record or a bench report uses.
pub static LN_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("layer_norm_fused"));

/// Test-only serialization for two-sided (`fused` advanced AND `eager`
/// unchanged) assertions against [`LN_DISPATCH_COUNTERS`]: it is one
/// process-wide static shared by every `#[test]` in this crate's unit-test
/// binary (`src/layer_norm.rs`'s own `mod tests` and `src/clip_text.rs`'s),
/// so an exact-equality read of its `eager` half is racy under the default
/// parallel test runner unless the read is exclusive. Mirrors
/// `crate::modernbert::DISPATCH_COUNTER_TEST_LOCK`'s SAME rationale for the
/// SEPARATE `tests/it` integration binary — that lock lives in a different
/// process and cannot serialize this one.
#[cfg(test)]
pub(crate) static DISPATCH_COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The fused kernel's domain, checked at the call site (family D / K2):
/// `x` and `weight` live on a device [`device_is_supported`] accepts,
/// share a dtype the kernel implements (F32, BF16, or F16 — F16 widened
/// in campaign #443 W2b, exactly where `jammi_kernels::cuda::layer_norm`
/// gained a compiled F16 dispatch arm backed by the SEPARATE
/// `cuda/layer_norm_f16.cu` translation unit), both are
/// contiguous (`LayerNormFused` refuses a strided view rather than risk
/// misreading the row grouping — see its module doc), `weight` is rank-1
/// matching `x`'s last dimension, and that dimension is within the
/// kernel's `MAX_HIDDEN` ceiling (a conservative validated bound, not a
/// hardware limit — see `MAX_HIDDEN`'s own doc). Returns the aggregate
/// predicate and the name of whichever check is the reason (the first
/// one evaluated, or a fixed "domain_ok" name when everything holds) —
/// the failing name is what a Fallback-mode log line or a Strict-mode
/// error names.
fn fused_admission_predicate(x: &Tensor, weight: &Tensor) -> (bool, &'static str) {
    if !device_is_supported(x.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if x.dtype() != weight.dtype() || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return (false, "dtype_f32_bf16_or_f16_matching_between_x_and_weight");
    }
    if !x.is_contiguous() {
        return (false, "x_contiguous");
    }
    if !weight.is_contiguous() {
        return (false, "weight_contiguous");
    }
    let Some(&hidden) = x.dims().last() else {
        return (false, "x_rank_at_least_1");
    };
    if weight.dims() != [hidden] {
        return (false, "weight_rank1_matches_x_last_dim");
    }
    if hidden == 0 || hidden > MAX_HIDDEN {
        return (false, "hidden_within_kernel_max_hidden");
    }
    (true, "domain_ok")
}

/// #460 (C-LN): the bias-carrying sibling of [`fused_admission_predicate`].
/// `x`/`weight` share the identical domain [`fused_admission_predicate`]
/// already checks (reused here, not re-derived — a single definition of
/// "does this `x`/`weight` pair admit"), plus `bias`'s OWN checks: dtype
/// matching `x` (the SAME F32/BF16/F16 restriction), contiguous, and
/// `[hidden]`-shaped — the identical rule this file already applies to
/// `weight`. Distinct predicate REASON strings for the `bias`-specific
/// checks (`..._bias` suffixes) so a Fallback-mode log line or a
/// Strict-mode error names exactly which operand failed, never conflating
/// a `weight` domain failure with a `bias` one.
fn fused_admission_predicate_biased(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> (bool, &'static str) {
    let (holds, predicate) = fused_admission_predicate(x, weight);
    if !holds {
        return (holds, predicate);
    }
    if bias.dtype() != x.dtype() {
        return (false, "dtype_f32_bf16_or_f16_matching_between_x_and_bias");
    }
    if !bias.is_contiguous() {
        return (false, "bias_contiguous");
    }
    let Some(&hidden) = x.dims().last() else {
        return (false, "x_rank_at_least_1");
    };
    if bias.dims() != [hidden] {
        return (false, "bias_rank1_matches_x_last_dim");
    }
    (true, "domain_ok")
}

/// #460 (C-LN): the three-way gate `jammi-encoders`' call site uses for
/// BOTH a `LayerNorm`'s `weight` and (when present) its `bias` — the same
/// policy `jammi_lora::lora_linear::frozen_weight_gate` applies to a LoRA
/// base's own weight/bias: `is_variable()` (tried FIRST — a `Var` also
/// reports `track_op() == true`, candle-core 0.11's `Tensor::track_op` is
/// `is_variable() || op.is_some()`) means the parameter is a genuine
/// trainable `Var`; an UNTRACKED leaf (`!track_op()`, a parameter loaded
/// straight from a `VarBuilder` with no upstream op) means it is a true
/// frozen leaf; a TRACKED non-`Var` — neither definitely frozen nor
/// definitely trainable — is a typed refusal rather than a silent
/// `false`.
///
/// This replaces a bare `weight.is_variable()`/`bias.is_variable()` at the
/// call site with a CHECKED invariant instead of an assumption: both
/// `weight` and `bias` are structurally leaf module parameters (loaded
/// straight from a `VarBuilder`, never produced by composing other
/// tensors) in every production path this crate ships today, so
/// `is_variable() == false` always meant "true frozen leaf" in practice —
/// but `LayerNormFused`'s own module doc names exactly the silent-`None`
/// landmine a bare `is_variable()` leaves open for the future (a
/// tracked-but-not-`Var` intermediate would silently read as "frozen",
/// and `bwd` would return `None` for a slot candle's own backward walk
/// later expects populated, panicking loudly at `grad not populated`
/// rather than training a grad-less parameter — a safe failure mode, but
/// only because that panic exists; this gate makes the refusal typed and
/// immediate instead of waiting on that downstream panic).
fn affine_needed_gate(t: &Tensor, which: &'static str) -> Result<bool, EncoderError> {
    if t.is_variable() {
        Ok(true)
    } else if !t.track_op() {
        Ok(false)
    } else {
        Err(EncoderError::Config(format!(
            "LayerNorm: {which} is a TRACKED tensor (carries an Op) but is not a Var -- a \
             LayerNorm affine parameter must be either a true frozen leaf or an explicitly \
             trainable Var; a tracked non-Var {which} would silently lose its own gradient \
             contribution"
        )))
    }
}

/// Layer normalisation over the last dimension with optional affine bias.
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
    training: bool,
}

/// True when `prefix`'s last `.`-separated segment is literally
/// `LayerNorm` — candle-nn 0.11.0's `VarBuilder::prefix()` is
/// `self.path.join(".")` (`var_builder.rs:124-126`), so this checks the
/// segment after the final `.` (or the whole string when there is no
/// `.` at all — a `LayerNorm` loaded at a `VarBuilder`'s own root; no
/// PRODUCTION call site does this today (the seam test below,
/// `tests::layer_norm_new_call_sites_are_pinned_to_the_known_set`'s
/// sibling seam test, constructs one deliberately: `vb.pp("LayerNorm")` on
/// a root `VarBuilder` — a dotless prefix, `"LayerNorm"` with no `.`
/// segment before it — not a `.pp`-less builder), but the boundary is the
/// same either way). This is the ONLY gate on
/// whether [`LayerNorm::new`]
/// ever consults the legacy `gamma`/`beta` names: a prefix ending in
/// `...gamma_scale` or `...LayerNormX` does not match, and a
/// `<parent>.gamma` tensor sitting one level ABOVE a `<parent>.LayerNorm`
/// prefix is never probed by a `VarBuilder` rooted at `<parent>.LayerNorm`
/// (candle probes the full joined path, never a parent of it) — see
/// `esc-086`'s boundary arm.
fn is_layer_norm_keyed(prefix: &str) -> bool {
    prefix.rsplit('.').next() == Some("LayerNorm")
}

/// Resolves which literal tensor name each affine axis should be read
/// from, given which of the modern (`weight`/`bias`) and legacy
/// (`gamma`/`beta`) names [`LayerNorm::new`] found present at a
/// confirmed `LayerNorm`-keyed prefix (see [`is_layer_norm_keyed`]).
/// Pure and table-testable: no `VarBuilder`, no tensor I/O — just the
/// four presence booleans a caller already probed via
/// `VarBuilder::contains_tensor`.
///
/// Mirrors HF transformers v4.51.3's `_fix_state_dict_key_on_load`
/// (`modeling_utils.py:4504-4511`: a key ending `LayerNorm.beta` is
/// rewritten to end `LayerNorm.bias`, `LayerNorm.gamma` to
/// `LayerNorm.weight`, logged) and transformers `main`'s `"legacy"`
/// `WeightRenaming` block (`conversion_mapping.py:1399-1408`, the same
/// two literal suffix patterns) — narrowed here to a KEY-SCOPED alias
/// (only consulted when the prefix itself ends in `LayerNorm`) rather
/// than HF's whole-state-dict key rewrite, since this crate loads one
/// module at a time under an already-`.pp()`-scoped `VarBuilder`.
///
/// Compare candle-nn `main`'s own `layer_norm` (`candle-nn/src/layer_norm.rs:153-166`
/// as of this writing), which has a MODULE-scoped fallback: it prefers
/// `weight` when present and falls back to `gamma` only on load
/// failure, with no refusal if a checkpoint happens to carry both.
/// jammi deliberately diverges on two points: the narrower KEY scope
/// (mirroring HF's own suffix-anchored rule exactly, rather than a
/// bare "try weight, then try gamma" at module scope) and a LOUD
/// collision refusal (a checkpoint carrying both names is almost
/// always a corrupted or half-converted checkpoint, not a legitimate
/// ambiguity to silently resolve).
///
/// The weight axis is resolved FIRST — a double collision (both the
/// weight axis AND the bias axis carrying both their modern and legacy
/// names) therefore always reports the weight axis, deterministically,
/// never the bias axis. `with_bias == false` callers pass `has_b =
/// has_beta = false` (see [`LayerNorm::new`]) — this function then
/// never returns a bias name, matching `with_bias`'s pre-existing
/// `with_bias.then(..)` gate.
fn resolve_affine_names(
    prefix: &str,
    has_w: bool,
    has_g: bool,
    has_b: bool,
    has_beta: bool,
    with_bias: bool,
) -> Result<(&'static str, Option<&'static str>), EncoderError> {
    if has_w && has_g {
        return Err(EncoderError::Config(format!(
            "LayerNorm::new: prefix `{prefix}` carries BOTH `weight` and \
             the legacy `gamma` for its weight axis -- refusing the \
             collision rather than guessing which is authoritative; drop \
             one of the two tensors from the checkpoint (see \
             ci/scripts/perf/convert_legacy_bert_checkpoint.py, which has \
             the same output-name collision property)"
        )));
    }
    let name_w = if has_g { "gamma" } else { "weight" };

    let name_b = if with_bias {
        if has_b && has_beta {
            return Err(EncoderError::Config(format!(
                "LayerNorm::new: prefix `{prefix}` carries BOTH `bias` and \
                 the legacy `beta` for its bias axis -- refusing the \
                 collision rather than guessing which is authoritative; \
                 drop one of the two tensors from the checkpoint (see \
                 ci/scripts/perf/convert_legacy_bert_checkpoint.py, which \
                 has the same output-name collision property)"
            )));
        }
        Some(if has_beta { "beta" } else { "bias" })
    } else {
        None
    };

    Ok((name_w, name_b))
}

impl LayerNorm {
    /// Load a LayerNorm under `vb`'s current prefix. `weight` and (when
    /// `with_bias` is true) `bias` are read from the safetensors layout
    /// expected at that prefix; for a `VarMap`/`Zeros`-backed builder that
    /// has never seen the name before, absent tensors are initialised to
    /// ones and zeros respectively (`Init::Const`) — this is NOT true for
    /// the frozen `VarBuilder::from_mmaped_safetensors` builder every
    /// production loader uses (`bert.rs`, `distilbert.rs`, `modernbert.rs`,
    /// `clip_text.rs`, `open_clip_vision.rs`, `htsat_audio.rs`), whose
    /// `SafeTensorWithRouting`/mmaped backend has no such fallback and
    /// hard-errors instead — see `esc-086`.
    ///
    /// ## Legacy `LayerNorm.gamma`/`LayerNorm.beta` names (`esc-086`)
    ///
    /// Google's original BERT checkpoints (and any checkpoint still
    /// carrying those names) name a LayerNorm's affine parameters
    /// `gamma`/`beta` rather than the modern `weight`/`bias`. When `vb`'s
    /// own prefix's LAST `.`-segment is literally `LayerNorm` (see
    /// [`is_layer_norm_keyed`] — e.g. `bert.rs`'s
    /// `vb.pp("LayerNorm")`/`vb.pp("attention.output.LayerNorm")`/
    /// `vb.pp("output.LayerNorm")`, `distilbert.rs`'s analogous
    /// `emb_vb.pp("LayerNorm")`), this constructor also probes for
    /// `gamma`/`beta` and aliases them onto the same weight/bias slots
    /// (see [`resolve_affine_names`] for the full name-resolution
    /// lattice, including the loud collision refusal when a checkpoint
    /// carries both a modern and a legacy name for the same axis).
    ///
    /// EVERY OTHER call site — every `LayerNorm::new` whose prefix does
    /// NOT end in a literal `LayerNorm` segment: DistilBERT's
    /// `sa_layer_norm`/`output_layer_norm`, ModernBERT's
    /// `attn_norm`/`mlp_norm`/`model.embeddings.norm`/`model.final_norm`,
    /// CLIP's `ln_1`/`ln_2`/`ln_final`, open_clip's `ln_1`/`ln_2`/`ln_pre`/
    /// `ln_post`, HTSAT's `norm`/`layernorm_before`/`layernorm_after` — is
    /// BYTE-FOR-BYTE today's pre-existing code path: no `gamma`/`beta`
    /// probe, no `contains_tensor` call at all, only ever `weight`/`bias`.
    ///
    /// This is a CHECKED invariant (test:
    /// `tests::layer_norm_new_call_sites_are_pinned_to_the_known_set`), not
    /// an assumption written by hand into this comment and left to drift:
    /// that test scans this crate's own `src/**/*.rs` for every literal
    /// `LayerNorm::new(` occurrence — EXCLUDING this file (`layer_norm.rs`)
    /// itself, whose own source text spells out that search pattern and
    /// this scan's own diagnostic messages (see
    /// `scan_layer_norm_new_call_sites`'s own doc for why: this file's 3
    /// `#[cfg(test)]`-module seam-test call sites (all three inside
    /// `direct_seam_non_layer_norm_keyed_prefix_containing_layer_norm_substring_is_not_aliased`
    /// — `vb.pp("sa_layer_norm")`, `.pp("LayerNormX")`, `.pp("LayerNorm")`)
    /// are therefore NOT part of the 26-occurrence count below — and pins
    /// the exact set. As of this
    /// writing there are 26 occurrences total — 22 production call sites
    /// (`bert.rs` 3, `distilbert.rs` 3, `modernbert.rs` 4, `clip_text.rs` 3,
    /// `open_clip_vision.rs` 4, `htsat_audio.rs` 5) plus 4 inside
    /// `#[cfg(test)]` modules — of which exactly 4 PRODUCTION sites are
    /// `LayerNorm`-keyed: `bert.rs`'s `.pp("LayerNorm")`,
    /// `.pp("attention.output.LayerNorm")`, `.pp("output.LayerNorm")`, and
    /// `distilbert.rs`'s `.pp("LayerNorm")`. The ONLY bare-`vb` call site
    /// (no `.pp(..)` at all) anywhere in the crate is
    /// `modernbert.rs:4216`, a `#[cfg(test)]`-gated `VarMap`-backed fixture
    /// (`final_norm_of_an_all_zero_row_is_exactly_zero`); `modernbert.rs`'s
    /// OTHER three test-mod sites (`9421`–`9423`) are `.pp("emb_norm")`/
    /// `.pp("final_norm")`/`.pp("mlp_norm")`, not bare — none of the four
    /// are `LayerNorm`-keyed, and none reach the alias branch. No
    /// `VarBuilder::zeros()` construction exists anywhere in this
    /// workspace today — the `VarMap`-backed test fixtures above are the
    /// only non-frozen builders in the crate, and they are safe from a
    /// `weight`+`gamma` collision for the mundane reason their prefixes are
    /// never `LayerNorm`-keyed in the first place, not because of any
    /// `Zeros`-backend `contains_tensor` behavior (a previous version of
    /// this doc cited that as the reason; it does not apply to any
    /// in-tree builder).
    ///
    /// A `<parent>.gamma` tensor sitting one level ABOVE a
    /// `<parent>.LayerNorm` prefix is never aliased into it (candle
    /// probes the full joined path, never a parent of it — a legacy
    /// name under a non-`LayerNorm`-keyed prefix is likewise never
    /// aliased, matching HF's own suffix-anchored scoping).
    ///
    /// Compare candle-nn `main`'s own `layer_norm` helper
    /// (`candle-nn/src/layer_norm.rs:153-166` as of this writing): it
    /// has a MODULE-scoped fallback (no `LayerNorm`-suffix gate) that
    /// prefers `weight` and falls back to `gamma` on load failure, with
    /// no refusal when both are present. jammi's narrower key scope
    /// mirrors HF's own rule exactly; its loud collision refusal never
    /// silently picks a winner. See [`resolve_affine_names`]'s own doc
    /// for the full citation set (HF transformers `v4.51.3`
    /// `modeling_utils.py:4504-4511`; transformers `main`'s `"legacy"`
    /// `WeightRenaming`, `conversion_mapping.py:1399-1408`).
    pub fn new(
        hidden_size: usize,
        eps: f64,
        with_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self, EncoderError> {
        let prefix = vb.prefix();
        if !is_layer_norm_keyed(&prefix) {
            // Byte-for-byte with pre-existing behavior: no `contains_tensor`
            // probe at all, no alias -- only a prefix whose last segment is
            // literally `LayerNorm` ever consults `gamma`/`beta` (see this
            // fn's own doc for the full list of untouched call sites).
            let weight = vb.get_with_hints(hidden_size, "weight", Init::Const(1.0))?;
            let bias = with_bias
                .then(|| vb.get_with_hints(hidden_size, "bias", Init::Const(0.0)))
                .transpose()?;
            return Ok(Self {
                weight,
                bias,
                eps,
                training: false,
            });
        }

        // `with_bias == false` never even calls `contains_tensor("beta")`
        // -- not merely "ignores the result" -- matching the pre-existing
        // `with_bias.then(..)` gate below.
        let has_w = vb.contains_tensor("weight");
        let has_g = vb.contains_tensor("gamma");
        let (has_b, has_beta) = if with_bias {
            (vb.contains_tensor("bias"), vb.contains_tensor("beta"))
        } else {
            (false, false)
        };
        let (name_w, name_b) =
            resolve_affine_names(&prefix, has_w, has_g, has_b, has_beta, with_bias)?;

        let weight = vb
            .get_with_hints(hidden_size, name_w, Init::Const(1.0))
            .map_err(|e| {
                EncoderError::Config(format!(
                    "LayerNorm::new: failed to load the weight axis at prefix \
                     `{prefix}` (tried modern `weight` and legacy `gamma`): {e}"
                ))
            })?;
        let bias = match name_b {
            None => None,
            Some(name_b) => Some(
                vb.get_with_hints(hidden_size, name_b, Init::Const(0.0))
                    .map_err(|e| {
                        EncoderError::Config(format!(
                            "LayerNorm::new: failed to load the bias axis at \
                             prefix `{prefix}` (tried modern `bias` and legacy \
                             `beta`): {e}"
                        ))
                    })?,
            ),
        };

        Ok(Self {
            weight,
            bias,
            eps,
            training: false,
        })
    }

    /// Switch between the fused eval forward and the gradient-carrying training
    /// forward.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// `[..., hidden] -> [..., hidden]`.
    ///
    /// #460 (C-LN): the `(Some(bias), true)` arm — every BERT/DistilBERT/
    /// CLIP-text LayerNorm — now ALSO dispatches through
    /// [`Self::forward_fused_or_fallback`], exactly like the bias-free
    /// `(None, true)` arm already did: bias presence is tensor STATE
    /// passed down to the fused-or-fallback decision, never a
    /// model-family branch in `forward` itself. Only `(_, false)` (eval)
    /// and `(None, false)` bias-free eval fall through to the pre-existing
    /// arms, byte-for-byte unchanged.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        match (&self.bias, self.training) {
            (Some(bias), false) if x.is_contiguous() => Ok(candle_nn::ops::layer_norm(
                x,
                &self.weight,
                bias,
                self.eps as f32,
            )?),
            (None, true) => self.forward_fused_or_fallback(x, None),
            (Some(bias), true) => self.forward_fused_or_fallback(x, Some(bias)),
            _ => self.slow(x),
        }
    }

    /// The training-mode arm, bias-free OR bias-carrying: dispatches to
    /// [`LayerNormFused`] (`bias.is_none()`) or `LayerNormBiasedFused`
    /// (`bias.is_some()`) when the respective domain holds, else falls
    /// back to [`Self::slow`] (recording which happened either way, under
    /// the SAME admission key `"layer_norm_fused"` — one LayerNorm-site
    /// capability, bias is tensor state, not a second key). See this
    /// module's doc for the full design and [`affine_needed_gate`] for why
    /// `dgamma_needed`/`dbeta_needed` are no longer a bare
    /// `is_variable()`.
    fn forward_fused_or_fallback(
        &self,
        x: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let (holds, predicate) = match bias {
            None => fused_admission_predicate(x, &self.weight),
            Some(b) => fused_admission_predicate_biased(x, &self.weight, b),
        };
        // Gate ordering (family D / adversarial advisory 6): evaluate
        // `affine_needed_gate` for weight (and bias, when present) BEFORE
        // `admit()`. A tracked-but-not-`Var` affine parameter is a typed
        // refusal (`EncoderError::Config`, not a panic) — if that refusal
        // fired AFTER `admit()` already recorded a `Fused` dispatch, the
        // `ln` fused counter would advance for a call that never actually
        // ran the kernel: a phantom dispatch a counter-based dispatch
        // assertion cannot distinguish from a real one. Evaluating first
        // and propagating via `?` means a refusal here never touches
        // [`LN_DISPATCH_COUNTERS`] at all.
        let dgamma_needed = affine_needed_gate(&self.weight, "weight")?;
        let dbeta_needed = match bias {
            Some(b) => Some(affine_needed_gate(b, "bias")?),
            None => None,
        };
        let outcome = admit(
            admission_mode(),
            "layer_norm_fused",
            predicate,
            holds,
            *LN_DISPATCH_COUNTERS,
        )?;
        match outcome {
            DispatchOutcome::Fused => match bias {
                None => Ok(apply2(
                    x,
                    &self.weight,
                    LayerNormFused::new(self.eps, dgamma_needed),
                )?),
                Some(b) => Ok(apply3(
                    x,
                    &self.weight,
                    b,
                    LayerNormBiasedFused::new(
                        self.eps,
                        dgamma_needed,
                        dbeta_needed
                            .expect("bias.is_some() above means dbeta_needed was computed as Some"),
                    ),
                )?),
            },
            DispatchOutcome::Eager => self.slow(x),
        }
    }

    /// `y = xhat * gamma [+ beta]`, matching torch's `layer_norm_cuda`,
    /// PINNED to torch 2.13.0
    /// (`aten/src/ATen/native/cuda/layer_norm_kernel.cu`'s
    /// `vectorized_layer_norm_kernel_impl`: "Computation is performed in
    /// T_ACC, X is cast to T_ACC and result is implicitly cast to T" —
    /// `out = gamma * (rstd * (x - mean)) + beta`, gamma AND beta both
    /// applied in the f32 accumulator before the SINGLE implicit cast to
    /// the output dtype) and jammi's own fused CUDA kernel
    /// (`cuda/layer_norm.cu:124`: `yr[i] =
    /// __float2bfloat16(xhat * __bfloat162float(gamma[i]))`, one
    /// `__float2bfloat16` call, at the end). `mean`/`variance`/`xhat`/the
    /// affine are ALL computed in `internal_dtype` (f32 whenever `x_dtype`
    /// is F16/BF16); `weight`/`bias` are upcast to `internal_dtype` for the
    /// affine rather than mixing dtypes, and the whole result is cast to
    /// `x_dtype` exactly once, at the very end.
    ///
    /// Previously this rounded `xhat` to `x_dtype` BEFORE multiplying by
    /// `weight` (and, when biased, added `bias` as a further `x_dtype`
    /// op) — two-to-three rounding points instead of one. A measured,
    /// non-vacuous divergence at production shape (`hidden=1024`,
    /// `batch=2`, `seq` in `{128, 512}`) is the RED control in
    /// `tests::layer_norm_slow_matches_truth_at_production_shape_seq128`/
    /// `_seq512` — see those tests' own printed mismatch counts for a
    /// reproducible figure (no number is hardcoded here; the committed
    /// test is the producer). This divergence is only OBSERVABLE where
    /// `internal_dtype != x_dtype` (an F16/BF16 backbone; F32/F64 make
    /// every `to_dtype` call below a same-dtype no-op) — but that is a
    /// DTYPE gate, not a training-vs-eval one. `forward` (above) names
    /// only two arms explicitly: `(Some(bias), false) if
    /// x.is_contiguous()` (candle's fused biased-eval fast path,
    /// `candle_nn::ops::layer_norm`, which already rounded once and so was
    /// never affected by this defect) and `(None, true)` (the fused-kernel
    /// training arm, which itself falls back to THIS function outside the
    /// fused domain). EVERY OTHER `(bias, training)` combination —
    /// `(None, false)`, bias-free EVAL, included — falls through the
    /// catch-all `_ => self.slow(x)`. Every ModernBERT LayerNorm is
    /// bias-free (`ModernBertConfig` has no `norm_bias` field), so
    /// ModernBERT's own eval/serving forward pass reaches `slow()` too,
    /// not only its training paths. Every served bias-free (ModernBERT)
    /// LayerNorm output on an F16/BF16 backbone — training-eager fallback,
    /// any `training=true` call that misses the fused kernel's admission
    /// domain, AND eval/serving itself (through this same catch-all) —
    /// therefore changes at the ULP level; F32-backbone serving is
    /// UNCHANGED BY THIS SPECIFIC DEFECT (`internal_dtype == x_dtype`
    /// there, so every `to_dtype` call below is a same-dtype no-op) — but
    /// see the SECOND, orthogonal divergence below, which is NOT
    /// dtype-scoped this way and DOES change F32 (and F64) output, on
    /// every path that reaches `slow()`, eval/serving included. The ONLY
    /// case this fix changes neither in call SHAPE nor in numerics is the
    /// biased, contiguous, eval fast path — but no ModernBERT LayerNorm is
    /// ever biased, so that carve-out never covers ModernBERT.
    ///
    /// A SECOND rounding-placement divergence, orthogonal to the one
    /// above: this function previously computed `centered.broadcast_div(&
    /// sqrt(variance + eps))` — a DIVISION — where torch's `rstd *`
    /// (quoted above), the fused CPU arm's `1.0 / sqrt(..)` multiply, and
    /// the fused CUDA arm's `rsqrtf` all take the RECIPROCAL first and
    /// MULTIPLY. Division and multiply-by-reciprocal are not bit-identical
    /// in floating point (the reciprocal is itself a rounded value, so
    /// `a / b` and `a * (1/b)` can round differently). This function now
    /// computes `(variance + eps).sqrt().recip()` and multiplies, matching
    /// every other placement's form.
    ///
    /// UNLIKE the double-rounding defect above, this placement change is
    /// NOT gated on `internal_dtype != x_dtype`: the `rstd` line runs
    /// identically regardless of dtype, so it changes output at EVERY
    /// dtype `slow()` supports, F32 and F64 included — the "F32-backbone
    /// serving is UNCHANGED" claim two paragraphs up applies ONLY to the
    /// double-rounding fix, not to this one. At F32, where
    /// `internal_dtype == x_dtype` makes every OTHER change in this
    /// function a same-dtype no-op, this `rstd` line is consequently the
    /// ONLY source of `slow()`'s F32 output changing at all — and the
    /// effect is large, not a stray ULP: on the same production-shape
    /// fixture — see `tests::slow_f32_reciprocal_form_is_bit_exact_and_diverges_from_division`,
    /// which measures live (`rows=256, hidden=1024`, `n=262144`), the division
    /// form disagrees with the reciprocal form on `74734/262144`
    /// elements — see that test's own printed count. Since
    /// bias-free eval (the ModernBERT serving path) reaches `slow()`
    /// through the catch-all named above, this is F32 ModernBERT's SERVED
    /// EMBEDDING output changing bitwise on `74734/262144` elements at
    /// this production shape — TOWARD torch's own reciprocal-then-multiply
    /// placement, away from the division form this line replaces. On the
    /// bf16/f16 arms, where `internal_dtype == F32` regardless of this
    /// fix, this SAME placement change is a much smaller, budget-visible
    /// effect: `tests::layer_norm_slow_matches_truth_at_production_shape_seq128`/
    /// `_seq512` (`REDUCTION_ORDER_BUDGET_FRACTION`'s doc) print BOTH
    /// `slow()`'s real reciprocal-form output AND a same-candle-fold
    /// (`sum_keepdim`) division-form comparator against the same scalar
    /// truth on every run, and assert the reciprocal form is NOT WORSE
    /// than that division form (`reciprocal-count <= division-count`) —
    /// see those tests' own printed pair for the live figures. Sharing
    /// `slow()`'s own reduction (rather than a hand-rolled scalar-loop
    /// division form) is what makes the two counts commensurable: any
    /// residual difference between them is attributable to the
    /// reciprocal-vs-division placement alone, not to a fold-order
    /// mismatch between a scalar loop and candle's SIMD-lane reduction —
    /// the F32 test above remains what actually discriminates this
    /// placement bit-exactly.
    ///
    /// Domain check (K2): `weight`'s (and, when biased, `bias`'s) dtype
    /// must match `x`'s own dtype — mirroring only the MATCHING half of
    /// `fused_admission_predicate`'s
    /// `dtype_f32_bf16_or_f16_matching_between_x_and_weight` check above,
    /// not its F32/BF16/F16 restriction: `slow()` is the fallback path for
    /// EVERY dtype `internal_dtype`'s match arm above accepts (F64
    /// included, not just F32/BF16/F16 — the fused kernel's tighter dtype
    /// domain does not apply here), so it only refuses a MISMATCH, never
    /// a dtype outside `{F32, BF16, F16}`. Before this check existed, a caller
    /// passing a mismatched-dtype
    /// weight got candle's own `broadcast_mul` dtype-mismatch error (the
    /// pre-fix code multiplied at `x_dtype` directly); the internal-dtype
    /// upcast this fix introduces (`weight.to_dtype(internal_dtype)`)
    /// would otherwise silently accept ANY weight dtype and produce a
    /// confident wrong number instead — a real domain-widening
    /// regression the fix must not introduce. See
    /// `tests::slow_refuses_a_dtype_mismatched_weight_instead_of_silently_upcasting`.
    fn slow(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let x_dtype = x.dtype();
        if self.weight.dtype() != x_dtype {
            return Err(EncoderError::Config(format!(
                "LayerNorm::slow: weight dtype {:?} does not match x dtype {:?} -- refusing \
                 rather than silently upcasting a mismatched-dtype weight into `internal_dtype` \
                 (mirrors only the MATCHING half of `fused_admission_predicate`'s \
                 `dtype_f32_bf16_or_f16_matching_between_x_and_weight` check -- slow() itself \
                 accepts any dtype `internal_dtype` handles, not just F32/BF16/F16)",
                self.weight.dtype(),
                x_dtype
            )));
        }
        if let Some(b) = &self.bias {
            if b.dtype() != x_dtype {
                return Err(EncoderError::Config(format!(
                    "LayerNorm::slow: bias dtype {:?} does not match x dtype {:?} -- same \
                     domain-validity refusal as the weight check above",
                    b.dtype(),
                    x_dtype
                )));
            }
        }
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden = x.dim(D::Minus1)?;
        let x_internal = x.to_dtype(internal_dtype)?;
        let mean = (x_internal.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let centered = x_internal.broadcast_sub(&mean)?;
        let variance = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let rstd = (variance + self.eps)?.sqrt()?.recip()?;
        let normalized = centered.broadcast_mul(&rstd)?;
        let weight_internal = self.weight.to_dtype(internal_dtype)?;
        let scaled_internal = normalized.broadcast_mul(&weight_internal)?;
        let out_internal = match &self.bias {
            None => scaled_internal,
            Some(b) => scaled_internal.broadcast_add(&b.to_dtype(internal_dtype)?)?,
        };
        Ok(out_internal.to_dtype(x_dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};
    use half::{bf16, f16};

    fn bias_free_ln(weight: Tensor, eps: f64, training: bool) -> LayerNorm {
        LayerNorm {
            weight,
            bias: None,
            eps,
            training,
        }
    }

    fn biased_ln(weight: Tensor, bias: Tensor, eps: f64, training: bool) -> LayerNorm {
        LayerNorm {
            weight,
            bias: Some(bias),
            eps,
            training,
        }
    }

    /// The positive half of the device clause: CPU must satisfy it (every
    /// other test in this file relies on that implicitly; this pins it
    /// explicitly as its own assertion on the predicate's return value,
    /// not just "the forward call happened to succeed").
    #[test]
    fn fused_admission_predicate_accepts_cpu_device() {
        let device = Device::Cpu;
        let hidden = 4;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 4], (hidden,), &device).unwrap();
        let (holds, predicate) = fused_admission_predicate(&x, &weight);
        assert!(holds, "CPU must satisfy the device clause: {predicate}");
    }

    /// The domain-widening PROOF (K2): F16 must now HOLD, not just fail to
    /// error — campaign #443 W2b's CUDA F16 dispatch arm
    /// (`jammi_kernels::cuda::layer_norm`'s `(DType::F16, DType::F16)`
    /// arm) is what makes this admission-widening sound; before that arm
    /// existed, an F16 `x`/`weight` pair was correctly refused here
    /// (`dtype_f32_bf16_or_f16_matching_between_x_and_weight`, née
    /// `dtype_f32_or_bf16_matching_between_x_and_weight`) — this test
    /// pins the flip, not merely its absence.
    #[test]
    fn fused_admission_predicate_now_accepts_matching_f16() {
        let device = Device::Cpu;
        let hidden = 4;
        let xv: Vec<f16> = (0..hidden).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
        let wv: Vec<f16> = (0..hidden).map(|_| f16::from_f32(1.0)).collect();
        let x = Tensor::from_slice(&xv, (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&wv, (hidden,), &device).unwrap();
        let (holds, predicate) = fused_admission_predicate(&x, &weight);
        assert!(holds, "matching F16 x/weight must now hold: {predicate}");
        assert_eq!(predicate, "domain_ok");
    }

    // -----------------------------------------------------------------
    // #460 (C-LN): `fused_admission_predicate_biased` oracles.
    // -----------------------------------------------------------------

    #[test]
    fn fused_admission_predicate_biased_accepts_a_valid_fixture() {
        let device = Device::Cpu;
        let hidden = 4;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 4], (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&[0.5f32; 4], (hidden,), &device).unwrap();
        let (holds, predicate) = fused_admission_predicate_biased(&x, &weight, &bias);
        assert!(holds, "{predicate}");
        assert_eq!(predicate, "domain_ok");
    }

    #[test]
    fn fused_admission_predicate_biased_refuses_a_mismatched_bias_dtype() {
        let device = Device::Cpu;
        let hidden = 4;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 4], (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&[bf16::from_f32(0.5); 4], (hidden,), &device).unwrap();
        let (holds, predicate) = fused_admission_predicate_biased(&x, &weight, &bias);
        assert!(!holds);
        assert_eq!(
            predicate,
            "dtype_f32_bf16_or_f16_matching_between_x_and_bias"
        );
    }

    #[test]
    fn fused_admission_predicate_biased_refuses_a_bias_shape_mismatch() {
        let device = Device::Cpu;
        let hidden = 4;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 4], (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&[0.5f32; 3], (3,), &device).unwrap();
        let (holds, predicate) = fused_admission_predicate_biased(&x, &weight, &bias);
        assert!(!holds);
        assert_eq!(predicate, "bias_rank1_matches_x_last_dim");
    }

    #[test]
    fn fused_admission_predicate_biased_refuses_a_non_contiguous_bias() {
        let device = Device::Cpu;
        let hidden = 4;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 4], (hidden,), &device).unwrap();
        let base = Tensor::from_slice(
            &[0.1f32, 0.0, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0],
            (4, 2),
            &device,
        )
        .unwrap();
        let bias = base.narrow(1, 0, 1).unwrap().squeeze(1).unwrap();
        assert!(!bias.is_contiguous());
        let (holds, predicate) = fused_admission_predicate_biased(&x, &weight, &bias);
        assert!(!holds);
        assert_eq!(predicate, "bias_contiguous");
    }

    /// A `weight`/`bias` domain failure (from the reused
    /// `fused_admission_predicate` call) is reported with THAT check's own
    /// reason, not silently swallowed into a `bias`-specific one — proves
    /// the early-return-on-`!holds` composition actually short-circuits.
    #[test]
    fn fused_admission_predicate_biased_propagates_the_weight_domain_failure_reason() {
        let device = Device::Cpu;
        let hidden = 4;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, hidden), &device).unwrap();
        // weight dtype mismatched against x -- a `fused_admission_predicate`
        // failure, not a bias-specific one.
        let weight = Tensor::from_slice(&[bf16::from_f32(1.0); 4], (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&[0.5f32; 4], (hidden,), &device).unwrap();
        let (holds, predicate) = fused_admission_predicate_biased(&x, &weight, &bias);
        assert!(!holds);
        assert_eq!(
            predicate,
            "dtype_f32_bf16_or_f16_matching_between_x_and_weight"
        );
    }

    // -----------------------------------------------------------------
    // #460 (C-LN): `affine_needed_gate` — the three-way lattice.
    // -----------------------------------------------------------------

    #[test]
    fn affine_needed_gate_untracked_leaf_is_false() {
        let device = Device::Cpu;
        let t = Tensor::from_slice(&[1.0f32; 4], (4,), &device).unwrap();
        assert!(!t.is_variable());
        assert!(!t.track_op());
        assert!(!affine_needed_gate(&t, "weight").unwrap());
    }

    #[test]
    fn affine_needed_gate_var_is_true() {
        let device = Device::Cpu;
        let v =
            Var::from_tensor(&Tensor::from_slice(&[1.0f32; 4], (4,), &device).unwrap()).unwrap();
        assert!(affine_needed_gate(v.as_tensor(), "weight").unwrap());
    }

    /// A tracked-but-not-`Var` intermediate is a typed refusal, not a
    /// silent `false` — the exact landmine a bare `is_variable()` would
    /// leave open, mirroring `jammi_lora::lora_linear::frozen_weight_gate`'s
    /// own `tracked_non_var_is_a_typed_refusal` test. `BackpropOp::new1`
    /// (candle-core 0.11's `op.rs`) only attaches an `Op` when its OWN
    /// argument already `track_op()`s — a plain leaf's `to_dtype` would
    /// just produce another untracked leaf, not the ambiguous state this
    /// test targets; starting from a `Var` and casting to a DIFFERENT
    /// dtype produces a tensor that IS tracked (inherits `track_op()` from
    /// its `Var` input) but is itself NOT a `Var`.
    #[test]
    fn affine_needed_gate_tracked_non_var_is_a_typed_refusal_not_a_panic() {
        let device = Device::Cpu;
        let v =
            Var::from_tensor(&Tensor::from_slice(&[1.0f32; 4], (4,), &device).unwrap()).unwrap();
        let tracked = v.as_tensor().to_dtype(DType::F64).unwrap();
        assert!(!tracked.is_variable());
        assert!(tracked.track_op(), "fixture must actually be tracked");
        let err = affine_needed_gate(&tracked, "weight")
            .expect_err("a tracked non-Var affine parameter must be a typed refusal");
        assert!(matches!(err, EncoderError::Config(_)));
    }

    /// End-to-end (adversarial F5 / gate-ordering advisory 6): a
    /// tracked-but-not-`Var` gamma reaches [`LayerNorm::forward`]'s
    /// bias-free `(None, true)` fused-or-fallback arm on a fixture that
    /// WOULD satisfy the fused admission domain (bf16, contiguous, hidden
    /// well within `MAX_HIDDEN`) — the domain predicate alone would say
    /// "fused". The refusal must still surface as a typed
    /// `EncoderError::Config` from `LayerNorm::forward` itself, AND
    /// [`LN_DISPATCH_COUNTERS`] must be UNTOUCHED (both `fused` and
    /// `eager`) — proving `affine_needed_gate` is evaluated, and its error
    /// propagated, strictly BEFORE `admit()` ever runs, not merely that
    /// the call fails somewhere before returning a tensor. Before the
    /// gate-ordering fix this test proves, `admit()` ran FIRST and would
    /// have already recorded a `Fused` dispatch that never actually
    /// produced an output — a phantom dispatch this counter-based
    /// assertion is built to catch.
    #[test]
    fn tracked_non_var_gamma_through_forward_is_a_typed_refusal_with_counters_untouched() {
        let _guard = DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let hidden = 8;
        let gv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(1.0 + i as f32 * 0.05))
            .collect();
        let xv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(i as f32 * 0.37 - 1.2))
            .collect();
        let x = Tensor::from_slice(&xv, (1, hidden), &device).unwrap();

        // The SAME "cast to a DIFFERENT dtype, then back" idiom as
        // `affine_needed_gate_tracked_non_var_is_a_typed_refusal_not_a_panic`,
        // round-tripped back to bf16 so the RESULT dtype matches `x` and
        // the fused domain predicate reports eligible — isolating the
        // affine gate as the ONLY reason this call can fail.
        let weight_var =
            Var::from_tensor(&Tensor::from_slice(&gv, (hidden,), &device).unwrap()).unwrap();
        let tracked_f32 = weight_var.as_tensor().to_dtype(DType::F32).unwrap();
        let tracked_bf16 = tracked_f32.to_dtype(DType::BF16).unwrap();
        assert!(!tracked_bf16.is_variable(), "fixture must not be a Var");
        assert!(tracked_bf16.track_op(), "fixture must actually be tracked");
        assert_eq!(tracked_bf16.dtype(), DType::BF16);

        let (holds, predicate) = fused_admission_predicate(&x, &tracked_bf16);
        assert!(
            holds,
            "fixture must satisfy the fused domain: {predicate} -- the test proves the \
             affine gate refuses it anyway, not that the domain predicate happens to fail"
        );

        let mut ln = LayerNorm {
            weight: tracked_bf16,
            bias: None,
            eps: 1e-5,
            training: true,
        };
        ln.set_training(true);

        let before = LN_DISPATCH_COUNTERS.snapshot();
        let err = ln
            .forward(&x)
            .expect_err("a tracked non-Var gamma must be a typed refusal, not a silent dispatch");
        let after = LN_DISPATCH_COUNTERS.snapshot();
        assert!(matches!(err, EncoderError::Config(_)));
        assert_eq!(
            (after.fused, after.eager),
            (before.fused, before.eager),
            "the typed refusal must never touch either dispatch counter \
             (before={before:?}, after={after:?})"
        );
    }

    /// End-to-end: a trainable `Var` bias on a bias-carrying, otherwise
    /// fused-eligible training LayerNorm must populate a real `dbeta`
    /// gradient AND be counted on the SAME `ln` fused counter the
    /// bias-free path uses.
    #[test]
    fn biased_training_with_a_var_bias_counts_fused_and_populates_dbeta() {
        let _guard = DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let hidden = 4;
        let rows = 2;
        let xv: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.29 - 0.7).sin() * 1.5)
            .collect();
        let x =
            Var::from_tensor(&Tensor::from_slice(&xv, (rows, hidden), &device).unwrap()).unwrap();
        let weight = Tensor::from_slice(&[1.1f32, 0.9, 1.0, 1.2], (hidden,), &device).unwrap();
        let bias = Var::from_tensor(
            &Tensor::from_slice(&[0.1f32, -0.1, 0.2, -0.2], (hidden,), &device).unwrap(),
        )
        .unwrap();

        let ln = LayerNorm {
            weight,
            bias: Some(bias.as_tensor().clone()),
            eps: 1e-5,
            training: true,
        };
        let before = LN_DISPATCH_COUNTERS.snapshot();
        let out = ln.forward(x.as_tensor()).unwrap();
        let after = LN_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused && after.eager == before.eager,
            "a Var bias must not prevent the fused biased dispatch, and must never fall back \
             to eager (before={before:?}, after={after:?})"
        );

        let grads = out.sum_all().unwrap().backward().unwrap();
        let dbeta: Vec<f32> = grads
            .get(&bias)
            .expect("dbeta_needed must be true for a trainable Var bias")
            .to_vec1()
            .unwrap();
        // `.sum_all()`'s upstream gradient is all-1.0, so `dbeta_i =
        // sum_rows(1.0) = rows` for every column, exactly.
        assert_eq!(dbeta, vec![rows as f32; hidden]);
    }

    /// The NEGATIVE half: a Metal device must be REJECTED. This IS
    /// hermetically testable with no `metal` feature on this crate at
    /// all: `candle_core` re-exports a `MetalDevice` type at its crate
    /// root regardless of whether ITS `metal` feature is on — the real
    /// backend's type when it is, and a public, zero-field dummy-backend
    /// unit struct (`pub struct MetalDevice;`, `dummy_metal_backend.rs`)
    /// when it is off — so `Device::Metal(MetalDevice)` is constructible
    /// today, unconditionally, with a bare unit-struct literal. This
    /// crate has no `metal` feature to gate on (declaring an empty one
    /// just to `#[cfg]` against it would be a phantom feature, and
    /// `cfg(feature = "metal")` on an undeclared feature trips rustc's
    /// `unexpected_cfgs` lint under `-D warnings`), so this test is
    /// unconditional: it exercises the dummy backend's zero-field
    /// `MetalDevice`, which is what candle-core actually compiles here.
    /// If this crate ever gains a real `metal` feature, `MetalDevice`
    /// becomes the real (non-unit) backend type and THIS specific
    /// construction stops compiling — a loud compile error flagging the
    /// exact test that needs replacing with a real Metal device/ordinal,
    /// not a silently-stale green test.
    #[test]
    fn device_is_supported_rejects_metal() {
        let metal = Device::Metal(candle_core::MetalDevice);
        assert!(
            !device_is_supported(&metal),
            "Metal must be rejected: LayerNormFused has no metal_fwd, and \
             candle's default metal_fwd errors rather than falling back"
        );
    }

    /// The eval-path bit-identity requirement: a `(bias.is_none(),
    /// training == false)` forward must be UNCHANGED by the fused
    /// kernel's existence, even on an input/weight pair that WOULD
    /// satisfy the fused admission domain if `training` were `true`
    /// (bf16, contiguous, `hidden` well within `MAX_HIDDEN`) — proving
    /// eval never reaches [`LayerNorm::forward_fused_or_fallback`]
    /// because the `match` in `forward` structurally routes it to
    /// `slow()`, not merely because this particular fixture happens to
    /// fail the domain check.
    #[test]
    fn eval_mode_forward_is_bit_identical_regardless_of_fused_eligibility() {
        // The `set_training(true)` forward below (see "Exercise the fused
        // arm" further down) bumps `LN_DISPATCH_COUNTERS` even though this
        // test never reads it — same lock discipline as every other
        // training-forward test in this module (see
        // `DISPATCH_COUNTER_TEST_LOCK`'s own doc).
        let _guard = DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let hidden = 8;
        let xv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(i as f32 * 0.37 - 1.2))
            .collect();
        let gv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(1.0 + i as f32 * 0.05))
            .collect();
        let x = Tensor::from_slice(&xv, (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&gv, (hidden,), &device).unwrap();
        assert!(x.is_contiguous());
        assert!(weight.is_contiguous());
        let (holds, _) = fused_admission_predicate(&x, &weight);
        assert!(
            holds,
            "fixture must satisfy the fused domain — the test proves eval \
             skips it anyway, not that the fixture happens to be ineligible"
        );

        let mut ln = bias_free_ln(weight, 1e-5, false);
        let before: Vec<Vec<bf16>> = ln.forward(&x).unwrap().to_vec2().unwrap();

        // Exercise the fused arm (this binary now has one) without
        // changing the eval call itself.
        ln.set_training(true);
        let _ = ln.forward(&x).unwrap();
        ln.set_training(false);

        let after: Vec<Vec<bf16>> = ln.forward(&x).unwrap().to_vec2().unwrap();
        assert_eq!(
            before, after,
            "eval-mode (training=false) forward must be byte-identical \
             before and after the fused kernel exists"
        );

        // And it is exactly `slow()` — eval's real, unchanged code path.
        let via_slow: Vec<Vec<bf16>> = ln.slow(&x).unwrap().to_vec2().unwrap();
        assert_eq!(before, via_slow);
    }

    /// #460 (C-LN): the biased path (BERT/DistilBERT) now DOES reach the
    /// fused arm in training mode — `forward`'s `(Some(bias), true)` arm
    /// dispatches through [`Self::forward_fused_or_fallback`] exactly like
    /// the bias-free `(None, true)` arm always has (one common
    /// architecture; bias presence is tensor state, not a model-family
    /// carve-out). This fixture (F32, contiguous, `hidden = 8`) satisfies
    /// [`fused_admission_predicate_biased`]'s domain, so `out_training`
    /// must be the FUSED kernel's own output — close to, but no longer
    /// bit-identical to, `slow()`'s (rule 15: the CPU biased fused row
    /// loop's reduction order differs from `slow()`'s candle-composed
    /// fold) — proved by a monotonic `LN_DISPATCH_COUNTERS` delta rather
    /// than an exact-equality claim (see
    /// `fused_training_path_matches_slow_within_tolerance_fwd_and_bwd`'s
    /// identical rationale for why `LN_DISPATCH_COUNTERS` is a
    /// process-wide static under parallel `cargo test`).
    ///
    /// EVAL is UNCHANGED: `(Some(bias), false)` still matches `forward`'s
    /// first arm (`candle_nn::ops::layer_norm` directly), byte-for-byte
    /// the pre-#460 code path — pinned here exactly, not within a
    /// tolerance.
    #[test]
    fn biased_layer_norm_training_now_dispatches_fused_eval_is_unaffected() {
        let _guard = DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let hidden = 8;
        let weight = Tensor::from_slice(&[1.3f32; 8], (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&[0.2f32; 8], (hidden,), &device).unwrap();
        let x = Tensor::from_slice(
            &[0.5f32, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 0.75],
            (1, hidden),
            &device,
        )
        .unwrap();

        let mut ln = LayerNorm {
            weight: weight.clone(),
            bias: Some(bias.clone()),
            eps: 1e-5,
            training: true,
        };
        let (holds, predicate) = fused_admission_predicate_biased(&x, &weight, &bias);
        assert!(
            holds,
            "fixture must satisfy the biased fused domain: {predicate}"
        );

        let before = LN_DISPATCH_COUNTERS.snapshot();
        let out_training: Vec<f32> = ln
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let after = LN_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused && after.eager == before.eager,
            "biased training must dispatch the fused kernel, and never fall back to eager \
             (before={before:?}, after={after:?})"
        );

        let expected_training: Vec<f32> = ln
            .slow(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Bound derivation (acceptance advisory, family J): `1e-4` is not a
        // round hand-picked number — it is a generous (~5x) margin over an
        // f32-fold-order bound derived from Higham (2002) Thm 4.2 for this
        // EXACT fixture. Thm 4.2: summing `n` floating-point terms in ANY
        // fixed order (fused's ascending-index row loop vs `slow()`'s
        // candle-composed `sum_keepdim`/`broadcast_*` tree — a DIFFERENT
        // fold order, not a wider/narrower one) bounds the computed sum's
        // absolute error by `(n-1) * u * Σ|x_i| + O(u²)`, `u` = f32 unit
        // roundoff ≈ 1.1921e-7 (`2^-23`). Two independent `n = hidden = 8`
        // reductions feed this op: `mean = Σx_i / 8` over `Σ|x_i| = 8.5`
        // (this fixture's own `x`), giving `(8-1) * 1.1921e-7 * 8.5 ≈
        // 7.09e-6`; and `var = Σ(x_i-mean)² / 8` over `Σ(x_i-mean)² ≈
        // 12.09` (this fixture's own deviations), giving `(8-1) * 1.1921e-7
        // * 12.09 ≈ 1.01e-5`. The variance error propagates through
        // `invvar = 1/sqrt(var+eps)` with local sensitivity `|d(invvar)/
        // d(var)| = 0.5*(var+eps)^-1.5 ≈ 0.5 * 1.512^-1.5 ≈ 0.269` here,
        // contributing `≈ 0.269 * 1.01e-5 ≈ 2.7e-6` to `invvar`'s own
        // error; that then scales `xhat * gamma` (`|xhat| ≤ 1.48`, `gamma =
        // 1.3` on this fixture) by roughly `1.48 * 1.3 * 2.7e-6 ≈ 5.2e-6`.
        // Summing every term (mean's own propagated contribution through
        // `xhat`, plus the two above) stays on the order of `1e-5` for
        // THIS fixture's magnitude — `1e-4` is therefore a ~5-10x safety
        // margin over the derived bound, tight enough to still catch a
        // real defect (every forced-defect leg in `cuda_parity.rs` diverges
        // by orders of magnitude more than this), not a number chosen to
        // make the test pass.
        for (i, (o, e)) in out_training
            .iter()
            .zip(expected_training.iter())
            .enumerate()
        {
            assert!(
                (o - e).abs() < 1e-4,
                "fused[{i}] = {o} vs slow()[{i}] = {e} (rule 15: fold-order divergence, \
                 not a defect — see this loop's own bound-derivation comment above; must \
                 stay within a tight tolerance, not bit-exact)"
            );
        }

        ln.set_training(false);
        let out_eval: Vec<f32> = ln
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let expected_eval: Vec<f32> = candle_nn::ops::layer_norm(&x, &weight, &bias, 1e-5)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            out_eval, expected_eval,
            "eval must remain byte-for-byte candle_nn::ops::layer_norm — unaffected by #460"
        );
    }

    /// Oracle 2 at the encoder level (per the fused-kernels plan's scope
    /// 7b, applied to the ACTUAL `slow()` this crate ships — the leaf
    /// `jammi-kernels` crate reproduces this composition in its own
    /// hermetic tests instead, since it cannot depend on this crate; see
    /// `jammi_kernels`' `tests/layer_norm_oracles.rs`). Compares the real
    /// dispatch path (`forward` with `bias.is_none() && training`)
    /// against `slow()` on the identical input, fwd AND bwd.
    #[test]
    fn fused_training_path_matches_slow_within_tolerance_fwd_and_bwd() {
        let device = Device::Cpu;
        let hidden = 8;
        let rows = 3;
        let xv: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.31 - 1.5).sin() * 3.0)
            .collect();
        let gv: Vec<f32> = (0..hidden).map(|i| 0.8 + i as f32 * 0.1).collect();

        let x_fused =
            Var::from_tensor(&Tensor::from_slice(&xv, (rows, hidden), &device).unwrap()).unwrap();
        let w_fused =
            Var::from_tensor(&Tensor::from_slice(&gv, (hidden,), &device).unwrap()).unwrap();
        let mut ln_fused = bias_free_ln(w_fused.as_tensor().clone(), 1e-5, true);
        ln_fused.training = true;

        let (holds, predicate) = fused_admission_predicate(x_fused.as_tensor(), &ln_fused.weight);
        assert!(holds, "fixture must be fused-eligible: {predicate}");
        // `LN_DISPATCH_COUNTERS` is one process-wide static shared with
        // every other test in this binary — under parallel test
        // execution an exact before+1 delta would be racy, so this only
        // asserts monotonic increase (other concurrent tests can only
        // add to it, never subtract).
        let before = LN_DISPATCH_COUNTERS.snapshot();
        let out_fused = ln_fused.forward(&x_fused).unwrap();
        let after = LN_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "this fixture must actually dispatch the fused kernel, not fall back \
             (before={before:?}, after={after:?})"
        );

        let x_eager =
            Var::from_tensor(&Tensor::from_slice(&xv, (rows, hidden), &device).unwrap()).unwrap();
        let w_eager =
            Var::from_tensor(&Tensor::from_slice(&gv, (hidden,), &device).unwrap()).unwrap();
        let ln_eager = bias_free_ln(w_eager.as_tensor().clone(), 1e-5, true);
        let out_eager = ln_eager.slow(&x_eager).unwrap();

        let vf: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ve: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs slow() {e}");
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
            assert!((f - e).abs() < 1e-3, "dx[{i}]: fused {f} vs slow() {e}");
        }

        // `w_fused`/`w_eager` are both real `Var`s (trainable parameters
        // in this fixture) — `self.weight.is_variable()` must therefore
        // have set `dgamma_needed = true` on the fused call, and
        // `dgamma`'s slot must be populated and match the eager
        // composition's gradient for `gamma`. Before the
        // `is_variable()`-driven fix, this fixture was constructing an
        // UNSOUND state (a trainable `Var` gamma paired with a hardcoded
        // `dgamma_needed = false`): `grads_fused.get(&w_fused)` would
        // have been `None` here — no panic, just a silently missing
        // gradient a real AdamW step would skip (`backprop.rs:674-677`).
        let dgf: Vec<f32> = grads_fused
            .get(&w_fused)
            .expect(
                "dgamma_needed must be true for a trainable Var gamma \
                 (self.weight.is_variable()) — this must not be None",
            )
            .to_vec1()
            .unwrap();
        let dge: Vec<f32> = grads_eager.get(&w_eager).unwrap().to_vec1().unwrap();
        for (i, (f, e)) in dgf.iter().zip(dge.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dgamma[{i}]: fused {f} vs slow() {e}");
        }
    }

    /// The domain-widening regression check (K2): a BF16 `x` paired with
    /// an F32 `weight` must be REFUSED, not silently upcast into
    /// `internal_dtype` and rounded down to a confident wrong bf16
    /// number. This is the exact mismatch
    /// `fused_admission_predicate`'s own
    /// `dtype_f32_bf16_or_f16_matching_between_x_and_weight` check refuses on
    /// the fused path; `slow()` must refuse it too.
    #[test]
    fn slow_refuses_a_dtype_mismatched_weight_instead_of_silently_upcasting() {
        let device = Device::Cpu;
        let hidden = 8;
        let xv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(i as f32 * 0.1))
            .collect();
        let x = Tensor::from_slice(&xv, (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 8], (hidden,), &device).unwrap();
        let ln = bias_free_ln(weight, 1e-5, true);
        let err = ln
            .slow(&x)
            .expect_err("mismatched weight/x dtype must error, not silently compute");
        assert!(
            matches!(err, EncoderError::Config(_)),
            "expected a Config error naming the dtype mismatch, got {err:?}"
        );
    }

    /// The bias-side twin of the check above (K2, same mechanism, the
    /// SEPARATE `if let Some(b) = &self.bias` guard at `layer_norm.rs`):
    /// a BF16 `x`/`weight` paired with an F32 `bias` must be REFUSED, not
    /// silently upcast into `internal_dtype`. This is the only other
    /// domain-widening edge `slow()`'s dtype guard covers, and it had no
    /// dedicated test before this one — the biased arm is live for
    /// `bert.rs`, `distilbert.rs`, and `clip_text.rs`'s LayerNorms.
    #[test]
    fn slow_refuses_a_dtype_mismatched_bias_instead_of_silently_upcasting() {
        let device = Device::Cpu;
        let hidden = 8;
        let xv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(i as f32 * 0.1))
            .collect();
        let wv: Vec<bf16> = (0..hidden).map(|_| bf16::from_f32(1.0)).collect();
        let x = Tensor::from_slice(&xv, (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&wv, (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&[0.0f32; 8], (hidden,), &device).unwrap();
        let ln = biased_ln(weight, bias, 1e-5, true);
        let err = ln
            .slow(&x)
            .expect_err("mismatched bias/x dtype must error, not silently compute");
        assert!(
            matches!(err, EncoderError::Config(_)),
            "expected a Config error naming the dtype mismatch, got {err:?}"
        );
    }

    /// Deterministic LCG walk producing PRODUCTION-AMPLITUDE f32 values in
    /// `[-half_width, half_width)`, tracked by its literal seed/multiplier/
    /// increment (not RNG-crate state) — the same convention
    /// `crate::test_support::deterministic_fill_varmap` uses at a
    /// narrower range, widened here so the bf16-rounded fixture spans
    /// several bf16 ULP steps and actually exercises a rounding-placement
    /// difference rather than a range where every rounding decision lands
    /// the same way regardless of where the cast sits.
    fn lcg_fixture(mut state: u32, n: usize, half_width: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
                (unit - 0.5) * 2.0 * half_width
            })
            .collect()
    }

    /// A from-scratch (no candle tensor ops, no `jammi_kernels` import)
    /// f32-accumulated, ascending-index, two-pass reference for the
    /// bias-free LayerNorm epilogue, rounded to bf16 EXACTLY ONCE at the
    /// end — the "torch placement" `slow()`'s doc pins to torch 2.13.0.
    /// Independently re-derived, not imported, from
    /// `jammi_kernels::ops::layer_norm`'s own private
    /// `mean_var_f32`/`ln_fwd_row_bf16` (this crate cannot import that
    /// private fn anyway) — the SAME fixed fold order (family J), so a
    /// bug shared by both implementations would not silently cancel.
    ///
    /// This fold order is NOT guaranteed to bit-match candle's own
    /// `Tensor::sum_keepdim` at production `hidden`: `sum_keepdim`'s CPU
    /// backend uses a SIMD-lane partial-sum reduction on targets where
    /// `neon`/`avx2`/`simd128` is enabled (candle-core 0.11.0's
    /// `cpu/mod.rs::vec_sum`), a DIFFERENT (still IEEE-754-correct, just
    /// differently associated) fold order than this function's plain
    /// left-to-right accumulation. That is a real, small,
    /// reduction-order-only divergence at production width — see
    /// `REDUCTION_ORDER_BUDGET_FRACTION`'s doc — orthogonal to the
    /// rounding-PLACEMENT defect `slow()`'s fix addresses.
    fn scalar_layer_norm_truth_bf16(
        x: &[bf16],
        gamma: &[bf16],
        hidden: usize,
        eps: f64,
    ) -> Vec<bf16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                out.push(bf16::from_f32(xhat * gamma[i].to_f32()));
            }
        }
        out
    }

    /// The CANDLE-FOLD division-form comparator
    /// [`layer_norm_slow_matches_truth_at_production_shape`] measures
    /// against `slow()`'s own real reciprocal-form output. Composed from
    /// the SAME candle tensor ops `slow()` itself calls (`to_dtype`,
    /// `sum_keepdim`, `broadcast_sub`, `sqr`, `broadcast_mul`), and
    /// therefore the SAME reduction fold `slow()`'s own `sum_keepdim`
    /// calls take on whatever host runs this test (see
    /// [`scalar_layer_norm_truth_bf16`]'s own doc on why that fold is not
    /// portable across hosts) — differing from `slow()` ONLY in
    /// `centered.broadcast_div(&std)`, where `slow()` takes the
    /// reciprocal first and multiplies (the pre-round-3 `slow()`
    /// placement this function reproduces).
    ///
    /// This is what makes the comparison in
    /// [`layer_norm_slow_matches_truth_at_production_shape`] COMMENSURABLE:
    /// a hand-rolled scalar-loop division form (as this file previously
    /// used here) shares NEITHER `slow()`'s reduction fold nor its op
    /// sequence, so any residual difference between it and `slow()`'s real
    /// output would be a mix of placement AND fold-order noise, not
    /// placement alone. Sharing the fold isolates the placement effect
    /// exactly the way [`f32_div_truth`]/[`f32_rstd_multiply_truth`]
    /// already do at F32 below — this is that pair's bf16 analog, added
    /// only to make the bf16 A/B fair; production `slow()` itself is
    /// untouched.
    fn candle_fold_division_form_bf16(x: &Tensor, gamma: &Tensor, eps: f64) -> Tensor {
        let hidden = x.dim(D::Minus1).unwrap();
        let x_f32 = x.to_dtype(DType::F32).unwrap();
        let mean = (x_f32.sum_keepdim(D::Minus1).unwrap() / hidden as f64).unwrap();
        let centered = x_f32.broadcast_sub(&mean).unwrap();
        let variance =
            (centered.sqr().unwrap().sum_keepdim(D::Minus1).unwrap() / hidden as f64).unwrap();
        let std = (variance + eps).unwrap().sqrt().unwrap();
        let normalized = centered.broadcast_div(&std).unwrap();
        let gamma_f32 = gamma.to_dtype(DType::F32).unwrap();
        let scaled = normalized.broadcast_mul(&gamma_f32).unwrap();
        scaled.to_dtype(DType::BF16).unwrap()
    }

    /// The PRE-FIX formula this commit removes: round `xhat` to bf16
    /// BEFORE multiplying by `gamma` — `bf16(bf16(xhat) * gamma)`, two
    /// rounding points instead of one. A deliberately WRONG
    /// reimplementation kept ONLY as this oracle's non-vacuity control:
    /// proves the fixture actually exercises the rounding-placement
    /// difference (mismatches against the truth on a stated,
    /// asserted-positive count), not a fixture that happens to round the
    /// same way regardless of where the cast sits.
    fn scalar_layer_norm_double_round_bf16(
        x: &[bf16],
        gamma: &[bf16],
        hidden: usize,
        eps: f64,
    ) -> Vec<bf16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                let xhat_bf16 = bf16::from_f32(xhat); // ROUND #1 (the pre-fix defect).
                out.push(bf16::from_f32(xhat_bf16.to_f32() * gamma[i].to_f32()));
                // ROUND #2.
            }
        }
        out
    }

    /// A PARTIAL-regression variant of
    /// [`scalar_layer_norm_double_round_bf16`]: double-rounds `xhat` (the
    /// pre-fix defect) on only the first `bad_rows` rows and single-rounds
    /// (correctly) every other row. This is the shape a REALISTIC
    /// regression takes — a bug that corrupts a subset of rows, not the
    /// whole tensor — used to prove `REDUCTION_ORDER_BUDGET_FRACTION` is
    /// tight enough to catch a ~1%-of-rows regression, not just the
    /// every-row worst case [`scalar_layer_norm_double_round_bf16`]
    /// already covers.
    fn scalar_layer_norm_partial_double_round_bf16(
        x: &[bf16],
        gamma: &[bf16],
        hidden: usize,
        eps: f64,
        bad_rows: usize,
    ) -> Vec<bf16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            let double_round_this_row = r < bad_rows;
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                if double_round_this_row {
                    let xhat_bf16 = bf16::from_f32(xhat); // ROUND #1 (the pre-fix defect).
                    out.push(bf16::from_f32(xhat_bf16.to_f32() * gamma[i].to_f32()));
                    // ROUND #2.
                } else {
                    out.push(bf16::from_f32(xhat * gamma[i].to_f32()));
                }
            }
        }
        out
    }

    /// The ONLY source of disagreement between `slow()` (candle
    /// `Tensor::sum_keepdim`, SIMD-lane reduction on this crate's own
    /// dev/CI targets) and [`scalar_layer_norm_truth_bf16`] (ascending-
    /// index fold) that survives the one-rounding fix AND the
    /// reciprocal-vs-division placement fix (`slow()`'s doc, just above
    /// its `rstd` line) is reduction-ORDER noise in the mean/variance sums
    /// straddling a bf16 rounding boundary for a handful of elements —
    /// not a rounding-PLACEMENT bug.
    ///
    /// Used ONLY by the BIAS-FREE arm's two tests
    /// (`layer_norm_slow_matches_truth_at_production_shape_seq128`/
    /// `_seq512`) — the biased and F16 arms each derive their OWN budget
    /// constant below ([`BIASED_REDUCTION_ORDER_BUDGET_FRACTION`],
    /// [`F16_REDUCTION_ORDER_BUDGET_FRACTION`]), from their own measured
    /// residuals, rather than reusing this one. A single shared constant
    /// derived only from the bias-free arms previously gave the OTHER
    /// three consuming arms far less headroom than the 10×-over-
    /// measurement this doc claims: at the values measured on this
    /// branch (residuals printed by
    /// [`layer_norm_slow_matches_truth_at_production_shape_biased_seq128`],
    /// [`layer_norm_slow_matches_truth_at_production_shape_biased_seq512`],
    /// and [`layer_norm_slow_matches_truth_at_production_shape_f16`]),
    /// this constant's `93`/`371` budgets left the biased arm
    /// only `93/34 ≈ 2.7×` / `371/148 ≈ 2.5×` headroom and the F16 arm
    /// only `93/59 ≈ 1.58×` — nowhere near the `10×` this doc's own
    /// derivation promises, and tight enough that a shift in libm/SIMD
    /// behavior on a different CI runner could flake those three arms
    /// even though the constant's OWN derivation (below) was sound for
    /// the two arms it was measured from.
    ///
    /// Derivation (not a value tightened to zero, and not a loose
    /// round-number guess): `layer_norm_slow_matches_truth_at_production_shape`
    /// prints the measured `slow()`-vs-truth mismatch count at both
    /// production shapes it covers, on this crate's own dev/CI target,
    /// AFTER both placement fixes above —
    ///
    /// * `rows=256, hidden=1024` (seq 128): `5/262144` = `1.91e-5`
    /// * `rows=1024, hidden=1024` (seq 512): `37/1048576` = `3.53e-5`
    ///
    /// This constant is `10×` the LARGER of those two measured fractions
    /// (`10 * 3.53e-5 = 3.529e-4`), i.e. headroom over the measurement,
    /// not the measurement itself — a different libm/SIMD width on
    /// another CI runner shifting the exact mismatch count by less than
    /// 10× does not flake this test. At the two production shapes above
    /// that resolves to element budgets of `ceil(262144 * 3.529e-4) = 93`
    /// (seq 128) and `ceil(1048576 * 3.529e-4) = 371` (seq 512) — both
    /// comfortably above the measured 5 and 37, and both tight enough
    /// that a partial regression touching only ~1% of rows still trips
    /// it (this same test's partial-double-round control double-rounds
    /// only `floor(rows * 0.01)` rows and ASSERTS its own mismatch count
    /// exceeds this budget — 526 vs 93 at seq 128, 2576 vs 371 at seq
    /// 512, both measured, printed, and re-checked live on every run
    /// (printed by [`layer_norm_slow_matches_truth_at_production_shape_seq128`]
    /// and [`layer_norm_slow_matches_truth_at_production_shape_seq512`]),
    /// not assumed).
    /// The whole-tensor double-rounding control (every row, not just
    /// ~1%) is checked against a separate, looser `budget * 5` bound
    /// only — see that assertion's own text for why: it exists to prove
    /// non-vacuity (the fixture exercises the rounding-placement bug at
    /// all), not to pin an exact headroom multiple that would go stale
    /// on its own.
    ///
    /// The measured counts quoted above (`5`, `37`, and every other
    /// mismatch figure this doc or `slow()`'s own doc cites) are
    /// HOST-FOLD-SPECIFIC: they come from candle's `Tensor::sum_keepdim`,
    /// whose CPU backend takes a SIMD-lane partial-sum reduction on
    /// `neon`/`avx2`/`simd128` targets and a plain scalar fold otherwise
    /// (`candle-core-0.11.0` `cpu/mod.rs::vec_sum`) — a genuinely
    /// different (still IEEE-754-correct) fold order per host
    /// architecture, not just a different compiler. None of these figures
    /// are asserted as fixed constants anywhere in this file for exactly
    /// that reason (a fixed cross-architecture hash of a SIMD-fold value
    /// is not portable — see the F32 discriminator test's own history);
    /// the `10×` headroom this budget is built from is what absorbs that
    /// host-to-host drift, not an assumption that the exact counts are
    /// architecture-invariant.
    const REDUCTION_ORDER_BUDGET_FRACTION: f64 = 3.529e-4;

    /// The BIASED arm's own reduction-order budget — the full torch form
    /// `slow()`'s doc quotes (`gamma` AND `beta` both in the epilogue).
    /// Derived the SAME way [`REDUCTION_ORDER_BUDGET_FRACTION`] is, but
    /// from the biased arm's OWN measured residuals rather than the
    /// bias-free arm's, since the two arms exercise a DIFFERENT candle-op
    /// sequence (the biased arm has an extra `broadcast_add` for `beta`)
    /// and there is no structural reason their reduction-order noise
    /// floors should coincide:
    ///
    /// * `rows=256, hidden=1024` (seq 128): `34/262144` = `1.297e-4`
    /// * `rows=1024, hidden=1024` (seq 512): `148/1048576` = `1.412e-4`
    ///
    /// `10×` the larger of those (`10 * 1.412e-4 = 1.412e-3`, rounded up
    /// slightly for the same reason `REDUCTION_ORDER_BUDGET_FRACTION` is)
    /// resolves to element budgets of `ceil(262144 * 1.412e-3) = 371`
    /// (seq 128, `10.9×` headroom over the measured 34) and
    /// `ceil(1048576 * 1.412e-3) = 1481` (seq 512, `10.0×` headroom over
    /// the measured 148) — both printed and re-checked live by
    /// `layer_norm_slow_matches_truth_at_production_shape_biased_seq128`/
    /// `_seq512`, not assumed.
    const BIASED_REDUCTION_ORDER_BUDGET_FRACTION: f64 = 1.412e-3;

    /// The F16 arm's own reduction-order budget. Only ONE production
    /// shape is exercised for F16 (`layer_norm_slow_matches_truth_at_production_shape_f16`,
    /// seq 128 only — see that test's own doc for why a single fixture is
    /// sufficient), so this constant is `10×` that single measured
    /// fraction directly, not the larger of two:
    ///
    /// * `rows=256, hidden=1024` (seq 128): `59/262144` = `2.2507e-4`
    ///
    /// `10 * 2.2507e-4 = 2.2507e-3`, rounded up slightly, resolves to an
    /// element budget of `ceil(262144 * 2.251e-3) = 591` — `10.0×`
    /// headroom over the measured 59, printed and re-checked live by that
    /// test, not assumed.
    const F16_REDUCTION_ORDER_BUDGET_FRACTION: f64 = 2.251e-3;

    /// Biting oracle (family F: measured live against an independently-
    /// derived reference, not a same-code tautology) at PRODUCTION
    /// shape — `hidden=1024`, `rows = batch * seq` for `batch=2`, `seq in
    /// {128, 512}` — calling the REAL `LayerNorm::slow` (not a
    /// reimplementation of it): `jammi-kernels` is a leaf crate and
    /// cannot reach this function at all (see that crate's
    /// `tests/layer_norm_oracles.rs` module doc), so THIS is the only
    /// place in the workspace that can exercise `slow()`'s actual
    /// dispatch against an independent numeric truth.
    ///
    /// Reverting this file's production `slow()` hunk (restoring the
    /// pre-fix two-round `normalized.to_dtype(x_dtype)?.broadcast_mul(&weight)`
    /// form) turns this test RED: `slow()`'s output then matches
    /// [`scalar_layer_norm_double_round_bf16`] almost everywhere instead
    /// of the truth reference, so `mismatch_vs_truth` blows past
    /// `REDUCTION_ORDER_BUDGET_FRACTION`'s budget.
    fn layer_norm_slow_matches_truth_at_production_shape(rows: usize, hidden: usize, seed: u32) {
        let device = Device::Cpu;
        let eps = 1e-5f64;
        let n = rows * hidden;

        let xf = lcg_fixture(seed, n, 24.0);
        let gf = lcg_fixture(seed.wrapping_add(0x9E37_79B9), hidden, 2.0);
        let x_bf16: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
        let g_bf16: Vec<bf16> = gf.iter().map(|&v| bf16::from_f32(v)).collect();
        assert!(
            x_bf16.iter().all(|v| v.to_f32().is_finite()),
            "fixture x must be finite before any bit compare"
        );
        assert!(
            g_bf16.iter().all(|v| v.to_f32().is_finite()),
            "fixture gamma must be finite before any bit compare"
        );

        let x = Tensor::from_slice(&x_bf16, (rows, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&g_bf16, (hidden,), &device).unwrap();
        let ln = bias_free_ln(weight.clone(), eps, true);

        let slow_out: Vec<bf16> = ln
            .slow(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let fused_out: Vec<bf16> = apply2(&x, &weight, LayerNormFused::new(eps, false))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            slow_out.iter().all(|v| v.to_f32().is_finite()),
            "slow() output must be finite before any bit compare"
        );
        assert!(
            fused_out.iter().all(|v| v.to_f32().is_finite()),
            "fused output must be finite before any bit compare"
        );

        let truth = scalar_layer_norm_truth_bf16(&x_bf16, &g_bf16, hidden, eps);
        assert!(
            truth.iter().all(|v| v.to_f32().is_finite()),
            "truth output must be finite before any bit compare"
        );

        // The fused CPU arm runs the SAME ascending-scalar algorithm this
        // truth reference does (independently re-derived, not imported) —
        // no candle-tensor-op reduction is involved on either side, so
        // this one IS bit-exact, unconditionally.
        assert_eq!(
            fused_out, truth,
            "LayerNormFused's CPU arm must be bit-exact vs the scalar truth (same fixed fold order)"
        );

        let mismatch_vs_truth = slow_out
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let budget = ((n as f64) * REDUCTION_ORDER_BUDGET_FRACTION).ceil() as usize;
        println!(
            "layer_norm_slow_matches_truth_at_production_shape(rows={rows}, hidden={hidden}): \
             slow() vs truth mismatches = {mismatch_vs_truth}/{n} (budget {budget})"
        );
        assert!(
            mismatch_vs_truth <= budget,
            "slow() diverged from the f32-round-once truth on {mismatch_vs_truth}/{n} \
             elements, past the {budget}-element reduction-order budget — this is the \
             rounding-PLACEMENT regression the fix restores, not reduction-order noise"
        );

        // Reciprocal-vs-division placement effect on THIS bf16 fixture,
        // measured COMMENSURABLY (orthogonal to the double-rounding RED
        // control below): `division_form` shares `slow()`'s own candle
        // `sum_keepdim` fold (built by `candle_fold_division_form_bf16`,
        // the SAME op sequence `slow()` uses except for the `rstd` line
        // itself), so any residual difference between it and
        // `mismatch_vs_truth` above (both diffed against the SAME scalar
        // truth reference) is attributable to the reciprocal-vs-division
        // placement alone, not to a fold-order mismatch between a scalar
        // loop and a SIMD-lane reduction — unlike a hand-rolled
        // scalar-loop division form, which would conflate the two.
        // `slow()`'s own doc cites this printed pair, live, for how much
        // smaller this placement's effect is at bf16 (where
        // `internal_dtype == F32` regardless of `x_dtype`) than at F32
        // itself (where it is the ONLY source of divergence — see
        // `slow_f32_reciprocal_form_is_bit_exact_and_diverges_from_division`).
        let division_form: Vec<bf16> = candle_fold_division_form_bf16(&x, &weight, eps)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            division_form.iter().all(|v| v.to_f32().is_finite()),
            "division-form residual reference output must be finite before any bit compare"
        );
        let mismatch_division_form = division_form
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "layer_norm_slow_matches_truth_at_production_shape(rows={rows}, hidden={hidden}): \
             division-form slow() vs truth = {mismatch_division_form}/{n}"
        );
        println!(
            "layer_norm_slow_matches_truth_at_production_shape(rows={rows}, hidden={hidden}): \
             reciprocal-form slow() vs truth = {mismatch_vs_truth}/{n}"
        );
        assert!(
            mismatch_vs_truth <= mismatch_division_form,
            "reciprocal-form slow() ({mismatch_vs_truth}/{n}) must not be WORSE than the \
             same-candle-fold division-form comparator ({mismatch_division_form}/{n}) against \
             the same scalar truth — both share slow()'s own sum_keepdim fold, so this A/B is \
             commensurable, and a regression here would mean the reciprocal placement made bf16 \
             output strictly worse, not merely different"
        );

        // RED CONTROL (non-vacuity): the pre-fix double-rounding formula
        // must differ from truth on a stated, ASSERTED-POSITIVE count —
        // proving the fixture actually exercises the rounding-placement
        // difference, and that its magnitude swamps the reduction-order
        // budget above (so the two mechanisms are told apart, not
        // conflated).
        let double_round = scalar_layer_norm_double_round_bf16(&x_bf16, &g_bf16, hidden, eps);
        assert!(
            double_round.iter().all(|v| v.to_f32().is_finite()),
            "double-round control output must be finite before any bit compare"
        );
        let mismatch_double_round = double_round
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "layer_norm_slow_matches_truth_at_production_shape(rows={rows}, hidden={hidden}): \
             double-round control vs truth mismatches = {mismatch_double_round}/{n}"
        );
        assert!(
            mismatch_double_round > 0,
            "RED control is vacuous: the double-rounding formula matched the truth on every \
             element (mismatch count 0) — this fixture does not exercise the \
             rounding-placement difference at all"
        );
        assert!(
            mismatch_double_round > budget * 5,
            "RED control's divergence ({mismatch_double_round}) must swamp the \
             reduction-order budget ({budget}) by a wide margin, or it is not actually \
             distinguishing the rounding-placement bug from ordinary reduction-order noise"
        );

        // PARTIAL-REGRESSION CONTROL: the RED control above double-rounds
        // EVERY row, which is the easiest possible case to catch. Prove
        // the budget is actually tight enough to flag a realistic
        // regression that only corrupts ~1% of rows — the shape a real
        // bug (e.g. a mis-scoped SIMD lane, an off-by-one tile boundary)
        // would take, not a whole-tensor formula swap.
        let bad_rows = ((rows as f64) * 0.01).floor().max(1.0) as usize;
        let partial_double_round =
            scalar_layer_norm_partial_double_round_bf16(&x_bf16, &g_bf16, hidden, eps, bad_rows);
        assert!(
            partial_double_round.iter().all(|v| v.to_f32().is_finite()),
            "partial double-round control output must be finite before any bit compare"
        );
        let mismatch_partial_double_round = partial_double_round
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "layer_norm_slow_matches_truth_at_production_shape(rows={rows}, hidden={hidden}): \
             partial ({bad_rows}/{rows} rows) double-round control vs truth mismatches = \
             {mismatch_partial_double_round}/{n} (budget {budget})"
        );
        assert!(
            mismatch_partial_double_round > budget,
            "the reduction-order budget ({budget}) is too loose: a partial regression that \
             double-rounds only {bad_rows}/{rows} rows produced {mismatch_partial_double_round} \
             mismatches, which must exceed the budget for the budget to be a useful regression \
             detector rather than dead code"
        );
    }

    #[test]
    fn layer_norm_slow_matches_truth_at_production_shape_seq128() {
        // batch=2, seq=128, hidden=1024 -> rows=256.
        layer_norm_slow_matches_truth_at_production_shape(2 * 128, 1024, 0xC0FF_EE01);
    }

    #[test]
    fn layer_norm_slow_matches_truth_at_production_shape_seq512() {
        // batch=2, seq=512, hidden=1024 -> rows=1024.
        layer_norm_slow_matches_truth_at_production_shape(2 * 512, 1024, 0xC0FF_EE02);
    }

    /// An F32 reference for `slow()`'s bias-free epilogue that reuses
    /// candle's OWN `Tensor::sum_keepdim` for both mean and variance —
    /// the exact same reduction `slow()` performs internally — rather
    /// than the hand-rolled ascending-index scalar loop
    /// [`scalar_layer_norm_truth_bf16`] uses. Sharing the fold order this
    /// way (family J: a fixed, explicit fold order is what makes a
    /// numeric claim checkable at all) removes reduction-order as a free
    /// variable entirely: at F32, `internal_dtype == x_dtype`, so every
    /// `to_dtype` call `slow()` makes is a same-dtype no-op, and the ONLY
    /// remaining degree of freedom between this function and `slow()` is
    /// whether `rstd` is computed as a reciprocal-then-multiply (this
    /// function, matching `slow()`'s current form) or a division (see
    /// [`f32_div_truth`] below). That makes this an exact, zero-tolerance
    /// oracle — not a budgeted one like the bf16/f16 arms above, which
    /// tolerate real SIMD-lane reduction-order noise from a DIFFERENT
    /// fold order.
    fn f32_rstd_multiply_truth(x: &Tensor, gamma: &Tensor, eps: f64) -> Tensor {
        let hidden = x.dim(D::Minus1).unwrap();
        let mean = (x.sum_keepdim(D::Minus1).unwrap() / hidden as f64).unwrap();
        let centered = x.broadcast_sub(&mean).unwrap();
        let variance =
            (centered.sqr().unwrap().sum_keepdim(D::Minus1).unwrap() / hidden as f64).unwrap();
        let rstd = (variance + eps).unwrap().sqrt().unwrap().recip().unwrap();
        let normalized = centered.broadcast_mul(&rstd).unwrap();
        normalized.broadcast_mul(gamma).unwrap()
    }

    /// The division-form TWIN of [`f32_rstd_multiply_truth`] — identical
    /// in every other respect (same `sum_keepdim` calls, same fold order)
    /// except `centered.broadcast_div(&std)` where the function above
    /// takes the reciprocal first and multiplies. This is the PRE-ROUND-3
    /// formula `slow()`'s `rstd` line replaced (see that line's own doc).
    /// Kept ONLY as this oracle's RED, non-vacuity control: division and
    /// multiply-by-reciprocal are not bit-identical in floating point (the
    /// reciprocal is itself a rounded value), so this must diverge from
    /// [`f32_rstd_multiply_truth`] — proving the fixture actually
    /// distinguishes the two placements at F32, where the bf16/f16
    /// double-rounding fix's own oracles are silent (that fix is a
    /// same-dtype no-op at F32; this one is not).
    fn f32_div_truth(x: &Tensor, gamma: &Tensor, eps: f64) -> Tensor {
        let hidden = x.dim(D::Minus1).unwrap();
        let mean = (x.sum_keepdim(D::Minus1).unwrap() / hidden as f64).unwrap();
        let centered = x.broadcast_sub(&mean).unwrap();
        let variance =
            (centered.sqr().unwrap().sum_keepdim(D::Minus1).unwrap() / hidden as f64).unwrap();
        let std = (variance + eps).unwrap().sqrt().unwrap();
        let normalized = centered.broadcast_div(&std).unwrap();
        normalized.broadcast_mul(gamma).unwrap()
    }

    /// The F32 discriminator for the reciprocal-vs-division rounding-
    /// PLACEMENT fix at `slow()`'s `rstd` line (family D/F/J): proves,
    /// against a same-fold-order reference, that `slow()`'s F32 output
    /// actually depends on taking the reciprocal
    /// first rather than dividing — closing the mutation survivor found
    /// on `3b3dbde` (reverting the `rstd` line back to
    /// `centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?` left
    /// every existing bf16/f16 test green, since their reduction-order
    /// BUDGET was loose enough to absorb the extra divergence — see
    /// `REDUCTION_ORDER_BUDGET_FRACTION`'s doc). F32 has no such budget to
    /// hide behind: `internal_dtype == x_dtype` there, so the ONLY
    /// difference between `slow()`'s real output and
    /// [`f32_rstd_multiply_truth`]'s same-fold-order reference is the
    /// `rstd` line itself, making an exact (not budgeted) bit-compare
    /// possible, and reverting that one line turns the whole tensor's
    /// output — not a stray 1-in-93 rounding-boundary element — into the
    /// division form's numbers instead.
    #[test]
    fn slow_f32_reciprocal_form_is_bit_exact_and_diverges_from_division() {
        let device = Device::Cpu;
        let eps = 1e-5f64;
        // batch=2, seq=128, hidden=1024 -> rows=256 -- the same production
        // shape as the bf16 seq128 oracle above.
        let (rows, hidden) = (2 * 128, 1024);
        let n = rows * hidden;

        let xf = lcg_fixture(0xF32B_EED1, n, 24.0);
        let gf = lcg_fixture(0xF32B_EED2, hidden, 2.0);
        assert!(xf.iter().all(|v| v.is_finite()), "fixture x must be finite");
        assert!(
            gf.iter().all(|v| v.is_finite()),
            "fixture gamma must be finite"
        );

        let x = Tensor::from_slice(&xf, (rows, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&gf, (hidden,), &device).unwrap();
        let ln = bias_free_ln(weight.clone(), eps, true);

        let slow_out: Vec<f32> = ln
            .slow(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            slow_out.iter().all(|v| v.is_finite()),
            "slow() output must be finite before any bit compare"
        );

        let truth_out: Vec<f32> = f32_rstd_multiply_truth(&x, &weight, eps)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            truth_out.iter().all(|v| v.is_finite()),
            "truth output must be finite before any bit compare"
        );
        assert_eq!(
            slow_out, truth_out,
            "slow()'s F32 output must be BIT-EXACT vs a same-fold-order (candle \
             sum_keepdim) reciprocal-multiply reference -- no reduction-order budget \
             applies at F32, since internal_dtype == x_dtype makes every to_dtype call \
             a same-dtype no-op"
        );

        // RED CONTROL (non-vacuity): the pre-round-3 division form must
        // diverge from the reciprocal-multiply truth on a stated,
        // ASSERTED-POSITIVE count, at F32, where the bf16/f16 oracles
        // above have no visibility into this specific placement at all.
        let div_out: Vec<f32> = f32_div_truth(&x, &weight, eps)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            div_out.iter().all(|v| v.is_finite()),
            "division-form control output must be finite before any bit compare"
        );
        let mismatch_div_vs_recip = slow_out
            .iter()
            .zip(div_out.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "slow_f32_reciprocal_form_is_bit_exact_and_diverges_from_division: \
             division-form vs slow() (reciprocal form) mismatches = {mismatch_div_vs_recip}/{n}"
        );
        assert!(
            mismatch_div_vs_recip > 0,
            "RED control is vacuous: the division form matched slow()'s reciprocal-form \
             output on every element -- this fixture does not exercise the \
             reciprocal-vs-division placement difference at F32 at all"
        );
    }

    /// Biased twin of [`scalar_layer_norm_truth_bf16`]: the SAME
    /// f32-accumulated, ascending-index, round-once-at-the-end reference,
    /// extended with the affine bias term (`out = gamma * (rstd * (x -
    /// mean)) + beta`, the full torch form quoted at `slow()`'s doc) —
    /// the arm every non-ModernBERT encoder's LayerNorm (`bert.rs`,
    /// `distilbert.rs`, `clip_text.rs`) is actually configured with.
    /// `gamma` AND `beta` are both applied in f32 before the single final
    /// round, exactly as `slow()`'s post-fix biased arm does.
    fn scalar_layer_norm_truth_bf16_biased(
        x: &[bf16],
        gamma: &[bf16],
        beta: &[bf16],
        hidden: usize,
        eps: f64,
    ) -> Vec<bf16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                let y = xhat * gamma[i].to_f32() + beta[i].to_f32();
                out.push(bf16::from_f32(y));
            }
        }
        out
    }

    /// Biased twin of [`scalar_layer_norm_double_round_bf16`]: the
    /// pre-fix biased-arm defect this commit removes — `xhat` rounded to
    /// bf16 before multiplying by `gamma` (ROUND #1), that product
    /// rounded to bf16 before adding `beta` (ROUND #2), then the sum
    /// rounded again (ROUND #3) — three rounding points instead of one.
    /// Kept ONLY as this oracle's non-vacuity control.
    fn scalar_layer_norm_double_round_bf16_biased(
        x: &[bf16],
        gamma: &[bf16],
        beta: &[bf16],
        hidden: usize,
        eps: f64,
    ) -> Vec<bf16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                let xhat_bf16 = bf16::from_f32(xhat); // ROUND #1.
                let scaled = bf16::from_f32(xhat_bf16.to_f32() * gamma[i].to_f32()); // ROUND #2.
                out.push(bf16::from_f32(scaled.to_f32() + beta[i].to_f32())); // ROUND #3.
            }
        }
        out
    }

    /// Biased analog of `layer_norm_slow_matches_truth_at_production_shape`
    /// (biting oracle, family F): calls the REAL `LayerNorm::slow` with a
    /// non-`None` `bias`, the arm the bias-free sweep above never
    /// exercises (`fused_admission_predicate`'s domain and
    /// `LayerNormFused` cover ONLY the bias-free case — every biased
    /// LayerNorm always falls to `slow()`, per `forward`'s `(bias,
    /// training)` match). Mutation testing on `b0c0a44` found this arm
    /// (`layer_norm.rs`'s `Some(b) =>
    /// scaled_internal.broadcast_add(&b.to_dtype(internal_dtype)?)`)
    /// survives reverting to the pre-fix double-rounding biased form with
    /// every existing test staying green — this oracle closes that gap.
    fn layer_norm_slow_matches_truth_at_production_shape_biased(
        rows: usize,
        hidden: usize,
        seed: u32,
    ) {
        let device = Device::Cpu;
        let eps = 1e-5f64;
        let n = rows * hidden;

        let xf = lcg_fixture(seed, n, 24.0);
        let gf = lcg_fixture(seed.wrapping_add(0x9E37_79B9), hidden, 2.0);
        let bf = lcg_fixture(seed.wrapping_add(0x1234_5678), hidden, 1.0);
        let x_bf16: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
        let g_bf16: Vec<bf16> = gf.iter().map(|&v| bf16::from_f32(v)).collect();
        let b_bf16: Vec<bf16> = bf.iter().map(|&v| bf16::from_f32(v)).collect();
        assert!(
            x_bf16.iter().all(|v| v.to_f32().is_finite()),
            "fixture x must be finite before any bit compare"
        );
        assert!(
            g_bf16.iter().all(|v| v.to_f32().is_finite()),
            "fixture gamma must be finite before any bit compare"
        );
        assert!(
            b_bf16.iter().all(|v| v.to_f32().is_finite()),
            "fixture beta must be finite before any bit compare"
        );

        let x = Tensor::from_slice(&x_bf16, (rows, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&g_bf16, (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&b_bf16, (hidden,), &device).unwrap();
        let ln = biased_ln(weight, bias, eps, true);

        let slow_out: Vec<bf16> = ln
            .slow(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            slow_out.iter().all(|v| v.to_f32().is_finite()),
            "slow() output must be finite before any bit compare"
        );

        let truth = scalar_layer_norm_truth_bf16_biased(&x_bf16, &g_bf16, &b_bf16, hidden, eps);
        assert!(
            truth.iter().all(|v| v.to_f32().is_finite()),
            "truth output must be finite before any bit compare"
        );

        let mismatch_vs_truth = slow_out
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let budget = ((n as f64) * BIASED_REDUCTION_ORDER_BUDGET_FRACTION).ceil() as usize;
        println!(
            "layer_norm_slow_matches_truth_at_production_shape_biased(rows={rows}, \
             hidden={hidden}): slow() vs truth mismatches = {mismatch_vs_truth}/{n} \
             (budget {budget})"
        );
        assert!(
            mismatch_vs_truth <= budget,
            "biased slow() diverged from the f32-round-once truth on {mismatch_vs_truth}/{n} \
             elements, past the {budget}-element reduction-order budget"
        );

        // RED CONTROL (non-vacuity): the pre-fix double-rounding biased
        // formula must differ from truth on a stated, ASSERTED-POSITIVE
        // count that also exceeds the reduction-order budget.
        let double_round =
            scalar_layer_norm_double_round_bf16_biased(&x_bf16, &g_bf16, &b_bf16, hidden, eps);
        assert!(
            double_round.iter().all(|v| v.to_f32().is_finite()),
            "double-round control output must be finite before any bit compare"
        );
        let mismatch_double_round = double_round
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "layer_norm_slow_matches_truth_at_production_shape_biased(rows={rows}, \
             hidden={hidden}): double-round control vs truth mismatches = \
             {mismatch_double_round}/{n}"
        );
        assert!(
            mismatch_double_round > 0,
            "RED control is vacuous: the biased double-rounding formula matched the truth on \
             every element (mismatch count 0) — this fixture does not exercise the biased \
             rounding-placement difference at all"
        );
        assert!(
            mismatch_double_round > budget,
            "RED control's divergence ({mismatch_double_round}) must exceed the \
             reduction-order budget ({budget}), or it is not actually distinguishing the \
             biased rounding-placement bug from ordinary reduction-order noise"
        );
    }

    #[test]
    fn layer_norm_slow_matches_truth_at_production_shape_biased_seq128() {
        // batch=2, seq=128, hidden=1024 -> rows=256.
        layer_norm_slow_matches_truth_at_production_shape_biased(2 * 128, 1024, 0xB1A5_ED01);
    }

    #[test]
    fn layer_norm_slow_matches_truth_at_production_shape_biased_seq512() {
        // batch=2, seq=512, hidden=1024 -> rows=1024.
        layer_norm_slow_matches_truth_at_production_shape_biased(2 * 512, 1024, 0xB1A5_ED02);
    }

    /// F16 twin of [`scalar_layer_norm_truth_bf16`]: `slow()`'s
    /// `internal_dtype` match (`DType::F16 | DType::BF16 => DType::F32`)
    /// takes the SAME branch for F16 as for BF16 — this proves that
    /// branch is actually exercised and rounds correctly for the OTHER
    /// dtype it names, not just BF16. Only ONE shape/seed is run here
    /// (not the full seq-128/seq-512 sweep the BF16 oracle covers): the
    /// rounding-placement mechanism is dtype-independent (both dtypes
    /// hit the identical F32-internal code path), so a single fixture is
    /// sufficient to confirm the F16 arm is reached and correct.
    fn scalar_layer_norm_truth_f16(x: &[f16], gamma: &[f16], hidden: usize, eps: f64) -> Vec<f16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                out.push(f16::from_f32(xhat * gamma[i].to_f32()));
            }
        }
        out
    }

    #[test]
    fn layer_norm_slow_matches_truth_at_production_shape_f16() {
        let device = Device::Cpu;
        let eps = 1e-5f64;
        // batch=2, seq=128, hidden=1024 -> rows=256 -- the same
        // production shape as the BF16 seq128 case above.
        let (rows, hidden) = (2 * 128, 1024);
        let n = rows * hidden;

        let xf = lcg_fixture(0xF16E_0002, n, 24.0);
        let gf = lcg_fixture(0xF16E_0003, hidden, 2.0);
        let x_f16: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
        let g_f16: Vec<f16> = gf.iter().map(|&v| f16::from_f32(v)).collect();
        assert!(
            x_f16.iter().all(|v| v.to_f32().is_finite()),
            "fixture x must be finite before any bit compare"
        );
        assert!(
            g_f16.iter().all(|v| v.to_f32().is_finite()),
            "fixture gamma must be finite before any bit compare"
        );

        let x = Tensor::from_slice(&x_f16, (rows, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&g_f16, (hidden,), &device).unwrap();
        let ln = bias_free_ln(weight, eps, true);

        let slow_out: Vec<f16> = ln
            .slow(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            slow_out.iter().all(|v| v.to_f32().is_finite()),
            "slow() output must be finite before any bit compare"
        );

        let truth = scalar_layer_norm_truth_f16(&x_f16, &g_f16, hidden, eps);
        assert!(
            truth.iter().all(|v| v.to_f32().is_finite()),
            "truth output must be finite before any bit compare"
        );

        let mismatch_vs_truth = slow_out
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        // F16's own measured residual at this shape is 59/262144
        // (2.2507e-4), printed on every run by this test
        // ([`layer_norm_slow_matches_truth_at_production_shape_f16`]) --
        // see [`F16_REDUCTION_ORDER_BUDGET_FRACTION`]'s
        // doc for the derivation; this is what is measured, not a claim
        // about WHY it differs from the bf16 arms' own residuals.
        let budget = ((n as f64) * F16_REDUCTION_ORDER_BUDGET_FRACTION).ceil() as usize;
        println!(
            "layer_norm_slow_matches_truth_at_production_shape_f16: slow() vs truth \
             mismatches = {mismatch_vs_truth}/{n} (budget {budget})"
        );
        assert!(
            mismatch_vs_truth <= budget,
            "F16 slow() diverged from the f32-round-once truth on {mismatch_vs_truth}/{n} \
             elements, past the {budget}-element reduction-order budget"
        );
    }

    #[test]
    fn strict_mode_errors_instead_of_falling_back_on_a_failed_predicate() {
        // SAFETY (test-only): env var mutation is racy across threads in
        // general, but `admission_mode()` memoizes into a `OnceLock` the
        // first time it is called in this PROCESS — this test's value
        // only takes effect if it runs before anything else calls
        // `admission_mode()`. `cargo test`'s default per-test-thread
        // model makes ordering non-deterministic across the WHOLE binary,
        // so this test instead calls `jammi_kernels::admission::admit`
        // directly with an explicit `Strict` mode, exercising the exact
        // same code `forward_fused_or_fallback` runs without depending on
        // the env-var memoization's timing.
        use jammi_kernels::admission::{admit, AdmissionMode};
        let counters = jammi_kernels::admission::DispatchCounters::new();
        let err = admit(
            AdmissionMode::Strict,
            "layer_norm_fused",
            "x_contiguous",
            false,
            &counters,
        )
        .expect_err("a failed predicate in Strict mode must error");
        assert!(matches!(
            err,
            jammi_kernels::error::KernelError::StrictModeFallback {
                op: "layer_norm_fused",
                predicate: "x_contiguous"
            }
        ));
    }

    /// #460 round-1 item 5b: `JAMMI_KERNELS_STRICT`-mode driven through
    /// [`LayerNorm::forward`] ITSELF (not `admit()` directly, unlike
    /// [`strict_mode_errors_instead_of_falling_back_on_a_failed_predicate`]
    /// above) on the BIASED arm, with a mismatched-dtype bias making the
    /// fused domain predicate fail. `admission_mode()` memoizes into a
    /// process-wide `OnceLock` inside `jammi_kernels` (the exact hazard
    /// that test's own doc names) — mirroring `jammi_kernels::admission`'s
    /// own `admission_mode_reads_strict_from_the_real_env_var_in_a_fresh_process`,
    /// this spawns a fresh CHILD process of this ALREADY-COMPILED test
    /// binary, `--exact`-targeted at
    /// [`layer_norm_forward_biased_strict_mode_child_process_body`] below,
    /// which is the only way to observe a real `JAMMI_KERNELS_STRICT=1`
    /// env var read deterministically rather than racing every other test
    /// in this binary for who initializes the `OnceLock` first.
    #[test]
    fn layer_norm_forward_biased_strict_mode_surfaces_a_typed_error_in_a_fresh_process() {
        let exe = std::env::current_exe().expect("test binary path");
        let output = std::process::Command::new(exe)
            .args([
                "layer_norm::tests::layer_norm_forward_biased_strict_mode_child_process_body",
                "--exact",
                "--nocapture",
            ])
            .env("JAMMI_KERNELS_STRICT", "1")
            .env("LN_FORWARD_STRICT_CHILD", "1")
            .output()
            .expect("spawn child test binary");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child process assertion failed: stdout={stdout}\nstderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        // Non-vacuity (family F): a filter matching zero tests still exits
        // 0 — assert the child actually ran (and passed) exactly the one
        // test it was told to run.
        assert!(
            stdout.contains("1 passed"),
            "the child process must have actually run (and passed) exactly one test -- \
             stdout={stdout}"
        );
    }

    /// Only meaningful inside the child process the test above spawns
    /// (`LN_FORWARD_STRICT_CHILD` set) — a silent no-op otherwise, so a
    /// stray direct `cargo test` run of this exact name (without the real
    /// `JAMMI_KERNELS_STRICT=1` env var already having won the `OnceLock`
    /// race) never produces a false pass OR a false fail.
    #[test]
    fn layer_norm_forward_biased_strict_mode_child_process_body() {
        // Positive-condition guard, no early `return` — mirrors
        // `jammi_kernels::admission::tests::admission_mode_child_process_body`'s
        // exact idiom (a stray direct run of this exact test name outside
        // the child process above is then simply a no-op assertion-free
        // pass, never a false RED).
        if std::env::var_os("LN_FORWARD_STRICT_CHILD").is_some() {
            let device = Device::Cpu;
            let hidden = 8;
            let x = Tensor::from_slice(&[0.1f32; 8], (1, hidden), &device).unwrap();
            let weight = Tensor::from_slice(&[1.0f32; 8], (hidden,), &device).unwrap();
            // Mismatched dtype vs x/weight (bf16 bias against an F32
            // x/weight pair) fails the fused domain predicate
            // (`dtype_f32_bf16_or_f16_matching_between_x_and_bias`); in
            // Strict mode that failure must surface as a typed error
            // through `forward()` itself rather than silently falling
            // back to `slow()`.
            let bias_bf16 =
                Tensor::from_slice(&[bf16::from_f32(0.2); 8], (hidden,), &device).unwrap();
            let (holds, predicate) = fused_admission_predicate_biased(&x, &weight, &bias_bf16);
            assert!(!holds, "fixture must actually fail the domain: {predicate}");

            let mut ln = LayerNorm {
                weight,
                bias: Some(bias_bf16),
                eps: 1e-5,
                training: true,
            };
            ln.set_training(true);
            let err = ln
                .forward(&x)
                .expect_err("Strict mode must error on a failed predicate, not silently fall back");
            assert!(
                matches!(
                    err,
                    EncoderError::Kernel(
                        jammi_kernels::error::KernelError::StrictModeFallback { .. }
                    )
                ),
                "expected a typed StrictModeFallback wrapped in EncoderError::Kernel, got {err:?}"
            );
        }
    }

    /// The admission/counter key this crate dispatches a fused path under
    /// is a call-site literal, independent of the kernel op's `name()` by
    /// construction (`jammi_kernels::admission::counters_for`'s doc): an
    /// admission key names a consumer's fused PATH, which may compose
    /// several ops, so it can legitimately differ from any one
    /// `CustomOp`'s name (the LoRA consumer keys `"lora_linear_fused"`
    /// over the op named `"low_rank_residual_linear"`). Where this crate
    /// keys a path by the op's own name — layer-norm and softmax — that
    /// coincidence is what lets a counters snapshot be read side by side
    /// with the op's error payloads, so it is pinned here without a third
    /// literal: the registry entry each `*_DISPATCH_COUNTERS` resolves to
    /// must be the very entry `counters_for(op.name())` resolves to
    /// (`counters_for` hands back the same `&'static` for the same key).
    /// The `admit(..)` call sites' `op` argument is the same key by
    /// convention but has no read-back API (it feeds a log-once WARN and
    /// the `StrictModeFallback` payload), so it stays pinned only by the
    /// strict-mode tests' literal matches above.
    #[test]
    fn dispatch_counter_keys_agree_with_the_kernel_ops_names() {
        use candle_core::CustomOp2;
        use jammi_kernels::admission::counters_for;
        use jammi_kernels::ops::{LayerNormFused, SoftmaxLastDimFused};
        assert!(
            std::ptr::eq(
                counters_for(LayerNormFused::new(1e-5, false).name()),
                *LN_DISPATCH_COUNTERS
            ),
            "LN_DISPATCH_COUNTERS is keyed by a literal that drifted from LayerNormFused::name()"
        );
        assert!(
            std::ptr::eq(
                counters_for(SoftmaxLastDimFused::default().name()),
                *crate::modernbert::SOFTMAX_DISPATCH_COUNTERS
            ),
            "SOFTMAX_DISPATCH_COUNTERS is keyed by a literal that drifted from \
             SoftmaxLastDimFused::name()"
        );
    }

    // -- esc-086: legacy `LayerNorm.gamma`/`.beta` name resolution --------

    // -- B2 (`#423` narrow-fix round 2): a hermetic, source-scanning proof
    // of this file's own call-site-inventory claim (see `LayerNorm::new`'s
    // doc, "EVERY OTHER call site" paragraph) rather than a hand-copied
    // comment that can silently drift. --------------------------------

    /// One statically-recognised shape for the LAST (`VarBuilder`) argument
    /// of a `LayerNorm::new(..)` call, as extracted by
    /// [`scan_layer_norm_new_call_sites`]. Any OTHER shape (a
    /// `.pp(format!(..))` runtime-formatted prefix, a multi-step method
    /// chain, etc.) makes the scan panic rather than silently skip the call
    /// site -- see that function's own doc.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum VbArgShape {
        /// `<ident>.pp("<literal>")` -- the literal segment appended to
        /// whatever prefix `<ident>` already carries. A `.pp()` call only
        /// ever APPENDS a segment (never splits or reinterprets its
        /// argument), so the full joined prefix's last `.`-segment is
        /// always this literal's OWN last `.`-segment, independent of
        /// whatever prefix `<ident>` already had -- which is what makes
        /// checking [`is_layer_norm_keyed`] directly on the bare literal
        /// (rather than reconstructing the full joined path) sound.
        PpLiteral(String),
        /// A bare `VarBuilder` identifier, no `.pp(..)` at all -- the
        /// builder's own existing prefix is consulted unchanged.
        Bare,
    }

    /// One `LayerNorm::new(..)` occurrence found by
    /// [`scan_layer_norm_new_call_sites`].
    #[derive(Debug, Clone)]
    struct LayerNormNewCallSite {
        /// Path relative to `src/`, e.g. `"bert.rs"`.
        file: String,
        /// 1-indexed line of the `LayerNorm::new(` token itself (NOT the
        /// `.pp(..)` argument, which may be several lines later) --
        /// informational only; every hard assertion in
        /// [`layer_norm_new_call_sites_are_pinned_to_the_known_set`] is
        /// keyed on `(file, shape)` content, not on this line number, so a
        /// reformat that shifts line numbers without changing any call
        /// site's shape does not fail the pin (the audit's own directive:
        /// "close the class... not the named lines").
        line: usize,
        shape: VbArgShape,
        /// Whether this occurrence sits at or after this file's own
        /// `#[cfg(test)] mod tests {` boundary (see
        /// [`find_mod_tests_boundary_line`]).
        is_test: bool,
    }

    /// Recursively collects every `*.rs` path under `dir`, in a fixed,
    /// deterministic (sorted) order (family J) -- so
    /// [`scan_layer_norm_new_call_sites`]'s output order does not depend on
    /// the host filesystem's directory-listing order.
    fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let entries =
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|e| panic!("dir entry in {}: {e}", dir.display()));
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    /// The 1-indexed line number of a file's own `mod tests {` boundary,
    /// gated on the line immediately above it (ignoring blank lines) being
    /// literally `#[cfg(test)]` -- the exact shape every test module in
    /// this crate uses. `None` when the file has no such module at all
    /// (`bert.rs`/`distilbert.rs` today) -- every occurrence in such a file
    /// is then treated as production.
    fn find_mod_tests_boundary_line(content: &str) -> Option<usize> {
        let lines: Vec<&str> = content.lines().collect();
        let mut prev_non_blank: Option<&str> = None;
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed == "mod tests {" && prev_non_blank == Some("#[cfg(test)]") {
                return Some(i + 1);
            }
            if !trimmed.is_empty() {
                prev_non_blank = Some(trimmed);
            }
        }
        None
    }

    /// Balanced-paren extraction (string-literal-aware, so a `"` inside a
    /// tracked string can never desynchronise the depth count) of the
    /// argument-list text between the `(` at `open_paren_idx` and its
    /// matching `)`, EXCLUSIVE of both parens. `open_paren_idx` must point
    /// at the `(` byte itself. Byte-indexed rather than char-indexed: every
    /// tracked byte (`"`, `\`, `(`, `)`) is ASCII, and ASCII bytes can never
    /// be mistaken for a continuation/lead byte of a multi-byte UTF-8
    /// sequence (those are always `>= 0x80`), so scanning by byte value
    /// alone is safe even though the surrounding source text is not
    /// ASCII-only (e.g. an em dash in a doc comment) -- and both `start`
    /// and every returned slice boundary sit immediately after an ASCII
    /// byte, which is always a valid `str` char boundary.
    fn extract_balanced_args(content: &str, open_paren_idx: usize) -> String {
        let bytes = content.as_bytes();
        assert_eq!(
            bytes[open_paren_idx], b'(',
            "expected '(' at byte {open_paren_idx}"
        );
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        let start = open_paren_idx + 1;
        let mut i = open_paren_idx;
        while i < bytes.len() {
            let b = bytes[i];
            if in_string {
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_string = false;
                }
            } else {
                match b {
                    b'"' => in_string = true,
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            return content[start..i].to_string();
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        panic!("unbalanced parens scanning LayerNorm::new( starting at byte {open_paren_idx}");
    }

    /// Splits `args_text` on top-level (paren-depth-0, outside any string
    /// literal) commas -- the same string/paren-aware scan
    /// [`extract_balanced_args`] uses, so a `format!("{n}")`-style nested
    /// call's own internal commas (none exist in this crate's
    /// `LayerNorm::new` call sites today, but a future one might) never
    /// get mistaken for an argument separator.
    fn split_top_level_commas(args_text: &str) -> Vec<String> {
        let bytes = args_text.as_bytes();
        let mut parts = Vec::new();
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        let mut start = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            let b = bytes[i];
            if in_string {
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_string = false;
                }
            } else {
                match b {
                    b'"' => in_string = true,
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    b',' if depth == 0 => {
                        parts.push(args_text[start..i].to_string());
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        parts.push(args_text[start..].to_string());
        parts
    }

    /// Parses `<ident>.pp("<literal>")` exactly (no other punctuation
    /// permitted before/after), returning the literal. `None` for anything
    /// else, including a superficially similar `.pp("...")foo` or a
    /// non-identifier receiver.
    fn parse_pp_literal(text: &str) -> Option<String> {
        let idx = text.find(".pp(\"")?;
        let recv = &text[..idx];
        let mut recv_chars = recv.chars();
        match recv_chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return None,
        }
        if !recv_chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        let after = &text[idx + 5..];
        let end_quote = after.find('"')?;
        let literal = &after[..end_quote];
        if after[end_quote + 1..].trim() != ")" {
            return None;
        }
        Some(literal.to_string())
    }

    /// True for a bare identifier (`vb`, `frozen_vb`, ...) with no
    /// trailing `.pp(..)` (or any other punctuation) at all.
    fn is_bare_ident(text: &str) -> bool {
        let mut chars = text.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return false,
        }
        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// Scans this crate's own `src/**/*.rs` files (via
    /// `env!("CARGO_MANIFEST_DIR")`) for every literal `LayerNorm::new(`
    /// occurrence and classifies the LAST (`VarBuilder`) argument passed at
    /// each one. Comment-only mentions are skipped (the occurrence's own
    /// source line, trimmed, must not start with `//`) -- none exist in
    /// this crate today, but a future doc comment quoting the call
    /// literally must not be double-counted as a real call site.
    ///
    /// Three argument shapes are recognised, explicitly:
    ///  * `<ident>.pp("<literal>")` — [`VbArgShape::PpLiteral`].
    ///  * a bare `VarBuilder` identifier — [`VbArgShape::Bare`].
    ///  * `<ident>.pp(format!(..))` — NOT resolvable to a static literal;
    ///    this scan panics rather than silently treating it as non-keyed
    ///    (no in-tree call site takes this shape today).
    ///
    /// Any OTHER shape also panics, naming the file/line and raw text —
    /// silently skipping an unrecognised call site would let a future
    /// keyed-or-not site go unchecked by the pinned invariant this scan
    /// backs.
    fn scan_layer_norm_new_call_sites() -> Vec<LayerNormNewCallSite> {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        collect_rs_files(&src_dir, &mut files);
        files.sort();

        let mut sites = Vec::new();
        let pattern = "LayerNorm::new(";
        for path in &files {
            // This file (`layer_norm.rs` itself, the definition site) is
            // excluded: it can never contain a real call site, and its own
            // source text necessarily spells out the search pattern above
            // plus every diagnostic message this scan prints -- both of
            // which are literal `LayerNorm::new(`-shaped substrings that
            // would otherwise self-match (the pattern string itself, and
            // every panic!/println! message quoting it, would each look
            // like a malformed call site to this same scan).
            if path.file_name().and_then(|n| n.to_str()) == Some("layer_norm.rs") {
                continue;
            }
            let content = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let rel_name = path
                .strip_prefix(&src_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();
            let test_boundary_line = find_mod_tests_boundary_line(&content);

            let mut search_from = 0usize;
            while let Some(rel_idx) = content[search_from..].find(pattern) {
                let idx = search_from + rel_idx;
                search_from = idx + pattern.len();

                let line_start = content[..idx].rfind('\n').map(|p| p + 1).unwrap_or(0);
                if content[line_start..idx].trim_start().starts_with("//") {
                    continue;
                }

                let line = 1 + content[..idx].matches('\n').count();
                let open_paren_idx = idx + "LayerNorm::new".len();
                let args_text = extract_balanced_args(&content, open_paren_idx);
                let last_arg = split_top_level_commas(&args_text)
                    .into_iter()
                    .map(|s| s.trim().to_string())
                    .rfind(|s| !s.is_empty())
                    .unwrap_or_else(|| {
                        panic!("{rel_name}:{line}: LayerNorm::new(..) has no arguments at all")
                    });

                let shape = if let Some(literal) = parse_pp_literal(&last_arg) {
                    VbArgShape::PpLiteral(literal)
                } else if is_bare_ident(&last_arg) {
                    VbArgShape::Bare
                } else if last_arg.contains(".pp(format!(") {
                    panic!(
                        "{rel_name}:{line}: found a `.pp(format!(...))` LayerNorm::new(..) \
                         VarBuilder argument (`{last_arg}`) -- this scan cannot statically \
                         resolve a runtime-formatted prefix's literal segment; manually verify \
                         whether this site is LayerNorm-keyed and extend this scan (rather than \
                         silently skipping it) before trusting the pinned invariant"
                    );
                } else {
                    panic!(
                        "{rel_name}:{line}: unrecognised LayerNorm::new(..) VarBuilder-argument \
                         shape: `{last_arg}` -- this scan only recognises \
                         `<ident>.pp(\"<literal>\")` and a bare VarBuilder identifier; update \
                         this scan (and its pinned expectations) rather than silently skipping \
                         this call site"
                    );
                };

                let is_test = test_boundary_line.is_some_and(|b| line >= b);
                sites.push(LayerNormNewCallSite {
                    file: rel_name.clone(),
                    line,
                    shape,
                    is_test,
                });
            }
        }
        sites
    }

    /// [`is_layer_norm_keyed`]'s positive/negative boundary, pinned against
    /// REAL per-site literals harvested by [`scan_layer_norm_new_call_sites`]
    /// (not a hand-typed table that can silently drift -- a previous
    /// version of this test carried a PHANTOM `"emb_norm"` case that no
    /// in-tree call site ever actually writes; the real ModernBERT
    /// embeddings-norm prefix is `"model.embeddings.norm"`,
    /// `modernbert.rs:3358`), plus the two SYNTHETIC boundary strings
    /// `esc-086` names (`LayerNormX`, and a substring-but-not-suffix
    /// `gamma_scale`) that no in-tree site writes and therefore cannot be
    /// harvested.
    #[test]
    fn is_layer_norm_keyed_matches_only_a_trailing_layer_norm_segment() {
        let sites = scan_layer_norm_new_call_sites();
        let mut harvested_literals: Vec<String> = sites
            .iter()
            .filter_map(|s| match &s.shape {
                VbArgShape::PpLiteral(lit) => Some(lit.clone()),
                VbArgShape::Bare => None,
            })
            .collect();
        harvested_literals.sort();
        harvested_literals.dedup();
        assert!(
            !harvested_literals.is_empty(),
            "the scan harvested zero literals -- it is almost certainly broken, not that this \
             crate suddenly has no LayerNorm::new(..) call sites"
        );

        // Ground truth: exactly these DISTINCT literal shapes are
        // `LayerNorm`-keyed among everything this crate's source actually
        // writes today (see `layer_norm_new_call_sites_are_pinned_to_the_known_set`
        // for the full file-scoped pin).
        let known_keyed: &[&str] = &[
            "LayerNorm",
            "attention.output.LayerNorm",
            "output.LayerNorm",
        ];

        for literal in &harvested_literals {
            let expected = known_keyed.contains(&literal.as_str());
            assert_eq!(
                is_layer_norm_keyed(literal),
                expected,
                "harvested real literal `{literal}` expected is_layer_norm_keyed == {expected}"
            );
        }

        let synthetic_cases: &[(&str, bool)] = &[
            ("embeddings.LayerNormX", false),
            ("embeddings.gamma_scale", false),
        ];
        for (prefix, expected) in synthetic_cases {
            assert_eq!(
                is_layer_norm_keyed(prefix),
                *expected,
                "synthetic boundary prefix `{prefix}` expected is_layer_norm_keyed == {expected}"
            );
        }
    }

    /// B2 (`#423` narrow-fix round 2): CHECKS, not merely documents, this
    /// module's own call-site-inventory claim (see `LayerNorm::new`'s doc).
    /// A previous version of that doc claimed "17 sites ... the only
    /// bare-`vb` call sites are `modernbert.rs:4216`, `9421-9423`" — every
    /// part of that was wrong: there are 26 occurrences, not 17;
    /// `9421`-`9423` are `.pp(..)`-scoped, not bare; and the single
    /// bare-`vb` site is `4216` alone. This test makes a future silent
    /// drift (a new `.pp("LayerNorm")` site, a newly-bare production call,
    /// a changed total count) fail loudly instead of re-drifting the doc.
    #[test]
    fn layer_norm_new_call_sites_are_pinned_to_the_known_set() {
        let sites = scan_layer_norm_new_call_sites();
        for s in &sites {
            println!(
                "LayerNorm::new(..) site: {}:{} is_test={} shape={:?}",
                s.file, s.line, s.is_test, s.shape
            );
        }

        assert_eq!(
            sites.len(),
            26,
            "total LayerNorm::new(..) occurrence count drifted from the pinned 26 -- a call \
             site was added or removed; update this pin only after reviewing whether the \
             new/removed site is LayerNorm-keyed"
        );

        let production: Vec<&LayerNormNewCallSite> = sites.iter().filter(|s| !s.is_test).collect();
        let test_only: Vec<&LayerNormNewCallSite> = sites.iter().filter(|s| s.is_test).collect();
        assert_eq!(
            production.len(),
            22,
            "production call-site count drifted from the pinned 22"
        );
        assert_eq!(
            test_only.len(),
            4,
            "#[cfg(test)]-mod call-site count drifted from the pinned 4"
        );

        // The LayerNorm-keyed SET, checked via the REAL `is_layer_norm_keyed`
        // predicate (not a re-implementation) -- a future `.pp("LayerNorm")`
        // (or any other literal whose last `.`-segment is `LayerNorm`) site
        // fails this pin until reviewed.
        let mut keyed_files_and_literals: Vec<(String, String)> = production
            .iter()
            .filter_map(|s| match &s.shape {
                VbArgShape::PpLiteral(lit) if is_layer_norm_keyed(lit) => {
                    Some((s.file.clone(), lit.clone()))
                }
                _ => None,
            })
            .collect();
        keyed_files_and_literals.sort();

        let mut expected_keyed: Vec<(String, String)> = vec![
            ("bert.rs".to_string(), "LayerNorm".to_string()),
            (
                "bert.rs".to_string(),
                "attention.output.LayerNorm".to_string(),
            ),
            ("bert.rs".to_string(), "output.LayerNorm".to_string()),
            ("distilbert.rs".to_string(), "LayerNorm".to_string()),
        ];
        expected_keyed.sort();
        assert_eq!(
            keyed_files_and_literals, expected_keyed,
            "the LayerNorm-keyed call-site SET drifted from the pinned 4 (bert.rs's \
             `LayerNorm`/`attention.output.LayerNorm`/`output.LayerNorm`, distilbert.rs's \
             `LayerNorm`) -- review whether a newly-added or newly-renamed site should \
             legitimately alias legacy gamma/beta before updating this pin"
        );

        // No PRODUCTION site may be a bare-`vb` call (no `.pp(..)` at all)
        // -- only a `#[cfg(test)]`-gated fixture does that today, and it is
        // deliberately NOT LayerNorm-keyed (a fresh `VarBuilder::from_varmap`'s
        // own root prefix is empty).
        let production_bare: Vec<&&LayerNormNewCallSite> = production
            .iter()
            .filter(|s| s.shape == VbArgShape::Bare)
            .collect();
        assert!(
            production_bare.is_empty(),
            "a PRODUCTION LayerNorm::new(..) call site now uses a bare VarBuilder with no \
             `.pp(..)` at all: {production_bare:?} -- review this site by hand before trusting \
             the pin above (this scan's is_layer_norm_keyed reasoning assumes every production \
             site scopes its own segment via `.pp(\"<literal>\")`)"
        );

        let test_bare_files: Vec<&str> = test_only
            .iter()
            .filter(|s| s.shape == VbArgShape::Bare)
            .map(|s| s.file.as_str())
            .collect();
        assert_eq!(
            test_bare_files,
            vec!["modernbert.rs"],
            "the single bare-vb call site drifted from the pinned single modernbert.rs test \
             fixture"
        );
    }

    /// Independent, separately-authored reference for
    /// [`resolve_affine_names`]'s full name-resolution lattice (family F: a
    /// second derivation, not the same code re-run): collision on either
    /// read axis reports that axis as `None` (Err), weight axis checked
    /// first so a double collision always reports weight; otherwise each
    /// axis independently prefers its legacy name only when present
    /// without its modern counterpart, and `with_bias == false` never
    /// returns a bias name.
    fn resolve_affine_names_reference(
        has_w: bool,
        has_g: bool,
        has_b: bool,
        has_beta: bool,
        with_bias: bool,
    ) -> Option<(&'static str, Option<&'static str>)> {
        if has_w && has_g {
            return None;
        }
        let name_w = if has_g { "gamma" } else { "weight" };
        if !with_bias {
            return Some((name_w, None));
        }
        if has_b && has_beta {
            return None;
        }
        let name_b = if has_beta { "beta" } else { "bias" };
        Some((name_w, Some(name_b)))
    }

    /// The full [`resolve_affine_names`] lattice (B3, `#423` narrow-fix
    /// round 2): ALL 16 `(has_w, has_g, has_b, has_beta)` cells for
    /// `with_bias == true`, and all 4 `(has_w, has_g)` cells for
    /// `with_bias == false` -- generated by looping over the bool lattice
    /// rather than hand-picking 7 of the 16 cells (a previous version of
    /// this test covered only 7), with the expected outcome computed by
    /// [`resolve_affine_names_reference`], a SEPARATE tiny reimplementation
    /// (not the same code re-run).
    #[test]
    fn resolve_affine_names_lattice() {
        const BOOLS: [bool; 2] = [false, true];
        let mut cases_checked = 0usize;
        for has_w in BOOLS {
            for has_g in BOOLS {
                for with_bias in BOOLS {
                    let bias_axis: &[(bool, bool)] = if with_bias {
                        &[(false, false), (false, true), (true, false), (true, true)]
                    } else {
                        &[(false, false)]
                    };
                    for &(has_b, has_beta) in bias_axis {
                        cases_checked += 1;
                        let expected = resolve_affine_names_reference(
                            has_w, has_g, has_b, has_beta, with_bias,
                        );
                        let result = resolve_affine_names(
                            "test.prefix",
                            has_w,
                            has_g,
                            has_b,
                            has_beta,
                            with_bias,
                        );
                        let case_desc = format!(
                            "has_w={has_w} has_g={has_g} has_b={has_b} has_beta={has_beta} \
                             with_bias={with_bias}"
                        );
                        match expected {
                            Some(exp) => {
                                let got = result.unwrap_or_else(|e| {
                                    panic!("{case_desc}: expected Ok, got Err({e:?})")
                                });
                                assert_eq!(got, exp, "{case_desc}");
                            }
                            None => {
                                result
                                    .expect_err(&format!("{case_desc}: expected Err (collision)"));
                            }
                        }
                    }
                }
            }
        }
        // 2*2*4 (with_bias=true) + 2*2*1 (with_bias=false) = 16 + 4 = 20.
        assert_eq!(
            cases_checked, 20,
            "the lattice loop's own case count drifted from the pinned 16 + 4 = 20"
        );
    }

    /// Preserves a specific edge check the lattice loop above does NOT
    /// cover (it fixes `has_b = has_beta = false` for every `with_bias ==
    /// false` cell, matching how every real caller invokes this function):
    /// `with_bias == false` must ignore `has_b`/`has_beta` even when a
    /// (never-real) caller passes them as `true` -- pinning the function's
    /// OWN robustness, independent of every real call site already
    /// honoring the "never consult beta when bias-free" rule.
    #[test]
    fn resolve_affine_names_with_bias_false_ignores_bogus_bias_presence() {
        assert_eq!(
            resolve_affine_names("test.prefix", true, false, true, true, false).unwrap(),
            ("weight", None)
        );
        assert_eq!(
            resolve_affine_names("test.prefix", false, true, true, true, false).unwrap(),
            ("gamma", None)
        );
    }

    /// The double-collision determinism arm, isolated: when BOTH axes
    /// carry both names, the error message names the WEIGHT axis's
    /// candidate names (`weight`/`gamma`), never the bias axis's
    /// (`bias`/`beta`) -- pinned as its own test since the lattice test
    /// above only checks `Err`-ness, not message content.
    #[test]
    fn resolve_affine_names_double_collision_reports_weight_axis() {
        let err = resolve_affine_names("embeddings.LayerNorm", true, true, true, true, true)
            .expect_err("double collision must be Err");
        let msg = err.to_string();
        assert!(
            msg.contains("weight") && msg.contains("gamma"),
            "double collision message must name the weight axis's own \
             candidates: {msg}"
        );
        assert!(
            msg.contains("embeddings.LayerNorm"),
            "message must name the prefix: {msg}"
        );
    }

    /// B1b (`#423` narrow-fix round 2): arm6b's replacement, a DIRECT seam
    /// test (no in-tree model ever builds a `VarBuilder` at a synthetic
    /// `embeddings.LayerNormX`-style prefix, so this exercises a REAL
    /// production non-keyed prefix instead). A temp safetensors file
    /// carries ONLY `sa_layer_norm.gamma`/`.beta` (DistilBERT's own actual
    /// prefix, `distilbert.rs`'s `layer_vb.pp("sa_layer_norm")`) plus the
    /// SAME values under `LayerNorm.gamma`/`.beta` AND under
    /// `LayerNormX.gamma`/`.beta`, all in the SAME file. `sa_layer_norm`
    /// is NOT aliased by a `starts_with`/`contains`/case-insensitive
    /// mutant of [`is_layer_norm_keyed`] (round-2 fix: the ORIGINAL text
    /// here claimed otherwise, which is false for all three) — it
    /// neither starts with nor contains the literal `LayerNorm` (the
    /// underscore breaks both: `"sa_layer_norm"` vs `"LayerNorm"`), and
    /// lower-casing either side still leaves `"sa_layer_norm" !=
    /// "layernorm"`; only an ALWAYS-TRUE mutant, or one that strips/
    /// normalizes underscores before comparing, would alias `sa_layer_norm`
    /// specifically. The `starts_with`/`contains` mutant family is instead
    /// killed end to end by this SAME fixture's `LayerNormX.gamma`/`.beta`
    /// pair (below): `"LayerNormX"` DOES start with and contain the
    /// literal `LayerNorm`, so a `starts_with`/`contains` mutant WOULD
    /// (wrongly) alias it and find `gamma`/`beta` genuinely present — the
    /// real, exact-segment check must refuse it instead, even though the
    /// SAME file's genuinely `LayerNorm`-keyed prefix (proving the
    /// fixture's tensors ARE readable at all, not merely that
    /// `sa_layer_norm`/`LayerNormX` are malformed) does alias.
    #[test]
    fn direct_seam_non_layer_norm_keyed_prefix_containing_layer_norm_substring_is_not_aliased() {
        let device = Device::Cpu;
        let hidden = 4usize;
        let eps = 1e-5f64;
        let gamma_vals = [0.7f32, 1.3, -0.2, 2.1];
        let beta_vals = [0.05f32, -0.3, 0.9, -1.1];
        let gamma = Tensor::from_slice(&gamma_vals, (hidden,), &device).unwrap();
        let beta = Tensor::from_slice(&beta_vals, (hidden,), &device).unwrap();

        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        tensors.insert("sa_layer_norm.gamma".to_string(), gamma.clone());
        tensors.insert("sa_layer_norm.beta".to_string(), beta.clone());
        tensors.insert("LayerNorm.gamma".to_string(), gamma.clone());
        tensors.insert("LayerNorm.beta".to_string(), beta.clone());
        tensors.insert("LayerNormX.gamma".to_string(), gamma.clone());
        tensors.insert("LayerNormX.beta".to_string(), beta.clone());

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sa_layer_norm_direct_seam.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &device).unwrap() };

        // `sa_layer_norm` must NOT be aliased: no `weight` there, and the
        // non-keyed branch never even probes `gamma`.
        let err = LayerNorm::new(hidden, eps, true, vb.pp("sa_layer_norm"))
            .err()
            .expect(
                "a non-LayerNorm-keyed prefix carrying only gamma/beta must be Err, not aliased",
            );
        assert!(
            matches!(err, EncoderError::Tensor(_)),
            "expected the non-keyed branch's plain `?`-propagated candle CannotFindTensor \
             error (EncoderError::Tensor), got {err:?}"
        );

        // `LayerNormX` is what actually kills the `starts_with`/`contains`
        // mutant family end to end through this seam (see this test's own
        // doc): it DOES start with and contain the literal `LayerNorm`, so
        // either mutant would (wrongly) alias it — the real, exact-segment
        // check must refuse it just like `sa_layer_norm` above.
        let err_x = LayerNorm::new(hidden, eps, true, vb.pp("LayerNormX"))
            .err()
            .expect(
                "a `LayerNormX` prefix (starts-with/contains `LayerNorm` but not equal to it) \
                 carrying only gamma/beta must be Err, not aliased",
            );
        assert!(
            matches!(err_x, EncoderError::Tensor(_)),
            "expected the non-keyed branch's plain `?`-propagated candle CannotFindTensor \
             error (EncoderError::Tensor), got {err_x:?}"
        );

        // The SAME values, at a genuinely LayerNorm-keyed prefix in the
        // SAME file, DO alias -- and match bitwise.
        let ln = LayerNorm::new(hidden, eps, true, vb.pp("LayerNorm"))
            .expect("a genuinely LayerNorm-keyed prefix must alias gamma/beta");
        let got_weight: Vec<f32> = ln.weight.flatten_all().unwrap().to_vec1().unwrap();
        let got_bias: Vec<f32> = ln
            .bias
            .as_ref()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (g, w)) in gamma_vals.iter().zip(got_weight.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "weight[{i}] must equal legacy gamma bitwise"
            );
        }
        for (i, (b, w)) in beta_vals.iter().zip(got_bias.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                w.to_bits(),
                "bias[{i}] must equal legacy beta bitwise"
            );
        }
    }
}
