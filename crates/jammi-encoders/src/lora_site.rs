//! The one LoRA-WRAPPING decision every non-BERT-family tower shares.
//!
//! # Why WRAP and RESOLVE are split
//!
//! A LoRA site has THREE independent axes, and only the first two are the
//! same across towers:
//!
//! 1. the **selector name** (`module_name`) that
//!    [`jammi_lora::should_apply_lora`] matches a caller's `target_modules`
//!    against;
//! 2. the **adapter subpath** (`lora_subpath`) the A/B tensors are
//!    registered/loaded under inside the trainable `VarBuilder`;
//! 3. the **base-tensor locator** — where the frozen weight actually lives
//!    in the checkpoint.
//!
//! Axis 3 is genuinely per-tower and cannot be folded in here. `bert.rs`'s
//! own layer-scoped `LoraSite` can own it because every BERT-family base is
//! `linear(in, out, layer_vb.pp(module_name))` — one shape, derivable from
//! the selector name. The OpenCLIP towers' fused QKV site is not: its base
//! is a FLAT `in_proj_weight`/`in_proj_bias` pair read directly off the
//! attention `VarBuilder` (`crate::attention::MultiHeadAttention::load`),
//! with no `in_proj` sub-module to `pp` into at all, and a `[3*width,
//! width]` geometry no `(in_features, out_features)` pair from the selector
//! name would produce. HTSAT's `PatchMerging::reduction` is
//! `linear_no_bias`, and `ClapAudioProjection`'s pair sits outside the block
//! stack entirely.
//!
//! So this type takes an ALREADY-RESOLVED [`FrozenBase`] and owns only the
//! decision that is common: *does this site get an adapter, at what rank,
//! and where do its A/B tensors live?* Each tower's loader keeps resolving
//! its own bases exactly as it always has, which is also why the frozen
//! (unadapted) path stays byte-identical by construction — an unselected
//! site is `MaybeLoraLinear::Frozen(base)` around the very same `Linear` the
//! loader built before this module existed.
//!
//! The BERT family is deliberately NOT migrated onto this seam: it has its
//! own working, proven site builder, and a migration would re-open three
//! shipped surfaces for no behaviour change.

use candle_nn::{VarBuilder, VarMap};
use jammi_lora::{
    effective_rank, should_apply_lora, FrozenBase, LoraBuildConfig, LoraLinear, MaybeLoraLinear,
};

use crate::error::EncoderError;

/// The immutable per-indexed-unit LoRA context a tower's loader threads
/// through its sites, so each call carries only what actually varies: the
/// selector name and the adapter subpath.
pub(crate) struct LoraSite<'a> {
    /// The TRAINABLE `VarBuilder`, already scoped to the indexed unit (e.g.
    /// `resblocks.{n}`), so [`Self::wrap`]'s `lora_subpath` is a single leaf
    /// segment under it. Never the frozen backbone builder: LoRA A/B always
    /// live in F32 and in the caller's `VarMap` (training) or the adapter
    /// safetensors file (inference).
    pub(crate) lora_vb: &'a VarBuilder<'a>,
    /// The index of the repeating unit this site belongs to, or `None` for a
    /// site that belongs to NO numbered unit (a head-side projection). See
    /// [`jammi_lora::should_apply_lora`]'s own doc for what `None` means
    /// against an active `layers_to_transform` filter — it is a refusal, not
    /// a pass.
    pub(crate) layer_idx: Option<usize>,
    /// The caller's build config: which selectors match, at what rank/alpha,
    /// with which init mode, dropout, and seed.
    pub(crate) lora: &'a LoraBuildConfig<'a>,
    /// The `VarMap` the seeded A/B tensors are registered into on the
    /// training path.
    pub(crate) varmap: &'a VarMap,
}

impl LoraSite<'_> {
    /// Wrap an already-resolved frozen `base` in a LoRA adapter when the
    /// build config selects `module_name` at this site's `layer_idx`;
    /// otherwise hand it back as `MaybeLoraLinear::Frozen(base)` — the same
    /// weight, forwarded through `candle_nn::Linear::forward` bit-for-bit
    /// (`jammi_lora::FrozenBase::forward`'s own doc pins that preservation).
    ///
    /// `lora_subpath` is the adapter key prefix under [`Self::lora_vb`]; it
    /// is a separate argument from `module_name` on purpose (module doc,
    /// axes 1 and 2): a tower may select a site by one name and persist its
    /// weights under another.
    pub(crate) fn wrap(
        &self,
        base: FrozenBase,
        module_name: &str,
        lora_subpath: &str,
    ) -> Result<MaybeLoraLinear, EncoderError> {
        if !should_apply_lora(
            module_name,
            self.lora.target_modules,
            self.layer_idx,
            self.lora.layers_to_transform,
        ) {
            return Ok(MaybeLoraLinear::Frozen(base));
        }
        let rank = effective_rank(module_name, self.lora.lora_rank, self.lora.rank_pattern);
        let lora_linear = LoraLinear::new_with_base(
            base,
            rank,
            self.lora.lora_alpha,
            self.lora.use_rslora,
            self.lora.init_mode,
            self.lora.lora_dropout,
            self.lora.seed,
            self.varmap,
            &self.lora_vb.pp(lora_subpath),
        )?;
        Ok(MaybeLoraLinear::Lora(lora_linear))
    }
}

/// Owns the two borrow targets a "wraps nothing" [`LoraSite`] needs, so a
/// tower's frozen `load(vb, config)` entry point can run the SAME
/// construction code its builder does instead of keeping a second, parallel
/// loader that could drift from it.
///
/// [`LoraBuildConfig::frozen`] has an EMPTY `target_modules`, so
/// [`jammi_lora::should_apply_lora`] declines every site and
/// [`LoraSite::wrap`] returns `MaybeLoraLinear::Frozen(base)` at each one —
/// the owned `VarMap` is never written to and the trainable `VarBuilder`
/// (which the caller passes as the frozen backbone builder, there being no
/// trainable one) is never read from. That is why a tower built through this
/// holder is bit-identical to one built with no LoRA seam at all.
pub(crate) struct FrozenSiteHolder {
    varmap: VarMap,
    lora: LoraBuildConfig<'static>,
}

impl FrozenSiteHolder {
    pub(crate) fn new() -> Self {
        Self {
            varmap: VarMap::new(),
            lora: LoraBuildConfig::frozen(),
        }
    }

    /// A [`LoraSite`] that declines every site. `layer_idx` is `None`: with
    /// `frozen()`'s empty selector list the index is never consulted, and
    /// `None` states honestly that this holder knows of no indexed unit.
    pub(crate) fn site<'a>(&'a self, vb: &'a VarBuilder<'a>) -> LoraSite<'a> {
        LoraSite {
            lora_vb: vb,
            layer_idx: None,
            lora: &self.lora,
            varmap: &self.varmap,
        }
    }
}
