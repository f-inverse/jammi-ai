//! The ONE OpenCLIP residual-attention block, shared by both OpenCLIP
//! towers.
//!
//! `crate::clip_text` and `crate::open_clip_vision` load the same
//! architecture: `LN → fused-QKV MHSA → residual → LN → QuickGelu MLP →
//! residual`, under the same checkpoint path (`transformer.resblocks.{n}`)
//! with the same four LoRA site names (`in_proj`, `out_proj`, `c_fc`,
//! `c_proj`). Only three things differ, and all three are ARGUMENTS here
//! rather than a second copy of the block:
//!
//! 1. the MLP's intermediate width — the text tower's fixed `4 * width`
//!    versus the vision config's `mlp_ratio * width`;
//! 2. the attention mask — `Some(causal)` for the causally-masked text
//!    tower, `None` for the bidirectional vision tower
//!    ([`crate::attention::MultiHeadAttention::forward`]'s own `Option`);
//! 3. the ADAPTER KEY ROOT the traversal helpers prefix every key with —
//!    see "Two towers, two namespaces" below.
//!
//! # Two towers, two namespaces
//!
//! An OpenCLIP checkpoint holds BOTH towers, and `jammi-ai` holds ONE
//! `VarMap` per run. The two towers' adapter key namespaces are disjoint by
//! construction: the text tower lives at the checkpoint root and keys its
//! adapters `resblocks.{n}.{site}.lora_{a,b}`, while the vision tower's
//! weights live under `visual.` in every OpenCLIP safetensors file, so its
//! adapter keys are `visual.resblocks.{n}.{site}.lora_{a,b}`. Each tower
//! passes its own root as `adapter_root`, so building both into one
//! `VarMap` registers two independent sets of `Var`s rather than aliasing
//! one tower's weights onto the other — candle's `VarBuilder::get` returns
//! the ALREADY-REGISTERED `Var` for a name it has seen, so a shared key
//! namespace would silently collapse the second tower's adapter into the
//! first's: half the intended trainable parameters, one gradient stream
//! feeding two towers, and an exported adapter whose vision weights are
//! literally its text weights. Disjoint namespaces are what rules that out.

use std::collections::HashMap;

use candle_core::Tensor;
use candle_nn::{linear, VarBuilder};
use jammi_lora::{FrozenBase, MaybeLoraLinear};

use crate::attention::MultiHeadAttention;
use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::lora_site::LoraSite;

/// The checkpoint subtree both towers' residual blocks live under, relative
/// to that tower's own `VarBuilder` root.
const BLOCK_STACK_PATH: &str = "transformer.resblocks";

/// The MLP's two LoRA-wrappable linear sites, named exactly as the OpenCLIP
/// checkpoint names them — the selector string a caller writes in
/// `target_modules` AND the adapter subpath leaf.
const C_FC_SITE: &str = "c_fc";
/// See [`C_FC_SITE`].
const C_PROJ_SITE: &str = "c_proj";

/// The selector names a caller may write in `target_modules` to reach
/// EITHER OpenCLIP tower's LoRA sites — both towers load this same block,
/// so both have exactly these four and `AnyEncoder::lora_site_names`
/// returns this one list for `ClipText` and `OpenClipVision` alike.
///
/// Built from the four site-name constants themselves (two here, two in
/// `crate::attention`), in `ResidualAttentionBlock::lora_sites`' order, so
/// it cannot drift from the names the sites are actually wrapped under. A
/// test asserts every entry selects at least one real site on a fixture
/// while the union of all of them is exactly what `all-linear` selects.
pub(crate) const LORA_SITE_NAMES: &[&str] = &[
    crate::attention::IN_PROJ_SITE,
    crate::attention::OUT_PROJ_SITE,
    C_FC_SITE,
    C_PROJ_SITE,
];

/// Feed-forward MLP with QuickGelu activation.
struct Mlp {
    c_fc: MaybeLoraLinear,
    c_proj: MaybeLoraLinear,
}

impl Mlp {
    /// One construction path (`crate::lora_site`'s module doc): the bases
    /// are resolved here exactly as they always were and then offered to
    /// `site`, which declines every one of them under a
    /// `LoraBuildConfig::frozen()` config.
    fn load_with(
        vb: VarBuilder,
        width: usize,
        intermediate_size: usize,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let c_fc = linear(width, intermediate_size, vb.pp(C_FC_SITE))?;
        let c_proj = linear(intermediate_size, width, vb.pp(C_PROJ_SITE))?;
        Ok(Self {
            c_fc: site.wrap(FrozenBase::Dense(c_fc), C_FC_SITE, C_FC_SITE)?,
            c_proj: site.wrap(FrozenBase::Dense(c_proj), C_PROJ_SITE, C_PROJ_SITE)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let x = self.c_fc.forward(x)?;
        let x = crate::activations::quick_gelu(&x)?;
        Ok(self.c_proj.forward(&x)?)
    }

    fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 2] {
        [(C_FC_SITE, &self.c_fc), (C_PROJ_SITE, &self.c_proj)]
    }

    fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 2] {
        [(C_FC_SITE, &mut self.c_fc), (C_PROJ_SITE, &mut self.c_proj)]
    }
}

/// Residual transformer block: LN → MHSA → residual → LN → MLP → residual.
pub(crate) struct ResidualAttentionBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl ResidualAttentionBlock {
    /// `intermediate_size` is the owning tower's own MLP width (module doc,
    /// difference 1) — this block never re-derives it from a ratio.
    fn load_with(
        vb: VarBuilder,
        width: usize,
        heads: usize,
        intermediate_size: usize,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        // `with_bias=true`: OpenCLIP's `ln_1`/`ln_2` are affine (weight AND
        // bias), matching `candle_nn::layer_norm`'s `remove_mean=true,
        // affine=true` default this call replaced — same "weight"/"bias"
        // safetensors key names (`crate::layer_norm::LayerNorm::new`'s doc).
        // No `remove_mean=false` (RMSNorm-style) variant exists anywhere in
        // either tower's config, so the house class's fixed mean-removal is
        // exactly the behavior being replaced, not a silent narrowing.
        let ln_1 = LayerNorm::new(width, 1e-5, true, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::load_with(vb.pp("attn"), width, heads, site)?;
        let ln_2 = LayerNorm::new(width, 1e-5, true, vb.pp("ln_2"))?;
        let mlp = Mlp::load_with(vb.pp("mlp"), width, intermediate_size, site)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    /// `attn_mask`: `Some(causal)` for the causally-masked text tower,
    /// `None` for the bidirectional vision tower (module doc, difference 2).
    /// `None` skips the mask `broadcast_add` entirely rather than adding a
    /// zero mask — see [`MultiHeadAttention::forward`]'s own doc.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x, attn_mask)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }

    /// Propagates to the attention module (softmax arm AND its two LoRA
    /// sites' dropout), both residual-stream LayerNorms, and the MLP's two
    /// LoRA sites — every training-gated component this block owns.
    fn set_training(&mut self, training: bool) {
        self.attn.set_training(training);
        self.ln_1.set_training(training);
        self.ln_2.set_training(training);
        for (_, site) in self.mlp.lora_sites_mut() {
            site.set_training(training);
        }
    }

    /// This block's four LoRA sites paired with their names — the single
    /// source of the site→name map every traversal below walks. Order is
    /// fixed (attention before MLP, checkpoint order within each) so every
    /// traversal visits them identically.
    fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 4] {
        let [in_proj, out_proj] = self.attn.lora_sites();
        let [c_fc, c_proj] = self.mlp.lora_sites();
        [in_proj, out_proj, c_fc, c_proj]
    }

    /// The `&mut` twin of [`Self::lora_sites`], same names, same order.
    fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 4] {
        let [in_proj, out_proj] = self.attn.lora_sites_mut();
        let [c_fc, c_proj] = self.mlp.lora_sites_mut();
        [in_proj, out_proj, c_fc, c_proj]
    }

    /// The attention sublayer, for this crate's own tower-level oracles
    /// (which read the fused QKV weight's gradient through
    /// `MultiHeadAttention::in_proj_weight`). `#[cfg(test)]`-only: no
    /// production caller reaches past the block boundary.
    #[cfg(test)]
    pub(crate) fn attn(&self) -> &MultiHeadAttention {
        &self.attn
    }
}

/// Load the whole `transformer.resblocks.{0..layers}` stack from a
/// TOWER-scoped frozen `VarBuilder`. `site_for(n)` yields block `n`'s
/// [`LoraSite`] (already scoped to that block's adapter subtree and carrying
/// `layer_idx = Some(n)`).
pub(crate) fn load_blocks<'a>(
    vb: &VarBuilder,
    layers: usize,
    width: usize,
    heads: usize,
    intermediate_size: usize,
    site_for: &dyn Fn(usize) -> LoraSite<'a>,
) -> Result<Vec<ResidualAttentionBlock>, EncoderError> {
    let mut blocks = Vec::with_capacity(layers);
    for i in 0..layers {
        blocks.push(ResidualAttentionBlock::load_with(
            vb.pp(format!("{BLOCK_STACK_PATH}.{i}")),
            width,
            heads,
            intermediate_size,
            &site_for(i),
        )?);
    }
    Ok(blocks)
}

/// The adapter key prefix for block `n`'s `site`, under `adapter_root`
/// (module doc, difference 3). The ONE place the two towers' key shapes are
/// spelled, so `named_trainable_weights` / `load_weights` /
/// `dropout_positions` / `restore_dropout_positions` and the builder's
/// trainable-`VarBuilder` scoping can never disagree about a key.
fn site_key(adapter_root: &str, n: usize, site: &str) -> String {
    format!("{adapter_root}.{n}.{site}")
}

/// One trainable sub-`VarBuilder` per block, so a site registered at
/// `lora_subpath` lands on exactly the key [`site_key`] predicts. Built up
/// front (rather than per site) so the scoped builders outlive the site
/// closure the tower's loader takes.
pub(crate) fn block_var_builders<'a>(
    trainable_vb: &VarBuilder<'a>,
    adapter_root: &str,
    layers: usize,
) -> Vec<VarBuilder<'a>> {
    (0..layers)
        .map(|n| trainable_vb.pp(format!("{adapter_root}.{n}")))
        .collect()
}

/// Propagate training mode to every block.
pub(crate) fn set_training(blocks: &mut [ResidualAttentionBlock], training: bool) {
    for block in blocks {
        block.set_training(training);
    }
}

/// Trainable tensors across every LoRA-wrapped site, in the fixed traversal
/// order. Empty for a fully frozen stack.
pub(crate) fn trainable_params(blocks: &[ResidualAttentionBlock]) -> Vec<&Tensor> {
    let mut params = Vec::new();
    for block in blocks {
        for (_, lin) in block.lora_sites() {
            params.extend(lin.trainable_params());
        }
    }
    params
}

/// Named LoRA A/B tensors keyed `{adapter_root}.{n}.{site}.lora_{a,b}`.
pub(crate) fn named_trainable_weights(
    blocks: &[ResidualAttentionBlock],
    adapter_root: &str,
) -> Result<HashMap<String, Tensor>, EncoderError> {
    let mut out = HashMap::new();
    for (n, block) in blocks.iter().enumerate() {
        for (site, lin) in block.lora_sites() {
            out.extend(lin.named_weights(&site_key(adapter_root, n, site))?);
        }
    }
    Ok(out)
}

/// Restore LoRA A/B tensors from a [`named_trainable_weights`]-shaped map.
/// Missing keys are no-ops.
pub(crate) fn load_weights(
    blocks: &mut [ResidualAttentionBlock],
    weights: &HashMap<String, Tensor>,
    adapter_root: &str,
) {
    for (n, block) in blocks.iter_mut().enumerate() {
        for (site, lin) in block.lora_sites_mut() {
            lin.load_weights(weights, &site_key(adapter_root, n, site));
        }
    }
}

/// Per-site dropout-stream positions keyed
/// `{adapter_root}.{n}.{site}.dropout` — the `.dropout` leaf is
/// [`jammi_lora::MaybeLoraLinear::collect_dropout_position`]'s own, appended
/// to the same site prefix [`named_trainable_weights`] uses.
pub(crate) fn dropout_positions(
    blocks: &[ResidualAttentionBlock],
    adapter_root: &str,
) -> Result<HashMap<String, u64>, EncoderError> {
    let mut out = HashMap::new();
    for (n, block) in blocks.iter().enumerate() {
        for (site, lin) in block.lora_sites() {
            lin.collect_dropout_position(&site_key(adapter_root, n, site), &mut out)?;
        }
    }
    Ok(out)
}

/// Restore each LoRA site's dropout-stream position from a
/// [`dropout_positions`]-shaped map. Missing keys are no-ops.
pub(crate) fn restore_dropout_positions(
    blocks: &[ResidualAttentionBlock],
    adapter_root: &str,
    positions: &HashMap<String, u64>,
) -> Result<(), EncoderError> {
    for (n, block) in blocks.iter().enumerate() {
        for (site, lin) in block.lora_sites() {
            lin.restore_dropout_position(&site_key(adapter_root, n, site), positions)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two towers' key sets are disjoint at EVERY block index and site,
    /// which is the whole point of the split (module doc). Asserted on the
    /// key builder itself so the property holds for any layer count, not
    /// just the tiny fixture the tower-level oracle uses.
    #[test]
    fn the_two_tower_roots_produce_disjoint_site_keys() {
        let sites = ["in_proj", "out_proj", C_FC_SITE, C_PROJ_SITE];
        let key_set = |root: &str| -> Vec<String> {
            (0..8)
                .flat_map(|n| {
                    sites
                        .iter()
                        .map(move |s| site_key(root, n, s))
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let text = key_set(crate::clip_text::ADAPTER_BLOCK_ROOT);
        let vision = key_set(crate::open_clip_vision::ADAPTER_BLOCK_ROOT);
        assert_eq!(text[0], "resblocks.0.in_proj");
        assert_eq!(vision[0], "visual.resblocks.0.in_proj");
        for k in &text {
            assert!(
                !vision.contains(k),
                "text key {k} also appears in the vision key set"
            );
        }
    }
}
