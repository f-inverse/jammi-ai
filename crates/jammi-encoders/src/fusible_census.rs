//! The structural CENSUS of a built tower's fusible seams — how many times
//! ONE training forward calls each of them.
//!
//! # Why this type exists
//!
//! A fused-kernel profile's positive-proof equation is
//!
//! ```text
//! fused + eager == calls x batches
//! ```
//!
//! per admission key. The left side is measured (the process-wide
//! `jammi_kernels::admission` counters); `batches` is measured (the number of
//! training forwards the run took). `calls` — how many times ONE forward
//! reaches that seam — had NO witness at all: it was a number a reader
//! derived by hand from a config and a mental model of the architecture, and
//! a wrong derivation there turns the whole equation from a proof into a
//! restatement of whatever the counters happened to say.
//!
//! [`FusibleSiteCensus`] is that witness. It is produced by WALKING THE BUILT
//! STRUCTURE — the `Vec<Layer>` the loader actually built, the
//! `MaybeLoraLinear` arm each site actually landed on, the `Option<LayerNorm>`
//! a layer actually holds — never by arithmetic over a config struct. The
//! distinction is the whole point: config arithmetic reproduces the loader's
//! own assumptions, so it agrees with the loader exactly when the loader is
//! right and exactly when it is wrong. A structural walk disagrees the moment
//! the built tower stops matching the config (a site the selector declined, a
//! pre-norm a family omits on layer 0, a stage with no downsample), which is
//! the only case where a `calls` witness earns anything.
//!
//! # Per-forward semantics: `training == true`
//!
//! Every count here is per ONE forward at `training == true`. That is not a
//! convention, it is where the seams live: each of the three admits inside a
//! training-only arm and short-circuits before any admission decision in
//! eval (`crate::layer_norm::LayerNorm::forward`'s eval arms,
//! `crate::activations::gelu_erf`'s `if !training` early return,
//! `jammi_lora::LoraLinear::forward`'s eval early return). So an EVAL forward
//! contributes exactly `0` to BOTH the `fused` and the `eager` side of every
//! pair — an eval pass is not "all eager", it is absent from the counters
//! entirely, and a profile that mixed eval passes into `batches` would be
//! comparing a count of training forwards against a count of all forwards.
//!
//! The census is a property of the built tower, not of a batch: it does not
//! depend on batch size, sequence length, device or dtype. Those decide
//! whether a given call lands on `fused` or on `eager`; they never change how
//! many calls there are.

use serde::{Deserialize, Serialize};

/// Per-forward call counts for the three fusible seams a tower can hold.
///
/// Read through [`crate::AnyEncoder::fusible_site_census`], which is total
/// over the encoder enum: every variant answers, and a variant that
/// architecturally has no such seam answers `0` rather than declining (see
/// that method's own doc for the per-family table).
///
/// Each field pairs with exactly one `jammi_kernels::admission` key:
///
/// | field | admission key |
/// |---|---|
/// | [`Self::lora_sites_wrapped`] | `lora_linear_fused` |
/// | [`Self::layer_norms`] | `layer_norm_fused` |
/// | [`Self::gelu_seam_calls_per_forward`] | `gelu_erf_fused` |
///
/// so the profile's per-key equation reads its `calls` term straight off
/// this struct: `fused + eager == <field> * batches`, with `batches` the
/// number of TRAINING forwards (see the module doc for why eval forwards
/// count for nothing on either side).
///
/// `Serialize`/`Deserialize` because a durable job record carries the census
/// alongside the counters it explains: a counter total with no `calls`
/// witness recorded next to it cannot be re-checked after the fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusibleSiteCensus {
    /// How many linear sites the built tower's ONE training forward actually
    /// takes a `lora_linear_fused` admission decision for —
    /// `jammi_lora::MaybeLoraLinear::takes_lora_linear_admission`, counted
    /// over the tower's own site traversal — the SAME traversal its adapter
    /// export walks, so the census cannot see a site set the checkpoint
    /// writer does not.
    ///
    /// This is a NARROWER question than "is this site adapted"
    /// (`jammi_lora::MaybeLoraLinear::is_lora`, that method's own doc records
    /// the caveat this field's doc used to omit): `LoraLinear::forward`
    /// branches on `FrozenBase::Dense` vs `FrozenBase::Quantized` BEFORE it
    /// ever reaches `admit()`, so a `Lora` site over a `FrozenBase::Quantized`
    /// base — the shape a QLoRA backbone builds for every adapted site — is
    /// adapted (`is_lora() == true`) but structurally composes and takes NO
    /// `lora_linear_fused` admission decision at all. This field counts
    /// exactly the sites that DO take that decision: `Lora` sites whose base
    /// is `FrozenBase::Dense`.
    ///
    /// `0` for a fully frozen backbone, and that zero is load-bearing: a
    /// frozen tower takes NO `lora_linear_fused` admission decision at all,
    /// so `fused + eager` must be `0` for it too. `0` is ALSO the correct
    /// answer for a fully-adapted QLoRA backbone whose bases are all
    /// quantized — same admission-side zero, a different structural reason.
    /// Unwrapped (`Frozen`) sites are deliberately NOT counted either way —
    /// they forward through `candle_nn::Linear`/`QuantizedLinear` directly
    /// and never reach the seam.
    ///
    /// This is a count of SITES taking the admission decision, not of
    /// trainable tensors and not of adapted sites: an adapted site owns two
    /// trainable tensors (`lora_a`, `lora_b`) regardless of its base storage,
    /// so on a fully-dense-base tower this field equals half of
    /// `trainable_params().len()`, but on a mixed or fully-quantized-base
    /// QLoRA tower it is strictly LESS than half — the quantized-base
    /// adapted sites still contribute their own `lora_a`/`lora_b` to
    /// `trainable_params()` while contributing `0` here.
    pub lora_sites_wrapped: usize,
    /// How many house `crate::layer_norm::LayerNorm` instances the built
    /// tower holds on its forward path — counted by walking the struct, so a
    /// family that omits one (ModernBERT's layer-0 pre-norm is `None`, an
    /// HTSAT stage with no `downsample` has no merging norm) contributes what
    /// it actually holds rather than what a `2 * layers` formula would
    /// predict.
    ///
    /// One instance is one admission decision per training forward: the
    /// training arm admits ONCE per `forward` call under the single key
    /// `layer_norm_fused`, whether or not the norm carries a bias (bias is
    /// tensor state inside that one key, not a second key — see
    /// `crate::layer_norm::LayerNorm::forward`'s own doc).
    ///
    /// Norms that are NOT the house type are outside this count by
    /// construction, because they cannot reach the seam: HTSAT's
    /// `candle_nn::BatchNorm` fusion-block norms are the live example.
    pub layer_norms: usize,
    /// How many times ONE training forward calls the house GELU seam
    /// (`crate::activations::gelu_erf`), and therefore how many
    /// `gelu_erf_fused` admission decisions it takes.
    ///
    /// `0` is a real answer, not a missing one, and it means two different
    /// architectural things worth keeping apart: ModernBERT's FFN is a GeGLU
    /// whose fused path is a DIFFERENT key (`geglu_fused`), and the OpenCLIP
    /// towers' MLP activation is `quick_gelu`, which has no fused seam at
    /// all. Neither will ever move the `gelu_erf_fused` counters, so for both
    /// the equation `fused + eager == 0 * batches` is the correct — and
    /// falsifiable — claim.
    pub gelu_seam_calls_per_forward: usize,
}
