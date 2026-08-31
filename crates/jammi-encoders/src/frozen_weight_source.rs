//! The candle-boundary seam a caller (jammi-ai's wave-3 GGUF loader) plugs
//! into `Bert`/`DistilBert`/`ModernBert`'s construction to supply a
//! GGUF-quantized weight in place of the default dense safetensors load —
//! WITHOUT this crate doing any GGUF file I/O itself. `pub mod` (not
//! `pub(crate)`) at the crate boundary: `FrozenWeightLookup` is a PARAMETER
//! TYPE every construction-site builder method accepts, so it must be
//! nameable from a downstream crate.
//!
//! # Why a seam here, and why it stops at this one type
//!
//! Issue #351's wave 3 (GGUF/k-quant inference in `jammi-ai`) needs a way
//! to hand `Bert`/`DistilBert`/`ModernBert`'s construction a
//! `jammi_lora::FrozenBase::Quantized(..)` for a given named tensor instead
//! of the frozen safetensors `VarBuilder` load every EXISTING path uses.
//! This crate has no GGUF reader, no k-quant block format knowledge, and no
//! opinion on WHERE a caller's weight map comes from — [`FrozenWeightLookup`]
//! is the entire seam: one `Fn(&str) -> Result<Option<FrozenBase>,
//! EncoderError>`, threaded as an OPTIONAL builder parameter through every
//! construction site that currently loads a base `Linear`. Wave 3
//! implements the closure over its own GGUF-loaded weight map and never
//! needs to touch `jammi-lora` or this crate again — the entire "does this
//! module have a quantized override" decision lives on the wave-3 side of
//! the closure boundary.
//!
//! # Byte-identical when unset (K2 / additive-only)
//!
//! Every EXISTING construction path (every `*Builder::build` call in this
//! workspace today) never calls the new `.weight_source(..)` builder
//! method, so `weight_source` stays `None` at every site — the per-name
//! lookup step becomes a single `match None { .. }` no-op, and the ORIGINAL
//! `linear(in_features, out_features, module_vb)` call (or
//! `linear_no_bias` for ModernBERT) runs exactly as it always has. Passing
//! `Some(lookup)` where `lookup` always returns `Ok(None)` is likewise a
//! byte-identical no-op — the seam's presence changes nothing about a
//! caller that does not use it.

use jammi_lora::FrozenBase;

use crate::error::EncoderError;

/// A per-tensor-name lookup a caller supplies at build time to override a
/// construction site's default (load a Dense `Linear` from the frozen
/// safetensors `VarBuilder`) with a pre-built [`FrozenBase`] — Dense OR
/// GGUF-quantized. `name` is the tensor's FULLY QUALIFIED dotted path (the
/// same string `candle_nn::VarBuilder::prefix()` returns for the
/// module-scoped `VarBuilder` that would otherwise load it — e.g.
/// `"bert.encoder.layer.3.attention.self.query"`), so a caller keys its
/// lookup the SAME way a safetensors reader addresses that tensor — never a
/// short/relative name a caller would have to re-derive the layer/prefix
/// context for.
///
/// Returns:
/// - `Ok(None)` — no override for this name; the caller falls back to the
///   existing Dense-from-`VarBuilder` load, BYTE-IDENTICAL to every prior
///   release (the ONLY behavior when no lookup is supplied at all, module
///   doc, and the correct fallback when a lookup IS supplied but simply
///   does not cover this particular tensor — e.g. a partially-quantized
///   checkpoint).
/// - `Ok(Some(base))` — use `base` (Dense or Quantized) directly, skipping
///   the `VarBuilder` load entirely.
/// - `Err(e)` — the lookup itself failed (e.g. a malformed GGUF entry for a
///   name the caller DOES recognize) — a typed, loud refusal (K2) rather
///   than a silent fallback to Dense, which would hide a real load failure
///   behind a plausible-looking successful build.
///
/// A plain `Fn` trait object, not a new named trait: every implementor is a
/// closure over a wave-3-owned weight map, and a trait with exactly one
/// method would add a vtable indirection with no benefit a `dyn Fn` does
/// not already provide.
pub type FrozenWeightLookup<'a> = dyn Fn(&str) -> Result<Option<FrozenBase>, EncoderError> + 'a;

#[cfg(test)]
mod tests {
    use super::*;

    /// A lookup that always misses (`Ok(None)`) is a legal, well-typed
    /// implementor — the shape every EXISTING call site's implicit `None`
    /// seam is equivalent to (module doc's "byte-identical when unset").
    #[test]
    fn an_always_miss_lookup_type_checks_and_returns_none() {
        let lookup: &FrozenWeightLookup = &|_name: &str| Ok(None);
        assert!(lookup("anything").unwrap().is_none());
    }

    /// A lookup that reports a typed failure for an unrecognized-but-
    /// present name — the `Err` arm's contract (module doc).
    #[test]
    fn a_failing_lookup_type_checks_and_returns_a_typed_error() {
        let lookup: &FrozenWeightLookup = &|name: &str| {
            Err(EncoderError::Config(format!(
                "no quantized entry for {name}"
            )))
        };
        assert!(lookup("some.tensor").is_err());
    }
}
