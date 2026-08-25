//! BIO span decoder for NER: converts token-level predictions to entity
//! spans.

use std::collections::HashMap;

use crate::error::{NumericsError, Result};
use crate::ner::types::Entity;

/// Decode BIO-tagged token predictions into entity spans.
///
/// # Errors
///
/// Returns [`NumericsError::InvalidInput`] naming the offending token index
/// if any non-skipped token's logit row contains a non-finite (`NaN` or
/// `±inf`) value. This is checked BEFORE the internal softmax step: a raw
/// logit is a real-number domain value, and a `NaN` or `+inf` element
/// corrupts the entire softmax row to `NaN` (a `-inf` element happens not
/// to, but is refused anyway as an equally invalid logit). Refusing at the
/// logit edge, before a corrupted row could produce a plausible-looking but
/// meaningless decoded label, surfaces a diverged model's corrupted logits
/// as an error instead.
pub fn decode_bio_spans(
    token_logits: &[Vec<f32>],
    offsets: &[(usize, usize)],
    attention_mask: &[u32],
    id2label: &HashMap<u32, String>,
    original_text: &str,
) -> Result<Vec<Entity>> {
    let mut entities = Vec::new();
    let mut current: Option<PartialEntity> = None;

    for (idx, logits) in token_logits.iter().enumerate() {
        if idx >= attention_mask.len() || attention_mask[idx] == 0 {
            continue;
        }

        let (start_byte, end_byte) = offsets.get(idx).copied().unwrap_or((0, 0));
        // Skip special tokens (offset (0, 0))
        if start_byte == 0 && end_byte == 0 {
            continue;
        }

        if logits.iter().any(|x| !x.is_finite()) {
            return Err(NumericsError::InvalidInput(format!(
                "decode_bio_spans: token {idx} has a non-finite logit"
            )));
        }

        let probs = softmax(logits);
        let (pred_idx, confidence) = argmax(&probs);
        let label = id2label
            .get(&(pred_idx as u32))
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{pred_idx}"));

        if let Some(entity_type) = label.strip_prefix("B-") {
            if let Some(partial) = current.take() {
                entities.push(partial.finalize(original_text));
            }
            current = Some(PartialEntity {
                label: entity_type.to_string(),
                start: start_byte,
                end: end_byte,
                total_confidence: confidence,
                token_count: 1,
            });
        } else if let Some(entity_type) = label.strip_prefix("I-") {
            let type_matches = current.as_ref().is_some_and(|p| p.label == entity_type);
            if type_matches {
                let partial = current.as_mut().unwrap();
                partial.end = end_byte;
                partial.total_confidence += confidence;
                partial.token_count += 1;
            } else if let Some(partial) = current.take() {
                entities.push(partial.finalize(original_text));
            }
        } else if let Some(partial) = current.take() {
            entities.push(partial.finalize(original_text));
        }
    }

    if let Some(partial) = current.take() {
        entities.push(partial.finalize(original_text));
    }

    Ok(entities)
}

struct PartialEntity {
    label: String,
    start: usize,
    end: usize,
    total_confidence: f32,
    token_count: usize,
}

impl PartialEntity {
    fn finalize(self, original_text: &str) -> Entity {
        let text = if self.end <= original_text.len() {
            original_text[self.start..self.end].to_string()
        } else {
            String::new()
        };
        Entity {
            label: self.label,
            start: self.start,
            end: self.end,
            text,
            confidence: self.total_confidence / self.token_count as f32,
        }
    }
}

/// A `NaN` or `+inf` element in `logits` corrupts every output element to
/// `NaN` (verified: at that element's position, `x - max` is `NaN` —
/// `f32::max`'s NaN-ignoring fold leaves `max` finite for a `NaN` logit, but
/// `NaN - finite = NaN`; for a `+inf` logit, `max` becomes `+inf` and
/// `+inf - +inf = NaN` — either way `exps` gets one `NaN` entry, and
/// `sum: f32 = exps.iter().sum()` is NaN-propagating, so every `e / sum`
/// below is `NaN`). A lone `-inf` element does NOT corrupt the row (its
/// shifted value is `-inf`, `exp(-inf) = 0.0`, a well-defined zero
/// probability for that class) — this function still requires the caller
/// to validate `logits` finite (`decode_bio_spans` does, before its
/// `softmax` call), because a raw logit is a real-number domain value and
/// `±inf` is not a value normal (non-corrupted) model output produces, not
/// because every non-finite input happens to corrupt this particular
/// computation.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Argmax over per-token class probabilities.
///
/// `values` is provably `NaN`-free on the only reachable call site: the
/// caller, [`decode_bio_spans`], refuses a non-finite logit row before
/// calling [`softmax`], and softmax of an all-finite row cannot itself
/// produce a `NaN` (its `sum` is at least `1.0`, from the `max`-shifted
/// element whose `exp(0) = 1`, so no `0.0 / 0.0` is possible). `f32::total_cmp`
/// is used here anyway to pin the fold order explicitly rather than lean on
/// that precondition holding forever: `f32::partial_cmp` returns `None` for
/// a `NaN` comparison, which `Iterator::max_by` would either need to
/// `.unwrap()` (panicking) or collapse to an arbitrary `Ordering` (silently
/// picking an unspecified winner) if a `NaN` ever did reach this function.
///
/// `total_cmp`'s NaN handling is NOT "NaN is always maximal": it defines a
/// total order over every IEEE-754 bit pattern
/// (`-NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN`), so a *positive*
/// `NaN` (e.g. Rust's `f32::NAN`, bit pattern `0x7fc00000`) sorts as the
/// maximum, but a *negative* `NaN` (bit pattern `0xffc00000`) sorts as the
/// MINIMUM — strictly below `f32::NEG_INFINITY` (verified: `f32::from_bits
/// (0xffc00000).total_cmp(&f32::NEG_INFINITY)` is `Less`). The tests below
/// exercise only the reachable, positive-`NaN` case as this function's own
/// contract, not a state [`decode_bio_spans`] can produce.
fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0))
}

#[cfg(test)]
mod argmax_tests {
    use super::argmax;

    #[test]
    fn argmax_picks_finite_max_when_no_nan() {
        let (idx, val) = argmax(&[0.1, 0.7, 0.2]);
        assert_eq!(idx, 1);
        assert_eq!(val, 0.7);
    }

    #[test]
    fn argmax_returns_nan_index_when_a_nan_is_present() {
        // This is `argmax`'s own contract, not a state its only caller
        // (`decode_bio_spans`) can produce (that caller refuses a
        // non-finite logit row before this function is ever called — see
        // `decode_bio_spans`'s doc). Rust's `f32::NAN` literal is a
        // *positive* NaN (`0x7fc00000`), which `total_cmp` places at the
        // IEEE-754-total-order maximum, so it wins `max_by` here.
        let (idx, val) = argmax(&[0.1, f32::NAN, 0.2]);
        assert_eq!(idx, 1);
        assert!(val.is_nan());
    }

    #[test]
    fn argmax_of_all_nan_is_deterministic() {
        // This is `argmax`'s own contract, not a state its only caller can
        // produce (see the note above). `Iterator::max_by` returns the last
        // of several equally-maximal elements; total_cmp treats every
        // (default, positive) NaN as bit-identical, so an all-NaN row
        // deterministically resolves to the last index rather than an
        // arbitrary one.
        let values = [f32::NAN, f32::NAN, f32::NAN];
        let (idx, val) = argmax(&values);
        assert_eq!(idx, values.len() - 1);
        assert!(val.is_nan());
    }
}
