//! BIO span decoder for NER: converts token-level predictions to entity
//! spans.

use std::collections::HashMap;

use crate::ner::types::Entity;

/// Decode BIO-tagged token predictions into entity spans.
pub fn decode_bio_spans(
    token_logits: &[Vec<f32>],
    offsets: &[(usize, usize)],
    attention_mask: &[u32],
    id2label: &HashMap<u32, String>,
    original_text: &str,
) -> Vec<Entity> {
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

    entities
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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Argmax over per-token class probabilities.
///
/// A `NaN` in `values` (e.g. a diverged model producing corrupted logits) is
/// not provably unreachable here — unlike [`crate::retrieval`]'s
/// integer-sourced ideal-gain sort — so the comparator cannot rely on
/// finiteness. `f32::total_cmp` is used instead of `partial_cmp`, which
/// gives `NaN` the maximal position in its IEEE-754 total order; this
/// matches `torch.argmax`'s own documented "NaN is propagated" convention
/// (a `NaN` element compares as the maximum, so `argmax` returns its index)
/// rather than silently picking an arbitrary non-NaN winner. The returned
/// confidence is then itself `NaN` — a visibly non-finite value a caller can
/// detect with `is_finite()`, not a plausible-looking wrong number.
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
        // torch.argmax's documented convention: a NaN element is treated as
        // the maximal value, so its index wins. total_cmp gives the same
        // outcome here — NaN sorts as the IEEE-754 maximum.
        let (idx, val) = argmax(&[0.1, f32::NAN, 0.2]);
        assert_eq!(idx, 1);
        assert!(val.is_nan());
    }

    #[test]
    fn argmax_of_all_nan_is_deterministic() {
        // `Iterator::max_by` returns the last of several equally-maximal
        // elements; total_cmp treats every (default, positive) NaN as
        // bit-identical, so an all-NaN row deterministically resolves to the
        // last index rather than an arbitrary one.
        let values = [f32::NAN, f32::NAN, f32::NAN];
        let (idx, val) = argmax(&values);
        assert_eq!(idx, values.len() - 1);
        assert!(val.is_nan());
    }
}
