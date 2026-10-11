//! Word-level BIO decoder for NER: token-level predictions to entity spans.
//!
//! A token classifier labels every subword token, but a subword is not a unit
//! a reader names: "Hooli" tokenized as `Ho`, `##oli` is one word, and its two
//! tokens can disagree (`B-PER`, `B-LOC`). The decoder therefore labels each
//! WORD by its first subtoken's prediction — the convention a token classifier
//! is trained under, its labels aligned to each word's first subtoken — and
//! groups words, never tokens, into entities.

use std::collections::HashMap;

use crate::error::{NumericsError, Result};
use crate::ner::types::Entity;

/// One row's tokenization, as the decoder reads it: each token's byte offsets
/// in the text, the word the tokenizer's pre-tokenizer assigned it (`None` for
/// a special token), and whether it is a real token (`1`) or padding (`0`).
#[derive(Debug, Clone, Copy)]
pub struct TokenizedRow<'a> {
    /// Each token's `(start, end)` byte offsets in the text.
    pub offsets: &'a [(usize, usize)],
    /// Each token's word index; `None` for a special token.
    pub word_ids: &'a [Option<u32>],
    /// `1` for a real token, `0` for padding.
    pub attention_mask: &'a [u32],
}

/// Decode a row's token-level predictions into entity spans.
///
/// Each word takes the label and the confidence of its first subtoken and
/// spans from that subtoken's start to its last subtoken's end. Words then
/// group into entities by their BIO tags: `B-X` begins an `X`, `I-X` continues
/// an open `X` and begins one when none is open, and `O` ends the open entity.
/// A label with no `B-`/`I-` prefix continues an entity of its own type, as an
/// IO-tagged checkpoint names them.
///
/// # Errors
///
/// [`NumericsError::InvalidInput`] naming the offending token index if any real
/// token's logit row contains a non-finite (`NaN` or `±inf`) value — checked
/// before the softmax, where a `NaN` or `+inf` element would corrupt the whole
/// row into a plausible-looking but meaningless label — or if a span does not
/// fall on the text's character boundaries.
pub fn decode_bio_spans(
    token_logits: &[Vec<f32>],
    row: &TokenizedRow<'_>,
    id2label: &HashMap<u32, String>,
    original_text: &str,
) -> Result<Vec<Entity>> {
    let words = words(token_logits, row, id2label)?;
    let mut entities = Vec::new();
    let mut open: Option<Partial> = None;
    for (_, word) in words {
        open = match (Tag::of(&word.label), open.take()) {
            (Tag::Outside, current) => {
                entities.extend(current.map(|p| p.finish(original_text)).transpose()?);
                None
            }
            (Tag::Inside(kind), Some(mut current)) if current.label == kind => {
                current.extend(&word);
                Some(current)
            }
            (Tag::Begin(kind) | Tag::Inside(kind), current) => {
                entities.extend(current.map(|p| p.finish(original_text)).transpose()?);
                Some(Partial::begin(kind, &word))
            }
        };
    }
    entities.extend(open.map(|p| p.finish(original_text)).transpose()?);
    Ok(entities)
}

/// A word's prediction: its first subtoken's label and confidence, over the
/// byte span of all its subtokens.
struct Word {
    label: String,
    start: usize,
    end: usize,
    confidence: f32,
}

/// The row's words, in order: consecutive real tokens sharing a word id are
/// one word.
fn words(
    token_logits: &[Vec<f32>],
    row: &TokenizedRow<'_>,
    id2label: &HashMap<u32, String>,
) -> Result<Vec<(u32, Word)>> {
    let mut words: Vec<(u32, Word)> = Vec::new();
    for (idx, logits) in token_logits.iter().enumerate() {
        let real = row.attention_mask.get(idx).is_some_and(|&m| m != 0);
        let Some(word_id) = row.word_ids.get(idx).copied().flatten().filter(|_| real) else {
            continue;
        };
        if logits.iter().any(|x| !x.is_finite()) {
            return Err(NumericsError::InvalidInput(format!(
                "decode_bio_spans: token {idx} has a non-finite logit"
            )));
        }
        let (start, end) = row.offsets.get(idx).copied().ok_or_else(|| {
            NumericsError::InvalidInput(format!("decode_bio_spans: token {idx} has no offsets"))
        })?;
        match words.last_mut() {
            Some((id, word)) if *id == word_id => word.end = end,
            _ => {
                let (class, confidence) = argmax(&softmax(logits));
                let label = id2label
                    .get(&(class as u32))
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{class}"));
                words.push((
                    word_id,
                    Word {
                        label,
                        start,
                        end,
                        confidence,
                    },
                ));
            }
        }
    }
    Ok(words)
}

/// A word's BIO tag and the entity type it names.
enum Tag<'a> {
    Begin(&'a str),
    Inside(&'a str),
    Outside,
}

impl<'a> Tag<'a> {
    fn of(label: &'a str) -> Self {
        if label == "O" {
            Self::Outside
        } else if let Some(kind) = label.strip_prefix("B-") {
            Self::Begin(kind)
        } else {
            Self::Inside(label.strip_prefix("I-").unwrap_or(label))
        }
    }
}

/// An entity being assembled from consecutive words.
struct Partial {
    label: String,
    start: usize,
    end: usize,
    confidence_sum: f32,
    words: usize,
}

impl Partial {
    fn begin(label: &str, word: &Word) -> Self {
        Self {
            label: label.to_string(),
            start: word.start,
            end: word.end,
            confidence_sum: word.confidence,
            words: 1,
        }
    }

    fn extend(&mut self, word: &Word) {
        self.end = word.end;
        self.confidence_sum += word.confidence;
        self.words += 1;
    }

    fn finish(self, original_text: &str) -> Result<Entity> {
        let text = original_text.get(self.start..self.end).ok_or_else(|| {
            NumericsError::InvalidInput(format!(
                "decode_bio_spans: span {}..{} is not on the text's character boundaries",
                self.start, self.end
            ))
        })?;
        Ok(Entity {
            label: self.label,
            start: self.start,
            end: self.end,
            text: text.to_string(),
            confidence: self.confidence_sum / self.words as f32,
        })
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
