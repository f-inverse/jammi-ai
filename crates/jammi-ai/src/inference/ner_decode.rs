//! BIO span decoder for NER: converts token-level predictions to entity spans.

use std::collections::HashMap;

use serde::Serialize;

/// A decoded named entity span.
#[derive(Debug, Clone, Serialize)]
pub struct EntitySpan {
    /// The entity text extracted from the original input.
    pub text: String,
    /// Entity type (e.g., "PER", "ORG") — without B-/I- prefix.
    pub label: String,
    /// Character start position in the original text.
    pub start: usize,
    /// Character end position (exclusive) in the original text.
    pub end: usize,
    /// Average softmax confidence across entity tokens.
    pub confidence: f32,
}

/// Decode BIO-tagged token predictions into entity spans.
pub fn decode_bio_spans(
    token_logits: &[Vec<f32>],
    offsets: &[(usize, usize)],
    attention_mask: &[u32],
    id2label: &HashMap<u32, String>,
    original_text: &str,
) -> Vec<EntitySpan> {
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
    fn finalize(self, original_text: &str) -> EntitySpan {
        let text = if self.end <= original_text.len() {
            original_text[self.start..self.end].to_string()
        } else {
            String::new()
        };
        EntitySpan {
            text,
            label: self.label,
            start: self.start,
            end: self.end,
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

fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id2label() -> HashMap<u32, String> {
        [
            (0, "O"),
            (1, "B-PER"),
            (2, "I-PER"),
            (3, "B-ORG"),
            (4, "I-ORG"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect()
    }

    #[test]
    fn decode_simple_bio_sequence() {
        let id2label = make_id2label();
        let offsets = vec![
            (0, 0),   // [CLS]
            (0, 4),   // John
            (5, 10),  // works
            (11, 13), // at
            (14, 20), // Google
            (0, 0),   // [SEP]
        ];
        let attention_mask = vec![1, 1, 1, 1, 1, 1];
        let token_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0, 0.0], // [CLS]
            vec![0.0, 10.0, 0.0, 0.0, 0.0], // John → B-PER
            vec![10.0, 0.0, 0.0, 0.0, 0.0], // works → O
            vec![10.0, 0.0, 0.0, 0.0, 0.0], // at → O
            vec![0.0, 0.0, 0.0, 10.0, 0.0], // Google → B-ORG
            vec![10.0, 0.0, 0.0, 0.0, 0.0], // [SEP]
        ];

        let entities = decode_bio_spans(
            &token_logits,
            &offsets,
            &attention_mask,
            &id2label,
            "John works at Google",
        );

        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "John");
        assert_eq!(entities[0].label, "PER");
        assert_eq!(entities[1].text, "Google");
        assert_eq!(entities[1].label, "ORG");
    }

    #[test]
    fn decode_multi_token_entity() {
        let id2label = make_id2label();
        let offsets = vec![(0, 0), (0, 3), (4, 8), (9, 13), (0, 0)];
        let attention_mask = vec![1, 1, 1, 1, 1];
        let token_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 10.0, 0.0], // B-ORG
            vec![0.0, 0.0, 0.0, 0.0, 10.0], // I-ORG
            vec![0.0, 0.0, 0.0, 0.0, 10.0], // I-ORG
            vec![10.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let entities = decode_bio_spans(
            &token_logits,
            &offsets,
            &attention_mask,
            &id2label,
            "New York City",
        );

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "New York City");
        assert_eq!(entities[0].label, "ORG");
    }

    #[test]
    fn decode_all_o_returns_empty() {
        let id2label = make_id2label();
        let offsets = vec![(0, 0), (0, 5), (0, 0)];
        let attention_mask = vec![1, 1, 1];
        let token_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let entities =
            decode_bio_spans(&token_logits, &offsets, &attention_mask, &id2label, "hello");
        assert!(entities.is_empty());
    }
}
