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

fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0))
}
