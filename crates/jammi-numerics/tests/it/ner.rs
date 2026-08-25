use std::collections::HashMap;

use approx::assert_abs_diff_eq;
use jammi_numerics::ner::{decode_bio_spans, Entity, NerMetrics};

fn id2label() -> HashMap<u32, String> {
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
    let id2label = id2label();
    let offsets = vec![(0, 0), (0, 4), (5, 10), (11, 13), (14, 20), (0, 0)];
    let attention_mask = vec![1, 1, 1, 1, 1, 1];
    let token_logits = vec![
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 10.0, 0.0, 0.0, 0.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 10.0, 0.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let entities = decode_bio_spans(
        &token_logits,
        &offsets,
        &attention_mask,
        &id2label,
        "John works at Google",
    )
    .unwrap();
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].text, "John");
    assert_eq!(entities[0].label, "PER");
    assert_eq!(entities[1].text, "Google");
    assert_eq!(entities[1].label, "ORG");
}

#[test]
fn decode_multi_token_entity() {
    let id2label = id2label();
    let offsets = vec![(0, 0), (0, 3), (4, 8), (9, 13), (0, 0)];
    let attention_mask = vec![1, 1, 1, 1, 1];
    let token_logits = vec![
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 10.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 10.0],
        vec![0.0, 0.0, 0.0, 0.0, 10.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let entities = decode_bio_spans(
        &token_logits,
        &offsets,
        &attention_mask,
        &id2label,
        "New York City",
    )
    .unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "New York City");
    assert_eq!(entities[0].label, "ORG");
}

#[test]
fn decode_all_o_returns_empty() {
    let id2label = id2label();
    let offsets = vec![(0, 0), (0, 5), (0, 0)];
    let attention_mask = vec![1, 1, 1];
    let token_logits = vec![
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let entities =
        decode_bio_spans(&token_logits, &offsets, &attention_mask, &id2label, "hello").unwrap();
    assert!(entities.is_empty());
}

#[test]
fn decode_refuses_non_finite_logit() {
    // Through the public entry point: a NaN logit at a non-skipped token
    // must be refused before it reaches softmax/argmax, not silently
    // decoded into a plausible-looking label.
    let id2label = id2label();
    let offsets = vec![(0, 0), (0, 4), (0, 0)];
    let attention_mask = vec![1, 1, 1];
    let token_logits = vec![
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, f32::NAN, 0.0, 0.0, 0.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let result = decode_bio_spans(&token_logits, &offsets, &attention_mask, &id2label, "John");
    assert!(
        result.is_err(),
        "a non-finite logit must be refused, not decoded into a plausible label"
    );
}

#[test]
fn decode_refuses_infinite_logit() {
    let id2label = id2label();
    let offsets = vec![(0, 0), (0, 4), (0, 0)];
    let attention_mask = vec![1, 1, 1];
    let token_logits = vec![
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, f32::INFINITY, 0.0, 0.0, 0.0],
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let result = decode_bio_spans(&token_logits, &offsets, &attention_mask, &id2label, "John");
    assert!(result.is_err());
}

#[test]
fn decode_ignores_non_finite_logit_on_a_skipped_special_token() {
    // A logit row at a special-token position (offset (0, 0), or masked by
    // attention_mask) is never fed to softmax, so it must not trip the
    // refusal — only non-skipped rows are validated.
    let id2label = id2label();
    let offsets = vec![(0, 0), (0, 4), (0, 0)];
    let attention_mask = vec![1, 1, 1];
    let token_logits = vec![
        vec![f32::NAN, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 10.0, 0.0, 0.0, 0.0],
        vec![f32::NAN, 0.0, 0.0, 0.0, 0.0],
    ];
    let entities =
        decode_bio_spans(&token_logits, &offsets, &attention_mask, &id2label, "John").unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "John");
}

fn gold_entity(label: &str, start: usize, end: usize) -> Entity {
    Entity {
        label: label.into(),
        start,
        end,
        text: String::new(),
        confidence: 0.0,
    }
}

#[test]
fn ner_metrics_perfect_match() {
    let gold = vec![vec![gold_entity("PER", 0, 5), gold_entity("ORG", 10, 15)]];
    let pred = gold.clone();
    let m = NerMetrics::compute(&pred, &gold);
    assert_abs_diff_eq!(m.precision, 1.0, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.recall, 1.0, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.f1, 1.0, epsilon = f64::EPSILON);
}

#[test]
fn ner_metrics_no_predictions() {
    let gold = vec![vec![gold_entity("PER", 0, 5)]];
    let pred = vec![vec![]];
    let m = NerMetrics::compute(&pred, &gold);
    assert_eq!(m.precision, 0.0);
    assert_eq!(m.recall, 0.0);
    assert_eq!(m.f1, 0.0);
}

#[test]
fn ner_metrics_partial_overlap() {
    let gold = vec![vec![gold_entity("PER", 0, 5), gold_entity("LOC", 10, 15)]];
    let pred = vec![vec![gold_entity("PER", 0, 5), gold_entity("ORG", 20, 25)]];
    let m = NerMetrics::compute(&pred, &gold);
    assert_abs_diff_eq!(m.precision, 0.5, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.recall, 0.5, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.f1, 0.5, epsilon = f64::EPSILON);
}

#[test]
fn ner_metrics_ignore_text_and_confidence_in_equality() {
    // Two entities with same label/boundaries but different text/confidence
    // must be treated as equal.
    let gold = vec![vec![Entity {
        label: "PER".into(),
        start: 0,
        end: 5,
        text: String::new(),
        confidence: 0.0,
    }]];
    let pred = vec![vec![Entity {
        label: "PER".into(),
        start: 0,
        end: 5,
        text: "John".into(),
        confidence: 0.95,
    }]];
    let m = NerMetrics::compute(&pred, &gold);
    assert_abs_diff_eq!(m.precision, 1.0, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.recall, 1.0, epsilon = f64::EPSILON);
}

#[test]
fn ner_metrics_per_type() {
    let gold = vec![vec![
        gold_entity("PER", 0, 5),
        gold_entity("PER", 10, 15),
        gold_entity("LOC", 20, 25),
    ]];
    let pred = vec![vec![gold_entity("PER", 0, 5), gold_entity("LOC", 20, 25)]];
    let m = NerMetrics::compute(&pred, &gold);
    let per = m.per_type.get("PER").unwrap();
    assert_abs_diff_eq!(per.precision, 1.0, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(per.recall, 0.5, epsilon = f64::EPSILON);
    assert_eq!(per.support, 2);
    let loc = m.per_type.get("LOC").unwrap();
    assert_abs_diff_eq!(loc.precision, 1.0, epsilon = f64::EPSILON);
    assert_eq!(loc.support, 1);
}

#[test]
fn ner_metrics_empty_inputs() {
    let m = NerMetrics::compute(&[], &[]);
    assert_eq!(m.precision, 0.0);
    assert_eq!(m.recall, 0.0);
    assert_eq!(m.f1, 0.0);
    assert!(m.per_type.is_empty());
}
