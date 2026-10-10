use std::collections::HashMap;

use approx::assert_abs_diff_eq;
use jammi_numerics::ner::{decode_bio_spans, Entity, NerMetrics, Span, TokenizedRow};

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

/// A logit row that predicts class `class` with near-certainty.
fn predicts(class: usize) -> Vec<f32> {
    let mut row = vec![0.0; 5];
    row[class] = 10.0;
    row
}

/// Decode `text` tokenized as `tokens`: each token's byte offsets and the word
/// it belongs to (`None` for a special token), every token real.
fn decode(
    text: &str,
    tokens: &[((usize, usize), Option<u32>)],
    logits: &[Vec<f32>],
    id2label: &HashMap<u32, String>,
) -> jammi_numerics::error::Result<Vec<Entity>> {
    let offsets: Vec<(usize, usize)> = tokens.iter().map(|(o, _)| *o).collect();
    let word_ids: Vec<Option<u32>> = tokens.iter().map(|(_, w)| *w).collect();
    let attention_mask = vec![1; tokens.len()];
    let row = TokenizedRow {
        offsets: &offsets,
        word_ids: &word_ids,
        attention_mask: &attention_mask,
    };
    decode_bio_spans(logits, &row, id2label, text)
}

const SPECIAL: ((usize, usize), Option<u32>) = ((0, 0), None);

#[test]
fn decode_simple_bio_sequence() {
    let tokens = [
        SPECIAL,
        ((0, 4), Some(0)),
        ((5, 10), Some(1)),
        ((11, 13), Some(2)),
        ((14, 20), Some(3)),
        SPECIAL,
    ];
    let logits = [0, 1, 0, 0, 3, 0].map(predicts).to_vec();
    let entities = decode("John works at Google", &tokens, &logits, &id2label()).unwrap();
    assert_eq!(entities.len(), 2);
    assert_eq!(
        (entities[0].label.as_str(), entities[0].text.as_str()),
        ("PER", "John")
    );
    assert_eq!(
        (entities[1].label.as_str(), entities[1].text.as_str()),
        ("ORG", "Google")
    );
}

#[test]
fn decode_multi_word_entity() {
    let tokens = [
        SPECIAL,
        ((0, 3), Some(0)),
        ((4, 8), Some(1)),
        ((9, 13), Some(2)),
        SPECIAL,
    ];
    let logits = [0, 3, 4, 4, 0].map(predicts).to_vec();
    let entities = decode("New York City", &tokens, &logits, &id2label()).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "New York City");
    assert_eq!(entities[0].label, "ORG");
}

#[test]
fn decode_all_o_returns_empty() {
    let tokens = [SPECIAL, ((0, 5), Some(0)), SPECIAL];
    let logits = [0, 0, 0].map(predicts).to_vec();
    assert!(decode("hello", &tokens, &logits, &id2label())
        .unwrap()
        .is_empty());
}

/// "Hooli" tokenizes as `Ho`, `##oli`: one word whose subtokens disagree.
/// The word takes its first subtoken's label and spans both, so it is one
/// entity — never `Ho` and `oli` as two entities of two types.
#[test]
fn a_word_takes_its_first_subtokens_label_and_spans_all_of_them() {
    let tokens = [
        SPECIAL,
        ((0, 2), Some(0)),
        ((2, 5), Some(0)),
        ((6, 10), Some(1)),
        SPECIAL,
    ];
    let logits = [0, 3, 1, 0, 0].map(predicts).to_vec();
    let entities = decode("Hooli rose", &tokens, &logits, &id2label()).unwrap();
    assert_eq!(entities.len(), 1, "one word is one entity: {entities:?}");
    assert_eq!(
        (entities[0].label.as_str(), entities[0].text.as_str()),
        ("ORG", "Hooli")
    );
}

/// A word tagged `I-X` with no open `X` begins one, as a tagger that never
/// emitted the `B-` still named the entity; an `I-` of another type than the
/// open entity's begins a new entity too.
#[test]
fn an_inside_tag_with_no_open_entity_of_its_type_begins_one() {
    let tokens = [
        SPECIAL,
        ((0, 3), Some(0)),
        ((4, 8), Some(1)),
        ((9, 14), Some(2)),
        SPECIAL,
    ];
    let logits = [0, 2, 4, 0, 0].map(predicts).to_vec();
    let entities = decode("Ann Acme hired", &tokens, &logits, &id2label()).unwrap();
    let found: Vec<(&str, &str)> = entities
        .iter()
        .map(|e| (e.label.as_str(), e.text.as_str()))
        .collect();
    assert_eq!(found, [("PER", "Ann"), ("ORG", "Acme")]);
}

/// An IO-tagged checkpoint names its types without a `B-`/`I-` prefix:
/// consecutive words of one type are one entity of that type.
#[test]
fn an_unprefixed_label_continues_an_entity_of_its_own_type() {
    let id2label: HashMap<u32, String> = [(0, "O"), (1, "PER")]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
    let tokens = [
        SPECIAL,
        ((0, 3), Some(0)),
        ((4, 9), Some(1)),
        ((10, 15), Some(2)),
        SPECIAL,
    ];
    let logit = |class: usize| {
        let mut row = vec![0.0; 2];
        row[class] = 10.0;
        row
    };
    let logits = vec![logit(0), logit(1), logit(1), logit(0), logit(0)];
    let entities = decode("Ada Byron wrote", &tokens, &logits, &id2label).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(
        (entities[0].label.as_str(), entities[0].text.as_str()),
        ("PER", "Ada Byron")
    );
}

#[test]
fn decode_refuses_a_non_finite_logit_on_any_real_token() {
    // A NaN on a word's second subtoken is refused as surely as on its first:
    // a diverged model is refused before any row reaches softmax/argmax.
    let tokens = [SPECIAL, ((0, 2), Some(0)), ((2, 5), Some(0)), SPECIAL];
    let mut logits = [0, 3, 3, 0].map(predicts).to_vec();
    logits[2][1] = f32::NAN;
    assert!(decode("Hooli", &tokens, &logits, &id2label()).is_err());
    logits[2][1] = f32::INFINITY;
    assert!(decode("Hooli", &tokens, &logits, &id2label()).is_err());
}

#[test]
fn decode_ignores_non_finite_logit_on_a_special_or_padded_token() {
    // A special token (no word) or a padded position is never fed to softmax,
    // so its logits must not trip the refusal.
    let offsets = vec![(0, 0), (0, 4), (0, 0), (0, 0)];
    let word_ids = vec![None, Some(0), None, Some(1)];
    let attention_mask = vec![1, 1, 1, 0];
    let row = TokenizedRow {
        offsets: &offsets,
        word_ids: &word_ids,
        attention_mask: &attention_mask,
    };
    let mut logits = [0, 1, 0, 0].map(predicts).to_vec();
    logits[0][0] = f32::NAN;
    logits[2][0] = f32::NAN;
    logits[3][0] = f32::NAN;
    let entities = decode_bio_spans(&logits, &row, &id2label(), "John").unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "John");
}

fn span(label: &str, start: usize, end: usize) -> Span {
    Span {
        label: label.into(),
        start,
        end,
    }
}

fn predicted(label: &str, start: usize, end: usize, text: &str, confidence: f32) -> Entity {
    Entity {
        label: label.into(),
        start,
        end,
        text: text.into(),
        confidence,
    }
}

#[test]
fn ner_metrics_perfect_match() {
    let gold = vec![vec![span("PER", 0, 5), span("ORG", 10, 15)]];
    let pred = vec![vec![
        predicted("PER", 0, 5, "Alice", 0.9),
        predicted("ORG", 10, 15, "Acme.", 0.8),
    ]];
    let m = NerMetrics::compute(&pred, &gold);
    assert_abs_diff_eq!(m.precision, 1.0, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.recall, 1.0, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.f1, 1.0, epsilon = f64::EPSILON);
}

#[test]
fn ner_metrics_no_predictions() {
    let gold = vec![vec![span("PER", 0, 5)]];
    let pred = vec![vec![]];
    let m = NerMetrics::compute(&pred, &gold);
    assert_eq!(m.precision, 0.0);
    assert_eq!(m.recall, 0.0);
    assert_eq!(m.f1, 0.0);
}

#[test]
fn ner_metrics_partial_overlap() {
    let gold = vec![vec![span("PER", 0, 5), span("LOC", 10, 15)]];
    let pred = vec![vec![
        predicted("PER", 0, 5, "Alice", 0.9),
        predicted("ORG", 20, 25, "Acmes", 0.6),
    ]];
    let m = NerMetrics::compute(&pred, &gold);
    assert_abs_diff_eq!(m.precision, 0.5, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.recall, 0.5, epsilon = f64::EPSILON);
    assert_abs_diff_eq!(m.f1, 0.5, epsilon = f64::EPSILON);
}

#[test]
fn ner_metrics_per_type() {
    let gold = vec![vec![
        span("PER", 0, 5),
        span("PER", 10, 15),
        span("LOC", 20, 25),
    ]];
    let pred = vec![vec![
        predicted("PER", 0, 5, "Alice", 0.9),
        predicted("LOC", 20, 25, "Paris", 0.7),
    ]];
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
