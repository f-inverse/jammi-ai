//! A BERT-family checkpoint is served through the head its architecture
//! trained with, and RoBERTa's encoder numbers positions as RoBERTa does.
//!
//! Each fixture is a tiny random checkpoint `transformers` itself wrote,
//! beside what `transformers` computed for a fixed batch of texts
//! (`tests/fixtures/generate_tiny_bert_family_heads.py`):
//!
//! - `tiny_bert_classifier`: `BertForSequenceClassification`, whose classifier
//!   reads the pooler's `tanh(dense(h_cls))`, not the raw `[CLS]` state.
//! - `tiny_roberta_classifier`: `RobertaForSequenceClassification`, encoder
//!   under `roberta.`, head `out_proj(tanh(dense(h_cls)))`.
//! - `tiny_roberta`: a raw `RobertaModel`, whose positions start at
//!   `pad_token_id + 1`.
//!
//! The generator refuses to write a fixture whose head or position offset
//! moves the output by less than 0.05, so a served model that skips either
//! fails here by far more than the tolerance.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use tempfile::{tempdir, TempDir};

use jammi_ai::session::InferenceSession;
use jammi_datafusion::{ModelSource, ModelTask};

/// The issue's bound: every class probability within 1e-4 of `transformers`.
const PROBABILITY_TOLERANCE: f32 = 1e-4;

/// Cosine floor for an embedding against the one `transformers`' hidden
/// states give: fp32 kernel-order rounding only.
const MIN_COSINE: f32 = 1.0 - 1e-5;

#[derive(serde::Deserialize)]
struct ClassifierGoldens {
    texts: Vec<String>,
    id2label: BTreeMap<String, String>,
    probabilities: Vec<Vec<f32>>,
}

#[derive(serde::Deserialize)]
struct EncoderGoldens {
    texts: Vec<String>,
    last_hidden_state: Vec<Vec<Vec<f32>>>,
}

fn fixture(name: &str) -> PathBuf {
    jammi_test_utils::fixture(name)
}

fn goldens<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let raw = std::fs::read_to_string(fixture(name).join("goldens.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

async fn session() -> (Arc<InferenceSession>, TempDir) {
    let dir = tempdir().unwrap();
    let config = jammi_test_utils::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    (session, dir)
}

/// The fixture's texts, run as one padded batch through the model's own forward.
async fn forward(name: &str, task: ModelTask, texts: &[String]) -> jammi_datafusion::BackendOutput {
    let (session, _dir) = session().await;
    let source = ModelSource::local(fixture(name));
    let guard = session
        .model_cache()
        .get_or_load(&source, task)
        .await
        .unwrap_or_else(|e| panic!("{name} loads for {task:?}: {e}"));
    let content: Vec<ArrayRef> = vec![Arc::new(StringArray::from(texts.to_vec())) as ArrayRef];
    guard.model.forward(&content, task).await.unwrap()
}

/// Every row's served class probabilities match `transformers`' to within
/// the tolerance, in the checkpoint's own label order.
async fn assert_classifier_matches_transformers(name: &str) {
    let goldens: ClassifierGoldens = goldens(name);
    let output = forward(name, ModelTask::Classification, &goldens.texts).await;
    let scores = &output.string_outputs[1];
    for (row, (text, expected)) in goldens.texts.iter().zip(&goldens.probabilities).enumerate() {
        assert!(output.row_status[row], "row {row} ({text:?}) was served");
        let served: BTreeMap<String, f32> = serde_json::from_str(&scores[row]).unwrap();
        for (index, want) in expected.iter().enumerate() {
            let label = &goldens.id2label[&index.to_string()];
            let got = served[label];
            assert!(
                (got - want).abs() <= PROBABILITY_TOLERANCE,
                "{name} row {row} ({text:?}) `{label}`: served {got}, transformers {want}"
            );
        }
    }
}

#[tokio::test]
async fn a_bert_classifier_scores_through_its_pooler_as_transformers_does() {
    assert_classifier_matches_transformers("tiny_bert_classifier").await;
}

#[tokio::test]
async fn a_roberta_classifier_scores_through_its_dense_head_as_transformers_does() {
    assert_classifier_matches_transformers("tiny_roberta_classifier").await;
}

/// A RoBERTa embedding is the mean of the hidden states `transformers`
/// computes with RoBERTa's positions, normalized; BERT's 0-based positions
/// give a different vector.
#[tokio::test]
async fn a_roberta_encoder_embeds_with_roberta_positions() {
    let goldens: EncoderGoldens = goldens("tiny_roberta");
    let output = forward("tiny_roberta", ModelTask::TextEmbedding, &goldens.texts).await;
    let width = goldens.last_hidden_state[0][0].len();
    for (row, (text, hidden)) in goldens
        .texts
        .iter()
        .zip(&goldens.last_hidden_state)
        .enumerate()
    {
        assert!(output.row_status[row], "row {row} ({text:?}) was served");
        let mut expected = vec![0.0_f32; width];
        for token in hidden {
            for (sum, x) in expected.iter_mut().zip(token) {
                *sum += x / hidden.len() as f32;
            }
        }
        let served = &output.float_outputs[0][row * width..(row + 1) * width];
        let cosine = cosine(served, &expected);
        assert!(
            cosine >= MIN_COSINE,
            "tiny_roberta row {row} ({text:?}): cosine {cosine} to transformers' mean hidden state"
        );
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}
