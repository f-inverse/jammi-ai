//! P1 — CPU↔GPU parity for `infer` (`Ner`) over a **ModernBERT** token
//! classifier.
//!
//! gpu-parity-cell: ModernBert × Ner
//!
//! NER is generic over `forward_hidden` (no per-architecture wrapper family
//! the way Classification has `SeqClassifier`), but it is still a distinct
//! served path from `classification_parity`'s `ModernBert × Classification`
//! cell: NER pools **every** token position (`(batch, seq, hidden)` straight
//! into the token classifier), where Classification pulls a single CLS row
//! (the `narrow(seq=0).squeeze(1)` non-contiguous layout `classification_parity`
//! guards). Token classification is argmax over a softmax, so the decisive
//! parity signal is exact label/span agreement — not a cosine floor over a
//! continuous vector. The same `tiny_modernbert_ner` model runs over the
//! same `tiny_ner_corpus.parquet` fixture on a GPU-pinned and a CPU-pinned session;
//! every row's decoded entity spans (keyed by `_row_id`, so the comparison
//! is row-exact regardless of scan order) must carry the same
//! `(label, start, end)` set, with each entity's mean softmax confidence
//! matching within a generous absolute tolerance (the same class of
//! reduction noise `classification_parity` tolerates for its score
//! distribution).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, StringArray};
use tempfile::TempDir;

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use jammi_numerics::ner::types::Entity;

use crate::harness;
use crate::skip_without_gpu;

/// Register the cookbook NER corpus (`id`, `text`) as a source named
/// `"corpus"` on `session`.
async fn add_ner_corpus(session: &Arc<InferenceSession>) {
    session
        .add_source(
            "corpus",
            SourceType::File,
            SourceConnection {
                url: Some(harness::cookbook_fixture_url("tiny_ner_corpus.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

/// Serve `infer` (Ner) over the corpus `text` column and read back each
/// successfully-tagged row's decoded entity spans, keyed by `_row_id` so a
/// CPU and GPU run compare row-exact regardless of scan order.
async fn keyed_entities(session: &Arc<InferenceSession>) -> HashMap<String, Vec<Entity>> {
    let source = ModelSource::parse(&harness::local_model_id("tiny_modernbert_ner"));
    let (batches, _) = session
        .infer(
            "corpus",
            &source,
            ModelTask::Ner,
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let mut out = HashMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("_row_id")
            .expect("infer output has _row_id")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_row_id is Utf8");
        let entities_col = batch
            .column_by_name("entities")
            .expect("infer output has entities")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("entities is Utf8");
        let status_col = batch
            .column_by_name("_status")
            .expect("infer output has _status")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_status is Utf8");
        for i in 0..batch.num_rows() {
            if status_col.value(i) != "ok" {
                continue;
            }
            let entities: Vec<Entity> =
                serde_json::from_str(entities_col.value(i)).expect("entities JSON parses");
            out.insert(ids.value(i).to_string(), entities);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ner_parity_cpu_vs_gpu_over_modernbert() {
    skip_without_gpu!();

    let cpu_dir = TempDir::new().unwrap();
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    add_ner_corpus(&cpu).await;
    let cpu_entities = keyed_entities(&cpu).await;

    let gpu_dir = TempDir::new().unwrap();
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    add_ner_corpus(&gpu).await;
    let gpu_entities = keyed_entities(&gpu).await;

    assert!(
        !cpu_entities.is_empty(),
        "CPU produced no tagged rows — fixture broken"
    );
    assert_eq!(
        gpu_entities.len(),
        cpu_entities.len(),
        "GPU tagged {} rows but CPU tagged {} — the ModernBERT GPU NER forward dropped rows",
        gpu_entities.len(),
        cpu_entities.len(),
    );

    for (key, cpu_row) in &cpu_entities {
        let gpu_row = gpu_entities
            .get(key)
            .unwrap_or_else(|| panic!("GPU has no entities for row {key}"));

        // `Entity`'s `PartialEq` is defined on `(label, start, end)` only, so
        // this is the exact span/label agreement the argmax decode must
        // reproduce identically on both devices.
        assert_eq!(
            cpu_row, gpu_row,
            "row {key}: CPU and GPU decoded different entity spans/labels \
             (CPU {cpu_row:?} vs GPU {gpu_row:?})",
        );

        // And each matched entity's mean softmax confidence agrees up to
        // reduction noise (same tolerance class as classification_parity's
        // score-distribution comparison).
        for (cpu_e, gpu_e) in cpu_row.iter().zip(gpu_row) {
            assert!(
                (cpu_e.confidence - gpu_e.confidence).abs() < 1e-2,
                "row {key} entity {:?}: CPU confidence {} vs GPU {} diverge beyond tolerance",
                cpu_e.label,
                cpu_e.confidence,
                gpu_e.confidence,
            );
        }
    }
}
