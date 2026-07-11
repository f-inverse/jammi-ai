// Hermetic tests for the configurable inference compute precision
// (`GpuConfig::compute_precision` / a per-model `config.json` override):
// F16 encoder inference for embedding, classification, and NER; its fold into
// the materialization identity for both the `Embedding` and `Inference`
// model-producing paths; and the fail-loud bf16 refusal. Uses the tiny BERT /
// ModernBERT-classifier / ModernBERT-NER fixtures checked into
// `cookbook/fixtures/` — no network access required. Candle's CPU backend
// supports F16, so this runs fully on CPU.

use crate::common;

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, StringArray};
use jammi_ai::model::backend::candle::CandleBackend;
use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
use jammi_ai::model::resolver::ModelResolver;
use jammi_ai::model::{LoadedModel, ModelSource, ModelTask};
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::StorageUrl;
use jammi_numerics::distance::cosine_similarity;
use jammi_numerics::ComputePrecision;
use tempfile::TempDir;

const TEXT: &str = "the quick brown fox jumps over the lazy dog";

fn tiny_bert_source() -> ModelSource {
    ModelSource::local(common::cookbook_fixture("tiny_bert"))
}

fn tiny_modernbert_classifier_source() -> ModelSource {
    ModelSource::local(common::cookbook_fixture("tiny_modernbert_classifier"))
}

fn tiny_modernbert_ner_source() -> ModelSource {
    ModelSource::local(common::cookbook_fixture("tiny_modernbert_ner"))
}

/// Resolve + load a fixture through the live engine path
/// (`ModelResolver::resolve` (local) → `CandleBackend::load`) at the given
/// compute precision.
async fn resolve_and_load(
    source: &ModelSource,
    task: ModelTask,
    precision: ComputePrecision,
) -> LoadedModel {
    let catalog_dir = TempDir::new().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, common::test_artifact_store()).unwrap();
    let resolved = resolver.resolve(source, task, None).await.unwrap();

    let backend = CandleBackend;
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: precision,
    };
    backend.load(&resolved, &device_config).unwrap()
}

/// Embed `text` through the live embedding path (`LoadedModel::forward` →
/// `forward_embedding` → `CandleTextForward::forward_pooled` → L2-normalize).
fn embed(model: &LoadedModel, text: &str) -> Vec<f32> {
    let content: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text])) as ArrayRef];
    let output = model.forward(&content, ModelTask::TextEmbedding).unwrap();
    output.float_outputs[0].clone()
}

/// (a) F16 inference produces a valid embedding: finite, correct dimension,
/// unit-norm.
/// (b) It differs from the F32 embedding of the same input (precision is
/// actually taking effect, not silently staying F32) while remaining close by
/// cosine similarity (the two precisions encode the same semantic content).
#[tokio::test]
async fn f16_embedding_is_valid_and_active_but_close_to_f32() {
    let model_f32 = resolve_and_load(
        &tiny_bert_source(),
        ModelTask::TextEmbedding,
        ComputePrecision::F32,
    )
    .await;
    let model_f16 = resolve_and_load(
        &tiny_bert_source(),
        ModelTask::TextEmbedding,
        ComputePrecision::F16,
    )
    .await;

    let vec_f32 = embed(&model_f32, TEXT);
    let vec_f16 = embed(&model_f16, TEXT);

    // (a) A valid embedding: correct dimension (tiny_bert hidden_size = 32),
    // every component finite, unit L2 norm (the embedding path L2-normalizes).
    assert_eq!(vec_f16.len(), 32, "tiny_bert hidden_size is 32");
    assert!(
        vec_f16.iter().all(|x| x.is_finite()),
        "every component of an F16-computed embedding must be finite: {vec_f16:?}"
    );
    let norm: f32 = vec_f16.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-2,
        "F16 embedding must be unit-norm (L2-normalized), got norm {norm}"
    );

    // (b) F16 is active, not silently F32: the two runs must NOT be
    // bit-identical...
    assert_ne!(
        vec_f32, vec_f16,
        "an F16 compute-precision run must not be bit-identical to F32 — \
         precision is not taking effect"
    );
    // ...yet the two precisions encode the same semantic content, so they
    // stay close by cosine similarity.
    let cos = cosine_similarity(&vec_f32, &vec_f16);
    assert!(
        cos > 0.99,
        "F16 and F32 embeddings of the same input should be nearly identical \
         by cosine similarity, got {cos}"
    );
}

/// A session config rooted at `dir`, with `gpu.compute_precision` overridden.
async fn session_at_precision(
    dir: &std::path::Path,
    precision: ComputePrecision,
) -> jammi_ai::session::InferenceSession {
    let mut config = common::test_config(dir);
    config.gpu.compute_precision = precision;
    jammi_ai::session::InferenceSession::new(config)
        .await
        .unwrap()
}

async fn session_with_patents_at(
    dir: &std::path::Path,
    precision: ComputePrecision,
) -> jammi_ai::session::InferenceSession {
    let session = session_at_precision(dir, precision).await;
    session
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
}

/// (c) The compute precision is folded into the embedding materialization
/// identity (K7): two runs of the *same* model over the *same* input,
/// differing only in `compute_precision`, must produce different
/// `DefinitionHash`es — never collide on one identity.
#[tokio::test]
async fn embedding_compute_precision_changes_the_materialization_identity() {
    let (record_f32, hash_f32) =
        run_embedding_and_read_definition_hash(ComputePrecision::F32).await;
    let (record_f16, hash_f16) =
        run_embedding_and_read_definition_hash(ComputePrecision::F16).await;

    assert_ne!(record_f32.table_name, record_f16.table_name);
    assert_ne!(
        hash_f32, hash_f16,
        "F32 and F16 runs of the same model over the same input must not \
         collide on one materialization identity"
    );
}

/// Run `generate_text_embeddings` for the tiny BERT fixture at `precision`
/// and read back the persisted `.materialization.json` sidecar's
/// `definition_hash` — the real production path (`EmbeddingPipeline::run`),
/// not a hand-built descriptor.
async fn run_embedding_and_read_definition_hash(
    precision: ComputePrecision,
) -> (
    jammi_db::catalog::result_repo::ResultTableRecord,
    jammi_db::store::manifest::DefinitionHash,
) {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), precision).await;

    let (record, _outcome) = session
        .generate_text_embeddings(
            "patents",
            &common::cookbook_fixture("tiny_bert").display().to_string(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let parquet_url = StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = session
        .result_store()
        .read_materialization_manifest(&parquet_url)
        .await
        .unwrap()
        .expect("a freshly finalized embedding table must carry a materialization manifest");

    (record, manifest.definition_hash)
}

/// K7, the `Inference` path specifically (classification/NER/regression) —
/// not just `Embedding`: two `infer()` runs of the identical model over the
/// identical input, differing only in `compute_precision`, must produce
/// different `DefinitionHash`es. This is the regression guard for the bug
/// 06f22e5 shipped precision-blind: it folded `compute_precision` into
/// `ProducingDescriptor::Embedding` only, leaving `Inference` collision-prone.
/// Folding it into `ModelIdentity` (part of `MaterializationEnv`, shared by
/// every model-producing descriptor) fixes both at once — this proves the
/// `Inference` half specifically.
#[tokio::test]
async fn classification_compute_precision_changes_the_materialization_identity() {
    let hash_f32 = run_classification_and_read_definition_hash(ComputePrecision::F32).await;
    let hash_f16 = run_classification_and_read_definition_hash(ComputePrecision::F16).await;

    assert_ne!(
        hash_f32, hash_f16,
        "F32 and F16 classification runs of the same model over the same input \
         must not collide on one materialization identity"
    );
}

async fn run_classification_and_read_definition_hash(
    precision: ComputePrecision,
) -> jammi_db::store::manifest::DefinitionHash {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), precision).await;
    let model_source = tiny_modernbert_classifier_source();

    session
        .infer(
            "patents",
            &model_source,
            ModelTask::Classification,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let records = session
        .catalog()
        .find_result_tables(
            "patents",
            Some(ModelTask::Classification),
            Some(&model_source.to_string()),
        )
        .await
        .unwrap();
    let record = records
        .last()
        .expect("infer() must have materialized a result table");
    let parquet_url = StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = session
        .result_store()
        .read_materialization_manifest(&parquet_url)
        .await
        .unwrap()
        .expect("a freshly finalized inference table must carry a materialization manifest");
    manifest.definition_hash
}

/// Crash-coverage (adversarial-audit finding): the embedding/image/audio/
/// regression paths cast their head output to `F32` before `to_vec::<f32>()`,
/// but classification did not — so at `F16` the classifier emits `F16`
/// logits and `to_vec2::<f32>()` hits candle's `UnexpectedDType`. RED before
/// the classification-logits cast, GREEN after: this proves the cast fixed a
/// real crash, not merely "some assertion passes".
#[tokio::test]
async fn classification_at_f16_produces_finite_confidence() {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), ComputePrecision::F16).await;

    let results = session
        .infer(
            "patents",
            &tiny_modernbert_classifier_source(),
            ModelTask::Classification,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    assert!(
        !results.is_empty(),
        "F16 classification must produce at least one batch"
    );
    let batch = &results[0];
    let confidence_col = batch
        .column_by_name("confidence")
        .expect("confidence column should exist")
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let status_col = batch
        .column_by_name("_status")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut saw_ok_row = false;
    for i in 0..batch.num_rows() {
        if status_col.value(i) == "ok" {
            saw_ok_row = true;
            let confidence = confidence_col.value(i);
            assert!(
                confidence.is_finite(),
                "F16 classification confidence must be finite, got {confidence}"
            );
        }
    }
    assert!(
        saw_ok_row,
        "expected at least one successfully classified row"
    );
}

/// The NER peer of the classification crash-coverage test above: the token
/// classifier's logits hit the same `to_vec3::<f32>()` `UnexpectedDType`
/// crash at `F16` before the NER-logits cast.
#[tokio::test]
async fn ner_at_f16_produces_valid_entity_spans() {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), ComputePrecision::F16).await;

    let results = session
        .infer(
            "patents",
            &tiny_modernbert_ner_source(),
            ModelTask::Ner,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    assert!(
        !results.is_empty(),
        "F16 NER must produce at least one batch"
    );
    let batch = &results[0];
    let entities_col = batch
        .column_by_name("entities")
        .expect("entities column should exist")
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let status_col = batch
        .column_by_name("_status")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut saw_ok_row = false;
    for i in 0..batch.num_rows() {
        if status_col.value(i) == "ok" {
            saw_ok_row = true;
            let json: serde_json::Value = serde_json::from_str(entities_col.value(i))
                .expect("F16 NER entities must be valid JSON");
            assert!(json.is_array(), "F16 NER entities should be a JSON array");
        }
    }
    assert!(
        saw_ok_row,
        "expected at least one successfully NER-tagged row"
    );
}

/// bf16 is deferred: an inference request resolving to `BF16` must fail
/// loud with the typed "not yet supported" engine error, never silently run
/// (which would risk the INVALID_PTX class on non-sm_80 hardware) and never
/// silently fall back to another precision.
#[tokio::test]
async fn bf16_inference_request_is_rejected_loudly() {
    let catalog_dir = TempDir::new().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, common::test_artifact_store()).unwrap();
    let source = tiny_bert_source();
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();

    let backend = CandleBackend;
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: ComputePrecision::BF16,
    };
    match backend.load(&resolved, &device_config) {
        Err(JammiError::Model { message, .. }) => {
            assert!(
                message.contains("bf16") && message.to_lowercase().contains("not yet supported"),
                "expected a typed bf16-not-yet-supported refusal, got: {message}"
            );
        }
        Err(other) => panic!("expected a typed JammiError::Model bf16 refusal, got: {other}"),
        Ok(_) => panic!(
            "expected a typed JammiError::Model bf16 refusal, but the model loaded successfully"
        ),
    }
}
