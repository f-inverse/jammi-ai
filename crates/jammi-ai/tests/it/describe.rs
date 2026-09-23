//! A model is described from its files — its identity, its width, its
//! regression head's form — before, and without, its weights are
//! materialized; a loaded model reports the description it was
//! materialized from. So a process that only plans a model's run (a
//! query tier whose plans are placed elsewhere) holds no weights, and what
//! it plans against is what the executing process records.
//!
//! Each proof drives the SAME `ModelCache` serving uses — `describe` beside
//! `get_or_load` — over the hermetic `tiny_bert` fixture staged the way
//! `pooling_config.rs` stages it, and over a GGUF checkpoint built the way
//! `gguf_qlora.rs` builds one. A description computed a second way beside
//! the load (a re-read of `config.json`, a second digest routine) fails the
//! equalities below the moment either drifts.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use jammi_numerics::WeightQuantization;
use tempfile::{tempdir, TempDir};

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;

use crate::gguf_qlora::{small_fixture, write_gguf_checkpoint, write_json, write_tokenizer};
use crate::pooling_config::{build_local_model_dir, cls_pooling_config, mean_pooling_config};

async fn session() -> (Arc<InferenceSession>, TempDir) {
    let dir = tempdir().unwrap();
    let config = jammi_test_utils::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    (session, dir)
}

/// A one-row `(id, text)` relation to plan an inference over.
fn scan() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["a sentence to embed"])),
        ],
    )
    .unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

/// The description of a local safetensors model is what loading it
/// reports — identity, width, head form — and the load materializes from
/// the very description the submitter read (the digest is hashed once).
/// Describing holds no weights; loading does.
#[tokio::test]
async fn a_local_model_is_described_as_it_is_loaded() {
    let (session, dir) = session().await;
    let model_dir = dir.path().join("model");
    build_local_model_dir(&model_dir, Some(&mean_pooling_config()));
    let source = ModelSource::local(&model_dir);
    let cache = session.model_cache();

    let described = cache
        .describe(&source, ModelTask::TextEmbedding, None)
        .await
        .expect("a local model describes");
    assert!(
        cache.resident_models_for_test().await.is_empty(),
        "describing a model materializes nothing"
    );
    assert_eq!(described.identity().model_id, source.to_string());
    assert_eq!(described.backend_kind(), "candle");
    assert_eq!(described.embedding_dim(), 32, "tiny_bert's hidden size");
    assert_eq!(described.regression_form(), None);
    assert_eq!(described.quantization(), None);

    let guard = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .expect("the described model loads");
    let loaded = guard.model.description();
    assert_eq!(loaded.identity(), described.identity());
    assert_eq!(loaded.embedding_dim(), described.embedding_dim());
    assert_eq!(loaded.regression_form(), described.regression_form());
    assert!(
        Arc::ptr_eq(loaded, &described),
        "the load materialized from the description the submitter read, never a second one"
    );
    assert_eq!(
        cache.resident_models_for_test().await,
        vec![jammi_ai::model::ModelId::from(&source)],
        "loading is what materializes the model"
    );
}

/// A GGUF checkpoint's quantization format is a header fact: the
/// description reports it without reading a tensor, and the load's
/// identity is the description's.
#[tokio::test]
async fn a_gguf_model_is_described_as_it_is_loaded() {
    let (session, dir) = session().await;
    let (tensors, config, sites) = small_fixture(&Device::Cpu);
    let model_dir = dir.path().join("gguf_model");
    write_json(&model_dir, "config.json", &config);
    write_tokenizer(&model_dir);
    write_gguf_checkpoint(&model_dir, &tensors, &sites, GgmlDType::Q8_0);
    let source = ModelSource::local(&model_dir);
    let cache = session.model_cache();

    let described = cache
        .describe(&source, ModelTask::TextEmbedding, None)
        .await
        .expect("a GGUF model describes");
    assert_eq!(described.quantization(), Some(WeightQuantization::Q8_0));
    assert!(cache.resident_models_for_test().await.is_empty());

    let guard = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .expect("the described GGUF model loads");
    assert_eq!(guard.model.description().identity(), described.identity());
    assert_eq!(
        guard.model.description().embedding_dim(),
        described.embedding_dim()
    );
}

/// Planning an inference over a relation — the output schema's width and
/// regression columns are the model's — materializes nothing: the plan is
/// built against the description, and the weights are the executing
/// process's to load.
#[tokio::test]
async fn planning_an_inference_holds_no_weights() {
    let (session, dir) = session().await;
    let model_dir = dir.path().join("model");
    build_local_model_dir(&model_dir, Some(&mean_pooling_config()));
    let source = ModelSource::local(&model_dir);

    let plan = session
        .annotate_plan(
            scan(),
            &source,
            ModelTask::TextEmbedding,
            &["text".to_string()],
            "id",
        )
        .await
        .expect("the inference plans");
    let vector = plan
        .schema()
        .field_with_name("vector")
        .expect("the planned schema carries the embedding column")
        .clone();
    assert!(
        matches!(vector.data_type(), DataType::FixedSizeList(_, 32)),
        "the column's width is the described model's: {:?}",
        vector.data_type()
    );
    assert!(
        session
            .model_cache()
            .resident_models_for_test()
            .await
            .is_empty(),
        "planning the inference materialized the model"
    );
}

/// A description follows its files: mutating an output-affecting file in
/// place under a constant model id yields a fresh description with a
/// different digest — the same bounded-staleness contract a warm
/// `get_or_load` keeps.
#[tokio::test]
async fn a_description_follows_its_files() {
    let (session, dir) = session().await;
    let model_dir = dir.path().join("model");
    build_local_model_dir(&model_dir, Some(&mean_pooling_config()));
    let source = ModelSource::local(&model_dir);
    let cache = session.model_cache();

    let before = cache
        .describe(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let again = cache
        .describe(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    assert!(
        Arc::ptr_eq(&before, &again),
        "unchanged files are described once"
    );

    std::fs::write(
        model_dir.join("1_Pooling/config.json"),
        serde_json::to_string(&cls_pooling_config()).unwrap(),
    )
    .unwrap();
    let after = cache
        .describe(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    assert!(
        !Arc::ptr_eq(&before, &after),
        "changed files are described again"
    );
    assert_ne!(
        before.content_digest(),
        after.content_digest(),
        "the pooling declaration is output-affecting"
    );
}
