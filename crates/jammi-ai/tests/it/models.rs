use jammi_ai::model::{resolver::ModelResolver, WeightsFormat};
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;
use jammi_db::catalog::Catalog;
use std::sync::Arc;
use tempfile::tempdir;

#[cfg(feature = "live-hub-tests")]
use jammi_ai::concurrency::GpuScheduler;
#[cfg(feature = "live-hub-tests")]
use jammi_ai::model::{
    backend::DeviceConfig, cache::ModelCache, hub::HubSource, tokenizer::TokenizerWrapper, ModelId,
};

/// A [`HubSource`] built exactly the way the session choke point builds one
/// — `[models]` defaults plus the real process environment — for the
/// two live tests below that talk to the tokenizer endpoint directly rather
/// than through a `ModelResolver`.
#[cfg(feature = "live-hub-tests")]
fn live_hub_source() -> HubSource {
    HubSource::from_config(&jammi_db::config::ModelsConfig::default(), &|k: &str| {
        std::env::var(k).ok()
    })
    .unwrap()
}

// --- HF Hub resolution (live only) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn resolve_hf_hub_sentence_transformer() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();

    assert_eq!(
        resolved.model_id,
        ModelId("sentence-transformers/all-MiniLM-L6-v2".into())
    );
    assert_eq!(resolved.task, ModelTask::TextEmbedding);
    assert!(
        !resolved.weights_paths.is_empty(),
        "Should have at least one weights file"
    );
    assert!(resolved.config_path.exists(), "config.json should exist");
    assert!(
        resolved.tokenizer.is_some(),
        "Sentence transformer should have a tokenizer"
    );
    assert!(
        resolved.estimated_memory > 0,
        "Memory estimate should be positive"
    );
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn resolve_hf_hub_selects_safetensors_weights() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();

    assert_eq!(resolved.weights_format, WeightsFormat::Safetensors);
}

// --- Local path resolution ---

/// A minimal but genuinely VALID (zero-tensor) safetensors file: the 8-byte
/// little-endian header-length prefix followed by that many bytes of an
/// empty JSON object. `estimate_safetensors_residency` parses the header at
/// RESOLVE time (mirroring the GGUF arm's own resolve-time header parse), so
/// a placeholder like `b"fake-weights"` is not a safetensors file at all and
/// fails to resolve, exactly as it would fail to LOAD. These backend/path-resolution tests only need SOME file
/// at the expected name that a real safetensors header-parse accepts; they
/// never load it.
fn write_minimal_safetensors(path: &std::path::Path) {
    let header = b"{}";
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    std::fs::write(path, buf).unwrap();
}

#[tokio::test]
async fn resolve_local_path_with_safetensors() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("local_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
    write_minimal_safetensors(&model_dir.join("model.safetensors"));
    std::fs::write(model_dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::local(&model_dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();

    assert_eq!(resolved.weights_format, WeightsFormat::Safetensors);
    assert!(resolved
        .weights_paths
        .iter()
        .any(|p| p.ends_with("model.safetensors")));
}

/// An ONNX export is not a weights file the engine loads: a directory that
/// carries only `model.onnx` is refused, naming the files it would load.
#[tokio::test]
async fn resolve_local_refuses_a_directory_with_only_onnx_weights() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("onnx_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
    std::fs::write(model_dir.join("model.onnx"), b"onnx-export").unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::local(&model_dir);
    let Err(error) = resolver.resolve(&source, ModelTask::TextEmbedding).await else {
        panic!("a directory with only ONNX weights must not resolve");
    };
    let message = error.to_string();
    assert!(
        message.contains("model.safetensors") && message.contains("model.gguf"),
        "the refusal names the weights the engine loads: {message}"
    );
    assert!(
        !message.contains("model.onnx"),
        "the refusal never offers ONNX as a loadable format: {message}"
    );
}

/// An ONNX export appearing beside `model.safetensors` changes nothing a
/// resolve loads: the directory resolves to the same safetensors file
/// before and after.
#[tokio::test]
async fn an_onnx_export_beside_safetensors_changes_nothing_a_resolve_loads() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("onnx_appears_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
    write_minimal_safetensors(&model_dir.join("model.safetensors"));

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let source = ModelSource::local(&model_dir);

    let before = resolver
        .resolve(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();
    std::fs::write(model_dir.join("model.onnx"), b"onnx-export").unwrap();
    let after = resolver
        .resolve(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();

    assert_eq!(before.weights_paths, after.weights_paths);
    assert_eq!(after.weights_format, WeightsFormat::Safetensors);
    assert!(after.weights_paths[0].ends_with("model.safetensors"));
}

// --- Tokenizer encoding (live only) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn tokenizer_encode_batch_produces_padded_output() {
    let hf_api = live_hub_source();
    let repo = hf_api
        .api()
        .model("sentence-transformers/all-MiniLM-L6-v2".into());
    let tokenizer_path = repo.get("tokenizer.json").unwrap();

    let tokenizer = TokenizerWrapper::from_file(&tokenizer_path).unwrap();

    let encoding = tokenizer
        .encode_batch(&["hello", "hello world foo bar"], None)
        .unwrap();

    assert_eq!(encoding.input_ids.len(), 2);
    assert_eq!(
        encoding.input_ids[0].len(),
        encoding.input_ids[1].len(),
        "Sequences should be padded to same length"
    );
    assert_eq!(
        encoding.attention_masks[0].len(),
        encoding.attention_masks[1].len(),
    );

    let short_mask = &encoding.attention_masks[0];
    let long_mask = &encoding.attention_masks[1];
    let short_ones: u32 = short_mask.iter().sum();
    let long_ones: u32 = long_mask.iter().sum();
    assert!(
        short_ones < long_ones,
        "Shorter sequence should have fewer active tokens"
    );
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn tokenizer_encode_batch_with_truncation() {
    let hf_api = live_hub_source();
    let repo = hf_api
        .api()
        .model("sentence-transformers/all-MiniLM-L6-v2".into());
    let tokenizer_path = repo.get("tokenizer.json").unwrap();

    let tokenizer = TokenizerWrapper::from_file(&tokenizer_path).unwrap();

    let long_text = "word ".repeat(1000);
    let encoding = tokenizer.encode_batch(&[&long_text], Some(32)).unwrap();

    assert!(
        encoding.input_ids[0].len() <= 32,
        "Truncation should cap sequence length at max_length"
    );
}

// --- Model cache (live only) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn cache_get_or_load_returns_guard_with_ref_count() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        Arc::clone(&catalog),
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        devices: vec![-1],
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    assert!(std::mem::size_of_val(&*guard.model) > 0);

    let guard2 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    assert!(Arc::ptr_eq(&guard.model, &guard2.model));
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn cache_ref_count_decrements_on_guard_drop() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        Arc::clone(&catalog),
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        devices: vec![-1],
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let guard1 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    let guard2 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    drop(guard2);

    let guard3 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    assert!(Arc::ptr_eq(&guard1.model, &guard3.model));
}

#[test]
fn activation_memory_scaling() {
    use jammi_ai::model::ModelDimensions;

    let dims = ModelDimensions {
        hidden_size: 384,
        num_layers: 6,
        num_attention_heads: 12,
        intermediate_size: 1536,
    };

    // Batch linearity: memory scales linearly with batch size
    let mem_b1 = dims.estimate_activation_memory(1, 128);
    let mem_b32 = dims.estimate_activation_memory(32, 128);
    assert!(mem_b32 > mem_b1);
    assert_eq!(
        mem_b32,
        mem_b1 * 32,
        "Should scale linearly with batch size"
    );

    // Seq-len superlinearity: doubling seq_len more than doubles memory (attention is quadratic)
    let mem_s128 = dims.estimate_activation_memory(1, 128);
    let mem_s256 = dims.estimate_activation_memory(1, 256);
    assert!(
        mem_s256 > mem_s128 * 2,
        "Doubling seq_len should more than double memory (attention is quadratic), \
         got mem_s128={mem_s128}, mem_s256={mem_s256}"
    );
}

// --- Preload (live only) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn preload_loads_model_into_cache_without_returning_guard() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        Arc::clone(&catalog),
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        devices: vec![-1],
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    cache
        .preload(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    assert!(std::mem::size_of_val(&*guard.model) > 0);
}

// --- Single-flight (live only) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn single_flight_concurrent_loads_coalesce() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        Arc::clone(&catalog),
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        devices: vec![-1],
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = Arc::new(ModelCache::new(resolver, device_config, scheduler));

    let cache1 = Arc::clone(&cache);
    let cache2 = Arc::clone(&cache);

    let (g1, g2) = tokio::join!(
        tokio::spawn(async move {
            cache1
                .get_or_load(
                    &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                    ModelTask::TextEmbedding,
                )
                .await
        }),
        tokio::spawn(async move {
            cache2
                .get_or_load(
                    &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                    ModelTask::TextEmbedding,
                )
                .await
        }),
    );

    let guard1 = g1.unwrap().unwrap();
    let guard2 = g2.unwrap().unwrap();
    assert!(Arc::ptr_eq(&guard1.model, &guard2.model));
}

// --- Eviction (live only) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn eviction_skips_model_with_active_guard() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        Arc::clone(&catalog),
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        devices: vec![-1],
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    assert!(std::mem::size_of_val(&*guard.model) > 0);
}

// --- Failure paths ---

#[tokio::test]
async fn resolve_local_missing_config_returns_error() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("broken_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("model.safetensors"), b"fake-weights").unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::local(&model_dir);
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    assert!(
        result.is_err(),
        "Missing config.json should fail resolution"
    );
}

#[tokio::test]
async fn resolve_local_empty_directory_returns_error() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("empty_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::local(&model_dir);
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    assert!(result.is_err(), "Empty directory should fail resolution");
}

#[tokio::test]
async fn resolve_nonexistent_local_path_returns_error() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::local("/nonexistent/path/to/model");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    assert!(result.is_err(), "Nonexistent path should fail resolution");
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn cache_load_failure_clears_in_flight_state() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        Arc::clone(&catalog),
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        devices: vec![-1],
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let result = cache
        .get_or_load(
            &ModelSource::hf("nonexistent-org/nonexistent-model-xyz"),
            ModelTask::TextEmbedding,
        )
        .await;
    assert!(result.is_err());

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();
    assert!(std::mem::size_of_val(&*guard.model) > 0);
}

/// Registering a model with task `Ner` round-trips through the catalog
/// without losing its variant — proves the `models.task` write/read path
/// goes through `ModelTask::as_db_str` / `try_from_db_str` end-to-end
/// (catches a regression where the catalog stored a raw string then
/// `parse_model_row` decoded a different variant).
#[tokio::test]
async fn ner_model_round_trips_through_catalog() {
    use jammi_db::catalog::model_repo::RegisterModelParams;

    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();

    catalog
        .register_model(RegisterModelParams {
            model_id: "tenant/ner-model",
            version: 1,
            model_type: "huggingface",
            backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
            task: ModelTask::Ner,
            base_model_id: None,
            external_location: None,
            config_json: None,
        })
        .await
        .unwrap();

    let fetched = catalog
        .get_model("tenant/ner-model")
        .await
        .unwrap()
        .expect("model just registered should be present");
    assert_eq!(fetched.task, ModelTask::Ner);
    assert_eq!(
        fetched.backend,
        jammi_db::catalog::model_repo::ModelBackendKind::Candle
    );
}

// =============================================================================
// A `model_type == "fine-tuned"` catalog record with a broken
// lineage/adapter pointer is a typed refusal, never a silent fall-back to
// the base model. `crate::tower_adapters`/`crate::fine_tune`'s own
// `*_serves_cold_after_restart` tests pin the end-to-end, happy-path
// mechanism (`ModelCache::get_or_load`'s post-load bookkeeping never
// clobbers a fine-tuned row); these tests pin the resolver's OWN
// defense-in-depth refusal directly, independent of how a record ends up
// broken.
// =============================================================================

/// A fine-tuned record with no location (a directly-registered row whose
/// adapter bundle was never attached) must refuse to
/// resolve — never silently resolve to the unadapted base model.
#[tokio::test]
async fn fine_tuned_record_without_a_location_refuses_to_resolve() {
    use jammi_db::catalog::model_repo::RegisterModelParams;

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let base_dir = crate::common::cookbook_fixture("tiny_bert");
    let base_id = format!("local:{}", base_dir.display());

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:broken-artifact-path",
            version: 1,
            model_type: "fine-tuned",
            backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            external_location: None,
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:broken-artifact-path");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record with no location must refuse to resolve, never \
             silently serve the unadapted base model"
        ),
        Err(e) => e,
    };
    let message = err.to_string();
    assert!(
        message.contains("jammi:fine-tuned:broken-artifact-path"),
        "refusal must name the broken model id, got: {message}"
    );
    assert!(
        message.contains("no location"),
        "refusal must name the missing pointer, got: {message}"
    );
}

/// A fine-tuned record whose `base_model_id` is `None` (the lineage pointer
/// was never written, or was lost) must refuse to resolve — never silently
/// fall through to resolving it as an ordinary directly-registered model,
/// which would misread its adapter-only artifact directory as a full
/// checkpoint.
#[tokio::test]
async fn fine_tuned_record_without_base_model_id_refuses_to_resolve() {
    use jammi_db::catalog::model_repo::RegisterModelParams;

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:broken-base-id",
            version: 1,
            model_type: "fine-tuned",
            backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            external_location: Some("/nonexistent/adapter/prefix"),
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:broken-base-id");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record with no base_model_id must refuse to resolve, never \
             silently fall through to treating it as a directly-registered model"
        ),
        Err(e) => e,
    };
    let message = err.to_string();
    assert!(
        message.contains("jammi:fine-tuned:broken-base-id"),
        "refusal must name the broken model id, got: {message}"
    );
    assert!(
        message.contains("base_model_id"),
        "refusal must name the missing pointer, got: {message}"
    );
}

/// The negative-control seam: a fine-tuned record whose adapter bundle
/// WAS published successfully but whose `adapter.safetensors` has since gone
/// missing on disk (a partial delete, artifact-store corruption — the exact
/// shape the cold-restart integration tests' negative control produces) must
/// still refuse through the REAL resolver, and with the SAME typed
/// `JammiError::Model` variant every other refusal in this arm raises.
///
/// `ArtifactStore::fetch_artifact`'s own `file://` in-place verification
/// (`ArtifactStore::verify_files`) raises on the missing file as
/// `JammiError::Storage`/`JammiError::Io`; `ModelResolver::try_catalog_lookup`
/// must map it to `JammiError::Model`, or a caller matching on that variant
/// (as this file's other fine-tuned-record tests, and the cold-restart
/// integration tests' negative control, do) sees the wrong variant despite the
/// refusal firing and naming the file. This
/// uses a real `file://` `ArtifactStore` (never `memory://`) so the missing
/// file is a genuine on-disk absence, matching production.
#[tokio::test]
async fn fine_tuned_adapter_bundle_missing_file_refuses_as_typed_model_error() {
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let base_dir = crate::common::cookbook_fixture("tiny_bert");
    let base_id = format!("local:{}", base_dir.display());

    let artifacts_root = dir.path().join("artifacts");
    let store = Arc::new(
        ArtifactStore::with_root(
            StorageUrl::parse(artifacts_root.to_str().unwrap()).unwrap(),
            StorageRegistry::new(),
            dir.path().join("artifact_cache"),
        )
        .unwrap(),
    );
    let prefix = crate::common::finalize_fine_tuned_model(
        &catalog,
        &store,
        "jammi:fine-tuned:missing-adapter-file",
        &base_id,
        &[
            (
                "adapter_config.json".to_string(),
                bytes::Bytes::from_static(b"{}"),
            ),
            (
                "adapter.safetensors".to_string(),
                bytes::Bytes::from_static(b"weights"),
            ),
        ],
    )
    .await;
    std::fs::remove_file(std::path::PathBuf::from(prefix.path()).join("adapter.safetensors"))
        .unwrap();

    let resolver = ModelResolver::new(catalog, store, crate::common::test_hub_source()).unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:missing-adapter-file");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record whose adapter.safetensors was deleted must refuse to \
             resolve through the real resolver, never silently serve the unadapted base"
        ),
        Err(e) => e,
    };
    assert!(
        matches!(err, jammi_db::error::JammiError::Model { .. }),
        "the refusal must be the SAME typed JammiError::Model variant every other \
         fine-tuned-record refusal in this arm raises, got a different variant: {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("adapter.safetensors"),
        "refusal must name the missing file, got: {message}"
    );
    assert!(
        message.contains("integrity check"),
        "a manifest-listed key truly absent is an INTEGRITY failure and must say so, distinct \
         from an unpublished-bundle refusal, got: {message}"
    );
}

/// The flip side of the missing-FILE test above: the
/// `manifest.json` itself is absent — no bundle was ever published at this
/// prefix at all. This is NOT bundle corruption (there is no manifest in
/// hand to say anything is corrupt); it must be a DIFFERENT message than the
/// integrity-failure refusal, though the SAME `JammiError::Model` variant.
#[tokio::test]
async fn fine_tuned_adapter_bundle_unpublished_refuses_as_typed_model_error() {
    use jammi_db::catalog::model_repo::RegisterModelParams;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let base_dir = crate::common::cookbook_fixture("tiny_bert");
    let base_id = format!("local:{}", base_dir.display());

    let artifacts_root = dir.path().join("artifacts");
    let store = Arc::new(
        ArtifactStore::with_root(
            StorageUrl::parse(artifacts_root.to_str().unwrap()).unwrap(),
            StorageRegistry::new(),
            dir.path().join("artifact_cache"),
        )
        .unwrap(),
    );
    // A prefix the catalog points at that was NEVER written, so no
    // manifest exists. Stands in for a
    // never-published bundle, a misdirected pointer, or a clobbered
    // pointer left aimed at the wrong directory.
    let never_published = artifacts_root.join("ghost-bundle");
    let prefix = format!("file://{}", never_published.display());

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:unpublished-bundle",
            version: 1,
            model_type: "fine-tuned",
            backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            external_location: Some(&prefix),
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(catalog, store, crate::common::test_hub_source()).unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:unpublished-bundle");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record whose location names a prefix nothing was ever \
             published at must refuse to resolve, never silently serve the unadapted base"
        ),
        Err(e) => e,
    };
    assert!(
        matches!(err, jammi_db::error::JammiError::Model { .. }),
        "the refusal must be the SAME typed JammiError::Model variant the integrity-failure \
         sibling test raises, got a different variant: {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("no adapter bundle is published"),
        "the message must say no bundle is published, never call this corruption, got: {message}"
    );
    assert!(
        !message.contains("integrity check"),
        "an unpublished bundle is NOT an integrity failure — no manifest is in hand to say \
         anything is corrupt, got: {message}"
    );
}

/// Id-shape backstop: a corrupted catalog (e.g. a bookkeeping write that
/// rewrote a resolved id's `model_type` unconditionally) can leave a
/// `jammi:fine-tuned:{job_id}` row typed as `"huggingface"` instead of
/// `"fine-tuned"`. Nothing else ever
/// mints this reserved prefix (`fine_tuned_model_id` is its sole producer),
/// so this shape can only be corruption, never an honestly-registered base
/// model that happens to share the naming convention. `try_catalog_lookup`
/// must refuse it by name rather than resolve it as a base checkpoint —
/// serving the unadapted base with no signal is the failure these tests
/// exist to catch.
#[tokio::test]
async fn fine_tuned_prefix_with_wrong_model_type_refuses_to_resolve() {
    use jammi_db::catalog::model_repo::RegisterModelParams;

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let base_dir = crate::common::cookbook_fixture("tiny_bert");
    let base_id = format!("local:{}", base_dir.display());

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:corrupted-by-old-build",
            version: 1,
            model_type: "huggingface",
            backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            external_location: Some(base_dir.to_str().unwrap()),
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:corrupted-by-old-build");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    let err = match result {
        Ok(_) => panic!(
            "a jammi:fine-tuned: id whose row is typed 'huggingface' must refuse to \
             resolve, never silently serve the row's location as an ordinary \
             base checkpoint"
        ),
        Err(e) => e,
    };
    let message = err.to_string();
    assert!(
        message.contains("jammi:fine-tuned:corrupted-by-old-build"),
        "refusal must name the broken model id, got: {message}"
    );
    assert!(
        message.contains("huggingface"),
        "refusal must name the row's actual (wrong) model_type, got: {message}"
    );
}

/// The OTHER pointer-corruption seam: a fine-tuned record whose
/// location string does not even parse as a storage URL (an unknown
/// scheme) is itself a corrupted CATALOG RECORD — never a storage-layer
/// transport fault — so `ModelResolver::try_catalog_lookup` must refuse
/// with the SAME typed `JammiError::Model` variant the sibling missing-file
/// refusal above raises, naming both the model id and the invalid pointer.
#[tokio::test]
async fn fine_tuned_adapter_bundle_corrupted_pointer_refuses_as_typed_model_error() {
    use jammi_db::catalog::model_repo::RegisterModelParams;

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let base_dir = crate::common::cookbook_fixture("tiny_bert");
    let base_id = format!("local:{}", base_dir.display());

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:corrupted-pointer",
            version: 1,
            model_type: "fine-tuned",
            backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            external_location: Some("not-a-real-scheme://nonsense"),
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:corrupted-pointer");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record whose location does not parse as a storage URL must \
             refuse to resolve, never silently serve the unadapted base"
        ),
        Err(e) => e,
    };
    assert!(
        matches!(err, jammi_db::error::JammiError::Model { .. }),
        "an unparseable location is a corrupted catalog record, not a storage-layer \
         transport fault — it must be the SAME typed JammiError::Model variant every other \
         fine-tuned-record refusal in this arm raises, got a different variant: {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("jammi:fine-tuned:corrupted-pointer"),
        "refusal must name the broken model id, got: {message}"
    );
    assert!(
        message.contains("not-a-real-scheme"),
        "refusal must name the invalid pointer string, got: {message}"
    );
}

#[cfg(all(unix, feature = "unprivileged-tests"))]
#[tokio::test]
async fn fine_tuned_adapter_bundle_permission_fault_is_not_a_typed_model_error() {
    use std::os::unix::fs::PermissionsExt;

    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;
    jammi_test_resources::assert_permissions_enforced();

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let base_dir = crate::common::cookbook_fixture("tiny_bert");
    let base_id = format!("local:{}", base_dir.display());

    let artifacts_root = dir.path().join("artifacts");
    let store = Arc::new(
        ArtifactStore::with_root(
            StorageUrl::parse(artifacts_root.to_str().unwrap()).unwrap(),
            StorageRegistry::new(),
            dir.path().join("artifact_cache"),
        )
        .unwrap(),
    );
    let prefix = crate::common::finalize_fine_tuned_model(
        &catalog,
        &store,
        "jammi:fine-tuned:permission-fault-bundle",
        &base_id,
        &[
            (
                "adapter_config.json".to_string(),
                bytes::Bytes::from_static(b"{}"),
            ),
            (
                "adapter.safetensors".to_string(),
                bytes::Bytes::from_static(b"weights"),
            ),
        ],
    )
    .await;
    let weights_path = std::path::PathBuf::from(prefix.path()).join("adapter.safetensors");

    std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o000)).unwrap();

    let resolver = ModelResolver::new(catalog, store, crate::common::test_hub_source()).unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:permission-fault-bundle");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding).await;

    // Restore permissions unconditionally so the tempdir's own Drop cleanup
    // never has to fight the chmod.
    let _ = std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o644));

    let err = match result {
        Ok(_) => panic!(
            "resolving a fine-tuned model whose adapter file is permission-denied must fail, \
             never silently serve a vector"
        ),
        Err(e) => e,
    };
    assert!(
        matches!(
            err,
            jammi_db::error::JammiError::Storage(jammi_db::storage::StorageError::Io { .. })
        ),
        "a permission/transport fault reading an INTACT bundle is NOT this model's fault — it \
         must propagate as the SAME variant both real reload surfaces actually raise for a \
         transport fault, StorageError::Io (never be folded into JammiError::Model, which a \
         gRPC client maps to InvalidArgument instead of Internal), got: {err:?}"
    );
}
