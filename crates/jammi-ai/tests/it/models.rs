use jammi_ai::model::{resolver::ModelResolver, BackendType, ModelSource, ModelTask};
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
/// (H3/H4) — `[models]` defaults plus the real process environment — for the
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
        .resolve(&source, ModelTask::TextEmbedding, None)
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
async fn resolve_hf_hub_selects_candle_for_safetensors_model() {
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
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();

    assert_eq!(resolved.backend, BackendType::Candle);
}

// --- Local path resolution ---

/// A minimal but genuinely VALID (zero-tensor) safetensors file: the 8-byte
/// little-endian header-length prefix followed by that many bytes of an
/// empty JSON object. Issue #431's `estimate_safetensors_residency` now
/// parses the header at RESOLVE time (mirroring the GGUF arm's own
/// resolve-time header parse) — a placeholder like the pre-#431
/// `b"fake-weights"` this file used to write is not a safetensors file at
/// all and correctly fails to resolve now, exactly as it would already have
/// failed to LOAD. These backend/path-resolution tests only need SOME file
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
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();

    assert_eq!(resolved.backend, BackendType::Candle);
    assert!(resolved
        .weights_paths
        .iter()
        .any(|p| p.ends_with("model.safetensors")));
}

#[tokio::test]
async fn resolve_local_path_with_onnx() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("onnx_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
    std::fs::write(model_dir.join("model.onnx"), b"fake-onnx").unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::local(&model_dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();

    assert_eq!(resolved.backend, BackendType::Ort);
    assert!(resolved
        .weights_paths
        .iter()
        .any(|p| p.ends_with("model.onnx")));
}

/// Audit round 62, adversarial round 10 (the weights slot's cold-side
/// backend flip): `model.onnx` appearing beside an already-existing
/// `model.safetensors` must flip a FRESH resolve's backend selection to ORT
/// — the same `has_onnx` preference `resolve_local` always applies,
/// independent of load order. This is the cold-side half of
/// `model::backend::candle::digest_fingerprint_audit62_tests::weights_slot_alternate_arm_appearing_trips_the_probe`,
/// which proves the WARM staleness probe detects the appearance; this test
/// proves what a subsequent cold reload actually does once it fires.
#[tokio::test]
async fn resolve_local_prefers_onnx_once_it_appears_alongside_existing_safetensors() {
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

    // Before model.onnx exists: Candle, via model.safetensors.
    let resolved_before = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    assert_eq!(resolved_before.backend, BackendType::Candle);

    // model.onnx APPEARS beside the existing model.safetensors.
    std::fs::write(model_dir.join("model.onnx"), b"fake-onnx").unwrap();

    // A FRESH resolve of the SAME directory now prefers ORT — proving the
    // staleness the warm probe detects (candle.rs's peer test) corresponds
    // to a REAL change in what a cold reload would do, not merely an inert
    // file the resolver ignores.
    let resolved_after = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    assert_eq!(
        resolved_after.backend,
        BackendType::Ort,
        "a fresh resolve of a directory where model.onnx now exists alongside \
         model.safetensors must prefer ORT (resolve_local's has_onnx branch), \
         even though an earlier resolve of the SAME directory picked Candle"
    );
    assert!(resolved_after
        .weights_paths
        .iter()
        .any(|p| p.ends_with("model.onnx")));
}

// --- Backend selection heuristic ---

#[tokio::test]
async fn backend_hint_overrides_heuristic() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("hint_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
    write_minimal_safetensors(&model_dir.join("model.safetensors"));
    std::fs::write(model_dir.join("model.onnx"), b"fake-onnx").unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(
        catalog,
        crate::common::test_artifact_store(),
        crate::common::test_hub_source(),
    )
    .unwrap();

    let source = ModelSource::local(&model_dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, Some(BackendType::Candle))
        .await
        .unwrap();

    assert_eq!(
        resolved.backend,
        BackendType::Candle,
        "Hint should override heuristic"
    );
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
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .unwrap();

    assert!(std::mem::size_of_val(&*guard.model) > 0);

    let guard2 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
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
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let guard1 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .unwrap();

    let guard2 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .unwrap();

    drop(guard2);

    let guard3 = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
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
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    cache
        .preload(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .unwrap();

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
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
        memory_fraction: 1.0,
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
                    None,
                )
                .await
        }),
        tokio::spawn(async move {
            cache2
                .get_or_load(
                    &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
                    ModelTask::TextEmbedding,
                    None,
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
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
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
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
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
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
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
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
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
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config, scheduler);

    let result = cache
        .get_or_load(
            &ModelSource::hf("nonexistent-org/nonexistent-model-xyz"),
            ModelTask::TextEmbedding,
            None,
        )
        .await;
    assert!(result.is_err());

    let guard = cache
        .get_or_load(
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
            ModelTask::TextEmbedding,
            None,
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
            backend: "candle",
            task: ModelTask::Ner,
            base_model_id: None,
            artifact_path: None,
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
    assert_eq!(fetched.backend, "candle");
}

// =============================================================================
// esc-089: a `model_type == "fine-tuned"` catalog record with a broken
// lineage/adapter pointer is a typed refusal, never a silent fall-back to
// the base model. `crate::tower_adapters`/`crate::fine_tune`'s own
// `*_serves_cold_after_restart` tests pin the end-to-end, happy-path
// mechanism this unit fixed (`ModelCache::get_or_load`'s post-load
// bookkeeping no longer clobbers a fine-tuned row); these two pin the
// resolver's OWN defense-in-depth refusal directly, independent of how a
// record ends up broken.
// =============================================================================

/// A fine-tuned record whose `artifact_path` is `None` (the finalize CAS
/// never ran, or its pointer was lost after the fact) must refuse to
/// resolve — never silently resolve to the unadapted base model.
#[tokio::test]
async fn fine_tuned_record_without_artifact_path_refuses_to_resolve() {
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
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            artifact_path: None,
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
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record with no artifact_path must refuse to resolve, never \
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
        message.contains("artifact_path"),
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
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: Some("/nonexistent/adapter/prefix"),
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
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
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

/// esc-089's negative-control seam: a fine-tuned record whose adapter bundle
/// WAS published successfully but whose `adapter.safetensors` has since gone
/// missing on disk (a partial delete, artifact-store corruption — the exact
/// shape the cold-restart integration tests' negative control produces) must
/// still refuse through the REAL resolver, and with the SAME typed
/// `JammiError::Model` variant every other refusal in this arm raises.
///
/// Before this test's fix: `ArtifactStore::fetch_artifact`'s own `file://`
/// in-place verification (`ArtifactStore::verify_files`) already raised loudly
/// on the missing file, but as `JammiError::Storage`/`JammiError::Io` —
/// `ModelResolver::try_catalog_lookup` propagated it via `?` untouched. A
/// caller matching on `JammiError::Model` (as this file's other two esc-089
/// tests, and the cold-restart integration tests' negative control, do) would
/// see the wrong variant despite the refusal firing and naming the file. This
/// uses a real `file://` `ArtifactStore` (never `memory://`) so the missing
/// file is a genuine on-disk absence, matching production.
#[tokio::test]
async fn fine_tuned_adapter_bundle_missing_file_refuses_as_typed_model_error() {
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
    let prefix = store
        .put_artifact(
            &["broken-bundle"],
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
        .await
        .unwrap();
    std::fs::remove_file(std::path::PathBuf::from(prefix.path()).join("adapter.safetensors"))
        .unwrap();

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:missing-adapter-file",
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            artifact_path: Some(prefix.as_str()),
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(catalog, store, crate::common::test_hub_source()).unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:missing-adapter-file");
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
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

/// The flip side of the missing-FILE test above (round-3 audit F2): the
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
    // A prefix the catalog points at that was NEVER written — no
    // `put_artifact` call at all, so no manifest exists. Stands in for a
    // never-published bundle, a misdirected pointer, or a pre-fix clobbered
    // pointer left aimed at the wrong directory.
    let never_published = artifacts_root.join("ghost-bundle");
    let prefix = format!("file://{}", never_published.display());

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:unpublished-bundle",
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            artifact_path: Some(&prefix),
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(catalog, store, crate::common::test_hub_source()).unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:unpublished-bundle");
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record whose artifact_path names a prefix nothing was ever \
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

/// esc-089 backstop: a catalog corrupted by a pre-fix build (the model
/// cache's post-load bookkeeping used to rewrite ANY resolved id's
/// `model_type` unconditionally) can leave a `jammi:fine-tuned:{job_id}`
/// row typed as `"huggingface"` instead of `"fine-tuned"`. Nothing else ever
/// mints this reserved prefix (`fine_tuned_model_id` is its sole producer),
/// so this shape can only be corruption, never an honestly-registered base
/// model that happens to share the naming convention. `try_catalog_lookup`
/// must refuse it by name rather than resolve it as a base checkpoint —
/// serving the unadapted base with no signal is exactly the esc-089 failure
/// this whole unit exists to close.
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
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            artifact_path: Some(base_dir.to_str().unwrap()),
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
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
    let err = match result {
        Ok(_) => panic!(
            "a jammi:fine-tuned: id whose row is typed 'huggingface' must refuse to \
             resolve, never silently serve the row's artifact_path as an ordinary \
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

/// esc-089's OTHER pointer-corruption seam: a fine-tuned record whose
/// `artifact_path` string does not even parse as a storage URL (an unknown
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
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            artifact_path: Some("not-a-real-scheme://nonsense"),
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
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;
    let err = match result {
        Ok(_) => panic!(
            "a fine-tuned record whose artifact_path does not parse as a storage URL must \
             refuse to resolve, never silently serve the unadapted base"
        ),
        Err(e) => e,
    };
    assert!(
        matches!(err, jammi_db::error::JammiError::Model { .. }),
        "an unparseable artifact_path is a corrupted catalog record, not a storage-layer \
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

/// esc-089's flip side: a fine-tuned record whose adapter bundle is
/// published and INTACT, but whose backing file becomes unreadable for a
/// reason that has nothing to do with the bundle's own integrity — a
/// permission fault standing in for a transient object-store outage, via
/// real fault injection (`chmod`, mirroring
/// `crates/jammi-ai/src/fine_tune/trainer.rs`'s own technique) — must NOT
/// be folded into the typed `JammiError::Model` refusal the sibling
/// missing-file test above pins. `ArtifactStore::fetch_artifact` re-types
/// ONLY a manifest-promised key that is truly absent
/// (`object_store::Error::NotFound`) into an integrity failure
/// (`StorageError::Layout`); every other driver fault — including a
/// permission-denied open, which `object_store`'s `LocalFileSystem` folds
/// into `Error::Generic` (its own `UnableToOpenFile` is a private local
/// error, never constructed outside that crate), never `NotFound` — stays
/// `StorageError::Io` and must reach the caller unchanged, so a gRPC client sees `Internal`
/// (a transient-outage shape), never `InvalidArgument` (a bad-request
/// shape), for a fault that is not this model's fault at all.
///
/// The require-gate polarity every `chmod` permission-fault probe in this
/// suite shares (esc-089): `probe` performs the fault-injection premise
/// check itself — "can this process still read/write through a chmod'd
/// path?" — and returns `true` if the fault was BYPASSED (root, or a
/// mode-ignoring filesystem). A bypass is normally a loud, `eprintln`'d skip:
/// the fault-injection premise the caller needs simply does not hold on this
/// host. But under `JAMMI_REQUIRE_POSIX_PERMS=1` (the CI lane that is
/// SUPPOSED to run unprivileged with real POSIX permission enforcement) a
/// bypass is instead a hard `panic!` — silently returning `true` in that lane
/// would let a permission-fault regression go completely uncaught.
///
/// This is a thin local wrapper of the same canonical shape carried by every
/// other `chmod`/permission-fault probe in this crate (`ci/kernel-oracle-
/// helpers.txt`'s KO-7 registry is `(file, fn)`-scoped: a shared helper
/// defined in `common/mod.rs` cannot be registered for a call site in a
/// DIFFERENT file, so each file that needs this polarity carries its own
/// copy rather than delegating).
///
/// Returns `true` if the caller must restore permissions and skip; `false` if
/// the fault was genuinely injected and the test should proceed.
#[cfg(unix)]
fn chmod_bypassed(test_name: &str, probe: impl FnOnce() -> bool) -> bool {
    let bypassed = probe();
    if bypassed {
        if std::env::var_os("JAMMI_REQUIRE_POSIX_PERMS").is_some() {
            panic!(
                "JAMMI_REQUIRE_POSIX_PERMS is set but '{test_name}' could not inject its \
                 permission fault (root, or a mode-ignoring filesystem) — the fault-injection \
                 premise this test needs does not hold; a silent skip is not acceptable here"
            );
        }
        eprintln!("{test_name}: chmod bypassed (root?) — skipping");
    }
    bypassed
}

#[cfg(unix)]
#[tokio::test]
async fn fine_tuned_adapter_bundle_permission_fault_is_not_a_typed_model_error() {
    use std::os::unix::fs::PermissionsExt;

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
    let prefix = store
        .put_artifact(
            &["permission-fault-bundle"],
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
        .await
        .unwrap();
    let weights_path = std::path::PathBuf::from(prefix.path()).join("adapter.safetensors");

    // PROBE: chmod the file unreadable, then confirm the process actually
    // cannot read it — root (and a mode-ignoring filesystem) bypasses this,
    // in which case the fault-injection premise this test needs never holds.
    // Shared require-gate polarity (esc-089): under
    // `JAMMI_REQUIRE_POSIX_PERMS=1` a bypass panics rather than skipping.
    std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o000)).unwrap();
    let bypassed = chmod_bypassed(
        "fine_tuned_adapter_bundle_permission_fault_is_not_a_typed_model_error",
        || std::fs::read(&weights_path).is_ok(),
    );
    if bypassed {
        let _ = std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o644));
        return;
    }

    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:permission-fault-bundle",
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some(&base_id),
            artifact_path: Some(prefix.as_str()),
            config_json: None,
        })
        .await
        .unwrap();

    let resolver = ModelResolver::new(catalog, store, crate::common::test_hub_source()).unwrap();
    let source = ModelSource::hf("jammi:fine-tuned:permission-fault-bundle");
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;

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
