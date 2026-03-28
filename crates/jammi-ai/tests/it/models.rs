#[allow(unused_imports)]
use jammi_ai::concurrency::GpuScheduler;
#[allow(unused_imports)]
use jammi_ai::model::{
    backend::DeviceConfig, cache::ModelCache, resolver::ModelResolver, tokenizer::TokenizerWrapper,
    BackendType, ModelId, ModelSource, ModelTask,
};
use jammi_engine::catalog::Catalog;
use std::sync::Arc;
use tempfile::tempdir;

// --- HF Hub resolution (live only) ---

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn resolve_hf_hub_sentence_transformer() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
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
        resolved.tokenizer_path.is_some(),
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
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2");
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .unwrap();

    assert_eq!(resolved.backend, BackendType::Candle);
}

// --- Local path resolution ---

#[tokio::test]
async fn resolve_local_path_with_safetensors() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("local_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), b"fake-weights").unwrap();
    std::fs::write(model_dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::local(&model_dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
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

    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::local(&model_dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .unwrap();

    assert_eq!(resolved.backend, BackendType::Ort);
    assert!(resolved
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
    std::fs::write(model_dir.join("model.safetensors"), b"fake-weights").unwrap();
    std::fs::write(model_dir.join("model.onnx"), b"fake-onnx").unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::local(&model_dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, Some(BackendType::Candle))
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
    let hf_api = hf_hub::api::sync::Api::new().unwrap();
    let repo = hf_api.model("sentence-transformers/all-MiniLM-L6-v2".into());
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
    let hf_api = hf_hub::api::sync::Api::new().unwrap();
    let repo = hf_api.model("sentence-transformers/all-MiniLM-L6-v2".into());
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
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
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
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
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
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
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
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
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
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
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

#[test]
fn resolve_local_missing_config_returns_error() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("broken_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    std::fs::write(model_dir.join("model.safetensors"), b"fake-weights").unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::local(&model_dir);
    let result = resolver.resolve(&source, ModelTask::TextEmbedding, None);
    assert!(
        result.is_err(),
        "Missing config.json should fail resolution"
    );
}

#[test]
fn resolve_local_empty_directory_returns_error() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("empty_model");
    std::fs::create_dir_all(&model_dir).unwrap();

    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::local(&model_dir);
    let result = resolver.resolve(&source, ModelTask::TextEmbedding, None);
    assert!(result.is_err(), "Empty directory should fail resolution");
}

#[test]
fn resolve_nonexistent_local_path_returns_error() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(catalog).unwrap();

    let source = ModelSource::local("/nonexistent/path/to/model");
    let result = resolver.resolve(&source, ModelTask::TextEmbedding, None);
    assert!(result.is_err(), "Nonexistent path should fail resolution");
}

#[cfg(feature = "live-hub-tests")]
#[tokio::test]
async fn cache_load_failure_clears_in_flight_state() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let resolver = ModelResolver::new(Arc::clone(&catalog)).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
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
