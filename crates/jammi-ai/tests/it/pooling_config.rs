//! Config-driven pooling: the text-embedding path must pool with the
//! strategy a model's `1_Pooling/config.json` declares, not an unconditional
//! mean. Reuses the hermetic `tiny_bert` fixture (32-dim, 1 layer) checked
//! into `cookbook/fixtures/tiny_bert/` — no network access required.
//!
//! The oracle: dir A (CLS-declared) and dir B (mean-declared) share the
//! IDENTICAL weights/tokenizer/config and differ only in `1_Pooling/`, so
//! their pooled vectors must differ on the same input — the mean-vs-CLS
//! identity check. On `origin/main` (pooling hardcoded to `Mean`), assertion
//! (3) below is false: both dirs mean-pool and their vectors are identical.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use candle_core::{DType, Device};
use candle_nn::VarMap;
use jammi_ai::model::backend::candle::CandleBackend;
use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
use jammi_ai::model::resolver::ModelResolver;
use jammi_ai::model::{LoadedModel, ModelSource, ModelTask};
use jammi_db::catalog::Catalog;
use jammi_encoders::{Bert, BertConfig, Pooling};
use tempfile::tempdir;

const TINY_BERT_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];
const TEXT: &str = "the quick brown fox jumps over the lazy dog";

/// Copy the hermetic `tiny_bert` fixture's weights/config/tokenizer into
/// `dst`, then optionally write a `1_Pooling/config.json` declaring
/// `pooling_flags`. `None` reproduces a bare BERT repo that ships no
/// `1_Pooling/` subfolder at all.
fn build_local_model_dir(dst: &Path, pooling_flags: Option<&serde_json::Value>) {
    std::fs::create_dir_all(dst).unwrap();
    let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
    for name in TINY_BERT_FILES {
        std::fs::copy(fixture.join(name), dst.join(name)).unwrap();
    }
    if let Some(flags) = pooling_flags {
        let pooling_dir = dst.join("1_Pooling");
        std::fs::create_dir_all(&pooling_dir).unwrap();
        std::fs::write(
            pooling_dir.join("config.json"),
            serde_json::to_string(flags).unwrap(),
        )
        .unwrap();
    }
}

fn cls_pooling_config() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": true,
        "pooling_mode_mean_tokens": false,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

fn mean_pooling_config() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": false,
        "pooling_mode_mean_tokens": true,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

/// Resolve + load `dir` through the live engine path: `ModelResolver::resolve`
/// (local) → `CandleBackend::load`.
async fn resolve_and_load(dir: &Path) -> LoadedModel {
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store()).unwrap();
    let source = ModelSource::local(dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();

    let backend = CandleBackend;
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    backend.load(&resolved, &device_config).unwrap()
}

/// Embed `text` through the live embedding path (`LoadedModel::forward` →
/// `forward_embedding` → `CandleTextForward::forward_pooled`).
fn embed(model: &LoadedModel, text: &str) -> Vec<f32> {
    let content: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text])) as ArrayRef];
    let output = model.forward(&content, ModelTask::TextEmbedding).unwrap();
    output.float_outputs[0].clone()
}

/// Independent oracle: load the SAME `tiny_bert` weights directly through
/// `jammi_encoders::Bert` with an explicit `.pooling(strategy)`, bypassing
/// the resolver/`candle.rs` config-dispatch path under test entirely.
/// Assertions (1)/(2) compare the engine's config-driven output against this
/// reference, not against another invocation of the code under test.
fn reference_pooled(fixture_dir: &Path, text: &str, strategy: Pooling) -> Vec<f32> {
    let device = Device::Cpu;
    let config_str = std::fs::read_to_string(fixture_dir.join("config.json")).unwrap();
    let config: BertConfig = serde_json::from_str(&config_str).unwrap();
    let weights = fixture_dir.join("model.safetensors");
    let bert = Bert::builder()
        .pooling(strategy)
        .build(&[&weights], &config, &device, &VarMap::new())
        .unwrap();

    let tokenizer = jammi_ai::model::tokenizer::TokenizerWrapper::from_file(
        &fixture_dir.join("tokenizer.json"),
    )
    .unwrap();
    let encoding = tokenizer.encode_batch(&[text], None).unwrap();
    let rows = encoding.input_ids.len();
    let cols = encoding.input_ids[0].len();
    let flat_ids: Vec<u32> = encoding.input_ids.into_iter().flatten().collect();
    let flat_mask: Vec<u32> = encoding.attention_masks.into_iter().flatten().collect();
    let input_ids = candle_core::Tensor::from_vec(flat_ids, (rows, cols), &device).unwrap();
    let mask = candle_core::Tensor::from_vec(flat_mask, (rows, cols), &device).unwrap();

    let pooled = bert.forward(&input_ids, &mask).unwrap();
    pooled
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()[0]
        .clone()
}

fn approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

#[tokio::test]
async fn cls_declared_pooling_differs_from_mean_declared_pooling() {
    let tmp = tempdir().unwrap();
    let dir_a = tmp.path().join("cls_model");
    let dir_b = tmp.path().join("mean_model");
    let dir_c = tmp.path().join("no_pooling_file_model");

    build_local_model_dir(&dir_a, Some(&cls_pooling_config()));
    build_local_model_dir(&dir_b, Some(&mean_pooling_config()));
    build_local_model_dir(&dir_c, None);

    let model_a = resolve_and_load(&dir_a).await;
    let model_b = resolve_and_load(&dir_b).await;
    let model_c = resolve_and_load(&dir_c).await;

    let vec_a = embed(&model_a, TEXT);
    let vec_b = embed(&model_b, TEXT);
    let vec_c = embed(&model_c, TEXT);

    let reference_cls = reference_pooled(&dir_a, TEXT, Pooling::Cls);
    let reference_mean = reference_pooled(&dir_b, TEXT, Pooling::Mean);

    // (1) dir A (CLS-declared) pools exactly as CLS pooling of the hidden
    // states — verified against the independent `jammi_encoders::Bert`
    // reference built with an explicit `.pooling(Cls)`.
    assert!(
        approx_eq(&vec_a, &reference_cls),
        "dir A (CLS-declared) should equal the independent CLS-pooled reference:\n\
         got      {vec_a:?}\n\
         expected {reference_cls:?}"
    );

    // (1b) dir B (mean-declared) pools exactly as mean pooling of the hidden
    // states.
    assert!(
        approx_eq(&vec_b, &reference_mean),
        "dir B (mean-declared) should equal the independent mean-pooled reference:\n\
         got      {vec_b:?}\n\
         expected {reference_mean:?}"
    );

    // (2) THE oracle assertion: A's vector must differ from B's on the same
    // input. This is RED on origin/main (pooling hardcoded to `Mean` in the
    // builder sites that the live `forward_pooled` path never reads from
    // anyway) — both dirs mean-pool there, so `vec_a == vec_b`.
    assert!(
        !approx_eq(&vec_a, &vec_b),
        "CLS-declared and mean-declared pooling must produce different vectors \
         on the same input; got identical vectors {vec_a:?} — pooling is not \
         being read from 1_Pooling/config.json"
    );

    // (3) No `1_Pooling/` directory at all → falls back to Mean, matching
    // dir B exactly (the historical sentence-transformers default for bare
    // BERT repos).
    assert!(
        approx_eq(&vec_c, &vec_b),
        "a model with no 1_Pooling/config.json should resolve to mean pooling, \
         matching the mean-declared dir:\n\
         got      {vec_c:?}\n\
         expected {vec_b:?}"
    );
}

/// A present-but-syntactically-invalid `1_Pooling/config.json` must hard-error
/// at resolve time, never silently collapse into the "genuinely absent" case
/// that drives the mean-pooling fallback. Reproduces the real failure mode: a
/// truncated/corrupt write of the pooling declaration.
#[tokio::test]
async fn corrupt_pooling_config_json_is_a_hard_error_at_resolve() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("corrupt_pooling_model");
    // Reuses the same fixture-copy helper as the mean/CLS cases, then
    // overwrites the (well-formed) `1_Pooling/config.json` it wrote with
    // syntactically invalid JSON — not merely the wrong shape.
    build_local_model_dir(&dir, Some(&mean_pooling_config()));
    std::fs::write(dir.join("1_Pooling/config.json"), b"{ not valid json").unwrap();

    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store()).unwrap();
    let source = ModelSource::local(&dir);
    let result = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await;

    assert!(
        result.is_err(),
        "a present-but-unparseable 1_Pooling/config.json must hard-error at \
         resolve time, never silently fall back to mean pooling"
    );
}

#[tokio::test]
async fn unsupported_pooling_mode_fails_model_load() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("lasttoken_model");
    build_local_model_dir(
        &dir,
        Some(&serde_json::json!({
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": false,
            "pooling_mode_lasttoken": true,
        })),
    );

    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store()).unwrap();
    let source = ModelSource::local(&dir);
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();

    let backend = CandleBackend;
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let result = backend.load(&resolved, &device_config);
    assert!(
        result.is_err(),
        "a pooling mode the engine cannot represent must fail loudly, never \
         silently fall back to mean pooling"
    );
}
