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

// `pub(crate)`: `tests/it/content_digest.rs` (esc-057's fix test, `closes_escape:
// esc-057`) reuses this exact fixture-copy helper to build/mutate model dirs
// through the identical live resolve→load path this file already exercises,
// rather than a second, independently-drifting copy of the same fixture-
// staging logic.
pub(crate) const TINY_BERT_FILES: [&str; 3] =
    ["config.json", "model.safetensors", "tokenizer.json"];
const TEXT: &str = "the quick brown fox jumps over the lazy dog";

/// Copy the hermetic `tiny_bert` fixture's weights/config/tokenizer into
/// `dst`, then optionally write a `1_Pooling/config.json` declaring
/// `pooling_flags`. `None` reproduces a bare BERT repo that ships no
/// `1_Pooling/` subfolder at all.
pub(crate) fn build_local_model_dir(dst: &Path, pooling_flags: Option<&serde_json::Value>) {
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

/// Write `dst/preprocessor_config.json` (F-2, `closes_escape`-adjacent audit
/// round 62 finding): a bare BERT repo's `tiny_bert` fixture ships no such
/// file, so a `1_Pooling`-style presence/absence test needs a helper to add
/// one. `pub(crate)`: shared with `content_digest.rs` (the digest-fold
/// mutation test) and `cache_staleness.rs` (the appearance-tripwire test).
pub(crate) fn write_preprocessor_config(dst: &Path, cfg: &serde_json::Value) {
    std::fs::write(
        dst.join("preprocessor_config.json"),
        serde_json::to_string(cfg).unwrap(),
    )
    .unwrap();
}

/// A harmless, syntactically-valid `preprocessor_config.json` body — its
/// content doesn't need to be a real CLAP feature-extractor geometry for
/// F-2's digest-fold / appearance tests (`tiny_bert` is a plain BERT
/// text-embedding model, never routed through the CLAP audio front-end that
/// actually interprets these keys), only present-and-parseable-as-JSON.
pub(crate) fn sample_preprocessor_config() -> serde_json::Value {
    serde_json::json!({
        "feature_extractor_type": "ClapFeatureExtractor",
        "sampling_rate": 48000,
    })
}

// `pub(crate)`: shared with `tests/it/content_digest.rs` — see `TINY_BERT_FILES`'s doc.
pub(crate) fn cls_pooling_config() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": true,
        "pooling_mode_mean_tokens": false,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

// `pub(crate)`: shared with `tests/it/content_digest.rs` — see `TINY_BERT_FILES`'s doc.
pub(crate) fn mean_pooling_config() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": false,
        "pooling_mode_mean_tokens": true,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

/// Add an inert marker key to a pooling declaration so its serialized byte
/// LENGTH deliberately differs from an equivalent config without the marker
/// — audit round 62, F-4a: a straight `cls_pooling_config()` ⇄
/// `mean_pooling_config()` swap (each has exactly one `true`/4 chars and
/// five `false`/5 chars, just at different keys) is byte-length-IDENTICAL
/// when both are serialized the same way, so a staleness-tripwire test built
/// on that swap alone would rest entirely on sub-second mtime resolution to
/// detect the mutation — never asserted, and not portable to a coarser
/// filesystem clock. `pooling_from_config` only inspects keys prefixed
/// `pooling_mode_`, so this key is invisible to production pooling-selection
/// logic; it exists purely to make the on-disk byte length change,
/// independent of which flags are true/false or of `to_string` vs
/// `to_string_pretty` formatting (a second, ACCIDENTAL source of length
/// difference this helper deliberately does not rely on either).
pub(crate) fn with_length_marker(mut cfg: serde_json::Value) -> serde_json::Value {
    cfg.as_object_mut()
        .expect("pooling config fixtures are always JSON objects")
        .insert(
            "_test_length_marker".to_string(),
            serde_json::Value::String("x".to_string()),
        );
    cfg
}

/// Resolve + load `dir` through the live engine path: `ModelResolver::resolve`
/// (local) → `CandleBackend::load`. `pub(crate)`: shared with
/// `tests/it/content_digest.rs` (esc-057's fix test) — see `TINY_BERT_FILES`'s
/// doc.
pub(crate) async fn resolve_and_load(dir: &Path) -> LoadedModel {
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
