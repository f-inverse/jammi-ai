//! esc-057 fix test (`closes_escape: esc-057`): a model-content digest folds
//! into `ModelIdentity` so `1_Pooling/config.json`, tokenizer, and weights
//! bytes are output-affecting relative to the bare `model_id` string a local
//! model names a model by — the local `model_id` IS the directory path, so a
//! two-directory form cannot exercise the collision the escape reports (two
//! different paths are two different `model_id`s regardless of content). Each
//! assertion below mutates ONE model directory IN PLACE, under a CONSTANT
//! `model_id` (the same path, resolved twice), and proves the recorded
//! `DefinitionHash` differs before vs. after.
//!
//! Reuses `pooling_config.rs`'s fixture-copy helpers (`build_local_model_dir`,
//! `resolve_and_load`, `TINY_BERT_FILES`, `cls_pooling_config`,
//! `mean_pooling_config`) — the identical hermetic `tiny_bert` fixture and the
//! identical live `ModelResolver::resolve` → `CandleBackend::load` path that
//! file already exercises, so this proves the content-digest fold through the
//! SAME production code the pooling-dispatch tests already trust.
//!
//! Reverting the production digest computation
//! (`model::backend::candle::compute_model_content_digest`) — e.g. back to a
//! constant/omitted `content_digest` on `ModelIdentity` — makes assertions
//! (a)–(c) below RED; the tests themselves are unchanged either way.

use std::path::Path;

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_db::store::manifest::{
    ComputeDevice, DefinitionHash, MaterializationEnv, MaterializationManifest, ModelIdentity,
    ProducingDescriptor,
};
use tempfile::tempdir;

use crate::pooling_config::{
    build_local_model_dir, cls_pooling_config, mean_pooling_config, resolve_and_load,
    TINY_BERT_FILES,
};

/// `tiny_bert`'s hidden size (see `pooling_config.rs`'s fixture doc) — the
/// fixed `dimensions` every descriptor below shares.
const DIMENSIONS: usize = 32;

/// Resolve + load `dir` through the live engine path and fold the result into
/// the SAME `ModelIdentity` / `ProducingDescriptor::Embedding` /
/// `MaterializationEnv` shape `pipeline/embedding.rs::EmbeddingPipeline::run`
/// builds in production, then return the resulting `DefinitionHash` — the
/// real fold (`MaterializationManifest::definition_of`), not a hand-rolled
/// stand-in. `model_id` is `dir`'s canonical string (`ModelSource::local`'s
/// `Display`), so calling this twice on the SAME `dir` — mutated in place
/// between calls — holds `model_id` constant across both calls, exactly as
/// esc-057 requires.
async fn definition_hash_for(dir: &Path) -> DefinitionHash {
    let model = resolve_and_load(dir).await;
    let model_id = ModelSource::local(dir).to_string();

    let identity = ModelIdentity {
        model_id: model_id.clone(),
        backend: model.backend_kind().to_string(),
        compute_precision: model.compute_precision(),
        content_digest: model
            .content_digest()
            .expect("the Candle backend always reports a Sha256 content digest"),
    };
    let descriptor = ProducingDescriptor::Embedding {
        model_id,
        task: ModelTask::TextEmbedding,
        source_id: "esc057_source".to_string(),
        columns: vec!["text".to_string()],
        key_column: "id".to_string(),
        dimensions: DIMENSIONS,
    };
    let env = MaterializationEnv::new(ComputeDevice::Cpu, vec![identity]);

    MaterializationManifest::definition_of(&descriptor, &env)
        .expect("definition_of must succeed for a well-formed descriptor/env pair")
}

/// (a) Mutating `1_Pooling/config.json`'s bytes in place — under a CONSTANT
/// `model_id` (the same directory path, resolved before and after) — must
/// change `definition_hash`. RED before the content-digest fold: pooling
/// bytes were not an output-affecting determinant of `ModelIdentity` at all.
#[tokio::test]
async fn pooling_config_bytes_mutation_changes_the_definition_hash() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    build_local_model_dir(&dir, Some(&mean_pooling_config()));

    let hash_before = definition_hash_for(&dir).await;

    // In-place mutation: overwrite the SAME file at the SAME path with a
    // different (still valid) pooling declaration. `dir` — and therefore
    // `model_id` — never changes.
    std::fs::write(
        dir.join("1_Pooling/config.json"),
        serde_json::to_string(&cls_pooling_config()).unwrap(),
    )
    .unwrap();

    let hash_after = definition_hash_for(&dir).await;

    assert_ne!(
        hash_before, hash_after,
        "mutating 1_Pooling/config.json bytes in place under a constant model_id \
         must change definition_hash (esc-057)"
    );
}

/// (b) The tokenizer peer of (a): mutating `tokenizer.json`'s bytes in place
/// under a constant `model_id` must change `definition_hash`. The mutation
/// (a trailing newline appended to the file) is a byte-level change that
/// stays valid, parseable JSON — `serde_json`/the `tokenizers` crate ignore
/// trailing whitespace — so the model still loads successfully both times;
/// only the recorded content digest is under test here, not tokenizer
/// behavior.
#[tokio::test]
async fn tokenizer_bytes_mutation_changes_the_definition_hash() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    build_local_model_dir(&dir, Some(&mean_pooling_config()));

    let hash_before = definition_hash_for(&dir).await;

    let tokenizer_path = dir.join("tokenizer.json");
    let mut bytes = std::fs::read(&tokenizer_path).unwrap();
    bytes.push(b'\n');
    std::fs::write(&tokenizer_path, &bytes).unwrap();

    let hash_after = definition_hash_for(&dir).await;

    assert_ne!(
        hash_before, hash_after,
        "mutating tokenizer.json bytes in place under a constant model_id must \
         change definition_hash (esc-057)"
    );
}

/// (c) The weights peer of (a)/(b): mutating `model.safetensors`'s bytes in
/// place under a constant `model_id` must change `definition_hash`. Flips the
/// LAST byte of the file — safetensors lays out a length-prefixed JSON header
/// followed by raw tensor bytes to EOF, so the final byte is always inside
/// the raw tensor data (never the header), and flipping it changes the file's
/// bytes without altering its shape/dtype/offset metadata — the model still
/// mmaps and loads successfully both times.
#[tokio::test]
async fn weights_bytes_mutation_changes_the_definition_hash() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    build_local_model_dir(&dir, Some(&mean_pooling_config()));

    let hash_before = definition_hash_for(&dir).await;

    let weights_path = dir.join("model.safetensors");
    let mut bytes = std::fs::read(&weights_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&weights_path, &bytes).unwrap();

    let hash_after = definition_hash_for(&dir).await;

    assert_ne!(
        hash_before, hash_after,
        "mutating model.safetensors bytes in place under a constant model_id \
         must change definition_hash (esc-057)"
    );
}

/// (d) The non-vacuous control: TWO SEPARATE model directories with
/// byte-identical content — same weights/config/tokenizer/pooling, copied
/// independently — must report the IDENTICAL content digest. (Their full
/// `ModelIdentity`/`definition_hash` legitimately differ, because a local
/// `model_id` is the directory path and the two dirs live at different
/// paths — this control isolates the digest computation itself via
/// `LoadedModel::content_digest()`, proving it is a deterministic function of
/// file bytes, not of directory path, mtimes, or filesystem enumeration
/// order.) Without this, assertions (a)–(c) alone could not distinguish "the
/// digest is a real content hash" from "the digest happens to vary with
/// SOMETHING every time" (e.g. a timestamp or an unstable readdir order).
#[tokio::test]
async fn byte_identical_model_dirs_produce_the_identical_content_digest() {
    let tmp = tempdir().unwrap();
    let dir_a = tmp.path().join("model_a");
    let dir_b = tmp.path().join("model_b");
    build_local_model_dir(&dir_a, Some(&mean_pooling_config()));
    build_local_model_dir(&dir_b, Some(&mean_pooling_config()));

    // `build_local_model_dir` always copies the same fixture bytes and writes
    // the same serialized pooling JSON, but assert the raw bytes are actually
    // identical too — otherwise this control would be vacuous (asserting
    // digests match without first proving the inputs match).
    for name in TINY_BERT_FILES {
        assert_eq!(
            std::fs::read(dir_a.join(name)).unwrap(),
            std::fs::read(dir_b.join(name)).unwrap(),
            "fixture file {name} must be byte-identical between dir_a and dir_b for this \
             control to be non-vacuous"
        );
    }
    assert_eq!(
        std::fs::read(dir_a.join("1_Pooling/config.json")).unwrap(),
        std::fs::read(dir_b.join("1_Pooling/config.json")).unwrap(),
        "1_Pooling/config.json must be byte-identical between dir_a and dir_b for this \
         control to be non-vacuous"
    );

    let model_a = resolve_and_load(&dir_a).await;
    let model_b = resolve_and_load(&dir_b).await;

    let digest_a = model_a.content_digest().unwrap();
    let digest_b = model_b.content_digest().unwrap();

    assert_eq!(
        digest_a, digest_b,
        "byte-identical model directories at different paths must produce the \
         identical content digest (determinism, the non-vacuous control)"
    );
}
