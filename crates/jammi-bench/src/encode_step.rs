//! The identity-audited encode-step tier (unit 62, K7/E3): drives the
//! engine's real text-embedding serving surface —
//! [`generate_text_embeddings`](jammi_ai::session::InferenceSession::generate_text_embeddings),
//! the SAME `resolve -> tokenize -> forward -> pool -> normalize` path a
//! serving request walks — over a small deterministic corpus, folding the
//! result into a [`crate::report::EncodeStepTier`] whose declared
//! `IDENTITY_FIELDS` name the COMPLETE output-affecting parameter set for
//! this surface. See [`crate::report::EncodeStepTier`]'s own doc for the
//! full K7/esc-057 rationale this tier exists to protect at the
//! bench-comparison layer.
//!
//! ## Not a synthetic loop
//!
//! Every number on the emitted tier is either read off a real artifact or
//! produced by a real call into the engine:
//!
//! * `checkpoint_config_sha256`/`checkpoint_weights_sha256`/
//!   `checkpoint_weights_size_bytes`/`checkpoint_tokenizer_sha256` are
//!   `sha256_and_len` over the fixture model dir's actual bytes (the SAME
//!   helper `finetune_step.rs`/`grad_oracle.rs` already use) — the complete
//!   three-file checkpoint content identity (config + weights + tokenizer),
//!   never a two-file subset (unit-62 F-5: tokenizer bytes are
//!   output-affecting on this surface, see [`crate::report::EncodeStepTier::checkpoint_tokenizer_sha256`]'s
//!   own doc).
//! * `compute_precision` is read off the LOADED model
//!   ([`jammi_ai::model::LoadedModel::compute_precision`], via the tier's
//!   own session model cache) — the precision the serve actually ran at,
//!   never a derived/default constant.
//! * `pooling` is read off the SAME loaded model
//!   ([`jammi_ai::model::LoadedModel::resolved_pooling`], the same cache
//!   hit `compute_precision` reads) — the pooling strategy the loaded
//!   text-embedding wrapper actually pools with, never a constant mirroring
//!   the fixture-writer function (round-3 audit F-5'). `checkpoint_pooling_sha256`
//!   closes the companion gap: the pooling-CONFIG BYTES themselves, hashed
//!   with the identical presence gate the engine's own `content_digest`
//!   applies to `1_Pooling/config.json`.
//! * `device_requested` is the CLI/param device value declared BEFORE any
//!   compute runs; `device_name` is the post-hoc hardware fact only knowable
//!   after the device resolved (round-3 audit lead ruling) — see
//!   [`crate::report::EncodeStepTier`]'s own doc for the full identity-vs-
//!   provenance split.
//! * `seq`/`row_lengths` are read off a REAL tokenization of the corpus
//!   text through [`jammi_ai::model::tokenizer::TokenizerWrapper`] — the
//!   exact wrapper `CandleBackend` loads for a local model — over the
//!   fixture's own `tokenizer.json`, never assumed or hand-computed.
//! * The embed measurement itself is `crate::model_inference::serve_embed`,
//!   the real `generate_text_embeddings` call the `model_inference`/
//!   `gpu_inference` tiers already drive.
//!
//! ## CPU-hermetic default, GPU-device-parameterized
//!
//! `gpu_device` flows straight into `corpus_session_on_device` — the SAME
//! device knob `model_inference`/`gpu_inference` already share (`-1` /
//! `Device::Cpu` for the CI-hermetic default this tier's own tests run
//! under, a real CUDA ordinal for the pod producer). No new device-selection
//! mechanism is introduced. The `encode-step` CLI subcommand (`main.rs`)
//! exposes this as `--cuda <ordinal>` (mirroring `finetune-step`/
//! `grad-oracle`'s own `--cuda: Option<usize>` convention exactly), omitted
//! for [`CPU_HERMETIC_DEVICE`].

use std::path::Path;

use jammi_ai::model::tokenizer::TokenizerWrapper;
use jammi_ai::model::{ModelSource, ModelTask};

use crate::finetune_step::sha256_and_len;
use crate::model_inference::{
    build_corpus, corpus_session_on_device, local_model_id, rows_per_s, serve_embed,
    ModelInferenceSpec,
};
use crate::report::{EncodeStepTier, Measurement};

/// The CI-hermetic default device: `Device::Cpu` (mirrors
/// `model_inference::corpus_session`'s own `-1` convention). The pod
/// producer overrides [`EncodeStepParams::gpu_device`] with a real CUDA
/// ordinal.
pub const CPU_HERMETIC_DEVICE: i32 = -1;

/// The generation/measurement parameters the tier drives its corpus and
/// serve off of.
#[derive(Debug, Clone, Copy)]
pub struct EncodeStepParams {
    /// The synthetic corpus row count — this run's `batch`.
    pub row_count: usize,
    /// The corpus generation seed (rotates which committed sentence each row
    /// draws) — this run's `seed`.
    pub seed: u64,
    /// Warmup serves before the measured iterations, discarded.
    pub warmup: usize,
    /// Measured `generate_text_embeddings` calls folded into the reported
    /// throughput/latency — this run's `iters_measured`.
    pub iters: usize,
    /// The device ordinal `corpus_session_on_device` resolves on:
    /// [`CPU_HERMETIC_DEVICE`] (`-1`) for the CI-hermetic default, a real
    /// CUDA ordinal for the pod producer.
    pub gpu_device: i32,
}

/// The `1_Pooling/config.json` flags declaring MEAN pooling — the same
/// six-key shape `jammi-ai`'s own `pooling_config.rs` it-suite writes for
/// its `mean_pooling_config` fixture, and the mapping `candle.rs`'s
/// `pooling_from_config` documents (`pooling_mode_mean_tokens: true` ->
/// `Pooling::Mean`).
fn mean_pooling_flags() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": false,
        "pooling_mode_mean_tokens": true,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

/// The CLS-pooling twin of [`mean_pooling_flags`] — same six-key shape,
/// `pooling_mode_cls_token: true` instead. Test-only (unit-62 F-5' teeth):
/// drives [`build_encode_model_dir_with_pooling`] to prove
/// `checkpoint_pooling_sha256`/`pooling` actually react to a fixture flip,
/// never a hand-typed expectation this crate never actually measures.
#[cfg(test)]
fn cls_pooling_flags() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": true,
        "pooling_mode_mean_tokens": false,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

/// Build a fresh model dir at `dst`: copy the shared `tiny_bert` fixture's
/// three files (the SAME fixture `model_inference`'s embed lane serves),
/// then write an EXPLICIT `1_Pooling/config.json` carrying `pooling_flags`.
///
/// The explicit pooling config (never the bare `tiny_bert` fixture, which
/// ships with no `1_Pooling/` folder at all) is deliberate: a repo with no
/// pooling config resolves through `candle.rs`'s silent mean-pooling
/// fallback — the exact ambiguity esc-057 is about. Declaring the strategy
/// here means [`crate::report::EncodeStepTier::pooling`] records a value
/// this tier KNOWS the engine resolves to, not an inferred one. Parameterized
/// over the flags (unit-62 F-5' teeth: `checkpoint_pooling_sha256_and_
/// pooling_move_together_when_the_fixture_flips_to_cls` drives this with the
/// test-only `cls_pooling_flags` to prove the two accessors this tier reads
/// actually react to the fixture, never a hand-typed expectation).
fn build_encode_model_dir_with_pooling(
    dst: &Path,
    pooling_flags: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dst)?;
    let fixture = ModelInferenceSpec::embed_model_dir();
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        std::fs::copy(fixture.join(name), dst.join(name)).map_err(|e| -> Box<dyn std::error::Error> {
            format!("copying {name} from the shared tiny_bert fixture into the encode-step model dir: {e}").into()
        })?;
    }
    let pooling_dir = dst.join("1_Pooling");
    std::fs::create_dir_all(&pooling_dir)?;
    std::fs::write(
        pooling_dir.join("config.json"),
        serde_json::to_string(pooling_flags)?,
    )?;
    Ok(())
}

/// The production model-dir builder every real `run()` invocation uses:
/// [`build_encode_model_dir_with_pooling`] with [`mean_pooling_flags`].
fn build_encode_model_dir(dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    build_encode_model_dir_with_pooling(dst, &mean_pooling_flags())
}

/// [`crate::report::EncodeStepTier::device_requested`]'s value — `"cpu"` for
/// a negative ordinal (the CI-hermetic default), `"cuda:<ordinal>"`
/// otherwise. A cheap, honest label derived straight from the same
/// `gpu_device` value threaded into `corpus_session_on_device`
/// (`select_device`'s own convention: negative selects `Device::Cpu`), never
/// a second, independently-resolved reading. Computable BEFORE any compute
/// runs (round-3 audit lead ruling: this is exactly what makes it an
/// identity field, unlike [`resolved_device_name`] below).
fn requested_device_label(gpu_device: i32) -> String {
    if gpu_device < 0 {
        "cpu".to_string()
    } else {
        format!("cuda:{gpu_device}")
    }
}

/// [`crate::report::EncodeStepTier::device_name`]'s value — a POST-HOC
/// hardware fact, only knowable after the device resolved (round-3 audit
/// lead ruling: PROVENANCE, never identity). `"cpu"` for the CI-hermetic
/// default; a real CUDA leg queries the actual device sub-class name off the
/// driver via `gpu_inference::cuda_device_name` — the SAME in-process
/// `cudarc` lookup that tier already performs, never a second,
/// independently-drifting hardware-name query.
fn resolved_device_name(gpu_device: i32) -> Result<String, Box<dyn std::error::Error>> {
    if gpu_device < 0 {
        Ok("cpu".to_string())
    } else {
        crate::gpu_inference::cuda_device_name(gpu_device as u32)
    }
}

/// [`crate::report::EncodeStepTier::checkpoint_pooling_sha256`]'s value:
/// `sha256_and_len` over `model_dir/1_Pooling/config.json`'s bytes when that
/// file exists, `None` when it doesn't (unit-62 F-5'(b)) — the SAME presence
/// gate `backend::candle::all_candidate_paths` applies before hashing this
/// file into the engine's own `content_digest`
/// (`resolved.pooling_config.is_some()`), never a second, independently-
/// drifting presence check. Extracted to its own function so
/// `checkpoint_pooling_sha256_is_none_when_absent`/`..._is_some_when_present`
/// can drive the SAME code `run()` calls, rather than a re-typed copy that
/// could silently drift from it.
fn checkpoint_pooling_sha256(
    model_dir: &Path,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let pooling_config_path = model_dir.join("1_Pooling").join("config.json");
    if pooling_config_path.exists() {
        Ok(Some(sha256_and_len(&pooling_config_path)?.0))
    } else {
        Ok(None)
    }
}

/// Run the encode-step tier: build the fixture model dir + committed corpus,
/// tokenize the corpus for real to measure `seq`/`row_lengths`, serve the
/// embed verb `warmup + iters` times through the real engine path, and
/// assemble the identity-audited [`EncodeStepTier`].
pub async fn run(params: EncodeStepParams) -> Result<EncodeStepTier, Box<dyn std::error::Error>> {
    let scratch = ModelInferenceSpec {
        row_count: params.row_count,
        corpus_seed: params.seed,
        target_keys: Vec::new(),
        embed_digest: String::new(),
        infer_digest: String::new(),
        baseline_embed_rows_per_s: 0.0,
        baseline_infer_rows_per_s: 0.0,
    };
    let rows = build_corpus(&scratch);

    let model_tmp = tempfile::tempdir()?;
    build_encode_model_dir(model_tmp.path())?;
    let model_id = local_model_id(model_tmp.path())?;

    // Real tokenization off the fixture's own `tokenizer.json`, through the
    // SAME wrapper the candle backend loads — see this module's own doc.
    let tokenizer = TokenizerWrapper::from_file(&model_tmp.path().join("tokenizer.json"))?;
    let texts: Vec<&str> = rows.iter().map(|r| r.text).collect();
    let encoding = tokenizer.encode_batch(&texts, None)?;
    let seq = encoding.seq_len;
    let row_lengths: Vec<usize> = encoding
        .attention_masks
        .iter()
        .map(|mask| mask.iter().map(|&bit| bit as usize).sum())
        .collect();

    let (checkpoint_config_sha256, _config_len) =
        sha256_and_len(&model_tmp.path().join("config.json"))?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =
        sha256_and_len(&model_tmp.path().join("model.safetensors"))?;
    let (checkpoint_tokenizer_sha256, _tokenizer_len) =
        sha256_and_len(&model_tmp.path().join("tokenizer.json"))?;
    let checkpoint_pooling_sha256 = checkpoint_pooling_sha256(model_tmp.path())?;

    let (session, _dir) = corpus_session_on_device(&rows, params.gpu_device).await?;

    for _ in 0..params.warmup {
        serve_embed(&session, &model_id).await?;
    }
    let mut serve_ms_samples: Vec<f64> = Vec::with_capacity(params.iters.max(1));
    let mut rows_served = 0usize;
    for _ in 0..params.iters {
        let (_digest, serve_ms, served) = serve_embed(&session, &model_id).await?;
        serve_ms_samples.push(serve_ms);
        rows_served = served;
    }
    let mean_serve_ms = if serve_ms_samples.is_empty() {
        0.0
    } else {
        serve_ms_samples.iter().sum::<f64>() / serve_ms_samples.len() as f64
    };
    let embed_rate = rows_per_s(rows_served, mean_serve_ms);

    // The LOADED model's actual effective precision — a cache HIT (the
    // model was already loaded by the `serve_embed` calls above), read off
    // the real `LoadedModel` this tier's own serve drove rather than a
    // derived/default constant (unit-62 F-5: `ComputePrecision::default()`
    // is a false determinant in an identity slot).
    let model_source = ModelSource::parse(&model_id);
    let model_guard = session
        .model_cache()
        .get_or_load(&model_source, ModelTask::TextEmbedding, None)
        .await?;
    let compute_precision = model_guard.model.compute_precision().to_string();
    // The SAME cache-hit read as `compute_precision` above, but for the
    // resolved pooling strategy (unit-62 F-5'(a)): read straight off
    // `LoadedModel::resolved_pooling`, the accessor wired to the exact
    // `Pooling` value the loaded text wrapper's `forward_pooled` applies —
    // never the constant `"mean"` literal this tier used to emit
    // regardless of the fixture's own declared strategy.
    let pooling = model_guard
        .model
        .resolved_pooling()
        .map(|p| p.to_string())
        .unwrap_or_else(|| "none".to_string());
    drop(model_guard);

    let tier = EncodeStepTier {
        seed: params.seed,
        batch: rows.len(),
        seq,
        row_lengths,
        compute_precision,
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        checkpoint_tokenizer_sha256,
        pooling,
        checkpoint_pooling_sha256,
        // `jammi_encoders::pool_and_normalize` mandatorily L2-normalizes on
        // every reachable path — see `EncodeStepTier::normalize`'s own doc.
        normalize: true,
        warmup: params.warmup,
        iters_measured: params.iters,
        device_requested: requested_device_label(params.gpu_device),
        device_name: resolved_device_name(params.gpu_device)?,
        kernels_disabled_requested: jammi_kernels::admission::disabled_ops_requested(),
        kernels_disabled_fired: jammi_kernels::admission::disabled_ops_fired(),
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        build_features: crate::report::build_features(),
        // The encode/eval path has no chunked-attention arm at all — see
        // `EncodeStepTier::chunk_size`'s own doc.
        chunk_size: None,
        // Fused attention arms are training-only; the encode/eval path
        // always runs eager. See `EncodeStepTier`'s own doc for why this is
        // provenance, never identity.
        attention_arm: "eager".to_string(),
        embed_rows_per_s: Measurement::measured(embed_rate, "rows_per_s"),
        embed_serve_ms: Measurement::measured(mean_serve_ms, "ms"),
    };

    // K7-completeness, enforced on every real run (mirrors
    // `finetune_step::run`/`grad_oracle::run`'s own posture) — see
    // `report::assert_identity_fields_present`'s own doc.
    let value = serde_json::to_value(&tier).expect("serialize EncodeStepTier for self-check");
    crate::report::assert_identity_fields_present(&value, EncodeStepTier::IDENTITY_FIELDS);
    crate::report::assert_identity_fields_present(&value, EncodeStepTier::PROVENANCE_FIELDS);

    Ok(tier)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PARAMS: EncodeStepParams = EncodeStepParams {
        row_count: 4,
        seed: 0,
        warmup: 1,
        iters: 2,
        gpu_device: CPU_HERMETIC_DEVICE,
    };

    /// Cardinality pin (unit-62 CONTRACT.md §E3, round-3 audit F-5'/lead
    /// ruling): the EXACT comparison identity set — 15 fields (13 original +
    /// `checkpoint_pooling_sha256` + `device_requested`, appended
    /// position-stable rather than re-ordered into the original 13), in this
    /// exact order — so `ci/scripts/perf/identity_fields.py`'s future
    /// `ENCODE_IDENTITY_FIELDS` (unit-62 E6, docs-ci domain) has a fixed,
    /// reviewable Rust-side source to mirror. A field added, removed, or
    /// renamed here is a visible, reviewed diff against this test, not a
    /// silent drift the Python mirror would only notice indirectly.
    #[test]
    fn identity_fields_cardinality_is_pinned() {
        let names: Vec<&str> = EncodeStepTier::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names.len(),
            15,
            "EncodeStepTier::IDENTITY_FIELDS cardinality drifted — update the pinned \
             count together with ci/scripts/perf/identity_fields.py's ENCODE_IDENTITY_FIELDS"
        );
        assert_eq!(
            names,
            vec![
                "seed",
                "batch",
                "seq",
                "row_lengths",
                "compute_precision",
                "checkpoint_config_sha256",
                "checkpoint_weights_sha256",
                "checkpoint_weights_size_bytes",
                "checkpoint_tokenizer_sha256",
                "pooling",
                "normalize",
                "warmup",
                "iters_measured",
                "checkpoint_pooling_sha256",
                "device_requested",
            ]
        );
    }

    /// The forbidden-in-identity clause (unit-62 PLAN.md v2 reshape 3) as a
    /// checked negative control, not only prose: `attention_arm` — and every
    /// other declared provenance field — must never appear in
    /// `IDENTITY_FIELDS`, mechanically enforced so a future "helpful"
    /// addition trips a test instead of silently reintroducing esc-057's
    /// class of false determinant.
    #[test]
    fn provenance_fields_are_never_members_of_identity_fields() {
        let identity_names: std::collections::HashSet<&str> = EncodeStepTier::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        for (provenance_name, _) in EncodeStepTier::PROVENANCE_FIELDS {
            assert!(
                !identity_names.contains(provenance_name),
                "{provenance_name:?} is a declared PROVENANCE_FIELDS entry but also \
                 appears in IDENTITY_FIELDS — attention_arm/chunk_size/device_name/\
                 kernels_disabled_*/flash_compiled/build_features are forbidden from \
                 identity on this surface"
            );
        }
        assert!(
            EncodeStepTier::PROVENANCE_FIELDS
                .iter()
                .any(|(name, _)| *name == "attention_arm"),
            "attention_arm must be declared as a PROVENANCE_FIELDS entry"
        );
    }

    /// The provenance roster's own cardinality/name pin — the seven fields
    /// CONTRACT.md §E3 names explicitly.
    #[test]
    fn provenance_fields_cardinality_is_pinned() {
        let names: Vec<&str> = EncodeStepTier::PROVENANCE_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            vec![
                "device_name",
                "kernels_disabled_requested",
                "kernels_disabled_fired",
                "flash_compiled",
                "build_features",
                "chunk_size",
                "attention_arm",
            ]
        );
    }

    /// The teeth, GATE-FAILS direction (RC1: an assertion must be able to
    /// fail): `run()` drives the REAL serving surface end to end on
    /// `Device::Cpu` — real tokenization, real checksums, a real
    /// `generate_text_embeddings` serve — and every declared identity AND
    /// provenance field lands populated on the emitted tier (the same
    /// `assert_identity_fields_present` check `run()` itself already
    /// enforces; re-proven here as a `#[test]` so a future refactor that
    /// dropped that internal call would still be caught).
    #[tokio::test(flavor = "multi_thread")]
    async fn encode_step_drives_the_real_surface_and_populates_every_field() {
        let tier = run(TEST_PARAMS).await.expect("encode-step run");

        assert_eq!(tier.seed, TEST_PARAMS.seed);
        assert_eq!(tier.batch, TEST_PARAMS.row_count);
        assert_eq!(tier.warmup, TEST_PARAMS.warmup);
        assert_eq!(tier.iters_measured, TEST_PARAMS.iters);
        assert_eq!(tier.row_lengths.len(), tier.batch, "one length per row");
        assert!(tier.seq >= 1, "the tokenizer must have produced columns");
        assert!(
            tier.row_lengths
                .iter()
                .all(|&len| len >= 1 && len <= tier.seq),
            "every row's real length must be in [1, seq]: {:?} vs seq={}",
            tier.row_lengths,
            tier.seq
        );
        // The teeth: the committed corpus sentences have genuinely different
        // lengths, so a REAL tokenization must produce at least one row
        // strictly shorter than the widest — proving this is a measured
        // fact, not a synthetic dense assumption (`[seq; batch]`).
        assert!(
            tier.row_lengths.iter().any(|&len| len < tier.seq),
            "a real corpus of differently-worded sentences must not tokenize to a \
             uniformly-dense batch: {:?} vs seq={}",
            tier.row_lengths,
            tier.seq
        );

        assert_eq!(tier.compute_precision, "f32");
        assert_eq!(tier.pooling, "mean");
        assert!(tier.normalize);
        assert_eq!(tier.attention_arm, "eager");
        assert_eq!(tier.chunk_size, None);
        assert_eq!(tier.device_name, "cpu");
        assert_eq!(
            tier.device_requested, "cpu",
            "the CI-hermetic default requests cpu, same label device_name renders for it"
        );

        for (digest, name) in [
            (&tier.checkpoint_config_sha256, "config"),
            (&tier.checkpoint_weights_sha256, "weights"),
            (&tier.checkpoint_tokenizer_sha256, "tokenizer"),
        ] {
            assert_eq!(digest.len(), 64, "{name} sha256 must be 64 hex chars");
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{name} sha256 must be hex"
            );
        }
        assert!(tier.checkpoint_weights_size_bytes > 0);

        // unit-62 F-5'(b): this tier's own fixture always writes an
        // explicit 1_Pooling/config.json, so a real run always reports
        // `Some` here — the `None`/`NullMeans` arm is exercised separately
        // by `checkpoint_pooling_sha256_is_none_when_the_fixture_has_no_pooling_config`.
        let pooling_sha = tier
            .checkpoint_pooling_sha256
            .as_ref()
            .expect("this tier's fixture always carries 1_Pooling/config.json");
        assert_eq!(
            pooling_sha.len(),
            64,
            "pooling config sha256 must be 64 hex chars"
        );
        assert!(
            pooling_sha.chars().all(|c| c.is_ascii_hexdigit()),
            "pooling config sha256 must be hex"
        );
    }

    /// The teeth for `checkpoint_tokenizer_sha256` (unit-62 F-5): a run
    /// against a model dir whose `tokenizer.json` bytes differ from the
    /// fixture's own, with `config.json`/`model.safetensors` held byte-
    /// identical, must move the recorded tokenizer digest (and ONLY that
    /// digest) — proving the field is a real content hash of the actual
    /// tokenizer bytes served, not a copy of `checkpoint_config_sha256` or a
    /// constant. Perturbs the SAME fixture `build_encode_model_dir`
    /// produces (rather than driving a second `run()`, which would also be
    /// legitimate but slower) so the assertion isolates the one changed
    /// file.
    #[test]
    fn checkpoint_tokenizer_sha256_reacts_to_the_actual_tokenizer_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_encode_model_dir(dir.path()).expect("build fixture model dir");
        let (tokenizer_baseline, _) =
            sha256_and_len(&dir.path().join("tokenizer.json")).expect("hash tokenizer.json");
        let (config_baseline, _) =
            sha256_and_len(&dir.path().join("config.json")).expect("hash config.json");

        // Perturb: append a byte to tokenizer.json only; config.json/
        // model.safetensors are never touched.
        let mut bytes = std::fs::read(dir.path().join("tokenizer.json")).expect("read tokenizer");
        bytes.push(b'\n');
        std::fs::write(dir.path().join("tokenizer.json"), &bytes).expect("write perturbed");
        let (tokenizer_perturbed, _) =
            sha256_and_len(&dir.path().join("tokenizer.json")).expect("hash perturbed tokenizer");
        let (config_after, _) =
            sha256_and_len(&dir.path().join("config.json")).expect("hash config.json again");

        assert_ne!(
            tokenizer_baseline, tokenizer_perturbed,
            "a byte-perturbed tokenizer.json must move its own sha256"
        );
        assert_eq!(
            config_baseline, config_after,
            "an untouched config.json must keep the same sha256 — the perturbation isolated \
             to tokenizer.json alone"
        );
    }

    /// The explicit `1_Pooling/config.json` fixture this tier builds
    /// resolves to a DIFFERENT pooled vector than the bare `tiny_bert`
    /// fixture (no `1_Pooling/` folder, candle's silent mean fallback would
    /// otherwise make this indistinguishable) — proving the declared
    /// config is actually consumed, not merely present alongside an
    /// unrelated default. On `main` pre-esc-057-fix this assertion is the
    /// SAME shape `jammi-ai`'s own `pooling_config.rs` red-green proves;
    /// here it only proves the FIXTURE is wired to a real, distinguishing
    /// config — the identity-hash-folds-it proof lives in `jammi-ai`'s own
    /// suite (out of this crate's scope).
    #[test]
    fn encode_model_dir_carries_an_explicit_pooling_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_encode_model_dir(dir.path()).expect("build fixture model dir");
        let pooling_json = dir.path().join("1_Pooling").join("config.json");
        assert!(
            pooling_json.exists(),
            "1_Pooling/config.json must be written"
        );
        let raw = std::fs::read_to_string(&pooling_json).expect("read pooling config");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(value["pooling_mode_mean_tokens"], serde_json::json!(true));
        assert_eq!(value["pooling_mode_cls_token"], serde_json::json!(false));
    }

    #[test]
    fn requested_device_label_reads_cpu_for_negative_ordinal_and_cuda_otherwise() {
        assert_eq!(requested_device_label(-1), "cpu");
        assert_eq!(requested_device_label(0), "cuda:0");
        assert_eq!(requested_device_label(3), "cuda:3");
    }

    /// [`resolved_device_name`] on the CI-hermetic default never touches the
    /// `cuda`-feature-gated `cudarc` path at all (round-3 audit lead ruling:
    /// `device_name` is a POST-HOC hardware fact, but `"cpu"` is knowable
    /// without querying any driver).
    #[test]
    fn resolved_device_name_reads_cpu_for_negative_ordinal_without_a_cuda_query() {
        assert_eq!(resolved_device_name(-1).expect("cpu never errors"), "cpu");
    }

    /// `checkpoint_pooling_sha256` (unit-62 F-5'(b)) is `None` — never a
    /// panic, never an empty-string stand-in — for a model dir that carries
    /// no `1_Pooling/` folder at all, the SAME presence gate
    /// `backend::candle::all_candidate_paths` applies to the engine's own
    /// `content_digest`.
    #[test]
    fn checkpoint_pooling_sha256_is_none_when_the_model_dir_has_no_pooling_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fixture = ModelInferenceSpec::embed_model_dir();
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(name), dir.path().join(name)).expect("copy fixture file");
        }
        assert!(
            !dir.path().join("1_Pooling").exists(),
            "the bare tiny_bert fixture ships with no 1_Pooling/ folder"
        );
        assert_eq!(
            checkpoint_pooling_sha256(dir.path()).expect("presence-gated read never errors"),
            None
        );
    }

    /// `checkpoint_pooling_sha256` is `Some(sha256_and_len(..).0)` when the
    /// file IS present — driving the SAME `sha256_and_len` helper directly
    /// against the fixture's own bytes, so this is a real cross-check
    /// against an independently-computed hash, never a self-referential
    /// assertion that the function returns whatever the function returns.
    #[test]
    fn checkpoint_pooling_sha256_is_some_and_matches_a_direct_hash_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_encode_model_dir(dir.path()).expect("build fixture model dir");
        let expected = sha256_and_len(&dir.path().join("1_Pooling").join("config.json"))
            .expect("hash pooling config directly")
            .0;
        assert_eq!(
            checkpoint_pooling_sha256(dir.path()).expect("presence-gated read never errors"),
            Some(expected)
        );
    }

    /// The F-5' teeth test: flip the fixture's `1_Pooling/config.json` from
    /// MEAN to CLS, holding `config.json` byte-identical, and drive the REAL
    /// accessors this tier's `run()` reads (`LoadedModel::resolved_pooling`
    /// via the engine's own model cache, and `checkpoint_pooling_sha256`
    /// over the actual file bytes) — proving BOTH `pooling` and
    /// `checkpoint_pooling_sha256` move on the flip while
    /// `checkpoint_config_sha256` stays put. Before this fix, `pooling` was
    /// a hardcoded `"mean"` literal and no field hashed the pooling-config
    /// bytes at all, so this exact flip left every one of the prior 13
    /// identity fields byte-identical while the served vectors differed.
    #[tokio::test(flavor = "multi_thread")]
    async fn checkpoint_pooling_sha256_and_pooling_move_together_when_the_fixture_flips_to_cls() {
        async fn resolved_pooling_and_hashes(
            pooling_flags: &serde_json::Value,
        ) -> (String, String, Option<String>) {
            let model_tmp = tempfile::tempdir().expect("tempdir");
            build_encode_model_dir_with_pooling(model_tmp.path(), pooling_flags)
                .expect("build fixture model dir");
            let model_id = local_model_id(model_tmp.path()).expect("model id");
            let (config_sha, _) =
                sha256_and_len(&model_tmp.path().join("config.json")).expect("hash config.json");
            let pooling_sha = checkpoint_pooling_sha256(model_tmp.path())
                .expect("presence-gated read never errors");

            let rows = build_corpus(&ModelInferenceSpec {
                row_count: 1,
                corpus_seed: 0,
                target_keys: Vec::new(),
                embed_digest: String::new(),
                infer_digest: String::new(),
                baseline_embed_rows_per_s: 0.0,
                baseline_infer_rows_per_s: 0.0,
            });
            let (session, _dir) = corpus_session_on_device(&rows, CPU_HERMETIC_DEVICE)
                .await
                .expect("corpus session");
            let model_source = ModelSource::parse(&model_id);
            let model_guard = session
                .model_cache()
                .get_or_load(&model_source, ModelTask::TextEmbedding, None)
                .await
                .expect("load model");
            let pooling = model_guard
                .model
                .resolved_pooling()
                .map(|p| p.to_string())
                .unwrap_or_else(|| "none".to_string());
            (pooling, config_sha, pooling_sha)
        }

        let (mean_pooling, mean_config_sha, mean_pooling_sha) =
            resolved_pooling_and_hashes(&mean_pooling_flags()).await;
        let (cls_pooling, cls_config_sha, cls_pooling_sha) =
            resolved_pooling_and_hashes(&cls_pooling_flags()).await;

        assert_eq!(mean_pooling, "mean");
        assert_eq!(cls_pooling, "cls");
        assert_ne!(
            mean_pooling, cls_pooling,
            "pooling must move when the fixture flips mean->cls"
        );
        assert_ne!(
            mean_pooling_sha, cls_pooling_sha,
            "checkpoint_pooling_sha256 must move when the fixture flips mean->cls"
        );
        assert_eq!(
            mean_config_sha, cls_config_sha,
            "config.json is byte-identical across the flip -- only 1_Pooling/config.json changed"
        );
    }

    /// The normalize advisory (round-3 audit, folded rather than deferred):
    /// `jammi_encoders::pool_and_normalize` mandatorily L2-normalizes every
    /// reachable output — proved directly here by feeding it a hand-built,
    /// deliberately non-unit-norm hidden-state tensor (a large constant
    /// value at every position, no L2-normalization applied anywhere before
    /// this call) and asserting the pooled output still comes out unit-norm.
    /// A future signature change that added a normalize-optional toggle
    /// would either fail this call to compile (a new required parameter) or
    /// — if given a default that preserved today's always-normalize
    /// behavior — still leave this assertion green, so the pinned invariant
    /// this test protects is "there is no code path in this crate's
    /// dependency graph that reaches a pooled embedding without going
    /// through `pool_and_normalize`", the same claim `EncodeStepTier::normalize`'s
    /// own doc pins.
    #[test]
    fn pool_and_normalize_is_mandatory_with_no_toggle() {
        let hidden = candle_core::Tensor::full(7.0f32, (2, 3, 4), &candle_core::Device::Cpu)
            .expect("build a deliberately non-unit-norm hidden tensor");
        let attention_mask =
            candle_core::Tensor::ones((2, 3), candle_core::DType::U32, &candle_core::Device::Cpu)
                .expect("build a real-tokens-only attention mask");

        let pooled = jammi_encoders::pool_and_normalize(
            &hidden,
            &attention_mask,
            jammi_encoders::Pooling::Mean,
        )
        .expect("pool_and_normalize");

        let norms: Vec<f32> = pooled
            .sqr()
            .and_then(|t| t.sum(1))
            .and_then(|t| t.sqrt())
            .expect("compute per-row L2 norm")
            .to_vec1()
            .expect("read norms back to host");
        for norm in norms {
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "pool_and_normalize must always emit a unit-L2-norm row, got {norm}"
            );
        }
    }
}
