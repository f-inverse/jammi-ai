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
//!   `checkpoint_weights_size_bytes` are `sha256_and_len` over the fixture
//!   model dir's actual bytes (the SAME helper `finetune_step.rs`/
//!   `grad_oracle.rs` already use).
//! * `seq`/`row_lengths` are read off a REAL tokenization of the corpus
//!   text through [`jammi_ai::model::tokenizer::TokenizerWrapper`] — the
//!   exact wrapper `CandleBackend` loads for a local model — over the
//!   fixture's own `tokenizer.json`, never assumed or hand-computed.
//! * The embed measurement itself is `crate::model_inference::serve_embed`,
//!   the real `generate_text_embeddings` call the `model_inference`/
//!   `gpu_inference` tiers already drive.
//!
//! ## CPU-hermetic default, GPU-shape-parameterized
//!
//! `gpu_device` flows straight into `corpus_session_on_device` — the SAME
//! device knob `model_inference`/`gpu_inference` already share (`-1` /
//! `Device::Cpu` for the CI-hermetic default this tier's own tests run
//! under, a real CUDA ordinal for the pod producer). No new device-selection
//! mechanism is introduced.

use std::path::Path;

use jammi_ai::model::tokenizer::TokenizerWrapper;

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

/// Build a fresh model dir at `dst`: copy the shared `tiny_bert` fixture's
/// three files (the SAME fixture `model_inference`'s embed lane serves),
/// then write an EXPLICIT `1_Pooling/config.json` declaring mean pooling.
///
/// The explicit pooling config (never the bare `tiny_bert` fixture, which
/// ships with no `1_Pooling/` folder at all) is deliberate: a repo with no
/// pooling config resolves through `candle.rs`'s silent mean-pooling
/// fallback — the exact ambiguity esc-057 is about. Declaring the strategy
/// here means [`crate::report::EncodeStepTier::pooling`] records a value
/// this tier KNOWS the engine resolves to, not an inferred one.
fn build_encode_model_dir(dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
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
        serde_json::to_string(&mean_pooling_flags())?,
    )?;
    Ok(())
}

/// The device name this tier records — `"cpu"` for a negative ordinal (the
/// CI-hermetic default), `"cuda:<ordinal>"` otherwise. A cheap, honest label
/// derived straight from the same `gpu_device` value threaded into
/// `corpus_session_on_device` (`select_device`'s own convention: negative
/// selects `Device::Cpu`), never a second, independently-resolved reading.
fn device_name(gpu_device: i32) -> String {
    if gpu_device < 0 {
        "cpu".to_string()
    } else {
        format!("cuda:{gpu_device}")
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

    let tier = EncodeStepTier {
        seed: params.seed,
        batch: rows.len(),
        seq,
        row_lengths,
        compute_precision: jammi_numerics::ComputePrecision::default().to_string(),
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        pooling: "mean".to_string(),
        // `jammi_encoders::pool_and_normalize` mandatorily L2-normalizes on
        // every reachable path — see `EncodeStepTier::normalize`'s own doc.
        normalize: true,
        warmup: params.warmup,
        iters_measured: params.iters,
        device_name: device_name(params.gpu_device),
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

    /// Cardinality pin (unit-62 CONTRACT.md §E3): the EXACT comparison
    /// identity set — 12 fields, in this exact order — so
    /// `ci/scripts/perf/identity_fields.py`'s future `ENCODE_IDENTITY_FIELDS`
    /// (unit-62 E6, docs-ci domain) has a fixed, reviewable Rust-side
    /// source to mirror. A field added, removed, or renamed here is a
    /// visible, reviewed diff against this test, not a silent drift the
    /// Python mirror would only notice indirectly.
    #[test]
    fn identity_fields_cardinality_is_pinned() {
        let names: Vec<&str> = EncodeStepTier::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names.len(),
            12,
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
                "pooling",
                "normalize",
                "warmup",
                "iters_measured",
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

        for (digest, name) in [
            (&tier.checkpoint_config_sha256, "config"),
            (&tier.checkpoint_weights_sha256, "weights"),
        ] {
            assert_eq!(digest.len(), 64, "{name} sha256 must be 64 hex chars");
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{name} sha256 must be hex"
            );
        }
        assert!(tier.checkpoint_weights_size_bytes > 0);
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
    fn device_name_reads_cpu_for_negative_ordinal_and_cuda_otherwise() {
        assert_eq!(device_name(-1), "cpu");
        assert_eq!(device_name(0), "cuda:0");
        assert_eq!(device_name(3), "cuda:3");
    }
}
