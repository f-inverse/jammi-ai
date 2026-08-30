//! On-GPU throughput/latency observability — the GPU peer of the CPU-hermetic
//! `model_inference` tier, serving the engine's two real GPU-model verbs,
//! embedding (`generate_text_embeddings`) and classification (`infer`), on a
//! real CUDA device.
//!
//! ## Why this exists next to the CPU-hermetic `model_inference` tier
//!
//! `model_inference` deliberately serves on `Device::Cpu` so it runs inside the
//! hermetic `cargo test` net; it proves byte-determinism and a coarse CPU
//! code-path rate, but says nothing about the real device. This tier is its GPU
//! peer: it serves the same two verbs over the same tiny committed bundles on
//! `gpu.device = 0`, and measures the two quantities a GPU optimization
//! actually moves — sustained throughput (rows/s) and per-serve tail latency
//! (p50/p99). It is gated behind the `cuda` feature and the `live-gpu-tests`
//! lane so default `cargo test` stays hermetic.
//!
//! ## Scope: embedding and classification
//!
//! Both verbs are measured. The embedding forward is the primary GPU workload
//! open optimizations target (the encoder forward, its precision, its
//! attention kernel); the classification (`infer`) lane serves the same
//! `tiny_modernbert_classifier` bundle the `gpu_capability` suite's
//! `classification_parity` cell already establishes as a validated GPU path
//! (CPU↔GPU parity hard-gated there), so measuring its throughput/latency here
//! is no longer premature.
//!
//! ## What is device-independent (hard-gated) vs device-dependent (recorded)
//!
//! An absolute throughput or tail latency is a property of `code × device ×
//! pod-conditions`, not of the code alone: the prove lane runs on an ephemeral
//! heterogeneous rented fleet (SXM4 / PCIe A100s, no pinning), so the same code
//! measures a different rate on every pod it happens to land on. Pinning an
//! absolute floor against that fleet gates pod variance, not a code regression.
//!
//! CPU↔GPU parity *is* a device-independent property of the code, and is
//! hard-gated in the separate `gpu_capability` suite, not here: the served
//! output must fall within tolerance of the CPU path. Cross-repeat
//! determinism is not a hard gate anywhere in this codebase — GPU float
//! bit-equality across repeats is not a property this codebase asserts — so
//! this tier records only the device-dependent quantities: throughput
//! (rows/s), tail latency (p50/p99), and cross-repeat determinism, all
//! *recorded* as observability
//! tagged with the concrete device that produced them, never asserted against a
//! remembered constant.
//!
//! Row conservation IS hard-gated on the classification lane, and it is a
//! correctness property, not a perf one: `infer`'s per-row annotate semantics
//! can silently drop a row whose forward errored, so the lane asserts the
//! scored row count equals the corpus row count on every serve — the same
//! invariant `classification_parity` checks CPU↔GPU, checked here run-to-run
//! on the GPU alone.
//!
//! The designated perf-regression gate for when a GPU-specific optimization
//! lands (fp8, cuDNN / flash-attention, a memory scheduler, batch coalescing) is
//! a **within-run A/B ratio** — parent-HEAD vs the PR change, measured back to
//! back on the *same* pod in the *same* run, so the device and its conditions
//! cancel by construction — not a resurrected absolute baseline.
//!
//! ## Emitted shape
//!
//! `run()` assembles a [`crate::report::GpuInferenceTier`], and the
//! `gpu-inference-scale` subcommand prints it as the `tiers.gpu_inference` field
//! of the one stable JSON [`crate::report::Report`] document on stdout (no other
//! output format). See [`crate::report::GpuInferenceTier`] for the exact field
//! shape and the diffability contract (stable keys, no per-lane timestamps).

use std::error::Error;
use std::path::Path;

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_db::store::manifest::ComputeDevice;

use crate::finetune_step::sha256_and_len;
use crate::model_inference::{
    build_corpus, corpus_session_on_device, local_model_id, rows_per_s, serve_embed,
    serve_infer_all, ModelInferenceSpec, Row,
};
use crate::report::{GpuInferenceTier, GpuLane, Measurement};

/// The CUDA device ordinal the tier serves on. Device 0 is the prove-lane A100.
const GPU_DEVICE: i32 = 0;

/// The generation parameters the tier drives its corpus and measurement off of.
#[derive(Debug, Clone, Copy)]
pub struct GpuInferenceParams {
    /// The synthetic corpus row count.
    pub row_count: usize,
    /// The corpus generation seed — this run's [`GpuInferenceTier::corpus_seed`].
    pub corpus_seed: u64,
    /// Serves discarded before the measured iterations — this run's
    /// [`GpuInferenceTier::warmup`]. Previously an implicit, hardcoded 1 (the
    /// single "first serve" call each lane used to establish its determinism
    /// baseline); now a real, emitted identity field (issue #335) so a
    /// within-run A/B comparator can state both legs discarded the same
    /// number of serves before timing.
    pub warmup: usize,
    /// Measured serves (after warmup).
    pub iters: usize,
}

/// sha256 (hex) of `dir`'s three checkpoint files, `(config, weights,
/// tokenizer)` — the SAME `sha256_and_len` helper
/// `finetune_step`/`encode_step`/`grad_oracle` already use, never a second,
/// independently-drifting hashing implementation. The weights' byte length
/// (this helper's second return value) is deliberately dropped here: unlike
/// [`crate::report::EncodeStepTier::checkpoint_weights_size_bytes`], this
/// tier admits only the three content hashes to identity (see
/// [`GpuInferenceTier`]'s own doc) — a byte count is redundant with a sha256
/// of the same file for detecting a content change.
fn checkpoint_hashes(dir: &Path) -> Result<(String, String, String), Box<dyn Error>> {
    let (config, _) = sha256_and_len(&dir.join("config.json"))?;
    let (weights, _) = sha256_and_len(&dir.join("model.safetensors"))?;
    let (tokenizer, _) = sha256_and_len(&dir.join("tokenizer.json"))?;
    Ok((config, weights, tokenizer))
}

/// The p50 and p99 of a set of per-serve latencies (ms), by nearest-rank on the
/// sorted samples. `p99` of a short sample is its slowest serve — the honest tail
/// for the small `iters` a coarse net runs.
fn percentiles_ms(mut latencies_ms: Vec<f64>) -> (f64, f64) {
    latencies_ms.sort_by(|a, b| a.total_cmp(b));
    let n = latencies_ms.len();
    let rank = |p: f64| -> usize {
        // Nearest-rank: ceil(p · n) clamped into [1, n], then 0-indexed.
        (((p * n as f64).ceil() as usize).clamp(1, n)) - 1
    };
    (latencies_ms[rank(0.50)], latencies_ms[rank(0.99)])
}

/// Serve the embed verb `warmup + iters` times on the GPU session, returning the
/// measured lane (throughput + p50/p99 + the determinism verdict). Recorded
/// observability only — no gate, per the module docs.
///
/// `warmup` serves are executed and their results entirely discarded (not
/// folded into the determinism baseline or any latency sample) — issue #335
/// made this an explicit, caller-controlled count (previously a hardcoded
/// implicit 1, whose own digest doubled as the determinism baseline). The
/// FIRST of the `iters` MEASURED serves is the determinism baseline instead;
/// every measured serve, including that first one, is folded into
/// `latencies_ms`.
async fn measure_embed_lane(
    session: &std::sync::Arc<jammi_ai::session::InferenceSession>,
    model_id: &str,
    warmup: usize,
    iters: usize,
) -> Result<GpuLane, Box<dyn Error>> {
    for _ in 0..warmup {
        serve_embed(session, model_id).await?;
    }

    let mut latencies_ms = Vec::with_capacity(iters);
    let mut deterministic = true;
    let mut first_digest: Option<String> = None;
    let mut rows = 0usize;
    for _ in 0..iters {
        let (digest, serve_ms, served_rows) = serve_embed(session, model_id).await?;
        rows = served_rows;
        match &first_digest {
            None => first_digest = Some(digest),
            Some(baseline) if *baseline != digest => deterministic = false,
            Some(_) => {}
        }
        latencies_ms.push(serve_ms);
    }

    let (p50_ms, p99_ms) = percentiles_ms(latencies_ms);
    // Throughput at the median serve — the representative steady-state rate.
    let rate = rows_per_s(rows, p50_ms);

    Ok(GpuLane {
        rows,
        rows_per_s: Measurement::measured(rate, "rows_per_s"),
        p50_ms: Measurement::measured(p50_ms, "ms"),
        p99_ms: Measurement::measured(p99_ms, "ms"),
        deterministic,
    })
}

/// Assert a classification serve scored every corpus row — row conservation is
/// a correctness property, not a perf one. `infer`'s per-row annotate semantics
/// silently drop a row whose forward errored (the RoPE-contiguity bug
/// `gpu_capability`'s `classification_parity` regression-guards CPU↔GPU), so a
/// scored count short of the corpus size is real data loss and must fail the
/// tier loudly rather than ride as a smaller-but-quietly-accepted rate.
fn assert_row_conservation(scored_rows: usize, expected_rows: usize) -> Result<(), Box<dyn Error>> {
    if scored_rows != expected_rows {
        return Err(format!(
            "gpu-inference classification lane scored {scored_rows} rows but the corpus has \
             {expected_rows} — a per-row forward failure silently dropped a row on GPU"
        )
        .into());
    }
    Ok(())
}

/// Serve the classification (`infer`) verb `warmup + iters` times on the GPU
/// session, returning the measured lane. Mirrors [`measure_embed_lane`]'s
/// throughput/latency/determinism measurement (including its issue #335
/// caller-controlled `warmup` count and "first MEASURED serve is the
/// determinism baseline" convention), plus one hard gate
/// [`measure_embed_lane`] has no need for: every serve, including warmup, must
/// score exactly `expected_rows` ([`assert_row_conservation`]) — the same
/// row-conservation invariant `classification_parity` checks CPU↔GPU, checked
/// here run-to-run on the GPU alone.
async fn measure_infer_lane(
    session: &std::sync::Arc<jammi_ai::session::InferenceSession>,
    model_id: &str,
    expected_rows: usize,
    warmup: usize,
    iters: usize,
) -> Result<GpuLane, Box<dyn Error>> {
    for _ in 0..warmup {
        let (_digest, _serve_ms, warm_rows) = serve_infer_all(session, model_id).await?;
        assert_row_conservation(warm_rows, expected_rows)?;
    }

    let mut latencies_ms = Vec::with_capacity(iters);
    let mut deterministic = true;
    let mut first_digest: Option<String> = None;
    let mut rows = 0usize;
    for _ in 0..iters {
        let (digest, serve_ms, scored_rows) = serve_infer_all(session, model_id).await?;
        assert_row_conservation(scored_rows, expected_rows)?;
        rows = scored_rows;
        match &first_digest {
            None => first_digest = Some(digest),
            Some(baseline) if *baseline != digest => deterministic = false,
            Some(_) => {}
        }
        latencies_ms.push(serve_ms);
    }

    let (p50_ms, p99_ms) = percentiles_ms(latencies_ms);
    // Throughput at the median serve — the representative steady-state rate.
    let rate = rows_per_s(rows, p50_ms);

    Ok(GpuLane {
        rows,
        rows_per_s: Measurement::measured(rate, "rows_per_s"),
        p50_ms: Measurement::measured(p50_ms, "ms"),
        p99_ms: Measurement::measured(p99_ms, "ms"),
        deterministic,
    })
}

/// Assert the session actually resolved to a CUDA device, so a CPU-fallback
/// (a non-`cuda` build, or a machine with no GPU) fails the tier loudly instead
/// of silently recording CPU numbers on the GPU lane.
fn require_cuda(device: ComputeDevice) -> Result<u32, Box<dyn Error>> {
    match device {
        ComputeDevice::Cuda { ordinal } => Ok(ordinal),
        other => Err(format!(
            "gpu-inference requires a CUDA device but the session resolved to {other:?}; \
             build jammi-bench with --features cuda and run on a GPU"
        )
        .into()),
    }
}

/// The concrete CUDA device sub-class name (e.g. `NVIDIA A100-SXM4-80GB`) for
/// the ordinal the tier serves on — the provenance tag that makes a recorded
/// rate/latency interpretable on an ephemeral heterogeneous fleet where the same
/// code measures a different number on every pod it lands on.
///
/// Queried in-process through cudarc's device API (candle re-exports cudarc as
/// `candle_core::cuda::cudarc`) rather than shelling out: it is the same driver
/// handle the session's own CUDA backend already opened, needs no subprocess,
/// and cannot silently return another host's GPU.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_device_name(ordinal: u32) -> Result<String, Box<dyn Error>> {
    use candle_core::cuda::cudarc::driver::result as cuda;
    cuda::init()?;
    let device = cuda::device::get(ordinal as i32)?;
    Ok(cuda::device::get_name(device)?)
}

/// Non-`cuda`-feature stub: unreachable in practice, since [`require_cuda`]
/// already fails the tier before this is ever called on a build with no CUDA
/// backend compiled in. Exists only so the tier compiles in the default
/// CPU-hermetic build. Also reused by `encode_step::resolved_device_name`
/// (unit-62 round-3 audit: `EncodeStepTier::device_name` now queries the
/// SAME real hardware string on a `--cuda` leg, never a second,
/// independently-drifting lookup).
#[cfg(not(feature = "cuda"))]
pub(crate) fn cuda_device_name(_ordinal: u32) -> Result<String, Box<dyn Error>> {
    Err("gpu-inference built without the cuda feature; no device to name".into())
}

/// Run the GPU-inference tier: serve the embed and classification verbs on
/// `gpu.device = 0` and record throughput, p50/p99 tail latency, and
/// cross-repeat determinism for each, tagged with the concrete device that
/// served them. The classification lane additionally hard-gates row
/// conservation (see [`measure_infer_lane`]).
///
/// Every declared [`GpuInferenceTier::IDENTITY_FIELDS`]/
/// [`GpuInferenceTier::PROVENANCE_FIELDS`] entry is asserted present on the
/// assembled tier before it is returned (issue #335's D4/K7-completeness
/// contract) — the SAME `assert_identity_fields_present` self-check
/// [`crate::encode_step::run`]/[`crate::finetune_step::run`] already enforce
/// on every real invocation.
pub async fn run(params: GpuInferenceParams) -> Result<GpuInferenceTier, Box<dyn Error>> {
    let rows = build_corpus_from(params.row_count, params.corpus_seed);
    let (session, _dir) = corpus_session_on_device(&rows, GPU_DEVICE).await?;
    let ordinal = require_cuda(session.compute_device())?;
    let device_name = cuda_device_name(ordinal)?;

    let embed_dir = ModelInferenceSpec::embed_model_dir();
    let embed_id = local_model_id(&embed_dir)?;
    let (
        embed_checkpoint_config_sha256,
        embed_checkpoint_weights_sha256,
        embed_checkpoint_tokenizer_sha256,
    ) = checkpoint_hashes(&embed_dir)?;
    let embed = measure_embed_lane(&session, &embed_id, params.warmup, params.iters).await?;

    let infer_dir = ModelInferenceSpec::classifier_model_dir();
    let infer_id = local_model_id(&infer_dir)?;
    let (
        infer_checkpoint_config_sha256,
        infer_checkpoint_weights_sha256,
        infer_checkpoint_tokenizer_sha256,
    ) = checkpoint_hashes(&infer_dir)?;
    let infer = measure_infer_lane(
        &session,
        &infer_id,
        params.row_count,
        params.warmup,
        params.iters,
    )
    .await?;

    // The embed bundle's actually-resolved precision, read off the real
    // `LoadedModel` rather than a derived/default constant (mirrors
    // `encode_step::run`'s own read of the same accessor, unit-62 F-5).
    // This call resolves to the SAME cache key `measure_embed_lane` above
    // already populated (`ModelSource::parse(&embed_id)` +
    // `ModelTask::TextEmbedding` + `None` backend hint — the exact tuple
    // `EmbeddingPipeline::run`'s own `get_or_load` call resolves
    // `serve_embed`'s `generate_text_embeddings` through, `pipeline/
    // embedding.rs`), so this is a cache HIT on any real invocation, not a
    // second cold load — round-1 adversarial audit advisory: verified by
    // reading both call sites' cache-key inputs, not merely asserted. The
    // one theoretical exception (eviction under real memory pressure
    // between the two calls) is not something this small, single-model
    // tier is expected to hit in practice, but the call is correct either
    // way — a cold `get_or_load` still returns the SAME real, resolved
    // `LoadedModel`, just paying a reload cost this comment does not
    // depend on for correctness.
    let model_source = ModelSource::parse(&embed_id);
    let model_guard = session
        .model_cache()
        .get_or_load(&model_source, ModelTask::TextEmbedding, None)
        .await?;
    let compute_precision = model_guard.model.compute_precision().to_string();
    drop(model_guard);

    let tier = GpuInferenceTier {
        device: format!("cuda:{ordinal}"),
        device_name,
        corpus_seed: params.corpus_seed,
        row_count: params.row_count,
        warmup: params.warmup,
        iters: params.iters,
        corpus_sha256: crate::model_inference::corpus_sha256(params.corpus_seed, params.row_count),
        compute_precision,
        embed_checkpoint_config_sha256,
        embed_checkpoint_weights_sha256,
        embed_checkpoint_tokenizer_sha256,
        infer_checkpoint_config_sha256,
        infer_checkpoint_weights_sha256,
        infer_checkpoint_tokenizer_sha256,
        kernels_disabled_requested: jammi_kernels::admission::disabled_ops_requested(),
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        build_features: crate::report::build_features(),
        embed,
        infer,
    };

    let value = serde_json::to_value(&tier).expect("serialize GpuInferenceTier for self-check");
    crate::report::assert_identity_fields_present(&value, GpuInferenceTier::IDENTITY_FIELDS);
    crate::report::assert_identity_fields_present(&value, GpuInferenceTier::PROVENANCE_FIELDS);

    Ok(tier)
}

/// Build the corpus for a `(row_count, seed)` without a full spec — the tier
/// draws only these two fields off [`ModelInferenceSpec`], so a throwaway spec is
/// the honest way to reuse the one corpus generator (no second copy of the
/// seed-rotation logic).
fn build_corpus_from(row_count: usize, corpus_seed: u64) -> Vec<Row> {
    let scratch = ModelInferenceSpec {
        row_count,
        corpus_seed,
        target_keys: Vec::new(),
        embed_digest: String::new(),
        infer_digest: String::new(),
        baseline_embed_rows_per_s: 0.0,
        baseline_infer_rows_per_s: 0.0,
    };
    build_corpus(&scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_pick_median_and_tail() {
        let (p50, p99) = percentiles_ms(vec![10.0, 20.0, 30.0, 40.0, 100.0]);
        assert_eq!(p50, 30.0, "median of five is the third");
        assert_eq!(p99, 100.0, "p99 of a short sample is its slowest serve");
    }

    #[test]
    fn percentiles_single_sample() {
        let (p50, p99) = percentiles_ms(vec![42.0]);
        assert_eq!(p50, 42.0);
        assert_eq!(p99, 42.0);
    }

    /// A JSON fixture carrying every [`GpuInferenceTier::IDENTITY_FIELDS`]/
    /// [`GpuInferenceTier::PROVENANCE_FIELDS`] entry, correctly populated —
    /// the flat `{field: value}` shape [`crate::report::assert_identity_fields_present`]
    /// reads. Hermetic (no CUDA/GPU): this is a hand-built stand-in for a
    /// real tier's serialized shape, not a live `run()`.
    fn identity_complete_fixture() -> serde_json::Value {
        serde_json::json!({
            "corpus_seed": 0,
            "row_count": 256,
            "warmup": 2,
            "iters": 20,
            "corpus_sha256": "0".repeat(64),
            "compute_precision": "f32",
            "embed_checkpoint_config_sha256": "a".repeat(64),
            "embed_checkpoint_weights_sha256": "b".repeat(64),
            "embed_checkpoint_tokenizer_sha256": "c".repeat(64),
            "infer_checkpoint_config_sha256": "d".repeat(64),
            "infer_checkpoint_weights_sha256": "e".repeat(64),
            "infer_checkpoint_tokenizer_sha256": "f".repeat(64),
            "device_name": "NVIDIA A100-SXM4-80GB",
            "kernels_disabled_requested": Vec::<String>::new(),
            "flash_compiled": true,
            "build_features": ["cuda"],
        })
    }

    /// Cardinality pin (issue #335 D4): the EXACT identity set this tier
    /// declares, in this exact order — `ci/scripts/perf/identity_fields.py`'s
    /// `GPU_INFERENCE_IDENTITY_FIELDS` mirrors this list EXACTLY. A field
    /// added, removed, or renamed here is a visible, reviewed diff against
    /// this test. 12 entries (round-1 adversarial audit B1's completeness
    /// fold-in: `row_count`/`iters`/`corpus_sha256` added to the original 9
    /// — `row_count` closes the manufactured-2x attack, `iters` closes an
    /// already-emitted-but-uncompared field, `corpus_sha256` closes the
    /// "reworded sentence, same seed/row_count" gap).
    #[test]
    fn identity_fields_cardinality_is_pinned() {
        let names: Vec<&str> = GpuInferenceTier::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            vec![
                "corpus_seed",
                "row_count",
                "warmup",
                "iters",
                "corpus_sha256",
                "compute_precision",
                "embed_checkpoint_config_sha256",
                "embed_checkpoint_weights_sha256",
                "embed_checkpoint_tokenizer_sha256",
                "infer_checkpoint_config_sha256",
                "infer_checkpoint_weights_sha256",
                "infer_checkpoint_tokenizer_sha256",
            ]
        );
    }

    /// The disjointness negative control (mirrors
    /// `encode_step::tests::provenance_fields_are_never_members_of_identity_fields`):
    /// no [`GpuInferenceTier::PROVENANCE_FIELDS`] entry may ever also appear
    /// in [`GpuInferenceTier::IDENTITY_FIELDS`] — a future "helpful" addition
    /// that reintroduces a post-hoc/build-only fact as a comparison key trips
    /// this test rather than silently reintroducing esc-057's class of false
    /// determinant.
    #[test]
    fn provenance_fields_are_never_members_of_identity_fields() {
        let identity_names: std::collections::HashSet<&str> = GpuInferenceTier::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        for (provenance_name, _) in GpuInferenceTier::PROVENANCE_FIELDS {
            assert!(
                !identity_names.contains(provenance_name),
                "{provenance_name:?} is a declared PROVENANCE_FIELDS entry but also appears in \
                 IDENTITY_FIELDS"
            );
        }
    }

    /// The positive control: a fixture carrying every declared identity AND
    /// provenance field, correctly populated, passes
    /// `assert_identity_fields_present` for both consts — proves
    /// [`identity_complete_fixture`] itself is a faithful stand-in before the
    /// RED teeth test below relies on it.
    #[test]
    fn identity_complete_fixture_passes_the_assertion() {
        let value = identity_complete_fixture();
        crate::report::assert_identity_fields_present(&value, GpuInferenceTier::IDENTITY_FIELDS);
        crate::report::assert_identity_fields_present(&value, GpuInferenceTier::PROVENANCE_FIELDS);
    }

    /// THE TEETH (RED direction — an assertion must be able to fail): removing
    /// ANY single declared [`GpuInferenceTier::IDENTITY_FIELDS`] entry from an
    /// otherwise-complete fixture must panic `assert_identity_fields_present`
    /// — proving the D4 identity-completeness self-check `run()` performs on
    /// every real invocation actually bites, rather than vacuously passing
    /// regardless of what the tier serializes. Swept over every declared
    /// field, not just one, so a future field addition is automatically
    /// covered by this same sweep.
    #[test]
    fn removing_any_identity_field_from_the_fixture_panics_the_assertion() {
        for (field, _) in GpuInferenceTier::IDENTITY_FIELDS {
            let mut value = identity_complete_fixture();
            value
                .as_object_mut()
                .expect("fixture is a JSON object")
                .remove(*field);
            let result = std::panic::catch_unwind(|| {
                crate::report::assert_identity_fields_present(
                    &value,
                    GpuInferenceTier::IDENTITY_FIELDS,
                );
            });
            assert!(
                result.is_err(),
                "removing identity field {field:?} from the fixture must panic \
                 assert_identity_fields_present — it did not"
            );
        }
    }

    /// The provenance twin of the identity teeth test above: removing ANY
    /// declared [`GpuInferenceTier::PROVENANCE_FIELDS`] entry must also
    /// panic — provenance fields carry the SAME presence/non-null guarantee
    /// as identity fields (see that const's own doc), just never as a
    /// comparison key.
    #[test]
    fn removing_any_provenance_field_from_the_fixture_panics_the_assertion() {
        for (field, _) in GpuInferenceTier::PROVENANCE_FIELDS {
            let mut value = identity_complete_fixture();
            value
                .as_object_mut()
                .expect("fixture is a JSON object")
                .remove(*field);
            let result = std::panic::catch_unwind(|| {
                crate::report::assert_identity_fields_present(
                    &value,
                    GpuInferenceTier::PROVENANCE_FIELDS,
                );
            });
            assert!(
                result.is_err(),
                "removing provenance field {field:?} from the fixture must panic \
                 assert_identity_fields_present — it did not"
            );
        }
    }
}
