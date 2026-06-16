//! The CPU-hermetic model-inference tier: the engine's GPU-model serving verbs
//! [`generate_text_embeddings`](InferenceSession::generate_text_embeddings) (the
//! `generate_embeddings` path) and [`infer`](InferenceSession::infer), driven on
//! `Device::Cpu` over a TINY committed model bundle so the regression net runs
//! in `cargo test` with no HuggingFace download and no network.
//!
//! This is the inference-serving peer of [`crate::context_predictor`]: a portable
//! committed *digest* on one side, a same-box serving *rate* on the other. It
//! covers the two remaining scale-relevant model verbs, each over its own tiny
//! bundle and its own real engine path.
//!
//! ## The A/B split — why CI gates a tiny-model digest, not the scaling SLO
//!
//! These are GPU-model inference rates, and the representative full-scale rate —
//! rows/s through a production-size embedding/classification model on a GPU — is
//! not a number a CPU runner can produce. So the cell is split:
//!
//! * **(A) the representative full-scale rate** is captured off-box in the
//!   cookbook against a real model on a real device. That number is the scaling
//!   SLO; it does NOT live here and is not produced by this tier. This tier only
//!   documents the seam.
//! * **(B) the CPU-hermetic gate** — this module — drives the *same engine fns*
//!   over a tiny committed bundle on `Device::Cpu`. It is the portable
//!   code-path-regression net: a regression in the resolve → tokenize → forward →
//!   adapt path moves the committed digest and trips `cargo test`.
//!
//! The two lanes here mirror that priority:
//!
//! 1. **The determinism digest (the cell-(d) anchor)** — the engine's serving
//!    path is byte-deterministic on `Device::Cpu` over a fixed model and fixed
//!    inputs (the engine's own
//!    `embedding_vectors_are_semantically_meaningful_and_reproducible` contract:
//!    encoding the same text twice yields identical vectors). So a committed
//!    checksum of the served output — the embedding vectors for the embed verb,
//!    the full per-row score distributions for the infer verb — gates for
//!    equality. A regression in the forward, the pooling, the tokenizer
//!    dispatch, the label mapping, or the output adapter moves the bits and trips
//!    the gate; a perturbed input or a different model moves it too (the teeth).
//! 2. **The serving throughput (coarse code-path net, NOT the SLO)** — rows/s the
//!    tiny model serves through the real verb on this box, gated against a
//!    committed same-box baseline by [`crate::rate_gate`]. This catches a *code
//!    path* regression (lost batching, an accidental per-row model reload, a
//!    dropped fast path) — it is emphatically NOT the scaling number, which is
//!    the cookbook (A) value over a real model. The rustdoc and the report both
//!    say so explicitly so the cell is not over-claimed.
//!
//! ## Why two bundles, two verbs
//!
//! `generate_embeddings` and `infer` are distinct engine paths: the embed verb
//! pools encoder hidden states into a vector and persists a result table; the
//! infer verb (here in its `Classification` task) runs a classifier head and
//! adapts a label + score distribution. Driving each over its own tiny bundle
//! exercises the two real paths, so a regression in either moves its own digest.
//!
//! ## The committed artifact bundles
//!
//! `baselines/model_inference.json` carries the synthetic corpus spec (so the
//! gate regenerates the exact deterministic text rows), the infer digest's target
//! keys (a deterministic prefix, folded in key order so any box folds the same
//! per-row scores), the two digests, and the two same-box baselines. The tiny
//! model bundles
//! (`config.json` + `model.safetensors` + `tokenizer.json`, ~100 KB each) live
//! beside it under `baselines/embed_model/` and `baselines/classifier_model/`.
//! They are the engine's own committed tiny fixtures (a 32-dim 1-layer BERT and a
//! 32-dim 1-layer ModernBERT classifier); the rebuild re-derives every committed
//! value by serving over those bundles and copies them in from the engine's
//! fixture tree so the bench is self-contained.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};

use crate::report::{DigestGate, Measurement, ModelInferenceTier, RateVerdict};

/// The source id the synthetic corpus registers under. Generic — names no
/// consumer; the corpus is a neutral family of short factual sentences.
const SOURCE_ID: &str = "corpus";
/// The text column the verbs read.
const TEXT_COLUMN: &str = "text";
/// The key column carrying each row's stable id into the result table.
const KEY_COLUMN: &str = "_row_id";

/// The deterministic synthetic corpus: a fixed family of short sentences across
/// two topics. Fixed (not random) text because the real model tokenizes and
/// embeds it — a random byte soup would tokenize to mostly the pad/unk token and
/// make the forward pass trivial (the vacuity trap). These are real words the
/// tiny tokenizer splits into distinct tokens, so the forward actually attends
/// over varied input and the digest reflects a non-trivial serve.
///
/// Indexed deterministically by `(seed-rotated index mod len)` so the corpus
/// shape (`row_count`) is a spec knob while the sentences stay a fixed,
/// reviewable set.
const SENTENCES: [&str; 8] = [
    "quantum computing in superconducting systems",
    "topological quantum error correction codes",
    "spin coherence in trapped ion qubits",
    "lattice gauge theory on a quantum simulator",
    "crispr gene editing for inherited disease",
    "ribosome structure and protein synthesis",
    "mitochondrial dna and cellular respiration",
    "enzyme kinetics in metabolic pathways",
];

/// The committed model-inference spec: the corpus generation parameters, the
/// committed target keys, the two served digests, and the two same-box serving
/// baselines. The on-disk `baselines/model_inference.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInferenceSpec {
    /// The synthetic corpus row count — the gate regenerates exactly this many
    /// deterministic rows, so every box serves the same input set.
    pub row_count: usize,
    /// The corpus generation seed (rotates which sentence each row draws).
    pub corpus_seed: u64,
    /// The infer digest's target row keys, in fold order — a deterministic prefix
    /// of the corpus rows, so any box folds the same per-row score distributions
    /// in the same order regardless of the served batch shape. (The embed digest
    /// folds the whole persisted `vector` column in storage order and needs no
    /// key list.)
    pub target_keys: Vec<String>,
    /// The committed embed digest: the checksum the engine's
    /// `generate_text_embeddings` produced over the corpus and the committed embed
    /// bundle when the spec was cut, folded over the persisted vector column.
    pub embed_digest: String,
    /// The committed infer digest: the checksum the engine's `infer`
    /// (`Classification`) produced over the committed targets and the committed
    /// classifier bundle — folded over the full per-row score distributions.
    pub infer_digest: String,
    /// The committed same-box embed serving baseline, rows/s. A coarse
    /// code-path-regression reference over the tiny model — NOT the full-scale
    /// SLO, which is the cookbook (A) value.
    pub baseline_embed_rows_per_s: f64,
    /// The committed same-box infer serving baseline, rows/s. Same coarse
    /// code-path net, not the SLO.
    pub baseline_infer_rows_per_s: f64,
}

impl ModelInferenceSpec {
    /// The crate-relative path to the committed spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("model_inference.json")
    }

    /// The crate-relative directory the committed embed bundle lives in.
    pub fn embed_model_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("embed_model")
    }

    /// The crate-relative directory the committed classifier bundle lives in.
    pub fn classifier_model_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("classifier_model")
    }

    /// Load the committed spec from `baselines/model_inference.json`.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(Self::path())?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// A `local:`-prefixed model id for a committed bundle directory — the resolver's
/// no-network branch (`ModelSource::Local`), so the serve loads off the committed
/// safetensors and never reaches HuggingFace.
fn local_model_id(dir: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!(
        "local:{}",
        dir.to_str()
            .ok_or("committed model dir is not valid UTF-8")?
    ))
}

/// One synthetic row: its stable key and its text.
struct Row {
    id: String,
    text: &'static str,
}

/// Build the deterministic synthetic corpus: `row_count` rows, each drawing a
/// fixed sentence by a seed-rotated index so the corpus is byte-stable from the
/// committed seed.
fn build_corpus(spec: &ModelInferenceSpec) -> Vec<Row> {
    (0..spec.row_count)
        .map(|i| {
            let idx = (spec.corpus_seed as usize).wrapping_add(i) % SENTENCES.len();
            Row {
                id: format!("row_{i}"),
                text: SENTENCES[idx],
            }
        })
        .collect()
}

/// Stand up a hermetic `Device::Cpu` session over the synthetic corpus: write the
/// source parquet (`_row_id`, `text`, and a `y` placeholder so the schema matches
/// the engine's source-table shape) and register it. Holds the
/// [`tempfile::TempDir`] so the artifacts outlive the session.
///
/// `gpu.device = -1` forces `Device::Cpu` (the engine's `select_device`), so the
/// serve runs CPU-only with no GPU and no network.
async fn corpus_session(
    rows: &[Row],
) -> Result<(Arc<InferenceSession>, tempfile::TempDir), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let config = JammiConfig {
        artifact_dir: dir.path().to_path_buf(),
        gpu: GpuConfig {
            device: -1,
            ..Default::default()
        },
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await?);
    session.register_query_functions();

    let schema = Arc::new(Schema::new(vec![
        Field::new(KEY_COLUMN, DataType::Utf8, false),
        Field::new(TEXT_COLUMN, DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let texts: Vec<&str> = rows.iter().map(|r| r.text).collect();
    let ys: Vec<f64> = (0..rows.len()).map(|i| i as f64).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(texts)),
            Arc::new(Float64Array::from(ys)),
        ],
    )?;
    let path = dir.path().join("source.parquet");
    let url = StorageUrl::parse(path.to_str().ok_or("source path is not valid UTF-8")?)?;
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None)?;
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema)).await?;
    writer.write_batch(&batch).await?;
    writer.close().await?;
    session
        .add_source(
            SOURCE_ID,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", path.to_str().unwrap())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;

    Ok((session, dir))
}

/// FNV-1a over a byte stream — the same stable, crate-free checksum the
/// propagation and context-predictor tiers fold with, rendered as fixed-width
/// hex. Pure arithmetic; any change to the served bytes flips it.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn mix(&mut self, byte: u8) {
        self.0 ^= byte as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn mix_f32(&mut self, value: f32) {
        for b in value.to_bits().to_le_bytes() {
            self.mix(b);
        }
    }
    fn mix_str(&mut self, value: &str) {
        for b in value.as_bytes() {
            self.mix(*b);
        }
    }
    fn finish(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Serve the embed verb over the corpus through the engine's real
/// `generate_text_embeddings` (the `generate_embeddings` path), read the
/// persisted vectors back through the public `read_vectors`, and fold the vector
/// floats into an FNV digest.
///
/// The fold is over the full persisted `vector` column in result-table storage
/// order, which is the embedding pipeline's deterministic source-scan write order
/// — so the digest is reload-invariant (the `digests_are_reload_invariant` test
/// proves it). A length tag per vector makes the row boundaries explicit so a
/// dimensionality change cannot be masked by a float collision.
///
/// Returns `(digest, serve_wall_ms, rows_served)`. The serve wall-time is over
/// the whole `generate_text_embeddings` call (the rate's numerator-time); the
/// read-back is the deterministic fold, not part of the measured serve.
///
/// `model_id` selects the bundle the serve resolves — the committed embed bundle
/// for the gate, a different bundle for the teeth test.
async fn serve_embed(
    session: &Arc<InferenceSession>,
    model_id: &str,
) -> Result<(String, f64, usize), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let table = session
        .generate_text_embeddings(SOURCE_ID, model_id, &[TEXT_COLUMN.to_string()], KEY_COLUMN)
        .await?;
    let serve_ms = start.elapsed().as_secs_f64() * 1_000.0;

    let vectors = session.read_vectors(&table).await?;
    let mut fnv = Fnv::new();
    for vector in &vectors {
        for v in vector {
            fnv.mix_f32(*v);
        }
        fnv.mix(vector.len() as u8); // row boundary + dimensionality tag
    }
    Ok((fnv.finish(), serve_ms, vectors.len()))
}

/// Serve the infer verb over the corpus through the engine's real `infer`
/// (`Classification`), then fold the full per-row score distribution
/// (`all_scores_json`) of each committed target, in committed-target order, into
/// an FNV digest.
///
/// The digest is over `all_scores_json` rather than the bare `label`, because the
/// score distribution is the classifier head's full output — a regression in the
/// forward, the pooling, or the softmax moves a score even when the argmax label
/// is unchanged, so this is the regression-sensitive quantity. Returns `(digest,
/// serve_wall_ms, rows_served)`.
async fn serve_infer(
    session: &Arc<InferenceSession>,
    model_id: &str,
    target_keys: &[String],
) -> Result<(String, f64, usize), Box<dyn std::error::Error>> {
    let source = ModelSource::parse(model_id);
    let start = Instant::now();
    let batches = session
        .infer(
            SOURCE_ID,
            &source,
            ModelTask::Classification,
            &[TEXT_COLUMN.to_string()],
            KEY_COLUMN,
        )
        .await?;
    let serve_ms = start.elapsed().as_secs_f64() * 1_000.0;

    // Index `_row_id -> all_scores_json` across the served batches so the fold
    // walks the committed targets in committed order, independent of batch shape.
    let mut by_key: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("_row_id")
            .ok_or("infer output missing _row_id")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("_row_id is not a string column")?;
        let scores = batch
            .column_by_name("all_scores_json")
            .ok_or("infer output missing all_scores_json")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("all_scores_json is not a string column")?;
        for i in 0..batch.num_rows() {
            if !ids.is_null(i) && !scores.is_null(i) {
                by_key.insert(ids.value(i).to_string(), scores.value(i).to_string());
            }
        }
    }

    let mut fnv = Fnv::new();
    for key in target_keys {
        let scores = by_key
            .get(key)
            .ok_or_else(|| format!("infer produced no scores for target key {key}"))?;
        fnv.mix_str(scores);
        fnv.mix(0x00); // key boundary
    }
    Ok((fnv.finish(), serve_ms, target_keys.len()))
}

/// A rows/s rate, or `0.0` when the serve was instantaneous (a degenerate
/// measurement the rate gate fails closed on rather than dividing by zero).
fn rows_per_s(rows: usize, wall_ms: f64) -> f64 {
    if wall_ms > 0.0 {
        rows as f64 / (wall_ms / 1_000.0)
    } else {
        0.0
    }
}

/// Re-derive a committed digest digest-only (dropping the serve timing): stand up
/// the corpus session and serve the named verb over the named bundle. The path
/// the digest gates re-run, with `model_dir` explicit so a teeth test can point
/// at a different bundle. Test-only — the production `run`/`rebuild` paths use
/// [`serve_embed`]/[`serve_infer`] directly because they also need the timing.
#[cfg(test)]
async fn fold_embed_digest(
    spec: &ModelInferenceSpec,
    model_dir: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let rows = build_corpus(spec);
    let (session, _dir) = corpus_session(&rows).await?;
    let model_id = local_model_id(model_dir)?;
    let (digest, _ms, _n) = serve_embed(&session, &model_id).await?;
    Ok(digest)
}

/// The infer-verb peer of [`fold_embed_digest`]. Test-only.
#[cfg(test)]
async fn fold_infer_digest(
    spec: &ModelInferenceSpec,
    model_dir: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let rows = build_corpus(spec);
    let (session, _dir) = corpus_session(&rows).await?;
    let model_id = local_model_id(model_dir)?;
    let (digest, _ms, _n) = serve_infer(&session, &model_id, &spec.target_keys).await?;
    Ok(digest)
}

/// Run the model-inference tier against the committed spec: serve both verbs over
/// their committed bundles, fold both digests, and assemble the tier with the two
/// digest gates and the two same-box rate gates.
///
/// The digests gate for byte-equality (the portable cell-(d) anchor); the
/// serving throughputs are coarse same-box code-path nets gated by the
/// relative-drop [`crate::rate_gate`] — NOT the scaling SLO, which is the
/// cookbook (A) full-scale value.
pub async fn run(
    spec: &ModelInferenceSpec,
) -> Result<ModelInferenceTier, Box<dyn std::error::Error>> {
    let rows = build_corpus(spec);

    // Embed lane: a fresh session so the serve wall-time is a clean single-call
    // measurement, not contaminated by the infer serve.
    let (session, _dir) = corpus_session(&rows).await?;
    let embed_id = local_model_id(&ModelInferenceSpec::embed_model_dir())?;
    let (embed_digest, embed_ms, embed_rows) = serve_embed(&session, &embed_id).await?;
    let embed_rate = rows_per_s(embed_rows, embed_ms);

    // Infer lane: a fresh session likewise.
    let (session, _dir) = corpus_session(&rows).await?;
    let infer_id = local_model_id(&ModelInferenceSpec::classifier_model_dir())?;
    let (infer_digest, infer_ms, infer_rows) =
        serve_infer(&session, &infer_id, &spec.target_keys).await?;
    let infer_rate = rows_per_s(infer_rows, infer_ms);

    let embed_gate = crate::rate_gate::RateGate::evaluate(
        embed_rate,
        spec.baseline_embed_rows_per_s,
        crate::rate_gate::DEFAULT_REGRESSION_THRESHOLD,
    );
    let infer_gate = crate::rate_gate::RateGate::evaluate(
        infer_rate,
        spec.baseline_infer_rows_per_s,
        crate::rate_gate::DEFAULT_REGRESSION_THRESHOLD,
    );

    Ok(ModelInferenceTier {
        targets: spec.target_keys.len(),
        embed_rows_per_s: Measurement::measured(embed_rate, "rows_per_s"),
        embed_serve_ms: Measurement::measured(embed_ms, "ms"),
        embed_rate_gate: Some(RateVerdict {
            measured_pairs_per_s: embed_gate.measured,
            baseline_pairs_per_s: embed_gate.baseline,
            threshold: embed_gate.threshold,
            floor_pairs_per_s: embed_gate.floor,
            passed: embed_gate.passed,
            detail: embed_gate.detail(),
        }),
        embed_digest: DigestGate {
            passed: embed_digest == spec.embed_digest,
            measured: embed_digest,
            committed: spec.embed_digest.clone(),
        },
        infer_rows_per_s: Measurement::measured(infer_rate, "rows_per_s"),
        infer_serve_ms: Measurement::measured(infer_ms, "ms"),
        infer_rate_gate: Some(RateVerdict {
            measured_pairs_per_s: infer_gate.measured,
            baseline_pairs_per_s: infer_gate.baseline,
            threshold: infer_gate.threshold,
            floor_pairs_per_s: infer_gate.floor,
            passed: infer_gate.passed,
            detail: infer_gate.detail(),
        }),
        infer_digest: DigestGate {
            passed: infer_digest == spec.infer_digest,
            measured: infer_digest,
            committed: spec.infer_digest.clone(),
        },
    })
}

/// Whether all four gates held — the verdict the subcommand maps to its exit code
/// and the `cargo test` gate asserts: both digests matched their committed values
/// AND both serving throughputs cleared their same-box floors.
pub fn gates_passed(tier: &ModelInferenceTier) -> bool {
    tier.embed_digest.passed
        && tier.infer_digest.passed
        && tier.embed_rate_gate.as_ref().is_none_or(|v| v.passed)
        && tier.infer_rate_gate.as_ref().is_none_or(|v| v.passed)
}

/// Re-derive the committed spec from a fresh serve over the committed bundles:
/// regenerate the corpus, serve both verbs, and record both digests and both
/// same-box serving baselines. The off-box one-shot that writes
/// `baselines/model_inference.json`; CI only ever loads and re-serves.
///
/// The bundles themselves are the engine's own committed tiny fixtures, copied
/// into the bench's `baselines/` so it is self-contained — they are not
/// regenerated here (there is no training step; they are pre-built tiny models).
pub async fn rebuild_spec(
    params: ModelInferenceParams,
) -> Result<ModelInferenceSpec, Box<dyn std::error::Error>> {
    let mut spec = ModelInferenceSpec {
        row_count: params.row_count,
        corpus_seed: params.corpus_seed,
        target_keys: Vec::new(),
        embed_digest: String::new(),
        infer_digest: String::new(),
        baseline_embed_rows_per_s: 0.0,
        baseline_infer_rows_per_s: 0.0,
    };

    let rows = build_corpus(&spec);
    spec.target_keys = rows
        .iter()
        .take(params.target_count)
        .map(|r| r.id.clone())
        .collect();

    // Embed lane: serve, digest, and record the same-box baseline.
    let (session, _dir) = corpus_session(&rows).await?;
    let embed_id = local_model_id(&ModelInferenceSpec::embed_model_dir())?;
    let (embed_digest, embed_ms, embed_rows) = serve_embed(&session, &embed_id).await?;
    spec.embed_digest = embed_digest;
    spec.baseline_embed_rows_per_s = rows_per_s(embed_rows, embed_ms);

    // Infer lane likewise.
    let (session, _dir) = corpus_session(&rows).await?;
    let infer_id = local_model_id(&ModelInferenceSpec::classifier_model_dir())?;
    let (infer_digest, infer_ms, infer_rows) =
        serve_infer(&session, &infer_id, &spec.target_keys).await?;
    spec.infer_digest = infer_digest;
    spec.baseline_infer_rows_per_s = rows_per_s(infer_rows, infer_ms);

    Ok(spec)
}

/// The generation parameters a rebuild draws the committed spec from — the corpus
/// shape and how many targets the digests fold over.
#[derive(Debug, Clone, Copy)]
pub struct ModelInferenceParams {
    /// The synthetic corpus row count.
    pub row_count: usize,
    /// The corpus generation seed.
    pub corpus_seed: u64,
    /// How many of the corpus rows the digests fold over.
    pub target_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed spec is well-formed: a positive corpus, a non-empty target
    /// set within the corpus, both digests of the FNV width, and both baselines
    /// positive.
    #[test]
    fn committed_spec_is_well_formed() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        assert!(spec.row_count >= 1, "corpus must have rows");
        assert!(!spec.target_keys.is_empty(), "the digests need targets");
        assert!(
            spec.target_keys.len() <= spec.row_count,
            "targets are a prefix of the corpus"
        );
        for (digest, name) in [(&spec.embed_digest, "embed"), (&spec.infer_digest, "infer")] {
            assert_eq!(digest.len(), 16, "{name} digest is a 64-bit FNV hex string");
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{name} digest must be hex"
            );
        }
        assert!(
            spec.baseline_embed_rows_per_s > 0.0 && spec.baseline_infer_rows_per_s > 0.0,
            "committed baselines must be positive"
        );
    }

    /// The committed model bundles are present and complete: each verb's bundle
    /// carries the three files the resolver's local branch loads. A missing file
    /// would make the serve fail to even resolve, so this guards the committed
    /// artifacts' completeness.
    #[test]
    fn committed_model_bundles_are_present() {
        for dir in [
            ModelInferenceSpec::embed_model_dir(),
            ModelInferenceSpec::classifier_model_dir(),
        ] {
            for file in ["config.json", "model.safetensors", "tokenizer.json"] {
                assert!(
                    dir.join(file).exists(),
                    "committed model bundle missing {file} under {}",
                    dir.display()
                );
            }
        }
    }

    /// The teeth, DIGESTS-CLEAR direction: serving both verbs over the committed
    /// bundles reproduces both committed digests. A regression in either real
    /// serving path moves its bits and trips this.
    #[tokio::test(flavor = "multi_thread")]
    async fn reserve_matches_committed_digests() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        let embed = fold_embed_digest(&spec, &ModelInferenceSpec::embed_model_dir())
            .await
            .expect("embed fold runs over the committed bundle");
        assert_eq!(
            embed, spec.embed_digest,
            "the re-served embed digest drifted"
        );
        let infer = fold_infer_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("infer fold runs over the committed bundle");
        assert_eq!(
            infer, spec.infer_digest,
            "the re-served infer digest drifted"
        );
    }

    /// The teeth, GATE-FAILS direction for the EMBED verb (RC1: an assertion must
    /// be able to fail). Serving the *classifier* bundle through the embed verb
    /// produces a different embedding (a different model, different weights and
    /// hidden geometry), so the embed digest moves — proving the gate reacts to
    /// the served model, not a constant. The committed embed bundle reproduces the
    /// committed digest (the contrast that gives the perturbation its teeth).
    #[tokio::test(flavor = "multi_thread")]
    async fn embed_over_a_different_model_changes_the_digest() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        let correct = fold_embed_digest(&spec, &ModelInferenceSpec::embed_model_dir())
            .await
            .expect("committed embed serve");
        assert_eq!(
            correct, spec.embed_digest,
            "the committed embed bundle must reproduce the committed embed digest"
        );
        let perturbed = fold_embed_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("perturbed embed serve");
        assert_ne!(
            perturbed, spec.embed_digest,
            "embedding through a different model must move the embed digest"
        );
    }

    /// The teeth, GATE-FAILS direction for the INFER verb: serving over a
    /// perturbed corpus (a different seed rotates which sentences the rows draw,
    /// so the classifier scores different text) produces a different infer digest.
    /// This proves the digest reacts to the served *input*, the regression a
    /// silently-changed tokenization or input-column wiring would cause.
    #[tokio::test(flavor = "multi_thread")]
    async fn infer_over_perturbed_input_changes_the_digest() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        let correct = fold_infer_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("committed infer serve");
        assert_eq!(
            correct, spec.infer_digest,
            "the committed corpus must reproduce the committed infer digest"
        );

        let mut perturbed_spec = spec.clone();
        perturbed_spec.corpus_seed = spec.corpus_seed.wrapping_add(1);
        let perturbed =
            fold_infer_digest(&perturbed_spec, &ModelInferenceSpec::classifier_model_dir())
                .await
                .expect("perturbed infer serve");
        assert_ne!(
            perturbed, spec.infer_digest,
            "classifying different input text must move the infer digest"
        );
    }

    /// The gate-fails direction at the harness level: a tampered committed digest
    /// fails [`gates_passed`], proving the verdict reacts to the committed value.
    #[tokio::test(flavor = "multi_thread")]
    async fn tampered_committed_digest_fails_the_gate() {
        let mut spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        spec.embed_digest = "deadbeefdeadbeef".to_string();
        let tier = run(&spec).await.expect("tier still runs");
        assert!(
            !gates_passed(&tier),
            "a tampered committed digest must trip the gate"
        );
    }

    /// Both digests are reload-invariant: two independent folds of each committed
    /// bundle (each a fresh session + serve) produce byte-identical digests. This
    /// is the deterministic-serving contract the gate rests on — were the serve
    /// reload-sensitive, the committed digest would be a moving target and the
    /// gate meaningless (the vacuity trap).
    #[tokio::test(flavor = "multi_thread")]
    async fn digests_are_reload_invariant() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        let embed_one = fold_embed_digest(&spec, &ModelInferenceSpec::embed_model_dir())
            .await
            .expect("first embed fold");
        let embed_two = fold_embed_digest(&spec, &ModelInferenceSpec::embed_model_dir())
            .await
            .expect("second embed fold");
        assert_eq!(embed_one, embed_two, "embed serve must be reload-invariant");
        assert_eq!(
            embed_one, spec.embed_digest,
            "and equal to the committed digest"
        );

        let infer_one = fold_infer_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("first infer fold");
        let infer_two = fold_infer_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("second infer fold");
        assert_eq!(infer_one, infer_two, "infer serve must be reload-invariant");
        assert_eq!(
            infer_one, spec.infer_digest,
            "and equal to the committed digest"
        );
    }

    /// The committed serving baselines gate with teeth: a run at the baseline
    /// clears the gate, a run past the threshold fails it (RC1). Asserts the
    /// committed baselines are well-formed, generously-thresholded same-box
    /// references — the coarse code-path net, not the full-scale SLO.
    #[test]
    fn committed_baselines_gate_with_teeth() {
        use crate::rate_gate::{RateGate, DEFAULT_REGRESSION_THRESHOLD};

        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        for rate in [
            spec.baseline_embed_rows_per_s,
            spec.baseline_infer_rows_per_s,
        ] {
            assert!(rate > 0.0, "committed baseline must be positive");
            assert!(
                RateGate::evaluate(rate, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
                "a run at the committed baseline must pass"
            );
            let floor = rate * (1.0 - DEFAULT_REGRESSION_THRESHOLD);
            assert!(
                !RateGate::evaluate(floor - 1.0, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
                "a rate below the floor must fail the gate"
            );
        }
    }
}
