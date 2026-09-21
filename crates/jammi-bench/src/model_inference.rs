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
//!   adapt path moves the digest off the in-process baseline and trips `cargo test`.
//!
//! The two lanes here mirror that priority:
//!
//! 1. **The determinism digest (the cell-(d) anchor)** — the engine's serving
//!    path is byte-deterministic on `Device::Cpu` over a fixed model and fixed
//!    inputs **on a machine** (the engine's own
//!    `embedding_vectors_are_semantically_meaningful_and_reproducible` contract:
//!    encoding the same text twice yields identical vectors). The served output is
//!    `f32` (embedding vectors; score distributions), and an `f32` forward's exact
//!    bits are NOT identical across CPUs (SIMD/FMA/BLAS reduction order differs), so
//!    the gate does not assert a committed cross-machine constant — that would
//!    spuriously "fail" on any box but the rebuild box. It asserts the real,
//!    same-machine property: serve twice on the running box and the two digests
//!    agree. A regression in the forward, the pooling, the tokenizer dispatch, the
//!    label mapping, or the output adapter is caught by the relative perturbation
//!    teeth — a perturbed input or a different model moving the digest off the
//!    in-process baseline computed on the same box.
//! 2. **The serving overhead (the code-path net, NOT the SLO)** — what the
//!    serving plan costs over calling the loaded model directly, measured as two
//!    dimensionless numbers in ONE process on ONE box and gated against committed
//!    budgets by [`crate::rate_gate`]. See "The overhead gate" below. This catches
//!    a *code path* regression (lost batching, an accidental per-row model
//!    reload, a dropped fast path, a plan that grew a per-call cost) — it is
//!    emphatically NOT the scaling number, which is the cookbook (A) value over a
//!    real model.
//!
//! ## The overhead gate: a same-process ratio, never a committed rate
//!
//! A rows/s rate is a property of `code × box`. A committed rate therefore gates
//! only on the box that emitted it: on a faster box the floor sits far below
//! anything the code can regress to, and the gate passes whatever happens. The
//! only baseline that tracks the box is one measured on it, in the same run, and
//! the only such baseline that needs no second build is a reference workload in
//! the same process.
//!
//! The reference is the DIRECT leg: the same rows, in the same `batch_size`
//! chunks, through [`jammi_ai::model::LoadedModel::forward`] — the loaded
//! model's own tokenize → forward → pool → normalize — with no plan, no catalog
//! and no result table. Everything the serving verb adds sits between the two
//! legs, and everything the box contributes (its cores, its memory, its BLAS
//! path) is in both. Both legs are served over a row sweep, interleaved
//! (plan, direct, direct, plan) so a drift of the box over the run lands on both
//! alike, and each leg's fastest serve per row count is fitted to
//! [`crate::timing::CostFit`]. The sweep stays inside the range where a serve
//! IS two-term: the embeddings sink builds an ANN segment whose cost grows
//! faster than its rows, which by a few thousand rows bends the curve away from
//! any straight line (a fit residual the gate itself refuses, below). Two
//! numbers are gated:
//!
//! * **`per_row_ratio`** = `plan.per_row_ms / direct.per_row_ms` — what a row
//!   costs through the plan, in units of what the bare model charges for it.
//!   Lost batching, a per-row reload or a per-row copy moves this.
//! * **`fixed_rows`** = `plan.fixed_ms / direct.per_row_ms` — what one call costs
//!   before it serves a row, in rows of bare-model work. A plan that grew a
//!   per-call cost moves this. (Milliseconds are not portable; row-equivalents
//!   are, to the extent planning and the forward scale alike across boxes — the
//!   weaker of the two cancellations, which is why both terms are gated apart.)
//!
//! Box interference only ever ADDS to a serve, so the gate folds each leg's
//! fastest serves over up to three fresh-session sweeps before it fails: a
//! regression in the code is in every sweep, a slow stretch of the box is not.
//! And it refuses to gate on a fit that does not describe its points: a plan
//! whose relative residual leaves `MAX_TWO_TERM_RESIDUAL` is no longer two-term
//! over the sweep — the signature of a cost that went superlinear — and fails
//! as such.
//!
//! The ratio holds across boxes, not across THREAD POSTURES: a many-thread
//! compute pool makes the tiny forward several times dearer, which shrinks the
//! plan's share of a row. The spec therefore records the `RAYON_NUM_THREADS` its
//! budgets were measured under, and the tier refuses to gate under any other.
//!
//! What the ratio deliberately cannot see is a regression INSIDE the forward —
//! it slows both legs alike. That is not this tier's to catch: a 32-dim
//! one-layer forward says nothing about a production encoder's, and the forward
//! is measured where it is real (the GPU tiers and the PyTorch reference).
//!
//! ## Why two bundles, two verbs
//!
//! `generate_embeddings` and `infer` are distinct engine paths: the embed verb
//! pools encoder hidden states into a vector and persists a result table; the
//! infer verb (here in its `Classification` task) runs a classifier head and
//! adapts a label + score distribution. Driving each over its own tiny bundle
//! exercises the two real paths, so a regression in either moves its own digest.
//!
//! ## The committed spec + the referenced fixtures
//!
//! `baselines/model_inference.json` carries the synthetic corpus spec (so the
//! gate regenerates the exact deterministic text rows), the infer digest's target
//! keys (a deterministic prefix, folded in key order so any box folds the same
//! per-row scores), the two digests, the overhead sweep's row counts and the two
//! verbs' overhead budgets. The tiny
//! model bundles themselves are NOT copied here — they are REFERENCED from the
//! engine's shared fixture tree (`cookbook/fixtures/tiny_bert` and
//! `tiny_modernbert_classifier` — a 32-dim 1-layer BERT and a 32-dim 1-layer
//! ModernBERT classifier), the same fixtures the `jammi-encoders` tests reference
//! by relative path. The rebuild re-derives every committed value by serving over
//! those referenced bundles; nothing is duplicated.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use jammi_ai::model::{LoadedModel, ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, InferenceConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};

use crate::rate_gate::{RateGate, DEFAULT_REGRESSION_THRESHOLD};
use crate::report::{
    CostVerdict, DeterminismGate, Measurement, ModelInferenceTier, OverheadLane, TwoTermVerdict,
};
use crate::timing::{per_second, CostFit, ServeStats};

/// The source id the synthetic corpus registers under. Generic — names no
/// consumer; the corpus is a neutral family of short factual sentences.
pub(crate) const SOURCE_ID: &str = "corpus";
/// The text column the verbs read.
pub(crate) const TEXT_COLUMN: &str = "text";
/// The key column carrying each row's stable id into the result table.
pub(crate) const KEY_COLUMN: &str = "_row_id";

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
    /// bundle when the spec was cut, on the rebuild box, folded over the persisted
    /// vector column. A documented same-box reference (the vectors are `f32`, whose
    /// exact bits vary by CPU) — never asserted for cross-machine equality.
    pub embed_digest: String,
    /// The committed infer digest: the checksum the engine's `infer`
    /// (`Classification`) produced over the committed targets and the committed
    /// classifier bundle, on the rebuild box — folded over the full per-row score
    /// distributions. A documented same-box reference (`f32` scores), never asserted
    /// for cross-machine equality.
    pub infer_digest: String,
    /// The compute-thread count (`RAYON_NUM_THREADS`) the overhead budgets were
    /// measured under. The ratio of the plan to the bare forward depends on how
    /// the forward is threaded — a pool of many threads makes a tiny forward
    /// several times dearer and the plan's share of a row correspondingly
    /// smaller — so a budget holds only under the posture it was measured in,
    /// and [`run`] refuses any other.
    pub compute_threads: usize,
    /// The row counts the overhead sweep serves, ascending. At the smallest a
    /// serve is its fixed cost; at the largest it is its rows — the sweep needs
    /// both ends to separate the two terms.
    pub overhead_rows: Vec<usize>,
    /// The embed verb's committed overhead budget.
    pub embed_overhead: OverheadBudget,
    /// The infer verb's committed overhead budget.
    pub infer_overhead: OverheadBudget,
}

/// One verb's committed serving-overhead budget: the two dimensionless costs of
/// the serving plan over the direct leg (see the module doc's "The overhead
/// gate"), as the rebuild measured them. The gate's ceiling is
/// `budget / (1 − threshold)`; the threshold is the harness's single
/// [`DEFAULT_REGRESSION_THRESHOLD`], never stored here.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OverheadBudget {
    /// `plan.per_row_ms / direct.per_row_ms`.
    pub per_row_ratio: f64,
    /// `plan.fixed_ms / direct.per_row_ms`.
    pub fixed_rows: f64,
}

impl ModelInferenceSpec {
    /// The crate-relative path to the committed spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("model_inference.json")
    }

    /// The shared engine fixture the embed bundle is read from — REFERENCED, not
    /// copied (the `jammi-encoders` tests reference this same `tiny_bert` fixture
    /// the same way). A tiny real BERT; the resolver's no-network `local:` branch
    /// loads it on `Device::Cpu`.
    pub fn embed_model_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
    }

    /// The shared engine fixture the classifier bundle is read from — REFERENCED,
    /// not copied (same idiom as the `jammi-encoders` tests). A tiny real ModernBERT
    /// classifier loaded via the resolver's no-network `local:` branch.
    pub fn classifier_model_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/tiny_modernbert_classifier")
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
pub(crate) fn local_model_id(dir: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!(
        "local:{}",
        dir.to_str()
            .ok_or("committed model dir is not valid UTF-8")?
    ))
}

/// One corpus row: its stable key and its text.
pub(crate) struct Row {
    pub(crate) id: String,
    pub(crate) text: String,
}

/// sha256 (hex) content hash of the RESOLVED corpus this `(corpus_seed,
/// row_count)` pair ACTUALLY produces — [`build_corpus`]'s own OUTPUT row
/// texts (in row order), never the raw `(SENTENCES, corpus_seed, row_count)`
/// INPUTS independently re-derived (hashing the whole [`SENTENCES`] array
/// plus the two scalar inputs directly, WITHOUT calling [`build_corpus`]
/// itself, would miss a bug that changed the SELECTION/ROTATION rule — e.g.
/// `% 4` in place of the real `% SENTENCES.len()` — which produces a
/// DIFFERENT resolved corpus under the SAME raw inputs: structurally blind
/// to exactly the class of bug this identity field exists to catch). Exists so
/// `GpuInferenceTier::corpus_sha256` (the within-run A/B identity
/// contract) has a single content hash
/// standing in for "the corpus this leg SERVED came from the same generator,
/// via the same selection logic" — the same belt-and-suspenders role
/// `checkpoint_*_sha256` plays for a model bundle: `corpus_seed`/`row_count`
/// alone are two SCALARS that can each independently be held fixed while
/// EITHER [`SENTENCES`]' own text OR `build_corpus`'s own selection rule is
/// edited, which neither scalar would ever move on its own.
///
/// Same-architecture evidence only: `row_count`'s own byte width is pinned to
/// `u64` in [`corpus_sha256_of_rows`] below, but [`Row`]/[`build_corpus`]
/// themselves are written in terms of `usize`-indexed Rust collections, so
/// this function makes no cross-architecture (32-bit vs 64-bit) guarantee —
/// not a live concern for the within-run A/B comparator this field backs,
/// which always compares two legs built/run on the SAME pod,
/// but disclosed here honestly rather than silently assumed.
pub(crate) fn corpus_sha256(corpus_seed: u64, row_count: usize) -> String {
    corpus_sha256_of_rows(&build_corpus(corpus_seed, row_count))
}

/// The core hashing primitive [`corpus_sha256`] wraps: sha256 over each
/// row's TEXT (in order, null-byte separated — sentence text never contains
/// a literal null byte, so this delimiter is unambiguous), then the row
/// COUNT as a pinned-width `u64` (belt-and-suspenders alongside the
/// delimiter scheme, which already makes the digest length-sensitive on its
/// own). Split out from [`corpus_sha256`] (rather than folded inline) so a
/// test can drive it against a DELIBERATELY alternative row selection — see
/// `tests::corpus_sha256_reacts_to_a_changed_selection_rule_not_just_a_changed_byte_layout`
/// for the teeth this split makes possible.
fn corpus_sha256_of_rows(rows: &[Row]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update(row.text.as_bytes());
        hasher.update(b"\0");
    }
    hasher.update((rows.len() as u64).to_le_bytes());
    hex::encode(hasher.finalize())
}

/// Build the deterministic synthetic corpus: `row_count` rows, each drawing a
/// fixed sentence by a seed-rotated index so the corpus is byte-stable from the
/// committed seed.
pub(crate) fn build_corpus(corpus_seed: u64, row_count: usize) -> Vec<Row> {
    (0..row_count)
        .map(|i| Row {
            id: format!("row_{i}"),
            text: SENTENCES[(corpus_seed as usize).wrapping_add(i) % SENTENCES.len()].to_string(),
        })
        .collect()
}

/// How a corpus session serves: the device and the two `[inference]` knobs that
/// shape a serving plan. Everything else is the engine's default.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ServeShape {
    /// The device ordinal the engine's `select_device` resolves: negative is
    /// `Device::Cpu`, otherwise a CUDA ordinal the box must actually have.
    pub(crate) gpu_device: i32,
    /// `[inference] batch_size` — rows per model forward.
    pub(crate) batch_size: usize,
    /// `[inference] partitions` — the plan's inference fan-out.
    pub(crate) partitions: usize,
    /// `[gpu] compute_precision` — the precision a model loads at unless its
    /// own `config.json` declares one.
    pub(crate) compute_precision: jammi_numerics::ComputePrecision,
}

impl ServeShape {
    /// The engine's default `[inference]` shape on `gpu_device`.
    pub(crate) fn on_device(gpu_device: i32) -> Self {
        let defaults = InferenceConfig::default();
        Self {
            gpu_device,
            batch_size: defaults.batch_size,
            partitions: defaults.partitions,
            compute_precision: GpuConfig::default().compute_precision,
        }
    }

    /// The hermetic shape: the engine's defaults on `Device::Cpu`, no GPU and
    /// no network.
    pub(crate) fn cpu() -> Self {
        Self::on_device(-1)
    }
}

/// Write `rows` as the source Parquet a corpus session serves: `_row_id`,
/// `text`, and a `y` placeholder so the schema matches the engine's
/// source-table shape. Written through the engine's own writer, in row order.
pub(crate) async fn write_corpus(
    rows: &[Row],
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(KEY_COLUMN, DataType::Utf8, false),
        Field::new(TEXT_COLUMN, DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let ys: Vec<f64> = (0..rows.len()).map(|i| i as f64).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(texts)),
            Arc::new(Float64Array::from(ys)),
        ],
    )?;
    let url = StorageUrl::parse(path.to_str().ok_or("corpus path is not valid UTF-8")?)?;
    let handle = StorageRegistry::new().handle_for(&url, None)?;
    let mut writer = ObjectParquetWriter::open(&handle, schema).await?;
    writer.write_batch(&batch).await?;
    writer.close().await?;
    Ok(())
}

/// Stand up a session that serves the corpus Parquet at `corpus` (one
/// [`write_corpus`] wrote) under `shape`, keeping its artifacts in
/// `artifact_dir`.
///
/// `gpu.require_gpu` is set to `gpu_device >= 0` — the SAME convention
/// `jammi-ai`'s own `gpu_capability` harness pins
/// (`tests/gpu_capability/harness.rs::config_for`: `require_gpu: device >= 0`)
/// — so a requested CUDA ordinal that the box cannot actually satisfy fails
/// loud with a typed `JammiError::Gpu` at the FIRST model load
/// (`CandleBackend::load`'s `select_device(device_config)?`), rather than
/// silently degrading to `Device::Cpu` with only a `tracing::warn!` (a
/// `--cuda N` leg must never be able to publish
/// `device_requested:"cuda:N"` + a real `device_name` for a run that actually
/// executed on CPU). A negative ordinal is untouched (`require_gpu` computes to
/// `false`, and `select_device` selects `Device::Cpu` unconditionally for it
/// regardless of the flag).
pub(crate) async fn session_over(
    corpus: &std::path::Path,
    artifact_dir: &std::path::Path,
    shape: ServeShape,
) -> Result<Arc<InferenceSession>, Box<dyn std::error::Error>> {
    let config = JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: GpuConfig {
            device: shape.gpu_device,
            require_gpu: shape.gpu_device >= 0,
            compute_precision: shape.compute_precision,
            ..Default::default()
        },
        inference: InferenceConfig {
            batch_size: shape.batch_size,
            partitions: shape.partitions,
            ..Default::default()
        },
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await?);
    session.install_query_functions();
    session
        .add_source(
            SOURCE_ID,
            SourceType::File,
            SourceConnection {
                url: Some(format!(
                    "file://{}",
                    corpus.to_str().ok_or("corpus path is not valid UTF-8")?
                )),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;
    Ok(session)
}

/// [`write_corpus`] then [`session_over`] in one scratch directory, which the
/// returned [`tempfile::TempDir`] keeps alive for as long as the session
/// serves from it. The shape the CPU tier ([`ServeShape::cpu`]) and the GPU
/// perf-baseline tier share.
pub(crate) async fn corpus_session(
    rows: &[Row],
    shape: ServeShape,
) -> Result<(Arc<InferenceSession>, tempfile::TempDir), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let corpus = dir.path().join("source.parquet");
    write_corpus(rows, &corpus).await?;
    let session = session_over(&corpus, dir.path(), shape).await?;
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
/// order — for a never-refreshed table (this bench never refreshes) the raw
/// base bytes in the embedding pipeline's deterministic key-order write order;
/// a versioned table would read through its masked provider in `_row_id`
/// order — so the digest is reproducible on a machine (the tier's on-box
/// determinism gate folds it twice and asserts equality) — see [`fold_vectors`].
///
/// Returns `(digest, serve_wall_ms, rows_served)`. The serve wall-time is over
/// the whole `generate_text_embeddings` call (the rate's numerator-time); the
/// read-back is the deterministic fold, not part of the measured serve.
///
/// `model_id` selects the bundle the serve resolves — the committed embed bundle
/// for the gate, a different bundle for the teeth test.
pub(crate) async fn serve_embed(
    session: &Arc<InferenceSession>,
    model_id: &str,
) -> Result<(String, f64, usize), Box<dyn std::error::Error>> {
    let (table, serve_ms) = serve_embed_table(session, model_id).await?;
    let vectors = read_served_vectors(session, table).await?;
    Ok((fold_vectors(&vectors), serve_ms, vectors.len()))
}

/// The FNV digest of a served vector column, in the order given: every float's
/// bits, then a length tag per vector so the row boundaries are explicit and a
/// dimensionality change cannot be masked by a float collision.
pub(crate) fn fold_vectors(vectors: &[Vec<f32>]) -> String {
    let mut fnv = Fnv::new();
    for vector in vectors {
        for v in vector {
            fnv.mix_f32(*v);
        }
        fnv.mix(vector.len() as u8); // row boundary + dimensionality tag
    }
    fnv.finish()
}

/// One timed `generate_text_embeddings` call: the result table it committed and
/// the wall time of the whole call, milliseconds — the rows read from the
/// source, forwarded, and PERSISTED. Nothing is read back inside the span.
pub(crate) async fn serve_embed_table(
    session: &Arc<InferenceSession>,
    model_id: &str,
) -> Result<(jammi_db::catalog::result_repo::ResultTableRecord, f64), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let (table, _) = session
        .generate_text_embeddings(
            SOURCE_ID,
            model_id,
            &[TEXT_COLUMN.to_string()],
            KEY_COLUMN,
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await?;
    Ok((table, start.elapsed().as_secs_f64() * 1_000.0))
}

/// The persisted `vector` column of a served table, in result-table storage
/// order — for a never-refreshed table, the embedding pipeline's key-order
/// write order.
pub(crate) async fn read_served_vectors(
    session: &Arc<InferenceSession>,
    table: jammi_db::catalog::result_repo::ResultTableRecord,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let pin = session.result_store().pin_current_version(table).await?;
    Ok(session.read_vectors(&pin).await?)
}

/// Serve the infer verb (`Classification`) over the corpus through the
/// engine's real `infer`, and index the served `(_row_id, all_scores_json)`
/// rows into a map — the shared raw serve both [`serve_infer`] (folds a
/// committed target subset) and [`serve_infer_all`] (folds every scored row)
/// build their digest on top of. Returns `(by_key, serve_wall_ms)`; the wall
/// time is over the whole `infer` call, matching [`serve_embed`]'s timing
/// convention.
async fn infer_scores_by_key(
    session: &Arc<InferenceSession>,
    model_id: &str,
) -> Result<(std::collections::HashMap<String, String>, f64), Box<dyn std::error::Error>> {
    let source = ModelSource::parse(model_id);
    let start = Instant::now();
    let (batches, _) = session
        .infer(
            SOURCE_ID,
            &source,
            ModelTask::Classification,
            &[TEXT_COLUMN.to_string()],
            KEY_COLUMN,
            jammi_db::store::CachePolicy::Bypass,
        )
        .await?;
    let serve_ms = start.elapsed().as_secs_f64() * 1_000.0;

    // Index `_row_id -> all_scores_json` across the served batches so a caller
    // can fold in whatever order (committed targets, or every scored key
    // sorted) independent of the served batch shape.
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
    Ok((by_key, serve_ms))
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
pub(crate) async fn serve_infer(
    session: &Arc<InferenceSession>,
    model_id: &str,
    target_keys: &[String],
) -> Result<(String, f64, usize), Box<dyn std::error::Error>> {
    let (by_key, serve_ms) = infer_scores_by_key(session, model_id).await?;

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

/// Serve the infer verb over EVERY corpus row (rather than a fixed committed
/// target subset) and fold every scored row's `all_scores_json`, in sorted
/// `_row_id` order (so the fold is reproducible independent of served batch
/// order), into an FNV digest. Returns `(digest, serve_wall_ms, scored_row_count)`.
///
/// The scored-row count is the row-conservation signal a caller needs: `infer`'s
/// per-row annotate semantics can silently drop a row whose forward errored (the
/// regression `gpu_capability`'s `classification_parity` guards on CPU↔GPU
/// output), so a caller comparing this count against the input row count catches
/// that same class of silent data loss on whichever device it served.
pub(crate) async fn serve_infer_all(
    session: &Arc<InferenceSession>,
    model_id: &str,
) -> Result<(String, f64, usize), Box<dyn std::error::Error>> {
    let (by_key, serve_ms) = infer_scores_by_key(session, model_id).await?;

    let mut keys: Vec<&String> = by_key.keys().collect();
    keys.sort();
    let mut fnv = Fnv::new();
    for key in &keys {
        fnv.mix_str(&by_key[*key]);
        fnv.mix(0x00); // key boundary
    }
    Ok((fnv.finish(), serve_ms, keys.len()))
}

/// Fold a digest digest-only (dropping the serve timing): stand up the corpus
/// session and serve the named verb over the named bundle. Used by the determinism
/// test (folding twice on the running box) and the teeth tests (with `model_dir`
/// explicit so a perturbation can point at a different bundle). Test-only — the
/// production `run`/`rebuild` paths use [`serve_embed`]/[`serve_infer`] directly.
#[cfg(test)]
async fn fold_embed_digest(
    spec: &ModelInferenceSpec,
    model_dir: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let rows = build_corpus(spec.corpus_seed, spec.row_count);
    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
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
    let rows = build_corpus(spec.corpus_seed, spec.row_count);
    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
    let model_id = local_model_id(model_dir)?;
    let (digest, _ms, _n) = serve_infer(&session, &model_id, &spec.target_keys).await?;
    Ok(digest)
}

/// A serving verb the overhead gate measures: which task its model loads for,
/// which committed bundle it serves, and its timed plan leg.
#[derive(Debug, Clone, Copy)]
enum Verb {
    /// `generate_text_embeddings` over the embed bundle.
    Embed,
    /// `infer` (`Classification`) over the classifier bundle.
    Infer,
}

impl Verb {
    fn task(self) -> ModelTask {
        match self {
            Verb::Embed => ModelTask::TextEmbedding,
            Verb::Infer => ModelTask::Classification,
        }
    }

    fn model_dir(self) -> std::path::PathBuf {
        match self {
            Verb::Embed => ModelInferenceSpec::embed_model_dir(),
            Verb::Infer => ModelInferenceSpec::classifier_model_dir(),
        }
    }

    /// The wall time of one serve of this verb through the engine's plan,
    /// milliseconds.
    async fn plan_ms(
        self,
        session: &Arc<InferenceSession>,
        model_id: &str,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        match self {
            Verb::Embed => Ok(serve_embed_table(session, model_id).await?.1),
            Verb::Infer => Ok(infer_scores_by_key(session, model_id).await?.1),
        }
    }
}

/// The wall time of the DIRECT leg, milliseconds: `texts` forwarded through the
/// loaded model in `batch_size` chunks, in row order — the same rows in the same
/// chunks the plan forwards, with nothing around them.
pub(crate) fn direct_forward_ms(
    model: &LoadedModel,
    texts: &StringArray,
    batch_size: usize,
    task: ModelTask,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start = Instant::now();
    for offset in (0..texts.len()).step_by(batch_size) {
        let chunk: ArrayRef = Arc::new(texts.slice(offset, batch_size.min(texts.len() - offset)));
        model.forward(&[chunk], task)?;
    }
    Ok(start.elapsed().as_secs_f64() * 1_000.0)
}

/// Warm rounds served and discarded before the measured rounds of one sweep
/// point: the first pays the model load and the plan's first-use costs.
const OVERHEAD_WARMUP_ROUNDS: usize = 2;
/// Measured rounds per sweep point. A round serves plan, direct, direct, plan,
/// so each leg's statistic is the fastest of twice this many serves.
const OVERHEAD_ROUNDS: usize = 5;

/// How a sweep's plan leg is shaped, against a direct leg that always forwards
/// in the engine's default `batch_size` chunks.
#[derive(Debug, Clone, Copy)]
struct OverheadSweep<'a> {
    verb: Verb,
    corpus_seed: u64,
    rows: &'a [usize],
    /// The plan leg's serve shape. [`ServeShape::cpu`] for the gate; a test
    /// degrades it to prove the gate sees the regression it exists for.
    plan_shape: ServeShape,
}

/// How many fresh-session sweeps one lane may fold before its verdict stands.
/// Interference from the box only ever ADDS to a serve — a slow stretch, a
/// session that landed badly — so each leg's fastest serve, folded over more
/// sessions, converges on the cost of the code from above. A regression in the
/// code is in every session and survives the fold; a bad stretch is not, and
/// does not. The gate folds until it passes or the attempts run out; the
/// rebuild folds all of them.
const OVERHEAD_ATTEMPTS: usize = 3;

/// The ceiling on the plan fit's relative residual: past it the serve is not
/// two-term over the sweep, its two fitted terms describe nothing, and the gate
/// fails rather than gate on them. A cost that turned superlinear inside the
/// sweep — the quadratic blow-up a regression net exists for — lands here.
const MAX_TWO_TERM_RESIDUAL: f64 = 0.10;

/// Each leg's fastest serve at each of a sweep's row counts, milliseconds.
#[derive(Debug, Clone)]
struct OverheadMins {
    plan_ms: Vec<f64>,
    direct_ms: Vec<f64>,
}

impl OverheadMins {
    /// The fastest serves across both — see [`OVERHEAD_ATTEMPTS`].
    fn fold(self, other: Self) -> Self {
        let fastest =
            |a: Vec<f64>, b: Vec<f64>| a.into_iter().zip(b).map(|(a, b)| a.min(b)).collect();
        Self {
            plan_ms: fastest(self.plan_ms, other.plan_ms),
            direct_ms: fastest(self.direct_ms, other.direct_ms),
        }
    }

    /// Both legs' cost models over `rows`.
    fn fits(&self, rows: &[usize]) -> Result<OverheadFits, Box<dyn std::error::Error>> {
        let fit = |mins: &[f64], leg: &str| {
            let points: Vec<(usize, f64)> =
                rows.iter().copied().zip(mins.iter().copied()).collect();
            CostFit::least_squares(&points).ok_or_else(|| {
                format!("the {leg} leg's sweep over {rows:?} rows determines no two-term cost: {mins:?} ms")
            })
        };
        Ok(OverheadFits {
            plan: fit(&self.plan_ms, "plan")?,
            direct: fit(&self.direct_ms, "direct")?,
        })
    }
}

/// The two legs' fitted cost models over one row sweep.
struct OverheadFits {
    plan: CostFit,
    direct: CostFit,
}

impl OverheadFits {
    /// The two gated numbers — see the module doc's "The overhead gate".
    fn overhead(&self) -> OverheadBudget {
        OverheadBudget {
            per_row_ratio: self.plan.per_row_ms / self.direct.per_row_ms,
            fixed_rows: self.plan.fixed_ms / self.direct.per_row_ms,
        }
    }
}

/// Serve both legs of `sweep` at every row count, interleaved, over fresh
/// sessions, and keep each leg's fastest serve per row count.
async fn measure_overhead(
    sweep: OverheadSweep<'_>,
) -> Result<OverheadMins, Box<dyn std::error::Error>> {
    let model_id = local_model_id(&sweep.verb.model_dir())?;
    let source = ModelSource::parse(&model_id);
    let direct_batch_size = ServeShape::cpu().batch_size;
    let mut plan_min_ms = Vec::with_capacity(sweep.rows.len());
    let mut direct_min_ms = Vec::with_capacity(sweep.rows.len());
    for &row_count in sweep.rows {
        let rows = build_corpus(sweep.corpus_seed, row_count);
        let texts = StringArray::from_iter_values(rows.iter().map(|r| r.text.as_str()));
        let (session, _dir) = corpus_session(&rows, sweep.plan_shape).await?;
        let guard = session
            .model_cache()
            .get_or_load(&source, sweep.verb.task(), None)
            .await?;
        let direct =
            || direct_forward_ms(&guard.model, &texts, direct_batch_size, sweep.verb.task());

        let mut plan_ms = Vec::with_capacity(2 * OVERHEAD_ROUNDS);
        let mut direct_ms = Vec::with_capacity(2 * OVERHEAD_ROUNDS);
        for round in 0..OVERHEAD_WARMUP_ROUNDS + OVERHEAD_ROUNDS {
            let plan_first = sweep.verb.plan_ms(&session, &model_id).await?;
            let direct_pair = [direct()?, direct()?];
            let plan_last = sweep.verb.plan_ms(&session, &model_id).await?;
            if round >= OVERHEAD_WARMUP_ROUNDS {
                plan_ms.extend([plan_first, plan_last]);
                direct_ms.extend(direct_pair);
            }
        }
        let stats =
            |samples: &[f64]| ServeStats::of(samples).ok_or("an overhead leg measured no serve");
        plan_min_ms.push(stats(&plan_ms)?.min_ms);
        direct_min_ms.push(stats(&direct_ms)?.min_ms);
    }
    Ok(OverheadMins {
        plan_ms: plan_min_ms,
        direct_ms: direct_min_ms,
    })
}

/// [`measure_overhead`] folded over every one of [`OVERHEAD_ATTEMPTS`] — what a
/// budget is rebuilt from.
async fn settled_overhead(
    sweep: OverheadSweep<'_>,
) -> Result<OverheadBudget, Box<dyn std::error::Error>> {
    let mut mins = measure_overhead(sweep).await?;
    for _ in 1..OVERHEAD_ATTEMPTS {
        mins = mins.fold(measure_overhead(sweep).await?);
    }
    Ok(mins.fits(sweep.rows)?.overhead())
}

/// Gate one measured cost against its committed budget, carrying the full
/// arithmetic into the report.
fn cost_verdict(measured: f64, budget: f64) -> CostVerdict {
    let gate = RateGate::evaluate_cost(measured, budget, DEFAULT_REGRESSION_THRESHOLD);
    CostVerdict {
        measured: gate.measured,
        budget: gate.budget,
        threshold: gate.threshold,
        ceiling: gate.ceiling,
        passed: gate.passed,
        detail: gate.detail(),
    }
}

/// Measure one verb's overhead lane and gate it against `budget`: fold
/// fresh-session sweeps until every gate holds or [`OVERHEAD_ATTEMPTS`] are
/// spent, and report the last fold.
async fn overhead_lane(
    sweep: OverheadSweep<'_>,
    budget: OverheadBudget,
) -> Result<OverheadLane, Box<dyn std::error::Error>> {
    let mut mins = measure_overhead(sweep).await?;
    let mut attempts = 1;
    loop {
        let fits = mins.fits(sweep.rows)?;
        let measured = fits.overhead();
        let lane = OverheadLane {
            rows: sweep.rows.to_vec(),
            rounds: OVERHEAD_ROUNDS,
            attempts,
            plan_rows_per_s: sweep
                .rows
                .iter()
                .zip(&mins.plan_ms)
                .next_back()
                .map_or_else(
                    || Measurement::not_yet_measured("rows_per_s"),
                    |(&rows, &ms)| Measurement::measured(per_second(rows, ms), "rows_per_s"),
                ),
            per_row_ratio: cost_verdict(measured.per_row_ratio, budget.per_row_ratio),
            fixed_rows: cost_verdict(measured.fixed_rows, budget.fixed_rows),
            two_term: TwoTermVerdict {
                relative_residual_rms: fits.plan.relative_residual_rms,
                ceiling: MAX_TWO_TERM_RESIDUAL,
                passed: fits.plan.relative_residual_rms <= MAX_TWO_TERM_RESIDUAL,
            },
            plan: fits.plan,
            direct: fits.direct,
            plan_min_ms: mins.plan_ms.clone(),
            direct_min_ms: mins.direct_ms.clone(),
        };
        if lane.passed() || attempts == OVERHEAD_ATTEMPTS {
            return Ok(lane);
        }
        mins = mins.fold(measure_overhead(sweep).await?);
        attempts += 1;
    }
}

impl<'a> OverheadSweep<'a> {
    /// The sweep the gate and the rebuild both serve: the engine's default
    /// serve shape on `Device::Cpu`.
    fn gated(verb: Verb, corpus_seed: u64, rows: &'a [usize]) -> Self {
        Self {
            verb,
            corpus_seed,
            rows,
            plan_shape: ServeShape::cpu(),
        }
    }
}

/// The compute-thread count this process was pinned to — `RAYON_NUM_THREADS`,
/// the variable the compute pool sizes itself from. Unset is refused: an
/// unpinned pool is the box's core count, which no committed budget was
/// measured under.
fn pinned_compute_threads() -> Result<usize, Box<dyn std::error::Error>> {
    let raw = std::env::var("RAYON_NUM_THREADS").map_err(|_| {
        "the overhead gate needs RAYON_NUM_THREADS set: a budget holds only under the \
         compute-thread count it was measured at"
    })?;
    match raw.parse::<usize>() {
        Ok(threads) if threads >= 1 => Ok(threads),
        _ => Err(format!("RAYON_NUM_THREADS={raw:?} is not a thread count").into()),
    }
}

/// Run the model-inference tier against the committed spec: serve both verbs over
/// their committed bundles TWICE on this box, fold both determinism gates, and
/// measure and gate both verbs' serving overhead.
///
/// Each digest gate asserts same-machine determinism — serve twice on the running
/// box and the two digests agree (the served output is `f32`, so the committed
/// digest rides as a same-box reference, not a cross-machine constant). Each
/// overhead lane is the same-process plan-over-direct ratio the module doc
/// describes — NOT the scaling SLO, which is the cookbook (A) full-scale value.
pub async fn run(
    spec: &ModelInferenceSpec,
) -> Result<ModelInferenceTier, Box<dyn std::error::Error>> {
    let compute_threads = pinned_compute_threads()?;
    if compute_threads != spec.compute_threads {
        return Err(format!(
            "the overhead budgets were measured at RAYON_NUM_THREADS={}, this run is at \
             {compute_threads}: the ratio is not comparable across thread postures",
            spec.compute_threads
        )
        .into());
    }
    let rows = build_corpus(spec.corpus_seed, spec.row_count);

    // Each digest is folded over a fresh session, twice, so the gate is
    // reload-invariant determinism and not a warm cache agreeing with itself.
    let embed_id = local_model_id(&ModelInferenceSpec::embed_model_dir())?;
    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
    let (embed_digest_first, _ms, _n) = serve_embed(&session, &embed_id).await?;
    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
    let (embed_digest_second, _ms, _n) = serve_embed(&session, &embed_id).await?;

    let infer_id = local_model_id(&ModelInferenceSpec::classifier_model_dir())?;
    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
    let (infer_digest_first, _ms, _n) = serve_infer(&session, &infer_id, &spec.target_keys).await?;
    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
    let (infer_digest_second, _ms, _n) =
        serve_infer(&session, &infer_id, &spec.target_keys).await?;

    Ok(ModelInferenceTier {
        targets: spec.target_keys.len(),
        embed_overhead: overhead_lane(
            OverheadSweep::gated(Verb::Embed, spec.corpus_seed, &spec.overhead_rows),
            spec.embed_overhead,
        )
        .await?,
        embed_digest: DeterminismGate::new(
            embed_digest_first,
            embed_digest_second,
            spec.embed_digest.clone(),
        ),
        infer_overhead: overhead_lane(
            OverheadSweep::gated(Verb::Infer, spec.corpus_seed, &spec.overhead_rows),
            spec.infer_overhead,
        )
        .await?,
        infer_digest: DeterminismGate::new(
            infer_digest_first,
            infer_digest_second,
            spec.infer_digest.clone(),
        ),
    })
}

/// Whether every gate held — the verdict the subcommand maps to its exit code:
/// both verbs were deterministic across two same-machine serves AND both verbs'
/// overhead lanes held ([`OverheadLane::passed`]).
pub fn gates_passed(tier: &ModelInferenceTier) -> bool {
    tier.embed_digest.passed
        && tier.infer_digest.passed
        && tier.embed_overhead.passed()
        && tier.infer_overhead.passed()
}

/// Re-derive the committed spec from a fresh serve over the committed bundles:
/// regenerate the corpus, serve both verbs, and record both digests and both
/// overhead budgets as measured. The one-shot that writes
/// `baselines/model_inference.json`; CI only ever loads and re-serves.
///
/// The bundles themselves are the engine's own shared tiny fixtures, REFERENCED
/// from `cookbook/fixtures/` (not copied) — they are not regenerated here (there
/// is no training step; they are pre-built tiny models).
pub async fn rebuild_spec(
    params: ModelInferenceParams<'_>,
) -> Result<ModelInferenceSpec, Box<dyn std::error::Error>> {
    let rows = build_corpus(params.corpus_seed, params.row_count);
    let target_keys: Vec<String> = rows
        .iter()
        .take(params.target_count)
        .map(|r| r.id.clone())
        .collect();

    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
    let embed_id = local_model_id(&ModelInferenceSpec::embed_model_dir())?;
    let (embed_digest, _ms, _n) = serve_embed(&session, &embed_id).await?;

    let (session, _dir) = corpus_session(&rows, ServeShape::cpu()).await?;
    let infer_id = local_model_id(&ModelInferenceSpec::classifier_model_dir())?;
    let (infer_digest, _ms, _n) = serve_infer(&session, &infer_id, &target_keys).await?;

    let sweep = |verb| OverheadSweep::gated(verb, params.corpus_seed, params.overhead_rows);
    Ok(ModelInferenceSpec {
        compute_threads: pinned_compute_threads()?,
        row_count: params.row_count,
        corpus_seed: params.corpus_seed,
        target_keys,
        embed_digest,
        infer_digest,
        overhead_rows: params.overhead_rows.to_vec(),
        embed_overhead: settled_overhead(sweep(Verb::Embed)).await?,
        infer_overhead: settled_overhead(sweep(Verb::Infer)).await?,
    })
}

/// The generation parameters a rebuild draws the committed spec from — the corpus
/// shape, how many targets the digests fold over, and the overhead sweep.
#[derive(Debug, Clone, Copy)]
pub struct ModelInferenceParams<'a> {
    /// The synthetic corpus row count.
    pub row_count: usize,
    /// The corpus generation seed.
    pub corpus_seed: u64,
    /// How many of the corpus rows the digests fold over.
    pub target_count: usize,
    /// The row counts the overhead sweep serves, ascending.
    pub overhead_rows: &'a [usize],
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`corpus_sha256`] is a real content hash, not a constant: a different
    /// seed OR a different row count each move the digest — the two scalar
    /// inputs [`GpuInferenceTier::corpus_seed`]/`::row_count` already cover
    /// as separate identity fields, folded into this one for the
    /// belt-and-suspenders reason this function's own doc states.
    #[test]
    fn corpus_sha256_reacts_to_seed_and_row_count() {
        let base = corpus_sha256(0, 4);
        assert_ne!(
            base,
            corpus_sha256(1, 4),
            "a different seed must move the digest"
        );
        assert_ne!(
            base,
            corpus_sha256(0, 8),
            "a different row_count must move the digest"
        );
        assert_eq!(
            base,
            corpus_sha256(0, 4),
            "the SAME (seed, row_count) must reproduce the SAME digest deterministically"
        );
    }

    /// A hand-computed hash over a manually-perturbed sentence array,
    /// entirely bypassing `corpus_sha256`'s own real selection logic, would
    /// be STRUCTURALLY BLIND to the class of bug this identity field actually
    /// exists to catch — a changed SELECTION/ROTATION RULE, not merely a
    /// changed byte layout at the same selected indices. This constructs
    /// an ALTERNATIVE resolved corpus using a DELIBERATELY WRONG rotation
    /// formula (`% 4` in place of the real `% SENTENCES.len()` == `% 8`)
    /// and proves the REAL
    /// `corpus_sha256_of_rows` primitive discriminates between the two: if
    /// some FUTURE bug in `build_corpus`'s own selection formula silently
    /// changed which sentence a row index draws, this digest would move
    /// for the SAME `(corpus_seed, row_count)` input.
    #[test]
    fn corpus_sha256_reacts_to_a_changed_selection_rule_not_just_a_changed_byte_layout() {
        let corpus_seed = 0u64;
        let row_count = 8usize;

        let real_rows = build_corpus(corpus_seed, row_count);
        let real_digest = corpus_sha256_of_rows(&real_rows);
        assert_eq!(
            real_digest,
            corpus_sha256(corpus_seed, row_count),
            "corpus_sha256 must equal corpus_sha256_of_rows(build_corpus(..)) -- the public \
             wrapper must actually call the real selection rule, never bypass it"
        );

        // The alternative selection rule: `% 4` in place of the real
        // `% SENTENCES.len()` (== `% 8`) -- genuinely selects a DIFFERENT
        // sentence starting at row index 4 (row 4 draws SENTENCES[0]
        // under `% 4` vs SENTENCES[4] under the real `% 8`).
        let alternative_rows: Vec<Row> = (0..row_count)
            .map(|i| {
                let idx = (corpus_seed as usize).wrapping_add(i) % 4;
                Row {
                    id: format!("row_{i}"),
                    text: SENTENCES[idx].to_string(),
                }
            })
            .collect();
        let alternative_digest = corpus_sha256_of_rows(&alternative_rows);

        assert_ne!(
            real_digest, alternative_digest,
            "corpus_sha256 must react to a changed SELECTION RULE (here: % 4 in place of the real \
             % SENTENCES.len()), not merely to a changed byte layout at the same selected indices \
             -- if this assertion fails, the digest is structurally blind to a rotation-rule \
             regression"
        );
    }

    /// The committed spec is well-formed: a positive corpus, a non-empty target
    /// set within the corpus, both digests of the FNV width, a sweep that can
    /// separate two cost terms, and budgets that are possible costs.
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
        assert!(spec.compute_threads >= 1, "budgets need a thread posture");
        assert!(
            spec.overhead_rows.len() >= 2 && spec.overhead_rows.windows(2).all(|w| w[0] < w[1]),
            "the overhead sweep needs at least two ascending row counts to separate a fixed \
             from a per-row cost: {:?}",
            spec.overhead_rows
        );
        for (budget, name) in [
            (spec.embed_overhead, "embed"),
            (spec.infer_overhead, "infer"),
        ] {
            assert!(
                [budget.per_row_ratio, budget.fixed_rows]
                    .iter()
                    .all(|cost| *cost > 0.0 && cost.is_finite()),
                "{name}: a budget is a positive finite cost: {budget:?}"
            );
        }
    }

    /// The referenced shared model fixtures are present and complete: each verb's
    /// fixture carries the three files the resolver's local branch loads. A missing
    /// file would make the serve fail to even resolve, so this guards the referenced
    /// fixtures' completeness.
    #[test]
    fn referenced_model_fixtures_are_present() {
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

    /// The portable determinism gate (DIGESTS-CLEAR direction): serving each verb
    /// over its committed bundle twice on THIS box (each a fresh session + serve)
    /// produces the byte-identical digest. This is the engine's real contract —
    /// same-machine, reload-invariant byte-identity for an `f32` serve — and is true
    /// on any CI box by construction, unlike a committed cross-machine constant,
    /// which `f32` SIMD/FMA/BLAS differences would spuriously break.
    #[tokio::test(flavor = "multi_thread")]
    async fn reserve_is_deterministic_on_this_machine() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        let embed_one = fold_embed_digest(&spec, &ModelInferenceSpec::embed_model_dir())
            .await
            .expect("first embed fold");
        let embed_two = fold_embed_digest(&spec, &ModelInferenceSpec::embed_model_dir())
            .await
            .expect("second embed fold");
        assert_eq!(
            embed_one, embed_two,
            "two same-machine embed serves disagreed — embed is not deterministic on this box"
        );

        let infer_one = fold_infer_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("first infer fold");
        let infer_two = fold_infer_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("second infer fold");
        assert_eq!(
            infer_one, infer_two,
            "two same-machine infer serves disagreed — infer is not deterministic on this box"
        );
    }

    /// The teeth, GATE-FAILS direction for the EMBED verb (an assertion must
    /// be able to fail). Serving the *classifier* bundle through the embed verb
    /// produces a different embedding (a different model, different weights and
    /// hidden geometry), so the embed digest moves off the IN-PROCESS baseline (the
    /// committed embed bundle served on THIS box) — proving the gate reacts to the
    /// served model. Both serves run on the same machine, so the teeth are portable.
    #[tokio::test(flavor = "multi_thread")]
    async fn embed_over_a_different_model_changes_the_digest() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        let baseline = fold_embed_digest(&spec, &ModelInferenceSpec::embed_model_dir())
            .await
            .expect("committed embed serve");
        let perturbed = fold_embed_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("perturbed embed serve");
        assert_ne!(
            perturbed, baseline,
            "embedding through a different model must move the embed digest"
        );
    }

    /// The teeth, GATE-FAILS direction for the INFER verb: serving over a
    /// perturbed corpus (a different seed rotates which sentences the rows draw,
    /// so the classifier scores different text) produces a different infer digest
    /// than the IN-PROCESS baseline (the committed corpus served on THIS box). This
    /// proves the digest reacts to the served *input*, the regression a
    /// silently-changed tokenization or input-column wiring would cause. Both serves
    /// run on the same machine, so the teeth are portable.
    #[tokio::test(flavor = "multi_thread")]
    async fn infer_over_perturbed_input_changes_the_digest() {
        let spec =
            ModelInferenceSpec::load().expect("baselines/model_inference.json must be present");
        let baseline = fold_infer_digest(&spec, &ModelInferenceSpec::classifier_model_dir())
            .await
            .expect("committed infer serve");

        let mut perturbed_spec = spec.clone();
        perturbed_spec.corpus_seed = spec.corpus_seed.wrapping_add(1);
        let perturbed =
            fold_infer_digest(&perturbed_spec, &ModelInferenceSpec::classifier_model_dir())
                .await
                .expect("perturbed infer serve");
        assert_ne!(
            perturbed, baseline,
            "classifying different input text must move the infer digest"
        );
    }

    /// The teeth, GATE-FAILS direction, on the regression the per-row gate exists
    /// for: a plan that lost its batching. The budget is measured HERE, under
    /// this test's own box and thread posture, from the healthy plan; then
    /// `batch_size = 1` makes the plan forward every row alone — the real
    /// engine path, really degraded, not an inflated budget — while the direct
    /// leg keeps the default chunks. The per-row cost must leave its ceiling.
    /// A tenth of the healthy fixed cost, as a budget, must fail the per-call
    /// gate the same way: neither verdict is vacuous.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_plan_that_lost_its_batching_fails_the_gate_a_healthy_plan_set() {
        // Two points are enough to separate the two terms, and keep a plan
        // forwarding one row at a time affordable in a debug build.
        let healthy = OverheadSweep::gated(Verb::Embed, 11, &[16, 256]);
        let budget = measure_overhead(healthy)
            .await
            .and_then(|mins| Ok(mins.fits(healthy.rows)?.overhead()))
            .expect("healthy sweep");

        let unbatched = overhead_lane(
            OverheadSweep {
                plan_shape: ServeShape {
                    batch_size: 1,
                    ..ServeShape::cpu()
                },
                ..healthy
            },
            budget,
        )
        .await
        .expect("unbatched lane");
        assert!(
            !unbatched.per_row_ratio.passed,
            "a plan forwarding one row at a time must fail the per-row gate: {}",
            unbatched.per_row_ratio.detail
        );

        let starved = overhead_lane(
            healthy,
            OverheadBudget {
                fixed_rows: budget.fixed_rows / 10.0,
                ..budget
            },
        )
        .await
        .expect("starved lane");
        assert!(!starved.fixed_rows.passed, "{}", starved.fixed_rows.detail);
    }
}
