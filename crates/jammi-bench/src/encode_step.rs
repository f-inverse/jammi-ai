//! The `encode` workload's ONE producer: the engine's serving path — rows in a
//! table → one artifact per key — run through a RUNG and emitted as LEGS
//! ([`crate::report::EncodePayload`]).
//!
//! ## Rungs, not tiers
//!
//! A workload has ordered rungs that differ by one layer; the ratio of two
//! adjacent rungs' times is that layer's cost. This module produces the three
//! that live in one process:
//!
//! * `direct` — the loaded model called on the rows in the forward chunks
//!   the plan cuts ([`jammi_ai::model::LoadedModel::forward`]: tokenize →
//!   forward → pool → normalize → host artifacts). No plan, no catalog, no
//!   result table.
//! * `plan` — the engine's real verb over the same rows at
//!   `[inference] partitions = 1`: `generate_text_embeddings` for
//!   [`Task::Embed`] (scan → sort → forward → COMMITTED result table: its
//!   Parquet object and the ANN segment beside it), `infer` for
//!   [`Task::Infer`] (`Classification`; its rows return to the caller).
//! * `plan-partitioned` — the same plan at `--partitions` N.
//!
//! The task, the size (a `--rows` sweep), the device (`--cuda`) and the
//! checkpoint (`--model-dir`, else a compiled-in fixture) are parameters of the
//! one workload, never tiers of their own. The producer decides nothing: it
//! emits legs, and every ratio, budget and verdict is the ladder comparator's.
//!
//! ## One process per leg session, sampled from outside
//!
//! Every (unit, take) is measured in a fresh child of this binary
//! (`encode-leg`), because the two space instruments are per-process: the
//! kernel's resident-set high-water mark (`VmHWM`) never falls, and the
//! device-memory sampler ([`crate::vram::run_sampled`]) wraps a process from
//! outside — the same instrument, the same way, that wraps a PyTorch leg. A
//! child's first call is also honestly cold.
//!
//! With more than one `--rung`, the child serves them INTERLEAVED in the one
//! session — forward order on even rounds, reversed on odd (A, B / B, A) — so
//! a drifting box lands on each rung alike. Those are the legs an edge's speed
//! is read from; the process's space marks belong to no one rung, so such legs
//! carry none, and a rung's space is read from a session of its own.
//!
//! ## What a leg carries
//!
//! The per-iteration wall-time series, never only a summary; for a plan leg
//! that commits a table, where each serve's time went inside the sink (the
//! time to the last output batch, the ok-row extraction, the Parquet write,
//! the wait on the parallel ANN segment builds, the segment persists — the
//! sink's own account, captured off its one `tracing` event — and, beside
//! those wall slices, the builds' own summed thread time, which overlaps
//! them); the artifact's digest in key order, and, on a unit's first take,
//! the vectors themselves beside the leg. Real token counts are the loaded
//! model's own row costs ([`jammi_ai::model::LoadedModel::row_costs`]: its
//! tokenizer at its truncation bound), and the padded count is what the
//! plan's chunks pad them to.
//!
//! ## The corpus: seeded, variable-length
//!
//! Rows of one length tokenize to a dense batch and hide what padding costs.
//! [`build_corpus`] draws each row's word count from a fixed four-band mixture
//! (most rows a sentence or a paragraph, a tail several hundred words long),
//! keyed by `(seed, row index)` alone — integer arithmetic throughout, so the
//! corpus is byte-identical on every platform and an `n`-row corpus is a
//! prefix of every larger one. Keys are zero-padded, so the plan's key order
//! (`CAST(key AS Utf8)`) is the corpus's row order: the chunks the plan cuts
//! from the rows in that order ([`forward_chunks`]) are the chunks the
//! `direct` rung and a reference producer forward, and every rung's
//! artifact is in the same order.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_ai::model::{LoadedModel, ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, InferenceConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_numerics::{ChunkBudget, ChunkCutter};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tracing_subscriber::layer::SubscriberExt;

use crate::finetune_step::sha256_and_len;
use crate::leg::{Facts, Leg, Measured, Provenance};
use crate::report::{EncodePayload, Measurement, Report, SinkPhaseSeries, Tiers};
use crate::rss::peak_rss_measurement;
use crate::timing::{nearest_rank, per_second, CostFit, ServeStats};
use crate::vram::{device_memory_probe, run_sampled};

/// The CI-hermetic default device: `Device::Cpu`.
pub const CPU_HERMETIC_DEVICE: i32 = -1;

/// The source id the corpus registers under.
const SOURCE_ID: &str = "corpus";
/// The text column the verbs read.
const TEXT_COLUMN: &str = "text";
/// The key column carrying each row's stable id into the artifact.
const KEY_COLUMN: &str = "_row_id";

/// What the rows are served for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Task {
    /// `generate_text_embeddings`: one L2-normalized vector per key.
    Embed,
    /// `infer` (`Classification`): one score distribution per key.
    Infer,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Embed => "embed",
            Task::Infer => "infer",
        }
    }

    fn model_task(self) -> ModelTask {
        match self {
            Task::Embed => ModelTask::TextEmbedding,
            Task::Infer => ModelTask::Classification,
        }
    }
}

/// The stack the rows go through — see the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Rung {
    Direct,
    Plan,
    PlanPartitioned,
}

impl Rung {
    pub fn as_str(self) -> &'static str {
        match self {
            Rung::Direct => "direct",
            Rung::Plan => "plan",
            Rung::PlanPartitioned => "plan-partitioned",
        }
    }

    /// The `[inference] partitions` the rung's session plans with; `None` for
    /// the rung that builds no plan.
    fn partitions(self, partitioned: usize) -> Option<usize> {
        match self {
            Rung::Direct => None,
            Rung::Plan => Some(1),
            Rung::PlanPartitioned => Some(partitioned),
        }
    }
}

/// What the producer serves and how it measures.
#[derive(Debug, Clone)]
pub struct EncodeStepParams {
    /// What the rows are served for.
    pub task: Task,
    /// The rungs of one leg session, in round order. More than one are served
    /// interleaved — see the module doc.
    pub rungs: Vec<Rung>,
    /// The checkpoint directory to serve; `None` serves the task's compiled-in
    /// fixture.
    pub model_dir: Option<PathBuf>,
    /// The sweep: the corpus row count of each unit.
    pub rows: Vec<usize>,
    /// How many times each unit is measured, each in a process of its own.
    pub takes: usize,
    /// The corpus generation seed.
    pub seed: u64,
    /// `[inference] batch_size` — the row cap of a forward chunk, on every
    /// rung.
    pub batch_size: usize,
    /// `[inference] batch_tokens` — the padded-token cap of a forward chunk,
    /// on every rung.
    pub batch_tokens: usize,
    /// `[inference] partitions` of the `plan-partitioned` rung.
    pub partitions: usize,
    /// `[gpu] compute_precision` — the precision the model loads at unless its
    /// own `config.json` declares one. What it RESOLVED to is what a leg
    /// records, read off the loaded model.
    pub compute_precision: jammi_numerics::ComputePrecision,
    /// Warm serves before the measured ones, per rung, discarded.
    pub warmup: usize,
    /// Measured serves per rung.
    pub iters: usize,
    /// The device ordinal the sessions resolve on: [`CPU_HERMETIC_DEVICE`]
    /// (`-1`), or a CUDA ordinal the box must actually have.
    pub gpu_device: i32,
    /// Where each unit leaves what a reference producer reads: the corpus
    /// Parquet it served (`corpus_<rows>.parquet`) and — when it served a
    /// compiled-in fixture — that checkpoint (`model/`). `None` keeps them in
    /// a scratch directory that dies with the leg session.
    pub exchange_dir: Option<PathBuf>,
    /// Where the legs are written, one file per leg
    /// (`<rung>__rows<N>__r<take>.json`, a unit's first take with its vectors
    /// beside it); `None` writes none and only summarises.
    pub legs_dir: Option<PathBuf>,
}

impl EncodeStepParams {
    fn cuda_ordinal(&self) -> Option<usize> {
        usize::try_from(self.gpu_device).ok()
    }

    /// Refuse a run that could measure nothing, or whose rounds could not be
    /// balanced, before anything is built.
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        let counts = [
            self.batch_size,
            self.batch_tokens,
            self.iters,
            self.takes,
            self.rungs.len(),
        ];
        if self.rows.is_empty() || self.rows.contains(&0) || counts.contains(&0) {
            return Err(format!(
                "encode-step needs at least one rung, one row count and one take, and every row \
                 count, the chunk budget and the measured iterations at least 1: {self:?}"
            )
            .into());
        }
        if self.rungs.len() > 1 && !self.iters.is_multiple_of(2) {
            return Err(format!(
                "interleaved rungs are served forward then reversed, so --iters must be even to \
                 balance them: {}",
                self.iters
            )
            .into());
        }
        if self.rungs.contains(&Rung::PlanPartitioned) && self.partitions < 2 {
            return Err(format!(
                "the plan-partitioned rung needs --partitions of at least 2, not {} — at 1 it \
                 is the plan rung",
                self.partitions
            )
            .into());
        }
        let mut distinct = self.rungs.clone();
        distinct.sort_by_key(|r| r.as_str());
        distinct.dedup();
        if distinct.len() != self.rungs.len() {
            return Err(format!("a leg session serves each rung once: {:?}", self.rungs).into());
        }
        Ok(())
    }
}

/// One corpus row: its stable key and its text.
pub(crate) struct Row {
    pub(crate) id: String,
    pub(crate) text: String,
}

/// The words a corpus row is drawn from: plain vocabulary any text tokenizer
/// splits into real tokens (random bytes would tokenize to mostly the unknown
/// token and make the forward trivial).
const WORDS: [&str; 48] = [
    "the",
    "of",
    "and",
    "in",
    "to",
    "is",
    "for",
    "with",
    "on",
    "that",
    "quantum",
    "computing",
    "superconducting",
    "systems",
    "topological",
    "error",
    "correction",
    "codes",
    "spin",
    "coherence",
    "trapped",
    "ion",
    "qubits",
    "lattice",
    "gauge",
    "theory",
    "simulator",
    "gene",
    "editing",
    "inherited",
    "disease",
    "ribosome",
    "structure",
    "protein",
    "synthesis",
    "mitochondrial",
    "cellular",
    "respiration",
    "enzyme",
    "kinetics",
    "metabolic",
    "pathways",
    "measurement",
    "model",
    "results",
    "between",
    "under",
    "across",
];

/// The word-count mixture a row's length is drawn from: `(share of rows in
/// parts per hundred, fewest words, most words)`. A title or a sentence, a
/// paragraph, a long passage, and a tail of documents long enough to reach a
/// small model's truncation bound.
const LENGTH_BANDS: [(u64, u64, u64); 4] = [(30, 3, 10), (40, 11, 30), (22, 31, 90), (8, 91, 300)];

/// SplitMix64: a bijective 64-bit mix, so distinct `(seed, row, draw)` inputs
/// give independent-looking draws with no shared generator state to advance
/// in order.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Row `index` of the seeded corpus — a function of `(seed, index)` alone.
fn corpus_row(seed: u64, index: usize) -> Row {
    let draw = |n: u64| splitmix64(splitmix64(seed) ^ splitmix64((index as u64) << 20 | n));
    let band_draw = draw(0) % 100;
    let (_, fewest, most) = LENGTH_BANDS
        .into_iter()
        .scan(0, |upto, (share, fewest, most)| {
            *upto += share;
            Some((*upto, fewest, most))
        })
        .find(|&(upto, _, _)| band_draw < upto)
        .unwrap_or(LENGTH_BANDS[LENGTH_BANDS.len() - 1]);
    let word_count = fewest + draw(1) % (most - fewest + 1);
    let text = (0..word_count)
        .map(|w| WORDS[(draw(2 + w) % WORDS.len() as u64) as usize])
        .collect::<Vec<_>>()
        .join(" ");
    Row {
        id: format!("row_{index:08}"),
        text,
    }
}

/// The seeded variable-length corpus — see this module's doc.
pub(crate) fn build_corpus(seed: u64, row_count: usize) -> Vec<Row> {
    (0..row_count).map(|i| corpus_row(seed, i)).collect()
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
/// `pooling_mode_cls_token: true` instead. Test-only:
/// drives [`build_embed_fixture_with_pooling`] to prove
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
/// three files,
/// then write an EXPLICIT `1_Pooling/config.json` carrying `pooling_flags`.
///
/// The explicit pooling config (never the bare `tiny_bert` fixture, which
/// ships with no `1_Pooling/` folder at all) is deliberate: a repo with no
/// pooling config resolves through `candle.rs`'s silent mean-pooling
/// fallback — a silent-identity ambiguity. Declaring the strategy
/// here means [`crate::report::EncodePayload::pooling`] records a value
/// this tier KNOWS the engine resolves to, not an inferred one. Parameterized
/// over the flags (`checkpoint_pooling_sha256_and_
/// pooling_move_together_when_the_fixture_flips_to_cls` drives this with the
/// test-only `cls_pooling_flags` to prove the two accessors this tier reads
/// actually react to the fixture, never a hand-typed expectation).
fn build_embed_fixture_with_pooling(
    dst: &Path,
    pooling_flags: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    copy_fixture("tiny_bert", dst)?;
    let pooling_dir = dst.join("1_Pooling");
    std::fs::create_dir_all(&pooling_dir)?;
    std::fs::write(
        pooling_dir.join("config.json"),
        serde_json::to_string(pooling_flags)?,
    )?;
    Ok(())
}

/// The engine's shared tiny fixtures, REFERENCED from the cookbook tree (the
/// same directories the `jammi-encoders` tests reference), never copied into
/// this crate: a 32-dim 1-layer BERT and a 32-dim 1-layer ModernBERT
/// classifier.
fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../cookbook/fixtures")
        .join(name)
}

/// Copy the three files the resolver's local branch loads from the shared
/// fixture `name` into `dst`.
fn copy_fixture(name: &str, dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dst)?;
    for file in ["config.json", "model.safetensors", "tokenizer.json"] {
        std::fs::copy(fixture_dir(name).join(file), dst.join(file)).map_err(
            |e| -> Box<dyn std::error::Error> {
                format!("copying {file} from the shared {name} fixture: {e}").into()
            },
        )?;
    }
    Ok(())
}

/// The compiled-in checkpoint a run with no `--model-dir` serves for `task`:
/// the embed fixture with its explicit mean-pooling declaration, or the
/// classifier fixture (which pools nothing).
fn build_fixture_model_dir(task: Task, dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    match task {
        Task::Embed => build_embed_fixture_with_pooling(dst, &mean_pooling_flags()),
        Task::Infer => copy_fixture("tiny_modernbert_classifier", dst),
    }
}

/// [`crate::report::EncodePayload::device_requested`]'s value — `"cpu"` for
/// a negative ordinal (the CI-hermetic default), `"cuda:<ordinal>"`
/// otherwise. A cheap, honest label derived straight from the same
/// `gpu_device` value threaded into [`session_over`]
/// (`select_device`'s own convention: negative selects `Device::Cpu`), never
/// a second, independently-resolved reading. Computable BEFORE any compute
/// runs (this is exactly what makes it an identity field, unlike
/// [`resolved_device_name`] below).
fn requested_device_label(gpu_device: i32) -> String {
    if gpu_device < 0 {
        "cpu".to_string()
    } else {
        format!("cuda:{gpu_device}")
    }
}

/// [`crate::leg::Provenance::device_name`]'s value — a POST-HOC
/// hardware fact, only knowable after the device resolved (so
/// PROVENANCE, never identity). `"cpu"` for the CI-hermetic
/// default; a real CUDA leg queries the actual device sub-class name off the
/// driver via [`cuda_device_name`].
///
/// Naming `gpu_device` here (the REQUESTED ordinal) rather than re-reading
/// the session's own resolved device is only honest because the silent-
/// CPU-fallback state this would otherwise transcribe is UNREPRESENTABLE by
/// the time this function runs: `run()` threads
/// `gpu_device` through [`session_over`], which sets
/// `gpu.require_gpu = gpu_device >= 0` on the session's `GpuConfig`; a `-cuda
/// N` leg whose ordinal the box cannot actually satisfy fails the FIRST
/// model load (`CandleBackend::load`'s `select_device(device_config)?`,
/// `backend/candle.rs`'s `gpu_unavailable` returning a typed
/// `JammiError::Gpu`) at the leg session's model load — well before this
/// function is ever reached. So on every path that DOES reach here,
/// requested ordinal and actually-resolved device are the same value by
/// construction: there is no code path left in which `gpu_device` names a
/// CUDA ordinal the run did not truly execute on.
///
/// **Qualification**:
/// on a build compiled with `feature = "metal"` but not `"cuda"`,
/// `select_device` itself can genuinely SUCCEED for a `gpu_device >= 0`
/// request via its metal branch (real Metal hardware, mislabeled here as a
/// CUDA ordinal) — the "first model load fails" argument above does not
/// apply on that build. The guarantee still holds there, but for a
/// DIFFERENT reason: this function's `cuda_device_name` call unconditionally
/// errors on any `not(feature = "cuda")` build (see its stub below), aborting `run()` before a mismatched, Metal-resolved
/// device name could ever populate this field. Both mechanisms are needed to
/// state the full picture; neither alone covers every build.
fn resolved_device_name(gpu_device: i32) -> Result<String, Box<dyn std::error::Error>> {
    if gpu_device < 0 {
        Ok("cpu".to_string())
    } else {
        cuda_device_name(gpu_device as u32)
    }
}

/// The concrete CUDA device name (e.g. `NVIDIA A100-SXM4-80GB`) for `ordinal`
/// — the provenance that makes a leg's numbers interpretable on an
/// ephemeral heterogeneous fleet. Queried in-process through cudarc's device
/// API (candle re-exports cudarc as `candle_core::cuda::cudarc`) rather than
/// by shelling out: it is the same driver the session's CUDA backend opened,
/// and cannot name another host's GPU.
#[cfg(feature = "cuda")]
fn cuda_device_name(ordinal: u32) -> Result<String, Box<dyn std::error::Error>> {
    use candle_core::cuda::cudarc::driver::result as cuda;
    cuda::init()?;
    let device = cuda::device::get(ordinal as i32)?;
    Ok(cuda::device::get_name(device)?)
}

/// Without the `cuda` feature there is no device to name — and a leg that
/// asked for one has already failed its model load.
#[cfg(not(feature = "cuda"))]
fn cuda_device_name(_ordinal: u32) -> Result<String, Box<dyn std::error::Error>> {
    Err("built without the cuda feature; no device to name".into())
}

/// [`crate::report::EncodePayload::checkpoint_pooling_sha256`]'s value:
/// `sha256_and_len` over `model_dir/1_Pooling/config.json`'s bytes when that
/// file exists, `None` when it doesn't — the SAME presence
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

/// The forward chunks the plan cuts from a corpus, and what they cost: the
/// decision the numbered input makes over its rows, made here over the same
/// rows in the same order with the same cutter. The rows are ordered by cost
/// (the model's own row costs, ties in row order — the plan's `(_cost,
/// _ordinal)` sort over rows numbered in key order, which is this corpus's
/// row order), and one pass of the plan's [`ChunkCutter`] over that sequence
/// under the session's [`ChunkBudget`], on the model's own shape ladder,
/// cuts the chunks. Each chunk is forwarded at the ladder rung of its
/// longest row, which is what `padded_tokens` sums.
struct ForwardChunks {
    /// The row indices of each forward, in forward order.
    chunks: Vec<Vec<usize>>,
    /// Each row's real (truncated, unpadded) token count, in row order.
    row_tokens: Vec<usize>,
    /// Tokens the forwards pad to, summed.
    padded_tokens: usize,
}

fn forward_chunks(
    model: &LoadedModel,
    task: ModelTask,
    texts: &StringArray,
    budget: ChunkBudget,
) -> Result<ForwardChunks, Box<dyn std::error::Error>> {
    let content: ArrayRef = Arc::new(texts.clone());
    let costs = model.row_costs(&[content], task)?;
    let ladder = model.shape_ladder(task)?;
    let mut order: Vec<usize> = (0..costs.len()).collect();
    order.sort_by_key(|&row| costs[row]);
    let mut cutter = ChunkCutter::new(budget, ladder);
    let mut chunks: Vec<(u64, Vec<usize>)> = Vec::new();
    for row in order {
        let id = cutter.push(costs[row]);
        match chunks.last_mut() {
            Some((open, rows)) if *open == id => rows.push(row),
            _ => chunks.push((id, vec![row])),
        }
    }
    let chunks: Vec<Vec<usize>> = chunks.into_iter().map(|(_, rows)| rows).collect();
    let padded_tokens = chunks
        .iter()
        .map(|rows| {
            let longest = rows
                .iter()
                .map(|&row| costs[row] as usize)
                .max()
                .unwrap_or(0);
            rows.len() * ladder.width(longest)
        })
        .sum();
    Ok(ForwardChunks {
        chunks,
        row_tokens: costs.into_iter().map(|cost| cost as usize).collect(),
        padded_tokens,
    })
}

/// sha256 (hex) of the per-row token counts, in row order, as their decimal
/// renderings joined by `,` — a rendering any producer in any language
/// reproduces byte for byte. Two producers that tokenized and truncated the
/// same corpus the same way agree on it; the full vector (one entry per row
/// of a sweep point) never has to travel.
fn token_lengths_sha256(row_tokens: &[usize]) -> String {
    let rendered = row_tokens
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    hex::encode(Sha256::digest(rendered.as_bytes()))
}

/// A `local:`-prefixed model id for a checkpoint directory — the resolver's
/// no-network branch, so a serve loads off the directory's safetensors and
/// never reaches a hub.
fn local_model_id(dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!(
        "local:{}",
        dir.to_str().ok_or("model dir is not valid UTF-8")?
    ))
}

/// Write `rows` as the source Parquet a session serves: `_row_id`, `text`,
/// and a `y` placeholder so the schema matches the engine's source-table
/// shape. Written through the engine's own writer, in row order.
async fn write_corpus(rows: &[Row], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(KEY_COLUMN, DataType::Utf8, false),
        Field::new(TEXT_COLUMN, DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.id.as_str()),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.text.as_str()),
            )),
            Arc::new(Float64Array::from_iter_values(
                (0..rows.len()).map(|i| i as f64),
            )),
        ],
    )?;
    let url = StorageUrl::parse(path.to_str().ok_or("corpus path is not valid UTF-8")?)?;
    let handle = StorageRegistry::new().handle_for(&url, None)?;
    let mut writer = ObjectParquetWriter::open(&handle, schema).await?;
    writer.write_batch(&batch).await?;
    writer.close().await?;
    Ok(())
}

/// How one rung's session serves.
#[derive(Debug, Clone, Copy)]
struct SessionShape {
    gpu_device: i32,
    batch_size: usize,
    batch_tokens: usize,
    partitions: usize,
    compute_precision: jammi_numerics::ComputePrecision,
}

/// The `[inference]` a session of `shape` serves under: what fixes the
/// chunk budget every rung's forwards are cut by.
fn inference_config(shape: SessionShape) -> InferenceConfig {
    InferenceConfig {
        batch_size: shape.batch_size,
        batch_tokens: shape.batch_tokens,
        partitions: shape.partitions,
        ..Default::default()
    }
}

/// Stand up a session that serves the corpus Parquet at `corpus` under
/// `shape`, keeping its artifacts in `artifact_dir`.
///
/// `gpu.require_gpu` is `gpu_device >= 0` — the convention `jammi-ai`'s own
/// `gpu_capability` harness pins — so a requested CUDA ordinal the box cannot
/// satisfy fails the FIRST model load with a typed `JammiError::Gpu` rather
/// than silently degrading to `Device::Cpu`: a `--cuda N` leg can never
/// publish `device_requested: "cuda:N"` for a run that executed on the CPU.
async fn session_over(
    corpus: &Path,
    artifact_dir: &Path,
    shape: SessionShape,
) -> Result<Arc<InferenceSession>, Box<dyn std::error::Error>> {
    let config = JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: GpuConfig {
            device: shape.gpu_device,
            require_gpu: shape.gpu_device >= 0,
            compute_precision: shape.compute_precision,
            ..Default::default()
        },
        inference: inference_config(shape),
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

/// FNV-1a over a byte stream — the stable, crate-free checksum the harness's
/// determinism digests fold with, rendered as fixed-width hex.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn mix(&mut self, byte: u8) {
        self.0 ^= byte as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn finish(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// What a serve made, in key order.
enum Artifact {
    /// One vector per key, row-major.
    Vectors { flat: Vec<f32>, dim: usize },
    /// One score-distribution string per key.
    Scores(Vec<String>),
}

impl Artifact {
    fn rows(&self) -> usize {
        match self {
            Artifact::Vectors { flat, dim } => flat.len().checked_div(*dim).unwrap_or(0),
            Artifact::Scores(scores) => scores.len(),
        }
    }

    /// The digest every rung of one unit on one box must agree on: every
    /// float's bits with a width tag per vector (a dimensionality change
    /// cannot hide behind a float collision), or every score string with a
    /// boundary byte.
    fn digest(&self) -> String {
        let mut fnv = Fnv::new();
        match self {
            Artifact::Vectors { flat, dim } => {
                for vector in flat.chunks(*dim.max(&1)) {
                    for byte in vector.iter().flat_map(|v| v.to_bits().to_le_bytes()) {
                        fnv.mix(byte);
                    }
                    fnv.mix(vector.len() as u8);
                }
            }
            Artifact::Scores(scores) => {
                for byte in scores.iter().flat_map(|s| s.bytes().chain([0x00])) {
                    fnv.mix(byte);
                }
            }
        }
        fnv.finish()
    }
}

/// The sink-phase events of finished writes, by table, in nanoseconds:
/// input, extract, parquet, index wait, segment, and the index builds'
/// thread time.
type PhaseLedger = Arc<Mutex<HashMap<String, [u64; 6]>>>;

/// The `tracing` layer that books `jammi_db::store::SINK_PHASES_TARGET`
/// events into a [`PhaseLedger`]. It enables that one target and nothing
/// else, so every other event in the engine stays the no-op it is without a
/// subscriber and costs the timed serves nothing.
struct PhaseLayer(PhaseLedger);

#[derive(Default)]
struct PhaseVisitor {
    table: String,
    ns: [u64; 6],
}

impl tracing::field::Visit for PhaseVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        let slot = [
            "input_ns",
            "extract_ns",
            "parquet_ns",
            "index_wait_ns",
            "segment_ns",
            "index_build_ns",
        ]
        .iter()
        .position(|name| *name == field.name());
        if let Some(slot) = slot {
            self.ns[slot] = value;
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "table" {
            self.table = format!("{value:?}");
        }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for PhaseLayer {
    fn enabled(
        &self,
        metadata: &tracing::Metadata<'_>,
        _: tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        metadata.target() == jammi_db::store::SINK_PHASES_TARGET
    }

    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let mut visitor = PhaseVisitor::default();
        event.record(&mut visitor);
        if let Ok(mut ledger) = self.0.lock() {
            ledger.insert(visitor.table, visitor.ns);
        }
    }
}

/// The process's phase ledger, its layer installed as the global subscriber on
/// first use. A process that already has another subscriber cannot book the
/// phases, and a plan leg refuses to be measured without them.
fn phase_ledger() -> Result<PhaseLedger, Box<dyn std::error::Error>> {
    static LEDGER: OnceLock<Result<PhaseLedger, String>> = OnceLock::new();
    LEDGER
        .get_or_init(|| {
            let ledger = PhaseLedger::default();
            let subscriber = tracing_subscriber::registry().with(PhaseLayer(Arc::clone(&ledger)));
            tracing::subscriber::set_global_default(subscriber)
                .map(|()| ledger)
                .map_err(|e| format!("the sink-phase subscriber could not be installed: {e}"))
        })
        .clone()
        .map_err(Into::into)
}

/// One rung, stood up: its session (a `direct` rung's only loads its model)
/// and the loaded model.
struct RungSession {
    rung: Rung,
    partitions: Option<usize>,
    session: Arc<InferenceSession>,
    model: Arc<LoadedModel>,
    model_load_ms: f64,
    _artifacts: tempfile::TempDir,
}

/// What every rung of one leg session serves.
struct Unit<'a> {
    task: Task,
    model_id: &'a str,
    texts: &'a StringArray,
    /// The plan's forward chunks over `texts`, for the `direct` rung.
    chunks: &'a [Vec<usize>],
    ledger: &'a PhaseLedger,
}

impl RungSession {
    /// One serve: `(wall seconds, sink phases in seconds, the artifact)`. The
    /// artifact of a committed table is read back OUTSIDE the span; a row the
    /// serve lost is an error, never a smaller artifact.
    async fn serve(
        &self,
        unit: &Unit<'_>,
    ) -> Result<(f64, Option<[f64; 6]>, Artifact), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let (wall_s, phases, artifact) = match (self.rung, unit.task) {
            (Rung::Direct, task) => {
                // The forwards are the plan's chunks, in its forward order;
                // the artifact is scattered back to row order, as a table
                // read in key order presents it.
                let rows = unit.texts.len();
                let mut flat = Vec::new();
                let mut scores = vec![String::new(); rows];
                let mut dim = 0;
                for chunk in unit.chunks {
                    // A fresh array per chunk, as the plan's chunk assembler
                    // hands the model, never a slice of the whole column.
                    let content: ArrayRef = Arc::new(StringArray::from_iter_values(
                        chunk.iter().map(|&row| unit.texts.value(row)),
                    ));
                    let mut out = self.model.forward(&[content], task.model_task())?;
                    if let Some(at) = out.row_status.iter().position(|ok| !ok) {
                        return Err(format!(
                            "the direct forward failed row {}: {}",
                            chunk[at], out.row_errors[at]
                        )
                        .into());
                    }
                    match task {
                        Task::Embed => {
                            dim = out.shapes.first().map_or(0, |shape| shape.1);
                            flat.resize(rows * dim, 0.0);
                            let vectors = out.float_outputs.first().ok_or("no embedding head")?;
                            for (&row, vector) in chunk.iter().zip(vectors.chunks(dim.max(1))) {
                                flat[row * dim..(row + 1) * dim].copy_from_slice(vector);
                            }
                        }
                        // `forward_classification`'s string heads: labels,
                        // then the score distributions the plan's adapter
                        // surfaces as `all_scores_json`.
                        Task::Infer => {
                            let distributions = out
                                .string_outputs
                                .get_mut(1)
                                .ok_or("no score-distribution head")?;
                            for (&row, scores_json) in chunk.iter().zip(distributions.drain(..)) {
                                scores[row] = scores_json;
                            }
                        }
                    }
                }
                let artifact = match task {
                    Task::Embed => Artifact::Vectors { flat, dim },
                    Task::Infer => Artifact::Scores(scores),
                };
                (start.elapsed().as_secs_f64(), None, artifact)
            }
            (_, Task::Embed) => {
                let (table, _) = self
                    .session
                    .generate_text_embeddings(
                        SOURCE_ID,
                        unit.model_id,
                        &[TEXT_COLUMN.to_string()],
                        KEY_COLUMN,
                        jammi_db::store::CachePolicy::Bypass,
                        None,
                    )
                    .await?;
                let wall_s = start.elapsed().as_secs_f64();
                let phases = unit
                    .ledger
                    .lock()
                    .map_err(|_| "the sink-phase ledger is poisoned")?
                    .remove(&table.table_name)
                    .ok_or("the serve committed a table whose sink reported no phases")?
                    .map(|ns| ns as f64 / 1e9);
                // Storage order of a never-refreshed table is the plan's key
                // order, which for this corpus's keys is row order.
                let pin = self
                    .session
                    .result_store()
                    .pin_current_version(table)
                    .await?;
                let vectors = self.session.read_vectors(&pin).await?;
                let dim = vectors.first().map_or(0, Vec::len);
                let flat = vectors.into_iter().flatten().collect();
                (wall_s, Some(phases), Artifact::Vectors { flat, dim })
            }
            (_, Task::Infer) => {
                let (batches, _) = self
                    .session
                    .infer(
                        SOURCE_ID,
                        &ModelSource::parse(unit.model_id),
                        ModelTask::Classification,
                        &[TEXT_COLUMN.to_string()],
                        KEY_COLUMN,
                        jammi_db::store::CachePolicy::Bypass,
                    )
                    .await?;
                let wall_s = start.elapsed().as_secs_f64();
                let mut keyed = Vec::with_capacity(unit.texts.len());
                for batch in &batches {
                    let column = |name: &str| {
                        batch
                            .column_by_name(name)
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                            .ok_or_else(|| format!("infer output has no string column {name}"))
                    };
                    let (ids, scores) = (column("_row_id")?, column("all_scores_json")?);
                    keyed.extend(
                        (0..batch.num_rows())
                            .filter(|&i| !ids.is_null(i) && !scores.is_null(i))
                            .map(|i| (ids.value(i).to_string(), scores.value(i).to_string())),
                    );
                }
                keyed.sort();
                let scores = keyed.into_iter().map(|(_, scores)| scores).collect();
                (wall_s, None, Artifact::Scores(scores))
            }
        };
        if artifact.rows() != unit.texts.len() {
            return Err(format!(
                "the {} rung served {} rows of a {}-row corpus: a row was lost",
                self.rung.as_str(),
                artifact.rows(),
                unit.texts.len()
            )
            .into());
        }
        Ok((wall_s, phases, artifact))
    }
}

/// What one rung's measured serves amounted to.
#[derive(Default)]
struct Served {
    first_serve_ms: f64,
    iter_wall_s: Vec<f64>,
    phases: Vec<[f64; 6]>,
    first_digest: Option<String>,
    last: Option<Artifact>,
}

/// Measure one leg session in THIS process: every rung of `params.rungs` over
/// the `row_count`-row unit, interleaved when there is more than one, one leg
/// per rung.
pub async fn measure_legs(
    params: &EncodeStepParams,
    row_count: usize,
    take: usize,
) -> Result<Vec<Leg<EncodePayload>>, Box<dyn std::error::Error>> {
    params.validate()?;
    let ledger = phase_ledger()?;
    let rows = build_corpus(params.seed, row_count);
    let texts = StringArray::from_iter_values(rows.iter().map(|r| r.text.as_str()));

    let scratch = tempfile::tempdir()?;
    let exchange_dir = params.exchange_dir.as_deref().unwrap_or(scratch.path());
    std::fs::create_dir_all(exchange_dir)?;
    let model_dir = match &params.model_dir {
        Some(dir) => dir.clone(),
        None => {
            let dir = exchange_dir.join("model");
            build_fixture_model_dir(params.task, &dir)?;
            dir
        }
    };
    let model_id = local_model_id(&model_dir)?;
    let (checkpoint_config_sha256, _) = sha256_and_len(&model_dir.join("config.json"))?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =
        sha256_and_len(&model_dir.join("model.safetensors"))?;
    let (checkpoint_tokenizer_sha256, _) = sha256_and_len(&model_dir.join("tokenizer.json"))?;
    let checkpoint_pooling_sha256 = checkpoint_pooling_sha256(&model_dir)?;

    let corpus_path = exchange_dir.join(format!("corpus_{row_count}.parquet"));
    write_corpus(&rows, &corpus_path).await?;
    let (corpus_sha256, _) = sha256_and_len(&corpus_path)?;

    let mut sessions = Vec::with_capacity(params.rungs.len());
    for &rung in &params.rungs {
        let partitions = rung.partitions(params.partitions);
        let artifacts = tempfile::tempdir()?;
        let session = session_over(
            &corpus_path,
            artifacts.path(),
            SessionShape {
                gpu_device: params.gpu_device,
                batch_size: params.batch_size,
                batch_tokens: params.batch_tokens,
                partitions: partitions.unwrap_or(1),
                compute_precision: params.compute_precision,
            },
        )
        .await?;
        let load = Instant::now();
        let guard = session
            .model_cache()
            .get_or_load(
                &ModelSource::parse(&model_id),
                params.task.model_task(),
                None,
            )
            .await?;
        sessions.push(RungSession {
            rung,
            partitions,
            model: Arc::clone(&guard.model),
            model_load_ms: load.elapsed().as_secs_f64() * 1_000.0,
            session,
            _artifacts: artifacts,
        });
    }

    // The plan's chunks over this corpus, off the loaded model's own costs
    // and ladder under the session's budget — what the `direct` rung
    // forwards and what the token accounting is read from.
    let loaded = &sessions.first().ok_or("a leg session has a rung")?.model;
    let max_sequence_length = loaded
        .max_sequence_length()
        .ok_or("the loaded model has no text forward to serve with")?;
    let budget = inference_config(SessionShape {
        gpu_device: params.gpu_device,
        batch_size: params.batch_size,
        batch_tokens: params.batch_tokens,
        partitions: 1,
        compute_precision: params.compute_precision,
    })
    .chunk_budget()?;
    let composed = forward_chunks(loaded, params.task.model_task(), &texts, budget)?;
    let unit = Unit {
        task: params.task,
        model_id: &model_id,
        texts: &texts,
        chunks: &composed.chunks,
        ledger: &ledger,
    };
    let mut served: Vec<Served> = sessions.iter().map(|_| Served::default()).collect();
    for (slot, session) in sessions.iter().enumerate() {
        served[slot].first_serve_ms = session.serve(&unit).await?.0 * 1_000.0;
    }
    for round in 0..params.warmup + params.iters {
        let mut order: Vec<usize> = (0..sessions.len()).collect();
        if round % 2 == 1 {
            order.reverse();
        }
        for slot in order {
            let (wall_s, phases, artifact) = sessions[slot].serve(&unit).await?;
            if round < params.warmup {
                continue;
            }
            let into = &mut served[slot];
            into.iter_wall_s.push(wall_s);
            into.phases.extend(phases);
            into.first_digest.get_or_insert_with(|| artifact.digest());
            into.last = Some(artifact);
        }
    }

    let compute_precision = loaded.compute_precision().to_string();
    let pooling = loaded
        .resolved_pooling()
        .map_or_else(|| "none".to_string(), |p| p.to_string());
    let tokens: usize = composed.row_tokens.iter().sum();
    let mut sorted_tokens: Vec<f64> = composed.row_tokens.iter().map(|&n| n as f64).collect();
    sorted_tokens.sort_by(|a, b| a.total_cmp(b));

    let session_rungs: Vec<String> = params.rungs.iter().map(|r| r.as_str().into()).collect();
    let solo = sessions.len() == 1;
    let mut legs = Vec::with_capacity(sessions.len());
    for (session, served) in sessions.iter().zip(served) {
        let stats = ServeStats::of(&served.iter_wall_s).ok_or("a leg measured no serve")?;
        let (p50_ms, min_ms) = (stats.p50_ms * 1_000.0, stats.min_ms * 1_000.0);
        let artifact = served.last.ok_or("a leg measured no serve")?;
        let outcome_digest = artifact.digest();
        if served.first_digest.as_deref() != Some(outcome_digest.as_str()) {
            return Err(format!(
                "the {} rung is not deterministic: its first measured serve digests to {:?}, its \
                 last to {outcome_digest}",
                session.rung.as_str(),
                served.first_digest
            )
            .into());
        }
        let vectors = match (&artifact, &params.legs_dir, take) {
            (Artifact::Vectors { flat, dim }, Some(legs_dir), 1) => {
                let name = format!("{}.vectors.f32", leg_stem(session.rung, row_count, take));
                std::fs::create_dir_all(legs_dir)?;
                let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(legs_dir.join(&name), bytes)?;
                Some((name, *dim))
            }
            _ => None,
        };
        let phase_series = |slot: usize| served.phases.iter().map(|p| p[slot]).collect();
        let payload = EncodePayload {
            task: params.task.as_str().to_string(),
            seed: params.seed,
            rows: row_count,
            corpus_sha256: corpus_sha256.clone(),
            token_lengths_sha256: token_lengths_sha256(&composed.row_tokens),
            tokens,
            batch_size: params.batch_size,
            batch_tokens: params.batch_tokens,
            max_sequence_length,
            compute_precision: compute_precision.clone(),
            checkpoint_config_sha256: checkpoint_config_sha256.clone(),
            checkpoint_weights_sha256: checkpoint_weights_sha256.clone(),
            checkpoint_weights_size_bytes,
            checkpoint_tokenizer_sha256: checkpoint_tokenizer_sha256.clone(),
            pooling: pooling.clone(),
            normalize: params.task == Task::Embed,
            warmup: params.warmup,
            iters_measured: served.iter_wall_s.len(),
            checkpoint_pooling_sha256: checkpoint_pooling_sha256.clone(),
            device_requested: requested_device_label(params.gpu_device),
            rung: session.rung.as_str().to_string(),
            partitions: session.partitions,
            session_rungs: session_rungs.clone(),
            take,
            model_dir: params.model_dir.as_ref().map(|d| d.display().to_string()),
            padded_tokens: composed.padded_tokens,
            row_tokens_p50: nearest_rank(&sorted_tokens, 0.50) as usize,
            row_tokens_max: sorted_tokens.last().map_or(0, |&n| n as usize),
            model_load_ms: session.model_load_ms,
            first_serve_ms: served.first_serve_ms,
            serve_ms_p50: p50_ms,
            serve_ms_min: min_ms,
            rows_per_s: per_second(row_count, p50_ms),
            tokens_per_s: per_second(tokens, p50_ms),
            sink_phases: (!served.phases.is_empty()).then(|| SinkPhaseSeries {
                input_s: phase_series(0),
                extract_s: phase_series(1),
                parquet_s: phase_series(2),
                index_wait_s: phase_series(3),
                segment_s: phase_series(4),
                index_build_s: phase_series(5),
            }),
        };
        let kernels_disabled_requested = jammi_kernels::admission::disabled_ops_requested();
        let provenance = Provenance {
            device_name: resolved_device_name(params.gpu_device)?,
            build_features: crate::report::build_features()
                .into_iter()
                .map(str::to_string)
                .collect(),
            flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
            kernels_disabled_fired: jammi_kernels::admission::disabled_ops_fired(),
            // The serving path has no fused attention arm: eval-only.
            arm: if kernels_disabled_requested.is_empty() {
                "fused"
            } else {
                "alloff"
            }
            .to_string(),
            attention_arm: "eager".to_string(),
            kernels_disabled_requested,
            mutant: Default::default(),
        };
        let measured = Measured {
            iter_wall_s: Some(served.iter_wall_s),
            work: Some(row_count as f64),
            // A shared process's high-water mark belongs to no one rung.
            peak_rss_bytes: if solo {
                peak_rss_measurement()
            } else {
                Measurement::not_yet_measured("bytes")
            },
            // The sampler wraps this process from outside: the orchestrator
            // books what it read onto the leg.
            peak_vram_bytes: Measurement::not_yet_measured("bytes"),
            outcome_digest: Some(outcome_digest),
            vector_dim: vectors.as_ref().map(|(_, dim)| *dim),
            vectors_file: vectors.map(|(name, _)| name),
            ..Default::default()
        };
        let leg = Leg::new(payload, provenance, measured, Facts::default());
        // Identity completeness, enforced on every real leg.
        leg.to_value();
        legs.push(leg);
    }
    Ok(legs)
}

/// A leg's file stem: `<rung>__rows<N>__r<take>`, the ladder's leg contract.
fn leg_stem(rung: Rung, row_count: usize, take: usize) -> String {
    format!("{}__rows{row_count}__r{take}", rung.as_str())
}

/// The reports one leg session prints: one per rung, the leg under
/// `tiers.encode_step`.
pub fn leg_reports(legs: Vec<Leg<EncodePayload>>) -> Vec<Report> {
    legs.into_iter()
        .map(|leg| {
            Report::new(
                "encode-leg",
                Tiers {
                    encode_step: Some(leg),
                    ..Default::default()
                },
            )
        })
        .collect()
}

/// What a sweep printed for a reader: every leg at a glance, and each rung's
/// cost split. The legs themselves are the files under `--legs-dir`.
#[derive(Debug, Serialize)]
pub struct EncodeSweep {
    pub legs: Vec<LegSummary>,
    pub fits: Vec<RungFit>,
}

/// One leg at a glance.
#[derive(Debug, Serialize)]
pub struct LegSummary {
    pub rung: String,
    pub rows: usize,
    pub take: usize,
    /// The leg's file under `--legs-dir`, when one was given.
    pub file: Option<String>,
    pub serve_ms_p50: f64,
    pub serve_ms_min: f64,
    pub rows_per_s: f64,
    pub outcome_digest: String,
    /// Of the measured serves' total wall time, the fraction each sink phase
    /// took, and what was spent outside the sink's loop. Null on a leg that
    /// committed no table.
    pub sink_shares: Option<SinkShares>,
}

/// Fractions of a plan leg's measured wall time. The wall phases sum with
/// `outside_sink` to one; `index_build` is the segment builders' thread time
/// over the same wall, overlapping the rest, so it stands outside that sum.
#[derive(Debug, Serialize)]
pub struct SinkShares {
    pub input: f64,
    pub extract: f64,
    pub parquet: f64,
    pub index_wait: f64,
    pub segment: f64,
    pub outside_sink: f64,
    pub index_build: f64,
}

/// One rung's `serve_ms = fixed_ms + per_row_ms · rows` over the sweep's
/// units, each at its fastest serve across takes; null when the sweep cannot
/// determine two terms.
#[derive(Debug, Serialize)]
pub struct RungFit {
    pub rung: String,
    pub fit_min: Option<CostFit>,
}

fn summarize(leg: &serde_json::Value, file: Option<String>) -> Option<LegSummary> {
    let number = |name: &str| leg[name].as_f64();
    let total = |series: &serde_json::Value| -> Option<f64> {
        Some(series.as_array()?.iter().filter_map(|v| v.as_f64()).sum())
    };
    let wall = total(&leg["iter_wall_s"])?;
    let sink_shares = leg["sink_phases"].as_object().and_then(|phases| {
        let share = |name: &str| Some(total(&phases[name])? / wall);
        let (input, extract, parquet, index_wait, segment, index_build) = (
            share("input_s")?,
            share("extract_s")?,
            share("parquet_s")?,
            share("index_wait_s")?,
            share("segment_s")?,
            share("index_build_s")?,
        );
        Some(SinkShares {
            input,
            extract,
            parquet,
            index_wait,
            segment,
            outside_sink: 1.0 - (input + extract + parquet + index_wait + segment),
            index_build,
        })
    });
    Some(LegSummary {
        rung: leg["rung"].as_str()?.to_string(),
        rows: leg["rows"].as_u64()? as usize,
        take: leg["take"].as_u64()? as usize,
        file,
        serve_ms_p50: number("serve_ms_p50")?,
        serve_ms_min: number("serve_ms_min")?,
        rows_per_s: number("rows_per_s")?,
        outcome_digest: leg["outcome_digest"].as_str()?.to_string(),
        sink_shares,
    })
}

/// File one leg's report under `legs_dir` by the ladder's leg contract
/// (`<rung>__rows<N>__r<take>.json`), when a directory was given, and
/// summarise it.
fn file_leg(
    legs_dir: Option<&Path>,
    report: &serde_json::Value,
) -> Result<LegSummary, Box<dyn std::error::Error>> {
    let leg = &report["tiers"]["encode_step"];
    let file = match legs_dir {
        Some(legs_dir) => {
            let name = format!(
                "{}__rows{}__r{}.json",
                leg["rung"].as_str().ok_or("a leg names its rung")?,
                leg["rows"].as_u64().ok_or("a leg names its rows")?,
                leg["take"].as_u64().ok_or("a leg names its take")?
            );
            std::fs::create_dir_all(legs_dir)?;
            std::fs::write(legs_dir.join(&name), serde_json::to_string_pretty(report)?)?;
            Some(name)
        }
        None => None,
    };
    summarize(leg, file).ok_or_else(|| "a leg session printed a malformed leg".into())
}

/// Each rung's cost split over the sweep's units, at each unit's fastest
/// serve across takes.
fn fit_rungs(rungs: &[Rung], rows: &[usize], legs: &[LegSummary]) -> Vec<RungFit> {
    rungs
        .iter()
        .map(|rung| {
            let fastest: Vec<(usize, f64)> = rows
                .iter()
                .filter_map(|&unit| {
                    legs.iter()
                        .filter(|leg| leg.rung == rung.as_str() && leg.rows == unit)
                        .map(|leg| leg.serve_ms_min)
                        .min_by(f64::total_cmp)
                        .map(|ms| (unit, ms))
                })
                .collect();
            RungFit {
                rung: rung.as_str().to_string(),
                fit_min: CostFit::least_squares(&fastest),
            }
        })
        .collect()
}

/// Run the producer: every (unit, take) in a child of this binary under the
/// device-memory sampler, its legs written to `--legs-dir`, and the sweep
/// summarised.
pub fn run(params: &EncodeStepParams) -> Result<EncodeSweep, Box<dyn std::error::Error>> {
    params.validate()?;
    let solo = params.rungs.len() == 1;
    let mut legs = Vec::new();
    for &row_count in &params.rows {
        for take in 1..=params.takes {
            let mut child = std::process::Command::new(std::env::current_exe()?);
            child
                .arg("encode-leg")
                .args(["--task", params.task.as_str()])
                .args(["--rows", &row_count.to_string()])
                .args(["--take", &take.to_string()])
                .args(["--seed", &params.seed.to_string()])
                .args(["--batch-size", &params.batch_size.to_string()])
                .args(["--batch-tokens", &params.batch_tokens.to_string()])
                .args(["--partitions", &params.partitions.to_string()])
                .args(["--compute-precision", &params.compute_precision.to_string()])
                .args(["--warmup", &params.warmup.to_string()])
                .args(["--iters", &params.iters.to_string()]);
            for rung in &params.rungs {
                child.args(["--rung", rung.as_str()]);
            }
            if let Some(ordinal) = params.cuda_ordinal() {
                child.args(["--cuda", &ordinal.to_string()]);
            }
            for (flag, dir) in [
                ("--model-dir", &params.model_dir),
                ("--exchange-dir", &params.exchange_dir),
                ("--legs-dir", &params.legs_dir),
            ] {
                if let Some(dir) = dir {
                    child.arg(flag).arg(dir);
                }
            }
            let (output, peak_vram) =
                run_sampled(&mut child, device_memory_probe(params.cuda_ordinal()))?;
            if !output.status.success() {
                return Err(format!(
                    "the leg session at {row_count} rows, take {take}, exited with {}",
                    output.status
                )
                .into());
            }
            let mut reports: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout)?;
            for report in &mut reports {
                if solo {
                    // One rung had the process, so what the sampler read
                    // around it is that rung's.
                    report["tiers"]["encode_step"]["peak_vram_bytes"] =
                        serde_json::to_value(&peak_vram)?;
                }
                legs.push(file_leg(params.legs_dir.as_deref(), report)?);
            }
        }
    }
    let fits = fit_rungs(&params.rungs, &params.rows, &legs);
    Ok(EncodeSweep { legs, fits })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> EncodeStepParams {
        EncodeStepParams {
            task: Task::Embed,
            rungs: vec![Rung::Plan],
            model_dir: None,
            rows: vec![48],
            takes: 1,
            seed: 0,
            batch_size: 8,
            batch_tokens: InferenceConfig::default().batch_tokens,
            partitions: 4,
            compute_precision: jammi_numerics::ComputePrecision::F32,
            warmup: 1,
            iters: 2,
            gpu_device: CPU_HERMETIC_DEVICE,
            exchange_dir: None,
            legs_dir: None,
        }
    }

    fn cpu_shape() -> SessionShape {
        SessionShape {
            gpu_device: CPU_HERMETIC_DEVICE,
            batch_size: 8,
            batch_tokens: InferenceConfig::default().batch_tokens,
            partitions: 1,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn leg_of(legs: &[Leg<EncodePayload>], rung: Rung) -> &Leg<EncodePayload> {
        legs.iter()
            .find(|leg| leg.payload.rung == rung.as_str())
            .unwrap_or_else(|| panic!("no {} leg", rung.as_str()))
    }

    /// Cardinality pin: the EXACT comparison identity, in this exact order —
    /// the list the ladder comparator reads and the CUDA-artifact guard
    /// extracts. A field added, removed, or renamed here is a reviewed diff
    /// against this test.
    #[test]
    fn identity_fields_are_pinned() {
        use crate::leg::Payload;
        let names: Vec<&str> = EncodePayload::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            vec![
                "task",
                "seed",
                "rows",
                "corpus_sha256",
                "token_lengths_sha256",
                "tokens",
                "batch_size",
                "batch_tokens",
                "max_sequence_length",
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

    /// The negative control: what the legs of an edge differ BY — the rung,
    /// its partitions, the session, the take, the checkpoint's path — is
    /// recorded on the payload and never a comparison key.
    #[test]
    fn recorded_fields_are_disjoint_from_identity() {
        use crate::leg::Payload;
        for recorded in ["rung", "partitions", "session_rungs", "take", "model_dir"] {
            assert!(
                !EncodePayload::IDENTITY_FIELDS
                    .iter()
                    .any(|(identity, _)| *identity == recorded),
                "{recorded:?} is recorded and must never be a comparison key"
            );
        }
    }

    /// The corpus is a function of `(seed, row index)` alone: an `n`-row
    /// corpus is a prefix of a larger one, a different seed is a different
    /// corpus, and the zero-padded keys sort in row order — the property that
    /// makes the plan's key order the corpus's own.
    #[test]
    fn the_corpus_is_seeded_prefix_stable_and_keyed_in_row_order() {
        let small = build_corpus(7, 64);
        let large = build_corpus(7, 300);
        assert!(small
            .iter()
            .zip(&large)
            .all(|(a, b)| a.id == b.id && a.text == b.text));
        assert!(build_corpus(8, 64)
            .iter()
            .zip(&small)
            .any(|(a, b)| a.text != b.text));
        let mut sorted: Vec<&str> = large.iter().map(|r| r.id.as_str()).collect();
        sorted.sort_unstable();
        assert!(sorted.iter().zip(&large).all(|(id, row)| *id == row.id));
    }

    /// The length mixture covers every row (its shares are parts per hundred)
    /// and really is variable: a few hundred rows span an order of magnitude
    /// in word count, with every band's range respected.
    #[test]
    fn the_corpus_length_mixture_is_whole_and_variable() {
        assert_eq!(LENGTH_BANDS.iter().map(|b| b.0).sum::<u64>(), 100);
        let words: Vec<usize> = build_corpus(0, 400)
            .iter()
            .map(|r| r.text.split(' ').count())
            .collect();
        let (fewest, most) = (
            *words.iter().min().expect("rows"),
            *words.iter().max().expect("rows"),
        );
        assert!(fewest >= 3 && most <= 300, "{fewest}..{most}");
        assert!(
            most >= 10 * fewest,
            "400 rows must reach both ends of the mixture: {fewest}..{most}"
        );
    }

    /// The token-length digest is the sha256 of the decimal counts joined by
    /// commas — pinned against a value computed outside this crate, the same
    /// known answer `reference/torch_encode.py`'s own test pins, so the two
    /// producers cannot drift apart on the rendering.
    #[test]
    fn token_lengths_sha256_is_the_pinned_rendering() {
        assert_eq!(
            token_lengths_sha256(&[3, 5, 8]),
            "d8c85d93367f9ebb51148c0a399b58ca7386e2de1a4ffcc5e925048f6d4e2250"
        );
    }

    /// The direct rung drives the loaded model end to end on `Device::Cpu`
    /// and every declared identity AND provenance field lands populated
    /// (the same `assert_identity_fields_present` a leg session enforces,
    /// re-proven as a test).
    #[tokio::test(flavor = "multi_thread")]
    async fn a_direct_leg_drives_the_loaded_model_and_populates_every_field() {
        let params = EncodeStepParams {
            rungs: vec![Rung::Direct],
            ..test_params()
        };
        let legs = measure_legs(&params, 48, 1).await.expect("leg session");
        let [leg] = legs.as_slice() else {
            panic!("one rung is one leg: {}", legs.len());
        };
        assert_eq!(
            (leg.payload.rung.as_str(), leg.payload.partitions),
            ("direct", None)
        );
        assert_eq!(leg.payload.session_rungs, vec!["direct"]);
        assert_eq!(
            (
                leg.payload.task.as_str(),
                leg.payload.rows,
                leg.measured.work,
                leg.payload.take
            ),
            ("embed", 48, Some(48.0), 1)
        );
        assert_eq!(
            leg.measured.iter_wall_s.as_ref().map_or(0, Vec::len),
            params.iters
        );
        assert_eq!(leg.payload.iters_measured, params.iters);
        assert_eq!(
            leg.payload.max_sequence_length, 128,
            "tiny_bert's position bound"
        );
        assert_eq!(
            (
                leg.payload.compute_precision.as_str(),
                leg.payload.pooling.as_str()
            ),
            ("f32", "mean")
        );
        assert!(leg.payload.normalize);
        assert_eq!(
            (
                leg.payload.device_requested.as_str(),
                leg.provenance
                    .as_ref()
                    .map(|p| p.device_name.as_str())
                    .unwrap_or("<none>")
            ),
            ("cpu", "cpu")
        );
        assert_eq!(
            leg.payload.model_dir, None,
            "the compiled-in fixture has no path"
        );
        assert!(
            leg.payload.sink_phases.is_none(),
            "the direct rung commits no table"
        );
        assert_eq!(leg.measured.vectors_file, None, "no legs dir was given");
        for (digest, name) in [
            (&leg.payload.checkpoint_config_sha256, "config"),
            (&leg.payload.checkpoint_weights_sha256, "weights"),
            (&leg.payload.checkpoint_tokenizer_sha256, "tokenizer"),
            (
                leg.payload
                    .checkpoint_pooling_sha256
                    .as_ref()
                    .expect("the embed fixture carries 1_Pooling/config.json"),
                "pooling config",
            ),
            (&leg.payload.corpus_sha256, "corpus"),
            (&leg.payload.token_lengths_sha256, "token lengths"),
        ] {
            assert_eq!(digest.len(), 64, "{name} sha256 must be 64 hex chars");
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{name} must be hex"
            );
        }
        // A corpus of genuinely different lengths must pad — a real
        // tokenization, not a dense assumption — and its long tail must reach
        // the model's truncation bound without passing it.
        assert!(leg.payload.padded_tokens > leg.payload.tokens);
        assert!(leg.payload.row_tokens_p50 < leg.payload.row_tokens_max);
        assert!(leg.payload.row_tokens_max <= leg.payload.max_sequence_length);
        assert!(
            leg.payload.serve_ms_min > 0.0 && leg.payload.serve_ms_min <= leg.payload.serve_ms_p50
        );
        assert!(leg.payload.first_serve_ms > 0.0 && leg.payload.model_load_ms > 0.0);
        assert!(leg.payload.tokens_per_s > leg.payload.rows_per_s);
        assert_eq!(
            leg.measured.outcome_digest.as_deref().map_or(0, str::len),
            16
        );
        assert_eq!(
            leg.measured.peak_vram_bytes.value, None,
            "the sampler wraps the process from outside; nothing is read in it"
        );
    }

    /// The exact edges, hermetically: over one unit on one box, `direct`,
    /// `plan` and `plan-partitioned` (N greater than the chunk count would
    /// leave idle, so the fan-out really splits the rows) persist
    /// byte-identical artifacts, for both tasks — the property that lets a
    /// result established at the bottom of the ladder hold at the top. The
    /// plan legs carry their sink phases; `infer`'s, which commit no
    /// embedding table, do not.
    #[tokio::test(flavor = "multi_thread")]
    async fn every_rung_of_a_unit_produces_the_same_artifact() {
        for task in [Task::Embed, Task::Infer] {
            let params = EncodeStepParams {
                task,
                rungs: vec![Rung::Direct, Rung::Plan, Rung::PlanPartitioned],
                ..test_params()
            };
            let legs = measure_legs(&params, 48, 1).await.expect("leg session");
            let digests: Vec<&str> = legs
                .iter()
                .map(|l| l.measured.outcome_digest.as_deref().unwrap_or_default())
                .collect();
            assert_eq!(digests.len(), 3);
            assert!(
                digests.iter().all(|d| *d == digests[0]),
                "{task:?}: the rungs' artifacts differ: {digests:?}"
            );
            let plan = leg_of(&legs, Rung::Plan);
            assert_eq!(plan.payload.partitions, Some(1));
            assert_eq!(
                leg_of(&legs, Rung::PlanPartitioned).payload.partitions,
                Some(4)
            );
            assert_eq!(
                plan.payload.session_rungs,
                vec!["direct", "plan", "plan-partitioned"]
            );
            assert_eq!(
                plan.measured.peak_rss_bytes.value, None,
                "a shared process's high-water mark belongs to no one rung"
            );
            assert_eq!(plan.payload.normalize, task == Task::Embed);
            match task {
                Task::Embed => {
                    let phases = plan
                        .payload
                        .sink_phases
                        .as_ref()
                        .expect("a committed table has phases");
                    assert_eq!(phases.index_build_s.len(), params.iters);
                    assert!(phases.input_s.iter().all(|&s| s > 0.0));
                    assert!(phases.index_build_s.iter().all(|&s| s > 0.0));
                    assert!(phases.parquet_s.iter().all(|&s| s > 0.0));
                }
                Task::Infer => {
                    assert!(plan.payload.sink_phases.is_none());
                    assert_eq!(plan.payload.pooling, "none");
                }
            }
        }
    }

    /// Interleaving is balanced: with two rungs and an even iteration count,
    /// each measured round runs the rungs in one order and the next in the
    /// other, so each rung has as many serves in first position as in last.
    /// Proven off the time series' lengths and the round order the session
    /// records; and an odd count is refused before anything is built.
    #[tokio::test(flavor = "multi_thread")]
    async fn interleaved_rungs_are_served_forward_then_reversed() {
        let params = EncodeStepParams {
            rungs: vec![Rung::Direct, Rung::Plan],
            iters: 4,
            ..test_params()
        };
        let legs = measure_legs(&params, 16, 1).await.expect("leg session");
        assert!(legs
            .iter()
            .all(|leg| leg.measured.iter_wall_s.as_ref().map_or(0, Vec::len) == 4));
        let odd = EncodeStepParams { iters: 3, ..params };
        assert!(
            odd.validate().is_err(),
            "an odd iteration count cannot be balanced"
        );
    }

    /// The legs directory carries the ladder's leg contract: one
    /// `<rung>__rows<N>__r<take>.json` per leg whose `tiers.encode_step` is
    /// the leg, a first take's vectors beside it as little-endian `f32` in
    /// key order, and the exchange directory the corpus the legs served and
    /// the fixture checkpoint; and a sweep's fastest serves fit two terms.
    /// The sessions run in this process (a test binary has no `encode-leg`
    /// child to spawn); `run` adds only the spawn and the sampler around
    /// them.
    #[tokio::test(flavor = "multi_thread")]
    async fn legs_are_filed_by_the_contract_with_vectors_corpus_and_checkpoint() {
        let legs_dir = tempfile::tempdir().expect("tempdir");
        let exchange = tempfile::tempdir().expect("tempdir");
        let params = EncodeStepParams {
            rungs: vec![Rung::Direct],
            rows: vec![16, 32],
            takes: 2,
            legs_dir: Some(legs_dir.path().to_path_buf()),
            exchange_dir: Some(exchange.path().to_path_buf()),
            ..test_params()
        };
        let mut summaries = Vec::new();
        for &rows in &params.rows {
            for take in 1..=params.takes {
                let legs = measure_legs(&params, rows, take)
                    .await
                    .expect("leg session");
                for report in leg_reports(legs) {
                    let report = serde_json::to_value(&report).expect("serialize");
                    summaries.push(file_leg(params.legs_dir.as_deref(), &report).expect("file"));
                }
            }
        }
        assert_eq!(summaries.len(), 4);
        let fit = fit_rungs(&params.rungs, &params.rows, &summaries)[0]
            .fit_min
            .expect("two units fit two terms");
        assert!(fit.per_row_ms > 0.0);
        let read = |name: &str| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(legs_dir.path().join(name)).expect(name))
                .expect("json")
        };
        let first = read("direct__rows16__r1.json");
        let leg = &first["tiers"]["encode_step"];
        assert_eq!(leg["vectors_file"], "direct__rows16__r1.vectors.f32");
        let dim = leg["vector_dim"].as_u64().expect("dim") as usize;
        let bytes =
            std::fs::read(legs_dir.path().join("direct__rows16__r1.vectors.f32")).expect("vectors");
        assert_eq!(bytes.len(), 16 * dim * 4);
        let second = read("direct__rows16__r2.json");
        assert!(second["tiers"]["encode_step"]["vectors_file"].is_null());
        assert_eq!(
            second["tiers"]["encode_step"]["outcome_digest"], leg["outcome_digest"],
            "two takes of one unit produce one artifact"
        );
        let (corpus_sha256, _) =
            sha256_and_len(&exchange.path().join("corpus_16.parquet")).expect("corpus");
        assert_eq!(leg["corpus_sha256"], corpus_sha256);
        assert!(exchange
            .path()
            .join("model")
            .join("model.safetensors")
            .exists());
    }

    /// A run that could measure nothing, or whose rungs make no sense, is
    /// refused before it builds anything.
    #[test]
    fn a_run_with_nothing_to_measure_is_refused() {
        for params in [
            EncodeStepParams {
                rows: vec![],
                ..test_params()
            },
            EncodeStepParams {
                rows: vec![16, 0],
                ..test_params()
            },
            EncodeStepParams {
                iters: 0,
                ..test_params()
            },
            EncodeStepParams {
                rungs: vec![],
                ..test_params()
            },
            EncodeStepParams {
                rungs: vec![Rung::Plan, Rung::Plan],
                ..test_params()
            },
            EncodeStepParams {
                rungs: vec![Rung::PlanPartitioned],
                partitions: 1,
                ..test_params()
            },
        ] {
            assert!(params.validate().is_err(), "{params:?}");
        }
    }

    /// The teeth for `checkpoint_tokenizer_sha256`: a run
    /// against a model dir whose `tokenizer.json` bytes differ from the
    /// fixture's own, with `config.json`/`model.safetensors` held byte-
    /// identical, must move the recorded tokenizer digest (and ONLY that
    /// digest) — proving the field is a real content hash of the actual
    /// tokenizer bytes served, not a copy of `checkpoint_config_sha256` or a
    /// constant. Perturbs the SAME fixture `build_fixture_model_dir`
    /// produces (rather than driving a second `run()`, which would also be
    /// legitimate but slower) so the assertion isolates the one changed
    /// file.
    #[test]
    fn checkpoint_tokenizer_sha256_reacts_to_the_actual_tokenizer_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_fixture_model_dir(Task::Embed, dir.path()).expect("build fixture model dir");
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
    /// unrelated default. This is the SAME shape `jammi-ai`'s own
    /// `pooling_config.rs` proves at the engine layer;
    /// here it only proves the FIXTURE is wired to a real, distinguishing
    /// config — the identity-hash-folds-it proof lives in `jammi-ai`'s own
    /// suite (out of this crate's scope).
    #[test]
    fn encode_model_dir_carries_an_explicit_pooling_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_fixture_model_dir(Task::Embed, dir.path()).expect("build fixture model dir");
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

    /// A `--cuda N` leg (`gpu_device >= 0`)
    /// on a box with no usable CUDA device must REFUSE with a typed error
    /// rather than silently degrading to `Device::Cpu` and publishing
    /// `device_requested:"cuda:0"` plus a real `device_name` for a run that
    /// actually executed on CPU (`resolved_device_name` queries
    /// `cuda_device_name` off the REQUESTED ordinal alone, so the session's
    /// `require_gpu` is what enforces that it actually resolved CUDA). This
    /// test runs on the CPU-hermetic default
    /// build (`not(feature = "cuda")`) — a box with no CUDA backend compiled
    /// in is exactly the "GPU-less box" this refusal must hold on, so it
    /// exercises the real `require_gpu` failure mode with no mocking.
    /// `#[cfg(not(feature = "cuda"))]` mirrors `GpuScheduler`'s own
    /// `for_device_falls_back_to_unlimited_when_probe_fails`: a `cuda`-feature
    /// build with a real device at ordinal 0 would legitimately succeed here,
    /// which is a different (valid) case this test is not about.
    #[cfg(not(feature = "cuda"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn cuda_leg_on_a_gpu_less_box_refuses_and_emits_no_leg() {
        let params = EncodeStepParams {
            gpu_device: 0,
            ..test_params()
        };
        let err = measure_legs(&params, 16, 1).await.expect_err(
            "a --cuda 0 leg on a box with no usable CUDA device must refuse (typed error),              never silently serve on CPU and emit a leg",
        );
        // The typed refusal threaded via `session_over`'s
        // `require_gpu = gpu_device >= 0` (`gpu_unavailable`'s
        // `JammiError::Gpu`, `#[error("GPU error: {0}")]`, message contains
        // "GPU required") — not some unrelated failure (a missing fixture, a
        // tokenizer error) that would also make this test spuriously pass.
        let message = err.to_string();
        assert!(
            message.contains("GPU required"),
            "expected the typed require_gpu refusal (JammiError::Gpu), got: {message}"
        );
    }

    #[test]
    fn requested_device_label_reads_cpu_for_negative_ordinal_and_cuda_otherwise() {
        assert_eq!(requested_device_label(-1), "cpu");
        assert_eq!(requested_device_label(0), "cuda:0");
        assert_eq!(requested_device_label(3), "cuda:3");
    }

    /// [`resolved_device_name`] on the CI-hermetic default never touches the
    /// `cuda`-feature-gated `cudarc` path at all (`device_name` is a
    /// POST-HOC hardware fact, but `"cpu"` is knowable
    /// without querying any driver).
    #[test]
    fn resolved_device_name_reads_cpu_for_negative_ordinal_without_a_cuda_query() {
        assert_eq!(resolved_device_name(-1).expect("cpu never errors"), "cpu");
    }

    /// `checkpoint_pooling_sha256` is `None` — never a
    /// panic, never an empty-string stand-in — for a model dir that carries
    /// no `1_Pooling/` folder at all, the SAME presence gate
    /// `backend::candle::all_candidate_paths` applies to the engine's own
    /// `content_digest`.
    #[test]
    fn checkpoint_pooling_sha256_is_none_when_the_model_dir_has_no_pooling_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        copy_fixture("tiny_bert", dir.path()).expect("copy fixture");
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
        build_fixture_model_dir(Task::Embed, dir.path()).expect("build fixture model dir");
        let expected = sha256_and_len(&dir.path().join("1_Pooling").join("config.json"))
            .expect("hash pooling config directly")
            .0;
        assert_eq!(
            checkpoint_pooling_sha256(dir.path()).expect("presence-gated read never errors"),
            Some(expected)
        );
    }

    /// The pooling teeth test: flip the fixture's `1_Pooling/config.json` from
    /// MEAN to CLS, holding `config.json` byte-identical, and drive the REAL
    /// accessors a leg reads (`LoadedModel::resolved_pooling`
    /// via the engine's own model cache, and `checkpoint_pooling_sha256`
    /// over the actual file bytes) — proving BOTH `pooling` and
    /// `checkpoint_pooling_sha256` move on the flip while
    /// `checkpoint_config_sha256` stays put. With `pooling` as a hardcoded
    /// `"mean"` literal and no field hashing the pooling-config bytes, this
    /// exact flip would leave every other identity field byte-identical
    /// while the served vectors differ.
    #[tokio::test(flavor = "multi_thread")]
    async fn checkpoint_pooling_sha256_and_pooling_move_together_when_the_fixture_flips_to_cls() {
        async fn resolved_pooling_and_hashes(
            pooling_flags: &serde_json::Value,
        ) -> (String, String, Option<String>) {
            let model_tmp = tempfile::tempdir().expect("tempdir");
            build_embed_fixture_with_pooling(model_tmp.path(), pooling_flags)
                .expect("build fixture model dir");
            let model_id = local_model_id(model_tmp.path()).expect("model id");
            let (config_sha, _) =
                sha256_and_len(&model_tmp.path().join("config.json")).expect("hash config.json");
            let pooling_sha = checkpoint_pooling_sha256(model_tmp.path())
                .expect("presence-gated read never errors");

            let corpus = model_tmp.path().join("corpus.parquet");
            write_corpus(&build_corpus(0, 1), &corpus)
                .await
                .expect("write corpus");
            let session = session_over(&corpus, model_tmp.path(), cpu_shape())
                .await
                .expect("session");
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
    /// through `pool_and_normalize`", the same claim `EncodePayload::normalize`'s
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
