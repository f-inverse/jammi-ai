//! The train-run tier: one fine-tune of one seed on one rung of the ladder,
//! driving the REAL trainer (`jammi_ai::fine_tune::trainer::TrainingLoop`)
//! and, above the first rung, the REAL job path around it — then one
//! evaluation of what the run published, through the public per-example
//! held-out seam (`TrainingLoop::evaluate_held_out`).
//!
//! ## The rungs, and what makes two of them one run
//!
//! [`Rung`] names how the same job reaches the trainer: `resident` drives
//! `TrainingLoopBuilder` over in-memory rows; `streamed` submits the job
//! through the session and lets an embedded worker materialise its training
//! set, stream the rows back and publish the adapter; `placed` and `shape-d`
//! run that job on other processes (`crate::plane`). Adjacent rungs differ
//! by one layer and must publish a byte-identical adapter, so every rung
//! trains the SAME run:
//!
//! - the rows in the training set's committed order ([`committed_order`]:
//!   the engine sorts a training set by its projected tuple, so the
//!   `resident` rows are sorted the same way rather than fed in file order);
//! - one `TrainingLoop::run` over every epoch (the job path runs a job
//!   whole), every epoch's checkpoint kept (`keep_last_n_checkpoints` is
//!   the run's epochs) so the trajectory is scored off what the run wrote;
//! - one `FineTuneConfig` ([`base_config`]) carrying the seed: the LoRA
//!   init is a function of `(seed, parameter name)` and the dropout streams
//!   of `seed`, on every rung.
//!
//! The run's own metrics — its per-epoch walls by phase, the kernel
//! dispatches inside its epoch loop, the disable list its process resolved
//! — are read back from the trainer's run metrics on every rung, so a leg
//! reports the process that trained, wherever that was.
//!
//! ## What the leg measures
//!
//! The evaluation never enters the run: the untrained adapter, every epoch's
//! checkpoint and the published adapter are loaded into a fresh loop over
//! the same spec ([`TrainingLoop::load_weights`]) and scored on the held-out
//! fixture and on a train-side probe batch. The endpoint is the PUBLISHED
//! adapter — what the job serves — never the last epoch's weights, which
//! differ when an earlier epoch's monitored loss was the best.
//!
//! Timing: `epoch_walls` are the trainer's own phases per epoch, on every
//! rung; the per-iteration series is each epoch's whole wall; the stations a
//! rung adds — claim latency, placement, training-set materialisation,
//! artifact publish — are reported on their own ([`Stations`]) and never
//! folded into an epoch's phases.
//!
//! ## The arm is provenance, not identity
//!
//! Which fused-kernel families run is `JAMMI_KERNELS_DISABLE`'s to say, read
//! once per process by `jammi_kernels::admission`; every process a rung
//! starts inherits it. `arm` records what the process resolved — `fused`
//! when nothing was requested off, `eager` otherwise — and the disable list
//! and dispatch counters record what ran. All of it is provenance the
//! ladder's rung premises prove, never identity.
//!
//! ## Held-out split is disjoint from the internal train/val split
//!
//! `TrainingDataLoader::split` carves an early-stopping validation slice out
//! of the training rows; the held-out fixture is a separate, disjoint set
//! that never enters the run and is scored in its committed order, which is
//! scoring identity. Disjointness is the fixture's own guarantee
//! (`train_ids_sha256.json` against `heldout_ids.txt`).
//!
//! ## What a run in another framework pairs against
//!
//! Every run writes the untrained adapter to `<work_dir>/`
//! [`INITIAL_ADAPTER_FILE`] and records its digest; `train_token_ids_sha256`
//! and `heldout_token_ids_sha256` digest the token batches the trainer's own
//! tokenization produces for one epoch and one held-out pass; the outcome
//! carries a time axis (`epoch_walls`, each trajectory point's cumulative
//! walls) and the cost carries space (`peak_rss_bytes`, `peak_vram_bytes`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use candle_core::Device;
use candle_nn::VarMap;

use jammi_ai::fine_tune::collective::BlockingCall;
use jammi_ai::fine_tune::data::{TextChunk, TrainingDataLoader};
use jammi_ai::fine_tune::source::TrainingSource;
use jammi_ai::fine_tune::spec::{
    admit_training_spec, submit_admitted_training, TrainingCommon, TrainingSpec, DEFAULT_WORLD_SIZE,
};
use jammi_ai::fine_tune::target::{EncoderAdaptersTarget, TrainingTarget};
use jammi_ai::fine_tune::trainer::{
    tokenize_and_bucket, AppliedLearningRate, KernelDispatchCount, KernelDispatches, TrainingLoop,
    TrainingLoopBuilder,
};
use jammi_ai::fine_tune::training_job::fine_tuned_model_id;
use jammi_ai::fine_tune::worker::{artifact_files_digest, published_artifact_digest};
use jammi_ai::fine_tune::{
    EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig, FineTuneMethod, LrSchedule,
};
use jammi_ai::model::arch::{self, EncoderFamily};
use jammi_ai::model::backend::candle::CandleBackend;
use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
use jammi_ai::model::tokenizer::{BatchEncoding, TokenizerWrapper};
use jammi_ai::model::{BackendType, LoadedModel, ModelId, ResolvedModel, TokenizerSource};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::config::JammiConfig;
use jammi_db::storage::{StorageRegistry, StorageUrl};
use jammi_db::store::{ArtifactStore, CachePolicy, LocalArtifact};
use jammi_db::ModelTask;
use jammi_encoders::AnyEncoder;
use jammi_lora::{AdapterConfig, LoraInitMode};

use crate::finetune_step::{attention_arm, sha256_and_len};
use crate::leg::{
    DispatchCounters, Facts, Leg, Measured, MutantStamp, Provenance, RanOn, Stations, Timeline,
    TrajectoryPoint,
};
use crate::report::TrainRunPayload;
use crate::vram::{device_memory_probe, VramWindow};

fn sendify<E: std::fmt::Display>(e: E) -> Box<dyn std::error::Error + Send + Sync> {
    e.to_string().into()
}

pub fn parse_lora_init(s: &str) -> Result<LoraInitMode, String> {
    match s {
        "zeros_b" => Ok(LoraInitMode::ZerosB),
        "gaussian" => Ok(LoraInitMode::Gaussian),
        other => Err(format!(
            "--lora-init '{other}' is invalid: expected 'zeros_b' or 'gaussian'"
        )),
    }
}

/// [`parse_lora_init`]'s inverse — the token this tier records in
/// [`crate::report::TrainRunPayload::lora_init`], byte-identical to what a
/// caller passes on the command line.
pub fn lora_init_as_str(mode: LoraInitMode) -> &'static str {
    match mode {
        LoraInitMode::ZerosB => "zeros_b",
        LoraInitMode::Gaussian => "gaussian",
    }
}

/// Which embedding objective this run trains. The committed held-out fixture
/// is TRIPLET-shaped (`anchor_id\tpositive_id\tnegative_id`); both families
/// train over the SAME committed fixture — `Triplet` natively (all three columns),
/// `Mnrl` via the (anchor, positive) PROJECTION of the identical rows in the
/// identical committed order (see [`project_to_pairs`]): dropping the mined
/// negative column and letting the rest of each batch supply in-batch
/// negatives instead. This tier does not itself choose — the CALLER selects
/// via `--objective`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// `EmbeddingLoss::Triplet { margin }` — explicit mined negative per row.
    Triplet,
    /// `EmbeddingLoss::MultipleNegativesRanking { temperature }` — in-batch
    /// negatives only, over the fixture's (anchor, positive) projection.
    Mnrl,
}

impl Objective {
    pub fn as_str(self) -> &'static str {
        match self {
            Objective::Triplet => "triplet",
            Objective::Mnrl => "mnrl",
        }
    }
}

impl std::str::FromStr for Objective {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "triplet" => Ok(Objective::Triplet),
            "mnrl" => Ok(Objective::Mnrl),
            other => Err(format!(
                "--objective '{other}' is invalid: expected 'triplet' or 'mnrl'"
            )),
        }
    }
}

/// Project committed [`IdTriplet`] rows to the MNRL (anchor, positive) shape
/// — the SAME rows, in the SAME committed order, with the mined negative
/// column DROPPED: the (anchor, positive) projection serves MNRL losslessly,
/// with in-batch negatives replacing the mined negative. A pure function (no
/// id, no shuffling) so both the train split
/// and the held-out fixture project identically — [`run`] calls this for
/// both when [`Objective::Mnrl`] is selected.
fn project_to_pairs(pairs: &[IdTriplet]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|p| (p.anchor.clone(), p.positive.clone()))
        .collect()
}

/// One (anchor, positive, negative) text triplet, keyed by a stable id — the
/// shape both the train split and the held-out fixture are supplied in,
/// regardless of which [`Objective`] this run trains. The committed held-out
/// fixture (`cookbook/fixtures/finetune_heldout`) mines an
/// EXPLICIT negative per row (`heldout_ids.txt`'s
/// `anchor_id\tpositive_id\tnegative_id` shape); [`Objective::Triplet`]
/// consumes all three columns natively, [`Objective::Mnrl`] consumes only
/// the (anchor, positive) projection ([`project_to_pairs`]) — see
/// [`crate::report::TrainRunPayload::margin`]'s doc for the field-naming
/// note.
#[derive(Debug, Clone)]
pub struct IdTriplet {
    pub id: String,
    pub anchor: String,
    pub positive: String,
    pub negative: String,
}

/// [`IdTriplet`]'s media twin: one (anchor, positive, negative) triplet whose
/// three members are the RAW ENCODED BYTES of an image or audio file (a PNG,
/// a WAV — whatever the corpus committed), keyed by a stable id.
///
/// The bytes, not a path, are what crosses this boundary: the trainer's media
/// path decodes each member itself (`image::load_from_memory` for pixels, the
/// CLAP front end for clips), exactly as the text path receives strings
/// rather than filenames. `main.rs::load_train_media_jsonl` resolves each
/// `*_path` against the JSONL's own directory and reads it, so a corpus tree
/// is relocatable and this struct never carries a path that could go stale
/// between load and use.
///
/// `*_sha256` are MEASURED off the bytes actually read (never transcribed
/// from a manifest), so a leg's provenance can name its inputs' digests the
/// way the text path's `train_pairs_file_sha256` names its file's.
#[derive(Debug, Clone)]
pub struct MediaTriplet {
    /// Stable row id — the anchor's id, matching [`IdTriplet::id`]'s role.
    pub id: String,
    /// Encoded bytes of the anchor file.
    pub anchor: Vec<u8>,
    /// Encoded bytes of the positive file.
    pub positive: Vec<u8>,
    /// Encoded bytes of the negative file.
    pub negative: Vec<u8>,
    /// sha256 (hex) of `anchor`, measured off those bytes.
    pub anchor_sha256: String,
    /// sha256 (hex) of `positive`, measured off those bytes.
    pub positive_sha256: String,
    /// sha256 (hex) of `negative`, measured off those bytes.
    pub negative_sha256: String,
}

/// The CONTENT digest of a media corpus — sha256 over each row's three
/// member digests (`anchor_sha256`, `positive_sha256`, `negative_sha256`,
/// each itself measured off the file bytes the loader read), lowercase hex,
/// concatenated in the SLICE'S OWN ORDER with no separator.
///
/// No separator is needed and none is used: every contribution is exactly
/// 64 hex characters, so the concatenation is uniquely decodable and no two
/// distinct corpora can fold to the same pre-image (the ambiguity a
/// variable-length concatenation would carry). The ORDER is part of the
/// digest by design — for the train split that is manifest order, for the
/// held-out fixture it is the committed scoring order the caller supplied,
/// which is exactly the order each corpus is consumed in.
///
/// Feeds [`crate::report::TrainRunPayload::train_media_sha256`]/
/// `heldout_media_sha256`; see those fields' docs for why a media leg needs
/// a content digest that the manifest digests cannot provide.
pub fn media_corpus_sha256(rows: &[MediaTriplet]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update(row.anchor_sha256.as_bytes());
        hasher.update(row.positive_sha256.as_bytes());
        hasher.update(row.negative_sha256.as_bytes());
    }
    hex::encode(hasher.finalize())
}

// Oracle hook: a per-thread sleep [`RowSet::loader`] applies to itself
// when nonzero, test-only (`#[cfg(test)]`, so this does not exist in a
// release build). Module-scoped (not inside `mod tests`) so `RowSet::loader`
// itself — OUTSIDE that module — can read it; `mod tests`' own
// `use super::*;` re-exposes it to the test that sets it
// (`train_run_wall_s_excludes_the_loader_build`). Thread-local, never a
// process-global `AtomicU64`: `cargo test` runs tests concurrently on
// separate threads, and `run` here always executes entirely on ONE
// thread (a `tokio::task::spawn_blocking` closure never hops threads mid-body),
// so setting/resetting it on the calling thread cannot leak into a
// concurrently-running, unrelated test.

/// A borrowed view of ONE modality's rows — the single shape every loader
/// construction in [`run`] goes through, so the train split, the
/// held-out fixture and the train-side probe can never disagree about which
/// modality this run is in.
enum RowSet<'a> {
    /// Text rows (`--task text_embedding`).
    Text(&'a [IdTriplet]),
    /// Media rows (`--task image_embedding` / `audio_embedding`).
    Media(&'a [MediaTriplet]),
}

impl<'a> RowSet<'a> {
    fn len(&self) -> usize {
        match self {
            RowSet::Text(rows) => rows.len(),
            RowSet::Media(rows) => rows.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Each row's stable id, in row order — what `evaluate_held_out` keys
    /// its per-example losses by.
    fn ids(&self) -> Vec<String> {
        match self {
            RowSet::Text(rows) => rows.iter().map(|r| r.id.clone()).collect(),
            RowSet::Media(rows) => rows.iter().map(|r| r.id.clone()).collect(),
        }
    }

    /// Refuse a media row whose three members are not three DISTINCT files,
    /// decided on the sha256 each member carries — MEASURED off the bytes
    /// this run read (`main.rs::read_media_member`), never on the paths,
    /// which two rows can spell differently for one file.
    ///
    /// A row whose positive IS its anchor contributes `max(0, 0 - d(a,n) +
    /// margin)` to the triplet loss no matter what the model does with it:
    /// a constant that moves with no gradient signal about the positive
    /// pair at all. A row whose negative is its anchor is worse — the loss
    /// pushes the anchor away from itself. Both are silent: they produce a
    /// perfectly plausible number. Refusing here is what makes the measured
    /// digests load-bearing rather than decorative provenance.
    ///
    /// Text rows are unaffected (`Ok`): the text corpora this tier consumes
    /// are committed fixtures whose partition is anchored by
    /// `heldout_ids_sha256`, so this refusal applies to media rows only.
    fn validate_media_members_are_distinct(&self, label: &str) -> Result<(), String> {
        let RowSet::Media(rows) = self else {
            return Ok(());
        };
        for (i, row) in rows.iter().enumerate() {
            let dup = if row.anchor_sha256 == row.positive_sha256 {
                Some(("anchor", "positive", &row.anchor_sha256))
            } else if row.anchor_sha256 == row.negative_sha256 {
                Some(("anchor", "negative", &row.anchor_sha256))
            } else if row.positive_sha256 == row.negative_sha256 {
                Some(("positive", "negative", &row.positive_sha256))
            } else {
                None
            };
            if let Some((a, b, sha)) = dup {
                return Err(format!(
                    "finetune-run: {label} row {i} (id {:?}) has the same file in its {a} and \
                     {b} slots (sha256 {sha}) — a triplet whose members are not three distinct \
                     files contributes a model-independent constant to the loss, so this run \
                     would report a plausible number measured on a degenerate example",
                    row.id
                ));
            }
        }
        Ok(())
    }

    /// The first `n` rows, in row order (the train-side probe's batch).
    fn take(&self, n: usize) -> RowSet<'a> {
        match self {
            RowSet::Text(rows) => RowSet::Text(&rows[..n]),
            RowSet::Media(rows) => RowSet::Media(&rows[..n]),
        }
    }

    /// Build the trainer's loader for these rows under `objective`.
    ///
    /// Text keeps both objectives (Triplet natively, MNRL over the
    /// (anchor, positive) projection — see [`project_to_pairs`]). Media has
    /// only the triplet shape: `TrainingDataLoader` exposes no media PAIRS
    /// constructor, so an MNRL media leg is a TYPED refusal rather than a
    /// silent fallback onto the triplet loss under an MNRL label — the two
    /// objectives are not interchangeable and a leg mislabelled that way
    /// would be unpairable with every other MNRL leg.
    fn loader(
        &self,
        objective: Objective,
    ) -> Result<TrainingDataLoader, Box<dyn std::error::Error + Send + Sync>> {
        match (self, objective) {
            (RowSet::Text(rows), Objective::Triplet) => Ok(TrainingDataLoader::from_triplets(
                rows.iter()
                    .map(|r| (r.anchor.clone(), r.positive.clone(), r.negative.clone()))
                    .collect(),
            )),
            (RowSet::Text(rows), Objective::Mnrl) => {
                Ok(TrainingDataLoader::from_pairs(project_to_pairs(rows)))
            }
            (RowSet::Media(rows), Objective::Triplet) => {
                Ok(TrainingDataLoader::from_media_triplets(
                    rows.iter()
                        .map(|r| (r.anchor.clone(), r.positive.clone(), r.negative.clone()))
                        .collect(),
                ))
            }
            (RowSet::Media(_), Objective::Mnrl) => Err(
                "finetune-run: --objective mnrl is not available for a media task — the \
                 trainer's media loader carries the (anchor, positive, negative) triplet \
                 shape only, and running the triplet loss under an MNRL label would make this \
                 leg unpairable with every real MNRL leg. Use --objective triplet."
                    .into(),
            ),
        }
    }
}

/// The texts one chunk sends through the encoder, in the order the trainer's
/// `encode_chunk` joins its groups for the single forward it runs per chunk:
/// anchors, then positives, then (for a triplet chunk) negatives.
fn joined_chunk_texts(chunk: &TextChunk) -> Result<Vec<String>, String> {
    let groups: Vec<&Vec<String>> = match chunk {
        TextChunk::Pairs { anchors, positives } => vec![anchors, positives],
        TextChunk::Triplet {
            anchors,
            positives,
            negatives,
        } => vec![anchors, positives, negatives],
        _ => {
            return Err(
                "finetune-run: internal: a text loader built by this tier yields only pair \
                 and triplet chunks"
                    .into(),
            )
        }
    };
    Ok(groups.into_iter().flatten().cloned().collect())
}

/// sha256 (hex) over a sequence of token batches, in order. Per batch the
/// hasher takes, little-endian: `rows` as `u32`, `cols` as `u32`, then every
/// `input_ids` entry row-major as `u32`, then every `attention_mask` entry
/// row-major as `u32`. The shape prefix keeps batch boundaries and padded
/// widths inside the digest, so two streams that flatten to the same ids
/// under different partitions or paddings do not collide.
///
/// `crates/jammi-bench/reference/torch_finetune_run.py` computes the same
/// layout over the batches it feeds its own model.
pub(crate) fn token_batches_sha256(batches: &[BatchEncoding]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for batch in batches {
        let rows = batch.input_ids.len() as u32;
        let cols = batch.input_ids.first().map_or(0, Vec::len) as u32;
        hasher.update(rows.to_le_bytes());
        hasher.update(cols.to_le_bytes());
        for plane in [&batch.input_ids, &batch.attention_masks] {
            for id in plane.iter().flatten() {
                hasher.update(id.to_le_bytes());
            }
        }
    }
    hex::encode(hasher.finalize())
}

/// Tokenize every chunk of `loader` exactly as the trainer does for `pass` —
/// the trainer's own two tokenization functions, over the trainer's own
/// chunking — and digest the result with [`token_batches_sha256`].
fn loader_token_batches(
    tokenizer: &TokenizerWrapper,
    loader: &TrainingDataLoader,
    batch_size: usize,
    effective_max: usize,
) -> Result<Vec<BatchEncoding>, Box<dyn std::error::Error + Send + Sync>> {
    loader
        .text_chunks(batch_size)
        .iter()
        .map(|chunk| {
            let texts = joined_chunk_texts(chunk)?;
            let (encoding, _rows, _cols) =
                tokenize_and_bucket(tokenizer, &texts, effective_max, None)?;
            Ok(encoding)
        })
        .collect()
}

/// Parameters [`run`] drives the tier off. The CPU-hermetic smoke test and
/// the real pod producer share this one shape — only which rows/checkpoint
/// they pass differs.
#[derive(Debug, Clone)]
pub struct FinetuneRunParams {
    /// Directory holding `config.json` + `model.safetensors` (+ optionally
    /// `tokenizer.json` — REQUIRED for the real `EncoderAdapters` target;
    /// see this module's doc).
    pub model_dir: PathBuf,
    /// Which tower of `model_dir`'s checkpoint this run trains, and hence
    /// which of the two row vectors below carries this run's data (see
    /// [`Task`]'s own doc). [`Task::Text`] is the default.
    pub task: Task,
    /// Training rows for a TEXT task (disjoint from `heldout_pairs`; see
    /// this module's doc). Empty for a media task.
    pub train_pairs: Vec<IdTriplet>,
    /// The held-out fixture's rows for a TEXT task, in the fixture's
    /// COMMITTED order — this order is scoring identity (the batch partition
    /// IS identity), never re-sorted or shuffled by this
    /// tier. Empty for a media task.
    pub heldout_pairs: Vec<IdTriplet>,
    /// Training rows for a MEDIA task — the decoded file BYTES of each
    /// row's three clips/images, read by `main.rs::load_train_media_jsonl`
    /// off the paths `--train-jsonl` named. Empty for a text task.
    pub train_media: Vec<MediaTriplet>,
    /// The held-out fixture's rows for a MEDIA task, in committed order.
    /// Empty for a text task.
    pub heldout_media: Vec<MediaTriplet>,
    /// sha256 (hex) of the `--train-jsonl` FILE's own raw bytes, MEASURED by
    /// `main.rs::load_train_jsonl` off the file this run actually opened —
    /// never a caller-transcribed digest, and NOT the same quantity as the
    /// committed fixture manifest's own `dataset_sha256` (a Merkle over
    /// per-pair digests, built off-process); see
    /// [`crate::report::TrainRunPayload::train_pairs_file_sha256`]'s own doc
    /// for why this field carries a distinct name.
    pub train_pairs_file_sha256: String,
    /// sha256 (hex) of the held-out id list's committed content — likewise
    /// caller-supplied (the fixture manifest's own `heldout_ids_sha256`).
    pub heldout_ids_sha256: String,
    /// sha256 (hex) of the `--heldout-jsonl` FILE's own raw bytes, MEASURED
    /// by `main.rs::load_heldout_fixture` off the file this run actually
    /// opened — the held-out TEXT is a total determinant of every per-
    /// example loss `d_i`, so (like `heldout_ids_sha256`) it must be
    /// content-anchored, never merely trusted by filename.
    pub heldout_pairs_sha256: String,
    pub seed: u64,
    /// Optimizer/schedule/objective knobs. `early_stopping_patience` MUST be
    /// `10_000` (the never-stops idiom) — [`run`] refuses a
    /// smaller value rather than silently letting an early-stopped run
    /// masquerade as a full-budget one.
    pub epochs: usize,
    pub eval_cadence: usize,
    pub batch_size: usize,
    /// The job's learning rate — positive, as admission requires of any job.
    pub learning_rate: f64,
    /// `--zero-lr-control`: run this exact job with every optimizer step
    /// applied at learning rate zero
    /// ([`AppliedLearningRate::Zero`]) — the negative control for every
    /// "learning happened" reading this tier emits. The job the run admits
    /// and records is still the real one, at [`Self::learning_rate`]; the
    /// control is a property of how it is RUN, set on the loop below
    /// admission, so the request edge's refusal of a non-positive rate stays
    /// whole. A control leg reports `lr: 0.0`, the rate it was measured at.
    pub applied_learning_rate: AppliedLearningRate,
    pub lr_schedule: LrSchedule,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub gradient_accumulation_steps: usize,
    pub validation_fraction: f64,
    pub early_stopping_patience: usize,
    pub early_stopping_metric: EarlyStoppingMetric,
    pub max_grad_norm: f64,
    /// Which embedding objective this run trains (see [`Objective`]'s own
    /// doc).
    pub objective: Objective,
    /// The Triplet objective's margin — used to build `EmbeddingLoss::Triplet`
    /// when `objective == Objective::Triplet`; ignored (but still a valid,
    /// caller-supplied value) otherwise.
    pub margin: f64,
    /// The MNRL objective's similarity-scale knob — used to build
    /// `EmbeddingLoss::MultipleNegativesRanking` when
    /// `objective == Objective::Mnrl`; ignored otherwise. `20.0` is the
    /// standard default (`jammi_wire::fine_tune::EmbeddingLoss::MultipleNegativesRanking`'s
    /// own doc: "temperature is the similarity scale; 20.0 is the standard
    /// default" — the same value `jammi_ai::fine_tune::trainer::PAIRWISE_SCALE`
    /// falls back to when no MNRL config is set).
    pub temperature: f64,
    pub matryoshka_dims: Vec<usize>,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    /// `--lora-init`: which LoRA initialization mode this run's adapters are
    /// built under. Default [`LoraInitMode::ZerosB`]. Threaded into BOTH
    /// construction sites ([`build_encoder_adapters`]'s `LoraBuildConfig`
    /// and [`base_config`]'s `FineTuneConfig::init_lora_weights`), never
    /// one: a run whose encoder
    /// adapters were built `Gaussian` while the trainer's own config still
    /// said `ZerosB` would carry a `FineTuneConfig` that DISAGREES with the
    /// weights it is training (the config is what a resume/adapter-save
    /// path re-reads), so splitting them would be a silent
    /// provenance-vs-reality fork, not a smaller change.
    ///
    /// Why the flag exists: a BF16 pre-flight needs a
    /// non-identity adapter at step 0 — under `ZerosB` every LoRA `A` has
    /// `dL/dA = 0` at the first step (`B = 0` kills the gradient path), so
    /// a "every LoRA Var has a non-zero gradient" bf16 check is VACUOUS in
    /// that mode and would pass on a dtype path that never worked.
    /// IDENTITY on the emitted tier (see
    /// [`crate::report::TrainRunPayload::lora_init`]).
    pub lora_init: LoraInitMode,
    /// `--expect-kernels-disabled`: the op key set this invocation CLAIMS
    /// `JAMMI_KERNELS_DISABLE` carries, sorted and
    /// deduplicated by
    /// [`jammi_kernels::admission::parse_disable_list`] — the SAME parser
    /// the env var itself is read through, never a second one (a divergent
    /// duplicate-preserving parser would hard-fail a VALID leg). `None` is the
    /// ordinary, unchecked case: the run makes no claim about
    /// `JAMMI_KERNELS_DISABLE` at all, and the leg's `arm` states what the
    /// process resolved. The witness that a decision leg ran unlabeled (no
    /// ambient contamination) is the driver script's to supply.
    ///
    /// When `Some`, [`run`] enforces THREE separate things a
    /// `JAMMI_KERNELS_DISABLE` leg can each fail independently — the leg is
    /// INVALID, never a datum, if any of them does:
    ///
    /// 1. At START (before any device/checkpoint/tensor work): this field
    ///    must equal
    ///    [`jammi_kernels::admission::disabled_ops_requested`] EXACTLY —
    ///    the SAME set EQUALITY
    ///    [`crate::finetune_step::FinetuneStepParams::expect_kernels_disabled`]
    ///    uses. (Not a SUBSET check: a subset check would hide an ambient
    ///    extra key.) The failure mode the check exists for — a
    ///    dropped, mistyped, unforwarded, or ambient-contaminated env var —
    ///    is caught either way: a dropped var makes the requested set
    ///    empty, a mistyped key is absent from it, and an ambient extra key
    ///    makes the sets unequal.
    /// 2. At END: [`jammi_kernels::admission::unmatched_disables`] must be
    ///    empty — a requested key that never disabled a live `admit` call
    ///    is a typo in the disable list, not evidence the forced-eager arm
    ///    ran (that module's own "safety property").
    /// 3. At END: every named key's `fused` dispatch counter
    ///    ([`jammi_kernels::admission::snapshot_all`]) must read `0`. (1)
    ///    and (2) together still admit a leg where the key fired eager on
    ///    one call site while a DIFFERENT site dispatched the same key
    ///    fused — the eager-twin premise this whole flag exists to prove
    ///    would be false, with nothing else to catch it.
    pub expect_kernels_disabled: Option<Vec<String>>,
    pub target_modules: Vec<String>,
    /// Optional restriction of LoRA injection to specific layer indices
    /// (`--layers-to-transform`; `jammi_lora::should_apply_lora`'s own doc:
    /// `Some(ids)` requires `layer_idx` to appear in it). `None` means no
    /// restriction — every layer matching `target_modules` gets a LoRA
    /// adapter (both [`build_encoder_adapters`]'s `LoraBuildConfig`s and
    /// [`base_config`]'s `FineTuneConfig::layers_to_transform` receive this
    /// value).
    pub layers_to_transform: Option<Vec<usize>>,
    pub backbone_dtype: jammi_numerics::ComputePrecision,
    pub max_seq_length: usize,
    /// CALLER-declared premise for the `admission_is_dense` report field
    /// (`--expect-dense`, default `false`, matching the committed fixture's
    /// padded transport) — mirrors `arm`'s declared-vs-resolved posture, not
    /// `expect_kernels_disabled`'s: the encoder decides dense-vs-padded per
    /// forward off the mask and this tier reads no per-forward signal back,
    /// so there is no process-resolved value to validate the claim against
    /// the way `disabled_ops_requested()` validates
    /// `expect_kernels_disabled`. The value is therefore recorded exactly as
    /// stated, never measured — a downstream merger checks it against the
    /// fixture's own known shape, the same way it checks any other
    /// caller-declared premise leg.
    pub expect_dense: bool,
    /// Which rung of the ladder this leg is — how the run reaches the
    /// trainer ([`Rung`]). Provenance on the emitted tier, never identity.
    pub rung: Rung,
    /// Where a rung above `streamed` runs (`crate::plane`).
    pub plane: crate::plane::PlaneParams,
    /// CUDA ordinal, or `None` for CPU (the CPU-hermetic smoke path).
    pub cuda_device: Option<usize>,
    /// Scratch directory this run's catalog sqlite file, artifact store, and
    /// training-scratch tempdirs live under — the CALLER owns its lifetime
    /// (kept alive for the whole run).
    pub work_dir: PathBuf,

    // ── Mutant provenance ───────────────────────────────────────────────
    //
    // These three are HONEST-LABELING fields, never identity or provenance
    // in [`crate::report::TrainRunPayload::IDENTITY_FIELDS`]/
    // [`crate::report::TrainRunPayload::PROVENANCE_FIELDS`]'s sense: a
    // mutant leg is an ordinary `fused`-arm run (same config, same
    // checkpoint, same fixture) with one AdamW-update-scaling patch
    // substituted into the binary this process was compiled from — nothing
    // about the RUN's own identity/provenance tuple differs from a clean
    // `fused` leg (`mutants/README.md`'s "what M1 does NOT touch"), so
    // stamping the mutant's name onto either comparison tuple would make a
    // mutant leg permanently unpairable with the clean legs it exists to be
    // compared against (the "arm is provenance, never identity" logic
    // applies here a fortiori: identity/provenance name
    // WHAT WAS MEASURED, this names WHICH BINARY did the measuring). They
    // are a caller-declared self-report — this process cannot verify from
    // inside itself that it was actually built from the claimed patch —
    // exactly as honest as a person naming themselves, never a measured or
    // derived fact; the downstream `ab_merge.py` mutant-dose-ladder mode
    // reads them by these exact key names to attribute a dose column's legs
    // to a specific, auditable mutant patch (`mutants/README.md`'s own
    // recorded fields).
    //
    // All three are OPTIONAL and all-or-none: [`run`] refuses (typed error,
    // not a panic) unless EITHER all three were never touched (`None`) OR
    // all three are non-empty after trimming — a non-emptiness count, not a
    // mere `is_some()` presence count, since the CLI happily parses
    // `--mutant-base-sha ""` (and any
    // whitespace-only value) into `Some(_)`; a trio that is explicitly
    // supplied but empty/whitespace in every position is refused too, not
    // silently treated as the ordinary non-mutant case — see [`run`]'s
    // own leading validation block, which also shape-checks
    // `mutant_base_sha` (7-40 hex chars) and `mutant_patch_sha256` (exactly
    // 64 hex chars) once the trio clears the emptiness gate. A normal
    // (non-mutant) leg supplies `None` for all three, and the emitted JSON
    // omits all three keys entirely (`#[serde(skip_serializing_if =
    // "Option::is_none")]` on [`crate::report::TrainRunPayload`]'s mirror
    // fields), so a normal leg's report carries no mutant keys.
    /// `--mutant-id`: the mutant's own label (e.g. `"eps-0.10"` — no-producer:
    /// an illustrative example label, not a measurement — see
    /// `docs/plans/63-how-well/mutants/README.md`'s dose-family naming).
    /// Trimmed and checked for non-emptiness by `run`; the STAMPED
    /// value (in the returned tier) is the trimmed string.
    pub mutant_id: Option<String>,
    /// `--mutant-base-sha`: the git commit sha this mutant's patch was cut
    /// against. Trimmed and shape-checked (7-40 hex chars) by `run`;
    /// the STAMPED value is the trimmed string.
    pub mutant_base_sha: Option<String>,
    /// `--mutant-patch-sha256`: sha256 (hex) of the mutant patch's own
    /// content — the "auditable" half of "attributable to a specific,
    /// auditable mutant patch". Trimmed and shape-checked (exactly 64 hex
    /// chars) by `run`; the STAMPED value is the trimmed string.
    pub mutant_patch_sha256: Option<String>,
}

impl FinetuneRunParams {
    /// This run's TRAIN rows, in the modality [`Self::task`] selects.
    fn train_rows(&self) -> RowSet<'_> {
        match self.task {
            Task::Text => RowSet::Text(&self.train_pairs),
            Task::Image | Task::Audio => RowSet::Media(&self.train_media),
        }
    }

    /// This run's HELD-OUT rows, in the modality [`Self::task`] selects.
    fn heldout_rows(&self) -> RowSet<'_> {
        match self.task {
            Task::Text => RowSet::Text(&self.heldout_pairs),
            Task::Image | Task::Audio => RowSet::Media(&self.heldout_media),
        }
    }

    /// Refuse a params value whose rows do not match its `--task`.
    ///
    /// The four row vectors are independent fields, so nothing in the type
    /// stops a caller from supplying text rows under `--task
    /// audio_embedding`. Left unchecked, the mismatched vector would simply
    /// be EMPTY and the run would die downstream on an opaque "0 rows is not
    /// a nonzero multiple of --batch" — naming the real cause here costs one
    /// comparison and turns a confusing failure into a correct one. The
    /// wrong-modality vector must be empty too: a run carrying both would
    /// leave a reader unable to tell which set produced the reported losses.
    fn validate_rows_match_task(&self) -> Result<(), String> {
        let (wrong_train, wrong_heldout, wrong_name) = match self.task {
            Task::Text => (self.train_media.len(), self.heldout_media.len(), "media"),
            Task::Image | Task::Audio => (self.train_pairs.len(), self.heldout_pairs.len(), "text"),
        };
        if wrong_train > 0 || wrong_heldout > 0 {
            return Err(format!(
                "finetune-run: --task {} carries {wrong_name} rows it cannot train \
                 ({wrong_train} train, {wrong_heldout} held-out) — a run that carried both \
                 modalities' rows would leave a reader unable to tell which set produced its \
                 reported losses",
                self.task.as_str()
            ));
        }
        if self.train_rows().is_empty() {
            return Err(format!(
                "finetune-run: --task {} has no train rows — --train-jsonl must carry the \
                 {} row shape for this task (see main.rs's row structs)",
                self.task.as_str(),
                self.task.modality().name(),
            ));
        }
        Ok(())
    }
}

/// A held-out example-mean loss point measured after one training epoch —
/// [`TrajectoryPoint`] is the serialized shape; this pairs it
/// with the model_type dispatch this module needs internally.
struct Trajectory {
    points: Vec<TrajectoryPoint>,
}

/// Which TOWER of the resolved checkpoint this run fine-tunes — the value
/// of the `--task` flag ([`FinetuneRunParams::task`]).
///
/// A BERT-family checkpoint has exactly one tower and only
/// [`Task::Text`] is meaningful for it; an OpenCLIP checkpoint has
/// two (text and vision) and an HF-CLAP checkpoint's audio half is the one
/// this tier trains, so for those the task is the ONLY thing that decides
/// which tower gets the LoRA adapters. The default is
/// [`Task::Text`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Train a text tower over `--train-jsonl`'s `*_text` columns
    /// (`--task text_embedding`).
    Text,
    /// Train a vision tower over `--train-jsonl`'s `*_path` columns
    /// (`--task image_embedding`; see `main.rs`'s media row shape).
    Image,
    /// Train an audio tower over `--train-jsonl`'s `*_path` columns
    /// (`--task audio_embedding`).
    Audio,
}

impl Task {
    /// The CLI spelling — the same string [`std::str::FromStr`] parses.
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Text => "text_embedding",
            Task::Image => "image_embedding",
            Task::Audio => "audio_embedding",
        }
    }

    /// Which modality this task's rows carry. The single place the
    /// task→modality mapping lives, so the row loader and the encoder
    /// dispatch can never disagree about what a `--task` value means.
    pub fn modality(self) -> jammi_encoders::Modality {
        match self {
            Task::Text => jammi_encoders::Modality::Text,
            Task::Image => jammi_encoders::Modality::Image,
            Task::Audio => jammi_encoders::Modality::Audio,
        }
    }

    /// The catalog's own task enum for this run's registered model row.
    pub fn model_task(self) -> ModelTask {
        match self {
            Task::Text => ModelTask::TextEmbedding,
            Task::Image => ModelTask::ImageEmbedding,
            Task::Audio => ModelTask::AudioEmbedding,
        }
    }
}

impl std::str::FromStr for Task {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text_embedding" => Ok(Task::Text),
            "image_embedding" => Ok(Task::Image),
            "audio_embedding" => Ok(Task::Audio),
            other => Err(format!(
                "--task '{other}' is invalid: expected 'text_embedding', 'image_embedding', or \
                 'audio_embedding'"
            )),
        }
    }
}

/// Whether `family`'s TRAINING-mode attention routes through the house
/// whole-attention-block admission cascade, and therefore whether
/// [`fused_dispatch_proof_gate`]'s attention-counter control is meaningful
/// for it.
///
/// The BERT-family encoders call `admit` on exactly one of the block or flash
/// cascade for every training-mode attention forward
/// (`forward_training_attention`'s doc), so all-zero counters over a run that
/// took optimizer steps proves the encoder never left eval mode. The three
/// cross-modal towers compose `jammi_encoders::attention_softmax` directly —
/// a plain `softmax_last_dim`/`softmax` fork with NO admission counter behind
/// it — so those counters read zero for a perfectly healthy tower run.
/// Applying the attention control to a tower would refuse every valid media
/// leg; asserting it "passes" would be worse still, a control that cannot
/// fail. The towers get their own live control instead (the LoRA linear
/// dispatch counters — see [`fused_dispatch_proof_gate`]).
///
/// A free function rather than a method because [`EncoderFamily`] is
/// `jammi-ai`'s type (ONE architecture predicate for the workspace); this
/// is a bench-tier fact about a measurement control, not a
/// property of the architecture itself.
pub(crate) fn attention_cascade_is_live(family: EncoderFamily) -> bool {
    match family {
        EncoderFamily::Bert | EncoderFamily::DistilBert | EncoderFamily::ModernBert => true,
        EncoderFamily::OpenClip | EncoderFamily::ClapAudio => false,
    }
}

/// One checkpoint directory, resolved ONCE per run: which files this tier
/// actually reads and which architecture family they name.
///
/// Every consumer in this module — the base-model load, the encoder build,
/// the reported `checkpoint_config_sha256`/`checkpoint_weights_sha256`, and
/// the catalog row's `model_type` — reads THESE fields, so the digests a run
/// reports are digests of the bytes it actually opened. Joining
/// `"config.json"`/`"model.safetensors"` onto `model_dir` by hand would
/// silently miss an OpenCLIP checkpoint, whose files are named differently.
pub(crate) struct Checkpoint {
    /// The architecture family, resolved from [`Self::config_json`] by the
    /// workspace's ONE predicate (`jammi_ai::model::arch::EncoderFamily::from_config`).
    pub(crate) family: EncoderFamily,
    /// The config file that was actually found (one of
    /// `jammi_ai::model::arch::CONFIG_CANDIDATE_NAMES`).
    pub(crate) config_path: PathBuf,
    /// Its parsed contents.
    pub(crate) config_json: serde_json::Value,
    /// The weights file that was actually found (one of
    /// `jammi_ai::model::arch::CANDLE_WEIGHTS_CANDIDATE_NAMES`).
    pub(crate) weights_path: PathBuf,
}

impl Checkpoint {
    /// Resolve `model_dir` through the ONE chain — `jammi_ai::model::arch`'s
    /// `config_candidates`/`weights_candidates`, the same frozen precedence
    /// the resolver, the serving loader and the model-content fingerprint
    /// walk use. This tier deliberately does NOT keep a private
    /// candidate list: a bench that disagreed with serving about which file
    /// in a directory is "the weights" would report digests for bytes no
    /// serving load ever reads.
    ///
    /// Every failure is a typed refusal naming the directory and what was
    /// tried — never a silent fallback onto a file that does not exist.
    pub(crate) fn resolve(
        model_dir: &Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_path = arch::config_candidates(model_dir).ok_or_else(|| {
            format!(
                "finetune-run: {} holds none of {:?} — this tier needs a checkpoint config to \
                 resolve the architecture family",
                model_dir.display(),
                arch::CONFIG_CANDIDATE_NAMES
            )
        })?;
        let weights_path = arch::weights_candidates(model_dir).ok_or_else(|| {
            format!(
                "finetune-run: {} holds none of {:?} — this tier builds LoRA-injected encoders \
                 from a Candle-loadable weights file",
                model_dir.display(),
                arch::CANDLE_WEIGHTS_CANDIDATE_NAMES
            )
        })?;
        // The shared Candle chain's last candidate is `model.gguf`. A GGUF
        // checkpoint resolves fine for SERVING but cannot be fine-tuned: the
        // `jammi-encoders` builders this tier drives take safetensors paths,
        // and quantized weights carry no trainable base to wrap. Refusing by
        // name here beats a downstream safetensors parse error that names a
        // `.gguf` file — and it keeps ONE chain rather than a private,
        // silently-divergent shorter one.
        if weights_path.file_name().and_then(|n| n.to_str()) == Some(arch::GGUF_WEIGHTS_FILENAME) {
            return Err(format!(
                "finetune-run: {} is a GGUF checkpoint — this training tier builds LoRA-injected \
                 encoders from safetensors weights only (quantized weights carry no trainable \
                 base to wrap)",
                weights_path.display()
            )
            .into());
        }
        let config_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)
                .map_err(|e| format!("finetune-run: {}: {e}", config_path.display()))?;
        let family = EncoderFamily::from_config(&config_json).ok_or_else(|| {
            let named = config_json
                .get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("<absent>");
            format!(
                "finetune-run: unsupported model_type '{named}' in {} — this tier supports \
                 'bert' (and its config-compatible aliases roberta/camembert/xlm-roberta), \
                 'modernbert' and 'distilbert' text checkpoints, an OpenCLIP checkpoint \
                 (discriminated by its 'model_cfg' key; jammi calls the family 'open_clip'), \
                 and an HF-CLAP audio checkpoint ('clap_audio_model')",
                config_path.display()
            )
        })?;
        Ok(Checkpoint {
            family,
            config_path,
            config_json,
            weights_path,
        })
    }
}

/// Build the frozen base model this run's `EncoderAdapters` target needs —
/// ONLY for its TOKENIZER (`TrainingLoop::encode_texts`'s `EncoderAdapters`
/// branch reads `base.tokenizer`, never `base`'s own forward; see that
/// method's doc). Constructed via the REAL `CandleBackend::load` public
/// trait method — the same backend a serving `InferenceSession` loads
/// through — over a hand-built [`ResolvedModel`] naming this run's
/// `model_dir` directly (mirroring the CLI-flag `--model-dir` convention
/// every other tier in this crate already uses, rather than the catalog-
/// registered `ModelSource` resolution production job submission goes
/// through — that resolution is a job-submission/catalog concern, not
/// something a bench tier that already knows its model path needs to
/// replicate).
///
/// The tokenizer requirement is scoped to [`Task::Text`]: only the
/// text path turns rows into token ids. A media task's rows are already
/// encoded images/clips the tower's own front end consumes, so demanding a
/// `tokenizer.json` there would refuse every legitimate audio checkpoint
/// (the committed `htsat_clap_tiny` fixture ships none, and needs none).
fn load_base_model(
    checkpoint: &Checkpoint,
    model_dir: &Path,
    task: Task,
    device_config: &DeviceConfig,
) -> Result<Arc<LoadedModel>, Box<dyn std::error::Error + Send + Sync>> {
    let config_path = checkpoint.config_path.clone();
    let model_config = checkpoint.config_json.clone();
    let tokenizer_path = model_dir.join("tokenizer.json");
    let has_tokenizer = tokenizer_path.exists();
    if task == Task::Text && !has_tokenizer {
        return Err(format!(
            "finetune-run: {} has no tokenizer.json — the EncoderAdapters training target \
             requires a real tokenizer to turn the fixture's text pairs into token ids (this \
             tier never falls back to synthetic ids the way finetune-step does, because a \
             synthetic-id run would never touch the committed fixture's real text at all)",
            model_dir.display()
        )
        .into());
    }
    let pooling_config_path = model_dir.join("1_Pooling/config.json");
    let pooling_config = if pooling_config_path.exists() {
        Some(serde_json::from_str(&std::fs::read_to_string(
            &pooling_config_path,
        )?)?)
    } else {
        None
    };
    // A media tower's front end reads its own `preprocessor_config.json`
    // (mel-bin count, fusion window, target sample rate for CLAP) — it is
    // OPTIONAL for the text path and supplied here whenever the checkpoint
    // ships one, so an audio base loads with the same front-end geometry a
    // serving `InferenceSession` would give it.
    let preprocessor_config_path = model_dir.join("preprocessor_config.json");
    let preprocessor_config = if preprocessor_config_path.exists() {
        Some(serde_json::from_str(&std::fs::read_to_string(
            &preprocessor_config_path,
        )?)?)
    } else {
        None
    };
    let resolved = ResolvedModel {
        model_id: ModelId(model_dir.display().to_string()),
        backend: BackendType::Candle,
        weights_format: jammi_ai::model::WeightsFormat::Safetensors,
        task: task.model_task(),
        config_path,
        weights_paths: vec![checkpoint.weights_path.clone()],
        tokenizer: has_tokenizer.then_some(TokenizerSource::HuggingFaceJson(tokenizer_path)),
        model_config,
        preprocessor_config,
        pooling_config,
        base_model_id: None,
        adapter_path: None,
        estimated_memory: 0,
    };
    let backend = CandleBackend;
    Ok(Arc::new(backend.load(&resolved, device_config)?))
}

/// Build a FRESH LoRA-injected encoder over `model_dir`, deterministic from
/// `(seed, target_modules, lora_rank, lora_alpha, lora_dropout)` —
/// [`jammi_lora::LoraInitMode::ZerosB`], the SAME init mode
/// `crate::finetune_step::build_fixture` uses, so two fresh builds at the
/// identical config start from byte-identical LoRA weights (the property
/// this tier's resume-cycling relies on: epoch k+1's fresh build, THEN
/// overwritten by the epoch-k checkpoint restore, never needs the fresh
/// init itself to carry any information — the restore always wins for every
/// registered `Var` name).
/// The LoRA site names (`--target-modules` selectors) the tower selected by
/// `(family, task)` actually carries — quoted verbatim into the
/// zero-trainable refusal so an operator who pointed ModernBERT selectors at
/// an OpenCLIP checkpoint is told what to write instead of being left to
/// guess.
///
/// These strings are the selector constants each tower's own loader passes
/// to `LoraSite::wrap` (`jammi-encoders`' crate-private `IN_PROJ_SITE`,
/// `C_FC_SITE`, `QUERY_SITE`, … — not importable from here). Because they
/// are transcribed rather than shared, they are held honest MECHANICALLY:
/// `tests::refusal_site_names_are_selectors_that_really_train` builds each
/// tower with exactly the names this function returns and asserts the build
/// yields a NON-empty trainable set, so a name that went stale fails the
/// suite instead of misdirecting an operator.
fn tower_site_names(family: EncoderFamily, task: Task) -> &'static str {
    match (family, task) {
        (EncoderFamily::Bert, _) => "query, key, value, dense",
        (EncoderFamily::DistilBert, _) => "q_lin, k_lin, v_lin, out_lin, lin1, lin2",
        (EncoderFamily::ModernBert, _) => "Wqkv, Wo, Wi",
        (EncoderFamily::OpenClip, _) => "in_proj, out_proj, c_fc, c_proj",
        (EncoderFamily::ClapAudio, _) => {
            "query, key, value, attention_output, intermediate_dense, output_dense, reduction, \
             linear1, linear2"
        }
    }
}

/// # `(family, task)` dispatch
///
/// The checkpoint's [`EncoderFamily`] says which architectures a directory
/// can build; `task` says WHICH TOWER of it to inject LoRA into. The three
/// BERT-family arms accept only [`Task::Text`] (they have one
/// tower); an OpenCLIP checkpoint builds its text tower for
/// [`Task::Text`] and its vision tower for
/// [`Task::Image`]; an HF-CLAP checkpoint builds its HTSAT audio
/// tower for [`Task::Audio`]. Every other pairing is a typed
/// refusal naming both halves — never a silent fallback onto the family's
/// "default" tower, which would train a text tower while the caller's rows
/// were images.
///
/// The emitted [`AdapterConfig`] records the family's own architecture id in
/// `model_type` AND, for the multi-tower families, `tower` — the two fields
/// together are what a serving load needs to pick the right tower to install
/// the adapter on (`jammi_lora::AdapterConfig::tower`'s own doc). A
/// single-tower family leaves `tower` at `None`.
#[allow(clippy::too_many_arguments)]
fn build_encoder_adapters(
    checkpoint: &Checkpoint,
    task: Task,
    target_modules: &[String],
    layers_to_transform: &Option<Vec<usize>>,
    lora_rank: usize,
    lora_alpha: f64,
    lora_dropout: f64,
    // Which LoRA initialization the built adapters use — threaded from
    // `FinetuneRunParams::lora_init`.
    lora_init: LoraInitMode,
    backbone_dtype: jammi_numerics::ComputePrecision,
    seed: u64,
    device: &Device,
    varmap: &VarMap,
) -> Result<(AnyEncoder, AdapterConfig), Box<dyn std::error::Error + Send + Sync>> {
    let family = checkpoint.family;
    let model_type = family.adapter_model_type();
    let config_json = &checkpoint.config_json;
    let weights = checkpoint.weights_path.clone();
    let dtype = jammi_encoders::compute_precision_to_dtype(backbone_dtype);
    let empty_ranks: HashMap<String, usize> = HashMap::new();
    let lora_dropout_opt = (lora_dropout > 0.0).then_some(lora_dropout as f32);
    let lora_build_1 = jammi_lora::LoraBuildConfig {
        target_modules,
        layers_to_transform,
        lora_rank,
        lora_alpha,
        use_rslora: false,
        lora_dropout: lora_dropout_opt,
        rank_pattern: &empty_ranks,
        init_mode: lora_init,
        seed,
        dropout_seed: seed,
    };
    let (mut encoder, tower) = match (family, task) {
        (EncoderFamily::ModernBert, Task::Text) => {
            let cfg: jammi_encoders::ModernBertConfig =
                serde_json::from_value(config_json.clone())?;
            let m = jammi_encoders::ModernBert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            (AnyEncoder::ModernBert(m), None)
        }
        (EncoderFamily::Bert, Task::Text) => {
            let cfg: jammi_encoders::BertConfig = serde_json::from_value(config_json.clone())?;
            let m = jammi_encoders::Bert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            (AnyEncoder::Bert(m), None)
        }
        (EncoderFamily::DistilBert, Task::Text) => {
            let cfg: jammi_encoders::DistilBertConfig =
                serde_json::from_value(config_json.clone())?;
            let m = jammi_encoders::DistilBert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            (AnyEncoder::DistilBert(m), None)
        }
        (EncoderFamily::OpenClip, Task::Text) => {
            let cfg = jammi_encoders::ClipTextConfig::from_open_clip_config(config_json)?;
            let m = jammi_encoders::ClipText::builder()
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            (AnyEncoder::ClipText(m), Some(jammi_lora::Tower::Text))
        }
        (EncoderFamily::OpenClip, Task::Image) => {
            let cfg = jammi_encoders::OpenClipVisionConfig::from_open_clip_config(config_json)?;
            let m = jammi_encoders::OpenClipVisionTransformer::builder()
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            (
                AnyEncoder::OpenClipVision(m),
                Some(jammi_lora::Tower::Vision),
            )
        }
        (EncoderFamily::ClapAudio, Task::Audio) => {
            let cfg = jammi_encoders::HtsatAudioConfig::from_hf_clap_config(config_json)?;
            let m = jammi_encoders::HtsatAudio::builder()
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            (
                AnyEncoder::Htsat(Box::new(m)),
                Some(jammi_lora::Tower::Audio),
            )
        }
        (family, task) => {
            return Err(format!(
                "finetune-run: --task '{}' has no tower on a '{}' checkpoint ({}). This tier \
                 trains: 'bert'/'modernbert'/'distilbert' with --task text_embedding; \
                 'open_clip' with --task text_embedding (the CLIP text tower) or \
                 image_embedding (the OpenCLIP vision tower); 'clap_audio_model' with --task \
                 audio_embedding (the HTSAT audio tower)",
                task.as_str(),
                family.adapter_model_type(),
                checkpoint.config_path.display(),
            )
            .into())
        }
    };
    // ZERO trainable Vars on the just-built encoder must refuse loudly here
    // — UNCONDITIONALLY, mirroring `finetune_step.rs`'s `build_fixture`
    // precedent EXACTLY ("no trainable LoRA tensors — target_modules matched
    // nothing"), never gated on whether `target_modules` itself was empty or
    // merely matched nothing on this architecture. This tier is a TRAINING
    // tier, not an inference load: with zero trainable `Var`s, candle-core's
    // backward pass tracks gradients only through `is_variable()` nodes, so
    // `loss.backward()` differentiates nothing, and the optimizer step
    // silently no-ops over an empty var list rather than warning — a run
    // configured this way would silently "train" nothing while reporting
    // plausible-looking numbers. `jammi_lora::LoraBuildConfig::frozen`'s
    // documented "no LoRA" convenience (an explicitly empty
    // `target_modules`) REMAINS valid for non-training (inference-load)
    // consumers of that crate — this refusal is scoped to this tier's
    // `build_encoder_adapters` only; `jammi-lora` itself is untouched. This
    // also covers the CLI's real failure mode: `main.rs`'s
    // `--target-modules` default (`"Wqkv,Wo,Wi"`, ModernBERT selectors)
    // matches nothing on `bert`'s or `distilbert`'s
    // `query`/`key`/`value`/`dense`-style naming.
    if encoder.trainable_params().is_empty() {
        // TWO producers can land here —
        // `target_modules` matching no linear at all, or a `layers_to_transform`
        // restriction excluding every layer `target_modules` WOULD otherwise
        // have matched (e.g. `Some([99])` on a fixture with fewer layers, or
        // `Some([1])` when the matching selector only exists on layer 0). The
        // message must name `layers_to_transform` whenever it is `Some` —
        // never blame `target_modules`/"correct the selectors" alone on the
        // exact N-twin profile path (one selector +
        // `layers_to_transform: Some([0])`), where an off-by-one layer index
        // is the likeliest operator error, not a bad selector string.
        let restriction = match layers_to_transform {
            Some(layers) => format!(" restricted to layers {layers:?}"),
            None => String::new(),
        };
        return Err(format!(
            "finetune-run: target_modules {target_modules:?}{restriction} yielded zero \
             trainable LoRA tensors on model_type '{model_type}' (--task {}) — this training \
             tier requires LoRA to actually train something. Correct the selectors for this \
             architecture (the CLI's own default, 'Wqkv,Wo,Wi', is ModernBERT-only and \
             matches nothing on bert/distilbert); this tower's own LoRA site names are: {}. \
             Or check layers_to_transform for an off-by-one layer index if it is set",
            task.as_str(),
            tower_site_names(family, task),
        )
        .into());
    }
    // `ModernBert::builder().build(..)` /
    // `Bert::builder().build(..)` construct a FRESH encoder in EVAL mode
    // (`training: false` at construction — see each builder's own `build`)
    // — a plain `TrainingTarget::EncoderAdapters(..)` wrap of that fresh
    // encoder therefore drives every forward through the EVAL attention
    // path, and `ModernBertAttention::forward`'s `self.training` gate never
    // reaches `forward_training_attention` at all, so the fused
    // whole-attention-block kernel (the A/B's entire measurand) cannot
    // dispatch in EITHER arm — the fused leg is silently indistinguishable
    // from a bug that always runs eval. `TrainingLoop::run()` itself never
    // calls `set_training(true)` for the ordinary (non-GradCache)
    // per-batch path either (only `mine_hard_negatives`/`run_gradcache_epoch`
    // do, and neither applies to `EncoderAdapters`), so this call is NOT
    // redundant with anything the trainer does — mirrors
    // `finetune_step.rs`'s own `build_fixture`'s `encoder.set_training(true)`
    // call EXACTLY, so this tier's fresh-per-epoch build starts from the
    // same training-mode precondition that tier's fixture always has. This
    // is belt-and-braces alongside `run`'s own dispatch-counter proof below
    // (`attention_block_fused_dispatches`/`attention_block_eager_dispatches`/
    // `attention_block_flash_*`): a caller must not depend on either check in
    // isolation to catch a future regression in the other.
    encoder.set_training(true);
    // Read the flag BACK off the object. The dispatch-
    // counter gate below is the process-wide proof for the BERT family, but
    // the three cross-modal towers have no attention-admission counter at
    // all (`attention_cascade_is_live`'s doc) — for them this
    // readback is the only place a `set_training` that silently failed to
    // reach every block would be caught. It is a mechanism check, not a
    // tautology: `set_training` walks each tower's blocks and sites, and
    // `is_training` reports the state the FORWARD path will actually read.
    if !encoder_is_training(&encoder) {
        return Err(format!(
            "finetune-run: the freshly built '{model_type}' encoder for --task {} did not \
             report training mode after set_training(true) — its LoRA sites would run \
             dropout-free and off the tape, so this run would not measure the fine-tune step \
             this tier claims to measure",
            task.as_str(),
        )
        .into());
    }
    // A SECOND `LoraBuildConfig` (identical values) — `lora_build_1` above
    // was already MOVED into `.lora(...)`, and `AdapterConfig::from_build`
    // needs its own borrow; the type is a plain, cheap struct of scalars and
    // borrowed slices, so building it twice is free.
    //
    // "Identical values" is load-bearing, not incidental: this one is what
    // the SAVED `adapter_config.json` records (`AdapterConfig::from_build`),
    // so a `lora_init` threaded into `lora_build_1` alone would ship an
    // adapter whose declared init mode contradicts the weights actually
    // built — hence BOTH sites read the same `lora_init` parameter, never
    // one literal and one variable.
    let lora_build_2 = jammi_lora::LoraBuildConfig {
        target_modules,
        layers_to_transform,
        lora_rank,
        lora_alpha,
        use_rslora: false,
        lora_dropout: lora_dropout_opt,
        rank_pattern: &empty_ranks,
        init_mode: lora_init,
        seed,
        dropout_seed: seed,
    };
    // `model_type` is the base ARCHITECTURE id (`EncoderFamily::
    // adapter_model_type`); `tower` says WHICH tower of a multi-tower
    // checkpoint these adapters install on, and stays `None` for the
    // single-tower BERT family — so an adapter saved by an existing text leg
    // carries exactly the bytes it always has (`jammi_lora::AdapterConfig::
    // tower` is `#[serde(default)]`, and `None` is skipped on the wire).
    let adapter_cfg = AdapterConfig::from_build(model_type, &lora_build_2, backbone_dtype);
    let adapter_cfg = match tower {
        Some(t) => adapter_cfg.with_tower(t),
        None => adapter_cfg,
    };
    Ok((encoder, adapter_cfg))
}

/// Whether `encoder` reports TRAINING mode — the per-variant accessor
/// `AnyEncoder` does not (yet) expose as one method.
///
/// `jammi_encoders::ModernBert` and the three cross-modal towers each carry
/// an `is_training()` of their own; `Bert`/`DistilBert` do not, and this
/// function reports `true` for them rather than inventing a state it cannot
/// read. That is honest, not vacuous: those two families are covered by the
/// process-wide attention-dispatch control instead
/// ([`fused_dispatch_proof_gate`], live for every BERT-family run), which is
/// exactly the coverage the towers lack. Neither family is left with no
/// control at all.
fn encoder_is_training(encoder: &AnyEncoder) -> bool {
    match encoder {
        AnyEncoder::ModernBert(m) => m.is_training(),
        AnyEncoder::ClipText(m) => m.is_training(),
        AnyEncoder::OpenClipVision(m) => m.is_training(),
        AnyEncoder::Htsat(m) => m.is_training(),
        AnyEncoder::Bert(_) | AnyEncoder::DistilBert(_) => true,
    }
}

/// Build the [`FineTuneConfig`] every epoch leg of this run shares, the
/// run's FULL `epochs` included. A leg stops after its own epoch through
/// `TrainingLoopBuilder::epoch_limit`, never by carrying a shorter `epochs`:
/// the LR schedule's horizon, the step-checkpoint cadence and the run's last
/// step are all derived from `config.epochs`, so a per-leg count would
/// compress the schedule into each leg and the cycle would no longer train
/// the run it claims to.
fn base_config(params: &FinetuneRunParams) -> FineTuneConfig {
    FineTuneConfig {
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        lora_dropout: params.lora_dropout,
        learning_rate: params.learning_rate,
        epochs: params.epochs,
        batch_size: params.batch_size,
        max_seq_length: params.max_seq_length,
        embedding_loss: Some(match params.objective {
            Objective::Triplet => EmbeddingLoss::Triplet {
                margin: params.margin,
            },
            Objective::Mnrl => EmbeddingLoss::MultipleNegativesRanking {
                temperature: params.temperature,
            },
        }),
        classification_loss: None,
        regression_loss: None,
        // Every epoch's checkpoint is retained and published: the trajectory
        // is scored off what the run wrote, on every rung. The write itself
        // happens on every run (the resume checkpoint) and is charged to the
        // checkpoint phase, never to the steps.
        keep_last_n_checkpoints: Some(params.epochs as u32),
        quantile_levels: Vec::new(),
        gradient_accumulation_steps: params.gradient_accumulation_steps,
        validation_fraction: params.validation_fraction,
        early_stopping_patience: params.early_stopping_patience,
        warmup_steps: params.warmup_steps,
        lr_schedule: params.lr_schedule,
        early_stopping_metric: params.early_stopping_metric,
        target_modules: params.target_modules.clone(),
        layers_to_transform: params.layers_to_transform.clone(),
        use_rslora: false,
        rank_pattern: HashMap::new(),
        // The THIRD site `--lora-init` reaches: the
        // `FineTuneConfig` the trainer itself carries. `build_encoder_adapters`
        // above is what actually initializes this tier's weights, so this
        // one is the config-vs-reality consistency half — a `FineTuneConfig`
        // still claiming `ZerosB` while the encoder was built `Gaussian`
        // would be a silent fork between what ran and what a resume /
        // adapter-save path re-reads.
        init_lora_weights: params.lora_init,
        backbone_dtype: params.backbone_dtype,
        weight_decay: params.weight_decay,
        max_grad_norm: params.max_grad_norm,
        cached: false,
        hard_negatives: Default::default(),
        matryoshka_dims: params.matryoshka_dims.clone(),
        seed: params.seed,
    }
}

/// The all-zero-attention validity gate: `Err` iff this run
/// took at least one optimizer step (`cumulative_steps > 0`) but every one
/// of the four training-mode attention dispatch counters
/// (`attention_block_fused_dispatches`, `attention_block_eager_dispatches`,
/// `attention_block_flash_fused_dispatches`,
/// `attention_block_flash_declined_dispatches`) reads `0`.
///
/// Not scoped to `model_type == "modernbert"`: BERT and DistilBERT
/// admit the SAME fused whole-attention-block kernel through the SAME
/// predicate, dispatched by TENSOR STATE (`head_dim == 64`), never by
/// architecture name (`jammi_encoders::attention_cascade`'s doc) — so a
/// `bert`- or `distilbert`-arch leg at `head_dim == 64` calls `admit` on
/// exactly one of the block or flash cascade for every training-mode
/// attention forward, exactly as a `modernbert` leg always has
/// (`forward_training_attention`'s doc). Gating this check on the model
/// NAME would silently let a BERT/DistilBERT leg that never called
/// `encoder.set_training(true)` through unnoticed — admission is by
/// COUNTERS, not by name, so this refusal is too, for every `model_type`
/// this tier supports.
///
/// For ANY architecture's leg that took at least one optimizer step, all
/// four dispatch counters reading zero at once is not a legitimate
/// "declined by domain" outcome (that reads `N eager / 0 fused`, never
/// `0/0/0/0`), it is proof the encoder never entered training mode at all
/// (a fresh `builder().build(..)` starts `training: false`, and
/// `TrainingLoop::run`'s ordinary per-batch path does not flip it via
/// `encoder.set_training(true)`). Refusing loudly here beats silently
/// emitting a plausible-looking report for a run that measured the eval
/// path — this check does NOT depend on `encoder.set_training(true)` being
/// called correctly upstream: it reads the same counters a downstream
/// merger's fused-proof gate reads, independent of how this process got
/// there. Extracted to a pure function (mirroring
/// [`crate::finetune_step::attention_arm`]'s own extraction) so the widened
/// admission-by-counters behaviour is unit-testable without a real
/// end-to-end training run.
///
/// # Two families, two live counters
///
/// Adding the three cross-modal towers made "all four attention counters
/// zero" ambiguous: a CLIP-text / OpenCLIP-vision / HTSAT tower composes
/// `jammi_encoders::attention_softmax` directly — a `softmax_last_dim` vs
/// `softmax` fork with NO admission counter behind it — so a perfectly
/// healthy tower run reads `0/0/0/0` on every attention counter. Applying
/// the attention control to a tower would refuse every valid media leg;
/// exempting a tower with nothing in its place would leave the whole media
/// half of this tier with a control that cannot fail, which is worse.
///
/// So the gate reads the counter that IS live per family
/// ([`attention_cascade_is_live`]): the BERT family keeps the
/// attention-block/flash cascade check unchanged, and a tower family is
/// checked against the LoRA-linear dispatch counters instead. Every LoRA
/// site on a tower calls `admit` on `lora_linear_fused` for every forward it
/// takes, so `fused + eager == 0` over a run with `cumulative_steps > 0`
/// proves the LoRA-injected encoder was never forwarded at all — the tower
/// analogue of the BERT failure this gate was built for. It does NOT prove
/// the training-mode softmax arm was taken; that is what
/// [`build_encoder_adapters`]'s `is_training()` readback covers, and neither
/// check is a substitute for the other.
// Eight counters + the family + the step count is one argument over
// clippy's default. They are a flat list of INDEPENDENT process-wide
// readings with no natural grouping — bundling them into a struct would add
// a type whose only job is to satisfy a lint, and would make each call site
// read further from the counters it names. Same posture (and same allow) as
// `build_encoder_adapters` above.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_dispatch_proof_gate(
    family: EncoderFamily,
    cumulative_steps: usize,
    attention_block_fused_dispatches: u64,
    attention_block_eager_dispatches: u64,
    attention_block_flash_fused_dispatches: u64,
    attention_block_flash_declined_dispatches: u64,
    lora_linear_fused_dispatches: u64,
    lora_linear_eager_dispatches: u64,
) -> Result<(), String> {
    if cumulative_steps == 0 {
        return Ok(());
    }
    let model_type = family.adapter_model_type();
    if attention_cascade_is_live(family) {
        if attention_block_fused_dispatches == 0
            && attention_block_eager_dispatches == 0
            && attention_block_flash_fused_dispatches == 0
            && attention_block_flash_declined_dispatches == 0
        {
            return Err(format!(
                "finetune-run: fused-dispatch-proof failure — this {model_type} run took \
                 {cumulative_steps} optimizer step(s) but the training-mode attention path never \
                 dispatched in either arm (attention_block_fused_dispatches, \
                 attention_block_eager_dispatches, attention_block_flash_fused_dispatches, and \
                 attention_block_flash_declined_dispatches are all 0). The encoder was never put \
                 into training mode via encoder.set_training(true), so this run measured the eval \
                 path, not the fine-tune step this tier claims to measure — INVALID run, not a \
                 datum."
            ));
        }
        return Ok(());
    }
    if lora_linear_fused_dispatches == 0 && lora_linear_eager_dispatches == 0 {
        return Err(format!(
            "finetune-run: lora-dispatch-proof failure — this {model_type} run took \
             {cumulative_steps} optimizer step(s) but no LoRA site was ever forwarded \
             (lora_linear_fused_dispatches and lora_linear_eager_dispatches are both 0). This \
             tower family has no whole-attention-block admission counter to read \
             (attention_cascade_is_live), so the LoRA linear cascade is the \
             live proof that the adapted encoder took part in this run at all — INVALID run, \
             not a datum."
        ));
    }
    Ok(())
}

/// The LoRA sites a fine-tune campaign adapts when a caller names none:
/// ModernBERT's attention and MLP input linears. One constant behind every
/// `--target-modules` default in this crate — the run, the step, the
/// gradient oracle — and behind `kernel-arm`'s census, so an arm derived
/// with no selector is derived on exactly the sites the legs will adapt.
pub const DEFAULT_TARGET_MODULES: &str = "Wqkv,Wo,Wi";

/// The run protocol this tier defaults to — the learning rate, epoch count
/// and evaluation cadence a leg gets when its command line names none, on
/// this producer and on its PyTorch twin alike (the twin's own defaults are
/// held equal to these by `test_torch_finetune_run_mirrors.py`). They are the
/// tier's, not the engine's: a job that names none gets
/// `FineTuneConfig::default`'s `2e-4` over 3 epochs, and at that rate over
/// the committed 1372 pairs the held-out loss is lowest at the FIRST epoch
/// boundary on every seed and rises from there (a twelve-seed pilot: 3.17 →
/// 3.22 → 3.27 over three epochs while the train probe falls 3.32 → 2.45) —
/// the minimum is censored by the evaluation cadence and the run overfits
/// before its second evaluation, so a reader judging at the trajectory's
/// minimum reads its first point. A quarter of that rate over four epochs,
/// evaluated at every epoch, puts the minimum inside the trajectory and
/// keeps the learning movement (~0.13 of held-out loss from the untrained
/// model) far above the seed spread (~0.03) and the repeat floor (0,
/// bit-identical repeats).
pub const DEFAULT_LEARNING_RATE: f64 = 5e-5;
/// See [`DEFAULT_LEARNING_RATE`].
pub const DEFAULT_EPOCHS: usize = 4;
/// See [`DEFAULT_LEARNING_RATE`]: every epoch is evaluated, so the
/// trajectory's minimum is never between two evaluations.
pub const DEFAULT_EVAL_CADENCE: usize = 1;

/// The untrained adapter's file name inside a run's `--work-dir` — see this
/// module's doc ("What a run in another framework pairs against").
pub const INITIAL_ADAPTER_FILE: &str = "initial_adapter.safetensors";

/// Write `varmap` — a FRESH build's trainable tensors, nothing trained — to
/// `path` as safetensors, and return the sha256 (hex) of the bytes written.
/// The digest is measured off the file, so it names exactly what a reader of
/// that file will load.
fn dump_initial_adapter(
    varmap: &VarMap,
    path: &Path,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    varmap
        .save(path)
        .map_err(|e| format!("finetune-run: writing {}: {e}", path.display()))?;
    let (sha256, _len) = sha256_and_len(path).map_err(sendify)?;
    Ok(sha256)
}

/// The realized token-batch digests of one run
/// ([`crate::report::TrainRunPayload::train_token_ids_sha256`] /
/// `heldout_token_ids_sha256`) — `None` on a media task, whose rows are never
/// tokenized.
struct TokenDigests {
    train: Option<String>,
    heldout: Option<String>,
}

impl TokenDigests {
    /// Tokenize what one epoch of `TrainingLoop::run` and one
    /// `evaluate_held_out` pass will tokenize, through the trainer's own
    /// functions and with the trainer's own length bound
    /// (`min(max_seq_length, the encoder's positional capacity)`), and digest
    /// each stream. Runs once, outside every timed span.
    fn measure(
        params: &FinetuneRunParams,
        base_model: &LoadedModel,
        encoder_max_seq_length: Option<usize>,
        train_rows: &RowSet<'_>,
        heldout_rows: &RowSet<'_>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (Task::Text, Some(encoder_max_seq_length)) = (params.task, encoder_max_seq_length)
        else {
            return Ok(Self {
                train: None,
                heldout: None,
            });
        };
        let tokenizer = base_model
            .tokenizer()
            .ok_or("finetune-run: internal: a text run's base model carries no tokenizer")?;
        let effective_max = params.max_seq_length.min(encoder_max_seq_length);
        let batch = params.batch_size;

        let (train_split, val_split) = train_rows
            .loader(params.objective)?
            .split(params.validation_fraction);
        let mut epoch_batches =
            loader_token_batches(tokenizer, &train_split, batch, effective_max)?;
        // The validation pass runs only when the run monitors `val_loss`
        // (`TrainingLoop::run` skips it entirely under `train_loss`).
        if params.early_stopping_metric == EarlyStoppingMetric::ValLoss {
            epoch_batches.extend(loader_token_batches(
                tokenizer,
                &val_split,
                batch,
                effective_max,
            )?);
        }
        let heldout_batches = loader_token_batches(
            tokenizer,
            &heldout_rows.loader(params.objective)?,
            batch,
            effective_max,
        )?;
        Ok(Self {
            train: Some(token_batches_sha256(&epoch_batches)),
            heldout: Some(token_batches_sha256(&heldout_batches)),
        })
    }
}

/// A rung of the train-run ladder: how the SAME fine-tune reaches the
/// trainer. Adjacent rungs differ by one layer, and the layers above
/// `resident` are the engine's job path, then where that job runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rung {
    /// [`TrainingLoopBuilder`] driven directly over in-memory rows
    /// (`TrainingSource::Resident`): the trainer alone.
    Resident,
    /// The job path, in this process: the pair source registered, the job
    /// admitted and submitted through the session, its training set
    /// materialised, the rows streamed back through the training-set table's
    /// scan, the adapter published through the artifact store.
    Streamed,
    /// The same job, claimed by one process and placed on a Ballista
    /// executor in another.
    Placed,
    /// The same job on the deployed topology's role configs: a scheduler, a
    /// query tier the job is submitted through over the public surface, and
    /// compute processes that claim and train it.
    ShapeD,
}

impl Rung {
    pub fn as_str(self) -> &'static str {
        match self {
            Rung::Resident => "resident",
            Rung::Streamed => "streamed",
            Rung::Placed => "placed",
            Rung::ShapeD => "shape-d",
        }
    }
}

impl std::str::FromStr for Rung {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "resident" => Ok(Rung::Resident),
            "streamed" => Ok(Rung::Streamed),
            "placed" => Ok(Rung::Placed),
            "shape-d" => Ok(Rung::ShapeD),
            other => Err(format!(
                "unknown --rung {other:?}; expected resident, streamed, placed, or shape-d"
            )),
        }
    }
}

/// The run metrics the trainer writes and the worker finalizes
/// (`TrainingResult::metrics_json`, `jobs.result`), read back by the fields
/// every rung reports from.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RunMetrics {
    pub final_loss: f64,
    pub total_steps: usize,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: chrono::DateTime<chrono::Utc>,
    pub epoch_walls: Vec<jammi_ai::fine_tune::trainer::EpochWall>,
    /// The media front end's wall inside the run's steps; absent on a text
    /// task.
    #[serde(default)]
    pub media_front_end_wall_s: Option<f64>,
    pub kernel_dispatches: KernelDispatches,
    pub kernels_disabled: KernelsDisabled,
    #[serde(default)]
    pub timeline: Option<Timeline>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct KernelsDisabled {
    pub requested: Vec<String>,
    pub fired: Vec<String>,
}

impl RunMetrics {
    pub fn parse(json: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        serde_json::from_str(json)
            .map_err(|e| format!("finetune-run: the run's metrics do not parse: {e}").into())
    }

    /// The trainer's own run span: its first statement to its final adapter
    /// written.
    pub fn train_run_wall_s(&self) -> f64 {
        (self.completed_at - self.started_at).as_seconds_f64()
    }

    /// The stations around this run, from its timeline; `submitted_at` is
    /// the job row's creation, or none for a run that was never queued.
    pub fn stations(&self, submitted_at: Option<chrono::DateTime<chrono::Utc>>) -> Stations {
        self.timeline.as_ref().map_or_else(Stations::default, |t| {
            Stations::of(t, submitted_at, self.completed_at)
        })
    }
}

/// The files a run ended on — the final adapter as trained, or as
/// published and fetched back.
pub enum Bundle {
    /// The trainer's own directory, held alive here.
    Trained(tempfile::TempDir),
    /// A published bundle, verified against its manifest.
    Published(LocalArtifact),
}

impl Bundle {
    pub fn dir(&self) -> &Path {
        match self {
            Bundle::Trained(dir) => dir.path(),
            Bundle::Published(local) => local.dir(),
        }
    }

    /// The bundle's files digest — the ONE fold both forms share
    /// (`artifact_files_digest` over the trained directory's files, which
    /// are exactly the files a publish lists).
    pub fn digest(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Bundle::Trained(dir) => artifact_files_digest(dir.path()).map_err(sendify),
            Bundle::Published(local) => published_artifact_digest(local).map_err(sendify),
        }
    }
}

/// What a rung hands back once its run has ended: the artifact and the
/// epoch checkpoints to evaluate, the run's own metrics, and where it ran.
pub struct TrainedRun {
    pub bundle: Bundle,
    /// Every encoder forward the run's own training loop took, when that
    /// loop ran in THIS process (the `resident` rung) — the same window
    /// the run's kernel dispatch counters are taken over, so the two
    /// divide. A rung whose trainer ran behind the job path — in an
    /// embedded worker, on an executor, on a compute process — has no
    /// handle on that loop, and the leg reports no count rather than a
    /// zero that reads like a measurement.
    pub forwards: Option<u64>,
    /// Every epoch's checkpoint, in epoch order — `keep_last_n_checkpoints`
    /// set to the run's epochs keeps them all.
    pub epoch_bundles: Vec<LocalArtifact>,
    pub metrics: RunMetrics,
    /// When the job was queued; none on a rung with no queue.
    pub submitted_at: Option<chrono::DateTime<chrono::Utc>>,
    pub ran_on: RanOn,
}

/// Rows of a TEXT run in the training set's committed order
/// (`full_tuple_v1`): sorted by the projected tuple the objective trains
/// on, so the `resident` rung and every rung above it train the same
/// sequence. A run's objective decides the projection, and so the key —
/// MNRL projects the triplet to its pair. Held-out rows are never reordered:
/// their committed order is scoring identity.
pub(crate) fn committed_order(rows: &[IdTriplet], objective: Objective) -> Vec<IdTriplet> {
    let mut sorted = rows.to_vec();
    match objective {
        Objective::Triplet => sorted.sort_by(|a, b| {
            (&a.anchor, &a.positive, &a.negative).cmp(&(&b.anchor, &b.positive, &b.negative))
        }),
        Objective::Mnrl => {
            sorted.sort_by(|a, b| (&a.anchor, &a.positive).cmp(&(&b.anchor, &b.positive)))
        }
    }
    sorted
}

/// The JSONL source the job path's rungs register: the training rows'
/// text columns under the names the training-set producer projects.
pub(crate) fn write_training_source(
    rows: &[IdTriplet],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut body = String::new();
    for row in rows {
        let line = serde_json::json!({
            "anchor": row.anchor,
            "positive": row.positive,
            "negative": row.negative,
        });
        body.push_str(&line.to_string());
        body.push('\n');
    }
    std::fs::write(path, body)?;
    Ok(())
}

/// Everything a rung's run and the evaluation after it share: the resolved
/// checkpoint, the device, the run's configuration, and a local file-backed
/// catalog and artifact store — the trainer's own construction surface,
/// which the `resident` rung trains against and every rung evaluates
/// against.
pub struct RunContext {
    pub checkpoint: Checkpoint,
    pub device: Device,
    pub device_config: DeviceConfig,
    pub base_model: Arc<LoadedModel>,
    pub config: FineTuneConfig,
    pub catalog: Arc<Catalog>,
    pub artifact_store: Arc<ArtifactStore>,
    pub artifact_dir: PathBuf,
    pub job_id: String,
    pub train_rows: Vec<IdTriplet>,
    /// The leg's scratch, absolute: what the store's root and every file a
    /// fleet member reads are named by.
    pub work_dir: PathBuf,
}

impl RunContext {
    fn open(
        params: &FinetuneRunParams,
        train_rows: Vec<IdTriplet>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = match params.cuda_device {
            Some(ordinal) => Device::new_cuda(ordinal)?,
            None => Device::Cpu,
        };
        let gpu_device = params.cuda_device.map(|o| o as i32).unwrap_or(-1);
        let device_config = DeviceConfig {
            gpu_device,
            // One device: a bench leg runs on the ordinal it was given.
            devices: vec![gpu_device],
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: params.backbone_dtype,
        };
        let checkpoint = Checkpoint::resolve(&params.model_dir)?;
        let base_model =
            load_base_model(&checkpoint, &params.model_dir, params.task, &device_config)?;

        // The work dir names the artifact store's root, which reads its
        // bundles back by path: absolute, never as given.
        std::fs::create_dir_all(&params.work_dir)?;
        let work_dir = std::fs::canonicalize(&params.work_dir)?;
        let catalog_dir = work_dir.join("catalog");
        std::fs::create_dir_all(&catalog_dir)?;
        let catalog =
            Arc::new(tokio::runtime::Handle::current().block_on(Catalog::open(&catalog_dir))?);
        let artifact_store_root = work_dir.join("artifacts");
        std::fs::create_dir_all(&artifact_store_root)?;
        let artifact_cache = work_dir.join("artifact_cache");
        std::fs::create_dir_all(&artifact_cache)?;
        let store_url = StorageUrl::parse(
            artifact_store_root
                .to_str()
                .ok_or("finetune-run: work_dir is not valid UTF-8")?,
        )?;
        let artifact_store = Arc::new(ArtifactStore::with_root(
            store_url,
            StorageRegistry::new(),
            artifact_cache,
        )?);
        let artifact_dir = work_dir.join("training");
        std::fs::create_dir_all(&artifact_dir)?;

        // A real admitted job row in the local catalog: the trainer's
        // durable epoch checkpoints are staged under a claimed job, so the
        // `resident` rung trains under one exactly as a worker's attempt
        // does; the evaluation loops of every rung are built against the
        // same row.
        let job_id = format!("finetune-run-{}-{}", params.seed, {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        });
        let model_row_id = format!("{}::finetune-run", params.model_dir.display());
        let model_catalog_pk = format!("{model_row_id}::1");
        tokio::runtime::Handle::current().block_on(catalog.register_model(
            RegisterModelParams {
                model_id: &model_row_id,
                version: 1,
                model_type: checkpoint.family.adapter_model_type(),
                backend: "candle",
                task: params.task.model_task(),
                base_model_id: None,
                external_location: None,
                config_json: None,
            },
        ))?;
        let config = base_config(params);
        let output_model_id = fine_tuned_model_id(&job_id);
        let training_spec = TrainingSpec::FineTune {
            source: format!("finetune-run:{}", params.model_dir.display()),
            columns: params.objective.columns(),
            method: FineTuneMethod::Lora,
            task: params.task.model_task(),
            common: TrainingCommon {
                base_model: model_row_id.clone(),
                config: config.clone(),
                world_size: DEFAULT_WORLD_SIZE,
                cache: CachePolicy::Bypass,
            },
        };
        let admitted = admit_training_spec(&JammiConfig::default(), training_spec)?;
        tokio::runtime::Handle::current().block_on(submit_admitted_training(
            &catalog,
            &admitted,
            &job_id,
            &model_catalog_pk,
            &output_model_id,
            0,
            None,
        ))?;
        tokio::runtime::Handle::current()
            .block_on(catalog.claim_next(
                RESIDENT_WORKER_ID,
                &["fine_tune"],
                std::time::Duration::from_secs(3600),
            ))?
            .ok_or("finetune-run: the just-created training job was not claimable")?;

        Ok(Self {
            checkpoint,
            device,
            device_config,
            base_model,
            config,
            catalog,
            artifact_store,
            artifact_dir,
            job_id,
            train_rows,
            work_dir,
        })
    }

    /// A fresh loop over this run's spec: the LoRA-injected encoder built
    /// from `(seed, target_modules)`, untrained. What the `resident` rung
    /// trains, and what every rung evaluates its checkpoints through.
    fn fresh_loop(
        &self,
        params: &FinetuneRunParams,
    ) -> Result<FreshLoop, Box<dyn std::error::Error + Send + Sync>> {
        let varmap = VarMap::new();
        let (encoder, adapter_cfg) = build_encoder_adapters(
            &self.checkpoint,
            params.task,
            &params.target_modules,
            &params.layers_to_transform,
            params.lora_rank,
            params.lora_alpha,
            params.lora_dropout,
            params.lora_init,
            params.backbone_dtype,
            params.seed,
            &self.device,
            &varmap,
        )?;
        let census = encoder.fusible_site_census();
        // A token-sequence bound: a media tower has none.
        let max_seq_length = match params.task {
            Task::Text => Some(encoder.max_seq_length().map_err(sendify)?),
            Task::Image | Task::Audio => None,
        };
        let target = TrainingTarget::EncoderAdapters(Box::new(EncoderAdaptersTarget {
            encoder,
            adapter_cfg,
        }));
        let training_loop = TrainingLoopBuilder::new(target, varmap.clone(), self.config.clone())
            .applied_learning_rate(params.applied_learning_rate)
            .base_model(Arc::clone(&self.base_model))
            .task(params.task.model_task())
            .job_id(self.job_id.clone())
            .catalog(Arc::clone(&self.catalog))
            .artifact_dir(self.artifact_dir.clone())
            .device(self.device.clone())
            .cancel(Arc::new(AtomicBool::new(false)))
            .artifact_store(Arc::clone(&self.artifact_store))
            .build()?;
        Ok(FreshLoop {
            training_loop,
            varmap,
            census,
            max_seq_length,
        })
    }
}

/// A [`RunContext::fresh_loop`]: the loop, its trainable tensors, and what
/// was read off the encoder before it was moved into the target.
struct FreshLoop {
    training_loop: TrainingLoop,
    varmap: VarMap,
    census: jammi_encoders::FusibleSiteCensus,
    /// The encoder's positional capacity, the trainer's own length bound;
    /// none for a media tower.
    max_seq_length: Option<usize>,
}

/// The worker id the `resident` rung's local job row is claimed under.
const RESIDENT_WORKER_ID: &str = "finetune-run-resident";

impl Objective {
    /// The training-set columns this objective projects — the tuple the
    /// committed order sorts by.
    pub fn columns(self) -> Vec<String> {
        match self {
            Objective::Triplet => vec!["anchor".into(), "positive".into(), "negative".into()],
            Objective::Mnrl => vec!["anchor".into(), "positive".into()],
        }
    }
}

/// The `resident` rung: one `TrainingLoop::run` over the rows in memory.
fn train_resident(
    call: &BlockingCall,
    params: &FinetuneRunParams,
    ctx: &RunContext,
    train_rows: &RowSet<'_>,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    let FreshLoop {
        mut training_loop, ..
    } = ctx.fresh_loop(params)?;
    let train_loader = train_rows.loader(params.objective)?;
    let result = training_loop.run(call, TrainingSource::Resident(train_loader))?;
    let metrics = RunMetrics::parse(&result.metrics_json)?;
    // The job's retained checkpoints, from the catalog — the same window a
    // finalize publishes.
    let runtime = tokio::runtime::Handle::current();
    let retained = match ctx
        .config
        .keep_last_n_checkpoints
        .and_then(|keep| std::num::NonZeroUsize::new(keep as usize))
    {
        Some(keep) => runtime.block_on(ctx.artifact_store.retained_checkpoints(
            &ctx.catalog,
            &ctx.job_id,
            keep,
        ))?,
        None => Vec::new(),
    };
    let epoch_bundles = retained
        .iter()
        .map(|(_epoch, staged)| {
            runtime.block_on(ctx.artifact_store.fetch_artifact(staged.artifact().url()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TrainedRun {
        bundle: Bundle::Trained(result.artifact_dir),
        forwards: Some(training_loop.encoder_forwards()),
        epoch_bundles,
        metrics,
        submitted_at: None,
        ran_on: RanOn::this_process("bench"),
    })
}

/// The `streamed` rung: the job path in this process. The rows become a
/// registered file source; the job is admitted and submitted through the
/// session the way every submit edge submits it; an embedded worker claims
/// it, materialises its training set, streams the rows back and publishes
/// the adapter; the published bundles are fetched back to evaluate.
fn train_streamed(
    params: &FinetuneRunParams,
    ctx: &RunContext,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    use jammi_ai::fine_tune::worker::EmbeddedWorker;
    use jammi_db::config::{GpuConfig, WorkerConfig};
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let engine_dir = ctx.work_dir.join("engine");
    std::fs::create_dir_all(&engine_dir)?;
    let source_path = ctx.work_dir.join("training_source.jsonl");
    write_training_source(&ctx.train_rows, &source_path)?;
    let gpu_device = ctx.device_config.gpu_device;
    let config = JammiConfig {
        artifact_dir: engine_dir.clone(),
        gpu: GpuConfig {
            device: gpu_device,
            devices: Some(vec![gpu_device]),
            require_gpu: gpu_device >= 0,
            ..Default::default()
        },
        worker: WorkerConfig {
            enabled: true,
            idle_poll_secs: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let handle = tokio::runtime::Handle::current();
    let session = handle.block_on(InferenceSession::open(config))?;
    handle.block_on(session.add_source(
        STREAMED_SOURCE,
        SourceType::File,
        SourceConnection {
            url: Some(format!("file://{}", source_path.display())),
            format: Some(FileFormat::JsonLines),
            ..Default::default()
        },
    ))?;
    let worker = EmbeddedWorker::spawn(&session)?;
    let submitted_at = chrono::Utc::now();
    let job = handle.block_on(session.run_training_spec(TrainingSpec::FineTune {
        source: STREAMED_SOURCE.to_string(),
        columns: params.objective.columns(),
        method: FineTuneMethod::Lora,
        task: params.task.model_task(),
        common: TrainingCommon {
            base_model: format!("local:{}", params.model_dir.display()),
            config: ctx.config.clone(),
            world_size: DEFAULT_WORLD_SIZE,
            cache: CachePolicy::Bypass,
        },
    }))?;
    handle.block_on(job.wait())?;
    handle.block_on(worker.stop_and_join())?;
    let published = handle.block_on(published_run(&session, &job.job_id, &job.model_id))?;
    let ran_on = RanOn {
        role: "worker".to_string(),
        ..published.ran_on
    };
    Ok(TrainedRun {
        submitted_at: Some(submitted_at),
        forwards: None,
        ran_on,
        ..published
    })
}

/// The file source the job-path rungs register the training rows as.
pub const STREAMED_SOURCE: &str = "finetune_run_training";

/// A completed job's published run, read through `session`'s catalog and
/// artifact store: the final bundle and every epoch checkpoint the finalize
/// published, the metrics the worker recorded, and who held the attempt.
/// `ran_on.role` is the reader's to name.
pub async fn published_run(
    session: &Arc<InferenceSession>,
    job_id: &str,
    model_id: &str,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    let catalog = session.catalog().pinned_to_tenant(None);
    let record = catalog.get_job(job_id).await?;
    let result_json = record
        .result
        .as_deref()
        .ok_or_else(|| format!("finetune-run: job {job_id} completed with no result"))?;
    let jammi_ai::jobs::JobResult::Model {
        artifact_path,
        metrics,
        ..
    } = serde_json::from_str(result_json)?
    else {
        return Err(format!("finetune-run: job {job_id}'s result is not a model").into());
    };
    let metrics = RunMetrics::parse(
        metrics
            .as_deref()
            .ok_or_else(|| format!("finetune-run: job {job_id} recorded no metrics"))?,
    )?;
    let store = session.artifact_store();
    let bundle = store
        .fetch_artifact(&StorageUrl::parse(&artifact_path)?)
        .await?;
    let mut epoch_bundles = Vec::with_capacity(metrics.epoch_walls.len());
    for epoch in 0..metrics.epoch_walls.len() {
        let name = format!("{model_id}:epoch_{epoch}");
        let model = catalog
            .get_model(&name)
            .await?
            .ok_or_else(|| format!("finetune-run: epoch checkpoint {name} was not published"))?;
        let url = model
            .location
            .as_ref()
            .ok_or_else(|| format!("finetune-run: {name} names no artifact"))?
            .bundle_url()?;
        epoch_bundles.push(store.fetch_artifact(&url).await?);
    }
    let claimed_by = record
        .claimed_by
        .clone()
        .ok_or_else(|| format!("finetune-run: completed job {job_id} names no claimant"))?;
    let worker = catalog
        .list_workers()
        .await?
        .into_iter()
        .find(|w| w.instance_id == claimed_by);
    let ran_on = RanOn {
        instance_id: claimed_by.clone(),
        label: worker.as_ref().and_then(|w| w.label.clone()),
        host: worker.as_ref().and_then(|w| w.host.clone()),
        role: String::new(),
        evidence: vec![format!(
            "jobs.claimed_by={claimed_by} attempts={} releases={}",
            record.attempts, record.releases
        )],
    };
    let submitted_at = chrono::DateTime::parse_from_rfc3339(&record.created_at)
        .map(|t| t.with_timezone(&chrono::Utc))
        .map_err(|e| {
            format!(
                "finetune-run: job {job_id} created_at {:?}: {e}",
                record.created_at
            )
        })?;
    Ok(TrainedRun {
        bundle: Bundle::Published(bundle),
        forwards: None,
        epoch_bundles,
        metrics,
        submitted_at: Some(submitted_at),
        ran_on,
    })
}

/// The dispatch counters a leg reports, read off the run's own window
/// (`RunMetrics::kernel_dispatches`) by the op names each seam admits
/// under — the same window on every rung, wherever the run happened.
struct NamedDispatches {
    ln: KernelDispatchCount,
    rope: KernelDispatchCount,
    softmax: KernelDispatchCount,
    geglu: KernelDispatchCount,
    gelu: KernelDispatchCount,
    lora_epilogue: KernelDispatchCount,
    lora_linear: KernelDispatchCount,
    attention_block: KernelDispatchCount,
    adamw: KernelDispatchCount,
    attention_block_flash: KernelDispatchCount,
}

impl NamedDispatches {
    fn read(dispatches: &KernelDispatches) -> Self {
        Self {
            ln: dispatches.of("layer_norm_fused"),
            rope: dispatches.of("rope_fused"),
            softmax: dispatches.of("softmax_last_dim_fused"),
            geglu: dispatches.of("geglu_fused"),
            gelu: dispatches.of("gelu_erf_fused"),
            lora_epilogue: dispatches.of("lora_epilogue"),
            lora_linear: dispatches.of("lora_linear_fused"),
            attention_block: dispatches.of("attention_block_fused"),
            adamw: dispatches.of("adamw_step_fused"),
            attention_block_flash: dispatches.of("attention_block_flash"),
        }
    }
}

/// Run this tier on `params.rung`: the run itself through the rung's own
/// path, then one evaluation of what it published — the untrained adapter,
/// every epoch's checkpoint and the final adapter scored on the held-out
/// fixture through a fresh loop over the same spec. See this module's own
/// doc.
pub fn run(
    call: &BlockingCall,
    params: &FinetuneRunParams,
) -> Result<Leg<TrainRunPayload>, Box<dyn std::error::Error + Send + Sync>> {
    validate(params)?;
    let train_rows_owned = match params.task {
        Task::Text => committed_order(&params.train_pairs, params.objective),
        Task::Image | Task::Audio => Vec::new(),
    };
    let ctx = RunContext::open(params, train_rows_owned)?;
    let train_rows = match params.task {
        Task::Text => RowSet::Text(&ctx.train_rows),
        Task::Image | Task::Audio => RowSet::Media(&params.train_media),
    };
    let heldout_rows = params.heldout_rows();

    let (checkpoint_config_sha256, _config_len) =
        sha256_and_len(&ctx.checkpoint.config_path).map_err(sendify)?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =
        sha256_and_len(&ctx.checkpoint.weights_path).map_err(sendify)?;

    // The device-memory window opens before the run and closes after the
    // last evaluation, on every rung: on a job rung the run's process may
    // be another, and its device is the same box's.
    let vram = VramWindow::open(device_memory_probe(params.cuda_device));

    let trained = match params.rung {
        Rung::Resident => train_resident(call, params, &ctx, &train_rows)?,
        Rung::Streamed => train_streamed(params, &ctx)?,
        Rung::Placed | Rung::ShapeD => crate::plane::train_run::train_on_fleet(params, &ctx)?,
    };
    if trained.epoch_bundles.len() != params.epochs {
        return Err(format!(
            "finetune-run: the run published {} epoch checkpoints for {} epochs — every epoch's \
             checkpoint is kept (keep_last_n_checkpoints = epochs), so the trajectory can be \
             scored off what the run itself wrote",
            trained.epoch_bundles.len(),
            params.epochs
        )
        .into());
    }

    let FreshLoop {
        mut training_loop,
        varmap,
        census: fusible_site_census,
        max_seq_length,
    } = ctx.fresh_loop(params)?;
    let initial_adapter_sha256 =
        dump_initial_adapter(&varmap, &ctx.work_dir.join(INITIAL_ADAPTER_FILE))?;
    let token_digests = TokenDigests::measure(
        params,
        &ctx.base_model,
        max_seq_length,
        &train_rows,
        &heldout_rows,
    )?;

    let heldout_ids: Vec<String> = heldout_rows.ids();
    let heldout_loader = heldout_rows.loader(params.objective)?;
    let probe_len = params.batch_size.min(train_rows.len());
    if probe_len == 0 || !probe_len.is_multiple_of(params.batch_size) {
        return Err(format!(
            "finetune-run: {} train pairs is fewer than --batch {} — cannot build a train-side \
             learning-happened probe batch",
            train_rows.len(),
            params.batch_size
        )
        .into());
    }
    let probe_rows = train_rows.take(probe_len);
    let probe_ids: Vec<String> = probe_rows.ids();
    let probe_loader = probe_rows.loader(params.objective)?;

    // The untrained model anchors both series: the held-out origin the
    // run's learning is measured from (`held_out_at_init`), and the
    // train-side probe's index 0.
    let held_out_at_init = training_loop
        .evaluate_held_out(&heldout_loader, &heldout_ids)?
        .mean;
    let mut train_probe_series: Vec<f64> = Vec::with_capacity(params.epochs + 1);
    train_probe_series.push(
        training_loop
            .evaluate_held_out(&probe_loader, &probe_ids)?
            .mean,
    );
    let mut trajectory = Trajectory { points: Vec::new() };
    let mut run_wall_s_cumulative = 0.0;
    let mut steps_wall_s_cumulative = 0.0;
    for (epoch_idx, bundle) in trained.epoch_bundles.iter().enumerate() {
        training_loop.load_weights(&bundle.dir().join(jammi_lora::ADAPTER_WEIGHTS_FILE))?;
        let wall = trained.metrics.epoch_walls.get(epoch_idx).ok_or_else(|| {
            format!("finetune-run: the run's metrics carry no wall for epoch {epoch_idx}")
        })?;
        run_wall_s_cumulative += wall.run_s;
        steps_wall_s_cumulative += wall.steps_s;
        let is_final = epoch_idx + 1 == params.epochs;
        let due = params.eval_cadence > 0 && (epoch_idx + 1).is_multiple_of(params.eval_cadence);
        if due || is_final {
            let held_out = training_loop.evaluate_held_out(&heldout_loader, &heldout_ids)?;
            trajectory.points.push(TrajectoryPoint {
                epoch: epoch_idx,
                held_out_mean: held_out.mean,
                run_wall_s_cumulative: Some(run_wall_s_cumulative),
                steps_wall_s_cumulative: Some(steps_wall_s_cumulative),
                held_out_tie_fraction: Some(held_out.tie_fraction),
                held_out_batch_partition_sha256: Some(held_out.batch_partition_sha256),
            });
        }
        train_probe_series.push(
            training_loop
                .evaluate_held_out(&probe_loader, &probe_ids)?
                .mean,
        );
    }
    // The endpoint is the PUBLISHED adapter — what the run serves — scored
    // through the same loop.
    training_loop.load_weights(&trained.bundle.dir().join(jammi_lora::ADAPTER_WEIGHTS_FILE))?;
    let held_out = training_loop.evaluate_held_out(&heldout_loader, &heldout_ids)?;
    let outcome_digest = trained.bundle.digest()?;

    let peak_vram_bytes = vram.close();
    let peak_rss_bytes = crate::rss::peak_rss_measurement();

    let dispatches = NamedDispatches::read(&trained.metrics.kernel_dispatches);
    let cumulative_steps = trained.metrics.total_steps;
    if let Err(message) = fused_dispatch_proof_gate(
        ctx.checkpoint.family,
        cumulative_steps,
        dispatches.attention_block.fused,
        dispatches.attention_block.eager,
        dispatches.attention_block_flash.fused,
        dispatches.attention_block_flash.declined,
        dispatches.lora_linear.fused,
        dispatches.lora_linear.eager,
    ) {
        return Err(message.into());
    }

    let kernels_disabled_requested = trained.metrics.kernels_disabled.requested.clone();
    let kernels_disabled_fired = trained.metrics.kernels_disabled.fired.clone();
    let resolved_attention_arm = attention_arm(&kernels_disabled_requested).to_string();
    let kernels_disabled_expected = match &params.expect_kernels_disabled {
        None => Vec::new(),
        Some(expected) => {
            // (2) The disable list's own safety property, read off the run's
            // process: a requested key that never disabled a live dispatch
            // is a typo, not evidence the forced-eager arm ran.
            let unmatched: Vec<&String> = kernels_disabled_requested
                .iter()
                .filter(|key| !kernels_disabled_fired.contains(key))
                .collect();
            if !unmatched.is_empty() {
                return Err(format!(
                    "finetune-run: JAMMI_KERNELS_DISABLE named op key(s) that never disabled a \
                     live dispatch this run (INVALID run, not a datum): {unmatched:?} — \
                     --expect-kernels-disabled was {expected:?}"
                )
                .into());
            }
            // (3) The premise the flag exists to prove: no named key
            // dispatched fused anywhere in the run's window.
            let fused_leaks: Vec<String> = expected
                .iter()
                .map(|key| (key, trained.metrics.kernel_dispatches.of(key)))
                .filter(|(_, count)| count.fused != 0)
                .map(|(key, count)| format!("{key}: fused={} eager={}", count.fused, count.eager))
                .collect();
            if !fused_leaks.is_empty() {
                return Err(format!(
                    "finetune-run: --expect-kernels-disabled named op key(s) that STILL \
                     dispatched fused this run (INVALID run, not a datum — the forced-eager arm \
                     was not actually eager): {fused_leaks:?}"
                )
                .into());
            }
            let mut recorded = expected.clone();
            recorded.sort();
            recorded
        }
    };

    let max_grad_norm = (params.max_grad_norm > 0.0).then_some(params.max_grad_norm);
    let (train_media_sha256, heldout_media_sha256) = match params.task {
        Task::Text => (None, None),
        Task::Image | Task::Audio => (
            Some(media_corpus_sha256(&params.train_media)),
            Some(media_corpus_sha256(&params.heldout_media)),
        ),
    };
    let MutantLabels {
        id: mutant_id,
        base_sha: mutant_base_sha,
        patch_sha256: mutant_patch_sha256,
    } = mutant_labels(params)?;
    let stations = trained.metrics.stations(trained.submitted_at);
    let epoch_walls = trained.metrics.epoch_walls.clone();

    let payload = TrainRunPayload {
        seed: params.seed,
        task: params.task.as_str().to_string(),
        batch: params.batch_size,
        max_seq_length: params.max_seq_length,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        lora_dropout: params.lora_dropout,
        lora_init: lora_init_as_str(params.lora_init).to_string(),
        margin: match params.objective {
            Objective::Triplet => Some(params.margin),
            Objective::Mnrl => None,
        },
        target_modules: params.target_modules.clone(),
        layers_to_transform: params.layers_to_transform.clone(),
        backbone_dtype: format!("{:?}", params.backbone_dtype).to_lowercase(),
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        max_grad_norm,
        warmup: None,
        row_lengths: None,
        epochs: params.epochs,
        // The rate the run was MEASURED at: a control leg's is zero.
        lr: match params.applied_learning_rate {
            AppliedLearningRate::Scheduled => params.learning_rate,
            AppliedLearningRate::Zero => 0.0,
        },
        schedule: format!("{:?}", params.lr_schedule).to_lowercase(),
        warmup_steps: params.warmup_steps,
        weight_decay: params.weight_decay,
        grad_accum: params.gradient_accumulation_steps,
        validation_fraction: params.validation_fraction,
        train_pairs_file_sha256: params.train_pairs_file_sha256.clone(),
        train_media_sha256,
        heldout_ids_sha256: params.heldout_ids_sha256.clone(),
        heldout_pairs_sha256: params.heldout_pairs_sha256.clone(),
        heldout_media_sha256,
        heldout_batch_partition_sha256: held_out.batch_partition_sha256.clone(),
        train_token_ids_sha256: token_digests.train,
        heldout_token_ids_sha256: token_digests.heldout,
        embedding_loss: params.objective.as_str().to_string(),
        temperature: match params.objective {
            Objective::Triplet => None,
            Objective::Mnrl => Some(params.temperature),
        },
        matryoshka_dims: params.matryoshka_dims.clone(),
        early_stopping_patience: params.early_stopping_patience,
        early_stopping_metric: match params.early_stopping_metric {
            EarlyStoppingMetric::TrainLoss => "train_loss".to_string(),
            EarlyStoppingMetric::ValLoss => "val_loss".to_string(),
        },
        eval_cadence: params.eval_cadence,
        kernels_disabled_expected,
        fusible_site_census,
        split_rule: "positional_fraction_split".to_string(),
        batched_forward: true,
        steps_measured: cumulative_steps,
        forwards_measured: trained.forwards,
        // The rayon GLOBAL pool size this process actually executed
        // under, read via `jammi_ai::fine_tune::media_front_end_pool_threads()`
        // (ai-core's own seam — never `rayon::current_num_threads()` called
        // directly here, so this crate never gains a direct `rayon` dep) —
        // MACHINE/BUILD provenance, never identity. See
        // `Leg<TrainRunPayload>::rayon_pool_threads`'s own doc.
        rayon_pool_threads: jammi_ai::fine_tune::media_front_end_pool_threads(),
        initial_adapter_sha256,
        rung: params.rung.as_str().to_string(),
        final_epoch: params.epochs - 1,
        held_out_count: held_out.count,
        final_loss_diagnostic: trained.metrics.final_loss,
        train_run_wall_s: trained.metrics.train_run_wall_s(),
        media_front_end_wall_s: trained.metrics.media_front_end_wall_s,
        epoch_walls: epoch_walls.clone(),
    };
    let provenance = Provenance {
        device_name: crate::finetune_step::device_name(params.cuda_device),
        build_features: crate::report::build_features()
            .into_iter()
            .map(str::to_owned)
            .collect(),
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        arm: crate::kernel_arm::arm_label(&kernels_disabled_requested).to_string(),
        kernels_disabled_requested,
        kernels_disabled_fired,
        attention_arm: resolved_attention_arm,
        mutant: MutantStamp {
            mutant_id,
            mutant_base_sha,
            mutant_patch_sha256,
        },
        ran_on: Some(trained.ran_on),
    };
    let measured = Measured {
        // A training run's timed iteration is its optimizer step: every
        // epoch's `step_walls`, in run order.
        iter_wall_s: Some(
            epoch_walls
                .iter()
                .flat_map(|w| w.step_walls.iter().copied())
                .collect(),
        ),
        work: Some(train_rows.len() as f64),
        peak_rss_bytes,
        peak_vram_bytes,
        outcome_digest: Some(outcome_digest),
        held_out_example_mean: Some(held_out.mean),
        held_out_at_init: Some(held_out_at_init),
        trajectory: trajectory.points,
        stations,
        ..Default::default()
    };
    let facts = Facts {
        train_probe_series: Some(train_probe_series),
        // A DECLARED premise, not a measurement: the encoder decides
        // dense-vs-padded per forward off the mask and this tier reads no
        // per-forward `jammi_encoders::CompactedBatch` signal back to check
        // the caller against — unlike `kernels_disabled_requested`, which
        // reads a real process-resolved env-var state. Recorded verbatim
        // (CALLER-declared, default `false` matching the committed
        // fixture's padded transport) so a downstream merger's conjunctive
        // premise leg has a concrete, honestly-scoped, checkable fact
        // rather than an inferred one — see
        // `FinetuneRunParams::expect_dense`'s own doc.
        admission_is_dense: Some(params.expect_dense),
        tie_fraction: Some(held_out.tie_fraction),
        dispatch: Some(DispatchCounters {
            ln_fused_dispatches: dispatches.ln.fused,
            ln_eager_dispatches: dispatches.ln.eager,
            rope_fused_dispatches: dispatches.rope.fused,
            rope_eager_dispatches: dispatches.rope.eager,
            softmax_fused_dispatches: dispatches.softmax.fused,
            softmax_eager_dispatches: dispatches.softmax.eager,
            geglu_fused_dispatches: dispatches.geglu.fused,
            geglu_eager_dispatches: dispatches.geglu.eager,
            gelu_fused_dispatches: dispatches.gelu.fused,
            gelu_eager_dispatches: dispatches.gelu.eager,
            lora_epilogue_fused_dispatches: dispatches.lora_epilogue.fused,
            lora_epilogue_eager_dispatches: dispatches.lora_epilogue.eager,
            lora_linear_fused_dispatches: dispatches.lora_linear.fused,
            lora_linear_eager_dispatches: dispatches.lora_linear.eager,
            attention_block_fused_dispatches: dispatches.attention_block.fused,
            attention_block_eager_dispatches: dispatches.attention_block.eager,
            adamw_fused_dispatches: dispatches.adamw.fused,
            adamw_eager_dispatches: dispatches.adamw.eager,
            attention_block_flash_fused_dispatches: dispatches.attention_block_flash.fused,
            attention_block_flash_declined_dispatches: dispatches.attention_block_flash.declined,
        }),
    };
    let leg = Leg::new(payload, provenance, measured, facts);
    leg.to_value();
    Ok(leg)
}

/// Every refusal that needs no device, checkpoint or filesystem: the
/// parameter shape, the arm's declared-vs-resolved check, and the rungs'
/// own constraints.
fn validate(params: &FinetuneRunParams) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if params.early_stopping_patience < 10_000 {
        return Err(format!(
            "finetune-run: --early-stopping-patience {} is below 10_000, the \
             never-stops setting — a run that can early-stop before the configured epoch \
             budget cannot be paired at the FINAL epoch the paired sign test requires",
            params.early_stopping_patience
        )
        .into());
    }
    if params.epochs == 0 {
        return Err("finetune-run: --epochs 0 has no final epoch to measure".into());
    }
    if params.learning_rate.is_nan() || params.learning_rate <= 0.0 {
        return Err(format!(
            "finetune-run: --lr {} is not a positive learning rate. A job that cannot learn is \
             refused at admission; the negative control is this same job run with its updates \
             nulled — pass the sweep's --lr together with --zero-lr-control",
            params.learning_rate
        )
        .into());
    }
    if params.rung != Rung::Resident {
        if params.applied_learning_rate == AppliedLearningRate::Zero {
            return Err(format!(
                "finetune-run: --zero-lr-control is a way of running the trainer below \
                 admission, which the {} rung's job path never exposes — the control leg \
                 belongs to the resident rung, where the torch edge is judged",
                params.rung.as_str()
            )
            .into());
        }
        if params.task != Task::Text {
            return Err(format!(
                "finetune-run: --rung {} trains from a registered text source; a media \
                 task's rows are files, which only the resident rung reads",
                params.rung.as_str()
            )
            .into());
        }
    }
    mutant_labels(params)?;
    // `--expect-kernels-disabled`, check (1) of 3: set EQUALITY with this
    // process's real `JAMMI_KERNELS_DISABLE`; (2) and (3) read the run's own
    // process at the end of [`run`].
    if let Some(expected) = &params.expect_kernels_disabled {
        let requested = jammi_kernels::admission::disabled_ops_requested();
        if requested != *expected {
            return Err(format!(
                "finetune-run: --expect-kernels-disabled {expected:?} does not exactly equal \
                 this process's real JAMMI_KERNELS_DISABLE (which resolved to {requested:?}) — \
                 the env var was dropped, mistyped, not forwarded to this process, or carries \
                 key(s) beyond what this leg declared (INVALID run, not a datum)"
            )
            .into());
        }
    }
    params.validate_rows_match_task()?;
    let train_rows = params.train_rows();
    let heldout_rows = params.heldout_rows();
    train_rows.validate_media_members_are_distinct("--train-jsonl")?;
    heldout_rows.validate_media_members_are_distinct("--heldout-jsonl")?;
    if heldout_rows.is_empty() || !heldout_rows.len().is_multiple_of(params.batch_size) {
        return Err(format!(
            "finetune-run: {} held-out pairs is not a nonzero multiple of --batch {} (the seam \
             refuses this too, but failing here names the fixture, not an opaque trainer error)",
            heldout_rows.len(),
            params.batch_size
        )
        .into());
    }
    Ok(())
}

/// The mutant trio, trimmed and shape-checked, all-or-none — see
/// [`FinetuneRunParams::mutant_id`]. A trio that was never touched is three
/// `None`s; a fully labelled one is three non-empty values with a git sha
/// (7-40 hex) and a sha256 (64 hex); everything else is refused.
fn mutant_labels(
    params: &FinetuneRunParams,
) -> Result<MutantLabels, Box<dyn std::error::Error + Send + Sync>> {
    let trimmed = |raw: &Option<String>| {
        raw.as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let mutant_id = trimmed(&params.mutant_id);
    let mutant_base_sha = trimmed(&params.mutant_base_sha).map(|s| s.to_lowercase());
    let mutant_patch_sha256 = trimmed(&params.mutant_patch_sha256).map(|s| s.to_lowercase());
    let never_touched = params.mutant_id.is_none()
        && params.mutant_base_sha.is_none()
        && params.mutant_patch_sha256.is_none();
    let fully_labeled =
        mutant_id.is_some() && mutant_base_sha.is_some() && mutant_patch_sha256.is_some();
    if !never_touched && !fully_labeled {
        let describe = |raw: &Option<String>, trimmed: &Option<String>| match (raw, trimmed) {
            (None, _) => "absent".to_string(),
            (Some(_), None) => "supplied but empty-or-whitespace-only".to_string(),
            (Some(_), Some(v)) => format!("{v:?}"),
        };
        return Err(format!(
            "finetune-run: --mutant-id/--mutant-base-sha/--mutant-patch-sha256 are all-or-none \
             (a value that is empty or whitespace-only after trimming is not a real label, but \
             supplying one is also not the same as never touching the flag) — got \
             mutant_id={}, mutant_base_sha={}, mutant_patch_sha256={} (a partial or blank \
             mutant label cannot be attributed to a specific, auditable mutant patch)",
            describe(&params.mutant_id, &mutant_id),
            describe(&params.mutant_base_sha, &mutant_base_sha),
            describe(&params.mutant_patch_sha256, &mutant_patch_sha256),
        )
        .into());
    }
    if let (Some(base_sha), Some(patch_sha256)) =
        (mutant_base_sha.as_deref(), mutant_patch_sha256.as_deref())
    {
        let is_hex = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit());
        if !(7..=40).contains(&base_sha.len()) || !is_hex(base_sha) {
            return Err(format!(
                "finetune-run: --mutant-base-sha {base_sha:?} must be 7-40 hex chars (a git \
                 commit sha), got length {} after trim",
                base_sha.len()
            )
            .into());
        }
        if patch_sha256.len() != 64 || !is_hex(patch_sha256) {
            return Err(format!(
                "finetune-run: --mutant-patch-sha256 {patch_sha256:?} must be exactly 64 hex \
                 chars (a sha256 hex digest), got length {} after trim",
                patch_sha256.len()
            )
            .into());
        }
    }
    Ok(MutantLabels {
        id: mutant_id,
        base_sha: mutant_base_sha,
        patch_sha256: mutant_patch_sha256,
    })
}

/// A mutant leg's trimmed, shape-checked labels — all three, or none.
struct MutantLabels {
    id: Option<String>,
    base_sha: Option<String>,
    patch_sha256: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // ── `token_batches_sha256` ──────────────────────────────────────────
    //
    // The reference values are CPython's (`hashlib` + `struct.pack("<I")`
    // over the documented layout), so these hold the digest a second
    // implementation of that layout produces — which is the whole point of
    // the field: `crates/jammi-bench/reference/torch_finetune_run.py`
    // computes it independently and the two must agree. Its own test file
    // (`test_torch_finetune_run_mirrors.py`) pins the same three values.

    fn token_batch(input_ids: Vec<Vec<u32>>, attention_masks: Vec<Vec<u32>>) -> BatchEncoding {
        let seq_len = input_ids.first().map_or(0, Vec::len);
        BatchEncoding {
            offsets: input_ids
                .iter()
                .map(|row| vec![(0, 0); row.len()])
                .collect(),
            input_ids,
            attention_masks,
            seq_len,
        }
    }

    #[test]
    fn token_batches_sha256_is_the_documented_layout() {
        let padded = token_batch(
            vec![vec![2, 7, 3], vec![2, 9, 0]],
            vec![vec![1, 1, 1], vec![1, 1, 0]],
        );
        let single = token_batch(vec![vec![2, 5, 6, 3]], vec![vec![1, 1, 1, 1]]);
        assert_eq!(
            token_batches_sha256(&[padded, single]),
            "365019a4cf7bd591cb4801106620245616e8549e382bcaf1cdc5091d75681534"
        );
        // No batches: sha256 of the empty message.
        assert_eq!(
            token_batches_sha256(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// The same ids flattened under a different partition are a different
    /// stream: the shape prefix is what keeps batch boundaries in the digest.
    #[test]
    fn token_batches_sha256_distinguishes_partitions_of_the_same_ids() {
        let merged = token_batch(vec![vec![2, 7, 3, 2, 9, 0]], vec![vec![1, 1, 1, 1, 1, 0]]);
        let single = token_batch(vec![vec![2, 5, 6, 3]], vec![vec![1, 1, 1, 1]]);
        assert_eq!(
            token_batches_sha256(&[merged, single]),
            "b3d6b6770ac17ef7b9f80bc1503901c991eeed8692f10244fa48dc040316dc47"
        );
    }

    // ── `media_corpus_sha256` ───────────────────────────────────────────
    //
    // The reference values below are computed by an INDEPENDENT
    // implementation (CPython's `hashlib`, not this crate's `sha2`), so
    // these assertions check that this function's MECHANISM produces the
    // digest a third party computing the documented construction would —
    // not merely that it is self-consistent. Re-derive any of them with:
    //
    //   python3 -c "import hashlib; abc=hashlib.sha256(b'abc').hexdigest(); \
    //               print(hashlib.sha256((abc*3).encode()).hexdigest())"
    //
    // `SHA256_OF_ABC`/`SHA256_OF_DEF` are the published single-message
    // vectors those one-liners start from.

    /// sha256("abc") — the published NIST test vector.
    const SHA256_OF_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    /// sha256("def").
    const SHA256_OF_DEF: &str = "cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34";

    /// A row whose three members all carry `digest` — the member BYTES are
    /// irrelevant to [`media_corpus_sha256`] by construction (it folds the
    /// measured per-member digests, which the loader computed off those
    /// bytes), so they are left empty here to make that explicit.
    fn row_with(id: &str, digest: &str) -> MediaTriplet {
        MediaTriplet {
            id: id.to_string(),
            anchor: Vec::new(),
            positive: Vec::new(),
            negative: Vec::new(),
            anchor_sha256: digest.to_string(),
            positive_sha256: digest.to_string(),
            negative_sha256: digest.to_string(),
        }
    }

    #[test]
    fn media_corpus_sha256_matches_an_independent_implementations_value() {
        // hashlib: sha256((sha256("abc").hexdigest() * 3).encode())
        assert_eq!(
            media_corpus_sha256(&[row_with("r0", SHA256_OF_ABC)]),
            "c7690b2f3f908d77bc23701243fa861aa6f01ac3d80570a4b000a073600dadb9"
        );
        // hashlib: sha256((abc*3 + def*3).encode()) — two rows, in order.
        assert_eq!(
            media_corpus_sha256(&[row_with("r0", SHA256_OF_ABC), row_with("r1", SHA256_OF_DEF),]),
            "00bdccf4b4640d72d226b663b9035f89698e3191e22c96af59be590508564a19"
        );
    }

    /// ORDER is part of the digest — the held-out corpus is folded in
    /// COMMITTED SCORING order, so a fold that sorted (or otherwise
    /// normalized) its input would silently make two differently-ordered
    /// fixtures compare as one. Pinned against the independent
    /// implementation's own swapped-order value, not merely asserted
    /// `!=` (a function returning a constant would pass a bare `!=` against
    /// nothing).
    #[test]
    fn media_corpus_sha256_is_order_sensitive() {
        let forward =
            media_corpus_sha256(&[row_with("r0", SHA256_OF_ABC), row_with("r1", SHA256_OF_DEF)]);
        let swapped =
            media_corpus_sha256(&[row_with("r1", SHA256_OF_DEF), row_with("r0", SHA256_OF_ABC)]);
        assert_eq!(
            swapped, "443fe3bf3115710f962f245974e24f4e96722aa63db035440a6c9dd8e9d706cd",
            "hashlib: sha256((def*3 + abc*3).encode())"
        );
        assert_ne!(forward, swapped);
    }

    /// The negative control that makes the field worth carrying at all: a
    /// corpus whose MANIFEST is unchanged but whose file CONTENT differs
    /// (the exact thing `train_pairs_file_sha256` cannot see) must produce a
    /// different digest. The row id is held fixed here precisely so the only
    /// varying input is content.
    #[test]
    fn media_corpus_sha256_changes_when_only_the_content_changes() {
        let a = media_corpus_sha256(&[row_with("same-id", SHA256_OF_ABC)]);
        let b = media_corpus_sha256(&[row_with("same-id", SHA256_OF_DEF)]);
        assert_ne!(
            a, b,
            "swapping the bytes behind an unchanged manifest must change the content digest"
        );
    }

    /// An empty corpus folds to sha256 of the EMPTY message — stated, not
    /// left to a reader to assume, because `run` computes this field
    /// unconditionally on a media task and an "empty means unknown"
    /// reading would be wrong (this tier refuses an empty train/held-out
    /// row set upstream, so the value never stands in for a missing one).
    #[test]
    fn media_corpus_sha256_of_no_rows_is_the_empty_message_digest() {
        assert_eq!(
            media_corpus_sha256(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// `cookbook/fixtures/tiny_bert` — the SAME generic, committed fixture
    /// `finetune_run_smoke.rs` drives via the compiled CLI (BERT
    /// architecture, real tokenizer, no consumer shape), resolved relative
    /// to this crate's own manifest dir so this IN-PROCESS test needs no
    /// extra dev-dependency.
    fn tiny_bert_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
    }

    /// The committed OpenCLIP fixture — `open_clip_config.json` +
    /// `open_clip_model.safetensors` + `tokenizer.json`, i.e. a checkpoint
    /// that hard-coded `config.json`/`model.safetensors` joins cannot see at
    /// all.
    fn tiny_open_clip_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_open_clip")
    }

    /// The committed HF-CLAP audio fixture.
    fn htsat_clap_tiny_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/htsat_clap_tiny")
    }

    /// Resolve a model dir through the ONE chain, panicking with the real
    /// message if it refuses (a resolution failure in a test that expects a
    /// buildable checkpoint is a fixture problem, not the case under test).
    fn checkpoint_of(model_dir: &Path) -> Checkpoint {
        Checkpoint::resolve(model_dir)
            .unwrap_or_else(|e| panic!("resolve {}: {e}", model_dir.display()))
    }

    /// Build the tiny, CPU-hermetic, deterministic [`FinetuneRunParams`] the
    /// non-perturbation test below drives — 4 synthetic train triplets (2
    /// batches at `batch_size 2`), 2 held-out triplets (1 batch), 2 epochs,
    /// `eval_cadence 1`, a nonzero learning rate (so training actually moves
    /// the weights) and a fixed `seed` — two calls built from this same
    /// function differ ONLY in which `work_dir` they own and (via the
    /// caller) `probe_at_init`.
    fn non_perturbation_test_params(work_dir: PathBuf) -> FinetuneRunParams {
        let mk = |offset: usize, n: usize| -> Vec<IdTriplet> {
            (0..n)
                .map(|i| {
                    let k = offset + i;
                    IdTriplet {
                        id: format!("row-{k}"),
                        anchor: format!("synthetic anchor sentence number {k} about widgets"),
                        positive: format!(
                            "synthetic positive sentence number {k} about widgets too"
                        ),
                        negative: format!(
                            "synthetic negative sentence number {k} about gadgets instead"
                        ),
                    }
                })
                .collect()
        };
        FinetuneRunParams {
            model_dir: tiny_bert_model_dir(),
            task: Task::Text,
            train_pairs: mk(0, 4),
            heldout_pairs: mk(100, 2),
            train_media: Vec::new(),
            heldout_media: Vec::new(),
            train_pairs_file_sha256: "0".repeat(64),
            heldout_ids_sha256: "1".repeat(64),
            heldout_pairs_sha256: "2".repeat(64),
            seed: 7,
            epochs: 2,
            eval_cadence: 1,
            batch_size: 2,
            learning_rate: 0.01,
            applied_learning_rate: AppliedLearningRate::Scheduled,
            lr_schedule: LrSchedule::Constant,
            warmup_steps: 0,
            weight_decay: 0.0,
            gradient_accumulation_steps: 1,
            // Matches the CLI's own
            // `--validation-fraction` default (`FinetuneRunArgs`'s
            // `default_value_t = 0.1`), not an arbitrary `0.0` — with 4 rows
            // this still rounds to a 0-row internal val split
            // (`round(4 * 0.1) == 0`), which is harmless here because
            // `early_stopping_metric` is `TrainLoss` (the `ValLoss`-only
            // empty-loader refusal in `TrainingLoop::run` never fires for
            // this metric), so this exercises the CLI's real
            // knob value without altering what the resume-cycle actually
            // trains over.
            validation_fraction: 0.1,
            early_stopping_patience: 10_000,
            early_stopping_metric: EarlyStoppingMetric::TrainLoss,
            max_grad_norm: 0.0,
            objective: Objective::Triplet,
            margin: 0.3,
            temperature: 20.0,
            matryoshka_dims: Vec::new(),
            lora_rank: 2,
            lora_alpha: 4.0,
            // Not `0.0`: that would make `jammi_lora::LoraLinear`'s
            // `dropout_masks` field structurally `None` for every LoRA layer
            // (`build_encoder_adapters`'s `lora_dropout_opt = (lora_dropout
            // > 0.0).then_some(..)` — `0.0` maps to `None`), so the mask
            // channel `init_probe_does_not_perturb_..._bitwise`'s own doc
            // names ("draws no dropout mask and so touches no RNG stream")
            // would be ABSENT, not merely idle: deleting the
            // `with_dropout_disabled` bracket entirely could not turn
            // this test red, because there would be no dropout stream for a
            // broken bracket to leave un-disabled. `0.05` matches the
            // CLI's own `--lora-dropout` default
            // (`FinetuneRunArgs`'s `default_value_t = 0.05`), making the
            // channel live — see
            // `dropout_forward_counter_is_live_at_the_default_lora_dropout_and_held_still_under_eval_mode`
            // below for the committed proof that the channel is Some and
            // that toggling training mode is what actually gates it (the
            // falsifiable mechanism `with_dropout_disabled` relies on).
            lora_dropout: 0.05,
            // The `--lora-init` default, so every test built on these params
            // exercises the default path; `lora_init_mode_reaches_both_lora_build_sites` below
            // overrides it explicitly to prove the other mode is threaded.
            lora_init: LoraInitMode::ZerosB,
            // `None` = the ordinary, unchecked case (no
            // `--expect-kernels-disabled` claim), which is what every test
            // in this module runs under.
            expect_kernels_disabled: None,
            target_modules: vec!["query".to_string(), "value".to_string()],
            layers_to_transform: None,
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            max_seq_length: 16,
            expect_dense: false,
            rung: Rung::Resident,
            plane: crate::plane::PlaneParams::default(),
            cuda_device: None,
            work_dir,
            mutant_id: None,
            mutant_base_sha: None,
            mutant_patch_sha256: None,
        }
    }

    /// The mask-counter proof: with the CLI's own `lora_dropout` (`0.05`,
    /// matching [`non_perturbation_test_params`]'s live value — see that
    /// function's own doc), every LoRA-wrapped layer's dropout forward
    /// counter (`jammi_lora::LoraLinear::dropout_position`,
    /// `AnyEncoder::dropout_positions`) is `Some` (the mask channel
    /// `init_probe_does_not_perturb_the_training_trajectory_bitwise`'s own
    /// doc names — "draws no dropout mask and so touches no RNG stream" —
    /// is actually PRESENT here, not structurally absent as it is at
    /// `lora_dropout: 0.0`), that it ADVANCES on a training-mode forward
    /// (the state the extra init probe would leave the encoder in if
    /// `TrainingLoop::with_dropout_disabled`'s bracket were bypassed or
    /// removed), and that it HOLDS STILL when training mode is off first —
    /// exactly the `set_training(false)` / call / `set_training(true)`
    /// sequence that bracket performs (`jammi-ai/src/fine_tune/trainer.rs`'s
    /// own doc on `with_dropout_disabled`) — around every
    /// `evaluate_held_out` call.
    ///
    /// This asserts via the mask-counter state rather than a test-only
    /// shadow bypass of `jammi-ai`'s bracket: it proves the property
    /// [`init_probe_does_not_perturb_the_training_trajectory_bitwise`] pins
    /// is LIVE (can fail), rather than vacuously true, by directly
    /// exhibiting one code path (training-mode forward) that DOES perturb
    /// the counter and one (eval-mode forward) that provably does not — the
    /// exact dichotomy a broken bracket would erase.
    ///
    /// Mutation (in `jammi-ai`, not committable from this crate): make
    /// `TrainingLoop::with_dropout_disabled` in
    /// `jammi-ai/src/fine_tune/trainer.rs` skip the `self.set_training(false)`
    /// call and `init_probe_does_not_perturb_the_training_trajectory_bitwise`
    /// fails at `lora_dropout: 0.05` (`named_with != named_without`): the
    /// extra init probe's dropout draw perturbs the first epoch's own
    /// dropout-stream position, changing its trained weights. At
    /// `lora_dropout: 0.0` the same mutation produces NO failure — the
    /// channel the pin protects is structurally absent there, which is why
    /// the params use `0.05`.
    #[test]
    fn dropout_forward_counter_is_live_at_the_default_lora_dropout_and_held_still_under_eval_mode()
    {
        let varmap = VarMap::new();
        let (mut encoder, _adapter_cfg) = build_encoder_adapters(
            &checkpoint_of(&tiny_bert_model_dir()),
            Task::Text,
            &["query".to_string(), "value".to_string()],
            &None,
            2,
            4.0,
            0.05,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        )
        .expect("build encoder adapters at lora_dropout 0.05");

        let positions_at_build = encoder
            .dropout_positions()
            .expect("dropout_positions at build");
        assert!(
            !positions_at_build.is_empty(),
            "lora_dropout 0.05 must leave at least one LoRA layer's dropout_masks Some (the \
             channel must be structurally present, unlike a lora_dropout 0.0 config): \
             {positions_at_build:?}"
        );
        assert!(
            positions_at_build.values().all(|&p| p == 0),
            "a freshly built encoder must start every dropout counter at 0: \
             {positions_at_build:?}"
        );

        let input_ids = crate::finetune_step::synthetic_ids(2, 4, 256, 99, &Device::Cpu);
        let mask = candle_core::Tensor::ones((2, 4), candle_core::DType::U32, &Device::Cpu)
            .expect("mask tensor");

        // Training-mode forward — the state the extra probe call would run
        // in if `with_dropout_disabled`'s bracket were bypassed: the
        // counter must advance.
        encoder.set_training(true);
        encoder
            .forward(&input_ids, &mask)
            .expect("training-mode forward");
        let positions_after_hot = encoder
            .dropout_positions()
            .expect("dropout_positions after hot forward");
        assert_ne!(
            positions_at_build, positions_after_hot,
            "a training-mode forward must advance the dropout forward counter — this is what \
             proves the channel is genuinely live, and that a bypassed with_dropout_disabled \
             bracket (which would leave the encoder in exactly this training=true state during \
             evaluate_held_out) WOULD perturb it"
        );

        // Eval-mode forward — exactly what `encoder.set_training(false)`
        // leaves the encoder in for the duration of
        // `TrainingLoop::with_dropout_disabled`'s bracket: the counter must
        // hold still.
        encoder.set_training(false);
        let positions_before_cold = encoder
            .dropout_positions()
            .expect("dropout_positions before cold forward");
        encoder
            .forward(&input_ids, &mask)
            .expect("eval-mode forward");
        let positions_after_cold = encoder
            .dropout_positions()
            .expect("dropout_positions after cold forward");
        assert_eq!(
            positions_before_cold, positions_after_cold,
            "an eval-mode (training=false) forward must NOT advance the dropout forward counter \
             — this is the exact mechanism with_dropout_disabled relies on to make the probe \
             read-only"
        );
    }

    /// Write a synthetic, ephemeral DistilBERT `config.json` + random-weight
    /// `model.safetensors` to a fresh tempdir and return it (dropped —
    /// deleting the dir — once the caller's `TempDir` goes out of scope).
    /// `layers` is the number of transformer blocks the synthetic config/
    /// weights get (every existing call site passes `1`; the
    /// `--layers-to-transform` restriction test below passes `2`, needing a
    /// second layer to restrict AWAY from).
    ///
    /// No committed HF-shaped DistilBERT fixture exists under
    /// `cookbook/fixtures` (unlike `tiny_bert`), and this crate does not
    /// invent a new committed fixture family to get one — this mirrors
    /// `jammi-encoders`' own `tests/it/distilbert.rs::write_synthetic_weights`
    /// exactly (same tensor names/prefix, same generic random content,
    /// generated fresh every run), just with the `config.json`/
    /// `model.safetensors` pair laid out on disk the way `build_encoder_adapters`
    /// reads them (that function takes a `model_dir`, not a config value +
    /// weights slice).
    fn write_synthetic_distilbert_model_dir(layers: usize) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let device = Device::Cpu;
        let (hidden, heads, inter, vocab, max_pos) = (32usize, 2usize, 64usize, 100usize, 128usize);

        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({
                "model_type": "distilbert",
                "dim": hidden,
                "n_layers": layers,
                "n_heads": heads,
                "hidden_dim": inter,
                "vocab_size": vocab,
                "max_position_embeddings": max_pos,
            })
            .to_string(),
        )
        .expect("write config.json");

        let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
        let randn = |shape: (usize, usize)| -> candle_core::Tensor {
            candle_core::Tensor::randn(0f32, 0.02, shape, &device).expect("randn 2-D")
        };
        let randn_1d = |size: usize| -> candle_core::Tensor {
            candle_core::Tensor::randn(0f32, 0.02, (size,), &device).expect("randn 1-D")
        };
        let ones_1d = |size: usize| -> candle_core::Tensor {
            candle_core::Tensor::ones((size,), candle_core::DType::F32, &device).expect("ones 1-D")
        };
        let zeros_1d = |size: usize| -> candle_core::Tensor {
            candle_core::Tensor::zeros((size,), candle_core::DType::F32, &device)
                .expect("zeros 1-D")
        };

        tensors.insert(
            "distilbert.embeddings.word_embeddings.weight".into(),
            randn((vocab, hidden)),
        );
        tensors.insert(
            "distilbert.embeddings.position_embeddings.weight".into(),
            randn((max_pos, hidden)),
        );
        tensors.insert(
            "distilbert.embeddings.LayerNorm.weight".into(),
            ones_1d(hidden),
        );
        tensors.insert(
            "distilbert.embeddings.LayerNorm.bias".into(),
            zeros_1d(hidden),
        );
        for n in 0..layers {
            let prefix = format!("distilbert.transformer.layer.{n}");
            for lin in ["q_lin", "k_lin", "v_lin", "out_lin"] {
                tensors.insert(
                    format!("{prefix}.attention.{lin}.weight"),
                    randn((hidden, hidden)),
                );
                tensors.insert(format!("{prefix}.attention.{lin}.bias"), randn_1d(hidden));
            }
            tensors.insert(format!("{prefix}.sa_layer_norm.weight"), ones_1d(hidden));
            tensors.insert(format!("{prefix}.sa_layer_norm.bias"), zeros_1d(hidden));
            tensors.insert(format!("{prefix}.ffn.lin1.weight"), randn((inter, hidden)));
            tensors.insert(format!("{prefix}.ffn.lin1.bias"), randn_1d(inter));
            tensors.insert(format!("{prefix}.ffn.lin2.weight"), randn((hidden, inter)));
            tensors.insert(format!("{prefix}.ffn.lin2.bias"), randn_1d(hidden));
            tensors.insert(
                format!("{prefix}.output_layer_norm.weight"),
                ones_1d(hidden),
            );
            tensors.insert(format!("{prefix}.output_layer_norm.bias"), zeros_1d(hidden));
        }
        // `heads` only feeds the config's own `n_heads` divisibility check
        // inside `DistilBert::builder().build(..)` — no weight tensor is
        // keyed by head count, so it is read here only to keep the tuple
        // destructure honest about every field `tiny_config` (the
        // `jammi-encoders` analogue) names.
        let _ = heads;
        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors"))
            .expect("save synthetic model.safetensors");
        dir
    }

    /// `model_type` `"distilbert"` must construct via
    /// [`jammi_encoders::DistilBert::builder`] (mirroring the `"bert"`/
    /// `"modernbert"` arms exactly) and the resulting encoder must genuinely
    /// run this tier's step machinery: a training-mode forward over synthetic
    /// ids, wired the same way `dropout_forward_counter_is_live_...` above
    /// proves for `"bert"`.
    #[test]
    fn build_encoder_adapters_supports_distilbert() {
        let model_dir = write_synthetic_distilbert_model_dir(1);
        let varmap = VarMap::new();
        let (mut encoder, adapter_cfg) = build_encoder_adapters(
            &checkpoint_of(model_dir.path()),
            Task::Text,
            &["q_lin".to_string(), "v_lin".to_string()],
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        )
        .expect("build_encoder_adapters must support model_type \"distilbert\"");

        assert!(
            matches!(encoder, AnyEncoder::DistilBert(_)),
            "model_type \"distilbert\" must build an AnyEncoder::DistilBert variant"
        );
        assert_eq!(adapter_cfg.model_type, "distilbert");
        assert_eq!(encoder.hidden_size(), 32);

        let input_ids = crate::finetune_step::synthetic_ids(2, 4, 100, 11, &Device::Cpu);
        let mask = candle_core::Tensor::ones((2, 4), candle_core::DType::U32, &Device::Cpu)
            .expect("mask tensor");
        encoder.set_training(true);
        let pooled = encoder
            .forward(&input_ids, &mask)
            .expect("training-mode forward over the synthetic distilbert encoder");
        assert_eq!(pooled.dims(), &[2, 32]);
        assert!(
            !encoder.trainable_params().is_empty(),
            "a LoRA-injected distilbert encoder must report trainable params"
        );
    }

    /// `--layers-to-transform`
    /// plumbing — `Some([0])` on a TWO-layer encoder must wrap ONLY layer
    /// 0's matching linears, never layer 1's, so the trainable LoRA tensor
    /// count under the restriction is EXACTLY HALF the unrestricted
    /// (`None`) count for the identical `target_modules` (both layers are
    /// structurally identical in this synthetic fixture, so "half" is exact,
    /// not approximate).
    #[test]
    fn build_encoder_adapters_layers_to_transform_restricts_to_one_layer() {
        let model_dir = write_synthetic_distilbert_model_dir(2);
        let target_modules = ["q_lin".to_string(), "v_lin".to_string()];

        let varmap_all = VarMap::new();
        let (encoder_all, _cfg_all) = build_encoder_adapters(
            &checkpoint_of(model_dir.path()),
            Task::Text,
            &target_modules,
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap_all,
        )
        .expect("no layers_to_transform restriction: build must succeed over both layers");
        let trainable_all = encoder_all.trainable_params().len();

        let varmap_one = VarMap::new();
        let (encoder_one, _cfg_one) = build_encoder_adapters(
            &checkpoint_of(model_dir.path()),
            Task::Text,
            &target_modules,
            &Some(vec![0]),
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap_one,
        )
        .expect("layers_to_transform: Some([0]) must still build (layer 0 matches)");
        let trainable_one = encoder_one.trainable_params().len();

        assert!(
            trainable_one > 0,
            "layer 0 alone must still yield trainable LoRA tensors"
        );
        assert_eq!(
            trainable_all,
            2 * trainable_one,
            "restricting to one of two structurally-identical layers must exactly halve the \
             trainable LoRA tensor count: all={trainable_all}, one={trainable_one}"
        );
    }

    /// The zero-trainable refusal has TWO
    /// producers — `target_modules` matching nothing, and a
    /// `layers_to_transform` restriction excluding every site
    /// `target_modules` would otherwise have matched (e.g. `Some([99])` on
    /// a 2-layer fixture, or `Some([1])` when the matching selector only
    /// exists on layer 0). This drives the SECOND producer — a real
    /// selector (`q_lin`/`v_lin`, present on both layers) with
    /// `layers_to_transform: Some([99])`, a layer index this 2-layer
    /// fixture does not have — and asserts the refusal fires AND the
    /// message NAMES `layers_to_transform` (including its value `[99]`),
    /// not just `target_modules` (which, taken alone, would misdiagnose
    /// this as a bad-selector problem).
    #[test]
    fn build_encoder_adapters_names_layers_to_transform_when_it_causes_the_zero_trainable_refusal()
    {
        let model_dir = write_synthetic_distilbert_model_dir(2);
        let varmap = VarMap::new();
        let result = build_encoder_adapters(
            &checkpoint_of(model_dir.path()),
            Task::Text,
            &["q_lin".to_string(), "v_lin".to_string()],
            &Some(vec![99]),
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        );
        let err = match result {
            Ok(_) => panic!(
                "layers_to_transform: Some([99]) on a 2-layer fixture must exclude every \
                 site and refuse, not silently build a LoRA-free encoder"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("layers_to_transform"),
            "error must name layers_to_transform as a possible cause, not just \
             target_modules: {msg}"
        );
        assert!(
            msg.contains("99"),
            "error must include the actual layers_to_transform value: {msg}"
        );
    }

    /// Write a minimal `model_dir` whose `config.json` is `config` and whose
    /// `model.safetensors` merely EXISTS — enough for
    /// [`Checkpoint::resolve`]'s chain, which decides the family before any
    /// weight tensor is read.
    fn write_config_only_model_dir(config: serde_json::Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).expect("serialize config"),
        )
        .expect("write config.json");
        std::fs::write(dir.path().join("model.safetensors"), b"placeholder")
            .expect("write model.safetensors");
        dir
    }

    /// Negative control (at the resolution chain, where the family is
    /// decided): a `model_type` this tier does not
    /// know must still error, and the error must honestly name every family
    /// this tier DOES support — not silently coerce to BERT, and not go
    /// stale as new arms are added.
    ///
    /// `"gpt2"` is the sharp case the plan names: its config parses cleanly
    /// as a `BertConfig`, so a `_ => Bert` fallthrough would build and train
    /// something that reported plausible numbers under the wrong
    /// architecture id.
    #[test]
    fn checkpoint_resolve_rejects_unknown_model_type_and_names_every_family() {
        let dir = write_config_only_model_dir(serde_json::json!({
            "model_type": "gpt2",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 100,
            "max_position_embeddings": 128,
        }));
        let err = match Checkpoint::resolve(dir.path()) {
            Ok(_) => panic!("an unsupported model_type must be refused, not silently accepted"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("gpt2"),
            "error must name the rejected type: {msg}"
        );
        for supported in [
            "bert",
            "modernbert",
            "distilbert",
            "open_clip",
            "clap_audio_model",
        ] {
            assert!(
                msg.contains(supported),
                "error must honestly list every supported family, missing {supported:?}: {msg}"
            );
        }
    }

    /// The other side of the same predicate: a config with NO `model_type`
    /// key at all resolves as BERT, per
    /// `jammi_ai::model::arch::UNDECLARED_MODEL_TYPE_FAMILY` — the one owner
    /// of the "undeclared -> bert" rule every reader in this workspace
    /// (serving's own load, the fine-tune worker, and this tier) applies.
    /// A config that DECLARES an unrecognised string is a different case and
    /// stays a typed refusal (`checkpoint_resolve_rejects_unknown_model_type_and_names_every_family`
    /// above).
    #[test]
    fn checkpoint_resolve_resolves_an_absent_model_type_as_bert() {
        let dir = write_config_only_model_dir(serde_json::json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
        }));
        let checkpoint = Checkpoint::resolve(dir.path()).unwrap_or_else(|e| {
            panic!("an absent model_type must resolve as bert, not refuse: {e}")
        });
        assert_eq!(checkpoint.family, EncoderFamily::Bert);
    }

    /// A BERT config-compatible alias resolves to BERT — the alias set the
    /// serving text arm accepts, shared with this tier. Pinned here because
    /// consuming the shared predicate admits `"roberta"`, and that deserves
    /// a test that says so out loud.
    #[test]
    fn checkpoint_resolve_accepts_the_bert_config_aliases() {
        for alias in ["bert", "roberta", "camembert", "xlm-roberta"] {
            let dir = write_config_only_model_dir(serde_json::json!({
                "model_type": alias,
                "hidden_size": 32,
                "num_hidden_layers": 1,
            }));
            let checkpoint = Checkpoint::resolve(dir.path())
                .unwrap_or_else(|e| panic!("{alias} must resolve as a BERT-family config: {e}"));
            assert_eq!(checkpoint.family, EncoderFamily::Bert, "{alias}");
            assert_eq!(checkpoint.family.adapter_model_type(), "bert", "{alias}");
        }
    }

    /// A GGUF-only directory resolves through the SHARED Candle weights
    /// chain (so the bench and serving agree about which file is "the
    /// weights") and is then refused BY NAME as untrainable — never left to
    /// die in a safetensors parser pointed at a `.gguf` file.
    #[test]
    fn checkpoint_resolve_refuses_a_gguf_checkpoint_by_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({"model_type": "bert", "hidden_size": 32}).to_string(),
        )
        .expect("write config.json");
        std::fs::write(dir.path().join("model.gguf"), b"placeholder").expect("write model.gguf");
        let err = match Checkpoint::resolve(dir.path()) {
            Ok(_) => panic!("a GGUF checkpoint cannot be fine-tuned by this tier"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("GGUF"), "{err}");
        assert!(
            err.contains("model.gguf"),
            "the refusal must name the file: {err}"
        );
    }

    /// A NON-EMPTY `target_modules` that matches zero linear layers on the
    /// built encoder must REFUSE loudly, mirroring
    /// `finetune_step.rs`'s `build_fixture` precedent ("no trainable LoRA
    /// tensors — target_modules matched nothing"). This is the CLI's real
    /// failure mode: `main.rs`'s `--target-modules` default is
    /// `"Wqkv,Wo,Wi"` (ModernBERT selectors), which matches nothing on
    /// `tiny_bert`'s `query`/`key`/`value`/`dense` naming — today that
    /// silently profiles a LoRA-free model instead of refusing.
    #[test]
    fn build_encoder_adapters_refuses_nonempty_target_modules_that_match_nothing() {
        let varmap = VarMap::new();
        let unmatched = ["Wqkv".to_string(), "Wo".to_string(), "Wi".to_string()];
        let result = build_encoder_adapters(
            &checkpoint_of(&tiny_bert_model_dir()),
            Task::Text,
            &unmatched,
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        );
        let err = match result {
            Ok(_) => panic!(
                "target_modules {unmatched:?} match nothing on a bert encoder's \
                 query/key/value/dense naming — this must refuse, not silently build a \
                 LoRA-free encoder"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();
        for selector in &unmatched {
            assert!(
                msg.contains(selector.as_str()),
                "error must name the unmatched selector {selector:?}: {msg}"
            );
        }
        assert!(
            msg.contains("bert"),
            "error must name the model_type: {msg}"
        );
        // `layers_to_transform` is `&None` here, so
        // the message's " restricted to layers {layers:?}" clause must be
        // ABSENT — this is the negative control for
        // `build_encoder_adapters_names_layers_to_transform_when_it_causes_the_zero_trainable_refusal`'s
        // positive case: an implementation that unconditionally emitted
        // "restricted to layers []"/"restricted to layers None" regardless
        // of the actual `layers_to_transform` value would make THAT test's
        // `msg.contains("layers_to_transform")` pass vacuously off the
        // message's own unconditional tail clause (which also contains the
        // literal substring "layers_to_transform" via "check
        // layers_to_transform for an off-by-one layer index if it is set")
        // — this assertion is what actually binds the conditional
        // interpolation to the real `None` case.
        assert!(
            !msg.contains("restricted to layers"),
            "with layers_to_transform: None, the message must NOT claim a layer restriction \
             caused the refusal: {msg}"
        );
    }

    /// An explicitly EMPTY
    /// `target_modules` must ALSO refuse in this TRAINING tier, mirroring
    /// `finetune_step.rs`'s `build_fixture` precedent
    /// (`trainable.is_empty()`) EXACTLY — that check is unconditional on
    /// whether `target_modules` itself was empty or merely matched nothing.
    /// Rationale: with zero trainable `Var`s, candle-core 0.11's
    /// `backprop.rs` only tracks gradients through `is_variable()` nodes, so
    /// `loss.backward()` emits nothing to differentiate, and
    /// `candle_nn::optimizer`'s `AdamW::step` silently no-ops over an empty
    /// var list rather than warning — a caller who passes an empty
    /// `target_modules` to THIS tier would get a run that silently "trains"
    /// nothing, indistinguishable in its reported numbers from a genuine
    /// LoRA run. `jammi_lora::LoraBuildConfig::frozen`'s "no LoRA"
    /// convenience remains valid for non-training (inference-load)
    /// consumers — this refusal is scoped to this tier's `build_encoder_adapters`
    /// only; `jammi-lora` itself is untouched.
    #[test]
    fn build_encoder_adapters_refuses_empty_target_modules_too() {
        let varmap = VarMap::new();
        let result = build_encoder_adapters(
            &checkpoint_of(&tiny_bert_model_dir()),
            Task::Text,
            &[],
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        );
        let err = match result {
            Ok(_) => panic!(
                "an empty target_modules must ALSO refuse in this training tier — zero \
                 trainable Vars means the run would silently train nothing"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("bert"),
            "error must name the model_type: {msg}"
        );
    }

    /// The untrained adapter a run wrote into `work_dir`, in
    /// [`adapter_flat_f32`]'s shape.
    fn initial_adapter_flat_f32(work_dir: &Path) -> std::collections::BTreeMap<String, Vec<f32>> {
        adapter_flat_f32(&work_dir.join(INITIAL_ADAPTER_FILE))
    }

    /// The last epoch's checkpoint as the resident run staged it in the
    /// work dir's file-backed artifact store
    /// (`{job}/_checkpoints/{attempt}/epoch_{N}/adapter.safetensors`).
    fn final_epoch_adapter_flat_f32(
        work_dir: &Path,
        epochs: usize,
    ) -> std::collections::BTreeMap<String, Vec<f32>> {
        fn find(dir: &Path, leaf: &str) -> Option<PathBuf> {
            for entry in std::fs::read_dir(dir).ok()?.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if path.file_name().is_some_and(|n| n == leaf) {
                        return Some(path.join(jammi_lora::ADAPTER_WEIGHTS_FILE));
                    }
                    if let Some(found) = find(&path, leaf) {
                        return Some(found);
                    }
                }
            }
            None
        }
        let leaf = format!("epoch_{}", epochs - 1);
        let path = find(&work_dir.join("artifacts"), &leaf)
            .unwrap_or_else(|| panic!("no {leaf} checkpoint under the work dir's artifact store"));
        adapter_flat_f32(&path)
    }

    fn adapter_flat_f32(path: &Path) -> std::collections::BTreeMap<String, Vec<f32>> {
        candle_core::safetensors::load(path, &Device::Cpu)
            .unwrap_or_else(|e| panic!("load adapter {}: {e}", path.display()))
            .into_iter()
            .map(|(name, tensor)| {
                let flat = tensor
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap_or_else(|e| panic!("flatten initial var {name}: {e}"));
                (name, flat)
            })
            .collect()
    }

    /// The negative control: the same job, run with
    /// [`AppliedLearningRate::Zero`]. The whole loop runs — it takes every
    /// optimizer step an ordinary run takes — and nothing is learned: the
    /// final weights ARE the untrained adapter, bit for bit, and the train
    /// probe reads the same value before training and after every epoch, so
    /// the learning-happened delta a reader derives from it is exactly `0.0`.
    /// The leg reports the rate it was measured at, `0.0`.
    ///
    /// The ordinary run beside it is the control's own control: same params
    /// but for the applied rate, it leaves the untrained adapter and moves
    /// the probe — so the zero is the control's doing, not a fixture that
    /// could not learn.
    #[tokio::test]
    async fn the_zero_learning_rate_control_runs_every_step_and_learns_nothing() {
        async fn run_at(
            rate: AppliedLearningRate,
        ) -> (
            Leg<TrainRunPayload>,
            std::collections::BTreeMap<String, Vec<f32>>,
            std::collections::BTreeMap<String, Vec<f32>>,
        ) {
            let work_dir = tempfile::tempdir().expect("tempdir");
            let params = FinetuneRunParams {
                applied_learning_rate: rate,
                // MNRL, not the fixture's triplet default: these synthetic
                // rows sit on the triplet hinge's margin, where the probe
                // reads the margin whatever the adapter does, and a probe
                // that cannot move would make the control's flat series
                // prove nothing.
                objective: Objective::Mnrl,
                ..non_perturbation_test_params(work_dir.path().to_path_buf())
            };
            let epochs = params.epochs;
            let tier = BlockingCall::spawn_blocking(move |call| run(&call, &params))
                .await
                .expect("join run task")
                .expect("finetune-run");
            (
                tier,
                initial_adapter_flat_f32(work_dir.path()),
                final_epoch_adapter_flat_f32(work_dir.path(), epochs),
            )
        }

        let (control, control_initial, control_final) = run_at(AppliedLearningRate::Zero).await;
        let (trained, trained_initial, trained_final) =
            run_at(AppliedLearningRate::Scheduled).await;
        let probe_of = |leg: &Leg<TrainRunPayload>| {
            leg.facts
                .train_probe_series
                .clone()
                .expect("a train-run leg carries its probe series")
        };
        let init_of = |leg: &Leg<TrainRunPayload>| {
            leg.measured
                .held_out_at_init
                .expect("a train-run leg scores the untrained model")
        };
        let (control_probe, trained_probe) = (probe_of(&control), probe_of(&trained));
        let (control_init, trained_init) = (init_of(&control), init_of(&trained));

        assert_eq!(
            control.payload.lr, 0.0,
            "a control leg reports the rate it ran at"
        );
        assert_eq!(trained.payload.lr, 0.01);
        assert_eq!(
            control.payload.steps_measured, trained.payload.steps_measured,
            "the control takes every optimizer step the ordinary run takes"
        );
        assert!(control.payload.steps_measured > 0);

        assert_eq!(
            control_final, control_initial,
            "a zero applied rate must leave every trainable tensor at its untrained value"
        );
        let first = control_probe[0];
        assert!(
            control_probe.iter().all(|p| *p == first),
            "the control's train probe must not move: {:?}",
            control_probe
        );
        assert_eq!(
            first - control_probe[control_probe.len() - 1],
            0.0,
            "the learning-happened delta of a control leg is exactly zero"
        );
        // The untrained model's held-out loss is where a control run stays:
        // every point of its trajectory reads it, so its improvement from
        // init is exactly zero too.
        assert!(
            control
                .measured
                .trajectory
                .iter()
                .all(|p| p.held_out_mean == control_init),
            "{:?} vs init {}",
            control.measured.trajectory,
            control_init
        );
        // The ordinary run's learning is read off its weights and its train
        // probe above, not off its held-out loss: this fixture's two held-out
        // rows sit at MNRL's symmetric floor (`ln 2`) whatever the adapter
        // does, so the held-out origin is recorded here and its movement is
        // proven on the committed corpus by the cross-producer parity guard.
        assert_eq!(control_init, trained_init);

        assert_eq!(
            trained_initial, control_initial,
            "both runs start from the same untrained adapter"
        );
        assert_ne!(
            trained_final, trained_initial,
            "the ordinary run must have moved its adapter"
        );
        assert_ne!(
            trained_probe[0],
            trained_probe[trained_probe.len() - 1],
            "the ordinary run's train probe must have moved: {:?}",
            trained_probe
        );
    }

    /// A non-positive `--lr` is refused by name, pointing at the control —
    /// it is not a way to ask for one.
    #[tokio::test]
    async fn a_non_positive_learning_rate_is_refused_naming_the_control() {
        for learning_rate in [0.0, -1e-3, f64::NAN] {
            let work_dir = tempfile::tempdir().expect("tempdir");
            let params = FinetuneRunParams {
                learning_rate,
                ..non_perturbation_test_params(work_dir.path().to_path_buf())
            };
            let err = BlockingCall::spawn_blocking(move |call| run(&call, &params))
                .await
                .expect("join run task")
                .err()
                .unwrap_or_else(|| panic!("--lr {learning_rate} must be refused"))
                .to_string();
            assert!(err.contains("--zero-lr-control"), "{err}");
        }
    }

    /// `train_run_wall_s` must be a
    /// REAL, measured, nonzero wall-clock time (never a stub/hardcoded
    /// value), and — because it times ONLY `training_loop.run()` calls,
    /// excluding `build_encoder_adapters`, the resume-checkpoint fetch, and
    /// every `evaluate_held_out` call (see that field's own doc) — it must
    /// be STRICTLY LESS than the whole `run` invocation's own outer
    /// wall-clock, since this CPU-hermetic fixture's held-out/probe
    /// evaluations and encoder builds each take real, nonzero time too.
    #[tokio::test]
    async fn train_run_wall_s_is_measured_and_strictly_less_than_the_outer_wall_clock() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let params = non_perturbation_test_params(work_dir.path().to_path_buf());
        let outer_t0 = Instant::now();
        let tier = BlockingCall::spawn_blocking(move |call| run(&call, &params))
            .await
            .expect("join run task")
            .expect("finetune-run");
        let outer_wall_s = outer_t0.elapsed().as_secs_f64();

        assert!(
            tier.payload.train_run_wall_s > 0.0,
            "train_run_wall_s must be a real, measured, nonzero wall-clock time, got {}",
            tier.payload.train_run_wall_s
        );
        assert!(
            tier.payload.train_run_wall_s < outer_wall_s,
            "train_run_wall_s ({}) must be STRICTLY LESS than run's own outer wall-clock \
             ({}) -- it excludes build_encoder_adapters, the resume-checkpoint fetch, and every \
             evaluate_held_out call, all of which this fixture's real tokenizer/forward passes \
             make take nonzero time too; train_run_wall_s >= outer_wall_s would mean this field \
             is silently timing more than just training_loop.run()",
            tier.payload.train_run_wall_s,
            outer_wall_s
        );
    }

    /// Proven as emitted: a committed golden is one past run's bytes, so
    /// this test is what binds the declared Rust consts
    /// (`FinetuneRunTier::IDENTITY_FIELDS`'s `layers_to_transform` entry, and
    /// `train_run_wall_s` itself) to the ACTUAL bytes THIS build emits.
    /// Runs the real CPU-fixture path (the same `run` the smoke tests
    /// drive), wraps the resulting [`crate::report::TrainRunPayload`] in a
    /// real [`crate::report::Report`], serializes THAT (not the bare tier),
    /// and asserts at the `serde_json::Value` PATH level — never by reading
    /// the Rust struct fields back — that `tiers.finetune_run` carries both
    /// keys.
    ///
    /// Mutations: (1) the `layers_to_transform` half is doubly
    /// covered — adding `#[serde(skip_serializing_if =
    /// "Option::is_none")]` to that field makes THIS test fail before even
    /// reaching its own assertion, inside `run`'s own
    /// `assert_identity_fields_present` self-check (`IDENTITY_FIELDS names
    /// "layers_to_transform", absent on this report`) — so a second,
    /// independent mechanism already guards it. (2) Hardcoding
    /// `train_run_wall_s: 0.0` at [`run`]'s tier construction site makes
    /// this test's own `wall_s > 0.0` assertion fail
    /// (`tiers.finetune_run.train_run_wall_s must carry a real, measured,
    /// nonzero value in the emitted JSON, got 0`); the SAME mutation ALSO fails
    /// [`train_run_wall_s_is_measured_and_strictly_less_than_the_outer_wall_clock`]
    /// above (its `tier.payload.train_run_wall_s > 0.0` assertion), so this test is
    /// NOT that field's sole guard. The two tests cover DIFFERENT things:
    /// that one guards the plain in-struct Rust value
    /// (`tier.payload.train_run_wall_s`, never serialized); this one's unique
    /// contribution is proving the value actually survives
    /// `serde_json::to_value` NESTED under the real `tiers.finetune_run`
    /// JSON path
    /// — a producer that computed the field correctly but wired it to the
    /// wrong JSON key, or dropped it via a stray `skip_serializing_if`,
    /// would still pass the other test while failing this one.
    #[tokio::test]
    async fn finetune_run_tier_json_actually_emits_layers_to_transform_and_train_run_wall_s() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let params = non_perturbation_test_params(work_dir.path().to_path_buf());
        let tier = BlockingCall::spawn_blocking(move |call| run(&call, &params))
            .await
            .expect("join run task")
            .expect("finetune-run");

        let report = crate::report::Report::new(
            "finetune-run",
            crate::report::Tiers {
                finetune_run: Some(tier),
                ..Default::default()
            },
        );
        let value = serde_json::to_value(&report).expect("serialize Report for JSON-shape check");

        let finetune_run_json = value
            .get("tiers")
            .and_then(|t| t.get("finetune_run"))
            .expect("tiers.finetune_run must be present in the emitted JSON");

        assert!(
            finetune_run_json.get("layers_to_transform").is_some(),
            "tiers.finetune_run.layers_to_transform must be a present key in the emitted \
             JSON (Some(null) counts as present -- the KEY must exist, not merely the Rust \
             field): {finetune_run_json}"
        );
        let wall_s = finetune_run_json
            .get("train_run_wall_s")
            .and_then(|v| v.as_f64())
            .expect(
                "tiers.finetune_run.train_run_wall_s must be a present JSON number key in the \
                 emitted JSON",
            );
        assert!(
            wall_s > 0.0,
            "tiers.finetune_run.train_run_wall_s must carry a real, measured, nonzero value \
             in the emitted JSON, got {wall_s}"
        );
    }

    #[test]
    fn objective_from_str_round_trips_both_variants() {
        assert_eq!("triplet".parse::<Objective>().unwrap(), Objective::Triplet);
        assert_eq!("mnrl".parse::<Objective>().unwrap(), Objective::Mnrl);
        assert_eq!(Objective::Triplet.as_str(), "triplet");
        assert_eq!(Objective::Mnrl.as_str(), "mnrl");
    }

    #[test]
    fn objective_from_str_rejects_an_unknown_value() {
        let err = "cosent".parse::<Objective>().unwrap_err();
        assert!(err.contains("cosent"), "{err}");
        assert!(err.contains("triplet") && err.contains("mnrl"), "{err}");
    }

    /// Projection correctness: the MNRL
    /// loader's (anchor, positive) rows must be EXACTLY the same rows, in
    /// the SAME committed order, as the source triplets — negative column
    /// dropped, nothing reordered, nothing dropped or duplicated — against
    /// a small fixture with known ids.
    #[test]
    fn project_to_pairs_keeps_committed_order_and_drops_the_negative_column() {
        let pairs = vec![
            IdTriplet {
                id: "row-0".to_string(),
                anchor: "anchor 0".to_string(),
                positive: "positive 0".to_string(),
                negative: "negative 0".to_string(),
            },
            IdTriplet {
                id: "row-1".to_string(),
                anchor: "anchor 1".to_string(),
                positive: "positive 1".to_string(),
                negative: "negative 1".to_string(),
            },
            IdTriplet {
                id: "row-2".to_string(),
                anchor: "anchor 2".to_string(),
                positive: "positive 2".to_string(),
                negative: "negative 2".to_string(),
            },
        ];
        let projected = project_to_pairs(&pairs);
        assert_eq!(
            projected,
            vec![
                ("anchor 0".to_string(), "positive 0".to_string()),
                ("anchor 1".to_string(), "positive 1".to_string()),
                ("anchor 2".to_string(), "positive 2".to_string()),
            ],
            "projection must be the same rows, same order, negative column dropped"
        );
    }

    /// Negative control on the projection: an empty input projects to an
    /// empty output (no fabricated rows), and the projection never reads
    /// `negative` at all — swapping every `negative` field to a distinct,
    /// unique sentinel value must not change the projected output, proving
    /// the negative column truly plays no role.
    #[test]
    fn project_to_pairs_output_is_independent_of_the_negative_column() {
        let mut pairs = vec![IdTriplet {
            id: "row-0".to_string(),
            anchor: "anchor 0".to_string(),
            positive: "positive 0".to_string(),
            negative: "negative 0".to_string(),
        }];
        let before = project_to_pairs(&pairs);
        pairs[0].negative = "a completely different sentinel negative".to_string();
        let after = project_to_pairs(&pairs);
        assert_eq!(before, after);
        assert!(project_to_pairs(&[]).is_empty());
    }

    /// `--mutant-id`/`--mutant-base-sha`/
    /// `--mutant-patch-sha256` are all-or-none. This check fires FIRST, before
    /// any device/catalog/filesystem setup (see `run`'s own leading
    /// validation block), so a plain `#[test]` (no tokio runtime) suffices —
    /// the function returns before ever reaching a `Handle::current()` call.
    /// Exercises all three "exactly one of three" partial subsets, plus both
    /// "exactly two of three" subsets — every INCOMPLETE non-empty subset a
    /// caller could supply — proving the refusal is not merely reachable for
    /// one particular partial combination.
    #[test]
    fn mutant_provenance_flags_are_refused_when_partially_supplied() {
        let some = |s: &str| Some(s.to_string());
        let combinations: [(Option<String>, Option<String>, Option<String>); 6] = [
            (some("eps-0.10"), None, None),
            (None, some("f".repeat(40).as_str()), None),
            (None, None, some("a".repeat(64).as_str())),
            (some("eps-0.10"), some("f".repeat(40).as_str()), None),
            (some("eps-0.10"), None, some("a".repeat(64).as_str())),
            (
                None,
                some("f".repeat(40).as_str()),
                some("a".repeat(64).as_str()),
            ),
        ];
        for (mutant_id, mutant_base_sha, mutant_patch_sha256) in combinations {
            let mut params = non_perturbation_test_params(std::env::temp_dir());
            params.mutant_id = mutant_id.clone();
            params.mutant_base_sha = mutant_base_sha.clone();
            params.mutant_patch_sha256 = mutant_patch_sha256.clone();
            let result = BlockingCall::spawn_thread(move |call| run(&call, &params))
                .join()
                .expect("join run thread");
            let err = match result {
                Ok(_) => panic!(
                    "a partial mutant subset must be refused: mutant_id={mutant_id:?}, \
                     mutant_base_sha={mutant_base_sha:?}, \
                     mutant_patch_sha256={mutant_patch_sha256:?}"
                ),
                Err(e) => e,
            };
            let msg = err.to_string();
            assert!(
                msg.contains("all-or-none"),
                "refusal message must name the all-or-none rule: {msg}"
            );
        }
    }

    /// `run`'s `Ok` payload carries a `VarMap`, which has no `Debug`
    /// impl, so the stdlib `Result::expect_err` (which formats the `Ok`
    /// value on failure) cannot be used against it directly — this is the
    /// same shape as the `match result { Ok(_) => panic!(...), Err(e) => e
    /// }` pattern `mutant_provenance_flags_are_refused_when_partially_supplied`
    /// already uses above, pulled out so the tests below don't each
    /// repeat it.
    fn expect_refused(
        result: Result<Leg<TrainRunPayload>, Box<dyn std::error::Error + Send + Sync>>,
        context: &str,
    ) -> Box<dyn std::error::Error + Send + Sync> {
        match result {
            Ok(_) => panic!("{context}"),
            Err(e) => e,
        }
    }

    /// The all-or-none gate counts NON-EMPTINESS, not PRESENCE
    /// (`is_some()`): `--mutant-base-sha ""` (which the CLI happily parses
    /// into `Some(String::new())`) must not count as "present" and produce
    /// a half-labeled leg. All three explicitly supplied as the empty
    /// string must be refused — NOT silently treated
    /// as the ordinary non-mutant (`None`-trio) case, which is a distinct
    /// state (see [`mutant_provenance_all_absent_is_not_the_same_state_as_an_empty_trio`]).
    #[test]
    fn mutant_provenance_empty_string_trio_is_refused() {
        let mut params = non_perturbation_test_params(std::env::temp_dir());
        params.mutant_id = Some(String::new());
        params.mutant_base_sha = Some(String::new());
        params.mutant_patch_sha256 = Some(String::new());
        let result = BlockingCall::spawn_thread(move |call| run(&call, &params))
            .join()
            .expect("join run thread");
        let err = expect_refused(
            result,
            "an explicitly-empty trio must be refused, not treated as absent",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("all-or-none"),
            "refusal message must name the all-or-none rule: {msg}"
        );
        assert!(
            msg.contains("empty-or-whitespace-only"),
            "refusal message must name the empty-or-whitespace-only state: {msg}"
        );
    }

    /// A whitespace-only trio is exactly as
    /// un-attributable as an empty-string one — `trim()` must run BEFORE the
    /// emptiness check, not after (or not at all), so a padded value can't
    /// slip past as "present".
    #[test]
    fn mutant_provenance_whitespace_trio_is_refused() {
        let mut params = non_perturbation_test_params(std::env::temp_dir());
        params.mutant_id = Some("   ".to_string());
        params.mutant_base_sha = Some("\t\n".to_string());
        params.mutant_patch_sha256 = Some(" ".to_string());
        let result = BlockingCall::spawn_thread(move |call| run(&call, &params))
            .join()
            .expect("join run thread");
        let err = expect_refused(
            result,
            "an explicitly-whitespace trio must be refused, not treated as absent",
        );
        assert!(
            err.to_string().contains("all-or-none"),
            "refusal message must name the all-or-none rule: {}",
            err
        );
    }

    /// Exactly one of the three empty/whitespace
    /// (with the other two genuinely valid) must be refused just like the
    /// classic `None`-mixed-with-`Some` partial subsets above — the
    /// "half-labeled leg" shape a bare `is_some()` count would let through
    /// (all three `is_some()`, so it sees "3 present"). Exercised in all
    /// three flag positions.
    #[test]
    fn mutant_provenance_one_empty_among_three_is_refused() {
        let valid_id = || Some("eps-0.10".to_string());
        let valid_base = || Some("f".repeat(40));
        let valid_patch = || Some("a".repeat(64));
        let combinations: [(Option<String>, Option<String>, Option<String>); 3] = [
            (Some("  ".to_string()), valid_base(), valid_patch()),
            (valid_id(), Some(String::new()), valid_patch()),
            (valid_id(), valid_base(), Some("\t".to_string())),
        ];
        for (mutant_id, mutant_base_sha, mutant_patch_sha256) in combinations {
            let mut params = non_perturbation_test_params(std::env::temp_dir());
            params.mutant_id = mutant_id.clone();
            params.mutant_base_sha = mutant_base_sha.clone();
            params.mutant_patch_sha256 = mutant_patch_sha256.clone();
            let result = BlockingCall::spawn_thread(move |call| run(&call, &params))
                .join()
                .expect("join run thread");
            let err = expect_refused(
                result,
                &format!(
                    "one empty/whitespace among three (otherwise valid) values must be \
                     refused: mutant_id={mutant_id:?}, mutant_base_sha={mutant_base_sha:?}, \
                     mutant_patch_sha256={mutant_patch_sha256:?}"
                ),
            );
            assert!(
                err.to_string().contains("all-or-none"),
                "refusal message must name the all-or-none rule: {err}"
            );
        }
    }

    /// Malformed (non-hex or wrong-length) shas in
    /// an otherwise-complete, non-empty trio get their own typed, flag-named
    /// refusal — a "3 present" trio is necessary but not sufficient; the
    /// content must actually be a plausible sha. Fires before any
    /// device/catalog/filesystem setup, same as the emptiness gate, so a
    /// plain `#[test]` suffices.
    #[test]
    fn mutant_provenance_malformed_shas_are_refused() {
        let mut too_short_base = non_perturbation_test_params(std::env::temp_dir());
        too_short_base.mutant_id = Some("eps-0.10".to_string());
        too_short_base.mutant_base_sha = Some("abc123".to_string()); // 6 hex chars, below the 7 floor
        too_short_base.mutant_patch_sha256 = Some("a".repeat(64));
        let result = BlockingCall::spawn_thread(move |call| run(&call, &too_short_base))
            .join()
            .expect("join run thread");
        let err = expect_refused(result, "a too-short mutant-base-sha must be refused");
        assert!(
            err.to_string().contains("--mutant-base-sha"),
            "refusal must name the offending flag: {err}"
        );

        let mut non_hex_base = non_perturbation_test_params(std::env::temp_dir());
        non_hex_base.mutant_id = Some("eps-0.10".to_string());
        non_hex_base.mutant_base_sha = Some("not-a-hex-sha!!".to_string());
        non_hex_base.mutant_patch_sha256 = Some("a".repeat(64));
        let result = BlockingCall::spawn_thread(move |call| run(&call, &non_hex_base))
            .join()
            .expect("join run thread");
        let err = expect_refused(result, "a non-hex mutant-base-sha must be refused");
        assert!(
            err.to_string().contains("--mutant-base-sha"),
            "refusal must name the offending flag: {err}"
        );

        let mut wrong_len_patch = non_perturbation_test_params(std::env::temp_dir());
        wrong_len_patch.mutant_id = Some("eps-0.10".to_string());
        wrong_len_patch.mutant_base_sha = Some("f".repeat(40));
        wrong_len_patch.mutant_patch_sha256 = Some("a".repeat(63)); // one short of 64
        let result = BlockingCall::spawn_thread(move |call| run(&call, &wrong_len_patch))
            .join()
            .expect("join run thread");
        let err = expect_refused(result, "a wrong-length mutant-patch-sha256 must be refused");
        assert!(
            err.to_string().contains("--mutant-patch-sha256"),
            "refusal must name the offending flag: {err}"
        );

        let mut non_hex_patch = non_perturbation_test_params(std::env::temp_dir());
        non_hex_patch.mutant_id = Some("eps-0.10".to_string());
        non_hex_patch.mutant_base_sha = Some("f".repeat(40));
        non_hex_patch.mutant_patch_sha256 = Some("z".repeat(64)); // right length, not hex
        let result = BlockingCall::spawn_thread(move |call| run(&call, &non_hex_patch))
            .join()
            .expect("join run thread");
        let err = expect_refused(result, "a non-hex mutant-patch-sha256 must be refused");
        assert!(
            err.to_string().contains("--mutant-patch-sha256"),
            "refusal must name the offending flag: {err}"
        );
    }

    /// A fully-supplied, non-empty trio clears the
    /// gate and the STAMPED values (in the returned tier) are the TRIMMED
    /// strings, never the raw, whitespace-padded CLI input — driven through
    /// the REAL `run` end to end (not merely the leading validation
    /// block) over the tiny, CPU-hermetic `tiny_bert` fixture, so this also
    /// re-covers the "all three genuinely present" arm of the gate itself.
    #[tokio::test]
    async fn mutant_provenance_valid_trio_is_stamped_trimmed() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let mut params = non_perturbation_test_params(work_dir.path().to_path_buf());
        params.mutant_id = Some("  eps-0.10  ".to_string());
        params.mutant_base_sha = Some(format!("  {}  ", "f".repeat(40)));
        params.mutant_patch_sha256 = Some(format!("\t{}\n", "a".repeat(64)));

        let tier = BlockingCall::spawn_blocking(move |call| run(&call, &params))
            .await
            .expect("join run task")
            .expect("a fully-supplied, non-empty (once trimmed) trio must be accepted");

        let mutant = &tier.provenance.as_ref().expect("a jammi leg").mutant;
        assert_eq!(mutant.mutant_id, Some("eps-0.10".to_string()));
        assert_eq!(mutant.mutant_base_sha, Some("f".repeat(40)));
        assert_eq!(mutant.mutant_patch_sha256, Some("a".repeat(64)));
    }

    /// Pins only the TYPE-level distinction; it deliberately does not
    /// re-implement the gate's predicate inline (a test asserting a copy of
    /// the predicate against itself would be tautological). The real
    /// positive control for "all-absent clears the gate" is exercised on
    /// every test run:
    ///
    /// - In-process: `init_probe_does_not_perturb_the_training_trajectory_bitwise`
    ///   (this module) drives the REAL `run` end to end, twice, via
    ///   `non_perturbation_test_params` — whose `mutant_id`/`mutant_base_sha`/
    ///   `mutant_patch_sha256` are all `None` — and `.expect()`s `Ok(_)` both
    ///   times. If the all-or-none gate ever wrongly fired on the all-absent
    ///   case, that test would fail immediately with a panic, not silently
    ///   pass.
    /// - End to end via the compiled CLI: every case in
    ///   `finetune_run_smoke.rs` (`finetune_run_smoke_end_to_end_cpu_hermetic`,
    ///   `finetune_run_smoke_mnrl_end_to_end_cpu_hermetic`) invokes the real
    ///   `finetune-run` binary WITHOUT any `--mutant-*` flag and asserts a
    ///   zero exit — the same "never touched" state this module's gate must
    ///   pass.
    #[test]
    fn mutant_provenance_all_absent_is_not_the_same_state_as_an_empty_trio() {
        // Deliberately not a re-implementation of the gate: this pins the
        // TYPE-level distinction the gate itself relies on (never-supplied
        // vs. explicitly-supplied-but-empty) without repeating the gate's
        // own logic. The genuine behavioral proof lives in the tests named
        // in this test's doc comment above.
        let never_touched: Option<String> = None;
        let explicitly_empty: Option<String> = Some(String::new());
        assert_ne!(
            never_touched, explicitly_empty,
            "a caller who never touches a `--mutant-*` flag and a caller who supplies it as an \
             empty string are observably different `Option<String>` values, even though both \
             trim to no content"
        );
    }

    /// The all-zero-attention validity gate refuses an all-zero-counters run
    /// at `model_type` `"bert"` and `"distilbert"`, not just `"modernbert"`
    /// — admission by counters, never by architecture name. A gate reading
    /// `model_type == "modernbert"` literally would pass a `bert`/`distilbert`
    /// leg with real optimizer steps and all-zero attention counters —
    /// silently passing a run that never entered training mode.
    #[test]
    fn fused_dispatch_proof_gate_refuses_all_zero_attention_for_every_supported_model_type() {
        for family in [
            EncoderFamily::Bert,
            EncoderFamily::DistilBert,
            EncoderFamily::ModernBert,
        ] {
            let model_type = family.adapter_model_type();
            // LoRA counters deliberately NON-zero: this control must fire
            // on the ATTENTION counters alone for a BERT-family leg, never
            // borrow a pass from the tower-family branch.
            let result = fused_dispatch_proof_gate(family, 6, 0, 0, 0, 0, 9, 9);
            let msg = match result {
                Ok(()) => panic!(
                    "model_type {model_type:?} with 6 optimizer steps and all-zero attention \
                     counters must be refused, not silently accepted"
                ),
                Err(msg) => msg,
            };
            assert!(
                msg.contains("attention_block_fused_dispatches"),
                "error must name attention_block_fused_dispatches: {msg}"
            );
            assert!(
                msg.contains("attention_block_eager_dispatches"),
                "error must name attention_block_eager_dispatches: {msg}"
            );
            assert!(
                msg.contains("set_training(true)"),
                "error must name the set_training(true) premise that was never met: {msg}"
            );
            assert!(
                msg.contains(model_type),
                "error must name the model_type it refused: {msg}"
            );
        }
    }

    /// Two-sided control: the SAME gate must accept a `bert`/`distilbert`/
    /// `modernbert` leg whose counters show real dispatch (either arm), and
    /// must never fire when this run took zero optimizer steps at all (an
    /// eval-only or empty-epoch invocation — not this gate's concern).
    #[test]
    fn fused_dispatch_proof_gate_accepts_live_dispatch_and_zero_step_runs() {
        for family in [
            EncoderFamily::Bert,
            EncoderFamily::DistilBert,
            EncoderFamily::ModernBert,
        ] {
            // Fused arm fired. LoRA counters zero throughout: a BERT-family
            // leg is judged on its attention counters ONLY.
            assert!(fused_dispatch_proof_gate(family, 6, 3, 0, 0, 0, 0, 0).is_ok());
            // Eager arm fired (a legitimate by-design domain decline, e.g.
            // `head_dim != 64`).
            assert!(fused_dispatch_proof_gate(family, 6, 0, 3, 0, 0, 0, 0).is_ok());
            // Flash cascade fired.
            assert!(fused_dispatch_proof_gate(family, 6, 0, 0, 3, 0, 0, 0).is_ok());
            // Flash cascade declined (also a legitimate by-design outcome
            // for THIS gate's purposes — it only refuses when ALL FOUR are
            // zero at once).
            assert!(fused_dispatch_proof_gate(family, 6, 0, 0, 0, 3, 0, 0).is_ok());
        }
        // Zero optimizer steps: this gate never fires regardless of the
        // counters (an eval-only run, or an empty epoch count, is simply
        // outside this gate's scope) — a real run's `cumulative_steps` is
        // production-computed (the final leg's `TrainingResult::total_steps`),
        // not something this gate re-derives.
        assert!(fused_dispatch_proof_gate(EncoderFamily::Bert, 0, 0, 0, 0, 0, 0, 0).is_ok());
    }

    /// The `"bert"` counted-eager head16 shape, dedicated:
    /// `(attention_block_fused_dispatches,
    /// attention_block_eager_dispatches) == (0, 2)` are the EXACT counts
    /// `tests/finetune_run_smoke.rs`'s `tiny_bert` (`hidden_size: 32,
    /// num_attention_heads: 2` — `head_dim == 16 != 64`) end-to-end run
    /// measures for a 1-epoch, 2-batch leg: every training-mode attention
    /// forward reaches `admit("attention_block_fused", ..)` and is DECLINED
    /// by the `head_dim == 64` domain predicate, so
    /// it counts as eager, never fused — the flash cascade is consulted
    /// first and also declines (`attention_block_flash_declined_dispatches
    /// == 2`), never fires. So `bert` at `head_dim != 64` reads `(0, >0)`,
    /// never all-zero, and the gate must accept that shape, not refuse it.
    #[test]
    fn fused_dispatch_proof_gate_passes_bert_counted_eager_head16_shape() {
        assert!(fused_dispatch_proof_gate(EncoderFamily::Bert, 2, 0, 2, 0, 2, 0, 0).is_ok());
    }

    /// The complementary `"bert"` fused-arm shape, dedicated:
    /// `(attention_block_fused_dispatches,
    /// attention_block_eager_dispatches) == (>0, 0)` is what a `bert` leg
    /// at `head_dim == 64` (the tensor-state predicate
    /// `jammi_encoders::attention_cascade`'s own doc names, not this
    /// fixture) reads: the fused whole-attention-block
    /// kernel dispatches on every training-mode forward, the eager
    /// composition never runs. Boundary-value coverage alongside
    /// [`fused_dispatch_proof_gate_passes_bert_counted_eager_head16_shape`]'s
    /// `(0, >0)` case and
    /// [`fused_dispatch_proof_gate_refuses_all_zero_attention_for_every_supported_model_type`]'s
    /// `(0, 0)` refusal — all three shapes a `bert` leg's own counters can
    /// legitimately read, checked for `model_type == "bert"` specifically
    /// (not merely folded into the multi-architecture loop above).
    #[test]
    fn fused_dispatch_proof_gate_passes_bert_fused_head64_shape() {
        assert!(fused_dispatch_proof_gate(EncoderFamily::Bert, 2, 2, 0, 0, 0, 0, 0).is_ok());
    }

    // ── The three towers, one resolution chain ──────────────────────────

    /// The ONE-chain claim, made concrete: `tiny_open_clip` holds NEITHER
    /// `config.json` NOR `model.safetensors` — its files are
    /// `open_clip_config.json` / `open_clip_model.safetensors`, so joining
    /// the two hard-coded names onto `model_dir` by hand would leave this
    /// committed checkpoint invisible to the tier.
    #[test]
    fn checkpoint_resolve_finds_the_open_clip_pair_and_names_the_family() {
        let dir = tiny_open_clip_model_dir();
        assert!(
            !dir.join("config.json").exists() && !dir.join("model.safetensors").exists(),
            "this test is only meaningful while the fixture carries the open_clip_* names"
        );
        let checkpoint = Checkpoint::resolve(&dir).expect("resolve tiny_open_clip");
        assert_eq!(checkpoint.family, EncoderFamily::OpenClip);
        assert_eq!(checkpoint.family.adapter_model_type(), "open_clip");
        assert_eq!(
            checkpoint.config_path.file_name().and_then(|n| n.to_str()),
            Some("open_clip_config.json")
        );
        assert_eq!(
            checkpoint.weights_path.file_name().and_then(|n| n.to_str()),
            Some("open_clip_model.safetensors")
        );
    }

    /// The CLAP half of the same predicate, on the committed audio fixture:
    /// its `config.json` carries BOTH `model_type: "clap_audio_model"` and
    /// `architectures: ["ClapAudioModelWithProjection"]`, either of which
    /// the serving-side predicate accepts.
    #[test]
    fn checkpoint_resolve_names_the_clap_audio_family() {
        let checkpoint = checkpoint_of(&htsat_clap_tiny_model_dir());
        assert_eq!(checkpoint.family, EncoderFamily::ClapAudio);
        assert_eq!(checkpoint.family.adapter_model_type(), "clap_audio_model");
    }

    /// `--task text_embedding` on an OpenCLIP checkpoint builds the CLIP
    /// TEXT tower, with `in_proj`/`c_fc` (its real site names) trained and
    /// the adapter stamped `open_clip` + `Tower::Text`.
    #[test]
    fn build_encoder_adapters_builds_the_clip_text_tower() {
        let varmap = VarMap::new();
        let (encoder, adapter_cfg) = build_encoder_adapters(
            &checkpoint_of(&tiny_open_clip_model_dir()),
            Task::Text,
            &["in_proj".to_string(), "c_fc".to_string()],
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        )
        .expect("open_clip + text_embedding must build the CLIP text tower");
        assert!(
            matches!(encoder, AnyEncoder::ClipText(_)),
            "open_clip + text_embedding must select the ClipText variant"
        );
        assert_eq!(adapter_cfg.model_type, "open_clip");
        assert_eq!(adapter_cfg.tower, Some(jammi_lora::Tower::Text));
        assert!(
            !encoder.trainable_params().is_empty(),
            "in_proj/c_fc must select real LoRA sites on the CLIP text tower"
        );
        assert!(
            encoder_is_training(&encoder),
            "the built encoder must report training mode"
        );
    }

    /// The SAME checkpoint under `--task image_embedding` builds the OTHER
    /// tower — the sharp case for "the task selects the tower": one
    /// directory, one family, two different encoders and two different
    /// `AdapterConfig.tower` stamps.
    #[test]
    fn build_encoder_adapters_builds_the_open_clip_vision_tower() {
        let varmap = VarMap::new();
        let (encoder, adapter_cfg) = build_encoder_adapters(
            &checkpoint_of(&tiny_open_clip_model_dir()),
            Task::Image,
            &["in_proj".to_string(), "c_fc".to_string()],
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        )
        .expect("open_clip + image_embedding must build the OpenCLIP vision tower");
        assert!(
            matches!(encoder, AnyEncoder::OpenClipVision(_)),
            "open_clip + image_embedding must select the OpenClipVision variant"
        );
        assert_eq!(adapter_cfg.model_type, "open_clip");
        assert_eq!(adapter_cfg.tower, Some(jammi_lora::Tower::Vision));
        assert!(!encoder.trainable_params().is_empty());
        assert!(encoder_is_training(&encoder));
    }

    /// `--task audio_embedding` on the committed HF-CLAP fixture builds the
    /// HTSAT audio tower, stamped `clap_audio_model` + `Tower::Audio`.
    #[test]
    fn build_encoder_adapters_builds_the_htsat_audio_tower() {
        let varmap = VarMap::new();
        let (encoder, adapter_cfg) = build_encoder_adapters(
            &checkpoint_of(&htsat_clap_tiny_model_dir()),
            Task::Audio,
            &["query".to_string(), "value".to_string()],
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        )
        .expect("clap_audio_model + audio_embedding must build the HTSAT audio tower");
        assert!(
            matches!(encoder, AnyEncoder::Htsat(_)),
            "clap_audio_model + audio_embedding must select the Htsat variant"
        );
        assert_eq!(adapter_cfg.model_type, "clap_audio_model");
        assert_eq!(adapter_cfg.tower, Some(jammi_lora::Tower::Audio));
        assert!(!encoder.trainable_params().is_empty());
        assert!(encoder_is_training(&encoder));
    }

    /// A single-tower family leaves `tower` at `None` — this tier stamps a
    /// tower ONLY where a checkpoint genuinely has more than one.
    ///
    /// MEASURED, not assumed, on the wire: `jammi_lora::AdapterConfig::tower`
    /// carries `#[serde(default, skip_serializing_if = "Option::is_none")]`,
    /// so a BERT adapter's `adapter_config.json` OMITS the `tower` key
    /// entirely rather than serialising an explicit `"tower":null` — the
    /// shape every single-tower bundle has. This pins that the key is
    /// genuinely absent from the emitted bytes (not merely `None` after a
    /// round trip).
    #[test]
    fn a_bert_adapter_carries_no_tower_stamp() {
        let varmap = VarMap::new();
        let (_encoder, adapter_cfg) = build_encoder_adapters(
            &checkpoint_of(&tiny_bert_model_dir()),
            Task::Text,
            &["query".to_string(), "value".to_string()],
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        )
        .expect("bert + text_embedding must build");
        assert_eq!(adapter_cfg.model_type, "bert");
        assert_eq!(
            adapter_cfg.tower, None,
            "a single-tower family must not stamp a tower — that would change every existing \
             text leg's saved adapter bytes"
        );
        let json = serde_json::to_string(&adapter_cfg).expect("serialize adapter config");
        assert!(
            !json.contains("tower"),
            "a single-tower family must OMIT the tower key entirely (skip_serializing_if), not \
             emit a null value — the single-tower wire shape: {json}"
        );
        for stamped in ["\"text\"", "\"vision\"", "\"audio\""] {
            assert!(
                !json.contains(stamped),
                "a BERT adapter must never claim a tower it does not have: {json}"
            );
        }
        let reparsed: AdapterConfig =
            serde_json::from_str(&json).expect("the emitted config must round-trip");
        assert_eq!(reparsed.tower, None);
        assert_eq!(reparsed.model_type, "bert");
    }

    /// Negative control on the `(family, task)` pairing: a task with no
    /// tower on this family must be a TYPED refusal naming both halves,
    /// never a silent fallback onto the family's "default" tower — which
    /// would train a text tower while the caller's rows were images.
    #[test]
    fn build_encoder_adapters_refuses_a_task_with_no_tower_on_this_family() {
        let varmap = VarMap::new();
        for (dir, task, family_id) in [
            (tiny_bert_model_dir(), Task::Image, "bert"),
            (tiny_bert_model_dir(), Task::Audio, "bert"),
            (tiny_open_clip_model_dir(), Task::Audio, "open_clip"),
            (htsat_clap_tiny_model_dir(), Task::Text, "clap_audio_model"),
            (htsat_clap_tiny_model_dir(), Task::Image, "clap_audio_model"),
        ] {
            let result = build_encoder_adapters(
                &checkpoint_of(&dir),
                task,
                &["query".to_string(), "in_proj".to_string()],
                &None,
                2,
                4.0,
                0.0,
                LoraInitMode::ZerosB,
                jammi_numerics::ComputePrecision::F32,
                7,
                &Device::Cpu,
                &varmap,
            );
            let msg = match result {
                Ok(_) => panic!(
                    "--task {} on a {family_id} checkpoint has no tower and must be refused",
                    task.as_str()
                ),
                Err(e) => e.to_string(),
            };
            assert!(
                msg.contains(task.as_str()),
                "refusal must name the task it refused: {msg}"
            );
            assert!(
                msg.contains(family_id),
                "refusal must name the family it refused: {msg}"
            );
        }
    }

    /// The zero-trainable refusal on a tower must name that tower's REAL
    /// site names. `q_proj` is a BERT-ish selector that matches nothing on
    /// an OpenCLIP block (whose sites are `in_proj`/`out_proj`/`c_fc`/
    /// `c_proj`), so an operator who wrote it gets told what to write
    /// instead of "correct the selectors".
    #[test]
    fn zero_trainable_refusal_on_open_clip_names_the_real_sites() {
        let varmap = VarMap::new();
        let result = build_encoder_adapters(
            &checkpoint_of(&tiny_open_clip_model_dir()),
            Task::Text,
            &["q_proj".to_string()],
            &None,
            2,
            4.0,
            0.0,
            LoraInitMode::ZerosB,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        );
        let msg = match result {
            Ok(_) => panic!("q_proj matches nothing on an OpenCLIP tower — this must refuse"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("q_proj"),
            "must name the unmatched selector: {msg}"
        );
        assert!(msg.contains("open_clip"), "must name the family: {msg}");
        for site in ["in_proj", "out_proj", "c_fc", "c_proj"] {
            assert!(
                msg.contains(site),
                "refusal must name the tower's real site {site:?}: {msg}"
            );
        }
    }

    /// [`tower_site_names`] transcribes selector strings that live as
    /// crate-private constants in `jammi-encoders`. This makes the
    /// transcription a MEASUREMENT rather than a claim: every name the
    /// refusal offers is fed back through `build_encoder_adapters` on a real
    /// checkpoint of that family, and the build must yield a NON-empty
    /// trainable set. A stale name fails here instead of silently
    /// misdirecting an operator.
    #[test]
    fn refusal_site_names_are_selectors_that_really_train() {
        let distilbert_dir = write_synthetic_distilbert_model_dir(1);
        let modernbert_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/tiny_modernbert_local");
        let cases: Vec<(PathBuf, Task)> = vec![
            (tiny_bert_model_dir(), Task::Text),
            (distilbert_dir.path().to_path_buf(), Task::Text),
            (modernbert_dir, Task::Text),
            (tiny_open_clip_model_dir(), Task::Text),
            (tiny_open_clip_model_dir(), Task::Image),
            (htsat_clap_tiny_model_dir(), Task::Audio),
        ];
        for (dir, task) in cases {
            let checkpoint = checkpoint_of(&dir);
            let family = checkpoint.family;
            let selectors: Vec<String> = tower_site_names(family, task)
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            assert!(
                !selectors.is_empty(),
                "{family:?} must advertise at least one site name"
            );
            let varmap = VarMap::new();
            let (encoder, _cfg) = build_encoder_adapters(
                &checkpoint,
                task,
                &selectors,
                &None,
                2,
                4.0,
                0.0,
                LoraInitMode::ZerosB,
                jammi_numerics::ComputePrecision::F32,
                7,
                &Device::Cpu,
                &varmap,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "{family:?} + {} with its own advertised sites {selectors:?} must build: {e}",
                    task.as_str()
                )
            });
            assert!(
                !encoder.trainable_params().is_empty(),
                "{family:?}'s advertised site names {selectors:?} must select real LoRA sites"
            );
        }
    }

    /// The tower families' half of the dispatch-proof gate. Both
    /// directions, because a control that only ever passes is not
    /// a control:
    ///
    /// - all-zero ATTENTION counters must NOT refuse a tower leg (a tower
    ///   has no whole-attention-block admission counter at all, so refusing
    ///   there would reject every valid media run);
    /// - all-zero LORA counters over a run that took steps MUST refuse it
    ///   (the LoRA cascade is the live proof the adapted encoder was
    ///   forwarded).
    #[test]
    fn fused_dispatch_proof_gate_judges_tower_families_on_the_lora_counters() {
        for family in [EncoderFamily::OpenClip, EncoderFamily::ClapAudio] {
            // Zero attention counters, live LoRA: a healthy tower run.
            assert!(
                fused_dispatch_proof_gate(family, 6, 0, 0, 0, 0, 4, 0).is_ok(),
                "{family:?}: all-zero attention counters are the NORMAL reading for a tower"
            );
            assert!(fused_dispatch_proof_gate(family, 6, 0, 0, 0, 0, 0, 4).is_ok());
            // Live attention counters would not rescue a run whose LoRA
            // sites were never forwarded — the control must not be
            // satisfiable from the wrong counter.
            let msg = match fused_dispatch_proof_gate(family, 6, 9, 9, 9, 9, 0, 0) {
                Ok(()) => panic!(
                    "{family:?} with 6 optimizer steps and no LoRA dispatch at all must be \
                     refused, not rescued by unrelated attention counters"
                ),
                Err(msg) => msg,
            };
            assert!(
                msg.contains("lora_linear_fused_dispatches"),
                "refusal must name the counter it read: {msg}"
            );
            assert!(
                msg.contains(family.adapter_model_type()),
                "refusal must name the family: {msg}"
            );
            // Zero optimizer steps: out of scope for either branch.
            assert!(fused_dispatch_proof_gate(family, 0, 0, 0, 0, 0, 0, 0).is_ok());
        }
    }

    /// `attention_cascade_is_live` is the one predicate that decides which
    /// counter the gate reads; pinning it here keeps a future family from
    /// silently inheriting the wrong branch.
    #[test]
    fn only_the_bert_family_has_a_live_attention_cascade() {
        assert!(attention_cascade_is_live(EncoderFamily::Bert));
        assert!(attention_cascade_is_live(EncoderFamily::DistilBert));
        assert!(attention_cascade_is_live(EncoderFamily::ModernBert));
        assert!(!attention_cascade_is_live(EncoderFamily::OpenClip));
        assert!(!attention_cascade_is_live(EncoderFamily::ClapAudio));
    }

    /// `--task`'s CLI spelling round-trips, and an unknown value is a typed
    /// refusal naming all three (never a silent default to text, which would
    /// train the wrong tower on a media corpus).
    #[test]
    fn task_round_trips_through_its_cli_spelling() {
        for task in [Task::Text, Task::Image, Task::Audio] {
            assert_eq!(task.as_str().parse::<Task>(), Ok(task));
        }
        let err = "vision"
            .parse::<Task>()
            .expect_err("unknown task must refuse");
        for expected in ["text_embedding", "image_embedding", "audio_embedding"] {
            assert!(
                err.contains(expected),
                "refusal must list {expected:?}: {err}"
            );
        }
    }

    /// Task→modality and task→catalog-task are the two mappings the row
    /// loader and the catalog row read; pinned so they cannot drift apart.
    #[test]
    fn task_maps_to_one_modality_and_one_catalog_task() {
        assert_eq!(Task::Text.modality(), jammi_encoders::Modality::Text);
        assert_eq!(Task::Image.modality(), jammi_encoders::Modality::Image);
        assert_eq!(Task::Audio.modality(), jammi_encoders::Modality::Audio);
        assert_eq!(Task::Text.model_task(), ModelTask::TextEmbedding);
        assert_eq!(Task::Image.model_task(), ModelTask::ImageEmbedding);
        assert_eq!(Task::Audio.model_task(), ModelTask::AudioEmbedding);
    }

    /// A media triplet whose three members are not three DISTINCT files is
    /// refused — decided on the MEASURED sha256 of the bytes, not on the
    /// paths (two rows can spell one file two ways). The positive case
    /// (three distinct digests) must pass, so this is not a check that
    /// refuses everything.
    #[test]
    fn media_rows_must_carry_three_distinct_files() {
        let row = |a: &str, p: &str, n: &str| MediaTriplet {
            id: "row-0".to_string(),
            anchor: b"a".to_vec(),
            positive: b"p".to_vec(),
            negative: b"n".to_vec(),
            anchor_sha256: a.to_string(),
            positive_sha256: p.to_string(),
            negative_sha256: n.to_string(),
        };
        let ok = vec![row("aa", "bb", "cc")];
        RowSet::Media(&ok)
            .validate_media_members_are_distinct("--train-jsonl")
            .expect("three distinct digests must pass");

        for (rows, slots) in [
            (vec![row("aa", "aa", "cc")], ("anchor", "positive")),
            (vec![row("aa", "bb", "aa")], ("anchor", "negative")),
            (vec![row("aa", "bb", "bb")], ("positive", "negative")),
        ] {
            let msg =
                match RowSet::Media(&rows).validate_media_members_are_distinct("--train-jsonl") {
                    Ok(()) => panic!("a duplicated member in slots {slots:?} must be refused"),
                    Err(msg) => msg,
                };
            assert!(msg.contains(slots.0) && msg.contains(slots.1), "{msg}");
            assert!(msg.contains("--train-jsonl"), "must name the input: {msg}");
        }

        // Text rows are untouched by this check (see its own doc).
        let text = vec![IdTriplet {
            id: "t".into(),
            anchor: "x".into(),
            positive: "x".into(),
            negative: "x".into(),
        }];
        RowSet::Text(&text)
            .validate_media_members_are_distinct("--train-jsonl")
            .expect("the text path keeps its existing behaviour");
    }

    /// `--task` and the row vectors must agree: a media task carrying text
    /// rows (or the reverse) is refused by name rather than dying downstream
    /// on an opaque "0 rows is not a nonzero multiple of --batch".
    #[test]
    fn params_refuse_rows_that_do_not_match_the_task() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut params = non_perturbation_test_params(tmp.path().to_path_buf());
        // The baseline (text task, text rows) is accepted — without this the
        // refusals below could be passing for the wrong reason.
        params
            .validate_rows_match_task()
            .expect("a text task with text rows must be accepted");

        params.task = Task::Audio;
        let msg = params
            .validate_rows_match_task()
            .expect_err("a media task carrying text rows must be refused");
        assert!(msg.contains("audio_embedding"), "{msg}");
        assert!(msg.contains("text"), "{msg}");

        params.train_pairs.clear();
        params.heldout_pairs.clear();
        let msg = params
            .validate_rows_match_task()
            .expect_err("a media task with no media rows must be refused");
        assert!(msg.contains("no train rows"), "{msg}");
    }

    /// MNRL has no media loader, so an MNRL media leg is a typed refusal
    /// rather than a silent fallback onto the triplet loss under an MNRL
    /// label. The triplet arm of the same rows must succeed, so this is not
    /// a blanket "media is unsupported".
    #[test]
    fn media_rows_refuse_mnrl_and_accept_triplet() {
        let rows = vec![MediaTriplet {
            id: "row-0".into(),
            anchor: b"a".to_vec(),
            positive: b"p".to_vec(),
            negative: b"n".to_vec(),
            anchor_sha256: "aa".into(),
            positive_sha256: "bb".into(),
            negative_sha256: "cc".into(),
        }];
        RowSet::Media(&rows)
            .loader(Objective::Triplet)
            .expect("the triplet media loader must build");
        let err = match RowSet::Media(&rows).loader(Objective::Mnrl) {
            Ok(_) => panic!("mnrl over media rows must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("mnrl"), "{err}");
        assert!(
            err.contains("triplet"),
            "must name the usable objective: {err}"
        );
    }
}
