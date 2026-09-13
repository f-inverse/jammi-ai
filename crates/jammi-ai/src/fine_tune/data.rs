//! Training data loader: reads from DataFusion, produces training batches.
//!
//! Two modes:
//! - **Encode-in-loop** (`from_contrastive` / `from_triplets` /
//!   `from_media_triplets` / `from_rows`): stores raw inputs (text strings or
//!   encoded media blobs — audio clips or images). Use `text_chunks()` to get
//!   batches for model-in-loop training (encode through the base model,
//!   project through LoRA). Text and media chunks differ only in how the base
//!   model turns one example into an embedding — the loss, head, and
//!   optimizer are shared.
//! - **Precomputed** (`from_precomputed`): stores pre-built tensor batches.
//!   `batches()` returns them as-is. Used in tests.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow::array::RecordBatch;
use candle_core::Tensor;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::TrainingSetTable;

use super::worker::{extract_binary_column, extract_numeric_column, extract_string_column};
use crate::session::InferenceSession;

/// A training batch — either contrastive pairs or triplets.
#[derive(Clone)]
pub enum TrainingBatch {
    /// Contrastive: two embedding tensors plus target similarity scores.
    Contrastive {
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        scores: Tensor,
    },
    /// Pairs: anchor and positive embeddings only. In-batch negatives are the
    /// other rows' positives, so no explicit negative column is carried — the
    /// `MultipleNegativesRanking` objective scores each anchor against every
    /// positive in the batch.
    Pairs { anchors: Tensor, positives: Tensor },
    /// Triplet: anchor, positive, and negative embeddings.
    Triplet {
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    },
    /// Classification: embeddings + integer class labels.
    Classification {
        embeddings: Tensor,
        labels: Tensor, // shape (batch_size,) u32
    },
    /// NER: hidden states for all tokens + per-token labels.
    Ner {
        hidden_states: Tensor, // (batch, seq_len, hidden)
        labels: Tensor,        // (batch, seq_len) as i64, -100 for ignored tokens
    },
    /// Regression (S18): the distributional head's raw output plus the observed
    /// continuous targets. `input` is `(batch, k)` — the unconstrained head
    /// parameters (`k = 2` `(mean, raw_std)` for the Gaussian objectives,
    /// `k = levels` for the pinball objective); `target` is `(batch,)` the
    /// observed `y`. The proper-scoring loss reads a positive `σ` from
    /// `raw_std` via `floor + softplus`, so the head trains in the
    /// unconstrained space.
    Regression { input: Tensor, target: Tensor },
}

/// Format of training data, detected from column names.
#[derive(Debug, Clone, Copy)]
pub enum TrainingFormat {
    /// `text_a, text_b, score` — contrastive pairs with scores.
    Contrastive,
    /// `anchor, positive` — contrastive pairs with no explicit negatives. The
    /// `MultipleNegativesRanking` objective draws negatives from the rest of
    /// the batch.
    Pairs,
    /// `anchor, positive, negative` — text triplet format.
    Triplet,
    /// `anchor, positive, negative` — MEDIA triplet format. The three
    /// columns carry encoded binary blobs, not text: audio clips
    /// (WAV/FLAC/MP3/Ogg bytes) or images (PNG/JPEG/… bytes).
    ///
    /// The MODALITY is not carried by this variant and is not sniffed from
    /// the bytes: it is the job's own [`crate::model::ModelTask`]
    /// (`audio_embedding` / `image_embedding`), which the loader's caller
    /// already supplies and which the trainer dispatches its front end on.
    /// An encoded WAV and an encoded PNG are both `Vec<u8>`; the column
    /// shape genuinely cannot distinguish them, so the declared task is the
    /// authority rather than a byte-header guess.
    ///
    /// What makes a blob a "positive" (augmentation-similar or
    /// co-occurring-complementary) is the caller's data, not the trainer's
    /// concern — the loss only minimizes the triplet objective over whatever
    /// items the caller paired.
    MediaTriplet,
    /// Classification with label-to-index mapping.
    Classification { num_classes: usize },
    /// NER with BIO tag mapping.
    Ner { num_labels: usize },
    /// Regression (S18): `text, target` rows — one input text and one observed
    /// continuous outcome. The trainer encodes the text through the frozen base
    /// model + the distributional projection head, then scores the head's
    /// parameters against the target with the configured proper-scoring
    /// objective.
    Regression,
    /// Graph-supervised (S11): the rows were sampled from a graph (node text +
    /// edge table) by biased random walks into `(anchor, positive,
    /// [hard_negative])` pairs. It carries **no new loss** — it is a
    /// data-loading shape that drives the existing in-batch-negative
    /// (`Pairs`/MNRL) or `Triplet` path, selected by `has_negatives`:
    /// `false` → `Pairs` (in-batch negatives only), `true` → `Triplet` (the
    /// sampler mined structure-aware hard negatives). The variant is retained
    /// for provenance — a consumer can see the supervision came from the graph —
    /// while every downstream step reuses the Pairs/Triplet machinery.
    Graph { has_negatives: bool },
}

/// The concrete batch/loss shape a [`TrainingFormat`] resolves to once the
/// provenance-carrying `Graph` variant is mapped onto its in-batch-negative
/// shape. There is no `Graph` here by construction — a graph loader trains as
/// `Pairs` or `Triplet`, so the chunk/loss dispatch matches on this exhaustive
/// set without a phantom arm.
#[derive(Debug, Clone, Copy)]
enum UnderlyingFormat {
    Contrastive,
    Pairs,
    Triplet,
    MediaTriplet,
    Classification,
    Ner,
    Regression,
}

/// Every canonical training-format tag, in declaration order — the CLOSED set
/// [`TrainingFormat::format_tag`] maps onto.
///
/// The set is closed by the round-trip test below rather than by convention:
/// the tag of every variant must appear here, the entries must be pairwise
/// distinct, and [`TrainingFormat::from_format_tag`] must recover a value with
/// the same tag for each. A tag that changes is a *different* training set, so
/// a rename is a breaking change to every recorded definition hash, not a
/// cosmetic edit.
pub const TRAINING_FORMAT_TAGS: &[&str] = &[
    "contrastive",
    "pairs",
    "triplet",
    "media_triplet",
    "classification",
    "ner",
    "regression",
    "graph_pairs",
    "graph_triplet",
];

impl TrainingFormat {
    /// The canonical string tag a materialised training set records this format
    /// under — the ONE mapping from a [`TrainingFormat`] to the `format` field
    /// of
    /// [`ProducingDescriptor::TrainingSet`](jammi_db::store::manifest::ProducingDescriptor::TrainingSet).
    ///
    /// `jammi-db` depends on no jammi crate but `jammi-numerics`, so the
    /// descriptor folds a string rather than this enum, and the completeness
    /// burden lands here: **a format distinction this mapping does not spell is
    /// two different training sets colliding on one definition hash.** The match
    /// is exhaustive with no `_` arm, so a new variant cannot reach the
    /// descriptor without being given a tag.
    ///
    /// The two parameterised variants deliberately drop their parameter.
    /// `Classification`'s `num_classes` and `Ner`'s `num_labels` are *functions
    /// of the rows* — the count of distinct labels the loader observed — not
    /// independent choices a caller makes, so they cannot distinguish two
    /// training sets built from the same source, columns and task, and folding
    /// them would require reading the rows before naming the table that holds
    /// them. `Graph`'s `has_negatives` is not in that class: it changes the
    /// projected column set (a mined negative is a third column) and therefore
    /// the committed bytes, so it takes two distinct tags.
    pub fn format_tag(self) -> &'static str {
        match self {
            TrainingFormat::Contrastive => "contrastive",
            TrainingFormat::Pairs => "pairs",
            TrainingFormat::Triplet => "triplet",
            TrainingFormat::MediaTriplet => "media_triplet",
            TrainingFormat::Classification { .. } => "classification",
            TrainingFormat::Ner { .. } => "ner",
            TrainingFormat::Regression => "regression",
            TrainingFormat::Graph {
                has_negatives: false,
            } => "graph_pairs",
            TrainingFormat::Graph {
                has_negatives: true,
            } => "graph_triplet",
        }
    }

    /// The inverse of [`Self::format_tag`] over [`TRAINING_FORMAT_TAGS`]: the
    /// representative format a recorded tag names, with the two data-derived
    /// parameters at zero (the tag never carried them — see
    /// [`Self::format_tag`] — so no value of theirs is recoverable and zero is
    /// the neutral stand-in, never a claim about the rows).
    ///
    /// `None` for a tag this build does not know, which is how a table
    /// committed by a newer build is refused rather than read as some
    /// near-miss format.
    pub fn from_format_tag(tag: &str) -> Option<Self> {
        match tag {
            "contrastive" => Some(TrainingFormat::Contrastive),
            "pairs" => Some(TrainingFormat::Pairs),
            "triplet" => Some(TrainingFormat::Triplet),
            "media_triplet" => Some(TrainingFormat::MediaTriplet),
            "classification" => Some(TrainingFormat::Classification { num_classes: 0 }),
            "ner" => Some(TrainingFormat::Ner { num_labels: 0 }),
            "regression" => Some(TrainingFormat::Regression),
            "graph_pairs" => Some(TrainingFormat::Graph {
                has_negatives: false,
            }),
            "graph_triplet" => Some(TrainingFormat::Graph {
                has_negatives: true,
            }),
            _ => None,
        }
    }
}

impl TrainingFormat {
    /// The concrete shape a format trains as: a graph with mined hard negatives
    /// is a `Triplet`, one without is `Pairs`; every other format is itself.
    /// This is the single place that maps the provenance-carrying `Graph`
    /// variant onto the loss/batch machinery, so `text_chunks` /
    /// `in_batch_negative_texts` stay DRY.
    fn underlying(self) -> UnderlyingFormat {
        match self {
            TrainingFormat::Contrastive => UnderlyingFormat::Contrastive,
            TrainingFormat::Pairs => UnderlyingFormat::Pairs,
            TrainingFormat::Triplet => UnderlyingFormat::Triplet,
            TrainingFormat::MediaTriplet => UnderlyingFormat::MediaTriplet,
            TrainingFormat::Classification { .. } => UnderlyingFormat::Classification,
            TrainingFormat::Ner { .. } => UnderlyingFormat::Ner,
            TrainingFormat::Regression => UnderlyingFormat::Regression,
            TrainingFormat::Graph {
                has_negatives: true,
            } => UnderlyingFormat::Triplet,
            TrainingFormat::Graph {
                has_negatives: false,
            } => UnderlyingFormat::Pairs,
        }
    }
}

/// A chunk of text data for one training batch. The training loop encodes
/// these through the base model before computing loss.
#[derive(Debug)]
pub enum TextChunk {
    Contrastive {
        texts_a: Vec<String>,
        texts_b: Vec<String>,
        scores: Vec<f32>,
    },
    Pairs {
        anchors: Vec<String>,
        positives: Vec<String>,
    },
    Triplet {
        anchors: Vec<String>,
        positives: Vec<String>,
        negatives: Vec<String>,
    },
    /// One batch of MEDIA triplets. Each element is one encoded blob —
    /// audio clip or image, per the job's task — which the training loop
    /// runs through that modality's front end and then the tower (LoRA
    /// inside it, for an encoder-adapters run) or the frozen tower plus the
    /// LoRA projection head, exactly as the text path does for
    /// [`TextChunk::Triplet`].
    MediaTriplet {
        anchors: Vec<Vec<u8>>,
        positives: Vec<Vec<u8>>,
        negatives: Vec<Vec<u8>>,
    },
    Classification {
        texts: Vec<String>,
        labels: Vec<u32>,
    },
    Ner {
        texts: Vec<String>,
        /// Per-text entity spans as JSON strings (same format as inference output).
        entities_json: Vec<String>,
    },
    /// One batch of regression rows: input texts and their observed continuous
    /// targets. The training loop encodes the texts through the base model + the
    /// distributional head, then scores the head output against `targets`.
    Regression {
        texts: Vec<String>,
        targets: Vec<f32>,
    },
}

impl TextChunk {
    /// The number of rows this chunk carries — every field of a well-formed
    /// chunk is the same length, so any one of them reports it. A zero here
    /// is a valid, well-formed state (a zero-row rank at the trailing global
    /// batch, DESIGN.md §2, K2), not a malformed chunk: every producer of a
    /// [`TextChunk`] (`TrainingDataLoader::rows_to_text_chunk`) builds it by
    /// `.map().collect()` over a row slice that can itself be empty.
    pub fn row_count(&self) -> usize {
        match self {
            TextChunk::Contrastive { texts_a, .. } => texts_a.len(),
            TextChunk::Pairs { anchors, .. } => anchors.len(),
            TextChunk::Triplet { anchors, .. } => anchors.len(),
            TextChunk::MediaTriplet { anchors, .. } => anchors.len(),
            TextChunk::Classification { texts, .. } => texts.len(),
            TextChunk::Ner { texts, .. } => texts.len(),
            TextChunk::Regression { texts, .. } => texts.len(),
        }
    }
}

/// The flattened in-batch-negative view of a text loader: `(anchors,
/// positives, optional explicit negatives)`. Consumed by GradCache and
/// hard-negative mining, which treat the dataset as one in-batch-negative batch.
pub type InBatchNegativeTexts = (Vec<String>, Vec<String>, Option<Vec<String>>);

/// Internal storage: text rows already resident (from source, or the tests-only
/// synthetic constructors), precomputed batches (tests only), or a per-epoch
/// STREAM over a committed training set's Parquet row groups (the production
/// `FineTune` path — see [`StreamSource`]).
enum LoaderData {
    TextRows(Vec<TrainingRow>),
    Precomputed(Vec<TrainingBatch>),
    // Boxed: `StreamSource` (session handle, table record, columns, tokio
    // handle, mutexed stream state) dwarfs the other two variants, and every
    // `TrainingDataLoader` value pays that size regardless of which variant
    // it actually holds.
    Stream(Box<StreamSource>),
}

/// How a per-epoch training-set stream is bounded: the per-rank batch size
/// (also the DataFusion execution batch size the scoped read is pinned to,
/// so a polled `RecordBatch` is at most this many rows — see
/// [`open_row_range_stream`]) and how many such batches may be resident
/// (decoded, reserved, not yet handed to the caller) at once. Together these
/// are the residency bound acceptance (a) checks: `batch * prefetch` rows.
#[derive(Debug, Clone, Copy)]
pub struct StreamConfig {
    pub batch: usize,
    pub prefetch: usize,
}

/// The residency accounting a streamed epoch's background reader task
/// reserves against before decoding each chunk, and the consumer releases
/// (by dropping the returned permit) once it hands that chunk's rows to the
/// caller.
///
/// **The counting seam** acceptance (a) asserts on: [`Self::high_water_mark`]
/// is the largest number of rows ever simultaneously reserved. Backed by a
/// [`tokio::sync::Semaphore`] with `bound` permits — reserving `n` permits
/// for an `n`-row chunk BLOCKS (async) until enough residency frees up, so
/// the bound is enforced BY CONSTRUCTION, not merely measured after the
/// fact: a background task can never decode past it, only wait. **This is
/// why it cannot under-count** (the R-A this unit's contract asks for): the
/// unit test `residency_bound_try_acquire_refuses_past_the_shrunk_bound`
/// shrinks `bound` by one row and shows a reservation that fit at the full
/// bound is refused at `bound - 1`, non-blockingly (`try_acquire_many`) —
/// the semaphore's own permit count, not a value this type merely reports.
struct ResidencyBound {
    bound: usize,
    semaphore: Arc<tokio::sync::Semaphore>,
    high_water: Arc<std::sync::atomic::AtomicUsize>,
}

impl ResidencyBound {
    fn new(bound: usize) -> Self {
        Self {
            bound,
            semaphore: Arc::new(tokio::sync::Semaphore::new(bound)),
            high_water: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Reserve `n` rows' worth of residency, waiting (async) until enough is
    /// free. `n` must not exceed `bound` (the caller picks `batch <=
    /// batch * prefetch`, which always holds for `prefetch >= 1`) — a larger
    /// request can never be satisfied and would wait forever, which is the
    /// correct, visible failure mode for a caller that mis-sized its own
    /// bound rather than a value silently exceeding it.
    async fn reserve(&self, n: usize) -> tokio::sync::OwnedSemaphorePermit {
        let permit = Arc::clone(&self.semaphore)
            .acquire_many_owned(n as u32)
            .await
            .expect("the residency semaphore is never closed while its reader task is alive");
        let occupied = self.bound - self.semaphore.available_permits();
        self.high_water
            .fetch_max(occupied, std::sync::atomic::Ordering::SeqCst);
        permit
    }

    fn high_water_mark(&self) -> usize {
        self.high_water.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// One decoded, row-bounded piece of a per-epoch stream, plus the permit
/// that reserves its residency until the receiver drops it.
type StreamItem = Result<(Vec<TrainingRow>, tokio::sync::OwnedSemaphorePermit)>;

/// A committed training set, streamed a bounded window of rows at a time —
/// the mechanism `TrainingDataLoader::from_training_set_stream` builds and
/// `TrainingDataLoader::text_chunk_for_rank` (the `Stream` arm) drives.
///
/// Holds no decoded rows itself between calls: a fresh per-epoch stream opens
/// on `step == 0` (DESIGN.md §2 — "a per-epoch RecordBatch stream") and each
/// call pulls exactly one bounded chunk, tracked by `state`'s `next_step` so
/// a caller that asks in the sequential order the trainer always does (0, 1,
/// 2, ...) never re-reads a row. A caller that asks out of that order (a step
/// behind, or ahead of, `next_step`) gets a FRESH restart from row 0 with the
/// intervening chunks fast-forwarded (received and dropped) — correct for
/// any access pattern, but only genuinely bounded/efficient for the
/// sequential one the trainer's loop and this unit's acceptance tests use.
struct StreamSource {
    session: Arc<InferenceSession>,
    table: TrainingSetTable,
    columns: Vec<String>,
    /// This loader's OWN row range within the committed table — the train
    /// prefix or the validation suffix, fixed once by
    /// [`TrainingDataLoader::split`] and never touched afterward.
    range: Range<usize>,
    cfg: StreamConfig,
    /// The training loop runs on a `spawn_blocking` thread
    /// (`worker.rs::train_fine_tune`), so every read here bridges from that
    /// SYNC context back into the tokio runtime — captured once, at
    /// construction, on the ASYNC side that built this loader.
    runtime: tokio::runtime::Handle,
    state: Mutex<StreamState>,
}

enum StreamState {
    /// No epoch stream open — a fresh loader, or the last one fully drained.
    Idle,
    /// An epoch stream is open. `next_step` is the step this receiver's next
    /// `recv()` will satisfy.
    Open {
        next_step: usize,
        rx: tokio::sync::mpsc::UnboundedReceiver<StreamItem>,
        residency: Arc<ResidencyBound>,
    },
}

impl std::fmt::Debug for StreamSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamSource")
            .field("table", &self.table.table_name())
            .field("columns", &self.columns)
            .field("range", &self.range)
            .field("cfg", &self.cfg)
            .finish()
    }
}

/// Opens a [`datafusion::physical_plan::SendableRecordBatchStream`] over
/// `[range.start, range.end)` of `table`, reading with **no `ORDER BY`** at a
/// SCOPED `target_partitions = 1` — the mechanism behind acceptance (e). The
/// producer already writes a training set's rows sorted by the full
/// projected tuple before committing them
/// (`crates/jammi-db/src/store/mod.rs`'s `plan_training_set_rows`), so a
/// single-partition sequential scan visits row groups in exactly that
/// committed order with no re-sort needed; the physical plan for a plain
/// `SELECT * ... LIMIT ... OFFSET ...` (no `ORDER BY` clause at all) never
/// contains a `SortExec` to begin with, so there is nothing to prove
/// sortedness against.
///
/// The scoped session clones the CALLER's own
/// [`datafusion::execution::session_state::SessionState`] — a shallow,
/// `Arc`-shared clone, so it resolves the SAME tenant-gated catalog the
/// committed table's `jammi.{name}` binding lives in — and overrides only
/// `target_partitions` (pinned to 1) and `batch_size` (pinned to the
/// caller's per-rank `batch`, so a polled `RecordBatch` is at most `batch`
/// rows). The AMBIENT session's own configured `target_partitions` (1 or N
/// — acceptance (e) is checked at both) never reaches this scan.
async fn open_row_range_stream(
    session: &InferenceSession,
    table: &TrainingSetTable,
    range: Range<usize>,
    batch_size: usize,
) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
    let mut state = session.context().state();
    let scoped_config = state
        .config()
        .clone()
        .with_target_partitions(1)
        .with_batch_size(batch_size.max(1));
    *state.config_mut() = scoped_config;
    let scoped_ctx = datafusion::prelude::SessionContext::new_with_state(state);
    let sql = format!(
        "SELECT * FROM {} LIMIT {} OFFSET {}",
        table.sql_relation(),
        range.len(),
        range.start
    );
    let df = scoped_ctx.sql(&sql).await?;
    Ok(df.execute_stream().await?)
}

/// Everything one call to [`run_epoch_stream`] needs — bundled so the
/// function takes one argument instead of the seven independent pieces
/// (`clippy::too_many_arguments`).
struct EpochStreamJob {
    session: Arc<InferenceSession>,
    table: TrainingSetTable,
    columns: Vec<String>,
    format: TrainingFormat,
    range: Range<usize>,
    cfg: StreamConfig,
    residency: Arc<ResidencyBound>,
}

/// Runs on [`StreamSource::runtime`]: opens
/// [`open_row_range_stream`] and, for each polled `RecordBatch`, decodes it
/// into [`TrainingRow`]s under `job.format`/`job.columns`
/// ([`decode_record_batch`]), reserves that many rows against
/// `job.residency` (blocking this task, never the consumer, when the bound
/// is full), and sends `(rows, permit)` into `tx`. Exits silently once
/// `tx`'s receiver drops (the loader moved on without draining this epoch's
/// stream to the end — never reached by the trainer's own sequential
/// access, but a correct, non-panicking exit for any other caller).
async fn run_epoch_stream(job: EpochStreamJob, tx: tokio::sync::mpsc::UnboundedSender<StreamItem>) {
    use futures::StreamExt;

    let mut stream =
        match open_row_range_stream(&job.session, &job.table, job.range, job.cfg.batch).await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Err(e));
                return;
            }
        };
    while let Some(batch) = stream.next().await {
        let batch = match batch {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.send(Err(JammiError::from(e)));
                return;
            }
        };
        let rows = match decode_record_batch(job.format, &job.columns, &batch) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Err(e));
                return;
            }
        };
        if rows.is_empty() {
            continue;
        }
        let permit = job.residency.reserve(rows.len()).await;
        if tx.send(Ok((rows, permit))).is_err() {
            return; // The receiver (loader) was dropped mid-epoch.
        }
    }
}

/// Decode ONE polled `RecordBatch` into [`TrainingRow`]s under `format` — the
/// per-chunk analogue of `worker::build_training_data_loader`'s per-batch
/// loop bodies (this decodes exactly one batch; [`run_epoch_stream`] loops
/// over the stream calling it once per polled batch).
///
/// `format` is the loader's OWN already-fixed format (the committed table's
/// tag, decided once by `worker::detect_training_format` before the table was
/// even materialised) — never re-detected from `columns` here.
///
/// `TrainingFormat::Classification { .. }` is refused: classification needs
/// every row's label before any row's class INDEX is knowable
/// (`build_training_data_loader`'s own `label_to_idx` is built from the
/// whole label column, a genuinely whole-set dependency, not a per-chunk
/// one), so `TrainingDataLoader::from_training_set_stream` falls back to an
/// eager whole-table build for it rather than calling this function.
/// `TrainingFormat::Graph { .. }` maps onto `Pairs`/`Triplet` here (its own
/// `underlying()`) and decodes through those arms — this function would
/// handle it correctly if it were ever called, but at this commit
/// `worker::reconstruct_graph_loader` stays on its existing eager read (see
/// this unit's report for why) and never reaches this function.
fn decode_record_batch(
    format: TrainingFormat,
    columns: &[String],
    batch: &RecordBatch,
) -> Result<Vec<TrainingRow>> {
    let missing =
        |col: &str| JammiError::FineTune(format!("streamed batch missing column '{col}'"));
    let not_text = |col: &str| JammiError::FineTune(format!("streamed column '{col}' is not text"));
    match format.underlying() {
        UnderlyingFormat::Contrastive => {
            let a = extract_string_column(
                batch
                    .column_by_name("text_a")
                    .ok_or_else(|| missing("text_a"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("text_a"))?;
            let b = extract_string_column(
                batch
                    .column_by_name("text_b")
                    .ok_or_else(|| missing("text_b"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("text_b"))?;
            let score_col = batch
                .column_by_name("score")
                .ok_or_else(|| missing("score"))?;
            let scores: Vec<f32> = extract_numeric_column(score_col.as_ref())
                .map_err(|_| JammiError::FineTune("streamed 'score' is not numeric".into()))?;
            Ok((0..batch.num_rows())
                .map(|i| TrainingRow::Contrastive {
                    text_a: a[i].clone(),
                    text_b: b[i].clone(),
                    score: scores[i],
                })
                .collect())
        }
        UnderlyingFormat::Pairs => {
            let anchor = extract_string_column(
                batch
                    .column_by_name("anchor")
                    .ok_or_else(|| missing("anchor"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("anchor"))?;
            let positive = extract_string_column(
                batch
                    .column_by_name("positive")
                    .ok_or_else(|| missing("positive"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("positive"))?;
            Ok((0..batch.num_rows())
                .map(|i| TrainingRow::Pairs {
                    anchor: anchor[i].clone(),
                    positive: positive[i].clone(),
                })
                .collect())
        }
        UnderlyingFormat::Triplet => {
            let anchor = extract_string_column(
                batch
                    .column_by_name("anchor")
                    .ok_or_else(|| missing("anchor"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("anchor"))?;
            let positive = extract_string_column(
                batch
                    .column_by_name("positive")
                    .ok_or_else(|| missing("positive"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("positive"))?;
            let negative = extract_string_column(
                batch
                    .column_by_name("negative")
                    .ok_or_else(|| missing("negative"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("negative"))?;
            Ok((0..batch.num_rows())
                .map(|i| TrainingRow::Triplet {
                    anchor: anchor[i].clone(),
                    positive: positive[i].clone(),
                    negative: negative[i].clone(),
                })
                .collect())
        }
        UnderlyingFormat::MediaTriplet => {
            let anchor = extract_binary_column(
                batch
                    .column_by_name("anchor")
                    .ok_or_else(|| missing("anchor"))?
                    .as_ref(),
            )
            .ok_or_else(|| JammiError::FineTune("streamed 'anchor' is not binary".into()))?;
            let positive = extract_binary_column(
                batch
                    .column_by_name("positive")
                    .ok_or_else(|| missing("positive"))?
                    .as_ref(),
            )
            .ok_or_else(|| JammiError::FineTune("streamed 'positive' is not binary".into()))?;
            let negative = extract_binary_column(
                batch
                    .column_by_name("negative")
                    .ok_or_else(|| missing("negative"))?
                    .as_ref(),
            )
            .ok_or_else(|| JammiError::FineTune("streamed 'negative' is not binary".into()))?;
            Ok((0..batch.num_rows())
                .map(|i| TrainingRow::MediaTriplet {
                    anchor: anchor[i].clone(),
                    positive: positive[i].clone(),
                    negative: negative[i].clone(),
                })
                .collect())
        }
        UnderlyingFormat::Regression => {
            let text = extract_string_column(
                batch
                    .column_by_name("text")
                    .ok_or_else(|| missing("text"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("text"))?;
            let target_col = batch
                .column_by_name("target")
                .ok_or_else(|| missing("target"))?;
            let target = extract_numeric_column(target_col.as_ref()).map_err(|e| match e {
                super::worker::NumericColumnError::NotNumeric => JammiError::FineTune(format!(
                    "streamed regression 'target' is not numeric (Arrow type {})",
                    target_col.data_type()
                )),
                super::worker::NumericColumnError::Null(i) => JammiError::FineTune(format!(
                    "streamed regression 'target' has a null at row {i}"
                )),
                super::worker::NumericColumnError::Nan(i) => JammiError::FineTune(format!(
                    "streamed regression 'target' has a NaN at row {i}"
                )),
            })?;
            Ok((0..batch.num_rows())
                .map(|i| TrainingRow::Regression {
                    text: text[i].clone(),
                    target: target[i],
                })
                .collect())
        }
        UnderlyingFormat::Ner => {
            let text = extract_string_column(
                batch
                    .column_by_name("text")
                    .ok_or_else(|| missing("text"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("text"))?;
            let entities = extract_string_column(
                batch
                    .column_by_name("entities_json")
                    .ok_or_else(|| missing("entities_json"))?
                    .as_ref(),
            )
            .ok_or_else(|| not_text("entities_json"))?;
            Ok((0..batch.num_rows())
                .map(|i| TrainingRow::Ner {
                    text: text[i].clone(),
                    entities_json: entities[i].clone(),
                })
                .collect())
        }
        UnderlyingFormat::Classification => Err(JammiError::FineTune(format!(
            "classification cannot be decoded one streamed chunk at a time (columns {columns:?}): \
             the class index needs the WHOLE label column first"
        ))),
    }
}

/// The eager whole-table build [`TrainingDataLoader::from_training_set_stream`]
/// falls back to for `TrainingFormat::Classification` — mirrors
/// `worker::build_training_data_loader`'s classification arm exactly (same
/// `text`/`label` column names, same label→index derivation over the WHOLE
/// label set before any row's class index is built), just reading `range`
/// off the committed table directly rather than off an already-materialised
/// `Vec<RecordBatch>`.
async fn build_classification_loader_eager(
    session: &InferenceSession,
    table: &TrainingSetTable,
    range: Range<usize>,
) -> Result<TrainingDataLoader> {
    let sql = format!(
        "SELECT * FROM {} LIMIT {} OFFSET {}",
        table.sql_relation(),
        range.len(),
        range.start
    );
    let batches = session.sql(&sql).await?;
    let mut label_set = std::collections::BTreeSet::new();
    let mut rows: Vec<(String, String)> = Vec::new();
    for batch in &batches {
        let text_vals = extract_string_column(
            batch
                .column_by_name("text")
                .ok_or_else(|| JammiError::FineTune("missing column 'text'".into()))?
                .as_ref(),
        )
        .ok_or_else(|| JammiError::FineTune("'text' is not text".into()))?;
        let label_vals = extract_string_column(
            batch
                .column_by_name("label")
                .ok_or_else(|| JammiError::FineTune("missing column 'label'".into()))?
                .as_ref(),
        )
        .ok_or_else(|| JammiError::FineTune("'label' is not text".into()))?;
        for i in 0..batch.num_rows() {
            label_set.insert(label_vals[i].clone());
            rows.push((text_vals[i].clone(), label_vals[i].clone()));
        }
    }
    let label_to_idx: std::collections::HashMap<String, u32> = label_set
        .iter()
        .enumerate()
        .map(|(i, l)| (l.clone(), i as u32))
        .collect();
    let num_classes = label_to_idx.len();
    let indexed_rows: Vec<(String, u32)> = rows
        .into_iter()
        .map(|(text, label)| (text, label_to_idx[&label]))
        .collect();
    Ok(TrainingDataLoader::from_classification(
        indexed_rows,
        num_classes,
    ))
}

/// Loads training data and produces batches of tensors.
///
/// Two construction modes:
/// - `from_contrastive` / `from_triplets` / `from_rows`: stores text.
///   Use `text_chunks()` for model-in-loop training.
/// - `from_precomputed`: stores pre-built batches, `batches()` returns them as-is.
///   Used in tests to exercise divergence detection, early stopping, etc.
pub struct TrainingDataLoader {
    format: TrainingFormat,
    data: LoaderData,
}

/// Text data for one training example.
#[derive(Clone)]
enum TrainingRow {
    Contrastive {
        text_a: String,
        text_b: String,
        score: f32,
    },
    Pairs {
        anchor: String,
        positive: String,
    },
    Triplet {
        anchor: String,
        positive: String,
        negative: String,
    },
    MediaTriplet {
        anchor: Vec<u8>,
        positive: Vec<u8>,
        negative: Vec<u8>,
    },
    Classification {
        text: String,
        label: u32,
    },
    Ner {
        text: String,
        /// JSON-serialized entity spans.
        entities_json: String,
    },
    Regression {
        text: String,
        target: f32,
    },
}

impl TrainingDataLoader {
    /// Create a loader from contrastive pair rows.
    pub fn from_contrastive(rows: Vec<(String, String, f32)>) -> Self {
        Self {
            format: TrainingFormat::Contrastive,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, b, s)| TrainingRow::Contrastive {
                        text_a: a,
                        text_b: b,
                        score: s,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from classification rows (text + integer label).
    pub fn from_classification(rows: Vec<(String, u32)>, num_classes: usize) -> Self {
        Self {
            format: TrainingFormat::Classification { num_classes },
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(text, label)| TrainingRow::Classification { text, label })
                    .collect(),
            ),
        }
    }

    /// Create a loader from NER rows (text + JSON entity spans).
    pub fn from_ner(rows: Vec<(String, String)>, num_labels: usize) -> Self {
        Self {
            format: TrainingFormat::Ner { num_labels },
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(text, entities_json)| TrainingRow::Ner {
                        text,
                        entities_json,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from regression rows (input text + observed continuous
    /// target). The trainer encodes each text through the base model and the
    /// distributional projection head, then scores the head's parameters against
    /// the target with the configured proper-scoring objective (S18).
    pub fn from_regression(rows: Vec<(String, f32)>) -> Self {
        Self {
            format: TrainingFormat::Regression,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(text, target)| TrainingRow::Regression { text, target })
                    .collect(),
            ),
        }
    }

    /// Create a loader from triplet rows.
    pub fn from_triplets(rows: Vec<(String, String, String)>) -> Self {
        Self {
            format: TrainingFormat::Triplet,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p, n)| TrainingRow::Triplet {
                        anchor: a,
                        positive: p,
                        negative: n,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from contrastive pair rows `(anchor, positive)`. The
    /// `MultipleNegativesRanking` objective draws negatives from the rest of
    /// each batch, so no explicit negative column is needed.
    pub fn from_pairs(rows: Vec<(String, String)>) -> Self {
        Self {
            format: TrainingFormat::Pairs,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p)| TrainingRow::Pairs {
                        anchor: a,
                        positive: p,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from a graph (S11): sample the node-text + edge-table
    /// graph into `(anchor, positive, [hard_negative])` text rows by biased
    /// random walks, then store them as the underlying `Pairs` (no mined
    /// negatives) or `Triplet` (structure-mined hard negatives) rows. The loader
    /// reports [`TrainingFormat::Graph`] for provenance, but the rows feed the
    /// existing MNRL/Triplet path unchanged — S11 adds **no new loss**.
    ///
    /// The sampler enforces the text-bearing precondition (every edge endpoint
    /// must resolve to a [`super::graph_sampler::TextNode`]) and the collapse /
    /// false-negative guards, so any violation surfaces here as a typed error.
    ///
    /// **Circularity caveat:** if the edges are S9-similarity edges the
    /// supervision largely re-learns the base metric; genuine gain comes from
    /// declared / external edges (see [`super::graph_sampler`]).
    pub fn from_graph(sampler: &super::graph_sampler::GraphSampler) -> Result<Self> {
        let pairs = sampler.sample()?;
        // The whole dataset shares one shape: if any pair carries mined hard
        // negatives the format is Triplet, otherwise Pairs. The sampler emits a
        // uniform shape (hard_negatives is a single config knob), so the first
        // pair determines it; an empty dataset is already a sampler error.
        let has_negatives = pairs.first().is_some_and(|p| !p.hard_negatives.is_empty());

        let rows = pairs
            .into_iter()
            .map(|p| {
                if has_negatives {
                    // Use the first mined negative as the explicit triplet
                    // negative; the rest still contribute via in-batch negatives
                    // (MNRL appends the explicit one as an extra column).
                    let negative = p.hard_negatives.into_iter().next().ok_or_else(|| {
                        JammiError::FineTune(
                            "graph pair declared hard negatives but supplied none".into(),
                        )
                    })?;
                    Ok(TrainingRow::Triplet {
                        anchor: p.anchor,
                        positive: p.positive,
                        negative,
                    })
                } else {
                    Ok(TrainingRow::Pairs {
                        anchor: p.anchor,
                        positive: p.positive,
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            format: TrainingFormat::Graph { has_negatives },
            data: LoaderData::TextRows(rows),
        })
    }

    /// Create a loader from MEDIA triplet rows. Each element is
    /// `(anchor_bytes, positive_bytes, negative_bytes)` where every field is
    /// one encoded blob — an audio clip or an image, per the job's task (see
    /// [`TrainingFormat::MediaTriplet`] for why the task, not the bytes, is
    /// the modality authority). The trainer encodes these through that
    /// modality's front end and the base model; the contrastive objective is
    /// identical to the text triplet path.
    pub fn from_media_triplets(rows: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>) -> Self {
        Self {
            format: TrainingFormat::MediaTriplet,
            data: LoaderData::TextRows(
                rows.into_iter()
                    .map(|(a, p, n)| TrainingRow::MediaTriplet {
                        anchor: a,
                        positive: p,
                        negative: n,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader with N synthetic rows (for testing validation split logic).
    pub fn from_rows(count: usize) -> Self {
        Self {
            format: TrainingFormat::Contrastive,
            data: LoaderData::TextRows(
                (0..count)
                    .map(|i| TrainingRow::Contrastive {
                        text_a: format!("text_a_{i}"),
                        text_b: format!("text_b_{i}"),
                        score: 0.5,
                    })
                    .collect(),
            ),
        }
    }

    /// Create a loader from pre-built batches. `batches()` returns clones of
    /// these directly instead of generating placeholder tensors. Used in tests
    /// to supply crafted tensors that trigger specific training behaviors
    /// (divergence, early stopping, etc.).
    pub fn from_precomputed(batches: Vec<TrainingBatch>) -> Self {
        Self {
            format: TrainingFormat::Contrastive,
            data: LoaderData::Precomputed(batches),
        }
    }

    /// Build a per-epoch STREAMING loader over a committed training set
    /// (DESIGN.md §2), covering the WHOLE table (`0..table.record.row_count`)
    /// at construction — [`Self::split`] narrows it to a train prefix / a
    /// validation suffix with no I/O.
    ///
    /// `format = TrainingFormat::Classification { .. }` cannot be decoded a
    /// chunk at a time ([`decode_record_batch`]'s doc: the class index needs
    /// the WHOLE label column first), so this constructor falls back to an
    /// eager whole-table build for it — `build_classification_loader_eager`
    /// mirrors `worker::build_training_data_loader`'s classification arm
    /// exactly (same column names, same label→index derivation), so a
    /// classification run's residency is simply exempt from this unit's
    /// bound rather than silently wrong; every other format streams
    /// genuinely, bounded by `cfg`.
    pub async fn from_training_set_stream(
        session: Arc<InferenceSession>,
        table: TrainingSetTable,
        columns: Vec<String>,
        format: TrainingFormat,
        cfg: StreamConfig,
    ) -> Result<Self> {
        let range = 0..table.record.row_count;
        if matches!(format, TrainingFormat::Classification { .. }) {
            return build_classification_loader_eager(&session, &table, range).await;
        }
        let runtime = tokio::runtime::Handle::current();
        Ok(Self {
            format,
            data: LoaderData::Stream(Box::new(StreamSource {
                session,
                table,
                columns,
                range,
                cfg,
                runtime,
                state: Mutex::new(StreamState::Idle),
            })),
        })
    }

    /// Total number of data points (rows for text, batches for precomputed).
    pub fn len(&self) -> usize {
        match &self.data {
            LoaderData::TextRows(rows) => rows.len(),
            LoaderData::Precomputed(batches) => batches.len(),
            LoaderData::Stream(src) => src.range.len(),
        }
    }

    /// Whether the loader has no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of batches at the given batch size.
    pub fn num_batches(&self, batch_size: usize) -> usize {
        match &self.data {
            LoaderData::TextRows(rows) => {
                if batch_size == 0 || rows.is_empty() {
                    0
                } else {
                    rows.len().div_ceil(batch_size)
                }
            }
            LoaderData::Precomputed(batches) => batches.len(),
            LoaderData::Stream(src) => {
                if batch_size == 0 || src.range.is_empty() {
                    0
                } else {
                    src.range.len().div_ceil(batch_size)
                }
            }
        }
    }

    /// Deterministic split: last `fraction` of data goes to validation.
    ///
    /// The `Stream` arm does no I/O at all — it narrows the SAME committed
    /// table's row range by the identical `val_count = round(len *
    /// fraction)` arithmetic the `TextRows` arm uses, and hands each half a
    /// fresh, idle [`StreamState`] (a split loader has never opened an epoch
    /// stream yet, whatever the parent had done).
    pub fn split(&self, fraction: f64) -> Result<(TrainingDataLoader, TrainingDataLoader)> {
        match &self.data {
            LoaderData::TextRows(rows) => {
                let val_count = (rows.len() as f64 * fraction).round() as usize;
                let train_count = rows.len() - val_count;
                Ok((
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::TextRows(rows[..train_count].to_vec()),
                    },
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::TextRows(rows[train_count..].to_vec()),
                    },
                ))
            }
            LoaderData::Precomputed(batches) => {
                let val_count = (batches.len() as f64 * fraction).round() as usize;
                let train_count = batches.len() - val_count;
                Ok((
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Precomputed(batches[..train_count].to_vec()),
                    },
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Precomputed(batches[train_count..].to_vec()),
                    },
                ))
            }
            LoaderData::Stream(src) => {
                let len = src.range.len();
                let val_count = (len as f64 * fraction).round() as usize;
                let train_count = len - val_count;
                let train_range = src.range.start..src.range.start + train_count;
                let val_range = src.range.start + train_count..src.range.end;
                let fresh = |range: Range<usize>| {
                    Box::new(StreamSource {
                        session: Arc::clone(&src.session),
                        table: src.table.clone(),
                        columns: src.columns.clone(),
                        range,
                        cfg: src.cfg,
                        runtime: src.runtime.clone(),
                        state: Mutex::new(StreamState::Idle),
                    })
                };
                Ok((
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Stream(fresh(train_range)),
                    },
                    TrainingDataLoader {
                        format: self.format,
                        data: LoaderData::Stream(fresh(val_range)),
                    },
                ))
            }
        }
    }

    /// Indices of rows in this loader (for testing split logic).
    pub fn indices(&self) -> impl Iterator<Item = usize> {
        0..self.len()
    }

    /// Produce training batches (precomputed mode only).
    ///
    /// For text-based loaders: returns an error — use `text_chunks()` instead.
    /// For precomputed loaders: returns the pre-built batches.
    pub fn batches(&self, _batch_size: usize) -> Result<Vec<Result<TrainingBatch>>> {
        match &self.data {
            LoaderData::TextRows(_) | LoaderData::Stream(_) => Err(JammiError::FineTune(
                "Text-based loaders require model-in-loop encoding. Use text_chunks() instead."
                    .into(),
            )),
            LoaderData::Precomputed(batches) => Ok(batches.iter().map(|b| Ok(b.clone())).collect()),
        }
    }

    /// Produce text chunks for model-in-loop training. Each chunk is one
    /// batch of text data to be encoded through the base model.
    /// Only works for text-based loaders (from_contrastive/from_triplets/from_rows).
    /// Returns empty for precomputed loaders.
    ///
    /// Fallible (unlike the `TextRows`/`Precomputed` arms) because the
    /// `Stream` arm drains its own range whole — the SAME whole-set exemption
    /// [`Self::in_batch_negative_texts`] uses — which is real I/O. The
    /// trainer's MAIN loop never calls this for a `Stream` loader (it drives
    /// [`Self::text_chunk_for_rank`] instead, per-step and residency-bounded);
    /// this arm exists so `evaluate`/`evaluate_held_out` — which call
    /// `text_chunks` on the validation split, not the train prefix — still
    /// work correctly on a streamed loader, just without that bound.
    pub fn text_chunks(&self, batch_size: usize) -> Result<Vec<TextChunk>> {
        match &self.data {
            LoaderData::TextRows(rows) => Ok(rows
                .chunks(batch_size)
                .map(|chunk| self.rows_to_text_chunk(chunk))
                .collect()),
            LoaderData::Precomputed(_) => Ok(Vec::new()),
            LoaderData::Stream(src) => {
                let rows = self.stream_drain_all(src)?;
                Ok(rows
                    .chunks(batch_size.max(1))
                    .map(|chunk| self.rows_to_text_chunk(chunk))
                    .collect())
            }
        }
    }

    /// One [`TextChunk`] from a slice of this loader's own [`TrainingRow`]s —
    /// the single per-chunk row→chunk converter [`Self::text_chunks`] (every
    /// row, `batch_size` at a time) and [`Self::text_chunk_for_rank`] (one
    /// [`super::partition::PartitionSpec`]-selected slice at a time) both
    /// drive, so the two never risk decoding a chunk differently. `chunk` may
    /// be EMPTY — a zero-row rank (DESIGN.md §2) still produces a well-formed
    /// [`TextChunk`] with empty inner vectors, never a panic: every arm here
    /// is a plain `.map().collect()` over `chunk`, which is total on an empty
    /// slice.
    ///
    /// A `Graph` loader stores `Pairs`/`Triplet` rows, so it encodes through
    /// its underlying shape — the provenance variant carries no chunk shape
    /// of its own.
    fn rows_to_text_chunk(&self, chunk: &[TrainingRow]) -> TextChunk {
        match self.format.underlying() {
            UnderlyingFormat::Contrastive => TextChunk::Contrastive {
                texts_a: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Contrastive { text_a, .. } => text_a.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                texts_b: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Contrastive { text_b, .. } => text_b.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                scores: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Contrastive { score, .. } => *score,
                        _ => 0.0,
                    })
                    .collect(),
            },
            UnderlyingFormat::Pairs => TextChunk::Pairs {
                anchors: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Pairs { anchor, .. } => anchor.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                positives: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Pairs { positive, .. } => positive.clone(),
                        _ => String::new(),
                    })
                    .collect(),
            },
            UnderlyingFormat::Triplet => TextChunk::Triplet {
                anchors: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Triplet { anchor, .. } => anchor.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                positives: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Triplet { positive, .. } => positive.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                negatives: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Triplet { negative, .. } => negative.clone(),
                        _ => String::new(),
                    })
                    .collect(),
            },
            UnderlyingFormat::MediaTriplet => TextChunk::MediaTriplet {
                anchors: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::MediaTriplet { anchor, .. } => anchor.clone(),
                        _ => Vec::new(),
                    })
                    .collect(),
                positives: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::MediaTriplet { positive, .. } => positive.clone(),
                        _ => Vec::new(),
                    })
                    .collect(),
                negatives: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::MediaTriplet { negative, .. } => negative.clone(),
                        _ => Vec::new(),
                    })
                    .collect(),
            },
            UnderlyingFormat::Classification => TextChunk::Classification {
                texts: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Classification { text, .. } => text.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                labels: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Classification { label, .. } => *label,
                        _ => 0,
                    })
                    .collect(),
            },
            UnderlyingFormat::Ner => TextChunk::Ner {
                texts: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Ner { text, .. } => text.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                entities_json: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Ner { entities_json, .. } => entities_json.clone(),
                        _ => String::new(),
                    })
                    .collect(),
            },
            UnderlyingFormat::Regression => TextChunk::Regression {
                texts: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Regression { text, .. } => text.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                targets: chunk
                    .iter()
                    .map(|r| match r {
                        TrainingRow::Regression { target, .. } => *target,
                        _ => 0.0,
                    })
                    .collect(),
            },
        }
    }

    /// The [`TextChunk`] rank `spec.rank` of `spec.world` holds for global
    /// step `step`, over THIS loader's own row count as the train prefix
    /// (DESIGN.md §2, partition rule v1 — [`super::partition::PartitionSpec::
    /// rows_for_step`] computes the slice; this method decodes it through the
    /// SAME [`Self::rows_to_text_chunk`] converter [`Self::text_chunks`]
    /// uses). A rank whose slice is empty (a zero-row rank at the trailing
    /// global batch, K2) yields a well-formed [`TextChunk`] with empty inner
    /// vectors, never an out-of-bounds panic — `rows_for_step` never returns
    /// a range past `rows.len()`.
    ///
    /// Text-rows-backed loaders only (`Precomputed` has no row-level
    /// partition; a `Stream` loader reads its own range directly rather than
    /// slicing an in-memory `Vec` — see `worker.rs`'s `run_spec` for that
    /// path, exercised only at `spec.world == 1` at this commit).
    pub fn text_chunk_for_rank(
        &self,
        spec: &super::partition::PartitionSpec,
        step: usize,
    ) -> Result<TextChunk> {
        match &self.data {
            LoaderData::TextRows(rows) => {
                let range = spec.rows_for_step(rows.len(), step);
                Ok(self.rows_to_text_chunk(&rows[range]))
            }
            LoaderData::Precomputed(_) => Err(JammiError::FineTune(
                "a precomputed loader has no row-level partition".into(),
            )),
            LoaderData::Stream(src) => {
                // At this commit `run_spec` always builds rank 0 of world 1
                // (U4b spawns the per-rank readers a genuinely larger world
                // would need); refuse rather than silently reading rank 0's
                // rows for every rank.
                if spec.world != 1 || spec.rank != 0 {
                    return Err(JammiError::FineTune(format!(
                        "a streamed loader supports only rank 0 of world 1 at this commit \
                         (got rank {} of world {}); U4b spawns per-rank readers",
                        spec.rank, spec.world
                    )));
                }
                self.stream_next_chunk(src, spec.batch, step)
            }
        }
    }

    /// The `Stream` arm of [`Self::text_chunk_for_rank`]: serve step `step`
    /// of a per-epoch stream over `src`'s own range, opening a FRESH one
    /// whenever `step == 0` or the caller asks behind where the current
    /// stream already is (DESIGN.md §2 — "a per-epoch RecordBatch stream").
    ///
    /// Efficient and residency-bounded ONLY for the sequential access
    /// pattern the trainer's loop and this unit's acceptance tests use (`0,
    /// 1, 2, ...`, once per epoch): an out-of-order ask still returns the
    /// CORRECT chunk (by restarting and fast-forwarding, discarding the
    /// skipped chunks), never a wrong one, but pays the discarded I/O.
    fn stream_next_chunk(
        &self,
        src: &StreamSource,
        batch: usize,
        step: usize,
    ) -> Result<TextChunk> {
        let mut guard = src
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let needs_restart = step == 0
            || match &*guard {
                StreamState::Open { next_step, .. } => *next_step > step,
                StreamState::Idle => true,
            };
        if needs_restart {
            let cfg = StreamConfig {
                batch,
                prefetch: src.cfg.prefetch,
            };
            let residency = Arc::new(ResidencyBound::new(
                cfg.batch.saturating_mul(cfg.prefetch.max(1)),
            ));
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let job = EpochStreamJob {
                session: Arc::clone(&src.session),
                table: src.table.clone(),
                columns: src.columns.clone(),
                format: self.format,
                range: src.range.clone(),
                cfg,
                residency: Arc::clone(&residency),
            };
            src.runtime.spawn(run_epoch_stream(job, tx));
            *guard = StreamState::Open {
                next_step: 0,
                rx,
                residency,
            };
        }
        loop {
            let (current_step, item) = match &mut *guard {
                StreamState::Open { next_step, rx, .. } => {
                    let item = src.runtime.block_on(rx.recv());
                    let served = *next_step;
                    *next_step += 1;
                    (served, item)
                }
                StreamState::Idle => unreachable!("just opened Open above"),
            };
            match item {
                None => {
                    // The stream is exhausted: every step from here on is a
                    // zero-row state (K2), not an error.
                    return Ok(self.rows_to_text_chunk(&[]));
                }
                Some(Err(e)) => return Err(e),
                Some(Ok((rows, permit))) => {
                    // Handed to the caller now — release the residency this
                    // chunk reserved.
                    drop(permit);
                    if current_step == step {
                        return Ok(self.rows_to_text_chunk(&rows));
                    }
                    // `current_step < step`: an out-of-order ask asked ahead
                    // of a stream that had already progressed less far;
                    // discard and keep fast-forwarding.
                }
            }
        }
    }

    /// The detected training data format.
    pub fn format(&self) -> TrainingFormat {
        self.format
    }

    /// Every regression target in this loader, in row order — the whole-dataset
    /// view the trainer reduces into a fixed target scaler once before the
    /// loop (K3). `Ok(None)` for any non-regression loader (no targets to
    /// standardise) and for the precomputed test path (which supplies
    /// head/target tensors directly, not text rows).
    ///
    /// The named 4-byte-per-row scaler exemption from the residency bound
    /// (DESIGN.md §2): a `Stream` loader still collects this column ONCE,
    /// fully, into a single `Vec<f32>` — the same bit-identical two-pass
    /// `TargetScaler::from_targets` shape `Ok`/`TextRows` already gave —
    /// rather than through the bounded per-chunk path.
    ///
    /// Fallible (unlike the `TextRows`/`Precomputed` arms, which never touch
    /// I/O) because the `Stream` arm issues its own one-shot query; a
    /// non-regression or precomputed loader never reaches that query and
    /// never fails.
    pub fn regression_targets(&self) -> Result<Option<Vec<f32>>> {
        if !matches!(self.format, TrainingFormat::Regression) {
            return Ok(None);
        }
        match &self.data {
            LoaderData::TextRows(rows) => Ok(Some(
                rows.iter()
                    .filter_map(|row| match row {
                        TrainingRow::Regression { target, .. } => Some(*target),
                        _ => None,
                    })
                    .collect(),
            )),
            LoaderData::Precomputed(_) => Ok(None),
            LoaderData::Stream(src) => Ok(Some(stream_collect_target_column(src)?)),
        }
    }

    /// Flatten the in-batch-negative text rows into `(anchors, positives,
    /// negatives)`, the whole-dataset view GradCache and hard-negative mining
    /// consume. `negatives` is `Some` for a `Triplet` loader (explicit hard
    /// negatives) and `None` for a `Pairs` loader. Returns an error for any
    /// other format — only in-batch-negative training has this shape.
    ///
    /// A `Stream` loader drains its own range fully here — the named
    /// whole-set exemption (DESIGN.md §2): mining/GradCache are W=1-only
    /// (U4a's refusal) and "stream the table in and hold what they need",
    /// never bounded by the per-chunk residency check.
    pub fn in_batch_negative_texts(&self) -> Result<InBatchNegativeTexts> {
        let owned_rows;
        let rows: &[TrainingRow] = match &self.data {
            LoaderData::TextRows(rows) => rows,
            LoaderData::Precomputed(_) => {
                return Err(JammiError::FineTune(
                    "GradCache requires text rows, not precomputed batches".into(),
                ))
            }
            LoaderData::Stream(src) => {
                owned_rows = self.stream_drain_all(src)?;
                &owned_rows
            }
        };
        // A `Graph` loader is itself an in-batch-negative loader — it stores
        // `Pairs`/`Triplet` rows — so it resolves through `underlying()` and
        // flows into mining / GradCache exactly like a hand-built pair set.
        match self.format.underlying() {
            UnderlyingFormat::Pairs => {
                let mut anchors = Vec::with_capacity(rows.len());
                let mut positives = Vec::with_capacity(rows.len());
                for row in rows {
                    if let TrainingRow::Pairs { anchor, positive } = row {
                        anchors.push(anchor.clone());
                        positives.push(positive.clone());
                    }
                }
                Ok((anchors, positives, None))
            }
            UnderlyingFormat::Triplet => {
                let mut anchors = Vec::with_capacity(rows.len());
                let mut positives = Vec::with_capacity(rows.len());
                let mut negatives = Vec::with_capacity(rows.len());
                for row in rows {
                    if let TrainingRow::Triplet {
                        anchor,
                        positive,
                        negative,
                    } = row
                    {
                        anchors.push(anchor.clone());
                        positives.push(positive.clone());
                        negatives.push(negative.clone());
                    }
                }
                Ok((anchors, positives, Some(negatives)))
            }
            other => Err(JammiError::FineTune(format!(
                "GradCache applies only to in-batch-negative formats (pairs/triplet), not {other:?}"
            ))),
        }
    }

    /// Whether this loader was constructed from pre-built tensor batches
    /// (typically a test fixture) rather than text rows that must be
    /// encoded through a model.
    pub fn is_precomputed(&self) -> bool {
        matches!(self.data, LoaderData::Precomputed(_))
    }

    /// Fully drain a `Stream` loader's own range into `TrainingRow`s —
    /// the whole-set exemption every whole-dataset consumer
    /// (`in_batch_negative_texts`, and `text_chunks`'s `Stream` fallback for
    /// any non-main-loop caller such as `evaluate`) goes through. Blocks the
    /// calling (sync, `spawn_blocking`) thread on `src`'s captured tokio
    /// handle; never touches the bounded residency machinery
    /// (`ResidencyBound`) `run_epoch_stream` uses for the main loop.
    fn stream_drain_all(&self, src: &StreamSource) -> Result<Vec<TrainingRow>> {
        let format = self.format;
        let columns = src.columns.clone();
        let session = Arc::clone(&src.session);
        let table = src.table.clone();
        let range = src.range.clone();
        let batch = src.cfg.batch.max(1);
        src.runtime.block_on(async move {
            use futures::StreamExt;
            let mut stream = open_row_range_stream(&session, &table, range, batch).await?;
            let mut rows = Vec::new();
            while let Some(batch) = stream.next().await {
                let batch = batch.map_err(JammiError::from)?;
                rows.extend(decode_record_batch(format, &columns, &batch)?);
            }
            Ok(rows)
        })
    }

    /// Test-only counting seam: the residency bound's own
    /// [`ResidencyBound::high_water_mark`] for the current (or last-run) per-epoch
    /// stream — `None` for a non-`Stream` loader or a `Stream` loader whose
    /// epoch stream has never been opened. Acceptance (a) asserts on this.
    /// `test-hooks`-gated (this crate's own test targets always build with it
    /// on, via its `[dev-dependencies]` self-dependency) rather than
    /// `#[cfg(test)]` alone, so the `tests/it` integration binary — a
    /// SEPARATE compilation of this crate — can reach it too.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn stream_residency_high_water_mark(&self) -> Option<usize> {
        match &self.data {
            LoaderData::Stream(src) => {
                let guard = src
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match &*guard {
                    StreamState::Open { residency, .. } => Some(residency.high_water_mark()),
                    StreamState::Idle => None,
                }
            }
            _ => None,
        }
    }
}

/// The K3 scaler exemption's one-shot query: collect the WHOLE `target`
/// column of `src`'s own range into one `Vec<f32>`, in committed order — the
/// same single, unbounded read [`TrainingDataLoader::regression_targets`]'s
/// `TextRows` arm already gave (there, because everything was already
/// resident; here, by a dedicated single-column query rather than the
/// per-chunk residency-bounded path).
fn stream_collect_target_column(src: &StreamSource) -> Result<Vec<f32>> {
    let session = Arc::clone(&src.session);
    let table = src.table.clone();
    let range = src.range.clone();
    src.runtime.block_on(async move {
        use futures::StreamExt;
        // `batch_size` here only bounds how many rows arrive per polled
        // `RecordBatch` of this ONE-OFF collect; nothing here is
        // residency-bounded, so any positive value is fine — the caller's
        // configured per-rank batch keeps this consistent with the epoch
        // stream's own scan shape.
        let mut stream = open_row_range_stream(&session, &table, range.clone(), 8192).await?;
        let mut targets = Vec::with_capacity(range.len());
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(JammiError::from)?;
            let target_col = batch.column_by_name("target").ok_or_else(|| {
                JammiError::FineTune("streamed batch missing column 'target'".into())
            })?;
            let values = extract_numeric_column(target_col.as_ref()).map_err(|e| match e {
                super::worker::NumericColumnError::NotNumeric => JammiError::FineTune(format!(
                    "streamed regression 'target' is not numeric (Arrow type {})",
                    target_col.data_type()
                )),
                super::worker::NumericColumnError::Null(i) => JammiError::FineTune(format!(
                    "streamed regression 'target' has a null at row {i}"
                )),
                super::worker::NumericColumnError::Nan(i) => JammiError::FineTune(format!(
                    "streamed regression 'target' has a NaN at row {i}"
                )),
            })?;
            targets.extend(values);
        }
        Ok(targets)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R-A for acceptance (a): the residency bound's own permit count is what
    /// a reservation actually relies on, never merely reported. At the real
    /// bound (`batch * prefetch = 8`) a fresh `batch`-sized (4-row)
    /// reservation succeeds twice (filling the bound exactly) and a third
    /// is refused (`try_acquire_many`, non-blocking, so the probe cannot
    /// hang). Shrink the SAME configuration by exactly one row (bound 7) and
    /// the SECOND reservation — which fit at bound 8 — is now refused too:
    /// "make the bound one row too small and watch it fail."
    #[test]
    fn residency_bound_semaphore_refuses_one_row_past_a_shrunk_bound() {
        let batch = 4usize;
        let prefetch = 2usize;
        let bound = batch * prefetch; // 8

        let full = ResidencyBound::new(bound);
        let first = full.semaphore.try_acquire_many(batch as u32);
        assert!(first.is_ok(), "the first 4-row chunk must fit at bound 8");
        let second = full.semaphore.try_acquire_many(batch as u32);
        assert!(
            second.is_ok(),
            "the second 4-row chunk must ALSO fit — 4 + 4 == the bound exactly"
        );
        let third = full.semaphore.try_acquire_many(1);
        assert!(
            third.is_err(),
            "a bound of 8 must refuse a 9th row — the semaphore is exhausted"
        );

        // R-A: the SAME batch/prefetch, but the residency bound is
        // constructed one row too small (7, not 8).
        let shrunk = ResidencyBound::new(bound - 1);
        let first = shrunk.semaphore.try_acquire_many(batch as u32);
        assert!(first.is_ok(), "the first 4-row chunk still fits at bound 7");
        let second = shrunk.semaphore.try_acquire_many(batch as u32);
        assert!(
            second.is_err(),
            "at the shrunk bound (7), the SAME second 4-row chunk that fit at 8 must now be \
             refused — the bound is one row too small to hold two full chunks"
        );
    }

    /// Every [`TrainingFormat`] value the tag mapping must cover, in
    /// [`TRAINING_FORMAT_TAGS`] order. Hand-written, and ANCHORED to the
    /// compiler by [`enumeration_index`] below — a new variant fails to
    /// compile there until it is added here too, so the round-trip tests can
    /// never range over a stale subset.
    fn every_training_format() -> Vec<TrainingFormat> {
        vec![
            TrainingFormat::Contrastive,
            TrainingFormat::Pairs,
            TrainingFormat::Triplet,
            TrainingFormat::MediaTriplet,
            TrainingFormat::Classification { num_classes: 7 },
            TrainingFormat::Ner { num_labels: 5 },
            TrainingFormat::Regression,
            TrainingFormat::Graph {
                has_negatives: false,
            },
            TrainingFormat::Graph {
                has_negatives: true,
            },
        ]
    }

    /// The compiler anchor for [`every_training_format`]: an exhaustive match
    /// with no `_` arm mapping each variant to its position in that list.
    fn enumeration_index(format: TrainingFormat) -> usize {
        match format {
            TrainingFormat::Contrastive => 0,
            TrainingFormat::Pairs => 1,
            TrainingFormat::Triplet => 2,
            TrainingFormat::MediaTriplet => 3,
            TrainingFormat::Classification { .. } => 4,
            TrainingFormat::Ner { .. } => 5,
            TrainingFormat::Regression => 6,
            TrainingFormat::Graph {
                has_negatives: false,
            } => 7,
            TrainingFormat::Graph {
                has_negatives: true,
            } => 8,
        }
    }

    #[test]
    fn the_format_enumeration_is_the_whole_enum() {
        let all = every_training_format();
        for (i, format) in all.iter().enumerate() {
            assert_eq!(
                enumeration_index(*format),
                i,
                "{format:?} is out of position in every_training_format()"
            );
        }
        // Every index the anchor can return is occupied, so the list has no
        // hole a variant could hide in.
        let mut occupied: Vec<usize> = all.iter().map(|f| enumeration_index(*f)).collect();
        occupied.sort_unstable();
        assert_eq!(occupied, (0..all.len()).collect::<Vec<_>>());
    }

    /// The tag mapping is a bijection between the variants and
    /// [`TRAINING_FORMAT_TAGS`], and it round-trips through
    /// [`TrainingFormat::from_format_tag`].
    ///
    /// Injectivity is the load-bearing half: two variants sharing a tag are two
    /// different training sets colliding on one definition hash, which is
    /// exactly the failure the string-tag indirection risks.
    #[test]
    fn format_tags_are_a_bijection_and_round_trip() {
        let all = every_training_format();

        let tags: Vec<&str> = all.iter().map(|f| f.format_tag()).collect();
        assert_eq!(
            tags, TRAINING_FORMAT_TAGS,
            "the declared tag set and the variants' own tags must agree, in order"
        );

        let mut distinct = tags.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            tags.len(),
            "two formats share a tag, so two different training sets would              collide on one definition hash: {tags:?}"
        );

        for format in &all {
            let tag = format.format_tag();
            let recovered = TrainingFormat::from_format_tag(tag)
                .unwrap_or_else(|| panic!("tag {tag:?} is not decodable"));
            assert_eq!(recovered.format_tag(), tag);
        }
        for tag in TRAINING_FORMAT_TAGS {
            let recovered = TrainingFormat::from_format_tag(tag)
                .unwrap_or_else(|| panic!("declared tag {tag:?} is not decodable"));
            assert_eq!(&recovered.format_tag(), tag);
        }
    }

    /// The two data-derived parameters are dropped by design, and the one
    /// byte-affecting parameter is not.
    #[test]
    fn the_tag_drops_row_derived_parameters_and_keeps_the_shape_one() {
        assert_eq!(
            TrainingFormat::Classification { num_classes: 3 }.format_tag(),
            TrainingFormat::Classification { num_classes: 900 }.format_tag(),
            "num_classes is a function of the rows, not of the table's identity"
        );
        assert_eq!(
            TrainingFormat::Ner { num_labels: 3 }.format_tag(),
            TrainingFormat::Ner { num_labels: 900 }.format_tag(),
            "num_labels is a function of the rows, not of the table's identity"
        );
        assert_ne!(
            TrainingFormat::Graph {
                has_negatives: false
            }
            .format_tag(),
            TrainingFormat::Graph {
                has_negatives: true
            }
            .format_tag(),
            "a mined negative is a third projected column, so it is a different table"
        );
    }

    #[test]
    fn an_unknown_tag_is_refused_rather_than_approximated() {
        assert!(TrainingFormat::from_format_tag("graph").is_none());
        assert!(TrainingFormat::from_format_tag("Contrastive").is_none());
        assert!(TrainingFormat::from_format_tag("").is_none());
    }

    /// A regression loader carries `TrainingFormat::Regression` and chunks its
    /// rows into `TextChunk::Regression { texts, targets }` — the shape the
    /// trainer encodes through the distributional head. Pins the S18 data path
    /// from constructor to chunk.
    #[test]
    fn regression_loader_chunks_into_regression_text_chunks() {
        let loader = TrainingDataLoader::from_regression(vec![
            ("cheap".into(), 0.1),
            ("mid".into(), 0.5),
            ("dear".into(), 0.9),
        ]);
        assert!(matches!(loader.format(), TrainingFormat::Regression));
        assert_eq!(loader.len(), 3);

        let chunks = loader.text_chunks(2).unwrap();
        assert_eq!(chunks.len(), 2, "3 rows at batch 2 → two chunks");
        match &chunks[0] {
            TextChunk::Regression { texts, targets } => {
                assert_eq!(texts, &["cheap".to_string(), "mid".to_string()]);
                assert_eq!(targets, &[0.1, 0.5]);
            }
            _ => panic!("regression loader must yield a Regression chunk"),
        }
        match &chunks[1] {
            TextChunk::Regression { texts, targets } => {
                assert_eq!(texts, &["dear".to_string()]);
                assert_eq!(targets, &[0.9]);
            }
            _ => panic!("regression loader must yield a Regression chunk"),
        }
    }

    /// The validation split preserves the regression format on both halves.
    #[test]
    fn regression_split_keeps_format() {
        let loader = TrainingDataLoader::from_regression(
            (0..10).map(|i| (format!("r{i}"), i as f32)).collect(),
        );
        let (train, val) = loader.split(0.2).unwrap();
        assert!(matches!(train.format(), TrainingFormat::Regression));
        assert!(matches!(val.format(), TrainingFormat::Regression));
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    /// Acceptance (b): for W ∈ {1, 2, 4}, the MULTISET of rows over all ranks
    /// at each global step equals the W=1 batch at that step, on a
    /// `train_count` (7) that is NOT a multiple of `W·B` for any tested W —
    /// asserting a zero-row rank actually occurs for W=2 and W=4 (DESIGN.md
    /// §2; PRESSURE round-2 design finding 6). RED at base: neither
    /// `PartitionSpec` nor `text_chunk_for_rank` exist there.
    ///
    /// Per-determinant table (reported alongside this test in the
    /// eval-verdict): the fixture is chosen so W=2 hits its zero-row rank at
    /// the LAST step (rank 1) and W=4 hits it at the FIRST step (rank 3) —
    /// two different positions in the epoch, not the same one twice.
    #[test]
    fn partition_rule_multiset_matches_the_w1_batch_with_zero_row_ranks() {
        use super::super::partition::{PartitionRule, PartitionSpec};

        let train_count = 7usize;
        let per_rank_batch = 3usize;
        let loader = TrainingDataLoader::from_pairs(
            (0..train_count)
                .map(|i| (format!("a{i}"), format!("p{i}")))
                .collect(),
        );

        fn anchors_of(chunk: &TextChunk) -> Vec<String> {
            match chunk {
                TextChunk::Pairs { anchors, .. } => anchors.clone(),
                other => panic!("expected a Pairs chunk, got a different TextChunk arm: {other:?}"),
            }
        }

        for &world in &[1usize, 2, 4] {
            let w1_ref = PartitionSpec {
                rank: 0,
                world: 1,
                batch: per_rank_batch * world,
                rule: PartitionRule::BlockByGlobalBatch,
            };
            let mut zero_row_seen = false;
            let mut step = 0usize;
            loop {
                let w1_range = w1_ref.rows_for_step(train_count, step);
                if w1_range.is_empty() {
                    break;
                }
                let w1_chunk = loader.text_chunk_for_rank(&w1_ref, step).unwrap();
                let expected = anchors_of(&w1_chunk);

                let mut union = Vec::new();
                for rank in 0..world {
                    let spec = PartitionSpec {
                        rank,
                        world,
                        batch: per_rank_batch,
                        rule: PartitionRule::BlockByGlobalBatch,
                    };
                    let chunk = loader.text_chunk_for_rank(&spec, step).unwrap();
                    let rank_anchors = anchors_of(&chunk);
                    if rank_anchors.is_empty() {
                        zero_row_seen = true;
                    }
                    union.extend(rank_anchors);
                }
                assert_eq!(
                    union, expected,
                    "world={world} step={step}: the union over ranks must equal the W=1 batch"
                );
                step += 1;
            }
            if world > 1 {
                assert!(
                    zero_row_seen,
                    "world={world} on train_count={train_count}, batch={per_rank_batch} must hit \
                     a zero-row rank on this fixture, and none did"
                );
            }
        }
    }

    /// R-A for (b): shrinking a rank's batch to a size the fixture cannot
    /// possibly fill for a real chunk still returns a well-formed (empty)
    /// chunk rather than panicking, and a step past every rank's data is
    /// empty for every rank — the zero-row state is total, not a
    /// coincidence of the one fixture above.
    #[test]
    fn partition_rule_a_step_past_the_train_prefix_is_zero_rows_for_every_rank() {
        use super::super::partition::{PartitionRule, PartitionSpec};
        let train_count = 5usize;
        let loader = TrainingDataLoader::from_pairs(
            (0..train_count)
                .map(|i| (format!("a{i}"), format!("p{i}")))
                .collect(),
        );
        for world in [1usize, 2, 4] {
            for rank in 0..world {
                let spec = PartitionSpec {
                    rank,
                    world,
                    batch: 3,
                    rule: PartitionRule::BlockByGlobalBatch,
                };
                // Step 10 is far past any row this 5-row fixture could ever
                // reach at batch 3 for any tested world.
                let chunk = loader.text_chunk_for_rank(&spec, 10).unwrap();
                match chunk {
                    TextChunk::Pairs { anchors, positives } => {
                        assert!(anchors.is_empty() && positives.is_empty());
                    }
                    _ => panic!("expected a Pairs chunk"),
                }
            }
        }
    }
}
