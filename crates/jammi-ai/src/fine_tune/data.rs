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

use candle_core::Tensor;
use jammi_db::error::{JammiError, Result};

/// The train/validation split boundary over a row COUNT alone: the last `round(total * fraction)`
/// rows go to validation, so the train prefix is `[0, split_index(total, fraction))`. The ONE place
/// this arithmetic is spelled — [`TrainingDataLoader::split`] (over an already
/// in-memory row/batch count) and [`super::source::StreamedSet`]'s
/// `train_count` (over a catalog row count, no scan) both call this, so a
/// resident loader and a streamed source over the SAME `(total, fraction)`
/// can never disagree about where the boundary falls. Pinned for every
/// `total ∈ 0..=1000` and every fraction the config admits by
/// `source::split_index_matches_the_resident_split_boundary` in this
/// crate's test suite.
pub(crate) fn split_index(total: usize, fraction: f64) -> usize {
    let val_count = (total as f64 * fraction).round() as usize;
    total - val_count
}

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
    /// Classification: the trainable head's LOGITS (never the pre-head
    /// embeddings — the cross-rank gather point for this arm must
    /// sit downstream of the classification head, so the head is applied at
    /// BATCH-CONSTRUCTION time, in `TrainingLoop::encode_chunk`, rather than
    /// inside the loss dispatch) + integer class labels.
    Classification {
        logits: Tensor,
        labels: Tensor, // shape (batch_size,) u32
    },
    /// Regression: the distributional head's raw output plus the observed
    /// continuous targets. `input` is `(batch, k)` — the unconstrained head
    /// parameters (`k = 2` `(mean, raw_std)` for the Gaussian objectives,
    /// `k = levels` for the pinball objective); `target` is `(batch,)` the
    /// observed `y`. The proper-scoring loss reads a positive `σ` from
    /// `raw_std` via `floor + softplus`, so the head trains in the
    /// unconstrained space.
    Regression { input: Tensor, target: Tensor },
}

/// Format of training data, detected from column names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// the bytes: it is the job's own [`jammi_datafusion::ModelTask`]
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
    /// Classification: `text, label` rows, each label an index into the
    /// loader's vocabulary of class names.
    Classification,
    /// Regression: `text, target` rows — one input text and one observed
    /// continuous outcome. The trainer encodes the text through the frozen base
    /// model + the distributional projection head, then scores the head's
    /// parameters against the target with the configured proper-scoring
    /// objective.
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
    "regression",
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
    /// A mined hard negative is a third projected column ([`Self::in_batch`]
    /// gives `Triplet`, not `Pairs`) and therefore different committed bytes,
    /// so it takes a distinct tag.
    pub fn format_tag(self) -> &'static str {
        match self {
            TrainingFormat::Contrastive => "contrastive",
            TrainingFormat::Pairs => "pairs",
            TrainingFormat::Triplet => "triplet",
            TrainingFormat::MediaTriplet => "media_triplet",
            TrainingFormat::Classification => "classification",
            TrainingFormat::Regression => "regression",
        }
    }

    /// The inverse of [`Self::format_tag`] over [`TRAINING_FORMAT_TAGS`]: the
    /// format a recorded tag names.
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
            "classification" => Some(TrainingFormat::Classification),
            "regression" => Some(TrainingFormat::Regression),
            _ => None,
        }
    }
}

impl TrainingFormat {
    /// The format of `(anchor, positive[, negative])` rows: `Triplet` when each
    /// row carries a mined hard negative, `Pairs` (in-batch negatives only)
    /// when it does not. Where the rows came from — a projection, a graph
    /// sample — is the training set's descriptor's to record, not the format's.
    pub fn in_batch(has_hard_negatives: bool) -> Self {
        if has_hard_negatives {
            TrainingFormat::Triplet
        } else {
            TrainingFormat::Pairs
        }
    }
}

/// Training rows of one format, column-major: a training batch, or a
/// loader's whole row set — the same thing at a larger scale, so a batch is a
/// [`Self::select`] of the loader's rows. The training loop encodes a batch
/// through the base model before computing loss.
#[derive(Debug, Clone, PartialEq)]
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
    /// inside it, for an adapted-encoder run) or the frozen tower plus the
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
    /// batch), not a malformed chunk: [`Self::select`] of an empty range is
    /// one.
    pub fn row_count(&self) -> usize {
        match self {
            TextChunk::Contrastive { texts_a, .. } => texts_a.len(),
            TextChunk::Pairs { anchors, .. } => anchors.len(),
            TextChunk::Triplet { anchors, .. } => anchors.len(),
            TextChunk::MediaTriplet { anchors, .. } => anchors.len(),
            TextChunk::Classification { texts, .. } => texts.len(),
            TextChunk::Regression { texts, .. } => texts.len(),
        }
    }

    /// The rows in `rows`, as a chunk of the same format. `rows` lies within
    /// `0..self.row_count()`; an empty range is a well-formed empty chunk.
    pub fn select(&self, rows: std::ops::Range<usize>) -> TextChunk {
        fn take<T: Clone>(column: &[T], rows: &std::ops::Range<usize>) -> Vec<T> {
            column[rows.clone()].to_vec()
        }
        match self {
            TextChunk::Contrastive {
                texts_a,
                texts_b,
                scores,
            } => TextChunk::Contrastive {
                texts_a: take(texts_a, &rows),
                texts_b: take(texts_b, &rows),
                scores: take(scores, &rows),
            },
            TextChunk::Pairs { anchors, positives } => TextChunk::Pairs {
                anchors: take(anchors, &rows),
                positives: take(positives, &rows),
            },
            TextChunk::Triplet {
                anchors,
                positives,
                negatives,
            } => TextChunk::Triplet {
                anchors: take(anchors, &rows),
                positives: take(positives, &rows),
                negatives: take(negatives, &rows),
            },
            TextChunk::MediaTriplet {
                anchors,
                positives,
                negatives,
            } => TextChunk::MediaTriplet {
                anchors: take(anchors, &rows),
                positives: take(positives, &rows),
                negatives: take(negatives, &rows),
            },
            TextChunk::Classification { texts, labels } => TextChunk::Classification {
                texts: take(texts, &rows),
                labels: take(labels, &rows),
            },
            TextChunk::Regression { texts, targets } => TextChunk::Regression {
                texts: take(texts, &rows),
                targets: take(targets, &rows),
            },
        }
    }

    /// Split off the rows from `at` on, moving them — never cloning — into
    /// the returned chunk; `self` keeps the rows before `at`.
    pub fn split_off(&mut self, at: usize) -> TextChunk {
        match self {
            TextChunk::Contrastive {
                texts_a,
                texts_b,
                scores,
            } => TextChunk::Contrastive {
                texts_a: texts_a.split_off(at),
                texts_b: texts_b.split_off(at),
                scores: scores.split_off(at),
            },
            TextChunk::Pairs { anchors, positives } => TextChunk::Pairs {
                anchors: anchors.split_off(at),
                positives: positives.split_off(at),
            },
            TextChunk::Triplet {
                anchors,
                positives,
                negatives,
            } => TextChunk::Triplet {
                anchors: anchors.split_off(at),
                positives: positives.split_off(at),
                negatives: negatives.split_off(at),
            },
            TextChunk::MediaTriplet {
                anchors,
                positives,
                negatives,
            } => TextChunk::MediaTriplet {
                anchors: anchors.split_off(at),
                positives: positives.split_off(at),
                negatives: negatives.split_off(at),
            },
            TextChunk::Classification { texts, labels } => TextChunk::Classification {
                texts: texts.split_off(at),
                labels: labels.split_off(at),
            },
            TextChunk::Regression { texts, targets } => TextChunk::Regression {
                texts: texts.split_off(at),
                targets: targets.split_off(at),
            },
        }
    }
}

/// The flattened in-batch-negative view of a text loader: `(anchors,
/// positives, optional explicit negatives)`. Consumed by GradCache and
/// hard-negative mining, which treat the dataset as one in-batch-negative batch.
pub type InBatchNegativeTexts = (Vec<String>, Vec<String>, Option<Vec<String>>);

/// Internal storage: the rows already resident, as one chunk of their format
/// (from source, or the tests-only synthetic constructors), or precomputed
/// batches (tests only).
enum LoaderData {
    Rows(TextChunk),
    Precomputed(Vec<TrainingBatch>),
}

/// Loads training data and produces batches of tensors.
///
/// Two construction modes:
/// - `from_contrastive` / `from_triplets` / `from_rows`: stores text.
///   Use `text_chunks()` for model-in-loop training.
/// - `from_precomputed`: stores pre-built batches, `batches()` returns them as-is.
///   Used in tests to exercise divergence detection, early stopping, etc.
pub struct TrainingDataLoader {
    data: LoaderData,
    /// A classification loader's label vocabulary, `None` for every other
    /// format: the names of the classes its rows' indices count, shared
    /// read-only by both halves of a [`Self::split`] and every replica.
    label_vocab: Option<std::sync::Arc<super::decode::LabelVocabulary>>,
    /// The eager collected batches' pool reservation, held for this loader's
    /// own lifetime — `None` for every loader that never
    /// went through a pool-accounted collect (every `from_*` constructor
    /// below builds one this way; `training_set::read_back_with_reservation`'s
    /// caller attaches the real one via [`Self::with_reservation`]).
    /// `MemoryReservation`'s own `Drop` frees its held bytes back to the pool
    /// the instant the LAST reservation over it goes out of scope — no
    /// manual `Drop` impl is needed on this type for the same reason
    /// [`super::stream::OwnedChunk`] needs none.
    reservation: Option<datafusion::execution::memory_pool::MemoryReservation>,
}

impl TrainingDataLoader {
    /// A loader over `rows`, with no pool reservation and, unless a
    /// classification constructor sets one, no label vocabulary.
    fn of_rows(rows: TextChunk) -> Self {
        Self {
            data: LoaderData::Rows(rows),
            label_vocab: None,
            reservation: None,
        }
    }

    /// A second loader over the SAME rows (or precomputed batches), for
    /// another rank of an in-process gang: the rows and the vocabulary are
    /// shared, the pool reservation is NOT — it stays with the loader the
    /// eager read attached it to (rank 0's), and is freed when that one
    /// drops. A rank's loader is read-only over resident rows, so the copies
    /// see identical rows in identical order, which is what every rank's
    /// partition slicing assumes (`partition.rs`).
    pub(crate) fn replicate(&self) -> Self {
        Self {
            data: match &self.data {
                LoaderData::Rows(rows) => LoaderData::Rows(rows.clone()),
                LoaderData::Precomputed(batches) => LoaderData::Precomputed(batches.clone()),
            },
            label_vocab: self.label_vocab.clone(),
            reservation: None,
        }
    }

    /// Create a loader from contrastive pair rows.
    pub fn from_contrastive(rows: Vec<(String, String, f32)>) -> Self {
        let mut texts_a = Vec::with_capacity(rows.len());
        let mut texts_b = Vec::with_capacity(rows.len());
        let mut scores = Vec::with_capacity(rows.len());
        for (a, b, s) in rows {
            texts_a.push(a);
            texts_b.push(b);
            scores.push(s);
        }
        Self::of_rows(TextChunk::Contrastive {
            texts_a,
            texts_b,
            scores,
        })
    }

    /// Create a loader from classification rows (text + class index into
    /// `vocab`).
    pub fn from_classification(
        rows: Vec<(String, u32)>,
        vocab: super::decode::LabelVocabulary,
    ) -> Self {
        let (texts, labels) = rows.into_iter().unzip();
        Self {
            label_vocab: Some(std::sync::Arc::new(vocab)),
            ..Self::of_rows(TextChunk::Classification { texts, labels })
        }
    }

    /// A classification loader's class names in class-index order; `None`
    /// for every other format.
    pub fn class_labels(&self) -> Option<&[String]> {
        self.label_vocab
            .as_deref()
            .map(super::decode::LabelVocabulary::labels)
    }

    /// Create a loader from regression rows (input text + observed continuous
    /// target). The trainer encodes each text through the base model and the
    /// regression head, then scores the head's parameters against the target
    /// with the configured proper-scoring objective.
    pub fn from_regression(rows: Vec<(String, f32)>) -> Self {
        let (texts, targets) = rows.into_iter().unzip();
        Self::of_rows(TextChunk::Regression { texts, targets })
    }

    /// Create a loader from triplet rows.
    pub fn from_triplets(rows: Vec<(String, String, String)>) -> Self {
        let mut anchors = Vec::with_capacity(rows.len());
        let mut positives = Vec::with_capacity(rows.len());
        let mut negatives = Vec::with_capacity(rows.len());
        for (a, p, n) in rows {
            anchors.push(a);
            positives.push(p);
            negatives.push(n);
        }
        Self::of_rows(TextChunk::Triplet {
            anchors,
            positives,
            negatives,
        })
    }

    /// Create a loader from contrastive pair rows `(anchor, positive)`. The
    /// `MultipleNegativesRanking` objective draws negatives from the rest of
    /// each batch, so no explicit negative column is needed.
    pub fn from_pairs(rows: Vec<(String, String)>) -> Self {
        let (anchors, positives) = rows.into_iter().unzip();
        Self::of_rows(TextChunk::Pairs { anchors, positives })
    }

    /// Create a loader from a graph: sample the node-text + edge-table
    /// graph into `(anchor, positive, [hard_negative])` text rows by biased
    /// random walks, then store them as the underlying `Pairs` (no mined
    /// negatives) or `Triplet` (structure-mined hard negatives) rows
    /// ([`TrainingFormat::in_batch`]). The rows feed the existing MNRL/Triplet
    /// path unchanged — a graph sample adds **no new loss**; that the rows came
    /// from a graph is recorded by the training set's descriptor.
    ///
    /// The sampler enforces the text-bearing precondition (every edge endpoint
    /// must resolve to a [`super::graph_sampler::TextNode`]) and the collapse /
    /// false-negative guards, so any violation surfaces here as a typed error.
    ///
    /// **Circularity caveat:** if the edges are similarity edges the
    /// supervision largely re-learns the base metric; genuine gain comes from
    /// declared / external edges (see [`super::graph_sampler`]).
    ///
    /// Has no production caller (a job trains from the materialised table,
    /// decoded like any pairs/triplet training set, never a
    /// re-sample-in-place); kept as a direct, table-free constructor over an
    /// ALREADY-BUILT `GraphSampler` for other callers (tests, and any future
    /// caller with no table to materialise). A caller comparing the result
    /// against a real job's output over the same node/edge SET must sort
    /// its nodes/edges with
    /// [`super::graph_sampler::sort_into_graph_read_order`] before calling
    /// [`super::graph_sampler::GraphSampler::build`] — this function samples
    /// an already-built sampler and cannot apply that rule itself.
    pub fn from_graph(sampler: &super::graph_sampler::GraphSampler) -> Result<Self> {
        sampler.config().validate_for_training()?;
        let pairs = sampler.sample()?;
        // The format is decided from the CONFIG (`hard_negatives > 0`), never
        // re-derived from which negatives the FIRST sampled pair happens to
        // carry — a misleading representative row must never decide the
        // whole set's shape. `GraphSampler::sample`'s own empty-negative-pool
        // refusal guarantees every pair below carries at least one negative
        // whenever `has_negatives` is true, so the `ok_or_else` below is
        // unreachable in practice — a typed refusal, never a panic, should
        // that invariant ever break.
        if sampler.config().hard_negatives == 0 {
            return Ok(Self::from_pairs(
                pairs.into_iter().map(|p| (p.anchor, p.positive)).collect(),
            ));
        }
        let rows = pairs
            .into_iter()
            .map(|p| {
                // The pair's one mined negative is the explicit triplet
                // negative (MNRL appends it as an extra column beside the
                // in-batch negatives).
                let negative = p.hard_negatives.into_iter().next().ok_or_else(|| {
                    JammiError::Internal(
                        "graph pair declared hard negatives but supplied none".into(),
                    )
                })?;
                Ok((p.anchor, p.positive, negative))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_triplets(rows))
    }

    /// Create a loader from MEDIA triplet rows. Each element is
    /// `(anchor_bytes, positive_bytes, negative_bytes)` where every field is
    /// one encoded blob — an audio clip or an image, per the job's task (see
    /// [`TrainingFormat::MediaTriplet`] for why the task, not the bytes, is
    /// the modality authority). The trainer encodes these through that
    /// modality's front end and the base model; the contrastive objective is
    /// identical to the text triplet path.
    pub fn from_media_triplets(rows: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>) -> Self {
        let mut anchors = Vec::with_capacity(rows.len());
        let mut positives = Vec::with_capacity(rows.len());
        let mut negatives = Vec::with_capacity(rows.len());
        for (a, p, n) in rows {
            anchors.push(a);
            positives.push(p);
            negatives.push(n);
        }
        Self::of_rows(TextChunk::MediaTriplet {
            anchors,
            positives,
            negatives,
        })
    }

    /// Create a loader with N synthetic rows (for testing validation split logic).
    pub fn from_rows(count: usize) -> Self {
        Self::from_contrastive(
            (0..count)
                .map(|i| (format!("text_a_{i}"), format!("text_b_{i}"), 0.5))
                .collect(),
        )
    }

    /// Create a loader from pre-built batches. `batches()` returns clones of
    /// these directly instead of generating placeholder tensors. Used in tests
    /// to supply crafted tensors that trigger specific training behaviors
    /// (divergence, early stopping, etc.).
    pub fn from_precomputed(batches: Vec<TrainingBatch>) -> Self {
        Self {
            data: LoaderData::Precomputed(batches),
            label_vocab: None,
            reservation: None,
        }
    }

    /// Attach the eager collected read's pool reservation to this loader,
    /// moving ownership in: the bytes it reserved release only when this
    /// loader (or whichever half of a [`Self::split`] carries it) drops.
    /// `pub(crate)`: only `worker.rs`'s Resident
    /// construction site calls this — every `from_*` constructor above
    /// stays reservation-free by design.
    pub(crate) fn with_reservation(
        mut self,
        reservation: datafusion::execution::memory_pool::MemoryReservation,
    ) -> Self {
        self.reservation = Some(reservation);
        self
    }

    /// Total number of data points (rows for text, batches for precomputed).
    pub fn len(&self) -> usize {
        match &self.data {
            LoaderData::Rows(rows) => rows.row_count(),
            LoaderData::Precomputed(batches) => batches.len(),
        }
    }

    /// Whether the loader has no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of batches at the given batch size.
    pub fn num_batches(&self, batch_size: usize) -> usize {
        match &self.data {
            LoaderData::Rows(_) if batch_size == 0 => 0,
            LoaderData::Rows(rows) => rows.row_count().div_ceil(batch_size),
            LoaderData::Precomputed(batches) => batches.len(),
        }
    }

    /// Deterministic split: last `fraction` of data goes to validation.
    ///
    /// The train/validation boundary itself is `split_index` — the SAME
    /// arithmetic a [`super::source::StreamedSet`] uses to derive its own
    /// `train_count` from a row COUNT alone, so a resident
    /// loader's split and a streamed source's window never disagree about
    /// where the boundary falls for the same `(total, fraction)`.
    ///
    /// **The eager reservation moves to the TRAIN half**: `self`'s ENTIRE currently-held
    /// reservation is carved, by `MemoryReservation::split`, into a fresh reservation the returned
    /// train loader owns — so the pool accounting follows the loader that
    /// actually stays resident through every epoch, never the transient
    /// pre-split original or the validation half this run reads only
    /// occasionally.
    ///
    /// **Consumes `self` and moves rows, never clones them.** `self`'s rows
    /// are truncated in place ([`TextChunk::split_off`]) — the validation
    /// half's rows are MOVED out (no row is ever cloned by this call). Taking `self` by value also matters for accounting:
    /// `MemoryReservation::split` drains atomically, so `self`'s reservation
    /// sits at size zero after a first call, and a second `split` on the
    /// SAME loader would hand the new "train" half a reservation carrying
    /// zero bytes silently — a correct-looking loader whose pool accounting
    /// has gone stale. A second `split` on a moved loader is a COMPILE error
    /// (the moved-value diagnostic the doctest below pins).
    ///
    /// ```compile_fail,E0382
    /// use jammi_ai::fine_tune::data::TrainingDataLoader;
    ///
    /// let loader = TrainingDataLoader::from_rows(4);
    /// let (train, _val) = loader.split(0.25);
    /// // `loader` was moved into the call above; a second `split` on it
    /// // cannot compile.
    /// let (_train2, _val2) = loader.split(0.25);
    /// # let _ = train;
    /// ```
    pub fn split(self, fraction: f64) -> (TrainingDataLoader, TrainingDataLoader) {
        let train_reservation = self.reservation.as_ref().map(|r| r.split(r.size()));
        let (train, val) = match self.data {
            LoaderData::Rows(mut rows) => {
                let val_rows = rows.split_off(split_index(rows.row_count(), fraction));
                (LoaderData::Rows(rows), LoaderData::Rows(val_rows))
            }
            LoaderData::Precomputed(mut batches) => {
                let val_batches = batches.split_off(split_index(batches.len(), fraction));
                (
                    LoaderData::Precomputed(batches),
                    LoaderData::Precomputed(val_batches),
                )
            }
        };
        // Both halves index one vocabulary: a validation row's class index
        // means what a training row's does.
        (
            TrainingDataLoader {
                data: train,
                label_vocab: self.label_vocab.clone(),
                reservation: train_reservation,
            },
            TrainingDataLoader {
                data: val,
                label_vocab: self.label_vocab,
                reservation: None,
            },
        )
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
            LoaderData::Rows(_) => Err(JammiError::Internal(
                "a row loader feeds model-in-loop encoding through text_chunks(), not batches()"
                    .into(),
            )),
            LoaderData::Precomputed(batches) => Ok(batches.iter().map(|b| Ok(b.clone())).collect()),
        }
    }

    /// The rows `batch_size` at a time, as chunks for model-in-loop training.
    /// Empty for a precomputed loader.
    pub fn text_chunks(&self, batch_size: usize) -> Vec<TextChunk> {
        match &self.data {
            LoaderData::Rows(rows) => {
                let total = rows.row_count();
                (0..total)
                    .step_by(batch_size.max(1))
                    .map(|start| rows.select(start..(start + batch_size).min(total)))
                    .collect()
            }
            LoaderData::Precomputed(_) => Vec::new(),
        }
    }

    /// The [`TextChunk`] rank `spec.rank` of `spec.world` holds for global
    /// step `step`, over THIS loader's own row count as the train prefix
    /// (partition rule v1 — [`super::partition::PartitionSpec::
    /// rows_for_step`] computes the slice). A rank whose slice is empty (a
    /// zero-row rank at the trailing global batch) yields a well-formed
    /// empty [`TextChunk`], never an out-of-bounds panic — `rows_for_step`
    /// never returns a range past the row count.
    ///
    /// Row loaders only (`Precomputed` has no row-level partition).
    pub fn text_chunk_for_rank(
        &self,
        spec: &super::partition::PartitionSpec,
        step: usize,
    ) -> Result<TextChunk> {
        match &self.data {
            LoaderData::Rows(rows) => Ok(rows.select(spec.rows_for_step(rows.row_count(), step))),
            LoaderData::Precomputed(_) => Err(JammiError::Internal(
                "a precomputed loader has no row-level partition".into(),
            )),
        }
    }

    /// The format of this loader's rows; `None` for precomputed batches,
    /// which carry their own batch kinds.
    pub fn format(&self) -> Option<TrainingFormat> {
        let LoaderData::Rows(rows) = &self.data else {
            return None;
        };
        Some(match rows {
            TextChunk::Contrastive { .. } => TrainingFormat::Contrastive,
            TextChunk::Pairs { .. } => TrainingFormat::Pairs,
            TextChunk::Triplet { .. } => TrainingFormat::Triplet,
            TextChunk::MediaTriplet { .. } => TrainingFormat::MediaTriplet,
            TextChunk::Classification { .. } => TrainingFormat::Classification,
            TextChunk::Regression { .. } => TrainingFormat::Regression,
        })
    }

    /// Every regression target in this loader, in row order — the whole-dataset
    /// view the trainer reduces into a fixed target scaler once before the
    /// loop. `None` for any non-regression loader (no targets to
    /// standardise) and for the precomputed test path (which supplies
    /// head/target tensors directly, not text rows).
    pub fn regression_targets(&self) -> Option<Vec<f32>> {
        match &self.data {
            LoaderData::Rows(TextChunk::Regression { targets, .. }) => Some(targets.clone()),
            _ => None,
        }
    }

    /// Flatten the in-batch-negative text rows into `(anchors, positives,
    /// negatives)`, the whole-dataset view GradCache and hard-negative mining
    /// consume. `negatives` is `Some` for a `Triplet` loader (explicit hard
    /// negatives) and `None` for a `Pairs` loader. Returns an error for any
    /// other format — only in-batch-negative training has this shape.
    pub fn in_batch_negative_texts(&self) -> Result<InBatchNegativeTexts> {
        match &self.data {
            LoaderData::Rows(TextChunk::Pairs { anchors, positives }) => {
                Ok((anchors.clone(), positives.clone(), None))
            }
            LoaderData::Rows(TextChunk::Triplet {
                anchors,
                positives,
                negatives,
            }) => Ok((anchors.clone(), positives.clone(), Some(negatives.clone()))),
            LoaderData::Rows(_) => Err(JammiError::InvalidArgument {
                argument: "cached".into(),
                reason: format!(
                    "GradCache applies only to in-batch-negative formats (pairs/triplet), not {:?}",
                    self.format()
                ),
            }),
            LoaderData::Precomputed(_) => Err(JammiError::Internal(
                "GradCache requires text rows, not precomputed batches".into(),
            )),
        }
    }

    /// Whether this loader was constructed from pre-built tensor batches
    /// (typically a test fixture) rather than text rows that must be
    /// encoded through a model.
    pub fn is_precomputed(&self) -> bool {
        matches!(self.data, LoaderData::Precomputed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Precomputed` loader has no row-level partition, so
    /// `text_chunk_for_rank` refuses it outright
    /// rather than attempting to slice pre-built tensor batches by row. Dies
    /// if the refusal is removed (the `LoaderData::Precomputed(_) => Err(...)`
    /// arm at `text_chunk_for_rank`'s match) — the call would then need to
    /// fall through to some OTHER arm, which does not compile for an empty
    /// `Vec<TrainingBatch>` fed in here without inventing a bogus row slice.
    #[test]
    fn precomputed_loader_refuses_text_chunk_for_rank() {
        use super::super::partition::{PartitionRule, PartitionSpec};
        let loader = TrainingDataLoader::from_precomputed(Vec::new());
        let spec = PartitionSpec::for_test(0, 1, 4, PartitionRule::BlockByGlobalBatch);
        match loader.text_chunk_for_rank(&spec, 0) {
            Err(e) => assert!(
                e.to_string().contains("no row-level partition"),
                "expected the typed 'no row-level partition' refusal, got: {e}"
            ),
            Ok(_) => panic!("a Precomputed loader must refuse text_chunk_for_rank"),
        }
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
            TrainingFormat::Classification,
            TrainingFormat::Regression,
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
            TrainingFormat::Classification => 4,
            TrainingFormat::Regression => 5,
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

    /// The byte-affecting shape a mined negative adds is part of the tag.
    #[test]
    fn the_tag_keeps_the_shape_a_mined_negative_adds() {
        assert_ne!(
            TrainingFormat::in_batch(false).format_tag(),
            TrainingFormat::in_batch(true).format_tag(),
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
    /// trainer encodes through the distributional head. Pins the regression data path
    /// from constructor to chunk.
    #[test]
    fn regression_loader_chunks_into_regression_text_chunks() {
        let loader = TrainingDataLoader::from_regression(vec![
            ("cheap".into(), 0.1),
            ("mid".into(), 0.5),
            ("dear".into(), 0.9),
        ]);
        assert_eq!(loader.format(), Some(TrainingFormat::Regression));
        assert_eq!(loader.len(), 3);

        let chunks = loader.text_chunks(2);
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
        let (train, val) = loader.split(0.2);
        assert_eq!(train.format(), Some(TrainingFormat::Regression));
        assert_eq!(val.format(), Some(TrainingFormat::Regression));
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    /// For W ∈ {1, 2, 4}, the MULTISET of rows over all ranks
    /// at each global step equals the W=1 batch at that step, on a
    /// `train_count` (7) that is NOT a multiple of `W·B` for any tested W —
    /// asserting a zero-row rank actually occurs for W=2 and W=4.
    ///
    /// The fixture is chosen so W=2 hits its zero-row rank at
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
            let w1_ref = PartitionSpec::for_test(
                0,
                1,
                per_rank_batch * world,
                PartitionRule::BlockByGlobalBatch,
            );
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
                    let spec = PartitionSpec::for_test(
                        rank,
                        world,
                        per_rank_batch,
                        PartitionRule::BlockByGlobalBatch,
                    );
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

    /// Complements the multiset test above: shrinking a rank's batch to a size the fixture cannot
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
                let spec =
                    PartitionSpec::for_test(rank, world, 3, PartitionRule::BlockByGlobalBatch);
                // Step 10 is far past any row this 5-row fixture could ever
                // reach at batch 3 for any tested world.
                let chunk = loader.text_chunk_for_rank(&spec, 10).unwrap();
                match &chunk {
                    TextChunk::Pairs { anchors, positives } => {
                        assert!(anchors.is_empty() && positives.is_empty());
                    }
                    _ => panic!("expected a Pairs chunk"),
                }
            }
        }
    }
}
