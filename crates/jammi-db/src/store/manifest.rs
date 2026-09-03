//! The materialization contract: the verifiable identity every result table
//! carries so a later reader can assert **"this artifact is the output of
//! definition D over input-state S"** — without trusting a name, a path, or an
//! out-of-band convention.
//!
//! A result table is published as an immutable Parquet object (plus, for
//! embedding tables, an ANN-index sidecar bundle). This module adds a *separate*
//! `.materialization.json` sidecar — written for **every** result table, not
//! only embedding tables — carrying an in-toto-shaped attestation that binds
//! three things to the artifact's content digest:
//!
//! 1. a **definition hash** of *how* the table was produced — a canonical
//!    encoding of the [`ProducingDescriptor`] (the verb plus its typed
//!    parameters) together with the [`MaterializationEnv`] that affects its
//!    output (engine version, invoked-model identities, input backend kinds,
//!    **and the compute device**);
//! 2. the **immutable as-of anchors** of every input the producer read
//!    ([`InputAnchor`]); and
//! 3. the **producing-run identity and instant**.
//!
//! It backs one verb, `verify_materialization`, that recomputes the artifact
//! digest and reports a [`MatchVerdict`]. The engine ships the contract and the
//! verify primitive; it ships **no** policy — what a reader *does* with a
//! mismatch (refuse, alarm, fall back) is the reader's concern.
//!
//! # The two sidecars are distinct
//!
//! An embedding table's ANN bundle includes a `.manifest.json` sibling (the
//! USearch index's `version / dimensions / count / backend`; see
//! [`crate::storage::sidecar_layout`]). The materialization attestation is a
//! **different** file, `.materialization.json`, and concerns the *data*, never
//! the index. The two never collide: `.manifest.json` describes the search
//! accelerator; `.materialization.json` attests the Parquet data-of-record.
//!
//! # `compute` takes a producing *description*, not a plan
//!
//! Result-table producers in this engine are hand-built physical pipelines —
//! there is no single `LogicalPlan` to canonicalise (the SQL lane's
//! `LogicalPlan` is unrelated). So the definition hash is computed over a
//! [`ProducingDescriptor`]: a typed, deterministically-serialisable description
//! of the verb and its parameters that each producer fills in. Two runs of the
//! same producer with the same parameters over the same input anchors in the
//! same environment hash identically; any output-affecting change to the
//! description, the environment (including the compute device), or the inputs
//! changes the hash.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model_task::ModelTask;

// Re-exported (not merely imported) so a consumer that constructs or matches a
// `ModelIdentity` — every model-producing descriptor's environment carries a
// `Vec<ModelIdentity>` — reaches `ComputePrecision`'s type from this module,
// without its own direct `jammi-numerics` import.
pub use jammi_numerics::ComputePrecision;

/// Manifest format version. A change to the [`ProducingDescriptor`] shape — the
/// determinant set a producer folds into its [`DefinitionHash`] *and* records
/// verbatim for replay — bumps this so a reader detects an incompatible older
/// manifest as a typed [`ManifestError::UnsupportedManifestVersion`] rather than
/// comparing a stale hash computed over a different determinant set, or replaying
/// a recorded descriptor whose shape this build no longer understands. The
/// version is the *authoritative* signal that an on-disk manifest was written
/// under a different determinant set and is therefore incomparable/un-replayable
/// — but not the only line of defence: a genuinely old (pre-contract) manifest
/// whose JSON predates a since-added field is also rejected serde-first as a
/// typed [`ManifestError`] before the version is even read. Either rejection is
/// clean (the signal to re-emit); a reader must never silently trust a
/// version-mismatched or shape-mismatched manifest.
pub const MANIFEST_VERSION: u32 = 3;

/// Content hash of *how* a table was produced: a canonical encoding of the
/// [`ProducingDescriptor`] plus the [`MaterializationEnv`] that affects its
/// output. SHA-256, hex-encoded.
///
/// "Environment" is deliberately broad — the engine semantic version, the
/// identity of every model the producer invokes, the backend kinds of the
/// inputs, **and the compute device**. A hash that omits part of the execution
/// environment yields false "matches" when that hidden part changes (the Bazel
/// cross-compiler lesson): a model run on CPU vs CUDA yields different float
/// outputs but the same model identity, so the device is part of the world the
/// hash must cover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DefinitionHash(pub String);

impl DefinitionHash {
    /// The hex digest as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DefinitionHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Content digest of the materialised artifact itself: SHA-256 over the Parquet
/// object's bytes, hex-encoded. This is the in-toto "subject" — the thing a
/// verifier matches by digest, treating the subject as immutable.
///
/// The digest covers the Parquet **data**, not the ANN index sidecar: the index
/// is a derived accelerator reconstructible from the data, so a [`MatchVerdict`]
/// attests the data-of-record, not the search structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArtifactDigest(pub String);

impl ArtifactDigest {
    /// Compute the digest over the artifact's bytes.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(hex::encode(Sha256::digest(bytes)))
    }

    /// The hex digest as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ArtifactDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The compute device a model ran on. Part of [`MaterializationEnv`] because a
/// model produces different float outputs on CPU vs an accelerator while
/// carrying the same model identity — so the device is a determinant of the
/// output the definition hash must cover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDevice {
    /// CPU.
    Cpu,
    /// A CUDA device at the given ordinal.
    Cuda {
        /// CUDA device ordinal.
        ordinal: u32,
    },
    /// An Apple Metal device at the given ordinal.
    Metal {
        /// Metal device ordinal.
        ordinal: u32,
    },
}

/// The execution environment that affects a producer's output, hashed into the
/// [`DefinitionHash`] alongside the [`ProducingDescriptor`].
///
/// Carries the engine semantic version, the compute device, and the identities,
/// backend kinds, and compute precisions of every model the producer invokes —
/// the determinants of the output that are *not* part of the producing
/// description itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationEnv {
    /// Engine semantic version that produced the artifact (`CARGO_PKG_VERSION`).
    pub engine_version: String,
    /// The compute device the producer's model(s) ran on.
    pub device: ComputeDevice,
    /// The identity + backend kind of every model the producer invoked, in a
    /// stable order. Empty for a producer that invokes no model (e.g. a
    /// neighbor-graph derivation, a propagation kernel).
    pub models: Vec<ModelIdentity>,
}

impl MaterializationEnv {
    /// Build the environment for the current engine version and the given
    /// device + invoked models.
    pub fn new(device: ComputeDevice, models: Vec<ModelIdentity>) -> Self {
        Self {
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            device,
            models,
        }
    }
}

/// The identity + backend kind + compute precision + content digest of a
/// model an environment invoked. The canonical model id (HF repo or local
/// path string) plus the backend kind that ran it, the dtype it ran at, and a
/// digest of the model's on-disk content.
///
/// `compute_precision` folds in here — the same place `backend` does — rather
/// than into a single per-descriptor field, so it enters the definition hash
/// **uniformly for every model-producing descriptor** (`Inference` and
/// `Embedding` alike) the moment either records a `ModelIdentity`, instead of
/// requiring each new model-invoking `ProducingDescriptor` variant to
/// remember its own precision field. An `F16` run of a model is output-
/// affecting relative to an `F32` run of the same model over the same input —
/// two such runs must never collide on one materialization identity.
///
/// `content_digest` folds in the SAME way, for the SAME reason (esc-057):
/// pooling strategy (`1_Pooling/config.json`), tokenizer files, and model
/// weights are all output-affecting relative to the bare `model_id` string —
/// two directories that share one HF repo id but differ in any of those bytes
/// must never collide on one `DefinitionHash`. Folding one combined
/// [`ModelContentDigest`] here (rather than a bespoke pooling/tokenizer/
/// weights field on `ProducingDescriptor::Embedding`) keeps the determinant
/// uniform across every model-producing variant, exactly like
/// `compute_precision` above — `Inference` would otherwise collide on it
/// identically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Canonical model id as stored in `result_tables.model_id`.
    pub model_id: String,
    /// The backend kind that ran the model (`candle` / `ort` / `http`).
    pub backend: String,
    /// The compute precision the model ran at (the resolved per-model
    /// `config.json` override, or the global `GpuConfig::compute_precision`
    /// default).
    pub compute_precision: ComputePrecision,
    /// A digest of the model's on-disk content (config + pooling config +
    /// tokenizer files + weights), or the typed reason none could be
    /// computed. See [`ModelContentDigest`] for why this is not a bare
    /// `Option<String>`.
    pub content_digest: ModelContentDigest,
    /// The GGUF/k-quant weight-storage format the model's weights were
    /// loaded in, or `None` when the model ran unquantized (a dense
    /// `f32`/`f16`/`bf16` weight tensor — the `compute_precision` field above
    /// already names that case).
    ///
    /// `quantization` folds in here — the same place `compute_precision` and
    /// `content_digest` do, for the same reason (see their doc comments
    /// above): a `WeightQuantization` is a determinant of the weight BYTES a
    /// model loaded, so it must enter the definition hash **uniformly for
    /// every model-producing descriptor** the moment either records a
    /// `ModelIdentity`, rather than requiring each new model-invoking
    /// `ProducingDescriptor` variant to remember its own quantization field.
    /// A `Q4K`-quantized run of a model is output-affecting relative to a
    /// full-precision run of the same model over the same inputs — two such
    /// runs must never collide on one materialization identity.
    ///
    /// `#[serde(skip_serializing_if = "Option::is_none")]` means an absent
    /// key (a pre-feature row, or any row for a model that ran unquantized)
    /// serialises to identical canonical bytes as before this field existed
    /// — so every `DefinitionHash` recorded before this field was added is
    /// preserved exactly; only a `Some` quantization changes the hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<jammi_numerics::WeightQuantization>,
}

/// A model's content digest — the [`ModelIdentity`] determinant that closes
/// esc-057: a `model_id` string alone does not change when the referenced
/// directory's `1_Pooling/config.json`, tokenizer files, or weights bytes
/// change, so two genuinely different models could otherwise collide on one
/// [`DefinitionHash`]. A loader computes this once per model load — SHA-256
/// over the model's config, `1_Pooling/config.json`, tokenizer files, and
/// weights bytes — and threads it into every `ModelIdentity` it builds.
///
/// Deliberately **not** a bare `Option<String>`: an external-producer import
/// (`ProducingDescriptor::External`-adjacent rows built by
/// `pipeline::import`) has no local model directory to hash, and that
/// "no digest" state must say WHY, typed, so a reader can tell "genuinely no
/// content to hash" from "the loader forgot to compute one" — the same
/// question a bare `Option::None` can never answer once constructed. This is
/// the digest's complete presence lattice, expressed as one closed sum type
/// rather than an `Option<T>` field paired with a second, independently
/// omittable reason field (a "None carries a typed reason, never a silent
/// default" contract that a companion field could silently violate by being
/// left `None` itself).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
pub enum ModelContentDigest {
    /// SHA-256 hex digest of the model's on-disk content, computed once per
    /// model load.
    Sha256(String),
    /// No digest could be computed for this model invocation, with the
    /// typed reason why.
    Unavailable(ModelContentDigestUnavailableReason),
}

/// Why a [`ModelContentDigest`] is [`ModelContentDigest::Unavailable`] for a
/// given model invocation. A closed enum (not a free-form string) so a new
/// reason is a reviewed, compiler-visible addition, and so downstream
/// matching (e.g. an audit that should only ever see `ExternalImport`) breaks
/// loudly at compile time if a second reason is ever added without being
/// handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelContentDigestUnavailableReason {
    /// The model reached this environment through the external-producer
    /// import path (`pipeline::import`), which has no local model directory
    /// — no config, pooling config, tokenizer, or weights files — to hash.
    ExternalImport,
}

/// A canonical, deterministically-serialisable description of the verb that
/// produced a result table and its typed parameters — the input to the
/// definition hash in place of a logical plan.
///
/// Each result-table producer fills in exactly one variant from its own typed
/// parameters. `serde` with a stable (sorted-key) JSON encoding yields canonical
/// bytes that SHA-256 folds into the [`DefinitionHash`]. Two runs of the same
/// producer with the same parameters
/// serialise identically; any output-affecting parameter change changes the
/// bytes and therefore the hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "producer", rename_all = "snake_case")]
pub enum ProducingDescriptor {
    /// Inference output: a model run over a source's content columns, keyed by
    /// `key_column`. (`InferenceSession::infer`.)
    Inference {
        /// The model that ran inference, canonical id.
        model_id: String,
        /// The model task (embedding / classification / regression / …).
        task: ModelTask,
        /// The source the input rows were scanned from.
        source_id: String,
        /// The content columns fed to the model, in caller order.
        content_columns: Vec<String>,
        /// The key column that identifies each output row.
        key_column: String,
    },
    /// Embedding pipeline output: a model embedding over a source's columns.
    /// (`EmbeddingPipeline::run`.)
    Embedding {
        /// The embedding model, canonical id.
        model_id: String,
        /// The embedding task (text / image).
        task: ModelTask,
        /// The source the input rows were scanned from.
        source_id: String,
        /// The columns embedded, in caller order.
        columns: Vec<String>,
        /// The key column that identifies each output row.
        key_column: String,
        /// The embedding width.
        dimensions: usize,
    },
    /// Neighbor-graph derivation: a k-NN edge relation derived from an embedding
    /// table. (`NeighborGraphPipeline::write_edge_table`.)
    ///
    /// Every field below changes the emitted edge set or its determinism, so all
    /// are part of the definition: `k` sets the fan-out; `min_similarity_bits`
    /// prunes edges below a floor; `mutual` keeps only reciprocal edges; `self_exclude`
    /// drops (or keeps) the self-edge; `exact` selects the deterministic
    /// brute-force driver over the non-deterministic index-assisted one (so it is
    /// itself output-affecting); `exact_max_rows` is the ceiling that gates the
    /// exact driver; and `index_storage_precision` is the source table's ANN
    /// sidecar-index precision **at the moment the index-assisted driver ran**
    /// (`None` when `exact`, which reads Parquet directly and never touches the
    /// index, so the source table's precision cannot affect it). A quantized
    /// index (`F16`/`Int8`) yields a different approximate neighbour set than an
    /// `F32` one over the identical rows — the same non-determinism contract
    /// that already makes the index-assisted driver's own recall a determinant,
    /// so the precision it ran at must be recorded too. (The `resolve_keys`
    /// flag is *not* recorded: today a resolved endpoint equals its `_row_id`
    /// either way, so it does not affect the output — recording it would be a
    /// false determinant.)
    ///
    /// The retrieve→rescore `oversample` multiplier is deliberately **not**
    /// recorded here even though the index-assisted driver reads it — folding
    /// a live-mutable deployment/table config knob into a deterministic
    /// identity would itself be wrong (a later config change would silently
    /// stop matching a still-valid cached artifact). The omission is sound
    /// only because of a persisted-record invariant enforced at
    /// [`crate::store::ResultStore::create_table`]/
    /// [`crate::catalog::Catalog::create_result_table`]: `oversample` and
    /// `storage_precision` are always written together (migration 023), so
    /// no persisted row has `oversample = None` under a `storage_precision`
    /// for which [`crate::config::StoragePrecision::needs_rescore`] is true —
    /// an `F32` (or pre-023-`None`, which reads back as `F32`) row makes
    /// `oversample` inert (`needs_rescore() == false`, so the index-assisted
    /// merge is already exact and `oversample` cannot change it), and a
    /// rescoring row always carries a concrete `oversample`. If either write
    /// site ever persisted `storage_precision` and `oversample` independently,
    /// or `oversample` started mattering at `F32`, this omission would become
    /// a silent stale cache-hit. See the coupling tripwire test in
    /// `crates/jammi-db/tests/it/store.rs`
    /// (`create_table_couples_rescoring_precision_to_a_present_oversample`).
    NeighborGraph {
        /// The embedding result table the edges were derived from.
        source_table: String,
        /// The number of neighbours per node.
        k: usize,
        /// Edge-weight floor: edges below this `similarity` are dropped, by its
        /// IEEE-754 bit pattern (`f32::to_bits`) so the descriptor stays
        /// bit-exact and `Eq`/`Hash`-able. `None` keeps all `k` edges per node.
        min_similarity_bits: Option<u32>,
        /// Keep an edge only when its reverse also survives (reciprocal filter).
        mutual: bool,
        /// Whether the self-edge `(a, a)` is excluded.
        self_exclude: bool,
        /// Whether the deterministic, complete exact driver was forced (vs the
        /// non-deterministic index-assisted one).
        exact: bool,
        /// Row-count ceiling that gates the exact driver.
        exact_max_rows: usize,
        /// The source table's sidecar-index storage precision at build time,
        /// when the index-assisted driver ran (`exact = false`); `None` when
        /// `exact = true`.
        index_storage_precision: Option<crate::config::StoragePrecision>,
    },
    /// Graph-propagation output: K hops of feature propagation over a
    /// neighbor-graph, materialised as a new embedding table.
    /// (`propagate_embeddings` via `materialize_embedding_table`.)
    ///
    /// Propagation reads **two** inputs — the embedding table holding `X⁽⁰⁾` and
    /// the edge relation defining the graph — so both are anchored in
    /// [`MaterializationManifest::input_anchors`]; the edge relation is recorded
    /// here by id (`edge_source`) as the second determinant. The kernel knobs
    /// `direction`, `hops` (the *effective*, post-clamp depth), `alpha`,
    /// `weighting`, and `output` each change the propagated vectors or their
    /// dimensionality, so all are part of the definition.
    GraphPropagation {
        /// The embedding result table whose features were propagated.
        source_table: String,
        /// The edge relation the propagation read, with its full column
        /// bindings — the second input anchor's source. A staleness/lineage
        /// determinant independent of the kernel knobs: the same knobs over a
        /// different graph (or the same registered source read through different
        /// `src`/`dst`/weight/type/as-of columns) yield a different output, so
        /// the whole binding — not just the source id — is part of the
        /// definition and must replay losslessly.
        edge_source: EdgeSourceBinding,
        /// The propagation kernel's canonical id.
        kernel_id: String,
        /// Edge-direction the walk followed.
        direction: PropagationDirection,
        /// The number of hops actually run (clamped to the depth cap).
        hops: usize,
        /// APPNP teleport probability re-mixed each hop, recorded by its IEEE-754
        /// bit pattern (`f64::to_bits`) so the descriptor stays bit-exact and
        /// `Eq`/`Hash`-able — two runs with the same `α` hash identically, and a
        /// different `α` (down to the last bit) changes the hash.
        alpha_bits: u64,
        /// How neighbour contributions were weighted.
        weighting: PropagationWeighting,
        /// What the propagation emitted (final block vs Jumping-Knowledge concat).
        output: PropagationOutput,
        /// The output embedding width.
        dimensions: usize,
    },
    /// Context-set output: per-target pooled context vectors materialised as a
    /// new embedding table. The real producer is the
    /// `assemble_context`→`materialize_context` pair — `materialize_context` is a
    /// sink receiving pre-pooled rows, so the determinants are the
    /// `assemble_context` **recipe** (the `ContextRequest`) the whole batch
    /// shared. (`materialize_context` via `materialize_embedding_table`.)
    ///
    /// The per-target `query` vector and `exclude_key` are deliberately **not**
    /// recorded: a batch materialises one recipe over many targets, and those two
    /// fields vary per target — they are the *inputs over which* the recipe runs
    /// (and become the output table's row keys), not the recipe's definition.
    /// Recording one target's query in a batch-level descriptor would be a false
    /// determinant; the definition is the recipe every target was pooled under.
    ContextSet {
        /// The encoder's canonical id.
        encoder_id: String,
        /// The source whose rows were pooled per target.
        source_id: String,
        /// The **resolved** source embedding table the recipe actually pooled
        /// against — the concrete table name, never the user's `Option`. Even
        /// when the recipe left the table unset (resolve the source's newest
        /// embedding table), the producer records the table that resolution
        /// selected, so a recompute re-pools over the **same** table the recipe
        /// ran on. This matters because a materialised context set is itself a
        /// `kind=model` table for the same source: had the descriptor recorded
        /// the user's `None`, the default resolution on replay would re-select
        /// the newer context-set output and shadow the original source table,
        /// pooling over the wrong rows. An output-affecting determinant: the same
        /// recipe over a different source embedding table pools different vectors.
        /// (`Option` only because a pre-resolution descriptor fixture may carry
        /// `None`; every producer-written descriptor records `Some`.)
        embedding_table: Option<String>,
        /// Where each target's candidate members came from — ANN retrieval, a
        /// declared-edge walk, or both — the determinant that selects which
        /// neighbours are pooled.
        candidate_source: ContextCandidateSource,
        /// The label / outcome columns hydrated from the source per context row.
        value_columns: Vec<String>,
        /// The permutation-invariant pooling reduction.
        aggregator: ContextAggregator,
        /// Whether the leakage guard dropped each target's own row from its
        /// context before pooling.
        exclude_self: bool,
        /// The optional split predicate scoping the context (the train/target
        /// leakage line). `None` = no split scope.
        split: Option<String>,
        /// The pooled-vector width.
        dimensions: usize,
    },
    /// As-of temporal join output: each spine row matched to at most one fact
    /// row valid as-of the spine instant within its equality group, materialised
    /// as a new result table. (`asof_join` via the `AsofJoinExec` sort-merge.)
    ///
    /// The fields are the join's output-affecting parameters in a
    /// transport-neutral encoding — the temporal-engine enums live in the AI
    /// crate, so the descriptor records them as the canonical string/scalar
    /// tags the hash folds over. Two runs of the same join over the same input
    /// anchors hash identically; any change to a knob, a key, or the projection
    /// changes the bytes and therefore the hash.
    AsofJoin {
        /// The spine relation's catalog id.
        spine: String,
        /// The facts relation's catalog id.
        facts: String,
        /// The spine's equality ("by") columns, in declared order.
        spine_by: Vec<String>,
        /// The facts' equality ("by") columns, in declared order.
        facts_by: Vec<String>,
        /// The spine's temporal ordering column.
        spine_time: String,
        /// The facts' temporal ordering column.
        facts_time: String,
        /// Match direction (`backward` / `forward` / `nearest`).
        direction: AsofDirection,
        /// Boundary inclusivity (`inclusive` / `exclusive`).
        boundary: AsofBoundary,
        /// Optional look-back/forward limit, encoded as `(unit, magnitude)`
        /// where `unit` is `duration` (microseconds) or `steps`. `None` =
        /// unbounded look-back.
        tolerance: Option<AsofTolerance>,
        /// Tie-break: the secondary descending column, or `None` for the loud
        /// `error` policy.
        tie_break_column: Option<String>,
        /// Right-side projection columns, in output order. Empty = all non-key
        /// columns.
        project: Vec<String>,
    },
    /// A table produced by a verb the engine does not own: a consumer built the
    /// rows through its own producer and asked the engine only to publish them
    /// behind the materialization contract. The engine has no dispatch for a
    /// producer it did not define, so a `recompute` of an external table is the
    /// loud typed refusal [`crate::error::JammiError::NotRecomputable`], never a
    /// re-run guessed from the table's columns. The rest of the contract still
    /// holds unchanged: `verify_materialization` recomputes the artifact digest,
    /// and the definition hash folds these fields exactly like any other
    /// producer's.
    ///
    /// `params` carries the producer's full set of output-affecting parameters
    /// as canonical key/value pairs. Their **completeness is the producer's
    /// contract** — the same burden every variant above carries in its typed
    /// fields: a parameter that changes the output but is absent here makes two
    /// different productions serialise identically and therefore collide on one
    /// hash (a silent false match). A `BTreeMap` so the canonical encoding is
    /// key-ordered and stable regardless of insertion order.
    External {
        /// The canonical id of the verb the engine does not own — an opaque
        /// label naming the external producer (e.g. its stable pipeline id).
        /// Named `producer_id`, not `producer`, because the enum's serde tag key
        /// is already `producer` (it holds the variant discriminant `external`);
        /// a field also named `producer` would collide with it in the flattened
        /// internally-tagged encoding.
        producer_id: String,
        /// Every output-affecting parameter of the external producer, as
        /// canonical string key/value pairs. An omitted determinant is a silent
        /// false match, so completeness is the producer's responsibility.
        params: BTreeMap<String, String>,
    },
}

/// Match direction recorded in an [`ProducingDescriptor::AsofJoin`] — the
/// transport-neutral mirror of the AI crate's `MatchDirection`, so the
/// definition hash covers the direction without `jammi-db` depending on the
/// temporal-engine types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsofDirection {
    /// Most recent fact at/before the spine instant.
    Backward,
    /// First fact at/after the spine instant.
    Forward,
    /// Smallest absolute distance, ties toward the past.
    Nearest,
}

/// Boundary inclusivity recorded in an [`ProducingDescriptor::AsofJoin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsofBoundary {
    /// A fact at exactly the spine instant is eligible (`<=` / `>=`).
    Inclusive,
    /// Strict (`<` / `>`).
    Exclusive,
}

/// Look-back/forward limit recorded in an [`ProducingDescriptor::AsofJoin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsofTolerance {
    /// Microsecond limit for a temporal key.
    Duration(i64),
    /// Step limit for an integer key.
    Steps(i64),
}

/// Edge direction recorded in [`ProducingDescriptor::GraphPropagation`] and the
/// edge gather of [`ProducingDescriptor::ContextSet`] — the transport-neutral
/// mirror of the AI crate's `EdgeDirection`, so the definition hash covers the
/// walk direction without `jammi-db` depending on the graph types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropagationDirection {
    /// Follow `src → dst` edges (out-neighbours).
    Out,
    /// Follow `dst → src` edges (in-neighbours).
    In,
    /// Both directions count as adjacency.
    Undirected,
}

/// Neighbour-contribution weighting recorded in
/// [`ProducingDescriptor::GraphPropagation`] — the transport-neutral mirror of
/// the AI crate's `PropagationWeighting`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropagationWeighting {
    /// Random-walk normalisation `D̃^{-1}Ã` (the plain neighbour mean).
    Uniform,
    /// Symmetric normalisation `D̃^{-1/2}(A+I)D̃^{-1/2}` (the APPNP default).
    DegreeNormalized,
    /// Edge-weighted mean `Σ(w·x)/Σw` over the neighbourhood.
    EdgeSimilarity,
}

/// What a propagation emitted, recorded in
/// [`ProducingDescriptor::GraphPropagation`] — the transport-neutral mirror of
/// the AI crate's `PropagationOutput`. Changes the output dimensionality, so it
/// is part of the definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropagationOutput {
    /// Only the final `X⁽ᴷ⁾` block — a `d`-dim table in the input's space.
    Final,
    /// The per-hop blocks concatenated (Jumping Knowledge) — `(K+1)·d`-dim.
    JumpingKnowledge,
}

/// The pooling reduction recorded in [`ProducingDescriptor::ContextSet`] — the
/// transport-neutral mirror of the AI crate's `SetAggregator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextAggregator {
    /// Element-wise mean.
    Mean,
    /// Element-wise sum.
    Sum,
    /// Element-wise maximum.
    Max,
}

/// How a hybrid context merges its ANN and declared-edge candidate sets,
/// recorded in [`ContextCandidateSource::Hybrid`] — the transport-neutral mirror
/// of the AI crate's `HybridMerge`. An enum (not a bool) so per-edge-type merge
/// channels can be added; the merge is output-affecting (it selects which keys
/// survive into the pool), so it is part of the definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextHybridMerge {
    /// Union the candidate key sets (ANN first, then declared-edge members not
    /// already present), dedup, pool once.
    Union,
}

/// Where a context set's candidate members came from, recorded in
/// [`ProducingDescriptor::ContextSet`] — the transport-neutral mirror of the AI
/// crate's `ContextSource`. The candidate-set source is a determinant: the same
/// pooling over a different candidate set yields a different output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "candidate", rename_all = "snake_case")]
pub enum ContextCandidateSource {
    /// `search(query, k)` over the source's embedding table.
    Ann {
        /// Neighbourhood size.
        k: usize,
    },
    /// A declared-edge walk anchored at the target.
    Edges {
        /// The edge gather that produced the candidate keys.
        gather: ContextEdgeGather,
    },
    /// Union of the ANN and declared-edge candidate sets, pooled once.
    Hybrid {
        /// ANN neighbourhood size for the retrieval arm.
        ann_k: usize,
        /// The declared-edge gather for the edge arm.
        gather: ContextEdgeGather,
        /// How the two candidate sets merge — an output-determinant recorded so
        /// a second merge channel can't silently regress the descriptor.
        merge: ContextHybridMerge,
    },
}

/// A bounded declared-edge gather recorded in a [`ContextCandidateSource`] — the
/// transport-neutral mirror of the AI crate's `EdgeGather`, carrying every
/// output-affecting knob of the walk. `hops` is the *effective* (post-clamp)
/// depth, so the cap itself never needs recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextEdgeGather {
    /// The edge relation walked.
    pub edge_source: EdgeSourceBinding,
    /// The effective (post-clamp) number of hops walked.
    pub hops: usize,
    /// Per-node per-hop neighbour sample cap (GraphSAGE). `None` = exact.
    pub fanout: Option<usize>,
    /// The direction the walk followed.
    pub direction: PropagationDirection,
    /// Optional edge-type allow-list, in declared order.
    pub edge_types: Option<Vec<String>>,
    /// Optional minimum edge weight to traverse, by IEEE-754 bit pattern so the
    /// descriptor stays bit-exact and `Eq`/`Hash`-able. `None` = no floor.
    pub min_weight_bits: Option<u64>,
    /// Optional as-of pin (used with a registered source's as-of column).
    pub as_of: Option<String>,
}

/// Which edge relation an edge-reading verb walked, with its full column
/// bindings — the transport-neutral mirror of the AI crate's `EdgeSourceRef`,
/// shared by every descriptor that records an edge source (a
/// [`ProducingDescriptor::GraphPropagation`] and the gather of a
/// [`ContextEdgeGather`]). The `Registered` bindings are output-affecting (the
/// same source read through different `src`/`dst`/weight/type/as-of columns is a
/// different graph), so the mirror carries them all and a replay reconstructs
/// them losslessly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "edge_source", rename_all = "snake_case")]
pub enum EdgeSourceBinding {
    /// A `neighbor_graph` result table, by registered name.
    NeighborGraph {
        /// The registered result-table name.
        table_name: String,
    },
    /// A registered external edge source: its id plus the columns the walk read.
    Registered {
        /// The registered source id holding the edge rows.
        source_id: String,
        /// Column holding the edge's source endpoint.
        src_column: String,
        /// Column holding the edge's destination endpoint.
        dst_column: String,
        /// Optional edge-type column (for the `edge_types` filter).
        type_column: Option<String>,
        /// Optional edge-weight column (for the `min_weight` filter).
        weight_column: Option<String>,
        /// Optional as-of column (for the `as_of` pin).
        as_of_column: Option<String>,
    },
}

impl ProducingDescriptor {
    /// Canonical bytes for hashing: a JSON encoding with object keys sorted, so
    /// the byte stream is independent of struct field declaration order and
    /// stable across serde versions. Pure; no I/O.
    fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        let value = serde_json::to_value(self)
            .map_err(|e| ManifestError::UncanonicalDescriptor(e.to_string()))?;
        let canonical = canonicalize_json(&value);
        serde_json::to_vec(&canonical)
            .map_err(|e| ManifestError::UncanonicalDescriptor(e.to_string()))
    }
}

/// The materialization contract a producer supplies for one result table — the
/// producing description, the output-affecting environment, and the resolved
/// input anchors, grouped so the single funnel
/// ([`crate::store::ResultStore::finalize_with_manifest`]) takes one value
/// rather than three positional arguments. Borrows the descriptor and
/// environment (the producer owns them for the call's duration) and owns the
/// anchors (resolved per-write).
#[derive(Debug)]
pub struct Materialization<'a> {
    /// How the table was produced — the verb + its typed parameters.
    pub descriptor: &'a ProducingDescriptor,
    /// The output-affecting environment (engine version, compute device, models).
    pub env: &'a MaterializationEnv,
    /// The as-of state of every input the producer read, in producer order.
    pub inputs: Vec<InputAnchor>,
}

impl<'a> Materialization<'a> {
    /// Group a producer's contract inputs for the funnel.
    pub fn new(
        descriptor: &'a ProducingDescriptor,
        env: &'a MaterializationEnv,
        inputs: Vec<InputAnchor>,
    ) -> Self {
        Self {
            descriptor,
            env,
            inputs,
        }
    }
}

/// The immutable state-pointer of one input a producer read. A robust anchor is
/// a content-derived id, never a wall-clock timestamp: a timestamp resolves
/// against a prunable log and can drift or expire, while an id is stable. The
/// producer resolves "as of T" to an id at write time and carries the id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAnchor {
    /// The input relation's catalog id.
    pub source: String,
    /// The immutable state pointer, encoded per `kind`.
    pub anchor: AnchorValue,
    /// What kind of state pointer `anchor` holds.
    pub kind: AnchorKind,
}

impl InputAnchor {
    /// An immutable result table input: its artifact digest is its anchor.
    pub fn result_digest(source: impl Into<String>, digest: &ArtifactDigest) -> Self {
        Self {
            source: source.into(),
            anchor: AnchorValue(digest.0.clone()),
            kind: AnchorKind::ResultDigest,
        }
    }

    /// A mutable companion table input: its monotonic version at read time.
    pub fn mutable_version(source: impl Into<String>, version: u64) -> Self {
        Self {
            source: source.into(),
            anchor: AnchorValue(version.to_string()),
            kind: AnchorKind::MutableVersion,
        }
    }

    /// An external source exposing an as-of/version surface: the pinned value.
    pub fn source_version(source: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            anchor: AnchorValue(version.into()),
            kind: AnchorKind::SourceVersion,
        }
    }

    /// An external source with no version surface: the read instant only. The
    /// manifest records that this input is *not* reproducibly pinned.
    pub fn unpinned_at_instant(source: impl Into<String>, instant: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            anchor: AnchorValue(instant.into()),
            kind: AnchorKind::UnpinnedAtInstant,
        }
    }
}

/// What kind of immutable state pointer an [`InputAnchor`] carries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorKind {
    /// An immutable Parquet result table: its content digest *is* its anchor.
    ResultDigest,
    /// A mutable companion table: the catalog's monotonic version counter for
    /// that table at read time.
    MutableVersion,
    /// An external/federated source exposing an as-of/version column: the pinned
    /// value of that column (an Iceberg snapshot id, a Delta version, an LSN, a
    /// watermark).
    SourceVersion,
    /// An external source with no version surface. The anchor is the read
    /// instant only; the manifest records that this input is not reproducibly
    /// pinned, so a verifier downgrades its confidence honestly rather than
    /// claim a guarantee it cannot keep.
    UnpinnedAtInstant,
}

/// The immutable state pointer of one input, encoded per its [`AnchorKind`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnchorValue(pub String);

/// The attestation written beside every materialised table. Shaped after an
/// in-toto statement: a `subject` (the artifact digest) plus a predicate
/// (everything about how it was produced), so a consumer verifies by digest
/// match then evaluates the predicate against its own policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationManifest {
    /// in-toto subject: digest of the Parquet artifact this manifest attests to.
    pub artifact: ArtifactDigest,
    /// How it was produced (the "definition"): hash of descriptor + environment.
    pub definition_hash: DefinitionHash,
    /// The producing descriptor recorded **verbatim** — the typed verb + its
    /// output-affecting parameters. The [`definition_hash`](Self::definition_hash)
    /// folds this away into an opaque digest for comparison; the descriptor is
    /// also kept in the clear so a reader can **replay** the producer over the
    /// inputs' current state (the recompute action), which an opaque hash cannot
    /// drive. A reader that only verifies reads the hash; a reader that recomputes
    /// reads the descriptor.
    pub descriptor: ProducingDescriptor,
    /// The as-of state of every input, in producer order.
    pub input_anchors: Vec<InputAnchor>,
    /// Producing-run identity (a per-process id) — provenance, never the
    /// reproducibility anchor (that is the `input_anchors`).
    pub produced_by: String,
    /// Producing instant, RFC3339. Provenance metadata, never the anchor.
    pub produced_at: String,
    /// Engine semantic version that produced this artifact.
    pub engine_version: String,
    /// Manifest format version, so a future format change is a typed error.
    pub manifest_version: u32,
}

impl MaterializationManifest {
    /// Compute the manifest over a producing description, its environment,
    /// resolved input anchors, and the written artifact's digest. Pure: no I/O.
    ///
    /// The `definition_hash` folds the canonical descriptor bytes and the
    /// canonical environment bytes; the `input_anchors` carry the as-of state of
    /// every input but are deliberately **not** part of the definition hash —
    /// the definition is *how* a table is produced, the anchors are *over what*.
    /// (A consumer that wants a combined "code + data" identity composes the
    /// two, as Dagster composes `code_version` with input data versions.)
    pub fn compute(
        descriptor: &ProducingDescriptor,
        env: &MaterializationEnv,
        inputs: Vec<InputAnchor>,
        artifact: ArtifactDigest,
        produced_by: String,
        produced_at: String,
    ) -> Result<Self, ManifestError> {
        let definition_hash = Self::definition_of(descriptor, env)?;
        Ok(Self {
            artifact,
            definition_hash,
            descriptor: descriptor.clone(),
            input_anchors: inputs,
            produced_by,
            produced_at,
            engine_version: env.engine_version.clone(),
            manifest_version: MANIFEST_VERSION,
        })
    }

    /// The [`DefinitionHash`] a producer would record for `(descriptor, env)` —
    /// the *same* fold [`Self::compute`] performs, exposed so a cache probe can
    /// build the lookup key **before** the expensive compute (it has no artifact
    /// digest yet, only the definition). The probe key is therefore byte-identical
    /// to what the funnel records at finalize: a top-of-producer
    /// `definition_of(descriptor, env)` and the finalised manifest's
    /// `definition_hash` are the same value for the same inputs, which is what
    /// makes the memoization sound.
    pub fn definition_of(
        descriptor: &ProducingDescriptor,
        env: &MaterializationEnv,
    ) -> Result<DefinitionHash, ManifestError> {
        definition_hash(descriptor, env)
    }

    /// The unpinned inputs, by source id — the inputs whose anchor is an
    /// instant, not a reproducible id. Empty when every input is pinned.
    pub fn unpinned_inputs(&self) -> Vec<String> {
        self.input_anchors
            .iter()
            .filter(|a| a.kind == AnchorKind::UnpinnedAtInstant)
            .map(|a| a.source.clone())
            .collect()
    }

    /// Serialise the manifest to JSON bytes for the sidecar.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Parse a manifest from its sidecar bytes, rejecting any format version this
    /// build does not support.
    ///
    /// The persisted manifest carries both the *opaque* [`DefinitionHash`] (for
    /// comparison) and the [`ProducingDescriptor`] in the clear (for replay). When
    /// the descriptor's determinant set changes, the recorded descriptor's JSON
    /// shape changes. Two independent guards keep a stale shape from being
    /// trusted. **First**, serde itself: a genuinely old manifest whose JSON
    /// predates a since-added field (e.g. a pre-`descriptor` manifest) fails the
    /// `serde_json::from_slice` decode with a typed [`ManifestError`] before the
    /// version is even inspected. **Second**, the version guard, for a shape that
    /// still deserialises but was written under a different determinant set: any
    /// version that is not exactly [`MANIFEST_VERSION`] — older (a different,
    /// now-superseded determinant set) or newer (a format this build cannot
    /// read) — is a typed [`ManifestError::UnsupportedManifestVersion`]. The
    /// version is checked **before** the parsed manifest is handed back, so a
    /// reader never silently trusts a stale hash or replays a stale descriptor;
    /// whichever guard fires, the typed error is the signal to re-emit.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ManifestError> {
        let manifest: Self = serde_json::from_slice(bytes)?;
        if manifest.manifest_version != MANIFEST_VERSION {
            return Err(ManifestError::UnsupportedManifestVersion {
                found: manifest.manifest_version,
                supported: MANIFEST_VERSION,
            });
        }
        Ok(manifest)
    }
}

/// Compute the definition hash: SHA-256 over the canonical descriptor bytes and
/// the canonical environment bytes, length-prefixed and domain-separated so a
/// descriptor field can never alias an environment field.
fn definition_hash(
    descriptor: &ProducingDescriptor,
    env: &MaterializationEnv,
) -> Result<DefinitionHash, ManifestError> {
    let descriptor_bytes = descriptor.canonical_bytes()?;
    let env_value = serde_json::to_value(env)?;
    let env_bytes = serde_json::to_vec(&canonicalize_json(&env_value))?;

    let mut hasher = Sha256::new();
    hasher.update(b"jammi.materialization.definition.v1");
    hasher.update(b"\0descriptor\0");
    hasher.update((descriptor_bytes.len() as u64).to_le_bytes());
    hasher.update(&descriptor_bytes);
    hasher.update(b"\0env\0");
    hasher.update((env_bytes.len() as u64).to_le_bytes());
    hasher.update(&env_bytes);
    Ok(DefinitionHash(hex::encode(hasher.finalize())))
}

/// Recursively rewrite a JSON value into a canonical form: object keys sorted
/// lexically so the serialised byte stream is independent of insertion order.
/// Arrays keep their order (it is semantically significant — column order,
/// model order). Scalars pass through.
fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: std::collections::BTreeMap<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), canonicalize_json(v)))
                .collect();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize_json).collect())
        }
        other => other.clone(),
    }
}

/// The outcome of checking a materialised table against an expectation. The
/// engine returns a verdict; it never *acts* on one.
///
/// Every verdict attests the **Parquet data**, never the ANN search index: the
/// index is a derived accelerator reconstructible from the data, so a `Match`
/// asserts the data-of-record is the output of the expected definition, not that
/// any particular index bytes are present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum MatchVerdict {
    /// Recomputed artifact digest equals the manifest's, and (if the caller
    /// supplied one) the manifest's definition hash equals the expected one.
    Match,
    /// Digest or definition hash differs — the served artifact is not the output
    /// of the expected definition. Carries both sides for the caller.
    Mismatch {
        /// The expectation the caller supplied (or the manifest's own digest
        /// when the recomputed artifact digest itself diverged).
        expected: String,
        /// What was actually found.
        found: String,
    },
    /// The artifact verifies, but at least one input was `UnpinnedAtInstant`, so
    /// reproducibility cannot be fully asserted. Honest, not silent.
    MatchWithUnpinnedInputs {
        /// The source ids of the inputs that were not reproducibly pinned.
        unpinned: Vec<String>,
    },
    /// No manifest sidecar exists for the table. A truthful "unknown" — a
    /// pre-contract table, never a fabricated match.
    MissingManifest,
}

/// Errors raised by the materialization contract.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// A producing descriptor could not be canonicalised for hashing.
    #[error("producing descriptor is not canonicalisable: {0}")]
    UncanonicalDescriptor(String),
    /// The manifest sidecar is missing for a table that should carry one.
    #[error("manifest sidecar missing for table `{0}`")]
    MissingManifest(String),
    /// A `ready` table created after the contract landed carries no manifest —
    /// a torn write or a producer that bypassed the funnel. Distinct from a
    /// legitimate pre-contract table (which verifies as
    /// [`MatchVerdict::MissingManifest`]).
    #[error(
        "table `{0}` is post-contract but has no manifest sidecar (torn write or bypassed funnel)"
    )]
    PostContractManifestMissing(String),
    /// The manifest sidecar's format version is not the one this build supports —
    /// older (a superseded determinant set, so its definition hash is
    /// incomparable) or newer (a format this build cannot read). Either way the
    /// artifact must be re-emitted.
    #[error(
        "manifest format version {found} is incompatible with supported version {supported}; \
         re-emit the artifact"
    )]
    UnsupportedManifestVersion {
        /// The version found on disk.
        found: u32,
        /// The format version this build reads and writes.
        supported: u32,
    },
    /// JSON (de)serialisation of a descriptor / environment / manifest failed.
    #[error("manifest serialisation error: {0}")]
    Serde(#[from] serde_json::Error),
    /// A storage read/write of the sidecar failed.
    #[error(transparent)]
    Storage(#[from] crate::storage::StorageError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding_descriptor() -> ProducingDescriptor {
        ProducingDescriptor::Embedding {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
            task: ModelTask::TextEmbedding,
            source_id: "docs".into(),
            columns: vec!["title".into(), "body".into()],
            key_column: "_row_id".into(),
            dimensions: 384,
        }
    }

    fn cpu_env() -> MaterializationEnv {
        MaterializationEnv::new(
            ComputeDevice::Cpu,
            vec![ModelIdentity {
                model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
                backend: "candle".into(),
                compute_precision: ComputePrecision::F32,
                content_digest: ModelContentDigest::Sha256("cpu-fixture-digest".into()),
                quantization: None,
            }],
        )
    }

    #[test]
    fn definition_hash_is_deterministic() {
        let d = embedding_descriptor();
        let env = cpu_env();
        assert_eq!(
            definition_hash(&d, &env).unwrap(),
            definition_hash(&d, &env).unwrap()
        );
    }

    /// HASH-PRESERVATION GOLDEN (issue #351, wave 2): `quantization: None`
    /// must serialise to no key at all (`#[serde(skip_serializing_if =
    /// "Option::is_none")]`), never a present `null` — a present-but-null key
    /// would still change the canonical byte stream (and therefore every
    /// existing `DefinitionHash`) relative to a pre-feature row that never
    /// had this key in its JSON shape.
    #[test]
    fn quantization_none_serialises_to_no_key() {
        let identity = ModelIdentity {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("cpu-fixture-digest".into()),
            quantization: None,
        };
        let value = serde_json::to_value(&identity).unwrap();
        let object = value.as_object().unwrap();
        assert!(
            !object.contains_key("quantization"),
            "quantization: None must serialise to no key, got {value:#?}"
        );
    }

    /// HASH-PRESERVATION GOLDEN (issue #351, wave 2): the end-to-end
    /// `definition_hash` for the representative `embedding_descriptor()` /
    /// `cpu_env()` fixture (the same fixture `definition_hash_is_deterministic`
    /// above uses) must still equal the value this exact fixture hashed to
    /// BEFORE `ModelIdentity::quantization` existed. The literal below was
    /// computed at base commit `309d1a10` (the commit this unit's contract
    /// names as base; `jammi-db` is byte-identical between `309d1a10` and
    /// wave 1's `7c90b328`, which touched only `jammi-numerics`/
    /// `jammi-kernels`/`jammi-lora`/`jammi-encoders`) by temporarily adding a
    /// `#[test] fn temp_print_golden_hash() { panic!("{}",
    /// definition_hash(&embedding_descriptor(), &cpu_env()).unwrap()); }` to
    /// this same test module at that commit, running
    /// `cargo test -p jammi-db temp_print_golden_hash -- --nocapture`, and
    /// copying the printed hex out of the panic message — then discarding
    /// that scaffolding. If this test ever needs to change, no future edit
    /// may just update the literal to match a new computed value; that would
    /// silently rubber-stamp a migration. A migration is a new
    /// `MANIFEST_VERSION` and a fresh golden recomputed at the NEW base.
    ///
    /// `cpu_env()` stamps `engine_version` from `CARGO_PKG_VERSION` (see
    /// `MaterializationEnv::new`), and `engine_version` is a hash input BY
    /// DESIGN — `different_engine_version_changes_the_hash` below pins
    /// exactly that sensitivity. The golden literal was therefore computed
    /// against whatever `CARGO_PKG_VERSION` was at base commit `309d1a10`,
    /// which was `0.48.0` (`git show 309d1a10:Cargo.toml | grep -m1
    /// '^version'`) — NOT against "whatever version this crate happens to
    /// build as today". Left as `cpu_env()` hands it back, this test would
    /// red on every single-crate-untouched lockstep version bump (e.g. the
    /// 0.48.0 -> 0.49.0 release bump), which is not the migration this golden
    /// exists to catch. So this test — and ONLY this test, never `cpu_env()`
    /// itself, which other tests rely on for the CURRENT engine version —
    /// overrides `engine_version` back to the `309d1a10` value before
    /// hashing, pinning the golden against version drift while still
    /// exercising the real `definition_hash` end-to-end and still leaving the
    /// literal itself immovable.
    #[test]
    fn definition_hash_golden_is_preserved_across_the_quantization_fold() {
        let d = embedding_descriptor();
        let mut env = cpu_env();
        env.engine_version = "0.48.0".into();
        let hash = definition_hash(&d, &env).unwrap();
        assert_eq!(
            hash.as_str(),
            "bb0bb2f37aa2dcde1a2244d6e37f6ca9e8e73c04961c5009164eef72b426ecaa",
            "definition_hash for the embedding_descriptor()/cpu_env() fixture (pinned to \
             engine_version 0.48.0, the CARGO_PKG_VERSION at base commit 309d1a10) drifted from \
             the golden value computed at that base commit — the \
             `quantization: None` fold must be byte-identical to the pre-feature shape, \
             not a silent migration"
        );
    }

    /// DISTINCTNESS (issue #351, wave 2): identical env except
    /// `quantization: Some(Q4K)` vs `None` must hash differently — a
    /// quantized run is output-affecting relative to a full-precision run of
    /// the same model over the same inputs.
    #[test]
    fn quantization_some_differs_from_none() {
        let d = embedding_descriptor();
        let none_hash = definition_hash(&d, &cpu_env()).unwrap();

        let mut quantized_env = cpu_env();
        quantized_env.models[0].quantization = Some(jammi_numerics::WeightQuantization::Q4K);
        let some_hash = definition_hash(&d, &quantized_env).unwrap();

        assert_ne!(
            none_hash, some_hash,
            "quantization: Some(Q4K) must hash differently from quantization: None over an \
             otherwise-identical ModelIdentity"
        );
    }

    /// Serde round-trip (issue #351, wave 2): a pre-feature JSON row — one
    /// with no `quantization` key at all, modelling what is actually on disk
    /// from before this field existed — deserialises to `None` via
    /// `#[serde(default)]`; a `Some` quantization round-trips through the
    /// lowercase wire vocabulary `WeightQuantization` already defines
    /// (`#[serde(rename_all = "lowercase")]`), so `Q4K` reads back as the
    /// string `"q4k"`.
    #[test]
    fn quantization_serde_round_trips_and_pre_feature_rows_default_to_none() {
        // A pre-feature row: no "quantization" key in the JSON at all.
        let pre_feature_json = serde_json::json!({
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "candle",
            "compute_precision": "f32",
            "content_digest": {"state": "sha256", "value": "cpu-fixture-digest"},
        });
        let identity: ModelIdentity = serde_json::from_value(pre_feature_json).unwrap();
        assert_eq!(identity.quantization, None);

        // Some(Q4K) round-trips, and serialises as the lowercase wire tag.
        let mut with_quant = identity.clone();
        with_quant.quantization = Some(jammi_numerics::WeightQuantization::Q4K);
        let value = serde_json::to_value(&with_quant).unwrap();
        assert_eq!(value["quantization"], "q4k");
        let back: ModelIdentity = serde_json::from_value(value).unwrap();
        assert_eq!(
            back.quantization,
            Some(jammi_numerics::WeightQuantization::Q4K)
        );
    }

    #[test]
    fn changing_a_parameter_changes_the_hash() {
        let env = cpu_env();
        let base = definition_hash(&embedding_descriptor(), &env).unwrap();

        let mut other = embedding_descriptor();
        if let ProducingDescriptor::Embedding { columns, .. } = &mut other {
            columns.push("extra".into());
        }
        assert_ne!(base, definition_hash(&other, &env).unwrap());
    }

    fn external_descriptor() -> ProducingDescriptor {
        ProducingDescriptor::External {
            producer_id: "continual-context".into(),
            params: BTreeMap::from([
                ("purpose".into(), "rc".into()),
                ("revision".into(), "7".into()),
                ("dimensions".into(), "384".into()),
            ]),
        }
    }

    #[test]
    fn external_descriptor_round_trips_and_tags_by_producer() {
        let d = external_descriptor();
        // The internally-tagged discriminant is snake_case `external`; the
        // opaque producer id lives under the non-colliding `producer_id` key.
        let value = serde_json::to_value(&d).unwrap();
        assert_eq!(value["producer"], "external");
        assert_eq!(value["producer_id"], "continual-context");
        // A full round-trip preserves the producer id and every param.
        let back: ProducingDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn external_definition_hash_is_deterministic_and_param_complete() {
        let env = cpu_env();
        let base = definition_hash(&external_descriptor(), &env).unwrap();
        // Deterministic: the same producer + params fold to the same hash.
        assert_eq!(base, definition_hash(&external_descriptor(), &env).unwrap());

        // Completeness: changing any recorded param changes the hash, so an
        // external producer that records all its determinants never collides two
        // distinct productions onto one identity.
        let mut changed = external_descriptor();
        if let ProducingDescriptor::External { params, .. } = &mut changed {
            params.insert("revision".into(), "8".into());
        }
        assert_ne!(base, definition_hash(&changed, &env).unwrap());

        // A different opaque producer id is likewise a different definition.
        let mut other_producer = external_descriptor();
        if let ProducingDescriptor::External { producer_id, .. } = &mut other_producer {
            *producer_id = "resilience-perturbation".into();
        }
        assert_ne!(base, definition_hash(&other_producer, &env).unwrap());
    }

    #[test]
    fn different_device_changes_the_hash() {
        let d = embedding_descriptor();
        let cpu = definition_hash(&d, &cpu_env()).unwrap();
        let cuda = definition_hash(
            &d,
            &MaterializationEnv::new(
                ComputeDevice::Cuda { ordinal: 0 },
                vec![ModelIdentity {
                    model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
                    backend: "candle".into(),
                    compute_precision: ComputePrecision::F32,
                    content_digest: ModelContentDigest::Sha256("cpu-fixture-digest".into()),
                    quantization: None,
                }],
            ),
        )
        .unwrap();
        assert_ne!(
            cpu, cuda,
            "CPU and CUDA must hash differently — float outputs differ"
        );
    }

    #[test]
    fn different_engine_version_changes_the_hash() {
        let d = embedding_descriptor();
        let base = definition_hash(&d, &cpu_env()).unwrap();
        let mut bumped = cpu_env();
        bumped.engine_version = "0.0.0-other".into();
        assert_ne!(base, definition_hash(&d, &bumped).unwrap());
    }

    #[test]
    fn different_model_version_changes_the_hash() {
        let d = embedding_descriptor();
        let base = definition_hash(&d, &cpu_env()).unwrap();
        let other_model = MaterializationEnv::new(
            ComputeDevice::Cpu,
            vec![ModelIdentity {
                model_id: "sentence-transformers/all-MiniLM-L12-v2".into(),
                backend: "candle".into(),
                compute_precision: ComputePrecision::F32,
                content_digest: ModelContentDigest::Sha256("cpu-fixture-digest".into()),
                quantization: None,
            }],
        );
        assert_ne!(base, definition_hash(&d, &other_model).unwrap());
    }

    /// K7: `compute_precision` folds into `ModelIdentity` (part of
    /// `MaterializationEnv`), not into a per-descriptor field — so it enters
    /// the definition hash uniformly for *every* model-producing descriptor.
    /// This exercises the `Inference` descriptor specifically (`Embedding` is
    /// covered by `cpu_env`/`different_model_version_changes_the_hash` above)
    /// to prove the fold is not blind to the non-embedding model-producing
    /// path: an `F32` and an `F16` run of the identical model over the
    /// identical `Inference` descriptor must never collide on one identity.
    #[test]
    fn different_compute_precision_changes_the_hash_for_inference_too() {
        let d = ProducingDescriptor::Inference {
            model_id: "distilbert-base-uncased-finetuned-sst-2-english".into(),
            task: ModelTask::Classification,
            source_id: "reviews".into(),
            content_columns: vec!["text".into()],
            key_column: "_row_id".into(),
        };
        let f32_env = MaterializationEnv::new(
            ComputeDevice::Cpu,
            vec![ModelIdentity {
                model_id: "distilbert-base-uncased-finetuned-sst-2-english".into(),
                backend: "candle".into(),
                compute_precision: ComputePrecision::F32,
                content_digest: ModelContentDigest::Sha256("cpu-fixture-digest".into()),
                quantization: None,
            }],
        );
        let f16_env = MaterializationEnv::new(
            ComputeDevice::Cpu,
            vec![ModelIdentity {
                model_id: "distilbert-base-uncased-finetuned-sst-2-english".into(),
                backend: "candle".into(),
                compute_precision: ComputePrecision::F16,
                content_digest: ModelContentDigest::Sha256("cpu-fixture-digest".into()),
                quantization: None,
            }],
        );
        assert_ne!(
            definition_hash(&d, &f32_env).unwrap(),
            definition_hash(&d, &f16_env).unwrap(),
            "F32 and F16 runs of the same model over the same Inference descriptor \
             must never collide on one materialization identity"
        );
    }

    /// K7/esc-057: `content_digest` folds into `ModelIdentity` (part of
    /// `MaterializationEnv`) the same way `compute_precision` does — two
    /// identities that differ ONLY in the model's content digest (same
    /// `model_id`, `backend`, `compute_precision`) must never collide on one
    /// `DefinitionHash`. This is the regression guard for the live esc-057
    /// defect: a `model_id` string alone does not change when the referenced
    /// directory's pooling config / tokenizer / weights bytes change.
    #[test]
    fn different_content_digest_changes_the_hash() {
        let d = embedding_descriptor();
        let base = definition_hash(&d, &cpu_env()).unwrap();

        let mut other_digest = cpu_env();
        other_digest.models[0].content_digest =
            ModelContentDigest::Sha256("a-different-digest".into());
        assert_ne!(
            base,
            definition_hash(&d, &other_digest).unwrap(),
            "two ModelIdentity values differing only in content_digest must never \
             collide on one DefinitionHash (esc-057)"
        );
    }

    /// None-vs-Some: an `Unavailable` content digest (the external-producer
    /// import path, which has no local model directory to hash) must hash
    /// differently from a `Sha256` digest recorded for the identical
    /// `model_id`/`backend`/`compute_precision` — the typed "no digest"
    /// reason is itself part of the identity, never a value that silently
    /// collides with a real digest.
    #[test]
    fn unavailable_content_digest_differs_from_present_digest() {
        let d = embedding_descriptor();
        let present = definition_hash(&d, &cpu_env()).unwrap();

        let mut unavailable_env = cpu_env();
        unavailable_env.models[0].content_digest =
            ModelContentDigest::Unavailable(ModelContentDigestUnavailableReason::ExternalImport);
        let unavailable = definition_hash(&d, &unavailable_env).unwrap();

        assert_ne!(
            present, unavailable,
            "a present content_digest and an Unavailable one must hash differently"
        );
    }

    /// Determinism family, extended to the `Unavailable` arm: two runs over
    /// an environment whose model content digest is `Unavailable` (not just
    /// the `Sha256`-carrying arm `definition_hash_is_deterministic` already
    /// covers) must hash identically.
    #[test]
    fn definition_hash_is_deterministic_with_unavailable_content_digest() {
        let d = embedding_descriptor();
        let mut env = cpu_env();
        env.models[0].content_digest =
            ModelContentDigest::Unavailable(ModelContentDigestUnavailableReason::ExternalImport);
        assert_eq!(
            definition_hash(&d, &env).unwrap(),
            definition_hash(&d, &env).unwrap()
        );
    }

    /// The `ModelIdentity` fold is exhaustive-by-type: `model_identity_from_fields`
    /// destructures `ModelIdentityFields` and reconstructs `ModelIdentity` by
    /// named field, with no `..` elision on either side — a field added to
    /// either struct without a matching update on the other breaks this
    /// function's compilation, so the fixture can never silently go stale
    /// relative to the real type. Every field then gets its own non-default
    /// mutation asserted to move the hash (the non-vacuity guard the
    /// NeighborGraph/GraphPropagation/ContextSet families above already use),
    /// covering `content_digest`'s two states (`Sha256` and `Unavailable`)
    /// alongside `model_id`/`backend`/`compute_precision`.
    #[derive(Clone)]
    struct ModelIdentityFields {
        model_id: String,
        backend: String,
        compute_precision: ComputePrecision,
        content_digest: ModelContentDigest,
        quantization: Option<jammi_numerics::WeightQuantization>,
    }

    fn model_identity_from_fields(p: &ModelIdentityFields) -> ModelIdentity {
        let ModelIdentityFields {
            model_id,
            backend,
            compute_precision,
            content_digest,
            quantization,
        } = p.clone();
        ModelIdentity {
            model_id,
            backend,
            compute_precision,
            content_digest,
            quantization,
        }
    }

    fn env_with_model(identity: ModelIdentity) -> MaterializationEnv {
        MaterializationEnv::new(ComputeDevice::Cpu, vec![identity])
    }

    #[test]
    fn model_identity_each_field_moves_the_hash() {
        let base = ModelIdentityFields {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("base-digest".into()),
            quantization: None,
        };
        let d = embedding_descriptor();
        let base_hash =
            definition_hash(&d, &env_with_model(model_identity_from_fields(&base))).unwrap();

        let cases: &[LabelledMutation<ModelIdentityFields>] = &[
            ("model_id", |p| {
                p.model_id = "sentence-transformers/all-MiniLM-L12-v2".into()
            }),
            ("backend", |p| p.backend = "ort".into()),
            ("compute_precision", |p| {
                p.compute_precision = ComputePrecision::F16
            }),
            ("content_digest (different Sha256)", |p| {
                p.content_digest = ModelContentDigest::Sha256("other-digest".into())
            }),
            ("content_digest (Unavailable)", |p| {
                p.content_digest = ModelContentDigest::Unavailable(
                    ModelContentDigestUnavailableReason::ExternalImport,
                )
            }),
            ("quantization (None -> Some(Q4K))", |p| {
                p.quantization = Some(jammi_numerics::WeightQuantization::Q4K)
            }),
        ];
        for (label, mutate) in cases {
            let mut changed = base.clone();
            mutate(&mut changed);
            let changed_hash =
                definition_hash(&d, &env_with_model(model_identity_from_fields(&changed))).unwrap();
            assert_ne!(
                base_hash, changed_hash,
                "changing ModelIdentity `{label}` must change the definition hash"
            );
        }
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let manifest = MaterializationManifest::compute(
            &embedding_descriptor(),
            &cpu_env(),
            vec![InputAnchor::mutable_version("ref_ranges", 7)],
            ArtifactDigest::of_bytes(b"parquet-bytes"),
            "run-123".into(),
            "2026-06-17T00:00:00Z".into(),
        )
        .unwrap();
        let bytes = manifest.to_json_bytes().unwrap();
        let back = MaterializationManifest::from_json_bytes(&bytes).unwrap();
        assert_eq!(manifest, back);
    }

    fn manifest_at_version(version: u32) -> Vec<u8> {
        let mut manifest = MaterializationManifest::compute(
            &embedding_descriptor(),
            &cpu_env(),
            vec![],
            ArtifactDigest::of_bytes(b"x"),
            "run".into(),
            "2026-06-17T00:00:00Z".into(),
        )
        .unwrap();
        manifest.manifest_version = version;
        serde_json::to_vec(&manifest).unwrap()
    }

    #[test]
    fn newer_manifest_version_is_rejected() {
        assert!(matches!(
            MaterializationManifest::from_json_bytes(&manifest_at_version(MANIFEST_VERSION + 1)),
            Err(ManifestError::UnsupportedManifestVersion { found, supported })
                if found == MANIFEST_VERSION + 1 && supported == MANIFEST_VERSION
        ));
    }

    /// A genuinely old-shape manifest JSON: the current manifest with the
    /// `descriptor` field removed (an old manifest predates the recorded
    /// descriptor) and an older `manifest_version`. Models what is actually on
    /// disk from a prior format, not a current-shape struct with a flipped int.
    fn old_shape_manifest_without_descriptor(version: u32) -> Vec<u8> {
        let manifest = MaterializationManifest::compute(
            &embedding_descriptor(),
            &cpu_env(),
            vec![],
            ArtifactDigest::of_bytes(b"x"),
            "run".into(),
            "2026-06-17T00:00:00Z".into(),
        )
        .unwrap();
        let mut value = serde_json::to_value(&manifest).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("descriptor");
        object.insert("manifest_version".into(), serde_json::json!(version));
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn older_manifest_version_is_rejected_cleanly() {
        // An old (v1) manifest predates the recorded `ProducingDescriptor`, so a
        // real on-disk old manifest has *no* `descriptor` field. Decoding it
        // therefore fails serde-first (`missing field 'descriptor'`) — before the
        // version guard is even reached. That is still a clean rejection: a typed
        // `ManifestError`, the signal to re-emit, never a panic or a silent
        // stale-hash match. The version guard is the *second* line of defence (for
        // a shape that still deserialises under a superseded determinant set); a
        // truly old manifest is caught by the first. Either typed error is
        // acceptable — what must never happen is a silent accept.
        let err =
            MaterializationManifest::from_json_bytes(&old_shape_manifest_without_descriptor(1))
                .expect_err(
                    "an old descriptor-less manifest must be rejected, not silently accepted",
                );
        assert!(
            matches!(
                err,
                ManifestError::Serde(_) | ManifestError::UnsupportedManifestVersion { .. }
            ),
            "expected a clean typed rejection (serde or version guard), got {err:?}"
        );
    }

    // ---- Non-vacuity guard: every output-affecting param of every data-producer
    // variant must move the definition hash. A default-only round-trip passes
    // vacuously exactly where a descriptor is lossy, so each case below flips a
    // *non-default* value and asserts the hash changes — the regression guard
    // that would have caught the original lossy NeighborGraph/GraphPropagation/
    // ContextSet descriptors.

    fn no_model_env() -> MaterializationEnv {
        MaterializationEnv::new(ComputeDevice::Cpu, Vec::new())
    }

    /// A named mutation of a descriptor-fields fixture: a label (for the
    /// assertion message) paired with the closure that flips one output-affecting
    /// field to a non-default value.
    type LabelledMutation<T> = (&'static str, fn(&mut T));

    fn assert_each_change_moves_hash<T: Clone>(
        base: &T,
        env: &MaterializationEnv,
        to_descriptor: impl Fn(&T) -> ProducingDescriptor,
        mutations: &[LabelledMutation<T>],
    ) {
        let base_hash = definition_hash(&to_descriptor(base), env).unwrap();
        for (label, mutate) in mutations {
            let mut changed = base.clone();
            mutate(&mut changed);
            assert_ne!(
                base_hash,
                definition_hash(&to_descriptor(&changed), env).unwrap(),
                "changing `{label}` must change the definition hash (lossy descriptor otherwise)"
            );
        }
    }

    fn neighbor_graph_descriptor(p: &BuildNeighborGraphFields) -> ProducingDescriptor {
        ProducingDescriptor::NeighborGraph {
            source_table: "emb".into(),
            k: p.k,
            min_similarity_bits: p.min_similarity_bits,
            mutual: p.mutual,
            self_exclude: p.self_exclude,
            exact: p.exact,
            exact_max_rows: p.exact_max_rows,
            index_storage_precision: p.index_storage_precision,
        }
    }

    #[derive(Clone)]
    struct BuildNeighborGraphFields {
        k: usize,
        min_similarity_bits: Option<u32>,
        mutual: bool,
        self_exclude: bool,
        exact: bool,
        exact_max_rows: usize,
        index_storage_precision: Option<crate::config::StoragePrecision>,
    }

    #[test]
    fn neighbor_graph_each_param_moves_the_hash() {
        let base = BuildNeighborGraphFields {
            k: 10,
            min_similarity_bits: None,
            mutual: false,
            self_exclude: true,
            exact: false,
            exact_max_rows: 50_000,
            index_storage_precision: Some(crate::config::StoragePrecision::F32),
        };
        assert_each_change_moves_hash(
            &base,
            &no_model_env(),
            neighbor_graph_descriptor,
            &[
                ("k", |p| p.k = 25),
                ("min_similarity", |p| {
                    p.min_similarity_bits = Some(0.7_f32.to_bits())
                }),
                ("mutual", |p| p.mutual = true),
                ("self_exclude", |p| p.self_exclude = false),
                ("exact", |p| p.exact = true),
                ("exact_max_rows", |p| p.exact_max_rows = 1_000),
                ("index_storage_precision", |p| {
                    p.index_storage_precision = Some(crate::config::StoragePrecision::Int8)
                }),
            ],
        );
    }

    #[derive(Clone)]
    struct GraphPropagationFields {
        edge_source: EdgeSourceBinding,
        direction: PropagationDirection,
        hops: usize,
        alpha_bits: u64,
        weighting: PropagationWeighting,
        output: PropagationOutput,
        dimensions: usize,
    }

    fn graph_propagation_descriptor(p: &GraphPropagationFields) -> ProducingDescriptor {
        ProducingDescriptor::GraphPropagation {
            source_table: "emb".into(),
            edge_source: p.edge_source.clone(),
            kernel_id: "graph_propagate".into(),
            direction: p.direction,
            hops: p.hops,
            alpha_bits: p.alpha_bits,
            weighting: p.weighting,
            output: p.output,
            dimensions: p.dimensions,
        }
    }

    #[test]
    fn graph_propagation_each_param_moves_the_hash() {
        let base = GraphPropagationFields {
            edge_source: EdgeSourceBinding::NeighborGraph {
                table_name: "graph".into(),
            },
            direction: PropagationDirection::Out,
            hops: 2,
            alpha_bits: 0.1_f64.to_bits(),
            weighting: PropagationWeighting::DegreeNormalized,
            output: PropagationOutput::Final,
            dimensions: 384,
        };
        assert_each_change_moves_hash(
            &base,
            &no_model_env(),
            graph_propagation_descriptor,
            &[
                ("edge_source", |p| {
                    p.edge_source = EdgeSourceBinding::NeighborGraph {
                        table_name: "other_graph".into(),
                    }
                }),
                ("direction", |p| {
                    p.direction = PropagationDirection::Undirected
                }),
                ("hops", |p| p.hops = 3),
                ("alpha", |p| p.alpha_bits = 0.25_f64.to_bits()),
                ("weighting", |p| {
                    p.weighting = PropagationWeighting::EdgeSimilarity
                }),
                ("output", |p| p.output = PropagationOutput::JumpingKnowledge),
                ("dimensions", |p| p.dimensions = 768),
            ],
        );
    }

    #[test]
    fn graph_propagation_registered_edge_columns_move_the_hash() {
        // The proven HIGH bug: a registered edge source's column bindings are
        // output-affecting (the same source read through swapped `src`/`dst`,
        // or with a weight/type/as-of column, is a different graph). The
        // descriptor now carries the full binding, so flipping any of those
        // columns must move the definition hash — two propagations over the
        // *same* registered source with *different* columns must not collide.
        let base = GraphPropagationFields {
            edge_source: EdgeSourceBinding::Registered {
                source_id: "edges".into(),
                src_column: "from".into(),
                dst_column: "to".into(),
                type_column: None,
                weight_column: None,
                as_of_column: None,
            },
            direction: PropagationDirection::Out,
            hops: 2,
            alpha_bits: 0.1_f64.to_bits(),
            weighting: PropagationWeighting::DegreeNormalized,
            output: PropagationOutput::Final,
            dimensions: 384,
        };
        // The load-bearing case: swapping src/dst (the proven collision) must
        // move the hash — under the old `edge_source: String` descriptor both
        // recorded the bare source id and collided.
        let swapped = GraphPropagationFields {
            edge_source: EdgeSourceBinding::Registered {
                source_id: "edges".into(),
                src_column: "to".into(),
                dst_column: "from".into(),
                type_column: None,
                weight_column: None,
                as_of_column: None,
            },
            ..base.clone()
        };
        assert_ne!(
            definition_hash(&graph_propagation_descriptor(&base), &no_model_env()).unwrap(),
            definition_hash(&graph_propagation_descriptor(&swapped), &no_model_env()).unwrap(),
            "swapping src/dst on the registered edge source must move the hash — \
             two propagations over the same source with swapped columns are different graphs"
        );
        assert_each_change_moves_hash(
            &base,
            &no_model_env(),
            graph_propagation_descriptor,
            &[
                ("src_column", |p| {
                    if let EdgeSourceBinding::Registered { src_column, .. } = &mut p.edge_source {
                        *src_column = "s".into();
                    }
                }),
                ("dst_column", |p| {
                    if let EdgeSourceBinding::Registered { dst_column, .. } = &mut p.edge_source {
                        *dst_column = "d".into();
                    }
                }),
                ("type_column", |p| {
                    if let EdgeSourceBinding::Registered { type_column, .. } = &mut p.edge_source {
                        *type_column = Some("etype".into());
                    }
                }),
                ("weight_column", |p| {
                    if let EdgeSourceBinding::Registered { weight_column, .. } = &mut p.edge_source
                    {
                        *weight_column = Some("w".into());
                    }
                }),
                ("as_of_column", |p| {
                    if let EdgeSourceBinding::Registered { as_of_column, .. } = &mut p.edge_source {
                        *as_of_column = Some("valid_at".into());
                    }
                }),
            ],
        );
    }

    #[derive(Clone)]
    struct ContextSetFields {
        embedding_table: Option<String>,
        candidate_source: ContextCandidateSource,
        value_columns: Vec<String>,
        aggregator: ContextAggregator,
        exclude_self: bool,
        split: Option<String>,
        dimensions: usize,
    }

    fn context_set_descriptor(p: &ContextSetFields) -> ProducingDescriptor {
        ProducingDescriptor::ContextSet {
            encoder_id: "jammi:context-set".into(),
            source_id: "patents".into(),
            embedding_table: p.embedding_table.clone(),
            candidate_source: p.candidate_source.clone(),
            value_columns: p.value_columns.clone(),
            aggregator: p.aggregator,
            exclude_self: p.exclude_self,
            split: p.split.clone(),
            dimensions: p.dimensions,
        }
    }

    #[test]
    fn context_set_each_param_moves_the_hash() {
        let base = ContextSetFields {
            embedding_table: None,
            candidate_source: ContextCandidateSource::Ann { k: 5 },
            value_columns: vec!["label".into()],
            aggregator: ContextAggregator::Mean,
            exclude_self: true,
            split: None,
            dimensions: 32,
        };
        assert_each_change_moves_hash(
            &base,
            &no_model_env(),
            context_set_descriptor,
            &[
                ("embedding_table", |p| {
                    p.embedding_table = Some("pinned_source_table".into())
                }),
                ("candidate_source.k", |p| {
                    p.candidate_source = ContextCandidateSource::Ann { k: 9 }
                }),
                ("candidate_source.kind", |p| {
                    p.candidate_source = ContextCandidateSource::Edges {
                        gather: ContextEdgeGather {
                            edge_source: EdgeSourceBinding::NeighborGraph {
                                table_name: "g".into(),
                            },
                            hops: 1,
                            fanout: None,
                            direction: PropagationDirection::Out,
                            edge_types: None,
                            min_weight_bits: None,
                            as_of: None,
                        },
                    }
                }),
                ("value_columns", |p| p.value_columns.push("extra".into())),
                ("aggregator", |p| p.aggregator = ContextAggregator::Max),
                ("exclude_self", |p| p.exclude_self = false),
                ("split", |p| p.split = Some("split = 'train'".into())),
                ("dimensions", |p| p.dimensions = 64),
            ],
        );
    }

    #[test]
    fn context_edge_gather_each_knob_moves_the_hash() {
        // The edge gather is a determinant set in its own right; flip each of its
        // knobs (with the gather embedded in a ContextSet) and assert the hash
        // moves, so a lossy gather mirror is caught too.
        let base = ContextSetFields {
            embedding_table: None,
            candidate_source: ContextCandidateSource::Edges {
                gather: ContextEdgeGather {
                    edge_source: EdgeSourceBinding::Registered {
                        source_id: "edges".into(),
                        src_column: "from".into(),
                        dst_column: "to".into(),
                        type_column: None,
                        weight_column: None,
                        as_of_column: None,
                    },
                    hops: 1,
                    fanout: None,
                    direction: PropagationDirection::Out,
                    edge_types: None,
                    min_weight_bits: None,
                    as_of: None,
                },
            },
            value_columns: Vec::new(),
            aggregator: ContextAggregator::Mean,
            exclude_self: true,
            split: None,
            dimensions: 32,
        };
        let with_gather = |mutate: fn(&mut ContextEdgeGather)| {
            let mut f = base.clone();
            if let ContextCandidateSource::Edges { gather } = &mut f.candidate_source {
                mutate(gather);
            }
            f
        };
        let base_hash = definition_hash(&context_set_descriptor(&base), &no_model_env()).unwrap();
        let cases: &[LabelledMutation<ContextEdgeGather>] = &[
            ("hops", |g| g.hops = 3),
            ("fanout", |g| g.fanout = Some(8)),
            ("direction", |g| g.direction = PropagationDirection::In),
            ("edge_types", |g| g.edge_types = Some(vec!["cites".into()])),
            ("min_weight", |g| {
                g.min_weight_bits = Some(0.5_f64.to_bits())
            }),
            ("as_of", |g| g.as_of = Some("2026-01-01".into())),
            ("edge_source", |g| {
                g.edge_source = EdgeSourceBinding::NeighborGraph {
                    table_name: "g".into(),
                }
            }),
        ];
        for (label, mutate) in cases {
            let f = with_gather(*mutate);
            assert_ne!(
                base_hash,
                definition_hash(&context_set_descriptor(&f), &no_model_env()).unwrap(),
                "changing gather `{label}` must change the definition hash"
            );
        }
    }

    fn asof_descriptor(direction: AsofDirection) -> ProducingDescriptor {
        ProducingDescriptor::AsofJoin {
            spine: "spine".into(),
            facts: "facts".into(),
            spine_by: vec!["acct".into()],
            facts_by: vec!["acct".into()],
            spine_time: "ts".into(),
            facts_time: "ts".into(),
            direction,
            boundary: AsofBoundary::Inclusive,
            tolerance: None,
            tie_break_column: None,
            project: vec!["px".into()],
        }
    }

    #[test]
    fn asof_join_each_knob_moves_the_hash() {
        let env = no_model_env();
        let base_d = asof_descriptor(AsofDirection::Backward);
        let base = definition_hash(&base_d, &env).unwrap();

        // direction
        assert_ne!(
            base,
            definition_hash(&asof_descriptor(AsofDirection::Forward), &env).unwrap()
        );

        // boundary, tolerance, tie-break, project, keys — flip each on a clone.
        let variants: Vec<(&str, ProducingDescriptor)> = vec![
            ("boundary", {
                let mut d = base_d.clone();
                if let ProducingDescriptor::AsofJoin { boundary, .. } = &mut d {
                    *boundary = AsofBoundary::Exclusive;
                }
                d
            }),
            ("tolerance", {
                let mut d = base_d.clone();
                if let ProducingDescriptor::AsofJoin { tolerance, .. } = &mut d {
                    *tolerance = Some(AsofTolerance::Steps(3));
                }
                d
            }),
            ("tie_break_column", {
                let mut d = base_d.clone();
                if let ProducingDescriptor::AsofJoin {
                    tie_break_column, ..
                } = &mut d
                {
                    *tie_break_column = Some("seq".into());
                }
                d
            }),
            ("project", {
                let mut d = base_d.clone();
                if let ProducingDescriptor::AsofJoin { project, .. } = &mut d {
                    project.push("py".into());
                }
                d
            }),
            ("spine_by", {
                let mut d = base_d.clone();
                if let ProducingDescriptor::AsofJoin { spine_by, .. } = &mut d {
                    spine_by.push("region".into());
                }
                d
            }),
        ];
        for (label, d) in variants {
            assert_ne!(
                base,
                definition_hash(&d, &env).unwrap(),
                "changing asof `{label}` must change the definition hash"
            );
        }
    }

    #[test]
    fn unpinned_inputs_are_reported() {
        let manifest = MaterializationManifest::compute(
            &embedding_descriptor(),
            &cpu_env(),
            vec![
                InputAnchor::mutable_version("pinned", 3),
                InputAnchor::unpinned_at_instant("federated", "2026-06-17T00:00:00Z"),
            ],
            ArtifactDigest::of_bytes(b"x"),
            "run".into(),
            "2026-06-17T00:00:00Z".into(),
        )
        .unwrap();
        assert_eq!(manifest.unpinned_inputs(), vec!["federated".to_string()]);
    }

    #[test]
    fn artifact_digest_is_content_addressed() {
        assert_eq!(
            ArtifactDigest::of_bytes(b"same"),
            ArtifactDigest::of_bytes(b"same")
        );
        assert_ne!(
            ArtifactDigest::of_bytes(b"a"),
            ArtifactDigest::of_bytes(b"b")
        );
    }
}
