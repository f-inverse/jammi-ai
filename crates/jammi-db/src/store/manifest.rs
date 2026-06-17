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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model_task::ModelTask;

/// Manifest format version. A change to the [`ProducingDescriptor`] shape — the
/// determinant set a producer folds into its [`DefinitionHash`] — bumps this so
/// a reader detects an incompatible older manifest as a typed
/// [`ManifestError::UnsupportedManifestVersion`] rather than comparing a stale
/// hash computed over a different determinant set. The hash is opaque in the
/// persisted manifest (the descriptor is folded away, not stored), so a version
/// mismatch is the *only* signal that an on-disk hash is incomparable: a reader
/// must reject it, never silently trust it.
pub const MANIFEST_VERSION: u32 = 2;

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
/// Carries the engine semantic version, the compute device, and the identities
/// and backend kinds of every model the producer invokes — the determinants of
/// the output that are *not* part of the producing description itself.
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

/// The identity + backend kind of a model an environment invoked. The canonical
/// model id (HF repo or local path string) plus the backend kind that ran it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Canonical model id as stored in `result_tables.model_id`.
    pub model_id: String,
    /// The backend kind that ran the model (`candle` / `ort` / `http`).
    pub backend: String,
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
    /// itself output-affecting); and `exact_max_rows` is the ceiling that gates
    /// the exact driver. (The `resolve_keys` flag is *not* recorded: today a
    /// resolved endpoint equals its `_row_id` either way, so it does not affect
    /// the output — recording it would be a false determinant.)
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
        /// The edge relation the propagation read, by id — the second input
        /// anchor's source. A staleness/lineage determinant independent of the
        /// kernel knobs: the same knobs over a different graph yield a different
        /// output.
        edge_source: String,
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
    pub edge_source: ContextEdgeSource,
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

/// Which edge relation a context gather walked, recorded in a
/// [`ContextEdgeGather`] — the transport-neutral mirror of the AI crate's
/// `EdgeSourceRef`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "edge_source", rename_all = "snake_case")]
pub enum ContextEdgeSource {
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
        let definition_hash = definition_hash(descriptor, env)?;
        Ok(Self {
            artifact,
            definition_hash,
            input_anchors: inputs,
            produced_by,
            produced_at,
            engine_version: env.engine_version.clone(),
            manifest_version: MANIFEST_VERSION,
        })
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
    /// The persisted manifest carries an *opaque* [`DefinitionHash`] — the
    /// [`ProducingDescriptor`] is folded into it, not stored — so the manifest's
    /// JSON shape does not change when the descriptor's determinant set changes.
    /// That makes the version the *only* signal that an on-disk hash was computed
    /// over a different determinant set and is therefore incomparable. The
    /// version is validated **before** the parsed manifest is handed back, so a
    /// reader never silently trusts a stale hash: any version that is not exactly
    /// [`MANIFEST_VERSION`] — older (a different, now-superseded determinant set)
    /// or newer (a format this build cannot read) — is a typed
    /// [`ManifestError::UnsupportedManifestVersion`], the signal to re-emit.
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
            }],
        );
        assert_ne!(base, definition_hash(&d, &other_model).unwrap());
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

    #[test]
    fn older_manifest_version_is_rejected_cleanly() {
        // An old (v1) manifest's opaque definition hash was computed over a
        // now-superseded descriptor determinant set, so it is incomparable.
        // The reader must reject it as a typed error — the signal to re-emit —
        // never a panic, never a silent stale-hash match. (v1 predates this
        // build's MANIFEST_VERSION = 2.)
        assert_eq!(
            MANIFEST_VERSION, 2,
            "this guard assumes v1 is the prior format"
        );
        assert!(matches!(
            MaterializationManifest::from_json_bytes(&manifest_at_version(1)),
            Err(ManifestError::UnsupportedManifestVersion { found: 1, supported })
                if supported == MANIFEST_VERSION
        ));
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
            ],
        );
    }

    #[derive(Clone)]
    struct GraphPropagationFields {
        edge_source: String,
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
            edge_source: "graph".into(),
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
                ("edge_source", |p| p.edge_source = "other_graph".into()),
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

    #[derive(Clone)]
    struct ContextSetFields {
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
                ("candidate_source.k", |p| {
                    p.candidate_source = ContextCandidateSource::Ann { k: 9 }
                }),
                ("candidate_source.kind", |p| {
                    p.candidate_source = ContextCandidateSource::Edges {
                        gather: ContextEdgeGather {
                            edge_source: ContextEdgeSource::NeighborGraph {
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
            candidate_source: ContextCandidateSource::Edges {
                gather: ContextEdgeGather {
                    edge_source: ContextEdgeSource::Registered {
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
                g.edge_source = ContextEdgeSource::NeighborGraph {
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
