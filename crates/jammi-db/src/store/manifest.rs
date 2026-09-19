//! The materialization contract: the verifiable identity every result table
//! carries so a later reader can assert **"this artifact is the output of
//! definition D over input-state S"** — without trusting a name, a path, or an
//! out-of-band convention.
//!
//! A result table is published as an immutable Parquet object (plus, for
//! embedding tables, an ANN-index sidecar bundle). This module adds a *separate*
//! `.materialization.json` sidecar — written for **every** result table, not
//! only embedding tables — carrying an in-toto-shaped attestation that binds
//! three things to the artifact's content digest (and, beside the digest, a
//! keyed inventory of the artifact's parts — [`LeafDigest`] per row group or
//! bundle file — so one partition can be verified without the rest):
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
///
/// **An ADDITIVE new [`ProducingDescriptor`] variant does not bump this
/// number** — [`ProducingDescriptor::TrainingSet`] landed without one. A
/// variant an older reader's `enum` definition does not know is rejected
/// serde-first, the same way a since-added FIELD is: deserializing an
/// unrecognised tagged-enum variant fails before `manifest_version` is even
/// read, so an older build can never silently misinterpret a newer
/// descriptor shape as one of its own known variants. What DOES bump this
/// number is a change to an EXISTING variant's determinant set (a field
/// added, removed, or reinterpreted within a variant the older reader
/// already knows) — that shape still deserializes under the old
/// definition, so nothing else would catch the older reader comparing a
/// stale hash computed over a different determinant set.
pub const MANIFEST_VERSION: u32 = 3;

/// The row-order rule version 1 of the training-set producer commits and
/// records in [`ProducingDescriptor::TrainingSet::order_rule`]: the rows are
/// ordered by the **full projected tuple** — every projected column, in
/// declared order, ascending, NULLs first.
///
/// Ordering by the full tuple rather than by a key column is what makes the
/// order a total function of the trainable data: the only rows that can tie
/// are byte-identical on every projected column, and such rows are
/// interchangeable for any consumer of the table (the reader cannot tell one
/// from the other), so the committed order is a pure function of the row
/// multiset even though a tie group's internal permutation is not pinned. No
/// engine-wide stable row identity exists on an arbitrary registered source,
/// so the projected tuple is the strongest total key available.
///
/// A versioned tag, not a bare `true`: a later rule (a different direction, a
/// different NULL placement, an added tie-break column) is a *different*
/// definition and must take a new tag, so a table committed under v1 is never
/// silently read as if it carried the newer order.
pub const TRAINING_SET_ORDER_RULE_V1: &str = "full_tuple_v1";

/// The graph fine-tune arm's read-order rule:
/// [`ProducingDescriptor::GraphTrainingSet::read_order_rule`]'s versioned
/// tag — the node scan ordered by `(id, text)` and the edge scan ordered by
/// `(src, dst)`, both ascending NULLS FIRST, so the sampler's input is a
/// function of the node/edge SET rather than of scan order. A versioned tag
/// for the same reason [`TRAINING_SET_ORDER_RULE_V1`] is: a later rule is a
/// *different* definition and must take a new tag.
pub const GRAPH_READ_ORDER_RULE_V1: &str = "graph_read_order_v1";

/// The leading column every [`ProducingDescriptor::GraphTrainingSet`] table
/// is written with: the sampler's own emission order,
/// ascending, assigned once per row at write time. Unlike
/// [`ProducingDescriptor::TrainingSet`], this variant carries no generic
/// `columns` list to derive a declared file sort order from (its fields are
/// the four named column bindings, not a projection) — every graph
/// training-set writer commits this ONE column name by convention, so a
/// reader (or a ListingTable registration declaring the file's order) names
/// it directly rather than re-deriving it from the descriptor.
pub const GRAPH_TRAINING_SET_ORDINAL_COLUMN: &str = "_ordinal";

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

/// The device KIND, discarding the ordinal — the determinant `jammi-ballista`'s
/// `InferenceExec::device_kind` and `JammiExecutionEngine`'s device refusal
/// compare against. Ordinals are never compared: a plan built on CUDA
/// ordinal 0 runs on an executor whose only CUDA device is ordinal 1 (the
/// ordinal is not output-affecting, and the codec carries none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDeviceKind {
    /// CPU.
    Cpu,
    /// A CUDA device, any ordinal.
    Cuda,
    /// An Apple Metal device, any ordinal.
    Metal,
}

impl ComputeDevice {
    /// This device's kind, discarding the ordinal.
    pub fn kind(&self) -> ComputeDeviceKind {
        match self {
            ComputeDevice::Cpu => ComputeDeviceKind::Cpu,
            ComputeDevice::Cuda { .. } => ComputeDeviceKind::Cuda,
            ComputeDevice::Metal { .. } => ComputeDeviceKind::Metal,
        }
    }
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
    /// The fused-kernel admission profile the producer's compute path ran
    /// under — a canonical, opaque string tag the producer maps its own
    /// `jammi-kernels` admission decision to (`jammi-db` depends on no jammi
    /// crate but `jammi-numerics`, so it cannot hold `jammi-kernels`' typed
    /// admission report directly; this is the SAME "db-local primitive
    /// standing in for a foreign type" shape [`ProducingDescriptor::TrainingSet::format`]
    /// already uses). A different admission outcome (fused vs. eager) can
    /// change the bits a training run produces at the same nominal
    /// precision, so it is a determinant of the output like the compute
    /// device and every invoked model's identity.
    ///
    /// **Residual, not fixed here:** [`Self::device`] folds only
    /// `ComputeDevice::Cuda { ordinal }` (an ordinal, never the GPU's
    /// actual compute capability), while
    /// `jammi_kernels::admission::flash_validated_arches`'s pod-parity gate
    /// admits or refuses an arch based on that REAL capability — two CUDA
    /// runs on different-architecture GPUs (differing compute capability,
    /// same ordinal convention) can therefore hash identically despite a
    /// genuine hardware-driven admission difference neither `device` nor
    /// this field folds.
    ///
    /// `None` for every producer that records no kernel-admission decision
    /// (every variant before [`ProducingDescriptor::FineTune`]).
    /// `#[serde(skip_serializing_if = "Option::is_none")]` means a `None`
    /// value serialises to no JSON key at all — the same hash-preservation
    /// contract [`ModelIdentity::quantization`] keeps — so this field's
    /// addition changes not one byte of any [`DefinitionHash`] computed
    /// before it existed.
    ///
    /// **Populated by the `FineTune` producer.** `jammi-db`
    /// cannot itself compute this value (it depends on no `jammi-kernels`
    /// type), so `jammi-ai`'s fine-tune worker builds the string via
    /// `jammi_kernels::admission::render_kernel_admission_profile` and hands
    /// it across through [`Self::with_kernel_admission_profile`] — the same
    /// "db-local primitive standing in for a foreign type" shape this
    /// field's own doc already describes. Every fact the rendered string
    /// carries is EX ANTE (known before training runs: this crate's own
    /// compiled build features, `admission_mode`, the disabled-op set, and
    /// the job's OWN declared training dtype — `FineTuneConfig::backbone_dtype`,
    /// never the base model's loaded `compute_precision()` (a different
    /// axis — see `render_kernel_admission_profile`'s own doc) — never a
    /// per-run OBSERVATION of
    /// which ops actually dispatched fused, which would make the value
    /// unknowable at the point a [`DefinitionHash`] is computed — before
    /// training runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_admission_profile: Option<String>,
}

impl MaterializationEnv {
    /// Build the environment for the current engine version and the given
    /// device + invoked models. [`Self::kernel_admission_profile`] starts
    /// `None`; set it with [`Self::with_kernel_admission_profile`].
    pub fn new(device: ComputeDevice, models: Vec<ModelIdentity>) -> Self {
        Self {
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            device,
            models,
            kernel_admission_profile: None,
        }
    }

    /// Record the fused-kernel admission profile the producer's compute path
    /// ran under (see [`Self::kernel_admission_profile`]'s doc for why this
    /// is an opaque tag rather than a typed `jammi-kernels` value).
    pub fn with_kernel_admission_profile(mut self, profile: impl Into<String>) -> Self {
        self.kernel_admission_profile = Some(profile.into());
        self
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
/// `content_digest` folds in the SAME way, for the SAME reason:
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

/// A model's content digest — the [`ModelIdentity`] determinant for the
/// model's bytes: a `model_id` string alone does not change when the referenced
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

/// [`ProducingDescriptor::GraphTrainingSet::sample`]'s fields — a db-local
/// mirror of `jammi-ai`'s `GraphSampleConfig` minus `min_negatives` (not
/// output-affecting; see the variant's own doc). `walk_length`,
/// `walks_per_node`, `hard_negatives` and `exclude_hops` widen from
/// `GraphSampleConfig`'s `usize` to `u64` for a hash fold whose width does
/// not depend on the compiling target; `return_p`/`in_out_q` fold as
/// `f64::to_bits()`, never the `f64` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSampleFields {
    /// Seed for the walk/negative RNG.
    pub seed: u64,
    /// node2vec walk length `L`.
    pub walk_length: u64,
    /// Walks started per node.
    pub walks_per_node: u64,
    /// node2vec return parameter `p`, as `f64::to_bits()`.
    pub return_p_bits: u64,
    /// node2vec in-out parameter `q`, as `f64::to_bits()`.
    pub in_out_q_bits: u64,
    /// Structure-aware hard negatives mined per pair (`0` = the `pairs`
    /// format; `>= 1` = `triplet`).
    pub hard_negatives: u64,
    /// Hops of the anchor's neighbourhood excluded from its negative pool.
    pub exclude_hops: u64,
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
    /// Training-set output: the rows a training run reads, projected from a
    /// source relation and committed in one canonical order as an immutable
    /// Parquet table of kind
    /// [`ResultTableKind::TrainingSet`](crate::catalog::result_repo::ResultTableKind::TrainingSet).
    /// ([`ResultStore::materialize_training_set`](crate::store::ResultStore::materialize_training_set).)
    ///
    /// The determinant set is exactly what changes the committed BYTES and
    /// their order: the source relation the rows were projected from, the
    /// projected columns in declared order, the model task those columns are
    /// read as, the training format the consumer parses them under, and the
    /// order rule the write committed. Nothing about *how a run consumes the
    /// table* is recorded — not the world size, not the per-rank batch, not
    /// the validation split, not the topology: those partition and slice a
    /// table that is already fixed, so folding them in would fragment one
    /// shared artifact into a per-run copy while changing not one committed
    /// byte. That omission is what lets runs of different shapes share one
    /// table — the definition half of the reuse key; the recorded input
    /// anchors must match too, and be pinned
    /// ([`ResultStore::materialize_training_set`](crate::store::ResultStore::materialize_training_set)).
    ///
    /// `format` is a **canonical string tag**, not the consuming crate's
    /// format type: `jammi-db` depends on no jammi crate but `jammi-numerics`,
    /// so the owning crate maps its own enum to a stable tag and this
    /// descriptor folds the tag. The completeness burden moves with it: a
    /// format distinction the tag does not spell is two different tables
    /// colliding on one hash, which is the mapping's contract to keep (the
    /// same burden [`Self::External`]'s `params` carries).
    TrainingSet {
        /// The source relation the rows were projected from, as the canonical
        /// text the producer actually ran — the query, never merely a source
        /// id. Two different projections, filters or joins over one registered
        /// source are two different training sets, and recording only the id
        /// would collide them on a single hash.
        source: String,
        /// The projected columns, in declared order. Also the order key (see
        /// `order_rule`), so their declared order is itself output-affecting:
        /// the same column set declared differently commits a different row
        /// order.
        columns: Vec<String>,
        /// The model task the projected columns are read as — the task decides
        /// which columns are inputs and which are targets, so it changes what
        /// the same bytes mean to a consumer.
        task: ModelTask,
        /// The training format the consumer parses the rows under, as its
        /// canonical string tag: the same columns under two formats are two
        /// different training sets.
        format: String,
        /// The row-order rule the write committed, as a versioned tag —
        /// [`TRAINING_SET_ORDER_RULE_V1`] today. A change to how the producer
        /// orders rows changes the committed byte order, so the rule is part
        /// of the definition and takes a NEW tag rather than silently
        /// re-ordering the rows an existing hash already names.
        order_rule: String,
    },
    /// A graph fine-tune's training set: the contrastive
    /// pairs a biased-walk sampler drew from a node-text source and an
    /// edge-table source, committed as a [`Self::TrainingSet`]-kind result
    /// table through the SAME materialisation funnel — but this variant, not
    /// [`Self::TrainingSet`], because the graph arm's row source is not a
    /// single SQL projection: it is TWO relations (nodes, edges) plus a
    /// sampling procedure, and `Self::TrainingSet::source` has no field wide
    /// enough to carry a walk configuration honestly.
    ///
    /// The determinant set is exactly what changes the committed BYTES: the
    /// two source relations and the four column bindings that resolve which
    /// of their columns feed the sampler, the model task, the format tag, the
    /// walk/negative-sampling knobs (`sample`), and the read-order rule the
    /// node/edge scans committed. `min_negatives` (a
    /// `GraphSampleConfig` field) is deliberately ABSENT from `sample`: it
    /// changes only how many rows a *successful* sample is allowed to have
    /// been willing to produce (a config-time validation ceiling on the
    /// negative pool), never which rows a successful sample actually emits:
    /// two runs differing ONLY in `min_negatives` emit byte-identical output
    /// over the same graph. `provenance`
    /// (`GraphFineTuneSources::provenance`) is likewise ABSENT: `GraphSampler
    /// ::sample` never reads it (only `has_declared_supervision`, an
    /// informational report, does), so it cannot change the sampled bytes
    /// either.
    ///
    /// `sample` mirrors `jammi-ai`'s `GraphSampleConfig` as a db-local
    /// primitive standing in for a foreign type — `jammi-db` depends on no
    /// jammi crate but `jammi-numerics` (crate-layering rule), the same constraint
    /// [`Self::TrainingSet::format`]'s canonical string tag and
    /// [`Self::FineTune::spec_canonical`]'s opaque JSON already accommodate.
    /// `return_p`/`in_out_q` fold as their IEEE-754 bit patterns
    /// (`f64::to_bits`) rather than the `f64` itself — a fixed, exact,
    /// byte-stable fold, never a float compared/hashed directly.
    GraphTrainingSet {
        /// Catalog source holding the node text.
        node_source: String,
        /// Catalog source holding the edges.
        edge_source: String,
        /// Column in `node_source` holding the node id.
        id_column: String,
        /// Column in `node_source` holding the node text.
        text_column: String,
        /// Column in `edge_source` holding the edge source endpoint.
        src_column: String,
        /// Column in `edge_source` holding the edge destination endpoint.
        dst_column: String,
        /// The model task the sampled rows are read as.
        task: ModelTask,
        /// The training format the consumer parses the rows under
        /// (`pairs` / `triplet`), as its canonical string tag — decided from
        /// `sample.hard_negatives`.
        format: String,
        /// The node2vec walk and structure-aware negative-sampling knobs.
        sample: GraphSampleFields,
        /// The read-order rule the node/edge scans committed —
        /// [`GRAPH_READ_ORDER_RULE_V1`] today.
        read_order_rule: String,
    },
    /// A trained model's output: a base model fine-tuned over a materialised
    /// [`Self::TrainingSet`], keyed under the catalog name
    /// `jammi:fine-tuned:{job_id}` (the handle and re-claim idempotency key).
    /// Recorded on the `models` row via the `model_materialization` migration
    /// (`models.definition_hash` / `input_anchors_json`, mirroring the
    /// `result_tables` columns migration 021 added); the `.materialization.json`
    /// sidecar path is derived from the artifact prefix, never recorded as a
    /// third column. Replay
    /// (`pipeline::recompute`) for this variant is **retrain**, never a
    /// re-derivation from the recorded fields.
    ///
    /// `jammi-db` depends on no jammi crate but `jammi-numerics`
    /// (crate-layering rule), so the two foreign types this
    /// variant would otherwise need to hold directly — `jammi-wire`'s
    /// `FineTuneConfig` and `jammi-ai`'s `TrainingSpec`/`TrainingCommon` —
    /// never appear here. Instead `spec_canonical` (below) is an OPAQUE,
    /// versioned canonical (sorted-key JSON) encoding of the **whole**
    /// `TrainingSpec::FineTune` variant (base model, the full
    /// `FineTuneConfig` — LoRA rank/alpha/dropout, `use_rslora`,
    /// `rank_pattern`, `init_lora_weights`, the training-time `backbone_dtype`,
    /// the deterministic `seed`, losses, `matryoshka_dims`,
    /// `quantile_levels`, `validation_fraction`, early stopping, `cached`,
    /// `hard_negatives`, …, and `TrainingCommon::world_size`), produced by
    /// `jammi-ai` from the owning types — the same "db-local primitive
    /// standing in for a foreign type" shape `TrainingSet::format` (above)
    /// already uses. Completeness of that whole-variant serialization (a new
    /// `FineTuneConfig`/`TrainingCommon` field never silently escaping it) is
    /// the exhaustive-destructuring completeness test in `jammi-wire` /
    /// `jammi-ai` — this variant's own completeness test instead covers
    /// every field named directly below.
    ///
    /// The base model's full identity (backend, compute precision, content
    /// digest, quantization) folds in via [`MaterializationEnv::models`] —
    /// the SAME uniform path [`Self::Embedding`]/[`Self::Inference`] already
    /// use for the model they invoke — so `base_model_id` (below) is the
    /// db-local mirror of `env.models[0].model_id`, not a second identity.
    /// The fused-kernel admission profile is an environment fact, not a
    /// spec knob: the `FineTune` producer calls
    /// [`MaterializationEnv::with_kernel_admission_profile`] with the
    /// EX ANTE string `jammi_kernels::admission::render_kernel_admission_profile`
    /// renders (build features, `admission_mode`, the disabled-op set, the
    /// job's OWN declared training dtype — `FineTuneConfig::backbone_dtype`,
    /// never the base model's loaded `compute_precision()`), so a build/policy difference that changes which
    /// ops CAN fuse is a determinant of this variant's [`DefinitionHash`] —
    /// see [`MaterializationEnv::kernel_admission_profile`]'s own doc for
    /// what is and is not folded. **Known limitation:** every OTHER
    /// model-invoking producer (`Self::Embedding`/`Self::Inference` —
    /// e.g. `jammi-ai`'s `pipeline::embedding::embedding_definition`)
    /// builds its `MaterializationEnv` with `kernel_admission_profile:
    /// None`; the field is populated ONLY for `FineTune`, so any
    /// admission-gated dispatch those other producers make is not a
    /// `DefinitionHash` determinant for them.
    ///
    /// `world_size` (`TrainingCommon`'s own topology field) and `collective`/
    /// `local_ranks` (below) are this variant's topology fields. The
    /// completeness test's exhaustive destructuring (no `..`) fails to
    /// compile when a field is added until it is bound and mutated, so the
    /// determinant set can never silently grow unaccounted-for.
    FineTune {
        /// The training-set table's own [`DefinitionHash`], hex — binds this
        /// fine-tune to the EXACT materialised
        /// [`ProducingDescriptor::TrainingSet`] definition it trained from,
        /// not merely "a table with this name" (which could be silently
        /// replaced by a later re-materialization under reuse rules a
        /// bare name could never detect).
        training_set_definition_hash: String,
        /// The training-set table's [`ArtifactDigest`], hex — the exact
        /// content the fine-tune consumed, distinct from the definition hash
        /// (two materialisations of the same definition over advanced inputs
        /// share no digest even though nothing about *how* they were built
        /// differs).
        training_set_artifact_digest: String,
        /// The training-set table's committed row count — an idle-seeming
        /// field that is nonetheless output-affecting: the same digest could
        /// only arise from one row count, but recording it directly (rather
        /// than requiring a reader to re-open the Parquet footer) makes the
        /// determinant self-contained in the descriptor, matching every
        /// other variant's "the descriptor alone states what changed the
        /// bytes" contract.
        training_set_row_count: u64,
        /// Sorted-key canonical JSON of the whole `TrainingSpec::FineTune`
        /// variant — see the variant doc above. Opaque to `jammi-db`.
        spec_canonical: String,
        /// The schema version `spec_canonical` (above) was encoded under —
        /// bumped by the owning crate whenever a field is added to or
        /// removed from the encoded shape, so a reader never compares two
        /// canonical strings encoded under different, silently-incompatible
        /// shapes as if they were the same kind of value.
        spec_schema_version: u32,
        /// The base model's canonical id — the db-local mirror of
        /// `env.models[0].model_id` (see the variant doc above); the full
        /// identity (backend, precision, content digest, quantization)
        /// folds in via [`MaterializationEnv::models`], uniformly with every
        /// other model-invoking variant.
        base_model_id: String,
        /// Data-parallel rank count this fine-tune trained over
        /// (`TrainingCommon::world_size`) — the summation order
        /// and batch layout depend on it, so two runs of the same spec at
        /// different world sizes are two different definitions.
        world_size: u32,
        /// The collective this run reduced over — a canonical, lowercase
        /// token (`"noop"`, `"local"`, `"peer"`, `"nccl"`), the db-local
        /// primitive standing in for `jammi-ai`'s `Collective` trait
        /// implementations (crate-layering rule: this crate
        /// depends on no jammi crate). Two otherwise-identical specs
        /// reduced over a DIFFERENT collective can disagree in their last
        /// bits (fold order, transport-specific rounding), so this is a
        /// determinant like every other field here, not metadata.
        collective: String,
        /// How many ranks THIS HOST ran on its own devices for this run —
        /// read off the topology the run executed at, never off a
        /// configured capacity: one for a single rank, the whole gang for
        /// an in-process (`"local"`) gang, one for the coordinator of a
        /// multi-host (`"peer"`) gang whose other ranks live elsewhere.
        /// Orthogonal to `world_size` above (the per-job identity field) and
        /// recorded alongside it, never instead of it.
        local_ranks: u32,
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
    /// An incremental refresh of an embedding table: the rows whose content
    /// hash changed (or were added) since `parent_version`, re-embedded with
    /// the same embedding parameters, plus the deletion-mask horizons for
    /// superseded and deleted keys. Recorded only in a `.version.json`
    /// (never in `.materialization.json`); replayed by a full
    /// `EmbeddingPipeline::run` over the current source into a NEW table.
    EmbeddingDelta {
        model_id: String,
        task: ModelTask,
        source_id: String,
        columns: Vec<String>,
        key_column: String,
        dimensions: usize,
        /// The version refreshed from.
        parent_version: i64,
        /// The parent's identity (the chain link).
        parent_identity: String,
        /// What a key the source no longer has became.
        deletes: DeletePolicy,
    },
    /// A compaction of a versioned embedding table: every live row of
    /// `parent_version` rewritten as one fragment + one segment, no inference.
    /// Recorded only in a `.version.json`; replayed like `EmbeddingDelta`.
    EmbeddingCompaction {
        model_id: String,
        task: ModelTask,
        source_id: String,
        columns: Vec<String>,
        key_column: String,
        dimensions: usize,
        /// The version compacted.
        parent_version: i64,
        /// The parent's identity (the chain link).
        parent_identity: String,
    },
}

/// What an incremental refresh does with a key the source no longer has.
/// Recorded in the delta descriptor (it is output-affecting).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeletePolicy {
    /// The key is masked out of every prior fragment and segment (the
    /// default: a table keeping rows its source lost is not "D over S").
    #[default]
    Tombstone,
    /// The key's current row is kept; a rolling-window table is the
    /// consumer's knowledge.
    Retain,
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
    /// The chain link of a version descriptor: the parent's identity for an
    /// `EmbeddingDelta` / `EmbeddingCompaction`, `None` for a base descriptor.
    pub fn parent_identity(&self) -> Option<&str> {
        match self {
            Self::EmbeddingDelta {
                parent_identity, ..
            }
            | Self::EmbeddingCompaction {
                parent_identity, ..
            } => Some(parent_identity.as_str()),
            _ => None,
        }
    }

    /// The columns a training-set table was committed in order of, or `None`
    /// for a descriptor that is not a training set. Producers are plural; the
    /// reader is one: everything that reads a training set asks this, never
    /// which producer wrote it.
    pub fn training_set_order_columns(&self) -> Option<Vec<String>> {
        match self {
            Self::TrainingSet { columns, .. } => Some(columns.clone()),
            Self::GraphTrainingSet { .. } => {
                Some(vec![GRAPH_TRAINING_SET_ORDINAL_COLUMN.to_string()])
            }
            _ => None,
        }
    }

    /// Canonical bytes for hashing: a JSON encoding with object keys sorted, so
    /// the byte stream is independent of struct field declaration order and
    /// stable across serde versions. Pure; no I/O.
    pub(crate) fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        let value = serde_json::to_value(self)
            .map_err(|e| ManifestError::UncanonicalDescriptor(e.to_string()))?;
        let canonical = canonicalize_json(&value);
        serde_json::to_vec(&canonical)
            .map_err(|e| ManifestError::UncanonicalDescriptor(e.to_string()))
    }

    /// Build a [`Self::TrainingSet`] descriptor — the tabular fine-tune arm's
    /// identity, at [`TRAINING_SET_ORDER_RULE_V1`]. A named constructor
    /// rather than deriving this inline at each call site: `TrainingSetSpec`
    /// takes ANY [`Self`] (a `Batches` caller needs
    /// [`Self::graph_training_set`] instead), and every production caller
    /// building the tabular identity goes through this one function so the
    /// two verbs' shapes cannot silently drift from each other.
    pub fn training_set(
        source: impl Into<String>,
        columns: Vec<String>,
        task: ModelTask,
        format: impl Into<String>,
    ) -> Self {
        Self::TrainingSet {
            source: source.into(),
            columns,
            task,
            format: format.into(),
            order_rule: TRAINING_SET_ORDER_RULE_V1.to_string(),
        }
    }

    /// Build a [`Self::GraphTrainingSet`] descriptor — the graph fine-tune
    /// arm's identity, at [`GRAPH_READ_ORDER_RULE_V1`].
    #[allow(clippy::too_many_arguments)]
    pub fn graph_training_set(
        node_source: impl Into<String>,
        edge_source: impl Into<String>,
        id_column: impl Into<String>,
        text_column: impl Into<String>,
        src_column: impl Into<String>,
        dst_column: impl Into<String>,
        task: ModelTask,
        format: impl Into<String>,
        sample: GraphSampleFields,
    ) -> Self {
        Self::GraphTrainingSet {
            node_source: node_source.into(),
            edge_source: edge_source.into(),
            id_column: id_column.into(),
            text_column: text_column.into(),
            src_column: src_column.into(),
            dst_column: dst_column.into(),
            task,
            format: format.into(),
            sample,
            read_order_rule: GRAPH_READ_ORDER_RULE_V1.to_string(),
        }
    }
}

/// The materialization contract a producer supplies for one result table — the
/// producing description, the output-affecting environment, and the resolved
/// input anchors, grouped so the single funnel
/// ([`crate::store::BuildingTable::finish`]) takes one value
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

/// A recorded materialization a reuse probe may match against a request: a
/// catalog row carrying the anchor set its bytes were produced over.
pub(crate) trait ReuseCandidate {
    /// The recorded input anchors as canonical JSON, or `None` for a row that
    /// records no materialization (never a match).
    fn recorded_anchors_json(&self) -> Option<&str>;
    /// Canonical-stamp creation time — the newest-first sort key.
    fn created_at(&self) -> &str;
    /// The row's unique name — the tie-break when two rows share a stamp.
    fn name(&self) -> &str;
}

/// A requested anchor set that can match a recorded one: every anchor in it
/// is a reproducible id. The engine's one reuse predicate is this type — its
/// constructor is the refusal of an unpinned request, [`Self::matches`] the
/// exact-set comparison and the order — so a probe that runs inside a catalog
/// transaction and one that fetches its candidates asynchronously
/// ([`exact_reuse_matches`]) decide identically.
#[derive(Debug, Clone)]
pub(crate) struct PinnedAnchors(Vec<InputAnchor>);

impl PinnedAnchors {
    /// `requested` as a matchable set, or `None` when it holds any
    /// [`AnchorKind::UnpinnedAtInstant`] anchor: an instant is not a
    /// reproducible id, so equal instants do not prove equal inputs and such
    /// a request matches nothing.
    pub(crate) fn of(requested: &[InputAnchor]) -> Option<Self> {
        requested
            .iter()
            .all(|a| a.kind != AnchorKind::UnpinnedAtInstant)
            .then(|| Self(requested.to_vec()))
    }

    /// Of `candidates` (the rows sharing the requested definition hash),
    /// those whose recorded anchor set EQUALS this one, newest first.
    ///
    /// - Anchors compare as a SET — a producer's declaration order is
    ///   incidental. A source appears at most once in a producer's anchor
    ///   set, so a length check plus containment in each direction is exact.
    /// - The order is the total key `(created_at DESC, name DESC)`, imposed
    ///   here rather than trusted from a catalog `ORDER BY`, so two rows
    ///   stamped in the same microsecond still resolve to one deterministic
    ///   winner.
    pub(crate) fn matches<C: ReuseCandidate>(
        &self,
        candidates: Vec<C>,
    ) -> Result<Vec<C>, serde_json::Error> {
        let mut exact = Vec::new();
        for candidate in candidates {
            let Some(anchors_json) = candidate.recorded_anchors_json() else {
                continue;
            };
            let recorded: Vec<InputAnchor> = serde_json::from_str(anchors_json)?;
            if anchor_sets_equal(&recorded, &self.0) {
                exact.push(candidate);
            }
        }
        exact.sort_by(|a, b| {
            b.created_at()
                .cmp(a.created_at())
                .then_with(|| b.name().cmp(a.name()))
        });
        Ok(exact)
    }
}

/// [`PinnedAnchors`] over candidates fetched on demand: `fetch` (the rows
/// sharing the requested definition hash) is never run for an unpinned
/// request.
pub(crate) async fn exact_reuse_matches<C, F, Fut>(
    requested: &[InputAnchor],
    fetch: F,
) -> crate::error::Result<Vec<C>>
where
    C: ReuseCandidate,
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = crate::error::Result<Vec<C>>>,
{
    match PinnedAnchors::of(requested) {
        Some(pinned) => Ok(pinned.matches(fetch().await?)?),
        None => Ok(Vec::new()),
    }
}

fn anchor_sets_equal(a: &[InputAnchor], b: &[InputAnchor]) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x)) && b.iter().all(|y| a.contains(y))
}

/// The immutable state pointer of one input, encoded per its [`AnchorKind`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnchorValue(pub String);

/// What one leaf of a [`MaterializationManifest`]'s inventory names — a
/// KEYED part of the artifact, never a position (adding a file to a bundle
/// renumbers nothing; a row group's key carries its own byte range).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LeafKey {
    /// A Parquet row group: its index in the footer and the byte range
    /// `[offset, offset + length)` of the file its column chunks occupy
    /// (from the footer's own column-chunk offsets — the dictionary page
    /// first when there is one).
    RowGroup {
        index: u32,
        offset: u64,
        length: u64,
    },
    /// A file of a model bundle, by its name in the bundle manifest.
    File { name: String },
}

/// One leaf of the inventory: a keyed part of the artifact and the SHA-256
/// of exactly that part. A peer verifies one partition against its leaf
/// without reading the rest; the whole-object [`ArtifactDigest`] stays the
/// subject every verifier matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeafDigest {
    pub key: LeafKey,
    pub digest: ArtifactDigest,
}

/// The leaf inventory of a Parquet object: one [`LeafKey::RowGroup`] leaf per
/// row group, in footer order, each digest over that row group's byte range.
/// Read from the footer the object itself carries, so a verifier with the
/// bytes recomputes exactly this (that is what `verify_partitions` does).
/// Bytes outside every row group — the magic, the footer, page indexes,
/// bloom filters — belong to no leaf; they are covered by the whole-object
/// digest, never by the inventory.
///
/// # Errors
///
/// [`ManifestError::ParquetFooter`] when the bytes carry no readable footer
/// or a row group's range falls outside them.
pub fn parquet_leaves(bytes: &[u8]) -> Result<Vec<LeafDigest>, ManifestError> {
    use parquet::file::metadata::ParquetMetaDataReader;
    let owned = bytes::Bytes::copy_from_slice(bytes);
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(&owned)
        .map_err(|e| ManifestError::ParquetFooter(e.to_string()))?;
    let mut leaves = Vec::with_capacity(metadata.num_row_groups());
    for (index, row_group) in metadata.row_groups().iter().enumerate() {
        let mut start = u64::MAX;
        let mut end = 0u64;
        for column in row_group.columns() {
            let (offset, length) = column.byte_range();
            start = start.min(offset);
            end = end.max(offset + length);
        }
        if start == u64::MAX {
            // A row group with no column chunks: an empty range at the end
            // of the previous one.
            start = end;
        }
        let range = usize::try_from(start)
            .ok()
            .zip(usize::try_from(end).ok())
            .filter(|(s, e)| s <= e && *e <= bytes.len())
            .ok_or_else(|| {
                ManifestError::ParquetFooter(format!(
                    "row group {index} names bytes [{start}, {end}) outside the {}-byte object",
                    bytes.len()
                ))
            })?;
        leaves.push(LeafDigest {
            key: LeafKey::RowGroup {
                index: u32::try_from(index).map_err(|_| {
                    ManifestError::ParquetFooter(format!("row group index {index} overflows u32"))
                })?,
                offset: start,
                length: end - start,
            },
            digest: ArtifactDigest::of_bytes(&bytes[range.0..range.1]),
        });
    }
    Ok(leaves)
}

/// The verdict of `verify_partitions`: every leaf of the inventory
/// recomputed from the bytes and compared by key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum PartitionVerdict {
    /// Every leaf recomputes to its recorded digest.
    Match,
    /// The table has no sidecar (pre-contract, or pre-`leaves`).
    MissingManifest,
    /// The first leaf whose recomputed digest differs — named by its key so
    /// a consumer knows WHICH partition is not the attested one.
    Mismatch {
        key: LeafKey,
        expected: String,
        found: String,
    },
    /// The bytes' own inventory has a different shape from the recorded
    /// one (a row group added or removed): no per-leaf comparison is
    /// meaningful.
    InventoryDiffers { expected: usize, found: usize },
}

/// The attestation written beside every materialised table. Shaped after an
/// in-toto statement: a `subject` (the artifact digest) plus a predicate
/// (everything about how it was produced), so a consumer verifies by digest
/// match then evaluates the predicate against its own policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationManifest {
    /// in-toto subject: digest of the Parquet artifact this manifest attests to.
    pub artifact: ArtifactDigest,
    /// The keyed inventory of the artifact's parts ([`LeafDigest`]): one leaf
    /// per Parquet row group for a result table ([`parquet_leaves`]), one
    /// per file for a model bundle. ADDITIVE to `artifact` — a peer verifies
    /// one partition against its leaf; the whole-object digest above stays
    /// the subject, the root of the version-identity chain, and what every
    /// verifier holding the bytes recomputes. REQUIRED: a sidecar without it
    /// was written before this field existed and reads as absent (see
    /// [`Self::from_json_bytes`]).
    pub leaves: Vec<LeafDigest>,
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
        leaves: Vec<LeafDigest>,
        produced_by: String,
        produced_at: String,
    ) -> Result<Self, ManifestError> {
        let definition_hash = Self::definition_of(descriptor, env)?;
        Ok(Self {
            artifact,
            leaves,
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
        let manifest: Self = match serde_json::from_slice(bytes) {
            Ok(m) => m,
            Err(shape) => {
                // The ONE shape rejection that is a known past format, not
                // corruption: an object at the CURRENT version with no
                // `leaves` — a sidecar written before the inventory existed.
                // Named, so a reader can treat exactly that as "no sidecar"
                // (re-materialise) while every other rejection — garbage, a
                // missing determinant field, another version — stays the
                // error it is.
                if let Ok(serde_json::Value::Object(object)) =
                    serde_json::from_slice::<serde_json::Value>(bytes)
                {
                    let at_current_version = object
                        .get("manifest_version")
                        .and_then(serde_json::Value::as_u64)
                        == Some(u64::from(MANIFEST_VERSION));
                    if at_current_version && !object.contains_key("leaves") {
                        return Err(ManifestError::PreLeavesSidecar);
                    }
                }
                return Err(ManifestError::Serde(shape));
            }
        };
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
    /// A sidecar at the current version with no `leaves` inventory — written
    /// before the inventory existed. The reader treats it as absent
    /// (re-materialise); it is never a hit and never a crash.
    #[error(
        "manifest sidecar predates the leaf inventory (no `leaves` field); re-emit the artifact"
    )]
    PreLeavesSidecar,
    /// The object's Parquet footer could not be read, so no inventory can be
    /// derived from (or verified against) its bytes.
    #[error("parquet footer unreadable: {0}")]
    ParquetFooter(String),
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

    /// HASH-PRESERVATION GOLDEN: `quantization: None`
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

    /// HASH-PRESERVATION GOLDEN: the end-to-end `definition_hash` for the
    /// representative `embedding_descriptor()` / `cpu_env()` fixture (the
    /// same fixture `definition_hash_is_deterministic` above uses) equals the
    /// value this fixture hashes to without any `ModelIdentity::quantization`
    /// key — so adding that `None` field changes no existing hash. Never
    /// update the literal to match a new computed value; that would silently
    /// rubber-stamp a migration. A migration is a new `MANIFEST_VERSION` and a
    /// fresh golden.
    ///
    /// `engine_version` is a hash input BY DESIGN
    /// (`different_engine_version_changes_the_hash` below pins that
    /// sensitivity), and `cpu_env()` stamps it from `CARGO_PKG_VERSION`. The
    /// golden was computed at engine version `0.48.0`, so this test — and ONLY
    /// this test, never `cpu_env()` itself, which other tests rely on for the
    /// CURRENT engine version — pins `engine_version` to `0.48.0` before
    /// hashing; otherwise every lockstep version bump would move it, which is
    /// not the migration this golden exists to catch.
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
             engine_version 0.48.0) drifted from the golden value — the \
             `quantization: None` fold must be byte-identical to the shape without the key, \
             not a silent migration"
        );
    }

    /// DISTINCTNESS: identical env except
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

    /// Serde round-trip: a pre-feature JSON row — one
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

    /// `compute_precision` folds into `ModelIdentity` (part of
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

    /// `content_digest` folds into `ModelIdentity` (part of
    /// `MaterializationEnv`) the same way `compute_precision` does — two
    /// identities that differ ONLY in the model's content digest (same
    /// `model_id`, `backend`, `compute_precision`) must never collide on one
    /// `DefinitionHash`: a `model_id` string alone does not change when the referenced
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
             collide on one DefinitionHash"
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
            vec![],
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
            vec![],
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
            vec![],
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
            vec![],
            "run".into(),
            "2026-06-17T00:00:00Z".into(),
        )
        .unwrap();
        assert_eq!(manifest.unpinned_inputs(), vec!["federated".to_string()]);
    }

    /// Every field of [`ProducingDescriptor::TrainingSet`], carried as a
    /// fixture whose shape the completeness test below destructures WITHOUT
    /// `..`, so a field added to the variant fails to compile here instead of
    /// silently escaping the definition hash.
    #[derive(Clone)]
    struct TrainingSetFields {
        source: String,
        columns: Vec<String>,
        task: ModelTask,
        format: String,
        order_rule: String,
    }

    fn training_set_descriptor(f: &TrainingSetFields) -> ProducingDescriptor {
        // Exhaustive construction: no `..`, so the fixture and the variant
        // stay in lock-step.
        let TrainingSetFields {
            source,
            columns,
            task,
            format,
            order_rule,
        } = f.clone();
        ProducingDescriptor::TrainingSet {
            source,
            columns,
            task,
            format,
            order_rule,
        }
    }

    /// A base fixture whose every field is a NON-default, distinguishable
    /// value: a mutation test over a fixture of defaults passes vacuously
    /// exactly where the identity is lossy.
    fn training_set_fields() -> TrainingSetFields {
        TrainingSetFields {
            source: "SELECT \"q\", \"a\" FROM jammi.support_tickets WHERE \"lang\" = 'en'".into(),
            columns: vec!["q".into(), "a".into()],
            task: ModelTask::TextEmbedding,
            format: "pairs".into(),
            order_rule: TRAINING_SET_ORDER_RULE_V1.to_string(),
        }
    }

    #[test]
    fn training_set_hash_is_deterministic() {
        let env = no_model_env();
        let f = training_set_fields();
        assert_eq!(
            definition_hash(&training_set_descriptor(&f), &env).unwrap(),
            definition_hash(&training_set_descriptor(&f), &env).unwrap(),
            "the same training-set definition must hash identically"
        );
    }

    /// Hash completeness: the field set the assertions below range over is the
    /// variant's own, taken by exhaustive destructuring (no `..`) — a new
    /// field breaks this test's compilation, which is the point.
    #[test]
    fn training_set_every_field_moves_the_hash() {
        // The `let` below is the enumeration of record: adding a field to the
        // variant fails to compile here until it is bound and mutated.
        let TrainingSetFields {
            source: _,
            columns: _,
            task: _,
            format: _,
            order_rule: _,
        } = training_set_fields();

        assert_each_change_moves_hash(
            &training_set_fields(),
            &no_model_env(),
            training_set_descriptor,
            &[
                ("source", |f| {
                    f.source = "SELECT \"q\", \"a\" FROM jammi.support_tickets".into()
                }),
                // A different column SET.
                ("columns", |f| f.columns.push("lang".into())),
                // …and the same set in a different ORDER: the columns are the
                // order key, so their declared order is output-affecting on
                // its own.
                ("columns order", |f| f.columns.reverse()),
                ("task", |f| f.task = ModelTask::Classification),
                ("format", |f| f.format = "triplets".into()),
                ("order_rule", |f| f.order_rule = "full_tuple_v2".into()),
            ],
        );
    }

    /// The device is part of the environment the hash folds, so the same
    /// training-set definition materialised on two devices is two identities —
    /// the environment leg of hash completeness for this variant, which the
    /// descriptor-only mutations above cannot show.
    #[test]
    fn training_set_hash_moves_with_the_device() {
        let d = training_set_descriptor(&training_set_fields());
        assert_ne!(
            definition_hash(&d, &MaterializationEnv::new(ComputeDevice::Cpu, Vec::new())).unwrap(),
            definition_hash(
                &d,
                &MaterializationEnv::new(ComputeDevice::Cuda { ordinal: 0 }, Vec::new())
            )
            .unwrap(),
        );
    }

    /// Every field of [`ProducingDescriptor::GraphTrainingSet`], carried as a
    /// fixture whose shape the completeness test below destructures WITHOUT
    /// `..`, so a field added to the variant fails to compile here instead of
    /// silently escaping the definition hash.
    #[derive(Clone)]
    struct GraphTrainingSetFields {
        node_source: String,
        edge_source: String,
        id_column: String,
        text_column: String,
        src_column: String,
        dst_column: String,
        task: ModelTask,
        format: String,
        sample: GraphSampleFields,
        read_order_rule: String,
    }

    fn graph_training_set_descriptor(f: &GraphTrainingSetFields) -> ProducingDescriptor {
        // Exhaustive construction: no `..`, so the fixture and the variant
        // stay in lock-step.
        let GraphTrainingSetFields {
            node_source,
            edge_source,
            id_column,
            text_column,
            src_column,
            dst_column,
            task,
            format,
            sample,
            read_order_rule,
        } = f.clone();
        ProducingDescriptor::GraphTrainingSet {
            node_source,
            edge_source,
            id_column,
            text_column,
            src_column,
            dst_column,
            task,
            format,
            sample,
            read_order_rule,
        }
    }

    /// A base fixture whose every field is a NON-default, distinguishable
    /// value: a mutation test over a fixture of defaults passes vacuously
    /// exactly where the identity is lossy.
    fn graph_training_set_fields() -> GraphTrainingSetFields {
        GraphTrainingSetFields {
            node_source: "kb_nodes".into(),
            edge_source: "kb_edges".into(),
            id_column: "id".into(),
            text_column: "text".into(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            task: ModelTask::TextEmbedding,
            format: "triplet".into(),
            sample: GraphSampleFields {
                seed: 42,
                walk_length: 4,
                walks_per_node: 2,
                return_p_bits: 1.0_f64.to_bits(),
                in_out_q_bits: 1.0_f64.to_bits(),
                hard_negatives: 2,
                exclude_hops: 1,
            },
            read_order_rule: GRAPH_READ_ORDER_RULE_V1.to_string(),
        }
    }

    #[test]
    fn graph_training_set_hash_is_deterministic() {
        let env = no_model_env();
        let f = graph_training_set_fields();
        assert_eq!(
            definition_hash(&graph_training_set_descriptor(&f), &env).unwrap(),
            definition_hash(&graph_training_set_descriptor(&f), &env).unwrap(),
            "the same graph training-set definition must hash identically"
        );
    }

    /// Hash completeness: the field set the assertions below range over is the
    /// variant's own, taken by exhaustive destructuring (no `..`) — a new
    /// field breaks this test's compilation, which is the point. Every
    /// mutation here is the "change one sample knob -> different hash"
    /// oracle, over the whole field table, executed per field.
    #[test]
    fn graph_training_set_every_field_moves_the_hash() {
        // The `let` below is the enumeration of record: adding a field to the
        // variant fails to compile here until it is bound and mutated.
        let GraphTrainingSetFields {
            node_source: _,
            edge_source: _,
            id_column: _,
            text_column: _,
            src_column: _,
            dst_column: _,
            task: _,
            format: _,
            sample: _,
            read_order_rule: _,
        } = graph_training_set_fields();

        assert_each_change_moves_hash(
            &graph_training_set_fields(),
            &no_model_env(),
            graph_training_set_descriptor,
            &[
                ("node_source", |f| f.node_source = "kb_nodes_v2".into()),
                ("edge_source", |f| f.edge_source = "kb_edges_v2".into()),
                ("id_column", |f| f.id_column = "node_id".into()),
                ("text_column", |f| f.text_column = "body".into()),
                ("src_column", |f| f.src_column = "from".into()),
                ("dst_column", |f| f.dst_column = "to".into()),
                ("task", |f| f.task = ModelTask::Classification),
                ("format", |f| f.format = "pairs".into()),
                ("sample.seed", |f| f.sample.seed = 7),
                ("sample.walk_length", |f| f.sample.walk_length = 8),
                ("sample.walks_per_node", |f| f.sample.walks_per_node = 6),
                ("sample.return_p_bits", |f| {
                    f.sample.return_p_bits = 2.0_f64.to_bits()
                }),
                ("sample.in_out_q_bits", |f| {
                    f.sample.in_out_q_bits = 0.5_f64.to_bits()
                }),
                ("sample.hard_negatives", |f| f.sample.hard_negatives = 5),
                ("sample.exclude_hops", |f| f.sample.exclude_hops = 2),
                ("read_order_rule", |f| {
                    f.read_order_rule = "graph_read_order_v2".into()
                }),
            ],
        );
    }

    /// `min_negatives` (a `GraphSampleConfig` field) has no home in
    /// [`GraphSampleFields`]: two runs differing ONLY in `min_negatives`
    /// emit byte-identical sampled output over the same graph, so it is not
    /// output-affecting and folding it in would collide two identical
    /// tables' definitions apart for no reason.
    /// Pinned here as the negative space `graph_training_set_every_field_
    /// moves_the_hash` cannot show (there is no `min_negatives` field to
    /// mutate): two descriptors built from otherwise-identical `sample`
    /// fields hash identically, full stop — the type simply carries no knob
    /// that could vary it.
    #[test]
    fn graph_training_set_hash_has_no_min_negatives_field() {
        let a = graph_training_set_descriptor(&graph_training_set_fields());
        let b = graph_training_set_descriptor(&graph_training_set_fields());
        assert_eq!(
            definition_hash(&a, &no_model_env()).unwrap(),
            definition_hash(&b, &no_model_env()).unwrap()
        );
    }

    /// The device is part of the environment the hash folds — the
    /// environment leg of hash completeness for this variant.
    #[test]
    fn graph_training_set_hash_moves_with_the_device() {
        let d = graph_training_set_descriptor(&graph_training_set_fields());
        assert_ne!(
            definition_hash(&d, &MaterializationEnv::new(ComputeDevice::Cpu, Vec::new())).unwrap(),
            definition_hash(
                &d,
                &MaterializationEnv::new(ComputeDevice::Cuda { ordinal: 0 }, Vec::new())
            )
            .unwrap(),
        );
    }

    /// Every field [`ProducingDescriptor::FineTune`] currently carries,
    /// captured as a fixture whose shape the completeness test below
    /// destructures WITHOUT `..` — a field added to the variant (e.g. new
    /// distributed-training topology fields) fails to compile here instead
    /// of silently escaping the definition hash.
    #[derive(Clone)]
    struct FineTuneFields {
        training_set_definition_hash: String,
        training_set_artifact_digest: String,
        training_set_row_count: u64,
        spec_canonical: String,
        spec_schema_version: u32,
        base_model_id: String,
        world_size: u32,
        collective: String,
        local_ranks: u32,
    }

    fn fine_tune_descriptor(f: &FineTuneFields) -> ProducingDescriptor {
        // Exhaustive construction: no `..`, so the fixture and the variant
        // stay in lock-step.
        let FineTuneFields {
            training_set_definition_hash,
            training_set_artifact_digest,
            training_set_row_count,
            spec_canonical,
            spec_schema_version,
            base_model_id,
            world_size,
            collective,
            local_ranks,
        } = f.clone();
        ProducingDescriptor::FineTune {
            training_set_definition_hash,
            training_set_artifact_digest,
            training_set_row_count,
            spec_canonical,
            spec_schema_version,
            base_model_id,
            world_size,
            collective,
            local_ranks,
        }
    }

    /// A base fixture whose every field is a NON-default, distinguishable
    /// value: a mutation test over a fixture of defaults passes vacuously
    /// exactly where the identity is lossy.
    fn fine_tune_fields() -> FineTuneFields {
        FineTuneFields {
            training_set_definition_hash: "a".repeat(64),
            training_set_artifact_digest: "b".repeat(64),
            training_set_row_count: 4096,
            spec_canonical: r#"{"base_model":"bert-base","config":{"lora_rank":8}}"#.into(),
            spec_schema_version: 1,
            base_model_id: "bert-base-uncased".into(),
            world_size: 2,
            collective: "local".into(),
            local_ranks: 2,
        }
    }

    fn base_model_identity() -> ModelIdentity {
        ModelIdentity {
            model_id: "bert-base-uncased".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("fine-tune-fixture-digest".into()),
            quantization: None,
        }
    }

    #[test]
    fn fine_tune_hash_is_deterministic() {
        let env = env_with_model(base_model_identity());
        let f = fine_tune_fields();
        assert_eq!(
            definition_hash(&fine_tune_descriptor(&f), &env).unwrap(),
            definition_hash(&fine_tune_descriptor(&f), &env).unwrap(),
            "the same fine-tune definition must hash identically"
        );
    }

    /// Hash completeness: the field set the assertions below range over is the
    /// variant's own, taken by exhaustive destructuring (no `..`) — a new
    /// field breaks this test's compilation, which is the point.
    #[test]
    fn fine_tune_every_field_moves_the_hash() {
        // The `let` below is the enumeration of record: adding a field to the
        // variant fails to compile here until it is bound and mutated.
        let FineTuneFields {
            training_set_definition_hash: _,
            training_set_artifact_digest: _,
            training_set_row_count: _,
            spec_canonical: _,
            spec_schema_version: _,
            base_model_id: _,
            world_size: _,
            collective: _,
            local_ranks: _,
        } = fine_tune_fields();

        assert_each_change_moves_hash(
            &fine_tune_fields(),
            &env_with_model(base_model_identity()),
            fine_tune_descriptor,
            &[
                ("training_set_definition_hash", |f| {
                    f.training_set_definition_hash = "c".repeat(64)
                }),
                ("training_set_artifact_digest", |f| {
                    f.training_set_artifact_digest = "d".repeat(64)
                }),
                ("training_set_row_count", |f| {
                    f.training_set_row_count = 4097
                }),
                ("spec_canonical", |f| {
                    f.spec_canonical =
                        r#"{"base_model":"bert-base","config":{"lora_rank":16}}"#.into()
                }),
                ("spec_schema_version", |f| f.spec_schema_version = 2),
                ("base_model_id", |f| {
                    f.base_model_id = "distilbert-base-uncased".into()
                }),
                ("world_size", |f| f.world_size = 4),
                ("collective", |f| f.collective = "nccl".into()),
                ("local_ranks", |f| f.local_ranks = 4),
            ],
        );
    }

    /// The base model's full identity folds in uniformly via
    /// `MaterializationEnv::models`, exactly like `Embedding`/`Inference` —
    /// a different base model's content digest must change the hash even
    /// though `base_model_id` (the descriptor's own db-local mirror) stays
    /// the same string.
    #[test]
    fn fine_tune_hash_moves_with_the_base_model_content_digest() {
        let d = fine_tune_descriptor(&fine_tune_fields());
        let base = definition_hash(&d, &env_with_model(base_model_identity())).unwrap();
        let mut other = base_model_identity();
        other.content_digest = ModelContentDigest::Sha256("a-different-base-digest".into());
        assert_ne!(base, definition_hash(&d, &env_with_model(other)).unwrap());
    }

    /// The device is part of the environment the hash folds — the
    /// environment leg of hash completeness for this variant, which the
    /// descriptor-only mutations above cannot show.
    #[test]
    fn fine_tune_hash_moves_with_the_device() {
        let d = fine_tune_descriptor(&fine_tune_fields());
        let cpu = MaterializationEnv::new(ComputeDevice::Cpu, vec![base_model_identity()]);
        let cuda = MaterializationEnv::new(
            ComputeDevice::Cuda { ordinal: 0 },
            vec![base_model_identity()],
        );
        assert_ne!(
            definition_hash(&d, &cpu).unwrap(),
            definition_hash(&d, &cuda).unwrap(),
        );
    }

    /// HASH-PRESERVATION GOLDEN: `kernel_admission_profile: None` must
    /// serialise to no key at all, never a present `null` — the same
    /// contract `quantization_none_serialises_to_no_key` pins for
    /// `ModelIdentity`, restated for `MaterializationEnv` so this field's
    /// addition changes not one byte of any pre-existing `DefinitionHash`.
    #[test]
    fn kernel_admission_profile_none_serialises_to_no_key() {
        let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
        let value = serde_json::to_value(&env).unwrap();
        let object = value.as_object().unwrap();
        assert!(
            !object.contains_key("kernel_admission_profile"),
            "kernel_admission_profile: None must serialise to no key, got {value:#?}"
        );
    }

    /// A manifest with no `kernel_admission_profile` — no key at all in the JSON, modelling what is
    /// actually on disk from before this field existed — deserialises to `None` via
    /// `#[serde(default)]`, the same pre-feature-row contract
    /// `quantization_serde_round_trips_and_pre_feature_rows_default_to_none`
    /// pins for `ModelIdentity`.
    #[test]
    fn kernel_admission_profile_pre_feature_manifest_deserialises_to_none() {
        let pre_feature_json = serde_json::json!({
            "engine_version": "0.0.0",
            "device": "cpu",
            "models": [],
        });
        let env: MaterializationEnv = serde_json::from_value(pre_feature_json).unwrap();
        assert_eq!(env.kernel_admission_profile, None);
    }

    /// A row recorded WITHOUT a profile never matches (hashes
    /// identically to) a row recorded WITH one over an otherwise-identical
    /// environment — the mirror direction of
    /// `fine_tune_hash_moves_with_the_kernel_admission_profile`, stated
    /// explicitly as the "prior manifests stay readable but distinguishable"
    /// property.
    #[test]
    fn kernel_admission_profile_absent_never_hashes_equal_to_present() {
        let d = fine_tune_descriptor(&fine_tune_fields());
        let absent = env_with_model(base_model_identity());
        let present = MaterializationEnv::new(ComputeDevice::Cpu, vec![base_model_identity()])
            .with_kernel_admission_profile("layer_norm=enabled");
        assert_ne!(
            definition_hash(&d, &absent).unwrap(),
            definition_hash(&d, &present).unwrap(),
            "a manifest with no recorded profile must never hash equal to one that recorded one"
        );
    }

    /// DISTINCTNESS: a `Some` kernel-admission profile must hash differently
    /// from `None` over an otherwise-identical environment — a fused-kernel
    /// run is output-affecting relative to an eager run of the same spec.
    #[test]
    fn fine_tune_hash_moves_with_the_kernel_admission_profile() {
        let d = fine_tune_descriptor(&fine_tune_fields());
        let bare = env_with_model(base_model_identity());
        let fused = MaterializationEnv::new(ComputeDevice::Cpu, vec![base_model_identity()])
            .with_kernel_admission_profile("lora_linear_fused_v1");
        assert_ne!(
            definition_hash(&d, &bare).unwrap(),
            definition_hash(&d, &fused).unwrap(),
            "a fused-kernel admission profile must change the hash relative to none recorded"
        );
    }

    /// A hash-completeness test over the `kernel_admission_profile`
    /// FIELD itself — this proves `MaterializationEnv`'s hash folds every
    /// LINE of an arbitrary, line-shaped `String` value, not that this
    /// specific literal matches `jammi_kernels::admission::render_kernel_admission_profile`'s
    /// real output (`jammi-db` cannot depend on `jammi-kernels` — leaf-crate
    /// rule — so this uses hand-written literals; the renderer's OWN fact
    /// set and its per-dtype-correctness are asserted directly against the
    /// real function in `jammi-kernels/src/admission.rs`'s own test suite,
    /// e.g. `render_kernel_admission_profile_includes_every_ex_ante_fact`
    /// and `render_kernel_admission_profile_resolves_the_disabled_check_per_dtype_not_across_all_dtypes`,
    /// and end to end — a real published `DefinitionHash` moving under a
    /// real `JAMMI_KERNELS_DISABLE` — by `jammi-ai`'s
    /// `kernel_admission_profile_names_a_real_disabled_op_and_moves_the_definition_hash`).
    /// The literals below are kept in the renderer's CURRENT shape
    /// (`report_key=<disabled|enabled|n/a>`, `name=bool`) for plausibility,
    /// but this test's OWN claim is narrower: two envs differing in exactly
    /// one line hash differently, and reverting that line restores equality
    /// (the control half — proves the OTHER lines are not secretly what
    /// moved the hash) — a property of `MaterializationEnv`'s own fold,
    /// true for ANY such string, real renderer or not.
    #[test]
    fn kernel_admission_profile_field_is_hash_complete_over_any_line_shaped_content() {
        let baseline = "layer_norm=enabled\n\
                         rope=enabled\n\
                         cast_scale=n/a\n\
                         cuda=false\n\
                         flash-attn=false\n\
                         metal=false\n\
                         admission_mode=Fallback\n\
                         dtype_class=F32";
        let d = fine_tune_descriptor(&fine_tune_fields());
        let base_hash = definition_hash(
            &d,
            &env_with_model(base_model_identity()).with_kernel_admission_profile(baseline),
        )
        .unwrap();

        // One perturbation per line this literal names — each must move the
        // hash relative to `baseline`, and reverting it must restore
        // equality (ruling out "always different" as a vacuous pass).
        let perturbations: &[(&str, &str)] = &[
            ("a disabled-op row line", "layer_norm=disabled"),
            ("a different row's line", "rope=disabled"),
            ("an n/a row line", "cast_scale=disabled"),
            ("the cuda build fact", "cuda=true"),
            ("the flash-attn build fact", "flash-attn=true"),
            ("the metal build fact", "metal=true"),
            ("admission_mode", "admission_mode=Strict"),
            ("dtype_class", "dtype_class=Bf16"),
        ];
        for (name, replacement) in perturbations {
            let target = replacement.split('=').next().unwrap();
            let mutated: String = baseline
                .lines()
                .map(|line| {
                    if line.starts_with(&format!("{target}=")) {
                        (*replacement).to_string()
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            assert_ne!(
                mutated, baseline,
                "test bug: perturbation {name:?} did not change the literal string"
            );
            let mutated_hash = definition_hash(
                &d,
                &env_with_model(base_model_identity())
                    .with_kernel_admission_profile(mutated.as_str()),
            )
            .unwrap();
            assert_ne!(
                base_hash, mutated_hash,
                "changing {name} alone must change DefinitionHash"
            );

            // Control: reverting the SAME line restores hash equality —
            // the hash difference above is attributable to this one line,
            // not some other channel this test forgot to hold constant.
            let reverted_hash = definition_hash(
                &d,
                &env_with_model(base_model_identity()).with_kernel_admission_profile(baseline),
            )
            .unwrap();
            assert_eq!(
                base_hash, reverted_hash,
                "control: re-using the baseline string must reproduce the baseline hash"
            );
        }
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

    /// The leaf inventory: one leaf per row group in footer order,
    /// each the digest of exactly that row group's bytes as the footer
    /// itself locates them; a byte flipped inside row group k changes leaf
    /// k and no other; the whole-object digest is untouched by the
    /// inventory and still catches a footer-only mutation.
    mod leaves {
        use std::sync::Arc;

        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use parquet::file::metadata::ParquetMetaDataReader;
        use parquet::file::properties::WriterProperties;

        use super::super::{
            parquet_leaves, ArtifactDigest, LeafKey, ManifestError, MaterializationManifest,
            MANIFEST_VERSION,
        };
        use super::{cpu_env, embedding_descriptor};

        fn three_row_group_parquet() -> Vec<u8> {
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from((0..6).collect::<Vec<i64>>()))],
            )
            .unwrap();
            let props = WriterProperties::builder()
                .set_max_row_group_row_count(Some(2))
                .build();
            let mut out = Vec::new();
            let mut writer = ArrowWriter::try_new(&mut out, schema, Some(props)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
            out
        }

        #[test]
        fn one_leaf_per_row_group_in_footer_order_each_the_footers_byte_range_digest() {
            let bytes = three_row_group_parquet();
            let footer = ParquetMetaDataReader::new()
                .parse_and_finish(&bytes::Bytes::from(bytes.clone()))
                .unwrap();
            assert_eq!(
                footer.num_row_groups(),
                3,
                "the fixture must span row groups"
            );
            let leaves = parquet_leaves(&bytes).unwrap();
            assert_eq!(leaves.len(), footer.num_row_groups());
            for (i, (leaf, rg)) in leaves.iter().zip(footer.row_groups()).enumerate() {
                // The oracle reads the footer ITSELF, not the leaf.
                let (start, end) = rg.columns().iter().fold((u64::MAX, 0), |(s, e), c| {
                    let (o, l) = c.byte_range();
                    (s.min(o), e.max(o + l))
                });
                let expected = ArtifactDigest::of_bytes(&bytes[start as usize..end as usize]);
                assert_eq!(leaf.digest, expected, "leaf {i}");
                assert_eq!(
                    leaf.key,
                    LeafKey::RowGroup {
                        index: i as u32,
                        offset: start,
                        length: end - start
                    }
                );
            }
        }

        #[test]
        fn a_byte_flipped_inside_row_group_k_changes_leaf_k_only_and_a_footer_flip_changes_no_leaf()
        {
            let bytes = three_row_group_parquet();
            let before = parquet_leaves(&bytes).unwrap();
            let LeafKey::RowGroup { offset, length, .. } = before[1].key.clone() else {
                panic!("row-group leaf");
            };
            let mut tampered = bytes.clone();
            let at = (offset + length / 2) as usize;
            tampered[at] ^= 0xff;
            let after = parquet_leaves(&tampered).unwrap();
            assert_ne!(after[1].digest, before[1].digest, "leaf 1 changed");
            assert_eq!(after[0], before[0], "leaf 0 untouched");
            assert_eq!(after[2], before[2], "leaf 2 untouched");
            assert_ne!(
                ArtifactDigest::of_bytes(&tampered),
                ArtifactDigest::of_bytes(&bytes)
            );
            // A footer-only mutation (inside the metadata, outside every row
            // group) changes no leaf — the whole-object digest is the
            // subject that catches it.
            let LeafKey::RowGroup {
                offset: last_off,
                length: last_len,
                ..
            } = before[2].key.clone()
            else {
                panic!("row-group leaf");
            };
            let footer_at = (last_off + last_len) as usize + 4;
            let mut footered = bytes.clone();
            footered[footer_at] ^= 0x01;
            assert_ne!(
                ArtifactDigest::of_bytes(&footered),
                ArtifactDigest::of_bytes(&bytes)
            );
            if let Ok(leaves) = parquet_leaves(&footered) {
                assert_eq!(leaves, before, "no leaf covers the footer");
            }
        }

        #[test]
        fn a_pre_leaves_sidecar_is_a_typed_pre_leaves_rejection_and_nothing_else_is() {
            let manifest = MaterializationManifest::compute(
                &embedding_descriptor(),
                &cpu_env(),
                vec![],
                ArtifactDigest::of_bytes(b"x"),
                vec![],
                "run".into(),
                "2026-06-17T00:00:00Z".into(),
            )
            .unwrap();
            let mut value = serde_json::to_value(&manifest).unwrap();
            value.as_object_mut().unwrap().remove("leaves");
            let pre = serde_json::to_vec(&value).unwrap();
            assert!(matches!(
                MaterializationManifest::from_json_bytes(&pre),
                Err(ManifestError::PreLeavesSidecar)
            ));
            // A NEWER version without leaves is never a pre-leaves miss (an
            // older binary must not re-materialise over it).
            let mut newer = serde_json::to_value(&manifest).unwrap();
            let object = newer.as_object_mut().unwrap();
            object.remove("leaves");
            object.insert(
                "manifest_version".into(),
                serde_json::json!(MANIFEST_VERSION + 1),
            );
            assert!(!matches!(
                MaterializationManifest::from_json_bytes(&serde_json::to_vec(&newer).unwrap()),
                Err(ManifestError::PreLeavesSidecar)
            ));
            // Garbage stays a serde error.
            assert!(matches!(
                MaterializationManifest::from_json_bytes(b"not json"),
                Err(ManifestError::Serde(_))
            ));
        }
    }
}
