# SPEC-02 — Materialization contract

> Part of the [60 — temporal correctness](./README.md) plan group. Independent of [`SPEC-01-asof-join.md`](./SPEC-01-asof-join.md) except at the shared result-store write path (see [`README.md` — Concurrent-session strategy](./README.md)); pinned to land first. Research rules: [`../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md`](../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md).
>
> **Status:** Implemented (the integration assumptions below were corrected against this engine during a pressure-test — there is no LogicalPlan-centric single funnel; see §3.3, §6, §6A).

## 1. Goal

A *materialization contract* is the verifiable identity a materialised table carries so that any later reader can assert **"this artifact is the output of definition D over input-state S"** — without trusting a name, a path, or an out-of-band convention. The engine already writes a result table as an immutable Parquet object plus, for embedding tables, an ANN-index sidecar set (`crates/jammi-db/src/store/`: `ResultTableInfo { table_name, parquet_url, index_url }`, whose layout helpers append `.usearch`/`.rowmap`/`.manifest.json`). This spec adds a *separate* `.materialization.json` sidecar — written for **every** result table, not only embedding tables — carrying a signed-shaped attestation that binds three things to the table's content digest: a **content hash of the producing description (the verb + its typed parameters) and the environment that affects its output** (engine version, compute device, invoked-model identities), the **immutable as-of anchors of every input** the producer read, and the **producing-run identity and instant**. It adds one verb, `verify_materialization`, that recomputes the artifact digest and reports whether it matches a caller-supplied expectation. The primitive carries only what every reproducibility-minded consumer needs; it ships **no** policy — what a reader *does* with a mismatch (refuse, alarm, fall back) is the reader's concern, not the engine's.

This is a seam, not a serving tier. The contract makes "the served value matches the definition that trained it, as of T" a checkable fact at a boundary; it does not build the boundary. That keeps the engine on the right side of [`../../PHILOSOPHY.md`](../../PHILOSOPHY.md): the engine ships the export/attestation contract; the KV store, the online tier, and the refuse-to-serve policy live in the consumer's composition.

## 2. Implementation shape

Single-capability change, landed atomically across `jammi-db` + `jammi-ai` + `jammi-wire` + `jammi-server` + PyO3 + the remote client + conformance on one branch.

1. **Contract** (`crates/jammi-db/src/store/manifest.rs`) — `ProducingDescriptor`, `MaterializationEnv` (with `ComputeDevice` + `ModelIdentity`), `Materialization` (the producer's `{descriptor, env, inputs}` bundle), `InputAnchor`/`AnchorKind`/`AnchorValue`, `DefinitionHash`, `ArtifactDigest`, `MaterializationManifest`, `MatchVerdict`, `ManifestError`. The definition hash and sidecar (de)serialisation are pure functions here.
2. **Funnel** (`store/mod.rs`) — `finalize_with_manifest` replaces the bare `finalize` (deleted) as the single `building -> ready` transition; `verify_materialization`, `result_digest_anchor`, manifest sidecar read/write; recovery (`recover_inner` + `reconcile_ready_manifests`) made manifest-aware. A `test-hooks` checkpoint (`maybe_signal_materialization`) marks the crash window.
3. **Catalog** — migration `021` (`definition_hash TEXT`, `input_anchors_json TEXT` on `result_tables`); `promote_result_table_with_manifest` flips status + persists the summary columns atomically; `ResultTableRecord` carries the two columns.
4. **Producers** — the five result-table producers (`infer`, the embedding pipeline, the neighbor-graph derivation, `propagate_embeddings`, `materialize_context`) each build their `ProducingDescriptor` + `MaterializationEnv` (engine version + effective compute device via `InferenceSession::compute_device` + invoked-model identities) + `InputAnchor`s and route through the funnel.
5. **Verb** — `Session::verify_materialization` → PyO3 `PyDatabase` binding → gRPC `CatalogService.VerifyMaterialization` + `RemoteDatabase` stub → `_PIPELINE_VERBS` in `test_conformance.py`.
6. **Tests** — `tests/it/materialization.rs` (the §11 verdict oracle + funnel + recovery), `tests/it/materialization_crash_recovery.rs` (the SIGKILL manifest-window crash), the `manifest.rs` unit tests (determinism / sensitivity / device / round-trip), and the migration-021 column test.

> **Note on the original plan (corrected during pressure-test).** The draft assumed a `LogicalPlan`-centric single funnel with a `store/plan_canonical.rs` and a `MaterializationManifest::compute(plan, …)`. This engine has no such plan for result-table producers and no pre-existing single funnel — both were corrected (§3.3, §6, §6A): the definition input is a `ProducingDescriptor`, and the funnel was *built* by replacing `finalize`.

## 3. Public API surface (exhaustive)

### 3.1 Identities (`store/manifest.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Content hash of *how* a table was produced: a canonical producing descriptor
/// plus the environment that affects its output. SHA-256, hex-encoded.
///
/// "Environment" is deliberately broad — the engine semantic version, the
/// identity (canonical name + version) of every model the plan invokes, and
/// the backend kinds of the inputs. The Bazel lesson is explicit here: a hash
/// that omits part of the execution environment yields false "matches" when
/// that hidden part changes[^bazel]. We hash the plan *and* its world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DefinitionHash(String);

/// Content digest of the materialised artifact itself (the Parquet object).
/// SHA-256 over the object bytes. This is the in-toto "subject" — the thing a
/// verifier matches by digest, "regardless of content type", treating the
/// subject as immutable[^intoto].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArtifactDigest(String);
```

### 3.2 Input anchors — the as-of half (`store/manifest.rs`)

```rust
/// The immutable state-pointer of one input the plan read. The robust anchor
/// is an opaque, content-derived id, never a wall-clock timestamp: Iceberg
/// reproducibility pins a job to a snapshot-id/tag precisely because a
/// timestamp resolves against a prunable log and can drift or expire, while
/// the snapshot id is stable[^iceberg]. We resolve "as of T" to an id at
/// write time and carry the id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAnchor {
    /// The input relation's catalog id.
    pub source: String,
    /// The immutable state pointer, encoded per `kind`.
    pub anchor: AnchorValue,
    pub kind: AnchorKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnchorKind {
    /// An immutable Parquet result table: its content digest *is* its anchor.
    ResultDigest,
    /// A mutable companion table: the catalog's monotonic version counter for
    /// that table at read time (cp9 mutable tables are append-addressed).
    MutableVersion,
    /// An external/federated source exposing an as-of/version column: the
    /// pinned value of that column (e.g. an Iceberg snapshot id, a Delta
    /// version, an LSN, a watermark).
    SourceVersion,
    /// An external source with no version surface. Anchor is the read instant
    /// only; the manifest records that this input is *not* reproducibly
    /// pinned, so a verifier can downgrade its confidence honestly rather
    /// than claim a guarantee it cannot keep.
    UnpinnedAtInstant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnchorValue(String);
```

### 3.3 The manifest (`store/manifest.rs`)

```rust
/// The attestation written beside every materialised table. Shaped after an
/// in-toto statement: a `subject` (the artifact digest) plus a predicate
/// (everything about how it was produced), so a consumer verifies by digest
/// match then evaluates the predicate against its own policy[^intoto]. The
/// vocabulary follows W3C PROV / OpenLineage — the artifact `wasGeneratedBy`
/// this run and `wasDerivedFrom` these inputs[^prov][^openlineage].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationManifest {
    /// in-toto subject: digest of the artifact this manifest attests to.
    pub artifact: ArtifactDigest,
    /// How it was produced (the "definition").
    pub definition_hash: DefinitionHash,
    /// The as-of state of every input, in plan order.
    pub input_anchors: Vec<InputAnchor>,
    /// Producing-run identity (a per-process ulid) and instant. The instant is
    /// provenance metadata, never the reproducibility anchor (that is the
    /// input_anchors); recording it follows OpenLineage's run `eventTime`[^openlineage].
    pub produced_by: String,
    pub produced_at: String, // RFC3339
    /// Engine semantic version that produced this artifact.
    pub engine_version: String,
    /// Manifest format version, so a future format change is detectable, never
    /// silently misread.
    pub manifest_version: u32,
}

impl MaterializationManifest {
    /// Compute over a producing *description* (not a plan — see below), its
    /// environment, resolved input anchors, and the written artifact's digest.
    /// Pure: no I/O.
    pub fn compute(
        descriptor: &ProducingDescriptor,
        env: &MaterializationEnv,
        inputs: Vec<InputAnchor>,
        artifact: ArtifactDigest,
        produced_by: String,
        produced_at: String,
    ) -> Result<Self, ManifestError>;
}
```

**Definition input — a producing descriptor, not a `LogicalPlan`.** This engine's
result-table producers are hand-built physical pipelines; there is no single
`LogicalPlan` to canonicalise (the only `LogicalPlan` lives in the SQL lane,
`tenant_scope.rs`, and is unrelated to result-table materialisation). So the
definition hash is computed over a `ProducingDescriptor` — a typed,
deterministically-serialisable description of the verb and its parameters that
each producer fills in:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "producer", rename_all = "snake_case")]
pub enum ProducingDescriptor {
    Inference { model_id, task, source_id, content_columns, key_column },
    Embedding { model_id, task, source_id, columns, key_column, dimensions },
    NeighborGraph { source_table, k },
    GraphPropagation { source_table, kernel_id, dimensions },
    ContextSet { encoder_id, source_id, dimensions },
}
```

The five producers and the variant each fills: `infer` →
`Inference`; the embedding pipeline → `Embedding`; the neighbor-graph
derivation → `NeighborGraph`; `propagate_embeddings` → `GraphPropagation`;
`materialize_context` → `ContextSet`. Canonicalisation is a sorted-key JSON
encoding (object keys sorted, array order preserved — column / model order is
significant), SHA-256-folded with the environment, length-prefixed and
domain-separated so a descriptor field can never alias an environment field.

**Environment — includes the compute device.** `MaterializationEnv` carries the
engine semantic version, the **compute device** (`Cpu | Cuda { ordinal } | Metal
{ ordinal }`), and the identity + backend kind of every invoked model:

```rust
pub struct MaterializationEnv {
    pub engine_version: String,
    pub device: ComputeDevice,        // CPU vs CUDA yields different floats
    pub models: Vec<ModelIdentity>,   // { model_id, backend } per invoked model
}
```

A model run on CPU vs CUDA yields different float outputs but the same model
identity. Omitting the device would yield a false `Match` across devices — the
Bazel cross-compiler failure the spec cites against itself — so the device is
part of the world the hash covers. The device recorded is the *effective* one
(resolved through the loader's `select_device`, including the CPU fallback when
a requested GPU is unavailable), never merely the one requested.

### 3.4 The verdict + verb (`store/manifest.rs`, `session.rs`)

```rust
/// The outcome of checking a materialised table against an expectation. The
/// engine returns a verdict; it never *acts* on one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchVerdict {
    /// Recomputed artifact digest equals the manifest's, and (if the caller
    /// supplied one) the manifest's definition_hash equals the expected one.
    Match,
    /// Digest or definition_hash differs — the served artifact is not the
    /// output of the expected definition. Carries both sides for the caller.
    Mismatch { expected: DefinitionHash, found: DefinitionHash },
    /// The artifact verifies, but at least one input was `UnpinnedAtInstant`,
    /// so reproducibility cannot be fully asserted. Honest, not silent.
    MatchWithUnpinnedInputs { unpinned: Vec<String> },
    /// No manifest sidecar exists for the table — a pre-contract table. A
    /// truthful "unknown," never a fabricated match.
    MissingManifest,
}

// On the Session surface (embedded `PyDatabase` + remote `RemoteDatabase`):
/// Recompute the artifact digest of a materialised table and compare it (and,
/// if given, an expected definition hash) against its manifest. Read-only.
///
/// **`Match` attests the DATA, not the index.** The digest covers the Parquet
/// object — the data-of-record — never the `.usearch`/`.rowmap` ANN sidecars,
/// which are a derived accelerator reconstructible from the data. A `Match`
/// asserts the data is the output of the expected definition, not that any
/// particular index bytes are present.
pub async fn verify_materialization(
    &self,
    table: &str,
    expected_definition: Option<DefinitionHash>,
) -> Result<MatchVerdict, JammiError>;
```

### 3.5 `ManifestError` (`store/manifest.rs`)

```rust
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("producing descriptor is not canonicalisable: {0}")]
    UncanonicalDescriptor(String),
    #[error("manifest sidecar missing for table `{0}`")]
    MissingManifest(String),
    #[error("manifest format version {found} is newer than supported {supported}")]
    UnsupportedManifestVersion { found: u32, supported: u32 },
    #[error(transparent)]
    Storage(#[from] crate::store::StorageError),
}
```

## 4. Catalog schema changes

Migration `021` (`MIGRATION_021_MATERIALIZATION_CONTRACT`), appended after the registered `020` in `crates/jammi-db/src/catalog/migrations.rs` per the append-only migration protocol (the migration sequence in [`../PLAN-META.md`](../PLAN-META.md) is stale — it stops at `004`; the live sequence in `crates/jammi-db/src/catalog/schema.rs` runs through `020`):

```sql
ALTER TABLE result_tables ADD COLUMN definition_hash   TEXT;
ALTER TABLE result_tables ADD COLUMN input_anchors_json TEXT;
```

The full manifest is the sidecar `.materialization.json`; the two catalog columns are the indexable summary (so `verify_materialization` and provenance queries need not open every sidecar). Append-only; never modifies a shipped migration. Pre-existing tables created before `021` have `NULL` here and `verify_materialization` returns `MissingManifest` for them — a truthful "unknown," not a fabricated match.

## 5. Semantics — the pinned decisions

### 5.1 What the definition hash covers

The canonical producing descriptor **and** its world: the descriptor's typed fields (the verb plus its parameters — model id, task, source, columns, key, dimensions, neighbour count, kernel id — serialised with sorted object keys and significant array order), plus the engine semantic version, **the compute device**, and the identity + backend kind of every model the producer invokes. This mirrors the cleanest published statement of the pattern — Dagster's `data_version` is "computed by hashing the `code_version` together with the data versions of all input assets"[^dagster] — and heeds the Bazel cross-compiler failure: a hash that omits a determinant of the output yields false cache hits / false matches[^bazel]. dbt's per-node checksum-driven `state:modified`[^dbt] is the same idea applied to SQL nodes.

### 5.2 Why an id, not a timestamp, is the as-of anchor

Carrying a wall-clock `T` is unsound: a timestamp is resolved against a log whose entries can be pruned/compacted, so the same `T` can resolve to different states over time (or stop resolving). Iceberg's reproducibility guidance is to pin to an immutable snapshot id (or a permanent tag), not a timestamp[^iceberg]; OpenLineage similarly defers the dataset `version` facet to the store's own version identity rather than inventing one[^openlineage]. SPEC-02 resolves `T` → an id at materialize time and stores the id; the `produced_at` timestamp is provenance, never the anchor.

### 5.3 Honest degradation for unpinnable inputs

Not every federated source exposes a version surface. Rather than fabricate an anchor or silently drop the input from the hash, the manifest records `UnpinnedAtInstant`, and `verify_materialization` returns `MatchWithUnpinnedInputs`. The engine never claims a reproducibility guarantee it cannot keep — the same discipline as SPEC-01's loud `AmbiguousMatch`.

### 5.4 No signing key in the engine

The manifest is *shaped* like an attestation (subject digest + predicate) but the engine does not own a signing identity — key custody is a deployment/consumer concern (Sigstore/DSSE keyless signing is the consumer's closure[^slsa]). The engine guarantees the *content* (digest + definition hash + anchors); a consumer that needs non-repudiation wraps the manifest in its own signed envelope. Putting a key in the engine would import a consumer's trust model into a primitive — a discipline-test failure.

## 6. Wiring

**There is no pre-existing single funnel — SPEC-02 builds one.** Five producers write result tables, each with its own `building -> ready` step: inference output (`session.rs`), the embedding pipeline (`pipeline/embedding.rs`), the neighbor-graph derivation (`pipeline/neighbor_graph.rs`), and graph-propagation + context-set (both via `ResultStore::materialize_embedding_table`). The shared low-level transition was `ResultStore::finalize(ctx, name, url, rows)`, which carried no descriptor / env / inputs.

SPEC-02 replaces it with the single `building -> ready` transition `ResultStore::finalize_with_manifest(ctx, name, url, rows, descriptor, env, inputs)` and **deletes the bare `finalize`** (no-backwards-compat). All five producers route through it, each supplying its `ProducingDescriptor` variant, a `MaterializationEnv` (engine version + effective compute device + invoked-model identities), and its resolved `InputAnchor`s. `finalize_with_manifest` performs the publish in a crash-safe order: compute the `ArtifactDigest` over the durable Parquet bytes; write the `.materialization.json` sidecar; register the table in DataFusion; then flip `building -> ready` and persist the two summary columns (`definition_hash`, `input_anchors_json`) in one transaction. Because the sidecar lands *before* the status flip — the same boundary the ANN sidecar uses — **no `ready` table ever lacks a manifest**, and because the funnel is single, **no table is produced without one**.

Input anchors are resolved by the producer at write time: a result-table input contributes its own `ArtifactDigest` (`ResultDigest`) — read from its own manifest, or recomputed for a pre-contract source; a mutable table its monotonic version (`MutableVersion`); a registered source, which exposes no as-of/version surface in open-core, its read instant honestly recorded as `UnpinnedAtInstant`.

## 6A. Recovery — the manifest is part of the crash-consistency boundary

SPEC-02 couples to the W2-T2 recovery sweep (`ResultStore::recover_inner`, #205), which reconciles `building` orphans against Parquet validity and rebuilds the ANN sidecar. The manifest joins that same atomic boundary:

1. **Write-before-flip.** `finalize_with_manifest` writes the `.materialization.json` sidecar *before* the `building -> ready` status flip (the same ordering the ANN sidecar uses), so a crash can never leave a `ready` table without a manifest.

2. **`building` reconciliation is manifest-aware.** A crash can leave a `building` row whose Parquet is valid but whose manifest never landed (the window between the Parquet close and the sidecar write). Recovery cannot reconstruct the `ProducingDescriptor` — it is not persisted as structured catalog data, it lives in the producer's call — so it **cannot synthesise a manifest**. The contract forbids promoting a manifest-less table, so such a row is reaped to `failed` and its bytes deleted, exactly as a torn/invalid Parquet is. A `building` row whose Parquet *and* sidecar both landed (a crash after the sidecar, before the flip) is promoted to `ready` with its summary columns backfilled from the sidecar.

3. **`ready` reconciliation distinguishes a bug from history.** `reconcile_ready_manifests` sweeps already-`ready` rows: a **post-contract** row (catalog `definition_hash` set — it was promoted under the contract) whose sidecar is now absent is a corruption (the attestation a verifier would read is gone) and is reaped to `failed`; a **pre-contract** row (`definition_hash IS NULL`, created before migration 021) legitimately has no sidecar and is left untouched, verifying as an honest `MissingManifest`. This is the post-021-bug vs legitimate-pre-021 distinction the contract requires.

A crash-injection test (`materialization_crash_recovery.rs`, feature `test-hooks`) proves (2): the `maybe_signal_materialization` hook fires inside `finalize_with_manifest` after the Parquet is durable and before the manifest write; the child parks, the parent `SIGKILL`s and restarts, and recovery reaps the manifest-less row — reusing the self-respawn + `SIGKILL` harness pattern from `mutable_crash_recovery.rs`.

## 7. Verb conventions (embed == remote)

`verify_materialization` follows the established path: a `Session` method (`crates/jammi-ai/src/local_session.rs`) → PyO3 binding on `PyDatabase` (`crates/jammi-python/src/database.rs`) → gRPC `CatalogService.VerifyMaterialization` handler + `RemoteDatabase` stub → `"verify_materialization"` in `_PIPELINE_VERBS` in `crates/jammi-python/tests/test_conformance.py`. (There is no Rust `Database` struct; the embedded surface is the `PyDatabase` pyclass over `InferenceSession`, reached through the shared `Session` wrapper, exactly as every other verb.) The verb is read-only and computed identically on both transports (the digest recomputation runs server-side; the verdict crosses the wire as a `oneof` mirroring `MatchVerdict`), so the conformance guard pins signature parity and the adversarial oracle (`tests/it/materialization.rs`) asserts each verdict. It is control-plane (a digest + verify), so it carries **no scale benchmark** (breadth-grid cell-(d) N/A): there is no result-set size to sweep, only a fixed digest recomputation over one table.

## 8. Module layout

```
crates/jammi-db/src/store/
  manifest.rs        +new  ProducingDescriptor, MaterializationEnv, ComputeDevice, ModelIdentity,
                           InputAnchor/AnchorKind/AnchorValue, DefinitionHash, ArtifactDigest,
                           MaterializationManifest, MatchVerdict, ManifestError (the frozen contract,
                           definition-hash + sidecar IO are pure functions here)
  mod.rs             CHANGED  finalize_with_manifest (the single building->ready funnel; bare `finalize`
                           DELETED); verify_materialization; read/write sidecar; result_digest_anchor;
                           recover_inner is manifest-aware + reconcile_ready_manifests
  mutable/test_hook.rs CHANGED  maybe_signal_materialization (the Parquet-written-but-manifest-not crash hook)
crates/jammi-db/src/catalog/schema.rs           CHANGED  MIGRATION_021_MATERIALIZATION_CONTRACT
crates/jammi-db/src/catalog/migrations.rs       CHANGED  register 021
crates/jammi-db/src/catalog/result_repo.rs      CHANGED  promote_result_table_with_manifest; record gains
                           definition_hash / input_anchors_json
crates/jammi-ai/src/session.rs                  CHANGED  compute_device() (effective device for the env)
crates/jammi-ai/src/local_session.rs            CHANGED  verify_materialization (the Session verb)
crates/jammi-ai/src/model/{mod,backend/candle}.rs CHANGED  LoadedModel::backend_kind; effective_compute_device
crates/jammi-ai/src/{session.rs, pipeline/embedding.rs, pipeline/neighbor_graph.rs,
                     pipeline/graph_propagation.rs, pipeline/context_set.rs}  CHANGED  the 5 producers
                           supply descriptor + env + inputs to finalize_with_manifest
crates/jammi-wire/proto/jammi/v1/catalog.proto  CHANGED  VerifyMaterialization rpc + messages
crates/jammi-wire/src/catalog.rs                CHANGED  match_verdict_{to,from}_proto
crates/jammi-server/src/grpc/catalog.rs         CHANGED  VerifyMaterialization handler
crates/jammi-python/src/database.rs             CHANGED  PyO3 verify_materialization binding
clients/python/jammi_client/_database.py        CHANGED  remote stub + verdict projection
crates/jammi-python/tests/test_conformance.py   CHANGED  _PIPELINE_VERBS += verify_materialization
crates/jammi-db/tests/it/materialization.rs                +new  adversarial oracle (the §11 verdicts) + funnel + recovery
crates/jammi-db/tests/it/materialization_crash_recovery.rs +new  SIGKILL crash-injection (manifest window)
docs/guide/src/materialization-contract.md      +new  cookbook recipe (§9) — SEPARATE post-tag deliverable (ch19)
```

## 9. Cookbook recipe (mandatory exit-criteria bullet)

`docs/guide/src/materialization-contract.md` — "Proving a table is what you think it is." Outline: materialise a table; read its manifest (definition hash, input anchors, run); recompute and `verify_materialization` → `Match`; change the producing query and show the definition hash changes (so a stale copy is detectable); show `MatchWithUnpinnedInputs` for an unversioned source. Closing note: this is the contract a consumer carries across a serving boundary so an online read can assert it corresponds to the offline definition — but the boundary itself is the consumer's to build. `SUMMARY.md` updated; `mdbook build` clean; samples compile under `mdbook test`.

## 10. Discipline-test example — a clinical-trial data fabric, not a feature store

**Hypothetical user: *Helix Trials*** — a small CRO assembling regulatory submissions. Nothing to do with AccuRisk, Lace, or any ML feature store. Helix materialises an analysis dataset by joining patient observations to lab reference ranges, and must later *prove to an auditor* that the submitted dataset is exactly the output of the locked analysis definition over the data as it stood at database-lock — not a re-run that quietly picked up later-corrected values.

```rust
let table = db.asof_join(observations.into(), reference_ranges.into(), spec).await?;
let manifest = db.read_manifest(table.name()).await?;
// ... months later, an auditor re-derives and checks ...
let verdict = db
    .verify_materialization(table.name(), Some(manifest.definition_hash.clone()))
    .await?;
assert_eq!(verdict, MatchVerdict::Match);
```

This passes the discipline test. None of *"patient," "lab," "audit," "submission," "feature,"* or *"store"* leaks into the engine; the engine provides the artifact-identity contract and the verify primitive, and Helix builds regulatory reproducibility on top — exactly as a quant proves a backtest used point-in-time data, or an ML consumer proves an online value matches the trained definition. A Jammi user who has never heard of any of those — or of Helix — gets the same primitive.

## 11. Exit criteria

1. **Determinism.** Materialising the same plan over the same inputs twice yields byte-identical `definition_hash`. `tests/it/definition_hash.rs`.
2. **Sensitivity.** Changing the plan (a projection, a literal, a model version) changes the `definition_hash`; changing only display aliases does not. Same file.
3. **Environment coverage.** A different engine version, a different invoked-model version, **or a different compute device (CPU vs CUDA)** yields a different hash — the Bazel lesson encoded as a test (`manifest.rs` unit tests: `different_engine_version_changes_the_hash`, `different_model_version_changes_the_hash`, `different_device_changes_the_hash`).
4. **Anchor correctness.** A result-table input contributes its own artifact digest; a mutable-table input contributes a version that increments after an insert; an unversioned source yields `UnpinnedAtInstant`.
5. **Round-trip.** Manifest written on materialize, read back identical; `ResultTableInfo.manifest` populated; catalog summary columns match the sidecar. `tests/it/manifest_roundtrip.rs`.
6. **Verdicts.** `verify_materialization` returns `Match` for an untouched table, `Mismatch` against a wrong expected hash, `MatchWithUnpinnedInputs` when an input was unpinned, `MissingManifest` for a pre-`021` table.
7. **No table escapes.** The bare `finalize` is deleted and `finalize_with_manifest` is the sole `building -> ready` transition, so the type system enforces that every one of the five producers (inference, embedding, neighbor-graph, graph-propagation, context-set) supplies a descriptor/env/inputs — a producer that tried to publish without one would not compile. The funnel test (`materialization.rs::the_funnel_persists_sidecar_and_summary_columns`) asserts a materialised table carries both the sidecar and the catalog summary columns.
8. **Embed == remote.** Conformance guard green for `verify_materialization`; identical verdict on both transports for a fixture.
9. **No band-aids.** Zero net new `#[allow(...)]`, `let _ =`, `// TODO`, `#[ignore]`; `cargo clippy --all-features -- -D warnings` and `cargo fmt --check` pass.
10. **Recipe.** §9 chapter renders and its samples compile.

## 12. Engineering-principles audit

| Principle | How this spec satisfies it |
|---|---|
| *Clean, functional style* | `MaterializationManifest::compute` is pure (inputs → manifest, no I/O); I/O lives only in the manifest-IO layer. Canonicalisation is a fold over the plan tree, stack-safe via an explicit work-stack. |
| *Clear boundaries* | Three concerns: the contract value + hashing + sidecar IO (`manifest.rs`, pure), the result-store funnel that emits it (`store/mod.rs`), and the producers that fill the descriptor. The verb depends on the store's read API, not its internals. |
| *DRY* | One materialization funnel stamps every table; no per-call-site manifest logic duplicated. One sidecar-path helper mirrors the existing `.usearch`/`.rowmap` convention rather than inventing a second layout. |
| *No backwards compatibility* | New module, new migration. Pre-`005` tables return `MissingManifest`, not a compatibility shim that fabricates a manifest. |
| *Type-driven design* | `DefinitionHash`/`ArtifactDigest`/`AnchorValue` newtypes (can't be confused); `AnchorKind`/`MatchVerdict` enums, never strings; `manifest_version` makes a future format change a typed `UnsupportedManifestVersion`, not a silent misparse. |
| *No band-aids* | Unpinnable inputs degrade to a typed `MatchWithUnpinnedInputs`/`UnpinnedAtInstant` rather than a fabricated anchor; the engine refuses to claim a guarantee it can't keep. No signing key smuggled into the engine (§5.4). |
| *Engine, not platform* | No serving tier, no KV, no policy, no "feature/online/governance" vocabulary; the §10 third-tenant is a CRO. The engine returns a verdict; acting on it is the consumer's. |
| *Atomic across the workspace* | §8 ships jammi-db + jammi-engine + jammi-ai + PyO3 + remote client + conformance test in one PR. |
| *Docs reflect current state* | The recipe describes the contract as it is; no journey markers; the migration is named for what it does. |

## 13. Open questions

1. **Signed envelope.** §5.4 keeps signing out of the engine. If a consumer demonstrates a need for engine-native non-repudiation that cannot live in their closure, revisit — but the default (content guarantee in the engine, signature in the consumer) is the discipline-respecting choice.
2. **Index sidecar in the digest.** The `ArtifactDigest` covers the Parquet object; should it also cover the `.usearch`/`.rowmap` ANN sidecars? For now the digest is the data-of-record (the Parquet); a search index is a derived accelerator reconstructible from it. Revisit if a consumer needs to attest the exact index bytes.
3. **Cross-engine-version verification.** A table produced by engine vX verified by engine vY: the definition hash includes the engine version, so a different verifier version reports `Mismatch` on definition even if the data is identical. Is that too strict? Deferred — the strict default is the loud-and-correct one; a "data-equal but definition-differs" verdict could be added if a consumer needs it.
4. **`derived_from` consolidation — resolved.** `input_anchors` is the source of truth for input provenance. The existing `derived_from` column (migration 013) is retained only as an indexed FK-lineage convenience and is no longer an *independent* provenance record: a producer that records `derived_from = X` always emits, in the same call, a `ResultDigest` `InputAnchor` for `X` (see `ResultStore::result_digest_anchor`, used by the neighbor-graph and graph-propagation producers), so the FK column is implied by — a projection of — the anchor set, never maintained separately. The richer, hashed form (`input_anchors_json`, migration 021) is authoritative; `derived_from` stays a queryable shorthand for the single-parent derivations, kept consistent by construction.

## 14. References

[^intoto]: in-toto Attestation Framework (ITE-6), Statement spec: an attestation has a `subject` (ResourceDescriptors, each with a `digest` map), a `predicateType` (TypeURI), and a `predicate`; "Subject artifacts are matched purely by digest, regardless of content type," and "Subjects are assumed to be immutable." Verification = compute the artifact digest, compare to `subject.digest`, check the predicate against policy. — https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md (fetched 2026-06-16).

[^iceberg]: Apache Iceberg — a snapshot is an immutable table state identified by a unique snapshot-id; the documented ML-reproducibility pattern is to pin a job to a snapshot-id or permanent tag so concurrent ingestion cannot change what the job reads; snapshot-id reads are deterministic, whereas timestamp reads resolve against metadata that can be expired/compacted. — https://iceberg.apache.org/spec/ ; https://lakefs.io/blog/iceberg-time-travel/ (fetched 2026-06-16).

[^dagster]: Dagster asset versioning — `data_version` "is computed by hashing the `code_version` together with the data versions of all input assets"; Dagster skips recomputation when code_version is unchanged AND input data_versions match. The cleanest published statement of "definition identity = hash(code + input identities)." — https://docs.dagster.io/guides/dagster/asset-versioning-and-caching (fetched 2026-06-16).

[^dbt]: dbt `state:modified` — change detection is per-node checksum-driven (each node carries a content checksum diffed against a baseline manifest); documented caveat: seeds >1 MiB cannot be content-hashed so dbt falls back to the file path — concrete evidence the mechanism is a content hash with a size cutoff. — https://docs.getdbt.com/reference/node-selection/state-comparison-caveats (fetched 2026-06-16).

[^bazel]: Bazel remote caching — an action's cache key is a digest of the action metadata + inputs (command line, input file digests, env); a cache hit pulls outputs from the content-addressable store by digest. Failure mode: Bazel does not hash tools outside the workspace (e.g. the system compiler), so two machines with different compilers can wrongly share a cache hit — the lesson that a definition hash must cover the whole execution environment or the "match" assertion is unsound. — https://github.com/bazelbuild/bazel/blob/master/site/en/remote/caching.md (fetched 2026-06-16).

[^prov]: W3C PROV-O — core types Entity/Activity/Agent; relations `prov:wasGeneratedBy` (entity ← producing activity), `prov:used` (activity → input), `prov:wasDerivedFrom` (entity ← source entity). The standard vocabulary for "what process produced this dataset, from what inputs." — https://www.w3.org/TR/prov-o/ (fetched 2026-06-16).

[^openlineage]: OpenLineage object model — Run/Job/Dataset; a RunEvent (START/COMPLETE) carries `runId` and `eventTime` and records input/output datasets; the dataset `version` facet is "the version of the dataset when versioning is defined by the data store" — i.e. lineage references the store's version id rather than inventing one. Motivates carrying `produced_by`/`produced_at` as provenance and deferring the reproducibility anchor to the input store's own id. — https://openlineage.io/docs/spec/object-model (fetched 2026-06-16).

[^slsa]: SLSA provenance is an in-toto predicate (builder identity, build instructions, environment, dependency digests), signed via DSSE envelopes, often keyless with Sigstore/cosign using the CI invocation identity — i.e. signing/key-custody is a build/deployment concern layered around the attestation, supporting §5.4's decision to keep signing in the consumer's closure rather than the engine. — https://slsa.dev/blog/2023/05/in-toto-and-slsa (fetched 2026-06-16).
