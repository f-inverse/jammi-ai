# SPEC-02 — Materialization contract

> Part of the [60 — temporal correctness](./README.md) plan group. Independent of [`SPEC-01-asof-join.md`](./SPEC-01-asof-join.md) except at the shared result-store write path (see [`README.md` — Concurrent-session strategy](./README.md)); pinned to land first. Research rules: [`../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md`](../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md).
>
> **Status:** Proposed (draft).

## 1. Goal

A *materialization contract* is the verifiable identity a materialised table carries so that any later reader can assert **"this artifact is the output of definition D over input-state S"** — without trusting a name, a path, or an out-of-band convention. The engine already writes a result table as an immutable Parquet object plus, for embedding tables, an ANN-index sidecar set (`crates/jammi-db/src/store/`: `ResultTableInfo { table_name, parquet_url, index_url }`, whose layout helpers append `.usearch`/`.rowmap`/`.manifest.json`). This spec adds a *separate* `.materialization.json` sidecar — written for **every** result table, not only embedding tables — carrying a signed-shaped attestation that binds three things to the table's content digest: a **content hash of the producing logical plan and the environment that affects its output**, the **immutable as-of anchors of every input** the plan read, and the **producing-run identity and instant**. It adds one verb, `verify_materialization`, that recomputes the artifact digest and reports whether it matches a caller-supplied expectation. The primitive carries only what every reproducibility-minded consumer needs; it ships **no** policy — what a reader *does* with a mismatch (refuse, alarm, fall back) is the reader's concern, not the engine's.

This is a seam, not a serving tier. The contract makes "the served value matches the definition that trained it, as of T" a checkable fact at a boundary; it does not build the boundary. That keeps the engine on the right side of [`../../PHILOSOPHY.md`](../../PHILOSOPHY.md): the engine ships the export/attestation contract; the KV store, the online tier, and the refuse-to-serve policy live in the consumer's composition.

## 2. Concurrent-session strategy

Single-capability spec. One sequential scaffold, two parallel subagents, integration.

### 2.A — Sequential scaffolding (~45 min, main session)

1. `crates/jammi-db/src/store/manifest.rs` — `MaterializationManifest`, `InputAnchor`, `DefinitionHash`, `ArtifactDigest`, `MatchVerdict`, `ManifestError` (the frozen contract, §3).
2. `crates/jammi-db/src/store/mod.rs` — extend `ResultTableInfo` with `manifest: MaterializationManifest`; add the manifest-write call to the existing materialization path (signature only, body `todo!()`).
3. `crates/jammi-db/src/catalog/schema.rs` — add `MIGRATION_021_MATERIALIZATION_CONTRACT` (DDL string adding `definition_hash TEXT`, `input_anchors_json TEXT` to `result_tables`). `021` is the next free number after the registered `020` (`MIGRATION_020_CHANNEL_TENANT_SCOPE`).
4. `crates/jammi-db/src/catalog/migrations.rs` — append `schema::MIGRATION_021_MATERIALIZATION_CONTRACT` to the migration slice.

Phase A ends when `cargo check -p jammi-db` passes and migration `021` applies against an empty SQLite DB in a unit test.

### 2.B — Two parallel subagents (~3 h wall-clock)

| Subagent | Files written | Responsibility | Independent test |
|---|---|---|---|
| **Hashing subagent** | `store/manifest.rs` (hashing impl), `store/plan_canonical.rs` | Canonicalise a `LogicalPlan` + environment into a stable `DefinitionHash`; compute the `ArtifactDigest` over the Parquet object | `crates/jammi-db/tests/it/definition_hash.rs` (determinism + sensitivity) |
| **Manifest-IO subagent** | `store/manifest.rs` (read/write), result-store wiring | Write the manifest sidecar on materialize; read it back; populate `ResultTableInfo` | `crates/jammi-db/tests/it/manifest_roundtrip.rs` |

**Coordination contract** (frozen in Phase A): `MaterializationManifest::compute(plan, env, inputs, digest) -> Result<Self, ManifestError>` and the sidecar path helper `materialization_url(base: &StorageUrl) -> StorageUrl` (appends `.materialization.json`, mirroring — but distinct from — the existing `.usearch`/`.rowmap`/`.manifest.json` index-sidecar convention).

### 2.C — Sequential integration (~45 min, main session)

1. Add `verify_materialization` to embedded `Database` + remote `RemoteDatabase`; add to `_PIPELINE_VERBS` in `test_conformance.py` (§7).
2. Backfill: ensure every existing materialization call site (embedding, fine-tune output, eval, and SPEC-01's `asof_join`) flows through the manifest-writing path — no table escapes without a manifest.
3. `cargo clippy --all-features -- -D warnings`, `cargo fmt --check`. No `#[allow(...)]`.

## 3. Public API surface (exhaustive)

### 3.1 Identities (`store/manifest.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Content hash of *how* a table was produced: a canonicalised logical plan
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
    /// Compute over a finished plan, its environment, resolved input anchors,
    /// and the written artifact's digest. Pure: no I/O.
    pub fn compute(
        plan: &LogicalPlan,
        env: &MaterializationEnv,
        inputs: Vec<InputAnchor>,
        artifact: ArtifactDigest,
    ) -> Result<Self, ManifestError>;
}
```

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
}

// On Database / RemoteDatabase:
/// Recompute the artifact digest of a materialised table and compare it (and,
/// if given, an expected definition hash) against its manifest. Read-only.
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
    #[error("logical plan is not canonicalisable: {0}")]
    UncanonicalPlan(String),
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

The canonical logical plan **and** its world: the plan's normalised structure (operators, expressions, projections — column *positions* and literal values, not display strings or generated aliases), plus the engine semantic version, the identity (canonical name + version) of every model the plan invokes, and the input backend kinds. This mirrors the cleanest published statement of the pattern — Dagster's `data_version` is "computed by hashing the `code_version` together with the data versions of all input assets"[^dagster] — and heeds the Bazel cross-compiler failure: a hash that omits a determinant of the output yields false cache hits / false matches[^bazel]. dbt's per-node checksum-driven `state:modified`[^dbt] is the same idea applied to SQL nodes.

### 5.2 Why an id, not a timestamp, is the as-of anchor

Carrying a wall-clock `T` is unsound: a timestamp is resolved against a log whose entries can be pruned/compacted, so the same `T` can resolve to different states over time (or stop resolving). Iceberg's reproducibility guidance is to pin to an immutable snapshot id (or a permanent tag), not a timestamp[^iceberg]; OpenLineage similarly defers the dataset `version` facet to the store's own version identity rather than inventing one[^openlineage]. SPEC-02 resolves `T` → an id at materialize time and stores the id; the `produced_at` timestamp is provenance, never the anchor.

### 5.3 Honest degradation for unpinnable inputs

Not every federated source exposes a version surface. Rather than fabricate an anchor or silently drop the input from the hash, the manifest records `UnpinnedAtInstant`, and `verify_materialization` returns `MatchWithUnpinnedInputs`. The engine never claims a reproducibility guarantee it cannot keep — the same discipline as SPEC-01's loud `AmbiguousMatch`.

### 5.4 No signing key in the engine

The manifest is *shaped* like an attestation (subject digest + predicate) but the engine does not own a signing identity — key custody is a deployment/consumer concern (Sigstore/DSSE keyless signing is the consumer's closure[^slsa]). The engine guarantees the *content* (digest + definition hash + anchors); a consumer that needs non-repudiation wraps the manifest in its own signed envelope. Putting a key in the engine would import a consumer's trust model into a primitive — a discipline-test failure.

## 6. Wiring

Every materialization path already funnels through the result-store write in `crates/jammi-db/src/store/`. SPEC-02 adds one step at the end of that funnel: after the Parquet object is written, compute its `ArtifactDigest`, build the `MaterializationManifest` from the plan + env + resolved input anchors, write the `.materialization.json` sidecar, and persist the two summary columns. Because the funnel is single, **no table can be produced without a manifest** — the integration step (§2.C item 2) audits every call site (embedding, fine-tune, eval, and SPEC-01 `asof_join`). Input anchors are resolved at read time from the catalog: a result-table input contributes its own `ArtifactDigest`; a mutable table its monotonic version; a federated source its as-of/version column value if it has one, else `UnpinnedAtInstant`.

## 7. Verb conventions (embed == remote)

`verify_materialization` follows the established path: Rust impl on `Database` → PyO3 binding → gRPC handler + `RemoteDatabase` stub → add `"verify_materialization"` to `_PIPELINE_VERBS` in `crates/jammi-python/tests/test_conformance.py`. The verb is read-only and computed identically on both transports (the digest recomputation runs server-side; the verdict crosses the wire), so the conformance guard pins signature parity and a fixture asserts identical verdicts embedded vs remote.

## 8. Module layout

```
crates/jammi-db/src/store/
  manifest.rs        +new  MaterializationManifest, InputAnchor, DefinitionHash, ArtifactDigest, MatchVerdict, ManifestError
  plan_canonical.rs  +new  LogicalPlan → stable canonical bytes
  mod.rs             CHANGED  ResultTableInfo gains `manifest`; materialize path writes the sidecar
crates/jammi-db/src/catalog/schema.rs           CHANGED  MIGRATION_021_MATERIALIZATION_CONTRACT
crates/jammi-db/src/catalog/migrations.rs       CHANGED  register 021
crates/jammi-ai/src/session.rs                  CHANGED  verify_materialization
crates/jammi-python/src/lib.rs                  CHANGED  PyO3 binding
clients/python/jammi_client/_database.py        CHANGED  remote stub
crates/jammi-python/tests/test_conformance.py   CHANGED  _PIPELINE_VERBS += verify_materialization
crates/jammi-db/tests/it/definition_hash.rs     +new
crates/jammi-db/tests/it/manifest_roundtrip.rs  +new
docs/guide/src/materialization-contract.md      +new  cookbook recipe (§9)
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
3. **Environment coverage.** A different engine version or a different invoked-model version yields a different hash — the Bazel lesson encoded as a test.
4. **Anchor correctness.** A result-table input contributes its own artifact digest; a mutable-table input contributes a version that increments after an insert; an unversioned source yields `UnpinnedAtInstant`.
5. **Round-trip.** Manifest written on materialize, read back identical; `ResultTableInfo.manifest` populated; catalog summary columns match the sidecar. `tests/it/manifest_roundtrip.rs`.
6. **Verdicts.** `verify_materialization` returns `Match` for an untouched table, `Mismatch` against a wrong expected hash, `MatchWithUnpinnedInputs` when an input was unpinned, `MissingManifest` for a pre-`021` table.
7. **No table escapes.** A test enumerates every materialization call site (embedding, fine-tune, eval, asof_join) and asserts each produces a manifest.
8. **Embed == remote.** Conformance guard green for `verify_materialization`; identical verdict on both transports for a fixture.
9. **No band-aids.** Zero net new `#[allow(...)]`, `let _ =`, `// TODO`, `#[ignore]`; `cargo clippy --all-features -- -D warnings` and `cargo fmt --check` pass.
10. **Recipe.** §9 chapter renders and its samples compile.

## 12. Engineering-principles audit

| Principle | How this spec satisfies it |
|---|---|
| *Clean, functional style* | `MaterializationManifest::compute` is pure (inputs → manifest, no I/O); I/O lives only in the manifest-IO layer. Canonicalisation is a fold over the plan tree, stack-safe via an explicit work-stack. |
| *Clear boundaries* | Three concerns, three units: canonicalisation (`plan_canonical.rs`), the manifest value + hashing (`manifest.rs`), and the result-store write that emits it. The verb depends on the store's read API, not its internals. |
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
4. **`derived_from` consolidation.** The existing provenance `derived_from` field overlaps with `input_anchors`. They should converge — `input_anchors` is the richer, hashed form. Whether to drop `derived_from` or express it as a view over `input_anchors` is a follow-up; not litigated here.

## 14. References

[^intoto]: in-toto Attestation Framework (ITE-6), Statement spec: an attestation has a `subject` (ResourceDescriptors, each with a `digest` map), a `predicateType` (TypeURI), and a `predicate`; "Subject artifacts are matched purely by digest, regardless of content type," and "Subjects are assumed to be immutable." Verification = compute the artifact digest, compare to `subject.digest`, check the predicate against policy. — https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md (fetched 2026-06-16).

[^iceberg]: Apache Iceberg — a snapshot is an immutable table state identified by a unique snapshot-id; the documented ML-reproducibility pattern is to pin a job to a snapshot-id or permanent tag so concurrent ingestion cannot change what the job reads; snapshot-id reads are deterministic, whereas timestamp reads resolve against metadata that can be expired/compacted. — https://iceberg.apache.org/spec/ ; https://lakefs.io/blog/iceberg-time-travel/ (fetched 2026-06-16).

[^dagster]: Dagster asset versioning — `data_version` "is computed by hashing the `code_version` together with the data versions of all input assets"; Dagster skips recomputation when code_version is unchanged AND input data_versions match. The cleanest published statement of "definition identity = hash(code + input identities)." — https://docs.dagster.io/guides/dagster/asset-versioning-and-caching (fetched 2026-06-16).

[^dbt]: dbt `state:modified` — change detection is per-node checksum-driven (each node carries a content checksum diffed against a baseline manifest); documented caveat: seeds >1 MiB cannot be content-hashed so dbt falls back to the file path — concrete evidence the mechanism is a content hash with a size cutoff. — https://docs.getdbt.com/reference/node-selection/state-comparison-caveats (fetched 2026-06-16).

[^bazel]: Bazel remote caching — an action's cache key is a digest of the action metadata + inputs (command line, input file digests, env); a cache hit pulls outputs from the content-addressable store by digest. Failure mode: Bazel does not hash tools outside the workspace (e.g. the system compiler), so two machines with different compilers can wrongly share a cache hit — the lesson that a definition hash must cover the whole execution environment or the "match" assertion is unsound. — https://github.com/bazelbuild/bazel/blob/master/site/en/remote/caching.md (fetched 2026-06-16).

[^prov]: W3C PROV-O — core types Entity/Activity/Agent; relations `prov:wasGeneratedBy` (entity ← producing activity), `prov:used` (activity → input), `prov:wasDerivedFrom` (entity ← source entity). The standard vocabulary for "what process produced this dataset, from what inputs." — https://www.w3.org/TR/prov-o/ (fetched 2026-06-16).

[^openlineage]: OpenLineage object model — Run/Job/Dataset; a RunEvent (START/COMPLETE) carries `runId` and `eventTime` and records input/output datasets; the dataset `version` facet is "the version of the dataset when versioning is defined by the data store" — i.e. lineage references the store's version id rather than inventing one. Motivates carrying `produced_by`/`produced_at` as provenance and deferring the reproducibility anchor to the input store's own id. — https://openlineage.io/docs/spec/object-model (fetched 2026-06-16).

[^slsa]: SLSA provenance is an in-toto predicate (builder identity, build instructions, environment, dependency digests), signed via DSSE envelopes, often keyless with Sigstore/cosign using the CI invocation identity — i.e. signing/key-custody is a build/deployment concern layered around the attestation, supporting §5.4's decision to keep signing in the consumer's closure rather than the engine. — https://slsa.dev/blog/2023/05/in-toto-and-slsa (fetched 2026-06-16).
