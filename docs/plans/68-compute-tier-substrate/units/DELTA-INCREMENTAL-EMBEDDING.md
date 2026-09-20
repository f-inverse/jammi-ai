# Incremental embedding — versioned embedding tables over immutable segments, content-hash deltas

An embedding table can be refreshed incrementally: a refresh re-embeds only the source rows whose
content changed, writes them as a new immutable fragment and ANN segment, hides superseded and
deleted keys behind a per-version deletion mask, and publishes the result as a new version of the
same logical table in one compare-and-set. Three actuators carry it — `refresh_embeddings`,
`compact_embeddings`, `expire_versions` — in
`crates/jammi-ai/src/pipeline/embedding_refresh.rs`, over the catalog layer in
`crates/jammi-db/src/catalog/version_repo.rs` and the store layer in `crates/jammi-db/src/store/`
(`version.rs`, `deletes.rs`, `masked_provider.rs`, `segment_set_cache.rs`, `building_version.rs`,
`content_hash.rs`). The user-facing description is `docs/guide/src/incremental-refresh.md`; this
document records the design positions and why each was taken.

Third-party behaviour a decision depends on is named by crate, version and symbol. The versions
are the ones `Cargo.lock` resolves today — `datafusion` 54.1.0 (and its `datafusion-common`,
`datafusion-physical-plan`, `datafusion-physical-expr`, `datafusion-physical-optimizer`,
`datafusion-catalog-listing` members), `arrow` / `parquet` 58.4.0, `object_store` 0.13.2,
`thiserror` 2.0.18 — unless a fact is marked as read at an earlier version.

## 1. Decisions

**D1 — The deletion set is a per-version Parquet sidecar with a VERSION horizon.**
`{table}__v{N}.deletes.parquet`, columns `_row_id Utf8 NOT NULL, _dead_through_version Int64 NOT
NULL`, sorted by `_row_id`, cumulative: version N's file is the whole mask
(`crates/jammi-db/src/store/deletes.rs::deletes_schema`, `::DeletionMask`). An entry `(K, h)` masks
`K` in every fragment and segment whose stamped `version <= h` and nowhere else
(`DeletionMask::is_masked`). Every fragment (in the manifest) and every segment
(`index_segments.version`) is stamped with its producing version; base artifacts are stamped with
the base version number (§2.1). A refresh producing N writes `entry[K] = max(entry[K], N-1)`
(`DeletionMask::raise`) for every superseded or deleted key, so artifacts stamped N are never
self-masked and an updated key present in two segments is unambiguous.

Why a version and not a segment id: segment ids are allocated by a read-max / insert /
collision-retry loop (`crates/jammi-db/src/store/mod.rs::ResultStore::append_segment_for_version`)
and a zero-row write allocates none, so the id space is not a stable horizon; the version number is
allocated once, before any artifact is written (D2). Why a sidecar and not catalog rows: it is
object-store native, there is one artifact per version whose digest folds into the version identity
(§2.6), reconcile already enumerates a table's objects by URL
(`crates/jammi-db/src/store/reconcile.rs::referenced_result_keys`), and the same file serves
`file://` and an object store — topology stays configuration. Row identity across fragments and
segments is `_row_id`, so one mask serves the ANN merge, the SQL/exact scan and
`read_vector_by_key`. Precedent: Iceberg equality deletes with sequence numbers (§6).

**D2 — One logical table, versions in a child table, monotonic allocation.** The `result_tables`
row (PK `table_name`) stays the identity every predicate renders and every `index_segments` row
references; "newest ready" resolution
(`crates/jammi-db/src/catalog/result_repo.rs::resolve_embedding_table`) is unchanged. A
version-per-row scheme was rejected: it would shadow tables in that resolution and break the
`derives_from` anchors dependents hold. Versions live in `result_table_versions` (§2.10) with
`result_tables.current_version` (the Iceberg current-snapshot-id + atomic swap shape) and
`result_tables.next_version INTEGER NOT NULL DEFAULT 0`, the monotonic allocator: a number is
allocated exactly once and never reused; a failed or crashed version keeps its number;
`expire_versions` and recovery delete rows and stamped artifacts and never touch `next_version`.
`current_version IS NULL` means never refreshed, and such a table behaves exactly as an unversioned
one.

Allocation is parent-pinned
(`crates/jammi-db/src/catalog/version_repo.rs::Catalog::allocate_result_table_version`): the
caller passes the `current_version` its delta was derived from and the allocating `UPDATE` requires
the row to still carry it, so a publish that landed in between refuses the allocation with
`ParentMoved` before `next_version` increments — a refusal never burns a number and never inserts a
`building` row whose manifest would name a parent the catalog no longer agrees with. Two refreshes
that both read the same parent before either publishes both allocate; the second to publish misses
`current_version = parent` and gets `ParentMoved` (D5). `ParentMoved` is the one classification for
every lost parent race (allocation, publish, base publish); `CasFailed` is the building-row lease
class — a CAS that matched no row because the row already left the state it was pinned to.

**D3 — The base version is published as manifest-write + CAS in ONE transaction, with no
`building` row.** On the first refresh (or compaction) of a table with `current_version IS NULL`,
`crates/jammi-ai/src/pipeline/embedding_refresh.rs::InferenceSession::ensure_base_version` reads
`next_version` (= B) and writes `{table}__v{B}.version.json` describing the table as it stands:
fragments = `[parquet_path]` stamped B, segments = the `index_segments` rows with `version IS NULL`
stamped B, no deletes, the descriptor copied verbatim from `.materialization.json`, `parent =
null`. Then `Catalog::publish_base_version` runs one transaction: insert the `ready` version row
(`identity`, `live_rows = row_count`, `masked_rows = 0`) and `UPDATE result_tables SET
current_version = B, next_version = B+1 WHERE table_name = $t AND status = 'ready' AND
current_version IS NULL AND next_version = B AND <tenant arm>` (one row, else rollback).

The base publish refuses a row that records no `dimensions` (`NotRefreshable { NotEmbeddingTable
}`), so a versioned row without `dimensions` cannot be created through this path, and a table with
no `.materialization.json` is `NotRecomputable`. A crash before the transaction leaves an orphan
`__v{B}.version.json` that the next attempt overwrites at the same path (same content modulo
`produced_by` / `produced_at`). Reconcile's object→row arm is age-gated — `apply = true` requires
`grace >=` the lease duration (`crates/jammi-db/src/store/reconcile.rs::ReconcileOptions`) — which
this design depends on; the publisher also re-probes the manifest after the CAS and re-PUTs it if
absent (idempotent). Two concurrent base publishers write the same path and exactly one CAS
applies; the loser absorbs its `ParentMoved`, re-reads `current_version = B` and proceeds with
parent B.

**identity(base) is the table's `.materialization.json` `artifact` hex exactly**, byte-equal to the
`ResultDigest` anchor dependents already hold, so publishing the base changes no downstream anchor
and a `NoChange` refresh leaves every `derives_from` dependent `Fresh`. Every later version,
compaction included, folds `parent_identity`; the base is the only chain root; only a recompute to
a NEW table starts a new chain (D9).

**D4 — `_content_hash` is a NULLABLE fifth column of `embedding_table_schema`.**
(`crates/jammi-db/src/store/schema.rs::embedding_table_schema`, `::CONTENT_HASH_COLUMN`.) `Utf8`
hex SHA-256 over the embedded columns in the descriptor's `columns` order, domain tag
`jammi.content_hash.v1`, one self-delimiting part per column `[tag u8][len u64 LE][bytes]` with
tags `s` (string, after the cast), `b` (binary-family bytes; dead for text tasks because
`validate_text_column` refuses binary, kept for the image path-string case) and `n` (null)
(`crates/jammi-db/src/store/content_hash.rs::content_hash_row`, `::CONTENT_HASH_DOMAIN`). The hash
is computed INSIDE the `jammi_content_hash` UDF
(`crates/jammi-ai/src/query/content_hash_udf.rs::ContentHashUdf`) over the RAW columns using the
inference runner's own rendering (`crates/jammi-ai/src/inference/mod.rs::validate_text_column`
plus `arrow::compute::cast`; rows are joined by `::arrow_to_texts`), so base embed, refresh and
inference share one rendering by construction. A SQL-side `CAST` with a coarser rendering was
rejected: it could hash distinct values equal, classify the row `Unchanged`, and serve a STALE
vector — a wrong answer, not a missed optimisation.

Model, task and device are folded once by the table's `definition_hash`, never per row. Only
`EmbeddingPipeline` writes real hashes; every hand-built embedding batch (a materialised context
set, a propagation, a fixture, a bench corpus) goes through the one builder
`crates/jammi-db/src/store/schema.rs::embedding_batch_with_null_hash` (D13). A NULL, malformed or
absent hash at refresh is `NotRefreshable { MissingContentHash }`; the remedy is one `recompute`.
Path-valued image columns hash the path, not the file bytes. Embedded-versus-remote byte identity
is unaffected: both transports run the same pipeline.

**D5 — `publish_version` is the sole commit point.**
(`crates/jammi-db/src/catalog/version_repo.rs::Catalog::publish_version`.) One transaction: renew
the version lease by CAS; `UPDATE result_table_versions SET status = 'ready', identity, live_rows,
masked_rows, completed_at, lease_expires_at = NULL` where the row is `(table, N)`, `building`,
under this `writer_id` (one row); `UPDATE result_tables SET current_version = N, row_count =
live_rows, input_anchors_json = $anchors WHERE table_name AND status = 'ready' AND current_version
= P AND <tenant arm>` (one row). Any zero-row statement rolls the whole transaction back. A miss on
the version row is classified like a building-table miss (`LeaseLost` / `CasFailed` / `RowGone` /
`TenantMismatch`); a miss on the table row goes through `Catalog::classify_ready_cas_miss`, which
yields `ParentMoved { expected, found }` when the row is `ready` but `current_version` disagrees
with `P`, and `CasFailed` for any non-`ready` shape. `parent < N` is checked before the
transaction opens rather than as a SQL conjunct, because a conjunct miss would be misnamed
`CasFailed { status: "ready" }`.

`<tenant arm>` is the STRICT pair `tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)` on the
allocation `UPDATE`, the base CAS, the publish CAS and every `result_table_versions` write (no arm
inside an admin scope): a scoped tenant can READ a GLOBAL table through the read-side OR-NULL
filter but cannot refresh, compact or expire it. The same comparison gates step 0
(`embedding_refresh.rs::InferenceSession::refreshable_record`), before the model load and the
scans, yielding `TenantMismatch`. `classify_ready_cas_miss` is owner-less on purpose: a
`result_tables` row that a concurrent recovery flipped to `building` must never read as
`LeaseLost` to a refresher; its re-read of the row is unscoped so `TenantMismatch` versus `RowGone`
is decidable.

Nothing is visible before the CAS: an unpublished delta segment row carries `version = N`, the
unversioned read path lists `WHERE version IS NULL`, the versioned path reads only a READY manifest
reached through `current_version`, and the DataFusion provider is re-bound only after the
transaction (D8). Retrying after `ParentMoved` (re-read the table row, restart from the new
parent), `CasFailed` or `LeaseLost` is the consumer's; all three map to gRPC `Aborted`.

**D6 — Actuators, no control loop.** `refresh_embeddings(table, RefreshOptions) -> RefreshReport`,
`compact_embeddings(table) -> RefreshReport`, `expire_versions(table, before) -> ExpiryReport`.
The engine ships the actuator and never the loop that pulls it (the stance
`crates/jammi-ai/src/pipeline/recompute.rs` states in its module doc): there is no background
compactor, GC loop or scheduler, and when to refresh, compact or expire is the consumer's. All
three verbs are exposed on the remote surface (`EmbeddingService` in
`crates/jammi-wire/proto/jammi/v1/embedding.proto`, `crates/jammi-client`, the Python client) — the
library is never less capable than the server. Catalog names describe mechanism, never governance
(`allocate_result_table_version`, `publish_version`, `fail_building_version`,
`list_live_building_versions`, `list_expired_building_versions`, `list_base_index_segments`,
`insert_index_segment_for_version`, `purge_segments_for_version`, `reap_version_artifacts`).

**D7 — Deletes default to tombstone; `DeletePolicy::Retain` is opt-in** and recorded in the delta
descriptor, so it folds into the version identity
(`crates/jammi-db/src/store/manifest.rs::DeletePolicy`). A table that keeps rows its source no
longer has is not "the definition over the source" in the materialization contract's sense
(`docs/guide/src/materialization-contract.md`); rolling-window knowledge is the consumer's.

**D8 — `bind_result_table(ctx, record)` is the one registration path for a ready table.**
(`crates/jammi-db/src/store/mod.rs::ResultStore::bind_result_table`.) `current_version` None → a
single `ListingTable` over the base Parquet; Some(N) → a `MaskedTableProvider` built from version
N's manifest and registered through `ResultTableSchemaProvider::add_result_table(name, provider,
owner)`; manifest unresolvable → the placeholder provider (D14). It is called at startup
(`ResultStore::load_existing_tables`), by `BuildingTable::finish` (always the None arm for a fresh
table), after the base publish and after every `publish_version`. `read_vectors`
(`crates/jammi-db/src/session.rs`) keeps the raw base-bytes read, byte- and order-identical, while
`current_version` is NULL, and reads `_row_id, vector ORDER BY _row_id` through the masked provider
once it is Some — the documented key order. `read_keyed_vectors` is exempt: it reads an external
URL.

The binding is per session and is rewritten only at session open and after THIS store's own
publish, so a session can hold a registration older than the catalog's `current_version` when a
sibling store publishes. An ad-hoc read served from it is a consistent older snapshot. A producer
that PERSISTS an artifact whose provenance names the table must not read through it: the refresh
and the compaction both read the parent through a provider built from the parent manifest they
resolved (`ResultStore::build_masked_provider`), and other persisting producers use
`ResultStore::pin_current_version` / `::pinned_provider` (`PinnedSource`), which resolve the anchor
and the read from one row fetch. Reading the session binding instead would let a stale binding
turn a compaction into the silent loss of every row added since.

**D9 — Replay of a versioned table is a full `EmbeddingPipeline::run` over the current source into
a NEW single-segment table**: value-equivalent, identity distinct, a new chain root.
`ResultStore::producing_descriptor` returns the CURRENT version's manifest descriptor (`Embedding`
at the base, `EmbeddingDelta` after a refresh, `EmbeddingCompaction` after a compaction), so both
replay arms in `crates/jammi-ai/src/pipeline/recompute.rs::replay_descriptor` are reachable and
the doc-parity check's replay leg is not vacuous. `.materialization.json`'s shape is unchanged; the
two variants are recorded only in `.version.json`, which carries its own `VERSION_FORMAT = 1`
reject-newer guard (`crates/jammi-db/src/store/version.rs::VERSION_FORMAT`).

**D10 — A deterministic total order at the refresh plan AND both base sites, with the null-key
check below the sort.** `InferenceExec` declares one output partition but forwards
`execute(partition)` to its input, and the optimizer parallelises the hash projection with a
round-robin repartition, so a plan executed at partition 0 would otherwise see a fraction of the
rows. One plan shape serves the embed pipeline, `infer` and the refresh
(`crates/jammi-ai/src/operator/ordered_input.rs::ordered_input`):

```text
scan (n partitions) → CoalescePartitionsExec → KeyCheckExec → SortExec → InferenceExec
```

Sort keys, a TOTAL order: `(CAST(key AS Utf8) ASC NULLS LAST, _content_hash ASC NULLS LAST)`, where
`_content_hash` is the `jammi_content_hash(...)` projection every source query carries (§2.2; at
the `infer` site it is an input-only column the fixed output schema never carries). Rows tied on
both keys have equal `_row_id` and equal content and are mutually substitutable: a vector is a
function of (text, batch composition, device), and every non-vector column of tied rows is
identical, so the byte stream is invariant under permuting ties. A partial key is not enough
because the arrow sort is unstable (`arrow-ord` `sort.rs::sort_unstable_by`) and the coalesce
interleave is nondeterministic. The physical `CastExpr` defaults to `safe: false`
(`datafusion-physical-expr` `expressions/cast.rs::DEFAULT_CAST_OPTIONS`), so a key that cannot
render fails loudly and never becomes a null.

DataFusion facts the shape depends on: `CoalescePartitionsExec` outputs `UnknownPartitioning(1)`;
`SortExec` with `preserve_partitioning = false` outputs one partition but sorts only the input
partition it is asked to execute and merely declares `Distribution::SinglePartition`
(`sorts/sort.rs::SortExec::required_input_distribution`) — satisfied by the optimizer in a planned
query, not in a hand-built plan, hence the explicit coalesce. With `[inference] partitions > 1`,
`crates/jammi-ai/src/operator/inference_exec.rs::wrap_with_split_and_merge` splits below and
merges above `InferenceExec` on an `_ordinal` column, preserving this order.

Memory: the global sort is blocking; its reservation is bounded by the session's `[engine]
memory_limit` pool (`crates/jammi-db/src/session.rs` installs a `GreedyMemoryPool`). The pipeline
collects every output batch anyway, so peak memory roughly doubles rather than changing class.
Consequence: a fresh table's row order is key order, and its bytes are identical across
`execution_threads` — the engine's same-box determinism contract.

**D11 — Uniqueness.** The initial embed tolerates duplicate source keys (a bare `SELECT`, the sink
writes every ok row, `SidecarIndex::add` overwrites its `row_index` entry); duplicates keep plan
order among themselves, made deterministic by D10. Refresh refuses: `NonUniqueKey { table, scan,
keys: Vec<(String, u64)> (<= 10), total }` after a COMPLETE scan, checked on BOTH the source scan
(`NonUniqueScan::Source`) and the parent's masked current-state scan (`NonUniqueScan::Parent` — two
physical rows under one `_row_id`; recompute once). A delta over a non-unique key space is
ambiguous, so it is refused before any new version is allocated. On a first refresh of an
unversioned table the base version published by step 1 is retained — downstream-invisible, since
identity(base) is the base artifact hex (D3). A null key on the same scan is refused first:
`InvalidKey` is raised by the stream at end of input, before the post-scan uniqueness check.

**D12 — Realized counts; an all-invalid delta is a legitimate publish.** `ResultSink::write_batch`
returns the ok row ids it realized (`crates/jammi-ai/src/pipeline/result_sink.rs`);
`RefreshReport.dropped_rows = |(Added ∪ Changed) \ realized|`. The mask entry for a dropped
`Changed` key is still written — a full re-embed drops that row too, so the result is
value-equivalent. An all-invalid delta does NOT inherit the base embed's all-invalid refusal: it is
a mask-only publish (no fragment, no segment, mask entries written, `row_count` reduced).
`live_rows` at publish is `COUNT(*)` over the masked provider of the not-yet-published manifest
(`ResultStore::count_live_rows` — exact, in row space, masked by the projection contract of §2.5);
`masked_rows = Σ fragment rows − live_rows`, where a delta fragment's rows are the sink's realized
count and the base fragment's are the table row's `row_count`. Emptiness policy is
backend-specific: candle marks an empty text a per-row error; the HTTP backend short-circuits an
empty input LIST and otherwise forwards texts verbatim, so per-row validity there is the remote's.

**D13 — The null-hash batch builder lives in `jammi-db`.**
`crates/jammi-db/src/store/schema.rs::embedding_batch_with_null_hash`, re-exported by
`jammi-test-utils` for test sites; `jammi-bench` and the `jammi-db` library sites call it directly.
One builder means no hand-built five-column batch can disagree about the hash column's shape.

**D14 — Reconcile, recovery, and the two failure classes of a version.**
`referenced_result_keys` protects, per ready table, every READY version row's manifest, fragment
and deletes URLs, and every LIVE-building version's deterministic `__v{N}*` URLs and `version = N`
segments (`Catalog::list_live_building_versions`). Expired-lease building versions are
unreferenced and are reaped by recovery's version arm
(`crates/jammi-db/src/store/mod.rs::ResultStore::recover_expired_versions`: claim → fail CAS →
`reap_version_artifacts`). Recovery never promotes a version — a delta is cheap to redo; the
promote-from-sidecar arm exists for whole tables only.

Two failure classes, kept apart:

1. **Manifest-RESOLUTION failure** — the current version row is not `ready`, or its manifest is
   definitively absent on an `exists()` probe — is the typed `VersionUnavailable { table, version
   }`, raised by `ResultStore::resolve_version_manifest` and therefore by the ANN path and the SQL
   path alike. "Only definitive absence fails the VERSION row" (`status = 'failed'` via
   `Catalog::fail_ready_version`, `current_version` unchanged, the `result_tables` row untouched;
   the version arm of `ResultStore::reconcile_ready_manifests`) is a rule about `exists()` probes
   and never about a scan `?`: a mid-scan not-found synthesised by the classifier (D20) is
   indistinguishable from definitive absence by shape. An `exists()` ERROR propagates as a storage
   error. For such a table `bind_result_table` registers a PLACEHOLDER provider
   (`crates/jammi-db/src/store/masked_provider.rs::PlaceholderProvider`): registered under the
   row's owner so tenant visibility still hides it from peers (no existence disclosure), schema =
   `embedding_table_schema(dimensions)` off the catalog row so planning succeeds, and every `scan`
   returns `DataFusionError::External(Box::new(JammiError::VersionUnavailable { .. }))`, which the
   classifier restores to the typed variant for every SQL caller. Registering nothing was rejected
   (a stringly planner not-found); registering the base provider was rejected (it would resurrect
   deleted rows). A versioned row with no `dimensions` is a catalog invariant violation:
   `bind_result_table` returns a catalog error, and startup logs it and continues.
2. **Segment-BUNDLE load failure** → the whole-table exact fallback, over the MASKED provider.

`reconcile_ready_manifests` keeps its table-level arm for `.materialization.json` only;
`purge_segments` and `delete_objects_after_cas` stay table-scoped and reachable only from the
table-level building/failed arms (invariant I-A2). The remedy for `VersionUnavailable` is
`recompute` (a new table).

**D15 — Byte identity is asserted at the VERSION identity.** The determinant set of a version's
bytes is f(row MULTISET, D10's total order, write-batch cadence, pinned writer properties). Writer
properties are pinned (ZSTD, 64K row groups, `crates/jammi-db/src/storage/writer.rs`). Cadence:
after the sort, output batches are `engine.batch_size`-row slices of the total order (DataFusion's
`ExternalSorter` is built with the session `batch_size` and emits through it on both its in-memory
and streaming-merge paths, `sorts/sort.rs::sort_batch_chunked`), and the runner re-chunks each
input batch into `inference.batch_size` slices with a short remainder per input batch — so the
cadence is deterministic in `(engine.batch_size, inference.batch_size, rows)` provided the sort
does not spill and the runner's OOM batch-halving (which persists for the run) does not fire. The
same operation on the same host with `engine.batch_size`, `inference.batch_size` and
`embedding.checkpoint_interval` equal on both configs yields equal identity, fragment digests,
deletes digest and counts on the embedded and the remote transport; `execution_threads` may
differ; `produced_by` / `produced_at` differ. Delta-versus-full and compaction-versus-chain are
different operations: value-equivalent, identity distinct.

**D16 — One appended migration.** `032_result_table_versions`
(`crates/jammi-db/src/catalog/migrations.rs`, DDL in `catalog/schema.rs`, §2.10). The migration
ledger is append-only; an applied migration is never renumbered.

**D17 — Sharding seam.** A sharded refresh is a design seam only (§2.9): `ComputeSpec::Embedding`
(`crates/jammi-ai/src/jobs.rs`) carries no key-range or target-version field, and this design adds
no wire change for it. What exists is every primitive a fan-out would call: a version number
allocated once before any write, version-stamped segment append under the version's lease, and a
single publish CAS.

**D18 — The Tier 2 line.** A TABLE-level monotonic source version is a freshness surface the
manifest already anticipates (`AnchorKind::MutableVersion`, which no producer emits and the
freshness layer resolves to `Undecidable`; `crates/jammi-db/src/store/freshness.rs` module doc).
ROW-level change tracking on mutable tables is a transition log and violates the leak-guard in
`docs/guide/src/philosophy.md#leak-guards` — it is not an engine primitive. The refresh covers
mutable tables through the generic full scan. `staleness` does not report "a delta is available";
that belongs with a table-level source version.

**D19 — Definition drift.** `definition_of(the Embedding descriptor rebuilt from the current
version's parameters, env)` ≠ `result_tables.definition_hash` → `DefinitionDrift { table, recorded,
current }` before any allocation (`embedding_refresh.rs::InferenceSession::check_definition_drift`);
the consumer runs `recompute`. The step-0 model load is this check's INPUT — the environment
(`embedding_dim`, backend kind, compute precision, content digest, quantization) is derived from
the loaded model — so the load is never deferred past the empty-delta return: `_content_hash`
excludes model identity by design (D4), and a pure model drift would otherwise read as `NoChange`.
`result_tables.definition_hash` is never rewritten by refresh or compaction: it names HOW the table
is produced; per-version identity lives on the version row. A row that records no
`definition_hash` is refused as `NotRefreshable` rather than folding an empty definition into the
identity chain.

**D20 — Error classification is structural: the manual `From<DataFusionError> for JammiError` IS
the classifier.** `JammiError::DataFusion(#[source] DataFusionError)` carries `#[source]` and not
`#[from]`: thiserror's `#[from]` implies `#[source]`, so dropping both would silently remove
`source()` from a public type. With the conversion written by hand
(`crates/jammi-db/src/error.rs`, §2.12), every existing `?` routes through the classifier by
construction, and `map_engine_error`'s wildcard still folds `DataFusion` and `Storage` alike, so
the gRPC code of an unclassified error is unchanged. Shapes, in order:

- **(a) owned destructuring** for the passthrough arm — `External(b)` →
  `b.downcast::<JammiError>()` returns the inner error itself; `Context(_, b)`, `ArrowError(b, _)`
  and `ParquetError(b)` are recursed BY VALUE (Box payloads). `Shared(Arc<_>)` cannot yield
  ownership and is a stated fidelity limit of this shape. `Diagnostic` and `Collection` are not
  recursed; at DataFusion 52.4 only the SQL planner's column/table-not-found and error-collector
  paths produced them, never with an `External(Box<JammiError>)` payload.
- **(c) resource exhaustion** — a `ResourcesExhausted` anywhere in the `source()` chain becomes the
  typed `JammiError::ResourcesExhausted`, checked before (b) so a pool exhaustion is never mistaken
  for an object-store miss.
- **(b) a borrowed `std::error::Error::source()` walk** for path-only classification: an
  `object_store::Error::NotFound { path, source }` found at any depth — through
  `ParquetError::External` (parquet's `From<object_store::Error>` wraps it there),
  `DataFusionError::ParquetError`, `ArrowError::ExternalError`, or the top-level
  `DataFusionError::ObjectStore` — becomes `JammiError::Storage(StorageError::Io { path, source:
  object_store::Error::NotFound { .. } })`, the EXISTING spelling every reader already matches; no
  new `StorageError` variant. Neither a top-level `ObjectStore` arm nor
  `DataFusionError::find_root` (which tracks only the lowest `DataFusionError`) reaches a NotFound
  nested under `ParquetError::External`, which is why the walk is by `source()`.

Every other DataFusion error keeps the shape `JammiError::DataFusion(e)` with `source()` intact.
The execute/collect sites the typed errors cross — the embed pipeline, `infer`, the runner's
input stream, `read_vector_by_key` — convert with `JammiError::from` rather than stringifying, so
`InvalidKey` reaches an embed caller and the placeholder's `VersionUnavailable` reaches
`search_by_id`'s query-by-example path typed.

**D21 — NULL keys are one typed refusal on all three paths (base embed, refresh, `infer`).** A
NULL in the key column is `JammiError::InvalidKey { column: String, null_count: u64 }`. `Schema {}`
was rejected because it states a type disagreement, `Source` because it names the source, not the
row. "Skip and count" exists nowhere: `RefreshReport` carries no invalid-key counter, and a refresh
that refuses leaves the previous version live (versions are immutable), so refusal is safe on every
path. Grounds: invalid input is refused typed at the input edge; the base paths have no report
carrier (`EmbeddingPipeline::run` and `infer` return a record or batches plus a `CacheOutcome`);
`infer`'s per-row disclosure contract (`docs/guide/src/generate-embeddings.md`; `_row_id` is
non-nullable) forbids a silent drop; without the check a null key survives the runner's cast and
fails the WHOLE run as a stringly arrow error from `RecordBatch::try_new`, after model calls.

Mechanism: the `KeyCheckExec` physical node (§2.11), a partition-preserving passthrough placed
BELOW the blocking `SortExec` of D10. It counts nulls in the raw key column across every input
batch, passes rows through unchanged, and at end of input raises `InvalidKey` with the EXACT total
if it is non-zero. Because `SortExec` emits nothing until its input is exhausted (its `execute`
drains `while let Some(batch) = input.next().await { … sorter.insert_batch(batch).await? }` before
sorting), `InferenceExec` pulls zero batches before the refusal: zero model invocations BY
CONSTRUCTION, no second scan of the source, an exact count. The runner-side check after its own
cast is defensive only — a hand-built plan that bypasses `KeyCheckExec` still gets the typed
variant.

**D22 — Wire fidelity for every caller-facing variant.** `InvalidKey`, `VersionUnavailable`,
`NotRefreshable`, `DefinitionDrift`, `NonUniqueKey` each have a `JammiErrorDetail` oneof arm
(`crates/jammi-wire/proto/jammi/v1/error.proto`, tags 32–36; `ParentMoved` is 39), encode/decode
arms and a round-trip entry
(`crates/jammi-wire/src/error.rs::every_owned_shape_variant_round_trips_to_itself`), and a
`map_engine_error` arm (`crates/jammi-server/src/grpc/wire.rs::map_engine_error`):

| Variant | gRPC code | Convention |
|---|---|---|
| `InvalidKey`, `NonUniqueKey` | `InvalidArgument` | data-shape faults of the caller's input, as `Schema` / `Source` |
| `DefinitionDrift` | `FailedPrecondition` | the environment, not the argument, must change before a retry |
| `NotRefreshable` | `FailedPrecondition` | "fix state, then retry", as `ModelReferenced` / `SourceBusy` |
| `VersionUnavailable` | `NotFound` | the absent-resource convention of `ModelNotFound` / `RowGone` |
| `ParentMoved`, `CasFailed`, `LeaseLost` | `Aborted` | the caller's view was stale; retry from a fresh read |

Python leaf classes (`clients/python/jammi/errors.py`): `InvalidKey(InvalidArgument)`,
`NonUniqueKey(InvalidArgument)`, `NotRefreshable(BackendError)`, `DefinitionDrift(BackendError)`,
`VersionUnavailable(BackendError)`; the embedded engine raises them by variant name
(`crates/jammi-python/src/error.rs::jammi_error_class`). Each leaf subclasses the class the remote
mapper already produces for its gRPC code (`clients/python/jammi/_database.py::_rpc_to_jammi` maps
`INVALID_ARGUMENT` → `InvalidArgument`, `UNIMPLEMENTED` → `NotSupportedOnBackend`, everything else
→ `BackendError`; the Python client decodes no status details), so `except InvalidArgument` /
`except BackendError` behave identically on both transports. Leaf-level parity on the remote
transport is out of scope (§5).

## 2. Design

### 2.1 Artifacts and catalog rows

Siblings of the table's Parquet via `crates/jammi-db/src/store/layout.rs`
(`::version_manifest_url`, `::version_fragment_url`, `::version_deletes_url`, `::segment_url`);
`{table}.parquet` stays `parquet_path` and is never rewritten or deleted while the table lives:

| Artifact | Written by | Stamp | Notes |
|---|---|---|---|
| `{table}.parquet` + `.materialization.json` | base embed | version B in the base manifest | five columns |
| `{table}__seg{id}.idx.*` | `append_segment` / `append_segment_for_version` | `index_segments.version` (NULL = base; read as B through a manifest) | one URL layout for both |
| `{table}__v{N}.parquet` | refresh N (delta rows), compaction N (all live rows) | N | five columns; absent when a delta realized zero rows |
| `{table}__v{N}.deletes.parquet` | refresh N | N | cumulative mask (D1); absent when empty |
| `{table}__v{N}.version.json` | base publish / refresh / compaction | N | `VersionManifest` |

`VersionManifest` (`crates/jammi-db/src/store/version.rs::VersionManifest`, `VERSION_FORMAT = 1`,
reject-newer → `IncompatibleFormat`): `version_format, table, version, parent, definition_hash,
delta { descriptor, input_anchors }, fragments [{ url, version, rows, digest }], segments [{
segment_id, version }], deletes { url, entries, digest } | null, live_rows, masked_rows, identity,
produced_by, produced_at, engine_version`.

Catalog `result_table_versions` (§2.10): PK `(table_name, version)`, `parent_version`, `status`
(`building` / `ready` / `failed`), `manifest_path`, `identity`, `live_rows`, `masked_rows`,
`writer_id`, `lease_expires_at`, `tenant_id` (the parent row's, read in the same transaction),
`created_at` (app-supplied), `completed_at`. `index_segments.version INTEGER` (NULL = base).
`result_tables.current_version INTEGER` (NULL = never refreshed), `result_tables.next_version
INTEGER NOT NULL DEFAULT 0`.

Readers of a version: `resolve_search_mode_local` (§2.4), `bind_result_table` and
`build_masked_provider` (§2.5), `verify_materialization` and `current_anchor` (§2.6),
`producing_descriptor` (§2.7), the refresh and the compaction (§2.3, §2.8), `expire_versions`
(§2.8), `referenced_result_keys` / `required_row_objects_present` and recovery (D14).

### 2.2 `_content_hash` production

`jammi_db::store::content_hash` holds the pure fold (`content_hash_row`, `ContentHash`) and an
Arrow kernel over already-rendered columns; `jammi_ai::query::content_hash_udf` registers the
variadic `ScalarUDF` `jammi_content_hash(...)`, which does the rendering.
`crates/jammi-ai/src/session.rs::InferenceSession::build_source_query` — shared by embed, `infer`
and refresh — projects `jammi_content_hash("c1", "c2", …) AS _content_hash` over the RAW columns,
with no SQL `CAST` (D4). `InferenceExecBuilder::passthrough(Vec<String>)` copies named input
columns to the output (the runner copies the arrays per emitted sub-batch); the embed and refresh
sites pass `["_content_hash"]`, `infer` passes nothing because its output schema is fixed.
`crates/jammi-ai/src/pipeline/result_sink.rs::filter_ok_and_extract_vectors` maps `_content_hash`
to the fifth column (NULL when absent). The UDF's typed refusals (a binary column for a text task;
a column the runner cannot render) surface through the classifier's shape (a). Every reader decodes
a stored hash with `ContentHash::from_hex`, which accepts exactly 64 lowercase hex characters. A
table with no `_content_hash` column, or a NULL hash, is `NotRefreshable { MissingContentHash }`.

### 2.3 Refresh algorithm — `refresh_embeddings(table, RefreshOptions { deletes }) -> RefreshReport`

`crates/jammi-ai/src/pipeline/embedding_refresh.rs::InferenceSession::refresh_embeddings`, exposed
on `local_session::Session` beside `recompute`.

0. **Gates** (`refreshable_record`). Tenant-scoped `get_result_table` plus the strict-pair
   comparison (D5); then `status = ready` (`NotReady`), `kind = model` with an embedding task
   (`NotEmbeddingTable`), and — when `current_version` is Some — a `ready` version row
   (`CurrentVersionUnavailable`); each failure is `NotRefreshable { reason }`. Read the current
   descriptor (§2.7) → `{model_id, task, source_id, columns, key_column, dimensions}`; load the
   model and build its environment; D19 drift check.
1. **Base publish** (D3) when `current_version IS NULL`; afterwards P = `current_version` is
   always Some.
2. **Parent.** `ResultStore::resolve_version_manifest(record, P)`: the version row must be `ready`
   and the manifest present, else `VersionUnavailable`; an `exists()` error propagates. Manifests
   are immutable once ready and are cached per `(table, version)` beside the loaded segment sets.
3. **Current state.** `_row_id, _content_hash` read through a masked provider built from P's
   manifest (not the session binding, D8) → `HashMap<String, ContentHash>` (~100 B/row in memory;
   spilling it is a seam). Duplicate `_row_id` → `NonUniqueKey { Parent }`; a NULL or malformed hash
   → `NotRefreshable { MissingContentHash }`.
4. **Source scan.** A hand-built plan — `build_source_query`'s physical plan →
   `CoalescePartitionsExec` → `KeyCheckExec(key_column)` (`ordered_input.rs::key_checked`) —
   streamed and drained completely by the classifier, which renders the key with the same Utf8
   cast the runner uses so the key space equals the stored `_row_id` space. A null key → the stream
   ends with `InvalidKey { column, null_count }` (no delta version allocated, no model call; a base
   version published by step 1 is retained). A duplicate key → collected, `NonUniqueKey { Source }`
   after the complete scan (<= 10 keys with exact counts). Each key is classified `Added` (not in
   current), `Changed` (hash differs) or `Unchanged`; `Deleted = current \ seen` unless `Retain`.
5. **Empty delta** → `RefreshReport { outcome: NoChange }`: nothing is written for the delta, no
   delta version is allocated, anchors are untouched; a base version published by step 1 is
   retained.
6. **Allocate** version N (`ResultStore::allocate_version` →
   `Catalog::allocate_result_table_version`, D2). One write transaction whose FIRST statement is
   `UPDATE result_tables SET next_version = next_version + 1 WHERE table_name = $t AND status =
   'ready' AND current_version = $P AND <tenant arm>` (one row, else `classify_ready_cas_miss`),
   then `SELECT next_version, current_version, …` in the same transaction → `N = next_version − 1`
   → `INSERT result_table_versions (t, N, P, 'building', manifest_path, writer_id, lease, tenant,
   created_at)`. The transaction issues NO read before the UPDATE: a read-then-write transaction on
   SQLite/WAL can fail `SQLITE_BUSY_SNAPSHOT`, which `busy_timeout` does not retry; on Postgres the
   row lock serialises allocators under the READ COMMITTED the catalog uses (REPEATABLE READ would
   abort the second allocator). The returned `BuildingVersion`
   (`crates/jammi-db/src/store/building_version.rs`) keeps the version lease renewed on the
   process's lease keeper; dropping it without a publish or abort spawns a best-effort `building →
   failed` CAS on the VERSION row with no byte deletion.
7. **Infer the delta.** The `(Added ∪ Changed)` keys, sorted, become an in-memory build side; plan =
   `HashJoinExec(Inner, CollectLeft, build = keys, probe = the source scan with its hash
   projection, on CAST(key AS Utf8) = key)` → `ordered_input` (coalesce → `KeyCheckExec`, a no-op
   guard here since step 4 already refused nulls → the D10 sort) → `InferenceExec` with
   `passthrough(["_content_hash"])`. `ResultSink::for_version_fragment` writes
   `{table}__v{N}.parquet` and a `SidecarIndex` at the table's persisted precision; `write_batch`
   returns the realized ok ids. Lease lost mid-stream → the sink is dropped and the refresh returns
   `LeaseLost`. Zero realized rows → the empty Parquet object is deleted: no fragment, no segment
   (D12).
8. **Segment.** `BuildingVersion::append_segment(index)` →
   `Catalog::insert_index_segment_for_version`: the same allocation loop as a base append, with
   the predicate "version row `(t, N)` is `building` under this `writer_id`" (tenant arm) in place
   of the table-row lease; it stamps `version = N` and saves the bundle second, so a save failure
   leaves a row with an absent bundle for the version's own reap.
9. **Deletes.** Start from the parent's mask; for `K ∈ Changed ∪ Deleted`: `entry[K] =
   max(entry[K], N−1)`. Before anything is written, a realized key that the mask already hides at
   version N is a typed refusal (`refuse_if_realized_key_is_masked`): it means the parent's
   horizons are not monotonic, and publishing would hide the version's own new rows. A typed error
   rather than a debug assertion, because an assertion compiles out of a release build. Write
   `{table}__v{N}.deletes.parquet` unless the mask is empty.
10. **Manifest.** `fragments = parent.fragments ∪ {new}` (`{url, version, rows, digest =
    ArtifactDigest::of_bytes}`), `segments = parent.segments ∪ {new}`, `deletes`,
    `delta.descriptor = ProducingDescriptor::EmbeddingDelta { model_id, task, source_id, columns,
    key_column, dimensions, parent_version: P, parent_identity, deletes: Tombstone | Retain }`,
    `delta.input_anchors = [unpinned_at_instant(source, now)]`, `live_rows` / `masked_rows` per
    D12, `identity` per §2.6. Write `{table}__v{N}.version.json`.
11. **Publish** (D5); then `bind_result_table` (which evicts the table's loaded segment sets) and
    `ann_cache.invalidate_source(source_id)`. On any publish failure — `ParentMoved` because a
    concurrent refresh or compaction already published past P, `CasFailed`, `LeaseLost` — the
    writer calls `BuildingVersion::abort`: `fail_building_version` CAS, then
    `reap_version_artifacts(N)`.
12. **Report** `RefreshReport { table, version, parent_version, inferred_rows, added, changed,
    deleted, unchanged, dropped_rows, live_rows, masked_rows, outcome: Published | NoChange }`
    (`crates/jammi-wire/src/embedding_refresh.rs`, one vocabulary for the local session and the
    remote client).

Failure modes: any error in 6–11 unwinds through `BuildingVersion`'s drop (version row → failed,
bytes left for recovery's reap); an explicit `abort()` deletes only artifacts stamped N
(`reap_version_artifacts`: `__v{N}.parquet`, `__v{N}.deletes.parquet`, `__v{N}.version.json`, and
`purge_segments_for_version(t, N)` = the `index_segments` rows with `version = N` plus their bundle
siblings) — never `{table}.parquet`, `.materialization.json`, or a `version IS NULL` segment. An
aborted or refused refresh therefore lands no terminal write on the table: the previous version
stays live. An expired lease → recovery (D14). `result_tables.status` / `writer_id` are never
touched by refresh or compaction (I-A2).

### 2.4 Read path — ANN

`ResultStore::resolve_search_mode_local(table)`: `current_version` None →
`list_base_index_segments` (`WHERE version IS NULL ORDER BY segment_id`) and the unversioned
path; Some(N) → the `SegmentSetCache` entry for `(table_name, Some(N))` → `LoadedSegmentSet {
index: Arc<SegmentedIndex>, mask: Arc<DeletionMask> }`, loaded by resolving version N's manifest
(resolution failure → `VersionUnavailable`, D14 class 1), loading ONLY its listed segments through
the content-addressed segment cache, loading the deletes file into a `DeletionMask`, and building
`SegmentedIndex::new_masked(segments_with_versions, mask)`. A bundle load failure → `None` → exact
fallback over the bound (masked) provider (D14 class 2) — never an index over the surviving
subset, which would silently make rows unsearchable. A never-refreshed `ready` table caches under
`(table_name, None)`: its base segment set is frozen, because a base segment insert requires the
table `building`. Eviction is per table, on `bind_result_table` (so on every publish) and on table
delete. Cache soundness: a version's SEMANTIC content is immutable once ready — the manifest is
written before the publish, a losing concurrent base publisher's PUT may land after the winner's
CAS differing only in `produced_by` / `produced_at`, which verify and identity exclude, and a
number is never reused.

`ResultStore::resolve_search_mode` is the placed entry for online search. When this process owns
every segment it delegates to `resolve_search_mode_local`, so the mask is applied. When at least
one segment is owned by a peer it builds the placed index from the flat `list_index_segments`
listing, which is not version-aware: peer-placed search over a refreshed table is outside this
design.

`SegmentedIndex` masked search (`crates/jammi-db/src/index/segment.rs`): at construction each
segment records `dead = |{K : mask[K] >= segment.version ∧ segment contains K}|`
(`SegmentedIndex::count_dead`; a `row_index` lookup per mask entry, O(|mask| × N_seg) once per
load). Per query and segment (`SegmentedIndex::live_candidates`): `w = over_fetch(m, N_seg)`; if
`dead > 0`, `w = ceil(w / (1 − dead/len))` capped at `len`; loop `hits = segment.search(q, w)` →
drop `K` where `mask[K] >= segment.version` → stop when `>= m` live or `w >= len`, else `w =
min(2w, len)`. An empty mask → exactly one search at the unmasked width, so
`n1_f32_search_is_byte_identical_to_the_lone_sidecar` holds unchanged. Candidates carry their
`SegmentId` into `search_final`; rescore reads the exact vector from the OWNING segment
(`SegmentedIndex::exact_in`), never the first segment that happens to index the same key; the
dedup pass stays as defence. Per-query cost: `N_seg` probes × widening rounds (one in the common
case) plus one candidate filter pass; memory O(|mask|) once per loaded set. The over-fetch factor
(`DEFAULT_SEGMENT_OVERFETCH_FACTOR`) is the tuning seam.

Accepted read semantics: a publish between the table-row read and the cache lookup serves the
older version, a consistent snapshot; an in-flight read holding a loaded set for a since-expired
version completes from memory (the rescore companion is an open file descriptor, which survives
unlink on POSIX; remote bundles sit in the local index cache); a read that must LOAD an expired
version fails the load → `None` → exact fallback over the currently bound provider; a mid-scan
object vanish surfaces as the typed `Storage(StorageError::Io { source:
object_store::Error::NotFound })` via D20, never a partial answer.

### 2.5 Read path — SQL / exact

`ResultStore::build_masked_provider` builds
`crates/jammi-db/src/store/masked_provider.rs::MaskedTableProvider` — one `ListingTable` per
fragment, each with its stamped version, under the manifest's mask. The `scan(projection, filters,
limit)` contract:

1. The pushed-down projection is remapped to always include `_row_id`. `COUNT(*)` and `SELECT
   vector` project no key and would otherwise be unmasked.
2. Each fragment is scanned with `limit = None`. `ListingTable` consumes `limit` two ways — file
   list truncation (`datafusion-catalog-listing` `table.rs::get_files_with_limit`) and
   `with_limit` on the scan — so a fetch pushed under the mask would return fewer than `LIMIT n`
   live rows.
3. `UnionExec` over the per-fragment scans, each wrapped in `MaskExec(fragment_version)` — an
   arrow filter on `_row_id`, skipped when no mask entry has a horizon >= that version.
4. `ProjectionExec` back to the caller's columns.

`supports_filters_pushdown` is not overridden, so the `FilterExec` stays above the mask. `MaskExec`
MUST NOT override `ExecutionPlan::supports_limit_pushdown` (default `false`) nor `with_fetch`
(default `None`): the physical `limit_pushdown` rule only pushes a fetch into a node that reports
support, so the outer fetch survives above the mask as a limit node; a unit assertion
(`masked_provider.rs::tests::mask_exec_never_accepts_a_pushed_limit`) pins both. The union schema
is pinned by passing the base fragment's inferred `SchemaRef` to every later fragment's
`ListingTable`; the inferred schema and the loaded mask are per-version-immutable and cached per
`(table, version)`, while the provider itself is rebuilt per call because building it registers
the fragment's object store on the calling `SessionContext`. One provider per fragment rather than
a multi-path `ListingTable`, because the mask is per fragment version.
`exact_vector_search`, `read_vector_by_key`, `read_vectors` (the Some arm, D8) and every user
`SELECT` therefore see only live rows. A never-refreshed table keeps its single provider; its
`EXPLAIN` shows no `MaskExec` / `UnionExec`. Under the `test-hooks` feature `MaskExec::execute`
parks at `masked_scan_test_hooks::maybe_park_before_masked_scan_drain(table)` before opening its
input stream — the pause point the mid-scan-vanish test uses.

Reader classes. Routed through the version: `resolve_search_mode_local`, startup / finish /
publish (`bind_result_table`), `read_vector_by_key`, `exact_vector_search`, `read_vectors` (Some),
`verify_materialization`, `current_anchor`, `producing_descriptor`,
`required_row_objects_present`, `referenced_result_keys` (a superset). Exempt, with reason:
`read_keyed_vectors` (external URL); `classify_expired_row` and the base index rebuild (building
rows only, I-A2); the table-level `purge_segments` / `delete_objects_after_cas`; the public
`list_index_segments` (a listing; it returns `version`); `staleness` / `lookup_cached` (unchanged:
anchors are what changes).

### 2.6 Identity and verification

`identity(base) = manifest.artifact` (D3). For every version N with parent P
(`VersionManifest::compute_identity`):

```text
SHA-256( "jammi.version.identity.v1"
       ‖ "\0parent\0"     ‖ len ‖ parent_identity
       ‖ "\0definition\0" ‖ len ‖ definition_hash
       ‖ "\0delta\0"      ‖ len ‖ canonical(delta.descriptor)
       ‖ "\0fragments\0"  ‖ Σ (len ‖ fragment.digest), manifest order
       ‖ "\0deletes\0"    ‖ len ‖ deletes.digest-or-"" )
```

Lengths are u64 LE; the fold is domain-separated in the same shape as the definition hash.
Identity folds the parent, the definition, the delta descriptor (delete policy included) and every
live artifact digest, so a version is a replayable, attributable producer output. Counts, segments
(the ANN index is never attested) and `produced_by` / `produced_at` are outputs, not inputs.
`ResultStore::current_anchor` for a table with `current_version` Some(N) returns version N's
identity from the catalog row, so a refresh that changed content advances dependents to `Stale {
InputAdvanced }` and one that did not leaves them `Fresh`. `verify_materialization` on a versioned
table runs the base check unchanged, then recomputes every listed fragment digest and the deletes
digest from the bytes, recomputes the identity, and compares it to the manifest and the catalog
row; `MatchVerdict::Mismatch` names the artifact. The user-facing statement is
`docs/guide/src/materialization-contract.md#versioned-tables-the-identity-chain`.

### 2.7 Replay

`ProducingDescriptor` (`crates/jammi-db/src/store/manifest.rs`) carries `EmbeddingDelta {
model_id, task, source_id, columns, key_column, dimensions, parent_version, parent_identity,
deletes }` and `EmbeddingCompaction { …the same parameters…, parent_version, parent_identity }`;
each has a `replay_descriptor` arm → `EmbeddingPipeline::run(…, CachePolicy::Bypass)` (D9).
`producing_descriptor` reads the current version's manifest when `current_version` is Some. Both
variants are listed in the maintainer guide's descriptor inventory, which
`ci/scripts/check_doc_parity.py` holds set-equal to the enum.

### 2.8 Compaction and expiry

`compact_embeddings(table)`: gates as §2.3 step 0 (no model load, no drift check — a compaction
runs no inference), base publish if needed, allocate N; read every live row `ORDER BY _row_id`
through a masked provider built from the PARENT manifest; write `{table}__v{N}.parquet` (all live
rows, carried hashes) and, when there is at least one row, one `SidecarIndex` at the table's
precision → a segment stamped N; no deletes; manifest `fragments = [new]`, `segments = [new]`,
descriptor `EmbeddingCompaction`, `delta.input_anchors = [result_digest(table,
parent_identity)]`; `result_tables.input_anchors_json` stays the parent's, because a compaction
reads no source; publish (D5). Precedent: Qdrant's vacuum. The threshold is the consumer's;
`live_rows` / `masked_rows` on every report are the inputs to that decision.

`expire_versions(table, before: i64) -> ExpiryReport { table, expired_versions, objects_deleted
}`: the current manifest is resolved first through `resolve_version_manifest`, because it IS the
retention set — an unresolvable or non-ready current version refuses the expiry rather than
reaping against an unchecked manifest. Then for each version row with `version < before AND
version != current_version AND status IN ('ready', 'failed')`: delete the row, then reap its
`__v{v}.version.json`, its `__v{v}.deletes.parquet`, and its fragment and every segment stamped
`v` that the CURRENT manifest does not list (`ResultStore::reap_expired_version`; a fragment
retained by reference stays). `{table}.parquet` and `.materialization.json` are never deleted
while the table lives — verify, `read_vectors`' NULL arm and recovery all depend on
`parquet_path` — so a compaction does not reclaim the base fragment. `next_version` is never
decremented. A never-refreshed table expires nothing.

### 2.9 Sharding seam (design only; no wire change)

Nothing here is built: there is no key-range or target-version field on `ComputeSpec::Embedding`,
no per-shard fragment naming, and the refresh runs in one process. The seam is that the design
admits a fan-out without changing any visibility rule. A coordinator would run §2.3 steps 0–6 and
9–11; each shard would infer one key range of the delta, write its own fragment (five columns,
sorted) plus one segment via `insert_index_segment_for_version` under the version's writer
identity — segment ids may interleave across shards, and a zero-row shard writes nothing — and
report `(url, digest, realized_rows, segment_id)`. The coordinator would build `fragments[]` from
the shard reports in partition order, never from an object listing (a missing report fails
typed), then write the deletes and the manifest and run `publish_version`. A failed shard fails
the publish; the version lease expires; recovery reaps. This is correct for n concurrent shards
because visibility depends only on the version number allocated once before the fan-out and on
the single CAS, and because masking depends only on version stamps, never on segment id order.
The engine has no job-dependency edges, so ordering the fan-in after its shards is the
orchestrating runtime's.

### 2.10 Migration DDL

One appended migration, `032_result_table_versions` (D16), portable across SQLite and Postgres
(`ci/scripts/check_sqlite_isms.py`):

```sql
CREATE TABLE result_table_versions (
    table_name       TEXT NOT NULL REFERENCES result_tables(table_name) ON DELETE CASCADE,
    version          INTEGER NOT NULL,
    parent_version   INTEGER,
    status           TEXT NOT NULL DEFAULT 'building',
    manifest_path    TEXT NOT NULL,
    identity         TEXT,
    live_rows        INTEGER,
    masked_rows      INTEGER,
    writer_id        TEXT,
    lease_expires_at TEXT,
    tenant_id        TEXT,
    created_at       TEXT NOT NULL,
    completed_at     TEXT,
    PRIMARY KEY (table_name, version)
);
CREATE INDEX idx_result_table_versions_lease ON result_table_versions(status, lease_expires_at);
ALTER TABLE result_tables ADD COLUMN current_version INTEGER;
ALTER TABLE result_tables ADD COLUMN next_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE index_segments ADD COLUMN version INTEGER;
```

`created_at` is app-supplied; one `ADD COLUMN` per statement; the lease column uses the same text
form as the result-table lease; `IndexSegment.version` is read as `Option<i64>`.

### 2.11 `KeyCheckExec` (D21)

`crates/jammi-ai/src/operator/key_check_exec.rs::KeyCheckExec { input, key_column, key_index }`,
schema = the input's. Properties: partitioning and ordering are the input's (it is only ever
placed above a `CoalescePartitionsExec`, so one partition); `supports_limit_pushdown` stays the
default `false` and `with_fetch` the default `None`, because a node that must see every row to
count can never accept a pushed fetch. `execute(p)` wraps `input.execute(p)` in a stream that, per
batch, adds the key column's `null_count()` to a counter and yields the batch unchanged; when the
input ends, if the total is non-zero it yields exactly one
`Err(DataFusionError::External(Box::new(JammiError::InvalidKey { column, null_count })))` and
then ends. The RAW key column is checked (the source's nulls); the sort's `CastExpr` above it
cannot add nulls (`safe: false`, D10). The error reaches the caller through `SortExec`'s drain →
the runner's input stream → the output channel → `collect` → `JammiError::from` (D20 shape (a)) →
`InvalidKey`.

`crates/jammi-ai/src/operator/ordered_input.rs` exposes `ordered_input(plan, key_column)` =
`SortExec(D10 keys, KeyCheckExec(CoalescePartitionsExec(plan), key_column))`, used by the embed
pipeline, `infer` and refresh step 7, and `key_checked(plan, key_column)` =
`KeyCheckExec(CoalescePartitionsExec(plan), key_column)` alone, used by refresh step 4, which has
no model below it. Under `jammi-ai`'s `test-hooks` feature (which enables `jammi-db`'s), the
runner counts `forward()` calls per source id
(`crates/jammi-ai/src/inference/runner.rs::test_hooks::forward_calls_for`), which is how the tests
observe "zero model invocations".

### 2.12 The classifier (D20)

```rust
impl From<DataFusionError> for JammiError {
    fn from(e: DataFusionError) -> Self {
        match unwrap_jammi(e) {                                // shape (a): owned passthrough
            Ok(inner) => inner,
            Err(e) => {
                if let Some(msg) = resources_exhausted_message(&e) {   // shape (c)
                    return JammiError::ResourcesExhausted { /* limit parsed from msg */ };
                }
                match not_found_path(&e) {                     // shape (b): borrowed source() walk
                    Some((path, original)) => JammiError::Storage(StorageError::Io {
                        path: path.clone(),
                        source: object_store::Error::NotFound { path, source: Box::from(original) },
                    }),
                    None => JammiError::DataFusion(e),
                }
            }
        }
    }
}
```

`unwrap_jammi(e) -> Result<JammiError, DataFusionError>` destructures by value: `External(b)` →
`b.downcast::<JammiError>()` (`Ok(j) => Ok(*j)`, `Err(b) => Err(External(b))`); `Context(msg,
inner)` → recurse, rebuilding `Context(msg, Box::new(back))` on a miss; `ArrowError(b, bt)` → if
`*b` is `ExternalError(inner)` downcast `inner`, rebuilding on a miss; `ParquetError(b)` → if `*b`
is `External(inner)` likewise; every other variant → `Err(e)` unchanged. `not_found_path(&e)`
walks `source()` from `e` and returns the first `object_store::Error::NotFound` it can
`downcast_ref` (its `path` and the original's `Display`). The unit tests beside it pin each shape:
`classifier_restores_a_nested_external_jammi_error`,
`classifier_types_a_nested_object_store_not_found`,
`classifier_keeps_other_errors_as_datafusion_with_source` (the `#[source]` attribute is live),
and the `classifier_types_*_resources_exhausted*` group.

## 3. Invariants and how each is preserved

- **Generic primitive, tenant isolation.** "Re-embed only what changed" names no consumer. The
  strict tenant arm is on every version CAS and on the step-0 read gate (D5); versions and
  segments inherit the table row's tenant; the placeholder is registered under the row's owner.
- **Embeddings are consumed through `search`.** The mask lives inside the read path; there is no
  new consumption verb and no raw-vector read of a version.
- **Topology is configuration; remote equals embedded.** One code path for SQLite and Postgres,
  for `file://` and an object store; the library exposes every verb the server does, and both
  transports return the same report and the same version identity (D15).
- **Replayable producers.** Two descriptor variants, two replay arms, reachable through
  `producing_descriptor` (D9).
- **Typed refusal at the input edge.** Null keys are refused typed on all three paths (D21); hash
  decode is validated; `.version.json` rejects a newer format; `DefinitionDrift`, `NonUniqueKey`,
  `NotRefreshable` and `VersionUnavailable` are raised at the edge; the base publish refuses a row
  without `dimensions`.
- **Append-only migrations.** One appended migration, `032_result_table_versions` (D16).
- **Lockstep versions across crates; additive wire.** Three RPCs and the `JammiErrorDetail` arms of
  D22 are additive: `reserved 15` is untouched and no field is renumbered or renamed.
- **Identity folds every input.** Parent + definition + delta descriptor + every live artifact
  digest; `_content_hash` folds input state; the delete policy is in the descriptor (§2.6).
- **I-A2 (engine-internal).** A `result_tables` row with `current_version` Some never re-enters
  `building`; refresh and compaction only `UPDATE current_version / next_version / row_count /
  input_anchors_json WHERE status = 'ready'`; the table-level purge arms are reachable only from
  building/failed table rows.
- **I-Mono (engine-internal).** `next_version` only increases; `publish_version` requires `parent <
  N`; a version's semantic content is immutable once ready; the `SegmentSetCache` keyed `(table,
  version)` is therefore sound across processes.

## 4. Properties the tests hold

SQL-error properties are asserted on the embedded engine only, because on the wire `DataFusion`
and `Storage` both fold to gRPC `Internal`. Tests are in `crates/jammi-ai/tests/it/refresh.rs`
unless another file is named.

- **Edit one row** → exactly one row inferred, one new segment stamped N, the mask holds `(key,
  N−1)`, search and `SELECT` see only the new vector (`edit_one_row_refresh_infers_exactly_one`).
  Editing a row to empty text is a mask-only publish with `dropped_rows = 1`
  (`edit_to_empty_text_publishes_a_mask_only_version`).
- **Delete one key** → no fragment, no segment, the key never returns from search or SQL while a
  direct probe of the base bundle still holds it (`delete_one_key`).
- **Recompute and compaction.** `recompute` of a versioned table is a new table with a new chain
  root (`recompute_of_a_versioned_table_is_a_new_table`); compaction yields one fragment and one
  segment, value-equivalent, identity distinct (`compact_yields_single_fragment_value_equivalent`);
  expiry refuses when the current version is not ready
  (`expire_versions_refuses_a_non_ready_current_version`).
- **Remote equals embedded at the version identity**
  (`crates/jammi-server/tests/it/grpc_remote_session.rs::remote_refresh_matches_local_identity`).
- **No change for a never-refreshed table** — no `MaskExec` / `UnionExec` in its plan
  (`crates/jammi-db/tests/it/masked_read.rs::never_refreshed_table_has_no_mask_in_its_plan`), and
  `n1_f32_search_is_byte_identical_to_the_lone_sidecar` in `index/segment.rs`.
- **Crash and manifest loss.** An expired version lease is reaped, the table stays `ready` and
  searchable (`expired_version_lease_is_reaped_and_the_table_stays_ready`); a vanished current
  manifest is the typed `VersionUnavailable` on search and SQL, and `recompute` yields a new table
  (`current_manifest_loss_is_typed_unavailable_and_recomputable`,
  `masked_read.rs::unresolvable_current_version_is_typed_unavailable`).
- **Concurrency.** Two refreshes on one parent publish exactly once
  (`concurrent_refreshes_on_one_parent_publish_exactly_once`); two base publishers do not fail the
  refresh (`concurrent_base_publish_does_not_fail_the_refresh`); readers see the parent until the
  publish (`concurrent_reads_see_the_parent_until_publish`); a stale session binding corrupts
  neither a refresh nor a compaction
  (`stale_process_binding_does_not_corrupt_a_concurrent_refresh`,
  `stale_process_binding_does_not_lose_rows_on_compaction`).
- **Allocation is monotonic and never reused**, on both catalog backends
  (`crates/jammi-db/tests/it/segment.rs::allocation_is_monotonic_and_never_reused`); masking is
  independent of segment id order (`segment.rs::masked_merge_is_independent_of_segment_id_order`);
  a masked row never appears and rescore reads the owning segment
  (`masked_read.rs::masked_row_never_appears_in_search`, `::rescore_reads_the_owning_segment`).
- **Drift** is refused before any allocation (`model_drift_is_refused_before_any_allocation`).
- **Restart parity** (`restart_serves_the_refreshed_version`).
- **Multi-partition completeness and thread-count invariance**
  (`multi_partition_refresh_is_complete`;
  `crates/jammi-ai/tests/it/content_hash.rs::multi_partition_embed_is_complete_and_thread_count_invariant`;
  `ordered_input.rs::tests::one_partition_below_the_sort_and_no_limit_pushdown`).
- **Replay reachability** (`producing_descriptor_is_the_delta_after_a_refresh`).
- **Uniqueness** on both scans (`non_unique_keys_are_refused_on_both_scans`).
- **Dependents stay fresh across a `NoChange` refresh**
  (`dependents_stay_fresh_across_a_no_change_refresh`).
- **Null keys** refuse before any model call, on embed, `infer` and refresh
  (`content_hash.rs::null_keys_are_refused_typed_before_any_model_call`,
  `null_keys_refuse_the_refresh_before_any_model_call`).
- **Mid-scan object vanish** is a typed not-found naming the fragment, never a partial count
  (`masked_read.rs::mid_scan_object_vanish_is_a_typed_not_found`).
- **Masked projection and limit** (`masked_read.rs::masked_projection_and_limit`).
- **Wire round-trip** of every variant (`every_owned_shape_variant_round_trips_to_itself`).
- **Tenant scope** gates the refresh before the model load
  (`tenant_scope_gates_the_refresh_before_the_model_load`; the `RefreshEmbeddings` case in
  `crates/jammi-server/tests/it/tenant_isolation_oracle.rs`, whose `every_rpc_is_covered` derives
  the RPC inventory from the service descriptor).
- **Versioned reads and verification** (`read_vectors_follows_the_refreshed_version_in_key_order`,
  `verify_follows_the_refreshed_version`).

## 5. Out of scope

Table-level changed-since surfaces and a `staleness` "delta available" answer; time-travel reads
of expired versions; background compaction, GC or scheduling; sharded refresh (§2.9); peer-placed
search over a refreshed table (§2.4); re-binding a session's registration when a sibling store
publishes (D8); refresh of propagation, context-set, imported or inference tables; hashing file
bytes behind path columns; spilling the current-state map; a faster hash; reclaiming
`{table}.parquet` after a compaction; a gRPC-level distinction between `DataFusion` and `Storage`
(both `Internal`); status-detail decoding in the Python remote client, which therefore ships
base-class parity (`InvalidArgument` / `BackendError` by gRPC code) while the embedded engine
raises the leaf classes — both catchable by the same base `except`.

## 6. References

- Apache Iceberg table spec (`format/spec.md`): snapshots and `current-snapshot-id`; atomic
  metadata swap; optimistic-commit retry; equality deletes apply only to data files with a lower
  sequence number; data files are immutable; rewrites create a new snapshot and applied deletes
  may be dropped → D1, D2, D5, §2.8.
- Lance table format (`lance.org/format/table`, `protos/table.proto`): a manifest per version
  listing fragments; per-version deletion files; deletions materialised by rewriting fragments and
  rebuilding indices → fragments never rewritten in place, one deletion artifact per version,
  compaction explicit. Lance's positional deletion encoding is NOT adopted: row identity here is
  the key.
- Qdrant storage/optimizer docs: non-appendable segments are read+delete only; vacuum rebuilds
  when the deleted fraction exceeds a threshold; atomic swap → masks over immutable segments; the
  threshold is policy, the rebuild is mechanism.
- `datafusion` 54.1.0, `parquet` / `arrow` 58.4.0, `object_store` 0.13.2, `thiserror` 2.0.18
  sources → D10, D15, D20, D21, §2.5, §2.11, §2.12.
