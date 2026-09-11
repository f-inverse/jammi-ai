# Refresh an Embedding Table Incrementally

An embedding table produced by `generate_embeddings` is a function of its
source: `D(S)`. When the source changes, `refresh_embeddings` re-embeds only
the rows whose content changed and publishes the result as a new **version**
of the same table — the table keeps its name, its ANN index and every
downstream anchor, and a reader sees exactly one version at a time. "Publish"
here is the storage-commit sense — the catalog's `publish_version` CAS that
makes a `building` version the table's current one — distinct from the
domain-lifecycle "publish/install/bind" a consumer builds on top of Jammi's
primitives (see [Philosophy](./philosophy.md#where-the-line-falls)).

```python
db.generate_embeddings("docs", model="sentence-transformers/all-MiniLM-L6-v2",
                       columns=["title", "body"], key_column="id")
# ... rows are edited, added, or deleted in the source ...
report = db.refresh_embeddings("docs__embedding__all-MiniLM-L6-v2__...")
# {'table': ..., 'version': 1, 'parent_version': 0, 'inferred_rows': 3,
#  'added': 1, 'changed': 2, 'deleted': 1, 'unchanged': 9996, 'dropped_rows': 0,
#  'live_rows': 9999, 'masked_rows': 3, 'outcome': 'published'}
```

```rust
use jammi_ai::pipeline::embedding_refresh::RefreshOptions;

let report = session.refresh_embeddings(&table, RefreshOptions::default()).await?;
```

Every transport exposes the same three verbs with the same report:
`Session::{refresh_embeddings, compact_embeddings, expire_versions}` in Rust,
`Database.refresh_embeddings / compact_embeddings / expire_versions` in Python
(embedded and remote alike), and `EmbeddingService.RefreshEmbeddings /
CompactEmbeddings / ExpireVersions` on gRPC. A refresh run through a local
session and the same refresh run through the data-plane client produce the
same version identity, fragment digests and counts (K4).

## What a refresh does

1. **Gates.** The table must be `ready`, produced by an embedding pipeline,
   and carry `_content_hash` (a table imported or produced before the column
   existed is `NotRefreshable { reason: missing_content_hash }` — run
   `recompute` once to get a refreshable table). The model is loaded and the
   table's recorded definition is compared with the definition the current
   environment would produce; a different model, backend, precision or
   quantization is `DefinitionDrift { table, recorded, current }` — nothing
   is allocated, and the consumer's remedy is `recompute`.
2. **Base version.** A never-refreshed table is first published as its base
   version, whose identity is the table's existing `.materialization.json`
   artifact digest, so publishing the base changes no downstream anchor.
3. **Diff.** The current version's `(_row_id, _content_hash)` pairs are read
   through the masked view and the source is scanned once, projecting the
   same content hash. Each source key is classified `added`, `changed` or
   `unchanged`; a current key the source no longer has is `deleted` (see the
   delete policy below).
4. **Nothing to do** → `outcome: no_change`, no version allocated, nothing
   written, every downstream anchor untouched (a dependent stays `Fresh`).
5. **Infer the delta.** A new version number is allocated (monotonic, never
   reused) and only the added and changed rows are run through the model —
   the plan is the source scan joined to the delta keys, sorted in the same
   key order the base uses, so the fragment's bytes are deterministic. The
   rows land in one Parquet **fragment** and one ANN **segment**, both stamped
   with the new version.
6. **Deletion mask.** Every superseded or deleted key is added to the
   version's cumulative mask with the horizon `N − 1`: the key is dead in
   every fragment and segment stamped at or below that horizon and live in
   the new fragment.
7. **Publish.** The version's manifest is written and one catalog transaction
   flips the version `building → ready` and moves the table's
   `current_version` forward. Two refreshes of one parent both allocate a
   number; the second to publish fails with `CasFailed`, its version row is
   marked `failed` and its artifacts are reaped. The previous version stays
   live until the swap, so a refused or failed refresh never changes what a
   reader sees.

## `RefreshReport`

| Field | Meaning |
|-------|---------|
| `version` | The published version (`published`), or the current one (`no_change`) |
| `parent_version` | The version the diff was computed against |
| `inferred_rows` | Rows the model actually ran on (`added + changed`) |
| `added` / `changed` / `deleted` / `unchanged` | The classification of the source's keys |
| `dropped_rows` | Rows asked for that the model did not realize (a per-row input failure — an empty text on the candle backend) |
| `live_rows` | Rows a reader sees after the publish |
| `masked_rows` | Physical rows in the version's fragments hidden by the mask |
| `outcome` | `published` or `no_change` |

`live_rows` and `masked_rows` are the inputs to a compaction decision; the
engine ships no threshold.

## Delete policy

`deletes` is `tombstone` (default) or `retain`:

- **`tombstone`** — a key the source no longer has is masked out of every
  prior fragment and segment; the table stays `D(S)`.
- **`retain`** — the key's current row is kept (a rolling-window source that
  drops old rows the table should still serve).

The policy is recorded in the version's descriptor, so two versions produced
under different policies have different identities.

## Uniqueness

The initial `generate_embeddings` tolerates duplicate source keys (every row is
written; the ANN index keeps the last one). A refresh does not: a duplicated
key on the source scan is `NonUniqueKey { table, scan: source, keys, total }`
after the complete scan (up to ten keys with their exact counts, plus the
total), and a parent version that already holds two physical rows under one
`_row_id` is `NonUniqueKey { scan: parent }`. Neither allocates a new version;
the remedy is to de-duplicate the source and, for a non-unique parent, to
`recompute` once.

## Null keys

A `NULL` in the key column is never a per-row event on any path —
`generate_embeddings`, a refresh, and `infer` all refuse the whole call with
`InvalidKey { column, null_count }` before the model runs. The count is exact,
zero rows are embedded, and nothing is written (a base version published by a
first refresh is retained; it is downstream-invisible).

## Empty texts

Per-row validity is the backend's: the candle backend marks an empty or null
text a per-row error (`dropped_rows`); the HTTP backend forwards texts
verbatim, so validity is the remote's. A delta whose every row is dropped is
still a legitimate publish — a mask-only version with no fragment and no
segment, so an edit-to-empty removes the row from the table.

## Reading a versioned table

A reader — `search`, `search_by_id`, every SQL `SELECT`, `read_vectors`,
`verify_materialization`, `staleness` — resolves the table's
`current_version` and sees only that version's live rows:

- **ANN.** The version's segments are merged under its mask; a masked
  candidate is dropped and the search widens until `k` live hits are found,
  and the exact rescore reads the segment that owns the hit.
- **SQL.** The version's fragments are unioned under the mask with `_row_id`
  always projected, so `COUNT(*)`, `SELECT vector` and `LIMIT n` all see live
  rows only.
- **Snapshots.** A publish between a reader's table-row read and its scan
  serves the older version consistently; a read in flight when its version
  expires completes from memory. A table whose current version's manifest
  cannot be resolved (the object is gone) is bound as a placeholder: planning
  succeeds, every scan is the typed `VersionUnavailable { table, version }`,
  and `recompute` is the remedy.
- **A never-refreshed table** is byte-identical to today's: no mask, no
  union, `read_vectors` reads the raw file.

Restarting the engine re-binds the current version; search results and
`SELECT` output are identical before and after.

## Compaction and expiry

`compact_embeddings(table)` rewrites the current version's live rows as one
fragment and one segment — no inference, vectors carried byte-for-byte — and
publishes it as a new version with the same ranking and an identity that
folds the parent's. `expire_versions(table, before)` deletes every
non-current version numbered below `before` and reaps its manifest, mask and
every fragment and segment the current version does not still reference;
`{table}.parquet`, its `.materialization.json` and the version allocator are
never touched.

```python
db.compact_embeddings(table)
db.expire_versions(table, before=db.refresh_embeddings(table)["version"])
```

## Identity and verification

The base version's identity is the table's artifact digest. Every later
version's identity is a hash chain over its parent's identity, the definition
hash, the delta descriptor, the fragment digests and the mask digest —
counts and the ANN index are outputs, never inputs. `verify_materialization`
on a versioned table recomputes every fragment and mask digest and the chain
and names the artifact that diverged; `staleness` on a dependent uses the
current version's identity, so a refresh that changed content advances the
dependent to `Stale { InputAdvanced }` and a `no_change` refresh leaves it
`Fresh`. `recompute` of a versioned table produces a **new** table with a
fresh chain root.

## Errors

| Error | Meaning | Remedy |
|-------|---------|--------|
| `InvalidKey { column, null_count }` | Null keys on the source | Fix the source |
| `NonUniqueKey { scan, keys, total }` | Duplicate keys on the source or the parent | De-duplicate; `recompute` a non-unique parent |
| `DefinitionDrift { recorded, current }` | The environment would produce a different definition | `recompute` |
| `NotRefreshable { reason }` | No content hash, not ready, not an embedding table, or the current version is unavailable | `recompute` |
| `VersionUnavailable { table, version }` | The current version's manifest cannot be resolved | `recompute` |
| `CasFailed` | Another refresh published first | Retry; the other refresh's result is current |

A storage object that vanishes under a running scan is the typed
`Storage(StorageError::Io { source: object_store::Error::NotFound })`, never a
partial answer; every other DataFusion error keeps its `source()` chain under
`JammiError::DataFusion`.

## Scope

A refresh diffs the whole source by content hash (Tier 1), which covers file,
federated and mutable sources alike. A table-level monotonic source version is
a freshness surface the manifest already anticipates and may be built later
(Tier 2); row-level change tracking on mutable tables is a transition log and
is never built. Refresh applies to tables produced by an embedding pipeline —
not to imported, propagated, context-set or inference tables.
