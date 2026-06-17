# Format Stability

Jammi persists several on-disk formats. Each one is **stamped** with a version
the writer records and the reader checks, so a file written by a newer build —
or by a backend whose serialized layout changed — is rejected as a *typed error*
rather than silently misparsed into wrong data. This page is the operator's
reference for every persisted format the engine owns, what its stability stamp
is, and how a reader reacts to a stamp it cannot honour.

The single principle: **a reader never guesses.** When a stamp is unreadable the
load fails loud with a typed error; the upgrade path is to **re-emit** the
artifact from its definition. There is no back-compat reader, no silent
downgrade, no default-to-version-1. (Re-emitting is cheap and exact: a result
table is the deterministic output of its producing definition over its pinned
input anchors — see [The Materialization Contract](./materialization-contract.md).)

## The per-format table

| Format | On-disk file | Stability stamp | Reject semantics on load |
|--------|--------------|-----------------|--------------------------|
| Materialization manifest | `.materialization.json` | `manifest_version` (`u32`) | **Reject-newer** — `found > MANIFEST_VERSION` → `ManifestError::UnsupportedManifestVersion` |
| ANN row map | `.rowmap` | leading `u32` version header | **Reject-newer** — `found > ROWMAP_VERSION` → `JammiError::IncompatibleFormat { artifact: "rowmap", .. }` |
| ANN sidecar manifest | `.manifest.json` | `version` (`u32`) | **Reject-newer** — `found > ANN_MANIFEST_VERSION` → `JammiError::IncompatibleFormat { artifact: "ann-manifest", .. }` |
| USearch ANN graph | `.usearch` | `backend_version` stamped in the sidecar `.manifest.json` | **Strict** — any mismatch with the linked USearch → `JammiError::IncompatibleFormat { artifact: "usearch-index", .. }` |
| Lexical (BM25) index | tantivy index dir | tantivy's own format tag | **Library-loud** — tantivy's `Index::open` fails with `IncompatibleIndex`, surfaced as `JammiError::Lexical` |
| Result-table data | `.parquet` | *none embedded* — its format-of-record version **is** the `.materialization.json` `manifest_version` | Schema-shape checked at read via `JammiError::Schema`; byte integrity caught by `verify_materialization` |

Two distinct kinds of stamp appear above, and the difference is deliberate:

- **Reject-newer** for formats that carry a *compatibility ordering*. An older
  or equal version is readable by construction (the layout only grew); only a
  newer version carries a layout this build does not know. This is the
  materialization manifest's idiom (`MaterializationManifest::from_json_bytes`),
  and the `.rowmap` and ANN `.manifest.json` follow it.
- **Strict** for the USearch `backend_version`, because the USearch serialized
  graph format carries **no** compatibility ordering between releases. A version
  that differs *at all* may mis-deserialise the graph and return wrong
  neighbours, so any inequality is incompatible — there is no "older is fine"
  here.

## Materialization manifest — reject-newer

`.materialization.json` carries `manifest_version` (`MANIFEST_VERSION`). The
reader rejects a newer version as the typed
`ManifestError::UnsupportedManifestVersion { found, supported }`. This is the
gold idiom every other stamped format is modeled on; the full contract is in
[The Materialization Contract](./materialization-contract.md). Its error lives
in its own domain (`ManifestError`) and is intentionally *not* folded into the
shared `IncompatibleFormat` variant — it carries the manifest-specific recovery
semantics the contract describes.

## ANN row map (`.rowmap`) — reject-newer

The `.rowmap` is the engine-owned mapping from a USearch internal id to the
Jammi `_row_id` string. It is a small binary file: a leading `u32` version
header, then length-prefixed UTF-8 entries. On load the reader checks the
header and rejects a version greater than `ROWMAP_VERSION` as
`JammiError::IncompatibleFormat { artifact: "rowmap", found, supported }`.

## ANN sidecar manifest (`.manifest.json`) — reject-newer + strict backend

The ANN sidecar's `.manifest.json` records the index metadata: `version`,
`dimensions`, `backend`, `backend_version`, `count`, the file names, and the
creation instant. On load it is deserialised as a **typed struct** (mirroring
`MaterializationManifest::from_json_bytes`), never by field-by-key
`serde_json::Value` lookups. The determinants of a safe load — `version`,
`dimensions`, and `backend_version` — are all **required**: a manifest missing
any of them is a hard decode error, never silently defaulted to a guess.

Two checks run on the deserialised manifest:

1. **`version`, reject-newer.** A `version` greater than `ANN_MANIFEST_VERSION`
   is `JammiError::IncompatibleFormat { artifact: "ann-manifest", .. }`.
2. **`backend_version`, strict.** The stamped USearch version is compared for
   exact equality against the linked `jammi_db::index::backend_version()`. Any
   mismatch is `JammiError::IncompatibleFormat { artifact: "usearch-index", .. }`.

## USearch ANN graph (`.usearch`) — strict backend version

The `.usearch` file is USearch's own serialized HNSW graph. Its serialized
header carries only the *major* version and gives no cross-release compatibility
guarantee, so a USearch upgrade can change the on-disk layout in a way that
deserialises into a structurally-valid-but-wrong graph — returning incorrect
nearest neighbours with no error. To close that silent-corruption path, the
engine stamps the full linked USearch version (`backend_version`) into the
sidecar `.manifest.json` at save and **strict-compares** it on load. The graph
itself is never trusted across a backend version change; the only safe action is
to re-emit the embedding table (which rebuilds the sidecar).

## Lexical (BM25) index — library-loud

The lexical retrieval sidecar is a tantivy index directory. Tantivy stamps its
own format version and refuses to open an index written by an incompatible
release: `Index::open` returns `IncompatibleIndex`, which the engine surfaces as
`JammiError::Lexical`. The engine adds no stamp of its own here — the library is
already loud, so a second stamp would be redundant machinery. The recovery is
the same: re-emit (re-index) the table.

## Result-table Parquet — no embedded stamp, by design

The result-table Parquet object carries **no** embedded format version. Its
format-of-record version is the `manifest_version` of the
`.materialization.json` sidecar written beside it: the manifest is the artifact's
identity, and the Parquet bytes are its *subject*. A reader does not need a
second, in-band version because:

- **Shape safety** is enforced at read time by the typed Arrow downcast. Every
  vector read goes through `store::vectors::extend_with_fixed_size_list_f32`,
  the single place in the engine that downcasts a vector column to
  `FixedSizeList<Float32>`; a missing column, a wrong Arrow type, or a
  non-`Float32` inner type is a typed `JammiError::Schema`, not a panic. Schema
  shape is checked from the data itself, so it needs no stamp.
- **Byte integrity** is content-addressed. The Parquet object is immutable and
  identified by its `ArtifactDigest` (SHA-256 over the bytes) recorded in the
  manifest, so any out-of-band byte change is caught by `verify_materialization`
  recomputing the digest — see
  [The Materialization Contract](./materialization-contract.md).

### The manifest-bypass Parquet read paths

Three engine read paths open a result-table Parquet object **directly**, without
first reading the `.materialization.json` manifest:

- `Session::read_vectors` — streams the whole `vector` column of an embedding
  table into one `Vec<f32>` per row.
- `Session::read_vector_by_key` — extracts a single row's `vector` by its
  `_row_id` (the resolver behind `search_by_id`'s query-by-example path).
- `store::register_parquet_table` — registers a Parquet URL as a DataFusion
  table under `jammi.{name}` for SQL scans.

These paths do not consult a format stamp, and that is correct: their safety
rests entirely on the typed `JammiError::Schema` downcast in
`store::vectors::extend_with_fixed_size_list_f32`, which validates the on-disk
Arrow schema shape directly from the data. A Parquet object whose schema does
not match what the read expects produces a typed `Schema` error, regardless of
how it was written. Out-of-band byte tampering on these immutable,
content-addressed objects is the `verify_materialization` digest check's
concern, not a per-read stamp's.

## Upgrade path: re-emit

For every stamped format above, the recovery from an incompatible stamp is the
same — **re-emit the artifact from its definition.** The engine ships no
back-compat reader and no in-place migrator: an ANN sidecar is rebuilt by
re-running the embedding producer, a tantivy index by re-indexing, a result
table by re-running its producing definition over its input anchors. Because a
result table is the deterministic output of a producing definition over pinned
inputs, re-emission is exact, not lossy. The typed rejection is the signal to
re-emit; it is never something to paper over with a default.
