# W3 — `Search` on the gRPC wire (edge-reachable similarity search)

**Status:** spec — pending review
**Owner:** TBD
**Estimated effort:** 3–4 days
**Workstream dependencies:** none (`search` exists; this exposes it). Pairs with W2 (R2) for the full edge audio path.
**Workstreams blocked by this:** W8 (edge consumers call `search` over gRPC-web)

> Naming note: the `W3` number is a placeholder.

## Motivation

`search` is the **sole embedding-consumption verb** (see [Design Philosophy](../guide/src/philosophy.md)).
It is reachable from the Rust library, Python, and Flight SQL — but **not** from the typed gRPC
surface. Flight SQL is HTTP/2 gRPC with bidirectional streaming, which edge-function runtimes
cannot speak (no HTTP/2 client; gRPC-web is the only available gRPC family). So an edge deployment
that runs the engine as a container sidecar and calls it over gRPC-web can **ingest** and
**encode** (`EmbeddingService` already serves those over tonic-web) but **cannot search** — the one
thing that actually consumes embeddings.

W3 closes that gap: add a `Search` RPC to `EmbeddingService`, served over gRPC-web like the
existing embedding verbs. It exposes an existing engine capability on the transport edge consumers
can reach; it adds no new consumption model.

## Current state (verified at spec time)

- `crates/jammi-ai/src/session.rs:327` — `pub async fn search(self: &Arc<Self>, source_id: &str, query: Vec<f32>, k: usize) -> Result<SearchBuilder>`. **`query` is a precomputed vector.**
- `crates/jammi-ai/src/search/builder.rs` — `SearchBuilder` exposes `filter(predicate: &str)`, `sort(column, descending)`, `limit(n)`, `select(columns)`, `join`, `annotate`, and `run() -> Vec<RecordBatch>` (rows carry the key/content columns + score + `retrieved_by`/`annotated_by`; the raw vector is stripped).
- Query vectors come from `encode_text_query(model_id, text)` (session.rs) and, for audio,
  `EncodeAudioQuery` — **already on the gRPC wire**.
- gRPC services (`crates/jammi-server/proto/jammi/v1/`): `EmbeddingService` (AddSource,
  GenerateAudioEmbeddings, EncodeAudioQuery), `SessionService`, `TriggerService`. **No `Search` RPC
  anywhere.** tonic-web is already enabled (the embedding verbs work over gRPC-web).
- **No query-by-id / search-by-example** in the engine — `search` only takes a vector.

## Change

### 1. `Search` RPC on `EmbeddingService` (`proto/jammi/v1/embedding.proto`)

```proto
rpc Search(SearchRequest) returns (SearchResponse);

message SearchRequest {
  string source_id = 1;
  oneof query {
    QueryVector query_vector = 2;   // search by a precomputed vector
    string      row_key      = 3;   // search by an existing row (query-by-example) — see D1
  }
  uint32 k = 4;
  optional string filter = 5;            // SQL predicate pushed into the search — see D3
  repeated string select = 6;            // projected columns (default: key only)
}
message QueryVector { repeated float values = 1; }   // packed f32

message SearchResponse { repeated SearchHit hits = 1; }
message SearchHit {
  string key   = 1;   // the source's key column
  float  score = 2;   // similarity / distance from AnnSearchExec
  // projected scalar columns when `select` is non-empty — see D2
  map<string, string> columns = 3;
}
```

### 2. Handler (`crates/jammi-server/src/grpc/embedding.rs`)

Decode the query → `session.search(source_id, vector, k)` (or `search_by_id`, D1) → apply `filter`
via `.filter(..)` and `select` via `.select(..)` on the builder → `.run()` → map each `RecordBatch`
row to a `SearchHit` (extract the key column + the score column; project `select` columns into
`columns`). Tenant scope via the existing `jammi-session-id` metadata path the other verbs use.
Errors map through the existing `JammiError → Status` conversion. gRPC-web framing is inherited
(tonic-web is already mounted).

### 3. (D1) Engine `search_by_id` for query-by-example

`search` takes a vector. Ranking *by an existing item* ("stems like this stem") is the common edge
case, and the item's vector is already in the result-table. Add:

```rust
// session.rs
pub async fn search_by_id(self: &Arc<Self>, source_id: &str, row_key: &str, k: usize)
    -> Result<SearchBuilder>
```

which resolves the row's stored vector **internally** (a scan of the result-table by key) and
delegates to `search`. The vector never crosses the boundary — this is consistent with the "no
raw-vector reads" line (query-by-example is generic; every vector DB has it). See **Decisions**.

## Decisions to ratify

- **D1 — query-by-id.** Add `search_by_id` (resolve the row's vector internally) **or** require
  clients to re-encode the query clip via `EncodeAudioQuery` then `Search(vector)`.
  *Recommend: add it.* It is what ranking-by-existing-item consumers want, avoids a re-encode
  round-trip, keeps the vector internal (philosophy-consistent), and is generic (query-by-example).
  The two coexist via the `oneof query`.
- **D2 — result encoding.** Structured `SearchHit` rows (key + score + projected scalar columns)
  **or** Arrow IPC bytes. *Recommend structured rows:* keeps the lightweight gRPC-web client simple
  (no Arrow reader in the edge bundle). Arrow IPC remains Flight SQL's job for heavy clients.
- **D3 — filter pushdown.** Expose the SQL `filter` predicate on the wire. *Recommend yes:* it is
  how a consumer combines its own coarse predicates with similarity ("push predicates into the
  search query") without the engine knowing any domain semantics.

## Tests

Hermetic by default: encode a handful of synthetic rows, `Search` by vector and by `row_key`,
assert ordering, `k`, `filter`, and `select` projection; a gRPC-web framing round-trip mirroring the
existing `EncodeAudioQuery` wire test. Live (opt-in) against a running server.

## Docs

- Extend `docs/guide/src/semantic-search.md` with the gRPC / edge path (encode-then-search, and
  query-by-id).
- Note `Search` over gRPC-web in `docs/guide/src/deploy-server.md`.

## Discipline check

- `Search` is the generic consumption verb; this exposes an existing engine capability on an
  additional transport (gRPC-web) that edge consumers reach. Generic — any edge deployment wants it.
- `search_by_id` is query-by-example, a standard vector-search primitive; it keeps the vector
  internal, so it does not reintroduce a raw-vector read.
- No consumer/tenant name appears. Additive: the new RPC does not change existing verbs.
