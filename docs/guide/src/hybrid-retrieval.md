# Hybrid Retrieval: Lexical (BM25) + Reciprocal-Rank Fusion

Dense vector search finds rows that *mean* the same thing as your query; lexical
(BM25) search finds rows that contain the same *words*. Each misses what the
other catches — dense search fumbles rare identifiers and exact phrases, lexical
search misses paraphrase. Hybrid retrieval runs both and fuses their rankings,
and it is the standard production recipe because it reliably beats either alone.

Jammi ships two pieces for this:

- A **lexical sidecar** (`LexicalIndex`) — a tantivy BM25 inverted index that
  rides beside a result table's Parquet object, the lexical peer of the USearch
  ANN sidecar.
- **Reciprocal-rank fusion** (`rrf_fuse`) — merges any number of ranked lists by
  *rank*, not score.

Fusing by rank is the whole point: BM25 scores and cosine similarities live on
incompatible scales, so averaging them is meaningless. RRF never looks at a raw
score — it sums `1 / (k_rrf + rank)` across the lists a row appears in, so the
fused order depends only on *where* a row landed in each list. The default
`k_rrf` is 60 (Cormack et al., SIGIR 2009; robust across 40–80).

## Build a lexical index

A `LexicalIndex` is built over `(row_id, text)` pairs — the `text` is whatever
text columns of the row you want searchable, joined by the caller. The analyzer
is configurable; `English` (lowercase + Porter stemming) is the default, and
`Raw` (lowercase, no stemming) is the escape hatch for text the English stemmer
would mangle (codes, identifiers, non-English).

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# fn ex() -> jammi_db::error::Result<()> {
use jammi_ai::index::{Analyzer, LexicalIndex};

let rows = vec![
    ("doc-1", "a method for reducing turbine blade vibration"),
    ("doc-2", "an apparatus for cooling turbine engine blades"),
    ("doc-3", "a recipe for baking sourdough bread"),
];

let lexical = LexicalIndex::build(rows, Analyzer::English)?;
let hits = lexical.search("turbine engine", 10)?;
for hit in &hits {
    println!("{} bm25={:.3} rank={}", hit.row_id, hit.bm25_score, hit.rank);
}
# Ok(()) }
```

Each `LexicalHit` carries the `row_id`, its raw `bm25_score`, and its 0-based
`rank` — the rank is what fusion consumes.

### Lifecycle and scope

The lexical sidecar's lifecycle equals the ANN sidecar's: it is built (and
rebuilt) with the table. An immutable result table that is rebuilt produces a
fresh sidecar; for a mutable-table source, re-ingesting the changed rows into a
new index is the caller's mode. Search applies no row-level filter — isolation
is table-level, exactly as the ANN `search` path: resolve the table through the
tenant-scoped catalog and hand the index only that table's rows.

## Fuse dense and lexical rankings

`rrf_fuse` takes a slice of ranked lists — each a best-first list of `_row_id`s —
and returns one fused ranking. The dense list is the ANN `search` result; the
lexical list is the `LexicalIndex` result. A third list (e.g. a graph-retrieval
channel) fuses identically, with no special-casing.

```rust,no_run
# extern crate jammi_ai;
use jammi_ai::query::{rrf_fuse, DEFAULT_K_RRF};

// Best-first row-id lists from each retriever.
let dense = vec!["doc-2", "doc-1", "doc-5"];   // ANN cosine order
let lexical = vec!["doc-1", "doc-2", "doc-9"]; // BM25 order

let fused = rrf_fuse(&[dense, lexical], DEFAULT_K_RRF);
for hit in &fused {
    println!("{} rrf={:.4}", hit.row_id, hit.rrf_score);
}
```

Rows that both retrievers surface rise to the top — cross-list agreement is
exactly what RRF rewards. The output is fully deterministic: it is sorted by
fused score descending, ties broken ascending by `row_id`, and it does **not**
depend on the order you pass the lists in. A row repeated within a single list
counts only once, at its best rank.

`k_rrf` is exposed, not forced. Larger values flatten the gap between adjacent
ranks (a deep-but-agreed-upon row matters more); smaller values sharpen the
reward for top-of-list placement. `DEFAULT_K_RRF` (60) is the recommended start.

## Record the evidence

BM25 contributions ride the built-in `bm25` evidence channel, the lexical peer
of `vector`'s `similarity`. It declares two columns — `bm25_score` (`Float32`)
and `bm25_rank` (`Int64`) — and a contribution is supplied to `merge_channels`
exactly as the `vector` channel's is, so a fused result carries both its dense
and its lexical provenance side by side. See
[Declare a Custom Provenance Channel](./declare-provenance-channel.md) for the
contribution mechanics; `bm25` needs no registration — it is seeded with the
catalog.
