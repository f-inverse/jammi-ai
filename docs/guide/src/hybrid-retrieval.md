# Hybrid Retrieval: Lexical (BM25) + Reciprocal-Rank Fusion

> **Measured companion:** for the long-form, executed-and-measured Python treatment, see [The Cookbook → Retrieval](https://f-inverse.github.io/jammi-ai/cookbook/chapters/10-retrieval/retrieval.html).

Dense vector search finds rows that *mean* the same thing as your query; lexical
(BM25) search finds rows that contain the same *words*. Each misses what the
other catches — dense search fumbles rare identifiers and exact phrases, lexical
search misses paraphrase. Hybrid retrieval runs both and fuses their rankings.

Jammi ships the three pieces:

- **`build_lexical_index`** — materialises a source's text as a lexical table,
  one `(_row_id, text)` row per source row.
- **`lexical_search`** — ranks a lexical table's rows against a text query by
  BM25 and hydrates the source's rows, exactly as `search` does for vectors.
- **Reciprocal-rank fusion** (`rrf_fuse`) — merges any number of ranked lists by
  *rank*, not score.

Fusing by rank is the whole point: BM25 scores and cosine similarities live on
incompatible scales, so averaging them is meaningless. RRF never looks at a raw
score — it sums `1 / (k_rrf + rank)` across the lists a row appears in, so the
fused order depends only on *where* a row landed in each list. The default
`k_rrf` is 60 (Cormack et al., SIGIR 2009; robust across 40–80).

## Build a lexical index

A lexical index is built over a registered source: its `key` column keys each
row, and its text `columns` are joined in order by a space into the row's text.
The analyzer decides how that text — and every query — is tokenised:
`"english"` (lowercase + Porter stemming) is the default, and `"raw"`
(lowercase, no stemming) is for text a stemmer would mangle (codes, identifiers,
other languages).

### Python

```python
db.add_source("patents", url="patents.parquet", format="parquet")
lexical = db.build_lexical_index("patents", columns=["title", "abstract"], key="id")
```

### Rust

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# async fn ex(session: &jammi_ai::Session) -> jammi_db::error::Result<()> {
use jammi_ai::local_session::{BuildLexicalIndex, LexicalAnalyzer};

let lexical = session
    .build_lexical_index(
        "patents",
        &BuildLexicalIndex {
            columns: vec!["title".into(), "abstract".into()],
            key_column: "id".into(),
            analyzer: LexicalAnalyzer::English,
        },
    )
    .await?;
# let _ = lexical;
# Ok(()) }
```

The lexical table is the index's data of record: it pins the text as it was
read, and it carries the same materialization manifest as every other derived
table, so `describe_table`, `staleness` and `recompute` apply to it unchanged.
Its inverted index is derived state — rebuilt in memory from the table's rows
the first time a process searches a version of it, and never stored.

## Search it

`lexical_search` takes a text query; its words are the query. Each analysed
term is one disjunctive clause, so a row matching more of the terms, or rarer
ones, ranks higher, and no query syntax is interpreted — a colon, a quote or a
minus sign is text. It returns the `k` best-ranked rows hydrated from the
source, with the same `filter` / `select` refinements `search` has: with a
`filter`, the `k` best-ranked rows that satisfy it.

### Python

```python
hits = db.lexical_search("patents", text="quantum error correction", k=10,
                         filter="year >= 2021")
print(hits.select(["id", "title", "bm25_score", "bm25_rank"]).to_pandas())
```

### Rust

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# async fn ex(session: &jammi_ai::Session) -> jammi_db::error::Result<()> {
use jammi_ai::local_session::LexicalSearchRequest;

let hits = session
    .lexical_search(LexicalSearchRequest {
        source_id: "patents".into(),
        text: "quantum error correction".into(),
        k: 10,
        lexical_table: None,
        filter: Some("year >= 2021".into()),
        select: Vec::new(),
    })
    .await?;
# let _ = hits;
# Ok(()) }
```

Each row carries the built-in `bm25` evidence channel's two columns —
`bm25_score` (`Float32`, the raw BM25 score) and `bm25_rank` (`Int64`, 0-based)
— and `retrieved_by == ["bm25"]`, the lexical peer of a dense search's `vector`
channel and `similarity`. `lexical_table` names which of a source's lexical
tables to search; unset, the newest.

## Fuse dense and lexical rankings

`rrf_fuse` takes ranked lists — each a best-first list of `_row_id`s — and
returns one fused ranking. The dense list is a `search` result's keys; the
lexical list is a `lexical_search` result's. A third list (e.g. a
graph-propagated search) fuses identically, with no special-casing.

### Python

```python
dense = db.search("patents", query=db.encode_query(model=model, query=q), k=50)
lexical = db.lexical_search("patents", text=q, k=50)
fused = db.rrf_fuse([dense.column("_row_id").to_pylist(),
                     lexical.column("_row_id").to_pylist()])
```

### Rust

```rust,no_run
# extern crate jammi_ai;
use jammi_ai::query::{rrf_fuse, DEFAULT_K_RRF};

// Best-first row-id lists from each retriever.
let dense = vec!["doc-2", "doc-1", "doc-5"];   // cosine order
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

## Scope

Lexical search applies no row-level filter of its own — isolation is
table-level, exactly as the ANN `search` path: the lexical table resolves
through the tenant-scoped catalog, so a tenant searches only the lexical tables
it built.
