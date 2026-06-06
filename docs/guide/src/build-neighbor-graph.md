# Building a similarity graph

`build_neighbor_graph` materializes the **k-nearest-neighbour graph** of an
existing embedding table: for every row it finds the `k` most similar rows
*within the same table* and writes one directed edge per pair as a queryable
edge relation.

## When to reach for it — and when not to

> **Use `search` for the neighbours of *specific* rows. Use
> `build_neighbor_graph` only when you need the *whole* edge set at once.**

If you want "rows like this row", call `search` (or `search_by_id`). It loads
the index once per query and hydrates results on demand — that is the
per-query path, and `build_neighbor_graph` is not for it.

`build_neighbor_graph` exists for **global-structure** work, where you consume
*all* edges as a durable artifact:

- near-duplicate detection / semantic dedup,
- clustering and connected components,
- entity resolution,
- generating training pairs for graph-aware fine-tuning.

For those, looping `search` over every row would reopen the index per row,
pay an `n`× hydration round-trip, and leave you with `n` detached result sets
instead of one catalogued, tenant-scoped, queryable table. The edge table this
verb writes closes that gap — and only for the global case.

## The edge relation

The result is an ordinary `result_table` you can query, join, and federate
like any other. One row per directed edge:

| Column | Type | Meaning |
|---|---|---|
| `src` | Utf8 | source node — the source key |
| `dst` | Utf8 | neighbour node — the source key |
| `rank` | Int32 | `1` = nearest, … `k` |
| `similarity` | Float32 | `1.0 - cosine_distance` |

`src` and `dst` are the embedding table's keys, so the edge table joins
**directly** to your source data — no detour through the embedding table.

## Approximate by default; exact on demand

The default driver is **index-assisted** (HNSW). It is fast (`n · log n`) but
its output is:

- **approximate** — HNSW recall is below 100%, so some true neighbours are
  missed, and
- **non-deterministic** — two builds can differ in the long tail of weak edges.

For dedup and clustering this is exactly what you want. When you need
**reproducible, auditable** edges, pass `exact = true`: a brute-force,
deterministic, complete `n²` pass (gated by a row-count ceiling, so it refuses
to run on very large tables).

## Example: near-duplicate detection

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::config::JammiConfig;
use jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph;
# async fn ex(config: JammiConfig, model_id: &str) -> jammi_db::error::Result<()> {
# let session = Arc::new(InferenceSession::new(config).await?);

// Embed the corpus first (any embedding model).
session
    .generate_text_embeddings("patents", model_id, &["abstract".into()], "id")
    .await?;

// Materialize the kNN graph, keeping only strong, reciprocal edges.
let edges = session
    .build_neighbor_graph(
        "patents",
        None, // resolve the latest embedding table
        &BuildNeighborGraph {
            k: 5,
            min_similarity: Some(0.9), // near-duplicate threshold
            mutual: true,              // both rows agree they are neighbours
            ..Default::default()
        },
    )
    .await?;

// The edge table is now an ordinary relation. Group by src to list each
// row's near-duplicates, joined directly to the source on the key.
let dupes = session
    .sql(&format!(
        "SELECT e.src, e.dst, e.similarity, p.title \
         FROM \"jammi.{}\" e \
         JOIN patents.public.patents p ON p.id = e.src \
         ORDER BY e.similarity DESC",
        edges.table_name
    ))
    .await?;
# Ok(()) }
```

## Example: graph traversal stays in SQL

`build_neighbor_graph` transports adjacency and weight — it never walks the
graph. Two-hop expansion, paths, and reachability are plain SQL over the edge
relation, on every transport:

```sql
-- Two-hop neighbours via a self-join on the edge table.
SELECT a.src AS origin, b.dst AS two_hops_away
FROM "jammi.<edge_table>" a
JOIN "jammi.<edge_table>" b ON b.src = a.dst
WHERE a.src = '<some_key>';
```

For deeper walks, use a `WITH RECURSIVE` CTE. There is no traversal operator
and no graph DSL — the edge table is just a relation.

## Example: training-data prep for graph-aware fine-tuning

The edge table is the raw material for neighbour-contrastive embedding
fine-tuning. Turn edges into `(anchor, positive, negative)` triplets — a
neighbour is a positive, a non-neighbour a negative — and feed them to the
existing triplet/contrastive fine-tune path. The walk policy, negative
sampling, and objective are yours; Jammi supplies the edges and the loss.
