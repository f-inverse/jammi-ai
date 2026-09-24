# Graphs and lineage

Build a k-nearest-neighbour graph over a corpus's embeddings, smooth the
embeddings over it, derive structure embeddings from the graph alone — then
ask the engine what produced each table and whether it still holds.

**When to use this pattern.** You derive tables from tables (a graph from
embeddings, smoothed vectors from a graph) and need to know, later, exactly
what produced a table, whether its bytes are still the ones recorded, whether
it is stale against the definition you run now, and what depends on it.

## What `example.py` does

1. Embeds the 20-row `tiny_corpus` with `tiny_bert`
2. `build_neighbor_graph` — a 3-NN edge table (60 edges)
3. `propagate_embeddings` — each vector mixed with its neighbours' (one hop)
4. `generate_structure_embeddings` — 8-dimensional vectors from the edges
5. `describe_table` — the propagated table's recorded producer, models and
   input anchors
6. `verify_materialization` — the bytes match the recorded digest
7. `staleness` — fresh against the recorded definition, stale (with the
   reason) against another
8. `derives_from` — the tables built from the base embeddings
9. `recompute` — re-run the graph's recorded producer; report what is
   downstream
10. `reconcile` — a dry-run cross-check of the catalog and the object store

## Run it

```bash
python cookbook/recipes/graph_and_lineage/example.py
```

Seconds on CPU. The guide's *Materialization contract* page explains the
manifest these verbs read.
