# Graphs and lineage

Build a k-nearest-neighbour graph over a corpus's embeddings, smooth the
embeddings over it, derive structure embeddings from the graph alone — then
ask the engine what produced each table and whether it still holds.

**When to use this pattern.** You derive tables from tables (a graph from
embeddings, smoothed vectors from a graph) and need to know, later, exactly
what produced a table, whether its bytes are still the ones recorded, whether
it is stale against the definition you run now, and what depends on it.

## Run it

```bash
python cookbook/recipes/graph_and_lineage/example.py
```

Seconds on CPU. The guide's *Materialization contract* page explains the
manifest these verbs read.
