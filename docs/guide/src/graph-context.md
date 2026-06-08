# Condition a Prediction on Declared-Edge Context (Bring Your Own Graph)

[`assemble_context`](./assemble-context.md) builds a target's context set from
its **embedding-similar** neighbours — `search(target, k)`. That is the right
neighbourhood when similarity *is* the relationship you want to condition on.
But often the relationship that matters is one only your domain declares: the
papers a paper **cites**, the products **co-purchased** with this one, the
concepts an entity is an **`is-a`** of. Those edges carry structure an embedding
metric does not reconstruct — and they are exactly the context a prediction is
most defensible conditioned on.

> A context set is a *search*. It is also a *walk* — and the walk you care about
> is the one only you can declare.

S16-G makes that first-class: a second **context source** for the same
`assemble_context`. You register an edge relation, and a target's context becomes
its bounded, target-anchored declared-edge neighbourhood — pooled through the
same permutation-invariant set encoder, under the same leakage and tenancy
contracts, decoded by the same calibrated predictor. The engine transports
adjacency; it never learns what an edge *means*.

## When this is the right tool

Reach for declared-edge context when a target's most informative neighbours are
the ones your graph names, not the ones the base metric happens to place nearby:
a citation classifier, a co-purchase recommender, a knowledge-graph-backed
labeller, a transaction-graph scorer. Use [ANN context](./assemble-context.md)
when similarity is the signal; use `Hybrid` (below) when declared edges are the
signal but the graph is sparse and you want similarity to densify it.

It is **not** a general graph-traversal verb. The gather is *target-anchored* and
*depth-/fan-out-bounded* — never a free walk — which is what keeps it inside the
tenant-scope guarantee.

## The edge source

Any relation with two key columns is an edge source. Register it like any other
source, then point the gather at it:

- a [similarity graph](./build-neighbor-graph.md) you already built
  (`neighbor_graph`), or
- an external edge table you register (two key columns, optionally a type and a
  weight column).

The edge endpoints are row keys: a neighbour joins to its stored vector and its
outcome columns by key, exactly as an ANN neighbour does.

## Assemble declared-edge context

### Rust

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
use jammi_ai::pipeline::context_set::{ContextRequest, ContextSource};
use jammi_ai::pipeline::graph_neighbourhood::{EdgeGather, EdgeSourceRef};

# async fn demo(session: Arc<InferenceSession>, target_vector: Vec<f32>) -> jammi_db::error::Result<()> {
// A registered edge relation: a node's declared neighbours.
let gather = EdgeGather::new(EdgeSourceRef::Registered {
    source_id: "citations".into(),
    src_column: "from_id".into(),
    dst_column: "to_id".into(),
    type_column: None,
    weight_column: None,
    as_of_column: None,
});

let mut request = ContextRequest::new("papers", target_vector, 0);
request.source = ContextSource::Edges(gather);
// The target's own row key is the gather anchor (and is excluded from its
// own context — the leakage guard).
request.exclude_key = Some("paper-42".into());
request.value_columns = vec!["topic".into()];

let context = session.assemble_context(&request).await?;
println!("conditioned on {} declared-edge neighbours", context.context_size);
# Ok(())
# }
```

### Python

```python
out = db.predict_with_context_predictor(
    model_id,
    source="papers",
    target_key="paper-42",
    edge_source="citations",     # a registered edge relation
    edge_src_column="from_id",
    edge_dst_column="to_id",
    edge_hops=1,                 # bounded depth (default 1)
    edge_fanout=25,              # sample at most 25 neighbours per node per hop
)
# The prediction carries how its context was assembled and which rows it used:
assert out["source"] == "edges"
print(out["context_ref"])        # the declared-edge member keys
```

## The bounds — and why they are caps, not knobs to crank

- **`hops` (default 1, hard-capped).** Beyond ~2–3 hops, neighbour pooling is
  Laplacian over-smoothing — the pooled vector washes out and *loses* signal.
  Depth is a precision/recall trade-off, not "more context is better."
- **`fanout` (sample, don't enumerate).** A high-degree node's neighbourhood is
  intractable to enumerate; `fanout` bounds the neighbours sampled per node per
  hop. The sample is seeded-deterministically from the target, so a gather
  reproduces byte-identically. `None` is exact (enumerate all) and uses no
  randomness; a truncated neighbourhood is reported, never silently dropped.
- **`edge_types` / `min_weight`.** Filter which edges the walk follows. Types are
  a *filter*, never *learned* aggregation — a consumer wanting learned
  multi-relational message passing runs it in a graph library and registers the
  resulting node embeddings back as a source.

## Hybrid: declared edges ∪ similarity

When the graph is sparsely connected, union the declared-edge neighbours with the
ANN neighbours and pool once — declared edges as the signal, similarity as the
densifier:

```rust,no_run
# extern crate jammi_ai;
# use jammi_ai::pipeline::context_set::{ContextSource, HybridMerge};
# use jammi_ai::pipeline::graph_neighbourhood::{EdgeGather, EdgeSourceRef};
# fn demo(gather: EdgeGather) -> ContextSource {
ContextSource::Hybrid {
    ann_k: 10,
    edges: gather,
    merge: HybridMerge::Union,
}
# }
```

## Check homophily before you trust it

A declared edge can be **heterophilous** — `cites` may connect *dissimilar*
items — and pooling over a heterophilous edge type degrades a prediction rather
than helping it. Declared-edge context is an *option on the spectrum*, never
unconditionally better than similarity. Before you rely on a type, read its
homophily:

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::pipeline::graph_neighbourhood::EdgeGather;
# async fn demo(session: Arc<InferenceSession>, gather: &EdgeGather) -> jammi_db::error::Result<()> {
// Per-edge-type label-agreement over a labelled set: a type near the label's
// chance rate is heterophilous — pooling over it is unlikely to help.
let homophily = session
    .homophily_by_edge_type(gather, "papers", "id", "topic")
    .await?;
for (edge_type, agreement) in &homophily {
    println!("{edge_type}: {agreement:.2} label agreement");
}
# Ok(())
# }
```

The decoder also receives the target's own (ego) features alongside the pooled
neighbour vector, so it can down-weight an unhelpful neighbourhood rather than
being forced to trust it.

## Coverage over graph context

A graph-conditioned prediction is decoded and
[conformally wrapped](./conformal-prediction.md) exactly like an ANN-conditioned
one, and it **always serves**. But graph correlation can break the
exchangeability that *marginal* split-conformal coverage assumes — so the served
prediction carries the assembly **`source` fact** and its member keys, and the
coverage claim is attributed, never silently presented as a guarantee. Choosing
whether to apply a group-conditional (Mondrian) or importance-weighted lever, and
which cohort or weights to use, is a governance decision the serving layer
*applies* but never *makes*. The engine surfaces the fact; governance chooses the
lever.

## Tenancy

An edge source is tenant-scoped like every other source. The gather runs inside
the session's tenant scope, so an edge whose endpoint belongs to another tenant
is filtered before it is ever materialised — a declared edge cannot leak one
tenant's rows into another's context.
