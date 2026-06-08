# Propagate Embeddings over a Graph (Decoupled GNN)

`propagate_embeddings` is the **forward pass of a graph convolution**, run as a
data-plane operation. For every row of an embedding table it replaces the row's
vector with an aggregate of its `k`-hop neighbourhood — `ÂᵏX` — and writes the
result as a new, ordinary embedding table (searchable, joinable, re-graphable).

This is the **propagate** half of a *decoupled* GNN. SGC showed the
nonlinearities between graph-conv layers are removable: precompute the
propagated features, then learn a simple head on top. APPNP added the teleport
restart that keeps deep propagation from collapsing. Neither needs autograd, an
architecture, or message-passing code — `ÂᵏX` is a graph join plus a grouped
vector average, and that is all this verb is.

It composes with anything that consumes an embedding table: search the
propagated vectors, evaluate them, build a neighbour graph over them, or
[fine-tune a head](./graph-supervised-finetune.md) on them (the SGC/APPNP order
— propagate first, then fine-tune).

## When propagation helps — measure homophily first

> **Smoothing helps only when neighbours share signal.**

Averaging a node with its neighbours *denoises* it **when the graph is
homophilous** — neighbours tend to be the same kind of thing (papers cite papers
on the same topic; co-purchased items share a category; KG entities of one type
link to one type). Then the propagated vectors cluster tighter and downstream
search / classification improves.

On a **heterophilous** graph — neighbours tend to *differ* — propagation mixes in
opposing signal and is **beaten by ignoring the graph entirely**. This is not a
silent failure mode to discover in production: measure it first. The
[per-edge-type homophily diagnostic](./graph-context.md) reports, for each edge
type, how often its endpoints share a label. Propagate over the homophilous
types; for genuinely heterophilous structure the answer is **learned attention**
(a later spec), not fixed averaging.

## The default is over-smoothing-safe

Iterated averaging is *exactly* the operation that collapses every node into one
indistinguishable point as the hop count grows (rank collapse). Three defaults
keep that in check:

- **PageRank-decay weighting** (`DegreeNormalized` + an `α`-teleport restart).
  Each hop re-mixes a share `α` of every node's *original* embedding back in, so
  a node stays anchored to itself however deep you go (the APPNP fix). `α`
  defaults to `0.1`.
- **Two hops** by default, **capped at three**. Beyond that, more hops add
  collapse, not signal.
- **Self-loops** (`Ã = A + I`). Every node aggregates over itself, so an
  **isolated node propagates to its own embedding** rather than vanishing, and
  the symmetric normalisation has no oscillating eigenmode.

## Weightings

| Weighting | Aggregation | Use |
|---|---|---|
| `DegreeNormalized` *(default)* | symmetric `Â = D̃^{-1/2}(A+I)D̃^{-1/2}`, with the `α`-teleport | the safe default (SGC/APPNP) |
| `Uniform` | random-walk mean `D̃^{-1}Ã` (each node = mean of itself + neighbours) | unweighted graphs, simplest smoothing |
| `EdgeSimilarity` | edge-weighted mean `Σ(w·x)/Σw` | use the edge weight as *fixed attention* (e.g. an S9 similarity edge); negative weights clamp to zero |

## Output: final block, or Jumping Knowledge

By default the output is the final propagated block `X⁽ᴷ⁾`, a `d`-dimensional
embedding table in the input's vector space.

`PropagationOutput::JumpingKnowledge` instead concatenates the per-hop blocks
`[X⁽⁰⁾ ‖ … ‖ X⁽ᴷ⁾]`, each L2-normalised before concat so the raw block does not
dominate cosine search. This lets a downstream head pick the right receptive
depth per node, but the output is `(K+1)·d`-dimensional and **indexes in its own
space** — do not search it against the original `d`-dimensional vectors.

## Example: propagate over a citation graph

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::config::JammiConfig;
use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
use jammi_ai::pipeline::graph_propagation::{PropagateRequest, PropagationWeighting};
# async fn ex(config: JammiConfig, model_id: &str) -> jammi_db::error::Result<()> {
# let session = Arc::new(InferenceSession::new(config).await?);

// Embed the documents first (any embedding model).
session
    .generate_text_embeddings("papers", model_id, &["abstract".into()], "id")
    .await?;

// Propagate over a declared citation edge source (src/dst are the paper ids,
// which are the embedding keys). Citations are undirected for smoothing.
let propagated = session
    .propagate_embeddings(
        &PropagateRequest::new(
            "papers",
            EdgeSourceRef::Registered {
                source_id: "citations".into(),
                src_column: "citing".into(),
                dst_column: "cited".into(),
                type_column: None,
                weight_column: None,
                as_of_column: None,
            },
        )
        .with_direction(EdgeDirection::Undirected)
        .with_weighting(PropagationWeighting::DegreeNormalized)
        .with_hops(2),
    )
    .await?;

// The result is an ordinary embedding table: search it, evaluate it, or graph
// it like any other.
let neighbours = session
    .sql(&format!(
        "SELECT _row_id FROM \"jammi.{}\" LIMIT 5",
        propagated.table_name
    ))
    .await?;
# let _ = neighbours;
# Ok(())
# }
```

## Propagating over an S9 similarity graph

You can also propagate over the [similarity graph](./build-neighbor-graph.md)
Jammi itself builds — pass its table name as the edge source:

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::config::JammiConfig;
use jammi_ai::pipeline::graph_neighbourhood::EdgeSourceRef;
use jammi_ai::pipeline::graph_propagation::PropagateRequest;
# async fn ex(config: JammiConfig, graph_table: &str) -> jammi_db::error::Result<()> {
# let session = Arc::new(InferenceSession::new(config).await?);
let propagated = session
    .propagate_embeddings(&PropagateRequest::new(
        "items",
        EdgeSourceRef::NeighborGraph {
            table_name: graph_table.into(),
        },
    ))
    .await?;
# let _ = propagated;
# Ok(())
# }
```

But note the caveat from [graph-supervised fine-tuning](./graph-supervised-finetune.md):
a similarity graph is k-NN *under the base metric*, so propagating over it mostly
re-averages things the model already thinks are close. **Declared** edges (a
citation network, a co-purchase log, a knowledge graph's typed relations) carry
structure the base metric does not already encode — that is where propagation
adds signal.

## Determinism

Propagation is deterministic: every fold, teleport, and weighted sum runs in
`f64` over a fixed `(node, neighbour)` order, so the output is **byte-identical**
regardless of how many threads the engine runs. It is the reproducible point on
the structure-aware spectrum — fixed averaging, no learned parameters.

## Bounds

The edge set is loaded under a row ceiling (`PropagateRequest::max_rows`); a
graph larger than that is refused loudly rather than risking an out-of-memory
pass. Whole-graph propagation beyond memory (chunking by the join) is future
work.
