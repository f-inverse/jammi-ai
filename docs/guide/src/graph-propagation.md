# Propagate Embeddings over a Graph (Decoupled GNN)

> **Measured companion:** for the long-form, executed-and-measured Python treatment, see [The Cookbook → Graph Signal Processing](https://f-inverse.github.io/jammi-ai/cookbook/chapters/02-analyze/analyze.html).

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

It needs an embedding table to start from. A graph whose nodes have no content
to embed is encoded from its [structure alone](./graph-structure.md) — the
same propagation, seeded from the graph.

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

## Output: how the per-hop blocks are read out

Every output is one readout of the hop history `[X⁽⁰⁾, …, X⁽ᴷ⁾]`: each block
weighed, optionally L2-normalised, and summed or concatenated.

| `PropagationOutput` | Reads | Output |
|---|---|---|
| `Final` *(default)* | the last block `X⁽ᴷ⁾`, as is | `d`-dim, in the input's space |
| `JumpingKnowledge` | every block, each L2-normalised, concatenated | `(K+1)·d`-dim, its own space |
| `WeightedSum { weights }` | each block `k` with `wₖ ≠ 0`, L2-normalised, weighed by `wₖ`, summed | `d`-dim, its own space |

`JumpingKnowledge` lets a downstream head pick the right receptive depth per
node; `WeightedSum` mixes chosen diffusion scales into one vector (the readout a
[structure embedding](./graph-structure.md) uses). Both index in **their own
space** — do not search them against the original `d`-dimensional vectors.
`weights` has exactly `hops + 1` entries, `weights[0]` weighing `X⁽⁰⁾` itself.

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
use jammi_db::store::CachePolicy;
# async fn ex(config: JammiConfig, model_id: &str) -> jammi_db::error::Result<()> {
# let session = Arc::new(InferenceSession::new(config).await?);

// Embed the documents first (any embedding model).
session
    .generate_text_embeddings("papers", model_id, &["abstract".into()], "id", CachePolicy::Bypass, None)
    .await?;

// Propagate over a declared citation edge source (src/dst are the paper ids,
// which are the embedding keys). Citations are undirected for smoothing.
let (propagated, _outcome) = session
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
        CachePolicy::Bypass,
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
use jammi_db::store::CachePolicy;
# async fn ex(config: JammiConfig, graph_table: &str) -> jammi_db::error::Result<()> {
# let session = Arc::new(InferenceSession::new(config).await?);
let (_propagated, _outcome) = session
    .propagate_embeddings(
        &PropagateRequest::new(
            "items",
            EdgeSourceRef::NeighborGraph {
                table_name: graph_table.into(),
            },
        ),
        CachePolicy::Bypass,
    )
    .await?;
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

Propagation is deterministic: every fold, teleport, and readout runs in `f64`
with one final `f32` cast, and each node's neighbours fold in one fixed order
(their keys), in one pass, by one accumulator. The output is **byte-identical**
regardless of `target_partitions`, of the order the edge relation's rows
arrive in, and of whether a hop spilled to disk. It is the reproducible point
on the structure-aware spectrum — fixed averaging, no learned parameters.

The output is `f32`, and `f32` bits are not the same on every CPU, so the
byte-identity is a property of one machine: `jammi-bench propagate` runs the
same propagation at `target_partitions` 1 and N over a sweep of graph sizes and
prints each run's digest of the key-sorted vectors (with its per-iteration
timings, its peak resident set and the vectors file), and the engine's tests
hold the two digests equal on the box they run on and hold that one hop fewer,
one more, or a different `α` moves them.

The operator is small enough to restate exactly: over the undirected edge *set*
(a pair listed twice, or in both directions, is one edge; a listed self-edge is
dropped), `Ã = A + I`, `d̃ = deg + 1`, and
`X⁽ᵏ⁾ = α·X⁽⁰⁾ + (1−α)·D̃^{-1/2} Ã D̃^{-1/2}·X⁽ᵏ⁻¹⁾` in `f64` with one final
`f32` cast. `crates/jammi-bench/reference/torch_propagate.py --impl exact` is
that restatement in `torch.sparse`, reading the same input files;
`--impl pyg` is `torch_geometric.nn.APPNP` (`SGConv` at `α = 0`), which computes
the same operator in `f32`.

## Scale

A hop is a join of the edge relation to the node state, a shuffle by node, and
a sorted fold — stock plan operators around one fold operator — so a
propagation runs **out of core**: its joins and sorts hold pool reservations
and spill, and the graph is bounded by the spill disk rather than by memory.
The pool is shared among the operators holding memory at that moment, so it
must hold a merge reservation and a batch for the sorts and the join of one
hop on every partition; below that the request is the typed
`ResourcesExhausted`, never an out-of-memory kill.

The graph is read **once**: the adjacency is snapshotted at the start of the
run as a working table in the result store, and every hop reads the snapshot.
An edge source that changes while a propagation runs cannot give hops that
disagree. A hop is one plan; its state is handed to the next hop as another
working table, reclaimed once read, and the snapshot when the run ends,
however it ends. The last hop's plan is written through the embedding sink, so it is placed on the
compute plane when one can hold it, exactly as `generate_embeddings` is.
