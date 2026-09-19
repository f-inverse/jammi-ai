# Fine-Tune from a Graph (Graph-Supervised)

> **Measured companion:** for the long-form, executed-and-measured Python treatment, see [The Cookbook → Representation Learning on Graphs](https://f-inverse.github.io/jammi-ai/cookbook/chapters/03-learn/learn.html).

Fine-tune embeddings so that **graph-neighbours are close in embedding space**.
This is node2vec / DeepWalk realised as Jammi config: it samples a graph into
contrastive `(anchor, positive, [hard_negative])` pairs and feeds them through
the *existing* fine-tune trainer. It authors **no GNN** — no message passing, no
new loss — it is a new *producer* of the pairs / triplet training data that drives
the same in-batch-negative (MNRL) / triplet objective as
[Fine-Tune for Your Domain](./fine-tuning.md).

Use it when your supervision is a **graph** rather than hand-built pairs: a
hierarchy, a crosswalk, a citation network, a set of coder-confirmed matches, or
the [neighbour graph](./build-neighbor-graph.md) Jammi itself builds.

## The load-bearing caveat: where the signal comes from

> **Declared edges teach; similarity edges echo.**

If you train on **S9-similarity edges** (the [neighbour
graph](./build-neighbor-graph.md), which is k-NN *under the base embedding
metric*), the walk-positives are mostly "things the model already thinks are
close" — so fine-tuning largely **re-learns the base metric**. That is a
degenerate feedback loop with little new signal.

Genuine gain comes from **declared / external edges** — structure the base
metric does *not* already encode:

- a **hierarchy** (parent/child categories),
- a **crosswalk** (version-A code ↔ version-B code),
- a **citation** / reference network,
- **coder-confirmed** pairs.

Tag your edges with their provenance. Similarity edges are an acceptable *weak
bootstrap* (e.g. to expand a sparse declared graph), but **never the sole
supervision**. The sampler tracks provenance and can report whether any declared
edge is present.

## Prepare the graph

Two sources: **node text** (what the encoder embeds) and **edges**.

`nodes.csv` — every node must be **text-bearing** (the encoder needs text;
pure-vector nodes are out of scope here):

```csv
id,text
c01,"acute myocardial infarction, initial"
c02,"acute myocardial infarction, subsequent"
c03,"benign essential hypertension"
```

`edges.csv` — directed edges; endpoints join to `id`:

```csv
src,dst
c01,c02
c02,c01
```

Register both as sources:

### Python

```python
db.add_source("nodes", path="/data/nodes.csv", format="csv")
db.add_source("edges", path="/data/edges.csv", format="csv")
```

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# use jammi_ai::session::InferenceSession;
# use jammi_db::source::{FileFormat, SourceConnection, SourceType};
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
for name in ["nodes", "edges"] {
    session.add_source(name, SourceType::File, SourceConnection {
        url: Some(format!("file:///data/{name}.csv")),
        format: Some(FileFormat::Csv),
        ..Default::default()
    }).await?;
}
# Ok(()) }
```

## Run the graph fine-tune

The sampler runs **biased random walks** (node2vec) over the edges: from each
node it walks `walk_length` (`L`) steps, biased by the return parameter `p` and
the in-out parameter `q`, and treats co-walked nodes as positives. `L > 1`
captures higher-order / community structure — `L = 1` is the degenerate 1-hop
case. Negatives are **in-batch** (every other pair's positive) plus
**structure-mined hard negatives** drawn from *outside* the anchor's
`exclude_hops`-hop neighbourhood (the false-negative guard — a node inside that
radius is likely a missing edge, i.e. a true positive).

### Python

```python
job = db.fine_tune_graph(
    node_source="nodes", id_column="id", text_column="text",
    edge_source="edges", src_column="src", dst_column="dst",
    base_model="local:/models/tiny_bert",
    edge_provenance="declared",   # "declared" teaches; "similarity" echoes
    walk_length=4, walks_per_node=2, return_p=1.0, in_out_q=1.0,
    graph_hard_negatives=1, exclude_hops=1, min_negatives=1,
    embedding_loss="mnrl",        # in-batch negatives (default); or "triplet"
    epochs=3, batch_size=8,
)
job.wait()
```

### Rust

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::fine_tune::FineTuneConfig;
# use jammi_ai::fine_tune::graph_sampler::{EdgeProvenance, GraphFineTuneSources, GraphSampleConfig};
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
let sources = GraphFineTuneSources {
    node_source: "nodes".into(), id_column: "id".into(), text_column: "text".into(),
    edge_source: "edges".into(), src_column: "src".into(), dst_column: "dst".into(),
    // Declared edges carry signal the base metric does not already encode.
    provenance: EdgeProvenance::Declared,
};
let sample = GraphSampleConfig {
    walk_length: 4, walks_per_node: 2, return_p: 1.0, in_out_q: 1.0,
    hard_negatives: 1, exclude_hops: 1, min_negatives: 1, seed: 0,
};
let job = session
    .fine_tune_graph(&sources, "local:/models/tiny_bert", sample, Some(FineTuneConfig::default()))
    .await?;
job.wait().await?;
# Ok(()) }
```

The output is a fine-tuned model; regenerate embeddings with it and they encode
the graph's structure ([`build_neighbor_graph`](./build-neighbor-graph.md),
search, and [propagation](./graph-propagation.md) all benefit).

## How it's materialised (and why the sample is reproducible)

Re-claiming the job (a lost lease, a retry) re-samples from the same seeded
config and re-reads the node/edge sources — the sample must be a function of
the node/edge *set*, not of whatever order the source happens to be scanned
in. Both scans carry an explicit order (ascending, by every projected
column), so two different physical layouts of the identical node/edge set
always sample byte-identical pairs; a node id that appears twice is refused,
typed, naming the id, rather than silently keeping "whichever row the scan
happened to see last."

The sampled pairs are materialised as an immutable, content-addressed
`TrainingSet`-kind table — the same producer funnel a tabular fine-tune's
source projection uses — so a `graph_fine_tune` job trains from a durable,
attested artifact rather than an ephemeral in-memory sample, and `recompute`
can replay it later over the current state of the node/edge sources. The
format (`pairs` / `triplet`) is
decided from `graph_hard_negatives` alone, never from what the first sampled
row happens to contain. If `graph_hard_negatives > 0` and some anchor's
entire candidate pool falls inside its own `exclude_hops`-hop neighbourhood
(nothing left outside the false-negative guard to mine a negative from), the
job fails with a typed error naming that anchor — never a row silently
trained with no negative.

`graph_fine_tune` supports an in-process multi-rank gang on ONE host: at
`world_size > 1` within `[worker] local_ranks`, the claiming worker itself
runs every rank, each over the SAME materialised table. It does **not**
yet support a multi-host (`Peer`) gang — `world_size` above `local_ranks`
is refused, typed, naming the reason (issue #538): a `Peer` member's own
read path over the table does not yet agree with rank 0's committed row
order, and there is no proof yet that the ranks' shards combine into a
correct gradient. Keep `world_size <= [worker] local_ranks` for now.

## Tuning knobs

| Knob | Effect |
| --- | --- |
| `walk_length` (`L`) | How far a positive can be. `1` = 1-hop only; `>1` = community structure. |
| `return_p` (`p`) | Large `p` discourages backtracking. |
| `in_out_q` (`q`) | `q < 1` explores outward (DFS-like); `q > 1` stays local (BFS-like). |
| `graph_hard_negatives` | Structure-mined hard negatives per pair. `0` = in-batch only. |
| `exclude_hops` | Hops of the anchor's neighbourhood excluded from its negatives (false-negative guard). |
| `min_negatives` | Minimum negative pool — guards against contrastive collapse on a tiny graph. |

## Compose with propagation

Both graph fine-tune and embedding
[**propagation**](./graph-propagation.md) encode homophily; stacking them
naively double-counts the same smoothing. The recommended order is **propagate
first, then fine-tune the head** (the SGC/APPNP decoupling) — not two independent
smoothing passes.

## Did it work? The circularity check

To confirm declared edges actually helped (and that you did not just re-learn the
base metric), evaluate on a **held-out golden set** — see [Did Structure Help? A
Graph-ML Evaluation Recipe](./graph-ml-eval.md):

1. Build two supervision graphs over the same nodes — one from **declared**
   edges, one from **S9-similarity** edges.
2. `fine_tune_graph` each; hold out a golden relevance set.
3. `eval_embeddings` the base model vs each fine-tune, with a paired
   significance test.
4. Expect the **declared-edge** model to beat the base significantly, and the
   **similarity-edge** model's gain to be near-zero — the degenerate feedback
   loop, measured rather than assumed.
