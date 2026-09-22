# Embed a Graph's Structure (You Have Only a Graph)

`generate_structure_embeddings` turns an **edge relation alone** into an
ordinary embedding table. It is for the graph whose nodes carry nothing an
encoder could read: a transaction graph of account ids, an entity graph of
opaque keys, a citation graph without abstracts. Every other door structure
enters through — a [graph-supervised fine-tune](./graph-supervised-finetune.md),
[propagation](./graph-propagation.md), a [declared-edge context](./graph-context.md)
— starts from content. This one starts from the graph.

It is training-free and deterministic: a data-plane encoder, not a learned
one. No message passing is fitted; the output is an embedding table like any
other — searchable, joinable, evaluable, cacheable, and placed on the compute
plane exactly as `generate_embeddings` output is.

## What it computes

The verb is a [propagation](./graph-propagation.md) whose initial features are
*generated* rather than read, and whose per-hop blocks are *mixed* rather than
dropped:

```text
X⁽⁰⁾ = the structural seed        one very sparse random projection row per
                                    node, scaled by its degree: d̃^β · r
X⁽ᵏ⁾ = P · X⁽ᵏ⁻¹⁾                 k = 1..K, the random walk P = D̃⁻¹(A + I)
out  = Σₖ wₖ · X⁽ᵏ⁾ / ‖X⁽ᵏ⁾‖₂     k = 0..K
```

Row `j` of `Pᵏ·R` is a random projection of node `j`'s `k`-step walk
distribution, so two nodes whose walks land in the same places — the same
community, the same role around a hub — get close rows, whatever their keys
are. The weights choose which walk lengths the embedding listens to. This is
FastRP (Chen et al. 2019) on the engine's own propagation operator; the
rustdoc of `jammi_ai::pipeline::graph_structure` states precisely where it
departs from the paper and why.

Three of those departures matter to a user:

- **Any graph is admissible, including a bipartite one.** The operator is the
  *lazy* walk over the self-loop-augmented graph. On a user ↔ item or
  account ↔ merchant graph the paper's walk alternates sides forever and its
  odd and even blocks disagree; the lazy walk places a node with its side
  under every block.
- **An isolated node is a fixed point**: every block is its own seed row, so
  it embeds to that row's direction rather than dividing by zero. Declare it
  with a self-edge `(v, v)` — every key the edge relation names is a node.
- **A node's seed row is a function of its key alone** (and its degree, when
  `β ≠ 0`). Adding a node or an edge leaves every other node's seed row
  bit-identical, and leaves the output of every node farther than `K` hops
  from the change bit-identical too (at `β = 0`, farther than `K − 1`: the
  change reaches a node only through the degree of the one it touched).

## Request

| Knob | Default | Meaning |
|---|---|---|
| `edge_source` / `direction` | — / `Undirected` | the graph, given exactly as a propagation takes it |
| `weighting` | `Uniform` | the walk operator (`DegreeNormalized` and `EdgeSimilarity` are the propagation's, unchanged) |
| `dimensions` | 256 | the embedding width `d` |
| `weights` | `[0, 0, 1, 1, 1]` | the readout weight of each block `0..=K` — the depth is `len − 1`; `weights[0]` weighs the seed itself |
| `beta` | `0` | the degree exponent of the seed scale `d̃^β`; negative damps hubs (the paper tunes it in `[−1, 0]`) |
| `sparsity` | `3` | the projection's `s`: entries are `±√s` with probability `1/(2s)` each, else `0` |
| `seed` | `0` | keys every node's projection stream |
| `hop_cap` | 8 | the most hops `weights` may ask for — the blocks are distinct scales read side by side, so this is not the smoothing cap |
| `key_column` | `None` | the source column holding the node keys; when set, `search` hydrates the source's columns |

Every invalid input is a typed refusal before anything is read: zero
dimensions, no weights, all-zero or non-finite weights, a depth past the cap,
a non-finite `beta`, a sparsity below one, a column the source lacks (named),
an edge source that does not exist, a graph with no node.

## Example: an account graph with no content

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::config::JammiConfig;
use jammi_ai::pipeline::graph_neighbourhood::EdgeSourceRef;
use jammi_ai::pipeline::graph_structure::StructureRequest;
use jammi_db::store::CachePolicy;
# async fn ex(config: JammiConfig) -> jammi_db::error::Result<()> {
# let session = Arc::new(InferenceSession::new(config).await?);

// `transfers` is a registered source of (payer, payee) rows; `accounts` is a
// registered source keyed by `account`. Neither carries text.
let (table, _outcome) = session
    .generate_structure_embeddings(
        &StructureRequest::new(
            "accounts",
            EdgeSourceRef::Registered {
                source_id: "transfers".into(),
                src_column: "payer".into(),
                dst_column: "payee".into(),
                type_column: None,
                weight_column: None,
                as_of_column: None,
            },
        )
        .with_key_column("account")
        .with_weights([0.0, 0.0, 1.0, 1.0, 1.0]),
        CachePolicy::Bypass,
    )
    .await?;

// Query by example: the accounts placed like this one, hydrated from
// `accounts` through the key column.
let similar = session
    .search_by_id("accounts", "acct-0007", 10, Some(&table.table_name), None)
    .await?
    .run()
    .await?;
# let _ = similar;
# Ok(())
# }
```

Python, embedded or remote — the same call:

```python
table = db.generate_structure_embeddings(
    "accounts",
    key_column="account",
    edge_source="transfers",
    edge_src_column="payer",
    edge_dst_column="payee",
    weights=[0, 0, 1, 1, 1],
)
```

## Searching a structure table

No encoder maps a query into this space, so there is nothing to search it
*with* but the table itself: **query by row key** (query-by-example — "nodes
placed like this one"), or by a vector read from the same table. A vector of
another width is refused by the index; a text encoder's vector of the same
width is a point of an unrelated space and ranks noise. Nothing in the engine
will encode text against a structure table, because nothing can.

## How well does it recover structure?

On a planted-partition graph of four communities of sixty nodes (an expected
in-community degree of ~9, cross-community ~2), with the defaults, each node's
five nearest neighbours under the structure embedding share its community
**99.2%** of the time; under the un-propagated seed alone (`weights = [1]`),
**22.3%** — chance, against a base rate of 24.7%. The test that measures this
(`crates/jammi-ai/tests/it/graph_structure.rs`) asserts floors of 0.85 and
0.35 respectively, and a gap of at least 0.5 between them.

## Determinism and scale

The output is byte-identical across `target_partitions`, across row orders of
the edge relation, across a graph declared with either endpoint first, and
whether or not a hop spilled — the propagation's [determinism
contract](./graph-propagation.md#determinism). A different seed, exponent,
sparsity or weight vector changes the table (and its definition hash, so a
`CachePolicy::Use` request never reuses across them).

The graph is never held in process memory: the plan runs out of core, bounded
by the spill disk and `[engine] memory_limit`, and refuses typed
(`ResourcesExhausted`) rather than being killed.

## References

- Chen, Sultan, Tian, Chen, Skiena 2019, *Fast and Accurate Network Embeddings
  via Very Sparse Random Projection (FastRP)*: <https://arxiv.org/abs/1908.11512>
- Achlioptas 2003, *Database-friendly random projections*; Li, Hastie, Church
  2006, *Very sparse random projections*.
