//! Graph feature propagation — the decoupled-GNN forward pass (SGC / APPNP),
//! a deterministic data-plane operation, no autograd and no architecture.
//!
//! # What it computes
//!
//! The forward pass of a graph convolution is, mathematically, repeatedly
//! replacing each node's embedding with an aggregate of its neighbours'
//! embeddings — `Âᵏ·X`. SGC showed the per-layer nonlinearities are removable:
//! precompute the propagated features, then learn a simple head (the *decoupled*
//! GNN). This module is the **propagate** half — it takes an embedding table and
//! a graph and emits a new embedding table where each row is its `k`-hop
//! neighbourhood aggregate. The **learn** half (fine-tune a head on the
//! propagated features) is the existing fine-tune path; the recommended order is
//! propagate-then-fine-tune, never two independent smoothing passes.
//!
//! # The recurrence
//!
//! With **self-loops** `Ã = A + I` and the symmetric normalisation
//! `Â = D̃^{-1/2}(A+I)D̃^{-1/2}` over the augmented degrees `d̃ = deg + 1`, the
//! default ([`PropagationWeighting::DegreeNormalized`]) iterates the APPNP
//! recurrence with an `α`-teleport restart:
//!
//! ```text
//! X⁽ᵏ⁾ = (1−α)·Â·X⁽ᵏ⁻¹⁾ + α·X⁽⁰⁾
//! ```
//!
//! The self-loop is load-bearing twice over: it removes the `−1` eigenmode that
//! makes a plain `D^{-1/2}AD^{-1/2}` oscillate, and it makes an **isolated node
//! propagate to its own `X⁽⁰⁾`** for free (its only neighbour is itself). The
//! `α`-restart is the oversmoothing fix: each hop re-anchors every node to its
//! original embedding, so deep propagation does not collapse the representations
//! into one low-rank subspace (APPNP). `α` defaults to `0.1`, hops to `2`, and
//! the hop count is capped at [`DEFAULT_HOP_CAP`].
//!
//! # How the aggregation runs (the fold trick)
//!
//! `Â·X` is a **grouped vector aggregation**: the `D̃^{-1/2}` factors are folded
//! into the per-node vectors as `O(nodes)` `f64` scaling, so the neighbour
//! aggregation over the self-loop-augmented adjacency is a plain element-wise
//! sum (symmetric) or mean ([`PropagationWeighting::Uniform`], the random-walk
//! `D̃^{-1}Ã`). That element-wise reduction is the P3 vector-aggregation
//! *operator* — propagation reuses `fold_vectors_in_order`, the shared
//! fixed-order reduction the UDAF accumulator also folds through, so
//! there is one reduction implementation — with no second normalisation pathway
//! and no per-element `vector_scale` function. It does **not** route the per-hop
//! aggregation through the SQL UDAF: the streaming aggregate cannot guarantee a
//! byte-identical result across partitionings (`f64` `+` is non-associative and a
//! parallel plan fixes neither the fold nor the merge order), so propagation
//! imposes its own fixed `(group, neighbour)` fold order over the **bounded**
//! collected adjacency in `f64` (the determinism contract below); the edge set is
//! bounded by [`PropagateRequest::max_rows`]. The `α`-teleport mix is then applied
//! per node in Rust (`f64`). The only weighting that needs per-edge weights —
//! [`PropagationWeighting::EdgeSimilarity`] — computes `Σ(w·x)/Σw` over the same
//! bounded rows, clamping the S9 cosine similarity (which lives in `[−1, 1]`) to
//! `max(sim, 0)`; a node whose neighbour weights sum to zero falls back to its
//! own `X⁽⁰⁾`.
//!
//! # Determinism
//!
//! Every fold, teleport, and weighted sum runs in `f64` with a single final
//! `f32` cast, over a total `(group, neighbour)` order. Because that order is
//! fixed and the whole pass runs in one Rust reduction (not the partition-split
//! SQL aggregate), the output is byte-identical regardless of how many execution
//! threads the engine runs — the reproducible point on the propagate/learn
//! spectrum (contrast S13's learned aggregation, and the streaming UDAF, whose
//! additive arms are only value-stable up to `f64` rounding across partitions).
//!
//! # Homophily
//!
//! Smoothing helps only when neighbours share signal. On a *heterophilous*
//! graph (neighbours tend to differ) propagation aggregates harmful signal and
//! is beaten by a structure-ignoring baseline; the homophily diagnostic
//! ([`InferenceSession::homophily_by_edge_type`]) measures this first, and the
//! learned-attention answer is S13. This module transports adjacency; it never
//! judges what an edge means.
//!
//! # References
//! - Wu et al. 2019, *Simplifying Graph Convolutional Networks (SGC)*:
//!   <https://arxiv.org/abs/1902.07153>
//! - Gasteiger et al. 2019, *Predict then Propagate (APPNP)*:
//!   <https://arxiv.org/abs/1810.05997>
//! - Xu et al. 2018, *Jumping Knowledge Networks*:
//!   <https://arxiv.org/abs/1806.03536>
//! - Zhu et al. 2020, *Beyond Homophily in GNNs*:
//!   <https://arxiv.org/abs/2006.11468>

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StringArray};
use arrow::datatypes::DataType;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};

use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef, DEFAULT_HOP_CAP};
use crate::query::vector_agg_udaf::{fold_vectors_in_order, VectorReduce};
use crate::session::InferenceSession;

/// Default number of propagation hops. Two hops is the homophily sweet spot;
/// beyond ~3 hops aggregation is over-smoothing (the `α`-restart mitigates it
/// but does not license unbounded depth).
pub const DEFAULT_PROPAGATE_HOPS: usize = 2;

/// Default APPNP teleport probability. Each hop re-mixes `α` of every node's
/// original `X⁽⁰⁾` back in, anchoring it against over-smoothing.
pub const DEFAULT_TELEPORT_ALPHA: f64 = 0.1;

/// Row-count ceiling on the loaded edge set. Above this the propagation refuses
/// rather than load the whole graph into memory and risk an OOM — mirrors
/// [`crate::pipeline::neighbor_graph::DEFAULT_EXACT_MAX_ROWS`]. Whole-graph
/// propagation beyond memory (chunking by the join) is documented future work,
/// not a silent failure.
pub const DEFAULT_PROPAGATE_MAX_ROWS: usize = 2_000_000;

/// How neighbour contributions are weighted when a hop aggregates them.
///
/// The default [`DegreeNormalized`](PropagationWeighting::DegreeNormalized) is
/// the symmetric `Â = D̃^{-1/2}(A+I)D̃^{-1/2}` of SGC/APPNP, paired with the
/// `α`-teleport restart — the PageRank-decay form that keeps propagation
/// anchored against over-smoothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PropagationWeighting {
    /// Random-walk normalisation `D̃^{-1}Ã`: each node's new vector is the plain
    /// mean of itself and its neighbours. No degree fold needed — the hop is a
    /// `vector_mean` over the self-loop-augmented neighbourhood.
    Uniform,
    /// Symmetric normalisation `D̃^{-1/2}(A+I)D̃^{-1/2}` (the default). The
    /// degree factors are folded into the per-node vectors, so the hop is a
    /// `vector_sum` over the augmented neighbourhood. Combined with the
    /// `α`-teleport this is the APPNP/PageRank-decay propagation.
    #[default]
    DegreeNormalized,
    /// Edge-weighted mean `Σ(w·x)/Σw` over the neighbourhood, where `w` is the
    /// declared edge weight clamped to `max(weight, 0)` ("fixed attention" from
    /// an S9 similarity edge). A node whose neighbour weights sum to zero falls
    /// back to its own `X⁽⁰⁾`.
    EdgeSimilarity,
}

/// What the propagation emits.
///
/// A typed mode rather than a boolean flag, because
/// [`JumpingKnowledge`](PropagationOutput::JumpingKnowledge) changes the output
/// dimensionality — concatenating the per-hop blocks yields `(K+1)·d` columns in
/// their own vector space, a same-space hazard a `bool` would hide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PropagationOutput {
    /// Emit only the final `X⁽ᴷ⁾` — a `d`-dimensional embedding table in the
    /// input's vector space.
    #[default]
    Final,
    /// Concatenate `[X⁽⁰⁾ ‖ X⁽¹⁾ ‖ … ‖ X⁽ᴷ⁾]`, each per-hop block L2-normalised
    /// before concat so the raw block's larger norm does not dominate cosine
    /// search (Jumping Knowledge). The output is `(K+1)·d`-dimensional and
    /// indexes in **its own space** — do not search it against `d`-dim vectors.
    JumpingKnowledge,
}

/// Which declared edge relation to propagate over and how the walk reads it. The
/// edge source and direction reuse the shared graph-neighbourhood config types;
/// the loader here is propagation's own bounded, tenant-scoped scan.
#[derive(Debug, Clone)]
pub struct PropagateRequest {
    /// The source whose embedding table holds `X⁽⁰⁾`.
    pub source_id: String,
    /// The specific embedding table to propagate, or `None` to resolve the
    /// source's default embedding table.
    pub embedding_table: Option<String>,
    /// The edge relation defining the graph.
    pub edge_source: EdgeSourceRef,
    /// How the direction of an edge is read when building the neighbourhood.
    pub direction: EdgeDirection,
    /// Number of hops. Clamped to `[1, hop_cap]`.
    pub hops: usize,
    /// The hard depth cap `hops` is clamped to.
    pub hop_cap: usize,
    /// Neighbour-contribution weighting.
    pub weighting: PropagationWeighting,
    /// APPNP teleport probability (the share of `X⁽⁰⁾` re-mixed each hop).
    /// Applied to [`DegreeNormalized`](PropagationWeighting::DegreeNormalized)
    /// and [`Uniform`](PropagationWeighting::Uniform); `0.0` disables the restart
    /// (plain SGC smoothing).
    pub alpha: f64,
    /// What to emit (final block, or the per-hop Jumping-Knowledge concat).
    pub output: PropagationOutput,
    /// Refuse if the loaded edge set exceeds this many rows.
    pub max_rows: usize,
}

impl PropagateRequest {
    /// A propagation with the over-smoothing-safe defaults: the symmetric
    /// degree-normalised `Â` with an `α=0.1` teleport restart, two hops capped at
    /// [`DEFAULT_HOP_CAP`], out-edges, final-block output, and the default row
    /// ceiling.
    pub fn new(source_id: impl Into<String>, edge_source: EdgeSourceRef) -> Self {
        Self {
            source_id: source_id.into(),
            embedding_table: None,
            edge_source,
            direction: EdgeDirection::Out,
            hops: DEFAULT_PROPAGATE_HOPS,
            hop_cap: DEFAULT_HOP_CAP,
            weighting: PropagationWeighting::DegreeNormalized,
            alpha: DEFAULT_TELEPORT_ALPHA,
            output: PropagationOutput::Final,
            max_rows: DEFAULT_PROPAGATE_MAX_ROWS,
        }
    }

    /// Builder: set the embedding table to propagate (default: the source's
    /// resolved embedding table).
    pub fn with_embedding_table(mut self, table: impl Into<String>) -> Self {
        self.embedding_table = Some(table.into());
        self
    }

    /// Builder: set how an edge's direction is read.
    pub fn with_direction(mut self, direction: EdgeDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Builder: set the hop count (clamped to `[1, hop_cap]` at run time).
    pub fn with_hops(mut self, hops: usize) -> Self {
        self.hops = hops;
        self
    }

    /// Builder: set the neighbour weighting.
    pub fn with_weighting(mut self, weighting: PropagationWeighting) -> Self {
        self.weighting = weighting;
        self
    }

    /// Builder: set the APPNP teleport probability.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: set the output mode (final block or Jumping-Knowledge concat).
    pub fn with_output(mut self, output: PropagationOutput) -> Self {
        self.output = output;
        self
    }

    /// The depth actually run: `hops` clamped to `[1, hop_cap]`.
    pub fn effective_hops(&self) -> usize {
        self.hops.clamp(1, self.hop_cap.max(1))
    }
}

/// The model-id provenance recorded on a propagated embedding table. Short and
/// fixed (the source embedding table is recorded separately in `derived_from`),
/// so the generated table name stays a sane length.
const PROPAGATE_MODEL_ID: &str = "graph_propagate";

/// One directed adjacency pair `(group, neighbour)` — the node whose new vector
/// is being aggregated and one node contributing to it. Self-loops are present
/// as `(v, v)`.
struct AdjacencyPair {
    group: String,
    neighbour: String,
}

/// A weighted neighbour contribution, used only by the edge-similarity path.
struct WeightedNeighbour {
    group: String,
    neighbour: String,
    weight: f64,
}

impl InferenceSession {
    /// Propagate an embedding table's features over a declared graph and
    /// materialise the result as a new, searchable embedding table.
    ///
    /// Loads `X⁽⁰⁾` from the source's embedding table and the tenant-scoped edge
    /// relation, then iterates [`PropagateRequest::effective_hops`] hops of the
    /// configured weighting with the `α`-teleport restart. The grouped neighbour
    /// aggregation reuses the engine's vector-aggregation *reduction operator*
    /// over a fixed `(group, neighbour)` fold order (the degree factors folded
    /// into the vectors in Rust) — it deliberately does not route the per-hop
    /// aggregation through the streaming SQL UDAF, which cannot promise a
    /// byte-identical result across partitionings; the teleport mix and any
    /// edge-weighting run per node in `f64`. The output is a normal `kind=Model`
    /// embedding table with a sidecar index, `derived_from` the source table.
    ///
    /// The edge load runs through the generic SQL surface so the tenant-scope
    /// analyzer rule scopes the scan — a cross-tenant endpoint is filtered before
    /// it reaches the adjacency.
    pub async fn propagate_embeddings(
        self: &Arc<Self>,
        request: &PropagateRequest,
    ) -> Result<ResultTableRecord> {
        let table = self
            .catalog()
            .resolve_embedding_table(&request.source_id, request.embedding_table.as_deref())
            .await?;
        let dimensions = table.dimensions.ok_or_else(|| {
            JammiError::Other(format!(
                "propagate_embeddings: embedding table '{}' carries no dimensions",
                table.table_name
            ))
        })? as usize;

        // X⁽⁰⁾: the source vectors keyed by `_row_id`, in a stable total order so
        // the materialised output is reproducible.
        let initial = self.load_initial_features(&table, dimensions).await?;
        if initial.is_empty() {
            return Err(JammiError::Other(format!(
                "propagate_embeddings: embedding table '{}' has no rows",
                table.table_name
            )));
        }

        // The full hop history `[X⁽⁰⁾, X⁽¹⁾, …, X⁽ᴷ⁾]`. `Final` output keeps the
        // last block; `JumpingKnowledge` concatenates them all.
        let history = match request.weighting {
            PropagationWeighting::EdgeSimilarity => {
                self.propagate_edge_similarity(request, &initial, dimensions)
                    .await?
            }
            PropagationWeighting::Uniform => {
                self.propagate_normalized(request, &initial, dimensions, Normalization::RandomWalk)
                    .await?
            }
            PropagationWeighting::DegreeNormalized => {
                self.propagate_normalized(request, &initial, dimensions, Normalization::Symmetric)
                    .await?
            }
        };

        let (rows, out_dim) = assemble_output(&initial, &history, request.output, dimensions);

        self.result_store()
            .materialize_embedding_table(
                self.context(),
                &request.source_id,
                PROPAGATE_MODEL_ID,
                Some(&table.table_name),
                &rows,
                out_dim,
            )
            .await
    }

    /// Load `X⁽⁰⁾` — every `(_row_id, vector)` of the embedding table — into a
    /// stable, `_row_id`-sorted list of `f64` vectors. The total order is what
    /// keeps the materialised output byte-identical across runs and partitions.
    async fn load_initial_features(
        &self,
        table: &ResultTableRecord,
        dimensions: usize,
    ) -> Result<Vec<NodeFeatures>> {
        let batches = self
            .sql(&format!(
                "SELECT _row_id, vector FROM \"jammi.{}\"",
                table.table_name
            ))
            .await?;

        let mut nodes: Vec<NodeFeatures> = Vec::new();
        for batch in &batches {
            let row_ids = read_row_id_column(batch, &table.table_name)?;
            let mut vectors: Vec<Vec<f32>> = Vec::new();
            jammi_db::store::vectors::extend_with_fixed_size_list_f32(
                batch,
                &table.table_name,
                "vector",
                &mut vectors,
            )?;
            for (row_id, vector) in row_ids.into_iter().zip(vectors) {
                if vector.len() != dimensions {
                    return Err(JammiError::Other(format!(
                        "propagate_embeddings: row '{row_id}' has width {} (expected {dimensions})",
                        vector.len()
                    )));
                }
                nodes.push(NodeFeatures {
                    row_id,
                    features: vector.into_iter().map(f64::from).collect(),
                });
            }
        }
        // Total order on `_row_id` so the hop loop, the aggregation join, and the
        // final write are all deterministic.
        nodes.sort_by(|a, b| a.row_id.cmp(&b.row_id));
        Ok(nodes)
    }

    /// The degree-normalised propagation path (random-walk or symmetric): iterate
    /// `α·X⁽⁰⁾ + (1−α)·Â·X` via the fold trick, the neighbour sum/mean folded
    /// deterministically over the bounded adjacency.
    async fn propagate_normalized(
        self: &Arc<Self>,
        request: &PropagateRequest,
        initial: &[NodeFeatures],
        dimensions: usize,
        norm: Normalization,
    ) -> Result<Vec<HashMap<String, Vec<f64>>>> {
        let pairs = self.load_adjacency_pairs(request, initial).await?;
        // Augmented degree d̃ = (neighbour count incl. the self-loop). Built from
        // the same pair list the aggregation groups over, so degree and
        // aggregation are consistent.
        let degree = augmented_degrees(&pairs);

        // The hop history starts at X⁽⁰⁾; each iteration appends X⁽ᵏ⁾.
        let mut history: Vec<HashMap<String, Vec<f64>>> = vec![initial
            .iter()
            .map(|n| (n.row_id.clone(), n.features.clone()))
            .collect()];
        let alpha = request.alpha;

        for _ in 0..request.effective_hops() {
            let current = history.last().expect("history seeded with X⁽⁰⁾");
            // Fold the source-side degree factor into the vectors before the
            // grouped aggregation. `RandomWalk` needs no fold (D̃^{-1}Ã = mean);
            // `Symmetric` scales by 1/√d̃ on both sides.
            let scaled: HashMap<String, Vec<f64>> = match norm {
                Normalization::RandomWalk => current.clone(),
                Normalization::Symmetric => current
                    .iter()
                    .map(|(id, v)| {
                        let f = inv_sqrt_degree(&degree, id);
                        (id.clone(), v.iter().map(|x| x * f).collect())
                    })
                    .collect(),
            };

            // The grouped neighbour aggregation, folded in a fixed neighbour
            // order so the result is byte-identical regardless of how the engine
            // would partition it (the determinism contract). `RandomWalk` divides
            // by the group count (the mean); `Symmetric` leaves the sum for the
            // source-side degree fold below.
            let aggregated = aggregate_neighbours(
                &pairs,
                &scaled,
                dimensions,
                norm == Normalization::RandomWalk,
            );

            // Final degree factor (`Symmetric` only) + the α-teleport mix, per
            // node in f64. A node with no aggregate (it appears in no pair —
            // impossible once self-loops are added, but defended) keeps X⁽⁰⁾.
            let mut next = HashMap::with_capacity(current.len());
            for node in initial {
                let id = &node.row_id;
                let neighbour_term = aggregated.get(id).map(|agg| match norm {
                    Normalization::Symmetric => {
                        let f = inv_sqrt_degree(&degree, id);
                        agg.iter().map(|x| x * f).collect::<Vec<f64>>()
                    }
                    Normalization::RandomWalk => agg.clone(),
                });
                next.insert(
                    id.clone(),
                    teleport_mix(&node.features, neighbour_term, alpha),
                );
            }
            history.push(next);
        }
        Ok(history)
    }

    /// The `EdgeSimilarity` path: a weighted mean `Σ(w·x)/Σw` per node, computed
    /// in Rust over the bounded weighted-edge rows (the only weighting that needs
    /// per-edge weights). Self-loops carry weight 1.
    async fn propagate_edge_similarity(
        self: &Arc<Self>,
        request: &PropagateRequest,
        initial: &[NodeFeatures],
        dimensions: usize,
    ) -> Result<Vec<HashMap<String, Vec<f64>>>> {
        let weighted = self.load_weighted_neighbours(request, initial).await?;

        // The hop history starts at X⁽⁰⁾; each iteration appends X⁽ᵏ⁾.
        let mut history: Vec<HashMap<String, Vec<f64>>> = vec![initial
            .iter()
            .map(|n| (n.row_id.clone(), n.features.clone()))
            .collect()];
        let alpha = request.alpha;

        for _ in 0..request.effective_hops() {
            let current = history.last().expect("history seeded with X⁽⁰⁾");
            // Per-node accumulator of Σ(w·x) and Σw.
            let mut sums: HashMap<&str, (Vec<f64>, f64)> = HashMap::new();
            for wn in &weighted {
                let Some(x) = current.get(&wn.neighbour) else {
                    // A neighbour with no feature vector (null / absent) carries
                    // no signal; it is skipped rather than zero-imputed.
                    continue;
                };
                let entry = sums
                    .entry(wn.group.as_str())
                    .or_insert_with(|| (vec![0.0; dimensions], 0.0));
                for (acc, xi) in entry.0.iter_mut().zip(x) {
                    *acc += wn.weight * xi;
                }
                entry.1 += wn.weight;
            }

            let mut next = HashMap::with_capacity(current.len());
            for node in initial {
                let id = &node.row_id;
                // Σw == 0 (every neighbour weight clamped away) → fall back to
                // X⁽⁰⁾; the self-loop weight 1 keeps Σw > 0 for every present
                // node, so this fires only for a genuinely all-negative
                // neighbourhood with the self-loop disabled — defended either way.
                let neighbour_term = sums.get(id.as_str()).and_then(|(sum, total)| {
                    (*total > 0.0).then(|| sum.iter().map(|s| s / total).collect::<Vec<f64>>())
                });
                next.insert(
                    id.clone(),
                    teleport_mix(&node.features, neighbour_term, alpha),
                );
            }
            history.push(next);
        }
        Ok(history)
    }

    /// Load the self-loop-augmented adjacency pairs for the `Uniform` /
    /// `DegreeNormalized` paths: the tenant-scoped declared edges (oriented per
    /// [`PropagateRequest::direction`]) plus a `(v, v)` self-loop for every node.
    async fn load_adjacency_pairs(
        self: &Arc<Self>,
        request: &PropagateRequest,
        initial: &[NodeFeatures],
    ) -> Result<Vec<AdjacencyPair>> {
        let edges = self.load_propagation_edges(request).await?;
        let mut pairs: Vec<AdjacencyPair> = Vec::with_capacity(edges.len() * 2 + initial.len());
        for (src, dst, _w) in &edges {
            // A declared `(v, v)` edge is dropped here: the `Ã = A + I` self-loop
            // injected below is the canonical one, so a declared self-edge must
            // not add a second `(v, v)` pair (which would inflate v's augmented
            // degree to deg+2, or deg+3 doubled under `Undirected`, skewing `Â`).
            if src == dst {
                continue;
            }
            push_oriented_pairs(&mut pairs, src, dst, request.direction);
        }
        // Ã = A + I: every node aggregates over itself, exactly once.
        for node in initial {
            pairs.push(AdjacencyPair {
                group: node.row_id.clone(),
                neighbour: node.row_id.clone(),
            });
        }
        // Total order on (group, neighbour) so the f64 neighbour fold runs in a
        // fixed order independent of the edge scan order — the byte-identical
        // determinism contract.
        pairs.sort_by(|a, b| (&a.group, &a.neighbour).cmp(&(&b.group, &b.neighbour)));
        Ok(pairs)
    }

    /// Load the self-loop-augmented *weighted* neighbours for the
    /// `EdgeSimilarity` path: each declared edge's weight clamped to
    /// `max(weight, 0)` (an absent weight is treated as 1.0 — an unweighted
    /// declared edge is a present, full-strength edge), plus a weight-1
    /// self-loop per node.
    async fn load_weighted_neighbours(
        self: &Arc<Self>,
        request: &PropagateRequest,
        initial: &[NodeFeatures],
    ) -> Result<Vec<WeightedNeighbour>> {
        let edges = self.load_propagation_edges(request).await?;
        let mut weighted: Vec<WeightedNeighbour> =
            Vec::with_capacity(edges.len() * 2 + initial.len());
        for (src, dst, w) in &edges {
            // A declared `(v, v)` edge is dropped: the canonical weight-1 self-loop
            // injected below is the only self-contribution, so a declared self-edge
            // must not add a second (weighted) one that would double-count v.
            if src == dst {
                continue;
            }
            // S9 cosine similarity lives in [−1, 1]; a negative-weight edge
            // carries anti-signal and is clamped to zero rather than subtracted.
            let weight = w.unwrap_or(1.0).max(0.0);
            for (group, neighbour) in oriented_endpoints(src, dst, request.direction) {
                weighted.push(WeightedNeighbour {
                    group: group.to_string(),
                    neighbour: neighbour.to_string(),
                    weight,
                });
            }
        }
        for node in initial {
            weighted.push(WeightedNeighbour {
                group: node.row_id.clone(),
                neighbour: node.row_id.clone(),
                weight: 1.0,
            });
        }
        Ok(weighted)
    }

    /// Load the declared edge rows `(src, dst, weight)` of the propagation's edge
    /// source, tenant-scoped, refusing past the row ceiling.
    ///
    /// The scan runs through the generic SQL surface so the tenant-scope analyzer
    /// rule injects the `tenant_id` predicate — a cross-tenant endpoint is
    /// filtered before it reaches the adjacency. This is propagation's own
    /// bounded scan; it deliberately does not reuse the graph-neighbourhood
    /// loaders (which drag whole-graph BFS into a bounded module).
    async fn load_propagation_edges(
        self: &Arc<Self>,
        request: &PropagateRequest,
    ) -> Result<Vec<(String, String, Option<f64>)>> {
        let (sql, weight_alias) = self.edge_scan_sql(&request.edge_source)?;
        let batches = self
            .context()
            .sql(&sql)
            .await
            .map_err(|e| JammiError::Other(format!("propagate: edge scan: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("propagate: edge collect: {e}")))?;

        let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
        if total > request.max_rows {
            return Err(JammiError::Config(format!(
                "propagate_embeddings: edge set of {total} rows exceeds the ceiling of {}; \
                 reduce the graph or raise PropagateRequest::max_rows",
                request.max_rows
            )));
        }

        let mut edges = Vec::with_capacity(total);
        for batch in &batches {
            let src = string_column(batch, "_src")?;
            let dst = string_column(batch, "_dst")?;
            let weight = weight_alias
                .map(|_| f64_column(batch, "_weight"))
                .transpose()?
                .flatten();
            for i in 0..batch.num_rows() {
                if src.is_null(i) || dst.is_null(i) {
                    continue;
                }
                let w = weight
                    .as_ref()
                    .filter(|a| !a.is_null(i))
                    .map(|a| a.value(i));
                edges.push((src.value(i).to_string(), dst.value(i).to_string(), w));
            }
        }
        Ok(edges)
    }

    /// Build the tenant-scoped edge-scan SQL for a [`EdgeSourceRef`], projecting
    /// canonical `_src`/`_dst`[/`_weight`] aliases. Returns the SQL plus the
    /// weight alias when the source carries one.
    fn edge_scan_sql(&self, edge_source: &EdgeSourceRef) -> Result<(String, Option<&'static str>)> {
        match edge_source {
            EdgeSourceRef::NeighborGraph { table_name } => {
                // S9 neighbor_graph tables register as the bare literal
                // `jammi.{name}`; `similarity` is the edge weight.
                Ok((
                    format!(
                        "SELECT arrow_cast(src, 'Utf8') AS _src, \
                         arrow_cast(dst, 'Utf8') AS _dst, \
                         arrow_cast(similarity, 'Float64') AS _weight \
                         FROM \"jammi.{table_name}\""
                    ),
                    Some("_weight"),
                ))
            }
            EdgeSourceRef::Registered {
                source_id,
                src_column,
                dst_column,
                weight_column,
                ..
            } => {
                let table = self.find_table_name(source_id)?;
                let mut projection = format!(
                    "arrow_cast(\"{src_column}\", 'Utf8') AS _src, \
                     arrow_cast(\"{dst_column}\", 'Utf8') AS _dst"
                );
                let weight_alias = weight_column.as_ref().map(|w| {
                    projection.push_str(&format!(", arrow_cast(\"{w}\", 'Float64') AS _weight"));
                    "_weight"
                });
                Ok((
                    format!("SELECT {projection} FROM \"{source_id}\".public.\"{table}\""),
                    weight_alias,
                ))
            }
        }
    }
}

/// A node's `_row_id` and its `f64` feature vector — the unit the hop loop folds.
struct NodeFeatures {
    row_id: String,
    features: Vec<f64>,
}

/// The two degree-normalised hops [`InferenceSession::propagate_normalized`]
/// runs — the internal counterpart of the public
/// [`PropagationWeighting`] arms that need a degree fold (`EdgeSimilarity` has
/// its own per-edge-weight path and never reaches here, so this enum makes that
/// impossible state unrepresentable rather than guarding it at run time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Normalization {
    /// `D̃^{-1}Ã` — the plain mean of a node and its neighbours, no degree fold.
    RandomWalk,
    /// `D̃^{-1/2}(A+I)D̃^{-1/2}` — the symmetric fold on both sides.
    Symmetric,
}

/// Push the adjacency pairs an edge `src→dst` contributes, oriented by
/// `direction` (an undirected edge contributes both ways). The `group` is the
/// node whose new vector aggregates the `neighbour`.
fn push_oriented_pairs(
    pairs: &mut Vec<AdjacencyPair>,
    src: &str,
    dst: &str,
    direction: EdgeDirection,
) {
    for (group, neighbour) in oriented_endpoints(src, dst, direction) {
        pairs.push(AdjacencyPair {
            group: group.to_string(),
            neighbour: neighbour.to_string(),
        });
    }
}

/// The `(group, neighbour)` endpoints an edge `src→dst` contributes under
/// `direction`. `Out`: `dst` is `src`'s out-neighbour, so `src` aggregates
/// `dst`. `In`: the reverse. `Undirected`: both.
fn oriented_endpoints<'a>(
    src: &'a str,
    dst: &'a str,
    direction: EdgeDirection,
) -> Vec<(&'a str, &'a str)> {
    match direction {
        EdgeDirection::Out => vec![(src, dst)],
        EdgeDirection::In => vec![(dst, src)],
        EdgeDirection::Undirected => vec![(src, dst), (dst, src)],
    }
}

/// The augmented degree d̃(v) of every group — the count of pairs it groups
/// over, which (with self-loops present) is `deg + 1`.
fn augmented_degrees(pairs: &[AdjacencyPair]) -> HashMap<String, f64> {
    let mut degree: HashMap<String, f64> = HashMap::new();
    for pair in pairs {
        *degree.entry(pair.group.clone()).or_insert(0.0) += 1.0;
    }
    degree
}

/// Fold each group's neighbour vectors into one aggregate, in `f64`, over the
/// fixed neighbour order the `pairs` slice already carries (sorted by
/// `(group, neighbour)` at load time).
///
/// The element-wise reduction is the engine's shared [`fold_vectors_in_order`]
/// — the *same* operator the P3 vector-aggregation UDAF folds through, applied
/// here over a canonical `(group, neighbour)` order. Propagation imposes that
/// fixed order deliberately: it buys the byte-identical determinism the streaming
/// SQL UDAF cannot guarantee across partitionings (`f64` `+` is non-associative
/// and a parallel plan fixes neither the fold nor the merge order), so the per-hop
/// aggregation runs in Rust over the **bounded** collected adjacency rather than
/// routing through the SQL aggregate.
///
/// `mean` selects [`VectorReduce::Mean`] (the random-walk `D̃^{-1}Ã`, dividing by
/// each group's neighbour count); otherwise [`VectorReduce::Sum`] returns the raw
/// sum for the caller's symmetric degree fold. A neighbour absent from `scaled`
/// (null / unmatched) contributes nothing.
fn aggregate_neighbours(
    pairs: &[AdjacencyPair],
    scaled: &HashMap<String, Vec<f64>>,
    dimensions: usize,
    mean: bool,
) -> HashMap<String, Vec<f64>> {
    let reduce = if mean {
        VectorReduce::Mean
    } else {
        VectorReduce::Sum
    };
    // Gather each group's present-neighbour vectors in the fixed pair order, then
    // reduce that ordered sequence through the shared lane operator. The pairs are
    // pre-sorted by (group, neighbour), so consecutive pairs share a group and the
    // per-group sequence is canonical.
    let mut grouped: HashMap<&str, Vec<&[f64]>> = HashMap::new();
    for pair in pairs {
        if let Some(x) = scaled.get(&pair.neighbour) {
            grouped.entry(pair.group.as_str()).or_default().push(x);
        }
    }
    grouped
        .into_iter()
        .map(|(group, vectors)| {
            (
                group.to_string(),
                fold_vectors_in_order(vectors, reduce, dimensions),
            )
        })
        .collect()
}

/// `1/√d̃(id)`, or `1.0` for a node absent from the degree map (it has only its
/// self-loop, d̃ = 1 — defensive; the self-loop guarantees presence).
fn inv_sqrt_degree(degree: &HashMap<String, f64>, id: &str) -> f64 {
    degree.get(id).map_or(1.0, |&d| 1.0 / d.sqrt())
}

/// The APPNP teleport mix `α·X⁽⁰⁾ + (1−α)·neighbour_term`. With no neighbour
/// term (an isolated node whose only contribution is itself, already folded into
/// the aggregate via the self-loop) the node keeps `X⁽⁰⁾`.
fn teleport_mix(initial: &[f64], neighbour_term: Option<Vec<f64>>, alpha: f64) -> Vec<f64> {
    match neighbour_term {
        Some(agg) => initial
            .iter()
            .zip(agg)
            .map(|(x0, agg)| alpha * x0 + (1.0 - alpha) * agg)
            .collect(),
        None => initial.to_vec(),
    }
}

/// Assemble the per-key output rows and their dimensionality from the hop
/// history `[X⁽⁰⁾, X⁽¹⁾, …, X⁽ᴷ⁾]`.
///
/// [`PropagationOutput::Final`] keeps the last block `X⁽ᴷ⁾`, cast to `f32` (a
/// `d`-dim table in the input's space). [`PropagationOutput::JumpingKnowledge`]
/// L2-normalises **every** per-hop block and concatenates them, so the
/// large-norm raw block does not dominate cosine search; the output is
/// `(K+1)·d`-dimensional, in its own space.
fn assemble_output(
    initial: &[NodeFeatures],
    history: &[HashMap<String, Vec<f64>>],
    output: PropagationOutput,
    dimensions: usize,
) -> (Vec<(String, Vec<f32>)>, usize) {
    // The history always carries X⁽⁰⁾ plus one block per hop; the last is X⁽ᴷ⁾.
    let blocks: &[HashMap<String, Vec<f64>>] = match output {
        PropagationOutput::Final => &history[history.len() - 1..],
        PropagationOutput::JumpingKnowledge => history,
    };
    let rows: Vec<(String, Vec<f32>)> = initial
        .iter()
        .map(|node| {
            let vector: Vec<f32> = match output {
                // The single final block, cast to f32 (no per-block normalise).
                PropagationOutput::Final => block_for(&blocks[0], node)
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
                // Every block, each L2-normalised, concatenated.
                PropagationOutput::JumpingKnowledge => blocks
                    .iter()
                    .flat_map(|block| l2_normalize(block_for(block, node)))
                    .collect(),
            };
            (node.row_id.clone(), vector)
        })
        .collect();
    let out_dim = dimensions * blocks.len();
    (rows, out_dim)
}

/// A node's vector in `block`, falling back to its `X⁽⁰⁾` if the block is missing
/// it (defensive — every block carries every node once self-loops are present).
fn block_for<'a>(block: &'a HashMap<String, Vec<f64>>, node: &'a NodeFeatures) -> &'a [f64] {
    block
        .get(&node.row_id)
        .map_or(&node.features, Vec::as_slice)
}

/// L2-normalise a block to unit norm, cast to `f32`. A zero vector stays zero
/// (no division by zero).
fn l2_normalize(block: &[f64]) -> Vec<f32> {
    let norm = block.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return block.iter().map(|&x| x as f32).collect();
    }
    block.iter().map(|&x| (x / norm) as f32).collect()
}

/// Read the `_row_id` column of an embedding batch into owned strings, casting
/// from whatever Arrow string family the Parquet scan returns.
fn read_row_id_column(batch: &RecordBatch, table: &str) -> Result<Vec<String>> {
    let col = batch
        .column_by_name("_row_id")
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: "_row_id".into(),
            expected: "Utf8".into(),
            actual: "missing".into(),
        })?;
    let utf8 =
        arrow::compute::cast(col.as_ref(), &DataType::Utf8).map_err(|e| JammiError::Schema {
            table: table.to_string(),
            column: "_row_id".into(),
            expected: "Utf8".into(),
            actual: format!("{:?} (cast failed: {e})", col.data_type()),
        })?;
    let arr = utf8
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: "_row_id".into(),
            expected: "Utf8".into(),
            actual: format!("{:?}", col.data_type()),
        })?;
    Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect())
}

/// Downcast a batch column to `StringArray`, casting from any Arrow string
/// family the scan returns (Parquet returns `Utf8View` on DataFusion 52+).
fn string_column(batch: &RecordBatch, name: &str) -> Result<StringArray> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| JammiError::Other(format!("propagate: column '{name}' is missing")))?;
    let utf8 = arrow::compute::cast(col.as_ref(), &DataType::Utf8)
        .map_err(|e| JammiError::Other(format!("propagate: column '{name}' is not Utf8: {e}")))?;
    utf8.as_any()
        .downcast_ref::<StringArray>()
        .cloned()
        .ok_or_else(|| JammiError::Other(format!("propagate: column '{name}' is not Utf8")))
}

/// Downcast (the cast-to-`Float64`) weight column, `None` when absent.
fn f64_column(batch: &RecordBatch, name: &str) -> Result<Option<arrow::array::Float64Array>> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    col.as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .cloned()
        .map(Some)
        .ok_or_else(|| {
            JammiError::Other(format!("propagate: weight column '{name}' is not Float64"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(spec: &[(&str, &str)]) -> Vec<AdjacencyPair> {
        spec.iter()
            .map(|(g, n)| AdjacencyPair {
                group: g.to_string(),
                neighbour: n.to_string(),
            })
            .collect()
    }

    #[test]
    fn oriented_endpoints_respect_direction() {
        assert_eq!(
            oriented_endpoints("a", "b", EdgeDirection::Out),
            [("a", "b")]
        );
        assert_eq!(
            oriented_endpoints("a", "b", EdgeDirection::In),
            [("b", "a")]
        );
        assert_eq!(
            oriented_endpoints("a", "b", EdgeDirection::Undirected),
            [("a", "b"), ("b", "a")]
        );
    }

    #[test]
    fn augmented_degree_counts_self_loop() {
        // a has neighbour b plus its self-loop → d̃ = 2.
        let p = pairs(&[("a", "b"), ("a", "a"), ("b", "b")]);
        let deg = augmented_degrees(&p);
        assert_eq!(deg["a"], 2.0);
        assert_eq!(deg["b"], 1.0);
        assert_eq!(inv_sqrt_degree(&deg, "a"), 1.0 / 2.0_f64.sqrt());
        // An absent node defaults to d̃ = 1.
        assert_eq!(inv_sqrt_degree(&deg, "missing"), 1.0);
    }

    #[test]
    fn teleport_mix_blends_initial_and_neighbour() {
        let x0 = vec![1.0, 0.0];
        let agg = vec![0.0, 2.0];
        let mixed = teleport_mix(&x0, Some(agg), 0.25);
        assert_eq!(mixed, vec![0.25, 1.5]);
    }

    #[test]
    fn teleport_mix_isolated_node_keeps_initial() {
        let x0 = vec![3.0, -1.0];
        assert_eq!(teleport_mix(&x0, None, 0.1), x0);
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let n = l2_normalize(&[3.0, 4.0]);
        let norm = (n[0] * n[0] + n[1] * n[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "block normalised to unit norm");
    }

    #[test]
    fn l2_normalize_zero_stays_zero() {
        assert_eq!(l2_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    /// A hop history `[X⁽⁰⁾, X⁽¹⁾, X⁽²⁾]` for one node, used to exercise the
    /// output assembly with multiple per-hop blocks.
    fn history_for(node: &str, blocks: &[Vec<f64>]) -> Vec<HashMap<String, Vec<f64>>> {
        blocks
            .iter()
            .map(|b| {
                let mut m = HashMap::new();
                m.insert(node.to_string(), b.clone());
                m
            })
            .collect()
    }

    #[test]
    fn jumping_knowledge_concats_every_normalized_hop_block() {
        let initial = vec![NodeFeatures {
            row_id: "a".into(),
            features: vec![3.0, 4.0],
        }];
        // Three blocks (X⁰, X¹, X²) → (K+1)·d = 3·2 = 6 dims.
        let history = history_for("a", &[vec![3.0, 4.0], vec![0.0, 5.0], vec![6.0, 8.0]]);
        let (rows, dim) =
            assemble_output(&initial, &history, PropagationOutput::JumpingKnowledge, 2);
        assert_eq!(dim, 6, "JK concatenates every per-hop block: (K+1)·d");
        let v = &rows[0].1;
        assert_eq!(v.len(), 6);
        // Each 2-dim block is independently unit-normalised before concat.
        for chunk in v.chunks(2) {
            let norm = (chunk[0] * chunk[0] + chunk[1] * chunk[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "each block is unit-normalised");
        }
    }

    #[test]
    fn final_output_keeps_dimension_and_last_block() {
        let initial = vec![NodeFeatures {
            row_id: "a".into(),
            features: vec![1.0, 2.0],
        }];
        let history = history_for("a", &[vec![1.0, 2.0], vec![5.0, 6.0]]);
        let (rows, dim) = assemble_output(&initial, &history, PropagationOutput::Final, 2);
        assert_eq!(dim, 2, "Final keeps d");
        assert_eq!(rows[0].1, vec![5.0, 6.0], "Final keeps the last block X⁽ᴷ⁾");
    }

    #[test]
    fn request_defaults_are_oversmoothing_safe() {
        let req = PropagateRequest::new(
            "src",
            EdgeSourceRef::NeighborGraph {
                table_name: "g".into(),
            },
        );
        assert_eq!(req.weighting, PropagationWeighting::DegreeNormalized);
        assert_eq!(req.alpha, DEFAULT_TELEPORT_ALPHA);
        assert_eq!(req.hops, DEFAULT_PROPAGATE_HOPS);
        assert_eq!(req.hop_cap, DEFAULT_HOP_CAP);
        assert_eq!(req.output, PropagationOutput::Final);
    }

    #[test]
    fn effective_hops_clamps_to_cap() {
        let req = PropagateRequest::new(
            "src",
            EdgeSourceRef::NeighborGraph {
                table_name: "g".into(),
            },
        )
        .with_hops(10);
        assert_eq!(req.effective_hops(), DEFAULT_HOP_CAP);
        assert_eq!(
            PropagateRequest::new(
                "src",
                EdgeSourceRef::NeighborGraph {
                    table_name: "g".into()
                }
            )
            .with_hops(0)
            .effective_hops(),
            1,
            "hops floor at 1"
        );
    }
}
