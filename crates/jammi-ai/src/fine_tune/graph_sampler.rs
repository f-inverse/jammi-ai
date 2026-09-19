//! Graph → contrastive-pair sampler: turns a graph (node text + edge table)
//! into the `(anchor, positive, [hard_negative])` text pairs the existing
//! fine-tune trainer consumes. This is node2vec/DeepWalk realised as a data
//! loader — it samples the graph, it does **not** author a GNN (no message
//! passing, no learned aggregation).
//!
//! # What it produces
//!
//! For each node, biased random walks (node2vec) over the edge table yield
//! co-walk *positives* — nodes that are graph-close at walk length `L`, not just
//! 1-hop neighbours (`L = 1` is the degenerate 1-hop special case). Each
//! `(anchor, positive)` pair drives the existing in-batch-negative objective
//! (MNRL), and the sampler additionally mines **structure-aware hard
//! negatives** — nodes that are reachable but *outside* the anchor's k-hop
//! neighbourhood, i.e. the sibling / near-but-not-neighbour discrimination dense
//! retrieval needs.
//!
//! # Text-bearing precondition (typed, not assumed)
//!
//! The encoder needs text input, so every graph node carries its text in
//! [`TextNode::text`] — there is no constructor for a node without text. An edge
//! endpoint that does not resolve to a [`TextNode`] is a typed error, never a
//! silent skip. Pure-vector nodes belong to graph propagation, not this sampler.
//!
//! # Circularity — the load-bearing caveat
//!
//! Training on **similarity edges** ([`EdgeProvenance::Similarity`]) largely
//! re-learns the base embedding metric: the edges were *drawn by* that metric,
//! so co-walk positives over them mostly restate "things the model already
//! thinks are close" → little new signal, a degenerate feedback loop. Genuine
//! gain comes from **declared / external edges** ([`EdgeProvenance::Declared`])
//! — hierarchy, crosswalk, citation, confirmed pairs — structure the base metric
//! does **not** already encode. The sampler tracks each edge's provenance and
//! [`GraphSampler::has_declared_supervision`] reports whether any declared edge
//! is present; similarity-only supervision is a weak bootstrap, never the sole
//! supervision. See the cookbook page for the full evaluation protocol.
//!
//! # Composition with graph propagation
//!
//! Fine-tuning the head on graph pairs and propagating embeddings over the
//! graph (`crate::pipeline::graph_propagation`) both encode homophily;
//! stacking them naively double-counts the same smoothing. The recommended
//! order is **propagate → fine-tune** — the SGC/APPNP decoupling — not two
//! independent smoothing passes.
//!
//! # References
//! - Perozzi et al. 2014, *DeepWalk*: <https://arxiv.org/abs/1403.6652>
//! - Grover & Leskovec 2016, *node2vec* (biased walks; p/q): <https://arxiv.org/abs/1607.00653>
//! - Karpukhin et al. 2020, *DPR* (in-batch + hard negatives): <https://arxiv.org/abs/2004.04906>
//! - Zhou et al. 2024, *Mitigating False Negatives in Dense Retrieval*: <https://arxiv.org/abs/2401.00165>

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use jammi_db::error::{JammiError, Result};

use crate::pipeline::graph_neighbourhood::{Adjacency, SplitMix64};

/// Where an edge came from — the load-bearing distinction for the circularity
/// contract. Tracked per edge so the sampler can report whether the supervision
/// carries any signal the base metric does not already encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeProvenance {
    /// External / declared structure: hierarchy, crosswalk, citation, a
    /// coder-confirmed pair. Independent of the base embedding metric, so it can
    /// teach the model something new. This is the supervision graph fine-tuning
    /// is *for*.
    Declared,
    /// A k-NN similarity edge: drawn *by* the base embedding metric. Training
    /// on it largely re-learns that metric (the degenerate feedback loop), so it
    /// is acceptable only as a weak bootstrap, never the sole supervision.
    Similarity,
}

/// One graph node, carrying the text the encoder will embed. The text is a
/// required field with no text-free constructor — the "nodes must be
/// text-bearing" precondition is the type, not a runtime assumption.
#[derive(Debug, Clone)]
pub struct TextNode {
    /// Stable node id; edge endpoints (`src`/`dst`) join to this.
    pub id: String,
    /// The node's text, encoded by the base model. Required.
    pub text: String,
}

impl TextNode {
    /// Construct a text-bearing node. There is deliberately no constructor that
    /// omits the text — a graph node without text cannot be fed to the encoder,
    /// so the type makes that unrepresentable.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

/// One directed edge of the supervision graph, with its provenance. Endpoints
/// are node ids that must resolve to [`TextNode`]s in the node set.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node id.
    pub src: String,
    /// Destination node id.
    pub dst: String,
    /// Where the edge came from (declared vs similarity).
    pub provenance: EdgeProvenance,
}

impl GraphEdge {
    /// A declared / external edge (hierarchy, crosswalk, citation, confirmed
    /// pair) — the supervision that carries genuine signal.
    pub fn declared(src: impl Into<String>, dst: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            provenance: EdgeProvenance::Declared,
        }
    }

    /// A similarity edge — a weak bootstrap only (see the circularity
    /// caveat in the module docs).
    pub fn similarity(src: impl Into<String>, dst: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            provenance: EdgeProvenance::Similarity,
        }
    }
}

/// Reorder an in-memory node/edge set into `GRAPH_READ_ORDER_RULE_V1`'s
/// order (nodes ascending by `(id, text)`, edges ascending by `(src, dst)`)
/// — the SAME full-tuple ascending order the real worker's two `ORDER BY`
/// scans (`fine_tune::worker::materialize_graph_training_set`) produce for
/// free from the source's own row order.
///
/// A reference sampler that builds a [`GraphSampler`] directly from an
/// in-memory fixture (to compare bytes against a real job over the SAME
/// node/edge SET) must call this FIRST — [`GraphSampler::build`] samples in
/// exactly the order it is handed, so a fixture in its own raw declaration
/// order (not already `(id, text)`/`(src, dst)`-sorted) samples different
/// bytes than the real job even though the SET is identical. Every
/// in-crate and `jammi-server` reference sampler calls this ONE function
/// rather than re-deriving the sort inline — a comparison against a
/// reference that skips it, or sorts differently, is silently vacuous
/// (two physical layouts of the identical set must sample identically;
/// this function is what makes a fixture's layout agree with a real
/// scan's).
pub fn sort_into_graph_read_order(nodes: &mut [TextNode], edges: &mut [GraphEdge]) {
    nodes.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.text.cmp(&b.text)));
    edges.sort_by(|a, b| a.src.cmp(&b.src).then_with(|| a.dst.cmp(&b.dst)));
}

/// The two sources and their column bindings a graph fine-tune reads from: a
/// node-text source (id + text) and an edge source (src + dst), plus the
/// provenance every edge in the edge source carries. Bundled so the
/// graph-fine-tune entry point takes one typed argument instead of a long
/// positional list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphFineTuneSources {
    /// Catalog source holding the node text.
    pub node_source: String,
    /// Column in `node_source` holding the node id (edge endpoints join to it).
    pub id_column: String,
    /// Column in `node_source` holding the text the encoder embeds.
    pub text_column: String,
    /// Catalog source holding the edges.
    pub edge_source: String,
    /// Column in `edge_source` holding the edge source endpoint.
    pub src_column: String,
    /// Column in `edge_source` holding the edge destination endpoint.
    pub dst_column: String,
    /// Provenance every edge in `edge_source` carries (the edge source is
    /// homogeneous in origin — declared *or* similarity, not mixed).
    pub provenance: EdgeProvenance,
}

/// Sampling configuration: node2vec walk knobs plus the structure-aware
/// negative-sampling knobs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSampleConfig {
    /// Walk length `L`. The positive for an anchor is drawn from the nodes the
    /// walk visits, so `L` controls how far up the graph the positive can be.
    /// `L = 1` is the degenerate 1-hop-only case; `L > 1` captures
    /// higher-order / community structure. Must be `>= 1`.
    pub walk_length: usize,
    /// Number of walks started from each node. More walks = more `(anchor,
    /// positive)` pairs per node and lower-variance positive sampling. Must be
    /// `>= 1`.
    pub walks_per_node: usize,
    /// node2vec return parameter `p`. A large `p` discourages immediately
    /// stepping back to the previous node (less backtracking); a small `p`
    /// encourages it. Must be `> 0`.
    pub return_p: f64,
    /// node2vec in-out parameter `q`. `q < 1` biases the walk outward (DFS-like,
    /// exploring further communities); `q > 1` keeps it local (BFS-like,
    /// structural equivalence). Must be `> 0`.
    pub in_out_q: f64,
    /// Number of structure-aware hard negatives to mine per `(anchor, positive)`
    /// pair. `0` emits `Pairs` (in-batch negatives only); `>= 1` emits `Triplet`
    /// with explicit hard negatives appended to the in-batch pool.
    pub hard_negatives: usize,
    /// Hops of the anchor's neighbourhood to exclude from its negative pool. A
    /// sampled "non-neighbour" within this radius is likely a *missing* edge (a
    /// true positive) → a false-negative gradient, so it is excluded. Must
    /// be `>= 1` so at least the direct neighbours are protected.
    pub exclude_hops: usize,
    /// Minimum number of distinct negatives (in-batch + mined) the training
    /// signal must be able to draw from, guarding against contrastive collapse
    /// on a tiny / disconnected graph. The sampler refuses to produce a dataset
    /// whose node count cannot supply this many negatives. Must be `>= 1`.
    pub min_negatives: usize,
    /// Seed for the walk/negative RNG so a run is reproducible.
    pub seed: u64,
}

impl Default for GraphSampleConfig {
    fn default() -> Self {
        Self {
            walk_length: 4,
            walks_per_node: 2,
            return_p: 1.0,
            in_out_q: 1.0,
            hard_negatives: 1,
            exclude_hops: 1,
            min_negatives: 1,
            seed: 0,
        }
    }
}

impl GraphSampleConfig {
    /// Validate the knobs; returns an error naming the first invalid field.
    pub fn validate(&self) -> Result<()> {
        if self.walk_length == 0 {
            return Err(JammiError::FineTune(
                "graph walk_length must be >= 1".into(),
            ));
        }
        if self.walks_per_node == 0 {
            return Err(JammiError::FineTune(
                "graph walks_per_node must be >= 1".into(),
            ));
        }
        // `is_finite` checked FIRST, short-circuiting `<= 0.0` (family
        // D/F): `NaN <= 0.0` is `false`, so a bare `self.return_p <= 0.0`
        // would silently PASS a NaN return_p/in_out_q through validation —
        // a confidently-wrong number, not a refused one — and
        // `biased_choice`'s `1.0 / return_p` weight computation would then
        // divide by NaN. Checking `!is_finite()` FIRST means a NaN or
        // Infinity is refused there, before `<= 0.0` (a comparison NaN
        // would make silently vacuous) ever runs.
        if !self.return_p.is_finite() || self.return_p <= 0.0 {
            return Err(JammiError::FineTune(format!(
                "graph return_p must be > 0 and finite (was {})",
                self.return_p
            )));
        }
        if !self.in_out_q.is_finite() || self.in_out_q <= 0.0 {
            return Err(JammiError::FineTune(format!(
                "graph in_out_q must be > 0 and finite (was {})",
                self.in_out_q
            )));
        }
        if self.exclude_hops == 0 {
            return Err(JammiError::FineTune(
                "graph exclude_hops must be >= 1 (the false-negative guard)".into(),
            ));
        }
        if self.min_negatives == 0 {
            return Err(JammiError::FineTune(
                "graph min_negatives must be >= 1 (collapse guard)".into(),
            ));
        }
        Ok(())
    }
}

/// One sampled training row: an anchor, its co-walk positive, and the
/// structure-mined hard negatives (empty when `hard_negatives == 0`).
#[derive(Debug, Clone, PartialEq)]
pub struct SampledPair {
    /// Anchor node text.
    pub anchor: String,
    /// Co-walk positive node text.
    pub positive: String,
    /// Structure-aware hard-negative node texts (k-hop-but-not-neighbour).
    pub hard_negatives: Vec<String>,
}

/// Samples a graph into contrastive text pairs via biased random walks.
///
/// Holds the resolved node texts and a directed adjacency (forward, for walks)
/// plus an undirected adjacency (for k-hop neighbourhood exclusion — a missing
/// edge is undirected by intent). Built once, then [`Self::sample`] draws the
/// pairs.
pub struct GraphSampler {
    /// Node id → text.
    nodes: HashMap<String, String>,
    /// Stable node-id ordering, so sampling iterates deterministically.
    node_ids: Vec<String>,
    /// Forward adjacency for walks: node → its out-neighbours.
    out_adj: HashMap<String, Vec<String>>,
    /// Undirected adjacency for the k-hop false-negative guard and the node2vec
    /// p/q bias: an edge in either direction makes the endpoints neighbours. The
    /// shared [`Adjacency`] whose bounded BFS the k-hop exclusion walks — one
    /// bounded-expansion core, shared with graph context assembly.
    undirected: Adjacency,
    /// Whether any edge is declared (vs similarity-only) — the circularity
    /// signal, computed at build time.
    has_declared: bool,
    config: GraphSampleConfig,
}

impl GraphSampler {
    /// Build a sampler from a node set and an edge set.
    ///
    /// Every edge endpoint must resolve to a [`TextNode`] — an unresolved
    /// endpoint is a typed error (the text-bearing precondition), never a silent
    /// skip. The node set must be non-empty and large enough that the negative
    /// pool can satisfy `config.min_negatives` (the collapse guard).
    pub fn build(
        nodes: Vec<TextNode>,
        edges: Vec<GraphEdge>,
        config: GraphSampleConfig,
    ) -> Result<Self> {
        config.validate()?;

        if nodes.is_empty() {
            return Err(JammiError::FineTune(
                "graph sampler needs at least one text-bearing node".into(),
            ));
        }

        // The node id is the sole ordering key a caller's
        // "ORDER BY id" scan can offer, so a duplicate id makes that key-only
        // order non-total — which of the two rows' text a tie-broken scan
        // would keep is undefined, and a silent "last write wins" would make
        // the sampled text a function of SCAN ORDER, not of the node SET.
        // Refused here, at build, naming the id — never a silent overwrite.
        let mut node_map = HashMap::with_capacity(nodes.len());
        let mut node_ids = Vec::with_capacity(nodes.len());
        for node in nodes {
            if node_map.contains_key(&node.id) {
                return Err(JammiError::FineTune(format!(
                    "duplicate node id '{}': every node id must appear exactly once in the \
                     node source; a duplicate id makes the read-order rule non-total (which \
                     row's text survives a tie-broken scan is undefined) — de-duplicate the \
                     source or make the ids unique",
                    node.id
                )));
            }
            node_ids.push(node.id.clone());
            node_map.insert(node.id, node.text);
        }

        // The negative pool for an anchor is "every node except the anchor's
        // excluded neighbourhood", so the absolute ceiling is `node_count - 1`.
        // If that cannot reach `min_negatives` the dataset would train against
        // too few negatives to avoid collapse — refuse up front.
        let max_possible_negatives = node_ids.len().saturating_sub(1);
        if max_possible_negatives < config.min_negatives {
            return Err(JammiError::FineTune(format!(
                "graph has {} nodes but min_negatives is {}: the negative pool \
                 (node_count - 1) cannot reach the minimum; lower min_negatives \
                 or supply more nodes",
                node_ids.len(),
                config.min_negatives
            )));
        }

        let mut out_adj: HashMap<String, Vec<String>> = HashMap::new();
        let mut undirected = Adjacency::new();
        let mut has_declared = false;
        for edge in &edges {
            for (which, endpoint) in [("src", &edge.src), ("dst", &edge.dst)] {
                if !node_map.contains_key(endpoint) {
                    return Err(JammiError::FineTune(format!(
                        "edge {which} endpoint '{endpoint}' has no text-bearing node; \
                         every edge endpoint must resolve to a TextNode (graph \
                         fine-tune nodes must be text-bearing — pure-vector nodes \
                         belong to graph propagation)"
                    )));
                }
            }
            if edge.provenance == EdgeProvenance::Declared {
                has_declared = true;
            }
            out_adj
                .entry(edge.src.clone())
                .or_default()
                .push(edge.dst.clone());
            // Undirected neighbourhood for the false-negative guard: both
            // directions count as adjacency.
            undirected.add_edge(edge.src.clone(), edge.dst.clone());
            undirected.add_edge(edge.dst.clone(), edge.src.clone());
        }

        Ok(Self {
            nodes: node_map,
            node_ids,
            out_adj,
            undirected,
            has_declared,
            config,
        })
    }

    /// Whether the graph carries any declared / external edge. `false` means the
    /// supervision is similarity-only — a weak bootstrap that largely
    /// re-learns the base metric (the circularity caveat). Consumers should
    /// treat similarity-only supervision as a bootstrap, never the sole signal.
    pub fn has_declared_supervision(&self) -> bool {
        self.has_declared
    }

    /// The sample config this sampler was built with: the
    /// authority `TrainingDataLoader::from_graph` reads `hard_negatives`
    /// from to decide `pairs` vs `triplet` — a CONFIG decision,
    /// never re-derived from which negatives the first sampled pair happens
    /// to carry (the first pair alone can be a misleading representative of
    /// the whole set).
    pub fn config(&self) -> &GraphSampleConfig {
        &self.config
    }

    /// A byte estimate for what this sampler holds RESIDENT for random
    /// access over its whole lifetime: every node id +
    /// text, the forward adjacency (`out_adj`, `biased_walk`'s own table),
    /// and the undirected adjacency (`k_hop_neighbourhood`'s bounded BFS,
    /// `biased_choice`'s p/q lookup). This is the shape `reconstruct_graph_
    /// loader`'s `training_set_graph_sample`-named `MemoryConsumer`
    /// reservation is sized against — the resident set node2vec's random
    /// access needs is O(|V|+|E|), unlike a training-set stream's
    /// per-chunk residency.
    ///
    /// Every string's own bytes PLUS the real, measured per-`String` and
    /// per-`HashMap`/`Vec`-slot struct overhead
    /// (`std::mem::size_of::<String>()`/`size_of::<usize>()`) the allocator
    /// actually pays beyond the text — a text-only floor (bytes alone, no
    /// overhead) silently under-reserves by roughly 2-3x the `Vec<String>`/
    /// `HashMap<String, _>` shapes this type holds; `size_of` is a
    /// compiler-verified platform constant, not a made-up multiplier. Still
    /// rounds UP, not an exact allocator accounting (no load-factor/
    /// capacity-growth headroom).
    pub fn resident_bytes(&self) -> usize {
        const STRING_OVERHEAD: usize = std::mem::size_of::<String>();
        const SLOT_OVERHEAD: usize = std::mem::size_of::<usize>();
        let nodes_bytes: usize = self
            .nodes
            .iter()
            .map(|(k, v)| k.len() + v.len() + 2 * STRING_OVERHEAD + SLOT_OVERHEAD)
            .sum();
        let node_ids_bytes: usize = self
            .node_ids
            .iter()
            .map(|s| s.len() + STRING_OVERHEAD)
            .sum();
        let out_adj_bytes: usize = self
            .out_adj
            .iter()
            .map(|(k, v)| {
                k.len()
                    + STRING_OVERHEAD
                    + SLOT_OVERHEAD
                    + v.iter()
                        .map(|s| s.len() + STRING_OVERHEAD + SLOT_OVERHEAD)
                        .sum::<usize>()
            })
            .sum();
        nodes_bytes + node_ids_bytes + out_adj_bytes + self.undirected.resident_bytes()
    }

    /// Sample the graph into `(anchor, positive, [hard_negative])` text rows,
    /// calling `emit` once per row AS IT IS PRODUCED — never building a
    /// `Vec<SampledPair>` of the whole output: a caller
    /// materialising the output (`fine_tune::worker::
    /// materialize_graph_training_set`) streams straight into its own
    /// bounded batch buffer, so peak residency is the sampler's own O(|V|+
    /// |E|) resident set (covered by [`Self::resident_bytes`]'s named
    /// reservation) plus ONE bounded chunk, never a second full-output copy.
    ///
    /// For each node, run `walks_per_node` biased walks of length `walk_length`;
    /// each distinct node visited after the start becomes one positive for that
    /// anchor. Per pair, mine `hard_negatives` structure-aware negatives drawn
    /// from outside the anchor's `exclude_hops`-hop neighbourhood (the
    /// false-negative guard). A node with no out-edges contributes no pairs (it
    /// has no graph structure to learn from). `emit` returning `Err` stops
    /// sampling immediately and propagates that error.
    pub fn sample_into(&self, mut emit: impl FnMut(SampledPair) -> Result<()>) -> Result<()> {
        let mut rng = SplitMix64::new(self.config.seed);
        let mut any = false;

        for start in &self.node_ids {
            // The anchor's protected neighbourhood: excluded from its negatives.
            let excluded = self.k_hop_neighbourhood(start);

            for _ in 0..self.config.walks_per_node {
                let walk = self.biased_walk(start, &mut rng);
                // Distinct co-walk nodes (excluding the anchor itself) are the
                // positives. Dedup keeps the pair set from being dominated by
                // self-loops on dense walks.
                let mut seen = HashSet::new();
                for visited in walk.iter().skip(1) {
                    if visited == start || !seen.insert(visited.clone()) {
                        continue;
                    }
                    let negatives = self.sample_negatives(start, visited, &excluded, &mut rng);
                    // An empty negative pool under a config
                    // that DEMANDS hard negatives is refused HERE, at sample
                    // time, naming the anchor — never emitted as a row with
                    // no negatives for `from_graph`'s later, generic
                    // `ok_or_else` to reject with no anchor context (a star
                    // graph's hub anchor can have its whole candidate pool
                    // excluded).
                    if self.config.hard_negatives > 0 && negatives.is_empty() {
                        return Err(JammiError::FineTune(format!(
                            "graph node '{start}' has an empty negative pool under \
                             exclude_hops={}: every candidate is within the anchor's excluded \
                             neighbourhood, so no structure-aware hard negative could be mined \
                             for this (anchor, positive) pair — lower hard_negatives, widen the \
                             graph, or reduce exclude_hops",
                            self.config.exclude_hops
                        )));
                    }
                    any = true;
                    emit(SampledPair {
                        anchor: self.text_of(start)?,
                        positive: self.text_of(visited)?,
                        hard_negatives: negatives
                            .iter()
                            .map(|id| self.text_of(id))
                            .collect::<Result<Vec<_>>>()?,
                    })?;
                }
            }
        }

        if !any {
            return Err(JammiError::FineTune(
                "graph sampling produced no pairs: every node is isolated (no \
                 out-edges). A graph with no edges carries no structure to learn."
                    .into(),
            ));
        }

        Ok(())
    }

    /// The whole-output convenience wrapper over [`Self::sample_into`] — for
    /// a caller with no reason to stream (tests; any future caller with a
    /// small enough graph that a `Vec<SampledPair>` is a non-issue). The
    /// production materialisation path
    /// (`fine_tune::worker::materialize_graph_training_set`) calls
    /// [`Self::sample_into`] directly and never this method, so its own
    /// residency is never charged the whole-output copy this convenience
    /// wrapper makes.
    pub fn sample(&self) -> Result<Vec<SampledPair>> {
        let mut pairs = Vec::new();
        self.sample_into(|pair| {
            pairs.push(pair);
            Ok(())
        })?;
        Ok(pairs)
    }

    /// One biased (node2vec) random walk of length `walk_length` starting at
    /// `start`. Returns the visited node ids including the start. The transition
    /// from `cur` (having come from `prev`) reweights each candidate `next` by
    /// node2vec's `α`: `1/p` if `next == prev` (return), `1` if `next` is also a
    /// neighbour of `prev` (distance 1, stay local), `1/q` otherwise (distance 2,
    /// explore outward).
    fn biased_walk(&self, start: &str, rng: &mut SplitMix64) -> Vec<String> {
        let mut walk = Vec::with_capacity(self.config.walk_length + 1);
        walk.push(start.to_string());

        // `prev` is the node visited before `cur`; `None` on the first step,
        // where there is no previous node to define the p/q bias.
        let mut prev: Option<String> = None;
        let mut cur = start.to_string();
        for _ in 0..self.config.walk_length {
            let Some(neighbours) = self.out_adj.get(&cur) else {
                break;
            };
            if neighbours.is_empty() {
                break;
            }
            let next = match &prev {
                None => neighbours[rng.below(neighbours.len())].clone(),
                Some(prev_id) => self.biased_choice(prev_id, neighbours, rng),
            };
            walk.push(next.clone());
            prev = Some(cur);
            cur = next;
        }
        walk
    }

    /// node2vec transition: pick `next` from `cur`'s neighbours, reweighted by
    /// the unnormalised `α(prev, next)`. Uses weighted reservoir-free roulette
    /// over the small neighbour list.
    fn biased_choice(&self, prev: &str, neighbours: &[String], rng: &mut SplitMix64) -> String {
        let prev_neighbours = self.undirected.neighbours(prev);
        let weights: Vec<f64> = neighbours
            .iter()
            .map(|next| {
                if next == prev {
                    // Return to the previous node.
                    1.0 / self.config.return_p
                } else if prev_neighbours.iter().any(|n| n == next) {
                    // `next` is also adjacent to `prev`: distance 1, stay local.
                    1.0
                } else {
                    // Distance 2: explore outward.
                    1.0 / self.config.in_out_q
                }
            })
            .collect();

        let total: f64 = weights.iter().sum();
        // All weights are strictly positive (p, q > 0), so total > 0.
        let mut target = rng.next_f64() * total;
        for (next, w) in neighbours.iter().zip(&weights) {
            target -= w;
            if target <= 0.0 {
                return next.clone();
            }
        }
        // Floating-point residue: fall back to the last candidate.
        neighbours[neighbours.len() - 1].clone()
    }

    /// The set of node ids within `exclude_hops` undirected hops of `start`
    /// (including `start`). These are excluded from `start`'s negative pool
    /// because a sampled "non-neighbour" inside this radius is most likely a
    /// missing edge — a true positive that would supply a false-negative
    /// gradient if used as a negative.
    ///
    /// Delegates to the shared [`Adjacency::bounded_frontier`] with the
    /// exclusion bounds (include the start, no fan-out so the whole
    /// neighbourhood is enumerated exactly) — the same bounded BFS the context
    /// gather drives, never a second implementation.
    fn k_hop_neighbourhood(&self, start: &str) -> HashSet<String> {
        self.undirected
            .bounded_frontier(start, self.config.exclude_hops, None, true, 0)
            .keys
            .into_iter()
            .collect()
    }

    /// Mine up to `hard_negatives` structure-aware negatives for the `(anchor,
    /// positive)` pair: nodes drawn uniformly from outside the anchor's
    /// excluded k-hop neighbourhood (and excluding the positive). These are the
    /// "reachable but not a neighbour" siblings that sharpen the discrimination,
    /// while the k-hop exclusion keeps a likely-missing-edge node out of the
    /// pool. Returns fewer than requested only when the safe pool is smaller.
    fn sample_negatives(
        &self,
        anchor: &str,
        positive: &str,
        excluded: &HashSet<String>,
        rng: &mut SplitMix64,
    ) -> Vec<String> {
        if self.config.hard_negatives == 0 {
            return Vec::new();
        }
        // Candidates: every node not in the anchor's protected neighbourhood and
        // not the positive. Iterated over the stable id order for determinism.
        let candidates: Vec<&String> = self
            .node_ids
            .iter()
            .filter(|id| id.as_str() != positive && !excluded.contains(*id))
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }

        let want = self.config.hard_negatives.min(candidates.len());
        let mut chosen = Vec::with_capacity(want);
        let mut used = HashSet::new();
        // Bounded sampling-without-replacement: try a few draws, then fall back
        // to a linear scan so we never loop unbounded on a small candidate set.
        let max_draws = want.saturating_mul(4).max(want);
        for _ in 0..max_draws {
            if chosen.len() == want {
                break;
            }
            let pick = candidates[rng.below(candidates.len())];
            if used.insert(pick.clone()) {
                chosen.push(pick.clone());
            }
        }
        if chosen.len() < want {
            for cand in &candidates {
                if chosen.len() == want {
                    break;
                }
                if used.insert((*cand).clone()) {
                    chosen.push((*cand).clone());
                }
            }
        }
        debug_assert!(
            chosen.iter().all(|n| n != anchor),
            "anchor is in its own excluded neighbourhood, never a negative"
        );
        chosen
    }

    /// Resolve a node id to its text; an unresolved id is a typed error (it
    /// should be impossible after `build`'s endpoint check, but the text-bearing
    /// contract is enforced, never assumed).
    fn text_of(&self, id: &str) -> Result<String> {
        self.nodes.get(id).cloned().ok_or_else(|| {
            JammiError::FineTune(format!("graph node '{id}' has no resolvable text"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A homophilous two-community graph: ids `a0..a3` form one clique, `b0..b3`
    /// another, joined by a single bridge edge. Declared edges throughout.
    fn two_community_graph() -> (Vec<TextNode>, Vec<GraphEdge>) {
        let ids = ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3"];
        let nodes = ids
            .iter()
            .map(|id| TextNode::new(*id, format!("text of {id}")))
            .collect();
        let mut edges = Vec::new();
        // Two cliques (undirected, so add both directions).
        for clique in [["a0", "a1", "a2", "a3"], ["b0", "b1", "b2", "b3"]] {
            for i in 0..clique.len() {
                for j in 0..clique.len() {
                    if i != j {
                        edges.push(GraphEdge::declared(clique[i], clique[j]));
                    }
                }
            }
        }
        // A single bridge.
        edges.push(GraphEdge::declared("a0", "b0"));
        edges.push(GraphEdge::declared("b0", "a0"));
        (nodes, edges)
    }

    #[test]
    fn config_rejects_zero_walk_length_and_exclude_hops() {
        assert!(GraphSampleConfig {
            walk_length: 0,
            ..GraphSampleConfig::default()
        }
        .validate()
        .is_err());
        assert!(GraphSampleConfig {
            exclude_hops: 0,
            ..GraphSampleConfig::default()
        }
        .validate()
        .is_err());
        assert!(GraphSampleConfig {
            min_negatives: 0,
            ..GraphSampleConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn build_rejects_dangling_edge_endpoint() {
        // Edge points at a node id with no TextNode → the text-bearing contract
        // is violated, a typed error, not a silent skip.
        let nodes = vec![TextNode::new("a", "alpha"), TextNode::new("b", "beta")];
        let edges = vec![GraphEdge::declared("a", "ghost")];
        let Err(err) = GraphSampler::build(nodes, edges, GraphSampleConfig::default()) else {
            panic!("dangling endpoint must be rejected");
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("text-bearing") && msg.contains("ghost"),
            "error should name the dangling endpoint and the precondition: {msg}"
        );
    }

    #[test]
    fn build_rejects_duplicate_node_id() {
        // A duplicate node id is refused, typed, naming the id — never a
        // silent "last text wins", which would make the sampled text a
        // function of scan order rather than of the node SET.
        let nodes = vec![
            TextNode::new("n0005", "FIRST text of node 5"),
            TextNode::new("n0006", "text of node 6"),
            TextNode::new("n0005", "SECOND text of node 5"),
        ];
        let edges = vec![GraphEdge::declared("n0005", "n0006")];
        let Err(err) = GraphSampler::build(nodes, edges, GraphSampleConfig::default()) else {
            panic!("duplicate node id must be rejected");
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("duplicate node id") && msg.contains("n0005"),
            "error should name the duplicated id: {msg}"
        );
    }

    #[test]
    fn build_keeps_duplicate_edges_asymmetry_with_duplicate_nodes() {
        // The asymmetry: a duplicate EDGE is benign (the projected
        // edge tuple carries no third field that can differ, so keeping both
        // copies is well-defined and output-affecting only in count, never in
        // content) while a duplicate NODE id is not (rejected above). Two
        // identical declared edges must both be accepted and both counted in
        // the adjacency the walk draws from.
        let nodes = vec![TextNode::new("a", "alpha"), TextNode::new("b", "beta")];
        let edges = vec![GraphEdge::declared("a", "b"), GraphEdge::declared("a", "b")];
        let sampler = GraphSampler::build(nodes, edges, GraphSampleConfig::default()).unwrap();
        assert_eq!(sampler.out_adj.get("a").map(|v| v.len()), Some(2));
    }

    #[test]
    fn build_rejects_too_few_nodes_for_min_negatives() {
        // 2 nodes → negative pool ceiling is 1; min_negatives 3 is unreachable.
        let nodes = vec![TextNode::new("a", "alpha"), TextNode::new("b", "beta")];
        let edges = vec![GraphEdge::declared("a", "b")];
        let cfg = GraphSampleConfig {
            min_negatives: 3,
            ..GraphSampleConfig::default()
        };
        let Err(err) = GraphSampler::build(nodes, edges, cfg) else {
            panic!("too-few-nodes must be rejected");
        };
        assert!(format!("{err}").contains("min_negatives"));
    }

    /// `NaN <= 0.0` is `false`, so a
    /// naive `<= 0.0` bound would silently let a NaN `return_p`/`in_out_q`
    /// pass validation. `validate` must refuse it, named, same as any other
    /// invalid value — never a confidently-wrong bias reaching
    /// `biased_choice`'s `1.0 / return_p`.
    #[test]
    fn validate_refuses_non_finite_return_p_and_in_out_q() {
        let nan_return_p = GraphSampleConfig {
            return_p: f64::NAN,
            ..GraphSampleConfig::default()
        };
        let err = nan_return_p.validate().unwrap_err();
        assert!(format!("{err}").contains("return_p"), "{err}");

        let inf_in_out_q = GraphSampleConfig {
            in_out_q: f64::INFINITY,
            ..GraphSampleConfig::default()
        };
        let err = inf_in_out_q.validate().unwrap_err();
        assert!(format!("{err}").contains("in_out_q"), "{err}");

        let neg_in_out_q = GraphSampleConfig {
            in_out_q: -1.0,
            ..GraphSampleConfig::default()
        };
        assert!(neg_in_out_q.validate().is_err());

        // The positive, finite default must still validate.
        assert!(GraphSampleConfig::default().validate().is_ok());
    }

    #[test]
    fn declared_vs_similarity_provenance_is_tracked() {
        // The circularity signal: a declared-edge graph reports declared
        // supervision; a similarity-only graph does not.
        let (nodes, _) = two_community_graph();
        let sim_edges = vec![
            GraphEdge::similarity("a0", "a1"),
            GraphEdge::similarity("a1", "a0"),
        ];
        let sampler =
            GraphSampler::build(nodes.clone(), sim_edges, GraphSampleConfig::default()).unwrap();
        assert!(
            !sampler.has_declared_supervision(),
            "similarity-only graph must report no declared supervision (the \
             circularity bootstrap-only case)"
        );

        let (nodes, declared_edges) = two_community_graph();
        let sampler =
            GraphSampler::build(nodes, declared_edges, GraphSampleConfig::default()).unwrap();
        assert!(
            sampler.has_declared_supervision(),
            "declared-edge graph must report declared supervision (genuine signal)"
        );
    }

    #[test]
    fn walk_positives_stay_in_community() {
        // On a homophilous graph, biased-walk positives for an `a`-node should be
        // overwhelmingly other `a`-nodes (same community), demonstrating the
        // structural property that fine-tuning will pull neighbours together.
        let (nodes, edges) = two_community_graph();
        let cfg = GraphSampleConfig {
            walk_length: 3,
            walks_per_node: 8,
            hard_negatives: 0,
            seed: 42,
            ..GraphSampleConfig::default()
        };
        let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
        let pairs = sampler.sample().unwrap();

        let a_anchor_pairs: Vec<_> = pairs
            .iter()
            .filter(|p| p.anchor.contains("text of a"))
            .collect();
        assert!(!a_anchor_pairs.is_empty(), "should have a-anchored pairs");
        let same_community = a_anchor_pairs
            .iter()
            .filter(|p| p.positive.contains("text of a"))
            .count();
        // The bridge to b0 means a few cross-community positives are expected,
        // but the strong majority must stay in-community.
        let ratio = same_community as f64 / a_anchor_pairs.len() as f64;
        assert!(
            ratio > 0.7,
            "walk positives should stay mostly in-community, got {ratio}"
        );
    }

    #[test]
    fn higher_order_positives_beat_one_hop_reach() {
        // A path a0 - a1 - a2 - a3 (directed forward). With L=1 the only positive
        // for a0 is a1; with L=3 the walk can reach a2 and a3. This is the
        // node2vec higher-order property: L>1 captures structure 1-hop misses.
        let nodes = (0..4)
            .map(|i| TextNode::new(format!("a{i}"), format!("text {i}")))
            .collect::<Vec<_>>();
        let edges = vec![
            GraphEdge::declared("a0", "a1"),
            GraphEdge::declared("a1", "a2"),
            GraphEdge::declared("a2", "a3"),
        ];

        let reach = |walk_length: usize| -> HashSet<String> {
            let cfg = GraphSampleConfig {
                walk_length,
                walks_per_node: 4,
                hard_negatives: 0,
                min_negatives: 1,
                seed: 7,
                ..GraphSampleConfig::default()
            };
            let sampler = GraphSampler::build(nodes.clone(), edges.clone(), cfg).unwrap();
            sampler
                .sample()
                .unwrap()
                .into_iter()
                .filter(|p| p.anchor == "text 0")
                .map(|p| p.positive)
                .collect()
        };

        let one_hop = reach(1);
        let multi_hop = reach(3);
        assert_eq!(
            one_hop,
            HashSet::from(["text 1".to_string()]),
            "L=1 reaches only the direct neighbour"
        );
        assert!(
            multi_hop.len() > one_hop.len(),
            "L=3 must reach beyond 1-hop (higher-order structure), got {multi_hop:?}"
        );
        assert!(
            multi_hop.contains("text 2") || multi_hop.contains("text 3"),
            "L=3 should reach a2 or a3"
        );
    }

    #[test]
    fn negatives_exclude_k_hop_neighbourhood() {
        // a0's 1-hop neighbourhood (itself, a1, a2, a3 via the clique, plus b0
        // via the bridge) must never appear as a negative for a0.
        let (nodes, edges) = two_community_graph();
        let cfg = GraphSampleConfig {
            walk_length: 2,
            walks_per_node: 4,
            hard_negatives: 3,
            exclude_hops: 1,
            seed: 99,
            ..GraphSampleConfig::default()
        };
        let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
        let pairs = sampler.sample().unwrap();

        // a0's excluded set under 1-hop: a0, a1, a2, a3, b0.
        let excluded_texts: HashSet<&str> = [
            "text of a0",
            "text of a1",
            "text of a2",
            "text of a3",
            "text of b0",
        ]
        .into_iter()
        .collect();
        for pair in pairs.iter().filter(|p| p.anchor == "text of a0") {
            for neg in &pair.hard_negatives {
                assert!(
                    !excluded_texts.contains(neg.as_str()),
                    "negative {neg} is inside a0's excluded 1-hop neighbourhood"
                );
            }
        }
    }

    #[test]
    fn sampling_is_reproducible_from_seed() {
        let (nodes, edges) = two_community_graph();
        let cfg = GraphSampleConfig {
            seed: 1234,
            ..GraphSampleConfig::default()
        };
        let s1 = GraphSampler::build(nodes.clone(), edges.clone(), cfg)
            .unwrap()
            .sample()
            .unwrap();
        let s2 = GraphSampler::build(nodes, edges, cfg)
            .unwrap()
            .sample()
            .unwrap();
        assert_eq!(s1, s2, "same seed must yield the same pairs");
    }

    #[test]
    fn k_hop_uses_the_shared_bounded_frontier() {
        // The fine-tune k-hop exclusion and the graph context gather share ONE
        // bounded BFS (`graph_neighbourhood::Adjacency::bounded_frontier`). This
        // asserts the sampler's `k_hop_neighbourhood` is exactly that shared core
        // driven with the exclusion bounds (undirected, include-start, no
        // fan-out) — the DRY guarantee, not a second implementation.
        use crate::pipeline::graph_neighbourhood::Adjacency;

        let (nodes, edges) = two_community_graph();
        let cfg = GraphSampleConfig {
            exclude_hops: 2,
            ..GraphSampleConfig::default()
        };
        let sampler = GraphSampler::build(nodes, edges.clone(), cfg).unwrap();

        // The same undirected adjacency, built directly, walked by the shared
        // provider with the exclusion bounds.
        let mut adj = Adjacency::new();
        for e in &edges {
            adj.add_edge(e.src.clone(), e.dst.clone());
            adj.add_edge(e.dst.clone(), e.src.clone());
        }
        let via_provider: HashSet<String> = adj
            .bounded_frontier("a0", 2, None, true, 0)
            .keys
            .into_iter()
            .collect();

        assert_eq!(
            sampler.k_hop_neighbourhood("a0"),
            via_provider,
            "k_hop is the shared bounded_frontier with the exclusion bounds — one BFS, two callers"
        );
    }

    #[test]
    fn isolated_graph_is_a_typed_error() {
        // Nodes but no edges → no structure → no pairs → typed error, not silent
        // empty.
        let nodes = vec![
            TextNode::new("a", "alpha"),
            TextNode::new("b", "beta"),
            TextNode::new("c", "gamma"),
        ];
        let sampler = GraphSampler::build(nodes, Vec::new(), GraphSampleConfig::default()).unwrap();
        assert!(sampler.sample().is_err());
    }

    /// A star graph: hub `c0` reaches every spoke directly, so
    /// `exclude_hops=1` excludes c0's WHOLE candidate pool — its negative
    /// pool is empty. Requesting `hard_negatives > 0` over this
    /// graph must refuse AT SAMPLE TIME, naming the hub, never silently emit
    /// a pair with no negatives for a later, generic caller to reject with
    /// no anchor context.
    fn star_graph() -> (Vec<TextNode>, Vec<GraphEdge>) {
        let n = 5;
        let nodes = (0..n)
            .map(|i| TextNode::new(format!("c{i}"), format!("t{i}")))
            .collect();
        let mut edges = Vec::new();
        for i in 1..n {
            edges.push(GraphEdge::declared("c0", format!("c{i}")));
            edges.push(GraphEdge::declared(format!("c{i}"), "c0"));
        }
        (nodes, edges)
    }

    #[test]
    fn sample_refuses_an_empty_negative_pool_when_hard_negatives_are_requested() {
        let (nodes, edges) = star_graph();
        let cfg = GraphSampleConfig {
            hard_negatives: 2,
            exclude_hops: 1,
            min_negatives: 1,
            seed: 1,
            ..GraphSampleConfig::default()
        };
        let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
        let Err(err) = sampler.sample() else {
            panic!("an empty negative pool under hard_negatives > 0 must be refused");
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("c0") && msg.contains("empty negative pool"),
            "the refusal should name the exhausted anchor: {msg}"
        );
    }

    /// The SAME star graph with `hard_negatives = 0` never mines a
    /// pool at all, so it succeeds — the format decision (`pairs`) is
    /// a config fact, not a property this graph's structure could violate.
    #[test]
    fn sample_succeeds_on_the_same_star_when_hard_negatives_is_zero() {
        let (nodes, edges) = star_graph();
        let cfg = GraphSampleConfig {
            hard_negatives: 0,
            exclude_hops: 1,
            min_negatives: 1,
            seed: 1,
            ..GraphSampleConfig::default()
        };
        let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
        let pairs = sampler.sample().unwrap();
        assert!(pairs.iter().all(|p| p.hard_negatives.is_empty()));
    }

    /// `resident_bytes` is a REAL measurement, not a
    /// placeholder — it moves when the resident text does, and a graph with
    /// more edges (a wider `out_adj`/`undirected`) reports MORE resident
    /// bytes than an isolated-looking one with identical node text alone.
    #[test]
    fn resident_bytes_reflects_node_text_and_adjacency_width() {
        let nodes = vec![
            TextNode::new("a", "alpha text"),
            TextNode::new("b", "beta text"),
            TextNode::new("c", "gamma text"),
        ];
        let cfg = GraphSampleConfig::default();
        let few_edges =
            GraphSampler::build(nodes.clone(), vec![GraphEdge::declared("a", "b")], cfg).unwrap();
        let more_edges = GraphSampler::build(
            nodes,
            vec![
                GraphEdge::declared("a", "b"),
                GraphEdge::declared("b", "c"),
                GraphEdge::declared("c", "a"),
            ],
            cfg,
        )
        .unwrap();

        let node_text_bytes = "a".len()
            + "alpha text".len()
            + "b".len()
            + "beta text".len()
            + "c".len()
            + "gamma text".len();
        assert!(
            few_edges.resident_bytes() >= node_text_bytes,
            "resident_bytes must be at least the node id+text bytes alone"
        );
        assert!(
            more_edges.resident_bytes() > few_edges.resident_bytes(),
            "a wider adjacency must report MORE resident bytes over the identical node text"
        );
    }
}
