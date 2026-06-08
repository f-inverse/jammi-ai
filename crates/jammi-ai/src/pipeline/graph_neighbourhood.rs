//! Shared, tenant-scoped graph-neighbourhood provider: the one bounded,
//! target-anchored neighbour gather both the fine-tune hard-negative guard
//! (S11) and declared-edge context assembly (S16-G) walk.
//!
//! # The one bounded-expansion core
//!
//! `Adjacency::bounded_frontier` is a breadth-first expansion from a start
//! node, capped at `hops` frontiers, optionally sampling at most `fanout`
//! neighbours per node per hop (GraphSAGE-style). It is the *only* bounded BFS
//! in the engine: the fine-tune sampler's k-hop neighbourhood exclusion drives
//! it with the degenerate bounds (undirected, include-start, no fanout), and the
//! context gather drives it with the full bounds (directional, neighbours-only,
//! seeded fan-out). There is deliberately no second BFS — a divergent
//! implementation would be the "special case per layer" the principles forbid.
//!
//! The walk is **target-anchored and depth/fan-out-bounded** — not a free
//! traversal verb. That boundary is what keeps neighbour gather inside the
//! tenant-scope guarantee (general N-hop traversal failed the two-tenant test
//! and stays retracted): a target's gather runs inside the analyzer-rule scope,
//! so no cross-tenant endpoint is ever materialised.
//!
//! # Determinism
//!
//! Fan-out sampling is seeded-deterministic: the seed is derived from the target
//! key (`seed_for_target`), so the same target reproduces a byte-identical
//! neighbour set across runs. `fanout: None` is exact (no sampling, fully
//! enumerated) and uses no randomness at all.
//!
//! # References
//! - Hamilton et al. 2017, *GraphSAGE* (fixed-size per-hop neighbour sampling):
//!   <https://arxiv.org/abs/1706.02216>
//! - Li et al. 2018, *Deeper Insights into GCNs* (over-smoothing — the hop cap):
//!   <https://arxiv.org/abs/1801.07606>

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Float64Array, RecordBatch, StringArray};
use datafusion::sql::TableReference;

use jammi_db::error::{JammiError, Result};

use crate::session::InferenceSession;

/// A small, fast, self-contained PRNG (SplitMix64) so the seeded fan-out sample
/// and the node2vec walk are reproducible from a seed without pulling a
/// dependency into the data path. Shared by the context gather (this module)
/// and the fine-tune sampler — one PRNG, two callers.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f64` in `[0, 1)`.
    pub(crate) fn next_f64(&mut self) -> f64 {
        // 53 mantissa bits → exact uniform in [0, 1).
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// A uniform index in `[0, n)`. `n` must be `> 0`.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize % n
    }
}

/// Derive a deterministic walk seed from the target key and a base seed, so a
/// gather for a given target reproduces byte-identically (the `exact` /
/// reproducibility contract) and varies across targets. SplitMix64-mixed — never
/// drawn from entropy.
pub(crate) fn seed_for_target(base_seed: u64, target: &str) -> u64 {
    // FNV-1a over the key bytes, then one SplitMix64 round mixed with the base.
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in target.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    SplitMix64::new(hash ^ base_seed).next_u64()
}

/// Which direction the gather follows an edge from the target's perspective.
/// Applied at adjacency-build time (an `Undirected` gather inserts both
/// directions), so the walk itself is direction-agnostic — it simply follows
/// whatever edges the adjacency holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// Follow `src → dst` edges (out-neighbours of the target).
    Out,
    /// Follow `dst → src` edges (in-neighbours of the target).
    In,
    /// Both directions count as adjacency.
    Undirected,
}

/// The result of a bounded gather: the visited neighbour keys in breadth-first
/// order, and whether any node's neighbourhood was fan-out-truncated. The
/// `truncated` flag makes the GraphSAGE cap honest — a truncated neighbourhood
/// is reported, never silently dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GatheredFrontier {
    pub keys: Vec<String>,
    pub truncated: bool,
}

/// A loaded, in-memory adjacency over an edge relation — the one representation
/// both the fine-tune k-hop exclusion and the context gather walk. Directed by
/// construction; an undirected gather inserts both directions at build time, and
/// edge-type / min-weight filtering is applied as edges are added (so the walk
/// itself needs no filter knobs and the adjacency carries only the surviving
/// neighbour keys).
#[derive(Debug, Default)]
pub(crate) struct Adjacency {
    edges: HashMap<String, Vec<String>>,
}

impl Adjacency {
    pub(crate) fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Add a directed edge `src → dst`. For an undirected adjacency the caller
    /// adds both `(src, dst)` and `(dst, src)`.
    pub(crate) fn add_edge(&mut self, src: impl Into<String>, dst: impl Into<String>) {
        self.edges.entry(src.into()).or_default().push(dst.into());
    }

    /// The out-neighbour keys of `node`, or an empty slice when it has none.
    pub(crate) fn neighbours(&self, node: &str) -> &[String] {
        self.edges.get(node).map_or(&[], Vec::as_slice)
    }

    /// Bounded breadth-first gather from `start`, capped at `hops` frontiers.
    ///
    /// At each node, if `fanout` is set and the node has more neighbours than
    /// the cap, at most `fanout` are sampled (seeded-deterministically from
    /// `seed`, processed in stable frontier order so the stream reproduces) and
    /// the truncation is reported; `None` enumerates all neighbours (exact, no
    /// randomness). The start is included in `keys` (first) iff `include_start`.
    /// Iterative frontier accumulator — stack-safe on any depth.
    pub(crate) fn bounded_frontier(
        &self,
        start: &str,
        hops: usize,
        fanout: Option<usize>,
        include_start: bool,
        seed: u64,
    ) -> GatheredFrontier {
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start.to_string());
        let mut order: Vec<String> = Vec::new();
        if include_start {
            order.push(start.to_string());
        }
        let mut frontier = vec![start.to_string()];
        let mut rng = SplitMix64::new(seed);
        let mut truncated = false;

        for _ in 0..hops {
            let mut next = Vec::new();
            for node in &frontier {
                let nbrs = self.neighbours(node);
                let selected = match fanout {
                    Some(f) if nbrs.len() > f => {
                        truncated = true;
                        sample_indices(nbrs.len(), f, &mut rng)
                    }
                    _ => (0..nbrs.len()).collect(),
                };
                for i in selected {
                    let dst = &nbrs[i];
                    if visited.insert(dst.clone()) {
                        order.push(dst.clone());
                        next.push(dst.clone());
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        GatheredFrontier {
            keys: order,
            truncated,
        }
    }
}

/// Pick `k` distinct indices from `[0, n)` via a partial Fisher-Yates over the
/// RNG: deterministic given the RNG state, and order-stable enough that the
/// sampled subset reproduces. `k` must be `<= n`.
fn sample_indices(n: usize, k: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.below(n - i);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

/// The default hard cap on gather depth. Beyond ~2–3 hops neighbour aggregation
/// is Laplacian over-smoothing — the pooled vector washes out, more hops do not
/// add signal (Li et al. 2018). `EdgeGather::hops` is clamped to this.
pub const DEFAULT_HOP_CAP: usize = 3;

/// Which declared edge relation to gather over, tenant-scoped.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeSourceRef {
    /// An S9 `neighbor_graph` result table (`src`/`dst`/`rank`/`similarity`).
    /// `similarity` carries the edge weight; edges are untyped.
    NeighborGraph {
        /// The registered result-table name (resolved as `jammi.{name}`).
        table_name: String,
    },
    /// A registered external edge source (two key columns + optional type /
    /// weight / as-of columns). "Bring your own graph": register the edge table,
    /// then gather over it under tenant scope.
    Registered {
        /// The registered source id (the catalog source holding the edge rows).
        source_id: String,
        /// Column holding the edge's source endpoint.
        src_column: String,
        /// Column holding the edge's destination endpoint.
        dst_column: String,
        /// Optional column holding the edge type (for `edge_types` filtering).
        type_column: Option<String>,
        /// Optional column holding the edge weight (for `min_weight` filtering).
        weight_column: Option<String>,
        /// Optional column carrying the edge's as-of marker. When present and an
        /// [`EdgeGather::as_of`] is supplied, the gather pins to rows at or
        /// before it — the temporal-drift anchor the continual loop reads.
        as_of_column: Option<String>,
    },
}

/// A bounded, target-anchored declared-edge gather: which edge source, how far
/// (`hops`, capped), how wide (`fanout`), which direction, and the optional
/// type / weight / as-of filters. The discipline boundary — depth- and
/// fan-out-bounded, never a free traversal — is what keeps it inside the
/// tenant-scope guarantee.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeGather {
    /// The edge relation to walk.
    pub edge_source: EdgeSourceRef,
    /// Hops of bounded BFS. Defaults 1; clamped to `hop_cap`.
    pub hops: usize,
    /// Per-node per-hop neighbour sample cap (GraphSAGE). `None` enumerates all
    /// (exact, deterministic).
    pub fanout: Option<usize>,
    /// Edge direction the walk follows from the target.
    pub direction: EdgeDirection,
    /// Optional edge-type allow-list. Edges whose type is absent (or untyped
    /// under a typed filter) are not traversed — a filter, never learned.
    pub edge_types: Option<Vec<String>>,
    /// Optional minimum edge weight to traverse.
    pub min_weight: Option<f64>,
    /// Optional as-of pin (used with a registered source's `as_of_column`).
    pub as_of: Option<String>,
    /// The hard depth cap `hops` is clamped to.
    pub hop_cap: usize,
}

impl EdgeGather {
    /// A gather with the bounded defaults: 1 hop, exact (no fan-out), outgoing,
    /// no filters, the default hop cap.
    pub fn new(edge_source: EdgeSourceRef) -> Self {
        Self {
            edge_source,
            hops: 1,
            fanout: None,
            direction: EdgeDirection::Out,
            edge_types: None,
            min_weight: None,
            as_of: None,
            hop_cap: DEFAULT_HOP_CAP,
        }
    }

    /// The depth actually walked: `hops` clamped to `[1, hop_cap]`.
    pub fn effective_hops(&self) -> usize {
        self.hops.clamp(1, self.hop_cap.max(1))
    }
}

/// One loaded edge row, normalised to string endpoints plus optional type and
/// weight — the shape both load paths (S9 table, registered source) produce and
/// [`build_adjacency`] consumes.
struct EdgeRow {
    src: String,
    dst: String,
    edge_type: Option<String>,
    weight: Option<f64>,
}

impl InferenceSession {
    /// Gather the bounded, target-anchored declared-edge neighbour keys of
    /// `target` (excluding the target itself), tenant-scoped.
    ///
    /// Loads the edge relation under tenant scope so a cross-tenant endpoint is
    /// filtered before it reaches the adjacency, builds the direction-/type-/
    /// weight-filtered [`Adjacency`], then runs the one bounded BFS. Sampling is
    /// seeded from the target, so the gather reproduces.
    pub(crate) async fn gather_edge_candidates(
        self: &Arc<Self>,
        gather: &EdgeGather,
        target: &str,
    ) -> Result<GatheredFrontier> {
        let adjacency = self.load_adjacency(gather).await?;
        Ok(adjacency.bounded_frontier(
            target,
            gather.effective_hops(),
            gather.fanout,
            false,
            seed_for_target(0, target),
        ))
    }

    /// Load the declared edge relation into an in-memory [`Adjacency`], applying
    /// direction, `edge_types`, and `min_weight` filters. The load runs through
    /// the generic SQL surface so the tenant-scope analyzer rule injects the
    /// `tenant_id` predicate on the scan.
    async fn load_adjacency(self: &Arc<Self>, gather: &EdgeGather) -> Result<Adjacency> {
        Ok(build_adjacency(self.load_edge_rows(gather).await?, gather))
    }

    /// Load the raw, tenant-scoped edge rows of a gather's edge source — the
    /// shared loader behind both the adjacency build and the homophily
    /// diagnostic. No `edge_types` / `min_weight` filtering here (the diagnostic
    /// reports across *all* types); the gather's filters are applied when the
    /// adjacency is built.
    async fn load_edge_rows(self: &Arc<Self>, gather: &EdgeGather) -> Result<Vec<EdgeRow>> {
        match &gather.edge_source {
            EdgeSourceRef::NeighborGraph { table_name } => {
                self.load_neighbor_graph_edges(table_name).await
            }
            EdgeSourceRef::Registered { .. } => self.load_registered_edges(gather).await,
        }
    }

    /// The per-edge-type **homophily** diagnostic: for each edge type, the rate
    /// at which its endpoints share a label over a labelled split. A declared
    /// edge can be *heterophilous* — connecting dissimilar targets — and naive
    /// neighbour pooling over a heterophilous type degrades rather than helps
    /// (Zhu et al. 2020). This surfaces the fact so a caller can decide; it never
    /// auto-disables a type (the engine transports adjacency, it does not judge
    /// what an edge means).
    ///
    /// `label_source_id` / `label_key_column` / `label_column` name the labelled
    /// relation joined to both endpoints (the same eval surface R1/R2 use); both
    /// the edge scan and the label scan run under tenant scope. Untyped edges are
    /// bucketed under `"(untyped)"`. An edge whose endpoints are not both labelled
    /// is skipped (it carries no agreement signal).
    pub async fn homophily_by_edge_type(
        self: &Arc<Self>,
        gather: &EdgeGather,
        label_source_id: &str,
        label_key_column: &str,
        label_column: &str,
    ) -> Result<HashMap<String, f64>> {
        let rows = self.load_edge_rows(gather).await?;
        let labels = self
            .load_label_map(label_source_id, label_key_column, label_column)
            .await?;

        // Per type: (agreeing endpoint pairs, labelled pairs). Agreement is a
        // categorical label match; a type's rate is agreeing / labelled.
        let mut tally: HashMap<String, (u64, u64)> = HashMap::new();
        for row in &rows {
            let (Some(ls), Some(ld)) = (labels.get(&row.src), labels.get(&row.dst)) else {
                continue;
            };
            let key = row
                .edge_type
                .clone()
                .unwrap_or_else(|| "(untyped)".to_string());
            let entry = tally.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if ls == ld {
                entry.0 += 1;
            }
        }

        Ok(tally
            .into_iter()
            .map(|(t, (agree, total))| (t, agree as f64 / total as f64))
            .collect())
    }

    /// Load a `key → label` map from a labelled relation, tenant-scoped. Labels
    /// are read as `Utf8` (homophily is categorical agreement). A null key or
    /// label drops the row.
    async fn load_label_map(
        self: &Arc<Self>,
        source_id: &str,
        key_column: &str,
        label_column: &str,
    ) -> Result<HashMap<String, String>> {
        let table = self.find_table_name(source_id)?;
        let sql = format!(
            "SELECT arrow_cast(\"{key_column}\", 'Utf8') AS _k, \
             arrow_cast(\"{label_column}\", 'Utf8') AS _v \
             FROM \"{source_id}\".public.\"{table}\""
        );
        let batches = self
            .context()
            .sql(&sql)
            .await
            .map_err(|e| JammiError::Other(format!("Homophily: label scan: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("Homophily: label collect: {e}")))?;

        let mut map = HashMap::new();
        for batch in &batches {
            let keys = string_column(batch, "_k")?;
            let vals = string_column(batch, "_v")?;
            for i in 0..batch.num_rows() {
                if !keys.is_null(i) && !vals.is_null(i) {
                    map.insert(keys.value(i).to_string(), vals.value(i).to_string());
                }
            }
        }
        Ok(map)
    }

    /// Load an S9 `neighbor_graph` result table's edges. The relation is
    /// registered as `jammi.{table_name}` (bare reference so a hyphenated name
    /// is not re-split on the dot); columns `src`/`dst` are the endpoints,
    /// `similarity` the weight.
    async fn load_neighbor_graph_edges(self: &Arc<Self>, table_name: &str) -> Result<Vec<EdgeRow>> {
        let table_ref = TableReference::bare(format!("jammi.{table_name}"));
        let batches = self
            .context()
            .table(table_ref.clone())
            .await
            .map_err(|e| JammiError::Other(format!("Edge load: resolve '{table_ref}': {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("Edge load: collect '{table_ref}': {e}")))?;
        collect_edge_rows(&batches, "src", "dst", None, Some("similarity"))
    }

    /// Load a registered external edge source's edges through the tenant-scoped
    /// SQL surface. The endpoints (and optional type / weight) are cast to a
    /// canonical shape; an as-of pin narrows the scan to rows at or before it.
    async fn load_registered_edges(self: &Arc<Self>, gather: &EdgeGather) -> Result<Vec<EdgeRow>> {
        let EdgeSourceRef::Registered {
            source_id,
            src_column,
            dst_column,
            type_column,
            weight_column,
            as_of_column,
        } = &gather.edge_source
        else {
            return Err(JammiError::Other(
                "load_registered_edges called on a non-registered edge source".into(),
            ));
        };
        let table = self.find_table_name(source_id)?;

        let mut projection = format!(
            "arrow_cast(\"{src_column}\", 'Utf8') AS _src, \
             arrow_cast(\"{dst_column}\", 'Utf8') AS _dst"
        );
        if let Some(t) = type_column {
            projection.push_str(&format!(", arrow_cast(\"{t}\", 'Utf8') AS _etype"));
        }
        if let Some(w) = weight_column {
            projection.push_str(&format!(", arrow_cast(\"{w}\", 'Float64') AS _weight"));
        }

        // The as-of pin is the user's own column predicate; the value is bound
        // through `arrow_cast` comparison, the column name a configured
        // identifier (quoted). When the source declares no as-of column the pin
        // is inapplicable and ignored (transport, not enforcement — the policy
        // is governance).
        let where_clause = match (as_of_column, &gather.as_of) {
            (Some(col), Some(asof)) => format!(
                " WHERE arrow_cast(\"{col}\", 'Utf8') <= '{}'",
                asof.replace('\'', "''")
            ),
            _ => String::new(),
        };

        let sql =
            format!("SELECT {projection} FROM \"{source_id}\".public.\"{table}\"{where_clause}");
        let batches = self
            .context()
            .sql(&sql)
            .await
            .map_err(|e| JammiError::Other(format!("Edge load: scan: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("Edge load: collect: {e}")))?;

        let type_alias = type_column.as_ref().map(|_| "_etype");
        let weight_alias = weight_column.as_ref().map(|_| "_weight");
        collect_edge_rows(&batches, "_src", "_dst", type_alias, weight_alias)
    }
}

/// Build a filtered [`Adjacency`] from loaded edge rows: apply `edge_types` and
/// `min_weight`, then insert each surviving edge per the gather direction (an
/// undirected gather inserts both ways).
fn build_adjacency(rows: Vec<EdgeRow>, gather: &EdgeGather) -> Adjacency {
    let mut adjacency = Adjacency::new();
    for row in rows {
        if let Some(allow) = &gather.edge_types {
            match &row.edge_type {
                Some(t) if allow.iter().any(|a| a == t) => {}
                // A typed allow-list never traverses an untyped or off-list edge.
                _ => continue,
            }
        }
        if let Some(min) = gather.min_weight {
            match row.weight {
                Some(w) if w >= min => {}
                // A min-weight filter never traverses an under-weight or
                // unweighted edge (it cannot be shown to satisfy the bound).
                _ => continue,
            }
        }
        match gather.direction {
            EdgeDirection::Out => adjacency.add_edge(row.src, row.dst),
            EdgeDirection::In => adjacency.add_edge(row.dst, row.src),
            EdgeDirection::Undirected => {
                adjacency.add_edge(row.src.clone(), row.dst.clone());
                adjacency.add_edge(row.dst, row.src);
            }
        }
    }
    adjacency
}

/// Read normalised edge rows from query batches. `src`/`dst` are read as
/// `Utf8`; `type`/`weight` columns are read when their alias is given. A row
/// with a null endpoint is dropped (it carries no edge).
fn collect_edge_rows(
    batches: &[RecordBatch],
    src_col: &str,
    dst_col: &str,
    type_col: Option<&str>,
    weight_col: Option<&str>,
) -> Result<Vec<EdgeRow>> {
    let mut rows = Vec::new();
    for batch in batches {
        let src = string_column(batch, src_col)?;
        let dst = string_column(batch, dst_col)?;
        let etype = type_col.map(|c| string_column(batch, c)).transpose()?;
        let weight = weight_col
            .map(|c| f64_column(batch, c))
            .transpose()?
            .flatten();
        for i in 0..batch.num_rows() {
            if src.is_null(i) || dst.is_null(i) {
                continue;
            }
            rows.push(EdgeRow {
                src: src.value(i).to_string(),
                dst: dst.value(i).to_string(),
                edge_type: etype
                    .as_ref()
                    .filter(|a| !a.is_null(i))
                    .map(|a| a.value(i).to_string()),
                weight: weight
                    .as_ref()
                    .filter(|a| !a.is_null(i))
                    .map(|a| a.value(i)),
            });
        }
    }
    Ok(rows)
}

/// Downcast a batch column to `StringArray`, erroring if it is not `Utf8`.
fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| JammiError::Other(format!("Edge load: column '{name}' is not Utf8")))
}

/// Downcast a batch column to a `Float64Array`, accepting either `Float64` (the
/// cast registered-source weight) or `Float32` (the S9 `similarity`). `None`
/// when the column is absent.
fn f64_column(batch: &RecordBatch, name: &str) -> Result<Option<Float64Array>> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        return Ok(Some(a.clone()));
    }
    if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        let widened: Float64Array = (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i) as f64))
            .collect();
        return Ok(Some(widened));
    }
    Err(JammiError::Other(format!(
        "Edge load: weight column '{name}' is not a float"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an undirected adjacency from id pairs (both directions) — the
    /// fine-tune-style graph.
    fn undirected(pairs: &[(&str, &str)]) -> Adjacency {
        let mut adj = Adjacency::new();
        for (a, b) in pairs {
            adj.add_edge(*a, *b);
            adj.add_edge(*b, *a);
        }
        adj
    }

    /// A star: `c` connected out to `n0..n{n-1}`.
    fn star(n: usize) -> Adjacency {
        let mut adj = Adjacency::new();
        for i in 0..n {
            adj.add_edge("c", format!("n{i}"));
        }
        adj
    }

    #[test]
    fn one_hop_gather_is_exact() {
        // star: c connected to n0..n3.
        let adj = undirected(&[("c", "n0"), ("c", "n1"), ("c", "n2"), ("c", "n3")]);
        let got = adj.bounded_frontier("c", 1, None, false, 0);
        assert!(!got.truncated);
        let set: HashSet<_> = got.keys.into_iter().collect();
        assert_eq!(
            set,
            ["n0", "n1", "n2", "n3"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        );
    }

    #[test]
    fn k_hop_reaches_further_than_one_hop() {
        // path a-b-c-d.
        let adj = undirected(&[("a", "b"), ("b", "c"), ("c", "d")]);
        let one = adj.bounded_frontier("a", 1, None, false, 0).keys;
        let two = adj.bounded_frontier("a", 2, None, false, 0).keys;
        assert_eq!(one, vec!["b".to_string()]);
        assert_eq!(two, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn include_start_keeps_the_seed_first() {
        let adj = undirected(&[("a", "b")]);
        let got = adj.bounded_frontier("a", 1, None, true, 0);
        assert_eq!(got.keys, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn fanout_caps_and_reports_truncation() {
        // c has 10 neighbours; fanout 3 keeps exactly 3 and flags truncation.
        let adj = star(10);
        let got = adj.bounded_frontier("c", 1, Some(3), false, seed_for_target(0, "c"));
        assert_eq!(got.keys.len(), 3, "fanout caps the neighbourhood");
        assert!(got.truncated, "truncation must be reported, never silent");
    }

    #[test]
    fn no_truncation_flag_when_under_fanout() {
        let adj = undirected(&[("c", "n0"), ("c", "n1")]);
        let got = adj.bounded_frontier("c", 1, Some(5), false, 0);
        assert_eq!(got.keys.len(), 2);
        assert!(!got.truncated);
    }

    #[test]
    fn seeded_sample_is_byte_identical_across_runs() {
        let adj = star(20);
        let seed = seed_for_target(42, "c");
        let a = adj.bounded_frontier("c", 1, Some(5), false, seed);
        let b = adj.bounded_frontier("c", 1, Some(5), false, seed);
        assert_eq!(a, b, "same target seed must reproduce the same sample");
    }

    #[test]
    fn exact_mode_uses_no_randomness() {
        // fanout None must enumerate all in stable adjacency order, seed-agnostic.
        let adj = star(6);
        let with_one_seed = adj.bounded_frontier("c", 1, None, false, 1);
        let with_other_seed = adj.bounded_frontier("c", 1, None, false, 999);
        assert_eq!(with_one_seed, with_other_seed);
        assert_eq!(with_one_seed.keys.len(), 6);
    }

    #[test]
    fn direction_is_a_build_time_property() {
        // Directed a → b only. An out-gather from a reaches b; from b reaches
        // nothing. The undirected build would reach both ways.
        let mut directed = Adjacency::new();
        directed.add_edge("a", "b");
        assert_eq!(
            directed.bounded_frontier("a", 1, None, false, 0).keys,
            vec!["b".to_string()]
        );
        assert!(directed
            .bounded_frontier("b", 1, None, false, 0)
            .keys
            .is_empty());
    }

    #[test]
    fn disconnected_start_gathers_nothing() {
        let adj = undirected(&[("x", "y")]);
        let got = adj.bounded_frontier("lonely", 3, None, false, 0);
        assert!(got.keys.is_empty());
        assert!(!got.truncated);
    }
}
