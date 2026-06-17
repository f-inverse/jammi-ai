//! Similarity-graph materialization: the k-nearest-neighbour graph of an
//! existing embedding table, written as a queryable edge relation.
//!
//! For every row of an embedding table this finds the row's `k` nearest
//! neighbours *within the same table* and emits one directed edge per
//! `(source, neighbour)` pair. The result is a catalogued, tenant-scoped
//! `result_table` of `kind = neighbor_graph` — adjacency + weight, nothing
//! more. It carries no model, no sidecar index, and no evidence channel; the
//! `similarity` weight is a plain column.
//!
//! This is for *global-structure* work — clustering, near-duplicate
//! detection, connected components, graph-aware training-data generation —
//! where the whole edge set is consumed as a durable artifact. For "neighbours
//! of *these* rows" compose `search` instead; this pipeline does not exist for
//! the per-query case.
//!
//! # Construction strategy
//!
//! Two drivers stand behind the [`NeighborGraphStrategy`] trait:
//!
//! - **Index-assisted** (default when a sidecar index is present): each row's
//!   vector queries the HNSW index for `k + 1` neighbours and drops the
//!   self-hit. The output is an *approximate* kNN graph — HNSW recall is below
//!   100%, so some true neighbours are missed — and it is *non-deterministic*
//!   across runs (insertion order, distance ties). This is the right default
//!   for dedup and clustering; a correctness-sensitive consumer forces the
//!   exact driver.
//! - **Exact**: brute-force over every pair via the same path `search` falls
//!   back to. It is deterministic and complete, at `n²` cost, so it is gated by
//!   an `exact_max_rows` ceiling. A consumer needing reproducible, auditable
//!   edges sets `exact = true`.
//!
//! # Cosine coupling
//!
//! `similarity = 1.0 - cosine_distance`, matching the ANN search path. This is
//! valid because every Jammi index is cosine (`SidecarIndex` hardcodes
//! `MetricKind::Cos`). If Jammi ever adds a non-cosine metric, this mapping
//! must read the index's metric and map per-metric or refuse.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::error::{JammiError, Result};
use jammi_db::index::VectorIndex;
use jammi_db::store::ResultStore;

use crate::session::InferenceSession;

/// Default ceiling on row count for the exact (brute-force) driver. Above this
/// an `exact = true` build refuses rather than run an `n²` pass that would
/// dwarf an index-assisted one; the same ceiling `search`'s fallback respects.
pub const DEFAULT_EXACT_MAX_ROWS: usize = 50_000;

/// Parameters for [`InferenceSession::build_neighbor_graph`].
#[derive(Debug, Clone)]
pub struct BuildNeighborGraph {
    /// Number of nearest neighbours per node (required, `>= 1`).
    pub k: usize,
    /// Drop any edge whose `similarity` is below this floor. `None` keeps all
    /// `k` edges per node.
    pub min_similarity: Option<f32>,
    /// Keep an edge `(a, b)` only when its reverse `(b, a)` also survives —
    /// the standard hubness-suppression filter for clustering/dedup. Default
    /// `false`.
    pub mutual: bool,
    /// Exclude the self-edge `(a, a)`. Default `true`.
    pub self_exclude: bool,
    /// Force the deterministic, complete exact driver. Default `false`
    /// (index-assisted when an index is present). See the module docs for the
    /// approximate/exact contract.
    pub exact: bool,
    /// Row-count ceiling for the exact driver. An `exact` build over a larger
    /// table is refused.
    pub exact_max_rows: usize,
    /// Resolve endpoints to the source key. Today the embedding table's
    /// `_row_id` *is* the stringified `key_column` value, so a resolved
    /// endpoint equals its `_row_id` and the edge table joins directly to
    /// source either way; the flag is retained for the contract and for a
    /// future where the two diverge. Default `true`.
    pub resolve_keys: bool,
}

impl Default for BuildNeighborGraph {
    fn default() -> Self {
        Self {
            k: 10,
            min_similarity: None,
            mutual: false,
            self_exclude: true,
            exact: false,
            exact_max_rows: DEFAULT_EXACT_MAX_ROWS,
            resolve_keys: true,
        }
    }
}

/// One node and its embedding vector, the unit the strategies consume.
pub struct Node {
    /// The node's `_row_id` (the stringified `key_column` value).
    pub row_id: String,
    /// The node's embedding vector.
    pub vector: Vec<f32>,
}

/// A directed, weighted edge before it is filtered and serialized.
struct Edge {
    src: String,
    dst: String,
    rank: i32,
    similarity: f32,
}

/// How to find each node's nearest neighbours. Two drivers implement it
/// (index-assisted and exact); a future NN-Descent driver fits the same shape.
pub trait NeighborGraphStrategy {
    /// Return up to `k` nearest neighbours of `node`, *excluding the node
    /// itself*, as `(neighbour_row_id, cosine_distance)` ordered nearest-first.
    /// The caller turns distance into `similarity` and assigns ranks, so a
    /// driver only owns "which neighbours, in what order".
    fn neighbours(&self, node: &Node, k: usize) -> Result<Vec<(String, f32)>>;

    /// Whether this driver's output is deterministic and complete across runs.
    /// `true` for the exact driver, `false` for index-assisted (HNSW recall +
    /// tie-ordering). Surfaced so callers can assert the contract.
    fn is_exact(&self) -> bool;
}

/// Index-assisted driver: each query goes to the HNSW sidecar index loaded
/// once. Queries `k + 1` and drops the self-hit so a full `k` survive.
struct IndexAssisted {
    index: jammi_db::index::sidecar::SidecarIndex,
}

impl NeighborGraphStrategy for IndexAssisted {
    fn neighbours(&self, node: &Node, k: usize) -> Result<Vec<(String, f32)>> {
        // Query k+1 because the node itself is (almost always) its own nearest
        // hit; drop it so a full k non-self neighbours remain.
        let raw = self.index.search(&node.vector, k + 1)?;
        Ok(drop_self_hit(raw, &node.row_id, k))
    }

    fn is_exact(&self) -> bool {
        false
    }
}

/// Exact driver: brute-force cosine over every node, in memory. Deterministic
/// and complete; gated by the ceiling at the call site.
struct Exact {
    nodes: Arc<Vec<Node>>,
}

impl NeighborGraphStrategy for Exact {
    fn neighbours(&self, node: &Node, k: usize) -> Result<Vec<(String, f32)>> {
        let mut scored: Vec<(String, f32)> = self
            .nodes
            .iter()
            .map(|other| {
                (
                    other.row_id.clone(),
                    jammi_numerics::distance::cosine_distance(&node.vector, &other.vector),
                )
            })
            .collect();
        // Stable, total order: distance ascending, then row_id ascending to
        // break ties — this is what makes the exact driver reproducible.
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        Ok(drop_self_hit(scored, &node.row_id, k))
    }

    fn is_exact(&self) -> bool {
        true
    }
}

/// Drop the single self-hit (a neighbour whose id equals the query node's) and
/// truncate to `k`. Only the first matching self-hit is removed so a genuine
/// duplicate-id collision (which the embedding key contract forbids) cannot
/// silently delete a real neighbour beyond the one self-reference.
fn drop_self_hit(
    mut neighbours: Vec<(String, f32)>,
    self_id: &str,
    k: usize,
) -> Vec<(String, f32)> {
    if let Some(pos) = neighbours.iter().position(|(id, _)| id == self_id) {
        neighbours.remove(pos);
    }
    neighbours.truncate(k);
    neighbours
}

/// The Arrow schema of an edge relation: `src`, `dst`, `rank`, `similarity`.
pub fn edge_table_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("similarity", DataType::Float32, false),
    ]))
}

/// Builds the k-nearest-neighbour edge relation of an embedding table.
pub struct NeighborGraphPipeline<'a> {
    session: &'a InferenceSession,
    result_store: &'a ResultStore,
}

impl<'a> NeighborGraphPipeline<'a> {
    pub fn new(session: &'a InferenceSession, result_store: &'a ResultStore) -> Self {
        Self {
            session,
            result_store,
        }
    }

    /// Materialize the edge relation of `source`'s embedding table.
    ///
    /// Resolves the embedding table through the tenant-scoped catalog path
    /// (the table the bound tenant would `search`), reads its nodes, picks the
    /// exact or index-assisted driver per [`BuildNeighborGraph`], and writes a
    /// new `neighbor_graph` result table through the shared `ResultStore`.
    pub async fn run(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        params: &BuildNeighborGraph,
    ) -> Result<ResultTableRecord> {
        if params.k == 0 {
            return Err(JammiError::Config(
                "build_neighbor_graph requires k >= 1".into(),
            ));
        }

        let source_table = self
            .session
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;

        let nodes = self.read_nodes(&source_table).await?;
        let edges = self.build_edges(&source_table, &nodes, params).await?;
        self.write_edge_table(&source_table, edges, params.k).await
    }

    /// Read every `(_row_id, vector)` pair from the embedding table's Parquet,
    /// in the order the engine scans it. Reuses the registered DataFusion table
    /// so cloud credentials and the tenant-scoped registration are inherited.
    async fn read_nodes(&self, table: &ResultTableRecord) -> Result<Vec<Node>> {
        let batches = self
            .session
            .sql(&format!(
                "SELECT _row_id, vector FROM \"jammi.{}\"",
                table.table_name
            ))
            .await?;

        let mut nodes = Vec::new();
        for batch in &batches {
            let row_ids = read_row_id_column(batch, &table.table_name)?;
            let mut vectors: Vec<Vec<f32>> = Vec::new();
            jammi_db::store::vectors::extend_with_fixed_size_list_f32(
                batch,
                &table.table_name,
                "vector",
                &mut vectors,
            )?;
            for (i, vector) in vectors.into_iter().enumerate() {
                nodes.push(Node {
                    row_id: row_ids[i].clone(),
                    vector,
                });
            }
        }
        Ok(nodes)
    }

    /// Select a driver and run it over every node, applying `self_exclude`,
    /// `min_similarity`, and `mutual` to produce the final edge list.
    async fn build_edges(
        &self,
        table: &ResultTableRecord,
        nodes: &[Node],
        params: &BuildNeighborGraph,
    ) -> Result<Vec<Edge>> {
        let strategy = self.resolve_strategy(table, nodes, params).await?;

        let mut edges: Vec<Edge> = Vec::new();
        for node in nodes {
            // `self_exclude = false` keeps the self-edge: the driver already
            // dropped the self-hit, so ask for one extra and prepend the
            // self-reference at rank 0 → distance 0 → similarity 1.0.
            let want = if params.self_exclude {
                params.k
            } else {
                params.k.saturating_sub(1)
            };
            let mut neighbours = strategy.neighbours(node, want)?;
            if !params.self_exclude {
                neighbours.insert(0, (node.row_id.clone(), 0.0));
            }

            for (rank0, (dst, distance)) in neighbours.into_iter().enumerate() {
                let similarity = 1.0 - distance;
                if let Some(floor) = params.min_similarity {
                    if similarity < floor {
                        continue;
                    }
                }
                edges.push(Edge {
                    src: node.row_id.clone(),
                    dst,
                    rank: (rank0 as i32) + 1,
                    similarity,
                });
            }
        }

        if params.mutual {
            edges = keep_mutual(edges);
        }
        Ok(edges)
    }

    /// Pick the construction driver. The exact driver runs when the caller
    /// forces it or the table has no sidecar index; otherwise the index-assisted
    /// driver loads the index once. An `exact` build over a table larger than
    /// the ceiling is refused rather than silently downgraded.
    async fn resolve_strategy(
        &self,
        table: &ResultTableRecord,
        nodes: &[Node],
        params: &BuildNeighborGraph,
    ) -> Result<Box<dyn NeighborGraphStrategy>> {
        let index = self.result_store.resolve_search_mode(table).await?;

        if params.exact || index.is_none() {
            if params.exact && nodes.len() > params.exact_max_rows {
                return Err(JammiError::Config(format!(
                    "exact neighbor-graph build over {} rows exceeds the ceiling of {}; \
                     run with exact = false to use the index-assisted driver",
                    nodes.len(),
                    params.exact_max_rows
                )));
            }
            return Ok(Box::new(Exact {
                nodes: Arc::new(clone_nodes(nodes)),
            }));
        }

        Ok(Box::new(IndexAssisted {
            index: index.expect("index presence checked above"),
        }))
    }

    /// Write the edge list as a new `neighbor_graph` result table derived from
    /// the source embedding table, then finalize it queryable.
    async fn write_edge_table(
        &self,
        source_table: &ResultTableRecord,
        edges: Vec<Edge>,
        k: usize,
    ) -> Result<ResultTableRecord> {
        // The edge table is a derivation: its `task` rides the source's so the
        // NOT NULL column round-trips, but `kind = NeighborGraph` excludes it
        // from embedding resolution and `create_table` gives it no sidecar.
        let table_info = self
            .result_store
            .create_table(
                &source_table.source_id,
                source_table.task,
                ResultTableKind::NeighborGraph,
                Some(&source_table.table_name),
                NEIGHBOR_GRAPH_MODEL_ID,
                None,
                None,
                None,
            )
            .await?;

        let schema = edge_table_schema();
        let batch = edges_to_batch(edges, schema.clone())?;
        let row_count = batch.num_rows();

        let mut writer = self
            .result_store
            .open_writer(&table_info.parquet_url, schema)
            .await?;
        writer.write_batch(&batch).await?;
        writer.close().await?;

        // The materialization contract: a neighbor-graph derivation invokes no
        // model, so the environment carries the engine version + device with an
        // empty model set; its sole input is the source embedding table, pinned
        // by its immutable content digest (`ResultDigest`).
        let descriptor = jammi_db::store::manifest::ProducingDescriptor::NeighborGraph {
            source_table: source_table.table_name.clone(),
            k,
        };
        let env = jammi_db::store::manifest::MaterializationEnv::new(
            self.session.compute_device(),
            Vec::new(),
        );
        let inputs = vec![self.result_store.result_digest_anchor(source_table).await?];

        self.result_store
            .finalize_with_manifest(
                self.session.context(),
                &table_info.table_name,
                &table_info.parquet_url,
                row_count,
                jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
            )
            .await?;

        self.session
            .catalog()
            .get_result_table(&table_info.table_name)
            .await?
            .ok_or_else(|| {
                JammiError::Catalog(format!(
                    "Neighbor-graph table '{}' not found after finalization",
                    table_info.table_name
                ))
            })
    }
}

/// Read the `_row_id` column of an embedding batch into owned strings.
///
/// DataFusion 52+ returns Parquet string columns as `Utf8View` by default
/// (older versions as `Utf8`/`LargeUtf8`), so the column is cast to `Utf8`
/// before the downcast rather than assuming one Arrow string family.
fn read_row_id_column(batch: &arrow::array::RecordBatch, table: &str) -> Result<Vec<String>> {
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

/// The `model_id` an edge table records. It carries no model, so the column
/// is a fixed marker naming the derivation kind rather than a model id; the
/// source embedding table is recorded separately in `derived_from`. Keeping it
/// short (not the source table name) keeps the generated table name a sane
/// length for the catalog and the storage path.
const NEIGHBOR_GRAPH_MODEL_ID: &str = "neighbor_graph";

/// Shallow-copy the nodes into an owned vec the exact driver can hold behind an
/// `Arc` for the duration of the build.
fn clone_nodes(nodes: &[Node]) -> Vec<Node> {
    nodes
        .iter()
        .map(|n| Node {
            row_id: n.row_id.clone(),
            vector: n.vector.clone(),
        })
        .collect()
}

/// Keep only reciprocal edges: `(a, b)` survives iff `(b, a)` is also present.
/// Rank and similarity are preserved from the forward edge.
fn keep_mutual(edges: Vec<Edge>) -> Vec<Edge> {
    let present: HashSet<(&str, &str)> = edges
        .iter()
        .map(|e| (e.src.as_str(), e.dst.as_str()))
        .collect();
    edges
        .iter()
        .filter(|e| present.contains(&(e.dst.as_str(), e.src.as_str())))
        .map(|e| Edge {
            src: e.src.clone(),
            dst: e.dst.clone(),
            rank: e.rank,
            similarity: e.similarity,
        })
        .collect()
}

/// Serialize the edge list into a single `RecordBatch` matching
/// [`edge_table_schema`].
fn edges_to_batch(edges: Vec<Edge>, schema: SchemaRef) -> Result<RecordBatch> {
    let src: StringArray = edges.iter().map(|e| Some(e.src.as_str())).collect();
    let dst: StringArray = edges.iter().map(|e| Some(e.dst.as_str())).collect();
    let rank: Int32Array = edges.iter().map(|e| Some(e.rank)).collect();
    let similarity: Float32Array = edges.iter().map(|e| Some(e.similarity)).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(src),
            Arc::new(dst),
            Arc::new(rank),
            Arc::new(similarity),
        ],
    )
    .map_err(|e| JammiError::Other(format!("Edge RecordBatch build: {e}")))
}
