//! The synthetic graph every graph workload's engine legs run over, and where
//! they run it: the planted-class graph and its features, the sources it is
//! registered under, a hermetic session at a partition count, the read of a
//! materialised table's rows in key order, and a host — this process, or a
//! fleet a request is submitted into as a job with its sink placed on an
//! executor. `propagate` and `structure` build on this and differ only in the
//! verb they run and the leg they file.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};

use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};

use crate::leg::RanOn;
use crate::plane::PlaneParams;

const NODES_SOURCE: &str = "nodes";
const EDGES_SOURCE: &str = "edges";
/// The model id the synthetic input embedding table is materialised under.
pub const INPUT_MODEL_ID: &str = "synthetic-embed";
const FEATURE_SEED: u64 = 0x00C0_FFEE_0001;

/// Keyed vector rows: a node id and its vector, the shape every leg's inputs
/// and outputs take.
pub(crate) type KeyedRows = Vec<(String, Vec<f32>)>;

/// The unit's input files under `legs_dir/input/<unit>/`: the vectors as
/// `x0.vectors.f32` + `x0.keys.txt` and the undirected edge list as
/// `edges.jsonl`, one `{"src", "dst"}` row per edge.
pub const INPUT_DIR: &str = "input";
/// The input vectors' stem.
pub const X0_STEM: &str = "x0";
/// The input edge list.
pub const EDGES_FILE: &str = "edges.jsonl";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphShape {
    /// Embedding dimensionality of `X⁽⁰⁾` and of the propagated output.
    pub dim: usize,
    /// Classes the graph wires within (a homophilous, variable-degree
    /// bounded-fan-out subgraph per class).
    pub n_classes: usize,
    /// Fan-out cap: node `i` wires to its next `1 + (i mod fan_out)` class-mates,
    /// so the per-node degree *varies* while the edge set stays
    /// `O(nodes · fan_out)`.
    pub fan_out: usize,
}

/// The shape every leg uses unless told otherwise.
pub const DEFAULT_SHAPE: GraphShape = GraphShape {
    dim: 16,
    n_classes: 4,
    fan_out: 4,
};

/// The Numerical-Recipes LCG the rest of the harness uses, for deterministic
/// no-crate synthetic fixture generation.
pub(crate) struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// A uniform draw in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A synthetic node: its id (the embedding key and edge endpoint) and the class
/// that drives both its feature centroid and its intra-class wiring.
pub(crate) struct Node {
    pub(crate) id: String,
    pub(crate) class: usize,
}

/// Build `n_classes · per_class` synthetic nodes — `per_class` nodes in each
/// class, ids `c{class}_{i}`. The class is the homophily signal the propagation
/// smooths over.
pub(crate) fn build_nodes(n_classes: usize, per_class: usize) -> Vec<Node> {
    let mut nodes = Vec::with_capacity(n_classes * per_class);
    for class in 0..n_classes {
        for i in 0..per_class {
            nodes.push(Node {
                id: format!("c{class}_{i}"),
                class,
            });
        }
    }
    nodes
}

/// A deterministic per-class centroid plus per-node noise drawn from the LCG, so
/// raw features are not trivially class-separable and the propagated mean moves
/// them measurably — the same fixture shape the engine's own propagation suite
/// uses, regenerated here from a seeded stream so the digest is reproducible.
fn node_vector(rng: &mut Lcg, class: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let centroid = (((class as f32 + 1.0) * (i as f32 + 1.0)) * 0.7).sin();
            // Per-node noise at the centroid-gap scale, from the seeded stream so
            // the whole vector is reproducible bit-for-bit on any box.
            let noise = (rng.unit() as f32 - 0.5) * 0.9;
            centroid + noise
        })
        .collect()
}

/// Build the `(id, vector)` embedding rows for the nodes, drawing each vector
/// from the seeded feature stream in node order so the draw is reproducible.
pub(crate) fn build_features(nodes: &[Node], dim: usize) -> Vec<(String, Vec<f32>)> {
    let mut rng = Lcg::new(FEATURE_SEED);
    nodes
        .iter()
        .map(|n| (n.id.clone(), node_vector(&mut rng, n.class, dim)))
        .collect()
}

/// Build the within-class bounded-fan-out edges — each node `i` connects to its
/// next `1 + (i mod fan_out)` class-mates cyclically, so the edge set is
/// `O(nodes · fan_out)`, bounded, not the `O(nodes²)` of a clique.
///
/// The per-node *variable* fan-out is load-bearing: a regular graph (every node
/// the same degree) reaches the symmetric-normalised APPNP fixed point in a
/// single hop, so hop count would not move the propagated output and a hop-count
/// regression would not move the digest. Varying the degree by node breaks
/// that regularity, so each additional hop genuinely re-mixes — the digest is
/// then sensitive to the hop count, the degree normalisation, and the
/// `α`-teleport alike. The graph stays homophilous (all edges within class), so
/// the propagated mean still denoises toward the class centroid (a non-trivial
/// transform of `X⁽⁰⁾`, not the identity).
///
/// Returns `(src, dst)` pairs; the edge relation is read undirected so a
/// symmetric `Â` is meaningful (APPNP/SGC assume undirected adjacency). The cyclic
/// offsets `1..=reach` never self-pair (the self-loop is the engine's `Ã = A+I`
/// augmentation, not a declared edge); a `BTreeSet` of the *unordered* pair
/// keys dedups the antipodal/overlap collisions deterministically, so the same
/// graph is emitted on any box.
pub(crate) fn build_edges(nodes: &[Node], fan_out: usize) -> Vec<(String, String)> {
    // Group node ids by class, preserving the node order.
    let mut by_class: HashMap<usize, Vec<&str>> = HashMap::new();
    for n in nodes {
        by_class.entry(n.class).or_default().push(&n.id);
    }
    let mut classes: Vec<usize> = by_class.keys().copied().collect();
    classes.sort_unstable();
    let mut edges = Vec::new();
    for class in classes {
        let members = &by_class[&class];
        let m = members.len();
        if m < 2 {
            continue;
        }
        // Dedup unordered pairs deterministically: a node's varying reach can wrap
        // onto a pair another node already emitted, and an undirected read would
        // otherwise double it. Sorting by the index pair keeps the emit order
        // stable across boxes.
        let mut seen: std::collections::BTreeSet<(usize, usize)> =
            std::collections::BTreeSet::new();
        for i in 0..m {
            // Variable per-node fan-out in `1..=fan_out`, capped at what the class
            // can supply without self-pairing.
            let reach = (1 + (i % fan_out.max(1))).min(m - 1);
            for off in 1..=reach {
                let j = (i + off) % m;
                if i == j {
                    continue;
                }
                let key = if i < j { (i, j) } else { (j, i) };
                seen.insert(key);
            }
        }
        for (i, j) in seen {
            edges.push((members[i].to_string(), members[j].to_string()));
        }
    }
    edges
}

/// Write an arrow batch to a fresh parquet file under `dir` and return its
/// `file://` URL, through the engine's own object-store parquet writer (the same
/// path the recall fixture writer uses) so the file the engine later scans is
/// written exactly as the engine reads it.
async fn write_parquet(
    dir: &std::path::Path,
    name: &str,
    schema: Arc<Schema>,
    batch: RecordBatch,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = dir.join(name);
    let url = StorageUrl::parse(path.to_str().ok_or("fixture path is not valid UTF-8")?)?;
    let registry = StorageRegistry::new();
    let handle = registry.handle_for(&url, None)?;
    let mut writer = ObjectParquetWriter::open(&handle, schema).await?;
    writer.write_batch(&batch).await?;
    writer.close().await?;
    Ok(format!("file://{}", path.to_str().unwrap()))
}

/// The names the synthetic graph is registered under: the constants in a
/// session of this leg's own, suffixed in a fleet's shared catalog.
#[derive(Debug, Clone)]
pub(crate) struct GraphSources {
    pub(crate) nodes: String,
    pub(crate) edges: String,
}

impl GraphSources {
    pub(crate) fn own() -> Self {
        Self {
            nodes: NODES_SOURCE.to_string(),
            edges: EDGES_SOURCE.to_string(),
        }
    }

    #[cfg(feature = "plane")]
    pub(crate) fn unique() -> Self {
        let suffix = crate::capture::unique_suffix();
        Self {
            nodes: format!("{NODES_SOURCE}_{suffix}"),
            edges: format!("{EDGES_SOURCE}_{suffix}"),
        }
    }
}

/// A hermetic `Device::Cpu` session of this leg's own, over a tempdir.
/// `target_partitions` is the DataFusion execution-thread count: the
/// determinism test varies it to exercise the byte-identical-across-partitions
/// contract.
pub(crate) async fn local_session(
    artifact_dir: &std::path::Path,
    target_partitions: usize,
) -> Result<Arc<InferenceSession>, Box<dyn std::error::Error>> {
    // CPU-hermetic config: device −1 forces CPU, the execution-thread count is the
    // partition knob, artifacts land in the tempdir.
    let mut config = JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: GpuConfig {
            device: -1,
            ..Default::default()
        },
        ..Default::default()
    };
    config.engine.execution_threads =
        std::num::NonZeroUsize::new(target_partitions).expect("a positive thread count");
    let session = Arc::new(InferenceSession::new(config).await?);
    session.install_query_functions();
    Ok(session)
}

/// Add the synthetic graph's node and edge relations to `session` as sources: the
/// nodes source (its parquet written under `dir`, `_row_id` and `class`) and
/// the edge relation (`src`, `dst`).
pub(crate) async fn add_graph_sources(
    session: &Arc<InferenceSession>,
    dir: &std::path::Path,
    sources: &GraphSources,
    nodes: &[Node],
    edges: &[(String, String)],
) -> Result<(), Box<dyn std::error::Error>> {
    let node_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("class", DataType::Int64, false),
    ]));
    let node_batch = RecordBatch::try_new(
        Arc::clone(&node_schema),
        vec![
            Arc::new(StringArray::from(
                nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                nodes.iter().map(|n| n.class as i64).collect::<Vec<_>>(),
            )),
        ],
    )?;
    let node_url = write_parquet(dir, "nodes.parquet", node_schema, node_batch).await?;
    session
        .add_source(
            &sources.nodes,
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(StringArray::from(
                edges.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                edges.iter().map(|(_, d)| d.as_str()).collect::<Vec<_>>(),
            )),
        ],
    )?;
    let edge_url = write_parquet(dir, "edges.parquet", edge_schema, edge_batch).await?;
    session
        .add_source(
            &sources.edges,
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;
    Ok(())
}

/// Materialise the synthetic input embedding table over the nodes source,
/// keyed by `_row_id` — the `X⁽⁰⁾` a propagation reads.
pub(crate) async fn materialize_features(
    session: &Arc<InferenceSession>,
    sources: &GraphSources,
    nodes: &[Node],
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let features = build_features(nodes, dim);
    let descriptor = jammi_db::store::manifest::ProducingDescriptor::ContextSet {
        encoder_id: INPUT_MODEL_ID.to_string(),
        source_id: sources.nodes.clone(),
        embedding_table: None,
        candidate_source: jammi_db::store::manifest::ContextCandidateSource::Ann { k: 5 },
        value_columns: Vec::new(),
        aggregator: jammi_db::store::manifest::ContextAggregator::Mean,
        exclude_self: true,
        split: None,
        dimensions: dim,
    };
    let env = jammi_db::store::manifest::MaterializationEnv::without_models();
    let inputs = vec![jammi_db::store::manifest::InputAnchor::unpinned_at_instant(
        &sources.nodes,
        "1970-01-01T00:00:00Z",
    )];
    session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            jammi_db::store::EmbeddingTableSpec {
                source_id: &sources.nodes,
                model_id: INPUT_MODEL_ID,
                derived_from: None,
                dimensions: dim,
                key_column: Some("_row_id"),
                text_columns: None,
            },
            &features,
            jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
            None,
        )
        .await?;
    Ok(())
}

/// Each node's augmented degree `d̃`: its distinct neighbours over the
/// undirected read of `edges`, plus the self-loop — the degree the engine's
/// structural seed scales by.
pub(crate) fn augmented_degrees(
    nodes: &[Node],
    edges: &[(String, String)],
) -> BTreeMap<String, u64> {
    let mut neighbours: BTreeMap<&str, std::collections::BTreeSet<&str>> = nodes
        .iter()
        .map(|n| (n.id.as_str(), std::collections::BTreeSet::new()))
        .collect();
    for (src, dst) in edges {
        if src != dst {
            neighbours
                .entry(src.as_str())
                .or_default()
                .insert(dst.as_str());
            neighbours
                .entry(dst.as_str())
                .or_default()
                .insert(src.as_str());
        }
    }
    neighbours
        .into_iter()
        .map(|(id, set)| (id.to_string(), set.len() as u64 + 1))
        .collect()
}

/// Read a materialised embedding table's `(_row_id, vector)` rows back, in a
/// stable `_row_id`-sorted order so the digest folds the bits in one canonical
/// sequence regardless of scan order.
pub(crate) async fn read_sorted_vectors(
    session: &Arc<InferenceSession>,
    table: &ResultTableRecord,
) -> Result<KeyedRows, Box<dyn std::error::Error>> {
    let batches = session
        .sql(&format!(
            "SELECT _row_id, vector FROM {}",
            jammi_db::store::result_table_relation(&table.table_name)
        ))
        .await?;
    let mut rows: KeyedRows = Vec::new();
    for batch in &batches {
        let ids = arrow::compute::cast(batch.column(0), &DataType::Utf8)?;
        let ids = ids
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("propagated _row_id column did not cast to Utf8")?;
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        jammi_db::store::vectors::extend_with_fixed_size_list_f32(
            batch,
            &table.table_name,
            "vector",
            &mut vectors,
        )?;
        for (i, vector) in vectors.into_iter().enumerate() {
            rows.push((ids.value(i).to_string(), vector));
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(rows)
}

/// The rung an engine leg claims: the verb in this process at one partition,
/// the same plan at `--partitions`, or the same request as a job on a fleet,
/// claimed by the submitter process and its sink placed on an executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum EngineRung {
    Plan,
    PlanPartitioned,
    Placed,
}

impl EngineRung {
    pub fn as_str(self) -> &'static str {
        match self {
            EngineRung::Plan => "plan",
            EngineRung::PlanPartitioned => "plan-partitioned",
            EngineRung::Placed => "placed",
        }
    }

    /// The `target_partitions` of the session this rung plans in;
    /// `partitioned` is the `plan-partitioned` rung's.
    pub fn target_partitions(self, partitioned: usize) -> usize {
        match self {
            EngineRung::Plan | EngineRung::Placed => 1,
            EngineRung::PlanPartitioned => partitioned,
        }
    }
}

/// Where a leg's verb runs: a session of this process's own, or a fleet the
/// request is submitted into as a job.
pub(crate) enum GraphHost {
    InProcess {
        session: Arc<InferenceSession>,
        sources: GraphSources,
    },
    #[cfg(feature = "plane")]
    Placed {
        fleet: crate::plane::fleet::RunningFleet,
        sources: GraphSources,
    },
}

impl GraphHost {
    /// Stand the host up for `rung`: a hermetic session at the rung's
    /// partition count, or a fleet whose compute claims `job_kinds`, labelled
    /// `label`.
    pub(crate) async fn stand_up(
        rung: EngineRung,
        partitions: usize,
        plane: &PlaneParams,
        workload: &str,
        label: &str,
        job_kinds: &'static [&'static str],
        artifact_dir: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match rung {
            EngineRung::Plan | EngineRung::PlanPartitioned => Ok(GraphHost::InProcess {
                session: local_session(artifact_dir, rung.target_partitions(partitions)).await?,
                sources: GraphSources::own(),
            }),
            #[cfg(feature = "plane")]
            EngineRung::Placed => {
                let _ = workload;
                let fleet =
                    crate::plane::fleet::RunningFleet::spawn_placed(plane, label, -1, job_kinds)
                        .await
                        .map_err(|e| e.to_string())?;
                Ok(GraphHost::Placed {
                    fleet,
                    sources: GraphSources::unique(),
                })
            }
            #[cfg(not(feature = "plane"))]
            EngineRung::Placed => {
                let _ = (label, job_kinds);
                Err(plane.refusal(workload, EngineRung::Placed.as_str()))
            }
        }
    }

    pub(crate) fn session(&self) -> &Arc<InferenceSession> {
        match self {
            GraphHost::InProcess { session, .. } => session,
            #[cfg(feature = "plane")]
            GraphHost::Placed { fleet, .. } => &fleet.session,
        }
    }

    pub(crate) fn sources(&self) -> &GraphSources {
        match self {
            GraphHost::InProcess { sources, .. } => sources,
            #[cfg(feature = "plane")]
            GraphHost::Placed { sources, .. } => sources,
        }
    }

    /// Run `spec` as a job on the fleet and return the table it committed;
    /// an in-process host has no fleet to submit to.
    #[cfg(feature = "plane")]
    pub(crate) async fn run_placed(
        &mut self,
        spec: jammi_ai::jobs::JobSpec,
        what: &str,
    ) -> Result<ResultTableRecord, Box<dyn std::error::Error>> {
        let GraphHost::Placed { fleet, .. } = self else {
            return Err(format!("{what}: an in-process host runs no placed job").into());
        };
        let job = fleet.session.enqueue(spec, 0).await?;
        let record = fleet
            .await_completed(&job.job_id, &format!("the placed {what} completes"))
            .await
            .map_err(|e| e.to_string())?;
        let result = record
            .result
            .as_deref()
            .ok_or_else(|| format!("the completed {what} job carries no result"))?;
        let jammi_ai::jobs::JobResult::Table { table, .. } = serde_json::from_str(result)? else {
            return Err(format!("the {what} job's result is not a table").into());
        };
        fleet
            .session
            .catalog()
            .get_result_table(&table)
            .await?
            .ok_or_else(|| format!("the {what} table {table} is not in the catalog").into())
    }

    /// Where the sink that committed `table_name` ran, when it left this
    /// process.
    #[cfg(feature = "plane")]
    pub(crate) async fn ran_on(
        &mut self,
        table_name: &str,
    ) -> Result<Option<RanOn>, Box<dyn std::error::Error>> {
        match self {
            GraphHost::InProcess { .. } => Ok(None),
            GraphHost::Placed { fleet, .. } => {
                use crate::plane::fleet::MemberRole;
                fleet
                    .placed_sink_ran_on(
                        table_name,
                        MemberRole::Submitter,
                        MemberRole::Submitter,
                        MemberRole::Executor,
                    )
                    .await
                    .map(Some)
                    .map_err(|e| e.to_string().into())
            }
        }
    }

    /// Without the plane, no sink leaves this process.
    #[cfg(not(feature = "plane"))]
    pub(crate) async fn ran_on(
        &mut self,
        _table_name: &str,
    ) -> Result<Option<RanOn>, Box<dyn std::error::Error>> {
        Ok(None)
    }
}
