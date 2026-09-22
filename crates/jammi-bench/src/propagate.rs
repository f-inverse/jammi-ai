//! The `propagate` workload's engine rungs, `plan`, `plan-partitioned` and
//! `placed`: the engine's
//! [`propagate_embeddings`](InferenceSession::propagate_embeddings)
//! (APPNP/SGC decoupled-GNN forward pass) over a synthetic graph and embedding
//! table at `target_partitions` 1 and N, and the same request as a job on
//! the compute plane, one leg per `(graph size, rung)`.
//!
//! The first two rungs are the same DataFusion-planned propagation at two
//! partition counts; the engine's contract is that the output is
//! byte-identical across runs and `target_partitions` **on a machine** — a
//! fixed `(group, neighbour)` fold order in `f64` with one final `f32` cast —
//! so the edge between them is digest equality. The `placed` rung is the
//! same request submitted as a job into a fleet's shared catalog, claimed by
//! the submitter process and its sink placed on an executor process
//! (`crate::plane`); its output is the same bytes, so that edge is digest
//! equality too. The output is `f32`, and `f32` bits are not identical across
//! CPUs, so a digest is compared only between legs of one machine and no
//! digest is committed.
//!
//! ## What a leg carries
//!
//! * `iter_wall_s`: the wall-clock of the whole `propagate_embeddings` call
//!   (load + fold + materialise), warm — on `placed`, of the whole job from
//!   submit to completed; `work` is edges × hops;
//! * on `placed`, where the sink ran (`ran_on`): the executor, with the
//!   submitter's placement line, the scheduler's binding and the executor's
//!   sink write as evidence — a leg whose submitter wrote the table is
//!   refused;
//! * `peak_rss_bytes`: the process's high-water mark — one leg per process;
//! * `outcome_digest` of the key-sorted propagated `f32` rows, and the rows
//!   themselves as `vectors_file` + `vector_dim`, the ladder's row-paired
//!   outcome;
//! * under the legs directory's `input/<unit>/`, the embedding table and the
//!   edge list the leg propagated — the files the PyTorch rungs read, so every
//!   rung of a unit runs over byte-identical input, named by `features_sha256`
//!   and `graph_edges_sha256`.
//!
//! ## What the hermetic tests hold
//!
//! All on the running box, over a fixture small enough to fold in seconds: two
//! legs of one point agree bit for bit; `plan` and `plan-partitioned` agree bit
//! for bit; a regressed parameter (one hop fewer, one more, a different `α`)
//! moves the digest, so the equality has teeth; the propagated vectors are not
//! the input; and the ladder reaches its verdict over filed legs.
//!
//! ## The synthetic graph
//!
//! Drawn deterministically from a seeded LCG and a pure wiring rule
//! ([`GraphShape`]), so a size names one graph and one embedding table on any
//! box. A unit is `edges<N>`, its undirected edge count.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::Serialize;

use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
use jammi_ai::pipeline::graph_propagation::{
    PropagateRequest, PropagationOutput, PropagationWeighting,
};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};

use crate::capture::{
    cpu_provenance, file_leg, leg_report, leg_stem, legs_per_point, vector_rows_digest,
    write_jsonl, write_vector_rows, IterationSeries,
};
use crate::leg::{Facts, Leg, Measured, Measurement, Payload, Provenance, RanOn};
use crate::plane::{PlaneArgs, PlaneParams};
use crate::report::{Nullable, Tiers};

/// The source id the synthetic node embedding table is registered under. Generic
/// — names no consumer; the fixture is a neutral graph of opaque node ids.
const NODES_SOURCE: &str = "nodes";
/// The source id the synthetic edge relation is registered under.
const EDGES_SOURCE: &str = "edges";
/// The model id stamped on the synthetic input embedding table, so the
/// propagation pins to it as its `X⁽⁰⁾` rather than to a prior propagation's
/// output (the IT suite's pin discipline).
const INPUT_MODEL_ID: &str = "synthetic-embed";

/// Seed for the synthetic node feature vectors. Distinct from the edge seed so
/// the feature draw and the graph wiring are independent streams.
const FEATURE_SEED: u64 = 0x00C0_FFEE_0001;

/// The synthetic graph's shape: what, with a node count, names the graph and its
/// embedding table.
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
struct Lcg {
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
struct Node {
    id: String,
    class: usize,
}

/// Build `n_classes · per_class` synthetic nodes — `per_class` nodes in each
/// class, ids `c{class}_{i}`. The class is the homophily signal the propagation
/// smooths over.
fn build_nodes(n_classes: usize, per_class: usize) -> Vec<Node> {
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
fn build_features(nodes: &[Node], dim: usize) -> Vec<(String, Vec<f32>)> {
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
fn build_edges(nodes: &[Node], fan_out: usize) -> Vec<(String, String)> {
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
struct GraphSources {
    nodes: String,
    edges: String,
}

impl GraphSources {
    fn own() -> Self {
        Self {
            nodes: NODES_SOURCE.to_string(),
            edges: EDGES_SOURCE.to_string(),
        }
    }

    #[cfg(feature = "plane")]
    fn unique() -> Self {
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
async fn local_session(
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

/// Stand the synthetic graph up in `session`: register the nodes source
/// (its parquet written under `dir`), materialise the synthetic embedding
/// table, and register the edge relation — the same setup shape the engine's
/// own propagation suite uses, driven here from the bench crate.
async fn register_graph(
    session: &Arc<InferenceSession>,
    dir: &std::path::Path,
    sources: &GraphSources,
    nodes: &[Node],
    edges: &[(String, String)],
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // nodes parquet: _row_id, class.
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

    // The synthetic input embedding table, keyed by `_row_id`.
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
    let env =
        jammi_db::store::manifest::MaterializationEnv::new(session.compute_device(), Vec::new());
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

    // edges parquet: src, dst.
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

/// Build the propagation request over the registered synthetic graph, pinned to
/// the synthetic input embedding table (never a prior propagation's output) and
/// read undirected so the symmetric `Â` is meaningful.
async fn build_request(
    session: &Arc<InferenceSession>,
    sources: &GraphSources,
    hops: usize,
    alpha: f64,
) -> Result<PropagateRequest, Box<dyn std::error::Error>> {
    let source_table = session
        .catalog()
        .find_result_tables(&sources.nodes, None, Some(INPUT_MODEL_ID))
        .await?
        .into_iter()
        .next()
        .ok_or("the synthetic source embedding table is missing")?;
    Ok(PropagateRequest::new(
        sources.nodes.clone(),
        EdgeSourceRef::Registered {
            source_id: sources.edges.clone(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        },
    )
    .with_embedding_table(source_table.table_name)
    .with_direction(EdgeDirection::Undirected)
    .with_weighting(PropagationWeighting::DegreeNormalized)
    .with_output(PropagationOutput::Final)
    .with_hops(hops)
    .with_alpha(alpha))
}

/// Read a materialised embedding table's `(_row_id, vector)` rows back, in a
/// stable `_row_id`-sorted order so the digest folds the bits in one canonical
/// sequence regardless of scan order.
async fn read_sorted_vectors(
    session: &Arc<InferenceSession>,
    table: &ResultTableRecord,
) -> Result<Vec<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
    let batches = session
        .sql(&format!(
            "SELECT _row_id, vector FROM {}",
            jammi_db::store::result_table_relation(&table.table_name)
        ))
        .await?;
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
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

/// The rung a `propagate` leg claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Rung {
    /// `propagate_embeddings` in this process at one partition.
    Plan,
    /// The same plan at `--partitions`.
    PlanPartitioned,
    /// The same request as a job on a fleet: claimed by the submitter
    /// process, its sink placed on an executor process.
    Placed,
}

impl Rung {
    pub fn as_str(self) -> &'static str {
        match self {
            Rung::Plan => "plan",
            Rung::PlanPartitioned => "plan-partitioned",
            Rung::Placed => "placed",
        }
    }

    /// The `target_partitions` of the session this rung plans in;
    /// `partitioned` is the `plan-partitioned` rung's.
    fn target_partitions(self, partitioned: usize) -> usize {
        match self {
            Rung::Plan | Rung::Placed => 1,
            Rung::PlanPartitioned => partitioned,
        }
    }
}

/// The propagation one `propagate` leg is asked to run: one graph size on one
/// rung.
#[derive(Debug, Clone)]
pub struct PropagateLegParams {
    /// The synthetic graph's shape.
    pub shape: GraphShape,
    /// Requested node count; spread evenly over the shape's classes.
    pub nodes: usize,
    /// The rung.
    pub rung: Rung,
    /// DataFusion `target_partitions` of the `plan-partitioned` rung — the
    /// one parameter separating it from `plan`.
    pub partitions: usize,
    /// Requested hop count `K`.
    pub hops: usize,
    /// Teleport probability `α`; `0` is SGC's `Âᴷ·X`.
    pub alpha: f64,
    /// Where the leg, its vectors and the unit's inputs are filed.
    pub legs_dir: PathBuf,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iterations: usize,
    /// The measured repeat this leg is filed as.
    pub take: usize,
    /// Where the `placed` rung runs.
    pub plane: PlaneParams,
}

/// The `propagate` leg's payload: what two legs must agree on, and what this
/// rung records about its run.
#[derive(Debug, Clone, Serialize)]
pub struct PropagatePayload {
    /// sha256 of the edge list every rung reads.
    pub graph_edges_sha256: String,
    /// sha256 of the input vectors file every rung reads.
    pub features_sha256: String,
    /// Node count.
    pub node_count: usize,
    /// Undirected edge count — the unit of the sweep.
    pub edge_count: usize,
    /// Vector width.
    pub dim: usize,
    /// Hops actually run — the request clamped to the engine's hop cap.
    pub hops: usize,
    /// Teleport probability `α`.
    pub alpha: f64,
    /// The operator: `D̃^{-1/2}(A+I)D̃^{-1/2}` over an undirected read.
    pub weighting: &'static str,
    /// The precision the vectors are held in.
    pub compute_precision: &'static str,
    /// The rung this leg claims.
    pub rung: &'static str,
    /// The unit of the sweep, `edges<N>`.
    pub unit: String,
    /// The measured repeat.
    pub take: usize,
    /// The session's `target_partitions`.
    pub target_partitions: usize,
    /// The hop count asked for, before the clamp.
    pub requested_hops: usize,
    /// The wiring rule's fan-out cap.
    pub fan_out: usize,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iters_measured: usize,
    /// The implementation that propagated.
    pub operator: &'static str,
}

impl Payload for PropagatePayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("graph_edges_sha256", Nullable::NonNull),
        ("features_sha256", Nullable::NonNull),
        ("node_count", Nullable::NonNull),
        ("edge_count", Nullable::NonNull),
        ("dim", Nullable::NonNull),
        ("hops", Nullable::NonNull),
        (
            "alpha",
            Nullable::NullMeans("no teleport: plain K-hop smoothing"),
        ),
        ("weighting", Nullable::NonNull),
        ("compute_precision", Nullable::NonNull),
    ];
}

/// The unit's input files under `legs_dir/input/<unit>/`: the vectors as
/// `x0.vectors.f32` + `x0.keys.txt` and the undirected edge list as
/// `edges.jsonl`, one `{"src", "dst"}` row per edge.
pub const INPUT_DIR: &str = "input";
/// The input vectors' stem.
pub const X0_STEM: &str = "x0";
/// The input edge list.
pub const EDGES_FILE: &str = "edges.jsonl";

/// Run one `propagate` leg and file it: write the unit's inputs (the files the
/// PyTorch rungs read), stand the same graph up in a session at
/// `target_partitions`, propagate it warm, and persist the key-sorted output
/// beside the leg. Returns the leg and its file name.
pub async fn run_leg(
    params: &PropagateLegParams,
) -> Result<(Leg<PropagatePayload>, String), Box<dyn std::error::Error>> {
    let shape = params.shape;
    let per_class = (params.nodes / shape.n_classes).max(2);
    let nodes = build_nodes(shape.n_classes, per_class);
    let edges = build_edges(&nodes, shape.fan_out);
    let unit = format!("edges{}", edges.len());
    let rung = params.rung.as_str();
    let stem = leg_stem(rung, &unit, params.take);

    let input_dir = params.legs_dir.join(INPUT_DIR).join(&unit);
    let x0 = write_vector_rows(&input_dir, X0_STEM, &build_features(&nodes, shape.dim))?;
    let edges_file = write_jsonl(
        &input_dir,
        EDGES_FILE,
        edges
            .iter()
            .map(|(src, dst)| serde_json::json!({"src": src, "dst": dst})),
    )?;

    let dir = tempfile::tempdir()?;
    let mut host = Host::stand_up(params, dir.path()).await?;
    let sources = host.sources().clone();
    register_graph(
        host.session(),
        dir.path(),
        &sources,
        &nodes,
        &edges,
        shape.dim,
    )
    .await?;
    let request = build_request(host.session(), &sources, params.hops, params.alpha).await?;

    let mut series = IterationSeries::new(params.warmup, params.iterations);
    let mut last = None;
    for _ in 0..series.total() {
        let start = Instant::now();
        let table = host.propagate(&request).await?;
        series.record(start.elapsed());
        last = Some(table);
    }
    let peak_rss_bytes = crate::rss::peak_rss_measurement();

    let table = last.ok_or("a propagate leg needs at least one iteration")?;
    let rows = read_sorted_vectors(host.session(), &table).await?;
    let ran_on = host.ran_on(&table.table_name).await?;
    let vectors = write_vector_rows(&params.legs_dir, &stem, &rows)?;
    let hops = request.effective_hops();

    let payload = PropagatePayload {
        graph_edges_sha256: edges_file.sha256,
        features_sha256: x0.file.sha256,
        node_count: nodes.len(),
        edge_count: edges.len(),
        dim: shape.dim,
        hops,
        alpha: params.alpha,
        weighting: "degree_normalized",
        compute_precision: "f32",
        rung,
        unit,
        take: params.take,
        target_partitions: params.rung.target_partitions(params.partitions),
        requested_hops: params.hops,
        fan_out: shape.fan_out,
        warmup: params.warmup,
        iters_measured: params.iterations,
        operator: "jammi_ai::session::InferenceSession::propagate_embeddings",
    };
    let measured = Measured {
        iter_wall_s: Some(series.into_seconds()),
        work: Some((edges.len() * hops) as f64),
        peak_rss_bytes,
        peak_vram_bytes: Measurement::not_yet_measured("bytes"),
        outcome_digest: Some(vector_rows_digest(&rows)),
        vectors_file: Some(format!("{stem}.vectors.f32")),
        vector_dim: Some(vectors.dim),
        ..Default::default()
    };
    let provenance = Provenance {
        ran_on,
        ..cpu_provenance()
    };
    let leg = Leg::new(payload, provenance, measured, Facts::default());
    let report = leg_report("propagate", leg.clone(), |leg| Tiers {
        propagate: Some(leg),
        ..Default::default()
    });
    let file = file_leg(&params.legs_dir, &stem, &report)?;
    Ok((leg, file))
}

/// Where a leg's propagation runs: a session of this process's own, or a
/// fleet the request is submitted into as a job.
enum Host {
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

impl Host {
    async fn stand_up(
        params: &PropagateLegParams,
        artifact_dir: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match params.rung {
            Rung::Plan | Rung::PlanPartitioned => Ok(Host::InProcess {
                session: local_session(
                    artifact_dir,
                    params.rung.target_partitions(params.partitions),
                )
                .await?,
                sources: GraphSources::own(),
            }),
            #[cfg(feature = "plane")]
            Rung::Placed => {
                let fleet = crate::plane::fleet::RunningFleet::spawn_placed(
                    &params.plane,
                    &format!("propagate-nodes{}-r{}", params.nodes, params.take),
                    -1,
                    &["propagate"],
                )
                .await
                .map_err(|e| e.to_string())?;
                Ok(Host::Placed {
                    fleet,
                    sources: GraphSources::unique(),
                })
            }
            #[cfg(not(feature = "plane"))]
            Rung::Placed => Err(params.plane.refusal("propagate", Rung::Placed.as_str())),
        }
    }

    fn session(&self) -> &Arc<InferenceSession> {
        match self {
            Host::InProcess { session, .. } => session,
            #[cfg(feature = "plane")]
            Host::Placed { fleet, .. } => &fleet.session,
        }
    }

    fn sources(&self) -> &GraphSources {
        match self {
            Host::InProcess { sources, .. } => sources,
            #[cfg(feature = "plane")]
            Host::Placed { sources, .. } => sources,
        }
    }

    /// One propagation of `request`: the table it committed.
    async fn propagate(
        &mut self,
        request: &PropagateRequest,
    ) -> Result<ResultTableRecord, Box<dyn std::error::Error>> {
        match self {
            Host::InProcess { session, .. } => {
                let (table, _) = session
                    .propagate_embeddings(request, jammi_db::store::CachePolicy::Bypass)
                    .await?;
                Ok(table)
            }
            #[cfg(feature = "plane")]
            Host::Placed { fleet, .. } => {
                let job = fleet
                    .session
                    .enqueue(
                        jammi_ai::jobs::JobSpec::Propagate {
                            request: request.clone(),
                            cache: jammi_db::store::CachePolicy::Bypass,
                        },
                        0,
                    )
                    .await?;
                let record = fleet
                    .await_completed(&job.job_id, "the placed propagation completes")
                    .await
                    .map_err(|e| e.to_string())?;
                let result = record
                    .result
                    .as_deref()
                    .ok_or("the completed propagation job carries no result")?;
                let jammi_ai::jobs::JobResult::Table { table, .. } = serde_json::from_str(result)?
                else {
                    return Err("the propagation job's result is not a table".into());
                };
                fleet
                    .session
                    .catalog()
                    .get_result_table(&table)
                    .await?
                    .ok_or_else(|| {
                        format!("the propagated table {table} is not in the catalog").into()
                    })
            }
        }
    }

    /// Where the sink that committed `table_name` ran, when it left this
    /// process.
    #[cfg(feature = "plane")]
    async fn ran_on(
        &mut self,
        table_name: &str,
    ) -> Result<Option<RanOn>, Box<dyn std::error::Error>> {
        match self {
            Host::InProcess { .. } => Ok(None),
            Host::Placed { fleet, .. } => {
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
    async fn ran_on(
        &mut self,
        _table_name: &str,
    ) -> Result<Option<RanOn>, Box<dyn std::error::Error>> {
        Ok(None)
    }
}

/// `propagate`'s flags: the size sweep, the rungs, and the operator's
/// parameters. One leg per `(nodes, rung, take)` point.
#[derive(Debug, Clone, clap::Args)]
pub struct PropagateArgs {
    /// Node counts to sweep, comma-separated.
    #[arg(long, value_delimiter = ',', required = true)]
    nodes: Vec<usize>,
    /// `target_partitions` of the `plan-partitioned` rung.
    #[arg(long, default_value_t = 4)]
    partitions: usize,
    /// The rungs to run: `plan`, `plan-partitioned`, `placed`; the first two
    /// by default. `placed` needs `--features plane`, the plane's backends
    /// in the environment and `--server-bin`.
    #[arg(long = "rung", value_enum, value_delimiter = ',', default_values = ["plan", "plan-partitioned"])]
    rungs: Vec<Rung>,
    #[arg(long, default_value_t = jammi_ai::pipeline::graph_propagation::DEFAULT_PROPAGATE_HOPS)]
    hops: usize,
    #[arg(long, default_value_t = jammi_ai::pipeline::graph_propagation::DEFAULT_TELEPORT_ALPHA)]
    alpha: f64,
    /// Where the legs are filed, `<rung>__edges<N>__r<take>.json` with the
    /// vectors beside, and the inputs under `input/edges<N>/`.
    #[arg(long)]
    legs_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    /// Measured repeats of each point, each in a process of its own.
    #[arg(long, default_value_t = 1)]
    takes: usize,
    /// The take a single point's run is filed as.
    #[arg(long, default_value_t = 1)]
    take: usize,
    #[command(flatten)]
    plane: PlaneArgs,
}

impl PropagateArgs {
    fn params(&self, nodes: usize, rung: Rung, take: usize) -> PropagateLegParams {
        PropagateLegParams {
            shape: DEFAULT_SHAPE,
            nodes,
            rung,
            partitions: self.partitions,
            hops: self.hops,
            alpha: self.alpha,
            legs_dir: self.legs_dir.clone(),
            warmup: self.warmup,
            iterations: self.iterations,
            take,
            plane: self.plane.clone().into(),
        }
    }

    /// Run the subcommand: one leg per point, and print the file names.
    pub async fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let points: Vec<(usize, Rung, usize)> = self
            .nodes
            .iter()
            .flat_map(|&n| {
                self.rungs
                    .iter()
                    .flat_map(move |&r| (1..=self.takes).map(move |t| (n, r, t)))
            })
            .collect();
        let (n, rung, _) = *points
            .first()
            .ok_or("propagate needs at least one --nodes value")?;
        let first = self.params(n, rung, if points.len() == 1 { self.take } else { 1 });
        let files = legs_per_point(
            &points,
            async move { run_leg(&first).await.map(|(_, file)| vec![file]) },
            |&(nodes, rung, take)| {
                let flags = [
                    ("--nodes", nodes.to_string()),
                    ("--rung", rung.as_str().to_string()),
                    ("--partitions", self.partitions.to_string()),
                    ("--hops", self.hops.to_string()),
                    ("--alpha", self.alpha.to_string()),
                    ("--warmup", self.warmup.to_string()),
                    ("--iterations", self.iterations.to_string()),
                    ("--take", take.to_string()),
                ];
                [
                    "propagate".into(),
                    "--legs-dir".into(),
                    (&self.legs_dir).into(),
                ]
                .into_iter()
                .chain(
                    flags
                        .into_iter()
                        .flat_map(|(flag, value)| [flag.into(), value.into()]),
                )
                .chain(PlaneParams::from(self.plane.clone()).child_args())
                .collect()
            },
        )
        .await?;
        println!("{}", serde_json::to_string_pretty(&files)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::capture::read_vector_rows;
    use crate::ladder::definition::Workload;
    use crate::ladder::verdict::OutcomeVerdict;
    use crate::ladder::{run_ladder, Axis, LadderArgs};

    const HOPS: usize = 2;
    const ALPHA: f64 = 0.1;

    /// One cold iteration of a 32-node fixture (4 classes × 8), filed under
    /// `legs_dir`.
    async fn leg(
        legs_dir: &std::path::Path,
        nodes: usize,
        hops: usize,
        alpha: f64,
        partitions: usize,
    ) -> (Leg<PropagatePayload>, String) {
        run_leg(&PropagateLegParams {
            shape: DEFAULT_SHAPE,
            nodes,
            rung: if partitions == 1 {
                Rung::Plan
            } else {
                Rung::PlanPartitioned
            },
            partitions,
            hops,
            alpha,
            legs_dir: legs_dir.to_path_buf(),
            warmup: 0,
            iterations: 1,
            take: 1,
            plane: PlaneParams::default(),
        })
        .await
        .expect("propagate leg runs")
    }

    /// The engine's real contract for an `f32` output: folding the same fixture
    /// twice on THIS box produces the byte-identical digest and byte-identical
    /// rows. True on any box by construction.
    #[tokio::test]
    async fn refold_is_deterministic_on_this_machine() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (first, file) = leg(a.path(), 32, HOPS, ALPHA, 1).await;
        let (second, _) = leg(b.path(), 32, HOPS, ALPHA, 1).await;
        assert_eq!(
            first.measured.outcome_digest,
            second.measured.outcome_digest
        );
        assert_eq!(
            first.payload.features_sha256,
            second.payload.features_sha256
        );
        assert_eq!(file, format!("plan__{}__r1.json", first.payload.unit));
        let rows = read_vector_rows(
            &a.path()
                .join(format!("plan__{}__r1.vectors.f32", first.payload.unit)),
            DEFAULT_SHAPE.dim,
        )
        .unwrap();
        assert_eq!(rows.len(), first.payload.node_count);
        assert_eq!(
            vector_rows_digest(&rows),
            first.measured.outcome_digest.unwrap()
        );
    }

    /// The `plan` → `plan-partitioned` edge is exact: same identity, same bits.
    #[tokio::test]
    async fn digest_is_invariant_across_target_partitions() {
        let dir = tempfile::tempdir().unwrap();
        let (one, _) = leg(dir.path(), 32, HOPS, ALPHA, 1).await;
        let (four, _) = leg(dir.path(), 32, HOPS, ALPHA, 4).await;
        assert_eq!(
            (one.payload.rung, four.payload.rung),
            ("plan", "plan-partitioned")
        );
        assert_eq!(one.measured.outcome_digest, four.measured.outcome_digest);
    }

    /// The equality above has teeth: a regressed parameter moves the digest.
    #[tokio::test]
    async fn perturbed_propagation_changes_the_digest() {
        let dir = tempfile::tempdir().unwrap();
        let (baseline, _) = leg(&dir.path().join("base"), 32, HOPS, ALPHA, 1).await;
        for (what, hops, alpha) in [
            ("one hop fewer", HOPS - 1, ALPHA),
            ("one hop more", HOPS + 1, ALPHA),
            ("a different teleport α", HOPS, ALPHA + 0.4),
        ] {
            let (perturbed, _) =
                leg(&dir.path().join(what.replace(' ', "_")), 32, hops, alpha, 1).await;
            assert_ne!(
                perturbed.measured.outcome_digest, baseline.measured.outcome_digest,
                "{what}"
            );
        }
    }

    /// The propagated vectors are not the input.
    #[tokio::test]
    async fn propagation_is_not_the_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (propagated, _) = leg(dir.path(), 32, HOPS, ALPHA, 1).await;
        let nodes = build_nodes(DEFAULT_SHAPE.n_classes, 8);
        let mut input = build_features(&nodes, DEFAULT_SHAPE.dim);
        input.sort_by(|a, b| a.0.cmp(&b.0));
        assert_ne!(
            Some(vector_rows_digest(&input)),
            propagated.measured.outcome_digest
        );
    }

    /// The hop count a leg is identified by is the depth actually run.
    #[tokio::test]
    async fn identity_records_the_hops_actually_run() {
        let dir = tempfile::tempdir().unwrap();
        let (capped, _) = leg(dir.path(), 32, 9, ALPHA, 1).await;
        assert_eq!(
            capped.payload.hops,
            jammi_ai::pipeline::graph_neighbourhood::DEFAULT_HOP_CAP
        );
        assert_eq!(capped.payload.requested_hops, 9);
    }

    /// The ladder runs end to end over filed legs: the exact edge between
    /// `plan` and `plan-partitioned` by digest, the cross-stack edges by row
    /// agreement. The reference rungs' legs are stand-ins — the `plan` legs
    /// filed under `torch` and `torch-geometric` — so the run exercises the
    /// comparator's path over the leg shape, not a second operator.
    #[tokio::test]
    async fn the_ladder_reaches_its_verdicts_over_filed_legs() {
        let dir = tempfile::tempdir().unwrap();
        let legs = dir.path().join("legs");
        for nodes in [32, 64] {
            let (_, plan) = leg(&legs, nodes, HOPS, ALPHA, 1).await;
            leg(&legs, nodes, HOPS, ALPHA, 4).await;
            for rung in ["torch", "torch-geometric"] {
                std::fs::copy(
                    legs.join(&plan),
                    legs.join(plan.replacen("plan__", &format!("{rung}__"), 1)),
                )
                .unwrap();
            }
        }
        let verdict = run_ladder(&LadderArgs {
            workload: Workload::Propagate,
            legs_dir: legs,
            out: None,
            from: None,
            to: Some("plan-partitioned".into()),
            axes: vec![Axis::Outcome],
            waive_control: false,
            law_dir: None,
            mutants: vec![],
            revision: None,
        })
        .unwrap();
        assert!(verdict.refusals.is_empty(), "{:?}", verdict.refusals);
        assert_eq!(verdict.edges.len(), 3);
        for edge in &verdict.edges {
            assert!(
                edge.refusals.is_empty(),
                "{}: {:?}",
                edge.edge,
                edge.refusals
            );
        }
        assert!(matches!(
            verdict.edges[0].outcome,
            Some(OutcomeVerdict::RowAgreement { .. })
        ));
        assert!(matches!(
            verdict.edges[1].outcome,
            Some(OutcomeVerdict::RowAgreement { .. })
        ));
        assert!(
            matches!(
                verdict.edges[2].outcome,
                Some(OutcomeVerdict::Digest { .. })
            ),
            "{:?}",
            verdict.edges[2].outcome
        );
    }
}
