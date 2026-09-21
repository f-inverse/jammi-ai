//! The `propagate` workload's engine rungs: the engine's
//! [`propagate_embeddings`](InferenceSession::propagate_embeddings) (APPNP/SGC
//! decoupled-GNN forward pass) over a synthetic graph and embedding table, one
//! leg per `(graph size, target_partitions)` point.
//!
//! `plan@1` and `plan@N` are the same DataFusion-planned propagation at two
//! partition counts; the engine's contract is that the output is byte-identical
//! across runs and `target_partitions` **on a machine** — a fixed
//! `(group, neighbour)` fold order in `f64` with one final `f32` cast — so the
//! edge between the two is digest equality. The output is `f32`, and `f32` bits
//! are not identical across CPUs (SIMD/FMA contraction, reduction order), so a
//! digest is compared only between legs of one machine and no digest is
//! committed.
//!
//! ## What a leg carries
//!
//! * the per-iteration wall-clock of the whole `propagate_embeddings` call
//!   (load + fold + materialise), warm;
//! * the process's peak resident set — one point per process, so a sweep's
//!   points do not inherit each other's peak;
//! * the digest of the key-sorted propagated `f32` bits, and the vectors file;
//! * beside it, under the size's `input/`, the embedding table and the edge
//!   list the leg propagated — the files the PyTorch rung reads, so every rung
//!   of a size runs over byte-identical input.
//!
//! ## What the hermetic tests hold
//!
//! All on the running box, over a fixture small enough to fold in seconds:
//! two legs of the same point agree bit for bit; `plan@1` and `plan@4` agree
//! bit for bit; a regressed parameter (one hop fewer, one more, a different
//! `α`) moves the digest, so the equality has teeth — a regression in the
//! APPNP `(1−α)·Â·X + α·X⁽⁰⁾` fold, the `D̃^{-1/2}` normalisation, the hop
//! count, the teleport or the self-loop augmentation moves the bits; and the
//! propagated vectors are not the input.
//!
//! ## The synthetic graph
//!
//! Drawn deterministically from a seeded LCG and a pure wiring rule
//! ([`GraphShape`]), so a size names one graph and one embedding table on any
//! box.

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

use crate::leg::{
    keyed_vector_digest, write_jsonl, write_keyed_vectors, Artifact, IterationSeries, Leg,
};
use crate::report::Measurement;

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

/// Stand up a hermetic `Device::Cpu` session over a synthetic graph: register
/// the nodes source, materialise the synthetic embedding table, and register the
/// edge relation — the same setup shape the engine's own propagation suite uses,
/// driven here from the bench crate over an in-process tempdir.
///
/// `target_partitions` is the DataFusion execution-thread count: the determinism
/// test varies it to exercise the byte-identical-across-partitions contract.
/// Holds the [`tempfile::TempDir`] in the returned tuple so the fixture files
/// outlive the session.
async fn graph_session(
    nodes: &[Node],
    edges: &[(String, String)],
    dim: usize,
    target_partitions: usize,
) -> Result<(Arc<InferenceSession>, tempfile::TempDir), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    // CPU-hermetic config: device −1 forces CPU, the execution-thread count is the
    // partition knob, artifacts land in the tempdir.
    let config = JammiConfig {
        artifact_dir: dir.path().to_path_buf(),
        gpu: GpuConfig {
            device: -1,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut config = config;
    config.engine.execution_threads =
        std::num::NonZeroUsize::new(target_partitions).expect("a positive thread count");

    let session = Arc::new(InferenceSession::new(config).await?);
    session.install_query_functions();

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
    let node_url = write_parquet(dir.path(), "nodes.parquet", node_schema, node_batch).await?;
    session
        .add_source(
            NODES_SOURCE,
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
        source_id: NODES_SOURCE.to_string(),
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
        NODES_SOURCE,
        "1970-01-01T00:00:00Z",
    )];
    session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            jammi_db::store::EmbeddingTableSpec {
                source_id: NODES_SOURCE,
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
    let edge_url = write_parquet(dir.path(), "edges.parquet", edge_schema, edge_batch).await?;
    session
        .add_source(
            EDGES_SOURCE,
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;

    Ok((session, dir))
}

/// Build the propagation request over the registered synthetic graph, pinned to
/// the synthetic input embedding table (never a prior propagation's output) and
/// read undirected so the symmetric `Â` is meaningful.
async fn build_request(
    session: &Arc<InferenceSession>,
    hops: usize,
    alpha: f64,
) -> Result<PropagateRequest, Box<dyn std::error::Error>> {
    let source_table = session
        .catalog()
        .find_result_tables(NODES_SOURCE, None, Some(INPUT_MODEL_ID))
        .await?
        .into_iter()
        .next()
        .ok_or("the synthetic source embedding table is missing")?;
    Ok(PropagateRequest::new(
        NODES_SOURCE,
        EdgeSourceRef::Registered {
            source_id: EDGES_SOURCE.into(),
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

/// What one `propagate` leg is asked to run: one graph size at one
/// `target_partitions`.
#[derive(Debug, Clone)]
pub struct PropagateLegParams {
    /// The synthetic graph's shape.
    pub shape: GraphShape,
    /// Requested node count; spread evenly over the shape's classes.
    pub nodes: usize,
    /// DataFusion `target_partitions` the session runs at — the one parameter
    /// separating the `plan@1` rung from `plan@N`.
    pub target_partitions: usize,
    /// Requested hop count `K`.
    pub hops: usize,
    /// Teleport probability `α`; `0` is SGC's `Âᴷ·X`.
    pub alpha: f64,
    /// Where the inputs and this leg's output are written.
    pub out: PathBuf,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iterations: usize,
}

/// What two `propagate` legs must agree on to be comparable.
#[derive(Debug, Serialize)]
pub struct PropagateIdentity {
    /// sha256 of the input vectors file every rung reads.
    pub x0_sha256: String,
    /// sha256 of the edge list every rung reads.
    pub edges_sha256: String,
    /// Node count.
    pub nodes: usize,
    /// Undirected edge count.
    pub edges: usize,
    /// Vector width.
    pub dim: usize,
    /// The wiring rule's fan-out cap.
    pub fan_out: usize,
    /// Hops actually run — the request clamped to the engine's hop cap.
    pub hops: usize,
    /// Teleport probability `α`.
    pub alpha: f64,
    /// The operator: `D̃^{-1/2}(A+I)D̃^{-1/2}` over an undirected read.
    pub weighting: &'static str,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iterations: usize,
}

/// Recorded, never compared.
#[derive(Debug, Serialize)]
pub struct PropagateProvenance {
    /// The implementation that propagated.
    pub operator: &'static str,
    /// The session's `target_partitions`.
    pub target_partitions: usize,
    /// The device the fold ran on.
    pub device: &'static str,
    /// The hop count asked for, before the clamp.
    pub requested_hops: usize,
}

/// What a `propagate` leg measured.
#[derive(Debug, Serialize)]
pub struct PropagateMeasured {
    /// Seconds per whole `propagate_embeddings` call (load + fold + materialise).
    pub iteration_s: Vec<f64>,
    /// The process's peak resident set (kernel high-water mark).
    pub peak_rss_bytes: Measurement,
    /// Peak device memory; the fold uses no device.
    pub peak_vram_bytes: Measurement,
    /// [`keyed_vector_digest`] of the key-sorted propagated vectors.
    pub digest: String,
    /// The key-sorted propagated vectors.
    pub vectors: Artifact,
}

/// One `propagate` leg.
pub type PropagateLeg = Leg<PropagateIdentity, PropagateProvenance, PropagateMeasured>;

/// The file stem of the input vectors under a size's `input/` directory.
pub const X0_STEM: &str = "x0";
/// The edge list under a size's `input/` directory: one `{"src", "dst"}` row per
/// undirected edge.
pub const EDGES_FILE: &str = "edges.jsonl";

/// Run one `propagate` leg: write the size's inputs (the files the PyTorch rung
/// reads), stand the same graph up in a session at `target_partitions`,
/// propagate it warm, and persist the key-sorted output.
pub async fn run_leg(
    params: &PropagateLegParams,
) -> Result<PropagateLeg, Box<dyn std::error::Error>> {
    let shape = params.shape;
    let per_class = (params.nodes / shape.n_classes).max(2);
    let nodes = build_nodes(shape.n_classes, per_class);
    let edges = build_edges(&nodes, shape.fan_out);

    let size_dir = params.out.join(format!("n{}", nodes.len()));
    let input_dir = size_dir.join("input");
    let x0 = write_keyed_vectors(&input_dir, X0_STEM, &build_features(&nodes, shape.dim))?;
    let edges_file = write_jsonl(
        &input_dir,
        EDGES_FILE,
        edges
            .iter()
            .map(|(src, dst)| serde_json::json!({"src": src, "dst": dst})),
    )?;

    let (session, _dir) =
        graph_session(&nodes, &edges, shape.dim, params.target_partitions).await?;
    let request = build_request(&session, params.hops, params.alpha).await?;

    let mut series = IterationSeries::new(params.warmup, params.iterations);
    let mut last = None;
    for _ in 0..series.total() {
        let start = Instant::now();
        let (table, _) = session
            .propagate_embeddings(&request, jammi_db::store::CachePolicy::Bypass)
            .await?;
        series.record(start.elapsed());
        last = Some(table);
    }
    let peak_rss_bytes = crate::rss::peak_rss_bytes();

    let table = last.ok_or("a propagate leg needs at least one iteration")?;
    let rows = read_sorted_vectors(&session, &table).await?;
    let leg_dir = size_dir.join(format!("p{}", params.target_partitions));

    Ok(Leg {
        workload: "propagate",
        rung: format!("plan@{}", params.target_partitions),
        identity: PropagateIdentity {
            x0_sha256: x0.sha256,
            edges_sha256: edges_file.sha256,
            nodes: nodes.len(),
            edges: edges.len(),
            dim: shape.dim,
            fan_out: shape.fan_out,
            hops: request.effective_hops(),
            alpha: params.alpha,
            weighting: "degree_normalized",
            warmup: params.warmup,
            iterations: params.iterations,
        },
        provenance: PropagateProvenance {
            operator: "jammi_ai::session::InferenceSession::propagate_embeddings",
            target_partitions: params.target_partitions,
            device: "cpu",
            requested_hops: params.hops,
        },
        measured: PropagateMeasured {
            iteration_s: series.into_seconds(),
            peak_rss_bytes,
            peak_vram_bytes: Measurement::not_yet_measured("bytes"),
            digest: keyed_vector_digest(&rows),
            vectors: write_keyed_vectors(&leg_dir, "propagated", &rows)?,
        },
    })
}

/// `propagate`'s flags: the size sweep, the partition counts, and the operator's
/// parameters. One leg per `(nodes, partitions)` point.
#[derive(Debug, Clone, clap::Args)]
pub struct PropagateArgs {
    /// Node counts to sweep, comma-separated.
    #[arg(long, value_delimiter = ',', required = true)]
    nodes: Vec<usize>,
    /// `target_partitions` values, comma-separated — `1,N` is the two plan rungs.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    partitions: Vec<usize>,
    #[arg(long, default_value_t = jammi_ai::pipeline::graph_propagation::DEFAULT_PROPAGATE_HOPS)]
    hops: usize,
    #[arg(long, default_value_t = jammi_ai::pipeline::graph_propagation::DEFAULT_TELEPORT_ALPHA)]
    alpha: f64,
    /// Where inputs and outputs are written: `n<nodes>/input/` and
    /// `n<nodes>/p<partitions>/`.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
}

impl PropagateArgs {
    fn params(&self, (nodes, target_partitions): (usize, usize)) -> PropagateLegParams {
        PropagateLegParams {
            shape: DEFAULT_SHAPE,
            nodes,
            target_partitions,
            hops: self.hops,
            alpha: self.alpha,
            out: self.out.clone(),
            warmup: self.warmup,
            iterations: self.iterations,
        }
    }

    /// Run the subcommand: one leg per point, printed as one report.
    pub async fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let points: Vec<(usize, usize)> = self
            .nodes
            .iter()
            .flat_map(|&n| self.partitions.iter().map(move |&p| (n, p)))
            .collect();
        let first = points
            .first()
            .map(|&point| self.params(point))
            .ok_or("propagate needs at least one --nodes and one --partitions value")?;
        let legs = crate::leg::leg_per_point(
            &points,
            async move { run_leg(&first).await },
            |&(nodes, partitions)| {
                let flags = [
                    ("--nodes", nodes.to_string()),
                    ("--partitions", partitions.to_string()),
                    ("--hops", self.hops.to_string()),
                    ("--alpha", self.alpha.to_string()),
                    ("--warmup", self.warmup.to_string()),
                    ("--iterations", self.iterations.to_string()),
                ];
                ["propagate".into(), "--out".into(), (&self.out).into()]
                    .into_iter()
                    .chain(
                        flags
                            .into_iter()
                            .flat_map(|(flag, value)| [flag.into(), value.into()]),
                    )
                    .collect()
            },
        )
        .await?;
        Ok(crate::leg::LegReport::new("propagate", legs).emit()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOPS: usize = 2;
    const ALPHA: f64 = 0.1;

    /// One cold iteration of the 32-node fixture (4 classes × 8) — the point the
    /// hermetic tests fold.
    async fn leg(
        out: &std::path::Path,
        hops: usize,
        alpha: f64,
        partitions: usize,
    ) -> PropagateLeg {
        run_leg(&PropagateLegParams {
            shape: DEFAULT_SHAPE,
            nodes: 32,
            target_partitions: partitions,
            hops,
            alpha,
            out: out.to_path_buf(),
            warmup: 0,
            iterations: 1,
        })
        .await
        .expect("propagate leg runs")
    }

    /// The engine's real contract for an `f32` output: folding the same fixture
    /// twice on THIS box produces the byte-identical digest and the
    /// byte-identical vectors file. True on any box by construction, whatever
    /// exact bits that box produces.
    #[tokio::test]
    async fn refold_is_deterministic_on_this_machine() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let first = leg(a.path(), HOPS, ALPHA, 1).await;
        let second = leg(b.path(), HOPS, ALPHA, 1).await;
        assert_eq!(first.measured.digest, second.measured.digest);
        assert_eq!(
            first.measured.vectors.sha256,
            second.measured.vectors.sha256
        );
        assert_eq!(first.identity.x0_sha256, second.identity.x0_sha256);
        assert_eq!(first.measured.iteration_s.len(), 1);
    }

    /// The `plan@1` → `plan@4` edge is exact: same identity, same bits. Were
    /// propagation partition-order-sensitive the digest would move with the
    /// partition count.
    #[tokio::test]
    async fn digest_is_invariant_across_target_partitions() {
        let dir = tempfile::tempdir().unwrap();
        let one = leg(dir.path(), HOPS, ALPHA, 1).await;
        let four = leg(dir.path(), HOPS, ALPHA, 4).await;
        assert_eq!(
            (one.rung.as_str(), four.rung.as_str()),
            ("plan@1", "plan@4")
        );
        assert_eq!(
            serde_json::to_value(&one.identity).unwrap(),
            serde_json::to_value(&four.identity).unwrap(),
            "the two rungs of an edge agree on identity"
        );
        assert_eq!(
            one.measured.digest, four.measured.digest,
            "propagation must be byte-identical across target_partitions on this box"
        );
    }

    /// The equality above has teeth: the SAME fixture folded through the SAME
    /// engine path at a regressed parameter moves the digest — one hop fewer, one
    /// hop more, a different teleport `α`. Measured against the in-process
    /// baseline on this box, so portable.
    #[tokio::test]
    async fn perturbed_propagation_changes_the_digest() {
        let dir = tempfile::tempdir().unwrap();
        let baseline = leg(&dir.path().join("base"), HOPS, ALPHA, 1).await;
        for (what, hops, alpha) in [
            ("one hop fewer", HOPS - 1, ALPHA),
            ("one hop more", HOPS + 1, ALPHA),
            ("a different teleport α", HOPS, ALPHA + 0.4),
        ] {
            let perturbed = leg(&dir.path().join(what.replace(' ', "_")), hops, alpha, 1).await;
            assert_ne!(
                perturbed.measured.digest, baseline.measured.digest,
                "{what} must change the digest"
            );
        }
    }

    /// The propagated vectors are not the input: a fixture on which propagation
    /// was a no-op would make every equality above vacuous.
    #[tokio::test]
    async fn propagation_is_not_the_identity() {
        let dir = tempfile::tempdir().unwrap();
        let propagated = leg(dir.path(), HOPS, ALPHA, 1).await;
        let nodes = build_nodes(DEFAULT_SHAPE.n_classes, 8);
        let mut input = build_features(&nodes, DEFAULT_SHAPE.dim);
        input.sort_by(|a, b| a.0.cmp(&b.0));
        assert_ne!(keyed_vector_digest(&input), propagated.measured.digest);
    }

    /// The hop count a leg is identified by is the depth actually run: a request
    /// past the engine's hop cap is clamped, and identity says so.
    #[tokio::test]
    async fn identity_records_the_hops_actually_run() {
        let dir = tempfile::tempdir().unwrap();
        let capped = leg(dir.path(), 9, ALPHA, 1).await;
        assert_eq!(
            capped.identity.hops,
            jammi_ai::pipeline::graph_neighbourhood::DEFAULT_HOP_CAP
        );
        assert_eq!(capped.provenance.requested_hops, 9);
    }
}
