//! The CPU-hermetic propagation tier: the engine's
//! [`propagate_embeddings`](InferenceSession::propagate_embeddings)
//! (APPNP/SGC decoupled-GNN forward pass) folded over a committed synthetic
//! graph+embedding fixture, gated on the engine's documented *determinism
//! contract* with propagation wall-time at named graph sizes as an un-gated
//! reference.
//!
//! This is the propagation analogue of [`crate::eval`]: every gated number is a
//! *deterministic fold* of a committed fixture through the engine's own
//! primitive. The engine's contract is that `propagate_embeddings` is
//! byte-identical across runs and `target_partitions` **on a machine** — a fixed
//! `(group, neighbour)` fold order in `f64` with one final `f32` cast. But the
//! output is `f32`, and `f32` results are NOT bit-identical *across* CPUs (SIMD/FMA
//! contraction, BLAS reduction order, and gemm tiling differ by machine). So the
//! portable property this tier gates is the real contract — *same-machine*
//! determinism — not a committed cross-machine bit-digest, which would spuriously
//! "fail" on any box other than the one the spec was cut on.
//!
//! ## What drives the digest — the real engine propagation path
//!
//! The digest folds through [`InferenceSession::propagate_embeddings`] (the
//! engine's [`jammi_ai::pipeline::graph_propagation`] primitive). The tier stands
//! up a hermetic `Device::Cpu` session, materialises a synthetic embedding
//! table over a synthetic graph, registers the edge relation, runs the real
//! propagation, reads the materialised output back, and checksums the
//! `_row_id`-sorted propagated `f32` bits. The `cargo test` gate proves the
//! contract portably and keeps teeth, all on the running box:
//!
//! * **same-machine determinism** — fold the committed fixture twice on this box
//!   and assert the two digests are equal to each other (true on any machine by
//!   construction, whatever exact bits that machine produces);
//! * **relative perturbation teeth** — fold the SAME fixture through the SAME real
//!   engine at a regressed parameter (a different hop count, the wrong `α`) and
//!   assert the perturbed digest differs from the in-process baseline computed on
//!   the same box. A regression in the propagation math — the APPNP
//!   `(1−α)·Â·X + α·X⁽⁰⁾` fold, the `D̃^{-1/2}` symmetric degree normalisation, the
//!   hop count, the `α`-teleport, or the self-loop augmentation — moves the bits and
//!   trips this, so the gate is non-vacuous.
//!
//! ## What is measured as reference, not gated
//!
//! Propagation wall-time at named graph sizes rides along as a [`Measurement`]
//! reference only — a wall-time is a property of the box, not the engine, so
//! gating it as a portable floor would be the un-gated-rate mistake (the
//! discipline the binding/training tiers' machine-dependent lanes follow). The
//! reference curve times the whole `propagate_embeddings` call (load + fold +
//! materialise) at ascending node counts with a bounded fan-out.
//!
//! ## Why a committed *spec*, not a committed digest constant
//!
//! The synthetic graph and embeddings are drawn deterministically from a seeded
//! LCG (the generator family the rest of the harness uses), so the committed
//! artifact is the *generation spec* (seeds, node/edge counts, dim, hops, α,
//! weighting). The committed `digest` is the fold the rebuild box produced when the
//! spec was cut — a documented **same-box reference** (like the machine-dependent
//! rate baselines elsewhere in the harness), never a hand-written digit string and
//! never asserted for cross-machine equality. The gate regenerates the exact same
//! fixture from the spec and re-folds it through the engine on the running box;
//! committing the spec rather than only a digest is the propagation mirror of
//! committing the corpus parquet: the inputs travel so the fold is re-derivable.
//!
//! ## Gate scale vs. timing scale
//!
//! The committed gate runs at a tractable node count so the hermetic `cargo test`
//! gate re-folds the digest in seconds — its job is to prove the engine folds the
//! *same bits twice on this box* off the committed fixture (same-machine
//! determinism is size-invariant in the sense that a regression shows at any size),
//! which a tractable point shows as faithfully as a huge one. The latency reference
//! curve sweeps the larger named
//! sizes the `propagate-scale` subcommand emits, mirroring the split the rest of
//! the harness documents between its committed gate slice and the larger on-box
//! measurement.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
use jammi_ai::pipeline::graph_propagation::{
    PropagateRequest, PropagationOutput, PropagationWeighting,
};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};

use crate::report::{DeterminismGate, Measurement, PropagateLatency, PropagateTier};

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

/// The committed propagation spec: the generation parameters the digest is folded
/// from, plus the digest (a same-box reference) and the named latency-reference
/// sizes. The on-disk `baselines/propagate.json` the tier and its gate read.
///
/// Nothing here is a hand-written digest: `digest` is the checksum the engine's
/// real propagation produced over the fixture this spec regenerates, on the rebuild
/// box. Because the output is `f32`, that digest is a documented same-box reference,
/// not a cross-machine constant — the gate asserts same-machine determinism, not
/// equality to this value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagateSpec {
    /// Embedding dimensionality the synthetic `X⁽⁰⁾` and the propagated output
    /// live in.
    pub dim: usize,
    /// Number of classes the synthetic graph wires within (a homophilous,
    /// variable-degree bounded-fan-out subgraph per class).
    pub n_classes: usize,
    /// Nodes per class in the *gated* fixture (the tractable digest size). The
    /// class count times this is the gated node count.
    pub gate_per_class: usize,
    /// Bounded fan-out cap: node `i` wires to its next `1 + (i mod fan_out)`
    /// class-mates, so the per-node degree *varies* (breaking the one-hop
    /// fixed-point convergence a regular graph would have) while the edge set
    /// stays `O(nodes · fan_out)`, under the engine's ceiling at the larger
    /// latency sizes. Shared by the gated fixture and the latency curve so both
    /// fold the same graph topology.
    pub fan_out: usize,
    /// APPNP hop count the gated digest is folded at.
    pub hops: usize,
    /// APPNP teleport probability `α` the gated digest is folded with.
    pub alpha: f64,
    /// The committed digest: the checksum of the propagated output the engine
    /// produced over the gated fixture when the spec was cut, on the rebuild box. A
    /// documented same-box reference (the output is `f32`, whose exact bits vary by
    /// CPU) — reported for human comparison, never asserted for cross-machine
    /// equality.
    pub digest: String,
    /// The node counts the un-gated latency reference is measured at, ascending.
    pub latency_nodes: Vec<usize>,
}

impl PropagateSpec {
    /// The crate-relative path to the committed propagation spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("propagate.json")
    }

    /// Load the committed spec from `baselines/propagate.json`.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(Self::path())?;
        Ok(serde_json::from_str(&json)?)
    }
}

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
/// regression would slip past the digest gate. Varying the degree by node breaks
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
    let driver = registry.driver_for(&url, None)?;
    let handle = JammiObjectStore::new(driver, url.clone());
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
    config.engine.execution_threads = target_partitions;

    let session = Arc::new(InferenceSession::new(config).await?);
    session.register_query_functions();

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
            NODES_SOURCE,
            INPUT_MODEL_ID,
            None,
            &features,
            dim,
            jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
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
            "SELECT _row_id, vector FROM \"jammi.{}\"",
            table.table_name
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

/// The stable checksum of a propagated output: an FNV-1a hash over the
/// `_row_id`-sorted rows, mixing each id's bytes and each vector lane's raw `f32`
/// bits, rendered as a fixed-width hex string.
///
/// Pure arithmetic over the exact output bits, no crate — the propagation
/// analogue of the recall floor's fold. Because the engine's propagation is
/// byte-identical across runs and partitions, this digest is a stable reference:
/// any change to the propagated bits (a different fold, hop count, or `α`) flips
/// it. The id bytes are folded in so a row *permutation* (were the sort to drift)
/// would also be caught, and the lane bits so a *value* change is caught.
fn digest(rows: &[(String, Vec<f32>)]) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    };
    for (id, vector) in rows {
        for b in id.bytes() {
            mix(b);
        }
        // A separator byte so `"ab","c"` and `"a","bc"` cannot collide.
        mix(0xff);
        for lane in vector {
            for b in lane.to_bits().to_le_bytes() {
                mix(b);
            }
        }
    }
    format!("{hash:016x}")
}

/// Fold the synthetic gated fixture through the real engine `propagate_embeddings`
/// and return the digest of the propagated output.
///
/// This is the path the digest gate re-runs: regenerate the fixture from the
/// spec, drive the real propagation on `Device::Cpu`, read the materialised
/// output back, and checksum it. `hops` / `alpha` are passed explicitly (not read
/// from the spec) so the gate-fails test can re-fold the SAME fixture at a
/// regressed depth and observe the digest move.
pub async fn fold_digest(
    spec: &PropagateSpec,
    hops: usize,
    alpha: f64,
    target_partitions: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let nodes = build_nodes(spec.n_classes, spec.gate_per_class);
    let edges = build_edges(&nodes, spec.fan_out);
    let (session, _dir) = graph_session(&nodes, &edges, spec.dim, target_partitions).await?;
    let request = build_request(&session, hops, alpha).await?;
    let table = session.propagate_embeddings(&request).await?;
    let rows = read_sorted_vectors(&session, &table).await?;
    Ok(digest(&rows))
}

/// Measure the propagation wall-time at one named graph size: build a fixture of
/// `per_class · n_classes` nodes, run the real propagation once, and time the
/// whole `propagate_embeddings` call. An un-gated, machine-dependent reference.
async fn measure_latency(
    spec: &PropagateSpec,
    nodes_target: usize,
) -> Result<PropagateLatency, Box<dyn std::error::Error>> {
    // Spread the requested node count evenly across the spec's classes; each
    // node wires to its next `spec.fan_out` class-mates, so the edge set is
    // bounded `O(nodes · fan_out)` and stays under the engine's ceiling.
    let per_class = (nodes_target / spec.n_classes).max(2);
    let nodes = build_nodes(spec.n_classes, per_class);
    let edges = build_edges(&nodes, spec.fan_out);
    let (session, _dir) = graph_session(&nodes, &edges, spec.dim, 1).await?;
    let request = build_request(&session, spec.hops, spec.alpha).await?;

    let start = Instant::now();
    let _table = session.propagate_embeddings(&request).await?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;

    Ok(PropagateLatency {
        nodes: nodes.len(),
        fan_out: spec.fan_out.min(per_class - 1),
        propagate_ms: Measurement::measured(elapsed_ms, "ms"),
    })
}

/// Run the propagation tier against the committed spec: re-fold the gated digest
/// through the real engine TWICE on this box and gate same-machine determinism
/// (`first == second`), then measure the un-gated latency reference at each named
/// size.
///
/// This is the path the `propagate-scale` subcommand drives and the `cargo test`
/// gate asserts. The output is `f32`, so its exact bits vary by CPU; the portable
/// contract is that the engine folds the *same* bits twice on a machine. The
/// committed digest rides into the [`DeterminismGate`] as a same-box reference, not
/// a gated constant. The latencies ride as reference [`Measurement`]s, never gated.
pub async fn run(spec: &PropagateSpec) -> Result<PropagateTier, Box<dyn std::error::Error>> {
    let first = fold_digest(spec, spec.hops, spec.alpha, 1).await?;
    let second = fold_digest(spec, spec.hops, spec.alpha, 1).await?;
    let digest_gate = DeterminismGate::new(first, second, spec.digest.clone());

    let mut latencies = Vec::with_capacity(spec.latency_nodes.len());
    for &n in &spec.latency_nodes {
        latencies.push(measure_latency(spec, n).await?);
    }

    Ok(PropagateTier {
        dim: spec.dim,
        hops: spec.hops,
        alpha: spec.alpha,
        weighting: "degree_normalized",
        digest: digest_gate,
        latencies,
    })
}

/// Whether the determinism gate held — the verdict the subcommand maps to its exit
/// code and the `cargo test` gate asserts: the two same-machine folds agreed. The
/// latency reference is un-gated, so it never enters the verdict.
pub fn gate_passed(tier: &PropagateTier) -> bool {
    tier.digest.passed
}

/// Re-derive the committed spec's digest from a fresh fold: regenerate the gated
/// fixture, fold it through the engine, and record the digest as a same-box
/// reference. The off-box one-shot that writes `baselines/propagate.json`; CI only
/// ever loads it and re-folds the fixture on its own box.
///
/// The latency-reference sizes are committed too (they shape the reference curve
/// the subcommand emits). The recorded digest is a documented same-box reference,
/// not a cross-machine gate value — the gate asserts same-machine determinism.
pub async fn rebuild_spec(
    dim: usize,
    n_classes: usize,
    gate_per_class: usize,
    fan_out: usize,
    hops: usize,
    alpha: f64,
    latency_nodes: &[usize],
) -> Result<PropagateSpec, Box<dyn std::error::Error>> {
    // A spec with a placeholder digest, so `fold_digest` can regenerate the same
    // fixture; the real digest replaces the placeholder below.
    let mut spec = PropagateSpec {
        dim,
        n_classes,
        gate_per_class,
        fan_out,
        hops,
        alpha,
        digest: String::new(),
        latency_nodes: latency_nodes.to_vec(),
    };
    spec.digest = fold_digest(&spec, hops, alpha, 1).await?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    /// Load a spec from an arbitrary directory's `propagate.json` (test seam).
    fn load_spec_from(dir: &Path) -> Result<PropagateSpec, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(dir.join("propagate.json"))?;
        Ok(serde_json::from_str(&json)?)
    }

    /// The committed spec is well-formed: a positive dim, at least two classes
    /// (so the graph has structure), a tractable gated size, and a non-empty
    /// digest. The digest is a hex string of the FNV width.
    #[test]
    fn committed_spec_is_well_formed() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");
        assert!(spec.dim > 0);
        assert!(
            spec.n_classes >= 2,
            "the graph needs structure to propagate"
        );
        assert!(spec.gate_per_class >= 2, "a class needs an edge");
        assert!(spec.fan_out >= 1, "each node needs at least one neighbour");
        assert!(spec.hops >= 1);
        assert!(spec.alpha >= 0.0 && spec.alpha < 1.0);
        assert_eq!(spec.digest.len(), 16, "digest is a 64-bit FNV hex string");
        assert!(
            spec.digest.chars().all(|c| c.is_ascii_hexdigit()),
            "digest must be hex"
        );
        assert!(!spec.latency_nodes.is_empty());
    }

    /// The portable determinism gate (DIGEST-CLEARS direction): folding the
    /// committed gated fixture through the engine's real `propagate_embeddings`
    /// twice on THIS box produces the byte-identical digest. This is the engine's
    /// real contract (same-machine byte-identity for an `f32` output) and is true on
    /// any CI box by construction, whatever exact bits that box produces — unlike a
    /// committed cross-machine constant, which `f32` SIMD/FMA/BLAS differences would
    /// spuriously break.
    #[tokio::test]
    async fn refold_is_deterministic_on_this_machine() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");
        let tier = run(&spec).await.expect("propagate tier runs over the spec");
        assert!(
            gate_passed(&tier),
            "two same-machine folds of the committed fixture disagreed — propagation \
             is not deterministic on this box: {} vs {}",
            tier.digest.first,
            tier.digest.second
        );
        assert_eq!(tier.digest.first, tier.digest.second);
    }

    /// The teeth, GATE-FAILS direction (RC1: an assertion must be able to fail).
    ///
    /// Perturbed propagations — the SAME committed fixture folded through the SAME
    /// real engine path with a regressed propagation *parameter* — must each
    /// produce a different digest, proving the gate catches the propagation-math
    /// regressions it exists to catch:
    ///
    /// * a **different hop count** (one fewer and one more than committed) — the
    ///   APPNP depth, the regression named in the deliverable. (The fixture's
    ///   per-node variable degree is what makes this discriminating: on a regular
    ///   graph APPNP converges in one hop and the hop count would not move the
    ///   output — see [`build_edges`].)
    /// * a **different `α`** (the teleport probability) — a changed APPNP fixed
    ///   point, the kind of regression a mis-wired restart constant would produce.
    ///
    /// The engine at the committed parameters folds a stable baseline ON THIS BOX —
    /// the in-process contrast that gives each perturbation its teeth, portable
    /// because both the baseline and the perturbed fold run on the same machine.
    #[tokio::test]
    async fn perturbed_propagation_changes_the_digest() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");

        // The in-process baseline: the committed parameters folded on THIS box. The
        // perturbations are measured against this, never against the committed
        // (same-box-reference) digest, so the teeth are portable.
        let baseline = fold_digest(&spec, spec.hops, spec.alpha, 1)
            .await
            .expect("baseline fold runs");

        // One fewer hop: a shallower APPNP fold than the baseline.
        let fewer_hops = fold_digest(&spec, spec.hops - 1, spec.alpha, 1)
            .await
            .expect("fewer-hops fold runs");
        assert_ne!(
            fewer_hops, baseline,
            "one fewer hop must change the digest (else a hop-count regression slips the gate)"
        );

        // One more hop: a deeper APPNP fold than the baseline.
        let more_hops = fold_digest(&spec, spec.hops + 1, spec.alpha, 1)
            .await
            .expect("more-hops fold runs");
        assert_ne!(
            more_hops, baseline,
            "one more hop must change the digest (else a hop-count regression slips the gate)"
        );

        // A different teleport probability: a changed APPNP fixed point. `α` is in
        // `[0, 1)`; perturb away from the committed value by a clear margin.
        let regressed_alpha = if spec.alpha < 0.5 {
            spec.alpha + 0.4
        } else {
            spec.alpha - 0.4
        };
        let wrong_alpha = fold_digest(&spec, spec.hops, regressed_alpha, 1)
            .await
            .expect("wrong-alpha fold runs");
        assert_ne!(
            wrong_alpha, baseline,
            "a different teleport α must change the digest (else an α regression slips the gate)"
        );
    }

    /// The engine's determinism contract, exercised through the tier: the gated
    /// fixture folds to a byte-identical digest across two `target_partitions`
    /// settings (1 and 4) ON THIS BOX. Both folds run on the test box, so the
    /// comparison is portable — were propagation partition-order-sensitive, the
    /// same-machine determinism gate would itself be a moving target.
    #[tokio::test]
    async fn digest_is_invariant_across_target_partitions() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");
        let one = fold_digest(&spec, spec.hops, spec.alpha, 1)
            .await
            .expect("fold at 1 partition");
        let four = fold_digest(&spec, spec.hops, spec.alpha, 4)
            .await
            .expect("fold at 4 partitions");
        assert_eq!(
            one, four,
            "propagation must be byte-identical across target_partitions on this box"
        );
    }

    /// `rebuild_spec` is the inverse of the gate: a fresh rebuild on THIS box, re-run
    /// through the gate, passes its same-machine determinism gate, and the digest it
    /// writes matches the gate's own fold on this box (both folds are on the running
    /// machine, so the round-trip is portable). Guards the off-box rebuilder against
    /// drifting from the fold idiom — without asserting the rebuilt digest equals the
    /// committed (same-box-reference) one, which need not hold across CI boxes.
    #[tokio::test]
    async fn rebuild_spec_round_trips_through_the_gate() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");
        let rebuilt = rebuild_spec(
            spec.dim,
            spec.n_classes,
            spec.gate_per_class,
            spec.fan_out,
            spec.hops,
            spec.alpha,
            &spec.latency_nodes,
        )
        .await
        .expect("rebuild runs");
        let tier = run(&rebuilt)
            .await
            .expect("tier runs over the rebuilt spec");
        assert!(
            gate_passed(&tier),
            "a freshly rebuilt spec must pass its same-machine determinism gate"
        );
        // The gate's fold on this box reproduces the digest the rebuild just wrote
        // (both folds are on the running machine) — the round-trip is exact same-box.
        assert_eq!(
            tier.digest.first, rebuilt.digest,
            "the gate's fold on this box must reproduce the rebuild's recorded digest"
        );
    }

    /// The propagated output is a non-trivial transform of `X⁽⁰⁾`: the digest of
    /// the propagated vectors (folded on THIS box) differs from the digest of the
    /// raw input features, so the gate is folding a real propagation, not the
    /// identity. (A degenerate fixture where propagation was a no-op would make the
    /// determinism gate pass vacuously.) Both digests are computed on the running
    /// machine, so the comparison is portable.
    #[tokio::test]
    async fn propagation_is_not_the_identity() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");
        let nodes = build_nodes(spec.n_classes, spec.gate_per_class);
        let mut input: Vec<(String, Vec<f32>)> = build_features(&nodes, spec.dim);
        input.sort_by(|a, b| a.0.cmp(&b.0));
        let input_digest = digest(&input);
        let propagated = fold_digest(&spec, spec.hops, spec.alpha, 1)
            .await
            .expect("propagated fold runs");
        assert_ne!(
            input_digest, propagated,
            "the propagated digest must differ from the raw input digest (propagation moved the \
             features), else the gate would pass on an identity fold"
        );
    }

    /// The latency reference is a real [`Measurement`], not a zero/None: every
    /// named size produces a measured wall-time. The value is machine-dependent
    /// (un-gated), but it must be present and positive — the honesty bar.
    #[tokio::test]
    async fn latency_reference_is_measured_not_stubbed() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");
        let tier = run(&spec).await.expect("tier runs");
        assert_eq!(tier.latencies.len(), spec.latency_nodes.len());
        for point in &tier.latencies {
            let ms = point
                .propagate_ms
                .value
                .expect("a measured latency, not a not-yet-measured stub");
            assert!(ms > 0.0, "propagation wall-time must be positive");
            assert!(point.nodes > 0);
        }
    }

    /// The `load_spec_from` seam reads a spec from an arbitrary directory.
    #[test]
    fn load_spec_from_reads_a_written_copy() {
        let spec = PropagateSpec::load().expect("baselines/propagate.json must be present");
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("propagate.json"),
            serde_json::to_string_pretty(&spec).unwrap(),
        )
        .unwrap();
        let loaded = load_spec_from(dir.path()).unwrap();
        assert_eq!(loaded.dim, spec.dim);
        assert_eq!(loaded.digest, spec.digest);
    }
}
