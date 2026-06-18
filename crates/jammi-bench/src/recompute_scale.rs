//! The CPU-hermetic recompute tier: the engine's
//! [`recompute(Downstream)`](jammi_ai::Session::recompute) bounded topological
//! sweep over a synthetic derived-table DAG.
//!
//! ## What is gated — the sweep's correctness, not a wall-time
//!
//! `recompute(Downstream)` re-invokes a table's recorded producer and then sweeps
//! every transitive dependent **once**, in topological (parent-before-child)
//! order — so each child re-resolves over its freshly-recomputed parent. That
//! invariant is the engine's contract and is box-independent, so the portable
//! property this tier gates is the *correctness of the sweep* — the right node
//! count, each recomputed exactly once, in topological order — not the
//! machine-dependent wall-time (the un-gated-rate discipline the rest of the
//! harness follows).
//!
//! The DAG is the canonical derived-table shape: a synthetic embedding table → a
//! neighbour-graph derived from it → a graph propagation over both. Recomputing
//! the neighbour-graph with `Downstream` must recompute the graph *and* its
//! propagation dependent (which anchors on the graph's digest), the propagation
//! strictly after the graph. A regression that dropped a dependent, re-ran one
//! twice, or inverted the order would move `recomputed_count` /
//! `topological_order_held` and trip the gate.
//!
//! ## What is measured as reference, not gated
//!
//! The whole sweep's wall-time at the named DAG size rides along as a
//! [`Measurement`] reference only — it scales with the producers re-run (the k-NN
//! rebuild + the propagation fold) and the box, so gating it as a portable floor
//! would be the un-gated-rate mistake.
//!
//! ## Why a committed *spec*, not a committed number
//!
//! The synthetic features are drawn from a seeded LCG (the generator family the
//! rest of the harness uses), so the committed artifact is the *generation spec*
//! (node count, dim, k). The gate regenerates the exact fixture and re-runs the
//! real engine sweep on the running box; the expected node count is derived from
//! the DAG shape the spec builds, never a hand-written constant.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use jammi_ai::pipeline::graph_neighbourhood::EdgeSourceRef;
use jammi_ai::pipeline::graph_propagation::PropagateRequest;
use jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph;
use jammi_ai::pipeline::recompute::Cascade;
use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::CachePolicy;

use crate::report::{Measurement, RecomputeScaleTier};

/// The source id the synthetic node table is registered under. Generic — names no
/// consumer; the fixture is a neutral set of opaque node ids.
const SOURCE_ID: &str = "nodes";
/// The model id stamped on the synthetic input embedding table.
const INPUT_MODEL_ID: &str = "synthetic-embed";
/// Seed for the synthetic feature vectors.
const FEATURE_SEED: u64 = 0x00C0_FFEE_5202;

/// The committed recompute spec: the synthetic fixture's size. The on-disk
/// `baselines/recompute.json` the tier reads.
///
/// `nodes` / `dim` / `k` size the DAG so the sweep does real producer work (a
/// k-NN rebuild + a propagation fold) while staying fast enough for the `cargo
/// test` gate. The gated node count is *derived* from the DAG shape (graph +
/// propagation = 2), not stored, so the spec carries only the fixture size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecomputeSpec {
    /// Number of synthetic nodes the embedding table holds.
    pub nodes: usize,
    /// Embedding dimensionality of the synthetic vectors.
    pub dim: usize,
    /// Neighbourhood size `k` the gated neighbour-graph is built at.
    pub k: usize,
}

impl RecomputeSpec {
    /// The crate-relative path to the committed spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("recompute.json")
    }

    /// Load the committed spec from `baselines/recompute.json`.
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

    fn unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Write an arrow batch to a fresh parquet file under `dir`, through the engine's
/// own object-store parquet writer, and return its `file://` URL.
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

/// Stand up a hermetic `Device::Cpu` session with a synthetic embedding table of
/// `nodes` rows in `dim` dimensions, registered so the derived producers resolve
/// it. The vectors are drawn from a seeded LCG so the fixture is reproducible.
async fn embedding_session(
    nodes: usize,
    dim: usize,
) -> Result<(Arc<InferenceSession>, tempfile::TempDir), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let config = JammiConfig {
        artifact_dir: dir.path().to_path_buf(),
        gpu: GpuConfig {
            device: -1,
            ..Default::default()
        },
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await?);
    session.register_query_functions();

    let node_schema = Arc::new(Schema::new(vec![Field::new(
        "_row_id",
        DataType::Utf8,
        false,
    )]));
    let ids: Vec<String> = (0..nodes).map(|i| format!("n{i}")).collect();
    let node_batch = RecordBatch::try_new(
        Arc::clone(&node_schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )) as ArrayRef],
    )?;
    let node_url = write_parquet(dir.path(), "nodes.parquet", node_schema, node_batch).await?;
    session
        .add_source(
            SOURCE_ID,
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;

    let mut rng = Lcg::new(FEATURE_SEED);
    let features: Vec<(String, Vec<f32>)> = ids
        .iter()
        .map(|id| {
            let v: Vec<f32> = (0..dim).map(|_| (rng.unit() as f32) - 0.5).collect();
            (id.clone(), v)
        })
        .collect();
    // A synthetic seed contract for the input embedding table — a context-set
    // descriptor over the source, anchored unpinned (the same shape every other
    // tier's synthetic embedding fixture records). The recompute target is the
    // DERIVED neighbour-graph built over this table, so this seed descriptor is
    // never itself replayed; it only gives the source table a valid manifest.
    let descriptor = jammi_db::store::manifest::ProducingDescriptor::ContextSet {
        encoder_id: INPUT_MODEL_ID.to_string(),
        source_id: SOURCE_ID.to_string(),
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
        SOURCE_ID,
        "1970-01-01T00:00:00Z",
    )];
    session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            jammi_db::store::EmbeddingTableSpec {
                source_id: SOURCE_ID,
                model_id: INPUT_MODEL_ID,
                derived_from: None,
                dimensions: dim,
            },
            &features,
            jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
        )
        .await?;

    Ok((session, dir))
}

/// Run the recompute tier against the committed spec: build the DAG (embedding →
/// neighbour-graph → propagation), run `recompute(graph, Downstream)`, and gate
/// that the sweep recomputed the right tables once each in topological order.
pub async fn run(spec: &RecomputeSpec) -> Result<RecomputeScaleTier, Box<dyn std::error::Error>> {
    let (session, _dir) = embedding_session(spec.nodes, spec.dim).await?;
    let svc = Session::new(Arc::clone(&session));

    // The DAG: emb → graph (anchors emb) → propagation (anchors emb + graph).
    let (graph, _) = session
        .build_neighbor_graph(
            SOURCE_ID,
            None,
            &BuildNeighborGraph {
                k: spec.k,
                exact: true,
                ..Default::default()
            },
            CachePolicy::Bypass,
        )
        .await?;
    let request = PropagateRequest::new(
        SOURCE_ID,
        EdgeSourceRef::NeighborGraph {
            table_name: graph.table_name.clone(),
        },
    );
    let (propagation, _) = session
        .propagate_embeddings(&request, CachePolicy::Bypass)
        .await?;

    // The Downstream sweep of the graph: it recomputes the graph and its
    // transitive dependent (the propagation), the propagation strictly after the
    // graph. Expected node count = 2 (graph + propagation).
    let expected_count = 2;
    let sweep_start = Instant::now();
    let report = svc
        .recompute(&graph.table_name, Cascade::Downstream)
        .await?;
    let sweep_ms = sweep_start.elapsed().as_secs_f64() * 1_000.0;

    let originals: Vec<&str> = report
        .recomputed
        .iter()
        .map(|t| t.original.as_str())
        .collect();
    let recomputed_count = originals.len();

    // Each node once (no duplicate originals).
    let mut unique = originals.clone();
    unique.sort_unstable();
    unique.dedup();
    let each_once = unique.len() == recomputed_count;

    // Topological order: the graph (parent) strictly before the propagation
    // (child that anchors on it).
    let graph_pos = originals.iter().position(|n| *n == graph.table_name);
    let prop_pos = originals.iter().position(|n| *n == propagation.table_name);
    let topological_order_held = matches!((graph_pos, prop_pos), (Some(g), Some(p)) if g < p);

    let passed = recomputed_count == expected_count && each_once && topological_order_held;

    Ok(RecomputeScaleTier {
        nodes: spec.nodes,
        recomputed_count,
        expected_count,
        topological_order_held,
        passed,
        sweep_ms: Measurement::measured(sweep_ms, "ms"),
    })
}

/// Whether the recompute correctness gate held — the verdict the subcommand maps
/// to its exit code and the `cargo test` gate asserts.
pub fn gate_passed(tier: &RecomputeScaleTier) -> bool {
    tier.passed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_spec_is_well_formed() {
        let spec = RecomputeSpec::load().expect("baselines/recompute.json must be present");
        assert!(
            spec.nodes >= 2,
            "a neighbour-graph needs at least two nodes"
        );
        assert!(spec.dim > 0);
        assert!(spec.k >= 1);
    }

    /// The Downstream sweep over the synthetic DAG recomputes exactly the graph
    /// and its propagation dependent, each once, the propagation after the graph —
    /// the box-independent topological-sweep invariant. Runs the real engine sweep
    /// on the test box.
    #[tokio::test]
    async fn downstream_sweep_recomputes_the_dag_in_topological_order() {
        let spec = RecomputeSpec::load().expect("baselines/recompute.json must be present");
        let tier = run(&spec).await.expect("recompute tier runs over the spec");
        assert_eq!(
            tier.recomputed_count, tier.expected_count,
            "the Downstream sweep must recompute every DAG node exactly once"
        );
        assert!(
            tier.topological_order_held,
            "the parent (graph) must be recomputed before its child (propagation)"
        );
        assert!(
            gate_passed(&tier),
            "the recompute correctness gate must hold"
        );
    }
}
