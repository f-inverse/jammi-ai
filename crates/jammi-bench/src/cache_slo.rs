//! The CPU-hermetic cache-hit SLO tier: the engine's opt-in producer
//! memoization ([`CachePolicy::Use`](jammi_db::store::CachePolicy)) measured on a
//! genuinely cacheable producer — the neighbour-graph, which anchors on the
//! immutable source-embedding-table `ResultDigest`, so the same build over the
//! same parent is a sound reuse.
//!
//! ## The gated property is the speed-up, not a wall-time
//!
//! A cache hit's win is *skipping the whole compute*. So the portable SLO is the
//! ratio `cold_build / warm_hit`, not either absolute (both are machine-
//! dependent — the un-gated-rate discipline the rest of the harness follows). The
//! tier stands up a hermetic `Device::Cpu` session over a synthetic embedding
//! table, runs:
//!
//! * a **cold** `Use` build — nothing cached yet, so the producer does the full
//!   read + k-NN edge build + write + finalize;
//! * a **warm** `Use` build over the *identical* `(definition, source-digest)` —
//!   the top-of-producer probe finds the prior table, confirms its bytes, and
//!   short-circuits, returning `CacheOutcome::Reused` with **no** compute;
//!
//! and gates that the warm hit cleared a committed minimum speed-up. The gate has
//! teeth: a probe that did not actually short-circuit (or a producer that
//! rebuilt under `Use`) would do the full work warm too, the ratio would collapse
//! toward 1, and the gate would fail. The committed `min_speedup` is a
//! conservative floor (the real ratio is far larger — a build vs a catalog lookup)
//! so the gate is robust to box-to-box timing noise while still proving the
//! short-circuit fires.
//!
//! ## Non-vacuity guard
//!
//! The warm build must report `Reused` (asserted), so a regression that silently
//! recomputed under `Use` is caught as a wrong *outcome*, not merely a slow ratio.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph;
use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::{CacheOutcome, CachePolicy};

use crate::report::{CacheSloTier, Measurement, SpeedupGate};

/// The source id the synthetic node table is registered under. Generic — names
/// no consumer; the fixture is a neutral set of opaque node ids.
const SOURCE_ID: &str = "nodes";
/// The model id stamped on the synthetic input embedding table.
const INPUT_MODEL_ID: &str = "synthetic-embed";
/// Seed for the synthetic feature vectors.
const FEATURE_SEED: u64 = 0x00C0_FFEE_5101;

/// The committed cache-SLO spec: the synthetic fixture's size + the minimum
/// speed-up the warm hit must clear. The on-disk `baselines/cache_slo.json` the
/// tier and its gate read.
///
/// `min_speedup` is the *only* gated number, and it is a conservative floor — a
/// catalog lookup + extant-bytes check is orders under the cold k-NN build, so a
/// modest committed floor proves the short-circuit fires without pinning a
/// machine-dependent absolute. `nodes` / `dim` / `k` size the fixture so the cold
/// build does enough work that the ratio is meaningful, while staying fast enough
/// for the `cargo test` gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSloSpec {
    /// Number of synthetic nodes the embedding table holds.
    pub nodes: usize,
    /// Embedding dimensionality of the synthetic vectors.
    pub dim: usize,
    /// Neighbourhood size `k` the gated neighbour-graph is built at.
    pub k: usize,
    /// The minimum speed-up `cold_build / warm_hit` the warm cache hit must clear.
    /// A conservative floor: the real ratio is far larger.
    pub min_speedup: f64,
}

impl CacheSloSpec {
    /// The crate-relative path to the committed spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("cache_slo.json")
    }

    /// Load the committed spec from `baselines/cache_slo.json`.
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
/// own object-store parquet writer (the same path the recall/propagate fixtures
/// use), and return its `file://` URL.
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
/// `nodes` rows in `dim` dimensions, registered so `build_neighbor_graph`
/// resolves it. The vectors are drawn from a seeded LCG so the fixture is
/// reproducible; their exact values are immaterial (the tier times the build, not
/// the edge content). Holds the [`tempfile::TempDir`] so the artifacts outlive
/// the session.
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

    // A nodes source (so the embedding table has a registered source to hang off).
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

    // The synthetic input embedding table, keyed by `_row_id`.
    let mut rng = Lcg::new(FEATURE_SEED);
    let features: Vec<(String, Vec<f32>)> = ids
        .iter()
        .map(|id| {
            let v: Vec<f32> = (0..dim).map(|_| (rng.unit() as f32) - 0.5).collect();
            (id.clone(), v)
        })
        .collect();
    let descriptor = jammi_db::store::manifest::ProducingDescriptor::ContextSet {
        encoder_id: INPUT_MODEL_ID.to_string(),
        source_id: SOURCE_ID.to_string(),
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

/// Run the cache-SLO tier against the committed spec: build the synthetic
/// embedding table, time a cold `Use` neighbour-graph build (nothing cached), then
/// time a warm `Use` build over the identical key (the probe short-circuits), and
/// gate the speed-up against the committed floor.
///
/// The warm build is asserted to report `CacheOutcome::Reused` — a regression that
/// silently recomputed under `Use` is caught as a wrong outcome here, before the
/// timing ratio is even consulted.
pub async fn run(spec: &CacheSloSpec) -> Result<CacheSloTier, Box<dyn std::error::Error>> {
    let (session, _dir) = embedding_session(spec.nodes, spec.dim).await?;
    let params = BuildNeighborGraph {
        k: spec.k,
        exact: true, // deterministic so the cold build does the full O(n²) work
        ..Default::default()
    };

    // Cold Use build: nothing cached, so the producer does the full compute.
    let cold_start = Instant::now();
    let (first, first_outcome) = session
        .build_neighbor_graph(SOURCE_ID, None, &params, CachePolicy::Use)
        .await?;
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1_000.0;
    if first_outcome != CacheOutcome::Computed {
        return Err("the cold Use build must Compute (nothing was cached yet)".into());
    }

    // Warm Use hit: the identical (definition, source-digest) short-circuits.
    let warm_start = Instant::now();
    let (_second, warm_outcome) = session
        .build_neighbor_graph(SOURCE_ID, None, &params, CachePolicy::Use)
        .await?;
    let warm_ms = warm_start.elapsed().as_secs_f64() * 1_000.0;
    match warm_outcome {
        CacheOutcome::Reused { table } if table == first.table_name => {}
        other => {
            return Err(format!(
                "the warm Use build must Reuse the cold table {}, got {other:?}",
                first.table_name
            )
            .into());
        }
    }

    let speedup = SpeedupGate::new(cold_ms, warm_ms, spec.min_speedup);
    Ok(CacheSloTier {
        k: spec.k,
        cold_build_ms: Measurement::measured(cold_ms, "ms"),
        warm_hit_ms: Measurement::measured(warm_ms, "ms"),
        speedup,
    })
}

/// Whether the cache-SLO gate held — the verdict the subcommand maps to its exit
/// code and the `cargo test` gate asserts: the warm hit cleared the committed
/// minimum speed-up (and, implicitly, reported `Reused` — [`run`] errors out
/// otherwise).
pub fn gate_passed(tier: &CacheSloTier) -> bool {
    tier.speedup.passed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_spec_is_well_formed() {
        let spec = CacheSloSpec::load().expect("baselines/cache_slo.json must be present");
        assert!(
            spec.nodes >= 2,
            "a neighbour-graph needs at least two nodes"
        );
        assert!(spec.dim > 0);
        assert!(spec.k >= 1);
        assert!(
            spec.min_speedup > 1.0,
            "a cache hit must be required to be FASTER than the cold build"
        );
    }

    /// The SLO gate clears on this box: a warm `Use` hit short-circuits the
    /// neighbour-graph build by at least the committed minimum speed-up. Both the
    /// cold and warm folds run on the test box, so the *ratio* is portable even
    /// though the absolute wall-times are machine-dependent.
    #[tokio::test]
    async fn warm_hit_clears_the_committed_speedup_floor() {
        let spec = CacheSloSpec::load().expect("baselines/cache_slo.json must be present");
        let tier = run(&spec).await.expect("cache-slo tier runs over the spec");
        assert!(
            gate_passed(&tier),
            "the warm cache hit ({:?}) did not clear the committed {}x speed-up over the \
             cold build ({:?}) — the top-of-producer probe did not short-circuit",
            tier.warm_hit_ms.value,
            spec.min_speedup,
            tier.cold_build_ms.value,
        );
    }

    /// Non-vacuity: the warm build genuinely Reused (else [`run`] errors), AND the
    /// measured speed-up is a real ratio of two measured wall-times — not a
    /// stubbed value a gate could pass on vacuously.
    #[tokio::test]
    async fn the_tier_measures_a_real_cold_and_warm_time() {
        let spec = CacheSloSpec::load().expect("baselines/cache_slo.json must be present");
        let tier = run(&spec).await.expect("cache-slo tier runs");
        let cold = tier.cold_build_ms.value.expect("a measured cold build");
        let warm = tier.warm_hit_ms.value.expect("a measured warm hit");
        assert!(cold > 0.0 && warm >= 0.0);
        assert_eq!(
            tier.speedup.measured_speedup,
            tier.speedup.cold_ms / tier.speedup.warm_ms.max(f64::MIN_POSITIVE),
            "the reported speed-up must be the real cold/warm ratio"
        );
    }
}
