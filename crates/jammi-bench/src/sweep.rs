//! The recall-vs-cost sweep: how ANN recall and its build/query cost move as the
//! HNSW knobs change, each point measured against the exact oracle over a
//! held-out query set.
//!
//! Three axes ([`crate::report::RecallSweepTier`] states the split in full):
//!
//! * **build** — sweep the construction knobs (`connectivity`, `build_expansion`).
//!   Each point is a *separately built* graph, so the cost is build time and
//!   on-disk size. The swept graphs are not committed (one full-scale graph is
//!   hundreds of MiB; N would blow the LFS budget), so this axis is an on-box
//!   *reference* — recall rides along as provenance, it is not a portable gate.
//! * **search** — sweep `search_expansion` (ef_search) over ONE frozen graph,
//!   re-dialed at query time. Recall rises and QPS falls as ef grows. Because it
//!   re-dials a single (committable) index, this axis is re-derivable — the
//!   portable recall-floor gate the cookbook re-runs against its own oracle.
//! * **precision** — sweep `storage_precision` (`F32` exact vs `Int8` quantized)
//!   and, at `Int8`, the retrieve→rescore `oversample`. Like the build axis,
//!   quantization is baked in at construction, so each point is a separately
//!   built graph and this axis is an on-box reference here — the portable,
//!   COMMITTED recall floor for the shipped `Int8` precision lives in a
//!   dedicated `cargo test` gate over a frozen `Int8` fixture bundle (mirroring
//!   the search axis's frozen-graph discipline), not in this on-box sweep.
//!
//! The exact ground truth does not depend on the ANN knobs, so it is computed
//! once per query and reused across every swept point.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::exact::exact_vector_search;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;

use crate::corpus;
use crate::operator_mirror::retrieve_then_rescore;
use crate::recall::recall_at_k_for_query;
use crate::report::{Measurement, PrecisionSweepPoint, RecallSweepTier, SweepPoint, RECALL_KS};

/// The k QPS is reported at — a typical retrieval breadth on the recall curve,
/// so the throughput number reflects a realistic search rather than a corner.
const QPS_K: usize = 10;

/// The build-knob grid: `(connectivity, build_expansion)` points whose build
/// cost (time, size) is the deliverable; recall must hold above its floor at
/// each. `0` means the backend default (USearch M=16 / ef_construction=128). The
/// points raise `build_expansion` (and one raises `connectivity`) so the
/// build-cost curve has visible spread.
const BUILD_GRID: &[AnnIndexConfig] = &[
    AnnIndexConfig {
        connectivity: 0,
        build_expansion: 64,
        search_expansion: 0,
        storage_precision: StoragePrecision::F32,
        oversample: Some(4),
    },
    AnnIndexConfig {
        connectivity: 0,
        build_expansion: 128,
        search_expansion: 0,
        storage_precision: StoragePrecision::F32,
        oversample: Some(4),
    },
    AnnIndexConfig {
        connectivity: 0,
        build_expansion: 256,
        search_expansion: 0,
        storage_precision: StoragePrecision::F32,
        oversample: Some(4),
    },
    AnnIndexConfig {
        connectivity: 32,
        build_expansion: 128,
        search_expansion: 0,
        storage_precision: StoragePrecision::F32,
        oversample: Some(4),
    },
];

/// The search-knob grid (ef_search), deliberately spanning BELOW the backend
/// default (64) so the recall sag at low ef is visible, not just the plateau
/// above it.
const EF_GRID: &[usize] = &[8, 16, 32, 64, 128, 256];

/// Run both sweep axes over a corpus + held-out query parquet, returning the tier.
///
/// `corpus_path` holds the vectors every graph is built over and the exact
/// oracle scores; `query_path` holds a disjoint held-out query set. The exact
/// top-`max(RECALL_KS)` is computed once per query and reused for every point.
pub async fn run(
    corpus_path: &Path,
    query_path: &Path,
) -> Result<RecallSweepTier, Box<dyn std::error::Error>> {
    const CORPUS_TABLE: &str = "sweep_corpus";
    const QUERY_TABLE: &str = "sweep_queries";

    let corpus_url = corpus::storage_url(corpus_path)?;
    let ctx = corpus::register(&corpus_url, CORPUS_TABLE).await?;
    let corpus_rows = corpus::load_vectors(&ctx, CORPUS_TABLE).await?;
    if corpus_rows.is_empty() {
        return Err("recall-sweep corpus is empty".into());
    }
    let dim = corpus_rows[0].1.len();

    let query_url = corpus::storage_url(query_path)?;
    let query_ctx = corpus::register(&query_url, QUERY_TABLE).await?;
    let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, QUERY_TABLE)
        .await?
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    if queries.is_empty() {
        return Err("recall-sweep query set is empty".into());
    }

    // Exact ground truth is independent of the ANN knobs: compute the top
    // max-k once per query and reuse for every swept point.
    let max_k = RECALL_KS.iter().copied().max().unwrap_or(0);
    let mut exact: Vec<Vec<(String, f32)>> = Vec::with_capacity(queries.len());
    for q in &queries {
        exact.push(exact_vector_search(&ctx, CORPUS_TABLE, q, max_k).await?);
    }

    let tmp = tempfile::tempdir()?;

    // BUILD axis: each point is a separately built graph — measure build time + size.
    let mut build_sweep = Vec::with_capacity(BUILD_GRID.len());
    for (i, cfg) in BUILD_GRID.iter().enumerate() {
        let base = tmp.path().join(format!("build_{i}"));
        let t0 = Instant::now();
        let mut index = SidecarIndex::new(dim, cfg, StoragePrecision::F32)?;
        for (id, v) in &corpus_rows {
            index.add(id, v)?;
        }
        index.build()?;
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        VectorIndex::save(&index, &base)?;
        let size = std::fs::metadata(base.with_extension("usearch"))?.len();
        let (recall, qps) = recall_and_qps(&index, &queries, &exact)?;
        build_sweep.push(SweepPoint {
            connectivity: cfg.connectivity,
            build_expansion: cfg.build_expansion,
            search_expansion: cfg.search_expansion,
            recall,
            build_time_ms: Measurement::measured(build_ms, "ms"),
            index_size_bytes: Measurement::measured(size as f64, "bytes"),
            search_qps: Measurement::measured(qps, "queries_per_s"),
        });
    }

    // SEARCH axis: build ONE default graph, then re-dial ef_search over it — the
    // query-time knob mutates a loaded graph, so every point shares one build.
    let base = tmp.path().join("search_base");
    {
        let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::F32)?;
        for (id, v) in &corpus_rows {
            index.add(id, v)?;
        }
        index.build()?;
        VectorIndex::save(&index, &base)?;
    }
    let mut search_sweep = Vec::with_capacity(EF_GRID.len());
    for &ef in EF_GRID {
        let cfg = AnnIndexConfig {
            search_expansion: ef,
            ..AnnIndexConfig::default()
        };
        let index = SidecarIndex::load(&base, &cfg, StoragePrecision::F32)?;
        let (recall, qps) = recall_and_qps(&index, &queries, &exact)?;
        search_sweep.push(SweepPoint {
            connectivity: cfg.connectivity,
            build_expansion: cfg.build_expansion,
            search_expansion: ef,
            recall,
            // The build cost is the shared base graph's, not this point's — the
            // re-dial builds nothing, so build metrics do not apply here.
            build_time_ms: Measurement::not_yet_measured("ms"),
            index_size_bytes: Measurement::not_yet_measured("bytes"),
            search_qps: Measurement::measured(qps, "queries_per_s"),
        });
    }

    // PRECISION axis: sweep `storage_precision` (and, for a quantized
    // precision, the retrieve→rescore `oversample`). Each point is a
    // SEPARATELY BUILT graph — quantization is baked in at construction,
    // unlike `search_expansion` — so this axis shares the build axis's
    // on-box-reference discipline. Every point is save+load round-tripped (not
    // measured on the fresh in-memory build) so a quantized point's rescore
    // reads back through the mmap'd `.rawf32` companion exactly the way a
    // loaded production index does — a freshly built, never-saved index has no
    // rescore companion yet, so its `get_exact` would silently fall back to
    // USearch's own lossy reconstruction and the rescore would recover nothing.
    let default_oversample =
        AnnIndexConfig::default().effective_oversample_for(StoragePrecision::Int8);
    let precision_grid: [(StoragePrecision, usize); 3] = [
        (StoragePrecision::F32, 1),
        (StoragePrecision::Int8, 1),
        (StoragePrecision::Int8, default_oversample),
    ];
    let mut precision_sweep = Vec::with_capacity(precision_grid.len());
    for (i, &(precision, oversample)) in precision_grid.iter().enumerate() {
        let base = tmp.path().join(format!("precision_{i}"));
        let t0 = Instant::now();
        let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default(), precision)?;
        for (id, v) in &corpus_rows {
            index.add(id, v)?;
        }
        index.build()?;
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        VectorIndex::save(&index, &base)?;
        let size = std::fs::metadata(base.with_extension("usearch"))?.len();
        let loaded = SidecarIndex::load(&base, &AnnIndexConfig::default(), precision)?;
        let (recall, qps) = recall_and_qps_with(&queries, &exact, |q, k| {
            if precision.needs_rescore() {
                retrieve_then_rescore(&loaded, q, k, oversample.max(1))
            } else {
                loaded.search(q, k)
            }
        })?;
        precision_sweep.push(PrecisionSweepPoint {
            precision,
            oversample,
            recall,
            build_time_ms: Measurement::measured(build_ms, "ms"),
            index_size_bytes: Measurement::measured(size as f64, "bytes"),
            search_qps: Measurement::measured(qps, "queries_per_s"),
        });
    }

    Ok(RecallSweepTier {
        backend_version: jammi_db::index::backend_version(),
        dim,
        corpus_rows: corpus_rows.len(),
        query_rows: queries.len(),
        build_sweep,
        search_sweep,
        precision_sweep,
    })
}

/// Recall@k for every k in [`RECALL_KS`] plus QPS at [`QPS_K`] over one index's
/// plain `search` — the `F32`/loaded-graph-default retrieval every build/search
/// axis point uses.
///
/// `exact[i]` is query `i`'s exact top-max-k; recall@k intersects its first `k`
/// with the retrieved top-`k`. QPS is the throughput of the `k == QPS_K` search
/// pass — measured on the very searches the recall curve already runs, so it
/// costs no extra work.
fn recall_and_qps(
    index: &SidecarIndex,
    queries: &[Vec<f32>],
    exact: &[Vec<(String, f32)>],
) -> Result<(BTreeMap<usize, Measurement>, f64), Box<dyn std::error::Error>> {
    recall_and_qps_with(queries, exact, |q, k| index.search(q, k))
}

/// Recall@k for every k in [`RECALL_KS`] plus QPS at [`QPS_K`], retrieving each
/// query's top-`k` through the caller-supplied `retrieve` closure rather than a
/// single fixed index — the shared measurement loop the build/search axes (a
/// plain `index.search`) and the precision axis (`search` or the two-stage
/// retrieve→rescore, depending on the point's precision) both drive.
fn recall_and_qps_with<F>(
    queries: &[Vec<f32>],
    exact: &[Vec<(String, f32)>],
    mut retrieve: F,
) -> Result<(BTreeMap<usize, Measurement>, f64), Box<dyn std::error::Error>>
where
    F: FnMut(&[f32], usize) -> jammi_db::error::Result<Vec<(String, f32)>>,
{
    let mut curve = BTreeMap::new();
    let mut qps = 0.0;
    for &k in &RECALL_KS {
        let t0 = Instant::now();
        let mut total = 0.0;
        for (i, q) in queries.iter().enumerate() {
            let ann = retrieve(q, k)?;
            total += recall_at_k_for_query(&ann, &exact[i], k);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        if k == QPS_K && elapsed > 0.0 {
            qps = queries.len() as f64 / elapsed;
        }
        curve.insert(
            k,
            Measurement::measured(total / queries.len() as f64, "fraction"),
        );
    }
    Ok((curve, qps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("scale")
            .join(name)
    }

    /// The sweep runs end-to-end over the committed fixture and emits a
    /// well-formed, schema-valid tier: every grid point present, every recall a
    /// fraction in [0, 1], every build point carrying a positive build time and
    /// a non-empty index. This is a SMOKE test of the sweep machinery — the
    /// committed fixture is tiny and saturated, so it asserts the *shape* of the
    /// output, NOT a recall-vs-cost curve (which is only meaningful at scale and
    /// only emitted on-box). It deliberately makes no monotonicity claim: at a
    /// few-thousand-row corpus build times are sub-millisecond and noise-
    /// dominated, so ordering across nearby knob values is not stable.
    #[tokio::test]
    async fn sweep_emits_schema_valid_tier_over_committed_fixture() {
        let corpus = fixture("corpus_vectors.parquet");
        let query = fixture("query_vectors.parquet");
        let tier = run(&corpus, &query).await.expect("sweep runs over fixture");

        assert!(
            !tier.backend_version.is_empty(),
            "backend version is recorded"
        );
        assert!(tier.dim > 0 && tier.corpus_rows > 0 && tier.query_rows > 0);
        assert_eq!(tier.build_sweep.len(), BUILD_GRID.len(), "all build points");
        assert_eq!(tier.search_sweep.len(), EF_GRID.len(), "all search points");

        for p in &tier.build_sweep {
            assert_recall_shape(&p.recall);
            assert!(
                p.build_time_ms.value.is_some_and(|v| v >= 0.0),
                "build point has a build time"
            );
            assert!(
                p.index_size_bytes.value.is_some_and(|v| v > 0.0),
                "build point has a non-empty index"
            );
            assert!(
                p.search_qps.value.is_some_and(|v| v > 0.0),
                "build point has a search throughput"
            );
        }
        for (p, &ef) in tier.search_sweep.iter().zip(EF_GRID) {
            assert_eq!(p.search_expansion, ef, "search point carries its ef");
            assert_recall_shape(&p.recall);
            assert!(
                p.search_qps.value.is_some_and(|v| v > 0.0),
                "search point has a throughput"
            );
        }
        assert_eq!(tier.precision_sweep.len(), 3, "all precision points");
        for p in &tier.precision_sweep {
            assert_recall_shape(&p.recall);
            assert!(
                p.build_time_ms.value.is_some_and(|v| v >= 0.0),
                "precision point has a build time"
            );
            assert!(
                p.index_size_bytes.value.is_some_and(|v| v > 0.0),
                "precision point has a non-empty index"
            );
            assert!(
                p.search_qps.value.is_some_and(|v| v > 0.0),
                "precision point has a search throughput"
            );
        }
        assert_eq!(
            tier.precision_sweep[0].precision,
            jammi_db::config::StoragePrecision::F32
        );
        assert_eq!(
            tier.precision_sweep[1].precision,
            jammi_db::config::StoragePrecision::Int8
        );
        assert_eq!(tier.precision_sweep[1].oversample, 1, "no-rescore point");
        assert_eq!(
            tier.precision_sweep[2].precision,
            jammi_db::config::StoragePrecision::Int8
        );
        assert!(
            tier.precision_sweep[2].oversample > 1,
            "shipped-default point oversamples"
        );
    }

    fn assert_recall_shape(recall: &BTreeMap<usize, Measurement>) {
        for &k in &RECALL_KS {
            let m = recall.get(&k).expect("recall has every k");
            let v = m.value.expect("recall is measured");
            assert!((0.0..=1.0).contains(&v), "recall@{k} = {v} is a fraction");
        }
    }
}
