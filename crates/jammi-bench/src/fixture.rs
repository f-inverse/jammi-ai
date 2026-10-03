//! Carving the committed held-out recall slice, and measuring the floors the
//! recall gate asserts over it.
//!
//! The slice under `crates/jammi-bench/fixtures/scale/` is a *provable
//! projection* of the full Git-LFS scale cache (corpus + held-out queries): a
//! deterministic sorted-`_row_id` subset of the same real embeddings, carved
//! small enough to ship in the engine git object store (no LFS) yet measured on
//! real vectors. [`build_held_out_fixture`] carves it once, off-box; the carve is
//! a closed function of its inputs (the two source parquets and the two subset
//! counts), so re-running it on the same cache reproduces the same slice.
//!
//! The slice carries no index. The gate ([`crate::recall`]) builds every table
//! it measures from the slice with the engine under test, and
//! [`measure_floors`] builds the same tables to measure the floors the gate
//! asserts — so a floor is re-measured, never carried over, when the engine's
//! builds change on purpose.
//!
//! ## The floor record
//!
//! `floor.json` is one [`FloorRecord`], written whole: the slice's provenance
//! (source counts, subset counts; the engine SHA is recorded in the commit),
//! and each recall@k *measured* on the slice beside the margin-subtracted
//! floor the gate asserts. A floor is the measured recall (or its bootstrap
//! CI's lower bound) minus a fixed safety margin, so the gate has headroom
//! against platform drift without becoming vacuous — never an invented round
//! number.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use jammi_db::config::StoragePrecision;

use crate::corpus;
use crate::recall::{self, Anchor, Recall, SINGLE_GRAPH};

/// Safety margin subtracted from a measured slice recall to set its committed
/// floor: `floor = anchored − MARGIN`.
///
/// The gate asserts `recall@k >= floor`, so the margin is the headroom a build
/// has before the gate trips — against float distance kernels that round
/// differently on another instruction set, not against a changed engine (a
/// deliberate change re-measures the floors). Never the bare measured number
/// nor an invented round value.
pub(crate) const FLOOR_MARGIN: f64 = 0.04;

/// Fixed headroom added atop the measured worst-case (single-graph − merged)
/// recall gap across every partitioning and k, to set the committed
/// [`SegmentFloors::single_graph_tracking_margin`] — the same "measured, then
/// pad a fixed safety amount" discipline as [`FLOOR_MARGIN`], applied to an
/// upper-bound gap rather than a lower-bound floor.
const TRACKING_MARGIN_HEADROOM: f64 = 0.02;

/// File name of the committed floor record, beside the slice.
const FLOOR_FILE: &str = "floor.json";

/// A quantized precision the floors are measured at, and the anchor its
/// floors are checked at.
pub(crate) struct QuantizedSpec {
    pub(crate) precision: StoragePrecision,
    pub(crate) anchor: Anchor,
}

/// The quantized precisions the floors are measured at — every precision
/// whose search runs a retrieve→rescore stage the floors exist to guard.
///
/// `Int8` is point-anchored. `Binary` is CI-anchored: its Hamming coarse stage
/// is noisy enough at this slice's query count that a point could pass or fail
/// on the draw of queries alone (see `recall.rs`'s "binary gate is a
/// confidence interval" section).
pub(crate) const QUANTIZED_PRECISIONS: &[QuantizedSpec] = &[
    QuantizedSpec {
        precision: StoragePrecision::Int8,
        anchor: Anchor::Point,
    },
    QuantizedSpec {
        precision: StoragePrecision::Binary,
        anchor: Anchor::CiLower,
    },
];

/// The committed `floor.json`.
#[derive(Debug, Serialize, Deserialize)]
pub struct FloorRecord {
    /// How the slice was carved from the full cache — the audit trail for "is
    /// this floor real".
    pub(crate) provenance: Provenance,
    /// The margin subtracted from each anchored recall to set its floor.
    pub(crate) margin: f64,
    /// The `F32` single graph's recall@k, keyed by k.
    pub(crate) recall: BTreeMap<usize, FloorEntry>,
    /// Each quantized precision's single graph, in [`QUANTIZED_PRECISIONS`]
    /// order.
    pub(crate) precision: Vec<QuantizedFloors>,
    /// The segment-merge floors.
    pub(crate) segment_merge: SegmentMergeFloors,
}

/// The slice's provenance — what was subset from what.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Provenance {
    /// Total corpus rows in the source cache.
    source_corpus_rows: usize,
    /// Total held-out query rows in the source cache.
    source_query_rows: usize,
    /// Corpus rows in this slice (first N by sorted `_row_id`).
    slice_corpus_rows: usize,
    /// Held-out query rows in this slice (first M by sorted `_row_id`).
    slice_query_rows: usize,
    /// Embedding dimensionality.
    dim: usize,
    /// Human-readable description of the projection.
    note: String,
}

/// One k's measured recall and the floor derived from it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct FloorEntry {
    /// The recall@k measured on the slice — the point estimate.
    pub(crate) measured: f64,
    /// The floor the gate asserts: the anchored recall minus
    /// [`FLOOR_MARGIN`], clamped at 0.
    pub(crate) floor: f64,
}

impl FloorEntry {
    fn of(recall: Recall) -> Self {
        Self {
            measured: recall.point,
            floor: (recall.anchored - FLOOR_MARGIN).max(0.0),
        }
    }
}

/// One quantized precision's single-graph floors: the retrieve→rescore
/// recovery proof.
///
/// The two curves differ only in the query-time oversample: `rescored` is the
/// deployment default (a wide candidate pool, exactly rescored), `no_rescore`
/// is `oversample = 1` (the naive quantized-graph-only result — nothing for
/// the rescore to recover, since it retrieves no more candidates than the
/// request asks for). The gap between them is the recall the retrieve→rescore
/// design recovers from quantization.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct QuantizedFloors {
    pub(crate) precision: StoragePrecision,
    /// Which value of each measurement its floor is anchored to.
    pub(crate) anchor: Anchor,
    /// The oversample `rescored` is measured at — this precision's
    /// [`StoragePrecision::default_oversample`].
    pub(crate) oversample: usize,
    /// recall@k at `oversample`, keyed by k.
    ///
    /// For `Binary` on this 2000-row slice, `k = 100` at `oversample = 32`
    /// approaches exhaustive (`k * oversample = 3200 > 2000` corpus rows), so
    /// its recovery there is partly a small-slice artifact rather than a
    /// property that holds at production scale.
    pub(crate) rescored: BTreeMap<usize, FloorEntry>,
    /// recall@k at `oversample = 1`, keyed by k.
    pub(crate) no_rescore: BTreeMap<usize, FloorEntry>,
}

/// The segment-merge floors.
///
/// Measured only at the quantized precisions: at `F32` `search_final` never
/// enters its rescore stage, so the merge's own per-segment over-fetch
/// (`jammi_db::index::segment::DEFAULT_SEGMENT_OVERFETCH_FACTOR`) is the only
/// thing between a segment's own HNSW recall and the merged answer — at this
/// slice's 1000-row-per-segment scale that floor sits at (or above) `1.0`,
/// guarding nothing. `Int8`/`Binary` segments are lossy enough on their own
/// that the retrieve→rescore-in-merge design has real work to do.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SegmentMergeFloors {
    /// Segments each partitioning splits the corpus into.
    pub(crate) segment_count: usize,
    /// Each quantized precision's floors, in [`QUANTIZED_PRECISIONS`] order.
    pub(crate) precisions: Vec<SegmentFloors>,
}

/// One quantized precision's segment-merge floors.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SegmentFloors {
    pub(crate) precision: StoragePrecision,
    /// Which value of each measurement its floor is anchored to.
    pub(crate) anchor: Anchor,
    /// The `search_final` retrieve→rescore oversample the merge is measured
    /// at — this precision's [`StoragePrecision::default_oversample`].
    pub(crate) oversample: usize,
    /// The single graph's recall@k at the same precision and oversample the
    /// tracking margin was derived from — recorded for the reader; the gate
    /// re-measures it.
    pub(crate) single_graph: BTreeMap<usize, f64>,
    /// The worst observed (single_graph − merged) gap across every
    /// partitioning and k, plus [`TRACKING_MARGIN_HEADROOM`].
    pub(crate) single_graph_tracking_margin: f64,
    /// Each partitioning's measured recall@k and its floor, keyed by
    /// partitioning name then k.
    pub(crate) partitionings: BTreeMap<String, BTreeMap<usize, FloorEntry>>,
}

/// Carve the held-out recall slice into `out_dir` from the full scale cache,
/// then measure its floors.
///
/// Reads the source corpus and held-out query parquets, takes the deterministic
/// first-`corpus_n` corpus rows and first-`query_n` query rows by sorted
/// `_row_id`, writes them as the slice's corpus and query parquets, and writes
/// the slice's [`FloorRecord`].
pub async fn build_held_out_fixture(
    corpus_src: &Path,
    query_src: &Path,
    out_dir: &Path,
    corpus_n: usize,
    query_n: usize,
) -> Result<FloorRecord, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out_dir)?;

    // Read both source sets back through the engine load path.
    let corpus_url = corpus::storage_url(corpus_src)?;
    let corpus_ctx = corpus::register(&corpus_url, "src_corpus").await?;
    let source_corpus = corpus::load_vectors(&corpus_ctx, "src_corpus").await?;

    let query_url = corpus::storage_url(query_src)?;
    let query_ctx = corpus::register(&query_url, "src_queries").await?;
    let source_queries = corpus::load_vectors(&query_ctx, "src_queries").await?;

    let source_corpus_rows = source_corpus.len();
    let source_query_rows = source_queries.len();
    let dim = source_corpus
        .first()
        .map(|(_, v)| v.len())
        .ok_or("source corpus is empty — nothing to subset")?;

    // Deterministic sorted-`_row_id` projections — the same slice on any box.
    let corpus_slice = corpus::sorted_row_id_subset(source_corpus, corpus_n);
    let query_slice = corpus::sorted_row_id_subset(source_queries, query_n);
    if corpus_slice.is_empty() || query_slice.is_empty() {
        return Err("subset counts yield an empty corpus or query slice".into());
    }

    // Verify the held-out invariant on the slice: no query id is in the corpus.
    let corpus_ids: std::collections::HashSet<&str> =
        corpus_slice.iter().map(|(id, _)| id.as_str()).collect();
    if let Some((id, _)) = query_slice
        .iter()
        .find(|(id, _)| corpus_ids.contains(id.as_str()))
    {
        return Err(format!(
            "query id {id} is also in the corpus slice — the query set is not held out"
        )
        .into());
    }

    corpus::write_vectors(
        &out_dir.join(recall::HELD_OUT_CORPUS_FILE),
        &corpus_slice,
        dim,
    )
    .await?;
    corpus::write_vectors(
        &out_dir.join(recall::HELD_OUT_QUERY_FILE),
        &query_slice,
        dim,
    )
    .await?;

    let provenance = Provenance {
        source_corpus_rows,
        source_query_rows,
        slice_corpus_rows: corpus_slice.len(),
        slice_query_rows: query_slice.len(),
        dim,
        note: "deterministic first-N-by-sorted-_row_id subset of the full scale cache; \
               corpus and queries disjoint by construction (held out in the source split)"
            .to_string(),
    };
    write_floors(out_dir, provenance).await
}

/// Re-measure the floors of the committed slice under `fixture_dir` with the
/// engine under test, keeping its provenance, and rewrite its [`FloorRecord`].
/// Run it after a deliberate change to how the engine builds or searches an
/// index.
pub async fn measure_floors(fixture_dir: &Path) -> Result<FloorRecord, Box<dyn std::error::Error>> {
    /// The one field a re-measure keeps: the floors it replaces need not parse.
    #[derive(Deserialize)]
    struct Carved {
        provenance: Provenance,
    }
    let Carved { provenance } = read_floor_json(fixture_dir)?;
    write_floors(fixture_dir, provenance).await
}

/// The `floor.json` committed beside the slice under `fixture_dir`, read as
/// `T` — the whole [`FloorRecord`], or the part of it a reader needs.
pub(crate) fn read_floor_json<T: serde::de::DeserializeOwned>(
    fixture_dir: &Path,
) -> Result<T, Box<dyn std::error::Error>> {
    let path = fixture_dir.join(FLOOR_FILE);
    let json =
        std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&json).map_err(|e| format!("parsing {}: {e}", path.display()))?)
}

/// Floors for a measured curve.
fn floors(curve: &BTreeMap<usize, Recall>) -> BTreeMap<usize, FloorEntry> {
    curve
        .iter()
        .map(|(&k, &recall)| (k, FloorEntry::of(recall)))
        .collect()
}

/// Build every table the gate builds from the slice under `dir` — the `F32`
/// single graph, and each quantized precision's single graph and segment
/// partitionings — measure their held-out recall@k, and write the slice's
/// [`FloorRecord`] under `provenance`.
async fn write_floors(
    dir: &Path,
    provenance: Provenance,
) -> Result<FloorRecord, Box<dyn std::error::Error>> {
    let slice = recall::load_slice(dir, "floors").await?;
    let work = tempfile::tempdir()?;

    let f32_graph = slice.build(work.path(), StoragePrecision::F32, &SINGLE_GRAPH)?;
    let f32_floors = floors(&slice.recall_curve(&f32_graph, 1, Anchor::Point)?);

    let mut precision = Vec::with_capacity(QUANTIZED_PRECISIONS.len());
    let mut segment_precisions = Vec::with_capacity(QUANTIZED_PRECISIONS.len());
    for spec in QUANTIZED_PRECISIONS {
        let oversample = spec.precision.default_oversample();
        let single = slice.build(work.path(), spec.precision, &SINGLE_GRAPH)?;
        let rescored = slice.recall_curve(&single, oversample, spec.anchor)?;
        let no_rescore = slice.recall_curve(&single, 1, spec.anchor)?;

        let mut partitionings = BTreeMap::new();
        let mut worst_gap = 0.0f64;
        for part in recall::SEGMENT_PARTITIONINGS {
            let segmented = slice.build(work.path(), spec.precision, part)?;
            let merged = slice.recall_curve(&segmented, oversample, spec.anchor)?;
            worst_gap = merged
                .iter()
                .map(|(k, r)| rescored[k].point - r.point)
                .fold(worst_gap, f64::max);
            partitionings.insert(part.name.to_string(), floors(&merged));
        }

        segment_precisions.push(SegmentFloors {
            precision: spec.precision,
            anchor: spec.anchor,
            oversample,
            single_graph: rescored.iter().map(|(&k, r)| (k, r.point)).collect(),
            single_graph_tracking_margin: worst_gap + TRACKING_MARGIN_HEADROOM,
            partitionings,
        });
        precision.push(QuantizedFloors {
            precision: spec.precision,
            anchor: spec.anchor,
            oversample,
            rescored: floors(&rescored),
            no_rescore: floors(&no_rescore),
        });
    }

    let record = FloorRecord {
        provenance,
        margin: FLOOR_MARGIN,
        recall: f32_floors,
        precision,
        segment_merge: SegmentMergeFloors {
            segment_count: recall::SEGMENT_COUNT,
            precisions: segment_precisions,
        },
    };
    std::fs::write(
        dir.join(FLOOR_FILE),
        serde_json::to_string_pretty(&record)? + "\n",
    )?;
    Ok(record)
}
