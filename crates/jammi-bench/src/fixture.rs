//! Building the committed held-out recall fixture from the real scale cache.
//!
//! The fixture under `crates/jammi-bench/fixtures/scale/` is a *provable
//! projection* of the full Git-LFS scale cache (corpus + frozen sidecar +
//! held-out queries): a deterministic sorted-`_row_id` subset of the same real
//! embeddings, carved small enough to ship in the engine git object store (no
//! LFS) yet measured on real vectors. It is built once, off-box, by this module
//! and committed; CI only ever *loads* it (the recall gate in [`crate::recall`]
//! never rebuilds the sidecar — USearch's default build is nondeterministic).
//!
//! The build is a closed function of its inputs (the two source parquets and the
//! two subset counts) under a sorted-`_row_id` projection, so re-running it on
//! the same cache reproduces the same corpus and query slices. The frozen
//! sidecar itself is *not* reproducible bit-for-bit (the nondeterministic build
//! is exactly why it is frozen and committed once), so the committed `.usearch`
//! is the single authority — this builder writes it once.
//!
//! ## Provenance recorded
//!
//! The builder writes `floor.json` with the recall@k *measured* on the slice and
//! the margin-subtracted floor the gate asserts, plus the slice provenance
//! (source counts, subset counts, the engine SHA is recorded in the commit). The
//! floor is the measured recall minus a fixed safety margin, so the gate has
//! headroom against load-path or USearch-version drift without becoming
//! vacuous — it is `measured − margin`, never an invented round number.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use jammi_db::config::AnnIndexConfig;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;

use crate::corpus;
use crate::recall;
use crate::report::RECALL_KS;

/// Safety margin subtracted from the measured slice recall to set the committed
/// floor: `floor = measured − MARGIN`.
///
/// The gate asserts `recall@k >= floor`, so the margin is the headroom the
/// frozen index has against load-path or USearch-version drift before the gate
/// trips. Matched to the full-cache golden's margin (0.04) so the small slice
/// and the 170k gate carry the same discipline — the floor is never the bare
/// measured number (which would trip on any drift) nor an invented round value.
const FLOOR_MARGIN: f64 = 0.04;

/// The committed floor record: per-k measured recall and the margin-subtracted
/// floor the gate asserts, plus the slice provenance.
///
/// This is the on-disk `floor.json` the gate reads. Every floor is a real
/// measurement minus [`FLOOR_MARGIN`]; nothing here is invented.
#[derive(Debug, Serialize)]
pub struct FloorRecord {
    /// How the slice was carved from the full cache — the audit trail for "is
    /// this floor real".
    pub provenance: Provenance,
    /// The margin subtracted from each measured recall to set its floor.
    pub margin: f64,
    /// Per-k floor record, keyed by k (serializes ascending).
    pub recall: BTreeMap<usize, FloorEntry>,
}

/// One k's measured recall and the floor derived from it.
#[derive(Debug, Serialize)]
pub struct FloorEntry {
    /// The recall@k measured on this committed slice (held-out queries vs. the
    /// frozen sidecar over the slice corpus).
    pub measured: f64,
    /// The floor the gate asserts: `measured − margin`, clamped at 0.
    pub floor: f64,
}

/// The slice's provenance — what was subset from what.
#[derive(Debug, Serialize)]
pub struct Provenance {
    /// Total corpus rows in the source cache.
    pub source_corpus_rows: usize,
    /// Total held-out query rows in the source cache.
    pub source_query_rows: usize,
    /// Corpus rows in this slice (first N by sorted `_row_id`).
    pub slice_corpus_rows: usize,
    /// Held-out query rows in this slice (first M by sorted `_row_id`).
    pub slice_query_rows: usize,
    /// Embedding dimensionality.
    pub dim: usize,
    /// Human-readable description of the projection.
    pub note: &'static str,
}

/// File name of the committed floor record.
const FLOOR_FILE: &str = "floor.json";

/// Build the held-out recall fixture into `out_dir` from the full scale cache.
///
/// Reads the source corpus and held-out query parquets, takes the deterministic
/// first-`corpus_n` corpus rows and first-`query_n` query rows by sorted
/// `_row_id`, writes them as the fixture's `corpus_vectors.parquet` /
/// `query_vectors.parquet`, freezes a sidecar over the corpus slice (the ONE
/// build — committed and never rebuilt), measures the held-out recall@k over the
/// slice, and writes `floor.json` with `floor = measured − margin`.
///
/// Run off-box once with `RAYON_NUM_THREADS=1` so the one sidecar build is
/// single-threaded; the resulting bundle is committed and CI only loads it.
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

    // Write the fixture parquets.
    let corpus_out = out_dir.join("corpus_vectors.parquet");
    let query_out = out_dir.join("query_vectors.parquet");
    corpus::write_vectors(&corpus_out, &corpus_slice, dim).await?;
    corpus::write_vectors(&query_out, &query_slice, dim).await?;

    // Freeze the sidecar over the corpus slice — the ONE build, committed, never
    // rebuilt by the gate.
    let sidecar_base = out_dir.join("frozen");
    freeze_sidecar(&sidecar_base, &corpus_slice, dim)?;

    // Measure the held-out recall over the freshly built fixture, then derive the
    // floor as measured − margin.
    let curve = recall::recall_curve_held_out(out_dir).await?;
    let mut recall = BTreeMap::new();
    for &k in &RECALL_KS {
        let measured = curve
            .get(&k)
            .and_then(|m| m.value)
            .ok_or_else(|| format!("recall@{k} missing from measured curve"))?;
        let floor = (measured - FLOOR_MARGIN).max(0.0);
        recall.insert(k, FloorEntry { measured, floor });
    }

    let record = FloorRecord {
        provenance: Provenance {
            source_corpus_rows,
            source_query_rows,
            slice_corpus_rows: corpus_slice.len(),
            slice_query_rows: query_slice.len(),
            dim,
            note: "deterministic first-N-by-sorted-_row_id subset of the full scale cache; \
                   corpus and queries disjoint by construction (held out in the source split)",
        },
        margin: FLOOR_MARGIN,
        recall,
    };
    std::fs::write(
        out_dir.join(FLOOR_FILE),
        serde_json::to_string_pretty(&record)?,
    )?;
    Ok(record)
}

/// Build a sidecar over `rows` and freeze it to `base`
/// (`.usearch`/`.rowmap`/`.manifest.json`). This is the one build the committed
/// fixture carries; the recall gate only ever loads what this writes.
fn freeze_sidecar(
    base: &Path,
    rows: &[(String, Vec<f32>)],
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default())?;
    for (id, v) in rows {
        index.add(id, v)?;
    }
    index.build()?;
    VectorIndex::save(&index, base)?;
    Ok(())
}
