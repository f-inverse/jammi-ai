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

use jammi_db::config::{AnnIndexConfig, StoragePrecision};
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
pub(crate) const FLOOR_MARGIN: f64 = 0.04;

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
    let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::F32)?;
    for (id, v) in rows {
        index.add(id, v)?;
    }
    index.build()?;
    VectorIndex::save(&index, base)?;
    Ok(())
}

/// File stem of the frozen `Int8` sidecar bundle, alongside the existing
/// `frozen` (`F32`) stem — the SAME held-out fixture corpus, quantized.
const FROZEN_INT8_STEM: &str = "frozen_int8";

/// The committed precision-recall floor record: the retrieve→rescore recovery
/// proof, measured on the same fixture slice [`build_held_out_fixture`] froze
/// the `F32` bundle over.
///
/// Two variants, both `Int8` (the shipped quantized precision), differing only
/// in the query-time `oversample`: [`Self::int8_rescored`] is the deployment
/// default (a wide candidate pool, exactly rescored), [`Self::int8_no_rescore`]
/// is `oversample = 1` (the naive quantized-graph-only result — nothing for
/// the rescore to recover, since it retrieves no more candidates than the
/// request asks for). The gap between them is the recall the retrieve→rescore
/// design recovers from quantization.
#[derive(Debug, Serialize)]
pub struct PrecisionFloorRecord {
    /// The retrieve→rescore oversample [`Self::int8_rescored`] measured at —
    /// the deployment's stamped default
    /// ([`jammi_db::config::AnnIndexConfig::oversample`]).
    pub oversample_rescored: usize,
    /// The oversample [`Self::int8_no_rescore`] measured at — always `1`.
    pub oversample_no_rescore: usize,
    /// Int8 + retrieve→rescore at `oversample_rescored`: measured recall@k and
    /// the margin-subtracted floor, keyed by k.
    pub int8_rescored: BTreeMap<usize, FloorEntry>,
    /// Int8 at `oversample_no_rescore = 1`: measured recall@k and the
    /// margin-subtracted floor, keyed by k.
    pub int8_no_rescore: BTreeMap<usize, FloorEntry>,
}

/// Build the frozen `Int8` sidecar over the ALREADY-COMMITTED fixture corpus
/// (`fixture_dir/corpus_vectors.parquet` — the same real embeddings the
/// existing frozen `F32` bundle indexes), freeze it to
/// `fixture_dir/frozen_int8.*`, measure the held-out recall@k for the
/// two-stage retrieve→rescore (at the deployment's default oversample) and the
/// naive no-rescore (`oversample = 1`) variants, and merge a `"precision"`
/// section into the existing committed `floor.json` (`floor = measured −
/// margin`, same discipline as [`build_held_out_fixture`]).
///
/// Unlike [`build_held_out_fixture`], this needs no access to the full
/// Git-LFS source cache — the `F32` fixture (built from that source) must
/// already be committed under `fixture_dir`; this function only reads it back
/// and adds the quantized-precision bundle + floor alongside it. Run off-box
/// once with `RAYON_NUM_THREADS=1` (the one Int8 sidecar build is
/// single-threaded, matching the frozen `F32` build); CI only ever loads what
/// this writes.
pub async fn build_precision_recall_fixture(
    fixture_dir: &Path,
) -> Result<PrecisionFloorRecord, Box<dyn std::error::Error>> {
    const CORPUS_TABLE: &str = "precision_fixture_corpus";
    const QUERY_TABLE: &str = "precision_fixture_queries";

    let corpus_path = fixture_dir.join("corpus_vectors.parquet");
    let query_path = fixture_dir.join("query_vectors.parquet");

    let corpus_url = corpus::storage_url(&corpus_path)?;
    let ctx = corpus::register(&corpus_url, CORPUS_TABLE).await?;
    let corpus_rows = corpus::load_vectors(&ctx, CORPUS_TABLE).await?;
    let dim = corpus_rows
        .first()
        .map(|(_, v)| v.len())
        .ok_or("fixture corpus is empty — nothing to quantize")?;

    let query_url = corpus::storage_url(&query_path)?;
    let query_ctx = corpus::register(&query_url, QUERY_TABLE).await?;
    let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, QUERY_TABLE)
        .await?
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    if queries.is_empty() {
        return Err("fixture query set is empty — nothing to measure recall over".into());
    }

    // Freeze the Int8 sidecar over the SAME corpus the committed F32 bundle
    // indexes — the ONE build, committed, never rebuilt by the gate.
    let int8_base = fixture_dir.join(FROZEN_INT8_STEM);
    let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Int8)?;
    for (id, v) in &corpus_rows {
        index.add(id, v)?;
    }
    index.build()?;
    VectorIndex::save(&index, &int8_base)?;

    let oversample_rescored =
        AnnIndexConfig::default().effective_oversample_for(StoragePrecision::Int8);
    let mut int8_rescored = BTreeMap::new();
    let mut int8_no_rescore = BTreeMap::new();
    for &k in &RECALL_KS {
        let rescored = recall::mean_recall_at_k_rescored(
            &ctx,
            CORPUS_TABLE,
            &int8_base,
            StoragePrecision::Int8,
            oversample_rescored,
            &queries,
            k,
        )
        .await?;
        int8_rescored.insert(
            k,
            FloorEntry {
                measured: rescored,
                floor: (rescored - FLOOR_MARGIN).max(0.0),
            },
        );

        let no_rescore = recall::mean_recall_at_k_rescored(
            &ctx,
            CORPUS_TABLE,
            &int8_base,
            StoragePrecision::Int8,
            1,
            &queries,
            k,
        )
        .await?;
        int8_no_rescore.insert(
            k,
            FloorEntry {
                measured: no_rescore,
                floor: (no_rescore - FLOOR_MARGIN).max(0.0),
            },
        );
    }

    let record = PrecisionFloorRecord {
        oversample_rescored,
        oversample_no_rescore: 1,
        int8_rescored,
        int8_no_rescore,
    };

    // Merge into the existing `floor.json` rather than overwrite it wholesale
    // — the F32 `"recall"`/`"provenance"` section is written by
    // `build_held_out_fixture` from the full LFS source, which this function
    // has no access to.
    let floor_path = fixture_dir.join(FLOOR_FILE);
    let existing = std::fs::read_to_string(&floor_path)?;
    let mut value: serde_json::Value = serde_json::from_str(&existing)?;
    value
        .as_object_mut()
        .ok_or("floor.json is not a JSON object")?
        .insert("precision".to_string(), serde_json::to_value(&record)?);
    std::fs::write(&floor_path, serde_json::to_string_pretty(&value)?)?;

    Ok(record)
}
