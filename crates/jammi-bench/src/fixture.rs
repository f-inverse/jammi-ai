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
use serde_json::value::RawValue;

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

/// File stem of the frozen `Binary` sidecar bundle, alongside `frozen` (`F32`)
/// and `frozen_int8` — the SAME held-out fixture corpus, sign-quantized to one
/// bit per dimension (USearch `B1`) and searched by Hamming distance.
const FROZEN_BINARY_STEM: &str = "frozen_binary";

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

/// The committed `Binary`-precision floor record — the same retrieve→rescore
/// recovery proof as [`PrecisionFloorRecord`], for [`StoragePrecision::Binary`]
/// rather than `Int8`.
///
/// `Binary`'s Hamming coarse stage needs a much wider oversample than `Int8`'s
/// cosine-ranked stage to recover comparable recall (see
/// [`jammi_db::config::StoragePrecision::default_oversample`]), so its
/// rescored oversample is its own field
/// ([`Self::binary_oversample_rescored`]) rather than sharing
/// [`PrecisionFloorRecord::oversample_rescored`], which stays Int8's `4`.
#[derive(Debug, Serialize)]
pub struct BinaryPrecisionFloorRecord {
    /// The retrieve→rescore oversample [`Self::binary_rescored`] measured at
    /// — `Binary`'s own
    /// [`jammi_db::config::StoragePrecision::default_oversample`] (`32`).
    pub binary_oversample_rescored: usize,
    /// The oversample [`Self::binary_no_rescore`] measured at — always `1`.
    pub binary_oversample_no_rescore: usize,
    /// Binary + retrieve→rescore at `binary_oversample_rescored`: measured
    /// recall@k and the margin-subtracted floor, keyed by k.
    ///
    /// On this 2000-row fixture, `k=100` at `oversample=32` approaches
    /// exhaustive (`k * oversample = 3200 > 2000` corpus rows), so its
    /// recovery is partly a small-fixture artifact rather than a property that
    /// holds at production scale — the floor is still measured live and
    /// margin-subtracted, but should not be over-read as "binary rescue
    /// recovers 100% of top-100 in general".
    pub binary_rescored: BTreeMap<usize, FloorEntry>,
    /// Binary at `binary_oversample_no_rescore = 1`: measured recall@k and the
    /// margin-subtracted floor, keyed by k.
    pub binary_no_rescore: BTreeMap<usize, FloorEntry>,
}

/// One quantized precision's measured retrieve→rescore recall@k curve (both
/// the deployment-default-oversample "rescored" variant and the
/// `oversample = 1` "no rescore" baseline), each already converted to a
/// [`FloorEntry`] (`floor = measured − margin`).
struct QuantizedRecallMeasurement {
    rescored: BTreeMap<usize, FloorEntry>,
    no_rescore: BTreeMap<usize, FloorEntry>,
}

/// Freeze a sidecar at `precision` over the ALREADY-COMMITTED fixture corpus
/// (`fixture_dir/corpus_vectors.parquet`) to `fixture_dir/{stem}.*` — the ONE
/// build, committed, never rebuilt by the gate — then measure the held-out
/// retrieve→rescore recall@k at `oversample_rescored` and at the naive
/// `oversample = 1` baseline.
///
/// Shared by [`build_precision_recall_fixture`] (`Int8`) and
/// [`build_binary_recall_fixture`] (`Binary`) — the two quantized precisions
/// differ only in their scalar kind and default oversample, not in how the
/// fixture is built or measured.
async fn freeze_and_measure_quantized_precision(
    fixture_dir: &Path,
    stem: &str,
    precision: StoragePrecision,
    oversample_rescored: usize,
) -> Result<QuantizedRecallMeasurement, Box<dyn std::error::Error>> {
    let corpus_table = format!("{stem}_fixture_corpus");
    let query_table = format!("{stem}_fixture_queries");

    let corpus_path = fixture_dir.join("corpus_vectors.parquet");
    let query_path = fixture_dir.join("query_vectors.parquet");

    let corpus_url = corpus::storage_url(&corpus_path)?;
    let ctx = corpus::register(&corpus_url, &corpus_table).await?;
    let corpus_rows = corpus::load_vectors(&ctx, &corpus_table).await?;
    let dim = corpus_rows
        .first()
        .map(|(_, v)| v.len())
        .ok_or("fixture corpus is empty — nothing to quantize")?;

    let query_url = corpus::storage_url(&query_path)?;
    let query_ctx = corpus::register(&query_url, &query_table).await?;
    let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, &query_table)
        .await?
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    if queries.is_empty() {
        return Err("fixture query set is empty — nothing to measure recall over".into());
    }

    // Freeze the quantized sidecar over the SAME corpus the committed F32
    // bundle indexes — the ONE build, committed, never rebuilt by the gate.
    let base = fixture_dir.join(stem);
    let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default(), precision)?;
    for (id, v) in &corpus_rows {
        index.add(id, v)?;
    }
    index.build()?;
    VectorIndex::save(&index, &base)?;

    let mut rescored = BTreeMap::new();
    let mut no_rescore = BTreeMap::new();
    for &k in &RECALL_KS {
        let rescored_recall = recall::mean_recall_at_k_rescored(
            &ctx,
            &corpus_table,
            &base,
            precision,
            oversample_rescored,
            &queries,
            k,
        )
        .await?;
        rescored.insert(
            k,
            FloorEntry {
                measured: rescored_recall,
                floor: (rescored_recall - FLOOR_MARGIN).max(0.0),
            },
        );

        let no_rescore_recall = recall::mean_recall_at_k_rescored(
            &ctx,
            &corpus_table,
            &base,
            precision,
            1,
            &queries,
            k,
        )
        .await?;
        no_rescore.insert(
            k,
            FloorEntry {
                measured: no_rescore_recall,
                floor: (no_rescore_recall - FLOOR_MARGIN).max(0.0),
            },
        );
    }

    Ok(QuantizedRecallMeasurement {
        rescored,
        no_rescore,
    })
}

/// Merge `fields` (a serialized record, top-level keys only) into the
/// `"precision"` object of the committed `floor.json`, creating that object if
/// it does not already exist. Additive — existing keys the caller's record
/// does not touch (e.g. another precision's floors) are left untouched, so
/// [`build_precision_recall_fixture`] and [`build_binary_recall_fixture`] can
/// each be run independently without clobbering the other's committed floors.
///
/// Every sibling key — both at the top level (`"margin"`/`"provenance"`/
/// `"recall"`) and inside the existing `"precision"` object (e.g. `Int8`'s
/// `int8_rescored`/`int8_no_rescore` when this call is writing `Binary`'s
/// keys) — is carried through as opaque raw JSON text ([`RawValue`]), never
/// reparsed into an `f64`. `serde_json`'s default number parser is not
/// exact-round-trip (the `float_roundtrip` cargo feature exists precisely
/// because reparsing an arbitrary decimal string can land 1 ULP off the
/// original value), so a naive read-into-`Value`-then-rewrite would silently
/// perturb an already-committed measured float that this call never touched
/// or re-measured — a measured number quietly turning into a different,
/// unmeasured one. Only the keys `fields` supplies are freshly written, from
/// numbers this run just measured; every other key's bytes are exactly what
/// was already committed.
fn merge_precision_floor_fields(
    fixture_dir: &Path,
    fields: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let floor_path = fixture_dir.join(FLOOR_FILE);
    let existing = std::fs::read_to_string(&floor_path)?;

    let mut top: BTreeMap<String, Box<RawValue>> = serde_json::from_str(&existing)?;
    let mut precision: BTreeMap<String, Box<RawValue>> = match top.get("precision") {
        Some(raw) => serde_json::from_str(raw.get())?,
        None => BTreeMap::new(),
    };

    // `"precision"`'s children sit one level deeper (4 spaces) than a
    // freshly-`to_string_pretty`'d value assumes (which starts its own
    // indentation at column 0, as if it were the whole document) — reindent
    // only the NEW value's lines by the difference. Untouched keys are never
    // touched here, so they never need reindenting: their raw bytes were
    // captured at, and are spliced back at, the exact same absolute depth.
    let fields_map = fields
        .as_object()
        .ok_or("precision floor record did not serialize to a JSON object")?;
    for (k, v) in fields_map {
        let rendered = indent_continuation_lines(&serde_json::to_string_pretty(v)?, 4);
        precision.insert(k.clone(), RawValue::from_string(rendered)?);
    }

    top.insert(
        "precision".to_string(),
        RawValue::from_string(render_object(&precision, 2)?)?,
    );

    std::fs::write(&floor_path, render_object(&top, 0)?)?;
    Ok(())
}

/// Prepend `spaces` worth of indentation to every line of `s` EXCEPT the
/// first — the first line follows immediately after `"key": ` on the same
/// line, so it needs no leading whitespace of its own; every subsequent line
/// is a fresh line that must line up at the target nesting depth.
fn indent_continuation_lines(s: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    s.lines()
        .enumerate()
        .map(|(i, line)| {
            if i == 0 {
                line.to_string()
            } else {
                format!("{pad}{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render `map` as a pretty-printed JSON object whose entries sit at
/// `indent + 2` spaces (the entries' own indentation; `indent` is the
/// indentation of the object's opening/closing braces themselves), splicing
/// each value's raw JSON bytes in verbatim.
///
/// Unlike `serde_json::to_string_pretty` over the map directly (which always
/// assumes the map is the top of its own document, indenting entries starting
/// at 2 spaces regardless of where the object actually nests), this renders at
/// the CALLER-SUPPLIED depth — the mechanism [`merge_precision_floor_fields`]
/// needs to splice an already-nested object (`"precision"`, or the whole
/// document) without disturbing the absolute indentation already baked into
/// untouched [`RawValue`] entries.
fn render_object(
    map: &BTreeMap<String, Box<RawValue>>,
    indent: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let entry_pad = " ".repeat(indent + 2);
    let mut out = String::from("{\n");
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str(&entry_pad);
        out.push_str(&serde_json::to_string(k)?);
        out.push_str(": ");
        out.push_str(v.get());
    }
    out.push('\n');
    out.push_str(&" ".repeat(indent));
    out.push('}');
    if indent == 0 {
        out.push('\n');
    }
    Ok(out)
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
    let oversample_rescored = StoragePrecision::Int8.default_oversample();
    let measurement = freeze_and_measure_quantized_precision(
        fixture_dir,
        FROZEN_INT8_STEM,
        StoragePrecision::Int8,
        oversample_rescored,
    )
    .await?;

    let record = PrecisionFloorRecord {
        oversample_rescored,
        oversample_no_rescore: 1,
        int8_rescored: measurement.rescored,
        int8_no_rescore: measurement.no_rescore,
    };
    merge_precision_floor_fields(fixture_dir, serde_json::to_value(&record)?)?;
    Ok(record)
}

/// Build the frozen `Binary` sidecar over the ALREADY-COMMITTED fixture corpus
/// (`fixture_dir/corpus_vectors.parquet` — the SAME real embeddings the
/// existing frozen `F32`/`Int8` bundles index), freeze it to
/// `fixture_dir/frozen_binary.*`, measure the held-out recall@k for the
/// two-stage retrieve→rescore at `Binary`'s own default oversample (`32`) and
/// the naive no-rescore (`oversample = 1`) baseline, and merge the
/// `binary_*` keys into the existing `"precision"` section of the committed
/// `floor.json` (`floor = measured − margin`, same discipline as
/// [`build_precision_recall_fixture`]).
///
/// Run off-box once with `RAYON_NUM_THREADS=1` (the one Binary sidecar build
/// is single-threaded, matching the frozen `F32`/`Int8` builds) after both the
/// `F32` and `Int8` fixtures are already committed; CI only ever loads what
/// this writes.
pub async fn build_binary_recall_fixture(
    fixture_dir: &Path,
) -> Result<BinaryPrecisionFloorRecord, Box<dyn std::error::Error>> {
    let oversample_rescored = StoragePrecision::Binary.default_oversample();
    let measurement = freeze_and_measure_quantized_precision(
        fixture_dir,
        FROZEN_BINARY_STEM,
        StoragePrecision::Binary,
        oversample_rescored,
    )
    .await?;

    let record = BinaryPrecisionFloorRecord {
        binary_oversample_rescored: oversample_rescored,
        binary_oversample_no_rescore: 1,
        binary_rescored: measurement.rescored,
        binary_no_rescore: measurement.no_rescore,
    };
    merge_precision_floor_fields(fixture_dir, serde_json::to_value(&record)?)?;
    Ok(record)
}
