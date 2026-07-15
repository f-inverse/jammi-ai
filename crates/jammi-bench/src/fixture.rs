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
use std::path::{Path, PathBuf};

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
    /// frozen sidecar over the slice corpus) — the point estimate.
    pub measured: f64,
    /// The floor the gate asserts. `measured − margin`, clamped at 0, for
    /// every point-anchored gate. [`BinaryPrecisionFloorRecord`]'s entries are
    /// the one exception — bootstrap-CI-anchored (`CI_lower − margin`), see
    /// [`build_binary_recall_fixture`].
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
    freeze_sidecar(&sidecar_base, &corpus_slice, dim, StoragePrecision::F32)?;

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

/// Build a sidecar over `rows` at `precision` and freeze it to `base`
/// (`.usearch`/`.rowmap`/`.manifest.json`, plus a `.rawf32` rescore companion
/// when `precision` is quantized). This is the one build the committed
/// fixture carries; the recall gate only ever loads what this writes.
fn freeze_sidecar(
    base: &Path,
    rows: &[(String, Vec<f32>)],
    dim: usize,
    precision: StoragePrecision,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default(), precision)?;
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
/// rather than `Int8`, but bootstrap-CI-anchored rather than point-anchored
/// (see [`build_binary_recall_fixture`]): each [`FloorEntry::floor`] here is
/// `CI_lower − margin`, not `measured − margin`.
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
    /// Binary + retrieve→rescore at `binary_oversample_rescored`: the measured
    /// POINT recall@k, and the bootstrap-CI-lower-bound-anchored floor
    /// (`CI_lower − margin`), keyed by k.
    ///
    /// On this 2000-row fixture, `k=100` at `oversample=32` approaches
    /// exhaustive (`k * oversample = 3200 > 2000` corpus rows), so its
    /// recovery is partly a small-fixture artifact rather than a property that
    /// holds at production scale — the floor is still measured live and
    /// margin-subtracted, but should not be over-read as "binary rescue
    /// recovers 100% of top-100 in general".
    pub binary_rescored: BTreeMap<usize, FloorEntry>,
    /// Binary at `binary_oversample_no_rescore = 1`: the measured POINT
    /// recall@k, and the bootstrap-CI-lower-bound-anchored floor, keyed by k.
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
/// Used by [`build_precision_recall_fixture`] (`Int8`), whose gate is
/// point-based. [`build_binary_recall_fixture`] (`Binary`) does NOT use this —
/// its gate is bootstrap-CI-based (see [`recall::recall_ci_at_k_rescored`]),
/// so it derives its `floor.json` entries from the CI's lower bound rather
/// than this helper's bare point `FloorEntry`; it freezes and measures inline
/// instead of sharing this point-only path.
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

    let fields_map = fields
        .as_object()
        .ok_or("precision floor record did not serialize to a JSON object")?;
    // `"precision"`'s children sit one level deeper (4 spaces) than a
    // freshly-`to_string_pretty`'d value assumes — see [`splice_raw_fields`].
    splice_raw_fields(&mut precision, fields_map, 4)?;

    top.insert(
        "precision".to_string(),
        RawValue::from_string(render_object(&precision, 2)?)?,
    );

    std::fs::write(&floor_path, render_object(&top, 0)?)?;
    Ok(())
}

/// Merge `fields` (a serialized record, top-level keys only) as a NEW
/// top-level key `key` of the committed `floor.json`, alongside `"margin"` /
/// `"provenance"` / `"recall"` / `"precision"`. Same additive,
/// raw-byte-preserving discipline as [`merge_precision_floor_fields`] — every
/// sibling top-level key this call does not touch is carried through
/// unreparsed — one level shallower (there is no nested container to descend
/// into first).
///
/// Used by [`build_segment_recall_fixture`] to add the `"segment_merge"`
/// section without disturbing any existing key `build_held_out_fixture` /
/// `build_precision_recall_fixture` / `build_binary_recall_fixture` already
/// wrote.
fn merge_top_level_floor_field(
    fixture_dir: &Path,
    key: &str,
    value: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let floor_path = fixture_dir.join(FLOOR_FILE);
    let existing = std::fs::read_to_string(&floor_path)?;

    let mut top: BTreeMap<String, Box<RawValue>> = serde_json::from_str(&existing)?;
    let mut wrapper = serde_json::Map::new();
    wrapper.insert(key.to_string(), value);
    // A fresh top-level key's entry sits at 2 spaces (the document root's own
    // entry depth) — see [`splice_raw_fields`].
    splice_raw_fields(&mut top, &wrapper, 2)?;

    std::fs::write(&floor_path, render_object(&top, 0)?)?;
    Ok(())
}

/// Insert each of `fields`' entries into `map` as raw JSON bytes, reindented
/// so its continuation lines land at `spaces` — the ONE reindent-and-splice
/// primitive [`merge_precision_floor_fields`] (splicing into the nested
/// `"precision"` object, `spaces = 4`) and [`merge_top_level_floor_field`]
/// (splicing a new top-level key, `spaces = 2`) share.
///
/// `serde_json`'s default number parser is not exact-round-trip (the
/// `float_roundtrip` cargo feature exists precisely because reparsing an
/// arbitrary decimal string can land 1 ULP off the original value), so a naive
/// read-into-`Value`-then-rewrite would silently perturb an already-committed
/// measured float this call never touched or re-measured — a measured number
/// quietly turning into a different, unmeasured one. Splicing raw bytes in
/// verbatim (only reindenting, never reparsing into `f64`) is what keeps every
/// untouched key byte-identical.
fn splice_raw_fields(
    map: &mut BTreeMap<String, Box<RawValue>>,
    fields: &serde_json::Map<String, serde_json::Value>,
    spaces: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for (k, v) in fields {
        let rendered = indent_continuation_lines(&serde_json::to_string_pretty(v)?, spaces);
        map.insert(k.clone(), RawValue::from_string(rendered)?);
    }
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
/// `fixture_dir/frozen_binary.*`, and merge the `binary_*` keys into the
/// existing `"precision"` section of the committed `floor.json`.
///
/// Unlike every other precision's floor ([`build_precision_recall_fixture`]'s
/// `measured − margin` on the bare point recall), `Binary`'s floor is derived
/// from a [`recall::recall_ci_at_k_rescored`] bootstrap CI:
/// `floor = CI_lower − MARGIN`. `measured` still records the point recall (for
/// readability — it is what a reader compares the floor against at a glance),
/// but the floor itself is anchored on the CI's lower bound, not the point —
/// see `recall.rs`'s module-level "binary gate is a confidence interval"
/// section for why a point-anchored floor produced a false positive at real
/// scale and a CI-anchored one does not. Prints the point/CI for both the
/// retrieve→rescore recovery (at `Binary`'s own default oversample) and the
/// naive `oversample = 1` baseline at every k, so the operator committing this
/// fixture sees the interval — not just the point — before the CI-gated
/// cargo test ever runs.
///
/// Run off-box once with `RAYON_NUM_THREADS=1` (the one Binary sidecar build
/// is single-threaded, matching the frozen `F32`/`Int8` builds) after both the
/// `F32` and `Int8` fixtures are already committed; CI only ever loads what
/// this writes.
pub async fn build_binary_recall_fixture(
    fixture_dir: &Path,
) -> Result<BinaryPrecisionFloorRecord, Box<dyn std::error::Error>> {
    let oversample_rescored = StoragePrecision::Binary.default_oversample();

    let corpus_table = format!("{FROZEN_BINARY_STEM}_fixture_corpus");
    let query_table = format!("{FROZEN_BINARY_STEM}_fixture_queries");

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

    // Freeze the Binary sidecar over the SAME corpus the committed F32/Int8
    // bundles index — the ONE build, committed, never rebuilt by the gate.
    let base = fixture_dir.join(FROZEN_BINARY_STEM);
    let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Binary)?;
    for (id, v) in &corpus_rows {
        index.add(id, v)?;
    }
    index.build()?;
    VectorIndex::save(&index, &base)?;

    eprintln!(
        "binary recall bootstrap CI ({} resamples, {:.0}% interval, seed {:#x}); \
         floor = CI_lower − {FLOOR_MARGIN}:",
        recall::RECALL_BOOTSTRAP_ITERATIONS,
        (1.0 - recall::RECALL_BOOTSTRAP_ALPHA) * 100.0,
        recall::RECALL_BOOTSTRAP_SEED
    );
    let mut binary_rescored = BTreeMap::new();
    let mut binary_no_rescore = BTreeMap::new();
    for &k in &RECALL_KS {
        let rescored = recall::recall_ci_at_k_rescored(
            &ctx,
            &corpus_table,
            &base,
            StoragePrecision::Binary,
            oversample_rescored,
            &queries,
            k,
        )
        .await?;
        let no_rescore = recall::recall_ci_at_k_rescored(
            &ctx,
            &corpus_table,
            &base,
            StoragePrecision::Binary,
            1,
            &queries,
            k,
        )
        .await?;
        eprintln!(
            "  k={k}: rescored point={:.4} ci=[{:.4}, {:.4}]  no_rescore point={:.4} ci=[{:.4}, {:.4}]",
            rescored.point,
            rescored.interval.lower,
            rescored.interval.upper,
            no_rescore.point,
            no_rescore.interval.lower,
            no_rescore.interval.upper,
        );
        binary_rescored.insert(
            k,
            FloorEntry {
                measured: rescored.point,
                floor: (rescored.interval.lower - FLOOR_MARGIN).max(0.0),
            },
        );
        binary_no_rescore.insert(
            k,
            FloorEntry {
                measured: no_rescore.point,
                floor: (no_rescore.interval.lower - FLOOR_MARGIN).max(0.0),
            },
        );
    }

    let record = BinaryPrecisionFloorRecord {
        binary_oversample_rescored: oversample_rescored,
        binary_oversample_no_rescore: 1,
        binary_rescored,
        binary_no_rescore,
    };
    merge_precision_floor_fields(fixture_dir, serde_json::to_value(&record)?)?;
    Ok(record)
}

/// How many segments each committed partitioning splits the fixture corpus
/// into. `2` is the smallest split that exercises the cross-segment merge at
/// all (a single segment is the `N = 1` byte-identity case
/// `jammi_db::index::segment`'s own unit tests already cover).
const SEGMENT_COUNT: usize = 2;

/// A committed row→segment assignment — the deterministic axis this fixture
/// varies in place of the HNSW build seed USearch does not let it pin (see
/// `recall.rs`'s module-level "segment axis" section). `assign(i, total)`
/// maps corpus row `i` of `total` (0-based, in the corpus parquet's
/// sorted-`_row_id` order) to a segment index in `0..SEGMENT_COUNT`.
struct SegmentPartitioning {
    /// Fixture file stem prefix: each segment freezes to
    /// `fixture_dir/{name}_{precision_key}_seg{i}.*`.
    name: &'static str,
    /// Row index, total row count → segment index.
    assign: fn(usize, usize) -> usize,
}

/// Contiguous halves: rows `[0, total/2)` land in segment 0, the rest in
/// segment 1 (generalizes to `SEGMENT_COUNT > 2` as even-width contiguous
/// blocks). Mirrors how an append-only table's segments actually accumulate —
/// each batch of newly-added rows is contiguous in insertion order.
fn contiguous_block_partition(i: usize, total: usize) -> usize {
    if total == 0 {
        return 0;
    }
    ((i * SEGMENT_COUNT) / total).min(SEGMENT_COUNT - 1)
}

/// Round-robin: row `i` lands in segment `i % SEGMENT_COUNT`. A structurally
/// different row→segment assignment from [`contiguous_block_partition`] over
/// the SAME rows — each corpus row's true nearest neighbours are scattered
/// across segments differently than under the contiguous split, so the two
/// partitionings are genuinely independent draws of "which segment a row's
/// neighbours end up in", not a relabeling of the same split.
fn interleaved_partition(i: usize, _total: usize) -> usize {
    i % SEGMENT_COUNT
}

/// The committed partitionings [`build_segment_recall_fixture`] freezes — at
/// least two, so the segment-merge floor holds across more than one
/// deterministic stand-in for an HNSW build seed.
const SEGMENT_PARTITIONINGS: &[SegmentPartitioning] = &[
    SegmentPartitioning {
        name: "seg_block",
        assign: contiguous_block_partition,
    },
    SegmentPartitioning {
        name: "seg_interleaved",
        assign: interleaved_partition,
    },
];

/// Fixed headroom added atop the measured worst-case (single-graph − merged)
/// recall gap across every committed partitioning and k, to set the committed
/// `single_graph_tracking_margin` — the same "measured, then pad a fixed
/// safety amount" discipline `recall.rs`'s `RESCORE_RECOVERY_MARGIN` and this
/// module's [`FLOOR_MARGIN`] both use, applied to an upper-bound gap rather
/// than a lower-bound floor.
const TRACKING_MARGIN_HEADROOM: f64 = 0.02;

/// One quantized precision [`build_segment_recall_fixture`] freezes segments
/// at and measures the merge over.
///
/// The segment-merge floor is only load-bearing at a precision where
/// [`StoragePrecision::needs_rescore`] is `true`: at `F32` `search_final` never
/// enters its rescore stage, so `oversample` is unused and the merge's own
/// per-segment over-fetch (`jammi_db::index::segment::DEFAULT_SEGMENT_OVERFETCH_FACTOR`)
/// is the only thing between a segment's own HNSW recall and the merged
/// answer — at this fixture's 1000-row-per-segment real-embedding scale that
/// floor turned out to sit at (or above) `1.0`, guarding nothing. `Int8`/
/// `Binary` segments are lossy enough on their own (see
/// [`build_precision_recall_fixture`] / [`build_binary_recall_fixture`]'s
/// single-graph numbers) that the retrieve→rescore-in-merge design has real
/// work to do, so these are the precisions this fixture's floor is measured
/// at.
struct SegmentPrecisionSpec {
    /// `floor.json` key under `"segment_merge"."precisions"`, and the segment
    /// file-stem infix (`fixture_dir/{partitioning}_{key}_seg{i}.*`).
    key: &'static str,
    precision: StoragePrecision,
    /// Stem of the ALREADY-COMMITTED single-graph bundle at this precision
    /// (`build_precision_recall_fixture`'s `frozen_int8` /
    /// `build_binary_recall_fixture`'s `frozen_binary`) the merge is tracked
    /// against.
    single_graph_stem: &'static str,
    /// Whether the floor (and the tracking-margin comparison) is anchored to
    /// the bare point mean (`Int8`) or the bootstrap-CI lower bound
    /// (`Binary`) — mirrors `recall.rs::tests::Anchor` and
    /// [`build_binary_recall_fixture`]'s "CI, not a point estimate"
    /// discipline, since `Binary`'s Hamming coarse stage is noisy enough at
    /// this small a per-segment row count that a single-sample point could
    /// invert.
    ci_anchored: bool,
}

/// The precisions [`build_segment_recall_fixture`] measures the segment merge
/// at — both quantized, both already load-bearing at the single-graph level
/// (see [`SegmentPrecisionSpec`]'s doc for why `F32` is excluded).
const SEGMENT_PRECISIONS: &[SegmentPrecisionSpec] = &[
    SegmentPrecisionSpec {
        key: "int8",
        precision: StoragePrecision::Int8,
        single_graph_stem: FROZEN_INT8_STEM,
        ci_anchored: false,
    },
    SegmentPrecisionSpec {
        key: "binary",
        precision: StoragePrecision::Binary,
        single_graph_stem: FROZEN_BINARY_STEM,
        ci_anchored: true,
    },
];

/// One precision's committed segment-merge floor record: the deployment
/// oversample it was measured at, which anchor (`"point"` / `"ci_lower"`)
/// its floors and tracking margin are checked against, the live single-graph
/// (`N = 1`) baseline recall@k the merge is checked to track, the derived
/// tracking margin, and the per-partitioning measured recall@k + floor.
#[derive(Debug, Serialize)]
pub struct SegmentPrecisionFloorRecord {
    /// The `search_final` retrieve→rescore oversample the merge is measured
    /// at — this precision's [`StoragePrecision::default_oversample`].
    pub oversample: usize,
    /// `"point"` or `"ci_lower"` — which value [`FloorEntry::measured`] and
    /// the tracking-margin comparison are anchored to; mirrors
    /// [`SegmentPrecisionSpec::ci_anchored`].
    pub anchor: &'static str,
    /// The single-graph (`N = 1`) baseline recall@k, measured live over the
    /// SAME corpus and queries via the already-committed frozen bundle at
    /// this precision — what the merge is checked to TRACK, not a floor
    /// asserted on its own.
    pub single_graph: BTreeMap<usize, f64>,
    /// The merge-vs-single-graph tracking margin: the worst observed
    /// (single_graph − merged) gap across every partitioning and k at this
    /// precision, plus [`TRACKING_MARGIN_HEADROOM`].
    pub single_graph_tracking_margin: f64,
    /// Per-partitioning measured recall@k and its margin-subtracted floor,
    /// keyed by partitioning name then k.
    pub partitionings: BTreeMap<String, BTreeMap<usize, FloorEntry>>,
}

/// The committed segment-merge floor record: how many segments each
/// partitioning splits the corpus into, and one [`SegmentPrecisionFloorRecord`]
/// per precision in [`SEGMENT_PRECISIONS`] — the `"segment_merge"` `floor.json`
/// section this whole record serializes to.
#[derive(Debug, Serialize)]
pub struct SegmentRecallFloorRecord {
    /// Segments each partitioning splits the corpus into.
    pub segment_count: usize,
    /// Per-precision floor record, keyed by [`SegmentPrecisionSpec::key`].
    pub precisions: BTreeMap<String, SegmentPrecisionFloorRecord>,
}

/// Build the committed segment-merge recall fixture: for each precision in
/// [`SEGMENT_PRECISIONS`] (`Int8`, `Binary`), split the ALREADY-COMMITTED
/// fixture corpus (`fixture_dir/corpus_vectors.parquet` — the SAME real
/// embeddings the frozen `F32`/`Int8`/`Binary` bundles index) into
/// [`SEGMENT_COUNT`] segments under each of [`SEGMENT_PARTITIONINGS`], freeze
/// one [`SidecarIndex`] per segment AT THAT PRECISION
/// (`fixture_dir/{partitioning}_{precision}_seg{i}.*` — ONE build per segment,
/// committed, never rebuilt by the gate), assemble them into a
/// [`jammi_db::index::SegmentedIndex`], and measure its held-out
/// `search_final` recall@k against the exact oracle — the SAME
/// retrieve→rescore-in-merge path `search_final` runs at deployment, since
/// [`StoragePrecision::needs_rescore`] is `true` for both.
///
/// Also measures the SAME corpus + queries through the already-committed
/// single-graph bundle at that precision, and records the worst observed
/// (single_graph − merged) gap plus [`TRACKING_MARGIN_HEADROOM`] as the
/// committed tracking margin. Merges a `"segment_merge"` top-level section
/// into the existing `floor.json` (`floor = measured − `[`FLOOR_MARGIN`]` at
/// `Int8`, `floor = CI_lower − `[`FLOOR_MARGIN`]` at `Binary` — same
/// discipline [`build_precision_recall_fixture`] /
/// [`build_binary_recall_fixture`] already use for the single-graph rows).
///
/// Run off-box once with `RAYON_NUM_THREADS=1` (every sidecar build here is
/// single-threaded, matching the other frozen bundles) after
/// `build-precision-recall-fixture` and `build-binary-recall-fixture` have
/// produced the single-graph `Int8`/`Binary` fixtures; the segment bundles
/// are committed and CI only ever loads them.
pub async fn build_segment_recall_fixture(
    fixture_dir: &Path,
) -> Result<SegmentRecallFloorRecord, Box<dyn std::error::Error>> {
    let corpus_path = fixture_dir.join("corpus_vectors.parquet");
    let query_path = fixture_dir.join("query_vectors.parquet");

    let corpus_url = corpus::storage_url(&corpus_path)?;
    let corpus_table = "segment_merge_fixture_corpus";
    let ctx = corpus::register(&corpus_url, corpus_table).await?;
    let corpus_rows = corpus::load_vectors(&ctx, corpus_table).await?;
    let dim = corpus_rows
        .first()
        .map(|(_, v)| v.len())
        .ok_or("fixture corpus is empty — nothing to segment")?;

    let query_url = corpus::storage_url(&query_path)?;
    let query_table = "segment_merge_fixture_queries";
    let query_ctx = corpus::register(&query_url, query_table).await?;
    let queries: Vec<Vec<f32>> = corpus::load_vectors(&query_ctx, query_table)
        .await?
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    if queries.is_empty() {
        return Err(
            "fixture query set is empty — nothing to measure segment-merge recall over".into(),
        );
    }

    let mut precisions: BTreeMap<String, SegmentPrecisionFloorRecord> = BTreeMap::new();
    for spec in SEGMENT_PRECISIONS {
        let oversample = spec.precision.default_oversample();

        // The SAME corpus + queries through the already-committed single-graph
        // (`N = 1`) bundle at THIS precision — measured live, never
        // transcribed from the `"precision"` section this same floor.json
        // already carries.
        let single_graph_base = fixture_dir.join(spec.single_graph_stem);
        let mut single_graph: BTreeMap<usize, f64> = BTreeMap::new();
        for &k in &RECALL_KS {
            let recall = recall::mean_recall_at_k_rescored(
                &ctx,
                corpus_table,
                &single_graph_base,
                spec.precision,
                oversample,
                &queries,
                k,
            )
            .await?;
            single_graph.insert(k, recall);
        }

        let mut partitionings: BTreeMap<String, BTreeMap<usize, FloorEntry>> = BTreeMap::new();
        let mut worst_gap = 0.0f64;
        for part in SEGMENT_PARTITIONINGS {
            let mut buckets: Vec<Vec<(String, Vec<f32>)>> = vec![Vec::new(); SEGMENT_COUNT];
            for (i, row) in corpus_rows.iter().enumerate() {
                let seg = (part.assign)(i, corpus_rows.len());
                buckets[seg].push(row.clone());
            }
            if let Some(empty_idx) = buckets.iter().position(Vec::is_empty) {
                return Err(format!(
                    "partitioning {} produced an empty segment {empty_idx} — cannot build a \
                     SidecarIndex over zero rows",
                    part.name
                )
                .into());
            }

            let mut segment_bases: Vec<PathBuf> = Vec::with_capacity(SEGMENT_COUNT);
            for (idx, bucket) in buckets.iter().enumerate() {
                let base = fixture_dir.join(format!("{}_{}_seg{idx}", part.name, spec.key));
                freeze_sidecar(&base, bucket, dim, spec.precision)?;
                segment_bases.push(base);
            }

            let mut per_k: BTreeMap<usize, FloorEntry> = BTreeMap::new();
            for &k in &RECALL_KS {
                let (point, anchor_value) = if spec.ci_anchored {
                    let ci = recall::recall_ci_at_k_segmented(
                        &ctx,
                        corpus_table,
                        &segment_bases,
                        spec.precision,
                        oversample,
                        &queries,
                        k,
                    )
                    .await?;
                    (ci.point, ci.interval.lower)
                } else {
                    let point = recall::mean_recall_at_k_segmented(
                        &ctx,
                        corpus_table,
                        &segment_bases,
                        spec.precision,
                        oversample,
                        &queries,
                        k,
                    )
                    .await?;
                    (point, point)
                };
                let gap = single_graph
                    .get(&k)
                    .copied()
                    .ok_or("single-graph recall missing for k")?
                    - point;
                worst_gap = worst_gap.max(gap);
                per_k.insert(
                    k,
                    FloorEntry {
                        measured: point,
                        floor: (anchor_value - FLOOR_MARGIN).max(0.0),
                    },
                );
            }
            partitionings.insert(part.name.to_string(), per_k);
        }

        precisions.insert(
            spec.key.to_string(),
            SegmentPrecisionFloorRecord {
                oversample,
                anchor: if spec.ci_anchored {
                    "ci_lower"
                } else {
                    "point"
                },
                single_graph,
                single_graph_tracking_margin: worst_gap + TRACKING_MARGIN_HEADROOM,
                partitionings,
            },
        );
    }

    let record = SegmentRecallFloorRecord {
        segment_count: SEGMENT_COUNT,
        precisions,
    };
    merge_top_level_floor_field(fixture_dir, "segment_merge", serde_json::to_value(&record)?)?;
    Ok(record)
}
