use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::VectorIndex;

/// Current rowmap format version.
const ROWMAP_VERSION: u32 = 1;

/// Current ANN `.manifest.json` format version. A future format change bumps
/// this so a reader rejects a newer manifest as a typed
/// [`JammiError::IncompatibleFormat`] rather than silently misparsing — the
/// `.rowmap` and materialization-manifest reject-newer idiom applied to the ANN
/// sidecar's metadata.
///
/// Bumped to `2`: every manifest now carries `scalar_kind`, the precision the
/// graph's own vectors are quantized to — a genuinely new required field, not
/// a cosmetic one, since [`SidecarIndex::load`] must reconstruct the USearch
/// handle with the *matching* quantization before calling USearch's own
/// `load` (USearch's own `Index::restore` does the same: read the header's
/// quantization, then construct before loading).
///
/// Bumped to `3`: a [`StoragePrecision::Binary`] manifest now also carries
/// `binary_threshold_kind` — required (hard error if absent) whenever
/// `scalar_kind` is `Binary` and the bundle holds at least one row, since
/// [`SidecarIndex::load`] must know which [`ThresholdKind`] reduction
/// produced the sibling `.threshold` companion before trusting it. A `Binary`
/// bundle written before this bump carries neither the field nor the
/// companion and must be rebuilt — it packed with the old fixed-`0` symmetric
/// threshold this version replaces.
const ANN_MANIFEST_VERSION: u32 = 3;

/// The rescore companion's file extension, alongside `.usearch` / `.rowmap` /
/// `.manifest.json`. Present only for a quantized-precision sidecar; a `F32`
/// index writes no companion (its own USearch vectors are already exact).
pub(crate) const RESCORE_COMPANION_EXTENSION: &str = "rawf32";

/// The per-dimension threshold τ companion's file extension, alongside
/// `.usearch` / `.rowmap` / `.manifest.json` / `.rawf32`. Present only for a
/// [`StoragePrecision::Binary`] sidecar with at least one row — every other
/// precision sign-packs nothing, so it carries no τ.
pub(crate) const THRESHOLD_COMPANION_EXTENSION: &str = "threshold";

/// The distance metric a sidecar index at `precision` is built and searched
/// with: `Binary` (USearch's `B1` scalar kind, one packed sign bit per
/// dimension) compares by Hamming (bit-differences) distance, since cosine
/// has no meaning over a bit-packed vector; every other precision keeps the
/// cosine metric every non-binary Jammi index has always used. The sole place
/// the storage-precision vocabulary maps onto USearch's `MetricKind`, shared
/// by [`index_options`] (build/load) and the on-disk manifest's `metric`
/// field (`SidecarIndex::save`), so the two can never drift apart.
fn ann_metric(precision: StoragePrecision) -> usearch::MetricKind {
    match precision {
        StoragePrecision::Binary => usearch::MetricKind::Hamming,
        StoragePrecision::F32 | StoragePrecision::F16 | StoragePrecision::Int8 => {
            usearch::MetricKind::Cos
        }
    }
}

/// The manifest `metric` string for a sidecar index at `precision` —
/// [`ann_metric`]'s name, in the vocabulary the `.manifest.json` field
/// already used (`"cosine"`) before `Binary` existed.
fn ann_metric_name(precision: StoragePrecision) -> &'static str {
    match ann_metric(precision) {
        usearch::MetricKind::Hamming => "hamming",
        _ => "cosine",
    }
}

/// Build the USearch index options for a sidecar of the given dimension and
/// storage precision.
///
/// This is the sole place USearch's field names appear: the public engine
/// surface speaks the HNSW primitive ([`AnnIndexConfig`]) plus the storage
/// vocabulary ([`StoragePrecision`]), and this function is the one boundary
/// that maps them onto `usearch::IndexOptions`. A `0` HNSW knob is carried
/// through verbatim — USearch treats `0` as "use the built-in default", so an
/// [`AnnIndexConfig::default`] reproduces the backend defaults exactly.
/// `precision` is taken as an explicit parameter rather than read off `ann`:
/// callers that reopen an *existing* graph (recovery rebuild, load) must pass
/// the precision recorded on the catalog row / manifest, never today's
/// deployment default, or a config change after a table was built would
/// silently rebuild it at the wrong precision. `metric` is derived from
/// `precision` via [`ann_metric`] rather than hardcoded, so a `Binary` build
/// gets Hamming while every other precision keeps cosine. `..Default::default()`
/// preserves the remaining options (notably `multi`).
fn index_options(
    dimensions: usize,
    ann: &AnnIndexConfig,
    precision: StoragePrecision,
) -> usearch::IndexOptions {
    usearch::IndexOptions {
        dimensions,
        metric: ann_metric(precision),
        quantization: precision.to_scalar_kind(),
        connectivity: ann.connectivity,
        expansion_add: ann.build_expansion,
        expansion_search: ann.search_expansion,
        ..Default::default()
    }
}

/// Which corpus-wide reduction [`fit_binary_threshold`] fits the
/// [`StoragePrecision::Binary`] sidecar's per-dimension threshold τ with.
///
/// Transformer embeddings are anisotropic (a large common-mean component
/// shared by nearly every row), so a fixed threshold at `0` collapses every
/// dimension aligned with that mean to a constant bit — `sign(v − τ)` at a
/// corpus-fit τ eliminates that collapse instead. `Mean` is the wave-2
/// validated baseline (per-dimension arithmetic mean directly cancels the
/// anisotropic offset); `Median` guarantees an exactly balanced 50/50 bit
/// split per dimension (maximum per-bit entropy) and is kept as the measured
/// alternative — see the `mean_vs_median_threshold_on_anisotropic_corpus`
/// test below for which one [`DEFAULT_BINARY_THRESHOLD_KIND`] picks and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ThresholdKind {
    /// Per-dimension arithmetic mean over the sampled corpus.
    Mean,
    /// Per-dimension median over the sampled corpus.
    Median,
}

/// The [`ThresholdKind`] a newly built [`StoragePrecision::Binary`] sidecar
/// fits τ with. Chosen by measurement (see
/// `mean_vs_median_threshold_measured_on_anisotropic_corpus` below): on a
/// synthetic anisotropic corpus (a strong shared per-dimension bias added to
/// uncorrelated noise — the shape that collapses dead bits under the old
/// fixed-`0` threshold), both reductions eliminate the collapse and beat the
/// old fixed threshold, but `Median`'s exactly-balanced 50/50 bit split
/// measured a higher brute-force recall@10 than `Mean` (0.685 vs. 0.675 on
/// that fixture) — the wave-2 mean baseline is a validated floor, not a
/// ceiling, and the balanced-bit property does translate into a small
/// measured win once an exact rescore follows the coarse Hamming stage. Kept
/// as the default; `Mean` stays available and measured alongside it.
const DEFAULT_BINARY_THRESHOLD_KIND: ThresholdKind = ThresholdKind::Median;

/// The maximum row count [`fit_binary_threshold`] fits τ over. A representative
/// per-dimension mean/median needs only a bounded sample, not the whole
/// corpus — this caps both the time and the (for [`ThresholdKind::Median`],
/// per-dimension sort) memory the fit costs on a large table.
const BINARY_THRESHOLD_SAMPLE_CAP: usize = 100_000;

/// Fit the per-dimension threshold τ (length `dimensions`) a
/// [`StoragePrecision::Binary`] sidecar sign-packs the corpus (at
/// [`SidecarIndex::build`]) and every query (at [`SidecarIndex::search`])
/// against — [`pack_threshold_bits`]'s `sign(v − τ)`, replacing the old fixed
/// `sign(v)`.
///
/// `vectors` is `dimensions`-wide `f32` records in internal-key order (the
/// same buffer [`SidecarIndex::exact_vectors`] accumulates). Only the first
/// `min(row_count, `[`BINARY_THRESHOLD_SAMPLE_CAP`]`)` rows are read — a
/// bounded, DETERMINISTIC (insertion-order, never random) sample, so the same
/// corpus always fits the same τ regardless of corpus size, satisfying the
/// same-corpus-same-codes determinism contract every other reduction in this
/// module upholds.
fn fit_binary_threshold(vectors: &[f32], dimensions: usize, kind: ThresholdKind) -> Vec<f32> {
    if dimensions == 0 {
        return Vec::new();
    }
    let total_rows = vectors.len() / dimensions;
    let sample_rows = total_rows.min(BINARY_THRESHOLD_SAMPLE_CAP);
    match kind {
        ThresholdKind::Mean => mean_threshold(vectors, dimensions, sample_rows),
        ThresholdKind::Median => median_threshold(vectors, dimensions, sample_rows),
    }
}

/// Per-dimension arithmetic mean over the first `sample_rows` records of
/// `vectors`. Accumulated in `f64` (a fixed left-to-right reduction order, one
/// pass) so a wide, deep corpus does not lose precision to `f32` summation
/// before the final per-dimension cast back down.
fn mean_threshold(vectors: &[f32], dimensions: usize, sample_rows: usize) -> Vec<f32> {
    let mut sums = vec![0f64; dimensions];
    for row in 0..sample_rows {
        let start = row * dimensions;
        for (d, &x) in vectors[start..start + dimensions].iter().enumerate() {
            sums[d] += f64::from(x);
        }
    }
    let n = (sample_rows.max(1)) as f64;
    sums.into_iter().map(|s| (s / n) as f32).collect()
}

/// Per-dimension median over the first `sample_rows` records of `vectors`.
/// One reused column buffer sorted per dimension (`f32::total_cmp` — the
/// same NaN-safe, deterministic-ordering primitive every other reduction in
/// this crate sorts floats with); the median of an even sample averages the
/// two central values.
fn median_threshold(vectors: &[f32], dimensions: usize, sample_rows: usize) -> Vec<f32> {
    if sample_rows == 0 {
        return vec![0.0; dimensions];
    }
    let mut column = vec![0f32; sample_rows];
    (0..dimensions)
        .map(|d| {
            for (row, slot) in column.iter_mut().enumerate() {
                *slot = vectors[row * dimensions + d];
            }
            column.sort_unstable_by(f32::total_cmp);
            let mid = sample_rows / 2;
            if sample_rows % 2 == 0 {
                (column[mid - 1] + column[mid]) / 2.0
            } else {
                column[mid]
            }
        })
        .collect()
}

/// Per-dimension threshold sign-bit packing for a [`StoragePrecision::Binary`]
/// sidecar: `v[i] > threshold[i]` → bit `1`, else (including exact equality
/// and every value below it) → bit `0`, packed LSB-first into `ceil(dim / 8)`
/// bytes. Unused high bits in the final byte are left `0` (`vec![0u8; ...]`
/// is zero-initialized, so this is a property of the construction, not a
/// separate masking step).
///
/// The ONE function both [`SidecarIndex::build`] (corpus rows, once τ is
/// fit) and [`SidecarIndex::search`] (queries) pack through — USearch's
/// Hamming metric counts every bit position in the buffer, padding included,
/// so packing the two sides against a different threshold would silently
/// bias every Hamming distance by a fixed, easy-to-miss amount (it would just
/// look like uniformly worse recall, never a crash). `threshold` is the
/// corpus-fit τ from [`fit_binary_threshold`] — an all-zero `threshold`
/// reproduces the old symmetric-sign-at-0 packing exactly, since `v[i] >
/// 0.0` is `sign(v[i] − 0)`.
fn pack_threshold_bits(v: &[f32], threshold: &[f32]) -> Vec<u8> {
    let n_bytes = v.len().div_ceil(8);
    let mut packed = vec![0u8; n_bytes];
    for (i, (&x, &t)) in v.iter().zip(threshold.iter()).enumerate() {
        if x > t {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

/// A positioned-read, id-keyed exact-`f32` companion for a quantized sidecar
/// index.
///
/// Fixed-size records (`dimensions` × `f32`, little-endian, native byte
/// order — read back on the same architecture family that wrote it) in the
/// same order as [`SidecarIndex`]'s internal USearch key (`0..row_map.len()`),
/// so the vector for internal key `k` sits at byte offset `k * dimensions *
/// 4` — O(1) random-access via a positioned read (served from the OS page
/// cache, never loaded resident up front), and never a per-query Parquet
/// re-scan. This is the *exact*-vector source a quantized index's own
/// `search`/`export` cannot serve: once USearch quantizes its stored vectors
/// (`F16`/`I8`), what it hands back is the lossy reconstructed value, not the
/// original `f32`.
///
/// A derived artifact: fully rebuildable from the Parquet result table (every
/// build site that constructs a [`SidecarIndex`] at a quantized precision
/// writes both the USearch graph and this companion from the same rows), so
/// it carries no independent durability guarantee beyond the sidecar bundle
/// it rides alongside.
struct RawVectorCompanion {
    file: std::fs::File,
    dimensions: usize,
}

impl RawVectorCompanion {
    /// Write the companion's bytes: `vectors` is `dimensions`-wide `f32`
    /// records concatenated in internal-key order (i.e. exactly the buffer
    /// [`SidecarIndex::add`] accumulates).
    fn write(path: &Path, dimensions: usize, vectors: &[f32]) -> Result<()> {
        debug_assert_eq!(
            vectors.len() % dimensions.max(1),
            0,
            "rescore companion buffer must hold whole records"
        );
        std::fs::write(path, bytemuck::cast_slice(vectors))?;
        Ok(())
    }

    /// Open an existing companion file read-only. Kept as an owned `File`
    /// (no mmap): every read below is a positioned `pread`, so the file
    /// descriptor is the only resource held for the companion's lifetime.
    fn open(path: &Path, dimensions: usize) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self { file, dimensions })
    }

    /// O(1) random read of the exact vector at `internal_key` via a
    /// positioned read (`pread`) at `internal_key * dimensions * 4` — no
    /// seek, no shared file-cursor mutation, safe to call concurrently from
    /// multiple readers of the same companion. `None` when the record falls
    /// outside the file (a stale key past this file's record count, or a
    /// truncated/missing file), mirroring the previous mmap-`get`'s graceful
    /// out-of-range behaviour rather than surfacing an I/O error.
    fn get(&self, internal_key: u64) -> Option<Vec<f32>> {
        use std::os::unix::fs::FileExt;

        let stride = self.dimensions * std::mem::size_of::<f32>();
        let start = (internal_key as usize).checked_mul(stride)?;
        let start = u64::try_from(start).ok()?;

        let mut record = vec![0f32; self.dimensions];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut record);
        self.file.read_exact_at(bytes, start).ok()?;
        Some(record)
    }
}

/// Sidecar ANN index backed by USearch, with a Jammi-owned `_row_id` mapping
/// and a JSON manifest.
///
/// Files produced per embedding table:
/// - `.usearch` — USearch serialized graph
/// - `.rowmap` — row_id mapping (internal_id → _row_id string)
/// - `.manifest.json` — metadata (version, dimensions, count, backend, created_at)
pub struct SidecarIndex {
    dimensions: usize,
    index: usearch::Index,
    row_map: Vec<String>,
    /// Reverse of `row_map`: `_row_id` → internal USearch key, so a stored
    /// vector can be fetched back by id via [`SidecarIndex::get`] without the
    /// caller keeping a second copy of the vectors. Holds only the ids (the same
    /// strings already in `row_map`), never the embeddings.
    row_index: HashMap<String, u64>,
    built: bool,
    /// The precision this index's own USearch-stored vectors are quantized
    /// to. `F32` means `get`/`export` already return the exact vector; a
    /// quantized precision means they don't, and `rescore` (or, mid-build,
    /// `exact_vectors`) is the exact-vector source instead.
    storage_precision: StoragePrecision,
    /// Exact `f32` vectors accumulated in internal-key order as [`Self::add`]
    /// is called, flushed into the `.rawf32` rescore companion by
    /// [`Self::save`]. `Some` (initially empty) only when
    /// [`StoragePrecision::needs_rescore`] — an `F32` build tracks nothing
    /// here because USearch's own storage is already exact. `None` after
    /// [`Self::load`]: a loaded index reads exact vectors back through
    /// `rescore`'s mmap, never rebuilds this buffer into memory.
    exact_vectors: Option<Vec<f32>>,
    /// The raw-`f32` rescore companion, present iff this index was loaded at
    /// a quantized precision (its sibling `.rawf32` file is opened at load
    /// time, not lazily; each vector is then served by a positioned read).
    /// `None` for an `F32` index.
    rescore: Option<RawVectorCompanion>,
    /// The per-dimension threshold τ a [`StoragePrecision::Binary`] index's
    /// sign-packing (both the corpus at [`Self::build`] and every query at
    /// [`Self::search`]) is applied against. `Some` (length `dimensions`)
    /// once a `Binary` index has been built ([`Self::build`] fits it from the
    /// accumulated corpus) or loaded ([`Self::load`] reads it back from the
    /// `.threshold` companion). `None` for every other precision, and for a
    /// `Binary` index that has neither been built nor loaded yet.
    binary_threshold: Option<Vec<f32>>,
    /// Which reduction ([`ThresholdKind`]) [`Self::binary_threshold`] was fit
    /// with. `Some` iff `binary_threshold` is — persisted in the manifest so
    /// a reload can confirm which threshold a bundle used.
    threshold_kind: Option<ThresholdKind>,
}

/// The load-relevant header of the ANN `.manifest.json` sidecar.
///
/// Deserialised as a typed struct (never field-by-field `Value` lookups) so the
/// load path mirrors [`crate::store::manifest::MaterializationManifest::from_json_bytes`]:
/// the determinants of a safe load — the manifest format `version`, the
/// embedding `dimensions`, the USearch `backend_version` the graph was
/// serialised by, and the `scalar_kind` it was quantized at — are all
/// required. A manifest missing any of them is a hard `serde` error (decoded
/// into a typed [`JammiError`]), never silently defaulted. The remaining
/// metadata (`count`, `metric`, `files`, `created_at`) is provenance the load
/// path does not consume, so it is ignored on read.
#[derive(Debug, Deserialize)]
struct AnnManifest {
    /// Manifest format version, checked reject-newer against
    /// [`ANN_MANIFEST_VERSION`].
    version: u32,
    /// Embedding width the graph was built over.
    dimensions: usize,
    /// The USearch version that serialised the graph, strict-compared against
    /// [`crate::index::backend_version`] — USearch gives no compatibility
    /// ordering, so any mismatch is incompatible.
    backend_version: String,
    /// The precision the graph's own vectors were quantized to at build time,
    /// strict-compared against the caller's expected precision (the catalog
    /// row's persisted `storage_precision`) — a mismatch means either the
    /// deployment default drifted since this table was built, or the graph is
    /// stale; either way the load must not silently reopen it as if it
    /// matched.
    scalar_kind: StoragePrecision,
    /// Which [`ThresholdKind`] the `.threshold` companion was fit with.
    /// Legitimately absent (`None`, via `#[serde(default)]`) for every
    /// non-`Binary` precision; [`SidecarIndex::load`] hard-errors if it is
    /// missing while `scalar_kind` is `Binary` and the bundle holds at least
    /// one row — a `Binary` manifest missing this field is a torn or
    /// pre-threshold-fix bundle, not a legitimate empty state.
    #[serde(default)]
    binary_threshold_kind: Option<ThresholdKind>,
}

impl SidecarIndex {
    /// Create a new empty sidecar index for vectors of the given dimension, at
    /// `precision`, tuned by the HNSW knobs in `ann`. The build-time knobs
    /// (`connectivity`, `build_expansion`) take effect as vectors are added;
    /// `search_expansion` governs queries against the resulting graph.
    /// `precision` is the table's resolved storage precision (a fresh table's
    /// deployment default, or an existing table's persisted catalog value for
    /// a rebuild) — never read off `ann` here, so a caller cannot forget which
    /// one applies.
    pub fn new(
        dimensions: usize,
        ann: &AnnIndexConfig,
        precision: StoragePrecision,
    ) -> Result<Self> {
        let index = usearch::Index::new(&index_options(dimensions, ann, precision))
            .map_err(|e| JammiError::Other(format!("USearch index creation: {e}")))?;

        Ok(Self {
            dimensions,
            index,
            row_map: Vec::new(),
            row_index: HashMap::new(),
            built: false,
            storage_precision: precision,
            exact_vectors: precision.needs_rescore().then(Vec::new),
            rescore: None,
            binary_threshold: None,
            threshold_kind: None,
        })
    }

    /// The precision this index's own USearch-stored vectors are quantized
    /// to — the value a loaded index verified against the catalog row at
    /// [`Self::load`], or the value it was constructed with at [`Self::new`].
    pub fn storage_precision(&self) -> StoragePrecision {
        self.storage_precision
    }

    /// Fetch the stored vector for `row_id`, or `None` if the id is not indexed.
    ///
    /// Reads the vector USearch already holds rather than asking the caller to
    /// keep its own id→vector map — the index is the single owner of the
    /// embeddings it was built over. At a quantized [`StoragePrecision`] this
    /// is USearch's own **lossy** reconstruction, not the original `f32` — use
    /// [`Self::get_exact`] when the exact vector is required (rescore).
    pub fn get(&self, row_id: &str) -> Result<Option<Vec<f32>>> {
        let Some(&key) = self.row_index.get(row_id) else {
            return Ok(None);
        };
        let mut out = Vec::new();
        let found = self
            .index
            .export(key, &mut out)
            .map_err(|e| JammiError::Other(format!("USearch get: {e}")))?;
        if found == 0 {
            return Ok(None);
        }
        out.truncate(self.dimensions);
        Ok(Some(out))
    }

    /// Fetch the **exact** `f32` vector for `row_id` — the retrieve→rescore
    /// source of truth. For a quantized index this reads the mmap'd rescore
    /// companion by internal key (O(1), no Parquet scan) when loaded, or the
    /// in-memory `exact_vectors` buffer for a freshly built index that
    /// has not yet been saved/loaded (the companion is only opened at
    /// [`Self::load`]). For an `F32` index (which carries no companion or
    /// buffer — its own vectors are already exact) this falls back to
    /// [`Self::get`]. `None` if the id is not indexed.
    ///
    /// A quantized index with neither the companion nor the in-memory buffer
    /// (a state the constructors never produce, but this guard refuses to
    /// paper over) is a hard error: silently falling back to [`Self::get`]
    /// here would hand the caller USearch's own **lossy** reconstruction under
    /// the name "exact", corrupting every rescore that reads it.
    pub fn get_exact(&self, row_id: &str) -> Result<Option<Vec<f32>>> {
        let Some(&key) = self.row_index.get(row_id) else {
            return Ok(None);
        };
        match &self.rescore {
            Some(companion) => Ok(companion.get(key)),
            None if self.storage_precision.needs_rescore() => match &self.exact_vectors {
                Some(vectors) => {
                    let start = key as usize * self.dimensions;
                    let end = start + self.dimensions;
                    Ok(vectors.get(start..end).map(<[f32]>::to_vec))
                }
                None => Err(JammiError::Other(format!(
                    "get_exact: quantized index ({:?}) has no rescore companion and no \
                     in-memory exact-vector buffer for '{row_id}' — refusing to fall back to \
                     USearch's lossy reconstruction",
                    self.storage_precision
                ))),
            },
            None => self.get(row_id),
        }
    }

    /// Save the sidecar bundle (`.usearch` + `.rowmap` + `.manifest.json` +,
    /// at a quantized precision, `.rawf32`).
    pub fn save(&self, base_path: &Path) -> Result<()> {
        // Save USearch index
        let usearch_path = base_path.with_extension("usearch");
        self.index
            .save(usearch_path.to_str().unwrap_or_default())
            .map_err(|e| JammiError::Other(format!("USearch save: {e}")))?;

        // Save rowmap: version (u32 LE) + entries (len_u32 LE + UTF-8 bytes)
        let rowmap_path = base_path.with_extension("rowmap");
        let mut file = std::fs::File::create(&rowmap_path)?;
        file.write_all(&ROWMAP_VERSION.to_le_bytes())?;
        for id in &self.row_map {
            let bytes = id.as_bytes();
            file.write_all(&(bytes.len() as u32).to_le_bytes())?;
            file.write_all(bytes)?;
        }

        // Save the raw-f32 rescore companion, only when this build tracked one
        // (a quantized precision with at least one vector added).
        if let Some(vectors) = self.exact_vectors.as_ref() {
            if !self.row_map.is_empty() {
                let companion_path = base_path.with_extension(RESCORE_COMPANION_EXTENSION);
                RawVectorCompanion::write(&companion_path, self.dimensions, vectors)?;
            }
        }

        // Save the Binary sidecar's per-dimension threshold τ companion, only
        // when `build` actually fit one (a `Binary` build with at least one
        // row added — `build` never sets `binary_threshold` on an empty
        // index).
        if let Some(threshold) = self.binary_threshold.as_ref() {
            let threshold_path = base_path.with_extension(THRESHOLD_COMPANION_EXTENSION);
            std::fs::write(&threshold_path, bytemuck::cast_slice(threshold))?;
        }

        // Save manifest
        let manifest_path = base_path.with_extension("manifest.json");
        let manifest = serde_json::json!({
            "version": ANN_MANIFEST_VERSION,
            "dimensions": self.dimensions,
            "count": self.row_map.len(),
            "metric": ann_metric_name(self.storage_precision),
            "backend": "usearch",
            "backend_version": crate::index::backend_version(),
            "scalar_kind": self.storage_precision,
            "binary_threshold_kind": self.threshold_kind,
            "files": {
                "index": usearch_path.file_name().and_then(|n| n.to_str()),
                "rowmap": rowmap_path.file_name().and_then(|n| n.to_str()),
            },
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

        Ok(())
    }

    /// Load a sidecar bundle from disk, applying the query-time HNSW knob from
    /// `ann` and strict-checking the graph's own stamped precision against
    /// `expected_precision` — the caller's catalog-persisted
    /// `storage_precision` for this table, never the deployment default.
    ///
    /// A loaded graph is frozen: `connectivity` was baked in at build time (and
    /// is repopulated from the serialized header), and `build_expansion` has no
    /// effect on an existing graph — so neither build knob is consequential
    /// here. `search_expansion` is the exception: USearch does not persist it in
    /// the serialized header, so the loaded handle carries the backend default
    /// until it is set explicitly. We re-apply it from `ann` when non-zero; a
    /// `0` leaves the backend default in place (today's behaviour).
    pub fn load(
        base_path: &Path,
        ann: &AnnIndexConfig,
        expected_precision: StoragePrecision,
    ) -> Result<Self> {
        // Load the manifest as a typed struct (mirroring
        // `MaterializationManifest::from_json_bytes`): a missing `version`,
        // `dimensions`, `backend_version`, or `scalar_kind` is a hard typed
        // error, never a silent default.
        let manifest_path = base_path.with_extension("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path)?;
        let manifest: AnnManifest = serde_json::from_slice(&manifest_bytes)?;

        // The ANN manifest format has a compatibility ordering — reject only a
        // NEWER version than this build can read.
        if manifest.version > ANN_MANIFEST_VERSION {
            return Err(JammiError::IncompatibleFormat {
                artifact: "ann-manifest".into(),
                found: manifest.version.to_string(),
                supported: ANN_MANIFEST_VERSION.to_string(),
            });
        }

        // STRICT precision validation: the on-disk graph's own quantization
        // must equal the catalog's persisted `storage_precision` for this
        // table. A mismatch is never a version-bump situation (both are
        // legal `StoragePrecision` values) — it means the graph was built
        // under a since-changed deployment default and would silently return
        // wrong-shaped (or wrongly quantized) results if reopened as if it
        // matched.
        if manifest.scalar_kind != expected_precision {
            return Err(JammiError::IncompatibleFormat {
                artifact: "ann-index-precision".into(),
                found: manifest.scalar_kind.to_string(),
                supported: expected_precision.to_string(),
            });
        }

        // STRICT backend-version validation: USearch's serialized graph format
        // gives no compatibility ordering, so a version that differs at all from
        // the linked USearch can silently mis-deserialise the graph and return
        // wrong neighbours. Any mismatch is incompatible — there is no
        // "reject-newer" here, only exact equality.
        let current_backend = crate::index::backend_version();
        if manifest.backend_version != current_backend {
            return Err(JammiError::IncompatibleFormat {
                artifact: "usearch-index".into(),
                found: manifest.backend_version,
                supported: current_backend.to_string(),
            });
        }

        let dimensions = manifest.dimensions;

        // Load rowmap
        let rowmap_path = base_path.with_extension("rowmap");
        let mut file = std::fs::File::open(&rowmap_path)?;
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        // Reject only a NEWER format than this build can read, mirroring the
        // materialization manifest's reject-newer idiom: the rowmap layout has a
        // compatibility ordering, so an older or equal stamp is readable while a
        // newer one carries a layout this build does not know.
        if version > ROWMAP_VERSION {
            return Err(JammiError::IncompatibleFormat {
                artifact: "rowmap".into(),
                found: version.to_string(),
                supported: ROWMAP_VERSION.to_string(),
            });
        }

        let mut row_map = Vec::new();
        loop {
            let mut len_bytes = [0u8; 4];
            match file.read_exact(&mut len_bytes) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf)?;
            row_map.push(
                String::from_utf8(buf)
                    .map_err(|e| JammiError::Other(format!("Invalid UTF-8 in rowmap: {e}")))?,
            );
        }

        // Load USearch index. Build knobs are inert for a frozen graph, so the
        // handle is created with backend defaults; only `search_expansion` is
        // re-applied below. `quantization` must match the on-disk graph's own
        // (just-verified) `scalar_kind` — USearch's own `Index::restore` does
        // the same (reads the header's quantization, then constructs before
        // loading), and constructing with the wrong one would misread the
        // serialized graph.
        let index = usearch::Index::new(&index_options(
            dimensions,
            &AnnIndexConfig::default(),
            manifest.scalar_kind,
        ))
        .map_err(|e| JammiError::Other(format!("USearch index creation for load: {e}")))?;

        let usearch_path = base_path.with_extension("usearch");
        index
            .load(usearch_path.to_str().unwrap_or_default())
            .map_err(|e| JammiError::Other(format!("USearch load: {e}")))?;

        if ann.search_expansion != 0 {
            index.change_expansion_search(ann.search_expansion);
        }

        let row_index: HashMap<String, u64> = row_map
            .iter()
            .enumerate()
            .map(|(key, id)| (id.clone(), key as u64))
            .collect();

        // A quantized precision expects a rescore companion beside the
        // bundle — it is a derived artifact every quantized build site writes
        // in the same `save`, so its absence here is a torn/incomplete bundle,
        // not a legitimate empty state. Fail loudly rather than silently
        // degrading every rescore into a quantized-only (lossy) result.
        let rescore = if manifest.scalar_kind.needs_rescore() && !row_map.is_empty() {
            let companion_path = base_path.with_extension(RESCORE_COMPANION_EXTENSION);
            Some(RawVectorCompanion::open(&companion_path, dimensions)?)
        } else {
            None
        };

        // A `Binary` bundle with at least one row expects the `.threshold`
        // companion beside it, and `binary_threshold_kind` on the manifest —
        // both written together by every `Binary` `save`, so either's
        // absence is a torn/incomplete (or pre-threshold-fix) bundle, not a
        // legitimate empty state. Fail loudly rather than silently reopening
        // it under the old fixed-`0` symmetric packing.
        let (binary_threshold, threshold_kind) =
            if manifest.scalar_kind == StoragePrecision::Binary && !row_map.is_empty() {
                let kind = manifest.binary_threshold_kind.ok_or_else(|| {
                    JammiError::Other(
                        "ann-manifest: a Binary sidecar's manifest is missing \
                         binary_threshold_kind — torn or pre-threshold-fix bundle"
                            .into(),
                    )
                })?;
                let threshold_path = base_path.with_extension(THRESHOLD_COMPANION_EXTENSION);
                let bytes = std::fs::read(&threshold_path)?;
                let threshold: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                if threshold.len() != dimensions {
                    return Err(JammiError::Other(format!(
                        "ann-threshold: expected {dimensions} τ values, found {}",
                        threshold.len()
                    )));
                }
                (Some(threshold), Some(kind))
            } else {
                (None, None)
            };

        Ok(Self {
            dimensions,
            index,
            row_map,
            row_index,
            built: true,
            storage_precision: manifest.scalar_kind,
            exact_vectors: None,
            rescore,
            binary_threshold,
            threshold_kind,
        })
    }
}

impl VectorIndex for SidecarIndex {
    fn add(&mut self, row_id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(JammiError::Other(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }
        let key = self.row_map.len() as u64;
        match self.storage_precision {
            // A Binary index's own USearch storage is `b1x8`-typed
            // sign-packed against a corpus-wide per-dimension threshold τ —
            // unknown until every row has been seen. So a Binary row is only
            // accumulated here (into `exact_vectors` below); [`Self::build`]
            // fits τ from the whole corpus and inserts every row into the
            // USearch graph in one bulk pass.
            StoragePrecision::Binary => {}
            StoragePrecision::F32 | StoragePrecision::F16 | StoragePrecision::Int8 => {
                // Reserve space if needed.
                if self.index.capacity() <= self.index.size() {
                    let new_cap = (self.index.capacity() + 1).max(64);
                    self.index
                        .reserve(new_cap)
                        .map_err(|e| JammiError::Other(format!("USearch reserve: {e}")))?;
                }
                self.index
                    .add(key, vector)
                    .map_err(|e| JammiError::Other(format!("USearch add: {e}")))?;
            }
        }
        if let Some(buf) = self.exact_vectors.as_mut() {
            buf.extend_from_slice(vector);
        }
        self.row_map.push(row_id.to_string());
        self.row_index.insert(row_id.to_string(), key);
        Ok(())
    }

    fn build(&mut self) -> Result<()> {
        // USearch builds incrementally during add() for every precision
        // EXCEPT Binary, whose corpus rows `add` only accumulated into
        // `exact_vectors` (see above) — this is where a Binary index's
        // per-dimension threshold τ is fit from the whole corpus and every
        // row is bulk-inserted into the USearch graph, sign-packed against
        // it. Every other precision was already inserted incrementally, so
        // this stays a no-op for them.
        if self.storage_precision == StoragePrecision::Binary && !self.row_map.is_empty() {
            let vectors = self.exact_vectors.as_ref().expect(
                "a Binary sidecar always tracks exact_vectors (StoragePrecision::Binary::needs_rescore() is true)",
            );
            let kind = DEFAULT_BINARY_THRESHOLD_KIND;
            let threshold = fit_binary_threshold(vectors, self.dimensions, kind);

            self.index
                .reserve(self.row_map.len())
                .map_err(|e| JammiError::Other(format!("USearch reserve: {e}")))?;
            for key in 0..self.row_map.len() as u64 {
                let start = key as usize * self.dimensions;
                let vector = &vectors[start..start + self.dimensions];
                let packed = pack_threshold_bits(vector, &threshold);
                self.index
                    .add(key, usearch::b1x8::from_u8s(&packed))
                    .map_err(|e| JammiError::Other(format!("USearch add: {e}")))?;
            }
            self.binary_threshold = Some(threshold);
            self.threshold_kind = Some(kind);
        }
        // We just mark it as built for correctness tracking.
        self.built = true;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if self.row_map.is_empty() {
            return Ok(Vec::new());
        }
        let actual_k = k.min(self.row_map.len());
        let matches = match self.storage_precision {
            // Same routing as `build`: a Binary graph's typed search path is
            // `search_b1x8`, fed the query packed through the identical
            // `pack_threshold_bits` (against the SAME corpus-fit τ) the
            // corpus rows were bulk-inserted with.
            StoragePrecision::Binary => {
                let threshold = self.binary_threshold.as_deref().ok_or_else(|| {
                    JammiError::Other(
                        "Binary search: no threshold τ available — index has not been built or \
                         loaded"
                            .into(),
                    )
                })?;
                let packed = pack_threshold_bits(query, threshold);
                self.index
                    .search(usearch::b1x8::from_u8s(&packed), actual_k)
            }
            StoragePrecision::F32 | StoragePrecision::F16 | StoragePrecision::Int8 => {
                self.index.search(query, actual_k)
            }
        }
        .map_err(|e| JammiError::Other(format!("USearch search: {e}")))?;

        let results: Vec<(String, f32)> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(&key, &dist)| {
                let idx = key as usize;
                self.row_map.get(idx).map(|id| (id.clone(), dist))
            })
            .collect();
        Ok(results)
    }

    fn save(&self, path: &Path) -> Result<()> {
        SidecarIndex::save(self, path)
    }

    fn len(&self) -> usize {
        self.row_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // USearch's built-in HNSW defaults — what a `0` knob resolves to. Pinned
    // here deliberately: a backend bump that shifts a default trips these
    // assertions rather than silently changing every index's recall/cost.
    const USEARCH_DEFAULT_CONNECTIVITY: usize = 16;
    const USEARCH_DEFAULT_EXPANSION_ADD: usize = 128;
    const USEARCH_DEFAULT_EXPANSION_SEARCH: usize = 64;

    #[test]
    fn knobs_map_onto_a_freshly_built_graph() {
        // Non-default build/query knobs flow through `index_options` into the
        // backing graph at construction time.
        let ann = AnnIndexConfig {
            connectivity: 32,
            build_expansion: 200,
            search_expansion: 100,
            ..AnnIndexConfig::default()
        };
        let idx = SidecarIndex::new(8, &ann, StoragePrecision::F32).unwrap();
        assert_eq!(idx.index.connectivity(), 32);
        assert_eq!(idx.index.expansion_add(), 200);
        assert_eq!(idx.index.expansion_search(), 100);
    }

    #[test]
    fn default_config_reproduces_backend_defaults() {
        // A zeroed config is the documented no-op: every knob resolves to the
        // backend's built-in default, so an unset deployment is unchanged.
        let idx = SidecarIndex::new(8, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
        assert_eq!(idx.index.connectivity(), USEARCH_DEFAULT_CONNECTIVITY);
        assert_eq!(idx.index.expansion_add(), USEARCH_DEFAULT_EXPANSION_ADD);
        assert_eq!(
            idx.index.expansion_search(),
            USEARCH_DEFAULT_EXPANSION_SEARCH
        );
    }

    #[test]
    fn load_reapplies_search_expansion_only() {
        // `search_expansion` is a query-time dial USearch does not persist in
        // the serialized header, so a load must re-apply it. The build knobs are
        // frozen into the graph and are NOT recovered from the config on load
        // (`connectivity` is read back from the header; `expansion_add` resets
        // to the backend default) — load honours `search_expansion` alone.
        let dir = tempdir().unwrap();
        let base = dir.path().join("knob_roundtrip");

        let build = AnnIndexConfig {
            connectivity: 32,
            build_expansion: 200,
            search_expansion: 0,
            ..AnnIndexConfig::default()
        };
        let mut idx = SidecarIndex::new(4, &build, StoragePrecision::F32).unwrap();
        idx.add("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx.save(&base).unwrap();

        // Load with a non-zero search_expansion → re-applied to the loaded graph.
        let loaded = SidecarIndex::load(
            &base,
            &AnnIndexConfig {
                search_expansion: 77,
                ..AnnIndexConfig::default()
            },
            StoragePrecision::F32,
        )
        .unwrap();
        assert_eq!(
            loaded.index.expansion_search(),
            77,
            "search_expansion must be re-applied on load"
        );

        // Load with a default (0) search_expansion → the backend default, not 0.
        let loaded_default =
            SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
        assert_eq!(
            loaded_default.index.expansion_search(),
            USEARCH_DEFAULT_EXPANSION_SEARCH
        );
    }

    /// Build and save a valid one-vector sidecar bundle at `base`, returning it
    /// ready for a tamper-then-reload teeth test.
    fn save_valid_bundle(base: &Path) {
        let mut idx =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
        idx.add("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx.save(base).unwrap();
    }

    /// Read the saved `.manifest.json` at `base` into a mutable JSON value.
    fn read_manifest_json(base: &Path) -> serde_json::Value {
        let manifest_path = base.with_extension("manifest.json");
        let bytes = std::fs::read(&manifest_path).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// Overwrite the `.manifest.json` at `base` with `value`.
    fn write_manifest_json(base: &Path, value: &serde_json::Value) {
        let manifest_path = base.with_extension("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string_pretty(value).unwrap()).unwrap();
    }

    /// The error from a load that must fail. `SidecarIndex` is intentionally not
    /// `Debug` (it wraps an opaque USearch handle), so the teeth tests cannot use
    /// `unwrap_err`; this funnels the failing load to its [`JammiError`].
    fn load_err(base: &Path) -> JammiError {
        match SidecarIndex::load(base, &AnnIndexConfig::default(), StoragePrecision::F32) {
            Ok(_) => panic!("expected load to fail, but it succeeded"),
            Err(e) => e,
        }
    }

    #[test]
    fn newer_rowmap_version_is_rejected() {
        // A `.rowmap` stamped one past this build's version is a typed
        // IncompatibleFormat rejection — the manifest reject-newer idiom applied
        // to the rowmap's binary header. Modeled on
        // `manifest.rs::newer_manifest_version_is_rejected`.
        let dir = tempdir().unwrap();
        let base = dir.path().join("rowmap_newer");
        save_valid_bundle(&base);

        // Rewrite only the leading u32 version of the .rowmap to ROWMAP_VERSION+1,
        // preserving the entry bytes after it.
        let rowmap_path = base.with_extension("rowmap");
        let mut bytes = std::fs::read(&rowmap_path).unwrap();
        bytes[..4].copy_from_slice(&(ROWMAP_VERSION + 1).to_le_bytes());
        std::fs::write(&rowmap_path, &bytes).unwrap();

        let err = load_err(&base);
        match err {
            JammiError::IncompatibleFormat {
                artifact,
                found,
                supported,
            } => {
                assert_eq!(artifact, "rowmap");
                assert_eq!(found, (ROWMAP_VERSION + 1).to_string());
                assert_eq!(supported, ROWMAP_VERSION.to_string());
            }
            other => panic!("expected IncompatibleFormat for rowmap, got {other:?}"),
        }
    }

    #[test]
    fn newer_ann_manifest_version_is_rejected() {
        // An ANN `.manifest.json` stamped one past this build's version is a
        // typed IncompatibleFormat rejection (the now-LIVE version stamp), not a
        // dead write-only field.
        let dir = tempdir().unwrap();
        let base = dir.path().join("ann_newer");
        save_valid_bundle(&base);

        let mut manifest = read_manifest_json(&base);
        manifest["version"] = serde_json::json!(ANN_MANIFEST_VERSION + 1);
        write_manifest_json(&base, &manifest);

        let err = load_err(&base);
        match err {
            JammiError::IncompatibleFormat {
                artifact,
                found,
                supported,
            } => {
                assert_eq!(artifact, "ann-manifest");
                assert_eq!(found, (ANN_MANIFEST_VERSION + 1).to_string());
                assert_eq!(supported, ANN_MANIFEST_VERSION.to_string());
            }
            other => panic!("expected IncompatibleFormat for ann-manifest, got {other:?}"),
        }
    }

    #[test]
    fn missing_manifest_version_is_a_hard_error() {
        // A manifest with no `version` field is a hard typed error (a serde
        // decode failure surfacing as JammiError::Json), never a silent
        // default-to-1.
        let dir = tempdir().unwrap();
        let base = dir.path().join("ann_no_version");
        save_valid_bundle(&base);

        let mut manifest = read_manifest_json(&base);
        manifest
            .as_object_mut()
            .unwrap()
            .remove("version")
            .expect("fixture manifest must have carried a version to remove");
        write_manifest_json(&base, &manifest);

        let err = load_err(&base);
        assert!(
            matches!(err, JammiError::Json(_)),
            "a missing version must be a hard decode error, got {err:?}"
        );
    }

    #[test]
    fn mismatched_usearch_backend_version_is_rejected() {
        // The stamped USearch `backend_version` is STRICT-compared on load: any
        // value other than the linked USearch's is incompatible, because the
        // serialized graph format carries no compatibility ordering and a bump
        // can silently mis-deserialise the graph. This is the silent-corruption
        // fix — without it the wrong-neighbours risk is undetected.
        let dir = tempdir().unwrap();
        let base = dir.path().join("backend_mismatch");
        save_valid_bundle(&base);

        let bogus = "0.0.0-not-the-linked-usearch";
        let mut manifest = read_manifest_json(&base);
        manifest["backend_version"] = serde_json::json!(bogus);
        write_manifest_json(&base, &manifest);

        let err = load_err(&base);
        match err {
            JammiError::IncompatibleFormat {
                artifact,
                found,
                supported,
            } => {
                assert_eq!(artifact, "usearch-index");
                assert_eq!(found, bogus);
                assert_eq!(supported, crate::index::backend_version());
            }
            other => panic!("expected IncompatibleFormat for usearch-index, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_scalar_kind_is_rejected() {
        // A graph built at one precision but expected (by the caller's catalog
        // row) at a different one is a typed IncompatibleFormat rejection — the
        // manifest reject-newer idiom applied to the scalar-kind determinant, so
        // a config drift since the table was built forces the caller to rebuild
        // rather than silently reopen the wrong-precision graph.
        let dir = tempdir().unwrap();
        let base = dir.path().join("scalar_kind_mismatch");
        save_valid_bundle(&base); // built at F32

        match SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::Int8) {
            Ok(_) => panic!("expected load to fail, but it succeeded"),
            Err(JammiError::IncompatibleFormat {
                artifact,
                found,
                supported,
            }) => {
                assert_eq!(artifact, "ann-index-precision");
                assert_eq!(found, "f32");
                assert_eq!(supported, "int8");
            }
            Err(other) => {
                panic!("expected IncompatibleFormat for ann-index-precision, got {other:?}")
            }
        }
    }

    #[test]
    fn f32_index_carries_no_rescore_companion() {
        // `F32`'s own USearch-stored vectors are already exact, so no `.rawf32`
        // sibling is written, and `get_exact` falls back to `get` rather than
        // reading a companion.
        let dir = tempdir().unwrap();
        let base = dir.path().join("f32_no_companion");

        let mut idx =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
        idx.add("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx.save(&base).unwrap();

        assert!(
            !base.with_extension(RESCORE_COMPANION_EXTENSION).exists(),
            "an F32 build must not write a rescore companion"
        );

        let loaded =
            SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
        assert!(loaded.rescore.is_none());
        assert_eq!(
            loaded.get_exact("a").unwrap(),
            Some(vec![1.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn quantized_index_writes_and_round_trips_the_rescore_companion() {
        // A quantized build (`Int8`) writes the `.rawf32` companion beside the
        // bundle; a loaded index reads it back exactly via `get_exact`, distinct
        // from `get`'s lossy USearch-quantized reconstruction.
        let dir = tempdir().unwrap();
        let base = dir.path().join("int8_companion");

        let vectors: [[f32; 4]; 3] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.25, 0.5, 0.75, 1.0],
        ];
        let mut idx =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::Int8).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            idx.add(&format!("row-{i}"), v).unwrap();
        }
        idx.build().unwrap();
        idx.save(&base).unwrap();

        assert!(
            base.with_extension(RESCORE_COMPANION_EXTENSION).exists(),
            "a quantized build must write the rescore companion"
        );

        let loaded =
            SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::Int8).unwrap();
        assert!(loaded.rescore.is_some());
        for (i, v) in vectors.iter().enumerate() {
            let exact = loaded.get_exact(&format!("row-{i}")).unwrap().unwrap();
            assert_eq!(
                exact,
                v.to_vec(),
                "get_exact must return the bit-exact f32 vector, not USearch's quantized \
                 reconstruction"
            );
        }
        assert_eq!(loaded.get_exact("row-missing").unwrap(), None);
    }

    #[test]
    fn get_exact_reads_the_in_memory_buffer_before_save() {
        // A freshly built quantized index — never saved or loaded — has no
        // rescore companion yet (`rescore` is only opened by `load`).
        // `get_exact` must still return the TRUE exact vector by reading the
        // in-memory `exact_vectors` buffer `add` accumulated, never falling
        // back to USearch's own lossy `get` under the "exact" name.
        let vectors: [[f32; 4]; 2] = [[1.0, 0.0, 0.0, 0.0], [0.25, 0.5, 0.75, 1.0]];
        let mut idx =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::Int8).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            idx.add(&format!("row-{i}"), v).unwrap();
        }
        idx.build().unwrap();
        assert!(
            idx.rescore.is_none(),
            "a pre-save index has no rescore companion yet"
        );

        for (i, v) in vectors.iter().enumerate() {
            let exact = idx.get_exact(&format!("row-{i}")).unwrap().unwrap();
            assert_eq!(
                exact,
                v.to_vec(),
                "get_exact on a pre-save quantized index must read the in-memory exact \
                 buffer, not USearch's own lossy quantized reconstruction"
            );
        }
    }

    #[test]
    fn get_exact_errors_on_a_quantized_index_with_neither_companion_nor_buffer() {
        // A state the constructors never produce (belt-and-suspenders): if a
        // quantized index somehow lost both its rescore companion and its
        // in-memory exact-vector buffer, `get_exact` must hard-error rather
        // than silently fall back to USearch's own lossy `get`.
        let mut idx =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::Int8).unwrap();
        idx.add("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx.exact_vectors = None;

        assert!(
            idx.get_exact("a").is_err(),
            "a quantized index with no rescore source of truth must error, not silently \
             return USearch's lossy reconstruction"
        );
    }

    // ─── Binary (B1) + Hamming ───────────────────────────────────────────────

    /// Deterministic, uncorrelated-looking `f32` values in `[-1, 1]` — a
    /// small stand-in for a real embedding row, seeded so every test run
    /// produces the identical corpus with no external `rand` dependency.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn synthetic_vector(seed: u64, dim: usize) -> Vec<f32> {
        let mut state = seed;
        (0..dim)
            .map(|_| {
                let bits = splitmix64(&mut state);
                ((bits >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0) as f32
            })
            .collect()
    }

    #[test]
    fn pack_threshold_bits_with_zero_threshold_matches_old_symmetric_sign_at_zero() {
        // dim = 9 -> ceil(9/8) = 2 bytes. An all-zero threshold reproduces
        // the pre-fix symmetric packing exactly: v > 0 -> bit 1 (including
        // the padding-adjacent index 8), else (including exactly 0.0) -> bit
        // 0, LSB-first within each byte, unused high bits left 0.
        let v = [1.0, -1.0, 0.0, 2.0, -0.5, 0.5, -3.0, 4.0, 0.1];
        let zero_threshold = [0.0f32; 9];
        let packed = pack_threshold_bits(&v, &zero_threshold);
        assert_eq!(packed.len(), 2);
        // Bits 0, 3, 5, 7 set (the positive entries) -> 1 + 8 + 32 + 128.
        assert_eq!(packed[0], 0b1010_1001);
        // Only bit 0 of the second byte is set; bits 1..7 are zero padding.
        assert_eq!(packed[1], 0b0000_0001);
    }

    #[test]
    fn pack_threshold_bits_shifts_the_boundary_per_dimension() {
        // A non-zero, per-dimension threshold moves the bit boundary away
        // from 0 independently per dimension — the asymmetric generalisation
        // `pack_sign_bits` (the old fixed-0 packer) could not express.
        let v = [0.4, 0.6, -0.1, -0.1];
        let threshold = [0.5, 0.5, -0.2, 0.0];
        let packed = pack_threshold_bits(&v, &threshold);
        // 0.4 <= 0.5 -> 0; 0.6 > 0.5 -> 1; -0.1 > -0.2 -> 1; -0.1 <= 0.0 -> 0.
        assert_eq!(packed, vec![0b0000_0110]);
    }

    #[test]
    fn binary_query_equal_to_corpus_vector_is_its_own_nearest_at_hamming_zero() {
        // The corpus rows and the query pack through the SAME
        // `pack_threshold_bits` against the SAME corpus-fit τ (both go
        // through `SidecarIndex::build`/`search`), so a query that IS
        // one of the corpus's own vectors must be its own nearest neighbour at
        // Hamming distance exactly 0 — any divergence between the add-side and
        // query-side packing would show up here as a nonzero distance.
        let dim = 64;
        let vectors: Vec<Vec<f32>> = (0..12).map(|i| synthetic_vector(i + 1, dim)).collect();

        let mut idx =
            SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Binary).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            idx.add(&format!("row-{i}"), v).unwrap();
        }
        idx.build().unwrap();

        let query = vectors[5].clone();
        let hits = idx.search(&query, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "row-5");
        assert_eq!(
            hits[0].1, 0.0,
            "a query bit-identical (post sign-threshold) to its own corpus row must be at \
             Hamming distance 0"
        );
    }

    #[test]
    fn binary_search_with_rescore_matches_exact_f32_baseline() {
        // Mirrors `int8_search_with_rescore_matches_exact_f32_baseline`
        // (`crates/jammi-ai/tests/it/storage_precision.rs`) one layer down, at
        // the `SidecarIndex` this engine unit owns: build a `Binary` sidecar
        // over a small corpus, run the retrieve->rescore mechanism production
        // uses (`jammi_ai::operator::ann_search_exec::retrieve_then_rescore`)
        // by hand over `SidecarIndex::search` + `get_exact`, and compare
        // against an independently-computed exact brute-force cosine ranking
        // over the ORIGINAL `f32` vectors.
        use jammi_numerics::distance::cosine_distance;

        let dim = 64;
        let corpus_len = 24;
        let vectors: Vec<(String, Vec<f32>)> = (0..corpus_len)
            .map(|i| (format!("row-{i}"), synthetic_vector(i as u64 + 100, dim)))
            .collect();

        let mut idx =
            SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Binary).unwrap();
        for (id, v) in &vectors {
            idx.add(id, v).unwrap();
        }
        idx.build().unwrap();

        // Save + reload before searching: `get_exact` serves the exact vector
        // from the `.rawf32` rescore companion only once it is open (populated
        // by `load`, mirroring exactly how production always searches a
        // quantized sidecar — through `resolve_search_mode`'s `load`, never a
        // still-building handle).
        let dir = tempdir().unwrap();
        let base = dir.path().join("binary_rescore_roundtrip");
        idx.save(&base).unwrap();
        let idx = SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::Binary)
            .unwrap();

        // A query that is a small perturbation of an existing corpus row, so
        // it has one clear exact nearest neighbour among otherwise
        // uncorrelated synthetic vectors.
        let mut query = vectors[9].1.clone();
        let noise = synthetic_vector(999, dim);
        for (q, n) in query.iter_mut().zip(noise.iter()) {
            *q += 0.05 * n;
        }

        let k = 5;
        // The Wave 1.5 spike's confirmed mandatory oversample for Binary.
        let oversample = StoragePrecision::Binary.default_oversample();
        let candidate_k = (k * oversample).min(vectors.len());

        let candidates = idx.search(&query, candidate_k).unwrap();
        assert_eq!(
            candidates.len(),
            vectors.len(),
            "oversample=32 must widen the Hamming coarse stage to cover this whole \
             24-row fixture"
        );
        let mut rescored: Vec<(String, f32)> = candidates
            .iter()
            .map(|(id, _)| {
                let exact = idx
                    .get_exact(id)
                    .unwrap()
                    .expect("every Hamming candidate must have an exact rescore companion entry");
                (id.clone(), cosine_distance(&query, &exact))
            })
            .collect();
        rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        rescored.truncate(k);

        let mut expected: Vec<(String, f32)> = vectors
            .iter()
            .map(|(id, v)| (id.clone(), cosine_distance(&query, v)))
            .collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        expected.truncate(k);

        let actual_ids: Vec<&str> = rescored.iter().map(|(id, _)| id.as_str()).collect();
        let expected_ids: Vec<&str> = expected.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            actual_ids, expected_ids,
            "Binary retrieve->rescore must recover the exact brute-force top-k row order"
        );
        assert_eq!(
            rescored[0].0, "row-9",
            "the perturbed corpus row must remain its own exact nearest neighbour"
        );
    }

    #[test]
    fn binary_manifest_records_binary_scalar_kind_and_hamming_metric() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("binary_manifest");

        let mut idx =
            SidecarIndex::new(8, &AnnIndexConfig::default(), StoragePrecision::Binary).unwrap();
        let vector = [1.0, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 3.0];
        idx.add("a", &vector).unwrap();
        idx.build().unwrap();
        idx.save(&base).unwrap();

        let manifest = read_manifest_json(&base);
        assert_eq!(manifest["scalar_kind"], "binary");
        assert_eq!(
            manifest["metric"], "hamming",
            "a Binary build's manifest must record the metric it actually searches with, \
             not a hardcoded 'cosine'"
        );
        assert_eq!(
            manifest["binary_threshold_kind"], "median",
            "a Binary build's manifest must record which ThresholdKind fit τ"
        );
        assert!(
            base.with_extension(THRESHOLD_COMPANION_EXTENSION).exists(),
            "a Binary build must write the .threshold companion"
        );

        let loaded =
            SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::Binary)
                .unwrap();
        assert_eq!(loaded.storage_precision(), StoragePrecision::Binary);
        assert!(
            loaded.rescore.is_some(),
            "a Binary build must write the rescore companion"
        );
        assert_eq!(loaded.get_exact("a").unwrap(), Some(vector.to_vec()));
        assert_eq!(
            loaded.threshold_kind,
            Some(ThresholdKind::Median),
            "a loaded Binary index must recover the ThresholdKind its manifest recorded"
        );
        assert_eq!(
            loaded.binary_threshold.as_deref(),
            Some(vector.as_slice()),
            "a single-row corpus's per-dimension mean/median IS that row, so its own τ \
             round-trips to the vector itself"
        );
    }

    #[test]
    fn binary_scalar_kind_mismatch_is_rejected() {
        // The strict scalar-kind reject-mismatch path (already exercised for
        // F32-vs-Int8 by `mismatched_scalar_kind_is_rejected`) also holds for
        // the new `Binary` variant: a graph built as `Binary` but expected as
        // `F32` must fail loudly, never silently reopen as if the (very
        // differently shaped) stored vectors matched.
        let dir = tempdir().unwrap();
        let base = dir.path().join("binary_scalar_kind_mismatch");

        let mut idx =
            SidecarIndex::new(8, &AnnIndexConfig::default(), StoragePrecision::Binary).unwrap();
        idx.add("a", &[1.0, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 3.0])
            .unwrap();
        idx.build().unwrap();
        idx.save(&base).unwrap();

        match SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::F32) {
            Ok(_) => panic!("expected load to fail, but it succeeded"),
            Err(JammiError::IncompatibleFormat {
                artifact,
                found,
                supported,
            }) => {
                assert_eq!(artifact, "ann-index-precision");
                assert_eq!(found, "binary");
                assert_eq!(supported, "f32");
            }
            Err(other) => {
                panic!("expected IncompatibleFormat for ann-index-precision, got {other:?}")
            }
        }
    }

    // ─── Asymmetric (mean-centered) threshold: wave-2 anisotropy fix ────────

    /// A shared, large positive per-dimension bias — the synthetic stand-in
    /// for a real embedding corpus's dominant common-mean component
    /// (`‖μ‖/E‖v‖ ≈ 0.97` measured on ModernBERT): every row's value at
    /// dimension `d` sits close to `bias[d]` (`[3.0, 3.5)`), with only a
    /// small (`[-1, 1]`) per-row noise residual distinguishing rows. A fixed
    /// threshold at `0` sees only the bias and collapses every dimension to
    /// a constant bit; a corpus-fit τ ≈ `bias` cancels it and exposes the
    /// noise residual.
    fn anisotropic_bias(dim: usize) -> Vec<f32> {
        let mut state = 1_000_000u64;
        (0..dim)
            .map(|_| {
                let bits = splitmix64(&mut state);
                let u = (bits >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
                (3.0 + 0.5 * u) as f32
            })
            .collect()
    }

    /// One anisotropic row: `bias[d] + noise[d]`, `noise` uniform in `[-1,
    /// 1]` via [`synthetic_vector`]. `bias` dominates (`>= 3.0` vs. noise's
    /// `<= 1.0` magnitude), so `v[d] > 0` for every row at every dimension —
    /// the fully-collapsed extreme of the real anisotropy this fix targets.
    fn anisotropic_vector(seed: u64, dim: usize, bias: &[f32]) -> Vec<f32> {
        let noise = synthetic_vector(seed, dim);
        bias.iter()
            .zip(noise.iter())
            .map(|(&b, &n)| b + n)
            .collect()
    }

    /// Popcount of `a XOR b` over equal-length packed-bit buffers — an exact
    /// (non-approximate) Hamming distance, used by the hand-rolled coarse
    /// stage below so the RED-side measurement is never contaminated by
    /// USearch's own HNSW approximation.
    fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    /// How many of `dim` dimensions have an IDENTICAL sign bit
    /// (`v[d] > threshold[d]`) across every row of `vectors` — USearch's
    /// Hamming metric can never discriminate on a collapsed dimension, since
    /// it contributes the same bit to every corpus row's code.
    fn count_collapsed_dims(vectors: &[Vec<f32>], dim: usize, threshold: &[f32]) -> usize {
        (0..dim)
            .filter(|&d| {
                let first = vectors[0][d] > threshold[d];
                vectors.iter().all(|v| (v[d] > threshold[d]) == first)
            })
            .count()
    }

    /// Exact brute-force cosine top-`k` over `corpus` — the ground-truth
    /// oracle every retrieve→rescore recall measurement below is checked
    /// against.
    fn brute_force_top_k(query: &[f32], corpus: &[(String, Vec<f32>)], k: usize) -> Vec<String> {
        use jammi_numerics::distance::cosine_distance;
        let mut ranked: Vec<(String, f32)> = corpus
            .iter()
            .map(|(id, v)| (id.clone(), cosine_distance(query, v)))
            .collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked.into_iter().map(|(id, _)| id).collect()
    }

    /// A hand-rolled retrieve→rescore over `corpus`'s codes at an arbitrary
    /// `threshold` — an exact brute-force Hamming coarse stage (never
    /// USearch's approximate HNSW), so this is reusable to measure ANY
    /// [`ThresholdKind`] fit (or the pre-fix all-zero threshold) on a level,
    /// backend-independent footing.
    fn manual_threshold_retrieve_then_rescore(
        query: &[f32],
        corpus: &[(String, Vec<f32>)],
        threshold: &[f32],
        k: usize,
        candidate_k: usize,
    ) -> Vec<String> {
        use jammi_numerics::distance::cosine_distance;
        let query_code = pack_threshold_bits(query, threshold);
        let mut candidates: Vec<(String, u32)> = corpus
            .iter()
            .map(|(id, v)| {
                (
                    id.clone(),
                    hamming_distance(&query_code, &pack_threshold_bits(v, threshold)),
                )
            })
            .collect();
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(candidate_k);

        let mut rescored: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|(id, _)| {
                let v = &corpus.iter().find(|(cid, _)| cid == &id).unwrap().1;
                (id, cosine_distance(query, v))
            })
            .collect();
        rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        rescored.truncate(k);
        rescored.into_iter().map(|(id, _)| id).collect()
    }

    /// A retrieve→rescore over a real, built [`SidecarIndex`] — the SAME
    /// production mechanism [`binary_search_with_rescore_matches_exact_f32_baseline`]
    /// exercises, factored out for the RED→GREEN oracle below.
    fn index_retrieve_then_rescore(
        index: &SidecarIndex,
        query: &[f32],
        k: usize,
        candidate_k: usize,
    ) -> Vec<String> {
        use jammi_numerics::distance::cosine_distance;
        let candidates = index.search(query, candidate_k).unwrap();
        let mut rescored: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|(id, _)| {
                let exact = index.get_exact(&id).unwrap().unwrap();
                (id, cosine_distance(query, &exact))
            })
            .collect();
        rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        rescored.truncate(k);
        rescored.into_iter().map(|(id, _)| id).collect()
    }

    /// Mean recall@k of `predicted` against `truth` (both already truncated
    /// to `k`), one query per pair.
    fn mean_recall_at_k(predicted: &[Vec<String>], truth: &[Vec<String>], k: usize) -> f64 {
        let total: usize = predicted
            .iter()
            .zip(truth.iter())
            .map(|(p, t)| p.iter().filter(|id| t.contains(*id)).count())
            .sum();
        total as f64 / (predicted.len() * k) as f64
    }

    #[test]
    fn asymmetric_threshold_eliminates_collapsed_dims_and_improves_recall_on_anisotropic_corpus() {
        // RED under the old fixed-0 symmetric threshold: a shared
        // per-dimension bias (>= 3.0) dominates a small (<= 1.0) per-row
        // noise residual, so `v[d] > 0` holds for EVERY row at EVERY
        // dimension — full collapse, the extreme of the real
        // ‖μ‖/E‖v‖ ≈ 0.97 anisotropy this fix targets. GREEN once τ is fit
        // at the corpus mean: it cancels the bias and exposes the noise
        // residual as genuine per-row bit variation. This test fails (both
        // assertions) if [`SidecarIndex::build`]'s Binary path is ever
        // reverted to the old fixed-0 packing.
        let dim = 64;
        let corpus_n = 300;
        let k = 10;
        let candidate_k = k * 4; // << corpus_n, so the coarse stage's
                                 // discriminative power actually matters —
                                 // a wider oversample would rescore the
                                 // whole corpus and hide any coarse-stage
                                 // difference.
        let bias = anisotropic_bias(dim);

        let corpus: Vec<(String, Vec<f32>)> = (0..corpus_n)
            .map(|i| {
                (
                    format!("row-{i}"),
                    anisotropic_vector(i as u64 + 1, dim, &bias),
                )
            })
            .collect();
        let corpus_vectors: Vec<Vec<f32>> = corpus.iter().map(|(_, v)| v.clone()).collect();

        // Held-out queries: fresh seeds, disjoint from the corpus's.
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|i| anisotropic_vector(500_000 + i, dim, &bias))
            .collect();

        // ── Collapsed-dim count ──
        let zero_threshold = vec![0f32; dim];
        let collapsed_at_zero = count_collapsed_dims(&corpus_vectors, dim, &zero_threshold);
        assert_eq!(
            collapsed_at_zero, dim,
            "the synthetic anisotropic corpus must fully collapse under the old fixed-0 \
             threshold (every dimension's bias dominates its noise residual)"
        );

        let mut idx =
            SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Binary).unwrap();
        for (id, v) in &corpus {
            idx.add(id, v).unwrap();
        }
        idx.build().unwrap();
        let fitted_threshold = idx.binary_threshold.clone().unwrap();
        let collapsed_at_mean = count_collapsed_dims(&corpus_vectors, dim, &fitted_threshold);
        assert!(
            collapsed_at_mean < collapsed_at_zero,
            "a corpus-fit τ must eliminate collapsed dims the old fixed-0 threshold produced: \
             zero={collapsed_at_zero} mean={collapsed_at_mean}"
        );
        assert_eq!(
            collapsed_at_mean, 0,
            "with 300 rows of genuine per-row noise, no dimension should remain collapsed once \
             τ cancels the shared bias, got {collapsed_at_mean}"
        );

        // ── Recall@10 ──
        let truth: Vec<Vec<String>> = queries
            .iter()
            .map(|q| brute_force_top_k(q, &corpus, k))
            .collect();
        let plain_sign_predicted: Vec<Vec<String>> = queries
            .iter()
            .map(|q| {
                manual_threshold_retrieve_then_rescore(q, &corpus, &zero_threshold, k, candidate_k)
            })
            .collect();
        let asymmetric_predicted: Vec<Vec<String>> = queries
            .iter()
            .map(|q| index_retrieve_then_rescore(&idx, q, k, candidate_k))
            .collect();

        let plain_sign_recall = mean_recall_at_k(&plain_sign_predicted, &truth, k);
        let asymmetric_recall = mean_recall_at_k(&asymmetric_predicted, &truth, k);
        assert!(
            asymmetric_recall > plain_sign_recall,
            "asymmetric (mean-centered) threshold must beat the old fixed-0 symmetric \
             threshold on this anisotropic corpus: plain_sign={plain_sign_recall:.3} \
             asymmetric={asymmetric_recall:.3}"
        );
    }

    #[test]
    fn mean_vs_median_threshold_measured_on_anisotropic_corpus() {
        // "Decide by measurement, not guess": fits BOTH ThresholdKind
        // reductions on the identical anisotropic fixture the RED→GREEN
        // oracle above uses, ranks each through the SAME hand-rolled exact
        // (non-approximate) coarse+rescore mechanism — so the comparison
        // measures only the two reductions' discriminative quality, never
        // USearch's HNSW approximation — and asserts
        // [`DEFAULT_BINARY_THRESHOLD_KIND`] is the one that measured at
        // least as well.
        let dim = 64;
        let corpus_n = 300;
        let k = 10;
        let candidate_k = k * 4;
        let bias = anisotropic_bias(dim);

        let corpus: Vec<(String, Vec<f32>)> = (0..corpus_n)
            .map(|i| {
                (
                    format!("row-{i}"),
                    anisotropic_vector(i as u64 + 1, dim, &bias),
                )
            })
            .collect();
        let flat: Vec<f32> = corpus.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|i| anisotropic_vector(500_000 + i, dim, &bias))
            .collect();
        let truth: Vec<Vec<String>> = queries
            .iter()
            .map(|q| brute_force_top_k(q, &corpus, k))
            .collect();

        let recall_for = |kind: ThresholdKind| -> f64 {
            let threshold = fit_binary_threshold(&flat, dim, kind);
            let predicted: Vec<Vec<String>> = queries
                .iter()
                .map(|q| {
                    manual_threshold_retrieve_then_rescore(q, &corpus, &threshold, k, candidate_k)
                })
                .collect();
            mean_recall_at_k(&predicted, &truth, k)
        };

        let mean_recall = recall_for(ThresholdKind::Mean);
        let median_recall = recall_for(ThresholdKind::Median);
        let default_recall = recall_for(DEFAULT_BINARY_THRESHOLD_KIND);

        assert!(
            default_recall >= mean_recall.max(median_recall) - f64::EPSILON,
            "DEFAULT_BINARY_THRESHOLD_KIND ({DEFAULT_BINARY_THRESHOLD_KIND:?}) must match the \
             measured best of the two kinds: mean={mean_recall:.3} median={median_recall:.3} \
             default={default_recall:.3}"
        );
    }

    #[test]
    fn binary_threshold_round_trips_through_save_and_load_and_query_uses_it() {
        let dim = 32;
        let mut idx =
            SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Binary).unwrap();
        let vectors: Vec<Vec<f32>> = (0..40).map(|i| synthetic_vector(i + 1, dim)).collect();
        for (i, v) in vectors.iter().enumerate() {
            idx.add(&format!("row-{i}"), v).unwrap();
        }
        idx.build().unwrap();
        let built_threshold = idx.binary_threshold.clone().unwrap();
        assert_eq!(idx.threshold_kind, Some(DEFAULT_BINARY_THRESHOLD_KIND));

        let dir = tempdir().unwrap();
        let base = dir.path().join("binary_threshold_roundtrip");
        idx.save(&base).unwrap();
        let loaded =
            SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::Binary)
                .unwrap();

        assert_eq!(
            loaded.binary_threshold,
            Some(built_threshold),
            "τ must round-trip bit-for-bit through the .threshold companion"
        );
        assert_eq!(loaded.threshold_kind, idx.threshold_kind);

        // The query must be thresholded with the SAME τ: a query
        // bit-identical to a corpus row (post-threshold) is its own nearest
        // neighbour at Hamming 0 whether we search the freshly-built or the
        // reloaded index.
        let query = vectors[7].clone();
        let hits_built = idx.search(&query, 1).unwrap();
        let hits_loaded = loaded.search(&query, 1).unwrap();
        assert_eq!(hits_built, hits_loaded);
        assert_eq!(hits_loaded[0].0, "row-7");
        assert_eq!(hits_loaded[0].1, 0.0);
    }

    #[test]
    fn binary_threshold_is_deterministic_across_builds_of_the_same_corpus() {
        let dim = 40;
        let vectors: Vec<(String, Vec<f32>)> = (0..50)
            .map(|i| (format!("row-{i}"), synthetic_vector(i + 1, dim)))
            .collect();

        let build = || {
            let mut idx =
                SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Binary)
                    .unwrap();
            for (id, v) in &vectors {
                idx.add(id, v).unwrap();
            }
            idx.build().unwrap();
            idx
        };

        let idx_a = build();
        let idx_b = build();

        assert_eq!(
            idx_a.binary_threshold, idx_b.binary_threshold,
            "the same corpus must fit an identical τ every build"
        );
        assert_eq!(idx_a.threshold_kind, idx_b.threshold_kind);

        // Same τ implies identical packed codes, so an identical query
        // returns identical results both times.
        let query = vectors[10].1.clone();
        assert_eq!(
            idx_a.search(&query, 5).unwrap(),
            idx_b.search(&query, 5).unwrap()
        );
    }

    #[test]
    fn rescore_is_byte_identical_regardless_of_binary_threshold() {
        // The exact-f32 rescore companion is populated from `exact_vectors`
        // (`add`'s ORIGINAL, un-thresholded vectors) and never touched by τ,
        // so `get_exact` must return byte-identical results no matter which
        // threshold the coarse Hamming stage used — τ can only ever change
        // the coarse CANDIDATE SET, never the exact rescore values.
        let dim = 64;
        let bias = anisotropic_bias(dim);
        let vectors: Vec<(String, Vec<f32>)> = (0..30)
            .map(|i| {
                (
                    format!("row-{i}"),
                    anisotropic_vector(i as u64 + 1, dim, &bias),
                )
            })
            .collect();

        let mut idx =
            SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Binary).unwrap();
        for (id, v) in &vectors {
            idx.add(id, v).unwrap();
        }
        idx.build().unwrap();

        let exact_with_fitted_threshold: Vec<Vec<f32>> = vectors
            .iter()
            .map(|(id, _)| idx.get_exact(id).unwrap().unwrap())
            .collect();
        let query = vectors[3].1.clone();
        let candidates_with_fitted = idx.search(&query, vectors.len()).unwrap();

        // Swap in a DIFFERENT threshold (the pre-fix all-zero one) directly
        // on the already-built index — a private-field test-only override,
        // never a public API — to isolate τ's effect to the coarse stage
        // alone; `exact_vectors`/the `.rawf32` rescore path never reads this
        // field.
        idx.binary_threshold = Some(vec![0.0; dim]);

        let exact_with_zero_threshold: Vec<Vec<f32>> = vectors
            .iter()
            .map(|(id, _)| idx.get_exact(id).unwrap().unwrap())
            .collect();
        let candidates_with_zero = idx.search(&query, vectors.len()).unwrap();

        assert_eq!(
            exact_with_fitted_threshold, exact_with_zero_threshold,
            "get_exact/rescore must be byte-identical regardless of the binary threshold — it \
             reads exact_vectors, which τ never touches"
        );
        assert_ne!(
            candidates_with_fitted, candidates_with_zero,
            "the coarse Hamming ranking DOES depend on τ — on this anisotropic corpus a \
             mismatched query-side threshold must change the candidate ranking, otherwise this \
             test is not distinguishing anything"
        );
    }
}
