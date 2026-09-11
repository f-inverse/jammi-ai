//! A table's ANN index as a **set of segments**.
//!
//! One embedding table's index is not a single graph but a set of immutable
//! [`SidecarIndex`] segments, each over a disjoint subset of the table's rows
//! with its own bundle. Appending a batch of new rows writes a *new* segment
//! and leaves every existing one untouched — the row-set grows without
//! rebuilding any graph. [`SegmentedIndex`] owns the read side: it fans a query
//! across every segment and merges the results under one total order, so a
//! caller searches the whole table through a single handle regardless of how
//! many segments back it. A table with one segment (`N = 1`, a freshly-built
//! table that has never been appended to) is the same operator at scale `1`,
//! not a special case.
//!
//! ## Two entry points, one comparability contract
//!
//! [`SegmentedIndex::search`] is the *raw candidate primitive*: it over-fetches
//! from each segment, concatenates, dedups by row id, and orders by
//! `(distance, row_id, segment_id)`. For an `F32` table those per-segment
//! distances are exact cosine and directly comparable across segments, so the
//! merged order is final. For a quantized (`F16` / `Int8`) or `Binary` table
//! they are per-segment *approximate* distances — and, for `Binary`, fit
//! against each segment's own corpus threshold τ, so not even on the same scale
//! across segments — so they are candidates only, never a final answer.
//!
//! [`SegmentedIndex::search_final`] is the *single final-results entry* every
//! consumer routes through. On a table that
//! [`needs_rescore`](StoragePrecision::needs_rescore) it retrieves an
//! oversampled candidate set through `search`, reads each candidate's exact
//! `f32` vector via its owning segment, recomputes cosine against the
//! query, and re-ranks by that exact distance — a corpus-independent order that
//! *is* comparable across segments. On an `F32` table it is `search` directly.
//! Routing every final read through `search_final` is what keeps a
//! multi-segment quantized table's answer exact-comparable: a consumer that
//! returned raw `search` output as final would surface per-segment
//! non-comparable distances as if they were a global ranking.
//!
//! ## Kernels
//!
//! `search_final` is expressed over three pure, sync kernels that carry no
//! transport: [`search_unit`] (one segment's hits at a width, rescored at the
//! segment for a `Final` phase on a quantized precision), [`merge`] (the total
//! order + dedup over units, keeping the winning segment id) and [`rescore`]
//! (exact cosine over named candidates; a missing exact vector is a hard
//! error). The same kernels run at a segment owner and at a coordinator
//! ([`crate::index::placed::PlacedIndex`]) — a placed search over `N` nodes is
//! the same merge, fanned out. The exact-vector lookup is a closure, so an
//! exact-read count is observable without instrumenting [`SidecarIndex`].
//!
//! Per precision: `F32` is one `Final` phase (no rescore). `F16` / `Int8`
//! approximate distances are comparable across segments (usearch scales each
//! vector by its own magnitude), so they retrieve at width, merge, truncate to
//! `k · oversample`, then rescore exactly those survivors — `candidate_k`
//! exact reads. `Binary` per-segment Hamming distances are fit against each
//! segment's own τ and are NOT on one scale, so every segment rescores its own
//! `width` hits before the merge and the merge runs on final distance —
//! `N · width` exact reads, paid only here, where a raw-Hamming truncation
//! would keep the wrong segment's row.
//!
//! ## Deferred seams
//!
//! Several extensions fit this shape but are not built here: an mmap `view()`
//! of a segment's vectors; re-quantization or compaction of the segment set (merging
//! many small segments into one, at which point the
//! authoritative-vector-on-re-embed question below becomes live); and a
//! compaction *policy*. A small appended segment — especially a `Binary` one,
//! whose τ is fit from few rows and is therefore noisy — has weaker per-segment
//! recall; the merge over-fetch feeding `search_final`'s exact rescore is what
//! keeps that from degrading the final answer, but compaction is the eventual
//! structural fix.

use std::collections::HashSet;
use std::sync::Arc;

use jammi_numerics::distance::cosine_distance;

use crate::config::StoragePrecision;
use crate::error::{JammiError, Result};
use crate::index::first_inadmissible_hit;
use crate::index::peer::SegmentSearchPhase;
use crate::index::sidecar::SidecarIndex;
use crate::index::VectorIndex;
use crate::store::deletes::DeletionMask;

/// A segment's position in its table's segment sequence — the `segment_id`
/// catalog column, starting at `0` for the first segment a fresh embedding
/// table writes. A newtype so it is never confused with a row count, a `k`, or
/// an internal USearch key at a call boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(pub i64);

/// The over-fetch multiplier each segment is searched at before the merge, when
/// there is more than one segment.
///
/// A `k`-nearest merge over `N` segments cannot ask each segment for only `k`:
/// the true global top-`k` may draw more than `k` of its members from one
/// segment, and an HNSW graph's own recall is imperfect, so each segment is
/// asked for `ceil(k * factor)` and the merge keeps the global top. `1.0` would
/// be no over-fetch (correct only for `N = 1`); `2.0` is the measured default —
/// generous enough that the merge's recall tracks a single graph's, cheap
/// enough that it does not dominate query cost.
///
/// SEAM: this factor's recall benefit is *recovering per-segment HNSW recall
/// loss* — it only bites when a segment's own graph returns less than its exact
/// top, a scale/seed-dependent property that is not robustly isolable at
/// deterministic unit scale (the merge-correctness unit tests below hold at
/// `1.0` too). Its floor is therefore guarded by a committed multi-seed recall
/// bench in `jammi-bench` (a follow-up), not by a single-seed unit assertion —
/// naming the gap rather than pretending a flaky unit test closes it.
pub const DEFAULT_SEGMENT_OVERFETCH_FACTOR: f32 = 2.0;

/// Per-segment fetch width for a merge that wants a global top-`m` over
/// `n_segments` segments. A single segment is asked for exactly `m` (its own
/// top-`m` *is* the global top-`m`, so no over-fetch — this is what makes `N=1`
/// byte-identical to a lone [`SidecarIndex::search`]); more than one is asked
/// for `ceil(m * DEFAULT_SEGMENT_OVERFETCH_FACTOR)`.
pub(crate) fn over_fetch(m: usize, n_segments: usize) -> usize {
    if n_segments <= 1 {
        m
    } else {
        (m as f32 * DEFAULT_SEGMENT_OVERFETCH_FACTOR).ceil() as usize
    }
}

/// The exact-vector lookup a rescore reads through: `Ok(None)` when the id is
/// not indexed by the segment the lookup is bound to.
pub type ExactLookup<'a> = dyn Fn(&str) -> Result<Option<Vec<f32>>> + 'a;

/// An [`ExactLookup`] dispatched by segment: called as `(winning_segment,
/// row_id)` for every exact read a multi-segment `search_final` makes.
pub(crate) type SegmentExactLookup<'a> = dyn Fn(SegmentId, &str) -> Result<Option<Vec<f32>>> + 'a;

/// One segment's hits for one query at `width` — the per-segment kernel.
///
/// Searches the graph at `width` (capped at the segment's row count by the
/// index itself, never padded). For a [`SegmentSearchPhase::Final`] phase on a
/// precision that [`needs_rescore`](StoragePrecision::needs_rescore), every
/// hit is rescored through `exact` + cosine so the returned distances are
/// final and comparable across segments (the `Binary` protocol); otherwise the
/// hits are returned as the graph produced them (`F32` exact, or `F16` /
/// `Int8` approximate candidates for an `Approximate` phase).
pub fn search_unit(
    segment: SegmentId,
    index: &SidecarIndex,
    query: &[f32],
    width: usize,
    phase: SegmentSearchPhase,
    exact: &ExactLookup<'_>,
) -> Result<Vec<(String, f32)>> {
    verify_query_width(segment, index, query)?;
    if width == 0 {
        return Ok(Vec::new());
    }
    let hits = index.search(query, width)?;
    if phase == SegmentSearchPhase::Final && index.storage_precision().needs_rescore() {
        rescore(segment, hits, exact, query)
    } else {
        admissible_or_err(segment, hits)
    }
}

/// The query must be exactly as wide as the index it is searched against.
///
/// Checked at the SEARCH ENTRY, against the INDEX's own width — the
/// authoritative one — rather than the catalog's `dimensions` column, which
/// is `Option<i32>` metadata with a live `None` branch. A wrong-width query
/// is a CALLER fault, and without this it surfaces as something else
/// entirely: at `Binary` the query packs to `ceil(len/8)` bytes, so an
/// over-long query passes usearch untouched and reaches `cosine_distance`,
/// which is where it panics.
pub fn verify_query_width(segment: SegmentId, index: &SidecarIndex, query: &[f32]) -> Result<()> {
    if query.len() != index.dimensions() {
        return Err(JammiError::Schema {
            table: format!("segment {}", segment.0),
            column: "query".into(),
            expected: format!("{} dimensions", index.dimensions()),
            actual: format!("{} dimensions", query.len()),
        });
    }
    Ok(())
}

/// Every distance a LOCAL kernel produces must be admissible
/// ([`distance_is_admissible`]). A violation here is a broken index — a
/// non-finite component in a stored vector, or a backend that stopped
/// guarding zero magnitude — not a peer's fault, so it is a typed engine
/// error naming the poisoned SEGMENT rather than a ladder failure.
fn admissible_or_err(segment: SegmentId, hits: Vec<(String, f32)>) -> Result<Vec<(String, f32)>> {
    if let Some((row_id, distance)) = first_inadmissible_hit(&hits) {
        return Err(JammiError::Other(format!(
            "segment {}: row '{row_id}' has a non-finite distance ({distance:?}) — the index or \
             its stored vectors are corrupt; a distance is the merge's sort key and the \
             user-visible similarity, so it is refused rather than ranked",
            segment.0
        )));
    }
    Ok(hits)
}

/// The merge kernel: concatenate every unit, order by `(distance ASC, row_id
/// ASC, segment_id ASC)`, dedup by row id keeping the nearest occurrence, and
/// truncate to `m`. Each survivor carries the segment id that won it — the
/// segment its exact vector is read from.
pub(crate) fn merge(
    units: Vec<(SegmentId, Vec<(String, f32)>)>,
    m: usize,
) -> Vec<(String, f32, SegmentId)> {
    if m == 0 {
        return Vec::new();
    }
    let mut merged: Vec<(String, f32, SegmentId)> = units
        .into_iter()
        .flat_map(|(segment, hits)| {
            hits.into_iter()
                .map(move |(row_id, distance)| (row_id, distance, segment))
        })
        .collect();
    merged.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.2.cmp(&b.2))
    });
    // Dedup by row id keeping the nearest: after the sort the first
    // occurrence of each id is its nearest, so a set-membership pass keeps
    // that one and drops the rest, stopping once `m` survivors are collected.
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, f32, SegmentId)> = Vec::with_capacity(m.min(merged.len()));
    for (row_id, distance, segment) in merged {
        if out.len() == m {
            break;
        }
        if seen.insert(row_id.clone()) {
            out.push((row_id, distance, segment));
        }
    }
    out
}

/// The rescore kernel: read every candidate's exact vector through `exact`,
/// recompute cosine distance against `query`, and order by `(distance,
/// row_id)`. A candidate whose exact vector is missing (present in a graph but
/// absent from its rescore companion — a torn bundle) is a hard error, never a
/// silent drop: a result set that quietly shrank would read as "fewer matches
/// exist" rather than "the index is broken".
pub fn rescore(
    segment: SegmentId,
    candidates: Vec<(String, f32)>,
    exact: &ExactLookup<'_>,
    query: &[f32],
) -> Result<Vec<(String, f32)>> {
    let mut rescored: Vec<(String, f32)> = Vec::with_capacity(candidates.len());
    for (row_id, _approx) in candidates {
        let vector = exact(&row_id)?.ok_or_else(|| {
            JammiError::Other(format!(
                "rescore: candidate '{row_id}' has no exact vector in its segment's rescore \
                 companion (corrupted or torn sidecar bundle)"
            ))
        })?;
        // The stored vector is the authority on width here (this kernel may
        // be reached with no index in hand), so the mismatch is typed before
        // `cosine_distance` — which now refuses it — can be called.
        if vector.len() != query.len() {
            return Err(JammiError::Schema {
                table: format!("segment {}", segment.0),
                column: "query".into(),
                expected: format!("{} dimensions", vector.len()),
                actual: format!("{} dimensions", query.len()),
            });
        }
        let distance = cosine_distance(query, &vector);
        rescored.push((row_id, distance));
    }
    rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    admissible_or_err(segment, rescored)
}

/// A loaded segment paired with its catalog id (the merge's final tie-break
/// and the owner of a candidate's exact vector) and its producing version
/// (the horizon the deletion mask is compared against).
struct Segment {
    id: SegmentId,
    version: i64,
    index: SidecarIndex,
    /// `|{K : mask[K] >= version ∧ contains(K)}|`, computed once at
    /// construction — how many of this segment's rows the mask hides, which
    /// sizes the widened fetch.
    dead: usize,
}

/// The read-side handle over a table's whole segment set. Built by
/// [`crate::store::ResultStore::resolve_search_mode`] from the segments loaded
/// through the content-addressed segment cache; searched through
/// [`Self::search_final`] by every final-results consumer.
///
/// A versioned table's set carries the version's [`DeletionMask`]: a hit
/// `K` from a segment stamped `v` is dropped when `mask[K] >= v`, so a
/// superseded or deleted key never surfaces from the segment that still
/// indexes it while the segment that re-embedded it (stamped later) serves it
/// unmasked. A never-refreshed table's set carries an empty mask and every
/// segment at version `0`, so nothing is filtered and each segment is searched
/// exactly once at today's width (the N=1 byte-identity contract).
pub struct SegmentedIndex {
    segments: Vec<Segment>,
    /// The table-level storage precision. Uniform across segments *by
    /// enforcement*, not assumption: a segment built at a drifted precision
    /// fails [`SidecarIndex::load`]'s strict `scalar_kind` check and never
    /// reaches this constructor, so every segment here loaded at the same
    /// precision — the one this field reports.
    storage_precision: StoragePrecision,
    mask: Arc<DeletionMask>,
}

impl SegmentedIndex {
    /// Assemble a segmented index from the segments loaded for one
    /// never-refreshed table, in `segment_id` order: no mask, every segment
    /// at version `0`.
    ///
    /// The set must be non-empty (a table with no segments resolves to the
    /// exact-search fallback upstream, never to an empty `SegmentedIndex`) and
    /// every segment must share one storage precision — guaranteed by the
    /// per-segment strict load check, and re-asserted here so a future loader
    /// that bypasses it cannot silently assemble a mixed-precision set whose
    /// merge would compare distances on different scales.
    pub fn new(segments: Vec<(SegmentId, SidecarIndex)>) -> Result<Self> {
        Self::new_masked(
            segments.into_iter().map(|(id, idx)| (id, 0, idx)).collect(),
            Arc::new(DeletionMask::empty()),
        )
    }

    /// Assemble a versioned table's set: every segment with its stamped
    /// version, under the version's cumulative deletion mask.
    pub fn new_masked(
        segments: Vec<(SegmentId, i64, SidecarIndex)>,
        mask: Arc<DeletionMask>,
    ) -> Result<Self> {
        let mut iter = segments.into_iter();
        let (first_id, first_version, first) = iter.next().ok_or_else(|| {
            JammiError::Other("SegmentedIndex requires at least one segment".into())
        })?;
        let storage_precision = first.storage_precision();
        let dead = Self::count_dead(&mask, first_version, &first);
        let mut out = vec![Segment {
            id: first_id,
            version: first_version,
            index: first,
            dead,
        }];
        for (id, version, index) in iter {
            if index.storage_precision() != storage_precision {
                return Err(JammiError::Other(format!(
                    "SegmentedIndex: segment {} loaded at {:?} but the set is {:?} — \
                     mixed-precision segment sets are unsearchable (distances are not comparable)",
                    id.0,
                    index.storage_precision(),
                    storage_precision
                )));
            }
            let dead = Self::count_dead(&mask, version, &index);
            out.push(Segment {
                id,
                version,
                index,
                dead,
            });
        }
        Ok(Self {
            segments: out,
            storage_precision,
            mask,
        })
    }

    fn count_dead(mask: &DeletionMask, version: i64, index: &SidecarIndex) -> usize {
        mask.entries()
            .filter(|(k, h)| *h >= version && index.contains(k))
            .count()
    }

    /// The precision every segment in this set was built and loaded at.
    pub fn storage_precision(&self) -> StoragePrecision {
        self.storage_precision
    }

    /// Total number of rows across every segment (physical rows, masked
    /// included).
    pub fn len(&self) -> usize {
        self.segments().map(|(_, index)| index.len()).sum()
    }

    /// Every segment with its id, in `segment_id` order — the local sources a
    /// placed search fans in-process.
    pub(crate) fn segments(&self) -> impl Iterator<Item = (SegmentId, &SidecarIndex)> {
        self.segments.iter().map(|s| (s.id, &s.index))
    }

    /// Whether the whole set indexes zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The version's mask.
    pub fn mask(&self) -> &Arc<DeletionMask> {
        &self.mask
    }

    /// Per segment: the live candidates for a global top-`m` — the segment's
    /// own search at the over-fetch width, masked; when the segment has dead
    /// rows the width is scaled by `1 / (1 - dead / len)`, capped at the
    /// segment's length, and doubled until `m` live hits survive or the whole
    /// segment has been asked for. A segment with no dead rows is searched
    /// exactly once at today's width.
    fn live_candidates(
        &self,
        seg: &Segment,
        query: &[f32],
        m: usize,
    ) -> Result<Vec<(String, f32, SegmentId)>> {
        let len = seg.index.len();
        let base = over_fetch(m, self.segments.len());
        let mut w = if seg.dead > 0 && len > seg.dead {
            let live_fraction = 1.0 - (seg.dead as f32 / len as f32);
            let scaled = (base as f32 / live_fraction).ceil() as usize;
            scaled.max(base).min(len)
        } else if seg.dead > 0 {
            len
        } else {
            base
        };
        loop {
            // Through `search_unit` at the CANDIDATE phase (never rescored
            // here — today's behaviour exactly), so the width guard and the
            // admissibility check apply to the masked/versioned path too:
            // these raw per-segment distances are the merge's sort key, and
            // for `F32` they are also the final answer.
            let hits = search_unit(
                seg.id,
                &seg.index,
                query,
                w,
                SegmentSearchPhase::Approximate,
                &|row_id| seg.index.get_exact(row_id),
            )?;
            let live: Vec<(String, f32, SegmentId)> = hits
                .into_iter()
                .filter(|(k, _)| !self.mask.is_masked(k, seg.version))
                .map(|(k, d)| (k, d, seg.id))
                .collect();
            if seg.dead == 0 || live.len() >= m || w >= len {
                return Ok(live);
            }
            w = (w * 2).min(len);
        }
    }

    /// The RAW candidate merge primitive: search each segment (masked), order
    /// by `(distance ASC, row_id ASC, segment_id ASC)`, dedup by row id keeping
    /// the nearest occurrence, and truncate to `m`. Each candidate carries the
    /// segment that owns it.
    fn search_candidates(&self, query: &[f32], m: usize) -> Result<Vec<(String, f32, SegmentId)>> {
        if m == 0 {
            return Ok(Vec::new());
        }
        let mut merged: Vec<(String, f32, SegmentId)> = Vec::new();
        for seg in &self.segments {
            merged.extend(self.live_candidates(seg, query, m)?);
        }
        merged.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.2.cmp(&b.2))
        });
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<(String, f32, SegmentId)> = Vec::with_capacity(m.min(merged.len()));
        for (row_id, distance, segment) in merged {
            if out.len() == m {
                break;
            }
            if seen.insert(row_id.clone()) {
                out.push((row_id, distance, segment));
            }
        }
        Ok(out)
    }

    /// The RAW candidate merge as `(row_id, distance)`.
    ///
    /// For an `F32` set the distances are exact cosine and comparable across
    /// segments, so this order is final. For a quantized / `Binary` set they
    /// are per-segment approximate candidates — deterministically ordered here
    /// for a stable candidate set, but **not** a final answer; only
    /// [`Self::search_final`] consumes them, feeding the exact rescore that
    /// makes the final order comparable. The dedup keeps the merge correct over
    /// the row-id space: a key present in two segments (re-embedded by a
    /// refresh) is masked in the older one by version, so the dedup is defence
    /// only.
    pub fn search(&self, query: &[f32], m: usize) -> Result<Vec<(String, f32)>> {
        Ok(self
            .search_candidates(query, m)?
            .into_iter()
            .map(|(k, d, _)| (k, d))
            .collect())
    }

    /// THE single final-results entry. Returns the exact top-`k` in a total
    /// order comparable across every segment. Sync, all-local: every segment
    /// here is resident in this process.
    ///
    /// Per precision (the module docs' protocol, over the kernels), every
    /// stage routed through [`Self::live_candidates`]'s masked, dead-row
    /// widened fetch — a masked or superseded key is never counted against a
    /// live `k`:
    ///
    /// - `F32`: the merge is already exact and final, so this is
    ///   `search(query, k)` and `oversample` is unused.
    /// - `F16` / `Int8`: retrieve `k * oversample` masked candidates through
    ///   [`Self::search_candidates`] (the over-fetch nests *under* this
    ///   candidate width by construction), read each survivor's exact `f32`
    ///   vector from the segment that OWNS it (never the first segment that
    ///   happens to index the same key), recompute cosine distance against
    ///   the query, then re-rank by `(distance, row_id)` and truncate to `k`
    ///   — exactly `candidate_k` exact reads.
    /// - `Binary`: each segment rescores its own masked
    ///   `over_fetch(candidate_k, N)` hits BEFORE the merge, and the merge
    ///   runs on final distance — per-segment Hamming distances are fit
    ///   against each segment's own τ and are not on one scale, so a merge
    ///   that truncated on them would keep the wrong segment's row. At `N = 1`
    ///   with an empty mask this is the same bytes as the quantized path
    ///   (`width = candidate_k`, one rescore of the same set).
    ///
    /// A candidate whose exact vector is missing (present in a graph but
    /// absent from its rescore companion — a torn bundle) is a hard error,
    /// never a silent drop (see [`rescore`]).
    pub fn search_final(
        &self,
        query: &[f32],
        k: usize,
        oversample: usize,
    ) -> Result<Vec<(String, f32)>> {
        self.search_final_with(query, k, oversample, &|segment, row_id| {
            self.exact_in(segment, row_id)
        })
    }

    /// [`Self::search_final`] with the exact-vector lookup injected: `exact`
    /// is called as `(winning_segment, row_id)` for every exact read the
    /// protocol makes, so a caller can count reads without instrumenting the
    /// segments. Production calls this with [`Self::exact_in`].
    pub(crate) fn search_final_with(
        &self,
        query: &[f32],
        k: usize,
        oversample: usize,
        exact: &SegmentExactLookup<'_>,
    ) -> Result<Vec<(String, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        match self.storage_precision {
            StoragePrecision::F32 => self.search(query, k),
            StoragePrecision::F16 | StoragePrecision::Int8 => {
                let candidate_k = k.saturating_mul(oversample).max(k);
                let survivors = self.search_candidates(query, candidate_k)?;
                // Group by winning segment so each group rescores through the
                // one segment that owns it — the same shape a coordinator uses
                // when the groups are split between local and remote owners.
                let mut by_segment: Vec<(SegmentId, Vec<(String, f32)>)> = Vec::new();
                for (row_id, approx, segment) in survivors {
                    match by_segment.iter_mut().find(|(s, _)| *s == segment) {
                        Some((_, group)) => group.push((row_id, approx)),
                        None => by_segment.push((segment, vec![(row_id, approx)])),
                    }
                }
                let mut rescored: Vec<(String, f32)> = Vec::with_capacity(candidate_k);
                for (segment, group) in by_segment {
                    rescored.extend(rescore(
                        segment,
                        group,
                        &|row_id| exact(segment, row_id),
                        query,
                    )?);
                }
                rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                rescored.truncate(k);
                Ok(rescored)
            }
            StoragePrecision::Binary => {
                let candidate_k = k.saturating_mul(oversample).max(k);
                let mut units = Vec::with_capacity(self.segments.len());
                for seg in &self.segments {
                    let live: Vec<(String, f32)> = self
                        .live_candidates(seg, query, candidate_k)?
                        .into_iter()
                        .map(|(row_id, distance, _)| (row_id, distance))
                        .collect();
                    let rescored = rescore(seg.id, live, &|row_id| exact(seg.id, row_id), query)?;
                    units.push((seg.id, rescored));
                }
                Ok(merge(units, k)
                    .into_iter()
                    .map(|(row_id, distance, _segment)| (row_id, distance))
                    .collect())
            }
        }
    }

    /// The exact `f32` vector for `row_id` read from `segment` — the segment
    /// the merge said owns the row. `None` when that segment does not index
    /// the id (or the id names no segment of this set). Internal to the crate:
    /// the only consumer is the rescore stage — callers ask for final results,
    /// not raw exact vectors.
    pub(crate) fn exact_in(&self, segment: SegmentId, row_id: &str) -> Result<Option<Vec<f32>>> {
        match self.segments.iter().find(|s| s.id == segment) {
            Some(seg) => seg.index.get_exact(row_id),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AnnIndexConfig;
    use crate::index::VectorIndex;

    /// Build one segment (a fully-built [`SidecarIndex`]) over `rows` at
    /// `precision`. A freshly-built quantized index serves `get_exact` from its
    /// in-memory buffer, so `search_final` rescores without any save/load.
    fn segment(rows: &[(&str, Vec<f32>)], precision: StoragePrecision) -> SidecarIndex {
        let dim = rows[0].1.len();
        let mut idx = SidecarIndex::new(dim, &AnnIndexConfig::default(), precision).unwrap();
        for (id, v) in rows {
            idx.add(id, v).unwrap();
        }
        idx.build().unwrap();
        idx
    }

    /// One segment's rows paired with the precision to build them at — the input
    /// shape [`segmented`] assembles a multi-segment fixture from.
    type SegmentSpec<'a> = (&'a [(&'a str, Vec<f32>)], StoragePrecision);

    fn segmented(parts: Vec<SegmentSpec<'_>>) -> SegmentedIndex {
        let segs = parts
            .into_iter()
            .enumerate()
            .map(|(i, (rows, precision))| (SegmentId(i as i64), segment(rows, precision)))
            .collect();
        SegmentedIndex::new(segs).unwrap()
    }

    /// Exact brute-force top-`k` over the union, ordered `(cosine_distance,
    /// row_id)` — the corpus-independent oracle every merge is checked against.
    fn brute_force(rows: &[(&str, Vec<f32>)], query: &[f32], k: usize) -> Vec<String> {
        let mut scored: Vec<(String, f32)> = rows
            .iter()
            .map(|(id, v)| (id.to_string(), cosine_distance(query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    fn ids(hits: &[(String, f32)]) -> Vec<String> {
        hits.iter().map(|(id, _)| id.clone()).collect()
    }

    /// Twelve well-separated 8-d unit-ish vectors, split into two disjoint
    /// halves — a recall=1.0 corpus (a tiny HNSW returns exact neighbours), so a
    /// merge that tracks the brute-force order is provably correct, not merely
    /// approximately so.
    fn corpus() -> Vec<(&'static str, Vec<f32>)> {
        vec![
            ("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
            ("b", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
            ("c", vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
            ("d", vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4]),
            ("e", vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]),
            ("f", vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6]),
            ("g", vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("h", vec![0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("i", vec![0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("j", vec![0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0]),
            ("k", vec![0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0]),
            ("l", vec![0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0]),
        ]
    }

    // Test 1 — N=1 byte-identity, F32: the single-segment merge is exactly the
    // lone SidecarIndex's own search (over_fetch(m,1)=m, no reordering) on a
    // tie-free corpus.
    #[test]
    fn n1_f32_search_is_byte_identical_to_the_lone_sidecar() {
        let rows = corpus();
        let lone = segment(&rows, StoragePrecision::F32);
        let seg = segmented(vec![(&rows, StoragePrecision::F32)]);
        for q in [&rows[0].1, &rows[6].1, &rows[11].1] {
            for k in [1usize, 3, 5] {
                assert_eq!(
                    ids(&seg.search(q, k).unwrap()),
                    ids(&lone.search(q, k).unwrap()),
                    "N=1 F32 merge must equal the lone sidecar search"
                );
            }
        }
    }

    // Test 1 — N=1 byte-identity, quantized: search_final over one Int8 segment
    // equals a manual retrieve-then-rescore over that same sidecar (the
    // single-sidecar retrieve-then-rescore path search_final folds in).
    #[test]
    fn n1_int8_search_final_equals_manual_retrieve_then_rescore() {
        let rows = corpus();
        let lone = segment(&rows, StoragePrecision::Int8);
        let seg = segmented(vec![(&rows, StoragePrecision::Int8)]);
        let oversample = 4;
        for q in [&rows[0].1, &rows[6].1] {
            for k in [1usize, 3, 5] {
                // Manual reference: raw candidates, exact-rescored, sorted.
                let cand = lone.search(q, k.saturating_mul(oversample).max(k)).unwrap();
                let mut manual: Vec<(String, f32)> = cand
                    .into_iter()
                    .map(|(id, _)| {
                        let exact = lone.get_exact(&id).unwrap().unwrap();
                        (id, cosine_distance(q, &exact))
                    })
                    .collect();
                manual.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                manual.truncate(k);
                assert_eq!(
                    ids(&seg.search_final(q, k, oversample).unwrap()),
                    ids(&manual),
                    "N=1 Int8 search_final must equal manual retrieve→rescore"
                );
            }
        }
    }

    // Test 2 — two-segment merged top-k equals the brute-force baseline over the
    // union (F32, recall=1.0 corpus).
    #[test]
    fn two_segment_f32_merge_equals_brute_force() {
        let rows = corpus();
        let (left, right) = rows.split_at(6);
        let seg = segmented(vec![
            (left, StoragePrecision::F32),
            (right, StoragePrecision::F32),
        ]);
        for q in [&rows[0].1, &rows[3].1, &rows[9].1] {
            for k in [1usize, 3, 6] {
                assert_eq!(
                    ids(&seg.search_final(q, k, 4).unwrap()),
                    brute_force(&rows, q, k),
                    "2-segment F32 merge must equal brute-force top-k"
                );
            }
        }
    }

    // Test 9 (part) — two-segment Binary search_final equals the exact
    // brute-force baseline: the per-segment Hamming candidates are
    // exact-rescored into one cross-segment comparable order.
    #[test]
    fn two_segment_binary_search_final_equals_brute_force() {
        let rows = corpus();
        let (left, right) = rows.split_at(6);
        let seg = segmented(vec![
            (left, StoragePrecision::Binary),
            (right, StoragePrecision::Binary),
        ]);
        // A wide oversample covers the whole 12-row corpus so the coarse Hamming
        // stage cannot drop a true neighbour before the exact rescore.
        for q in [&rows[0].1, &rows[6].1, &rows[11].1] {
            for k in [1usize, 3] {
                assert_eq!(
                    ids(&seg.search_final(q, k, 32).unwrap()),
                    brute_force(&rows, q, k),
                    "2-segment Binary search_final must equal exact brute-force top-k"
                );
            }
        }
    }

    // Test 9b (A1) — the Binary two-segment `search_final` at `k = 1`,
    // `oversample = 1` (so `candidate_k = 1`, `width = over_fetch(1, 2) = 2`).
    // Per-segment Hamming distances are fit against each segment's OWN τ, so
    // they are not on one scale: a merge that truncates the candidate set on
    // raw Hamming before any rescore keeps the wrong segment's row. The
    // fixture is constructed so the true nearest row (`b0`, cosine ≈ 0.001)
    // sits at Hamming 1 in segment B while a far row (`a0`, cosine ≈ 0.667)
    // sits at Hamming 0 in segment A — `search_final` must return `b0`.
    #[test]
    fn two_segment_binary_search_final_rescores_per_segment_before_the_merge() {
        // dim 8, only the first four dims non-zero.
        let seg_a: Vec<(&str, Vec<f32>)> = vec![
            ("a0", vec![0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("a1", vec![0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("a2", vec![0.5, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let seg_b: Vec<(&str, Vec<f32>)> = vec![
            ("b0", vec![0.0, 0.0, 1.0, 0.05, 0.0, 0.0, 0.0, 0.0]),
            ("b1", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("b2", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let rows: Vec<(&str, Vec<f32>)> = seg_a.iter().chain(seg_b.iter()).cloned().collect();
        let q = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let seg = segmented(vec![
            (&seg_a, StoragePrecision::Binary),
            (&seg_b, StoragePrecision::Binary),
        ]);
        // Fixture premise, asserted so a usearch ordering change is loud: the
        // raw per-segment Hamming order puts `a0` (Hamming 0 under τ_A) ahead
        // of `b0` (Hamming 1 under τ_B) — the cross-segment incomparability
        // the final-distance merge must not be fooled by.
        let raw = seg.search(&q, 1).unwrap();
        assert_eq!(
            ids(&raw),
            vec!["a0".to_string()],
            "fixture premise: raw Hamming merge keeps a0 (distance 0 under τ_A)"
        );
        assert_eq!(
            ids(&seg.search_final(&q, 1, 1).unwrap()),
            brute_force(&rows, &q, 1),
            "2-segment Binary search_final must equal exact brute-force top-1 even when \
             candidate_k = 1: each segment rescores before the merge"
        );
    }

    // Test 9 (correctness under truncation) — a two-segment quantized
    // `search_final` returns the exact brute-force top-k over the union even when
    // the candidate stage truncates *below* the corpus size, so the rescore does
    // not simply re-rank every row. This is the assertion that BITES: it fails if
    // the merge drops a true neighbour (a mis-ordered or mis-truncated candidate
    // set, a lost segment).
    //
    // It does NOT isolate `DEFAULT_SEGMENT_OVERFETCH_FACTOR` — the factor only
    // pays off when a segment's own HNSW recall is below 100%, which is not
    // robustly reproducible at deterministic unit scale (this test holds at
    // factor `1.0` too). That floor is seamed to a multi-seed recall bench; see
    // the const's SEAM note.
    #[test]
    fn two_segment_quantized_search_final_equals_brute_force_under_truncation() {
        fn normalize(mut v: Vec<f32>) -> Vec<f32> {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        }
        let dim = 32;
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;

        // 5 "near" rows at strictly increasing (distinct) distance to the query,
        // and 123 "far" rows orthogonal to it (distance ~1.0). The gap between
        // the two families is far larger than Int8 quantization error, so the
        // true top-5 are unambiguous in both exact and quantized space — the test
        // is deterministic, not seed-fragile.
        let mut all: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..5 {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            v[1] = 0.08 * i as f32;
            all.push((format!("near{i}"), normalize(v)));
        }
        for r in 0..123 {
            let mut v = vec![0.0f32; dim];
            v[2 + (r % 28)] = 1.0;
            v[3 + (r % 28)] += 0.01 * ((r % 5) as f32);
            all.push((format!("far{r:03}"), normalize(v)));
        }

        // Even/odd split → two 64-row segments with the near rows spread across
        // BOTH, so recovering the top-k exercises the cross-segment merge.
        let mut seg_a: Vec<(String, Vec<f32>)> = Vec::new();
        let mut seg_b: Vec<(String, Vec<f32>)> = Vec::new();
        for (idx, row) in all.iter().enumerate() {
            if idx % 2 == 0 {
                seg_a.push(row.clone());
            } else {
                seg_b.push(row.clone());
            }
        }

        let build = |rows: &[(String, Vec<f32>)]| {
            let mut idx =
                SidecarIndex::new(dim, &AnnIndexConfig::default(), StoragePrecision::Int8).unwrap();
            for (id, v) in rows {
                idx.add(id, v).unwrap();
            }
            idx.build().unwrap();
            idx
        };
        let seg = SegmentedIndex::new(vec![
            (SegmentId(0), build(&seg_a)),
            (SegmentId(1), build(&seg_b)),
        ])
        .unwrap();

        // candidate_k = k * oversample = 5 * 4 = 20 < 128, and per-segment fetch
        // ceil(20 * 2.0) = 40 < 64: BOTH the per-segment fetch and the merge
        // truncate before the exact rescore — the truncation regime, not
        // rescore-everything.
        let k = 5;
        let got: Vec<String> = seg
            .search_final(&query, k, 4)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let mut scored: Vec<(String, f32)> = all
            .iter()
            .map(|(id, v)| (id.clone(), cosine_distance(&query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let expected: Vec<String> = scored.into_iter().take(k).map(|(id, _)| id).collect();

        assert_eq!(
            got, expected,
            "merged Int8 search_final must equal the exact brute-force top-k even when the \
             candidate stage truncates below the corpus size"
        );
        assert!(
            expected.iter().all(|id| id.starts_with("near")),
            "the true top-k are the five near rows"
        );
    }

    // A3 (sync entry) — N=1 byte identity at EVERY precision: `search_final`
    // over one segment returns the identical `(row_id, distance)` bytes a
    // manual retrieve→rescore over the lone `SidecarIndex` produces
    // (`over_fetch(m, 1) = m`, one rescore of the same candidate set, the same
    // `(distance, row_id)` order). The Binary twin is the one the per-segment
    // rescore fix must not move: at N=1, `width = candidate_k`, so "rescore
    // per segment then merge" and "merge then rescore" are the same bytes.
    #[test]
    fn n1_search_final_is_byte_identical_to_the_lone_sidecar_at_every_precision() {
        let rows = corpus();
        for precision in [
            StoragePrecision::F32,
            StoragePrecision::F16,
            StoragePrecision::Int8,
            StoragePrecision::Binary,
        ] {
            let lone = segment(&rows, precision);
            let seg = segmented(vec![(&rows, precision)]);
            for q in [&rows[0].1, &rows[6].1, &rows[11].1] {
                for (k, oversample) in [(1usize, 1usize), (3, 4), (5, 32)] {
                    let reference: Vec<(String, f32)> = if precision.needs_rescore() {
                        let cand = lone.search(q, k.saturating_mul(oversample).max(k)).unwrap();
                        let mut manual: Vec<(String, f32)> = cand
                            .into_iter()
                            .map(|(id, _)| {
                                let exact = lone.get_exact(&id).unwrap().unwrap();
                                (id, cosine_distance(q, &exact))
                            })
                            .collect();
                        manual.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                        manual.truncate(k);
                        manual
                    } else {
                        lone.search(q, k).unwrap()
                    };
                    assert_eq!(
                        seg.search_final(q, k, oversample).unwrap(),
                        reference,
                        "{precision:?} k={k} oversample={oversample}: N=1 search_final must be \
                         byte-identical to the lone sidecar"
                    );
                }
            }
        }
    }

    /// A 128-row, 32-d corpus with 5 near rows and 123 far rows (test 9's
    /// generator), split even/odd into two 64-row halves — sized so the
    /// exact-read equalities below are reachable (`search` caps at the
    /// segment's row count and never pads).
    type Rows = Vec<(String, Vec<f32>)>;

    fn wide_corpus() -> (Rows, Rows, Vec<f32>) {
        fn normalize(mut v: Vec<f32>) -> Vec<f32> {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        }
        let dim = 32;
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let mut all: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..5 {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            v[1] = 0.08 * i as f32;
            all.push((format!("near{i}"), normalize(v)));
        }
        for r in 0..123 {
            let mut v = vec![0.0f32; dim];
            v[2 + (r % 28)] = 1.0;
            v[3 + (r % 28)] += 0.01 * ((r % 5) as f32);
            all.push((format!("far{r:03}"), normalize(v)));
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (idx, row) in all.into_iter().enumerate() {
            if idx % 2 == 0 {
                a.push(row);
            } else {
                b.push(row);
            }
        }
        (a, b, query)
    }

    fn wide_segment(rows: &[(String, Vec<f32>)], precision: StoragePrecision) -> SidecarIndex {
        let mut idx = SidecarIndex::new(32, &AnnIndexConfig::default(), precision).unwrap();
        for (id, v) in rows {
            idx.add(id, v).unwrap();
        }
        idx.build().unwrap();
        idx
    }

    /// Run `search_final` through a counting exact-vector closure and return
    /// `(hits, exact_reads)`.
    fn count_exact_reads(
        seg: &SegmentedIndex,
        query: &[f32],
        k: usize,
        oversample: usize,
    ) -> (Vec<(String, f32)>, usize) {
        let reads = std::cell::Cell::new(0usize);
        let hits = seg
            .search_final_with(query, k, oversample, &|segment, row_id| {
                reads.set(reads.get() + 1);
                seg.exact_in(segment, row_id)
            })
            .unwrap();
        (hits, reads.get())
    }

    // A2 — the exact-read count per precision, on the kernels through a
    // counting closure (no `SidecarIndex` instrumentation). `k = 5`,
    // `oversample = 4` → `candidate_k = 20`:
    //   F16 / Int8 at N=1 and N=2 → exactly 20 (rescore only the merge's
    //     survivors — today's count);
    //   Binary at N=2 → `2 · over_fetch(20, 2) = 80` (every segment rescores
    //     its own width before the merge — the multiplier is paid only here);
    //   F32 → 0.
    // The counted call is the production `search_final` composition, so the
    // bytes it returns are asserted against the uncounted entry too.
    #[test]
    fn exact_read_count_per_precision() {
        let (a, b, query) = wide_corpus();
        let all: Vec<(String, Vec<f32>)> = a.iter().chain(b.iter()).cloned().collect();
        let (k, oversample) = (5usize, 4usize);
        let candidate_k = k * oversample;
        assert_eq!(over_fetch(candidate_k, 2), 40);
        assert!(
            a.len() >= 40 && b.len() >= 40,
            "≥ 40 rows per Binary segment"
        );

        let expect = |precision: StoragePrecision, n: usize| -> usize {
            match precision {
                StoragePrecision::F32 => 0,
                StoragePrecision::F16 | StoragePrecision::Int8 => candidate_k,
                StoragePrecision::Binary => n * over_fetch(candidate_k, n),
            }
        };
        for precision in [
            StoragePrecision::F32,
            StoragePrecision::F16,
            StoragePrecision::Int8,
            StoragePrecision::Binary,
        ] {
            let n1 =
                SegmentedIndex::new(vec![(SegmentId(0), wide_segment(&all, precision))]).unwrap();
            let n2 = SegmentedIndex::new(vec![
                (SegmentId(0), wide_segment(&a, precision)),
                (SegmentId(1), wide_segment(&b, precision)),
            ])
            .unwrap();
            for (n, seg) in [(1usize, &n1), (2usize, &n2)] {
                let (hits, reads) = count_exact_reads(seg, &query, k, oversample);
                assert_eq!(
                    reads,
                    expect(precision, n),
                    "{precision:?} N={n}: exact-read count"
                );
                assert_eq!(
                    hits,
                    seg.search_final(&query, k, oversample).unwrap(),
                    "{precision:?} N={n}: the counted composition is the production entry"
                );
                assert_eq!(hits.len(), k);
            }
        }
    }

    /// A non-finite distance produced LOCALLY is a broken index, not a peer
    /// fault, so it is a typed engine error naming the poisoned SEGMENT
    /// rather than a ladder failure. The producer is a non-finite COMPONENT
    /// in a stored vector: `cosine_distance` guards zero magnitude but not
    /// that, so `denom` is NaN and the distance is NaN.
    #[test]
    fn a_non_finite_local_distance_is_a_typed_segment_error() {
        let query = corpus()[0].1.clone();
        for poison in [f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut stored = vec![0.0f32; 8];
            stored[0] = poison;
            // The rescore kernel, reached with no index in hand.
            let err = rescore(
                SegmentId(7),
                vec![("x".to_string(), 0.0)],
                &|_| Ok(Some(stored.clone())),
                &query,
            )
            .unwrap_err();
            let text = err.to_string();
            assert!(
                text.contains("segment 7") && text.contains("non-finite"),
                "{poison:?}: must name the poisoned segment: {text}"
            );

            // …and the search kernel, whose `Final` phase rescores in place.
            let rows = corpus();
            let index = segment(&rows, StoragePrecision::Int8);
            let err = search_unit(
                SegmentId(3),
                &index,
                &query,
                2,
                SegmentSearchPhase::Final,
                &|_| Ok(Some(stored.clone())),
            )
            .unwrap_err();
            assert!(err.to_string().contains("segment 3"), "{poison:?}: {}", err);
        }
    }

    /// A wrong-width query is refused at the SEARCH ENTRY with a typed error
    /// naming both widths — never a panic, and never a silent answer over a
    /// prefix. `Binary` is the case that used to panic: the query packs to
    /// `ceil(len/8)` bytes, so usearch accepts an over-long query and the
    /// fault only surfaces inside `cosine_distance`.
    #[test]
    fn a_wrong_width_query_is_a_typed_refusal_on_every_precision() {
        let rows = corpus(); // 8-wide
        for precision in [
            StoragePrecision::F32,
            StoragePrecision::F16,
            StoragePrecision::Int8,
            StoragePrecision::Binary,
        ] {
            let index = segment(&rows, precision);
            for width in [7usize, 9, 0] {
                let q = vec![0.5f32; width];
                let err = search_unit(
                    SegmentId(2),
                    &index,
                    &q,
                    2,
                    SegmentSearchPhase::Final,
                    &|row_id| index.get_exact(row_id),
                )
                .unwrap_err();
                let text = err.to_string();
                assert!(
                    text.contains("segment 2") && text.contains("8 dimensions"),
                    "{precision:?} width {width}: {text}"
                );
            }
            // The whole-set entry refuses it too (this is the path a
            // coordinator re-runs in process after a local load).
            let seg = segmented(vec![(&rows, precision)]);
            assert!(
                seg.search_final(&[0.5f32; 9], 1, 1).is_err(),
                "{precision:?}: search_final must refuse a 9-wide query on an 8-wide set"
            );
            assert!(
                seg.search(&[0.5f32; 9], 1).is_err(),
                "{precision:?}: the raw candidate entry refuses it as well"
            );
        }
    }

    /// usearch 2.25.1's `metric_cos_gt` carries an explicit zero-magnitude    /// usearch 2.25.1's `metric_cos_gt` carries an explicit zero-magnitude
    /// guard (zero-vs-nonzero → 1, zero-vs-zero → 0), so a zero corpus row —
    /// the case that would otherwise divide by zero — never yields a
    /// non-finite distance on ANY precision. Measured and pinned per
    /// precision so a usearch bump that drops the guard fails here rather
    /// than silently poisoning the merge's sort key. `Binary` is Hamming, not
    /// cosine, so it is pinned as "finite" rather than to a cosine value.
    ///
    /// This is also usearch's half of the zero-vs-zero divergence
    /// jammi-numerics pins in `distance.rs`: for a ZERO query against a zero
    /// row usearch answers 0.0 where `cosine_distance` answers 1.0. The
    /// divergence is documented on both sides, never normalised away.
    #[test]
    fn a_zero_corpus_row_is_finite_on_every_precision() {
        let zero = vec![0.0f32; 8];
        let probe = {
            let mut v = vec![0.0f32; 8];
            v[0] = 1.0;
            v
        };
        for (precision, expected) in [
            (StoragePrecision::F32, Some(1.0f32)),
            (StoragePrecision::F16, Some(1.0)),
            // 0.5 was the contract's figure on ITS fixture; measured here on
            // an 8-d corpus it is 0.6464466. The load-bearing property is the
            // same on every precision: FINITE, never NaN.
            (StoragePrecision::Int8, Some(0.646_446_6)),
            (StoragePrecision::Binary, None),
        ] {
            let rows: Vec<(&str, Vec<f32>)> =
                vec![("zero", zero.clone()), ("probe", probe.clone())];
            let index = segment(&rows, precision);
            let hits = index.search(&probe, 2).unwrap();
            let zero_hit = hits
                .iter()
                .find(|(id, _)| id == "zero")
                .expect("the zero row is indexed");
            assert!(
                zero_hit.1.is_finite(),
                "{precision:?}: a zero corpus row must not yield a non-finite distance, got {:?}",
                zero_hit.1
            );
            if let Some(expected) = expected {
                assert!(
                    (zero_hit.1 - expected).abs() < 1e-6,
                    "{precision:?}: usearch's measured zero-row distance moved: {:?} vs {expected}",
                    zero_hit.1
                );
            }
            // Zero query against the zero row — usearch's zero-vs-zero arm.
            let zq = index.search(&zero, 2).unwrap();
            for (id, d) in &zq {
                assert!(
                    d.is_finite(),
                    "{precision:?}: zero query on '{id}' gave {d:?}"
                );
            }
        }
    }

    // Test 5 — uniform total order including ties: two rows with an identical
    // vector tie on distance and are ordered by row_id, at N=1 and N=2, matching
    // the brute-force tiebreak.
    #[test]
    fn ties_break_on_row_id_at_every_n() {
        let dup = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rows = corpus();
        // Two deliberately-identical vectors (equal distance to any query).
        rows.push(("y", dup.clone()));
        rows.push(("z", dup.clone()));
        let query = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let n1 = segmented(vec![(&rows[..], StoragePrecision::F32)]);
        let (left, right) = rows.split_at(7);
        let n2 = segmented(vec![
            (left, StoragePrecision::F32),
            (right, StoragePrecision::F32),
        ]);
        let expected = brute_force(&rows, &query, 4);
        // The tie between y and z must resolve y-before-z (row_id order) in both.
        assert_eq!(ids(&n1.search(&query, 4).unwrap()), expected);
        assert_eq!(ids(&n2.search(&query, 4).unwrap()), expected);
        let y = expected.iter().position(|id| id == "y").unwrap();
        let z = expected.iter().position(|id| id == "z").unwrap();
        assert!(y < z, "row_id tiebreak orders y before z");
    }

    // Test 4 — determinism: two independently-assembled segmented indexes over
    // the same rows produce byte-identical ordering.
    #[test]
    fn merge_is_deterministic_across_independent_builds() {
        let rows = corpus();
        let (left, right) = rows.split_at(6);
        let query = &rows[4].1;
        let a = segmented(vec![
            (left, StoragePrecision::F32),
            (right, StoragePrecision::F32),
        ]);
        let b = segmented(vec![
            (left, StoragePrecision::F32),
            (right, StoragePrecision::F32),
        ]);
        assert_eq!(
            ids(&a.search(query, 6).unwrap()),
            ids(&b.search(query, 6).unwrap())
        );
    }

    // Test 10 — dedup by row id: a row id present in two segments (a constructed
    // violation of the disjointness invariant) surfaces ONCE (the nearest), not
    // twice.
    #[test]
    fn duplicate_row_id_across_segments_surfaces_once() {
        let left: Vec<(&str, Vec<f32>)> = vec![
            ("shared", vec![1.0, 0.0, 0.0, 0.0]),
            ("l1", vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let right: Vec<(&str, Vec<f32>)> = vec![
            ("shared", vec![0.9, 0.1, 0.0, 0.0]),
            ("r1", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let seg = segmented(vec![
            (&left, StoragePrecision::F32),
            (&right, StoragePrecision::F32),
        ]);
        let hits = seg.search(&[1.0, 0.0, 0.0, 0.0], 4).unwrap();
        let shared_count = hits.iter().filter(|(id, _)| id == "shared").count();
        assert_eq!(
            shared_count, 1,
            "a row id spanning two segments appears once"
        );
    }

    // Uniformity is enforced: a mixed-precision segment set is refused at
    // construction rather than silently merging incomparable distances.
    #[test]
    fn mixed_precision_segment_set_is_rejected() {
        let rows = corpus();
        let (left, right) = rows.split_at(6);
        let segs = vec![
            (SegmentId(0), segment(left, StoragePrecision::F32)),
            (SegmentId(1), segment(right, StoragePrecision::Int8)),
        ];
        assert!(SegmentedIndex::new(segs).is_err());
    }
}
