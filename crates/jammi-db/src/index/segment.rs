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
//! `f32` vector via [`SegmentedIndex::get_exact`], recomputes cosine against the
//! query, and re-ranks by that exact distance — a corpus-independent order that
//! *is* comparable across segments. On an `F32` table it is `search` directly.
//! Routing every final read through `search_final` is what keeps a
//! multi-segment quantized table's answer exact-comparable: a consumer that
//! returned raw `search` output as final would surface per-segment
//! non-comparable distances as if they were a global ranking.
//!
//! ## Deferred seams
//!
//! Several extensions fit this shape but are not built here: scatter-gather /
//! cross-node segment placement (each segment is independently loadable, so a
//! future distributed reader fans out the same merge); an mmap `view()` of a
//! segment's vectors; re-quantization or compaction of the segment set (merging
//! many small segments into one, at which point the
//! authoritative-vector-on-re-embed question below becomes live); and a
//! compaction *policy*. A small appended segment — especially a `Binary` one,
//! whose τ is fit from few rows and is therefore noisy — has weaker per-segment
//! recall; the merge over-fetch feeding `search_final`'s exact rescore is what
//! keeps that from degrading the final answer, but compaction is the eventual
//! structural fix.

use std::collections::HashSet;

use jammi_numerics::distance::cosine_distance;

use crate::config::StoragePrecision;
use crate::error::{JammiError, Result};
use crate::index::sidecar::SidecarIndex;
use crate::index::VectorIndex;

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
pub const DEFAULT_SEGMENT_OVERFETCH_FACTOR: f32 = 2.0;

/// Per-segment fetch width for a merge that wants a global top-`m` over
/// `n_segments` segments. A single segment is asked for exactly `m` (its own
/// top-`m` *is* the global top-`m`, so no over-fetch — this is what makes `N=1`
/// byte-identical to a lone [`SidecarIndex::search`]); more than one is asked
/// for `ceil(m * DEFAULT_SEGMENT_OVERFETCH_FACTOR)`.
fn over_fetch(m: usize, n_segments: usize) -> usize {
    if n_segments <= 1 {
        m
    } else {
        (m as f32 * DEFAULT_SEGMENT_OVERFETCH_FACTOR).ceil() as usize
    }
}

/// A loaded segment paired with its catalog id (the id is the merge's final
/// tie-break and identifies which bundle owns a given row).
struct Segment {
    id: SegmentId,
    index: SidecarIndex,
}

/// The read-side handle over a table's whole segment set. Built by
/// [`crate::store::ResultStore::resolve_search_mode`] from the segments loaded
/// through the content-addressed segment cache; searched through
/// [`Self::search_final`] by every final-results consumer.
pub struct SegmentedIndex {
    segments: Vec<Segment>,
    /// The table-level storage precision. Uniform across segments *by
    /// enforcement*, not assumption: a segment built at a drifted precision
    /// fails [`SidecarIndex::load`]'s strict `scalar_kind` check and never
    /// reaches this constructor, so every segment here loaded at the same
    /// precision — the one this field reports.
    storage_precision: StoragePrecision,
}

impl SegmentedIndex {
    /// Assemble a segmented index from the segments loaded for one table, in
    /// `segment_id` order.
    ///
    /// The set must be non-empty (a table with no segments resolves to the
    /// exact-search fallback upstream, never to an empty `SegmentedIndex`) and
    /// every segment must share one storage precision — guaranteed by the
    /// per-segment strict load check, and re-asserted here so a future loader
    /// that bypasses it cannot silently assemble a mixed-precision set whose
    /// merge would compare distances on different scales.
    pub fn new(segments: Vec<(SegmentId, SidecarIndex)>) -> Result<Self> {
        let mut iter = segments.into_iter();
        let (first_id, first) = iter.next().ok_or_else(|| {
            JammiError::Other("SegmentedIndex requires at least one segment".into())
        })?;
        let storage_precision = first.storage_precision();
        let mut out = vec![Segment {
            id: first_id,
            index: first,
        }];
        for (id, index) in iter {
            if index.storage_precision() != storage_precision {
                return Err(JammiError::Other(format!(
                    "SegmentedIndex: segment {} loaded at {:?} but the set is {:?} — \
                     mixed-precision segment sets are unsearchable (distances are not comparable)",
                    id.0,
                    index.storage_precision(),
                    storage_precision
                )));
            }
            out.push(Segment { id, index });
        }
        Ok(Self {
            segments: out,
            storage_precision,
        })
    }

    /// The precision every segment in this set was built and loaded at.
    pub fn storage_precision(&self) -> StoragePrecision {
        self.storage_precision
    }

    /// Total number of rows across every segment.
    pub fn len(&self) -> usize {
        self.segments.iter().map(|s| s.index.len()).sum()
    }

    /// Whether the whole set indexes zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The RAW candidate merge primitive: search each segment at the over-fetch
    /// width, concatenate, order by `(distance ASC, row_id ASC, segment_id
    /// ASC)`, dedup by row id keeping the nearest occurrence, and truncate to
    /// `m`.
    ///
    /// For an `F32` set the distances are exact cosine and comparable across
    /// segments, so this order is final. For a quantized / `Binary` set they
    /// are per-segment approximate candidates — deterministically ordered here
    /// for a stable candidate set, but **not** a final answer; only
    /// [`Self::search_final`] consumes them, feeding the exact rescore that
    /// makes the final order comparable. The dedup keeps the merge correct over
    /// the row-id space even though the S10 append-disjoint-rows invariant means
    /// no row id spans two segments today; which segment's vector wins for a
    /// re-embedded row is a compaction/retire concern, seamed, not built.
    pub fn search(&self, query: &[f32], m: usize) -> Result<Vec<(String, f32)>> {
        if m == 0 {
            return Ok(Vec::new());
        }
        let per_segment = over_fetch(m, self.segments.len());
        let mut merged: Vec<(String, f32, SegmentId)> = Vec::new();
        for seg in &self.segments {
            for (row_id, distance) in seg.index.search(query, per_segment)? {
                merged.push((row_id, distance, seg.id));
            }
        }
        merged.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.2.cmp(&b.2))
        });
        // Dedup by row id keeping the nearest: after the sort the first
        // occurrence of each id is its nearest, so a set-membership pass keeps
        // that one and drops the rest, stopping once `m` survivors are collected.
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<(String, f32)> = Vec::with_capacity(m.min(merged.len()));
        for (row_id, distance, _segment) in merged {
            if out.len() == m {
                break;
            }
            if seen.insert(row_id.clone()) {
                out.push((row_id, distance));
            }
        }
        Ok(out)
    }

    /// THE single final-results entry. Returns the exact top-`k` in a total
    /// order comparable across every segment.
    ///
    /// On a set that [`needs_rescore`](StoragePrecision::needs_rescore)
    /// (quantized or `Binary`) this is a two-stage retrieve→rescore: retrieve
    /// `k * oversample` candidates through [`Self::search`] (the merge
    /// over-fetch nests *under* this candidate width by construction), read each
    /// candidate's exact `f32` vector via [`Self::get_exact`], recompute cosine
    /// distance against the query, then re-rank by `(distance, row_id)` and
    /// truncate to `k`. The exact re-rank is corpus-independent, so a candidate
    /// drawn from any segment lands in one global order — the property a raw
    /// `search` cannot give a quantized multi-segment table. On an `F32` set the
    /// merge is already exact and final, so this is `search(query, k)` and
    /// `oversample` is unused.
    ///
    /// A candidate whose exact vector is missing (present in a graph but absent
    /// from its rescore companion — a torn bundle) is a hard error, never a
    /// silent drop: a result set that quietly shrank below `k` would read as
    /// "fewer matches exist" rather than "the index is broken".
    pub fn search_final(
        &self,
        query: &[f32],
        k: usize,
        oversample: usize,
    ) -> Result<Vec<(String, f32)>> {
        if !self.storage_precision.needs_rescore() {
            return self.search(query, k);
        }
        let candidate_k = k.saturating_mul(oversample).max(k);
        let candidates = self.search(query, candidate_k)?;
        let mut rescored: Vec<(String, f32)> = Vec::with_capacity(candidates.len());
        for (row_id, _approx) in candidates {
            let exact = self.get_exact(&row_id)?.ok_or_else(|| {
                JammiError::Other(format!(
                    "rescore: candidate '{row_id}' has no exact vector in its segment's rescore \
                     companion (corrupted or torn sidecar bundle)"
                ))
            })?;
            let distance = cosine_distance(query, &exact);
            rescored.push((row_id, distance));
        }
        rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        rescored.truncate(k);
        Ok(rescored)
    }

    /// The exact `f32` vector for `row_id`, dispatched to the segment that owns
    /// it. Segments are row-disjoint, so at most one returns `Some`; the first
    /// hit wins. `None` when no segment indexes the id.
    pub fn get_exact(&self, row_id: &str) -> Result<Option<Vec<f32>>> {
        for seg in &self.segments {
            if let Some(vector) = seg.index.get_exact(row_id)? {
                return Ok(Some(vector));
            }
        }
        Ok(None)
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
    // equals a manual retrieve-then-rescore over that same sidecar (the pre-S10
    // path search_final folds in).
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

    // Test 9 (recall floor) — a merged multi-segment quantized index's recall@k
    // is no worse than the same rows in a single segment, over a fixed query
    // set. The relative floor that forbids a factor that silently degrades
    // recall.
    #[test]
    fn merged_quantized_recall_is_not_below_single_segment() {
        let rows = corpus();
        let (left, right) = rows.split_at(6);
        let single = segmented(vec![(&rows[..], StoragePrecision::Int8)]);
        let merged = segmented(vec![
            (left, StoragePrecision::Int8),
            (right, StoragePrecision::Int8),
        ]);
        let k = 3;
        let (mut single_hits, mut merged_hits) = (0usize, 0usize);
        for (_, q) in &rows {
            let truth: std::collections::HashSet<String> =
                brute_force(&rows, q, k).into_iter().collect();
            let s = single.search_final(q, k, 8).unwrap();
            let m = merged.search_final(q, k, 8).unwrap();
            single_hits += ids(&s).iter().filter(|id| truth.contains(*id)).count();
            merged_hits += ids(&m).iter().filter(|id| truth.contains(*id)).count();
        }
        assert!(
            merged_hits >= single_hits,
            "merged recall ({merged_hits}) must be >= single-segment recall ({single_hits})"
        );
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
