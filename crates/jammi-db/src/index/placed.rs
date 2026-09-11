//! A table's ANN index as a **placed** set of segments: some resident in this
//! process, some owned by a peer — searched through one async entry.
//!
//! [`PlacedIndex`] is the read-side handle
//! [`crate::store::ResultStore::resolve_search_mode`] returns for the ONLINE
//! retrieval leaf. Its only search entry is [`PlacedIndex::search_final_placed`]
//! (async). When every source is local the handle wraps an all-local
//! [`SegmentedIndex`] and the async entry literally calls the sync
//! [`SegmentedIndex::search_final`] — same kernels, same bytes, today's
//! exact-read count at every `N`. When at least one source is remote the entry
//! runs the per-precision protocol over the peer transport and the bounded
//! failure ladder.
//!
//! The split is structural, not a flag: [`SegmentedIndex`] stays the sync,
//! all-local type (the batch consumers hold one across a whole build), and a
//! remote source cannot reach a sync path by construction — there is no
//! `SegmentedIndex` that contains a `Remote` source.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::peer::{
    ExactRescoreRequest, PeerAddr, PeerError, PeerFailureCounters, PeerFailureReason,
    PeerTransport, SegmentSearchPhase, SegmentSearchRequest, PEER_RPC_DEADLINE,
};
use crate::index::segment::{merge, over_fetch, rescore, search_unit};
use crate::index::sidecar::SidecarIndex;
use crate::index::{SegmentId, SegmentedIndex, VectorIndex};
use crate::storage::index_cache::SegmentIndexCache;
use crate::storage::StorageUrl;

/// One segment of a placed table: resident here, or owned elsewhere.
pub enum SegmentSource {
    /// A segment this process loaded through the segment cache.
    Local(SegmentId, SidecarIndex),
    /// A segment a peer owns. Nothing is loaded here unless the failure
    /// ladder's local-load rung admits it — hence the bundle URL and the
    /// catalog row count ride along.
    Remote {
        /// The segment's catalog id.
        segment_id: SegmentId,
        /// The rendezvous-ordered owners: first is the owner, second the one
        /// retry.
        owners: Vec<PeerAddr>,
        /// The catalog's row count for the segment (its contribution to
        /// [`PlacedIndex::len`] and the local-load estimate).
        row_count: usize,
        /// The segment bundle's base URL, for the local-load rung.
        index_url: StorageUrl,
    },
}

/// A segment a peer owns, as [`Placed::Mixed`] holds it.
pub(crate) struct RemoteSegment {
    pub(crate) segment_id: SegmentId,
    pub(crate) owners: Vec<PeerAddr>,
    pub(crate) row_count: usize,
    pub(crate) index_url: StorageUrl,
}

/// The two shapes a placed table takes.
pub(crate) enum Placed {
    /// Every segment is resident: the sync all-local index, searched as-is.
    /// An `Arc` so the force-local entry's ([`crate::store::ResultStore::
    /// resolve_search_mode_local`]) cached, version-aware, masked
    /// [`SegmentedIndex`] can be wrapped without a copy — the ONLINE entry
    /// never rebuilds an unmasked set out from under it.
    AllLocal(Arc<SegmentedIndex>),
    /// At least one segment is owned by a peer.
    Mixed {
        local: Vec<(SegmentId, SidecarIndex)>,
        remote: Vec<RemoteSegment>,
    },
}

/// The read-side handle over a table's whole placed segment set. Opaque: the
/// only search entry is [`Self::search_final_placed`].
pub struct PlacedIndex {
    pub(crate) inner: Placed,
    pub(crate) storage_precision: StoragePrecision,
    pub(crate) table_name: String,
    pub(crate) transport: Arc<dyn PeerTransport>,
    pub(crate) loader: Arc<SegmentIndexCache>,
    pub(crate) ann: AnnIndexConfig,
    /// `[server] peer_local_load_bytes`: the marginal-load admission budget
    /// one query may spend loading segments it does not own. `None` =
    /// unbounded.
    pub(crate) budget: Option<u64>,
    /// The table's embedding width, for the local-load estimate. `None` skips
    /// the local-load rung.
    pub(crate) dimensions: Option<i32>,
    pub(crate) counters: Arc<PeerFailureCounters>,
}

impl PlacedIndex {
    /// Assemble a placed index from `sources`, in `segment_id` order.
    ///
    /// With no `Remote` source this is `Placed::AllLocal(SegmentedIndex::new(...))`
    /// — the same constructor, the same uniformity check, the same type the
    /// batch consumers hold. With at least one `Remote` source it is
    /// `Placed::Mixed`; the local sources' precision uniformity is re-asserted
    /// here (a remote source's precision is asserted by the owner's strict
    /// load, which refuses a mismatching bundle). The set must be non-empty (a
    /// table with no segments resolves to the exact fallback upstream).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_sources(
        sources: Vec<SegmentSource>,
        table_name: &str,
        precision: StoragePrecision,
        transport: Arc<dyn PeerTransport>,
        loader: Arc<SegmentIndexCache>,
        ann: AnnIndexConfig,
        budget: Option<u64>,
        dimensions: Option<i32>,
        counters: Arc<PeerFailureCounters>,
    ) -> Result<Self> {
        if sources.is_empty() {
            return Err(JammiError::Other(
                "PlacedIndex requires at least one segment".into(),
            ));
        }
        let any_remote = sources
            .iter()
            .any(|s| matches!(s, SegmentSource::Remote { .. }));
        let inner = if any_remote {
            let mut local = Vec::new();
            let mut remote = Vec::new();
            for source in sources {
                match source {
                    SegmentSource::Local(id, index) => {
                        if index.storage_precision() != precision {
                            return Err(JammiError::Other(format!(
                                "PlacedIndex: segment {} loaded at {:?} but table '{table_name}' \
                                 is {precision:?} — mixed-precision segment sets are unsearchable \
                                 (distances are not comparable)",
                                id.0,
                                index.storage_precision(),
                            )));
                        }
                        local.push((id, index));
                    }
                    SegmentSource::Remote {
                        segment_id,
                        owners,
                        row_count,
                        index_url,
                    } => remote.push(RemoteSegment {
                        segment_id,
                        owners,
                        row_count,
                        index_url,
                    }),
                }
            }
            Placed::Mixed { local, remote }
        } else {
            let segments = sources
                .into_iter()
                .map(|s| match s {
                    SegmentSource::Local(id, index) => (id, index),
                    SegmentSource::Remote { .. } => unreachable!("no remote source"),
                })
                .collect();
            let index = SegmentedIndex::new(segments)?;
            if index.storage_precision() != precision {
                return Err(JammiError::Other(format!(
                    "PlacedIndex: segment set loaded at {:?} but table '{table_name}' is \
                     {precision:?}",
                    index.storage_precision(),
                )));
            }
            Placed::AllLocal(Arc::new(index))
        };
        Ok(Self {
            inner,
            storage_precision: precision,
            table_name: table_name.to_string(),
            transport,
            loader,
            ann,
            budget,
            dimensions,
            counters,
        })
    }

    /// Wrap an already-resolved, version-aware all-local [`SegmentedIndex`]
    /// (from [`crate::store::ResultStore::resolve_search_mode_local`]) as a
    /// [`PlacedIndex`], for [`crate::store::ResultStore::resolve_search_mode`]'s
    /// every-segment-local arm: the ONLINE entry must load exactly what the
    /// force-local entry loads — mask, `current_version` and all — never a
    /// second, unmasked read through [`Self::with_sources`]'s flat segment
    /// list. The precision is read off `index` rather than re-asserted: the
    /// force-local resolver already enforces the set's own uniformity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_local(
        index: Arc<SegmentedIndex>,
        table_name: &str,
        transport: Arc<dyn PeerTransport>,
        loader: Arc<SegmentIndexCache>,
        ann: AnnIndexConfig,
        budget: Option<u64>,
        dimensions: Option<i32>,
        counters: Arc<PeerFailureCounters>,
    ) -> Self {
        Self {
            storage_precision: index.storage_precision(),
            inner: Placed::AllLocal(index),
            table_name: table_name.to_string(),
            transport,
            loader,
            ann,
            budget,
            dimensions,
            counters,
        }
    }

    /// The precision every segment in this set is stored at.
    pub fn storage_precision(&self) -> StoragePrecision {
        self.storage_precision
    }

    /// Total number of rows across every segment: the loaded count for local
    /// segments plus the catalog row count for remote ones.
    pub fn len(&self) -> usize {
        match &self.inner {
            Placed::AllLocal(index) => index.len(),
            Placed::Mixed { local, remote } => {
                local.iter().map(|(_, index)| index.len()).sum::<usize>()
                    + remote.iter().map(|r| r.row_count).sum::<usize>()
            }
        }
    }

    /// Whether the whole set indexes zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether at least one segment is owned by a peer.
    pub fn has_remote(&self) -> bool {
        matches!(&self.inner, Placed::Mixed { .. })
    }

    /// THE single search entry over a placed set: the exact top-`k` in one
    /// total order comparable across every segment, wherever it lives.
    ///
    /// All-local: literally [`SegmentedIndex::search_final`] — the same
    /// kernels, the same bytes, today's exact-read count. Mixed: the
    /// per-precision protocol over the transport plus the failure ladder.
    pub async fn search_final_placed(
        &self,
        query: &[f32],
        k: usize,
        oversample: usize,
    ) -> Result<Vec<(String, f32)>> {
        match &self.inner {
            Placed::AllLocal(index) => index.search_final(query, k, oversample),
            Placed::Mixed { local, remote } => {
                self.search_mixed(local, remote, query, k, oversample).await
            }
        }
    }

    /// The `Mixed` arm: the per-precision protocol over the transport plus
    /// the bounded failure ladder, per remote segment, per phase.
    ///
    /// `N` = local + remote segment count; `candidate_k = max(k, k·oversample)`
    /// for a rescoring precision and `k` for `F32`; `width = over_fetch(candidate_k, N)`.
    /// Remote segments sharing one owner list go in ONE request per phase, in
    /// parallel across owners; local sources run [`search_unit`] in-process.
    ///
    /// - `F32` — one `Final` phase: every source returns its top-`width` exact
    ///   hits; `merge(units, k)`. 1 RTT.
    /// - `F16` / `Int8` — `Approximate` then `ExactRescore`: every source
    ///   returns approximate candidates (comparable across segments);
    ///   `merge(units, candidate_k)` truncates on approximate distance;
    ///   survivors are grouped by winning segment — local ones rescore through
    ///   [`rescore`] with the local exact lookup, remote ones in ONE
    ///   `ExactRescore` per owner; the coordinator sorts by `(distance,
    ///   row_id)` and truncates to `k`. 2 RTT; exactly `candidate_k` exact reads.
    /// - `Binary` — one `Final` phase: every source rescores all its `width`
    ///   hits locally and returns exact distances; `merge(units, k)` on final
    ///   distance. 1 RTT; `N·width` exact reads, paid only here.
    ///
    /// The ladder per remote segment (per phase): the owner under
    /// [`PEER_RPC_DEADLINE`] → one retry at the next rendezvous candidate →
    /// a local load through the segment cache under `2 × PEER_RPC_DEADLINE`,
    /// admitted only by the marginal-load budget with the table's dimensions
    /// known → [`JammiError::Unavailable`] naming the segment. Every rung emits
    /// a `warn!` and increments its counter.
    async fn search_mixed(
        &self,
        local: &[(SegmentId, SidecarIndex)],
        remote: &[RemoteSegment],
        query: &[f32],
        k: usize,
        oversample: usize,
    ) -> Result<Vec<(String, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let precision = self.storage_precision;
        let n = local.len() + remote.len();
        let candidate_k = if precision.needs_rescore() {
            k.saturating_mul(oversample).max(k)
        } else {
            k
        };
        let width = over_fetch(candidate_k, n);
        let phase = SegmentSearchPhase::for_precision(precision);
        // Bytes this query has loaded at the local-load rung so far — the
        // marginal-load admission runs per query, sequentially.
        let mut loaded_this_query: u64 = 0;
        // Remote segments this query loaded locally (rung 3): searched and
        // rescored here from now on.
        let mut locally_loaded: Vec<(SegmentId, SidecarIndex)> = Vec::new();

        // ---- Phase 1: per-segment units at `width` ----
        let mut units: Vec<(SegmentId, Vec<(String, f32)>)> = Vec::with_capacity(n);
        for (id, index) in local {
            units.push((
                *id,
                search_unit(index, query, width, phase, &|row_id| {
                    index.get_exact(row_id)
                })?,
            ));
        }
        let groups = owner_groups(remote);
        let searches = groups.iter().map(|group| async move {
            let req = SegmentSearchRequest {
                table_name: self.table_name.clone(),
                segment_ids: group.segments.iter().map(|s| s.segment_id).collect(),
                storage_precision: precision,
                query: query.to_vec(),
                width,
                phase,
            };
            let req = &req;
            self.call_with_retry(group, |owner| async move {
                self.transport
                    .segment_search(&owner, req, PEER_RPC_DEADLINE)
                    .await
            })
            .await
        });
        let results = futures::future::join_all(searches).await;
        for (group, result) in groups.iter().zip(results) {
            match result {
                Ok(remote_units) => {
                    for unit in remote_units {
                        units.push((unit.segment_id, unit.hits));
                    }
                }
                Err(last) => {
                    // Rung 3, per segment of the failed group.
                    for seg in &group.segments {
                        let index = self.load_locally(seg, &mut loaded_this_query, last).await?;
                        units.push((
                            seg.segment_id,
                            search_unit(&index, query, width, phase, &|row_id| {
                                index.get_exact(row_id)
                            })?,
                        ));
                        locally_loaded.push((seg.segment_id, index));
                    }
                }
            }
        }

        // ---- Final phases merge on final distance and are done ----
        if phase == SegmentSearchPhase::Final {
            return Ok(merge(units, k)
                .into_iter()
                .map(|(row_id, distance, _segment)| (row_id, distance))
                .collect());
        }

        // ---- Phase 2 (Approximate): merge, truncate, rescore the survivors ----
        let survivors = merge(units, candidate_k);
        let mut by_segment: Vec<(SegmentId, Vec<(String, f32)>)> = Vec::new();
        for (row_id, approx, segment) in survivors {
            match by_segment.iter_mut().find(|(s, _)| *s == segment) {
                Some((_, group)) => group.push((row_id, approx)),
                None => by_segment.push((segment, vec![(row_id, approx)])),
            }
        }
        let resident = |segment: SegmentId| -> Option<&SidecarIndex> {
            local
                .iter()
                .chain(locally_loaded.iter())
                .find(|(id, _)| *id == segment)
                .map(|(_, index)| index)
        };
        let mut rescored: Vec<(String, f32)> = Vec::with_capacity(candidate_k);
        // Remote survivors, grouped by the owner list of their segment.
        let mut remote_groups: Vec<(&OwnerGroup<'_>, RowIdsBySegment)> = Vec::new();
        for (segment, candidates) in by_segment {
            if let Some(index) = resident(segment) {
                rescored.extend(rescore(
                    candidates,
                    &|row_id| index.get_exact(row_id),
                    query,
                )?);
                continue;
            }
            let group = groups
                .iter()
                .find(|g| g.segments.iter().any(|s| s.segment_id == segment))
                .expect("every non-resident survivor came from a remote group");
            let row_ids: Vec<String> = candidates.into_iter().map(|(id, _)| id).collect();
            match remote_groups
                .iter_mut()
                .find(|(g, _)| std::ptr::eq(*g, group))
            {
                Some((_, groups_rows)) => groups_rows.push((segment, row_ids)),
                None => remote_groups.push((group, vec![(segment, row_ids)])),
            }
        }
        let rescores = remote_groups.iter().map(|(group, rows)| async move {
            let req = ExactRescoreRequest {
                table_name: self.table_name.clone(),
                storage_precision: precision,
                query: query.to_vec(),
                row_ids_by_segment: rows.clone(),
            };
            let req = &req;
            self.call_with_retry(group, |owner| async move {
                self.transport
                    .exact_rescore(&owner, req, PEER_RPC_DEADLINE)
                    .await
            })
            .await
        });
        let results = futures::future::join_all(rescores).await;
        for ((group, rows), result) in remote_groups.iter().zip(results) {
            match result {
                Ok(hits) => rescored.extend(hits),
                Err(last) => {
                    for (segment, row_ids) in rows {
                        let seg = group
                            .segments
                            .iter()
                            .find(|s| s.segment_id == *segment)
                            .expect("the survivor's segment is in its group");
                        let index = self.load_locally(seg, &mut loaded_this_query, last).await?;
                        rescored.extend(rescore(
                            row_ids.iter().map(|id| (id.clone(), 0.0)).collect(),
                            &|row_id| index.get_exact(row_id),
                            query,
                        )?);
                    }
                }
            }
        }
        rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        rescored.truncate(k);
        Ok(rescored)
    }

    /// Rungs 1–2 of the ladder for one owner group: the owner, then one retry
    /// at the next rendezvous candidate. `Err` carries the last failure's
    /// reason for rung 3's message.
    async fn call_with_retry<T, F, Fut>(
        &self,
        group: &OwnerGroup<'_>,
        call: F,
    ) -> std::result::Result<T, PeerFailureReason>
    where
        F: Fn(PeerAddr) -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, PeerError>>,
    {
        let mut last = PeerFailureReason::Unreachable;
        for (rung, owner) in group.owners.iter().take(2).enumerate() {
            match call(owner.clone()).await {
                Ok(value) => {
                    if rung == 1 {
                        self.counters.retry_ok.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(value);
                }
                Err(e) => {
                    self.counters.record(e.reason);
                    tracing::warn!(
                        table = self.table_name,
                        segment = e.segment.0,
                        owner = %e.owner,
                        reason = %e.reason,
                        rung = rung + 1,
                        "peer segment search failed"
                    );
                    last = e.reason;
                }
            }
        }
        Err(last)
    }

    /// Rung 3: load `seg` locally through the segment cache under
    /// `2 × PEER_RPC_DEADLINE`, admitted iff the table records its dimensions
    /// and `loaded_this_query + estimate(seg) ≤ budget`. Every refusal or
    /// failure is [`JammiError::Unavailable`] naming the segment.
    async fn load_locally(
        &self,
        seg: &RemoteSegment,
        loaded_this_query: &mut u64,
        last: PeerFailureReason,
    ) -> Result<SidecarIndex> {
        let resource = format!("segment {}/{}", self.table_name, seg.segment_id.0);
        let unavailable = |reason: String| {
            self.counters.unavailable.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                table = self.table_name,
                segment = seg.segment_id.0,
                owner = ?seg.owners.first(),
                reason = %reason,
                "placed segment unavailable"
            );
            JammiError::Unavailable {
                resource: resource.clone(),
                reason,
            }
        };
        let Some(dimensions) = self.dimensions else {
            return Err(unavailable(format!(
                "{last}; local load skipped: the table records no dimensions"
            )));
        };
        let estimate = local_load_estimate(seg.row_count, dimensions, self.storage_precision);
        if let Some(budget) = self.budget {
            if loaded_this_query.saturating_add(estimate) > budget {
                return Err(unavailable(format!(
                    "{last}; local load of ~{estimate} bytes refused by peer_local_load_bytes = \
                     {budget} ({loaded_this_query} already loaded by this query)"
                )));
            }
        }
        let load = self
            .loader
            .load_segment(&seg.index_url, &self.ann, self.storage_precision);
        match tokio::time::timeout(2 * PEER_RPC_DEADLINE, load).await {
            Ok(Ok(index)) => {
                *loaded_this_query += estimate;
                self.counters.local_load.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    table = self.table_name,
                    segment = seg.segment_id.0,
                    owner = ?seg.owners.first(),
                    reason = %last,
                    estimate_bytes = estimate,
                    "placed segment loaded locally after its owners failed"
                );
                Ok(index)
            }
            Ok(Err(e)) => Err(unavailable(format!("{last}; local load failed: {e}"))),
            Err(_elapsed) => Err(unavailable(format!(
                "{last}; local load exceeded {:?}",
                2 * PEER_RPC_DEADLINE
            ))),
        }
    }
}

/// The survivors an `ExactRescore` names, grouped by the segment that owns
/// each.
type RowIdsBySegment = Vec<(SegmentId, Vec<String>)>;

/// Remote segments sharing one owner list, so they ride ONE request per
/// phase and retry together.
struct OwnerGroup<'a> {
    owners: &'a [PeerAddr],
    segments: Vec<&'a RemoteSegment>,
}

fn owner_groups(remote: &[RemoteSegment]) -> Vec<OwnerGroup<'_>> {
    let mut groups: Vec<OwnerGroup<'_>> = Vec::new();
    for seg in remote {
        match groups
            .iter_mut()
            .find(|g| g.owners == seg.owners.as_slice())
        {
            Some(group) => group.segments.push(seg),
            None => groups.push(OwnerGroup {
                owners: &seg.owners,
                segments: vec![seg],
            }),
        }
    }
    groups
}

/// The local-load estimate for a segment: `row_count × (d × bytes(precision)
/// + 32 + 64)` — 4 (F32) / 2 (F16) / 1 (Int8) / `ceil(d/8)` total (Binary)
/// bytes of stored vector, 32 bytes of row-id strings (`row_map` +
/// `row_index`) and 64 bytes of per-node graph link overhead. A LOWER bound
/// for the quantized precisions: the rawf32 companion is excluded (an fd read
/// by `pread`, never resident); usearch's level-0 links and the row-id
/// `HashMap` are unmodelled.
pub(crate) fn local_load_estimate(
    row_count: usize,
    dimensions: i32,
    precision: StoragePrecision,
) -> u64 {
    let d = u64::try_from(dimensions).unwrap_or(0);
    let vector_bytes = match precision {
        StoragePrecision::F32 => d * 4,
        StoragePrecision::F16 => d * 2,
        StoragePrecision::Int8 => d,
        StoragePrecision::Binary => d.div_ceil(8),
    };
    (row_count as u64).saturating_mul(vector_bytes + 32 + 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::peer::NoPeers;
    use crate::storage::StorageRegistry;

    fn segment(rows: &[(&str, Vec<f32>)], precision: StoragePrecision) -> SidecarIndex {
        let dim = rows[0].1.len();
        let mut idx = SidecarIndex::new(dim, &AnnIndexConfig::default(), precision).unwrap();
        for (id, v) in rows {
            idx.add(id, v).unwrap();
        }
        idx.build().unwrap();
        idx
    }

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

    fn placed(sources: Vec<SegmentSource>, precision: StoragePrecision) -> PlacedIndex {
        let dir = tempfile::tempdir().unwrap();
        let loader = Arc::new(
            SegmentIndexCache::new(StorageRegistry::new(), dir.path().join("index")).unwrap(),
        );
        PlacedIndex::with_sources(
            sources,
            "t",
            precision,
            Arc::new(NoPeers),
            loader,
            AnnIndexConfig::default(),
            None,
            Some(8),
            Arc::new(PeerFailureCounters::default()),
        )
        .unwrap()
    }

    // A3 (placed entry) — with every segment local, `search_final_placed`
    // returns the identical `(row_id, distance)` bytes `SegmentedIndex::
    // search_final` returns for the same corpus / query / k / oversample, at
    // every precision, at N = 1 and N = 2.
    #[tokio::test]
    async fn all_local_placed_search_is_byte_identical_to_segmented_search_final() {
        for precision in [
            StoragePrecision::F32,
            StoragePrecision::F16,
            StoragePrecision::Int8,
            StoragePrecision::Binary,
        ] {
            let rows = corpus();
            let (left, right) = rows.split_at(6);
            for parts in [vec![&rows[..]], vec![left, right]] {
                let sync = SegmentedIndex::new(
                    parts
                        .iter()
                        .enumerate()
                        .map(|(i, r)| (SegmentId(i as i64), segment(r, precision)))
                        .collect(),
                )
                .unwrap();
                let placed = placed(
                    parts
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            SegmentSource::Local(SegmentId(i as i64), segment(r, precision))
                        })
                        .collect(),
                    precision,
                );
                assert!(!placed.has_remote());
                assert_eq!(placed.len(), sync.len());
                assert_eq!(placed.storage_precision(), precision);
                for q in [&rows[0].1, &rows[6].1, &rows[11].1] {
                    for (k, oversample) in [(1usize, 1usize), (3, 4), (5, 32)] {
                        let want = sync.search_final(q, k, oversample).unwrap();
                        let got = placed.search_final_placed(q, k, oversample).await.unwrap();
                        assert_eq!(
                            got,
                            want,
                            "{precision:?} N={} k={k} oversample={oversample}: the placed entry \
                             must return the sync entry's bytes",
                            parts.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn empty_source_set_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let loader = Arc::new(
            SegmentIndexCache::new(StorageRegistry::new(), dir.path().join("index")).unwrap(),
        );
        assert!(PlacedIndex::with_sources(
            Vec::new(),
            "t",
            StoragePrecision::F32,
            Arc::new(NoPeers),
            loader,
            AnnIndexConfig::default(),
            None,
            None,
            Arc::new(PeerFailureCounters::default()),
        )
        .is_err());
    }

    #[test]
    fn mixed_precision_local_sources_are_refused_in_every_shape() {
        let rows = corpus();
        let (left, right) = rows.split_at(6);
        let dir = tempfile::tempdir().unwrap();
        let loader = Arc::new(
            SegmentIndexCache::new(StorageRegistry::new(), dir.path().join("index")).unwrap(),
        );
        let build = |with_remote: bool| {
            let mut sources = vec![
                SegmentSource::Local(SegmentId(0), segment(left, StoragePrecision::F32)),
                SegmentSource::Local(SegmentId(1), segment(right, StoragePrecision::Int8)),
            ];
            if with_remote {
                sources.push(SegmentSource::Remote {
                    segment_id: SegmentId(2),
                    owners: vec![PeerAddr("127.0.0.1:1".into())],
                    row_count: 1,
                    index_url: StorageUrl::parse("/tmp/x").unwrap(),
                });
            }
            PlacedIndex::with_sources(
                sources,
                "t",
                StoragePrecision::F32,
                Arc::new(NoPeers),
                Arc::clone(&loader),
                AnnIndexConfig::default(),
                None,
                None,
                Arc::new(PeerFailureCounters::default()),
            )
        };
        assert!(
            build(false).is_err(),
            "AllLocal shape re-asserts uniformity"
        );
        assert!(build(true).is_err(), "Mixed shape re-asserts uniformity");
    }

    #[test]
    fn local_load_estimate_per_precision() {
        // 1000 rows × 128 dims: vector bytes + 96 per row.
        assert_eq!(
            local_load_estimate(1000, 128, StoragePrecision::F32),
            1000 * (512 + 96)
        );
        assert_eq!(
            local_load_estimate(1000, 128, StoragePrecision::F16),
            1000 * (256 + 96)
        );
        assert_eq!(
            local_load_estimate(1000, 128, StoragePrecision::Int8),
            1000 * (128 + 96)
        );
        assert_eq!(
            local_load_estimate(1000, 128, StoragePrecision::Binary),
            1000 * (16 + 96)
        );
        // Binary pads to whole bytes.
        assert_eq!(local_load_estimate(1, 12, StoragePrecision::Binary), 2 + 96);
    }
}
