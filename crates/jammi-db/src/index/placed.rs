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

use std::sync::Arc;

use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::peer::{PeerAddr, PeerFailureCounters, PeerTransport};
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
// The owners and bundle URL are read by the `Mixed` search arm (the protocol
// and ladder), which lands in the next commit.
#[allow(dead_code)]
pub(crate) struct RemoteSegment {
    pub(crate) segment_id: SegmentId,
    pub(crate) owners: Vec<PeerAddr>,
    pub(crate) row_count: usize,
    pub(crate) index_url: StorageUrl,
}

/// The two shapes a placed table takes.
pub(crate) enum Placed {
    /// Every segment is resident: the sync all-local index, searched as-is.
    AllLocal(SegmentedIndex),
    /// At least one segment is owned by a peer.
    Mixed {
        local: Vec<(SegmentId, SidecarIndex)>,
        remote: Vec<RemoteSegment>,
    },
}

/// The read-side handle over a table's whole placed segment set. Opaque: the
/// only search entry is [`Self::search_final_placed`].
// The transport / loader / budget fields are read by the `Mixed` search arm
// (the protocol and ladder), which lands in the next commit.
#[allow(dead_code)]
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
            Placed::AllLocal(index)
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

    /// The `Mixed` arm: the protocol and ladder land in the next commit.
    async fn search_mixed(
        &self,
        _local: &[(SegmentId, SidecarIndex)],
        remote: &[RemoteSegment],
        _query: &[f32],
        _k: usize,
        _oversample: usize,
    ) -> Result<Vec<(String, f32)>> {
        let segment = remote.first().map(|r| r.segment_id.0).unwrap_or(-1);
        Err(JammiError::Unavailable {
            resource: format!("segment {}/{segment}", self.table_name),
            reason: "placed search over remote segments is not built".into(),
        })
    }
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
}
