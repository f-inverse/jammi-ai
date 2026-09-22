//! The ANN segments of a result table's write, built in parallel from the
//! rows as they are written.
//!
//! A table's index is a set of segments ([`crate::index::segment`]), and an
//! HNSW graph is a function of its insertion order: a node's neighbours are
//! chosen by searching the graph as it stands when the node is inserted, so
//! two builds answer a query identically only if they inserted the same rows
//! in the same order. That is a property PER INDEX, not per table. The
//! builder therefore cuts the written rows, in the order they are written —
//! key order, the table's own — into segments of `rows_per_segment`
//! consecutive rows, and builds each on its own thread from its own rows in
//! that order. No build shares state with another, every segment is a
//! function of the row sequence and the budget alone — the layout is
//! identical across partition counts and executors — and the builds overlap
//! the write and each other. A search fans out over the segments as it does
//! for every appended table; a compaction rewrites the set at the same
//! budget.
//!
//! In-flight builds are bounded by the host's parallelism: a segment whose
//! turn has not come waits, holding back the stream — each pending segment
//! holds its rows' exact vectors, so the bound is what caps the memory
//! `rows_per_segment` segments would otherwise stack up.

use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::physical_plan::metrics::Time;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::sidecar::SidecarIndex;
use crate::index::VectorIndex;

/// The rows of the segment being gathered.
struct OpenSegment {
    row_ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

impl OpenSegment {
    fn new(capacity: usize) -> Self {
        Self {
            row_ids: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity),
        }
    }
}

/// Builds a table's ANN segments from its rows in write order. See the
/// module doc.
pub struct SegmentBuilder {
    dimensions: usize,
    ann: AnnIndexConfig,
    precision: StoragePrecision,
    rows_per_segment: NonZeroUsize,
    open: OpenSegment,
    /// The builds in progress, in segment order.
    builds: Vec<JoinHandle<Result<SidecarIndex>>>,
    permits: Arc<Semaphore>,
    /// The summed thread time of every build.
    build_time: Time,
}

impl SegmentBuilder {
    /// A builder of segments of `rows_per_segment` rows at `dimensions`,
    /// each built with `ann`'s knobs at `precision`; `build_time` accrues
    /// every build's own time.
    pub fn new(
        dimensions: usize,
        ann: AnnIndexConfig,
        precision: StoragePrecision,
        rows_per_segment: NonZeroUsize,
        build_time: Time,
    ) -> Self {
        let parallelism = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        Self {
            dimensions,
            ann,
            precision,
            rows_per_segment,
            open: OpenSegment::new(rows_per_segment.get()),
            builds: Vec::new(),
            permits: Arc::new(Semaphore::new(parallelism)),
            build_time,
        }
    }

    /// Take the next rows, in write order; every segment they complete
    /// starts building.
    pub async fn push(&mut self, row_ids: Vec<String>, vectors: Vec<Vec<f32>>) -> Result<()> {
        if row_ids.len() != vectors.len() {
            return Err(JammiError::Other(format!(
                "segment builder: {} row ids for {} vectors",
                row_ids.len(),
                vectors.len()
            )));
        }
        for (row_id, vector) in row_ids.into_iter().zip(vectors) {
            self.open.row_ids.push(row_id);
            self.open.vectors.push(vector);
            if self.open.row_ids.len() == self.rows_per_segment.get() {
                self.seal().await?;
            }
        }
        Ok(())
    }

    /// End of rows: the segments, in order, once every build is complete.
    /// An empty builder yields none.
    pub async fn finish(mut self) -> Result<Vec<SidecarIndex>> {
        if !self.open.row_ids.is_empty() {
            self.seal().await?;
        }
        let mut segments = Vec::with_capacity(self.builds.len());
        for build in self.builds {
            segments.push(build.await.map_err(|e| {
                JammiError::Other(format!("segment builder: a build thread failed: {e}"))
            })??);
        }
        Ok(segments)
    }

    /// Start building the open segment on its own thread, once a build slot
    /// is free.
    async fn seal(&mut self) -> Result<()> {
        let rows = std::mem::replace(
            &mut self.open,
            OpenSegment::new(self.rows_per_segment.get()),
        );
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|e| JammiError::Other(format!("segment builder: build slots closed: {e}")))?;
        let (dimensions, ann, precision, timer) = (
            self.dimensions,
            self.ann,
            self.precision,
            self.build_time.clone(),
        );
        self.builds.push(tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let _build = timer.timer();
            let mut index = SidecarIndex::new(dimensions, &ann, precision)?;
            for (row_id, vector) in rows.row_ids.iter().zip(&rows.vectors) {
                index.add(row_id, vector)?;
            }
            index.build()?;
            Ok(index)
        }));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::validate_query;
    use crate::index::QuerySource;

    fn vector(i: usize) -> Vec<f32> {
        let x = i as f32;
        vec![x.sin(), x.cos(), (x * 0.5).sin(), 1.0]
    }

    async fn segments_of(rows: usize, per_segment: usize) -> Vec<SidecarIndex> {
        let mut builder = SegmentBuilder::new(
            4,
            AnnIndexConfig::default(),
            StoragePrecision::F32,
            NonZeroUsize::new(per_segment).unwrap(),
            Time::new(),
        );
        // Batches of seven rows, so segments straddle batch boundaries.
        for start in (0..rows).step_by(7) {
            let end = (start + 7).min(rows);
            builder
                .push(
                    (start..end).map(|i| format!("{i:04}")).collect(),
                    (start..end).map(vector).collect(),
                )
                .await
                .unwrap();
        }
        builder.finish().await.unwrap()
    }

    /// Segments are consecutive runs of the write order at the budget, the
    /// last one shorter, each holding exactly its rows.
    #[tokio::test]
    async fn segments_are_consecutive_runs_of_the_write_order() {
        let segments = segments_of(23, 10).await;
        assert_eq!(
            segments.iter().map(SidecarIndex::len).collect::<Vec<_>>(),
            vec![10, 10, 3]
        );
        for (s, segment) in segments.iter().enumerate() {
            for i in 0..23 {
                assert_eq!(
                    segment.contains_row(&format!("{i:04}")),
                    i / 10 == s,
                    "row {i} belongs to segment {}",
                    i / 10
                );
            }
        }
    }

    /// The same rows in the same order build the same segments: every query
    /// answers bit-for-bit alike across two builds.
    #[tokio::test]
    async fn two_builds_over_the_same_rows_answer_alike() {
        let a = segments_of(50, 16).await;
        let b = segments_of(50, 16).await;
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            for q in [3usize, 17, 41] {
                let query = validate_query(vector(q), 4, QuerySource::Caller).unwrap();
                let hits = |index: &SidecarIndex| {
                    index
                        .search(&query, 5)
                        .unwrap()
                        .into_iter()
                        .map(|(id, d)| (id, d.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(hits(x), hits(y));
            }
        }
    }

    /// No rows, no segments.
    #[tokio::test]
    async fn no_rows_build_no_segment() {
        assert!(segments_of(0, 8).await.is_empty());
    }
}
