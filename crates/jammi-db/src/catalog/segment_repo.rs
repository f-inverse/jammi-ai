//! Catalog access for the ANN index-segment set (the index_segments table).
//!
//! A table's ANN index is a *set* of immutable segments (migration 025), one
//! catalog row per segment. This repo owns the three operations the segment set
//! needs: read the current maximum `segment_id` for a table (the allocator's
//! read half), insert one segment row at an explicitly chosen id surfacing a
//! primary-key collision (the allocator's write half), and list a table's
//! segments in id order (the reader). The *allocation loop* — read-max,
//! derive-the-bundle-URL-from-the-id, insert, retry on collision — lives one
//! level up in [`crate::store::ResultStore::append_segment`], because the
//! segment bundle's URL embeds the allocated id and URL layout is the store's
//! concern, not the catalog's.

use crate::catalog::backend::{Row, SqlValue, TxOptions};
use crate::catalog::Catalog;
use crate::error::Result;
use crate::tenant::TenantId;

/// One index_segments row: a segment's id within its table, the base URL of
/// its sidecar bundle (no extension — the layout helpers append
/// `.usearch` / `.rowmap` / `.manifest.json` / …), and its own row contribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSegment {
    /// Per-table segment sequence position, starting at `0`.
    pub segment_id: i64,
    /// The sidecar-bundle base URL for this segment.
    pub index_path: String,
    /// The number of rows this segment indexes.
    pub row_count: usize,
}

fn parse_segment(
    row: &Row<'_>,
) -> std::result::Result<IndexSegment, crate::catalog::backend::BackendError> {
    Ok(IndexSegment {
        segment_id: row.get::<i32>("segment_id")? as i64,
        index_path: row.get("index_path")?,
        row_count: row.get::<i32>("row_count")? as usize,
    })
}

impl Catalog {
    /// The largest `segment_id` currently recorded for `table_name`, or `None`
    /// when the table has no segments yet. The allocator adds `1` (or starts at
    /// `0`) and retries [`Self::insert_index_segment`] on the primary-key
    /// collision a concurrent appender racing to the same next id would cause.
    pub async fn max_index_segment_id(&self, table_name: &str) -> Result<Option<i64>> {
        let table_name = table_name.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT MAX(segment_id) AS max_id FROM index_segments \
                             WHERE table_name = $1",
                            &[SqlValue::TextOwned(table_name)],
                            |row| row.try_get::<i32>("max_id"),
                        )
                        .await
                    })
                },
            )
            .await?
            .flatten()
            .map(i64::from))
    }

    /// Insert one segment row at the caller-chosen `segment_id`, returning
    /// `true` when it landed and `false` when the `(table_name, segment_id)`
    /// primary key already held a row — the collision a concurrent appender
    /// that read the same [`Self::max_index_segment_id`] must retry against.
    /// `ON CONFLICT DO NOTHING` makes the write atomic and portable across both
    /// backends: the loser observes zero affected rows and re-reads the max.
    ///
    /// `tenant_id` is the owning tenant carried from the parent table (`None`
    /// for a GLOBAL table), passed explicitly rather than read off the session
    /// so a cross-tenant recovery rebuild under an admin scope stamps each
    /// segment with its table's own tenant.
    pub async fn insert_index_segment(
        &self,
        table_name: &str,
        tenant_id: Option<TenantId>,
        segment_id: i64,
        index_path: &str,
        row_count: usize,
    ) -> Result<bool> {
        let table_name = table_name.to_string();
        let index_path = index_path.to_string();
        let tenant_str = tenant_id.map(|t| t.to_string());
        let created_at = crate::catalog::backend::now_sortable();
        let affected = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO index_segments \
                           (table_name, segment_id, index_path, row_count, tenant_id, created_at) \
                         VALUES ($1, $2, $3, $4, $5, $6) \
                         ON CONFLICT (table_name, segment_id) DO NOTHING",
                        &[
                            SqlValue::TextOwned(table_name),
                            SqlValue::Int(segment_id),
                            SqlValue::TextOwned(index_path),
                            SqlValue::Int(row_count as i64),
                            SqlValue::from(tenant_str),
                            SqlValue::TextOwned(created_at),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(affected == 1)
    }

    /// Delete every segment row of `table_name`. Used by a recovery rebuild
    /// (which replaces the set with one fresh full-table segment) and by
    /// file-cleanup that clears the catalog rows after their bundles are
    /// deleted; a table *drop* reaps these rows via `ON DELETE CASCADE` instead.
    pub async fn delete_index_segments(&self, table_name: &str) -> Result<()> {
        let table_name = table_name.to_string();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM index_segments WHERE table_name = $1",
                        &[SqlValue::TextOwned(table_name)],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Every segment of `table_name`, ordered by `segment_id` — the merge input
    /// [`crate::store::ResultStore::resolve_search_mode`] loads into a
    /// [`crate::index::segment::SegmentedIndex`], and the file-cleanup set a
    /// table delete enumerates before the `ON DELETE CASCADE` reaps the rows.
    ///
    /// Not independently tenant-filtered: a caller reaches this only with a
    /// `table_name` it already resolved through the tenant-scoped
    /// `result_tables` read, so the segment set is scoped by its parent exactly
    /// as reading a column off that already-scoped row was.
    pub async fn list_index_segments(&self, table_name: &str) -> Result<Vec<IndexSegment>> {
        let table_name = table_name.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT segment_id, index_path, row_count FROM index_segments \
                             WHERE table_name = $1 ORDER BY segment_id",
                            &[SqlValue::TextOwned(table_name)],
                            parse_segment,
                        )
                        .await
                    })
                },
            )
            .await?)
    }
}
