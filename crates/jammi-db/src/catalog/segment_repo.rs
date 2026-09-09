//! Catalog access for the ANN index-segment set (`index_segments`).
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
use crate::catalog::result_repo::{read_cas_target, CasOutcome, ResultTableCas};
use crate::catalog::Catalog;
use crate::error::Result;

/// One `index_segments` row: a segment's id within its table, the base URL of
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

    /// Insert one segment row at the caller-chosen `segment_id` into the
    /// building table `cas` names, returning `true` when it landed and `false`
    /// when the `(table_name, segment_id)` primary key already held a row —
    /// the collision a concurrent appender that read the same
    /// [`Self::max_index_segment_id`] must retry against. `ON CONFLICT DO
    /// NOTHING` makes the write atomic and portable across both backends: the
    /// loser observes zero affected rows and re-reads the max.
    ///
    /// One transaction, lease check first: the parent row must still match
    /// `cas` (status `building`, the caller's ownership, the tenant arm in
    /// force) or the typed miss ([`crate::error::JammiError::LeaseLost`] and
    /// its siblings) is returned and nothing is inserted — a segment is never
    /// registered under a row its writer no longer owns. The segment inherits
    /// the parent row's own `tenant_id`, read in the same transaction, so a
    /// cross-tenant recovery rebuild under an admin scope stamps each segment
    /// with its table's tenant.
    pub async fn insert_index_segment(
        &self,
        cas: &ResultTableCas,
        segment_id: i64,
        index_path: &str,
        row_count: usize,
    ) -> Result<bool> {
        let cas_in_tx = cas.clone();
        let index_path = index_path.to_string();
        let created_at = crate::catalog::backend::now_sortable();
        let tenant = self.current_tenant();
        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let cas = cas_in_tx;
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> = Vec::new();
                    let predicate = cas.render_owner_exists(&mut params);
                    // `render_owner_exists` carries no status arm; a segment
                    // may only be appended to a row still `building`.
                    let owner_tenant = tx
                        .query_opt(
                            &format!(
                                "SELECT tenant_id FROM result_tables \
                                 WHERE table_name = $1 AND status = 'building' AND {predicate}"
                            ),
                            &params,
                            |row| row.try_get::<String>("tenant_id"),
                        )
                        .await?;
                    let Some(owner_tenant) = owner_tenant else {
                        return Ok(CasOutcome::Missed(read_cas_target(tx, &cas.table).await?));
                    };
                    let affected = tx
                        .execute(
                            "INSERT INTO index_segments \
                               (table_name, segment_id, index_path, row_count, tenant_id, created_at) \
                             VALUES ($1, $2, $3, $4, $5, $6) \
                             ON CONFLICT (table_name, segment_id) DO NOTHING",
                            &[
                                SqlValue::TextOwned(cas.table.clone()),
                                SqlValue::Int(segment_id),
                                SqlValue::TextOwned(index_path),
                                SqlValue::Int(row_count as i64),
                                SqlValue::from(owner_tenant),
                                SqlValue::TextOwned(created_at),
                            ],
                        )
                        .await?;
                    Ok(CasOutcome::Applied(affected == 1))
                })
            })
            .await?;
        match outcome {
            CasOutcome::Applied(landed) => Ok(landed),
            CasOutcome::Missed(target) => Err(Catalog::classify_cas_miss(cas, target)),
        }
    }

    /// Delete every segment row of the table `cas` names, under its ownership
    /// and tenant arms (status-agnostic: a purge follows a transition that
    /// already left `building` — the writer's own `failed`, recovery's claim,
    /// or a `ready` row whose lease is long cleared — and `writer_id` survives
    /// that transition as history, so the arm still names the purger). Used by
    /// a recovery rebuild (which replaces the set with one fresh full-table
    /// segment) and by file-cleanup that clears the catalog rows after their
    /// bundles are deleted; a table *drop* reaps these rows via `ON DELETE
    /// CASCADE` instead. Returns the number of segment rows deleted; a row
    /// the purger does not own yields `0`, never a cross-owner purge.
    pub async fn delete_index_segments(&self, cas: &ResultTableCas) -> Result<u64> {
        let cas = cas.clone();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> =
                        vec![SqlValue::TextOwned(cas.table.clone())];
                    let owner = cas.render_owner_exists(&mut params);
                    tx.execute(
                        &format!("DELETE FROM index_segments WHERE table_name = $1 AND {owner}"),
                        &params,
                    )
                    .await
                })
            })
            .await?)
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
