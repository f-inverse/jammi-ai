//! The placement arity guard: `ResultStore::resolve_search_mode` treats a
//! `SegmentPlacement::plan` reply whose length disagrees with the requested
//! segment count as a placement fault (`JammiError::Catalog`, naming both
//! counts) — never a silent zip that serves a subset of segments as if it
//! were the whole table.

use std::sync::Arc;

use async_trait::async_trait;
use jammi_datafusion::ModelTask;
use jammi_db::catalog::backend::BackendImpl;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::catalog::Catalog;
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::error::{JammiError, Result};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::{PeerAddr, SegmentId, SegmentPlacement, VectorIndex};
use jammi_db::store::{BuildingTable, ResultStore};
use tempfile::tempdir;

/// A placement that always returns a single, hard-coded, EMPTY owner list —
/// deliberately shorter than any multi-segment request — to force the
/// arity guard.
#[derive(Debug, Default)]
struct ShortAnswerPlacement;

#[async_trait]
impl SegmentPlacement for ShortAnswerPlacement {
    async fn plan(&self, _table: &str, _segments: &[SegmentId]) -> Result<Vec<Vec<PeerAddr>>> {
        Ok(vec![Vec::new()])
    }
}

async fn fresh_store(dir: &std::path::Path, placement: Arc<dyn SegmentPlacement>) -> ResultStore {
    let backend = BackendImpl::Sqlite(SqliteBackend::open(&dir.join("catalog.db")).await.unwrap());
    backend.migrate().await.unwrap();
    let catalog = Arc::new(Catalog::from_backend(backend));
    let ann = AnnIndexConfig {
        storage_precision: StoragePrecision::default(),
        ..AnnIndexConfig::default()
    };
    ResultStore::new(dir, catalog, ann)
        .unwrap()
        .with_placement(placement)
}

async fn two_segment_table(store: &ResultStore) -> BuildingTable {
    let table = store
        .create_table(
            "src",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "model",
            Some(4),
            Some("_row_id"),
            None,
            None,
        )
        .await
        .unwrap();
    for rows in [
        [("a", [1.0f32, 0.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0, 0.0])],
        [("c", [0.0, 0.0, 1.0, 0.0]), ("d", [0.0, 0.0, 0.0, 1.0])],
    ] {
        let mut idx =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::default()).unwrap();
        for (id, v) in rows {
            idx.add(id, &v).unwrap();
        }
        idx.build().unwrap();
        table.append_segment(&idx).await.unwrap();
    }
    table
}

// A placement reply shorter than the requested segment count is a typed,
// named-counts refusal — never a silent truncation that zips the short
// answer against the long request.
//
// Executed mutation (reverted): commenting out the
// `if owners.len() != segments.len() { return Err(...) }` guard in
// `ResultStore::resolve_search_mode` and re-running this test turns the
// `Err(Catalog(..))` this test asserts on into `Ok(..)` — the assertion on
// the error message fails with:
//   "placement arity guard did not fire: resolve_search_mode returned Ok"
// i.e. the guard's absence reds this test rather than leaving it vacuously
// green.
#[tokio::test]
async fn placement_reply_shorter_than_segment_count_is_a_named_arity_refusal() {
    let dir = tempdir().unwrap();
    let store = fresh_store(dir.path(), Arc::new(ShortAnswerPlacement)).await;
    let table = two_segment_table(&store).await;
    let record = store
        .catalog()
        .get_result_table(table.table_name())
        .await
        .unwrap()
        .unwrap();

    let result = store.resolve_search_mode(&record).await;
    let err = match result {
        Err(err) => err,
        Ok(_) => panic!("placement arity guard did not fire: resolve_search_mode returned Ok"),
    };

    let JammiError::Catalog(msg) = err else {
        panic!("expected JammiError::Catalog naming both counts, got a different variant");
    };
    assert!(
        msg.contains("1 owner lists for 2 segments"),
        "message must name both the returned-owners count (1) and the \
         requested-segments count (2): {msg}"
    );
}
