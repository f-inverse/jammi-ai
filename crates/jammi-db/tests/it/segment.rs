//! The ANN index-segment set (migration 025): incremental append without
//! rebuild, concurrent segment-id allocation, catalog round-trip + cascade, and
//! the `search_vectors` lane over a multi-segment quantized table.
//!
//! The pure merge order / rescore correctness lives in the `index::segment`
//! unit tests; these prove the catalog + storage + store wiring around it.

use std::sync::Arc;

use datafusion::prelude::SessionContext;
use jammi_db::catalog::backend::{BackendImpl, BackendKind};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::Catalog;
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::model_task::ModelTask;
use jammi_db::store::ResultStore;
use jammi_numerics::distance::cosine_distance;
use tempfile::tempdir;
use test_case::test_case;

async fn open_backend(kind: BackendKind, dir: &std::path::Path) -> Option<BackendImpl> {
    match kind {
        BackendKind::Sqlite => Some(BackendImpl::Sqlite(
            SqliteBackend::open(&dir.join("catalog.db")).await.unwrap(),
        )),
        BackendKind::Postgres => {
            let url = jammi_test_utils::pg_url_for_tests()?;
            Some(BackendImpl::Postgres(
                PostgresBackend::open_with_options(&url, 8, None)
                    .await
                    .expect("open postgres backend"),
            ))
        }
    }
}

async fn fresh_catalog(backend: BackendImpl) -> Arc<Catalog> {
    backend.migrate().await.unwrap();
    let catalog = Arc::new(Catalog::from_backend(backend));
    // The Postgres lane shares one DB across the run; clear the child table then
    // the parent so a cross-test scan sees only this test's rows.
    catalog
        .backend_arc()
        .transaction(Default::default(), |tx| {
            Box::pin(async move {
                tx.execute("DELETE FROM index_segments", &[]).await?;
                tx.execute("DELETE FROM result_tables", &[]).await
            })
        })
        .await
        .unwrap();
    catalog
}

fn store(dir: &std::path::Path, catalog: Arc<Catalog>, precision: StoragePrecision) -> ResultStore {
    let ann = AnnIndexConfig {
        storage_precision: precision,
        ..AnnIndexConfig::default()
    };
    ResultStore::new(dir, catalog, ann).unwrap()
}

/// Register a `building` embedding table and return its catalog record.
async fn building_table(store: &ResultStore) -> ResultTableRecord {
    let info = store
        .create_table(
            "src",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "model",
            Some(4),
            Some("_row_id"),
            None,
        )
        .await
        .unwrap();
    store
        .catalog()
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .unwrap()
}

/// Build a fully-built one-segment [`SidecarIndex`] over `rows` at `precision`.
fn built_index(rows: &[(&str, [f32; 4])], precision: StoragePrecision) -> SidecarIndex {
    let mut idx = SidecarIndex::new(4, &AnnIndexConfig::default(), precision).unwrap();
    for (id, v) in rows {
        idx.add(id, v).unwrap();
    }
    idx.build().unwrap();
    idx
}

// Test 3 — appending a second segment leaves the first segment's on-disk bundle
// byte-for-byte untouched, and both segments' rows become searchable through
// the merged index.
#[tokio::test]
async fn append_does_not_rebuild_prior_segments() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = store(dir.path(), catalog, StoragePrecision::F32);
    let table = building_table(&store).await;

    let seg0 = built_index(
        &[("a", [1.0, 0.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0, 0.0])],
        StoragePrecision::F32,
    );
    let id0 = store.append_segment(&table, &seg0).await.unwrap();

    // Snapshot segment 0's graph bytes and the dot-free discriminator naming.
    let segs = store
        .catalog()
        .list_index_segments(&table.table_name)
        .await
        .unwrap();
    assert_eq!(segs.len(), 1);
    assert!(
        segs[0]
            .index_path
            .contains(&format!("{}__seg0.idx", table.table_name)),
        "segment 0 URL is the dot-free {{table}}__seg0.idx discriminator: {}",
        segs[0].index_path
    );
    let seg0_usearch = jammi_test_utils::url_to_path(&segs[0].index_path).with_extension("usearch");
    let before = std::fs::read(&seg0_usearch).unwrap();

    // Append segment 1 over disjoint rows.
    let seg1 = built_index(
        &[("c", [0.0, 0.0, 1.0, 0.0]), ("d", [0.0, 0.0, 0.0, 1.0])],
        StoragePrecision::F32,
    );
    let id1 = store.append_segment(&table, &seg1).await.unwrap();
    assert_ne!(id0, id1, "the appended segment gets a fresh id");

    // Segment 0's bytes are unchanged — no rebuild.
    let after = std::fs::read(&seg0_usearch).unwrap();
    assert_eq!(
        before, after,
        "appending must not rewrite an existing segment"
    );

    // Both segments' rows are searchable through the merged index.
    let index = store.resolve_search_mode(&table).await.unwrap().unwrap();
    assert_eq!(index.len(), 4);
    let hit_c = index.search_final(&[0.0, 0.0, 1.0, 0.0], 1, 4).unwrap();
    assert_eq!(hit_c.first().map(|(id, _)| id.as_str()), Some("c"));
    let hit_a = index.search_final(&[1.0, 0.0, 0.0, 0.0], 1, 4).unwrap();
    assert_eq!(hit_a.first().map(|(id, _)| id.as_str()), Some("a"));
}

// Test 6 — concurrent appends never collide on a segment id (both backends),
// the catalog round-trips the set, and dropping the table cascades the segment
// rows away.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn concurrent_append_never_collides_and_cascades(kind: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = Arc::new(store(
        dir.path(),
        Arc::clone(&catalog),
        StoragePrecision::F32,
    ));
    let table = building_table(&store).await;

    // Fan out N concurrent appends; the allocator's read-max + insert +
    // PK-conflict retry must hand each a distinct id with no lost writes.
    const N: i64 = 8;
    let mut handles = Vec::new();
    for i in 0..N {
        let store = Arc::clone(&store);
        let table = table.clone();
        handles.push(tokio::spawn(async move {
            let idx = built_index(
                &[(&format!("r{i}"), [i as f32, 1.0, 0.0, 0.0])],
                StoragePrecision::F32,
            );
            store.append_segment(&table, &idx).await.unwrap()
        }));
    }
    let mut ids: Vec<i64> = Vec::new();
    for h in handles {
        ids.push(h.await.unwrap().0);
    }
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..N).collect::<Vec<_>>(),
        "each append got a distinct id, no collisions"
    );

    // Round-trip: the catalog lists exactly the N segments in id order.
    let segs = catalog
        .list_index_segments(&table.table_name)
        .await
        .unwrap();
    assert_eq!(segs.len(), N as usize);
    assert_eq!(
        segs.iter().map(|s| s.segment_id).collect::<Vec<_>>(),
        (0..N).collect::<Vec<_>>()
    );

    // ON DELETE CASCADE: dropping the parent result-table row reaps its segments.
    catalog
        .backend_arc()
        .transaction(Default::default(), |tx| {
            let name = table.table_name.clone();
            Box::pin(async move {
                tx.execute(
                    "DELETE FROM result_tables WHERE table_name = $1",
                    &[jammi_db::catalog::backend::SqlValue::TextOwned(name)],
                )
                .await
            })
        })
        .await
        .unwrap();
    assert!(
        catalog
            .list_index_segments(&table.table_name)
            .await
            .unwrap()
            .is_empty(),
        "ON DELETE CASCADE reaps the segment rows with the table"
    );
}

// Test 9 (non-rescore consumer) — `search_vectors` over a two-segment quantized
// table routes through `search_final`, so it returns the exact cross-segment
// comparable top-k (the brute-force baseline), never raw per-segment candidate
// distances.
#[tokio::test]
async fn search_vectors_over_two_int8_segments_equals_brute_force() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = store(dir.path(), catalog, StoragePrecision::Int8);
    let table = building_table(&store).await;

    let rows_left = [
        ("a", [1.0, 0.0, 0.0, 0.1]),
        ("b", [0.0, 1.0, 0.0, 0.2]),
        ("c", [0.0, 0.0, 1.0, 0.3]),
    ];
    let rows_right = [
        ("d", [0.9, 0.1, 0.0, 0.0]),
        ("e", [0.1, 0.9, 0.0, 0.0]),
        ("f", [0.0, 0.1, 0.9, 0.0]),
    ];
    store
        .append_segment(&table, &built_index(&rows_left, StoragePrecision::Int8))
        .await
        .unwrap();
    store
        .append_segment(&table, &built_index(&rows_right, StoragePrecision::Int8))
        .await
        .unwrap();

    let all: Vec<(&str, [f32; 4])> = rows_left.iter().chain(rows_right.iter()).copied().collect();
    let ctx = SessionContext::new();
    let k = 3;
    for (_, q) in &all {
        let hits = store.search_vectors(&ctx, &table, q, k).await.unwrap();
        let got: Vec<String> = hits.into_iter().map(|(id, _)| id).collect();

        let mut truth: Vec<(String, f32)> = all
            .iter()
            .map(|(id, v)| (id.to_string(), cosine_distance(q, v)))
            .collect();
        truth.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let expected: Vec<String> = truth.into_iter().take(k).map(|(id, _)| id).collect();

        assert_eq!(
            got, expected,
            "search_vectors over a 2-segment Int8 table must equal the exact brute-force top-k"
        );
    }
}

// Sanity: `Catalog::open` migrates through 025 so the `index_segments` table
// exists and starts empty.
#[tokio::test]
async fn migration_025_creates_an_empty_index_segments_table() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    assert!(catalog
        .list_index_segments("nonexistent")
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        catalog.max_index_segment_id("nonexistent").await.unwrap(),
        None
    );
}
