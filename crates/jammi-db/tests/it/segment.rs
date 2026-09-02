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
use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind, ResultTableRecord};
use jammi_db::catalog::segment_repo::IndexSegment;
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

/// Require-gate (KO-7) for the `JAMMI_TEST_PG_URL`-unset skip the
/// `open_backend(BackendKind::Postgres, ..)` call site in this file falls
/// through to: by default (unset) the Postgres arm still silently skips,
/// exactly as before — a lane that wants to REQUIRE the real Postgres arm
/// run (never silently skip it) sets `JAMMI_REQUIRE_PG`, and this call
/// panics instead.
fn require_live_pg(test_name: &str) {
    if std::env::var_os("JAMMI_REQUIRE_PG").is_some() {
        panic!(
            "{test_name}: JAMMI_REQUIRE_PG is set but JAMMI_TEST_PG_URL is unset -- this lane \
             must run the real Postgres arm, not skip it"
        );
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
        require_live_pg("concurrent_append_never_collides_and_cascades");
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

// ---------------------------------------------------------------------------
// The session verb: `JammiSession::list_index_segments`
// ---------------------------------------------------------------------------
//
// The `index_segments` rows were readable through no public surface: `sql()`
// federates result tables and external sources, not the catalog's own tables,
// and no session verb reached them. These pin the verb that closes that gap and
// — the part that matters — its tenant gate. `Catalog::list_index_segments` is
// NOT independently tenant-filtered (its rows are scoped by their parent), so
// the session resolves `table_name` through the tenant-filtered
// `get_result_table` first. An unresolvable table lists nothing, and it lists
// nothing IDENTICALLY whether it is unknown or simply another tenant's — the
// verb is not an existence oracle for a peer's table names.

/// A fresh per-test tenant id; never a fixed literal (sibling tests share a
/// catalog on the Postgres lane).
fn segment_tenant() -> jammi_db::TenantId {
    jammi_db::TenantId::from_uuid(uuid::Uuid::new_v4()).unwrap()
}

/// Create a `result_tables` row named `table` under whatever tenant the calling
/// session is bound to (the repo stamps `tenant_id` from the binding).
async fn seed_result_table(session: &jammi_db::session::JammiSession, table: &str) {
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            table_name: table,
            source_id: "seg_src",
            model_id: "seg_model",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "file:///tmp/seg.parquet",
            dimensions: Some(4),
            key_column: Some("id"),
            text_columns: None,
            storage_precision: StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
        })
        .await
        .unwrap();
}

/// Two segments on one table list in `segment_id` order through the session
/// verb — including when they were inserted out of order, so the ordering is
/// the query's, not the insertion's.
#[tokio::test]
async fn session_lists_a_tables_segments_in_segment_id_order() {
    let dir = tempdir().unwrap();
    let tenant = segment_tenant();
    let session = jammi_test_utils::make_test_session(BackendKind::Sqlite, dir.path())
        .await
        .expect("sqlite session")
        .with_tenant(tenant);

    seed_result_table(&session, "seg_table").await;
    // Inserted 1-then-0: the listing must still come back 0, 1.
    for (id, path, rows) in [
        (1_i64, "file:///idx/seg-1", 7_usize),
        (0, "file:///idx/seg-0", 3),
    ] {
        assert!(session
            .catalog()
            .insert_index_segment("seg_table", Some(tenant), id, path, rows)
            .await
            .unwrap());
    }

    let listed = session.list_index_segments("seg_table").await.unwrap();
    assert_eq!(
        listed,
        vec![
            IndexSegment {
                segment_id: 0,
                index_path: "file:///idx/seg-0".to_string(),
                row_count: 3,
            },
            IndexSegment {
                segment_id: 1,
                index_path: "file:///idx/seg-1".to_string(),
                row_count: 7,
            },
        ],
        "the session verb must return every segment of the table, ordered by segment_id"
    );

    // The verb reads the same rows the catalog-level API does — the session
    // adds the tenant gate, never a different projection.
    assert_eq!(
        listed,
        session
            .catalog()
            .list_index_segments("seg_table")
            .await
            .unwrap(),
        "the session verb must not reshape the catalog's rows"
    );
}

/// A table the session's tenant cannot resolve lists NOTHING — and lists the
/// same nothing an unknown table lists, so the verb cannot be used to probe
/// which table names a peer tenant owns. The CROSS-TENANT DENIAL case for this
/// verb.
#[tokio::test]
async fn session_hides_another_tenants_segments_and_an_unknown_table_alike() {
    let dir = tempdir().unwrap();
    let tenant_a = segment_tenant();
    let tenant_b = segment_tenant();
    let session = jammi_test_utils::make_test_session(BackendKind::Sqlite, dir.path())
        .await
        .expect("sqlite session")
        .with_tenant(tenant_a);

    seed_result_table(&session, "a_only_table").await;
    assert!(session
        .catalog()
        .insert_index_segment("a_only_table", Some(tenant_a), 0, "file:///idx/a-0", 5)
        .await
        .unwrap());

    // A sees its own segment.
    assert_eq!(
        session
            .list_index_segments("a_only_table")
            .await
            .unwrap()
            .len(),
        1,
        "tenant A must see the segment of its own table"
    );

    // B — scoped on the same session, the same path a gRPC request takes —
    // sees nothing.
    let b_view = session
        .with_tenant_scoped(tenant_b, |scope| async move {
            scope.list_index_segments("a_only_table").await
        })
        .await
        .unwrap();
    assert!(
        b_view.is_empty(),
        "CROSS-TENANT READ LEAK: tenant B saw tenant A's index segments: {b_view:?}"
    );

    // And the unknown-table answer is byte-identical, so the empty listing
    // leaks no existence signal.
    let unknown = session
        .with_tenant_scoped(tenant_b, |scope| async move {
            scope.list_index_segments("no_such_table_at_all").await
        })
        .await
        .unwrap();
    assert_eq!(
        b_view, unknown,
        "a peer's table and an unknown table must be indistinguishable through this verb"
    );

    // The bare catalog read is NOT gated — which is precisely why the session
    // resolves the parent row first. Pinning it here keeps the gate's reason
    // visible if the catalog layer ever changes.
    assert_eq!(
        session
            .catalog()
            .list_index_segments("a_only_table")
            .await
            .unwrap()
            .len(),
        1,
        "the catalog-level read is parent-scoped, not independently tenant-filtered"
    );
}
