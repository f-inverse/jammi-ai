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
use jammi_db::catalog::result_repo::{
    CreateResultTableParams, Owner, ResultTableCas, ResultTableKind, ResultTableRecord, TenantArm,
};
use jammi_db::catalog::segment_repo::IndexSegment;
use jammi_db::catalog::Catalog;
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::model_task::ModelTask;
use jammi_db::store::{BuildingTable, ResultStore};
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

/// Register a `building` embedding table and return the writer's handle (the
/// lease-owned row every segment append is a CAS against).
async fn building_table(store: &ResultStore) -> BuildingTable {
    store
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
        .unwrap()
}

/// The catalog record behind a building handle.
async fn record_of(store: &ResultStore, building: &BuildingTable) -> ResultTableRecord {
    store
        .catalog()
        .get_result_table(building.table_name())
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
    let id0 = table.append_segment(&seg0).await.unwrap();

    // Snapshot segment 0's graph bytes and the dot-free discriminator naming.
    let segs = store
        .catalog()
        .list_index_segments(table.table_name())
        .await
        .unwrap();
    assert_eq!(segs.len(), 1);
    assert!(
        segs[0]
            .index_path
            .contains(&format!("{}__seg0.idx", table.table_name())),
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
    let id1 = table.append_segment(&seg1).await.unwrap();
    assert_ne!(id0, id1, "the appended segment gets a fresh id");

    // Segment 0's bytes are unchanged — no rebuild.
    let after = std::fs::read(&seg0_usearch).unwrap();
    assert_eq!(
        before, after,
        "appending must not rewrite an existing segment"
    );

    // Both segments' rows are searchable through the merged index.
    let index = store
        .resolve_search_mode_local(&record_of(&store, &table).await)
        .await
        .unwrap()
        .unwrap();
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
    let table = Arc::new(building_table(&store).await);

    // Fan out N concurrent appends; the allocator's read-max + insert +
    // PK-conflict retry must hand each a distinct id with no lost writes.
    const N: i64 = 8;
    let mut handles = Vec::new();
    for i in 0..N {
        let table = Arc::clone(&table);
        handles.push(tokio::spawn(async move {
            let idx = built_index(
                &[(&format!("r{i}"), [i as f32, 1.0, 0.0, 0.0])],
                StoragePrecision::F32,
            );
            table.append_segment(&idx).await.unwrap()
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
        .list_index_segments(table.table_name())
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
            let name = table.table_name().to_string();
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
            .list_index_segments(table.table_name())
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
    table
        .append_segment(&built_index(&rows_left, StoragePrecision::Int8))
        .await
        .unwrap();
    table
        .append_segment(&built_index(&rows_right, StoragePrecision::Int8))
        .await
        .unwrap();

    let all: Vec<(&str, [f32; 4])> = rows_left.iter().chain(rows_right.iter()).copied().collect();
    let ctx = SessionContext::new();
    let k = 3;
    for (_, q) in &all {
        let hits = store
            .search_vectors(&ctx, &record_of(&store, &table).await, q, k)
            .await
            .unwrap();
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
            writer_id: None,
            lease: None,
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
            job_attempt: None,
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
            .insert_index_segment(
                &ResultTableCas {
                    table: "seg_table".to_string(),
                    tenant_arm: TenantArm::Strict(Some(tenant)),
                    owner: Owner::ExpiredLease,
                    lease_present: false,
                },
                id,
                path,
                rows,
            )
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
                version: None,
            },
            IndexSegment {
                segment_id: 1,
                index_path: "file:///idx/seg-1".to_string(),
                row_count: 7,
                version: None,
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
        .insert_index_segment(
            &ResultTableCas {
                table: "a_only_table".to_string(),
                tenant_arm: TenantArm::Strict(Some(tenant_a)),
                owner: Owner::ExpiredLease,
                lease_present: false,
            },
            0,
            "file:///idx/a-0",
            5,
        )
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

/// Promote a `building` fixture table to `ready` the way every versioned-table
/// test needs one: the row is `ready`, `current_version` NULL, `next_version`
/// 0 (a never-refreshed table).
async fn ready_table(store: &ResultStore) -> ResultTableRecord {
    let table = building_table(store).await;
    let name = table.table_name().to_string();
    table.detach();
    store
        .catalog()
        .update_result_table_status(
            &name,
            jammi_db::catalog::status::ResultTableStatus::Ready,
            0,
        )
        .await
        .unwrap();
    store
        .catalog()
        .get_result_table(&name)
        .await
        .unwrap()
        .unwrap()
}

// C2 — the version allocator is monotonic and never reuses a number: two
// allocations take 0 and 1; failing 1 keeps it; the next allocation takes 2;
// expiring 1 never lowers `next_version`. A segment appended under a version
// is stamped with it (excluded from the base set) and reaped with it; the
// base publish and the publish CAS swap `current_version` exactly once.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn allocation_is_monotonic_and_never_reused(kind: BackendKind) {
    use jammi_db::catalog::version_repo::{PublishVersion, VersionCas};

    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("allocation_is_monotonic_and_never_reused");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = store(dir.path(), Arc::clone(&catalog), StoragePrecision::F32);
    let table = ready_table(&store).await;
    assert_eq!(table.current_version, None);
    assert_eq!(table.next_version, 0);

    // Two allocations on the same parent: 0 then 1, both parent None.
    let v0 = store.allocate_version(&table).await.unwrap();
    assert_eq!((v0.version(), v0.parent_version()), (0, None));
    let v1 = store.allocate_version(&table).await.unwrap();
    assert_eq!((v1.version(), v1.parent_version()), (1, None));
    assert!(
        v1.manifest_url()
            .as_str()
            .ends_with(&format!("{}__v1.version.json", table.table_name)),
        "manifest path embeds the version: {}",
        v1.manifest_url()
    );

    // A segment appended under version 1 is stamped 1, excluded from the
    // base set, and listed under its version.
    let seg = v1
        .append_segment(&built_index(
            &[("k", [1.0, 0.0, 0.0, 0.0])],
            StoragePrecision::F32,
        ))
        .await
        .unwrap();
    let all = catalog
        .list_index_segments(&table.table_name)
        .await
        .unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].segment_id, seg.0);
    assert_eq!(all[0].version, Some(1));
    assert!(catalog
        .list_base_index_segments(&table.table_name)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        catalog
            .list_index_segments_for_version(&table.table_name, 1)
            .await
            .unwrap()
            .len(),
        1
    );
    let seg_usearch = jammi_test_utils::url_to_path(&all[0].index_path).with_extension("usearch");
    assert!(seg_usearch.exists());

    // Fail 1 (abort): the row is `failed`, its bundle and segment row reaped,
    // the number kept; the next allocation is 2.
    v1.abort().await.unwrap();
    let row1 = catalog
        .get_result_table_version(&table.table_name, 1)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(row1.status, "failed");
    assert!(
        !seg_usearch.exists(),
        "the aborted version's bundle is reaped"
    );
    assert!(catalog
        .list_index_segments(&table.table_name)
        .await
        .unwrap()
        .is_empty());
    let v2 = store.allocate_version(&table).await.unwrap();
    assert_eq!(v2.version(), 2);
    let after = catalog
        .get_result_table(&table.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after.next_version, 3);

    // Expiring the failed row never lowers `next_version`.
    assert!(catalog
        .delete_result_table_version(&table.table_name, 1)
        .await
        .unwrap());
    let after = catalog
        .get_result_table(&table.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after.next_version, 3);
    assert_eq!(
        catalog
            .list_result_table_versions(&table.table_name)
            .await
            .unwrap()
            .iter()
            .map(|v| v.version)
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    v0.detach();
    v2.detach();

    // The base publish is a CAS on `current_version IS NULL AND next_version
    // = $B`: on a fresh table it takes B = 0 and lands the ready row.
    let fresh = ready_table(&store).await;
    catalog
        .publish_base_version(&fresh.table_name, 0, "mem://base.version.json", "id0", 7)
        .await
        .unwrap();
    let fresh_row = catalog
        .get_result_table(&fresh.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (fresh_row.current_version, fresh_row.next_version),
        (Some(0), 1)
    );
    let base = catalog
        .get_result_table_version(&fresh.table_name, 0)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (base.status.as_str(), base.live_rows, base.masked_rows),
        ("ready", Some(7), Some(0))
    );
    // A second base publish misses the CAS (current_version is no longer
    // NULL): the ready row's `current_version` disagrees with the expected
    // parent (`None`), the same lost-race shape `ParentMoved` names for the
    // allocation and publish-table-row misses below — ONE classifier for
    // every ready-row parent mismatch, never `CasFailed`'s "left building".
    let miss = catalog
        .publish_base_version(&fresh.table_name, 1, "mem://x", "id1", 7)
        .await
        .expect_err("a second base publish must miss");
    assert!(
        matches!(
            miss,
            jammi_db::error::JammiError::ParentMoved {
                expected: None,
                found: Some(0),
                ..
            }
        ),
        "{miss:?}"
    );

    // publish_version: the delta allocated on parent 0 swaps current_version
    // to 1; a second allocation whose parent is 0 then misses at publish.
    let d1 = store.allocate_version(&fresh_row).await.unwrap();
    assert_eq!((d1.version(), d1.parent_version()), (1, Some(0)));
    let d2 = store.allocate_version(&fresh_row).await.unwrap();
    assert_eq!((d2.version(), d2.parent_version()), (2, Some(0)));
    let cas1 = VersionCas::writer(&fresh.table_name, 1, store.writer_id(), None);
    catalog
        .publish_version(PublishVersion {
            cas: &cas1,
            lease: std::time::Duration::from_secs(30),
            parent: Some(0),
            identity: "id-v1",
            live_rows: 8,
            masked_rows: 1,
            anchors_json: "[]",
        })
        .await
        .unwrap();
    let cas2 = VersionCas::writer(&fresh.table_name, 2, store.writer_id(), None);
    let miss = catalog
        .publish_version(PublishVersion {
            cas: &cas2,
            lease: std::time::Duration::from_secs(30),
            parent: Some(0),
            identity: "id-v2",
            live_rows: 8,
            masked_rows: 1,
            anchors_json: "[]",
        })
        .await
        .expect_err("the second publisher on a stale parent must miss");
    assert!(
        matches!(
            miss,
            jammi_db::error::JammiError::ParentMoved {
                expected: Some(0),
                found: Some(1),
                ..
            }
        ),
        "{miss:?}"
    );
    let row = catalog
        .get_result_table(&fresh.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (row.current_version, row.row_count, row.next_version),
        (Some(1), 8, 3)
    );
    let v2row = catalog
        .get_result_table_version(&fresh.table_name, 2)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        v2row.status, "building",
        "a missed publish rolls back the whole transaction"
    );
    d1.detach();
    d2.detach();
}

// O1 (DELTA fix round 1, audit a25424e2aa5e91337 F1) — the serialized
// interleaving the pre-fix §6.7 oracle could never build: A publishes
// FULLY (allocates AND publishes) from parent P, THEN B — still holding
// its OWN read of P from before A's publish — attempts to allocate. Before
// this fix, `allocate_result_table_version` re-read `current_version`
// itself and handed it back as B's parent, so B's allocation silently
// SUCCEEDED against A's new current_version and B's manifest would later
// disagree with its own `parent_version` (K7 broken). The fix pins the
// allocating UPDATE's WHERE clause to the caller's `expected_parent`: B's
// allocation now refuses typed (`ParentMoved`) BEFORE `next_version`
// increments, so B's stale attempt consumes no version number and inserts
// no `building` row (no manifest, no fragment ever gets a chance to be
// written for it). `refresh_embeddings` and `compact_embeddings` both
// allocate through this exact `ResultStore::allocate_version` /
// `Catalog::allocate_result_table_version` pair, so this one test covers
// the allocation-time guard both actuators share.
#[cfg_attr(test, test_case(BackendKind::Sqlite ; "sqlite"))]
#[cfg_attr(
    all(test, feature = "live-postgres-tests"),
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn allocation_refuses_when_the_parent_moved(kind: BackendKind) {
    use jammi_db::catalog::version_repo::{PublishVersion, VersionCas};

    let dir = tempdir().unwrap();
    let Some(backend) = open_backend(kind, dir.path()).await else {
        eprintln!("skipping {kind:?}: JAMMI_TEST_PG_URL unset");
        require_live_pg("allocation_refuses_when_the_parent_moved");
        return;
    };
    let catalog = fresh_catalog(backend).await;
    let store = store(dir.path(), Arc::clone(&catalog), StoragePrecision::F32);
    let table = ready_table(&store).await;

    // A publishes the base FULLY.
    catalog
        .publish_base_version(&table.table_name, 0, "mem://base.version.json", "id0", 5)
        .await
        .unwrap();

    // B's own read of the parent, captured BEFORE A's next (delta) publish:
    // `current_version = Some(0)`.
    let record_b = catalog
        .get_result_table(&table.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record_b.current_version, Some(0));

    // A now publishes FULLY again — a refresh from parent 0 to version 1.
    let a_version = store.allocate_version(&record_b).await.unwrap();
    assert_eq!(
        (a_version.version(), a_version.parent_version()),
        (1, Some(0))
    );
    let cas_a = VersionCas::writer(&table.table_name, 1, store.writer_id(), None);
    catalog
        .publish_version(PublishVersion {
            cas: &cas_a,
            lease: std::time::Duration::from_secs(30),
            parent: Some(0),
            identity: "id-a1",
            live_rows: 5,
            masked_rows: 0,
            anchors_json: "[]",
        })
        .await
        .unwrap();
    a_version.detach();
    let after_a = catalog
        .get_result_table(&table.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after_a.current_version, Some(1), "A fully published");
    let next_before = after_a.next_version;

    // B, STILL holding `record_b` (current_version = Some(0)), now attempts
    // to allocate — AFTER A has fully published. The allocation must refuse
    // typed, BEFORE `next_version` increments.
    let miss = store
        .allocate_version(&record_b)
        .await
        .expect_err("B's allocation from the older parent must refuse");
    assert!(
        matches!(
            miss,
            jammi_db::error::JammiError::ParentMoved {
                expected: Some(0),
                found: Some(1),
                ..
            }
        ),
        "{miss:?}"
    );

    // No version number was consumed and no `building` row was inserted for
    // B's refused attempt.
    let after_miss = catalog
        .get_result_table(&table.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        after_miss.next_version, next_before,
        "a refused allocation must not burn a version number"
    );
    assert!(
        catalog
            .get_result_table_version(&table.table_name, next_before)
            .await
            .unwrap()
            .is_none(),
        "a refused allocation must insert no building row"
    );

    // A re-derived B — reading the CURRENT parent — allocates and publishes
    // normally.
    let record_b2 = catalog
        .get_result_table(&table.table_name)
        .await
        .unwrap()
        .unwrap();
    let b2_version = store.allocate_version(&record_b2).await.unwrap();
    assert_eq!(
        (b2_version.version(), b2_version.parent_version()),
        (next_before, Some(1))
    );
    let cas_b2 = VersionCas::writer(&table.table_name, next_before, store.writer_id(), None);
    catalog
        .publish_version(PublishVersion {
            cas: &cas_b2,
            lease: std::time::Duration::from_secs(30),
            parent: Some(1),
            identity: "id-b2",
            live_rows: 5,
            masked_rows: 0,
            anchors_json: "[]",
        })
        .await
        .unwrap();
    b2_version.detach();
    let final_row = catalog
        .get_result_table(&table.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(final_row.current_version, Some(next_before));
}

// §6.14 — shard-ordering property: segments appended to one building version
// in either order yield an identical masked merge; masking depends only on
// the version stamps, never on segment id order.
#[tokio::test]
async fn masked_merge_is_independent_of_segment_id_order() {
    use jammi_db::index::segment::{SegmentId, SegmentedIndex};
    use jammi_db::store::deletes::DeletionMask;

    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = store(dir.path(), Arc::clone(&catalog), StoragePrecision::F32);
    let shard_a = [("a1", [1.0, 0.0, 0.0, 0.0]), ("k", [0.9, 0.1, 0.0, 0.0])];
    let shard_b = [("b1", [0.0, 1.0, 0.0, 0.0]), ("b2", [0.0, 0.0, 1.0, 0.0])];
    let base = [("k", [0.0, 0.0, 0.0, 1.0]), ("z", [0.5, 0.5, 0.0, 0.0])];
    let query = [0.9, 0.1, 0.0, 0.0];

    let mut results = Vec::new();
    for order in [[&shard_a[..], &shard_b[..]], [&shard_b[..], &shard_a[..]]] {
        let table = ready_table(&store).await;
        // The base is stamped with its own (earlier) version number; the
        // shards land in the next one.
        let base_version = store.allocate_version(&table).await.unwrap();
        let base_stamp = base_version.version();
        base_version.detach();
        let version = store.allocate_version(&table).await.unwrap();
        assert!(version.version() > base_stamp);
        let mut loaded = vec![(
            SegmentId(0),
            base_stamp,
            built_index(&base, StoragePrecision::F32),
        )];
        for shard in order {
            let seg = version
                .append_segment(&built_index(shard, StoragePrecision::F32))
                .await
                .unwrap();
            loaded.push((
                seg,
                version.version(),
                built_index(shard, StoragePrecision::F32),
            ));
        }
        // The base K is superseded by the shard's K: mask `(k, base_stamp)`.
        let mask = Arc::new(DeletionMask::from_entries([("k".to_string(), base_stamp)]));
        let index = SegmentedIndex::new_masked(loaded, mask).unwrap();
        let hits = index.search_final(&query, 6, 4).unwrap();
        assert_eq!(hits[0].0, "k");
        assert!(
            hits[0].1 < 1e-4,
            "the shard's K, never the masked base K: {hits:?}"
        );
        results.push(hits);
        version.detach();
    }
    assert_eq!(
        results[0], results[1],
        "the merge is independent of segment id order"
    );
}
