//! Phase 3 integration tests — tenant-scoped sessions deliver disjoint
//! views of mutable companion tables. Engine-only scope (no wire-surface
//! tests; those land with the ADR-01 substrate PR).

use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::JammiSession;
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::TenantId;
use jammi_test_utils::{make_test_session, unique_suffix};
use tempfile::tempdir;
use test_case::test_case;
use uuid::Uuid;

use crate::common;

/// Fetch a backend-parameterized session, skipping the test (with a warning,
/// never `#[ignore]`) when the Postgres arm has no `JAMMI_TEST_PG_URL`.
macro_rules! session_or_skip {
    ($backend:expr, $dir:expr) => {
        match make_test_session($backend, $dir.path()).await {
            Some(s) => s,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

fn widget_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

/// A fresh, well-formed, per-test tenant id — a random UUID, never a fixed
/// literal. The Postgres lane runs the whole matrix against one shared
/// database, and the `sources` / mutable-table-registry catalog rows are
/// global (not scoped to a per-test container the way a uniquely-named
/// mutable table is), so a fixed tenant literal shared across sibling tests
/// (or repeated runs) would accumulate rows in that tenant's read-scope and
/// break exact-list assertions below.
fn fresh_tenant() -> TenantId {
    TenantId::from_uuid(Uuid::new_v4()).unwrap()
}

/// Register a mutable companion table named `table_id`, keyed on `id`. Every
/// test picks its own unique `table_id` (via [`unique_suffix`]) so sibling
/// tests sharing the Postgres lane's one database never collide on the same
/// backing table.
async fn register_widgets(session: &JammiSession, table_id: &str) {
    let def =
        MutableTableDefinitionBuilder::new(MutableTableId::new(table_id).unwrap(), widget_schema())
            .primary_key(vec!["id".into()])
            .build()
            .unwrap();
    session.create_mutable_table(def).await.unwrap();
}

/// Two sessions in the same process with different tenant bindings see
/// disjoint row sets through the same mutable table.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn two_tenants_see_disjoint_rows(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();

    let session_a = session_or_skip!(backend, dir);
    register_widgets(&session_a, &widgets).await;
    let session_a = session_a.with_tenant(tenant_a);
    session_a
        .sql(&format!(
            "INSERT INTO mutable.public.{widgets} (id, name) VALUES (1, 'alpha')"
        ))
        .await
        .unwrap();

    let session_b = session_or_skip!(backend, dir).with_tenant(tenant_b);
    session_b
        .sql(&format!(
            "INSERT INTO mutable.public.{widgets} (id, name) VALUES (2, 'beta')"
        ))
        .await
        .unwrap();

    let rows_a = session_a
        .sql(&format!(
            "SELECT id, name FROM mutable.public.{widgets} ORDER BY id"
        ))
        .await
        .unwrap();
    let batch_a = arrow::compute::concat_batches(&rows_a[0].schema(), &rows_a).unwrap();
    let ids_a = batch_a
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids_a.len(), 1, "session_a should see only its own row");
    assert_eq!(ids_a.value(0), 1);

    let rows_b = session_b
        .sql(&format!(
            "SELECT id, name FROM mutable.public.{widgets} ORDER BY id"
        ))
        .await
        .unwrap();
    let batch_b = arrow::compute::concat_batches(&rows_b[0].schema(), &rows_b).unwrap();
    let ids_b = batch_b
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids_b.len(), 1);
    assert_eq!(ids_b.value(0), 2);
}

/// An `Unscoped` session sees only rows whose `tenant_id IS NULL`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn unscoped_session_sees_only_global_rows(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();

    // Unscoped session writes one row → tenant_id NULL.
    let session = session_or_skip!(backend, dir);
    register_widgets(&session, &widgets).await;
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{widgets} (id, name) VALUES (10, 'global')"
        ))
        .await
        .unwrap();

    // Scoped session writes one row → tenant_id = A.
    let session_a = session_or_skip!(backend, dir).with_tenant(tenant_a);
    session_a
        .sql(&format!(
            "INSERT INTO mutable.public.{widgets} (id, name) VALUES (20, 'a-only')"
        ))
        .await
        .unwrap();

    // A fresh Unscoped session should see only the global row.
    let session_unscoped = session_or_skip!(backend, dir);
    let rows = session_unscoped
        .sql(&format!(
            "SELECT id, name FROM mutable.public.{widgets} ORDER BY id"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let names = batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.len(), 1);
    assert_eq!(names.value(0), "global");
}

/// A scoped session sees its own rows plus globally-scoped rows
/// (`tenant_id IS NULL`).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn scoped_session_sees_own_plus_global(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();

    let session = session_or_skip!(backend, dir);
    register_widgets(&session, &widgets).await;
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{widgets} (id, name) VALUES (100, 'global')"
        ))
        .await
        .unwrap();

    let session_a = session_or_skip!(backend, dir).with_tenant(tenant_a);
    session_a
        .sql(&format!(
            "INSERT INTO mutable.public.{widgets} (id, name) VALUES (200, 'a')"
        ))
        .await
        .unwrap();

    let rows = session_a
        .sql(&format!(
            "SELECT id FROM mutable.public.{widgets} ORDER BY id"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let values: Vec<i64> = (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(values, vec![100, 200]);
}

/// Tenant binding persists across multiple queries on the same session.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn tenant_binding_is_sticky_across_queries(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();

    let session = session_or_skip!(backend, dir);
    register_widgets(&session, &widgets).await;
    let session = session.with_tenant(tenant_a);

    session
        .sql(&format!(
            "INSERT INTO mutable.public.{widgets} (id, name) VALUES (1, 'a1'), (2, 'a2')"
        ))
        .await
        .unwrap();

    let rows = session
        .sql(&format!(
            "SELECT COUNT(*) AS n FROM mutable.public.{widgets}"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let n = batch
        .column_by_name("n")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(n.value(0), 2);

    // Same session, second query: also tenant-scoped.
    let rows = session
        .sql(&format!(
            "SELECT name FROM mutable.public.{widgets} ORDER BY name"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let names = batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "a1");
    assert_eq!(names.value(1), "a2");
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn with_tenant_returns_same_session_id(backend: BackendKind) {
    // with_tenant is a builder that returns Self — it does not require a
    // SessionContext rebuild.
    let dir = tempdir().unwrap();
    let tenant_a = fresh_tenant();
    let session = session_or_skip!(backend, dir);
    assert!(session.tenant().is_none());

    let session = session.with_tenant(tenant_a);
    assert_eq!(session.tenant(), Some(tenant_a));
}

/// Two scoped sessions writing through the same `Catalog::register_source`
/// path see disjoint `list_sources` results — the tenant filter on read +
/// the tenant binding on write together enforce isolation.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn catalog_sources_isolated_by_tenant(backend: BackendKind) {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let dir = tempdir().unwrap();
    let suffix = unique_suffix();
    let src_a = format!("src_a_{suffix}");
    let src_b = format!("src_b_{suffix}");
    // Fresh per-test tenants: `sources` is a catalog-wide table, not scoped
    // to a per-test container, so a fixed tenant literal shared with sibling
    // tests would accumulate rows in this tenant's read-scope.
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();

    let session_a = session_or_skip!(backend, dir).with_tenant(tenant_a);
    session_a
        .add_source(
            &src_a,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let session_b = session_or_skip!(backend, dir).with_tenant(tenant_b);
    session_b
        .add_source(
            &src_b,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // `list_sources` for a scoped tenant also returns every globally-scoped
    // (`tenant_id IS NULL`) row — by design (a scoped session sees its own
    // rows plus global rows). On the shared Postgres lane that global pool
    // accumulates rows from every OTHER test in the suite that registered an
    // unscoped source, so filter down to this test's own unique-suffixed
    // names before asserting the set is exactly `{src_a}` / `{src_b}`.
    let sources_a = session_a.catalog().list_sources().await.unwrap();
    let ids_a: Vec<&str> = sources_a
        .iter()
        .map(|s| s.source_id.as_str())
        .filter(|id| id.ends_with(&suffix))
        .collect();
    assert_eq!(ids_a, vec![src_a.as_str()]);

    let sources_b = session_b.catalog().list_sources().await.unwrap();
    let ids_b: Vec<&str> = sources_b
        .iter()
        .map(|s| s.source_id.as_str())
        .filter(|id| id.ends_with(&suffix))
        .collect();
    assert_eq!(ids_b, vec![src_b.as_str()]);
}

/// `list_all_sources` enumerates sources across every tenant, while the
/// tenant-scoped `list_sources` stays filtered to its own rows plus the
/// globally-scoped (`tenant_id IS NULL`) rows. Session startup re-hydrates
/// source providers through the cross-tenant view so a worker that later
/// binds to any tenant can resolve that tenant's private sources.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn catalog_list_all_sources_sees_across_tenants(backend: BackendKind) {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let dir = tempdir().unwrap();
    let suffix = unique_suffix();
    let global_src = format!("global_src_{suffix}");
    let src_a = format!("src_a_{suffix}");
    let src_b = format!("src_b_{suffix}");
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();

    let parquet = || SourceConnection {
        url: Some(common::fixture_url("patents.parquet")),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    };

    // One global source plus one private source per tenant.
    let unscoped = session_or_skip!(backend, dir);
    unscoped
        .add_source(&global_src, SourceType::File, parquet())
        .await
        .unwrap();

    let session_a = session_or_skip!(backend, dir).with_tenant(tenant_a);
    session_a
        .add_source(&src_a, SourceType::File, parquet())
        .await
        .unwrap();

    let session_b = session_or_skip!(backend, dir).with_tenant(tenant_b);
    session_b
        .add_source(&src_b, SourceType::File, parquet())
        .await
        .unwrap();

    // Cross-tenant enumeration sees every source this test created (and,
    // on the shared Postgres lane, possibly siblings' — filter down to this
    // test's own unique-suffixed names before asserting the set).
    // Sort for a registration-order-independent set comparison: the catalog
    // orders by `created_at`, which ties across sub-millisecond inserts.
    let mut all: Vec<String> = session_or_skip!(backend, dir)
        .catalog()
        .list_all_sources()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.source_id)
        .filter(|id| id.ends_with(&suffix))
        .collect();
    all.sort();
    let mut expected = vec![global_src.clone(), src_a.clone(), src_b.clone()];
    expected.sort();
    assert_eq!(all, expected);

    // The tenant-scoped API stays filtered to tenant A's own + global rows.
    let mut scoped_a: Vec<String> = session_a
        .catalog()
        .list_sources()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.source_id)
        .filter(|id| id.ends_with(&suffix))
        .collect();
    scoped_a.sort();
    assert_eq!(scoped_a, vec![global_src, src_a]);
}

/// An unscoped session sees globally-scoped (NULL) rows; a scoped session
/// sees its own rows plus the NULL rows (consistent with the read-side
/// predicate-injection rule).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn catalog_unscoped_session_sees_global_only_after_scoped_writes(backend: BackendKind) {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let dir = tempdir().unwrap();
    let suffix = unique_suffix();
    let global_src = format!("global_src_{suffix}");
    let tenant_a_src = format!("tenant_a_src_{suffix}");
    let tenant_a = fresh_tenant();

    let unscoped = session_or_skip!(backend, dir);
    unscoped
        .add_source(
            &global_src,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let scoped = session_or_skip!(backend, dir).with_tenant(tenant_a);
    scoped
        .add_source(
            &tenant_a_src,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // A fresh unscoped session sees only the global row (filtered to this
    // test's own unique-suffixed names — the shared Postgres lane's global
    // pool may carry other tests' NULL-tenant rows too).
    let fresh_unscoped = session_or_skip!(backend, dir);
    let ids: Vec<String> = fresh_unscoped
        .catalog()
        .list_sources()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.source_id)
        .filter(|id| id.ends_with(&suffix))
        .collect();
    assert_eq!(ids, vec![global_src]);
}

/// SPEC-03 §12 #2 — one federated source carries a `tenant_id` column;
/// the analyzer rule injects a per-session filter that yields 6 rows for
/// tenant A and 4 rows for tenant B on the same on-disk Parquet table.
/// Verifies the read-side predicate-injection path end-to-end against a
/// local Parquet source, not just the engine-internal mutable table tested
/// elsewhere in this file.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn federated_source_tenant_column_filters_split_6_4(backend: BackendKind) {
    use arrow::array::{ArrayRef, RecordBatch};
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let dir = tempdir().unwrap();
    let pq_path = dir.path().join("notes.parquet");
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();
    let notes_src = format!("notes_{}", unique_suffix());

    // Build a 10-row batch: 6 for tenant A, 4 for tenant B.
    let schema = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("tenant_id", DataType::Utf8, true),
    ]));
    let note_ids = Int64Array::from((0..10_i64).collect::<Vec<_>>());
    let a_str = tenant_a.to_string();
    let b_str = tenant_b.to_string();
    let tenant_col: Vec<&str> = (0..10)
        .map(|i| {
            if i < 6 {
                a_str.as_str()
            } else {
                b_str.as_str()
            }
        })
        .collect();
    let tenant_col = StringArray::from(tenant_col);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(note_ids) as ArrayRef,
            Arc::new(tenant_col) as ArrayRef,
        ],
    )
    .unwrap();
    let file = std::fs::File::create(&pq_path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let url = format!("file://{}", pq_path.display());

    // Register the source ONCE — unscoped (tenant_id NULL on the catalog
    // row) — so both per-tenant sessions read it from the catalog on
    // reload. SPEC-03 §12 #2 calls for "one source registration, one
    // connection pool, no per-tenant table".
    {
        let registrar = session_or_skip!(backend, dir);
        registrar
            .add_source(
                &notes_src,
                SourceType::File,
                SourceConnection {
                    url: Some(url),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    // Session A: bind tenant A and declare the federated source's tenant
    // discriminator column. The source row in the catalog is `tenant_id
    // NULL`, so both per-tenant sessions can see it via the read-side
    // predicate (`tenant_id = $bound OR tenant_id IS NULL`).
    let session_a = session_or_skip!(backend, dir).with_tenant(tenant_a);
    session_a.set_source_tenant_column(&notes_src, Some("tenant_id".into()));

    let session_b = session_or_skip!(backend, dir).with_tenant(tenant_b);
    session_b.set_source_tenant_column(&notes_src, Some("tenant_id".into()));

    async fn count_for(session: &JammiSession, notes_src: &str) -> i64 {
        let rows = session
            .sql(&format!(
                "SELECT COUNT(*) AS n FROM {notes_src}.public.notes"
            ))
            .await
            .unwrap();
        let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
        batch
            .column_by_name("n")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    let n_a = count_for(&session_a, &notes_src).await;
    let n_b = count_for(&session_b, &notes_src).await;
    assert_eq!(n_a, 6, "session A must see exactly its 6 rows");
    assert_eq!(n_b, 4, "session B must see exactly its 4 rows");

    async fn collect_ids(session: &JammiSession, notes_src: &str) -> Vec<i64> {
        let rows = session
            .sql(&format!(
                "SELECT note_id FROM {notes_src}.public.notes ORDER BY note_id"
            ))
            .await
            .unwrap();
        let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
        let col = batch
            .column_by_name("note_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        (0..col.len()).map(|i| col.value(i)).collect::<Vec<i64>>()
    }
    let ids_a = collect_ids(&session_a, &notes_src).await;
    let ids_b = collect_ids(&session_b, &notes_src).await;
    let intersection: Vec<i64> = ids_a
        .iter()
        .copied()
        .filter(|id| ids_b.contains(id))
        .collect();
    assert!(
        intersection.is_empty(),
        "tenant A ids ({ids_a:?}) and B ids ({ids_b:?}) must be disjoint"
    );
}

/// A source's tenant discriminator persists in the catalog and is replayed on
/// reload. The `SourceTenantColumns` lookup is process-memory only, so without
/// persistence a federated source registered with a `tenant_column` would
/// reload after a restart with no scope — a latent cross-tenant read. Here the
/// column is set on the `SourceConnection` at registration (never via the
/// in-process setter), the session is dropped and rebuilt against the same
/// catalog DB, and the rebuilt session must replay the discriminator and emit
/// the scoping filter. A source registered with no discriminator reloads as
/// `None`, emitting no filter.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn source_tenant_column_persists_and_replays_on_reload(backend: BackendKind) {
    use arrow::array::{ArrayRef, RecordBatch};
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let dir = tempdir().unwrap();
    let suffix = unique_suffix();
    let notes_src = format!("notes_{suffix}");
    let docs_src = format!("public_docs_{suffix}");
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();

    // Two parquet files: `notes` carries tenancy under `customer_id`
    // (6 rows tenant A, 4 rows tenant B); `public_docs` carries no tenant
    // discriminator at all.
    let notes_path = dir.path().join("notes.parquet");
    let docs_path = dir.path().join("public_docs.parquet");

    let notes_schema = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("customer_id", DataType::Utf8, true),
    ]));
    let a_str = tenant_a.to_string();
    let b_str = tenant_b.to_string();
    let customer_col: Vec<&str> = (0..10)
        .map(|i| {
            if i < 6 {
                a_str.as_str()
            } else {
                b_str.as_str()
            }
        })
        .collect();
    let notes_batch = RecordBatch::try_new(
        Arc::clone(&notes_schema),
        vec![
            Arc::new(Int64Array::from((0..10_i64).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(StringArray::from(customer_col)) as ArrayRef,
        ],
    )
    .unwrap();
    {
        let file = std::fs::File::create(&notes_path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            Arc::clone(&notes_schema),
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        writer.write(&notes_batch).unwrap();
        writer.close().unwrap();
    }

    let docs_schema = Arc::new(Schema::new(vec![Field::new(
        "doc_id",
        DataType::Int64,
        false,
    )]));
    let docs_batch = RecordBatch::try_new(
        Arc::clone(&docs_schema),
        vec![Arc::new(Int64Array::from((0..5_i64).collect::<Vec<_>>())) as ArrayRef],
    )
    .unwrap();
    {
        let file = std::fs::File::create(&docs_path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            Arc::clone(&docs_schema),
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        writer.write(&docs_batch).unwrap();
        writer.close().unwrap();
    }

    // Register both sources against the catalog, then drop the session. The
    // discriminator is carried on the connection — never via
    // `set_source_tenant_column` — so the persist path is what's exercised.
    {
        let registrar = session_or_skip!(backend, dir);
        registrar
            .add_source(
                &notes_src,
                SourceType::File,
                SourceConnection {
                    url: Some(format!("file://{}", notes_path.display())),
                    format: Some(FileFormat::Parquet),
                    tenant_column: Some("customer_id".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        registrar
            .add_source(
                &docs_src,
                SourceType::File,
                SourceConnection {
                    url: Some(format!("file://{}", docs_path.display())),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    // Rebuild a fresh session against the SAME catalog DB. `reload_sources`
    // runs at construction and must replay the persisted discriminator — no
    // `set_source_tenant_column` call here.
    let session_a = session_or_skip!(backend, dir).with_tenant(tenant_a);
    let session_b = session_or_skip!(backend, dir).with_tenant(tenant_b);

    async fn count(session: &JammiSession, sql: &str) -> i64 {
        let rows = session.sql(sql).await.unwrap();
        let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    // The replayed discriminator scopes `notes`: A sees its 6, B its 4.
    assert_eq!(
        count(
            &session_a,
            &format!("SELECT COUNT(*) FROM {notes_src}.public.notes")
        )
        .await,
        6,
        "tenant A must see its 6 rows via the replayed customer_id filter"
    );
    assert_eq!(
        count(
            &session_b,
            &format!("SELECT COUNT(*) FROM {notes_src}.public.notes")
        )
        .await,
        4,
        "tenant B must see its 4 rows via the replayed customer_id filter"
    );

    // `public_docs` has no discriminator: every tenant sees all 5 rows, so the
    // reload replayed `None` and injected no spurious filter. (A wrongly
    // replayed filter would drop rows or fail to plan against the absent
    // column.)
    assert_eq!(
        count(
            &session_a,
            &format!("SELECT COUNT(*) FROM {docs_src}.public.public_docs")
        )
        .await,
        5,
        "an un-scoped source must reload as None — no spurious filter"
    );
    assert_eq!(
        count(
            &session_b,
            &format!("SELECT COUNT(*) FROM {docs_src}.public.public_docs")
        )
        .await,
        5,
        "the un-scoped source is visible in full to every tenant"
    );
}

/// `JammiSession::with_tenant_scoped` installs a Tokio task-local for the
/// duration of the closure that shadows the session's sticky shared
/// binding. Two concurrent tasks invoking the helper with different
/// tenants on the *same* `Arc<JammiSession>` each see their own tenant
/// inside the closure — no cross-pollution from the other task's binding.
///
/// This is the concurrency property that the helper exists for. Without
/// it, two gRPC handlers from different tenants sharing one
/// `Arc<JammiSession>` would race on the shared `Arc<RwLock<TenantContext>>`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_tenant_scoped_isolates_concurrent_tasks(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();

    let session = Arc::new(session_or_skip!(backend, dir));
    register_widgets(&session, &widgets).await;

    // Each task: enter its tenant's scope, insert a row tagged with its
    // tenant name, count rows visible inside the scope, and snapshot
    // `session.tenant()` from inside the closure. Both tasks run on the
    // same `Arc<JammiSession>` concurrently on a multi-thread runtime.
    async fn run_one(
        session: Arc<JammiSession>,
        tenant: TenantId,
        row_id: i64,
        row_name: &'static str,
        widgets: String,
    ) -> (Option<TenantId>, i64) {
        session
            .with_tenant_scoped(tenant, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.{widgets} (id, name) VALUES ({row_id}, '{row_name}')"
                    ))
                    .await
                    .unwrap();
                let observed = scope.tenant();
                let rows = scope
                    .sql(&format!("SELECT id FROM mutable.public.{widgets}"))
                    .await
                    .unwrap();
                let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
                let ids = batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let visible_id = ids.value(0);
                (observed, visible_id)
            })
            .await
    }

    // Launch many concurrent invocations on both tenants. The high
    // iteration count exists to stress the race window: without the
    // task-local override, each task would intermittently observe the
    // other tenant's binding around the await point inside the helper.
    let mut handles = Vec::new();
    for i in 0..16 {
        let session_a = Arc::clone(&session);
        let session_b = Arc::clone(&session);
        let id_a = (i * 2) + 1000;
        let id_b = (i * 2) + 1001;
        let widgets_a = widgets.clone();
        let widgets_b = widgets.clone();
        handles.push(tokio::spawn(async move {
            run_one(session_a, tenant_a, id_a, "alpha", widgets_a).await
        }));
        handles.push(tokio::spawn(async move {
            run_one(session_b, tenant_b, id_b, "beta", widgets_b).await
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let (observed, visible_id) = h.await.unwrap();
        let task_is_a = i % 2 == 0;
        let expected_tenant = if task_is_a { tenant_a } else { tenant_b };
        assert_eq!(
            observed,
            Some(expected_tenant),
            "task {i} observed wrong tenant inside scope",
        );
        // Every visible row id must belong to this task's tenant — the
        // session-internal mutable-table scan applies the tenant filter
        // based on the task-local override. Tenant A's task IDs are even
        // offsets from 1000; tenant B's are odd.
        let is_a_id = (visible_id - 1000) % 2 == 0;
        assert_eq!(
            is_a_id, task_is_a,
            "task {i} saw a row ({visible_id}) belonging to the other tenant",
        );
    }
}

/// The task-local override installed by `with_tenant_scoped` does not
/// mutate the session's sticky shared binding. After the closure
/// returns, `session.tenant()` reflects whatever the sticky binding was
/// before the scoped call — not the scope's tenant.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn with_tenant_scoped_does_not_mutate_sticky_binding(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();

    let session = session_or_skip!(backend, dir);
    register_widgets(&session, &widgets).await;
    // Sticky-bind tenant_b so we can observe that the scoped call to
    // tenant_a does not leak past the closure.
    let session = session.with_tenant(tenant_b);
    let session = Arc::new(session);

    assert_eq!(session.tenant(), Some(tenant_b));
    let observed_inside = session
        .with_tenant_scoped(tenant_a, |scope| async move { scope.tenant() })
        .await;
    assert_eq!(observed_inside, Some(tenant_a));
    // After the scope exits, the sticky binding (tenant_b) is restored
    // because it was never touched.
    assert_eq!(session.tenant(), Some(tenant_b));
}

/// Headline safety property for [`jammi_db::trigger::Subscriber::subscribe_scoped`].
///
/// A `gRPC` server-streaming handler enters `with_tenant_scoped(A)`, opens
/// a subscription, returns the stream to tonic, and the closure resolves —
/// the surrounding task-local binding clears the instant the closure
/// returns. Tonic then polls the stream from a task that has no tenant
/// binding of its own (no `with_tenant_scoped` wrapping its poll loop).
///
/// `subscribe_scoped` resolves the tenant filter at subscribe time, before
/// returning the stream, so the replay rows materialised into the stream
/// are filtered to the caller-supplied tenant regardless of what
/// `current_tenant()` reads at poll time. This test directly populates the
/// backing table with rows for two tenants, subscribes for tenant A from a
/// task with no binding, and verifies the polled output never includes
/// tenant B's rows.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_scoped_stream_remains_tenant_filtered_after_closure_returns(
    backend: BackendKind,
) {
    use arrow::array::RecordBatch;
    use futures::StreamExt;
    use jammi_db::catalog::backend::TxOptions;
    use jammi_db::catalog::topic_repo::TopicRepo;
    use jammi_db::source::mutable::MutableTableRegistry;
    use jammi_db::trigger::{
        InMemoryBroker, Offset, Predicate, Subscriber, TopicDefinition, TopicId, TriggerBroker,
    };
    use std::collections::BTreeMap;

    let dir = tempdir().unwrap();
    let session = Arc::new(session_or_skip!(backend, dir));
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();
    let topic_name = format!("global.events.{}", unique_suffix());

    // Build a global (unscoped) topic so both tenants can write to the
    // same backing table. The leak the PR fixes is on the read side; the
    // backing-table population happens by hand below so the test does not
    // depend on the publisher's tenant propagation path.
    let topic_schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let topic = TopicDefinition {
        id: TopicId::new(),
        name: topic_name,
        schema: Arc::clone(&topic_schema),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    broker.register_topic(&topic).await.unwrap();
    let topic_repo = TopicRepo::new(Arc::clone(session.catalog()), session.mutable_tables_arc());
    topic_repo.register_topic(&topic).await.unwrap();

    // Build a subscriber on top of the same backing-table registry the
    // session owns so the backing-table reads see the same data as the
    // hand-rolled inserts below.
    let registry: Arc<MutableTableRegistry> = session.mutable_tables_arc();
    let subscriber = Arc::new(Subscriber::new(Arc::clone(&broker), Arc::clone(&registry)));

    // Insert two rows for tenant A and two for tenant B straight into the
    // backing table, each pair under its own transaction with the right
    // `tx.set_tenant` binding so the rows carry the matching `tenant_id`.
    let backing_id = MutableTableId::new(topic.backing_table_name()).unwrap();
    let augmented_schema = topic.backing_table_schema();
    let row_for = |offset: i64, event: i64, label: &str| -> RecordBatch {
        use arrow::array::Int64Array;
        RecordBatch::try_new(
            Arc::clone(&augmented_schema),
            vec![
                Arc::new(Int64Array::from(vec![offset])),
                Arc::new(Int64Array::from(vec![0_i64])),
                Arc::new(Int64Array::from(
                    vec![chrono::Utc::now().timestamp_micros()],
                )),
                Arc::new(Int64Array::from(vec![event])),
                Arc::new(StringArray::from(vec![label])),
            ],
        )
        .unwrap()
    };

    let backend_arc = session.catalog().backend_arc();
    let registry_for_a = Arc::clone(&registry);
    let backing_for_a = backing_id.clone();
    let a_batch_one = row_for(0, 100, "a-one");
    let a_batch_two = row_for(1, 101, "a-two");
    backend_arc
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.set_tenant(Some(tenant_a));
                registry_for_a
                    .insert_batch(tx, &backing_for_a, &a_batch_one)
                    .await
                    .map_err(|e| {
                        jammi_db::catalog::backend::BackendError::Execution(e.to_string())
                    })?;
                registry_for_a
                    .insert_batch(tx, &backing_for_a, &a_batch_two)
                    .await
                    .map_err(|e| {
                        jammi_db::catalog::backend::BackendError::Execution(e.to_string())
                    })?;
                Ok::<(), jammi_db::catalog::backend::BackendError>(())
            })
        })
        .await
        .unwrap();

    let registry_for_b = Arc::clone(&registry);
    let backing_for_b = backing_id.clone();
    let b_batch_one = row_for(2, 200, "b-one");
    let b_batch_two = row_for(3, 201, "b-two");
    backend_arc
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.set_tenant(Some(tenant_b));
                registry_for_b
                    .insert_batch(tx, &backing_for_b, &b_batch_one)
                    .await
                    .map_err(|e| {
                        jammi_db::catalog::backend::BackendError::Execution(e.to_string())
                    })?;
                registry_for_b
                    .insert_batch(tx, &backing_for_b, &b_batch_two)
                    .await
                    .map_err(|e| {
                        jammi_db::catalog::backend::BackendError::Execution(e.to_string())
                    })?;
                Ok::<(), jammi_db::catalog::backend::BackendError>(())
            })
        })
        .await
        .unwrap();

    // Enter `with_tenant_scoped(A)` to subscribe for tenant A from inside a
    // scope. The returned `Subscription` must remain safe to poll outside
    // the scope — that is the exact pattern a downstream consumer's gRPC
    // handlers want.
    let subscription = session
        .with_tenant_scoped(tenant_a, |_scope| {
            let subscriber = Arc::clone(&subscriber);
            let topic = topic.clone();
            async move {
                subscriber
                    .subscribe_scoped(
                        &topic,
                        Some(tenant_a),
                        Predicate::match_all(),
                        Some(Offset::new(0, chrono::Utc::now())),
                    )
                    .await
                    .unwrap()
            }
        })
        .await;

    // Move the subscription onto a task that has no tenant binding of its
    // own — tonic's stream poller has the same shape: no surrounding
    // `with_tenant_scoped` wraps the polls.
    let polled = tokio::spawn(async move {
        let mut stream = subscription;
        let mut events: Vec<i64> = Vec::new();
        // The replay materialises four candidate rows; only A's two pass
        // the tenant filter baked in at subscribe time. Anything beyond
        // that must come from the live broker tail, which is empty here.
        while events.len() < 2 {
            let next = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
                .await
                .expect("subscribe stream blocked");
            let delivered = next.expect("stream ended early").unwrap();
            let col = delivered
                .batch
                .column_by_name("event_id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..col.len() {
                events.push(col.value(i));
            }
        }
        // Drain a tiny window further to confirm no tenant-B rows arrive.
        let extra =
            tokio::time::timeout(std::time::Duration::from_millis(100), stream.next()).await;
        assert!(
            extra.is_err(),
            "tenant-B rows leaked into a tenant-A subscription poll loop"
        );
        events
    })
    .await
    .unwrap();

    assert_eq!(polled, vec![100, 101], "tenant-A events only");
}

/// `with_admin_scope` lifts the analyzer rule and the mutable-table
/// provider's tenant filter so a cross-tenant administrative scan can
/// enumerate rows owned by every tenant. The session is unbound (no
/// sticky tenant, no `with_tenant_scoped` wrapper) when the admin scope
/// opens; without the bypass it would see only globally-scoped
/// (`tenant_id IS NULL`) rows.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn with_admin_scope_sees_across_tenants(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();
    let tenant_c = fresh_tenant();

    // Single session, used by every tenant in sequence — exercises the
    // session-shared mutable-table registry that the leak path threads
    // through.
    let session = Arc::new(session_or_skip!(backend, dir));
    register_widgets(&session, &widgets).await;

    for (tenant, id, name) in [
        (tenant_a, 1_i64, "a-row"),
        (tenant_b, 2_i64, "b-row"),
        (tenant_c, 3_i64, "c-row"),
    ] {
        let widgets = widgets.clone();
        session
            .with_tenant_scoped(tenant, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.{widgets} (id, name) VALUES ({id}, '{name}')"
                    ))
                    .await
                    .unwrap();
            })
            .await;
    }

    // Outside an admin scope, an unbound session sees zero tenant-tagged
    // rows (only globally-scoped, `tenant_id IS NULL` rows would surface).
    let rows = session
        .sql(&format!(
            "SELECT COUNT(*) AS n FROM mutable.public.{widgets}"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let n = batch
        .column_by_name("n")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, 0, "unbound session must not see any tenant-tagged rows");

    // Inside `with_admin_scope`, every row across every tenant is visible.
    let widgets_admin = widgets.clone();
    let ids: Vec<i64> = session
        .with_admin_scope(|admin| async move {
            let rows = admin
                .sql(&format!(
                    "SELECT id FROM mutable.public.{widgets_admin} ORDER BY id"
                ))
                .await
                .unwrap();
            let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
            let col = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect()
        })
        .await;

    assert_eq!(
        ids,
        vec![1, 2, 3],
        "admin scope must surface rows from every tenant"
    );
}

/// The admin-scope task-local clears the moment the closure resolves; a
/// subsequent SQL call on the same session is tenant-filtered again. A
/// stale bypass leaking past the closure would silently widen the read
/// surface of every later query and is exactly what the closure-shaped
/// API exists to prevent.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn admin_scope_does_not_leak_into_subsequent_calls(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let widgets = format!("widgets_{}", unique_suffix());
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();

    let session = Arc::new(session_or_skip!(backend, dir));
    register_widgets(&session, &widgets).await;

    {
        let widgets = widgets.clone();
        session
            .with_tenant_scoped(tenant_a, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.{widgets} (id, name) VALUES (1, 'a')"
                    ))
                    .await
                    .unwrap();
            })
            .await;
    }
    {
        let widgets = widgets.clone();
        session
            .with_tenant_scoped(tenant_b, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.{widgets} (id, name) VALUES (2, 'b')"
                    ))
                    .await
                    .unwrap();
            })
            .await;
    }

    // Admin scope: sees both rows.
    let widgets_admin = widgets.clone();
    let cross_tenant_count = session
        .with_admin_scope(|admin| async move {
            let rows = admin
                .sql(&format!(
                    "SELECT COUNT(*) AS n FROM mutable.public.{widgets_admin}"
                ))
                .await
                .unwrap();
            let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
            batch
                .column_by_name("n")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0)
        })
        .await;
    assert_eq!(cross_tenant_count, 2);

    // After the closure resolves, the same `sql` call on the unbound
    // session is tenant-filtered again — zero rows because none of the
    // inserted rows have `tenant_id IS NULL`.
    let rows = session
        .sql(&format!(
            "SELECT COUNT(*) AS n FROM mutable.public.{widgets}"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    let n = batch
        .column_by_name("n")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        n, 0,
        "admin-scope bypass leaked past the closure into a subsequent query"
    );

    // And a tenant-scoped query after that still sees only its own row.
    let visible_b = session
        .with_tenant_scoped(tenant_b, |scope| async move {
            let rows = scope
                .sql(&format!(
                    "SELECT id FROM mutable.public.{widgets} ORDER BY id"
                ))
                .await
                .unwrap();
            let batch = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
            let col = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<i64>>()
        })
        .await;
    assert_eq!(visible_b, vec![2]);
}

/// `Transaction::assert_tenant_matches` is the defence-in-depth write-side
/// guard. The sink calls it once per write_all; verify it rejects mismatches.
///
/// SQLite-only, not parameterized over Postgres: the assertion under test
/// (`Transaction::assert_tenant_matches`) is a pure in-memory comparison on
/// the `Transaction` struct — no SQL is sent for it. The surrounding
/// `backend.transaction()` BEGIN/COMMIT dialect is already exercised on
/// Postgres by the parameterized tests above, so a Postgres arm here would
/// duplicate that coverage without adding a new determinant.
#[tokio::test]
async fn transaction_tenant_guard_rejects_mismatch() {
    use jammi_db::catalog::backend::{BackendError, TxOptions};
    use jammi_db::catalog::backend_sqlite::SqliteBackend;
    use jammi_db::catalog::Catalog;
    use jammi_db::CatalogBackend;

    let dir = tempdir().unwrap();
    let backend = SqliteBackend::open(&dir.path().join("guard.db"))
        .await
        .unwrap();
    let _catalog = Catalog::from_backend(jammi_db::BackendImpl::Sqlite(backend.clone()));

    let bound = fresh_tenant();
    let other = fresh_tenant();
    let err = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.set_tenant(Some(bound));
                tx.assert_tenant_matches(Some(other), "widgets")
            })
        })
        .await
        .unwrap_err();
    match err {
        BackendError::TenantMismatch {
            table,
            expected,
            got,
        } => {
            assert_eq!(table, "widgets");
            assert_eq!(expected, Some(bound));
            assert_eq!(got, Some(other));
        }
        other => panic!("expected TenantMismatch, got {other:?}"),
    }
}
