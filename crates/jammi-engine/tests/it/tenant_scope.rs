//! Phase 3 integration tests — tenant-scoped sessions deliver disjoint
//! views of mutable companion tables. Engine-only scope (no wire-surface
//! tests; those land with the ADR-01 substrate PR).

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use jammi_engine::session::JammiSession;
use jammi_engine::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_engine::TenantId;
use tempfile::tempdir;

use crate::common;

fn widget_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn tenant_a() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap()
}

fn tenant_b() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap()
}

async fn register_widgets(session: &JammiSession) {
    let def = MutableTableDefinitionBuilder::new(
        MutableTableId::new("widgets").unwrap(),
        widget_schema(),
    )
    .primary_key(vec!["id".into()])
    .build()
    .unwrap();
    session.create_mutable_table(def).await.unwrap();
}

/// Two sessions in the same process with different tenant bindings see
/// disjoint row sets through the same mutable table.
#[tokio::test]
async fn two_tenants_see_disjoint_rows() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let session_a = JammiSession::new(cfg.clone()).await.unwrap();
    register_widgets(&session_a).await;
    let session_a = session_a.with_tenant(tenant_a());
    session_a
        .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (1, 'alpha')")
        .await
        .unwrap();

    let session_b = JammiSession::new(cfg)
        .await
        .unwrap()
        .with_tenant(tenant_b());
    session_b
        .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (2, 'beta')")
        .await
        .unwrap();

    let rows_a = session_a
        .sql("SELECT id, name FROM mutable.public.widgets ORDER BY id")
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
        .sql("SELECT id, name FROM mutable.public.widgets ORDER BY id")
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
#[tokio::test]
async fn unscoped_session_sees_only_global_rows() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    // Unscoped session writes one row → tenant_id NULL.
    let session = JammiSession::new(cfg.clone()).await.unwrap();
    register_widgets(&session).await;
    session
        .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (10, 'global')")
        .await
        .unwrap();

    // Scoped session writes one row → tenant_id = A.
    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    session_a
        .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (20, 'a-only')")
        .await
        .unwrap();

    // A fresh Unscoped session should see only the global row.
    let session_unscoped = JammiSession::new(cfg).await.unwrap();
    let rows = session_unscoped
        .sql("SELECT id, name FROM mutable.public.widgets ORDER BY id")
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
#[tokio::test]
async fn scoped_session_sees_own_plus_global() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let session = JammiSession::new(cfg.clone()).await.unwrap();
    register_widgets(&session).await;
    session
        .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (100, 'global')")
        .await
        .unwrap();

    let session_a = JammiSession::new(cfg)
        .await
        .unwrap()
        .with_tenant(tenant_a());
    session_a
        .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (200, 'a')")
        .await
        .unwrap();

    let rows = session_a
        .sql("SELECT id FROM mutable.public.widgets ORDER BY id")
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
#[tokio::test]
async fn tenant_binding_is_sticky_across_queries() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());
    let session = JammiSession::new(cfg).await.unwrap();
    register_widgets(&session).await;
    let session = session.with_tenant(tenant_a());

    session
        .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (1, 'a1'), (2, 'a2')")
        .await
        .unwrap();

    let rows = session
        .sql("SELECT COUNT(*) AS n FROM mutable.public.widgets")
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
        .sql("SELECT name FROM mutable.public.widgets ORDER BY name")
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

#[tokio::test]
async fn with_tenant_returns_same_session_id() {
    // with_tenant is a builder that returns Self — it does not require a
    // SessionContext rebuild.
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());
    let session = JammiSession::new(cfg).await.unwrap();
    assert!(session.tenant().is_none());

    let session = session.with_tenant(tenant_a());
    assert_eq!(session.tenant(), Some(tenant_a()));
}

/// Two scoped sessions writing through the same `Catalog::register_source`
/// path see disjoint `list_sources` results — the tenant filter on read +
/// the tenant binding on write together enforce isolation.
#[tokio::test]
async fn catalog_sources_isolated_by_tenant() {
    use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    session_a
        .add_source(
            "src_a",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let session_b = JammiSession::new(cfg)
        .await
        .unwrap()
        .with_tenant(tenant_b());
    session_b
        .add_source(
            "src_b",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sources_a = session_a.catalog().list_sources().await.unwrap();
    let ids_a: Vec<&str> = sources_a.iter().map(|s| s.source_id.as_str()).collect();
    assert_eq!(ids_a, vec!["src_a"]);

    let sources_b = session_b.catalog().list_sources().await.unwrap();
    let ids_b: Vec<&str> = sources_b.iter().map(|s| s.source_id.as_str()).collect();
    assert_eq!(ids_b, vec!["src_b"]);
}

/// An unscoped session sees globally-scoped (NULL) rows; a scoped session
/// sees its own rows plus the NULL rows (consistent with the read-side
/// predicate-injection rule).
#[tokio::test]
async fn catalog_unscoped_session_sees_global_only_after_scoped_writes() {
    use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let unscoped = JammiSession::new(cfg.clone()).await.unwrap();
    unscoped
        .add_source(
            "global_src",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let scoped = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    scoped
        .add_source(
            "tenant_a_src",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // A fresh unscoped session sees only the global row.
    let fresh_unscoped = JammiSession::new(cfg).await.unwrap();
    let ids: Vec<String> = fresh_unscoped
        .catalog()
        .list_sources()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.source_id)
        .collect();
    assert_eq!(ids, vec!["global_src".to_string()]);
}

/// SPEC-03 §12 #2 — one federated source carries a `tenant_id` column;
/// the analyzer rule injects a per-session filter that yields 6 rows for
/// tenant A and 4 rows for tenant B on the same on-disk Parquet table.
/// Verifies the read-side predicate-injection path end-to-end against a
/// local Parquet source, not just the engine-internal mutable table tested
/// elsewhere in this file.
#[tokio::test]
async fn federated_source_tenant_column_filters_split_6_4() {
    use arrow::array::{ArrayRef, RecordBatch};
    use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let dir = tempdir().unwrap();
    let pq_path = dir.path().join("notes.parquet");

    // Build a 10-row batch: 6 for tenant A, 4 for tenant B.
    let schema = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("tenant_id", DataType::Utf8, true),
    ]));
    let note_ids = Int64Array::from((0..10_i64).collect::<Vec<_>>());
    let a_str = tenant_a().to_string();
    let b_str = tenant_b().to_string();
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

    let cfg = common::test_config(dir.path());
    let url = format!("file://{}", pq_path.display());

    // Register the source ONCE — unscoped (tenant_id NULL on the catalog
    // row) — so both per-tenant sessions read it from the catalog on
    // reload. SPEC-03 §12 #2 calls for "one source registration, one
    // connection pool, no per-tenant table".
    {
        let registrar = JammiSession::new(cfg.clone()).await.unwrap();
        registrar
            .add_source(
                "notes",
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
    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    session_a.set_source_tenant_column("notes", Some("tenant_id".into()));

    let session_b = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_b());
    session_b.set_source_tenant_column("notes", Some("tenant_id".into()));

    async fn count_for(session: &JammiSession) -> i64 {
        let rows = session
            .sql("SELECT COUNT(*) AS n FROM notes.public.notes")
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

    let n_a = count_for(&session_a).await;
    let n_b = count_for(&session_b).await;
    assert_eq!(n_a, 6, "session A must see exactly its 6 rows");
    assert_eq!(n_b, 4, "session B must see exactly its 4 rows");

    async fn collect_ids(session: &JammiSession) -> Vec<i64> {
        let rows = session
            .sql("SELECT note_id FROM notes.public.notes ORDER BY note_id")
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
    let ids_a = collect_ids(&session_a).await;
    let ids_b = collect_ids(&session_b).await;
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

/// `Transaction::assert_tenant_matches` is the defence-in-depth write-side
/// guard. The sink calls it once per write_all; verify it rejects mismatches.
#[tokio::test]
async fn transaction_tenant_guard_rejects_mismatch() {
    use jammi_engine::catalog::backend::{BackendError, TxOptions};
    use jammi_engine::catalog::backend_sqlite::SqliteBackend;
    use jammi_engine::catalog::Catalog;
    use jammi_engine::CatalogBackend;

    let dir = tempdir().unwrap();
    let backend = SqliteBackend::open(&dir.path().join("guard.db"))
        .await
        .unwrap();
    let _catalog = Catalog::from_backend(jammi_engine::BackendImpl::Sqlite(backend.clone()));

    let bound = tenant_a();
    let other = tenant_b();
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
