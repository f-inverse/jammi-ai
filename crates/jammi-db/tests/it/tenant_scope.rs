//! Phase 3 integration tests — tenant-scoped sessions deliver disjoint
//! views of mutable companion tables. Engine-only scope (no wire-surface
//! tests; those land with the ADR-01 substrate PR).

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use jammi_db::session::JammiSession;
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::TenantId;
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

fn tenant_c() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9c").unwrap()
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
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

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

/// `list_all_sources` enumerates sources across every tenant, while the
/// tenant-scoped `list_sources` stays filtered to its own rows plus the
/// globally-scoped (`tenant_id IS NULL`) rows. Session startup re-hydrates
/// source providers through the cross-tenant view so a worker that later
/// binds to any tenant can resolve that tenant's private sources.
#[tokio::test]
async fn catalog_list_all_sources_sees_across_tenants() {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let parquet = || SourceConnection {
        url: Some(common::fixture_url("patents.parquet")),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    };

    // One global source plus one private source per tenant.
    let unscoped = JammiSession::new(cfg.clone()).await.unwrap();
    unscoped
        .add_source("global_src", SourceType::File, parquet())
        .await
        .unwrap();

    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    session_a
        .add_source("src_a", SourceType::File, parquet())
        .await
        .unwrap();

    let session_b = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_b());
    session_b
        .add_source("src_b", SourceType::File, parquet())
        .await
        .unwrap();

    // Cross-tenant enumeration sees every source regardless of binding.
    // Sort for a registration-order-independent set comparison: the catalog
    // orders by `created_at`, which ties across sub-millisecond inserts.
    let mut all: Vec<String> = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .catalog()
        .list_all_sources()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.source_id)
        .collect();
    all.sort();
    assert_eq!(
        all,
        vec![
            "global_src".to_string(),
            "src_a".to_string(),
            "src_b".to_string()
        ]
    );

    // The tenant-scoped API stays filtered to tenant A's own + global rows.
    let mut scoped_a: Vec<String> = session_a
        .catalog()
        .list_sources()
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.source_id)
        .collect();
    scoped_a.sort();
    assert_eq!(
        scoped_a,
        vec!["global_src".to_string(), "src_a".to_string()]
    );
}

/// An unscoped session sees globally-scoped (NULL) rows; a scoped session
/// sees its own rows plus the NULL rows (consistent with the read-side
/// predicate-injection rule).
#[tokio::test]
async fn catalog_unscoped_session_sees_global_only_after_scoped_writes() {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

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
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
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

/// A source's tenant discriminator persists in the catalog and is replayed on
/// reload. The `SourceTenantColumns` lookup is process-memory only, so without
/// persistence a federated source registered with a `tenant_column` would
/// reload after a restart with no scope — a latent cross-tenant read. Here the
/// column is set on the `SourceConnection` at registration (never via the
/// in-process setter), the session is dropped and rebuilt against the same
/// catalog DB, and the rebuilt session must replay the discriminator and emit
/// the scoping filter. A source registered with no discriminator reloads as
/// `None`, emitting no filter.
#[tokio::test]
async fn source_tenant_column_persists_and_replays_on_reload() {
    use arrow::array::{ArrayRef, RecordBatch};
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let dir = tempdir().unwrap();

    // Two parquet files: `notes` carries tenancy under `customer_id`
    // (6 rows tenant A, 4 rows tenant B); `public_docs` carries no tenant
    // discriminator at all.
    let notes_path = dir.path().join("notes.parquet");
    let docs_path = dir.path().join("public_docs.parquet");

    let notes_schema = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("customer_id", DataType::Utf8, true),
    ]));
    let a_str = tenant_a().to_string();
    let b_str = tenant_b().to_string();
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

    let cfg = common::test_config(dir.path());

    // Register both sources against the catalog, then drop the session. The
    // discriminator is carried on the connection — never via
    // `set_source_tenant_column` — so the persist path is what's exercised.
    {
        let registrar = JammiSession::new(cfg.clone()).await.unwrap();
        registrar
            .add_source(
                "notes",
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
                "public_docs",
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
    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    let session_b = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_b());

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
        count(&session_a, "SELECT COUNT(*) FROM notes.public.notes").await,
        6,
        "tenant A must see its 6 rows via the replayed customer_id filter"
    );
    assert_eq!(
        count(&session_b, "SELECT COUNT(*) FROM notes.public.notes").await,
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
            "SELECT COUNT(*) FROM public_docs.public.public_docs"
        )
        .await,
        5,
        "an un-scoped source must reload as None — no spurious filter"
    );
    assert_eq!(
        count(
            &session_b,
            "SELECT COUNT(*) FROM public_docs.public.public_docs"
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
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_tenant_scoped_isolates_concurrent_tasks() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let session = Arc::new(JammiSession::new(cfg).await.unwrap());
    register_widgets(&session).await;

    // Each task: enter its tenant's scope, insert a row tagged with its
    // tenant name, count rows visible inside the scope, and snapshot
    // `session.tenant()` from inside the closure. Both tasks run on the
    // same `Arc<JammiSession>` concurrently on a multi-thread runtime.
    async fn run_one(
        session: Arc<JammiSession>,
        tenant: TenantId,
        row_id: i64,
        row_name: &'static str,
    ) -> (Option<TenantId>, i64) {
        session
            .with_tenant_scoped(tenant, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.widgets (id, name) VALUES ({row_id}, '{row_name}')"
                    ))
                    .await
                    .unwrap();
                let observed = scope.tenant();
                let rows = scope
                    .sql("SELECT id FROM mutable.public.widgets")
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
        handles.push(tokio::spawn(async move {
            run_one(session_a, tenant_a(), id_a, "alpha").await
        }));
        handles.push(tokio::spawn(async move {
            run_one(session_b, tenant_b(), id_b, "beta").await
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let (observed, visible_id) = h.await.unwrap();
        let task_is_a = i % 2 == 0;
        let expected_tenant = if task_is_a { tenant_a() } else { tenant_b() };
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
#[tokio::test]
async fn with_tenant_scoped_does_not_mutate_sticky_binding() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let session = JammiSession::new(cfg).await.unwrap();
    register_widgets(&session).await;
    // Sticky-bind tenant_b so we can observe that the scoped call to
    // tenant_a does not leak past the closure.
    let session = session.with_tenant(tenant_b());
    let session = Arc::new(session);

    assert_eq!(session.tenant(), Some(tenant_b()));
    let observed_inside = session
        .with_tenant_scoped(tenant_a(), |scope| async move { scope.tenant() })
        .await;
    assert_eq!(observed_inside, Some(tenant_a()));
    // After the scope exits, the sticky binding (tenant_b) is restored
    // because it was never touched.
    assert_eq!(session.tenant(), Some(tenant_b()));
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
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_scoped_stream_remains_tenant_filtered_after_closure_returns() {
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
    let cfg = common::test_config(dir.path());
    let session = Arc::new(JammiSession::new(cfg).await.unwrap());

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
        name: "global.events".into(),
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

    let backend = session.catalog().backend_arc();
    let registry_for_a = Arc::clone(&registry);
    let backing_for_a = backing_id.clone();
    let a_batch_one = row_for(0, 100, "a-one");
    let a_batch_two = row_for(1, 101, "a-two");
    backend
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.set_tenant(Some(tenant_a()));
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
    backend
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.set_tenant(Some(tenant_b()));
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
        .with_tenant_scoped(tenant_a(), |_scope| {
            let subscriber = Arc::clone(&subscriber);
            let topic = topic.clone();
            async move {
                subscriber
                    .subscribe_scoped(
                        &topic,
                        Some(tenant_a()),
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
#[tokio::test]
async fn with_admin_scope_sees_across_tenants() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    // Single session, used by every tenant in sequence — exercises the
    // session-shared mutable-table registry that the leak path threads
    // through.
    let session = Arc::new(JammiSession::new(cfg).await.unwrap());
    register_widgets(&session).await;

    for (tenant, id, name) in [
        (tenant_a(), 1_i64, "a-row"),
        (tenant_b(), 2_i64, "b-row"),
        (tenant_c(), 3_i64, "c-row"),
    ] {
        session
            .with_tenant_scoped(tenant, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.widgets (id, name) VALUES ({id}, '{name}')"
                    ))
                    .await
                    .unwrap();
            })
            .await;
    }

    // Outside an admin scope, an unbound session sees zero tenant-tagged
    // rows (only globally-scoped, `tenant_id IS NULL` rows would surface).
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
        .unwrap()
        .value(0);
    assert_eq!(n, 0, "unbound session must not see any tenant-tagged rows");

    // Inside `with_admin_scope`, every row across every tenant is visible.
    let ids: Vec<i64> = session
        .with_admin_scope(|admin| async move {
            let rows = admin
                .sql("SELECT id FROM mutable.public.widgets ORDER BY id")
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
#[tokio::test]
async fn admin_scope_does_not_leak_into_subsequent_calls() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let session = Arc::new(JammiSession::new(cfg).await.unwrap());
    register_widgets(&session).await;

    session
        .with_tenant_scoped(tenant_a(), |scope| async move {
            scope
                .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (1, 'a')")
                .await
                .unwrap();
        })
        .await;
    session
        .with_tenant_scoped(tenant_b(), |scope| async move {
            scope
                .sql("INSERT INTO mutable.public.widgets (id, name) VALUES (2, 'b')")
                .await
                .unwrap();
        })
        .await;

    // Admin scope: sees both rows.
    let cross_tenant_count = session
        .with_admin_scope(|admin| async move {
            let rows = admin
                .sql("SELECT COUNT(*) AS n FROM mutable.public.widgets")
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
        .sql("SELECT COUNT(*) AS n FROM mutable.public.widgets")
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
        .with_tenant_scoped(tenant_b(), |scope| async move {
            let rows = scope
                .sql("SELECT id FROM mutable.public.widgets ORDER BY id")
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

/// Cross-tenant + global delete isolation for `remove_source`. The DELETE uses
/// the STRICT tenant predicate (`tenant_id = $cur OR (tenant_id IS NULL AND
/// $cur IS NULL)`), not the read path's `OR tenant_id IS NULL` leak: a tenant
/// session removes only a source it owns — never another tenant's, and never a
/// shared GLOBAL (`tenant_id IS NULL`) source it did not create. Only an
/// unscoped session manages GLOBAL rows.
///
/// Pre-fix (loose `OR tenant_id IS NULL` DELETE) this FAILS at the global-row
/// step: tenant B's `remove_source` deletes the GLOBAL row it does not own.
#[tokio::test]
async fn remove_source_refuses_cross_tenant_and_global() {
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());
    let parquet = || SourceConnection {
        url: Some(common::fixture_url("patents.parquet")),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    };

    // One GLOBAL source (unscoped), one private source per tenant. All share
    // the same backend (the same catalog file under `dir`).
    let unscoped = JammiSession::new(cfg.clone()).await.unwrap();
    unscoped
        .add_source("global_src", SourceType::File, parquet())
        .await
        .unwrap();

    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    session_a
        .add_source("src_a", SourceType::File, parquet())
        .await
        .unwrap();

    let session_b = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_b());
    session_b
        .add_source("src_b", SourceType::File, parquet())
        .await
        .unwrap();

    let exists = |sess: &JammiSession, id: &'static str| {
        let cat = sess.catalog().clone();
        async move { cat.get_source(id).await.unwrap().is_some() }
    };

    // 1) Tenant B cannot remove tenant A's private source.
    session_b.catalog().remove_source("src_a").await.unwrap();
    assert!(
        exists(&session_a, "src_a").await,
        "tenant B must not delete tenant A's source"
    );

    // 2) Tenant B cannot remove the GLOBAL source it did not create.
    session_b
        .catalog()
        .remove_source("global_src")
        .await
        .unwrap();
    assert!(
        exists(&unscoped, "global_src").await,
        "a tenant must not delete a shared GLOBAL source"
    );

    // 3) Tenant B removes its OWN source fine.
    session_b.catalog().remove_source("src_b").await.unwrap();
    assert!(
        !exists(&session_b, "src_b").await,
        "a tenant deletes its own source"
    );

    // 4) An unscoped session CAN remove the GLOBAL source.
    unscoped
        .catalog()
        .remove_source("global_src")
        .await
        .unwrap();
    assert!(
        !exists(&unscoped, "global_src").await,
        "an unscoped session manages GLOBAL sources"
    );
}

/// Cross-tenant + global delete isolation for `drop_topic`. Both the lookup and
/// the DELETE carry the STRICT tenant predicate, so a tenant session resolves
/// only a topic it owns — a GLOBAL (`tenant_id IS NULL`) topic surfaces
/// `TopicNotFound` and is never deleted, and (critically) its shared backing
/// table is never dropped. Only an unscoped session manages GLOBAL topics.
///
/// Pre-fix (loose lookup + loose DELETE) this FAILS: tenant B's `drop_topic`
/// deletes the GLOBAL topic's catalog row.
#[tokio::test]
async fn drop_topic_refuses_cross_tenant_and_global() {
    use jammi_db::trigger::{TopicDefinition, TopicId, TriggerError};
    use std::collections::BTreeMap;

    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let topic_schema = || {
        Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
        ]))
    };

    // Per-tenant sessions over the SAME backend (the same catalog file under
    // `dir`), mirroring production where the session's tenant binding rides on
    // the catalog AND the backing-table registry together. The unscoped
    // session owns GLOBAL rows.
    let unscoped = JammiSession::new(cfg.clone()).await.unwrap();
    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_a());
    let session_b = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(tenant_b());

    // GLOBAL topic (unscoped registration) + one private topic for tenant A.
    let global = TopicDefinition {
        id: TopicId::new(),
        name: "global.events".into(),
        schema: topic_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    unscoped.topic_repo().register_topic(&global).await.unwrap();

    let owned_a = TopicDefinition {
        id: TopicId::new(),
        name: "a.events".into(),
        schema: topic_schema(),
        tenant: Some(tenant_a()),
        broker_metadata: BTreeMap::new(),
    };
    session_a
        .topic_repo()
        .register_topic(&owned_a)
        .await
        .unwrap();

    let listed_for = |sess: &JammiSession, tenant: Option<TenantId>| {
        let repo = sess.topic_repo();
        async move {
            repo.list_topics(tenant)
                .await
                .unwrap()
                .into_iter()
                .map(|t| t.id)
                .collect::<Vec<_>>()
        }
    };

    // 1) Tenant B cannot drop tenant A's private topic (TopicNotFound, untouched).
    match session_b
        .topic_repo()
        .drop_topic(owned_a.id, Some(tenant_b()))
        .await
    {
        Err(TriggerError::TopicNotFound(_)) => {}
        other => panic!("expected TopicNotFound for cross-tenant drop, got {other:?}"),
    }
    assert!(
        listed_for(&session_a, Some(tenant_a()))
            .await
            .contains(&owned_a.id),
        "tenant B must not drop tenant A's topic"
    );

    // 2) Tenant B cannot drop the GLOBAL topic it did not create.
    match session_b
        .topic_repo()
        .drop_topic(global.id, Some(tenant_b()))
        .await
    {
        Err(TriggerError::TopicNotFound(_)) => {}
        other => panic!("expected TopicNotFound for global drop, got {other:?}"),
    }
    assert!(
        listed_for(&unscoped, None).await.contains(&global.id),
        "a tenant must not drop a shared GLOBAL topic"
    );

    // 3) An unscoped session CAN drop the GLOBAL topic.
    unscoped
        .topic_repo()
        .drop_topic(global.id, None)
        .await
        .unwrap();
    assert!(
        !listed_for(&unscoped, None).await.contains(&global.id),
        "an unscoped session manages GLOBAL topics"
    );

    // 4) Tenant A drops its OWN topic fine.
    session_a
        .topic_repo()
        .drop_topic(owned_a.id, Some(tenant_a()))
        .await
        .unwrap();
    assert!(
        !listed_for(&session_a, Some(tenant_a()))
            .await
            .contains(&owned_a.id),
        "a tenant drops its own topic"
    );
}

/// Cross-tenant + global delete isolation for `delete_result_tables_for_source`.
/// Both the SELECT (the disk-cleanup set) and the DELETE carry the STRICT tenant
/// predicate, so a tenant session deletes only its own result tables — never a
/// shared GLOBAL (`tenant_id IS NULL`) one it did not create. Only an unscoped
/// session manages GLOBAL rows, and the returned record set equals the deleted
/// set exactly.
///
/// Pre-fix (loose `OR tenant_id IS NULL` SELECT+DELETE) this FAILS: tenant B's
/// call deletes (and returns) the GLOBAL result-table row it does not own.
#[tokio::test]
async fn delete_result_tables_for_source_refuses_cross_tenant_and_global() {
    use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind};
    use jammi_db::catalog::Catalog;
    use jammi_db::ModelTask;

    let dir = tempdir().unwrap();
    let unscoped = Catalog::open(dir.path()).await.unwrap();
    let cat_a = unscoped.pinned_to_tenant(Some(tenant_a()));
    let cat_b = unscoped.pinned_to_tenant(Some(tenant_b()));

    async fn make(cat: &Catalog, name: &str, source: &str) {
        cat.create_result_table(CreateResultTableParams {
            table_name: name,
            source_id: source,
            model_id: "model",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "file:///tmp/rt.parquet",
            index_path: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();
    }

    // A GLOBAL result table for source "shared" (unscoped) + a private one for
    // tenant A on the same source id.
    make(&unscoped, "rt_global", "shared").await;
    make(&cat_a, "rt_a", "shared").await;

    // 1) Tenant B deleting "shared" deletes nothing — neither tenant A's nor the
    //    GLOBAL row. The returned (disk-cleanup) set is empty.
    let removed = cat_b
        .delete_result_tables_for_source("shared")
        .await
        .unwrap();
    assert!(
        removed.is_empty(),
        "tenant B must not delete another tenant's or a GLOBAL result table; got {removed:?}"
    );
    // `get_result_table` is itself tenant-scoped (read predicate
    // `tenant_id = $cur OR tenant_id IS NULL`), so check each row through a
    // scope that can see it: the GLOBAL row via the unscoped catalog, tenant
    // A's private row via tenant A's catalog.
    assert!(
        unscoped
            .get_result_table("rt_global")
            .await
            .unwrap()
            .is_some(),
        "the GLOBAL result table survives a foreign tenant's delete"
    );
    assert!(
        cat_a.get_result_table("rt_a").await.unwrap().is_some(),
        "tenant A's result table survives tenant B's delete"
    );

    // 2) Tenant A deletes its OWN row — and only that row.
    let removed_a = cat_a
        .delete_result_tables_for_source("shared")
        .await
        .unwrap();
    assert_eq!(
        removed_a
            .iter()
            .map(|r| r.table_name.as_str())
            .collect::<Vec<_>>(),
        vec!["rt_a"],
        "tenant A deletes exactly its own result table"
    );
    assert!(
        unscoped
            .get_result_table("rt_global")
            .await
            .unwrap()
            .is_some(),
        "the GLOBAL result table still survives after tenant A's delete"
    );

    // 3) An unscoped session CAN delete the GLOBAL row.
    let removed_global = unscoped
        .delete_result_tables_for_source("shared")
        .await
        .unwrap();
    assert_eq!(
        removed_global
            .iter()
            .map(|r| r.table_name.as_str())
            .collect::<Vec<_>>(),
        vec!["rt_global"],
        "an unscoped session deletes the GLOBAL result table"
    );
}
