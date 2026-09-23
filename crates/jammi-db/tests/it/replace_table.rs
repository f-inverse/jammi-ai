//! `CREATE OR REPLACE TABLE … AS` at the store, on both catalog backends:
//! the replacement is built under a row of its own and published under the
//! name by the one promote transaction that removes the old row — the old
//! table serves until then, the new one after, and the old bytes go; a
//! name nothing is under is simply created; a replacement whose target is a
//! live writer's `building` row is refused with nothing written; a
//! replacement abandoned before its swap is reaped by recovery, row and
//! bytes, and the table it named stays.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use jammi_datafusion::ModelTask;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::{
    CreateResultTableParams, ResultTableCas, ResultTableKind, ResultTableRecord,
};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::config::StoragePrecision;
use jammi_db::error::JammiError;
use jammi_db::session::QueryContext;
use jammi_db::storage::StorageUrl;
use jammi_db::store::{result_table_relation, CreateTableAs, ResultStore};
use tempfile::tempdir;
use test_case::test_case;

use crate::common::{fresh_catalog, memory_scan, objects_named, store_over, titled_rows};

/// Two rows the replacement carries instead of [`titled_rows`]'s three.
fn later_rows() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("title", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![4, 5])),
            Arc::new(StringArray::from(vec![
                "separator membrane",
                "current collector",
            ])),
        ],
    )
    .unwrap()
}

fn statement(name: &str, or_replace: bool, query: &str) -> CreateTableAs {
    CreateTableAs {
        name: name.to_string(),
        if_not_exists: false,
        or_replace,
        query: query.to_string(),
        sources: Vec::new(),
    }
}

async fn create(
    store: &ResultStore,
    ctx: &QueryContext,
    statement: &CreateTableAs,
    rows: RecordBatch,
) -> jammi_db::error::Result<Option<ResultTableRecord>> {
    store
        .create_table_as(ctx, statement, memory_scan(rows), ctx.task_ctx())
        .await
}

/// The `id`s of `"jammi.<name>"` as `ctx` resolves it now.
async fn ids(ctx: &QueryContext, name: &str) -> Vec<i64> {
    use arrow::array::AsArray;
    use arrow::datatypes::Int64Type;

    let batches = ctx
        .table(result_table_relation(name).table_reference())
        .await
        .unwrap()
        .sort(vec![datafusion::prelude::col("id").sort(true, true)])
        .unwrap()
        .collect()
        .await
        .unwrap();
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect()
}

async fn object_exists(store: &ResultStore, url: &StorageUrl) -> bool {
    let handle = store.open_parquet(url).unwrap();
    handle.exists(&handle.data_path().unwrap()).await.unwrap()
}

/// A successful replacement: the name moves to the new artifact in one
/// transaction — a reader resolves the new rows, the row under the name is
/// `ready` with a different `parquet_path`, and the old Parquet with its
/// attestation is gone; `OR REPLACE` on a name nothing is under creates
/// it; `DROP TABLE` afterwards reclaims what the name holds.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_replacement_moves_the_name_onto_the_new_artifact_and_reclaims_the_old(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let ctx = QueryContext::from(SessionContext::new());
    store.install_result_schema(&ctx).unwrap();
    let name = format!("recent_{}", jammi_test_utils::unique_suffix());

    // OR REPLACE with nothing under the name is a plain create: the row
    // lives under the name itself.
    let first = create(
        &store,
        &ctx,
        &statement(&name, true, "SELECT id, title FROM docs"),
        titled_rows(),
    )
    .await
    .unwrap()
    .expect("created");
    assert_eq!(first.table_name, name);
    assert!(first.parquet_path.ends_with(&format!("/{name}.parquet")));
    let old_url = StorageUrl::parse(&first.parquet_path).unwrap();
    assert_eq!(ids(&ctx, &name).await, vec![1, 2, 3]);

    let second = create(
        &store,
        &ctx,
        &statement(&name, true, "SELECT id, title FROM docs WHERE id > 3"),
        later_rows(),
    )
    .await
    .unwrap()
    .expect("replaced");
    assert_eq!(second.table_name, name, "published under the name");
    assert_eq!(second.status, ResultTableStatus::Ready.to_string());
    assert_eq!(second.kind, ResultTableKind::Statement);
    assert_eq!(second.row_count, 2);
    assert_ne!(
        second.parquet_path, first.parquet_path,
        "a different artifact"
    );
    assert_eq!(ids(&ctx, &name).await, vec![4, 5], "the new rows serve");
    assert!(
        !object_exists(&store, &old_url).await,
        "the old Parquet is reclaimed"
    );
    assert!(
        objects_named(dir.path(), &format!("{name}.")).is_empty(),
        "the old artifact and its attestation are gone: {:?}",
        objects_named(dir.path(), &format!("{name}."))
    );
    let rows: Vec<_> = catalog
        .list_result_tables_by_status(ResultTableStatus::Ready)
        .await
        .unwrap()
        .into_iter()
        .filter(|r| r.table_name.starts_with(&name))
        .map(|r| r.table_name)
        .collect();
    assert_eq!(rows, vec![name.clone()], "one row under the name, no other");

    // DROP TABLE reclaims everything the name holds now.
    let new_url = StorageUrl::parse(&second.parquet_path).unwrap();
    store.drop_result_table(&name).await.unwrap();
    assert!(!object_exists(&store, &new_url).await);
    assert!(objects_named(dir.path(), &name).is_empty());
    assert!(catalog.get_result_table(&name).await.unwrap().is_none());
}

/// A replacement whose write fails leaves the old table as it was —
/// readable, its row and bytes untouched — and leaves no row and no bytes
/// of its own.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_replacement_that_fails_to_write_leaves_the_old_table_and_nothing_of_its_own(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let ctx = QueryContext::from(SessionContext::new());
    store.install_result_schema(&ctx).unwrap();
    let name = format!("recent_{}", jammi_test_utils::unique_suffix());
    let first = create(
        &store,
        &ctx,
        &statement(&name, false, "SELECT id, title FROM docs"),
        titled_rows(),
    )
    .await
    .unwrap()
    .expect("created");

    // A child that faults as it is pulled: a cast no row of the titles
    // survives.
    let failing = SessionContext::new();
    let plan = failing
        .read_batch(titled_rows())
        .unwrap()
        .select(vec![
            datafusion::prelude::col("id"),
            datafusion::prelude::cast(datafusion::prelude::col("title"), DataType::Int64)
                .alias("title"),
        ])
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let err = store
        .create_table_as(
            &ctx,
            &statement(
                &name,
                true,
                "SELECT id, CAST(title AS BIGINT) AS title FROM docs",
            ),
            plan,
            ctx.task_ctx(),
        )
        .await
        .expect_err("the cast refuses the rows");
    assert!(
        !matches!(
            err,
            JammiError::RowGone { .. } | JammiError::CasFailed { .. }
        ),
        "the statement's own error, not the discard's: {err:?}"
    );

    let after = catalog
        .get_result_table(&name)
        .await
        .unwrap()
        .expect("the old row");
    assert_eq!(
        after.parquet_path, first.parquet_path,
        "the old row, untouched"
    );
    assert_eq!(after.status, ResultTableStatus::Ready.to_string());
    assert_eq!(ids(&ctx, &name).await, vec![1, 2, 3], "the old rows serve");
    for status in [ResultTableStatus::Building, ResultTableStatus::Failed] {
        let leftovers: Vec<_> = catalog
            .list_result_tables_by_status(status)
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.table_name.starts_with(&name))
            .map(|r| r.table_name)
            .collect();
        assert!(
            leftovers.is_empty(),
            "no row of the failed statement: {leftovers:?}"
        );
    }
    assert!(
        objects_named(dir.path(), "__replacement__").is_empty(),
        "no bytes of the failed statement: {:?}",
        objects_named(dir.path(), "__replacement__")
    );
}

/// A `building` row under `writer`'s live lease, seeded straight into the
/// catalog — the row a live statement holds while it writes.
async fn building_row(
    catalog: &jammi_db::catalog::Catalog,
    dir: &std::path::Path,
    table_name: &str,
    writer: &str,
    replaces: Option<&str>,
) {
    let parquet_path = format!("file://{}/{table_name}.parquet", dir.display());
    catalog
        .create_result_table(CreateResultTableParams {
            table_name,
            source_id: "docs",
            model_id: "statement",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Statement,
            derived_from: None,
            parquet_path: &parquet_path,
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::lease::canonical_stamp_now(),
            writer_id: Some(writer),
            lease: Some(std::time::Duration::from_secs(60)),
            job_attempt: None,
            replaces,
        })
        .await
        .unwrap();
}

/// The name a replacement is to take is a live writer's `building` row:
/// the promote is refused naming that status, and nothing is written — the
/// replacement's row stays `building` under its own writer, the live row
/// stays.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_replacement_of_a_live_writers_row_is_refused_with_nothing_written(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let name = format!("recent_{}", jammi_test_utils::unique_suffix());
    let staging = format!("{name}__replacement__test");
    let writer = format!("writer-{}", jammi_test_utils::unique_suffix());
    for (table, replaces) in [(&name, None), (&staging, Some(name.as_str()))] {
        building_row(&catalog, dir.path(), table, &writer, replaces).await;
    }

    let err = catalog
        .promote_result_table_with_manifest(
            &ResultTableCas::writer(&staging, &writer, None),
            2,
            "deadbeef",
            "[]",
        )
        .await
        .expect_err("a live writer's row is not replaced");
    assert!(
        matches!(&err, JammiError::CasFailed { table, status }
            if *table == name && *status == ResultTableStatus::Building.to_string()),
        "got {err:?}"
    );
    for (table, status) in [(&name, "building"), (&staging, "building")] {
        let row = catalog.get_result_table(table).await.unwrap().unwrap();
        assert_eq!(row.status, status, "{table} untouched");
        assert_eq!(row.table_name, *table);
    }
}

/// A replacement abandoned before its swap — its writer gone, its lease
/// expired — is reaped by recovery whatever its bytes say: the row and the
/// bytes go, and the table it was to replace is untouched.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_replacement_abandoned_before_its_swap_is_reaped_and_the_table_stays(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let ctx = QueryContext::from(SessionContext::new());
    store.install_result_schema(&ctx).unwrap();
    let name = format!("recent_{}", jammi_test_utils::unique_suffix());
    let first = create(
        &store,
        &ctx,
        &statement(&name, false, "SELECT id, title FROM docs"),
        titled_rows(),
    )
    .await
    .unwrap()
    .expect("created");

    // The replacement's row as a dead writer leaves it: no lease, and a
    // complete Parquet it never got to publish.
    let staging = format!("{name}__replacement__abandoned");
    let staging_url = StorageUrl::parse(&format!(
        "{}/{staging}.parquet",
        first.parquet_path.rsplit_once('/').unwrap().0
    ))
    .unwrap();
    catalog
        .create_result_table(CreateResultTableParams {
            table_name: &staging,
            source_id: "docs",
            model_id: "statement",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Statement,
            derived_from: None,
            parquet_path: staging_url.as_str(),
            dimensions: None,
            key_column: None,
            text_columns: None,
            storage_precision: StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::lease::canonical_stamp_now(),
            writer_id: None,
            lease: None,
            job_attempt: None,
            replaces: Some(&name),
        })
        .await
        .unwrap();
    let mut writer = store
        .open_writer(&staging_url, later_rows().schema())
        .await
        .unwrap();
    writer.write_batch(&later_rows()).await.unwrap();
    writer.close().await.unwrap();
    assert!(object_exists(&store, &staging_url).await);

    store.recover().await.unwrap();

    assert!(
        catalog.get_result_table(&staging).await.unwrap().is_none(),
        "the abandoned replacement's row is gone"
    );
    assert!(
        !object_exists(&store, &staging_url).await,
        "and its bytes with it"
    );
    let after = catalog.get_result_table(&name).await.unwrap().unwrap();
    assert_eq!(after.parquet_path, first.parquet_path);
    assert_eq!(after.status, ResultTableStatus::Ready.to_string());
    assert_eq!(ids(&ctx, &name).await, vec![1, 2, 3]);
}
