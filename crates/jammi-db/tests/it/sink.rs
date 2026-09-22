//! The result-table sink's lifecycle on both catalog backends: the row a
//! sink writes under is taken from its submitter by transfer CAS, held by
//! the writing process, handed back on success and failed under the
//! writer's own id on failure; a second take of a row that already moved
//! fails typed; a sink executed in-process writes the table the submitter
//! then finishes; a sink whose object lies outside a store's root is
//! refused typed when it arrives placed.

use arrow::array::StringArray;
use arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::session::QueryContext;
use jammi_db::storage::{StorageError, StorageUrl};
use jammi_db::store::manifest::{
    ComputeDevice, InputAnchor, Materialization, MaterializationEnv, ProducingDescriptor,
};
use jammi_db::store::{
    BuildingTable, ResultStore, ResultTableSinkSpec, SinkKind, SinkLease, SinkLeaseKind,
};
use tempfile::tempdir;
use test_case::test_case;

use crate::common::{fresh_catalog, memory_scan as scan, store_over, titled_rows as rows};

async fn building(store: &ResultStore, source: &str) -> BuildingTable {
    store
        .create_table(
            source,
            ModelTask::TextEmbedding,
            ResultTableKind::AsofJoin,
            None,
            "rows",
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap()
}

/// The spec a submitter's `BuildingTable` yields for a rows sink.
fn rows_spec(building: &BuildingTable) -> ResultTableSinkSpec {
    ResultTableSinkSpec {
        table_name: building.table_name().to_string(),
        parquet_url: building.parquet_url().clone(),
        tenant: building.tenant(),
        writer_id: building.writer_id().to_string(),
        storage_precision: building.storage_precision(),
        lease: SinkLeaseKind::Table,
        kind: SinkKind::Rows,
    }
}

async fn writer_of(store: &ResultStore, table: &str) -> Option<String> {
    store
        .catalog()
        .get_result_table(table)
        .await
        .unwrap()
        .and_then(|r| r.writer_id)
}

/// The lease moves from the submitter to the writing process and back:
/// after `take` the row is the writer's, a second take of the same spec
/// fails the transfer CAS typed (the row already moved), and after
/// `hand_back` the submitter renews it again and finishes the table.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_sink_lease_is_taken_from_the_submitter_and_handed_back(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let submitter = store_over(dir.path(), &catalog);
    let writer = store_over(dir.path(), &catalog);
    assert_ne!(submitter.writer_id(), writer.writer_id());
    let source = format!("docs-{}", jammi_test_utils::unique_suffix());
    let mut table = building(&submitter, &source).await;
    let spec = rows_spec(&table);

    let lease = SinkLease::take(&writer, &spec).await.unwrap();
    assert!(lease.is_live());
    assert_eq!(
        writer_of(&submitter, table.table_name()).await.as_deref(),
        Some(writer.writer_id()),
        "the row is the writing process's for the duration of the write"
    );

    let second = SinkLease::take(&writer, &spec)
        .await
        .err()
        .expect("a second launch of the same sink cannot take a row that already moved");
    assert!(
        matches!(second, JammiError::LeaseLost { ref table } if *table == spec.table_name),
        "expected LeaseLost, got {second:?}"
    );

    lease.hand_back(submitter.writer_id()).await.unwrap();
    assert_eq!(
        writer_of(&submitter, table.table_name()).await.as_deref(),
        Some(submitter.writer_id())
    );
    table.rehold();
    assert!(table.is_live());
    catalog
        .renew_lease(&table.cas(), submitter.lease_intervals().lease())
        .await
        .expect("the submitter renews the row it got back");
    table.abort().await.unwrap();
    assert_eq!(
        submitter
            .catalog()
            .get_result_table(&spec.table_name)
            .await
            .unwrap()
            .unwrap()
            .status,
        ResultTableStatus::Failed.to_string()
    );
}

/// A failed write fails the row under the writing process's own id: the
/// row is `failed` and the submitter's own transitions miss typed.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_failed_sink_fails_the_row_under_its_own_writer(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let submitter = store_over(dir.path(), &catalog);
    let writer = store_over(dir.path(), &catalog);
    let source = format!("docs-{}", jammi_test_utils::unique_suffix());
    let table = building(&submitter, &source).await;
    let spec = rows_spec(&table);

    let lease = SinkLease::take(&writer, &spec).await.unwrap();
    lease.fail().await.unwrap();
    let row = submitter
        .catalog()
        .get_result_table(table.table_name())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(row.status, ResultTableStatus::Failed.to_string());
    let err = table
        .abort()
        .await
        .expect_err("the submitter no longer owns a row the writer failed");
    assert!(
        matches!(err, JammiError::CasFailed { ref status, .. }
            if *status == ResultTableStatus::Failed.to_string()),
        "expected CasFailed naming `failed`, got {err:?}"
    );
}

/// The sink executed with no compute plane writes the table in-process
/// under the same lifecycle — a transfer from the writer to itself — and
/// the submitter finishes what the summary reports.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_sink_with_no_plane_writes_the_table_here(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let ctx = QueryContext::from(SessionContext::new());
    store.install_result_schema(&ctx).unwrap();
    let source = format!("docs-{}", jammi_test_utils::unique_suffix());
    let mut table = building(&store, &source).await;

    let summary = store
        .write_result_table(&mut table, SinkKind::Rows, scan(rows()), ctx.task_ctx())
        .await
        .unwrap();
    assert_eq!(
        (summary.input_rows, summary.rows, summary.segments.as_slice()),
        (3, 3, &[][..])
    );
    assert_eq!(
        writer_of(&store, table.table_name()).await.as_deref(),
        Some(store.writer_id())
    );

    let descriptor = ProducingDescriptor::Statement {
        query: format!("SELECT id, title FROM {source}"),
    };
    let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
    let record = table
        .finish(
            &ctx,
            summary.rows as usize,
            Materialization::new(
                &descriptor,
                &env,
                vec![InputAnchor::unpinned_at_instant(&source, "now")],
            ),
        )
        .await
        .unwrap();
    assert_eq!(record.row_count, 3);
    let read = ctx
        .table(jammi_db::store::result_table_relation(&record.table_name).table_reference())
        .await
        .unwrap()
        .sort(vec![
            jammi_db::store::verbatim_column("id").sort(true, false)
        ])
        .unwrap()
        .collect()
        .await
        .unwrap();
    let read = arrow::compute::concat_batches(&read[0].schema(), &read).unwrap();
    assert_eq!(read.num_rows(), 3);
    // The reader resolves strings as Utf8View; compare in Utf8.
    let titles =
        arrow::compute::cast(read.column_by_name("title").unwrap(), &DataType::Utf8).unwrap();
    assert_eq!(
        titles
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(2),
        "solid electrolyte"
    );
}

/// A sink that arrives placed is refused typed when its object is not
/// under this store's root, or its row is one this catalog does not hold.
#[tokio::test]
async fn a_placed_sink_outside_this_stores_root_or_naming_an_unknown_row_is_refused() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(BackendKind::Sqlite, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let source = format!("docs-{}", jammi_test_utils::unique_suffix());
    let table = building(&store, &source).await;

    let mut foreign = rows_spec(&table);
    foreign.parquet_url =
        StorageUrl::parse("file:///elsewhere/jammi_db/_global/x.parquet").unwrap();
    let err = store
        .adopt_placed_sink(foreign, scan(rows()))
        .await
        .expect_err("an object outside this store's root");
    assert!(
        matches!(err, JammiError::Storage(StorageError::InvalidUrl { .. })),
        "expected InvalidUrl, got {err:?}"
    );

    let mut unknown = rows_spec(&table);
    unknown.table_name = "no-such-table".into();
    let err = store
        .adopt_placed_sink(unknown, scan(rows()))
        .await
        .expect_err("a row this catalog does not hold");
    assert!(
        matches!(err, JammiError::RowGone { ref table } if table == "no-such-table"),
        "expected RowGone, got {err:?}"
    );

    let placed = store
        .adopt_placed_sink(rows_spec(&table), scan(rows()))
        .await
        .unwrap();
    assert!(placed.is_placed());
}

/// A result table is catalogued state every session sees: a table one
/// session's store finished after another session started resolves on
/// that other session through the catalog, bound on first use.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_table_finished_on_one_session_resolves_on_another_through_the_catalog(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let producer = store_over(dir.path(), &catalog);
    let producer_ctx = QueryContext::from(SessionContext::new());
    producer.install_result_schema(&producer_ctx).unwrap();
    let reader = store_over(dir.path(), &catalog);
    let reader_ctx = QueryContext::from(SessionContext::new());
    reader.install_result_schema(&reader_ctx).unwrap();
    reader.load_existing_tables(&reader_ctx).await.unwrap();

    let source = format!("docs-{}", jammi_test_utils::unique_suffix());
    let mut table = building(&producer, &source).await;
    let summary = producer
        .write_result_table(
            &mut table,
            SinkKind::Rows,
            scan(rows()),
            producer_ctx.task_ctx(),
        )
        .await
        .unwrap();
    let descriptor = ProducingDescriptor::Statement {
        query: format!("SELECT id, title FROM {source}"),
    };
    let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
    let record = table
        .finish(
            &producer_ctx,
            summary.rows as usize,
            Materialization::new(
                &descriptor,
                &env,
                vec![InputAnchor::unpinned_at_instant(&source, "now")],
            ),
        )
        .await
        .unwrap();

    let relation = jammi_db::store::result_table_relation(&record.table_name);
    let read = reader_ctx
        .table(relation.table_reference())
        .await
        .expect("the reader resolves the name through the catalog")
        .collect()
        .await
        .unwrap();
    assert_eq!(read.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
}
