//! The result-table sink's lifecycle on both catalog backends: a write
//! streams its plan's rows under the caller's `building` row, which the
//! caller then finishes; a write that fails aborts the row in place — the
//! row is `failed`, its bytes are gone, and the caller's handle is done.

use arrow::array::StringArray;
use arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;
use jammi_datafusion::ModelTask;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::{Producer, ResultTableKind};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::error::JammiError;
use jammi_db::session::QueryContext;
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{
    ComputeDevice, ContentDigest, InputAnchor, LocalRun, Materialization, MaterializationEnv,
    ModelIdentity, ModelRun, ProducingDescriptor,
};
use jammi_db::store::sink::ProducingEnvironment;
use jammi_db::store::{BuildingTable, ResultStore, ResultTableOrigin, SinkKind};
use jammi_numerics::ComputePrecision;
use tempfile::tempdir;
use test_case::test_case;

use crate::common::{fresh_catalog, memory_scan as scan, store_over, titled_rows as rows};

async fn building(store: &ResultStore, source: &str) -> BuildingTable {
    store
        .create_table(ResultTableOrigin {
            source_id: source,
            producer: Producer::Model {
                model_id: "rows".to_string(),
                task: ModelTask::TextEmbedding,
            },
            kind: ResultTableKind::AsofJoin,
            derived_from: None,
            dimensions: None,
            key_column: None,
            text_columns: None,
            job_attempt: None,
        })
        .await
        .unwrap()
}

async fn writer_of(store: &ResultStore, table: &str) -> Option<String> {
    store
        .catalog()
        .get_result_table(table)
        .await
        .unwrap()
        .and_then(|r| r.writer_id)
}

/// A write that fails aborts the caller's row in place: an empty training
/// set is refused typed after its object was opened, the row is `failed`
/// under the caller's writer, the object is gone, and the caller's handle
/// is done — a second abort through it misses on the `failed` row.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_failed_write_aborts_the_callers_row_in_place(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let ctx = QueryContext::from(SessionContext::new());
    let source = format!("docs-{}", jammi_test_utils::unique_suffix());
    let mut table = building(&store, &source).await;
    let source_query = format!("SELECT id, title FROM {source} WHERE 1 = 0");

    let err = store
        .write_result_table(
            &mut table,
            SinkKind::TrainingSet {
                columns: vec!["id".into()],
                source_query: source_query.clone(),
            },
            scan(rows().slice(0, 0)),
            ctx.task_ctx(),
        )
        .await
        .expect_err("an empty training set is refused");
    assert!(
        matches!(err, JammiError::EmptyTrainingSet { source_query: ref q } if *q == source_query),
        "expected EmptyTrainingSet, got {err:?}"
    );
    assert!(!table.is_live(), "the aborted handle is done");
    let row = store
        .catalog()
        .get_result_table(table.table_name())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(row.status, ResultTableStatus::Failed.to_string());
    let handle = store.open_parquet(table.parquet_url()).unwrap();
    assert!(
        !handle.exists(&handle.data_path().unwrap()).await.unwrap(),
        "the failed write's object is deleted"
    );
    let again = table.abort().await.expect_err("the row is already failed");
    assert!(
        matches!(again, JammiError::CasFailed { ref status, .. }
            if *status == ResultTableStatus::Failed.to_string()),
        "expected CasFailed naming `failed`, got {again:?}"
    );
}

/// A write streams the plan's rows under the caller's row, which stays the
/// caller's, and the caller finishes what the summary reports.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_sink_writes_the_table_under_the_callers_row(backend: BackendKind) {
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
        (
            summary.input_rows,
            summary.rows,
            summary.segments.as_slice()
        ),
        (3, 3, &[][..])
    );
    assert_eq!(
        writer_of(&store, table.table_name()).await.as_deref(),
        Some(store.writer_id())
    );

    // A store with no environment installed runs no model: its sink
    // reports the model-free environment.
    assert_eq!(summary.env, MaterializationEnv::without_models());

    let descriptor = ProducingDescriptor::Statement {
        query: format!("SELECT id, title FROM {source}"),
    };
    let record = table
        .finish(
            &ctx,
            summary.rows as usize,
            Materialization::new(
                &descriptor,
                &summary.env,
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

/// The environment a process installs on its store is what every sink it
/// runs reports, and what the table's manifest then records.
struct Installed(MaterializationEnv);

#[async_trait::async_trait]
impl ProducingEnvironment for Installed {
    async fn of(
        &self,
        _plan: &std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    ) -> jammi_db::error::Result<MaterializationEnv> {
        Ok(self.0.clone())
    }
}

#[tokio::test]
async fn a_sink_reports_the_environment_of_the_store_that_runs_it() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(BackendKind::Sqlite, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let ran_on = MaterializationEnv::of_models(
        ComputeDevice::Cuda { ordinal: 1 },
        vec![ModelIdentity {
            model_id: "test-model".into(),
            run: ModelRun::Local(LocalRun {
                backend: jammi_db::store::manifest::LocalBackend::Candle,
                compute_precision: ComputePrecision::BF16,
                content_digest: ContentDigest("it-fixture-digest".into()),
                quantization: None,
            }),
        }],
    );
    assert!(store.install_producing_environment(std::sync::Arc::new(Installed(ran_on.clone()))));
    assert!(
        !store.install_producing_environment(std::sync::Arc::new(Installed(
            MaterializationEnv::without_models()
        ))),
        "the environment is installed once"
    );
    let ctx = QueryContext::from(SessionContext::new());
    store.install_result_schema(&ctx).unwrap();
    let source = format!("docs-{}", jammi_test_utils::unique_suffix());
    let mut table = building(&store, &source).await;

    let summary = store
        .write_result_table(&mut table, SinkKind::Rows, scan(rows()), ctx.task_ctx())
        .await
        .unwrap();
    assert_eq!(summary.env, ran_on);

    let descriptor = ProducingDescriptor::Statement {
        query: format!("SELECT id, title FROM {source}"),
    };
    let record = table
        .finish(
            &ctx,
            summary.rows as usize,
            Materialization::new(
                &descriptor,
                &summary.env,
                vec![InputAnchor::unpinned_at_instant(&source, "now")],
            ),
        )
        .await
        .unwrap();
    let manifest = store
        .read_materialization_manifest(&StorageUrl::parse(&record.parquet_path).unwrap())
        .await
        .unwrap()
        .expect("a finished table has its sidecar");
    assert_eq!(manifest.env, ran_on);
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
    let env = MaterializationEnv::without_models();
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
