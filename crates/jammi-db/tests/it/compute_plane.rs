//! `CREATE TABLE … AS` issued to the session is a result table — a
//! `result_tables` row, bytes under the store's root, read on every session
//! bound to the catalog as `"jammi.<name>"` — written through the sink where
//! the installed compute plane says; `DROP TABLE` is the store's drop of it;
//! `CREATE TABLE` without a query, or over a query the engine could not
//! replay, is refused typed; a `SELECT` on the same session never reaches
//! the plane.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::{execute_stream, ExecutionPlan};
use futures::future::BoxFuture;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::compute_plane::{ComputePlane, Unheld};
use jammi_db::config::AnnIndexConfig;
use jammi_db::error::{JammiError, Result};
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;

use crate::common;

/// A plane that runs every plan it is handed under its own bare context —
/// the way an executor runs a plan under its own session, where the sink
/// arrives placed and writes — and counts them.
struct CountingPlane {
    submitted: AtomicUsize,
}

impl ComputePlane for CountingPlane {
    fn unheld(&self, _plan: &Arc<dyn ExecutionPlan>) -> BoxFuture<'static, Result<Option<Unheld>>> {
        Box::pin(async { Ok(None) })
    }

    fn place(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> BoxFuture<'static, Result<SendableRecordBatchStream>> {
        self.submitted.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let ctx = datafusion::prelude::SessionContext::new();
            Ok(execute_stream(plan, ctx.task_ctx())?)
        })
    }
}

/// A session over `patents` with a result store installed and the counting
/// plane as its compute plane.
async fn session_with_store(
    dir: &std::path::Path,
) -> (JammiSession, ResultStore, Arc<CountingPlane>, String) {
    let session = jammi_test_utils::make_test_session(BackendKind::Sqlite, dir).await;
    let store = ResultStore::new(
        dir,
        Arc::clone(session.catalog()),
        AnnIndexConfig::default(),
    )
    .unwrap();
    store.install_result_schema(session.context()).unwrap();
    let plane = Arc::new(CountingPlane {
        submitted: AtomicUsize::new(0),
    });
    assert!(session.compute_plane().install(plane.clone()));
    let patents = format!("patents_{}", jammi_test_utils::unique_suffix());
    session
        .add_source(
            &patents,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    (session, store, plane, patents)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_create_table_as_is_a_result_table_written_on_the_plane_and_a_select_never_is() {
    let dir = tempfile::tempdir().unwrap();
    let (session, store, plane, patents) = session_with_store(dir.path()).await;

    let expected = session
        .sql(&format!(
            "SELECT id, title FROM {patents}.public.patents WHERE year >= 2022 ORDER BY id"
        ))
        .await
        .unwrap();
    assert_eq!(
        plane.submitted.load(Ordering::SeqCst),
        0,
        "a SELECT serves its rows inline"
    );

    let created = session
        .sql(&format!(
            "CREATE TABLE recent AS SELECT id, title FROM {patents}.public.patents \
             WHERE year >= 2022 ORDER BY id"
        ))
        .await
        .unwrap();
    assert!(
        created.iter().all(|b| b.num_rows() == 0),
        "a CREATE TABLE AS returns no rows"
    );
    assert_eq!(
        plane.submitted.load(Ordering::SeqCst),
        1,
        "the CREATE TABLE AS submitted its sink once"
    );

    // The table is catalogued state: a `ready` `result_tables` row of the
    // statement kind, its bytes under the store's root.
    let record = session
        .catalog()
        .get_result_table("recent")
        .await
        .unwrap()
        .expect("the statement's result_tables row");
    assert_eq!(record.kind, ResultTableKind::Statement);
    assert_eq!(record.status, ResultTableStatus::Ready.to_string());
    assert_eq!(
        record.row_count,
        expected.iter().map(|b| b.num_rows()).sum::<usize>()
    );
    let url = StorageUrl::parse(&record.parquet_path).unwrap();
    assert!(store.holds_url(&url), "{url} lies under the store's root");
    let handle = store.open_parquet(&url).unwrap();
    assert!(handle.exists(&handle.data_path().unwrap()).await.unwrap());

    let read = session
        .sql("SELECT id, title FROM \"jammi.recent\" ORDER BY id")
        .await
        .unwrap();
    assert_eq!(plane.submitted.load(Ordering::SeqCst), 1);
    let concat = |batches: &[arrow::array::RecordBatch]| {
        arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
    };
    assert_eq!(concat(&read), concat(&expected));

    // DROP TABLE is the store's drop: the row, the bytes and the binding go.
    session.sql("DROP TABLE recent").await.unwrap();
    assert!(session
        .catalog()
        .get_result_table("recent")
        .await
        .unwrap()
        .is_none());
    assert!(!handle.exists(&handle.data_path().unwrap()).await.unwrap());
    session
        .sql("SELECT id FROM \"jammi.recent\"")
        .await
        .expect_err("a dropped table resolves nothing");
    let gone = session
        .sql("DROP TABLE recent")
        .await
        .expect_err("dropping a table that is not there is refused");
    assert!(
        matches!(gone, JammiError::RowGone { ref table } if table == "recent"),
        "expected RowGone, got {gone:?}"
    );
    session.sql("DROP TABLE IF EXISTS recent").await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_create_table_without_a_query_is_refused_typed() {
    let dir = tempfile::tempdir().unwrap();
    let (session, _store, plane, _patents) = session_with_store(dir.path()).await;
    let err = session
        .sql("CREATE TABLE empty_rows (id BIGINT, title VARCHAR)")
        .await
        .expect_err("a result table is what a query produced; there are no empty ones");
    assert!(
        matches!(err, JammiError::Schema { ref table, .. } if table == "empty_rows"),
        "expected the typed Schema refusal naming the table, got {err:?}"
    );
    assert!(session
        .catalog()
        .get_result_table("empty_rows")
        .await
        .unwrap()
        .is_none());
    assert_eq!(plane.submitted.load(Ordering::SeqCst), 0);
}

/// A `CREATE TABLE … AS` over a query the unparser cannot render back to
/// SQL is refused at planning, naming the node: the table would record a
/// definition no recompute could replay, so nothing is written and nothing
/// is submitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_create_table_as_over_a_query_that_cannot_replay_is_refused_typed() {
    let dir = tempfile::tempdir().unwrap();
    let (session, _store, plane, patents) = session_with_store(dir.path()).await;
    let err = session
        .sql(&format!(
            "CREATE TABLE lineage AS WITH RECURSIVE cited AS \
             (SELECT id FROM {patents}.public.patents WHERE id = 1 \
              UNION ALL SELECT id + 1 FROM cited WHERE id < 3) \
             SELECT id FROM cited"
        ))
        .await
        .expect_err("a recursive query renders to no SQL the engine can re-plan");
    assert!(
        matches!(err, JammiError::Schema { ref table, ref actual, .. }
            if table == "lineage" && actual.contains("RecursiveQuery")),
        "expected the typed Schema refusal naming the table and the node, got {err:?}"
    );
    assert!(session
        .catalog()
        .get_result_table("lineage")
        .await
        .unwrap()
        .is_none());
    assert_eq!(plane.submitted.load(Ordering::SeqCst), 0);
}
