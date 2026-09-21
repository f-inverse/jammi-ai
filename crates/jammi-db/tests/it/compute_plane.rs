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

/// The `(id, title)` rows of `"jammi.<name>"`, by id, and the `ready`
/// row's Parquet URL.
async fn rows_and_url(
    session: &JammiSession,
    name: &str,
) -> (Vec<arrow::array::RecordBatch>, StorageUrl) {
    let rows = session
        .sql(&format!(
            "SELECT id, title FROM \"jammi.{name}\" ORDER BY id"
        ))
        .await
        .unwrap();
    let record = session
        .catalog()
        .get_result_table(name)
        .await
        .unwrap()
        .expect("the row under the name");
    assert_eq!(record.status, ResultTableStatus::Ready.to_string());
    (rows, StorageUrl::parse(&record.parquet_path).unwrap())
}

fn concat(batches: &[arrow::array::RecordBatch]) -> arrow::array::RecordBatch {
    arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
}

async fn object_exists(store: &ResultStore, url: &StorageUrl) -> bool {
    let handle = store.open_parquet(url).unwrap();
    handle.exists(&handle.data_path().unwrap()).await.unwrap()
}

/// A `CREATE OR REPLACE TABLE … AS` whose query is refused at planning —
/// a relation that does not exist, a query that cannot replay — never
/// reaches the store: the old table's rows stay readable under its
/// unchanged row, and nothing was submitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replacement_refused_at_planning_leaves_the_old_table_readable() {
    let dir = tempfile::tempdir().unwrap();
    let (session, _store, plane, patents) = session_with_store(dir.path()).await;
    session
        .sql(&format!(
            "CREATE TABLE recent AS SELECT id, title FROM {patents}.public.patents \
             WHERE year >= 2022 ORDER BY id"
        ))
        .await
        .unwrap();
    let (before, url) = rows_and_url(&session, "recent").await;
    assert_eq!(plane.submitted.load(Ordering::SeqCst), 1);

    session
        .sql(&format!(
            "CREATE OR REPLACE TABLE recent AS SELECT id, title \
             FROM {patents}.public.no_such_table"
        ))
        .await
        .expect_err("an unknown relation refuses at planning");
    let err = session
        .sql(&format!(
            "CREATE OR REPLACE TABLE recent AS WITH RECURSIVE cited AS \
             (SELECT id FROM {patents}.public.patents WHERE id = 1 \
              UNION ALL SELECT id + 1 FROM cited WHERE id < 3) \
             SELECT id FROM cited"
        ))
        .await
        .expect_err("a query that cannot replay refuses at planning");
    assert!(
        matches!(err, JammiError::Schema { ref table, .. } if table == "recent"),
        "got {err:?}"
    );

    let (after, url_after) = rows_and_url(&session, "recent").await;
    assert_eq!(concat(&after), concat(&before), "the old rows serve");
    assert_eq!(url_after, url, "the old row, untouched");
    assert_eq!(
        plane.submitted.load(Ordering::SeqCst),
        1,
        "nothing was submitted"
    );
}

/// A `CREATE OR REPLACE TABLE … AS` whose input refuses mid-write — the
/// sink pulled rows the query could not produce — leaves the old table as
/// it was and no row or bytes of its own: the replacement's artifact is
/// reclaimed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replacement_that_refuses_mid_write_leaves_the_old_table_and_reclaims_its_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let (session, store, plane, patents) = session_with_store(dir.path()).await;
    session
        .sql(&format!(
            "CREATE TABLE recent AS SELECT id, title FROM {patents}.public.patents \
             WHERE year >= 2022 ORDER BY id"
        ))
        .await
        .unwrap();
    let (before, url) = rows_and_url(&session, "recent").await;

    session
        .sql(&format!(
            "CREATE OR REPLACE TABLE recent AS SELECT id, CAST(title AS BIGINT) AS title \
             FROM {patents}.public.patents"
        ))
        .await
        .expect_err("no title casts to an integer");
    assert_eq!(
        plane.submitted.load(Ordering::SeqCst),
        2,
        "the replacement's sink was submitted and failed there"
    );

    let (after, url_after) = rows_and_url(&session, "recent").await;
    assert_eq!(concat(&after), concat(&before), "the old rows serve");
    assert_eq!(url_after, url, "the old row, untouched");
    assert!(object_exists(&store, &url).await);
    for status in [ResultTableStatus::Building, ResultTableStatus::Failed] {
        let leftovers: Vec<_> = session
            .catalog()
            .list_result_tables_by_status(status)
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.table_name)
            .collect();
        assert!(
            leftovers.is_empty(),
            "no row of the failed statement: {leftovers:?}"
        );
    }
    let staged = common::objects_named(dir.path(), "__replacement__");
    assert!(
        staged.is_empty(),
        "no bytes of the failed statement: {staged:?}"
    );
}

/// A `CREATE OR REPLACE TABLE … AS` that completes serves the new rows
/// under the name from the moment it returns, on this session and on a
/// second one over the same catalog; the old bytes are gone; `DROP TABLE`
/// afterwards reclaims what the name holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replacement_that_completes_serves_the_new_rows_and_the_old_bytes_are_gone() {
    let dir = tempfile::tempdir().unwrap();
    let (session, store, plane, patents) = session_with_store(dir.path()).await;
    // A second session over the same catalog, bound to the old table
    // before the swap.
    let reader = store_over_session(dir.path(), &session).await;
    session
        .sql(&format!(
            "CREATE TABLE recent AS SELECT id, title FROM {patents}.public.patents \
             WHERE year >= 2022 ORDER BY id"
        ))
        .await
        .unwrap();
    let (before, old_url) = rows_and_url(&session, "recent").await;
    assert_eq!(concat(&read_recent(&reader).await), concat(&before));

    let expected = session
        .sql(&format!(
            "SELECT id, title FROM {patents}.public.patents WHERE year >= 2023 ORDER BY id"
        ))
        .await
        .unwrap();
    session
        .sql(&format!(
            "CREATE OR REPLACE TABLE recent AS SELECT id, title FROM {patents}.public.patents \
             WHERE year >= 2023 ORDER BY id"
        ))
        .await
        .unwrap();
    assert_eq!(plane.submitted.load(Ordering::SeqCst), 2);

    let (after, new_url) = rows_and_url(&session, "recent").await;
    assert_eq!(concat(&after), concat(&expected), "the new rows serve here");
    assert_eq!(
        concat(&read_recent(&reader).await),
        concat(&expected),
        "and on the second session, whose binding the catalog moved past"
    );
    assert_ne!(new_url, old_url);
    assert!(
        !object_exists(&store, &old_url).await,
        "the old bytes are gone"
    );
    assert!(object_exists(&store, &new_url).await);

    session.sql("DROP TABLE recent").await.unwrap();
    assert!(!object_exists(&store, &new_url).await);
    reader
        .table(jammi_db::store::result_table_relation("recent").table_reference())
        .await
        .expect_err("the second session finds no row either");
}

/// A reader on a second session over the same catalog that opened the old
/// table before the swap finishes its read: the rows it pulls after the
/// swap are the old table's, to the end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reader_that_opened_the_old_table_before_the_swap_finishes_its_read() {
    use futures::StreamExt;

    let dir = tempfile::tempdir().unwrap();
    let (session, _store, _plane, patents) = session_with_store(dir.path()).await;
    session
        .sql(&format!(
            "CREATE TABLE recent AS SELECT id, title FROM {patents}.public.patents ORDER BY id"
        ))
        .await
        .unwrap();
    let (before, _) = rows_and_url(&session, "recent").await;
    let total: usize = before.iter().map(|b| b.num_rows()).sum();
    assert!(total > 2, "enough rows for more than one batch: {total}");

    // The reader pulls two rows per batch, so its read spans the swap.
    let reader = store_over_session(dir.path(), &session).await;
    let mut stream = reader
        .table(jammi_db::store::result_table_relation("recent").table_reference())
        .await
        .unwrap()
        .execute_stream()
        .await
        .unwrap();
    let first = stream.next().await.expect("a first batch").unwrap();
    let mut read = first.num_rows();
    assert!(read < total, "the read is in flight");

    session
        .sql(&format!(
            "CREATE OR REPLACE TABLE recent AS SELECT id, title FROM {patents}.public.patents \
             WHERE year >= 2023 ORDER BY id"
        ))
        .await
        .unwrap();

    while let Some(batch) = stream.next().await {
        read += batch.unwrap().num_rows();
    }
    assert_eq!(read, total, "the reader finished the old table");
}

/// A second session's context over `session`'s catalog: its own store,
/// resolving result tables through the catalog, pulling two rows a batch.
async fn store_over_session(
    dir: &std::path::Path,
    session: &JammiSession,
) -> jammi_db::session::QueryContext {
    let store = ResultStore::new(
        dir,
        Arc::clone(session.catalog()),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let ctx = jammi_db::session::QueryContext::from(
        datafusion::prelude::SessionContext::new_with_config(
            datafusion::prelude::SessionConfig::new().with_batch_size(2),
        ),
    );
    // The context's config carries the store from here on; the provider
    // resolves through it.
    store.install_result_schema(&ctx).unwrap();
    ctx
}

async fn read_recent(ctx: &jammi_db::session::QueryContext) -> Vec<arrow::array::RecordBatch> {
    ctx.table(jammi_db::store::result_table_relation("recent").table_reference())
        .await
        .unwrap()
        .sort(vec![datafusion::prelude::col("id").sort(true, true)])
        .unwrap()
        .collect()
        .await
        .unwrap()
}
