//! A `CREATE TABLE … AS` issued to the session runs its query where the
//! installed compute plane says, and the table it registers holds the rows
//! the plane streamed back; a `SELECT` on the same session never reaches
//! the plane.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use datafusion::physical_plan::{execute_stream, ExecutionPlan};
use futures::future::BoxFuture;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::compute_plane::{ComputePlane, Submission};
use jammi_db::error::Result;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

use crate::common;

/// A plane that runs every plan it is handed under its own bare context —
/// the way an executor runs a plan under its own session — and counts
/// them.
struct CountingPlane {
    submitted: AtomicUsize,
}

impl ComputePlane for CountingPlane {
    fn submit(&self, plan: Arc<dyn ExecutionPlan>) -> BoxFuture<'static, Result<Submission>> {
        self.submitted.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let ctx = datafusion::prelude::SessionContext::new();
            let stream = execute_stream(plan, ctx.task_ctx())?;
            Ok(Submission::Placed(stream))
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_create_table_as_runs_its_query_on_the_plane_and_a_select_never_does() {
    let dir = tempfile::tempdir().unwrap();
    let session = jammi_test_utils::make_test_session(BackendKind::Sqlite, dir.path()).await;
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

    session
        .sql(&format!(
            "CREATE TABLE recent AS SELECT id, title FROM {patents}.public.patents \
             WHERE year >= 2022 ORDER BY id"
        ))
        .await
        .unwrap();
    assert_eq!(
        plane.submitted.load(Ordering::SeqCst),
        1,
        "the CREATE TABLE AS submitted its query once"
    );

    let created = session
        .sql("SELECT id, title FROM recent ORDER BY id")
        .await
        .unwrap();
    assert_eq!(plane.submitted.load(Ordering::SeqCst), 1);
    let concat = |batches: &[arrow::array::RecordBatch]| {
        arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
    };
    assert_eq!(concat(&created), concat(&expected));
}
