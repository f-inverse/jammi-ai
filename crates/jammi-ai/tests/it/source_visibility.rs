//! Two `InferenceSession`s over ONE catalog — a query-tier replica and a
//! compute replica sharing the `sources` table. A source registered through
//! one serves a SQL query, a `describe_source` and a submitted-and-claimed
//! job's source resolution on the other with no restart; once removed
//! through the first, every one of those on the second is a typed
//! not-found naming the source.

use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use jammi_ai::jobs::{execute_compute, ComputeSpec, JobResult};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::JobAttempt;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// Two sessions on one artifact dir: two catalog pools on one SQLite file.
async fn two_sessions() -> (
    Arc<InferenceSession>,
    Arc<InferenceSession>,
    tempfile::TempDir,
) {
    let dir = tempfile::TempDir::new().unwrap();
    let a = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let b = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    (a, b, dir)
}

fn embedding_spec(source_id: &str) -> ComputeSpec {
    ComputeSpec::Embedding(jammi_ai::local_session::EmbeddingRequest {
        source_id: source_id.to_string(),
        model_id: tiny_bert_model(),
        columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        modality: jammi_wire::request::Modality::Text,
        cache: CachePolicy::Bypass,
        dimensions: None,
    })
}

/// Run `spec` on `session` as a claimed queued job — the compute-replica
/// path a `JobWorker` takes after `claim_next`.
async fn run_claimed(
    session: &Arc<InferenceSession>,
    spec: &ComputeSpec,
) -> Result<String, JammiError> {
    let (job_id, instance_id, attempts) = common::submit_and_claim(session, spec).await;
    let attempt = JobAttempt {
        job_id: &job_id,
        instance_id: &instance_id,
        attempts,
    };
    match execute_compute(session, session.catalog(), spec, attempt).await? {
        JobResult::Table { table, .. } => Ok(table),
        JobResult::Model { .. } => panic!("an embedding job yields a table"),
    }
}

async fn count_patents(session: &InferenceSession) -> Result<i64, JammiError> {
    let batches = session
        .sql("SELECT COUNT(*) AS n FROM patents.public.patents")
        .await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT(*) is Int64")
        .value(0))
}

fn assert_source_not_found(err: &JammiError, source_id: &str) {
    assert!(
        matches!(err, JammiError::SourceNotFound { source_id: s } if s == source_id),
        "expected SourceNotFound for '{source_id}', got {err:?}"
    );
    assert!(
        err.to_string().contains(source_id),
        "the not-found message must name the source: {err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_source_registered_on_one_session_serves_a_query_a_describe_and_a_claimed_job_on_another()
{
    let (query_tier, compute, _dir) = two_sessions().await;

    query_tier
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // A plain query routed to the other replica.
    let rows = count_patents(&compute).await.unwrap();
    assert!(rows > 0);
    assert_eq!(rows, count_patents(&query_tier).await.unwrap());

    // `describe_source` on the other replica.
    let described = compute
        .catalog()
        .describe_source("patents")
        .await
        .unwrap()
        .expect("the other replica describes the source");
    assert_eq!(described.source_id, "patents");

    // A job claimed on the other replica resolves the source and completes.
    let table = run_claimed(&compute, &embedding_spec("patents"))
        .await
        .expect("a compute replica resolves a source a query-tier replica registered");
    let record = compute
        .catalog()
        .get_result_table(&table)
        .await
        .unwrap()
        .expect("the claimed job's table is catalogued");
    assert_eq!(record.row_count, usize::try_from(rows).unwrap());

    // Removal on one replica propagates to the other: the query, the
    // describe and a freshly claimed job all resolve not-found by name.
    query_tier.remove_source("patents").await.unwrap();

    let err = count_patents(&compute).await.unwrap_err();
    assert_source_not_found(&err, "patents");
    assert!(compute
        .catalog()
        .describe_source("patents")
        .await
        .unwrap()
        .is_none());
    let err = run_claimed(&compute, &embedding_spec("patents"))
        .await
        .expect_err("a job over a removed source fails");
    assert_source_not_found(&err, "patents");
}
