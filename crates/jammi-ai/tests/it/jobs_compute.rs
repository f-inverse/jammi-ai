//! Item 2/N1: every embedded synchronous compute verb goes through
//! `InferenceSession::run_now`, and a claimed job's second-and-later attempt
//! dispatches on `jobs.partial_result` before ever running the producer
//! again.
//!
//! K4's "embedded return == remote terminal payload" has no remote transport
//! to exercise in this crate (the wire redesign is a follow-up unit), so the
//! analogous, in-tree property this file proves instead is the ACTUAL
//! architectural seam item 2 introduces: `run_now`'s inline path and a
//! worker's claimed-queued-job path both bottom out at
//! `jammi_ai::jobs::execute_compute` — driving both directly must produce
//! byte-identical tables for the identical spec.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, StringArray};
use jammi_ai::fine_tune::worker::JobWorker;
use jammi_ai::jobs::{execute_compute, ComputeSpec, JobResult};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::result_repo::JobAttempt;
use jammi_db::catalog::status::JobExecution;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_patents() -> (Arc<InferenceSession>, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
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
    (session, dir)
}

/// Claim `spec` exactly as `JobWorker`'s poll loop would (a fresh
/// `execution = 'queued'` row, `claim_next`), returning the
/// `(job_id, instance_id, attempts)` tuple `execute_compute` needs.
async fn submit_and_claim(
    session: &Arc<InferenceSession>,
    spec: &ComputeSpec,
) -> (String, String, u32) {
    let job_id = uuid::Uuid::new_v4().to_string();
    let spec_json = serde_json::to_string(spec).unwrap();
    session
        .catalog()
        .submit_job(SubmitJobParams {
            job_id: &job_id,
            kind: spec.kind(),
            execution: JobExecution::Queued,
            spec: &spec_json,
            model_ref: None,
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let instance_id = session.instance_id().to_string();
    let lease = session.worker_intervals().unwrap().lease;
    let claimed = session
        .catalog()
        .claim_next(&instance_id, &[spec.kind()], lease)
        .await
        .unwrap()
        .expect("the just-submitted job must be claimable");
    assert_eq!(claimed.job_id, job_id);
    (job_id, instance_id, claimed.attempts)
}

/// item 3/K4: `generate_embeddings` (the `run_now` wrapper) and a
/// worker-claimed `embedding` job of the SAME spec, run through
/// `execute_compute` directly, materialise independent tables with the
/// identical `definition_hash` and byte-identical vectors.
#[tokio::test]
async fn embedding_run_now_and_a_claimed_job_are_byte_identical() {
    let (session, _dir) = session_with_patents().await;
    let modality = jammi_wire::request::Modality::Text;

    let (record_a, _) = session
        .generate_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            modality,
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let spec = ComputeSpec::Embedding {
        source_id: "patents".to_string(),
        model_id: tiny_bert_model(),
        columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        modality,
        cache: CachePolicy::Bypass,
    };
    let (job_id, instance_id, attempts) = submit_and_claim(&session, &spec).await;
    let job_attempt = JobAttempt {
        job_id: &job_id,
        instance_id: &instance_id,
        attempts,
    };
    let result_b = execute_compute(&session, &spec, job_attempt).await.unwrap();
    let table_b = match result_b {
        JobResult::Table { table, .. } => table,
        JobResult::Model { .. } => panic!("expected a Table result"),
    };
    let record_b = session
        .catalog()
        .get_result_table(&table_b)
        .await
        .unwrap()
        .unwrap();

    assert_ne!(
        record_a.table_name, record_b.table_name,
        "two independent materializations, never the same table"
    );
    assert_eq!(
        record_a.definition_hash, record_b.definition_hash,
        "the SAME spec over the SAME content must hash to the SAME definition \
         regardless of which entry point ran it"
    );
    assert_eq!(record_a.row_count, record_b.row_count);
    assert_eq!(record_a.dimensions, record_b.dimensions);

    let vectors_a = session.read_vectors(&record_a).await.unwrap();
    let vectors_b = session.read_vectors(&record_b).await.unwrap();
    assert_eq!(
        vectors_a, vectors_b,
        "run_now and a claimed queued job of the same spec must embed byte-identical vectors"
    );
}

/// item 3/K4, the `infer` verb: `run_now`'s wrapper and a claimed `infer`
/// job dispatched through `execute_compute` directly must read back
/// byte-identical rows (in the SAME `_row_id, _ordinal` order — item 6).
#[tokio::test]
async fn infer_run_now_and_a_claimed_job_are_byte_identical() {
    let (session, _dir) = session_with_patents().await;
    let source = jammi_ai::model::ModelSource::parse(&tiny_bert_model());

    let (batches_a, _) = session
        .infer(
            "patents",
            &source,
            ModelTask::TextEmbedding,
            &["abstract".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let spec = ComputeSpec::Infer {
        source_id: "patents".to_string(),
        model_id: tiny_bert_model(),
        task: ModelTask::TextEmbedding,
        content_columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        cache: CachePolicy::Bypass,
    };
    let (job_id, instance_id, attempts) = submit_and_claim(&session, &spec).await;
    let job_attempt = JobAttempt {
        job_id: &job_id,
        instance_id: &instance_id,
        attempts,
    };
    let result_b = execute_compute(&session, &spec, job_attempt).await.unwrap();
    let table_b = match result_b {
        JobResult::Table { table, .. } => table,
        JobResult::Model { .. } => panic!("expected a Table result"),
    };
    let batches_b = session
        .sql(&format!(
            "SELECT * FROM \"jammi.{table_b}\" ORDER BY _row_id, _ordinal"
        ))
        .await
        .unwrap();

    let ids_a: Vec<String> = row_ids(&batches_a);
    let ids_b: Vec<String> = row_ids(&batches_b);
    assert_eq!(
        ids_a, ids_b,
        "run_now and a claimed queued infer job of the same spec must read \
         back in the identical _row_id order"
    );
    assert!(
        !ids_a.is_empty(),
        "the fixture must produce at least one row"
    );
}

/// `_row_id` as `Utf8` (a batch straight off `InferenceSession::infer`'s
/// normalized read-back) or `Utf8View` (a raw `SELECT *` over the registered
/// result table, which the catalog's provider resolves as a view type — see
/// `InferenceSession::normalize_view_columns`'s doc) — both arms this file
/// compares must read equally regardless of which one produced the batch.
fn row_ids(batches: &[arrow::record_batch::RecordBatch]) -> Vec<String> {
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch.column_by_name("_row_id").unwrap();
        if let Some(utf8) = col.as_any().downcast_ref::<StringArray>() {
            for i in 0..utf8.len() {
                ids.push(utf8.value(i).to_string());
            }
        } else if let Some(view) = col.as_any().downcast_ref::<arrow::array::StringViewArray>() {
            for i in 0..view.len() {
                ids.push(view.value(i).to_string());
            }
        } else {
            panic!(
                "_row_id is neither Utf8 nor Utf8View: {:?}",
                col.data_type()
            );
        }
    }
    ids
}

/// N1: a job's SECOND attempt (after its lease expired and a peer reclaimed
/// it) must read `jobs.partial_result`, see the `ready` table the FIRST
/// attempt already wrote, and finish the job with it directly — never
/// re-materialize a second table for the same job.
///
/// Drives the actual worker dispatch (`JobWorker::run_claimed_job`, which
/// hits `run_claimed_compute_job`'s N1 branch internally for a compute
/// kind) rather than reimplementing the algorithm in the test.
#[tokio::test]
async fn n1_reclaimed_attempt_adopts_the_ready_partial_result_table() {
    let (session, _dir) = session_with_patents().await;
    let lease = Duration::from_millis(60);

    let spec = ComputeSpec::Embedding {
        source_id: "patents".to_string(),
        model_id: tiny_bert_model(),
        columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        modality: jammi_wire::request::Modality::Text,
        cache: CachePolicy::Bypass,
    };
    let job_id = uuid::Uuid::new_v4().to_string();
    let spec_json = serde_json::to_string(&spec).unwrap();
    session
        .catalog()
        .submit_job(SubmitJobParams {
            job_id: &job_id,
            kind: spec.kind(),
            execution: JobExecution::Queued,
            spec: &spec_json,
            model_ref: None,
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();

    // Attempt 1: claim under a short lease and execute for real — this is
    // the SAME `execute_compute` call `run_claimed_compute_job` makes, so it
    // performs the `create_table` CAS that stamps `jobs.partial_result` —
    // but deliberately never calls `finish_job` (simulating a worker that
    // crashed after materializing and before its terminal write).
    let worker_a_id = format!("{}-a", session.instance_id());
    let claimed1 = session
        .catalog()
        .claim_next(&worker_a_id, &[spec.kind()], lease)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(claimed1.attempts, 1);
    let job_attempt1 = JobAttempt {
        job_id: &job_id,
        instance_id: &worker_a_id,
        attempts: claimed1.attempts,
    };
    let result1 = execute_compute(&session, &spec, job_attempt1)
        .await
        .unwrap();
    let table1 = match result1 {
        JobResult::Table { table, .. } => table,
        JobResult::Model { .. } => panic!("expected a Table result"),
    };
    let record1 = session
        .catalog()
        .get_result_table(&table1)
        .await
        .unwrap()
        .expect("attempt 1 must have created its table");
    assert_eq!(record1.status, "ready");

    // The job row itself now carries `partial_result = table1` (written by
    // `create_table`'s own transaction) but is still `running` under
    // attempt 1's now-abandoned lease.
    let mid_record = session.catalog().get_job(&job_id).await.unwrap();
    assert_eq!(mid_record.partial_result.as_deref(), Some(table1.as_str()));

    // Let attempt 1's job lease expire, then a peer worker reclaims it —
    // this is what actually advances `attempts` to 2.
    tokio::time::sleep(lease * 4).await;
    session
        .catalog()
        .reclaim_expired_jobs(lease, 5)
        .await
        .unwrap();

    // `JobWorker`'s `worker_id` is always stamped from the session's own
    // `instance_id` (see `JobWorker::with_intervals_and_kinds`'s doc), so
    // attempt 2 is claimed under THAT identity — the one the `JobWorker`
    // under test will present for every lease-guarded read/write N1 issues.
    let worker_b_id = session.instance_id().to_string();
    let claimed2 = session
        .catalog()
        .claim_next(&worker_b_id, &[spec.kind()], Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the reclaimed job must be claimable again");
    assert_eq!(claimed2.job_id, job_id);
    assert_eq!(
        claimed2.attempts, 2,
        "reclaim must advance the attempt counter"
    );
    assert_eq!(claimed2.partial_result.as_deref(), Some(table1.as_str()));

    let worker_intervals = session.worker_intervals().unwrap();
    let worker = JobWorker::with_intervals_and_kinds(
        &session,
        worker_intervals,
        vec![spec.kind().to_string()],
    );
    worker.run_claimed_job(&session, claimed2).await;

    let finished = session.catalog().get_job(&job_id).await.unwrap();
    assert_eq!(
        finished.status, "completed",
        "N1's Ready arm must finish the job directly from the adopted table"
    );
    let result_json = finished
        .result
        .expect("a completed job must carry a result");
    let result: JobResult = serde_json::from_str(&result_json).unwrap();
    match result {
        JobResult::Table { table, .. } => {
            assert_eq!(
                table, table1,
                "N1 must finish with the FIRST attempt's table, never a duplicate"
            );
        }
        JobResult::Model { .. } => panic!("expected a Table result"),
    }

    // No second `neighbor_graph`/`embedding`-kind result table exists for
    // this source under this model — the adopt path never re-materialized.
    let all_for_source = session
        .catalog()
        .find_result_tables("patents", Some(ModelTask::TextEmbedding), None)
        .await
        .unwrap();
    let matching: Vec<_> = all_for_source
        .iter()
        .filter(|t| t.table_name == table1)
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "exactly one result table must exist for this job — no duplicate materialization"
    );
}
