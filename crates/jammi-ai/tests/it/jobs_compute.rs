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
    let result_b = execute_compute(&session, session.catalog(), &spec, job_attempt)
        .await
        .unwrap();
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
    assert_eq!(record_a.dimensions_raw(), record_b.dimensions_raw());

    let vectors_a = session.read_vectors(&record_a).await.unwrap();
    let vectors_b = session.read_vectors(&record_b).await.unwrap();
    assert_eq!(
        vectors_a, vectors_b,
        "run_now and a claimed queued job of the same spec must embed byte-identical vectors"
    );

    // Beyond the vectors: the two tables read back with the identical
    // schema and identical bytes in every column, row for row.
    let rows_a = session
        .sql(&format!(
            "SELECT * FROM \"jammi.{}\" ORDER BY _row_id",
            record_a.table_name
        ))
        .await
        .unwrap();
    let rows_b = session
        .sql(&format!(
            "SELECT * FROM \"jammi.{}\" ORDER BY _row_id",
            record_b.table_name
        ))
        .await
        .unwrap();
    assert_batches_byte_identical(&rows_a, &rows_b, &[]);
}

/// Every column of `a` and `b` — schema (names and types, after the
/// `Utf8View`→`Utf8` normalization a registered-table scan needs) and the
/// bytes of every row — must agree, except the columns named in `skip`
/// (per-row wall-clock latency legitimately differs between two runs).
fn assert_batches_byte_identical(
    a: &[arrow::record_batch::RecordBatch],
    b: &[arrow::record_batch::RecordBatch],
    skip: &[&str],
) {
    let a = normalized_single_batch(a);
    let b = normalized_single_batch(b);
    assert_eq!(
        a.schema().fields(),
        b.schema().fields(),
        "both arms must read back the identical schema"
    );
    assert!(
        a.num_rows() > 0,
        "the fixture must produce at least one row"
    );
    assert_eq!(a.num_rows(), b.num_rows());
    for (field, (col_a, col_b)) in a
        .schema()
        .fields()
        .iter()
        .zip(a.columns().iter().zip(b.columns()))
    {
        if skip.contains(&field.name().as_str()) {
            continue;
        }
        assert_eq!(
            col_a.to_data(),
            col_b.to_data(),
            "column `{}` must be byte-identical across the two arms",
            field.name()
        );
    }
}

/// Concatenate `batches` into one and cast every `Utf8View`/`BinaryView`
/// column to its plain encoding, so a batch straight off `infer` and a raw
/// `SELECT *` over the registered table compare on content, not on the
/// encoding the scan happened to choose.
fn normalized_single_batch(
    batches: &[arrow::record_batch::RecordBatch],
) -> arrow::record_batch::RecordBatch {
    use arrow::datatypes::{DataType, Field, Schema};
    let first = batches.first().expect("at least one batch");
    let batch = arrow::compute::concat_batches(&first.schema(), batches).unwrap();
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
        let target = match field.data_type() {
            DataType::Utf8View => Some(DataType::Utf8),
            DataType::BinaryView => Some(DataType::Binary),
            _ => None,
        };
        match target {
            Some(ty) => {
                columns.push(arrow::compute::cast(col, &ty).unwrap());
                fields.push(Arc::new(Field::new(field.name(), ty, field.is_nullable())));
            }
            None => {
                columns.push(Arc::clone(col));
                fields.push(Arc::clone(field));
            }
        }
    }
    arrow::record_batch::RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
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
    let result_b = execute_compute(&session, session.catalog(), &spec, job_attempt)
        .await
        .unwrap();
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
    // The full parity: schema (with `_ordinal` present and typed alike),
    // `_ordinal` values, and every column's bytes — only the per-row
    // latency is allowed to differ between two runs.
    assert!(
        batches_a[0].schema().column_with_name("_ordinal").is_some(),
        "an infer result carries `_ordinal`"
    );
    assert_batches_byte_identical(&batches_a, &batches_b, &["_latency_ms"]);
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
    let result1 = execute_compute(&session, session.catalog(), &spec, job_attempt1)
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

/// The escape `esc-110`'s own RED, on today's EXPIRY path (no RELEASE): a
/// compute job's first attempt is parked inside `BuildingTable::finish`
/// (its building row `building` under a live lease, `partial_result`
/// recorded), its job lease expires and its building lease is expired; the
/// second attempt's dispatch takes the claim-and-fail arm (`MaterializeAnew`),
/// clears the stale `partial_result`, and the attempt runs to `completed`
/// with its OWN `ready` table. Base: the second attempt's `create_result_table`
/// CAS (`… AND partial_result IS NULL`) matches 0 rows against the
/// once-written column → `JobAttemptSuperseded` → terminal `failed`.
#[serial_test::serial(materialization_park)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_compute_attempt_re_materializes_on_the_successor() {
    use jammi_db::catalog::result_repo::ResultTableCas;
    use jammi_db::store::mutable::test_hook::{arm, MaterializationPoint};

    let (session, _dir) = session_with_patents().await;
    let lease = Duration::from_millis(60);
    let writer_id = session.result_store().writer_id().to_string();

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

    // Attempt 1 (a peer instance): claimed under a short lease; its producer
    // parks inside `finish` with the building row leased and
    // `partial_result` pointing at it.
    let parked = arm(MaterializationPoint::Materialization, &writer_id);
    let worker_a_id = format!("{}-a", session.instance_id());
    let claimed1 = session
        .catalog()
        .claim_next(&worker_a_id, &[spec.kind()], lease)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(claimed1.attempts, 1);
    let attempt1 = {
        let session = Arc::clone(&session);
        let spec = spec.clone();
        let job_id = job_id.clone();
        let worker_a_id = worker_a_id.clone();
        tokio::spawn(async move {
            execute_compute(
                &session,
                session.catalog(),
                &spec,
                JobAttempt {
                    job_id: &job_id,
                    instance_id: &worker_a_id,
                    attempts: 1,
                },
            )
            .await
        })
    };
    parked
        .wait_parked()
        .await
        .expect("attempt 1 parks inside finish");
    let mid = session.catalog().get_job(&job_id).await.unwrap();
    let table1 = mid
        .partial_result
        .clone()
        .expect("attempt 1 recorded its building table");

    // Both leases expire: the job's by time, the building row's through the
    // test-only expiry (the catalog's own clock).
    tokio::time::sleep(lease * 4).await;
    session
        .catalog()
        .expire_lease_for_test(&ResultTableCas::writer(&table1, &writer_id, None))
        .await
        .unwrap();
    session
        .catalog()
        .reclaim_expired_jobs(lease, 5)
        .await
        .unwrap();

    // Attempt 2 under this session's own identity, driven through the real
    // worker dispatch.
    let claimed2 = session
        .catalog()
        .claim_next(
            session.instance_id(),
            &[spec.kind()],
            Duration::from_secs(30),
        )
        .await
        .unwrap()
        .expect("the requeued job is claimable");
    assert_eq!(claimed2.attempts, 2);
    assert_eq!(claimed2.partial_result.as_deref(), Some(table1.as_str()));
    let worker = JobWorker::with_intervals_and_kinds(
        &session,
        session.worker_intervals().unwrap(),
        vec![spec.kind().to_string()],
    );
    worker.run_claimed_job(&session, claimed2).await;

    let finished = session.catalog().get_job(&job_id).await.unwrap();
    assert_eq!(
        finished.status, "completed",
        "the successor must re-materialize, got {:?} ({:?})",
        finished.status, finished.error
    );
    let table2 = finished
        .partial_result
        .clone()
        .expect("the successor recorded its own table");
    assert_ne!(
        table2, table1,
        "a fresh table, never the failed predecessor's"
    );
    assert_eq!(
        session
            .catalog()
            .get_result_table(&table2)
            .await
            .unwrap()
            .unwrap()
            .status,
        "ready"
    );
    assert_eq!(
        session
            .catalog()
            .get_result_table(&table1)
            .await
            .unwrap()
            .unwrap()
            .status,
        "failed",
        "the predecessor's row was claimed and failed by the successor's dispatch"
    );

    // Let attempt 1's parked writer go: its own promote misses (the row is
    // `failed` under another owner) and the abandoned attempt errors out
    // without touching the successor's result.
    parked.release();
    let _ = attempt1.await.unwrap();
    assert_eq!(
        session.catalog().get_job(&job_id).await.unwrap().status,
        "completed"
    );
}
