//! `JammiExecutionEngine`'s per-stage refusals — device-kind mismatch and a
//! multi-partition placed-attempt stage — through the real engine, classified back to
//! the typed error the submitter restores.

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;

use ballista_executor::execution_engine::ExecutionEngine;

use jammi_ai::operator::placed_attempt_exec::PlacedAttemptExec;
use jammi_ai::session::InferenceSession;
use jammi_ballista::engine::JammiExecutionEngine;
use jammi_db::error::JammiError;
use jammi_db::store::manifest::ComputeDeviceKind;

async fn session() -> Arc<InferenceSession> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = jammi_test_utils::test_config(dir.path());
    let s = InferenceSession::new(cfg).await.expect("session builds");
    std::mem::forget(dir);
    Arc::new(s)
}

fn scan() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, true)]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(vec!["a"]))]).unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

/// A stage whose `InferenceExec` names a device kind different from this
/// executor's own `compute_device().kind()` is refused typed, never run.
/// The test session's default device is CPU (`jammi_test_utils::test_config`
/// sets no `[gpu]` override), so a descriptor explicitly stamped `Cuda`
/// mismatches it.
#[tokio::test]
async fn refuses_a_stage_whose_inference_exec_names_a_different_device_kind() {
    let session = session().await;
    assert_eq!(
        session.compute_device().kind(),
        ComputeDeviceKind::Cpu,
        "test precondition: the fixture session runs on CPU"
    );
    let plan = crate::inference_plan(&session, scan(), ComputeDeviceKind::Cuda, 1);

    let engine = JammiExecutionEngine::new(Arc::clone(&session));
    let err = engine
        .create_query_stage_exec(
            "job-1".to_string().into(),
            0,
            0,
            plan,
            "/tmp",
            &SessionConfig::default(),
        )
        .expect_err("a device-kind mismatch must be refused, never silently run");
    assert_device_kind_unheld(err);
}

/// The refusal an executor raises for a stage of another kind, as the
/// engine's classifier restores it from the stage-creation error: the
/// plan's own `Cuda`, this executor's `Cpu` as the one kind held.
fn assert_device_kind_unheld(err: datafusion::error::DataFusionError) {
    match JammiError::from(err) {
        JammiError::DeviceKindUnheld { required, held } => {
            assert_eq!(required, ComputeDeviceKind::Cuda);
            assert_eq!(held, vec![ComputeDeviceKind::Cpu]);
        }
        other => panic!("expected DeviceKindUnheld, got {other:?}"),
    }
}

/// The matching-kind arm: a descriptor whose kind agrees with the executor's
/// own device is NOT refused by the device-kind check (it proceeds to
/// `DefaultExecutionEngine`, which then fails for an unrelated reason — no
/// real `ShuffleWriterExec` wraps this plan — proving the device-kind gate itself let
/// it through rather than raising a false positive).
#[tokio::test]
async fn does_not_refuse_a_matching_device_kind() {
    let session = session().await;
    let own_kind = session.compute_device().kind();
    let plan = crate::inference_plan(&session, scan(), own_kind, 1);

    let engine = JammiExecutionEngine::new(Arc::clone(&session));
    let err = engine
        .create_query_stage_exec(
            "job-1".to_string().into(),
            0,
            0,
            plan,
            "/tmp",
            &SessionConfig::default(),
        )
        .expect_err(
            "this plan is not shuffle-writer-rooted, so DefaultExecutionEngine itself errors",
        );
    assert!(
        !matches!(JammiError::from(err), JammiError::DeviceKindUnheld { .. }),
        "a matching device kind must not be refused by the device-kind gate"
    );
}

/// The device-kind check also covers `PlacedAttemptExec` — its
/// descriptor's own stamped `device_kind` is compared against this
/// executor's kind the same way `InferenceExec`'s is. The test session's
/// default device is CPU, so a descriptor explicitly stamped `Cuda`
/// mismatches it.
#[tokio::test]
async fn refuses_a_placed_attempt_stage_whose_descriptor_names_a_different_device_kind() {
    let session = session().await;
    assert_eq!(
        session.compute_device().kind(),
        ComputeDeviceKind::Cpu,
        "test precondition: the fixture session runs on CPU"
    );
    let descriptor = jammi_ai::operator::placed_attempt_exec::PlacedAttempt {
        job_id: "job-1".to_string(),
        attempt: 0,
        submitter: "submitter-1".to_string(),
        device_kind: ComputeDeviceKind::Cuda,
        claimed_at: chrono::Utc::now(),
    };
    let plan: Arc<dyn ExecutionPlan> = Arc::new(PlacedAttemptExec::new(descriptor));

    let engine = JammiExecutionEngine::new(Arc::clone(&session));
    let err = engine
        .create_query_stage_exec(
            "job-1".to_string().into(),
            0,
            0,
            plan,
            "/tmp",
            &SessionConfig::default(),
        )
        .expect_err(
            "a device-kind mismatch on a PlacedAttemptExec must be refused, never silently run",
        );
    assert_device_kind_unheld(err);
}

/// A stage plan wrapping a `PlacedAttemptExec` under a
/// MULTI-partition node is refused typed — one attempt is one task, never a
/// multi-partition fan-out of its body. Two `PlacedAttemptExec` leaves
/// under a `UnionExec` (partition count 2, `PlacedAttemptExec` itself is always
/// single-partition) exercises the "under" wording literally: the refusal
/// looks at the STAGE's own output partitioning, not each leaf's.
#[tokio::test]
async fn refuses_a_placed_attempt_stage_with_more_than_one_partition() {
    let session = session().await;
    let descriptor = jammi_ai::operator::placed_attempt_exec::PlacedAttempt {
        job_id: "job-1".to_string(),
        attempt: 0,
        submitter: "submitter-1".to_string(),
        device_kind: ComputeDeviceKind::Cpu,
        claimed_at: chrono::Utc::now(),
    };
    let left: Arc<dyn ExecutionPlan> = Arc::new(PlacedAttemptExec::new(descriptor.clone()));
    let right: Arc<dyn ExecutionPlan> = Arc::new(PlacedAttemptExec::new(descriptor));
    let plan = datafusion::physical_plan::union::UnionExec::try_new(vec![left, right])
        .expect("two same-schema PlacedAttemptExec leaves union");
    assert_eq!(
        plan.properties().output_partitioning().partition_count(),
        2,
        "test precondition: the wrapping node is multi-partition"
    );

    let engine = JammiExecutionEngine::new(Arc::clone(&session));
    let err = engine
        .create_query_stage_exec(
            "job-1".to_string().into(),
            0,
            0,
            plan,
            "/tmp",
            &SessionConfig::default(),
        )
        .expect_err(
            "a multi-partition stage containing a PlacedAttemptExec must be refused, never run",
        );
    match JammiError::from(err) {
        JammiError::PlacedAttemptFanOut { job_id, partitions } => {
            assert_eq!(
                job_id, "job-1",
                "the descriptor's own job, never Ballista's"
            );
            assert_eq!(partitions, 2, "the partition count found");
        }
        other => panic!("expected PlacedAttemptFanOut, got {other:?}"),
    }
}
