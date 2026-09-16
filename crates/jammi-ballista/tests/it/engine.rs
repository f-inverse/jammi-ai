//! `JammiExecutionEngine` K7 device-kind refusal (contract `feat_500-wave4`
//! §9 B3).

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;

use ballista_executor::execution_engine::ExecutionEngine;

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::operator::inference_exec::InferenceExecBuilder;
use jammi_ai::session::InferenceSession;
use jammi_ballista::engine::JammiExecutionEngine;
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
    let node = InferenceExecBuilder::new(
        scan(),
        ModelSource::hf("m"),
        ModelTask::TextEmbedding,
        vec!["text".to_string()],
        "text".to_string(),
        "src-1".to_string(),
        Arc::clone(session.model_cache()),
    )
    .embedding_dim(Some(2))
    .device_kind(Some(ComputeDeviceKind::Cuda))
    .build()
    .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(node);

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
    let msg = err.to_string();
    assert!(
        msg.contains("K7"),
        "the refusal must name the property: {msg}"
    );
    assert!(msg.contains("Cuda") && msg.contains("Cpu"), "{msg}");
}

/// The matching-kind arm: a descriptor whose kind agrees with the executor's
/// own device is NOT refused by the K7 check (it proceeds to
/// `DefaultExecutionEngine`, which then fails for an unrelated reason — no
/// real `ShuffleWriterExec` wraps this plan — proving the K7 gate itself let
/// it through rather than raising a false positive).
#[tokio::test]
async fn does_not_refuse_a_matching_device_kind() {
    let session = session().await;
    let own_kind = session.compute_device().kind();
    let node = InferenceExecBuilder::new(
        scan(),
        ModelSource::hf("m"),
        ModelTask::TextEmbedding,
        vec!["text".to_string()],
        "text".to_string(),
        "src-1".to_string(),
        Arc::clone(session.model_cache()),
    )
    .embedding_dim(Some(2))
    .device_kind(Some(own_kind))
    .build()
    .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(node);

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
        !err.to_string().contains("K7"),
        "a matching device kind must not be refused by the K7 gate: {err}"
    );
}
