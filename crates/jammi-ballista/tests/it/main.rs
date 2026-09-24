mod cluster;
mod codec;
mod engine;
mod roles;

use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use jammi_ai::session::InferenceSession;
use jammi_datafusion::ComputeDeviceKind;
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;
use jammi_datafusion::RowOrder;
use jammi_datafusion::{plan_inference, InferenceSpec};
use jammi_numerics::ChunkBudget;

/// The spec of a text embedding over a scan's `text` column, keyed by it too,
/// placed on `device_kind` at a fan-out of `partitions`.
fn text_embedding_spec(device_kind: ComputeDeviceKind, partitions: usize) -> InferenceSpec {
    InferenceSpec {
        source: ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
        task: ModelTask::TextEmbedding,
        content_columns: vec!["text".to_string()],
        key_column: "text".to_string(),
        source_id: "src-1".to_string(),
        chunk: ChunkBudget {
            rows: NonZeroUsize::new(8).expect("8 is non-zero"),
            tokens: NonZeroUsize::new(4096).expect("4096 is non-zero"),
        },
        embedding_dim: Some(4),
        regression_form: None,
        passthrough: Vec::new(),
        device_kind,
        partitions: NonZeroUsize::new(partitions).expect("a fan-out is at least one"),
    }
}

/// The production inference plan over `scan`, in arrival order. The plan is
/// planned, encoded and staged by these tests, never executed, so the model
/// it names is never loaded.
fn inference_plan(
    session: &InferenceSession,
    scan: Arc<dyn ExecutionPlan>,
    device_kind: ComputeDeviceKind,
    partitions: usize,
) -> Arc<dyn ExecutionPlan> {
    plan_inference(
        scan,
        RowOrder::Arrival,
        text_embedding_spec(device_kind, partitions),
        session.inference_runtime(),
    )
    .expect("the inference plan builds")
}
