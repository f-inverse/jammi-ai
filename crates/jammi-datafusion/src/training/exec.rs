//! [`TrainingExec`]: a claimed training job as one task, run by the
//! [`TrainingRunner`] the node was bound to.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    stream::RecordBatchReceiverStreamBuilder, DisplayAs, DisplayFormatType, ExecutionPlan,
    Partitioning, PlanProperties,
};

use crate::device::ComputeDeviceKind;
use crate::error::{Error, Result};

/// A claimed training job's coordinates — everything a [`TrainingRunner`]
/// needs to take over the claim and re-derive the run from wherever the job
/// is recorded. No spec, kind, rank count or training set travels here: the
/// runner reconstructs the whole run from the record, and decides its
/// topology from the record's spec and its own process's ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingJob {
    /// The job.
    pub job_id: String,
    /// The claim's attempt number.
    pub attempt: u32,
    /// Who claimed the attempt and submitted this stage — the claim a runner
    /// takes over from.
    pub submitter: String,
    /// The device kind this job must run on: the submitter's own, stamped
    /// where it builds the stage. A placement binds the stage only to an
    /// executor holding this exact kind, the same kind match an
    /// [`InferenceExec`](crate::InferenceExec) carries.
    pub device_kind: ComputeDeviceKind,
    /// When the submitter claimed the attempt, on its own clock — the first
    /// station of the job's timeline, which no other process could stamp.
    pub claimed_at: chrono::DateTime<chrono::Utc>,
}

/// How a training job completed — the one thing a [`TrainingExec`] carries
/// back as a row. A job that failed is the runner's `Err`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingOutcome {
    /// The job trained and published an artifact with this digest.
    Trained {
        /// The published artifact's digest.
        artifact_digest: String,
    },
    /// The job completed by reusing an already-published artifact of the
    /// same definition: nothing trained and no bytes were written, so there
    /// is no digest of this job's own.
    Reused,
}

/// Runs a claimed training job on this process: the seam a consumer
/// implements with its own claim transfer, training loop and publish.
///
/// A failure is carried as [`Error::runtime`], so a consumer that knows the
/// runner's error type restores it from the stream's error.
#[async_trait]
pub trait TrainingRunner: Send + Sync {
    /// Run `job` to its outcome.
    async fn run(&self, job: TrainingJob) -> Result<TrainingOutcome>;
}

/// The runner a process that runs no training binds — a scheduler that
/// decodes plans but executes none, or a submitter that builds the stage
/// for another process to run: every job is refused with
/// [`Error::NoTrainingRunner`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTrainingRunner;

#[async_trait]
impl TrainingRunner for NoTrainingRunner {
    async fn run(&self, _job: TrainingJob) -> Result<TrainingOutcome> {
        Err(Error::NoTrainingRunner)
    }
}

fn outcome_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("outcome", DataType::Utf8, false),
        Field::new("artifact_digest", DataType::Utf8, true),
    ]))
}

fn outcome_batch(schema: &SchemaRef, outcome: &TrainingOutcome) -> DfResult<RecordBatch> {
    let (label, digest) = match outcome {
        TrainingOutcome::Trained { artifact_digest } => ("trained", Some(artifact_digest.as_str())),
        TrainingOutcome::Reused => ("reused", None),
    };
    let label: ArrayRef = Arc::new(StringArray::from(vec![label]));
    let digest: ArrayRef = Arc::new(StringArray::from(vec![digest]));
    Ok(RecordBatch::try_new(
        Arc::clone(schema),
        vec![label, digest],
    )?)
}

/// A claimed training job as one task: a leaf with a single partition that
/// runs its [`TrainingJob`] through the [`TrainingRunner`] it was bound to
/// and yields one row — `{ outcome: Utf8, artifact_digest: Utf8? }` — or the
/// runner's error, never both.
///
/// The runner is bound at construction, as an
/// [`InferenceExec`](crate::InferenceExec) binds its runtime: a codec's
/// decode binds the decoding process's own.
pub struct TrainingExec {
    job: TrainingJob,
    runner: Arc<dyn TrainingRunner>,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for TrainingExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainingExec")
            .field("job", &self.job)
            .finish_non_exhaustive()
    }
}

impl TrainingExec {
    /// `job` as one task, run by `runner`.
    pub fn new(job: TrainingJob, runner: Arc<dyn TrainingRunner>) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(outcome_schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            job,
            runner,
            properties: Arc::new(properties),
        }
    }

    /// The job this task runs.
    pub fn job(&self) -> &TrainingJob {
        &self.job
    }
}

impl DisplayAs for TrainingExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "TrainingExec: job_id={}, attempt={}, device_kind={}",
            self.job.job_id,
            self.job.attempt,
            self.job.device_kind.wire_str()
        )
    }
}

impl ExecutionPlan for TrainingExec {
    fn name(&self) -> &str {
        "TrainingExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "TrainingExec has no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Plan(format!(
                "TrainingExec has exactly one partition (0), got {partition}"
            )));
        }
        let schema = self.schema();
        let job = self.job.clone();
        let runner = Arc::clone(&self.runner);
        let mut builder = RecordBatchReceiverStreamBuilder::new(Arc::clone(&schema), 1);
        let tx = builder.tx();
        builder.spawn(async move {
            let outcome = runner.run(job).await.map_err(Error::into_df)?;
            tx.send(outcome_batch(&schema, &outcome)).await.ok();
            Ok(())
        });
        Ok(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use datafusion::physical_plan::common::collect;
    use datafusion::prelude::SessionContext;

    fn job(job_id: &str) -> TrainingJob {
        TrainingJob {
            job_id: job_id.to_string(),
            attempt: 1,
            submitter: "submitter-1".to_string(),
            device_kind: ComputeDeviceKind::Cpu,
            claimed_at: chrono::Utc::now(),
        }
    }

    /// A runner that answers every job with one scripted result.
    struct Scripted(std::sync::Mutex<Option<Result<TrainingOutcome>>>);

    #[async_trait]
    impl TrainingRunner for Scripted {
        async fn run(&self, _job: TrainingJob) -> Result<TrainingOutcome> {
            self.0
                .lock()
                .expect("the script is never poisoned")
                .take()
                .expect("each scripted runner runs once")
        }
    }

    async fn run(job: TrainingJob, runner: Arc<dyn TrainingRunner>) -> DfResult<Vec<RecordBatch>> {
        let plan = TrainingExec::new(job, runner);
        collect(plan.execute(0, SessionContext::new().task_ctx())?).await
    }

    fn strings(batch: &RecordBatch, column: usize) -> &StringArray {
        batch
            .column(column)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("the outcome columns are strings")
    }

    /// One task, one partition, the outcome schema; any other partition is
    /// refused.
    #[test]
    fn a_training_stage_is_one_partition_of_the_outcome_schema() {
        let plan = TrainingExec::new(job("schema-job"), Arc::new(NoTrainingRunner));
        assert_eq!(plan.schema().field(0).name(), "outcome");
        assert!(!plan.schema().field(0).is_nullable());
        assert_eq!(plan.schema().field(1).name(), "artifact_digest");
        assert!(plan.schema().field(1).is_nullable());
        assert_eq!(plan.properties().output_partitioning().partition_count(), 1);
        let Err(err) = plan.execute(1, SessionContext::new().task_ctx()) else {
            panic!("partition 1 must refuse");
        };
        assert!(err.to_string().contains("exactly one partition"), "{err}");
    }

    /// A process that runs no training refuses the job typed.
    #[tokio::test]
    async fn a_process_with_no_runner_refuses_the_job_typed() {
        let err = run(job("no-runner-job"), Arc::new(NoTrainingRunner))
            .await
            .expect_err("a process with no runner must refuse");
        assert!(
            matches!(Error::found_in(&err), Some(Error::NoTrainingRunner)),
            "{err}"
        );
    }

    /// A trained job yields one row carrying its digest; a reused one, one
    /// row with none.
    #[tokio::test]
    async fn a_completed_job_yields_one_row_of_its_outcome() {
        let trained = Scripted(std::sync::Mutex::new(Some(Ok(TrainingOutcome::Trained {
            artifact_digest: "deadbeef".into(),
        }))));
        let batches = run(job("trained-job"), Arc::new(trained)).await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(strings(&batches[0], 0).value(0), "trained");
        assert_eq!(strings(&batches[0], 1).value(0), "deadbeef");

        let reused = Scripted(std::sync::Mutex::new(Some(Ok(TrainingOutcome::Reused))));
        let batches = run(job("reused-job"), Arc::new(reused)).await.unwrap();
        assert_eq!(strings(&batches[0], 0).value(0), "reused");
        assert!(strings(&batches[0], 1).is_null(0));
    }

    /// A failed job's error is the stream's only item, restorable as the
    /// runner raised it.
    #[tokio::test]
    async fn a_failed_job_is_the_streams_only_item_as_the_runner_raised_it() {
        #[derive(Debug, thiserror::Error)]
        #[error("source '{0}' not found")]
        struct SourceNotFound(String);

        let failed = Scripted(std::sync::Mutex::new(Some(Err(Error::runtime(
            SourceNotFound("pairs".into()),
        )))));
        let err = run(job("failed-job"), Arc::new(failed))
            .await
            .expect_err("a failed job must surface as the stream's error");
        let Some(Error::Runtime(source)) = Error::found_in(&err) else {
            panic!("expected the runner's own error, got {err}");
        };
        let restored = source
            .downcast_ref::<SourceNotFound>()
            .expect("the runner's error type is restorable");
        assert_eq!(restored.0, "pairs");
    }
}
