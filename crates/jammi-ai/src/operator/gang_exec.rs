//! `GangExec` — the Ballista task a `Peer` gang runs as under placement.
//!
//! One task, single partition, zero children: `GangExec::execute` dispatches
//! to the process's installed [`crate::fine_tune::worker::PlacedGangRunner`]
//! and yields exactly one batch — `{ outcome: Utf8, artifact_digest: Utf8?
//! }` — or the runner's own error, never both.
//!
//! **Reaching the runner with no session in hand.** `GangExec::execute`
//! receives only a DataFusion `TaskContext` — on a Ballista executor this is
//! Ballista's own object, reconstructed on the executor side from the
//! scheduler's serialized `SessionConfig` (`ballista-core`'s task-exec
//! plumbing), never the submitter's `InferenceSession`. A `TaskContext`
//! session-config extension was the first candidate and was refused: an
//! extension set on the SUBMITTING session's `SessionConfig` never crosses
//! the wire to the EXECUTOR's reconstructed one — the two are different
//! processes' objects, connected only by the serialized plan bytes, and
//! Ballista's own decode never re-attaches a submitter-side extension. A
//! `jammi-server` process hosts exactly one [`crate::session::InferenceSession`]
//! (`OssServer::bind` builds one), so the runner this process installed is a
//! PROCESS-GLOBAL fact, not a hidden second source of truth — carried by
//! [`crate::fine_tune::worker::placed_gang_runner`], a single
//! `OnceLock<Weak<HostAdmission>>` set the one time this process's session
//! installs a [`crate::fine_tune::worker::PlacedGangRunner`]. See that
//! function's own doc for the refutation this shape survived.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    stream::RecordBatchReceiverStreamBuilder, DisplayAs, DisplayFormatType, ExecutionPlan,
    Partitioning, PlanProperties,
};
use jammi_db::store::manifest::ComputeDeviceKind;
use serde::{Deserialize, Serialize};

/// The coordinates of a placed gang's one Ballista task — everything
/// [`crate::fine_tune::worker::PlacedGangRunner::run`] needs to re-derive the
/// job from the catalog: no spec, no training-set identity travels here (the
/// executor reconstructs the whole run from the row, exactly as a claim
/// loop's own reclaim does).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GangDescriptor {
    pub job_id: String,
    pub attempt: u32,
    /// The job's own `world_size` — INFORMATIONAL only (the executor decides
    /// its own topology from this value and its OWN `[worker] local_ranks`
    /// when it runs `run_claimed_job_under`; `GangExec`'s codec/placement
    /// never reads it).
    pub world: u32,
    /// The submitting instance's id — [`crate::fine_tune::worker::
    /// WorkerJobError::Abandoned`]'s re-read compares the row's `claimed_by`
    /// against this, and [`Catalog::transfer_claim`](jammi_db::catalog::Catalog::transfer_claim)'s
    /// `$from` conjunct is this value verbatim.
    pub submitter: String,
    /// The device kind this gang must run on — the submitter's own
    /// [`crate::session::InferenceSession::compute_device`] kind, stamped at
    /// `JobWorker::submit_placed` (the device-kind rule
    /// this descriptor carries, the same one `InferenceExec::device_kind`
    /// carries: the required kind is the PLAN's own, never re-derived from
    /// "any GPU exists"). `placement::DevicePlacement` binds a `GangExec`
    /// stage only to an executor whose `compute_executors.devices` lists
    /// this exact kind; `JammiExecutionEngine`'s device-kind refusal compares it
    /// against the executing session's own kind the same way it does for
    /// `InferenceExec`.
    pub device_kind: ComputeDeviceKind,
}

/// What a placed gang's coordinator body ended as — the ONE thing `GangExec`
/// carries back across the Ballista wire (never a spec, never a row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacedOutcome {
    /// The attempt published `completed`; this is the SAME digest a member
    /// or an in-process `Peer` run would report (`adapter_files_digest`).
    Trained { artifact_digest: String },
    /// The attempt completed by reusing an already-published artifact of the
    /// same definition: no rank trained and no bytes were written, so there
    /// is no digest of this attempt's own.
    Reused,
    /// The attempt recorded a terminal `failed`, with this reason — a
    /// SUCCESSFUL run of the coordinator body that decided the job itself
    /// did not train, never a Ballista task fault.
    Failed { reason: String },
}

fn gang_output_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("outcome", DataType::Utf8, false),
        Field::new("artifact_digest", DataType::Utf8, true),
    ]))
}

fn outcome_batch(
    schema: &SchemaRef,
    outcome: &str,
    artifact_digest: Option<&str>,
) -> DfResult<RecordBatch> {
    let outcome_col: ArrayRef = Arc::new(StringArray::from(vec![outcome]));
    let digest_col: ArrayRef = Arc::new(StringArray::from(vec![artifact_digest]));
    Ok(RecordBatch::try_new(
        Arc::clone(schema),
        vec![outcome_col, digest_col],
    )?)
}

/// The placed gang's one Ballista task. See the module doc.
pub struct GangExec {
    descriptor: GangDescriptor,
    properties: Arc<PlanProperties>,
}

impl std::fmt::Debug for GangExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("GangExec")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl GangExec {
    /// One task, one partition — `Partitioning::UnknownPartitioning(1)`,
    /// bounded, incremental emission (a single terminal batch). Unconditional:
    /// this node is a leaf (no child `ExecutionPlan` — it drives the placed
    /// gang's coordinator/worker fleet directly via `descriptor`), so there is
    /// nothing to partition.
    pub fn new(descriptor: GangDescriptor) -> Self {
        let schema = gang_output_schema();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            descriptor,
            properties: Arc::new(properties),
        }
    }

    /// The descriptor this task runs.
    pub fn descriptor(&self) -> &GangDescriptor {
        &self.descriptor
    }
}

impl DisplayAs for GangExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "GangExec: job_id={}, attempt={}, world={}",
            self.descriptor.job_id, self.descriptor.attempt, self.descriptor.world
        )
    }
}

impl ExecutionPlan for GangExec {
    fn name(&self) -> &str {
        "GangExec"
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
                "GangExec has no children, got {}",
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
                "GangExec has exactly one partition (0), got {partition}"
            )));
        }
        let Some(runner) = crate::fine_tune::worker::placed_gang_runner() else {
            return Err(DataFusionError::Plan(
                "this process hosts no executor: no PlacedGangRunner is installed".into(),
            ));
        };
        let schema = self.schema();
        let descriptor = self.descriptor.clone();
        let job_id = descriptor.job_id.clone();
        let mut builder = RecordBatchReceiverStreamBuilder::new(schema.clone(), 1);
        let tx = builder.tx();
        builder.spawn(async move {
            match runner.run(descriptor).await {
                Ok(super::gang_exec::PlacedOutcome::Trained { artifact_digest }) => {
                    let batch = outcome_batch(&schema, "trained", Some(&artifact_digest))?;
                    tx.send(Ok(batch)).await.ok();
                    Ok(())
                }
                Ok(super::gang_exec::PlacedOutcome::Reused) => {
                    let batch = outcome_batch(&schema, "reused", None)?;
                    tx.send(Ok(batch)).await.ok();
                    Ok(())
                }
                Ok(super::gang_exec::PlacedOutcome::Failed { reason }) => {
                    tracing::warn!(job_id, reason, "placed gang ended failed");
                    let batch = outcome_batch(&schema, "failed", None)?;
                    tx.send(Ok(batch)).await.ok();
                    Ok(())
                }
                Err(e) => Err(DataFusionError::External(Box::new(e))),
            }
        });
        Ok(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fine_tune::worker::{HostAdmission, PlacedGangRunner};
    use datafusion::physical_plan::common::collect;
    use datafusion::prelude::SessionContext;
    use futures::future::BoxFuture;
    use jammi_db::catalog::instance::InstanceRegistration;
    use jammi_db::error::{JammiError, Result};
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn descriptor(job_id: &str) -> GangDescriptor {
        GangDescriptor {
            job_id: job_id.to_string(),
            attempt: 1,
            world: 2,
            submitter: "submitter-1".to_string(),
            device_kind: ComputeDeviceKind::Cpu,
        }
    }

    fn admission() -> Arc<HostAdmission> {
        HostAdmission::new(Arc::new(InstanceRegistration::new(
            "gang-exec-test",
            None,
            None,
            None,
            None,
        )))
    }

    /// One scripted result per `job_id`, consumed once (`test-hooks`-style
    /// registry — the same shape `training_test_hooks`'s `static ARMED`
    /// cells use — since [`PLACED_GANG_HOST`](crate::fine_tune::worker) is
    /// a genuine process-global that can only ever be installed once for
    /// the whole test binary: this ONE stub, scripted per descriptor by
    /// `job_id`, is what every scenario below drives, all from inside this
    /// ONE test function so their install order is deterministic.
    enum Script {
        Trained(String),
        Failed(String),
        Err(String),
    }

    fn scripts() -> &'static Mutex<HashMap<String, Script>> {
        static SCRIPTS: std::sync::OnceLock<Mutex<HashMap<String, Script>>> =
            std::sync::OnceLock::new();
        SCRIPTS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    struct StubRunner;

    impl PlacedGangRunner for StubRunner {
        fn run(&self, descriptor: GangDescriptor) -> BoxFuture<'static, Result<PlacedOutcome>> {
            let script = scripts().lock().unwrap().remove(&descriptor.job_id);
            Box::pin(async move {
                match script {
                    Some(Script::Trained(digest)) => Ok(PlacedOutcome::Trained {
                        artifact_digest: digest,
                    }),
                    Some(Script::Failed(reason)) => Ok(PlacedOutcome::Failed { reason }),
                    Some(Script::Err(msg)) => Err(JammiError::FineTune(msg)),
                    None => Err(JammiError::FineTune(format!(
                        "no script armed for job '{}'",
                        descriptor.job_id
                    ))),
                }
            })
        }
    }

    async fn run_plan(descriptor: GangDescriptor) -> DfResult<Vec<RecordBatch>> {
        let plan = GangExec::new(descriptor);
        let ctx = SessionContext::new();
        collect(plan.execute(0, ctx.task_ctx())?).await
    }

    /// The four GangExec-level oracles — sequenced in ONE test so the process-global
    /// runner's install order is deterministic (see [`Script`]'s doc):
    /// partition != 0 refuses BEFORE any runner is ever installed; no
    /// runner installed refuses typed; a `Trained` stub yields one batch
    /// with the digest; an `Err` stub's error is the stream's only item.
    #[tokio::test]
    async fn gang_exec_dispatches_through_the_process_global_runner_seam() {
        // Schema and partitioning, and the partition refusal — none of this
        // needs any runner at all.
        let plan = GangExec::new(descriptor("schema-job"));
        assert_eq!(plan.schema().field(0).name(), "outcome");
        assert!(!plan.schema().field(0).is_nullable());
        assert_eq!(plan.schema().field(1).name(), "artifact_digest");
        assert!(plan.schema().field(1).is_nullable());
        assert_eq!(plan.properties().output_partitioning().partition_count(), 1);
        let ctx = SessionContext::new();
        let err = match plan.execute(1, ctx.task_ctx()) {
            Ok(_) => panic!("partition 1 must refuse"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("exactly one partition"), "{err}");

        // No runner installed anywhere in this binary yet: a typed refusal.
        let no_runner_job = "no-runner-job";
        let err = run_plan(descriptor(no_runner_job))
            .await
            .expect_err("no runner installed must refuse");
        assert!(err.to_string().contains("no executor"), "{err}");

        // Install the ONE stub runner this whole test binary will ever see
        // for the placed-gang seam (the crate-global write-once fact).
        let admission = admission();
        assert!(admission.install_placed_gang_runner(Arc::new(StubRunner)));
        // A second install on the SAME instance is refused (write-once, the
        // `MemberDialer` shape).
        assert!(!admission.install_placed_gang_runner(Arc::new(StubRunner)));

        // A stub runner returning `Trained`: one batch, the digest carried.
        let trained_job = "trained-job";
        scripts()
            .lock()
            .unwrap()
            .insert(trained_job.to_string(), Script::Trained("deadbeef".into()));
        let batches = run_plan(descriptor(trained_job))
            .await
            .expect("a Trained runner must yield one batch");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let outcome = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(outcome.value(0), "trained");
        let digest = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(digest.value(0), "deadbeef");

        // A stub runner returning `Failed`: one batch, no digest.
        let failed_job = "failed-job";
        scripts()
            .lock()
            .unwrap()
            .insert(failed_job.to_string(), Script::Failed("no signal".into()));
        let batches = run_plan(descriptor(failed_job))
            .await
            .expect("a Failed runner must still yield one batch");
        assert_eq!(batches.len(), 1);
        let outcome = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(outcome.value(0), "failed");
        assert!(batches[0].column(1).is_null(0));

        // A stub runner returning `Err`: the stream's error is that error,
        // and no batch precedes it.
        let err_job = "err-job";
        scripts()
            .lock()
            .unwrap()
            .insert(err_job.to_string(), Script::Err("boom".into()));
        let err = run_plan(descriptor(err_job))
            .await
            .expect_err("an Err runner must surface as the stream's error");
        assert!(err.to_string().contains("boom"), "{err}");
    }
}
