//! Where a materialization's plan runs: on the compute plane when this
//! process holds the client role and a live executor can hold the plan, in
//! this process otherwise.
//!
//! [`ComputePlane`] is the one submit seam: this crate declares it, the
//! compute-plane crate implements it over its submit client, and the server
//! installs it on the session through [`ComputePlaneSlot::install`] when
//! the process is configured as a client — the engine never depends on the
//! implementation. Every submission a process makes — a materialization's
//! plan, a claimed gang's one task — goes through it. The slot rides in the
//! session's `SessionConfig` as an extension, so it reaches every plan
//! executed under a context derived from the session's — a per-request
//! Flight SQL state, a single-partition derivation — through the
//! `TaskContext` alone, and a context a caller built for itself carries no
//! slot and never routes.
//!
//! [`StatementClass`] states, in one place, which SQL statements are
//! materializations and why; [`MaterializationExec`] is the one physical
//! node through which a materialization's read/compute plan is executed —
//! the statement's, through [`MaterializationNode`] and its planner, and a
//! verb's, through [`execute_materialization`]. The plan beneath the node
//! is the plan the in-process run executes: nothing is rewritten for the
//! trip, the node only decides where its child runs. What a
//! materialization writes — a `MemTable` the process registers, a result
//! table the store writes, a companion table's sink — stays this process's:
//! the compute plane runs the read/compute part and streams its rows back.

use std::fmt;
use std::sync::{Arc, OnceLock};

use datafusion::common::{DFSchemaRef, Result as DfResult};
use datafusion::error::DataFusionError;
use datafusion::execution::context::{SessionState, TaskContext};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{
    CreateMemoryTable, DdlStatement, Expr, Extension, LogicalPlan, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execute_stream, DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties,
    Partitioning, PlanProperties,
};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use futures::future::BoxFuture;
use futures::TryStreamExt;

use crate::error::{JammiError, Result};
use crate::store::manifest::ComputeDeviceKind;

/// A compute plane a physical plan is submitted to: the plan's stages run
/// on the plane's executors and its output streams back. The plan crosses
/// as it is — the same operators the in-process run would execute —
/// and a failure on the plane reaches the caller as the same typed error
/// the in-process run would raise. Admission and placement are two verbs
/// so a caller that must run an unheld plan somewhere else — a
/// materialization in this process, a gang in this claimant's own body —
/// decides that BEFORE anything crosses the wire, and can tell a refusal
/// from a submission's failure.
pub trait ComputePlane: Send + Sync {
    /// Why the plane cannot hold `plan` right now, or `None` when a live
    /// executor can: the plan's [`PlanRequirements`] against the live
    /// inventory, decided before any task is scheduled.
    fn unheld(&self, plan: &Arc<dyn ExecutionPlan>) -> BoxFuture<'static, Result<Option<Unheld>>>;

    /// Submit `plan`; the stream is the plan's own output, its failure the
    /// plan's own typed error.
    fn place(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> BoxFuture<'static, Result<SendableRecordBatchStream>>;
}

/// What a plan asks of the executor that holds it — read off the plan's
/// own nodes, never off the process submitting it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PlanRequirements {
    /// The device kind a node of the plan is stamped with, if any (an
    /// inference's, a gang's — CPU is a kind too).
    pub device_kind: Option<ComputeDeviceKind>,
    /// An executor the plan must not land on: a gang's own submitter, whose
    /// process holds the claim for the whole await and would deadlock
    /// against its own task.
    pub excluded_executor: Option<String>,
}

/// Why the compute plane cannot hold a plan right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unheld {
    /// No registered executor is live.
    NoLiveExecutor,
    /// No live executor lists a device of the kind the plan requires. The
    /// fields are [`JammiError::DeviceKindUnheld`]'s, the error a submit
    /// that must not fall back raises from this refusal.
    NoExecutorOfKind {
        /// The kind the plan requires.
        required: ComputeDeviceKind,
        /// Every kind a live executor lists, distinct and in wire order.
        held: Vec<ComputeDeviceKind>,
    },
    /// The only live executor that could hold the plan is the one the plan
    /// excludes — its own submitter.
    OnlyTheSubmitter {
        /// The excluded executor's id.
        submitter: String,
    },
}

impl fmt::Display for Unheld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLiveExecutor => f.write_str("no live registered compute executor"),
            Self::NoExecutorOfKind { required, held } => {
                let unheld = JammiError::DeviceKindUnheld {
                    required: *required,
                    held: held.clone(),
                };
                write!(f, "{unheld}: no live registered compute executor lists it")
            }
            Self::OnlyTheSubmitter { submitter } => write!(
                f,
                "the only live registered compute executor able to hold the plan is its own \
                 submitter {submitter}"
            ),
        }
    }
}

/// The session's write-once slot for its [`ComputePlane`]. Empty on a
/// process holding no client role — every materialization runs in-process.
#[derive(Default)]
pub struct ComputePlaneSlot(OnceLock<Arc<dyn ComputePlane>>);

impl fmt::Debug for ComputePlaneSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePlaneSlot")
            .field("installed", &self.0.get().is_some())
            .finish()
    }
}

impl ComputePlaneSlot {
    /// Install the process's [`ComputePlane`] — once. `false` when one is
    /// already installed (the first stays).
    pub fn install(&self, plane: Arc<dyn ComputePlane>) -> bool {
        self.0.set(plane).is_ok()
    }

    /// The installed plane, if this process holds the client role.
    pub fn plane(&self) -> Option<Arc<dyn ComputePlane>> {
        self.0.get().cloned()
    }
}

/// Which side of the route a SQL statement is on, decided by the root of
/// its logical plan — the one place the class is stated:
///
/// - `CREATE TABLE … AS <query>` is a [`Self::Materialization`]: the
///   query's rows are collected into a table this process registers and no
///   caller waits on a row, so the query beneath is a materialization's
///   read/compute part.
/// - Everything else is [`Self::Inline`], running where it was issued: a
///   query's rows stream to the caller as they are produced; `INSERT`,
///   `UPDATE` and `DELETE` write a mutable companion table on this
///   process's catalog backend from the statement's own rows; `COPY … TO`
///   writes a file through this process's own store registry; the
///   remaining DDL, `SET`, `EXPLAIN` and transaction statements compute
///   nothing.
pub enum StatementClass {
    /// `CREATE TABLE … AS`, with the statement it came from.
    Materialization(CreateMemoryTable),
    /// Any other statement, as it was planned.
    Inline(LogicalPlan),
}

impl StatementClass {
    /// Classify `plan`, the logical plan of one statement.
    pub fn of(plan: LogicalPlan) -> Self {
        match plan {
            LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(cmd)) => Self::Materialization(cmd),
            other => Self::Inline(other),
        }
    }

    /// The plan the engine executes: a materialization's query wrapped in
    /// [`MaterializationNode`], so its physical plan roots in
    /// [`MaterializationExec`] and DataFusion's own statement execution
    /// writes the rows the node hands it; an inline statement unchanged.
    pub fn into_plan(self) -> LogicalPlan {
        match self {
            Self::Materialization(cmd) => {
                let input = MaterializationNode::new(Arc::unwrap_or_clone(cmd.input));
                LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(CreateMemoryTable {
                    input: Arc::new(LogicalPlan::Extension(Extension {
                        node: Arc::new(input),
                    })),
                    ..cmd
                }))
            }
            Self::Inline(plan) => plan,
        }
    }
}

/// The logical node marking a statement's query as a materialization's
/// read/compute part: pass-through in every respect (its input's schema,
/// no expressions of its own), planned into [`MaterializationExec`] by
/// [`MaterializationPlanner`].
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd)]
pub struct MaterializationNode {
    input: LogicalPlan,
}

impl MaterializationNode {
    /// Mark `input` as the read/compute part of a materialization.
    pub fn new(input: LogicalPlan) -> Self {
        Self { input }
    }
}

impl UserDefinedLogicalNodeCore for MaterializationNode {
    fn name(&self) -> &str {
        "Materialization"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Materialization")
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> DfResult<Self> {
        let [input] = <[LogicalPlan; 1]>::try_from(inputs).map_err(|inputs| {
            DataFusionError::Internal(format!(
                "Materialization has exactly one input, got {}",
                inputs.len()
            ))
        })?;
        Ok(Self { input })
    }
}

/// Plans [`MaterializationNode`] into [`MaterializationExec`] over its
/// input's physical plan; every other extension node is left to the next
/// planner.
#[derive(Debug, Default)]
pub struct MaterializationPlanner;

#[async_trait::async_trait]
impl ExtensionPlanner for MaterializationPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
        if node
            .as_any()
            .downcast_ref::<MaterializationNode>()
            .is_none()
        {
            return Ok(None);
        }
        let [input] = physical_inputs else {
            return Err(DataFusionError::Internal(format!(
                "Materialization has exactly one input, got {}",
                physical_inputs.len()
            )));
        };
        Ok(Some(Arc::new(MaterializationExec::new(Arc::clone(input)))))
    }
}

/// Execute `plan`, a materialization's read/compute part, where the
/// compute plane says: [`MaterializationExec`] over `plan`, executed under
/// `context`. The one call every verb that materializes a plan makes.
pub fn execute_materialization(
    plan: Arc<dyn ExecutionPlan>,
    context: Arc<TaskContext>,
) -> DfResult<SendableRecordBatchStream> {
    MaterializationExec::new(plan).execute(0, context)
}

/// The physical node a materialization's read/compute plan roots in. Its
/// one output partition is the child's whole output: submitted to the
/// [`ComputePlane`] the `TaskContext`'s session config carries when one is
/// installed and holds the child (the stream is the child's own, its
/// failure the typed error the in-process run would raise), executed in
/// this process otherwise — a context carrying no plane, a plane that
/// refuses the child unheld. The decision is made when the stream is
/// first polled, never at plan time: the executor inventory is read then.
#[derive(Debug)]
pub struct MaterializationExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl MaterializationExec {
    /// Root `input` in a materialization node.
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Self { input, properties }
    }

    /// The plan this node runs: the materialization's read/compute part,
    /// as the in-process run would execute it.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

impl DisplayAs for MaterializationExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MaterializationExec")
    }
}

impl ExecutionPlan for MaterializationExec {
    fn name(&self) -> &str {
        "MaterializationExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let [input] = <[Arc<dyn ExecutionPlan>; 1]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "MaterializationExec has exactly one child, got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self::new(input)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "MaterializationExec has one output partition, partition {partition} was asked for"
            )));
        }
        let plane = context
            .session_config()
            .get_extension::<ComputePlaneSlot>()
            .and_then(|slot| slot.plane());
        let input = Arc::clone(&self.input);
        let schema = input.schema();
        let stream = futures::stream::once(async move {
            let Some(plane) = plane else {
                return execute_stream(input, context);
            };
            let typed = |e: JammiError| DataFusionError::External(Box::new(e));
            if let Some(why) = plane.unheld(&input).await.map_err(typed)? {
                tracing::info!(
                    reason = %why,
                    "materialization runs in this process: the compute plane cannot hold it"
                );
                return execute_stream(input, context);
            }
            let stream = plane.place(input).await.map_err(typed)?;
            tracing::info!("materialization placed on the compute plane");
            Ok(stream)
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::physical_plan::collect;
    use datafusion::prelude::{SessionConfig, SessionContext};

    use super::*;

    fn rows() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("title", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap()
    }

    /// A context over one table `t` — a caller's own context, carrying the
    /// slot only when `slot` says so.
    fn context(slot: Option<Arc<ComputePlaneSlot>>) -> SessionContext {
        let config = match slot {
            Some(slot) => SessionConfig::new().with_extension(slot),
            None => SessionConfig::new(),
        };
        let ctx = SessionContext::new_with_config(config);
        let batch = rows();
        ctx.register_table(
            "t",
            Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap()),
        )
        .unwrap();
        ctx
    }

    /// The class of every statement shape the SQL surface takes, and the
    /// one that routes: `CREATE TABLE … AS` alone.
    #[tokio::test]
    async fn the_class_table_routes_a_create_table_as_and_nothing_else() {
        let ctx = context(None);
        let table: &[(&str, bool)] = &[
            ("CREATE TABLE u AS SELECT id, title FROM t", true),
            (
                "CREATE TABLE IF NOT EXISTS u AS SELECT id FROM t WHERE id > 1",
                true,
            ),
            ("SELECT id, title FROM t", false),
            ("SELECT count(*) FROM t", false),
            ("SELECT * FROM t ORDER BY id LIMIT 1", false),
            ("INSERT INTO t VALUES (4, 'd')", false),
            ("INSERT INTO t SELECT id + 10, title FROM t", false),
            ("EXPLAIN SELECT id FROM t", false),
            ("CREATE VIEW v AS SELECT id FROM t", false),
            ("SET datafusion.execution.batch_size = 8", false),
            ("DROP TABLE IF EXISTS u", false),
        ];
        for (sql, materialization) in table {
            let plan = ctx.state().create_logical_plan(sql).await.unwrap();
            let class = StatementClass::of(plan);
            assert_eq!(
                matches!(class, StatementClass::Materialization(_)),
                *materialization,
                "{sql}"
            );
            let routed = class.into_plan();
            let rooted = match &routed {
                LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(cmd)) => matches!(
                    cmd.input.as_ref(),
                    LogicalPlan::Extension(Extension { node })
                        if node.as_any().downcast_ref::<MaterializationNode>().is_some()
                ),
                _ => false,
            };
            assert_eq!(
                rooted, *materialization,
                "{sql}: the query is rooted iff routed"
            );
        }
    }

    /// What a fake plane answers a submission with.
    #[derive(Clone, Copy)]
    enum Answer {
        /// Held: the plane runs the plan under its own context, the way an
        /// executor runs it under its own session.
        Placed,
        /// Refused before anything is submitted.
        Unheld,
        /// Held, and the placement fails typed.
        Failing,
    }

    /// A plane that answers as told and counts the plans it was asked to
    /// hold and the plans it placed.
    struct FakePlane {
        answer: Answer,
        asked: AtomicUsize,
        placed: AtomicUsize,
    }

    impl FakePlane {
        fn new(answer: Answer) -> Arc<Self> {
            Arc::new(Self {
                answer,
                asked: AtomicUsize::new(0),
                placed: AtomicUsize::new(0),
            })
        }
    }

    impl ComputePlane for FakePlane {
        fn unheld(
            &self,
            _plan: &Arc<dyn ExecutionPlan>,
        ) -> BoxFuture<'static, Result<Option<Unheld>>> {
            self.asked.fetch_add(1, Ordering::SeqCst);
            let why = matches!(self.answer, Answer::Unheld).then(|| Unheld::NoExecutorOfKind {
                required: ComputeDeviceKind::Cuda,
                held: vec![ComputeDeviceKind::Cpu],
            });
            Box::pin(async move { Ok(why) })
        }

        fn place(
            &self,
            plan: Arc<dyn ExecutionPlan>,
        ) -> BoxFuture<'static, Result<SendableRecordBatchStream>> {
            self.placed.fetch_add(1, Ordering::SeqCst);
            let answer = self.answer;
            Box::pin(async move {
                match answer {
                    Answer::Placed | Answer::Unheld => {
                        execute_stream(plan, SessionContext::new().task_ctx())
                            .map_err(JammiError::from)
                    }
                    Answer::Failing => Err(JammiError::SourceNotFound {
                        source_id: "patents".into(),
                    }),
                }
            })
        }
    }

    async fn scan(ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
        ctx.sql("SELECT id, title FROM t ORDER BY id")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    /// The node's three routes: placed (the plane's stream, the plan
    /// asked about once and placed once), unheld (in-process, the plan
    /// asked about once and never placed — the refusal is the plane's),
    /// and no plane installed (in-process, nothing asked). The rows are
    /// the same on every route.
    #[tokio::test]
    async fn the_node_runs_its_child_where_the_plane_says() {
        for (answer, expect_placed) in [(Answer::Placed, 1), (Answer::Unheld, 0)] {
            let plane = FakePlane::new(answer);
            let slot = Arc::new(ComputePlaneSlot::default());
            assert!(slot.install(plane.clone()));
            assert!(!slot.install(plane.clone()), "write-once");
            let ctx = context(Some(slot));
            let plan = scan(&ctx).await;
            let out = collect(Arc::new(MaterializationExec::new(plan)), ctx.task_ctx())
                .await
                .unwrap();
            assert_eq!(out, vec![rows()]);
            assert_eq!(plane.asked.load(Ordering::SeqCst), 1);
            assert_eq!(plane.placed.load(Ordering::SeqCst), expect_placed);
        }

        let ctx = context(None);
        let plan = scan(&ctx).await;
        let out = collect(Arc::new(MaterializationExec::new(plan)), ctx.task_ctx())
            .await
            .unwrap();
        assert_eq!(out, vec![rows()]);

        let empty = Arc::new(ComputePlaneSlot::default());
        let ctx = context(Some(empty));
        let plan = scan(&ctx).await;
        let out = collect(Arc::new(MaterializationExec::new(plan)), ctx.task_ctx())
            .await
            .unwrap();
        assert_eq!(out, vec![rows()]);
    }

    /// A submission's failure is the statement's, typed: the engine's
    /// classifier restores the same variant and fields from the stream.
    #[tokio::test]
    async fn a_placed_failure_reaches_the_caller_typed() {
        let slot = Arc::new(ComputePlaneSlot::default());
        slot.install(FakePlane::new(Answer::Failing));
        let ctx = context(Some(slot));
        let plan = scan(&ctx).await;
        let err = collect(Arc::new(MaterializationExec::new(plan)), ctx.task_ctx())
            .await
            .expect_err("the plane's failure is the statement's");
        match JammiError::from(err) {
            JammiError::SourceNotFound { source_id } => assert_eq!(source_id, "patents"),
            other => panic!("expected SourceNotFound, got {other:?}"),
        }
    }
}
