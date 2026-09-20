//! Where a materialization runs: on the compute plane when this process
//! holds the client role and a live executor can hold the plan, in this
//! process otherwise.
//!
//! [`ComputePlane`] is the one submit seam: this crate declares it, the
//! compute-plane crate implements it over its submit client, and the server
//! installs it on the session through [`ComputePlaneSlot::install`] when
//! the process is configured as a client — the engine never depends on the
//! implementation. Every submission a process makes — a result table's
//! sink over its compute, a claimed gang's one task — goes through it. The
//! slot rides in the session's `SessionConfig` as an extension, so it
//! reaches every plan executed under a context derived from the session's
//! — a per-request Flight SQL state, a single-partition derivation —
//! through the `TaskContext` alone, and a context a caller built for itself
//! carries no slot and never routes.
//!
//! [`StatementClass`] states, in one place, which SQL statements the
//! result store executes and how: a `CREATE TABLE … AS` is a result-table
//! materialization rooted in [`crate::store::ResultTableSinkExec`] — the
//! same node every verb that materializes a result table roots in — and a
//! `DROP TABLE` is the store's drop of one. [`MaterializationPlanner`]
//! plans the statement's node into [`StoreStatementExec`], which runs the
//! store's verb under the session that planned it.

use std::fmt;
use std::sync::{Arc, OnceLock};

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, DFSchemaRef, Result as DfResult, TableReference};
use datafusion::error::DataFusionError;
use datafusion::execution::context::{SessionState, TaskContext};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{
    CreateMemoryTable, DdlStatement, Expr, Extension, LogicalPlan, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::SessionContext;
use futures::future::BoxFuture;
use futures::TryStreamExt;

use crate::error::{JammiError, Result};
use crate::session::QueryContext;
use crate::store::manifest::ComputeDeviceKind;
use crate::store::statement::CreateTableAs;
use crate::store::ResultStore;

/// A compute plane a physical plan is submitted to: the plan's stages run
/// on the plane's executors and its output streams back. The plan crosses
/// as it is — the same operators the in-process run would execute —
/// and a failure on the plane reaches the caller as the same typed error
/// the in-process run would raise. Admission and placement are two verbs
/// so a caller that must run an unheld plan somewhere else — a result
/// table's sink in this process, a gang in this claimant's own body —
/// decides that BEFORE anything crosses the wire, and can tell a refusal
/// from a submission's failure.
pub trait ComputePlane: Send + Sync {
    /// Why the plane cannot hold `plan` right now, or `None` when a live
    /// executor can: the plan's [`PlanRequirements`] against the live
    /// inventory, and whether the wire carries the plan at all, decided
    /// before any task is scheduled.
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
    /// No live executor lists a device of the kind the plan requires.
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
    /// The plan carries a node the wire cannot carry — a stream of this
    /// process's own rows, a handle only this process holds — so no other
    /// process can hold it.
    NotEncodable {
        /// The codec's own account of what it could not carry.
        detail: String,
    },
}

impl fmt::Display for Unheld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLiveExecutor => f.write_str("no live registered compute executor"),
            Self::NoExecutorOfKind { required, held } => {
                let held = held
                    .iter()
                    .map(|k| k.wire_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "no live registered compute executor lists a {} device (held: [{held}])",
                    required.wire_str()
                )
            }
            Self::OnlyTheSubmitter { submitter } => write!(
                f,
                "the only live registered compute executor able to hold the plan is its own \
                 submitter {submitter}"
            ),
            Self::NotEncodable { detail } => {
                write!(f, "the plan carries a node the wire cannot carry: {detail}")
            }
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

/// How a SQL statement is executed, decided by the root of its logical
/// plan — the one place the class is stated:
///
/// - `CREATE TABLE <name> AS <query>` is a [`Self::CreateTableAs`]: a
///   result table named `<name>` — bytes on the object store under the
///   store's root, a `result_tables` row, read on every replica through
///   the store's binding as `"jammi.<name>"` — materialized through
///   [`crate::store::ResultTableSinkExec`] over the query, where the
///   compute plane says. Never an in-memory table one replica holds.
/// - `DROP TABLE <name>` is a [`Self::DropTable`]: the store's drop of the
///   result table named `<name>` under the tenant in force
///   ([`ResultStore::drop_result_table`]).
/// - `CREATE TABLE <name> (<columns>)` — no query — is [`Self::Refused`]:
///   a result table is what a query produced; there are no empty ones. A
///   qualified name (`CREATE TABLE a.b AS`) is refused too: a result table
///   is named by one identifier.
/// - Everything else is [`Self::Inline`], running where it was issued: a
///   query's rows stream to the caller as they are produced; `INSERT`,
///   `UPDATE` and `DELETE` write a mutable companion table on this
///   process's catalog backend from the statement's own rows; `COPY … TO`
///   writes a file through this process's own store registry; the
///   remaining DDL, `SET`, `EXPLAIN` and transaction statements compute
///   nothing.
pub enum StatementClass {
    /// `CREATE TABLE … AS <query>`, with the query it materializes.
    CreateTableAs {
        /// What the statement asks for.
        statement: CreateTableAs,
        /// The query.
        input: LogicalPlan,
    },
    /// `DROP TABLE`.
    DropTable {
        /// The result table's name.
        name: String,
        /// `IF EXISTS`: an absent table is not an error.
        if_exists: bool,
    },
    /// A statement the store refuses, typed at planning.
    Refused(RefusedStatement),
    /// Any other statement, as it was planned.
    Inline(LogicalPlan),
}

/// A statement [`StatementClass`] refuses, and why.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct RefusedStatement {
    /// The table the statement named.
    pub name: String,
    /// Why.
    pub reason: RefusalReason,
}

/// Why a statement is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub enum RefusalReason {
    /// `CREATE TABLE` with a column list and no query.
    CreateTableWithoutQuery,
    /// A result table is named by one identifier, never `schema.table`.
    QualifiedName,
}

impl RefusedStatement {
    /// The typed error the refusal raises.
    pub fn error(&self) -> JammiError {
        let (expected, actual) = match self.reason {
            RefusalReason::CreateTableWithoutQuery => (
                "CREATE TABLE <name> AS <query>: a result table is what a query produced",
                "a column list and no query (no empty result tables)",
            ),
            RefusalReason::QualifiedName => (
                "one identifier: a result table is named by its bare name",
                "a qualified name",
            ),
        };
        JammiError::Schema {
            table: self.name.clone(),
            column: "<statement>".to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
}

/// The one identifier a result-table statement names, or the refusal.
fn statement_name(name: &TableReference) -> std::result::Result<String, RefusalReason> {
    match name.schema() {
        None => Ok(name.table().to_string()),
        Some(_) => Err(RefusalReason::QualifiedName),
    }
}

impl StatementClass {
    /// Classify `plan`, the logical plan of one statement.
    pub fn of(plan: LogicalPlan) -> Self {
        match plan {
            LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(cmd)) => Self::classify_create(cmd),
            LogicalPlan::Ddl(DdlStatement::DropTable(cmd)) => match statement_name(&cmd.name) {
                Ok(name) => Self::DropTable {
                    name,
                    if_exists: cmd.if_exists,
                },
                Err(reason) => Self::Refused(RefusedStatement {
                    name: cmd.name.to_string(),
                    reason,
                }),
            },
            other => Self::Inline(other),
        }
    }

    fn classify_create(cmd: CreateMemoryTable) -> Self {
        let refused = |reason| {
            Self::Refused(RefusedStatement {
                name: cmd.name.to_string(),
                reason,
            })
        };
        let name = match statement_name(&cmd.name) {
            Ok(name) => name,
            Err(reason) => return refused(reason),
        };
        if matches!(cmd.input.as_ref(), LogicalPlan::EmptyRelation(_)) {
            return refused(RefusalReason::CreateTableWithoutQuery);
        }
        let input = Arc::unwrap_or_clone(cmd.input);
        let definition = input.display_indent().to_string();
        let sources = scanned_relations(&input);
        Self::CreateTableAs {
            statement: CreateTableAs {
                name,
                if_not_exists: cmd.if_not_exists,
                or_replace: cmd.or_replace,
                definition,
                sources,
            },
            input,
        }
    }

    /// The plan the engine executes: a store statement's node
    /// ([`StoreStatementNode`]), planned into [`StoreStatementExec`] by
    /// [`MaterializationPlanner`] and run when the frame is collected; an
    /// inline statement unchanged.
    pub fn into_plan(self) -> LogicalPlan {
        let node = match self {
            Self::Inline(plan) => return plan,
            Self::CreateTableAs { statement, input } => StoreStatementNode {
                statement: StoreStatement::CreateTableAs(statement),
                input: vec![input],
            },
            Self::DropTable { name, if_exists } => StoreStatementNode {
                statement: StoreStatement::DropTable { name, if_exists },
                input: Vec::new(),
            },
            Self::Refused(refused) => StoreStatementNode {
                statement: StoreStatement::Refused(refused),
                input: Vec::new(),
            },
        };
        LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })
    }
}

/// Every relation `plan` scans, in plan order, as the statement spelled
/// it — the sources a `CREATE TABLE … AS` table is anchored on.
fn scanned_relations(plan: &LogicalPlan) -> Vec<String> {
    let mut sources = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            sources.push(scan.table_name.to_string());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .expect("collecting table scans cannot fail");
    sources
}

/// The statement the store executes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub enum StoreStatement {
    /// Materialize the node's one input as the named result table.
    CreateTableAs(CreateTableAs),
    /// Drop the named result table.
    DropTable {
        /// The result table's name.
        name: String,
        /// `IF EXISTS`.
        if_exists: bool,
    },
    /// Refuse, typed.
    Refused(RefusedStatement),
}

/// The logical node a store statement plans to: no output columns, one
/// input for a `CREATE TABLE … AS`, none otherwise.
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd)]
pub struct StoreStatementNode {
    statement: StoreStatement,
    input: Vec<LogicalPlan>,
}

impl StoreStatementNode {
    /// The statement this node runs.
    pub fn statement(&self) -> &StoreStatement {
        &self.statement
    }
}

fn empty_schema() -> DFSchemaRef {
    static EMPTY: OnceLock<DFSchemaRef> = OnceLock::new();
    Arc::clone(EMPTY.get_or_init(|| Arc::new(DFSchema::empty())))
}

impl UserDefinedLogicalNodeCore for StoreStatementNode {
    fn name(&self) -> &str {
        "StoreStatement"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.input.iter().collect()
    }

    fn schema(&self) -> &DFSchemaRef {
        static EMPTY: OnceLock<DFSchemaRef> = OnceLock::new();
        EMPTY.get_or_init(empty_schema)
    }

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.statement {
            StoreStatement::CreateTableAs(s) => write!(f, "CreateResultTable: name={}", s.name),
            StoreStatement::DropTable { name, .. } => write!(f, "DropResultTable: name={name}"),
            StoreStatement::Refused(r) => write!(f, "RefusedStatement: {}", r.error()),
        }
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> DfResult<Self> {
        if inputs.len() != self.input.len() {
            return Err(DataFusionError::Internal(format!(
                "StoreStatement has {} input(s), got {}",
                self.input.len(),
                inputs.len()
            )));
        }
        Ok(Self {
            statement: self.statement.clone(),
            input: inputs,
        })
    }
}

/// Plans [`StoreStatementNode`] into [`StoreStatementExec`] over the
/// planning session's own store and state; every other extension node is
/// left to the next planner. A session carrying no store (no result store
/// was installed on it) cannot run a store statement: refused typed.
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
        session_state: &SessionState,
    ) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<StoreStatementNode>() else {
            return Ok(None);
        };
        let typed = |e: JammiError| DataFusionError::External(Box::new(e));
        if let StoreStatement::Refused(refused) = &node.statement {
            return Err(typed(refused.error()));
        }
        let store = session_state
            .config()
            .get_extension::<ResultStore>()
            .ok_or_else(|| {
                typed(JammiError::Config(
                    "this session carries no result store: a CREATE TABLE … AS / DROP TABLE \
                     needs one installed"
                        .into(),
                ))
            })?;
        let ctx = QueryContext::from(SessionContext::new_with_state(session_state.clone()));
        Ok(Some(Arc::new(StoreStatementExec::new(
            node.statement.clone(),
            physical_inputs.to_vec(),
            ctx,
            ResultStore::clone(&store),
        ))))
    }
}

/// The physical node a store statement runs as: the store's verb under the
/// session that planned it, emitting no rows. A `CREATE TABLE … AS` roots
/// its child in [`crate::store::ResultTableSinkExec`] as every producer
/// does, so the write runs where the compute plane says.
pub struct StoreStatementExec {
    statement: StoreStatement,
    input: Vec<Arc<dyn ExecutionPlan>>,
    ctx: QueryContext,
    store: ResultStore,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for StoreStatementExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoreStatementExec")
            .field("statement", &self.statement)
            .finish_non_exhaustive()
    }
}

impl StoreStatementExec {
    fn new(
        statement: StoreStatement,
        input: Vec<Arc<dyn ExecutionPlan>>,
        ctx: QueryContext,
        store: ResultStore,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(empty_schema().inner().clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            statement,
            input,
            ctx,
            store,
            properties,
        }
    }

    async fn run(
        statement: StoreStatement,
        input: Vec<Arc<dyn ExecutionPlan>>,
        ctx: QueryContext,
        store: ResultStore,
        context: Arc<TaskContext>,
    ) -> Result<()> {
        match statement {
            StoreStatement::CreateTableAs(create) => {
                let [query] = <[Arc<dyn ExecutionPlan>; 1]>::try_from(input).map_err(|inputs| {
                    JammiError::Other(format!(
                        "CreateResultTable has exactly one input, got {}",
                        inputs.len()
                    ))
                })?;
                store
                    .create_table_as(&ctx, &create, query, context)
                    .await
                    .map(|_| ())
            }
            StoreStatement::DropTable { name, if_exists } => {
                match store.drop_result_table(&name).await {
                    Ok(_) => Ok(()),
                    Err(JammiError::RowGone { .. }) if if_exists => Ok(()),
                    Err(e) => Err(e),
                }
            }
            StoreStatement::Refused(refused) => Err(refused.error()),
        }
    }
}

impl DisplayAs for StoreStatementExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StoreStatementExec: {:?}", self.statement)
    }
}

impl ExecutionPlan for StoreStatementExec {
    fn name(&self) -> &str {
        "StoreStatementExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.input.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            self.statement.clone(),
            children,
            self.ctx.clone(),
            self.store.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "StoreStatementExec has one output partition, partition {partition} was asked for"
            )));
        }
        let schema = self.schema();
        let (statement, input, ctx, store) = (
            self.statement.clone(),
            self.input.clone(),
            self.ctx.clone(),
            self.store.clone(),
        );
        let stream = futures::stream::once(async move {
            Self::run(statement, input, ctx, store, context)
                .await
                .map(|()| futures::stream::empty())
                .map_err(|e| DataFusionError::External(Box::new(e)))
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;

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

    /// A context over one table `t`.
    fn context() -> SessionContext {
        let ctx = SessionContext::new();
        let batch = rows();
        ctx.register_table(
            "t",
            Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap()),
        )
        .unwrap();
        ctx
    }

    /// Which class every statement shape the SQL surface takes falls in:
    /// the store statements, the refusals, and everything else inline.
    #[tokio::test]
    async fn the_class_table_names_the_store_statements_and_the_refusals() {
        let ctx = context();
        let create = |name: &str, s: &StatementClass| matches!(s, StatementClass::CreateTableAs { statement, .. } if statement.name == name);
        type Expectation = Box<dyn Fn(&StatementClass) -> bool>;
        let table: Vec<(&str, Expectation)> = vec![
            (
                "CREATE TABLE u AS SELECT id, title FROM t",
                Box::new(move |s| create("u", s)),
            ),
            (
                "CREATE TABLE IF NOT EXISTS u AS SELECT id FROM t WHERE id > 1",
                Box::new(|s| {
                    matches!(s, StatementClass::CreateTableAs { statement, .. }
                        if statement.if_not_exists && statement.sources == ["t"])
                }),
            ),
            (
                "CREATE TABLE u (id BIGINT)",
                Box::new(|s| {
                    matches!(s, StatementClass::Refused(r)
                        if r.reason == RefusalReason::CreateTableWithoutQuery)
                }),
            ),
            (
                "CREATE TABLE a.b AS SELECT id FROM t",
                Box::new(
                    |s| matches!(s, StatementClass::Refused(r) if r.reason == RefusalReason::QualifiedName),
                ),
            ),
            (
                "DROP TABLE IF EXISTS u",
                Box::new(
                    |s| matches!(s, StatementClass::DropTable { name, if_exists: true } if name == "u"),
                ),
            ),
            (
                "SELECT id, title FROM t",
                Box::new(|s| matches!(s, StatementClass::Inline(_))),
            ),
            (
                "INSERT INTO t VALUES (4, 'd')",
                Box::new(|s| matches!(s, StatementClass::Inline(_))),
            ),
            (
                "EXPLAIN SELECT id FROM t",
                Box::new(|s| matches!(s, StatementClass::Inline(_))),
            ),
            (
                "CREATE VIEW v AS SELECT id FROM t",
                Box::new(|s| matches!(s, StatementClass::Inline(_))),
            ),
            (
                "SET datafusion.execution.batch_size = 8",
                Box::new(|s| matches!(s, StatementClass::Inline(_))),
            ),
        ];
        for (sql, expected) in table {
            let plan = ctx.state().create_logical_plan(sql).await.unwrap();
            let class = StatementClass::of(plan);
            assert!(expected(&class), "{sql}");
            let rooted = matches!(
                class.into_plan(),
                LogicalPlan::Extension(Extension { node })
                    if node.as_any().downcast_ref::<StoreStatementNode>().is_some()
            );
            assert_eq!(
                rooted,
                !sql.starts_with("SELECT")
                    && !sql.starts_with("INSERT")
                    && !sql.starts_with("EXPLAIN")
                    && !sql.starts_with("CREATE VIEW")
                    && !sql.starts_with("SET"),
                "{sql}: rooted in the store statement node iff a store statement"
            );
        }
    }

    /// A refused statement's error is typed and names the table.
    #[test]
    fn a_refusal_is_the_typed_schema_error_naming_the_table() {
        let refused = RefusedStatement {
            name: "u".into(),
            reason: RefusalReason::CreateTableWithoutQuery,
        };
        match refused.error() {
            JammiError::Schema { table, .. } => assert_eq!(table, "u"),
            other => panic!("expected Schema, got {other:?}"),
        }
    }
}
