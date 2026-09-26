//! Where a materialization runs: on the compute plane when this process
//! holds the client role and a live executor can hold the plan, in this
//! process otherwise.
//!
//! [`ComputePlane`] is the one submit seam: this crate declares it, the
//! compute-plane crate implements it over its submit client, and the server
//! installs it on the session through [`ComputePlaneSlot::install`] when
//! the process is configured as a client — the engine never depends on the
//! implementation. Every submission a process makes — a result table's
//! sink over its compute, a claimed training attempt's one task — goes
//! through it. The slot rides in the session's `SessionConfig` as an
//! extension, so it reaches every plan executed under a context derived
//! from the session's
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
    CreateMemoryTable, DdlStatement, DmlStatement, Expr, Extension, LogicalPlan,
    UserDefinedLogicalNode, UserDefinedLogicalNodeCore, WriteOp,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::SessionContext;
use datafusion::sql::unparser::plan_to_sql;
use futures::future::BoxFuture;
use futures::TryStreamExt;

use crate::error::{JammiError, Result};
use crate::session::QueryContext;
use crate::store::statement::CreateTableAs;
use crate::store::ResultStore;
use jammi_datafusion::ComputeDeviceKind;

/// A compute plane a physical plan is submitted to: the plan's stages run
/// on the plane's executors and its output streams back. The plan crosses
/// as it is — the same operators the in-process run would execute —
/// and a failure on the plane reaches the caller as the same typed error
/// the in-process run would raise. Admission and placement are two verbs
/// so a caller that must run an unheld plan somewhere else — a result
/// table's sink in this process, a training attempt in its claimant's
/// own body —
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

    /// The device kind this plane places models' plans onto, when the
    /// deployment names one; `None` places them onto the submitter's own.
    fn device_kind(&self) -> Option<ComputeDeviceKind> {
        None
    }
}

/// What a plan asks of the executor that holds it — read off the plan's
/// own nodes, never off the process submitting it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PlanRequirements {
    /// The device kind a node of the plan is stamped with, if any (an
    /// inference's, a placed training attempt's — CPU is a kind too).
    pub device_kind: Option<ComputeDeviceKind>,
    /// An executor the plan must not land on: a placed training attempt's
    /// own submitter, whose
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
///   compute plane says. Never an in-memory table one replica holds. The
///   table records the query as SQL — its plan rendered back by
///   DataFusion's unparser — so a recompute re-plans it under the catalog
///   of the day.
/// - `DROP TABLE <name>` is a [`Self::DropTable`]: the store's drop of the
///   result table named `<name>` under the tenant in force
///   ([`ResultStore::drop_result_table`]).
/// - `CREATE TABLE <name> (<columns>)` — no query — is [`Self::Refused`]:
///   a result table is what a query produced; there are no empty ones. A
///   qualified name (`CREATE TABLE a.b AS`) is refused too: a result table
///   is named by one identifier. So is a query the unparser cannot render
///   back to SQL (a `WITH RECURSIVE` query, a `VALUES` list), naming the
///   node: a definition that cannot replay is never recorded.
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub enum RefusalReason {
    /// `CREATE TABLE` with a column list and no query.
    CreateTableWithoutQuery,
    /// A result table is named by one identifier, never `schema.table`.
    QualifiedName,
    /// The query carries a node the unparser cannot render back to SQL, so
    /// the table could never replay.
    NotReplayable {
        /// The node, as the plan displays it.
        node: String,
    },
    /// An `UPDATE` or `DELETE` whose rows are chosen by more than a predicate
    /// over the target table's own columns — a subquery, or a join to
    /// another relation. A table provider receives a DML statement's
    /// predicate as filters over its own columns only, so anything beyond
    /// them would be dropped and the statement would rewrite rows it never
    /// selected.
    DmlBeyondTarget {
        /// The node or expression, as the plan displays it.
        node: String,
    },
}

impl RefusedStatement {
    /// The typed error the refusal raises.
    pub fn error(&self) -> JammiError {
        let (expected, actual) = match &self.reason {
            RefusalReason::CreateTableWithoutQuery => (
                "CREATE TABLE <name> AS <query>: a result table is what a query produced"
                    .to_string(),
                "a column list and no query (no empty result tables)".to_string(),
            ),
            RefusalReason::QualifiedName => (
                "one identifier: a result table is named by its bare name".to_string(),
                "a qualified name".to_string(),
            ),
            RefusalReason::NotReplayable { node } => (
                "a query the engine can render back to SQL, so the table replays".to_string(),
                format!("a `{node}` node the SQL unparser cannot render"),
            ),
            RefusalReason::DmlBeyondTarget { node } => (
                "an UPDATE / DELETE predicate over the target table's own columns".to_string(),
                format!("`{node}`: select the keys first, then name them in the predicate"),
            ),
        };
        JammiError::Schema {
            table: self.name.clone(),
            column: "<statement>".to_string(),
            expected,
            actual,
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
    pub fn of(plan: LogicalPlan) -> DfResult<Self> {
        Ok(match plan {
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
            LogicalPlan::Dml(dml) if matches!(dml.op, WriteOp::Update | WriteOp::Delete) => {
                Self::classify_rewrite(dml)?
            }
            other => Self::Inline(other),
        })
    }

    fn classify_rewrite(dml: DmlStatement) -> DfResult<Self> {
        Ok(match beyond_target(&dml.input, &dml.table_name)? {
            Some(node) => Self::Refused(RefusedStatement {
                name: dml.table_name.to_string(),
                reason: RefusalReason::DmlBeyondTarget { node },
            }),
            None => Self::Inline(LogicalPlan::Dml(dml)),
        })
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
        let query = match plan_to_sql(&input) {
            Ok(rendered) => rendered.to_string(),
            Err(_) => {
                return refused(RefusalReason::NotReplayable {
                    node: unrenderable_node(&input),
                })
            }
        };
        let sources = scanned_relations(&input);
        Self::CreateTableAs {
            statement: CreateTableAs {
                name,
                if_not_exists: cmd.if_not_exists,
                or_replace: cmd.or_replace,
                query,
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

/// The first part of a DML statement's `input` that chooses rows by more than
/// a predicate over `target`'s own columns: a node other than a projection,
/// a filter, or a scan of `target` (a join, a scan of another relation), or
/// a subquery expression inside a filter.
fn beyond_target(input: &LogicalPlan, target: &TableReference) -> DfResult<Option<String>> {
    let mut culprit = None;
    input.apply(|node| {
        let beyond = match node {
            LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) => None,
            LogicalPlan::TableScan(scan) => {
                (!scan.table_name.resolved_eq(target)).then(|| node.display().to_string())
            }
            LogicalPlan::Filter(filter) => filter
                .predicate
                .exists(|e| {
                    Ok(matches!(
                        e,
                        Expr::InSubquery(_) | Expr::Exists(_) | Expr::ScalarSubquery(_)
                    ))
                })?
                .then(|| filter.predicate.to_string()),
            other => Some(other.display().to_string()),
        };
        Ok(match beyond {
            Some(node) => {
                culprit = Some(node);
                TreeNodeRecursion::Stop
            }
            None => TreeNodeRecursion::Continue,
        })
    })?;
    Ok(culprit)
}

/// The node of `plan` the unparser cannot render, as the plan displays it:
/// a refused node every child of which renders on its own. `plan` itself
/// failed to render, so its root is the first candidate; the walk descends
/// only through refused nodes (a node that renders is skipped with its
/// subtree), so the last node it visits is one whose children all render.
fn unrenderable_node(plan: &LogicalPlan) -> String {
    let mut culprit = plan.display().to_string();
    plan.apply_with_subqueries(|node| {
        Ok(if plan_to_sql(node).is_ok() {
            TreeNodeRecursion::Jump
        } else {
            culprit = node.display().to_string();
            TreeNodeRecursion::Continue
        })
    })
    .expect("localizing the unrenderable node cannot fail");
    culprit
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
    use datafusion::catalog::{
        CatalogProvider, MemoryCatalogProvider, MemorySchemaProvider, SchemaProvider,
    };
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

    fn mem_table() -> Arc<MemTable> {
        let batch = rows();
        Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap())
    }

    /// A context over the same rows under the three spellings a statement
    /// scans: a bare `t`, a result table's `"jammi.recent"`, and a source's
    /// `src.public.t`.
    fn context() -> SessionContext {
        let ctx = SessionContext::new();
        ctx.register_table("t", mem_table()).unwrap();
        ctx.register_table(
            crate::store::result_table_relation("recent").table_reference(),
            mem_table(),
        )
        .unwrap();
        let schema = Arc::new(MemorySchemaProvider::new());
        schema.register_table("t".into(), mem_table()).unwrap();
        let catalog = MemoryCatalogProvider::new();
        catalog.register_schema("public", schema).unwrap();
        ctx.register_catalog("src", Arc::new(catalog));
        ctx
    }

    /// The `CREATE TABLE … AS` class over `sql`, or the refusal.
    async fn classify(ctx: &SessionContext, sql: &str) -> StatementClass {
        StatementClass::of(ctx.state().create_logical_plan(sql).await.unwrap()).unwrap()
    }

    /// Every query shape the class admits records as SQL that re-plans to
    /// the same rows and re-renders to the same text — the fixed point a
    /// replay rests on — over each relation spelling a statement scans.
    #[tokio::test]
    async fn an_admitted_query_records_as_sql_that_replays_to_the_same_rows() {
        let ctx = context();
        let queries = [
            "SELECT id, title FROM t WHERE id > 1 ORDER BY id LIMIT 2",
            "SELECT title, count(*) AS n FROM t GROUP BY title HAVING count(*) >= 1 ORDER BY title",
            "SELECT a.id, b.title FROM t a JOIN t b ON a.id = b.id ORDER BY a.id",
            "SELECT id FROM t UNION ALL SELECT id FROM t ORDER BY id",
            "SELECT DISTINCT title FROM t ORDER BY title",
            "WITH late AS (SELECT id FROM t WHERE id >= 2) SELECT id FROM late ORDER BY id",
            "SELECT id FROM t WHERE id IN (SELECT id FROM t WHERE id > 2)",
            "SELECT id, row_number() OVER (ORDER BY id) AS rn FROM t ORDER BY id",
            "SELECT 1 AS id, 'a' AS title",
            "SELECT id, title FROM \"jammi.recent\" WHERE id > 1 ORDER BY id",
            "SELECT id, title FROM src.public.t WHERE id > 1 ORDER BY id",
            "SELECT a.id, b.title FROM src.public.t a JOIN \"jammi.recent\" b ON a.id = b.id \
             ORDER BY a.id",
        ];
        for query in queries {
            let recorded = match classify(&ctx, &format!("CREATE TABLE u AS {query}")).await {
                StatementClass::CreateTableAs { statement, .. } => statement.query,
                StatementClass::Refused(refused) => {
                    panic!("{query}: refused {:?}", refused.reason)
                }
                _ => panic!("{query}: not a CREATE TABLE AS"),
            };
            let expected = ctx.sql(query).await.unwrap().collect().await.unwrap();
            let replayed = ctx.sql(&recorded).await.unwrap().collect().await.unwrap();
            let concat = |batches: &[RecordBatch]| {
                arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
            };
            assert_eq!(
                concat(&replayed),
                concat(&expected),
                "{query}: recorded as {recorded}"
            );
            let StatementClass::CreateTableAs { statement, .. } =
                classify(&ctx, &format!("CREATE TABLE u AS {recorded}")).await
            else {
                panic!("{recorded}: admitted again");
            };
            assert_eq!(statement.query, recorded, "{query}: a fixed point");
        }
    }

    /// A query the unparser cannot render is refused at planning naming
    /// the node — a `WITH RECURSIVE` query's `RecursiveQuery`, a `VALUES`
    /// list's `Values` — before anything is written.
    #[tokio::test]
    async fn a_query_the_unparser_cannot_render_is_refused_naming_the_node() {
        let ctx = context();
        let refusals = [
            (
                "CREATE TABLE cycles AS WITH RECURSIVE n AS \
                 (SELECT 1 AS v UNION ALL SELECT v + 1 FROM n WHERE v < 3) SELECT v FROM n",
                "cycles",
                "RecursiveQuery",
            ),
            (
                "CREATE TABLE listed AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS v(id, title)",
                "listed",
                "Values",
            ),
        ];
        for (sql, name, node_kind) in refusals {
            let StatementClass::Refused(refused) = classify(&ctx, sql).await else {
                panic!("{sql}: refused");
            };
            assert_eq!(refused.name, name);
            let RefusalReason::NotReplayable { node } = &refused.reason else {
                panic!("{sql}: expected NotReplayable, got {:?}", refused.reason);
            };
            assert!(
                node.starts_with(node_kind),
                "{sql}: the refusal names the node: {node}"
            );
            match refused.error() {
                JammiError::Schema { table, actual, .. } => {
                    assert_eq!(table, name);
                    assert!(actual.contains(node_kind), "{actual}");
                }
                other => panic!("expected Schema, got {other:?}"),
            }
        }
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
                "UPDATE t SET title = upper(title) WHERE id > 1 AND title <> 'b'",
                Box::new(|s| matches!(s, StatementClass::Inline(_))),
            ),
            (
                "DELETE FROM t WHERE id = 2",
                Box::new(|s| matches!(s, StatementClass::Inline(_))),
            ),
            (
                "DELETE FROM t WHERE id IN (SELECT id FROM t WHERE title = 'b')",
                Box::new(|s| {
                    matches!(s, StatementClass::Refused(r)
                        if matches!(r.reason, RefusalReason::DmlBeyondTarget { .. }))
                }),
            ),
            (
                "UPDATE t SET title = 'x' WHERE EXISTS (SELECT 1 FROM t)",
                Box::new(|s| {
                    matches!(s, StatementClass::Refused(r)
                        if matches!(r.reason, RefusalReason::DmlBeyondTarget { .. }))
                }),
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
            let class = StatementClass::of(plan).unwrap();
            assert!(expected(&class), "{sql}");
            let inline = matches!(class, StatementClass::Inline(_));
            let rooted = matches!(
                class.into_plan(),
                LogicalPlan::Extension(Extension { node })
                    if node.as_any().downcast_ref::<StoreStatementNode>().is_some()
            );
            assert_eq!(
                rooted, !inline,
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
