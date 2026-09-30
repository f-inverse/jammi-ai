//! The SQL statements the result store executes, and how.
//!
//! [`StatementClass`] states, in one place, which statements the store
//! executes: a `CREATE TABLE … AS` is a result-table materialization, a
//! `DROP TABLE` the store's drop of one, an `UPDATE`/`DELETE` of a mutable
//! companion table a row rewrite. [`MaterializationPlanner`] plans the
//! statement's node into [`StoreStatementExec`], which runs the store's
//! verb under the session that planned it.
//!
//! `CREATE TABLE … AS` as the store executes it: a result table named by
//! the statement, produced through the same sink and funnel every other
//! producer uses.
//!
//! `OR REPLACE` never loses the table it replaces. The statement's rows are
//! built under a row of their own that names the table it is to replace
//! ([`CreateResultTableParams::replaces`]); the old table serves under its
//! name until the new artifact is complete and attested, then the promote
//! transaction removes the old row and moves the new one onto the name —
//! a reader resolves one or the other, never none — and the old bytes are
//! reclaimed after it. A failure anywhere before that transaction — the
//! query refusing mid-write, the promote losing its lease — leaves the old
//! table as it was and discards the row and bytes the statement made. A replacement is not a version of the table it
//! replaces: a version is a keyed delta over a base artifact that stays,
//! while `OR REPLACE` produces a different artifact under a possibly
//! different definition and reclaims the old one, which is exactly what a
//! row of its own gives it.

use std::fmt;
use std::sync::{Arc, OnceLock};

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, DFSchemaRef, Result as DfResult, TableReference};
use datafusion::error::DataFusionError;
use datafusion::execution::context::{SessionState, TaskContext};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{
    CreateMemoryTable, DdlStatement, DmlStatement, Expr, ExprSchemable, Extension, LogicalPlan,
    Projection, TableScan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore, WriteOp,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::Statement as DfStatement;
use datafusion::sql::sqlparser::ast::{
    AssignmentTarget, Statement as SqlStatement, TableFactor, Update, UpdateTableFromKind,
};
use datafusion::sql::unparser::plan_to_sql;
use futures::TryStreamExt;
use tracing::warn;

use crate::catalog::result_repo::{Producer, ResultTableCas, ResultTableKind, ResultTableRecord};
use crate::error::{JammiError, Result};
use crate::session::QueryContext;
use crate::store::building::BuildingTable;
use crate::store::manifest::{InputAnchor, Materialization, ProducingDescriptor};
use crate::store::mutable::rewrite::RowRewriteNode;
use crate::store::{ResultStore, ResultTableOrigin, SinkKind};

#[cfg(doc)]
use crate::catalog::result_repo::CreateResultTableParams;

/// What a `CREATE TABLE <name> AS <query>` asks for.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct CreateTableAs {
    /// The result table's name — the `result_tables` primary key, read as
    /// `"jammi.<name>"`.
    pub name: String,
    /// `IF NOT EXISTS`: an existing table of this name is left as it is.
    pub if_not_exists: bool,
    /// `OR REPLACE`: an existing table of this name serves until the new
    /// one is complete, then the name moves to the new table in one catalog
    /// transaction and the old bytes are reclaimed. A name nothing is under
    /// is simply created.
    pub or_replace: bool,
    /// The `AS <query>` part as SQL — the table's definition, which the
    /// manifest hashes and a recompute re-plans.
    pub query: String,
    /// Every relation the query scans, as spelled — the table's input
    /// anchors, and its lineage's source.
    pub sources: Vec<String>,
}

/// The name a replacement of `name` is built under: unique per statement,
/// and never the name itself, which the table being replaced keeps until
/// the swap.
fn replacement_name(name: &str) -> String {
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%9f");
    let suffix = &uuid::Uuid::new_v4().simple().to_string()[..8];
    format!("{name}__replacement__{timestamp}_{suffix}")
}

impl ResultStore {
    /// Materialize `query`'s rows as the result table `statement` names:
    /// the row is created under the caller's tenant, written through the
    /// sink, attested with a
    /// [`ProducingDescriptor::Statement`] over the query and an
    /// unpinned anchor per scanned relation, and promoted `ready` — under
    /// the name, replacing the table there when `OR REPLACE` found one
    /// (see the module doc). `None` when `IF NOT EXISTS` found the table;
    /// an existing table without `OR REPLACE` or `IF NOT EXISTS` is refused
    /// typed. A statement that fails leaves no row and no bytes of its own
    /// behind.
    pub async fn create_table_as(
        &self,
        ctx: &QueryContext,
        statement: &CreateTableAs,
        query: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<Option<ResultTableRecord>> {
        let name = statement.name.as_str();
        let replaces = match self.catalog.get_result_table(name).await? {
            Some(_) if statement.or_replace => Some(name),
            Some(_) if statement.if_not_exists => return Ok(None),
            Some(_) => {
                return Err(JammiError::Catalog(format!(
                    "result table '{name}' already exists"
                )))
            }
            None => None,
        };
        let table_name = replaces.map_or_else(|| name.to_string(), replacement_name);
        // The lineage column names the first relation the query scans; a
        // query scanning none (`SELECT 1 AS id`) is its own source.
        let source_id = statement.sources.first().map_or(name, String::as_str);
        let building = self
            .create_named_table(
                table_name,
                ResultTableOrigin {
                    source_id,
                    producer: Producer::Derivation { task: None },
                    kind: ResultTableKind::Statement,
                    derived_from: None,
                    dimensions: None,
                    key_column: None,
                    text_columns: None,
                    job_attempt: None,
                },
                replaces,
            )
            .await?;
        let row = building.table_name().to_string();
        let cas = building.cas();
        match self
            .materialize_statement(ctx, statement, building, query, context)
            .await
        {
            Ok(record) => Ok(Some(record)),
            Err(e) => {
                self.discard_statement_row(&cas, &row).await;
                Err(e)
            }
        }
    }

    /// The statement's write and finish over its `building` row.
    async fn materialize_statement(
        &self,
        ctx: &QueryContext,
        statement: &CreateTableAs,
        mut building: BuildingTable,
        query: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<ResultTableRecord> {
        let summary = self
            .write_result_table(&mut building, SinkKind::Rows, query, context)
            .await?;
        let descriptor = ProducingDescriptor::Statement {
            query: statement.query.clone(),
        };
        let now = chrono::Utc::now().to_rfc3339();
        let inputs = statement
            .sources
            .iter()
            .map(|source| InputAnchor::unpinned_at_instant(source, now.clone()))
            .collect();
        building
            .finish(
                ctx,
                summary.rows as usize,
                Materialization::new(&descriptor, &summary.env, inputs),
            )
            .await
    }

    /// Discard the row a failed statement made, with its bytes: the
    /// `building -> failed` CAS under the statement's own writer (a miss is
    /// the row's answer — the sink already failed it, or recovery holds
    /// it), then the drop of the terminal row. A statement's failed row is
    /// nobody's evidence: the statement reported its error, and no name
    /// reaches the row. What this cannot remove is logged and left to
    /// `reconcile`; the statement's own error is what the caller sees.
    async fn discard_statement_row(&self, cas: &ResultTableCas, row: &str) {
        if let Err(e) = self.catalog.fail_building_table(cas).await {
            warn!(table = row, outcome = %e, "statement: the failed row was not this writer's to fail");
        }
        match self.drop_result_table(row).await {
            Ok(_) | Err(JammiError::RowGone { .. }) => {}
            Err(e) => {
                warn!(table = row, error = %e, "statement: the failed row was not discarded; reconcile reaps it");
            }
        }
    }
}

/// How a SQL statement is executed, decided by the root of its logical
/// plan — the one place the class is stated:
///
/// - `CREATE TABLE <name> AS <query>` is a [`Self::CreateTableAs`]: a
///   result table named `<name>` — bytes on the object store under the
///   store's root, a `result_tables` row, read on every replica through
///   the store's binding as `"jammi.<name>"` — materialized through
///   the result-table sink over the query. Never an in-memory table one
///   replica holds. The
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
/// - `UPDATE` / `DELETE` of a mutable companion table is a
///   [`Self::RewriteRows`]: the statement's own plan — whatever it joins,
///   filters or limits — selects the rows, and the table rewrites exactly
///   those, on this process's catalog backend
///   ([`RowRewriteNode`]). On any other table the statement is the table
///   provider's own, which receives only filters over the table's columns:
///   one choosing its rows by more than that is [`Self::Refused`].
/// - Everything else is [`Self::Inline`], running where it was issued: a
///   query's rows stream to the caller as they are produced; `INSERT`
///   writes a mutable companion table on this process's catalog backend
///   from the statement's own rows; `EXPLAIN` explains the statement as
///   classified here; `COPY … TO`
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
    /// `UPDATE` / `DELETE` of a mutable table's selected rows.
    RewriteRows(RowRewriteNode),
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
    /// An `UPDATE` or `DELETE` of a table other than a mutable table whose
    /// rows are chosen by more than a predicate over the target table's own
    /// columns — a subquery, or a join to another relation. Such a table's
    /// provider receives a DML statement's predicate as filters over its own
    /// columns only, so anything beyond them would be dropped and the
    /// statement would rewrite rows it never selected.
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
    /// Plan and classify the one statement `sql` holds under `state`.
    ///
    /// DataFusion's SQL planner refuses `UPDATE … FROM`; this plans it into
    /// the shape DataFusion gives every other `UPDATE` (see
    /// `plan_update_from`), so the class of an `UPDATE` does not depend on
    /// whether its rows come from a join.
    pub async fn plan(state: &SessionState, sql: &str) -> DfResult<Self> {
        let dialect = state.config().options().sql_parser.dialect;
        let plan = match state.sql_to_statement(sql, &dialect)? {
            DfStatement::Statement(statement) => match *statement {
                SqlStatement::Update(update) if update.from.is_some() => {
                    plan_update_from(state, update).await?
                }
                statement => {
                    state
                        .statement_to_plan(DfStatement::Statement(Box::new(statement)))
                        .await?
                }
            },
            statement => state.statement_to_plan(statement).await?,
        };
        Self::of(plan)
    }

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
                match RowRewriteNode::of(dml)? {
                    Ok(rewrite) => Self::RewriteRows(rewrite),
                    Err(dml) => Self::classify_provider_dml(dml)?,
                }
            }
            LogicalPlan::Explain(mut explain) => {
                explain.plan = Arc::new(Self::of(Arc::unwrap_or_clone(explain.plan))?.into_plan());
                Self::Inline(LogicalPlan::Explain(explain))
            }
            LogicalPlan::Analyze(mut analyze) => {
                analyze.input =
                    Arc::new(Self::of(Arc::unwrap_or_clone(analyze.input))?.into_plan());
                Self::Inline(LogicalPlan::Analyze(analyze))
            }
            other => Self::Inline(other),
        })
    }

    /// An `UPDATE` / `DELETE` the target's provider runs from filters over
    /// its own columns, or the refusal when the statement chooses rows by
    /// more than that.
    fn classify_provider_dml(dml: DmlStatement) -> DfResult<Self> {
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
    /// [`MaterializationPlanner`] and run when the frame is collected; a
    /// rewrite's [`RowRewriteNode`]; an inline statement unchanged.
    pub fn into_plan(self) -> LogicalPlan {
        let node = match self {
            Self::Inline(plan) => return plan,
            Self::RewriteRows(rewrite) => return rewrite.into_plan(),
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

/// The plan of `UPDATE <target> SET … FROM <relations> [WHERE …]`, in the
/// shape DataFusion plans an `UPDATE` without `FROM`: a `Dml` over a
/// projection of each target column's new value — its assignment cast to the
/// column's type, or the column as read — over the target joined to the
/// `FROM` relations and filtered by the predicate. The target is the join's
/// leftmost relation, so its columns as read are the first under the
/// projection ([`RowRewriteNode`] reads them there).
async fn plan_update_from(state: &SessionState, update: Update) -> DfResult<LogicalPlan> {
    if update.returning.is_some()
        || update.output.is_some()
        || update.or.is_some()
        || update.limit.is_some()
        || !update.order_by.is_empty()
    {
        return Err(DataFusionError::NotImplemented(
            "UPDATE … FROM takes only SET, FROM and WHERE".into(),
        ));
    }
    let target = match &update.table.relation {
        TableFactor::Table { name, alias, .. } => alias
            .as_ref()
            .map_or_else(|| name.to_string(), |alias| alias.name.to_string()),
        other => {
            return Err(DataFusionError::Plan(format!(
                "UPDATE names a table, not `{other}`"
            )))
        }
    };
    let relations = match update.from {
        Some(UpdateTableFromKind::BeforeSet(from) | UpdateTableFromKind::AfterSet(from)) => from,
        None => Vec::new(),
    };
    let selected = format!(
        "SELECT {target}.* FROM {}{}",
        std::iter::once(&update.table)
            .chain(&relations)
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        update
            .selection
            .as_ref()
            .map_or_else(String::new, |p| format!(" WHERE {p}"))
    );
    let LogicalPlan::Projection(read) = state.create_logical_plan(&selected).await? else {
        return Err(DataFusionError::Internal(format!(
            "`{selected}` plans to a projection"
        )));
    };
    let source = read.input;
    let target_scan = leftmost_scan(&source).ok_or_else(|| {
        DataFusionError::Internal(format!("`{selected}` scans its target leftmost"))
    })?;
    let normalize = state
        .config()
        .options()
        .sql_parser
        .enable_ident_normalization;
    let mut assigned = update
        .assignments
        .into_iter()
        .map(|assignment| {
            let AssignmentTarget::ColumnName(column) = assignment.target else {
                return Err(DataFusionError::NotImplemented(
                    "UPDATE assigns one column at a time, not a tuple".into(),
                ));
            };
            let column = column
                .0
                .last()
                .and_then(|part| part.as_ident())
                .map(|ident| match ident.quote_style {
                    None if normalize => ident.value.to_lowercase(),
                    _ => ident.value.clone(),
                })
                .ok_or_else(|| DataFusionError::Plan(format!("`{column}` names no column")))?;
            let value =
                state.create_logical_expr(&assignment.value.to_string(), source.schema())?;
            Ok((column, value))
        })
        .collect::<DfResult<std::collections::HashMap<_, _>>>()?;
    let columns = target_scan.source.schema();
    let values = read
        .expr
        .into_iter()
        .zip(columns.fields().iter())
        .map(|(as_read, field)| {
            let value = match assigned.remove(field.name()) {
                Some(value) => value.cast_to(field.data_type(), source.schema())?,
                None => as_read,
            };
            Ok(value.alias(field.name()))
        })
        .collect::<DfResult<Vec<_>>>()?;
    if let Some(column) = assigned.into_keys().next() {
        return Err(DataFusionError::Plan(format!(
            "UPDATE assigns `{column}`, which `{}` does not have",
            target_scan.table_name
        )));
    }
    Ok(LogicalPlan::Dml(DmlStatement::new(
        target_scan.table_name.clone(),
        Arc::clone(&target_scan.source),
        WriteOp::Update,
        Arc::new(LogicalPlan::Projection(Projection::try_new(
            values, source,
        )?)),
    )))
}

/// The table scan a plan of `SELECT … FROM <target>, …` reads its target
/// from: the leftmost relation under the filters and joins.
fn leftmost_scan(mut plan: &LogicalPlan) -> Option<&TableScan> {
    loop {
        plan = match plan {
            LogicalPlan::Filter(filter) => &filter.input,
            LogicalPlan::Join(join) => &join.left,
            LogicalPlan::SubqueryAlias(alias) => &alias.input,
            LogicalPlan::TableScan(scan) => return Some(scan),
            _ => return None,
        };
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
/// session that planned it, emitting no rows. A `CREATE TABLE … AS`
/// writes its child through the result-table sink as every producer does.
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
        StatementClass::plan(&ctx.state(), sql).await.unwrap()
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
    /// the store statements, the refusals, and everything else inline. The
    /// table here is not a mutable table, so an `UPDATE` / `DELETE` choosing
    /// rows beyond its own columns is refused.
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
                "UPDATE t SET title = o.title FROM t AS o WHERE t.id = o.id + 1",
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
            let class = StatementClass::plan(&ctx.state(), sql).await.unwrap();
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
