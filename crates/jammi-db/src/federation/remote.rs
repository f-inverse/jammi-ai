//! What a database source runs with the engine's meaning.
//!
//! Federation writes a sub-plan as the remote's SQL, and SQL that reads the
//! same can mean something else there. In MySQL `a || b` is a logical OR,
//! `DATE - DATE` subtracts the dates as `YYYYMMDD` numbers, an ascending sort
//! puts NULLs first and the unparser drops the `NULLS LAST` that would say
//! otherwise, `/` on integers is decimal division, and `=` on strings follows
//! the column's collation, case-insensitive by default. The remote answers
//! each without an error, and the answer is not the query's.
//!
//! So a remote is sent only what is known to mean there what it means to the
//! engine: an allowlist of plan nodes and expressions per remote ([`runs`]),
//! which [`LocalFunctionFederation`](super::LocalFunctionFederation) consults
//! before federating a sub-plan, and which gates the filters a scan that is
//! not federated hands its table ([`RemoteFilters`]). Anything else stays in
//! the engine, over the rows the remote returns: never a different answer,
//! at the cost of the rows a remote-side filter, sort or aggregate would have
//! saved. A construct joins the allowlist, or is written as the remote's
//! exact equivalent, only with a test that it means the same.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion::common::Statistics;
use datafusion::common::{Column, DFSchema, TableReference};
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{
    Aggregate, BinaryExpr, Expr, ExprSchemable, JoinType, LogicalPlan, Operator,
    TableProviderFilterPushDown, TableScan, TableSource, TableType,
};
use datafusion::physical_expr::{PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_federation::sql::{SQLExecutor, SQLFederationProvider, SQLTable, SQLTableSource};
use datafusion_federation::{get_table_source, FederatedTableProviderAdaptor};

/// A database whose meaning of SQL is not the engine's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "mysql"), allow(dead_code))] // its sources' feature is off
pub(crate) enum Remote {
    MySql,
}

/// A table of `remote`: `provider` scans it and `executor` runs the SQL
/// federation writes for it, and neither is handed a node or a filter the
/// remote would read differently.
#[cfg_attr(not(feature = "mysql"), allow(dead_code))] // its sources' feature is off
pub(crate) fn table_provider(
    remote: Remote,
    reference: TableReference,
    provider: Arc<dyn TableProvider>,
    executor: Arc<dyn SQLExecutor>,
) -> Arc<dyn TableProvider> {
    let schema = provider.schema();
    let source = SQLTableSource::new_with_table(
        Arc::new(SQLFederationProvider::new(executor)),
        Arc::new(RemoteTable {
            remote,
            reference,
            schema,
        }),
    );
    Arc::new(FederatedTableProviderAdaptor::new_with_provider(
        Arc::new(source),
        Arc::new(RemoteFilters { remote, provider }),
    ))
}

/// Whether every remote `plan` scans runs `plan` — each node of it, its
/// subqueries included — with the engine's meaning. A plan that scans no
/// such remote is run by it trivially.
pub(crate) fn runs(plan: &LogicalPlan) -> Result<bool> {
    let mut remotes = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            if let Some(remote) = remote_of(&scan.source)? {
                if !remotes.contains(&remote) {
                    remotes.push(remote);
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if remotes.is_empty() {
        return Ok(true);
    }
    let mut every = true;
    plan.apply_with_subqueries(|node| {
        every = remotes.iter().all(|remote| node_runs(*remote, node));
        Ok(if every {
            TreeNodeRecursion::Continue
        } else {
            TreeNodeRecursion::Stop
        })
    })?;
    Ok(every)
}

/// The remote a scan's table is a table of, when it is one of [`Remote`]'s.
fn remote_of(source: &Arc<dyn TableSource>) -> Result<Option<Remote>> {
    let Some(federated) = get_table_source(source)? else {
        return Ok(None);
    };
    let Some(source) = (federated.as_ref() as &dyn Any).downcast_ref::<SQLTableSource>() else {
        return Ok(None);
    };
    Ok((source.table.as_any())
        .downcast_ref::<RemoteTable>()
        .map(|table| table.remote))
}

/// Whether `remote` runs `node` itself — not its inputs — with the engine's
/// meaning.
fn node_runs(remote: Remote, node: &LogicalPlan) -> bool {
    match remote {
        Remote::MySql => mysql::node_runs(node),
    }
}

/// Whether `remote` evaluates `expr`, over `schema`, as the engine does.
fn expr_runs(remote: Remote, expr: &Expr, schema: &DFSchema) -> bool {
    match remote {
        Remote::MySql => mysql::expr_runs(expr, schema),
    }
}

/// MySQL's allowlist.
mod mysql {
    use super::*;

    pub(super) fn node_runs(node: &LogicalPlan) -> bool {
        match node {
            LogicalPlan::TableScan(scan) => scan_filters_run(scan),
            LogicalPlan::Projection(projection) => projection
                .expr
                .iter()
                .all(|expr| expr_runs(expr, projection.input.schema())),
            LogicalPlan::Filter(filter) => expr_runs(&filter.predicate, filter.input.schema()),
            LogicalPlan::Limit(_) | LogicalPlan::SubqueryAlias(_) => true,
            LogicalPlan::Aggregate(aggregate) => aggregate_runs(aggregate),
            LogicalPlan::Join(join) => {
                let schema = join.left.schema().join(join.right.schema());
                matches!(
                    join.join_type,
                    JoinType::Inner | JoinType::Left | JoinType::Right
                ) && join.on.iter().all(|(left, right)| {
                    compared(left, join.left.schema()) && compared(right, join.right.schema())
                }) && match (&join.filter, schema) {
                    (None, _) => true,
                    (Some(filter), Ok(schema)) => expr_runs(filter, &schema),
                    (Some(_), Err(_)) => false,
                }
            }
            // A sort: MySQL puts NULLs first when ascending, and orders
            // strings by collation. Every other node is unverified.
            _ => false,
        }
    }

    fn scan_filters_run(scan: &TableScan) -> bool {
        scan.filters.is_empty()
            || DFSchema::try_from_qualified_schema(scan.table_name.clone(), &scan.source.schema())
                .is_ok_and(|schema| scan.filters.iter().all(|f| expr_runs(f, &schema)))
    }

    /// A grouping by strings merges what the collation calls equal; an
    /// aggregate MySQL computes in another type (`AVG` in a four-place
    /// decimal) answers differently.
    fn aggregate_runs(aggregate: &Aggregate) -> bool {
        let schema = aggregate.input.schema();
        aggregate
            .group_expr
            .iter()
            .all(|expr| compared(expr, schema))
            && aggregate.aggr_expr.iter().all(|expr| {
                let Expr::AggregateFunction(function) = unaliased(expr) else {
                    return false;
                };
                let params = &function.params;
                if params.distinct || params.filter.is_some() || !params.order_by.is_empty() {
                    return false;
                }
                let arguments_run = params.args.iter().all(|arg| expr_runs(arg, schema));
                let argument = |test: fn(&DataType) -> bool| {
                    params
                        .args
                        .first()
                        .and_then(|arg| arg.get_type(schema).ok())
                        .is_some_and(|t| test(&t))
                };
                arguments_run
                    && match function.func.name() {
                        "count" => true,
                        "sum" => argument(|t| t.is_integer() || t.is_floating()),
                        "min" | "max" => argument(comparable),
                        _ => false,
                    }
            })
    }

    pub(super) fn expr_runs(expr: &Expr, schema: &DFSchema) -> bool {
        let typed = |expr: &Expr, test: fn(&DataType) -> bool| {
            expr.get_type(schema).is_ok_and(|t| test(&t))
        };
        match expr {
            Expr::Column(_) => true,
            Expr::Literal(value, _) => !value.data_type().is_nested(),
            Expr::Alias(alias) => expr_runs(&alias.expr, schema),
            Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
                expr_runs(inner, schema)
            }
            Expr::Negative(inner) => typed(inner, numeric) && expr_runs(inner, schema),
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                let operands = expr_runs(left, schema) && expr_runs(right, schema);
                operands
                    && match op {
                        Operator::And | Operator::Or => true,
                        Operator::Eq
                        | Operator::NotEq
                        | Operator::Lt
                        | Operator::LtEq
                        | Operator::Gt
                        | Operator::GtEq => typed(left, comparable) && typed(right, comparable),
                        Operator::Plus | Operator::Minus | Operator::Multiply => {
                            typed(left, numeric) && typed(right, numeric)
                        }
                        _ => false,
                    }
            }
            Expr::Between(between) => {
                compared(&between.expr, schema)
                    && compared(&between.low, schema)
                    && compared(&between.high, schema)
            }
            Expr::InList(in_list) => {
                compared(&in_list.expr, schema)
                    && in_list.list.iter().all(|item| compared(item, schema))
            }
            Expr::Cast(cast) => {
                expr_runs(&cast.expr, schema)
                    && cast.expr.get_type(schema).is_ok_and(|from| {
                        let to = cast.field.data_type();
                        from.is_integer() && (to.is_integer() || to.is_floating())
                    })
            }
            _ => false,
        }
    }

    /// `expr` runs, and its values compare as the engine compares them.
    fn compared(expr: &Expr, schema: &DFSchema) -> bool {
        expr_runs(expr, schema) && expr.get_type(schema).is_ok_and(|t| comparable(&t))
    }

    fn numeric(t: &DataType) -> bool {
        t.is_integer() || t.is_floating() || matches!(t, DataType::Decimal128(..))
    }

    /// Ordered and compared alike: numbers, booleans, dates and zone-less
    /// timestamps. Strings compare by collation; zoned timestamps by the
    /// session's zone.
    fn comparable(t: &DataType) -> bool {
        numeric(t)
            || matches!(
                t,
                DataType::Boolean
                    | DataType::Date32
                    | DataType::Date64
                    | DataType::Timestamp(_, None)
            )
    }

    fn unaliased(expr: &Expr) -> &Expr {
        match expr {
            Expr::Alias(alias) => unaliased(&alias.expr),
            expr => expr,
        }
    }
}

/// The remote table federation writes SQL for; it marks the table as
/// [`Remote`]'s so [`runs`] knows whose allowlist applies.
#[derive(Debug)]
struct RemoteTable {
    remote: Remote,
    reference: TableReference,
    schema: SchemaRef,
}

impl SQLTable for RemoteTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// `expr` with every column named by its name alone.
fn unqualified(expr: &Expr) -> Result<Expr> {
    expr.clone()
        .transform(|expr| {
            Ok(match expr {
                Expr::Column(column) => {
                    Transformed::yes(Expr::Column(Column::new_unqualified(column.name)))
                }
                expr => Transformed::no(expr),
            })
        })
        .data()
}

/// A remote's table as a scan that is not federated reads it: the filters it
/// is handed are only those the remote runs with the engine's meaning, the
/// rest left to the engine above the scan.
#[derive(Debug)]
struct RemoteFilters {
    remote: Remote,
    provider: Arc<dyn TableProvider>,
}

#[async_trait]
impl TableProvider for RemoteFilters {
    fn schema(&self) -> SchemaRef {
        self.provider.schema()
    }

    fn table_type(&self) -> TableType {
        self.provider.table_type()
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        // The filters name the table as the query did; the provider's schema
        // knows only its columns.
        let schema = DFSchema::try_from(self.schema().as_ref().clone())?;
        let runs = filters
            .iter()
            .map(|filter| Ok(expr_runs(self.remote, &unqualified(filter)?, &schema)))
            .collect::<Result<Vec<bool>>>()?;
        let handed: Vec<&Expr> = filters
            .iter()
            .zip(&runs)
            .filter_map(|(filter, runs)| runs.then_some(*filter))
            .collect();
        let mut answers = self
            .provider
            .supports_filters_pushdown(&handed)?
            .into_iter();
        Ok(runs
            .into_iter()
            .map(|runs| {
                if runs {
                    answers
                        .next()
                        .unwrap_or(TableProviderFilterPushDown::Unsupported)
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let scan = self
            .provider
            .scan(state, projection, filters, limit)
            .await?;
        Ok(Arc::new(RemoteScanExec::new(scan)))
    }
}

/// A remote's scan, which takes no filter or sort from the plan above it.
///
/// The SQL scan `datafusion-table-providers` gives a database source writes
/// the filters and the sort physical pushdown offers it into its statement,
/// in the remote's dialect and with the remote's meaning: a string filter by
/// MySQL's collation, an ascending sort with NULLs first. Above this node
/// both stay in the engine; the filters the remote runs as the engine does
/// already reached the scan, through [`RemoteFilters`]. A limit is the
/// remote's to apply — any `n` rows are any `n` rows.
#[derive(Debug)]
struct RemoteScanExec {
    input: Arc<dyn ExecutionPlan>,
}

impl RemoteScanExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self { input }
    }
}

impl DisplayAs for RemoteScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RemoteScanExec")
    }
}

impl ExecutionPlan for RemoteScanExec {
    fn name(&self) -> &str {
        "RemoteScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(children.swap_remove(0))))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.input.supports_limit_pushdown()
    }

    fn fetch(&self) -> Option<usize> {
        self.input.fetch()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let input = self.input.with_fetch(limit)?;
        Some(Arc::new(Self::new(input)))
    }

    /// No filter from above reaches the scan; written out, not left to the
    /// default, so a change to the default cannot quietly change it.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        Ok(FilterDescription::all_unsupported(
            &parent_filters,
            &self.children(),
        ))
    }

    /// No sort from above reaches the scan; written out for the same reason.
    fn try_pushdown_sort(
        &self,
        _order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        Ok(SortOrderPushdownResult::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow::datatypes::{Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::execution::context::{QueryPlanner, SessionState};
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion::physical_plan::displayable;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
    use datafusion::prelude::SessionContext;
    use datafusion::sql::unparser::dialect::{Dialect, MySqlDialect};
    use datafusion_federation::FederatedPlanner;

    use super::*;
    use crate::federation::{EngineFunctions, LocalFunctionFederation};

    /// MySQL as federation sees it, recording each statement it is sent.
    #[derive(Debug)]
    struct StubMySql {
        schema: SchemaRef,
        sent: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl SQLExecutor for StubMySql {
        fn name(&self) -> &str {
            "mysql"
        }

        fn compute_context(&self) -> Option<String> {
            Some("stub".into())
        }

        fn dialect(&self) -> Arc<dyn Dialect> {
            Arc::new(MySqlDialect {})
        }

        fn execute(
            &self,
            query: &str,
            schema: SchemaRef,
            _filters: &[Arc<dyn PhysicalExpr>],
        ) -> Result<SendableRecordBatchStream> {
            self.sent.lock().unwrap().push(query.to_string());
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                schema,
                futures::stream::empty(),
            )))
        }

        async fn table_names(&self) -> Result<Vec<String>> {
            Ok(vec!["t".into()])
        }

        async fn get_table_schema(&self, _table_name: &str) -> Result<SchemaRef> {
            Ok(Arc::clone(&self.schema))
        }
    }

    #[derive(Debug)]
    struct Planner;

    #[async_trait]
    impl QueryPlanner for Planner {
        async fn create_physical_plan(
            &self,
            plan: &LogicalPlan,
            state: &SessionState,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(FederatedPlanner::new())])
                .create_physical_plan(plan, state)
                .await
        }
    }

    /// A session federating as the engine's does, over a MySQL table `t`,
    /// and the statements MySQL is sent.
    fn session() -> (SessionContext, Arc<Mutex<Vec<String>>>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("status", DataType::Utf8, true),
            Field::new("bought", DataType::Date32, true),
            Field::new("delivered", DataType::Date32, true),
        ]));
        let sent = Arc::new(Mutex::new(Vec::new()));
        let executor = Arc::new(StubMySql {
            schema: Arc::clone(&schema),
            sent: Arc::clone(&sent),
        });
        let table = MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap();
        let provider = table_provider(
            Remote::MySql,
            TableReference::bare("t"),
            Arc::new(table),
            executor,
        );
        let base = SessionContext::new().state();
        let mut rules = base.optimizer().rules.clone();
        let at = rules
            .iter()
            .position(|rule| rule.name() == "scalar_subquery_to_join")
            .unwrap()
            + 1;
        rules.insert(
            at,
            Arc::new(LocalFunctionFederation::new(Arc::new(
                EngineFunctions::default(),
            ))),
        );
        let state = SessionStateBuilder::new_from_existing(base)
            .with_optimizer_rules(rules)
            .with_query_planner(Arc::new(Planner))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table("t", provider).unwrap();
        (ctx, sent)
    }

    /// The statements MySQL is sent to answer `sql`, and the physical plan.
    async fn sent(sql: &str) -> (Vec<String>, String) {
        let (ctx, sent) = session();
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap();
        let statements = sent.lock().unwrap().clone();
        let shown = displayable(plan.as_ref()).indent(false).to_string();
        (statements, shown)
    }

    #[tokio::test]
    async fn what_mysql_reads_differently_stays_in_the_engine() {
        for (sql, statement) in [
            // `||` is a logical OR in MySQL.
            (
                "SELECT status || '-x' AS v FROM t WHERE id = 1",
                "SELECT * FROM `t` WHERE (`t`.`id` = 1)",
            ),
            // `DATE - DATE` subtracts YYYYMMDD numbers in MySQL.
            (
                "SELECT delivered - bought AS d FROM t WHERE id = 1",
                "SELECT * FROM `t` WHERE (`t`.`id` = 1)",
            ),
            (
                "SELECT AVG(delivered - bought) AS d FROM t WHERE delivered IS NOT NULL",
                "SELECT * FROM `t` WHERE `t`.`delivered` IS NOT NULL",
            ),
            // MySQL sorts NULLs first when ascending; the unparser drops
            // the `NULLS LAST` that would say otherwise.
            (
                "SELECT id FROM t ORDER BY delivered ASC NULLS LAST",
                "SELECT `t`.`id`, `t`.`delivered` FROM `t`",
            ),
            (
                "SELECT id FROM t ORDER BY delivered LIMIT 2",
                "SELECT `t`.`id`, `t`.`delivered` FROM `t`",
            ),
            // MySQL has no `date_trunc`.
            (
                "SELECT date_trunc('month', bought) AS m FROM t WHERE id = 1",
                "SELECT * FROM `t` WHERE (`t`.`id` = 1)",
            ),
        ] {
            assert_eq!(sent(sql).await.0, [statement], "{sql}");
        }
    }

    #[tokio::test]
    async fn what_mysql_reads_as_the_engine_does_is_still_sent_to_it() {
        for (sql, statement) in [
            (
                "SELECT COUNT(*) AS n, MIN(id) AS lo FROM t WHERE id > 1",
                "SELECT count(1) AS `n`, min(`t`.`id`) AS `lo` FROM `t` WHERE (`t`.`id` > 1)",
            ),
            (
                "SELECT id FROM t WHERE id >= 2",
                "SELECT `t`.`id` FROM `t` WHERE (`t`.`id` >= 2)",
            ),
            (
                "SELECT * FROM t LIMIT 1",
                "SELECT `t`.`id`, `t`.`status`, `t`.`bought`, `t`.`delivered` FROM `t` LIMIT 1",
            ),
        ] {
            assert_eq!(sent(sql).await.0, [statement], "{sql}");
        }
    }

    /// A query whose every node above the scan stays in the engine is not
    /// federated at all; the scan's own filter and sort pushdown would write
    /// them into its statement, and does not.
    #[tokio::test]
    async fn a_scan_that_is_not_federated_takes_no_filter_or_sort_from_above() {
        for sql in [
            // `=` on strings follows MySQL's collation, case-insensitive.
            "SELECT id FROM t WHERE status = 'Delivered' ORDER BY id",
            "SELECT status, COUNT(*) AS n FROM t GROUP BY status ORDER BY status",
        ] {
            let (statements, plan) = sent(sql).await;
            assert!(statements.is_empty(), "{sql}: {statements:?}");
            let above = plan.find("SortExec").zip(plan.find("RemoteScanExec"));
            assert!(above.is_some_and(|(sort, scan)| sort < scan), "{plan}");
        }
        let (_, plan) = sent("SELECT id FROM t WHERE status = 'Delivered'").await;
        let above = plan.find("FilterExec").zip(plan.find("RemoteScanExec"));
        assert!(above.is_some_and(|(filter, scan)| filter < scan), "{plan}");
    }

    #[tokio::test]
    async fn the_mysql_allowlist() {
        let (ctx, _) = session();
        let runs = |sql: &'static str| {
            let ctx = &ctx;
            async move {
                let plan = ctx
                    .sql(&format!("SELECT {sql} FROM t"))
                    .await
                    .unwrap()
                    .into_unoptimized_plan();
                let LogicalPlan::Projection(projection) = plan else {
                    panic!("{plan}")
                };
                mysql::expr_runs(&projection.expr[0], projection.input.schema())
            }
        };
        for (expr, expected) in [
            ("id", true),
            ("id + 1", true),
            ("id * 2 - 1", true),
            ("id > 1 AND bought < delivered", true),
            ("id BETWEEN 1 AND 3", true),
            ("id IN (1, 2)", true),
            ("delivered IS NULL", true),
            ("CAST(id AS BIGINT)", true),
            ("id / 2", false),
            ("status = 'x'", false),
            ("status IN ('a', 'b')", false),
            ("status || 'x'", false),
            ("delivered - bought", false),
            ("upper(status)", false),
            ("date_trunc('month', bought)", false),
            ("CAST(status AS INT)", false),
        ] {
            assert_eq!(runs(expr).await, expected, "{expr}");
        }
    }
}
