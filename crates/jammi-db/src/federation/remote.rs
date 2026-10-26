//! What a database source runs with the engine's meaning.
//!
//! Federation writes a sub-plan as the remote's SQL, and SQL that reads the
//! same can mean something else there. In MySQL `a || b` is a logical OR,
//! `DATE - DATE` subtracts the dates as `YYYYMMDD` numbers, an ascending sort
//! puts NULLs first and the unparser drops the `NULLS LAST` that would say
//! otherwise, `/` on integers is decimal division, and `=` on strings follows
//! the column's collation, case-insensitive by default. In Postgres strings
//! order by the database's collation, a decimal `AVG` carries more places
//! than the engine planned, and DataFusion's own functions (`approx_distinct`,
//! `date_trunc`'s spellings, every function a session installs) do not exist.
//! The remote answers each without an error, or fails where the engine would
//! not, and the answer is not the query's.
//!
//! So a remote is sent only what is known to mean there what it means to the
//! engine: one allowlist of plan nodes and expressions, read through each
//! remote's [`Semantics`] — what it shares with the engine beyond the common
//! core. [`federable`] consults it before federation takes a sub-plan, and it
//! gates the filters a scan that is not federated hands its table
//! ([`RemoteFilters`]). Anything else stays in the engine, over the rows the
//! remote returns: never a different answer, at the cost of the rows a
//! remote-side filter, sort or aggregate would have saved. A construct joins
//! the allowlist, or a remote's semantics, only with a test that it means the
//! same there (`tests/it/remote_conformance.rs` runs each over the remote and
//! over the same rows in a file).

use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DFSchema, TableReference};
use datafusion::error::{DataFusionError, Result};
use datafusion::functions_aggregate::average::Avg;
use datafusion::functions_aggregate::count::Count;
use datafusion::functions_aggregate::min_max::{Max, Min};
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::{
    Aggregate, BinaryExpr, Expr, ExprSchemable, JoinType, LogicalPlan, Operator,
    SortExpr, TableProviderFilterPushDown, TableScan, TableSource, TableType,
};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::sql::sqlparser::ast::Statement;
use datafusion_federation::sql::{
    AstAnalyzer, SQLExecutor, SQLFederationProvider, SQLTable, SQLTableSource,
};
use datafusion_federation::{get_table_source, FederatedTableProviderAdaptor};

use super::RemoteExec;

/// A database whose meaning of SQL is not the engine's, one per database
/// source this build carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Remote {
    #[cfg(feature = "postgres")]
    Postgres,
    #[cfg(feature = "mysql")]
    MySql,
}

/// What a remote reads as the engine does, beyond the core every remote
/// shares: columns, literals, boolean logic, `IS [NOT] NULL`, ordering
/// comparisons and `BETWEEN` on numbers, booleans, dates and zone-less
/// timestamps, `+ - *` on numbers, integer-to-number casts, and `COUNT`,
/// numeric `SUM` and ordered `MIN`/`MAX`, over scans, projections, filters,
/// limits and inner, left and right joins.
#[derive(Debug, Clone, Copy)]
struct Semantics {
    /// `=`, `<>`, `IN`, grouping and join keys on strings are byte equality.
    /// MySQL's default collation folds case.
    string_equality: bool,
    /// `||` concatenates strings. In MySQL it is a logical OR.
    string_concat: bool,
    /// `/` and `%` on integers truncate toward zero, and on floats are IEEE.
    /// MySQL's integer `/` is decimal division.
    division: bool,
    /// A sort on an ordered, non-string key orders as the engine does,
    /// NULLs placed as the query says. MySQL puts NULLs first ascending and
    /// the unparser drops the `NULLS LAST` that would say otherwise.
    sorts: bool,
    /// `AVG` over integers and floats is the engine's value. MySQL's is a
    /// four-place decimal.
    average: bool,
    /// `COUNT(DISTINCT …)` counts what the engine counts (over keys that
    /// compare as it does).
    count_distinct: bool,
    /// `FULL JOIN` joins as the engine does.
    full_join: bool,
}

impl Remote {
    fn semantics(self) -> Semantics {
        match self {
            #[cfg(feature = "postgres")]
            Remote::Postgres => Semantics {
                string_equality: true,
                string_concat: true,
                division: true,
                sorts: true,
                average: true,
                count_distinct: true,
                full_join: true,
            },
            #[cfg(feature = "mysql")]
            Remote::MySql => Semantics {
                string_equality: false,
                string_concat: false,
                division: false,
                sorts: false,
                average: false,
                count_distinct: false,
                full_join: false,
            },
        }
    }
}

/// How a remote reads a table it cannot read as stored: a rewrite of every
/// statement that names the table.
pub(crate) type Reads = Arc<dyn Fn(Statement) -> Result<Statement> + Send + Sync>;

/// A table of `remote`: `provider` scans it and `executor` runs the SQL
/// federation writes for it — through `reads`, when the table is read
/// through one — and neither is handed a node or a filter the remote would
/// read differently.
pub(crate) fn table_provider(
    remote: Remote,
    reference: TableReference,
    provider: Arc<dyn TableProvider>,
    executor: Arc<dyn SQLExecutor>,
    reads: Option<Reads>,
) -> Arc<dyn TableProvider> {
    let schema = provider.schema();
    let source = SQLTableSource::new_with_table(
        Arc::new(SQLFederationProvider::new(executor)),
        Arc::new(RemoteTable {
            remote,
            reference,
            schema,
            reads,
        }),
    );
    Arc::new(FederatedTableProviderAdaptor::new_with_provider(
        Arc::new(source),
        Arc::new(RemoteFilters { remote, provider }),
    ))
}

/// Whether federation takes `plan` whole: it scans at least one remote, and
/// every remote it scans runs every node of it — its subqueries included —
/// with the engine's meaning. A plan that scans no remote has nothing to
/// federate.
pub(crate) fn federable(plan: &LogicalPlan) -> Result<bool> {
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
        return Ok(false);
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
    let semantics = remote.semantics();
    match node {
        LogicalPlan::TableScan(scan) => scan_runs(remote, scan),
        LogicalPlan::Projection(projection) => projection
            .expr
            .iter()
            .all(|expr| expr_runs(remote, expr, projection.input.schema())),
        LogicalPlan::Filter(filter) => expr_runs(remote, &filter.predicate, filter.input.schema()),
        LogicalPlan::Limit(_) | LogicalPlan::SubqueryAlias(_) => true,
        LogicalPlan::Aggregate(aggregate) => aggregate_runs(semantics, aggregate),
        LogicalPlan::Sort(sort) => {
            semantics.sorts
                && sort
                    .expr
                    .iter()
                    .all(|SortExpr { expr, .. }| ordered_by(semantics, expr, sort.input.schema()))
        }
        LogicalPlan::Join(join) => {
            let schema = join.left.schema().join(join.right.schema());
            let kind = match join.join_type {
                JoinType::Inner | JoinType::Left | JoinType::Right => true,
                JoinType::Full => semantics.full_join,
                _ => false,
            };
            kind && join.on.iter().all(|(left, right)| {
                equated(semantics, left, join.left.schema())
                    && equated(semantics, right, join.right.schema())
            }) && match (&join.filter, schema) {
                (None, _) => true,
                (Some(filter), Ok(schema)) => expr_runs(remote, filter, &schema),
                (Some(_), Err(_)) => false,
            }
        }
        // Every other node is unverified, or one the unparser does not write
        // as SQL the remote accepts (a `DISTINCT` over a filtered projection
        // names a table its `FROM` does not).
        _ => false,
    }
}

/// A scan runs when it is not `remote`'s (another source's scan sends
/// `remote` nothing) or when `remote` runs every filter on it.
fn scan_runs(remote: Remote, scan: &TableScan) -> bool {
    if !matches!(remote_of(&scan.source), Ok(Some(of)) if of == remote) {
        return true;
    }
    scan.filters.is_empty()
        || DFSchema::try_from_qualified_schema(scan.table_name.clone(), &scan.source.schema())
            .is_ok_and(|schema| scan.filters.iter().all(|f| expr_runs(remote, f, &schema)))
}

/// A grouping by keys the remote compares otherwise merges other rows; an
/// aggregate the remote computes in another type answers differently. An
/// aggregate directly over a projection is not sent at all: the unparser
/// writes the projection as a derived table and still names the base table
/// in the aggregate's columns, which the remote refuses.
fn aggregate_runs(semantics: Semantics, aggregate: &Aggregate) -> bool {
    let schema = aggregate.input.schema();
    !matches!(aggregate.input.as_ref(), LogicalPlan::Projection(_))
        && aggregate
        .group_expr
        .iter()
        .all(|expr| equated(semantics, expr, schema))
        && aggregate.aggr_expr.iter().all(|expr| {
            let Expr::AggregateFunction(function) = unaliased(expr) else {
                return false;
            };
            let params = &function.params;
            if params.filter.is_some() || !params.order_by.is_empty() {
                return false;
            }
            let argument_type = || {
                params
                    .args
                    .first()
                    .and_then(|arg| arg.get_type(schema).ok())
            };
            let arguments_run = params.args.iter().all(|arg| core_runs(semantics, arg, schema));
            let implementation = function.func.inner();
            let numeric_argument = || argument_type().is_some_and(|t| t.is_integer() || t.is_floating());
            let runs = if implementation.is::<Count>() {
                !params.distinct
                    || (semantics.count_distinct
                        && argument_type().is_some_and(|t| equatable(semantics, &t)))
            } else if implementation.is::<Sum>() {
                !params.distinct && numeric_argument()
            } else if implementation.is::<Min>() || implementation.is::<Max>() {
                argument_type().is_some_and(|t| ordered(&t))
            } else if implementation.is::<Avg>() {
                semantics.average && !params.distinct && numeric_argument()
            } else {
                false
            };
            arguments_run && runs
        })
}

/// Whether `remote` evaluates `expr`, over `schema`, as the engine does.
fn expr_runs(remote: Remote, expr: &Expr, schema: &DFSchema) -> bool {
    core_runs(remote.semantics(), expr, schema)
}

fn core_runs(semantics: Semantics, expr: &Expr, schema: &DFSchema) -> bool {
    let typed = |expr: &Expr, test: &dyn Fn(&DataType) -> bool| {
        expr.get_type(schema).is_ok_and(|t| test(&t))
    };
    let runs = |expr: &Expr| core_runs(semantics, expr, schema);
    match expr {
        Expr::Column(_) => true,
        Expr::Literal(value, _) => !value.data_type().is_nested(),
        Expr::Alias(alias) => runs(&alias.expr),
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => runs(inner),
        Expr::Negative(inner) => typed(inner, &numeric) && runs(inner),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            runs(left)
                && runs(right)
                && match op {
                    Operator::And | Operator::Or => true,
                    Operator::Eq | Operator::NotEq => {
                        typed(left, &|t| equatable(semantics, t))
                            && typed(right, &|t| equatable(semantics, t))
                    }
                    Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                        typed(left, &ordered) && typed(right, &ordered)
                    }
                    Operator::Plus | Operator::Minus | Operator::Multiply => {
                        typed(left, &numeric) && typed(right, &numeric)
                    }
                    Operator::Divide | Operator::Modulo => {
                        semantics.division
                            && (typed(left, &DataType::is_integer)
                                && typed(right, &DataType::is_integer)
                                || typed(left, &DataType::is_floating)
                                    && typed(right, &DataType::is_floating))
                    }
                    Operator::StringConcat => {
                        semantics.string_concat && typed(left, &string) && typed(right, &string)
                    }
                    _ => false,
                }
        }
        Expr::Between(between) => {
            ordered_by(semantics, &between.expr, schema)
                && ordered_by(semantics, &between.low, schema)
                && ordered_by(semantics, &between.high, schema)
        }
        Expr::InList(in_list) => {
            equated(semantics, &in_list.expr, schema)
                && in_list
                    .list
                    .iter()
                    .all(|item| equated(semantics, item, schema))
        }
        Expr::Cast(cast) => {
            runs(&cast.expr)
                && cast.expr.get_type(schema).is_ok_and(|from| {
                    let to = cast.field.data_type();
                    from.is_integer() && (to.is_integer() || to.is_floating())
                })
        }
        _ => false,
    }
}

/// `expr` runs, and its values compare for equality as the engine compares
/// them.
fn equated(semantics: Semantics, expr: &Expr, schema: &DFSchema) -> bool {
    core_runs(semantics, expr, schema)
        && expr
            .get_type(schema)
            .is_ok_and(|t| equatable(semantics, &t))
}

/// `expr` runs, and its values order as the engine orders them.
fn ordered_by(semantics: Semantics, expr: &Expr, schema: &DFSchema) -> bool {
    core_runs(semantics, expr, schema) && expr.get_type(schema).is_ok_and(|t| ordered(&t))
}

fn numeric(t: &DataType) -> bool {
    t.is_integer() || t.is_floating() || matches!(t, DataType::Decimal128(..))
}

fn string(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

/// Ordered and compared alike everywhere: numbers, booleans, dates and
/// zone-less timestamps. Strings order by collation; zoned timestamps by the
/// session's zone.
fn ordered(t: &DataType) -> bool {
    numeric(t)
        || matches!(
            t,
            DataType::Boolean | DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, None)
        )
}

/// Compared for equality as the engine compares: every ordered type, and
/// strings where the remote's equality is byte equality.
fn equatable(semantics: Semantics, t: &DataType) -> bool {
    ordered(t) || (semantics.string_equality && string(t))
}

fn unaliased(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(alias) => unaliased(&alias.expr),
        expr => expr,
    }
}

/// The remote table federation writes SQL for; it marks the table as
/// [`Remote`]'s so [`federable`] knows whose semantics apply, and carries
/// how the remote reads it.
struct RemoteTable {
    remote: Remote,
    reference: TableReference,
    schema: SchemaRef,
    reads: Option<Reads>,
}

impl std::fmt::Debug for RemoteTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteTable")
            .field("remote", &self.remote)
            .field("reference", &self.reference)
            .field("reads", &self.reads.is_some())
            .finish_non_exhaustive()
    }
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

    fn ast_analyzer(&self) -> Option<AstAnalyzer> {
        let reads = self.reads.clone()?;
        Some(Box::new(move |statement| reads(statement)))
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
/// rest left to the engine above the scan, and the scan sits behind a
/// [`RemoteExec`].
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
        // A provider answers once per filter it is handed, in order.
        let answers = self.provider.supports_filters_pushdown(&handed)?;
        if answers.len() != handed.len() {
            return Err(DataFusionError::Internal(format!(
                "a remote table's provider answered {} of the {} filters it was handed",
                answers.len(),
                handed.len()
            )));
        }
        let mut answers = answers.into_iter();
        runs.into_iter()
            .map(|runs| {
                if runs {
                    answers.next().ok_or_else(|| {
                        DataFusionError::Internal("a remote table's filter went unanswered".into())
                    })
                } else {
                    Ok(TableProviderFilterPushDown::Unsupported)
                }
            })
            .collect()
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
        Ok(Arc::new(RemoteExec::new(scan)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow::datatypes::{Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::execution::context::{QueryPlanner, SessionState};
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::logical_expr::{ScalarUDF, Signature, Volatility};
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_plan::displayable;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
    use datafusion::prelude::SessionContext;
    use datafusion::sql::unparser::dialect::Dialect;

    use super::*;
    use crate::federation::{RemoteFederation, RemotePlanner};

    /// A remote as federation sees it, recording each statement it is sent.
    #[derive(Debug)]
    struct Stub {
        remote: Remote,
        schema: SchemaRef,
        sent: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl SQLExecutor for Stub {
        fn name(&self) -> &str {
            match self.remote {
                #[cfg(feature = "postgres")]
                Remote::Postgres => "postgres",
                #[cfg(feature = "mysql")]
                Remote::MySql => "mysql",
            }
        }

        fn compute_context(&self) -> Option<String> {
            Some("stub".into())
        }

        fn dialect(&self) -> Arc<dyn Dialect> {
            match self.remote {
                #[cfg(feature = "postgres")]
                Remote::Postgres => Arc::new(datafusion::sql::unparser::dialect::PostgreSqlDialect {}),
                #[cfg(feature = "mysql")]
                Remote::MySql => Arc::new(datafusion::sql::unparser::dialect::MySqlDialect {}),
            }
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

    /// The engine's federation planning: the remote planner beneath
    /// DataFusion's own.
    #[derive(Debug)]
    struct Planner;

    #[async_trait]
    impl QueryPlanner for Planner {
        async fn create_physical_plan(
            &self,
            plan: &LogicalPlan,
            state: &SessionState,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(RemotePlanner::new())])
                .create_physical_plan(plan, state)
                .await
        }
    }

    /// `is_even(Int32) -> Boolean`, a function a session installs: it
    /// exists in the engine and on no remote.
    fn is_even() -> ScalarUDF {
        use datafusion::arrow::array::{BooleanArray, Int32Array};
        use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};

        #[derive(Debug, PartialEq, Eq, Hash)]
        struct IsEven(Signature);
        impl ScalarUDFImpl for IsEven {
            fn name(&self) -> &str {
                "is_even"
            }
            fn signature(&self) -> &Signature {
                &self.0
            }
            fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
                Ok(DataType::Boolean)
            }
            fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
                let arrays = ColumnarValue::values_to_arrays(&args.args)?;
                let ids = arrays[0].as_any().downcast_ref::<Int32Array>().unwrap();
                let even: BooleanArray = ids.iter().map(|id| id.map(|id| id % 2 == 0)).collect();
                Ok(ColumnarValue::Array(Arc::new(even)))
            }
        }
        ScalarUDF::new_from_impl(IsEven(Signature::exact(
            vec![DataType::Int32],
            Volatility::Immutable,
        )))
    }

    /// A session federating as the engine's does, over `remote`'s table `t`
    /// (and the same rows in a file, `f`), with `is_even` installed, and the
    /// statements the remote is sent.
    fn session(remote: Remote) -> (SessionContext, Arc<Mutex<Vec<String>>>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("status", DataType::Utf8, true),
            Field::new("amount", DataType::Decimal128(10, 2), true),
            Field::new("bought", DataType::Date32, true),
            Field::new("delivered", DataType::Date32, true),
        ]));
        let sent = Arc::new(Mutex::new(Vec::new()));
        let executor = Arc::new(Stub {
            remote,
            schema: Arc::clone(&schema),
            sent: Arc::clone(&sent),
        });
        let table = MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap();
        let provider = table_provider(
            remote,
            TableReference::bare("t"),
            Arc::new(table),
            executor,
            None,
        );
        let base = SessionContext::new().state();
        let mut rules = base.optimizer().rules.clone();
        let at = rules
            .iter()
            .position(|rule| rule.name() == "scalar_subquery_to_join")
            .unwrap()
            + 1;
        rules.insert(at, Arc::new(RemoteFederation::default()));
        let state = SessionStateBuilder::new_from_existing(base)
            .with_optimizer_rules(rules)
            .with_query_planner(Arc::new(Planner))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table("t", provider).unwrap();
        ctx.register_table(
            "f",
            Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap()),
        )
        .unwrap();
        ctx.register_udf(is_even());
        (ctx, sent)
    }

    /// The statements `remote` is sent to answer `sql`, and the physical plan.
    async fn sent(remote: Remote, sql: &str) -> (Vec<String>, String) {
        let (ctx, sent) = session(remote);
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

    /// Whether `remote` evaluates the expression `sql` over `t` as the engine
    /// does.
    async fn runs(remote: Remote, sql: &str) -> bool {
        let (ctx, _) = session(remote);
        let plan = ctx
            .sql(&format!("SELECT {sql} FROM t"))
            .await
            .unwrap()
            .into_unoptimized_plan();
        let LogicalPlan::Projection(projection) = plan else {
            panic!("{plan}")
        };
        expr_runs(remote, &projection.expr[0], projection.input.schema())
    }

    /// `node` sits above `RemoteExec` in `plan`.
    fn above_remote(plan: &str, node: &str) -> bool {
        plan.find(node)
            .zip(plan.find("RemoteExec"))
            .is_some_and(|(node, remote)| node < remote)
    }

    /// A function the engine has and no remote does stays in the engine, on
    /// every remote: an installed one, and DataFusion's own.
    #[tokio::test]
    async fn no_remote_is_sent_a_function_it_does_not_have() {
        for remote in REMOTES {
            for sql in [
                "SELECT id FROM t WHERE is_even(id)",
                "SELECT approx_distinct(id) AS n FROM t",
                "SELECT date_trunc('month', bought) AS m FROM t WHERE id = 1",
            ] {
                let (statements, plan) = sent(remote, sql).await;
                assert!(
                    statements.iter().all(|s| !s.contains("is_even")
                        && !s.contains("approx_distinct")
                        && !s.contains("date_trunc")),
                    "{remote:?} {sql}: {statements:?}\n{plan}"
                );
            }
        }
    }

    /// `unnest` stays in the engine: federation never hands the federation
    /// rule a plan holding one (it rebuilds an `Unnest` from expressions the
    /// node refuses), over a remote or a file.
    #[tokio::test]
    async fn unnest_stays_in_the_engine() {
        for remote in REMOTES {
            for table in ["t", "f"] {
                let sql = format!(
                    "SELECT unnest(string_to_array(status, ',')) AS item, COUNT(*) AS n \
                     FROM {table} GROUP BY unnest(string_to_array(status, ','))"
                );
                let (statements, _) = sent(remote, &sql).await;
                assert!(
                    statements.iter().all(|s| !s.contains("unnest")),
                    "{remote:?} {table}: {statements:?}"
                );
            }
        }
    }

    const REMOTES: [Remote; REMOTE_COUNT] = [
        #[cfg(feature = "postgres")]
        Remote::Postgres,
        #[cfg(feature = "mysql")]
        Remote::MySql,
    ];
    const REMOTE_COUNT: usize = cfg!(feature = "postgres") as usize + cfg!(feature = "mysql") as usize;

    #[cfg(feature = "mysql")]
    mod mysql {
        use super::*;

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
            ] {
                assert_eq!(sent(Remote::MySql, sql).await.0, [statement], "{sql}");
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
            ] {
                assert_eq!(sent(Remote::MySql, sql).await.0, [statement], "{sql}");
            }
        }

        /// A query whose every node above the scan stays in the engine is
        /// not federated at all; the scan's own filter and sort pushdown
        /// would write them into its statement, and does not.
        #[tokio::test]
        async fn a_scan_that_is_not_federated_takes_no_filter_or_sort_from_above() {
            for sql in [
                // `=` on strings follows MySQL's collation, case-insensitive.
                "SELECT id FROM t WHERE status = 'Delivered' ORDER BY id",
                "SELECT status, COUNT(*) AS n FROM t GROUP BY status ORDER BY status",
            ] {
                let (statements, plan) = sent(Remote::MySql, sql).await;
                assert!(statements.is_empty(), "{sql}: {statements:?}");
                assert!(above_remote(&plan, "SortExec"), "{plan}");
            }
            let (_, plan) = sent(Remote::MySql, "SELECT id FROM t WHERE status = 'Delivered'").await;
            assert!(above_remote(&plan, "FilterExec"), "{plan}");
        }
    }

    #[cfg(feature = "postgres")]
    mod postgres {
        use super::*;

        /// What Postgres reads otherwise stays in the engine: an ordering
        /// comparison or a sort on strings (Postgres orders by collation),
        /// and an `AVG` over decimals (Postgres carries more places than the
        /// engine planned).
        #[tokio::test]
        async fn what_postgres_reads_differently_stays_in_the_engine() {
            for (sql, never_sent, kept) in [
                ("SELECT id FROM t WHERE status < 'm'", "< 'm'", "FilterExec"),
                ("SELECT id FROM t ORDER BY status", "ORDER BY", "SortExec"),
                ("SELECT AVG(amount) AS a FROM t", "avg(", "AggregateExec"),
            ] {
                let (statements, plan) = sent(Remote::Postgres, sql).await;
                assert!(
                    statements
                        .iter()
                        .all(|s| !s.to_lowercase().contains(&never_sent.to_lowercase())),
                    "{sql}: {statements:?}"
                );
                assert!(above_remote(&plan, kept), "{sql}\n{plan}");
            }
        }

        #[tokio::test]
        async fn what_postgres_reads_as_the_engine_does_is_sent_to_it() {
            for sql in [
                "SELECT id FROM t WHERE status = 'SP'",
                "SELECT status || '-x' AS v FROM t WHERE id = 1",
                "SELECT id / 2 AS half FROM t WHERE id % 2 = 0",
                "SELECT id FROM t ORDER BY delivered ASC NULLS LAST LIMIT 2",
                "SELECT status, COUNT(*) AS n FROM t GROUP BY status",
                "SELECT COUNT(DISTINCT status) AS n FROM t",
                "SELECT AVG(id) AS a FROM t",
            ] {
                let (statements, plan) = sent(Remote::Postgres, sql).await;
                assert_eq!(statements.len(), 1, "{sql}: {statements:?}\n{plan}");
                assert!(
                    !plan.contains("FilterExec") && !plan.contains("AggregateExec"),
                    "{sql}: the remote answers it whole\n{plan}"
                );
            }
        }
    }

    /// The allowlist, per remote: which expressions each runs as the engine
    /// does, as `[postgres, mysql]`.
    #[tokio::test]
    async fn the_allowlist() {
        let column = |remote: Remote| match remote {
            #[cfg(feature = "postgres")]
            Remote::Postgres => 0,
            #[cfg(feature = "mysql")]
            Remote::MySql => 1,
        };
        for (expr, expected) in [
            ("id", [true, true]),
            ("id + 1", [true, true]),
            ("id * 2 - 1", [true, true]),
            ("id > 1 AND bought < delivered", [true, true]),
            ("id BETWEEN 1 AND 3", [true, true]),
            ("id IN (1, 2)", [true, true]),
            ("delivered IS NULL", [true, true]),
            ("CAST(id AS BIGINT)", [true, true]),
            ("id / 2", [true, false]),
            ("status = 'x'", [true, false]),
            ("status IN ('a', 'b')", [true, false]),
            ("status || 'x'", [true, false]),
            ("status < 'x'", [false, false]),
            ("amount / 2", [false, false]),
            ("delivered - bought", [false, false]),
            ("upper(status)", [false, false]),
            ("is_even(id)", [false, false]),
            ("date_trunc('month', bought)", [false, false]),
            ("CAST(status AS INT)", [false, false]),
            ("CAST(id AS VARCHAR)", [false, false]),
        ] {
            for remote in REMOTES {
                assert_eq!(
                    runs(remote, expr).await,
                    expected[column(remote)],
                    "{remote:?}: {expr}"
                );
            }
        }
    }
}
