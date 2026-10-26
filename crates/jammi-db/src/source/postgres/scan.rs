//! Reading a Postgres table's rows: every statement the engine sends
//! Postgres runs through [`rows`], whether federation wrote it
//! ([`PostgresExecutor`]) or a scan that is not federated did
//! ([`PostgresScan`]).
//!
//! The driver's own reading turns a failed connect into text
//! (`DataFusionError::Execution`), so an unreachable database
//! ([`super::pool::ReachablePool`]'s [`JammiError::Unavailable`]) would reach
//! the caller as an untyped fault; [`rows`] keeps the error as raised.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    project_schema, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::sql::sqlparser::dialect::PostgreSqlDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::unparser::dialect::Dialect;
use datafusion_federation::sql::SQLExecutor;
use datafusion_table_providers::sql::db_connection_pool::dbconnection::query_arrow;
use datafusion_table_providers::sql::db_connection_pool::DbConnectionPool;
use datafusion_table_providers::sql::sql_provider_datafusion::SqlTable;
use futures::TryStreamExt;

use super::columns::DerivedTable;

/// A driver connection pool, over its connection and parameter types.
type Pool<T, P> = Arc<dyn DbConnectionPool<T, P> + Send + Sync>;

/// The rows `sql` reads through `pool`, as `schema`.
fn rows<T: 'static, P: 'static>(
    pool: Pool<T, P>,
    sql: String,
    schema: SchemaRef,
) -> SendableRecordBatchStream {
    let projected = Arc::clone(&schema);
    let read = async move {
        let connection = pool.connect().await.map_err(DataFusionError::External)?;
        query_arrow(connection, sql, Some(projected))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))
    };
    Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(read).try_flatten(),
    ))
}

/// What federation hands the SQL it writes for a Postgres table: the
/// driver's executor in every respect but reading the rows ([`rows`]).
pub(super) struct PostgresExecutor<T: 'static, P: 'static> {
    table: Arc<SqlTable<T, P>>,
    pool: Pool<T, P>,
}

impl<T, P> PostgresExecutor<T, P> {
    pub(super) fn new(table: Arc<SqlTable<T, P>>, pool: Pool<T, P>) -> Self {
        Self { table, pool }
    }
}

impl<T, P> fmt::Debug for PostgresExecutor<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PostgresExecutor").finish_non_exhaustive()
    }
}

#[async_trait]
impl<T: 'static, P: 'static> SQLExecutor for PostgresExecutor<T, P> {
    fn name(&self) -> &str {
        SQLExecutor::name(self.table.as_ref())
    }

    fn compute_context(&self) -> Option<String> {
        self.table.compute_context()
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        SQLExecutor::dialect(self.table.as_ref())
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(rows(Arc::clone(&self.pool), query.to_string(), schema))
    }

    async fn table_names(&self) -> DataFusionResult<Vec<String>> {
        self.table.table_names().await
    }

    async fn get_table_schema(&self, table_name: &str) -> DataFusionResult<SchemaRef> {
        self.table.get_table_schema(table_name).await
    }
}

/// A Postgres table as a scan that is not federated reads it: the driver
/// writes the statement, read through the table's derived table when it has
/// one, and [`rows`] reads it.
pub(super) struct PostgresScan<T: 'static, P: 'static> {
    table: Arc<SqlTable<T, P>>,
    pool: Pool<T, P>,
    derived: Option<DerivedTable>,
}

impl<T, P> PostgresScan<T, P> {
    pub(super) fn new(
        table: Arc<SqlTable<T, P>>,
        pool: Pool<T, P>,
        derived: Option<DerivedTable>,
    ) -> Self {
        Self {
            table,
            pool,
            derived,
        }
    }

    /// The statement Postgres runs for a scan the driver writes as `sql`.
    fn statement(&self, sql: String) -> DataFusionResult<String> {
        let Some(derived) = &self.derived else {
            return Ok(sql);
        };
        let mut statements = Parser::parse_sql(&PostgreSqlDialect {}, &sql)
            .map_err(|e| DataFusionError::Plan(format!("reading `{sql}`: {e}")))?;
        let (Some(statement), None) = (statements.pop(), statements.pop()) else {
            return Err(DataFusionError::Plan(format!(
                "reading `{sql}`: it is not one statement"
            )));
        };
        Ok(derived.rewrite(statement)?.to_string())
    }
}

impl<T, P> fmt::Debug for PostgresScan<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PostgresScan")
            .field("derived", &self.derived)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<T: 'static, P: 'static> TableProvider for PostgresScan<T, P> {
    fn schema(&self) -> SchemaRef {
        self.table.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        self.table.supports_filters_pushdown(filters)
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let sql = self.statement(self.table.scan_to_sql(projection, filters, limit)?)?;
        let schema = project_schema(&self.schema(), projection)?;
        Ok(Arc::new(PostgresRows::new(Arc::clone(&self.pool), sql, schema)))
    }
}

/// One statement's rows, read from Postgres ([`rows`]).
struct PostgresRows<T: 'static, P: 'static> {
    pool: Pool<T, P>,
    sql: String,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl<T, P> PostgresRows<T, P> {
    fn new(pool: Pool<T, P>, sql: String, schema: SchemaRef) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            pool,
            sql,
            schema,
            properties,
        }
    }
}

impl<T, P> fmt::Debug for PostgresRows<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PostgresRows")
            .field("sql", &self.sql)
            .finish_non_exhaustive()
    }
}

impl<T, P> DisplayAs for PostgresRows<T, P> {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PostgresRows: sql={}", self.sql)
    }
}

impl<T: 'static, P: 'static> ExecutionPlan for PostgresRows<T, P> {
    fn name(&self) -> &str {
        "PostgresRows"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(rows(
            Arc::clone(&self.pool),
            self.sql.clone(),
            Arc::clone(&self.schema),
        ))
    }
}
