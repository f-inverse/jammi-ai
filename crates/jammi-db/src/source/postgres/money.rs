//! A Postgres `money` column served as the amount it holds.
//!
//! Postgres sends `money` in binary as an `int8` count of the currency's
//! minor unit, and the driver passes that count through as the value —
//! `12.34` reads as `1234`. Nor can the column be relabelled a decimal and
//! left as it is: every query that reaches the database compares or sums it
//! there, and Postgres has no operator between `money` and a number, so
//! `amount > 1` would be refused.
//!
//! A table with a `money` column is therefore served through a derived table
//! instead: every statement sent to Postgres reads it as
//! `(SELECT …, "amount"::numeric(19, s) AS "amount" FROM "prices") AS "prices"`,
//! so Postgres filters, sorts and sums `numeric`, the column is declared
//! `Decimal128(19, s)`, and the driver's own `numeric` conversion reads it.
//! `s` is the database's money scale (`lc_monetary`'s fractional digits,
//! read once when the source is registered); 19 digits hold every `int8`.
//! A federated query takes the derived table from [`MoneyTable`]'s AST
//! analyzer; a scan that is not federated takes it from [`MoneyScan`].

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::TableReference;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::sql::sqlparser::ast::{
    Ident, ObjectName, Statement, TableAlias, TableFactor, VisitMut, VisitorMut,
};
use datafusion::sql::sqlparser::dialect::PostgreSqlDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::unparser::dialect::PostgreSqlDialect as PostgresUnparserDialect;
use datafusion_federation::sql::{AstAnalyzer, SQLFederationProvider, SQLTable, SQLTableSource};
use datafusion_federation::FederatedTableProviderAdaptor;
use datafusion_table_providers::postgres::DynPostgresConnectionPool;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use datafusion_table_providers::sql::db_connection_pool::DbConnectionPool;
use datafusion_table_providers::sql::sql_provider_datafusion::{SqlExec, SqlTable};

/// Digits of a `numeric` that holds every `money` value: an `int8` has 19.
const PRECISION: u8 = 19;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The `money` columns of each table in `schema`, by table name.
pub(super) async fn money_columns(
    pool: &PostgresConnectionPool,
    schema: &str,
) -> Result<HashMap<String, HashSet<String>>, BoxError> {
    let connection = pool.connect_direct().await?;
    let rows = connection
        .conn
        .query(
            "SELECT table_name::text, column_name::text FROM information_schema.columns \
             WHERE table_schema = $1 AND data_type = 'money'",
            &[&schema],
        )
        .await?;
    let mut columns: HashMap<String, HashSet<String>> = HashMap::new();
    for row in rows {
        columns
            .entry(row.try_get(0)?)
            .or_default()
            .insert(row.try_get(1)?);
    }
    Ok(columns)
}

/// The database's money scale: the fractional digits `lc_monetary` gives a
/// `money` value, as Postgres itself converts one to `numeric`.
pub(super) async fn money_scale(pool: &PostgresConnectionPool) -> Result<i8, BoxError> {
    let connection = pool.connect_direct().await?;
    let row = connection
        .conn
        .query_one("SELECT scale('0'::money::numeric)", &[])
        .await?;
    let scale: i32 = row.try_get(0)?;
    Ok(i8::try_from(scale)?)
}

/// The provider for `table`, whose `money` columns are `money`, each served
/// as a `Decimal128(19, scale)`.
pub(super) async fn table_provider(
    pool: &Arc<PostgresConnectionPool>,
    table: TableReference,
    money: &HashSet<String>,
    scale: i8,
) -> Result<Arc<dyn TableProvider>, BoxError> {
    let dyn_pool: Arc<DynPostgresConnectionPool> = Arc::clone(pool) as _;
    let driver_schema = SqlTable::new("postgres", &dyn_pool, table.clone())
        .await?
        .schema();
    let schema = Arc::new(Schema::new_with_metadata(
        driver_schema
            .fields()
            .iter()
            .map(|field| {
                if money.contains(field.name()) {
                    Arc::new(Field::new(
                        field.name(),
                        DataType::Decimal128(PRECISION, scale),
                        field.is_nullable(),
                    ))
                } else {
                    Arc::clone(field)
                }
            })
            .collect::<Vec<_>>(),
        driver_schema.metadata().clone(),
    ));
    let derived = DerivedTable {
        table: table.clone(),
        columns: schema
            .fields()
            .iter()
            .map(|field| (field.name().clone(), money.contains(field.name())))
            .collect(),
        scale,
    };
    let sql_table = Arc::new(
        SqlTable::new_with_schema("postgres", &dyn_pool, Arc::clone(&schema), table)
            .with_dialect(Arc::new(PostgresUnparserDialect {})),
    );
    let source = SQLTableSource::new_with_table(
        Arc::new(SQLFederationProvider::new(Arc::clone(&sql_table) as _)),
        Arc::new(MoneyTable {
            schema,
            derived: derived.clone(),
        }),
    );
    Ok(Arc::new(FederatedTableProviderAdaptor::new_with_provider(
        Arc::new(source),
        Arc::new(MoneyScan {
            sql_table,
            pool: dyn_pool,
            derived,
        }),
    )))
}

/// The derived table a table with `money` columns is read through.
#[derive(Debug, Clone)]
struct DerivedTable {
    /// The table as the statements Postgres receives name it.
    table: TableReference,
    /// Every column of the table in order, and whether it is `money`.
    columns: Vec<(String, bool)>,
    scale: i8,
}

impl DerivedTable {
    /// `statement` with every reference to the table replaced by the derived
    /// table that reads its `money` columns as `numeric`.
    fn rewrite(&self, mut statement: Statement) -> DataFusionResult<Statement> {
        let mut visitor = ReplaceTable {
            derived: self,
            error: None,
        };
        let _ = statement.visit(&mut visitor);
        match visitor.error {
            Some(error) => Err(error),
            None => Ok(statement),
        }
    }

    /// The derived table standing in for `name`, aliased as the reference
    /// was or, unaliased, by the table's own name so qualified columns
    /// still resolve.
    fn factor(
        &self,
        name: &ObjectName,
        alias: Option<TableAlias>,
    ) -> DataFusionResult<TableFactor> {
        let select = self
            .columns
            .iter()
            .map(|(column, money)| {
                let column = Ident::with_quote('"', column);
                if *money {
                    format!("{column}::numeric({PRECISION}, {}) AS {column}", self.scale)
                } else {
                    column.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("SELECT {select} FROM {name}");
        let subquery = Parser::new(&PostgreSqlDialect {})
            .try_with_sql(&sql)
            .and_then(|mut parser| parser.parse_query())
            .map_err(|e| DataFusionError::Plan(format!("reading `money` as numeric: {e}")))?;
        Ok(TableFactor::Derived {
            lateral: false,
            subquery,
            alias: alias.or_else(|| {
                Some(TableAlias {
                    explicit: true,
                    name: Ident::with_quote('"', self.table.table()),
                    columns: vec![],
                    at: None,
                })
            }),
            sample: None,
        })
    }
}

/// Replaces each reference to one table with its [`DerivedTable`]. It runs
/// after a factor's children are visited, so the derived table — which
/// names the table again — is never itself revisited.
struct ReplaceTable<'a> {
    derived: &'a DerivedTable,
    error: Option<DataFusionError>,
}

impl VisitorMut for ReplaceTable<'_> {
    type Break = ();

    fn post_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
        let TableFactor::Table {
            name,
            alias,
            args: None,
            ..
        } = factor
        else {
            return ControlFlow::Continue(());
        };
        if !names(name).is_some_and(|reference| reference.resolved_eq(&self.derived.table)) {
            return ControlFlow::Continue(());
        }
        match self.derived.factor(name, alias.take()) {
            Ok(derived) => {
                *factor = derived;
                ControlFlow::Continue(())
            }
            Err(error) => {
                self.error = Some(error);
                ControlFlow::Break(())
            }
        }
    }
}

/// The table `name` refers to, when it is one of up to three identifiers.
fn names(name: &ObjectName) -> Option<TableReference> {
    let parts = name
        .0
        .iter()
        .map(|part| part.as_ident().map(|ident| ident.value.clone()))
        .collect::<Option<Vec<_>>>()?;
    match parts.as_slice() {
        [table] => Some(TableReference::bare(table.as_str())),
        [schema, table] => Some(TableReference::partial(schema.as_str(), table.as_str())),
        [catalog, schema, table] => Some(TableReference::full(
            catalog.as_str(),
            schema.as_str(),
            table.as_str(),
        )),
        _ => None,
    }
}

/// The remote table federation sends a query over: its `money` columns are
/// decimals, and the statement reads them through the derived table.
#[derive(Debug)]
struct MoneyTable {
    schema: SchemaRef,
    derived: DerivedTable,
}

impl SQLTable for MoneyTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_reference(&self) -> TableReference {
        self.derived.table.clone()
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn ast_analyzer(&self) -> Option<AstAnalyzer> {
        let derived = self.derived.clone();
        Some(Box::new(move |statement| derived.rewrite(statement)))
    }
}

/// The table scanned where a query is not federated: the driver's scan of
/// it, its statement reading through the derived table.
struct MoneyScan<T: 'static, P: 'static> {
    sql_table: Arc<SqlTable<T, P>>,
    pool: Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
    derived: DerivedTable,
}

impl<T, P> std::fmt::Debug for MoneyScan<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoneyScan")
            .field("derived", &self.derived)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<T: 'static, P: 'static> TableProvider for MoneyScan<T, P> {
    fn schema(&self) -> SchemaRef {
        self.sql_table.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        self.sql_table.supports_filters_pushdown(filters)
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let sql = self.sql_table.scan_to_sql(projection, filters, limit)?;
        let mut statements = Parser::parse_sql(&PostgreSqlDialect {}, &sql)
            .map_err(|e| DataFusionError::Plan(format!("reading `money` as numeric: {e}")))?;
        let (Some(statement), None) = (statements.pop(), statements.pop()) else {
            return Err(DataFusionError::Plan(format!(
                "reading `money` as numeric: `{sql}` is not one statement"
            )));
        };
        let sql = self.derived.rewrite(statement)?.to_string();
        Ok(Arc::new(SqlExec::new(
            projection,
            &self.schema(),
            Arc::clone(&self.pool),
            sql,
            Arc::new(PostgresUnparserDialect {}),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prices() -> DerivedTable {
        DerivedTable {
            table: TableReference::bare("prices"),
            columns: vec![("id".into(), false), ("amount".into(), true)],
            scale: 2,
        }
    }

    fn rewritten(derived: &DerivedTable, sql: &str) -> String {
        let statement = Parser::parse_sql(&PostgreSqlDialect {}, sql)
            .unwrap()
            .remove(0);
        derived.rewrite(statement).unwrap().to_string()
    }

    #[test]
    fn a_reference_reads_money_as_numeric_under_the_tables_own_name() {
        assert_eq!(
            rewritten(
                &prices(),
                r#"SELECT sum("prices"."amount") FROM "prices" WHERE "prices"."amount" > 1"#
            ),
            r#"SELECT sum("prices"."amount") FROM (SELECT "id", "amount"::NUMERIC(19,2) AS "amount" FROM "prices") AS "prices" WHERE "prices"."amount" > 1"#
        );
    }

    #[test]
    fn an_aliased_reference_keeps_its_alias_and_other_tables_are_untouched() {
        assert_eq!(
            rewritten(
                &prices(),
                r#"SELECT p."amount" FROM "public"."prices" AS p JOIN "items" AS i ON p."id" = i."id""#
            ),
            r#"SELECT p."amount" FROM (SELECT "id", "amount"::NUMERIC(19,2) AS "amount" FROM "public"."prices") AS p JOIN "items" AS i ON p."id" = i."id""#
        );
    }

    #[test]
    fn every_reference_is_replaced_in_subqueries_too() {
        let sql = rewritten(
            &prices(),
            r#"SELECT a."id" FROM "prices" AS a WHERE a."amount" > (SELECT avg(b."amount") FROM "prices" AS b)"#,
        );
        assert_eq!(sql.matches("::NUMERIC(19,2)").count(), 2, "{sql}");
    }

    #[test]
    fn a_column_name_is_quoted_whole() {
        let derived = DerivedTable {
            table: TableReference::bare("t"),
            columns: vec![(r#"pr"ice"#.into(), true)],
            scale: 0,
        };
        assert_eq!(
            rewritten(&derived, r#"SELECT * FROM "t""#),
            r#"SELECT * FROM (SELECT "pr""ice"::NUMERIC(19,0) AS "pr""ice" FROM "t") AS "t""#
        );
    }
}
