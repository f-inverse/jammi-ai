//! A Postgres table's columns, as the engine reads them.
//!
//! The driver names a column's Arrow type from its type name and converts
//! each value from its binary form, and neither covers every type, nor is
//! right for every type it covers: a domain reports its own name and fails
//! the whole source's registration, as `jsonb`, `citext` and `timetz` do;
//! `tsvector`, `inet`, `xml` and `bit` register and then fail every query
//! that selects them; `money` reads as its integer count of minor units; and
//! a `timestamp` before 1970 is dropped from its column, failing the batch.
//!
//! So the engine reads a table's columns from Postgres's own catalog
//! ([`columns`]), each through its effective type (a domain's base type),
//! and decides column by column how to read it ([`Read`]): as the driver
//! reads it, for a type the driver reads correctly, or through an
//! expression Postgres evaluates first — `numeric` for `money`, the instant
//! at UTC for a `timestamp`, `text` for every other type. A table with any
//! column read through an expression is read through a derived table
//! ([`DerivedTable`]) that every statement Postgres receives names in its
//! place, so Postgres filters, sorts and aggregates the values as read: a
//! federated query takes it from the table's AST analyzer, a scan that is
//! not federated in the scan's own statement ([`DerivedScan`]).

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, IntervalUnit, Schema, SchemaRef, TimeUnit};
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
use datafusion_table_providers::postgres::DynPostgresConnectionPool;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use datafusion_table_providers::sql::db_connection_pool::DbConnectionPool;
use datafusion_table_providers::sql::sql_provider_datafusion::{SqlExec, SqlTable};

use crate::federation::remote::{self, Reads, Remote};

/// Digits of a `numeric` that holds every `money` value: an `int8` has 19.
const MONEY_PRECISION: u8 = 19;

/// The driver's reading of an unconstrained `numeric`.
const NUMERIC_DEFAULT: (u8, i8) = (38, 20);

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// How the engine reads one column.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Read {
    /// As the driver reads the column's type.
    Direct,
    /// `money` as `numeric(19, scale)`, the database's money scale.
    Money { scale: i8 },
    /// A `timestamp` as the instant it names at UTC, which the driver reads
    /// for every value; the session's `TimeZone` is pinned to UTC.
    AtUtc,
    /// As `text`.
    Text,
}

/// One column of a table, as the engine reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Column {
    name: String,
    data_type: DataType,
    nullable: bool,
    /// The base type a domain column is cast to before it is read.
    domain: Option<String>,
    read: Read,
}

impl Column {
    /// The SQL that reads the column, or `None` when it is read as stored.
    fn expression(&self) -> Option<String> {
        let column = Ident::with_quote('"', &self.name).to_string();
        let value = match &self.domain {
            Some(base) => format!("({column})::{base}"),
            None => column,
        };
        match &self.read {
            Read::Direct => self.domain.as_ref().map(|_| value),
            Read::Money { scale } => Some(format!("({value})::numeric({MONEY_PRECISION}, {scale})")),
            Read::AtUtc => Some(format!("({value}) AT TIME ZONE 'UTC'")),
            Read::Text => Some(format!("({value})::text")),
        }
    }
}

/// One column as the catalog describes it.
struct Described {
    schema: String,
    table: String,
    name: String,
    not_null: bool,
    /// The effective type's name: a domain's base type.
    type_name: String,
    /// The effective type with its modifier, as Postgres spells it.
    formatted: String,
    domain: bool,
    /// An array's element type name.
    element: Option<String>,
}

/// The columns of every table, view and materialized view of every schema
/// the connecting role may use — the system schemas aside — in column
/// order, by `(schema, relation)`.
pub(super) async fn columns(
    pool: &PostgresConnectionPool,
) -> Result<BTreeMap<(String, String), Vec<Column>>, BoxError> {
    let connection = pool.connect_direct().await?;
    let rows = connection
        .conn
        .query(
            "SELECT n.nspname::text, c.relname::text, a.attname::text, a.attnotnull, \
                    et.typname::text, \
                    format_type(et.oid, CASE WHEN t.typtype = 'd' THEN t.typtypmod \
                                             ELSE a.atttypmod END), \
                    t.typtype = 'd', el.typname::text \
             FROM pg_attribute a \
             JOIN pg_class c ON c.oid = a.attrelid \
             JOIN pg_namespace n ON n.oid = c.relnamespace \
             JOIN pg_type t ON t.oid = a.atttypid \
             JOIN pg_type et ON et.oid = CASE WHEN t.typtype = 'd' THEN t.typbasetype \
                                              ELSE t.oid END \
             LEFT JOIN pg_type el ON el.oid = et.typelem AND et.typcategory = 'A' \
             WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') \
               AND n.nspname NOT LIKE 'pg\\_toast%' AND n.nspname NOT LIKE 'pg\\_temp%' \
               AND has_schema_privilege(n.oid, 'USAGE') \
               AND c.relkind IN ('r', 'p', 'v', 'm', 'f') \
               AND NOT c.relispartition AND a.attnum > 0 AND NOT a.attisdropped \
             ORDER BY n.nspname, c.relname, a.attnum",
            &[],
        )
        .await?;
    let described = rows
        .iter()
        .map(|row| {
            Ok(Described {
                schema: row.try_get(0)?,
                table: row.try_get(1)?,
                name: row.try_get(2)?,
                not_null: row.try_get(3)?,
                type_name: row.try_get(4)?,
                formatted: row.try_get(5)?,
                domain: row.try_get(6)?,
                element: row.try_get(7)?,
            })
        })
        .collect::<Result<Vec<_>, BoxError>>()?;
    let money_scale = if described.iter().any(|d| d.type_name == "money") {
        Some(money_scale(pool).await?)
    } else {
        None
    };
    let mut tables: BTreeMap<(String, String), Vec<Column>> = BTreeMap::new();
    for column in described {
        let (data_type, read) = reading(&column, money_scale)?;
        tables.entry((column.schema, column.table)).or_default().push(Column {
            name: column.name,
            data_type,
            nullable: !column.not_null,
            domain: column.domain.then_some(column.formatted),
            read,
        });
    }
    Ok(tables)
}

/// The database's money scale: the fractional digits `lc_monetary` gives a
/// `money` value, as Postgres itself converts one to `numeric`.
async fn money_scale(pool: &PostgresConnectionPool) -> Result<i8, BoxError> {
    let connection = pool.connect_direct().await?;
    let row = connection
        .conn
        .query_one("SELECT scale('0'::money::numeric)", &[])
        .await?;
    let scale: i32 = row.try_get(0)?;
    Ok(i8::try_from(scale)?)
}

/// The Arrow type a column is read as, and how. The types read as stored
/// are those the driver both names and converts correctly; every other type
/// is read through Postgres. `money_scale` is the database's, read whenever
/// a `money` column exists.
fn reading(column: &Described, money_scale: Option<i8>) -> Result<(DataType, Read), BoxError> {
    let direct = |data_type| (data_type, Read::Direct);
    Ok(match column.type_name.as_str() {
        "int2" => direct(DataType::Int16),
        "int4" => direct(DataType::Int32),
        "int8" => direct(DataType::Int64),
        "float4" => direct(DataType::Float32),
        "float8" => direct(DataType::Float64),
        "bool" => direct(DataType::Boolean),
        "numeric" => match numeric_modifier(&column.formatted) {
            Some((precision, scale)) => direct(DataType::Decimal128(precision, scale)),
            None if column.formatted == "numeric" => {
                direct(DataType::Decimal128(NUMERIC_DEFAULT.0, NUMERIC_DEFAULT.1))
            }
            None => (DataType::Utf8, Read::Text),
        },
        "text" | "varchar" | "bpchar" | "name" | "uuid" => direct(DataType::Utf8),
        "bytea" => direct(DataType::Binary),
        "date" => direct(DataType::Date32),
        "time" => direct(DataType::Time64(TimeUnit::Nanosecond)),
        "timestamptz" => direct(DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))),
        "interval" => direct(DataType::Interval(IntervalUnit::MonthDayNano)),
        "timestamp" => (DataType::Timestamp(TimeUnit::Nanosecond, None), Read::AtUtc),
        "money" => {
            let scale = money_scale.ok_or_else(|| {
                format!("`{}` is money, and the money scale was not read", column.name)
            })?;
            (DataType::Decimal128(MONEY_PRECISION, scale), Read::Money { scale })
        }
        _ => match column.element.as_deref().and_then(array_element) {
            Some(element) => direct(DataType::List(Arc::new(Field::new("item", element, true)))),
            None => (DataType::Utf8, Read::Text),
        },
    })
}

/// The element type of an array the driver reads as stored.
fn array_element(type_name: &str) -> Option<DataType> {
    Some(match type_name {
        "int2" => DataType::Int16,
        "int4" => DataType::Int32,
        "int8" => DataType::Int64,
        "float4" => DataType::Float32,
        "float8" => DataType::Float64,
        "bool" => DataType::Boolean,
        "text" | "varchar" | "bpchar" => DataType::Utf8,
        _ => return None,
    })
}

/// `(precision, scale)` of a `numeric(p, s)` or `numeric(p)` spelling.
fn numeric_modifier(formatted: &str) -> Option<(u8, i8)> {
    let inner = formatted.strip_prefix("numeric(")?.strip_suffix(')')?;
    let mut parts = inner.split(',').map(str::trim);
    let precision = parts.next()?.parse().ok()?;
    let scale = parts.next().map_or(Some(0), |s| s.parse().ok())?;
    Some((precision, scale))
}

/// The engine's schema of a table whose columns are `columns`.
pub(super) fn schema(columns: &[Column]) -> SchemaRef {
    Arc::new(Schema::new(
        columns
            .iter()
            .map(|c| Field::new(&c.name, c.data_type.clone(), c.nullable))
            .collect::<Vec<_>>(),
    ))
}

/// `table`, whose columns are `columns`, as a [`Remote::Postgres`] table:
/// the driver's scan and executor over the engine's schema, through a
/// derived table when any column is read through an expression.
pub(super) fn remote_table(
    pool: &Arc<PostgresConnectionPool>,
    table: TableReference,
    columns: &[Column],
) -> Arc<dyn TableProvider> {
    let pool: Arc<DynPostgresConnectionPool> = Arc::clone(pool) as _;
    let schema = schema(columns);
    let sql_table = Arc::new(
        SqlTable::new_with_schema("postgres", &pool, schema, table.clone())
            .with_dialect(Arc::new(PostgresUnparserDialect {})),
    );
    match DerivedTable::of(&table, columns) {
        None => remote::table_provider(
            Remote::Postgres,
            table,
            Arc::clone(&sql_table) as _,
            sql_table,
            None,
        ),
        Some(derived) => {
            let scan = Arc::new(DerivedScan {
                sql_table: Arc::clone(&sql_table),
                pool,
                derived: derived.clone(),
            });
            let reads: Reads = Arc::new(move |statement| derived.rewrite(statement));
            remote::table_provider(Remote::Postgres, table, scan, sql_table, Some(reads))
        }
    }
}

/// The derived table a table with columns read through an expression is
/// read through.
#[derive(Debug, Clone)]
struct DerivedTable {
    /// The table as the statements Postgres receives name it.
    table: TableReference,
    /// Every column of the table in order, and the SQL that reads it when it
    /// is not read as stored.
    columns: Vec<(String, Option<String>)>,
}

impl DerivedTable {
    /// The derived table `table` is read through, or `None` when every
    /// column is read as stored.
    fn of(table: &TableReference, columns: &[Column]) -> Option<Self> {
        let columns = columns
            .iter()
            .map(|c| (c.name.clone(), c.expression()))
            .collect::<Vec<_>>();
        columns
            .iter()
            .any(|(_, read)| read.is_some())
            .then(|| Self {
                table: table.clone(),
                columns,
            })
    }

    /// `statement` with every reference to the table replaced by the derived
    /// table that reads its columns.
    fn rewrite(&self, mut statement: Statement) -> DataFusionResult<Statement> {
        match statement.visit(&mut ReplaceTable { derived: self }) {
            ControlFlow::Continue(()) => Ok(statement),
            ControlFlow::Break(error) => Err(error),
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
            .map(|(column, read)| {
                let column = Ident::with_quote('"', column);
                match read {
                    Some(read) => format!("{read} AS {column}"),
                    None => column.to_string(),
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("SELECT {select} FROM {name}");
        let subquery = Parser::new(&PostgreSqlDialect {})
            .try_with_sql(&sql)
            .and_then(|mut parser| parser.parse_query())
            .map_err(|e| DataFusionError::Plan(format!("reading the columns of `{name}`: {e}")))?;
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

/// Replaces each reference to one table with its [`DerivedTable`], breaking
/// with the error when one cannot be built. It runs after a factor's
/// children are visited, so the derived table — which names the table
/// again — is never itself revisited.
struct ReplaceTable<'a> {
    derived: &'a DerivedTable,
}

impl VisitorMut for ReplaceTable<'_> {
    type Break = DataFusionError;

    fn post_visit_table_factor(
        &mut self,
        factor: &mut TableFactor,
    ) -> ControlFlow<DataFusionError> {
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
            Err(error) => ControlFlow::Break(error),
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

/// The table scanned where a query is not federated: the driver's scan of
/// it, its statement reading through the derived table.
struct DerivedScan<T: 'static, P: 'static> {
    sql_table: Arc<SqlTable<T, P>>,
    pool: Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
    derived: DerivedTable,
}

impl<T, P> std::fmt::Debug for DerivedScan<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DerivedScan")
            .field("derived", &self.derived)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<T: 'static, P: 'static> TableProvider for DerivedScan<T, P> {
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
        let mut statements = Parser::parse_sql(&PostgreSqlDialect {}, &sql).map_err(|e| {
            DataFusionError::Plan(format!("reading the columns of `{}`: {e}", self.derived.table))
        })?;
        let (Some(statement), None) = (statements.pop(), statements.pop()) else {
            return Err(DataFusionError::Plan(format!(
                "reading the columns of `{}`: `{sql}` is not one statement",
                self.derived.table
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

    fn described(type_name: &str, formatted: &str, domain: bool, element: Option<&str>) -> Described {
        Described {
            schema: "public".into(),
            table: "t".into(),
            name: "c".into(),
            not_null: false,
            type_name: type_name.into(),
            formatted: formatted.into(),
            domain,
            element: element.map(Into::into),
        }
    }

    fn column(name: &str, read: Read, domain: Option<&str>) -> Column {
        Column {
            name: name.into(),
            data_type: DataType::Utf8,
            nullable: true,
            domain: domain.map(Into::into),
            read,
        }
    }

    fn prices() -> DerivedTable {
        DerivedTable::of(
            &TableReference::bare("prices"),
            &[
                column("id", Read::Direct, None),
                column("amount", Read::Money { scale: 2 }, None),
            ],
        )
        .unwrap()
    }

    fn rewritten(derived: &DerivedTable, sql: &str) -> String {
        let statement = Parser::parse_sql(&PostgreSqlDialect {}, sql)
            .unwrap()
            .remove(0);
        derived.rewrite(statement).unwrap().to_string()
    }

    /// Every type the driver reads correctly is read as stored; a domain is
    /// cast to its base first; every other type is read through Postgres.
    #[test]
    fn each_type_is_read_as_the_driver_reads_it_or_through_postgres() {
        for (type_name, formatted, domain, element, data_type, read) in [
            ("int4", "integer", false, None, DataType::Int32, Read::Direct),
            ("numeric", "numeric(10,2)", false, None, DataType::Decimal128(10, 2), Read::Direct),
            ("numeric", "numeric", false, None, DataType::Decimal128(38, 20), Read::Direct),
            ("numeric", "numeric(7)", false, None, DataType::Decimal128(7, 0), Read::Direct),
            ("int4", "integer", true, None, DataType::Int32, Read::Direct),
            (
                "timestamp",
                "timestamp without time zone",
                false,
                None,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                Read::AtUtc,
            ),
            ("money", "money", false, None, DataType::Decimal128(19, 2), Read::Money { scale: 2 }),
            ("jsonb", "jsonb", false, None, DataType::Utf8, Read::Text),
            ("json", "json", false, None, DataType::Utf8, Read::Text),
            ("citext", "citext", false, None, DataType::Utf8, Read::Text),
            ("timetz", "time with time zone", false, None, DataType::Utf8, Read::Text),
            ("tsvector", "tsvector", false, None, DataType::Utf8, Read::Text),
            ("inet", "inet", false, None, DataType::Utf8, Read::Text),
            (
                "_int4",
                "integer[]",
                false,
                Some("int4"),
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                Read::Direct,
            ),
            ("_jsonb", "jsonb[]", false, Some("jsonb"), DataType::Utf8, Read::Text),
        ] {
            let column = described(type_name, formatted, domain, element);
            assert_eq!(
                reading(&column, Some(2)).unwrap(),
                (data_type, read),
                "{formatted}"
            );
        }
    }

    #[test]
    fn a_column_read_as_stored_names_itself_and_a_domain_reads_its_base() {
        assert_eq!(column("c", Read::Direct, None).expression(), None);
        assert_eq!(
            column("c", Read::Direct, Some("integer")).expression().as_deref(),
            Some(r#"("c")::integer"#)
        );
        assert_eq!(
            column("c", Read::Text, Some("numeric(4,1)")).expression().as_deref(),
            Some(r#"(("c")::numeric(4,1))::text"#)
        );
        assert_eq!(
            column("born", Read::AtUtc, None).expression().as_deref(),
            Some(r#"("born") AT TIME ZONE 'UTC'"#)
        );
    }

    #[test]
    fn a_table_read_as_stored_has_no_derived_table() {
        assert!(DerivedTable::of(
            &TableReference::bare("t"),
            &[column("id", Read::Direct, None), column("name", Read::Direct, None)]
        )
        .is_none());
    }

    #[test]
    fn a_reference_reads_through_the_derived_table_under_the_tables_own_name() {
        let sql = rewritten(
            &prices(),
            r#"SELECT sum("prices"."amount") FROM "prices" WHERE "prices"."amount" > 1"#,
        );
        assert!(
            sql.starts_with(r#"SELECT sum("prices"."amount") FROM (SELECT "id", "#)
                && sql.ends_with(r#" AS "amount" FROM "prices") AS "prices" WHERE "prices"."amount" > 1"#),
            "{sql}"
        );
        assert_eq!(sql.matches("NUMERIC(19, 2)").count() + sql.matches("NUMERIC(19,2)").count(), 1, "{sql}");
    }

    #[test]
    fn an_aliased_reference_keeps_its_alias_and_other_tables_are_untouched() {
        let sql = rewritten(
            &prices(),
            r#"SELECT p."amount" FROM "public"."prices" AS p JOIN "items" AS i ON p."id" = i."id""#,
        );
        assert!(sql.contains(r#"FROM "public"."prices") AS p JOIN "items" AS i"#), "{sql}");
    }

    #[test]
    fn every_reference_is_replaced_in_subqueries_too() {
        let sql = rewritten(
            &prices(),
            r#"SELECT a."id" FROM "prices" AS a WHERE a."amount" > (SELECT avg(b."amount") FROM "prices" AS b)"#,
        );
        assert_eq!(sql.matches(r#"AS "amount" FROM "prices")"#).count(), 2, "{sql}");
    }

    #[test]
    fn a_column_name_is_quoted_whole() {
        let derived = DerivedTable::of(
            &TableReference::bare("t"),
            &[column(r#"pr"ice"#, Read::Text, None)],
        )
        .unwrap();
        let sql = rewritten(&derived, r#"SELECT * FROM "t""#);
        assert!(sql.contains(r#"("pr""ice")"#) && sql.contains(r#"AS "pr""ice" FROM "t") AS "t""#), "{sql}");
    }
}
