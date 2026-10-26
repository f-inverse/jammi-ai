//! A database source answers as the engine does: every query here runs over
//! a Postgres table and over the same rows held in a Parquet file, and both
//! answer the same — whatever of it federation sends Postgres and whatever
//! it keeps in the engine. Each construct the Postgres allowlist admits is
//! here, and so is each it keeps out because Postgres reads it otherwise.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion::logical_expr::{create_udf, ColumnarValue, Volatility};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::{JammiSession, QueryFunction};
use jammi_db::source::{FileFormat, SourceConnection, SourceDefinition, SourceType};
use jammi_test_utils::{make_test_session, unique_suffix};
use parquet::arrow::ArrowWriter;
use sqlx::{Connection, Executor, PgConnection};
use tempfile::{tempdir, TempDir};

/// One row: `(id, status, qty, ratio, amount in cents, bought, delivered, tags)`.
type Row = (
    i32,
    Option<&'static str>,
    Option<i32>,
    Option<f64>,
    Option<i128>,
    Option<i32>,
    Option<i32>,
    Option<&'static str>,
);

/// Mixed-case strings, NULLs in every column, negative integers, amounts
/// whose average repeats, and dates on both sides of each other.
const ROWS: [Row; 8] = [
    (1, Some("SP"), Some(7), Some(0.5), Some(100), Some(17_532), Some(17_535), Some("a, b")),
    (2, Some("sp"), Some(-7), Some(1.25), Some(100), Some(17_533), None, Some("b")),
    (3, Some("b"), Some(0), Some(-2.0), Some(200), Some(17_534), Some(17_540), Some("a,c")),
    (4, Some("B"), None, None, None, None, Some(17_541), None),
    (5, Some("a"), Some(3), Some(4.5), Some(250), Some(17_536), Some(17_536), Some("c")),
    (6, None, Some(-3), Some(0.0), Some(-50), Some(17_537), Some(17_550), Some("b, c")),
    (7, Some("SP"), Some(10), Some(9.75), Some(125), Some(17_538), None, Some("a")),
    (8, Some("Sp"), Some(1), Some(3.0), Some(100), Some(17_539), Some(17_545), Some("")),
];

/// The rows as a record batch, typed as the Postgres table reads.
fn batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("qty", DataType::Int32, true),
        Field::new("ratio", DataType::Float64, true),
        Field::new("amount", DataType::Decimal128(10, 2), true),
        Field::new("bought", DataType::Date32, true),
        Field::new("delivered", DataType::Date32, true),
        Field::new("tags", DataType::Utf8, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(ROWS.iter().map(|r| r.0))),
        Arc::new(StringArray::from_iter(ROWS.iter().map(|r| r.1))),
        Arc::new(Int32Array::from_iter(ROWS.iter().map(|r| r.2))),
        Arc::new(Float64Array::from_iter(ROWS.iter().map(|r| r.3))),
        Arc::new(
            Decimal128Array::from_iter(ROWS.iter().map(|r| r.4))
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ),
        Arc::new(Date32Array::from_iter(ROWS.iter().map(|r| r.5))),
        Arc::new(Date32Array::from_iter(ROWS.iter().map(|r| r.6))),
        Arc::new(StringArray::from_iter(ROWS.iter().map(|r| r.7))),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}

/// SQL literal for an optional value.
fn literal<T: ToString>(value: Option<T>, quote: bool) -> String {
    match value {
        None => "NULL".into(),
        Some(v) if quote => format!("'{}'", v.to_string().replace('\'', "''")),
        Some(v) => v.to_string(),
    }
}

/// The date `days` after the epoch, as a Postgres literal.
fn date(days: Option<i32>) -> String {
    days.map_or("NULL".into(), |d| format!("DATE '1970-01-01' + {d}"))
}

/// The same rows in Postgres (`pg_*`) and in a file (`f_*`), and the session
/// reading both, with `is_even` installed.
struct Sources {
    session: JammiSession,
    postgres: String,
    file: String,
    _dirs: [TempDir; 3],
}

async fn sources() -> Sources {
    let pg_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_url(pg_dir.path()).await;
    let mut db = PgConnection::connect(&url).await.unwrap();
    let values = ROWS
        .iter()
        .map(|r| {
            format!(
                "({}, {}, {}, {}, {}, {}, {}, {})",
                r.0,
                literal(r.1, true),
                literal(r.2, false),
                literal(r.3, false),
                r.4.map_or("NULL".into(), |c| format!("{c} / 100.0")),
                date(r.5),
                date(r.6),
                literal(r.7, true),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    db.execute(
        format!(
            "CREATE TABLE orders (id INTEGER NOT NULL, status TEXT, qty INTEGER, \
             ratio DOUBLE PRECISION, amount NUMERIC(10, 2), bought DATE, delivered DATE, \
             tags TEXT); INSERT INTO orders VALUES {values};"
        )
        .as_str(),
    )
    .await
    .unwrap();
    db.close().await.unwrap();

    let files = tempdir().unwrap();
    let path = files.path().join("orders.parquet");
    write_parquet(&path, &batch());

    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    session.install_functions([QueryFunction::Scalar(is_even())]);
    let suffix = unique_suffix();
    let (postgres, file) = (format!("pg_{suffix}"), format!("f_{suffix}"));
    let definition = SourceDefinition::from_url(url.as_str(), None).unwrap();
    session
        .add_source(&postgres, definition.source_type, definition.connection)
        .await
        .unwrap();
    session
        .add_source(
            &file,
            SourceType::File,
            SourceConnection {
                url: Some(path.to_str().unwrap().to_string()),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    Sources {
        session,
        postgres,
        file,
        _dirs: [pg_dir, files, dir],
    }
}

fn write_parquet(path: &Path, batch: &RecordBatch) {
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(path).unwrap(), batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

/// `is_even(Int32) -> Boolean`, a function the session installs: the engine
/// has it and no database does.
fn is_even() -> datafusion::logical_expr::ScalarUDF {
    create_udf(
        "is_even",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| {
            let ids = ColumnarValue::values_to_arrays(args)?;
            let ids = ids[0].as_any().downcast_ref::<Int32Array>().unwrap();
            let even: BooleanArray = ids.iter().map(|id| id.map(|id| id % 2 == 0)).collect();
            Ok(ColumnarValue::Array(Arc::new(even)))
        }),
    )
}

impl Sources {
    /// `sql` with `{T}` naming the table, run over Postgres and over the
    /// file, each rendered as a table.
    async fn both(&self, sql: &str) -> (String, String) {
        let run = |source: &str| {
            let sql = sql.replace("{T}", &format!("{source}.public.orders"));
            async move {
                let batches = self
                    .session
                    .sql(&sql)
                    .await
                    .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
                pretty_format_batches(&batches).unwrap().to_string()
            }
        };
        (run(&self.postgres).await, run(&self.file).await)
    }
}

/// Every construct the Postgres allowlist admits answers as the engine does.
#[tokio::test]
async fn what_postgres_is_sent_it_answers_as_the_engine_does() {
    let sources = sources().await;
    for sql in [
        // String equality is byte equality: 'SP' is not 'sp'.
        "SELECT id FROM {T} WHERE status = 'SP' ORDER BY id",
        "SELECT id FROM {T} WHERE status <> 'SP' ORDER BY id",
        "SELECT id FROM {T} WHERE status IN ('SP', 'b') ORDER BY id",
        "SELECT status, COUNT(*) AS n FROM {T} GROUP BY status ORDER BY n, status",
        "SELECT COUNT(DISTINCT status) AS n FROM {T}",
        "SELECT DISTINCT status FROM {T} WHERE status IS NOT NULL ORDER BY status",
        "SELECT DISTINCT status FROM {T} ORDER BY status",
        "SELECT qty % 2 AS parity, COUNT(*) AS n, SUM(id) AS s FROM {T} GROUP BY qty % 2 ORDER BY 1",
        "SELECT status, SUM(qty) AS q FROM {T} WHERE id > 1 GROUP BY status ORDER BY q, status",
        "SELECT a.id, b.id AS other FROM {T} a JOIN {T} b ON a.status = b.status WHERE a.id < b.id ORDER BY 1, 2",
        // `||` concatenates, NULL in, NULL out.
        "SELECT id, status || '-x' AS v FROM {T} ORDER BY id",
        // Integer division truncates toward zero; `%` takes the dividend's sign.
        "SELECT id, qty / 2 AS h, qty % 4 AS m FROM {T} ORDER BY id",
        "SELECT id, ratio / 4.0 AS q FROM {T} ORDER BY id",
        // Sorts on ordered keys place NULLs as the query says.
        "SELECT id FROM {T} ORDER BY delivered ASC NULLS LAST, id LIMIT 5",
        "SELECT id FROM {T} ORDER BY delivered DESC NULLS FIRST, id LIMIT 5",
        "SELECT id FROM {T} ORDER BY qty, id",
        "SELECT id FROM {T} ORDER BY qty DESC, id",
        // Aggregates.
        "SELECT AVG(id) AS a, AVG(qty) AS q, AVG(ratio) AS r FROM {T}",
        "SELECT SUM(id) AS s, SUM(qty) AS q, SUM(amount) AS a FROM {T}",
        "SELECT MIN(bought) AS lo, MAX(delivered) AS hi, MIN(qty) AS q FROM {T}",
        "SELECT COUNT(*) AS n, COUNT(delivered) AS d FROM {T} WHERE qty > 0",
        // Joins of every kind.
        "SELECT a.id, b.id AS other FROM {T} a FULL JOIN {T} b ON a.id = b.qty ORDER BY 1, 2",
        "SELECT a.id, b.id AS other FROM {T} a LEFT JOIN {T} b ON a.qty = b.id ORDER BY 1, 2",
        // Comparisons and ranges on ordered types.
        "SELECT id FROM {T} WHERE bought BETWEEN DATE '2018-01-02' AND DATE '2018-01-05' ORDER BY id",
        "SELECT id FROM {T} WHERE delivered > bought AND qty >= 0 ORDER BY id",
        "SELECT id, CAST(qty AS BIGINT) * 3 - id AS e FROM {T} ORDER BY id",
    ] {
        let (postgres, engine) = sources.both(sql).await;
        assert_eq!(postgres, engine, "{sql}");
    }
}

/// What Postgres reads otherwise — strings ordered by collation, a decimal
/// `AVG`, DataFusion's own functions and the session's — stays in the
/// engine, and the answer is the engine's.
#[tokio::test]
async fn what_postgres_reads_otherwise_answers_as_the_engine_does() {
    let sources = sources().await;
    for sql in [
        "SELECT id FROM {T} WHERE status < 'm' ORDER BY id",
        "SELECT id, status FROM {T} ORDER BY status, id",
        "SELECT MIN(status) AS lo, MAX(status) AS hi FROM {T}",
        "SELECT AVG(amount) AS a FROM {T}",
        "SELECT approx_distinct(status) AS n FROM {T}",
        "SELECT id FROM {T} WHERE is_even(id) ORDER BY id",
        "SELECT id FROM {T} WHERE is_even(id) AND status = 'SP' ORDER BY id",
        "SELECT date_trunc('month', bought) AS m, COUNT(*) AS n FROM {T} GROUP BY 1 ORDER BY 1",
        "SELECT delivered - bought AS d FROM {T} ORDER BY id",
        "SELECT amount / 3 AS third FROM {T} ORDER BY id",
        "SELECT upper(status) AS u FROM {T} ORDER BY id",
        "SELECT unnest(string_to_array(tags, ',')) AS item, COUNT(*) AS n FROM {T} \
         GROUP BY unnest(string_to_array(tags, ',')) ORDER BY 1",
    ] {
        let (postgres, engine) = sources.both(sql).await;
        assert_eq!(postgres, engine, "{sql}");
    }
}
