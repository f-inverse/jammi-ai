//! Every Postgres column type reads as the value it holds, and none fails a
//! source: a domain, `jsonb`, `citext` and `timetz` register; `tsvector`,
//! `inet`, `xml` and `bit` read; a `timestamp` before 1970 (and 1900) reads
//! as itself — through a federated query and through a scan that is not.

use std::sync::Arc;

use arrow::array::{BooleanArray, Int32Array, RecordBatch};
use arrow::datatypes::DataType;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::logical_expr::{create_udf, ColumnarValue, Volatility};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::{JammiSession, QueryFunction};
use jammi_db::source::SourceDefinition;
use jammi_test_utils::{make_test_session, unique_suffix};
use sqlx::{Connection, Executor, PgConnection};
use tempfile::{tempdir, TempDir};

/// A database holding `plain(id)` and `kinds`, one column of every type the
/// driver does not read as stored, and the session reading it.
async fn session() -> (JammiSession, String, [TempDir; 2]) {
    let source_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_url(source_dir.path()).await;
    let mut db = PgConnection::connect(&url).await.unwrap();
    db.execute(
        "CREATE EXTENSION IF NOT EXISTS citext; \
         CREATE DOMAIN year AS integer; \
         CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy'); \
         CREATE TABLE plain (id INTEGER); INSERT INTO plain VALUES (1); \
         CREATE TABLE kinds ( \
             id INTEGER PRIMARY KEY, meta JSONB, doc JSON, email CITEXT, at TIMETZ, \
             words TSVECTOR, addr INET, x XML, flags BIT(4), released year, feeling mood, \
             born TIMESTAMP, born_tz TIMESTAMPTZ, day DATE, tags TEXT[], nums INTEGER[], \
             price MONEY); \
         INSERT INTO kinds VALUES \
             (1, '{\"a\": 1}', '{\"b\":2}', 'Ann@X.io', '10:30:00+02', 'a b', '10.0.0.1', \
              '<p>x</p>', B'1010', 1999, 'ok', '1958-12-08 10:30:00', \
              '1958-12-08 10:30:00+00', '1958-12-08', '{x,y}', '{1,2}', 12.34), \
             (2, NULL, NULL, 'bob@y.io', NULL, NULL, NULL, NULL, NULL, 2003, 'happy', \
              '1850-01-01 00:00:00', NULL, NULL, NULL, NULL, -7.25), \
             (3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
              '1973-07-01 00:00:00', NULL, NULL, NULL, NULL, NULL);",
    )
    .await
    .unwrap();
    db.close().await.unwrap();

    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    session.install_functions([QueryFunction::Scalar(create_udf(
        "is_odd",
        vec![DataType::Int32],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| {
            let ids = ColumnarValue::values_to_arrays(args)?;
            let ids = ids[0].as_any().downcast_ref::<Int32Array>().unwrap();
            let odd: BooleanArray = ids.iter().map(|id| id.map(|id| id % 2 != 0)).collect();
            Ok(ColumnarValue::Array(Arc::new(odd)))
        }),
    ))]);
    let source = format!("pg_{}", unique_suffix());
    let definition = SourceDefinition::from_url(url.as_str(), None).unwrap();
    session
        .add_source(&source, definition.source_type, definition.connection)
        .await
        .expect("a database with every type registers");
    (session, source, [source_dir, dir])
}

/// Each column of `batches`, its values as Arrow displays them.
fn displayed(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let options = FormatOptions::default().with_null("NULL");
    let width = batches.first().map_or(0, |b| b.num_columns());
    (0..width)
        .map(|column| {
            batches
                .iter()
                .flat_map(|batch| {
                    let formatter =
                        ArrayFormatter::try_new(batch.column(column).as_ref(), &options).unwrap();
                    (0..batch.num_rows())
                        .map(|row| formatter.value(row).to_string())
                        .collect::<Vec<_>>()
                })
                .collect()
        })
        .collect()
}

#[tokio::test]
async fn a_column_of_any_type_fails_no_table_of_its_source() {
    let (session, source, _dirs) = session().await;
    let batches = session
        .sql(&format!("SELECT COUNT(*) AS n FROM {source}.public.plain"))
        .await
        .unwrap();
    assert_eq!(displayed(&batches), [["1"]]);
}

/// `SELECT *` reads every column as the value it holds — federated whole,
/// and through a scan that is not federated (an installed function keeps
/// the query in the engine).
#[tokio::test]
async fn every_column_reads_as_the_value_it_holds() {
    let (session, source, _dirs) = session().await;
    for filter in ["", "WHERE is_odd(id) OR NOT is_odd(id)"] {
        let batches = session
            .sql(&format!(
                "SELECT meta, doc, email, at, words, addr, x, flags, released, feeling, born, \
                 born_tz, day, tags, nums, price FROM {source}.public.kinds {filter} ORDER BY id"
            ))
            .await
            .unwrap_or_else(|e| panic!("{filter}: {e:?}"));
        let columns = displayed(&batches);
        let first: Vec<&str> = columns.iter().map(|c| c[0].as_str()).collect();
        assert_eq!(
            first,
            [
                r#"{"a": 1}"#,
                r#"{"b":2}"#,
                "Ann@X.io",
                "10:30:00+02",
                "'a' 'b'",
                "10.0.0.1/32",
                "<p>x</p>",
                "1010",
                "1999",
                "ok",
                "1958-12-08T10:30:00",
                "1958-12-08T10:30:00Z",
                "1958-12-08",
                "[x, y]",
                "[1, 2]",
                "12.34",
            ],
            "{filter}"
        );
        assert_eq!(
            columns[10],
            [
                "1958-12-08T10:30:00",
                "1850-01-01T00:00:00",
                "1973-07-01T00:00:00"
            ],
            "{filter}"
        );
        assert_eq!(columns[15], ["12.34", "-7.25", "NULL"], "{filter}");
    }
}

/// Postgres compares the values as read: a `timestamp` at UTC, a domain as
/// its base, `citext` as text.
#[tokio::test]
async fn a_filter_on_a_column_read_through_postgres_compares_the_value_read() {
    let (session, source, _dirs) = session().await;
    for (filter, ids) in [
        ("born < TIMESTAMP '1970-01-01 00:00:00'", vec!["1", "2"]),
        ("born >= TIMESTAMP '1973-07-01 00:00:00'", vec!["3"]),
        ("released > 2000", vec!["2"]),
        ("email = 'Ann@X.io'", vec!["1"]),
        ("feeling = 'happy'", vec!["2"]),
        ("price > 1", vec!["1"]),
    ] {
        let batches = session
            .sql(&format!(
                "SELECT id FROM {source}.public.kinds WHERE {filter} ORDER BY id"
            ))
            .await
            .unwrap_or_else(|e| panic!("{filter}: {e:?}"));
        assert_eq!(displayed(&batches), [ids], "{filter}");
    }
    let batches = session
        .sql(&format!(
            "SELECT date_trunc('month', born) AS m FROM {source}.public.kinds ORDER BY id"
        ))
        .await
        .unwrap();
    assert_eq!(
        displayed(&batches),
        [[
            "1958-12-01T00:00:00",
            "1850-01-01T00:00:00",
            "1973-07-01T00:00:00"
        ]]
    );
}
