//! A list column breaks down by item in every form SQL writes it: the
//! select-list `unnest` and the lateral `CROSS JOIN unnest(...)`, over a file
//! and over a database, each answering the same counts.

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_test_utils::make_test_session;
use parquet::arrow::ArrowWriter;
use tempfile::{tempdir, TempDir};

/// `(id, items)`: delimited items, spaced and not, one repeated across rows.
const ROWS: [(i32, &str); 3] = [(1, "a, b"), (2, "b"), (3, "a,c")];

/// The items of `t.items`, split on commas with their spacing dropped.
const ITEMS: &str = "string_to_array(regexp_replace(t.items, '\\s*,\\s*', ',', 'g'), ',')";

/// The counts every form answers.
const COUNTS: &str = "+------+---+
| item | n |
+------+---+
| a    | 2 |
| b    | 2 |
| c    | 1 |
+------+---+";

fn forms(relation: &str) -> [String; 2] {
    [
        format!(
            "SELECT unnest({ITEMS}) AS item, COUNT(*) AS n FROM {relation} t \
             GROUP BY unnest({ITEMS}) ORDER BY 1"
        ),
        format!(
            "SELECT u.item, COUNT(*) AS n FROM {relation} t \
             CROSS JOIN unnest({ITEMS}) AS u(item) GROUP BY u.item ORDER BY 1"
        ),
    ]
}

async fn answers(session: &JammiSession, sql: &str) -> String {
    let batches = session
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    pretty_format_batches(&batches).unwrap().to_string()
}

async fn file_session() -> (JammiSession, [TempDir; 2]) {
    let files = tempdir().unwrap();
    let path = files.path().join("items.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("items", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values(ROWS.iter().map(|r| r.0))),
            Arc::new(StringArray::from_iter_values(ROWS.iter().map(|r| r.1))),
        ],
    )
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    session
        .add_source(
            "f",
            SourceType::File,
            SourceConnection {
                url: Some(path.to_str().unwrap().to_string()),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    (session, [files, dir])
}

#[tokio::test]
async fn a_file_column_breaks_down_by_item_in_every_form() {
    let (session, _dirs) = file_session().await;
    for sql in forms("f.public.items") {
        assert_eq!(answers(&session, &sql).await, COUNTS, "{sql}");
    }
}

/// A database source registered on `url` as `db`, in a fresh session.
#[cfg(any(feature = "live-postgres-tests", feature = "live-mysql-tests"))]
async fn database_session(url: &str) -> (JammiSession, TempDir) {
    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    let definition = jammi_db::source::SourceDefinition::from_url(url, None).unwrap();
    session
        .add_source("db", definition.source_type, definition.connection)
        .await
        .unwrap();
    (session, dir)
}

/// `INSERT` values for [`ROWS`].
#[cfg(any(feature = "live-postgres-tests", feature = "live-mysql-tests"))]
fn values() -> String {
    ROWS.iter()
        .map(|(id, items)| format!("({id}, '{items}')"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn a_postgres_column_breaks_down_by_item_in_every_form() {
    let source_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_holding(
        source_dir.path(),
        &format!(
            "CREATE TABLE items (id INTEGER, items TEXT); INSERT INTO items VALUES {};",
            values()
        ),
    )
    .await;
    let (session, _dir) = database_session(&url).await;
    for sql in forms("db.public.items") {
        assert_eq!(answers(&session, &sql).await, COUNTS, "{sql}");
    }
}

#[cfg(feature = "live-mysql-tests")]
#[tokio::test]
async fn a_mysql_column_breaks_down_by_item_in_every_form() {
    let url = crate::common::mysql::mysql_database_holding(&format!(
        "CREATE TABLE items (id INT, items TEXT); INSERT INTO items VALUES {};",
        values()
    ))
    .await;
    let (session, _dir) = database_session(&url).await;
    for sql in forms("db.public.items") {
        assert_eq!(answers(&session, &sql).await, COUNTS, "{sql}");
    }
}
