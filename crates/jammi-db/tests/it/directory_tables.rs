//! A directory registers as one table, its files the parts of it (#755). A
//! directory whose files do not agree on that table's columns is refused,
//! naming two of them, rather than read as the union of every file's columns
//! with NULLs wherever a file lacks one. A Parquet part may lack some of the
//! table's columns (a partition that added one), its rows reading them as
//! NULL; CSV parts, whose columns are read by position, must share one header;
//! JSON parts are the union of their records' keys, as one file of sparse
//! records is.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::util::pretty::pretty_format_batches;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::error::{JammiError, Result};
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_test_utils::{make_test_session, unique_suffix};
use parquet::arrow::ArrowWriter;
use tempfile::tempdir;
use test_case::test_case;

/// Write `columns` (name, values, nullable) to `path` as one Parquet file.
fn write_parquet(path: &Path, columns: Vec<(&str, ArrayRef, bool)>) {
    let batch = RecordBatch::try_from_iter_with_nullable(columns).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn int64s(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn float64s(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}

fn strings(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

/// A fresh directory `name` under `root`.
fn folder(root: &Path, name: &str) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `orders.parquet` and `customers.parquet`, two tables with no column in
/// common: the issue's folder, its columns nullable as pyarrow writes them.
fn unrelated_exports(root: &Path) -> PathBuf {
    let dir = folder(root, "mixed");
    write_parquet(
        &dir.join("orders.parquet"),
        vec![
            ("order_id", int64s(&[1, 2, 3]), true),
            ("amount", float64s(&[10.0, 20.0, 30.0]), true),
        ],
    );
    write_parquet(
        &dir.join("customers.parquet"),
        vec![
            ("customer_id", int64s(&[7, 8]), true),
            ("city", strings(&["Lisbon", "Porto"]), true),
        ],
    );
    dir
}

/// Register `dir` as file source `id` of `format`.
async fn add_directory(
    session: &JammiSession,
    id: &str,
    dir: &Path,
    format: FileFormat,
) -> Result<()> {
    session
        .add_source(
            id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", dir.display())),
                format: Some(format),
                ..Default::default()
            },
        )
        .await?;
    Ok(())
}

/// `err` refuses the source's URL, naming each of `files`.
fn assert_refused_naming(err: &JammiError, files: &[&str]) {
    let JammiError::InvalidArgument { argument, reason } = err else {
        panic!("a directory of two tables must be refused as an invalid url, got: {err}");
    };
    assert_eq!(argument, "url", "{reason}");
    for file in files {
        assert!(
            reason.contains(file),
            "the refusal must name {file}: {reason}"
        );
    }
}

/// `sql`'s rows, as a table.
async fn rows(session: &JammiSession, sql: &str) -> String {
    pretty_format_batches(&session.sql(sql).await.unwrap())
        .unwrap()
        .to_string()
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_directory_of_unrelated_parquet_files_is_refused_naming_two_of_them(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let id = format!("mixed_{}", unique_suffix());

    let err = add_directory(
        &session,
        &id,
        &unrelated_exports(dir.path()),
        FileFormat::Parquet,
    )
    .await
    .unwrap_err();

    assert_refused_naming(&err, &["orders.parquet", "customers.parquet"]);
    let sources = session.catalog().list_sources().await.unwrap();
    assert!(!sources.iter().any(|s| s.source_id == id));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_parquet_column_of_another_type_is_refused_naming_both_files(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let parts = folder(dir.path(), "parts");
    write_parquet(
        &parts.join("part-0.parquet"),
        vec![("id", Arc::new(Int32Array::from(vec![1])), false)],
    );
    write_parquet(
        &parts.join("part-1.parquet"),
        vec![("id", int64s(&[2]), false)],
    );

    let err = add_directory(
        &session,
        &format!("parts_{}", unique_suffix()),
        &parts,
        FileFormat::Parquet,
    )
    .await
    .unwrap_err();

    assert_refused_naming(
        &err,
        &["part-0.parquet", "part-1.parquet", "Int32", "Int64"],
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_csv_directory_whose_headers_differ_is_refused_naming_two_files(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    // CSV columns are read by position: the same columns in another order
    // would read "Hopper" as a first name.
    let reordered = folder(dir.path(), "people");
    std::fs::write(reordered.join("a.csv"), "first,last\nAda,Lovelace\n").unwrap();
    std::fs::write(reordered.join("b.csv"), "last,first\nHopper,Grace\n").unwrap();
    // Two tables, which today fail only once a query reads them.
    let tables = folder(dir.path(), "exports");
    std::fs::write(tables.join("orders.csv"), "order_id,amount\n1,10.0\n").unwrap();
    std::fs::write(tables.join("customers.csv"), "customer_id,city\n7,Lisbon\n").unwrap();

    for (dir, files) in [
        (&reordered, ["a.csv", "b.csv"]),
        (&tables, ["orders.csv", "customers.csv"]),
    ] {
        let id = format!("csv_{}", unique_suffix());
        let err = add_directory(&session, &id, dir, FileFormat::Csv)
            .await
            .unwrap_err();
        assert_refused_naming(&err, &files);
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn parts_of_one_table_still_read_as_one_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

    // A later partition added a column: the earlier rows read it as NULL.
    let grown = folder(dir.path(), "grown");
    write_parquet(
        &grown.join("part-0.parquet"),
        vec![
            ("id", int64s(&[1]), true),
            ("amount", float64s(&[1.0]), true),
        ],
    );
    write_parquet(
        &grown.join("part-1.parquet"),
        vec![
            ("id", int64s(&[2]), true),
            ("amount", float64s(&[2.0]), true),
            ("tax", float64s(&[0.2]), true),
        ],
    );
    // Parquet columns are read by name: their order and nullability may differ.
    let shuffled = folder(dir.path(), "shuffled");
    write_parquet(
        &shuffled.join("part-0.parquet"),
        vec![
            ("id", int64s(&[1]), false),
            ("name", strings(&["x"]), false),
        ],
    );
    write_parquet(
        &shuffled.join("part-1.parquet"),
        vec![("name", strings(&["y"]), true), ("id", int64s(&[2]), true)],
    );
    // CSV parts sharing one header.
    let csv = folder(dir.path(), "csv_parts");
    std::fs::write(csv.join("a.csv"), "first,last\nAda,Lovelace\n").unwrap();
    std::fs::write(csv.join("b.csv"), "first,last\nGrace,Hopper\n").unwrap();
    // JSON records carry their own keys: the directory is their union.
    let json = folder(dir.path(), "json_parts");
    std::fs::write(json.join("a.jsonl"), "{\"a\": 1}\n").unwrap();
    std::fs::write(json.join("b.jsonl"), "{\"b\": \"two\"}\n").unwrap();

    let suffix = unique_suffix();
    for (name, dir, format) in [
        ("grown", &grown, FileFormat::Parquet),
        ("shuffled", &shuffled, FileFormat::Parquet),
        ("csv_parts", &csv, FileFormat::Csv),
        ("json_parts", &json, FileFormat::JsonLines),
    ] {
        add_directory(&session, &format!("{name}_{suffix}"), dir, format)
            .await
            .unwrap();
    }

    let grown = rows(
        &session,
        &format!("SELECT id, amount, tax FROM grown_{suffix}.public.grown ORDER BY id"),
    )
    .await;
    assert_eq!(
        grown,
        "+----+--------+-----+
| id | amount | tax |
+----+--------+-----+
| 1  | 1.0    |     |
| 2  | 2.0    | 0.2 |
+----+--------+-----+"
    );
    let shuffled = rows(
        &session,
        &format!("SELECT id, name FROM shuffled_{suffix}.public.shuffled ORDER BY id"),
    )
    .await;
    assert_eq!(
        shuffled,
        "+----+------+
| id | name |
+----+------+
| 1  | x    |
| 2  | y    |
+----+------+"
    );
    let csv = rows(
        &session,
        &format!("SELECT first, last FROM csv_parts_{suffix}.public.csv_parts ORDER BY first"),
    )
    .await;
    assert_eq!(
        csv,
        "+-------+----------+
| first | last     |
+-------+----------+
| Ada   | Lovelace |
| Grace | Hopper   |
+-------+----------+"
    );
    let json = rows(
        &session,
        &format!("SELECT a, b FROM json_parts_{suffix}.public.json_parts ORDER BY a"),
    )
    .await;
    assert_eq!(
        json,
        "+---+-----+
| a | b   |
+---+-----+
| 1 |     |
|   | two |
+---+-----+"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_part_lacking_a_non_nullable_column_reads_it_as_null(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    // The partition that added `tax` declares it non-nullable; the earlier
    // partition, which lacks it, can only read it as NULL.
    let grown = folder(dir.path(), "grown");
    write_parquet(
        &grown.join("part-0.parquet"),
        vec![("id", int64s(&[1]), false)],
    );
    write_parquet(
        &grown.join("part-1.parquet"),
        vec![
            ("id", int64s(&[2]), false),
            ("tax", float64s(&[0.2]), false),
        ],
    );
    let id = format!("grown_{}", unique_suffix());
    add_directory(&session, &id, &grown, FileFormat::Parquet)
        .await
        .unwrap();

    let batches = session
        .sql(&format!(
            "SELECT id, tax FROM {id}.public.grown ORDER BY id"
        ))
        .await
        .unwrap();
    assert_eq!(
        pretty_format_batches(&batches).unwrap().to_string(),
        "+----+-----+
| id | tax |
+----+-----+
| 1  |     |
| 2  | 0.2 |
+----+-----+"
    );
    let schema = batches[0].schema();
    assert!(schema.field_with_name("tax").unwrap().is_nullable());
    assert!(!schema.field_with_name("id").unwrap().is_nullable());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_file_added_after_registration_is_refused_when_the_source_is_next_built(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let id = format!("growing_{}", unique_suffix());
    let exports = unrelated_exports(dir.path());
    let customers = std::fs::read(exports.join("customers.parquet")).unwrap();
    std::fs::remove_file(exports.join("customers.parquet")).unwrap();

    {
        let session = make_test_session(backend, dir.path()).await;
        add_directory(&session, &id, &exports, FileFormat::Parquet)
            .await
            .unwrap();
    }
    std::fs::write(exports.join("customers.parquet"), customers).unwrap();

    // The next process to build the source (here, a session over the same
    // catalog) lists the directory afresh and refuses it.
    let session = make_test_session(backend, dir.path()).await;
    let err = session
        .sql(&format!("SELECT * FROM {id}.public.mixed"))
        .await
        .unwrap_err()
        .to_string();
    for file in ["orders.parquet", "customers.parquet"] {
        assert!(err.contains(file), "the refusal must name {file}: {err}");
    }
}
