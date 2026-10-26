//! A filter on a federated source's column holds when the query also joins a
//! source the engine scans itself: every query here runs once over a
//! Postgres table joined with a file, and once over the same rows held in a
//! file, and both answer the same.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceDefinition, SourceType};
use jammi_test_utils::{make_test_session, unique_suffix};
use parquet::arrow::ArrowWriter;
use sqlx::{Connection, Executor, PgConnection};
use tempfile::{tempdir, TempDir};

const CUSTOMERS: [(i32, &str); 5] = [(1, "SP"), (2, "SP"), (3, "SP"), (4, "RJ"), (5, "RJ")];
const REGIONS: [(&str, &str); 2] = [("SP", "Southeast"), ("RJ", "Rio")];

fn write_parquet(path: &Path, columns: Vec<(&str, ArrayRef)>) {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), false))
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn file_source(path: &Path) -> SourceConnection {
    SourceConnection {
        url: Some(path.to_str().unwrap().to_string()),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }
}

/// The sources every query reads: `customers` in Postgres (`pg_*`) and in a
/// file (`cf_*`), and `regions` in a file (`rf_*`).
struct Sources {
    session: JammiSession,
    pg: String,
    customers_file: String,
    regions: String,
    _dirs: [TempDir; 3],
}

async fn sources() -> Sources {
    let pg_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_url(pg_dir.path()).await;
    let mut db = PgConnection::connect(&url).await.unwrap();
    let values = CUSTOMERS
        .iter()
        .map(|(id, state)| format!("({id}, '{state}')"))
        .collect::<Vec<_>>()
        .join(", ");
    db.execute(
        format!(
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, state TEXT NOT NULL); \
             INSERT INTO customers VALUES {values};"
        )
        .as_str(),
    )
    .await
    .unwrap();
    db.close().await.unwrap();

    let files = tempdir().unwrap();
    let customers_path = files.path().join("customers.parquet");
    write_parquet(
        &customers_path,
        vec![
            (
                "customer_id",
                Arc::new(Int32Array::from_iter_values(CUSTOMERS.iter().map(|c| c.0))) as ArrayRef,
            ),
            (
                "state",
                Arc::new(StringArray::from_iter_values(CUSTOMERS.iter().map(|c| c.1))),
            ),
        ],
    );
    let regions_path = files.path().join("regions.parquet");
    write_parquet(
        &regions_path,
        vec![
            (
                "state",
                Arc::new(StringArray::from_iter_values(REGIONS.iter().map(|r| r.0))) as ArrayRef,
            ),
            (
                "region",
                Arc::new(StringArray::from_iter_values(REGIONS.iter().map(|r| r.1))),
            ),
        ],
    );

    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    let suffix = unique_suffix();
    let (pg, customers_file, regions) = (
        format!("pg_{suffix}"),
        format!("cf_{suffix}"),
        format!("rf_{suffix}"),
    );
    let definition = SourceDefinition::from_url(url.as_str(), None).unwrap();
    session
        .add_source(&pg, definition.source_type, definition.connection)
        .await
        .unwrap();
    session
        .add_source(
            &customers_file,
            SourceType::File,
            file_source(&customers_path),
        )
        .await
        .unwrap();
    session
        .add_source(&regions, SourceType::File, file_source(&regions_path))
        .await
        .unwrap();
    Sources {
        session,
        pg,
        customers_file,
        regions,
        _dirs: [pg_dir, files, dir],
    }
}

impl Sources {
    /// `sql` with `{customers}` and `{regions}` naming the tables, run over
    /// the Postgres customers and over the file's, each rendered as a table.
    async fn both(&self, sql: &str) -> (String, String) {
        let run = |customers: String| {
            let sql = sql
                .replace("{customers}", &customers)
                .replace("{regions}", &format!("{}.public.regions", self.regions));
            async move {
                let batches = self.session.sql(&sql).await.unwrap();
                pretty_format_batches(&batches).unwrap().to_string()
            }
        };
        (
            run(format!("{}.public.customers", self.pg)).await,
            run(format!("{}.public.customers", self.customers_file)).await,
        )
    }
}

#[tokio::test]
async fn a_filter_on_a_postgres_column_holds_across_a_join_with_a_file() {
    let sources = sources().await;
    let (federated, _) = sources
        .both(
            "SELECT COUNT(*) AS n FROM {customers} c JOIN {regions} r ON c.state = r.state \
             WHERE c.state = 'SP'",
        )
        .await;
    assert!(federated.contains("| 3 |"), "{federated}");
}

#[tokio::test]
async fn a_join_of_postgres_and_a_file_answers_as_the_engine_alone_does() {
    let sources = sources().await;
    let join = "FROM {customers} c JOIN {regions} r ON c.state = r.state";
    let left = "FROM {customers} c LEFT JOIN {regions} r ON c.state = r.state";
    for sql in [
        format!("SELECT COUNT(*) AS n {join} WHERE c.state = 'SP'"),
        format!("SELECT COUNT(*) AS n {join} WHERE r.state = 'SP'"),
        format!("SELECT COUNT(*) AS n {join} WHERE c.state = 'SP' AND r.region = 'Southeast'"),
        format!("SELECT COUNT(*) AS n {join} WHERE c.customer_id > 2"),
        format!("SELECT c.customer_id, r.region {join} WHERE c.state = 'RJ' OR c.customer_id = 1 ORDER BY 1"),
        format!("SELECT c.customer_id, r.region {join} WHERE c.state <> r.region ORDER BY 1"),
        format!("SELECT c.customer_id, r.region {left} WHERE c.customer_id IN (2, 4) ORDER BY 1"),
        format!("SELECT r.region, COUNT(*) AS n {join} WHERE c.customer_id < 5 GROUP BY 1 ORDER BY 1"),
    ] {
        let (federated, engine) = sources.both(&sql).await;
        assert_eq!(federated, engine, "{sql}");
    }
}

/// The plan behind the answer: the filter on the Postgres column runs in a
/// `FilterExec` above the federated scan, not handed to it.
#[tokio::test]
async fn a_filter_on_a_postgres_column_stays_above_the_federated_scan() {
    use arrow::array::{Array, StringArray};

    let sources = sources().await;
    let batches = sources
        .session
        .sql(&format!(
            "EXPLAIN SELECT COUNT(*) AS n FROM {}.public.customers c \
             JOIN {}.public.regions r ON c.state = r.state WHERE c.state = 'SP'",
            sources.pg, sources.regions
        ))
        .await
        .unwrap();
    let physical = batches
        .iter()
        .find_map(|batch| {
            let kinds = batch.column(0).as_any().downcast_ref::<StringArray>()?;
            let plans = batch.column(1).as_any().downcast_ref::<StringArray>()?;
            (0..kinds.len())
                .find(|&i| kinds.value(i) == "physical_plan")
                .map(|i| plans.value(i).to_string())
        })
        .expect("EXPLAIN shows the physical plan");
    let filter = physical.find("FilterExec: state@");
    let scan = physical.find("RemoteExec");
    assert!(
        matches!((filter, scan), (Some(filter), Some(scan)) if filter < scan),
        "{physical}"
    );
}
