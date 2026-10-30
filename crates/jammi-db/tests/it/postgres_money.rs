//! A Postgres `money` column reads as the amount it holds — `12.34`, not the
//! `1234` of minor units Postgres sends in binary — whether the engine reads
//! its rows or Postgres answers the query: a filter on the column and an
//! aggregate over it are sent to Postgres and answered there.

use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::session::JammiSession;
use jammi_db::source::SourceDefinition;
use jammi_test_utils::{make_test_session, unique_suffix};
use tempfile::{tempdir, TempDir};

/// A session with a Postgres source `pg_<suffix>` holding
/// `prices(id INTEGER, amount MONEY)` — a fraction, a value under one, a
/// negative and a NULL — and `labels(id INTEGER, label TEXT)` naming them.
async fn session_with_prices() -> (JammiSession, String, [TempDir; 2]) {
    let source_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_holding(
        source_dir.path(),
        "CREATE TABLE prices (id INTEGER PRIMARY KEY, amount MONEY); \
         INSERT INTO prices VALUES (1, 12.34), (2, 0.50), (3, -7.25), (4, NULL); \
         CREATE TABLE labels (id INTEGER PRIMARY KEY, label TEXT NOT NULL); \
         INSERT INTO labels VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd');",
    )
    .await;

    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    let source_id = format!("pg_{}", unique_suffix());
    let definition = SourceDefinition::from_url(&jammi_test_utils::source_url(&url), None).unwrap();
    session
        .add_source(&source_id, definition.source_type, definition.connection)
        .await
        .unwrap();
    (session, source_id, [source_dir, dir])
}

/// The first column of `batches`, each value as Arrow displays it.
fn displayed(batches: &[RecordBatch]) -> Vec<String> {
    let options = FormatOptions::default().with_null("NULL");
    batches
        .iter()
        .flat_map(|batch| {
            let formatter = ArrayFormatter::try_new(batch.column(0).as_ref(), &options).unwrap();
            (0..batch.num_rows())
                .map(|row| formatter.value(row).to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn a_money_column_reads_as_its_amount() {
    let (session, source, _dirs) = session_with_prices().await;
    let batches = session
        .sql(&format!(
            "SELECT amount FROM {source}.public.prices ORDER BY id"
        ))
        .await
        .unwrap();
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &DataType::Decimal128(19, 2)
    );
    assert_eq!(displayed(&batches), ["12.34", "0.50", "-7.25", "NULL"]);
}

#[tokio::test]
async fn an_aggregate_over_a_money_column_is_its_amount() {
    let (session, source, _dirs) = session_with_prices().await;
    let sum = session
        .sql(&format!("SELECT SUM(amount) FROM {source}.public.prices"))
        .await
        .unwrap();
    assert_eq!(displayed(&sum), ["5.59"]);
    let max = session
        .sql(&format!("SELECT MAX(amount) FROM {source}.public.prices"))
        .await
        .unwrap();
    assert_eq!(displayed(&max), ["12.34"]);
}

#[tokio::test]
async fn a_filter_on_a_money_column_compares_amounts() {
    let (session, source, _dirs) = session_with_prices().await;
    let ids = |predicate: &'static str| {
        let (session, source) = (&session, &source);
        async move {
            let batches = session
                .sql(&format!(
                    "SELECT id FROM {source}.public.prices WHERE {predicate} ORDER BY id"
                ))
                .await
                .unwrap();
            displayed(&batches)
        }
    };
    assert_eq!(ids("amount > 1").await, ["1"]);
    assert_eq!(ids("amount < 0").await, ["3"]);
    assert_eq!(ids("amount = 0.5").await, ["2"]);
    assert_eq!(ids("amount BETWEEN 0 AND 12.34").await, ["1", "2"]);
    assert_eq!(ids("amount IS NULL").await, ["4"]);
}

#[tokio::test]
async fn a_money_column_joined_to_another_table_of_its_database_reads_as_its_amount() {
    let (session, source, _dirs) = session_with_prices().await;
    let batches = session
        .sql(&format!(
            "SELECT p.amount, l.label FROM {source}.public.prices p \
             JOIN {source}.public.labels l ON p.id = l.id \
             WHERE p.amount > 0 ORDER BY p.amount DESC"
        ))
        .await
        .unwrap();
    assert_eq!(displayed(&batches), ["12.34", "0.50"]);
}
