//! S7 — compound retrieval + inference in one Flight SQL round-trip.
//!
//! The `annotate` table function exposes model inference as a SQL relation, so
//! a remote caller runs `scan → inference` (and any `join`/`filter`/`select`
//! around it) in one SQL statement over the Flight SQL lane — no per-row RPC,
//! no bespoke `CompoundSearch` verb. This pins that wire contract: a Flight SQL
//! `SELECT … FROM annotate('local:<tiny_bert>', 'text_embedding',
//! 'patents.public.patents', 'id', 'abstract')` returns the inference output
//! rows (`_row_id` keyed back to the source `id`, plus the embedding `vector`),
//! and a SQL `WHERE`/`JOIN` composes over them.
//!
//! Hermetic: the encoder is the shipped local `tiny_bert` fixture (no network),
//! the corpus is the shipped patents parquet, and the assertions read the
//! decoded Arrow rows directly.

use std::net::SocketAddr;

use arrow::array::{Array, StringArray};
use arrow::datatypes::DataType;
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::grpc::session::SESSION_HEADER;
use jammi_test_utils::{cookbook_fixture, fixture};

use super::common::grpc::{channel, start_engine_server, EngineServer};

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

/// Register the shipped patents parquet on the running engine in-process (the
/// fixture exposes the same `Arc<InferenceSession>` the server drives), so the
/// Flight SQL query below scans a real source.
async fn add_patents(server: &EngineServer) {
    server
        .engine
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", fixture("patents.parquet").display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .expect("add patents source");
}

/// Run one Flight SQL statement and collect the result batches.
async fn flight_query(addr: SocketAddr, sql: &str) -> Vec<arrow::record_batch::RecordBatch> {
    let mut client = FlightSqlServiceClient::new(channel(addr).await);
    // No tenant bound; the patents source carries no tenant column, so its rows
    // are globally visible.
    client.set_header(SESSION_HEADER, "session-annotate");
    let info = client
        .execute(sql.to_string(), None)
        .await
        .expect("execute flight sql");
    let ticket = info
        .endpoint
        .first()
        .cloned()
        .expect("flight info endpoint")
        .ticket
        .expect("endpoint ticket");
    let stream = client.do_get(ticket).await.expect("do_get");
    stream.try_collect().await.expect("collect flight stream")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flight_annotate_runs_inference_as_a_sql_relation() {
    let server = start_engine_server().await;
    add_patents(&server).await;

    let sql = format!(
        "SELECT _row_id, _status, vector \
         FROM annotate('{model}', 'text_embedding', \
                       'patents.public.patents', 'id', 'abstract')",
        model = tiny_bert_model_id()
    );
    let batches = flight_query(server.addr, &sql).await;

    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        total > 0,
        "annotate must return one inference row per source row"
    );

    let batch = &batches[0];
    // The inference prefix carries `_row_id` (keyed from the source `id`) and a
    // `_status`; the embedding task appends a fixed-size `vector` column.
    let row_id = batch
        .column_by_name("_row_id")
        .expect("_row_id column")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_row_id is Utf8");
    assert!(
        !row_id.value(0).is_empty(),
        "_row_id keys back to the source row"
    );

    let status = batch
        .column_by_name("_status")
        .expect("_status column")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_status is Utf8");
    for i in 0..status.len() {
        assert_eq!(status.value(i), "ok", "every row's inference must succeed");
    }

    let vector_field = batch
        .schema()
        .field_with_name("vector")
        .expect("vector column")
        .clone();
    match vector_field.data_type() {
        DataType::FixedSizeList(_, dim) => {
            assert_eq!(*dim, 32, "tiny_bert embeds in 32 dimensions")
        }
        other => panic!("vector column should be a FixedSizeList, got {other:?}"),
    }

    let _ = server.shutdown.send(());
}

/// A `WHERE` over the annotated output composes — the compound query is open
/// SQL, not a closed verb. The predicate runs above the inference node (the
/// table function declares inference pushdown unsupported), so it filters the
/// model output rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flight_annotate_composes_with_a_filter() {
    let server = start_engine_server().await;
    add_patents(&server).await;

    let sql = format!(
        "SELECT _row_id FROM annotate('{model}', 'text_embedding', \
                       'patents.public.patents', 'id', 'abstract') \
         WHERE _status = 'ok'",
        model = tiny_bert_model_id()
    );
    let batches = flight_query(server.addr, &sql).await;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        total > 0,
        "the filter keeps the successfully-annotated rows"
    );

    let _ = server.shutdown.send(());
}

/// `annotate` over a relation joined back to its source: the caller projects an
/// inference column alongside an original source column, exactly the
/// "search → join → annotate" shape the lane exists for, expressed as plain SQL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flight_annotate_joins_back_to_source_columns() {
    let server = start_engine_server().await;
    add_patents(&server).await;

    let sql = format!(
        "SELECT p.title, a._status \
         FROM annotate('{model}', 'text_embedding', \
                       'patents.public.patents', 'id', 'abstract') AS a \
         JOIN patents.public.patents AS p \
           ON a._row_id = arrow_cast(p.id, 'Utf8')",
        model = tiny_bert_model_id()
    );
    let batches = flight_query(server.addr, &sql).await;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total > 0, "join back to the source must yield rows");

    let batch = &batches[0];
    // The joined source `title` rides alongside the annotated `_status` — one
    // round-trip, both retrieval and inference. The Parquet reader may produce
    // `title` as Utf8 or Utf8View, so assert via the type-general Arrow
    // formatter rather than a fixed downcast.
    let titles = batch
        .column_by_name("title")
        .expect("title column from the joined source");
    let format = arrow::util::display::FormatOptions::default();
    let fmt =
        arrow::util::display::ArrayFormatter::try_new(titles.as_ref(), &format).expect("formatter");
    assert!(
        (0..titles.len()).any(|i| !titles.is_null(i) && !fmt.value(i).to_string().is_empty()),
        "at least one joined row carries a source title"
    );
    assert!(batch.column_by_name("_status").is_some());

    let _ = server.shutdown.send(());
}
