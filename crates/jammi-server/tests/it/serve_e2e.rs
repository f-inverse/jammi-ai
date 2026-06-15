//! End-to-end test for `OssServer::run` — boot the full binary in
//! process with a temporary SQLite config, drive a gRPC call, an
//! `EmbeddingService/Search`, and a Flight SQL roundtrip against
//! `:flight_listen`, hit the HTTP side-channel on `:health_listen`, and
//! assert the substrate metrics moved (not just that they exist) before a
//! graceful shutdown.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use jammi_db::config::JammiConfig;
use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::{
    AddSourceRequest, FileFormat, SourceConnection, SourceKind,
};
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::search_request::Query as SearchQuery;
use jammi_server::grpc::proto::embedding::{GenerateEmbeddingsRequest, Modality, SearchRequest};
use jammi_server::grpc::proto::eval::eval_service_client::EvalServiceClient;
use jammi_server::grpc::proto::eval::EvalEmbeddingsRequest;
use jammi_server::runtime::OssServer;
use jammi_test_utils::{cookbook_fixture, fixture, test_config};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

/// Acquire two distinct loopback ports for health + flight by binding
/// `127.0.0.1:0` twice and immediately dropping the listeners. The
/// kernel issues unique ephemeral ports, so the bound addresses are
/// safe to hand to the OSS server seconds later — the race window is
/// the same one every other multi-listener integration test relies on.
async fn pick_two_ports() -> (String, String) {
    let l1 = TcpListener::bind("127.0.0.1:0").await.expect("bind 1");
    let l2 = TcpListener::bind("127.0.0.1:0").await.expect("bind 2");
    let a1 = l1.local_addr().expect("addr 1");
    let a2 = l2.local_addr().expect("addr 2");
    drop(l1);
    drop(l2);
    (a1.to_string(), a2.to_string())
}

fn test_oss_config(artifact_dir: &std::path::Path, health: &str, flight: &str) -> JammiConfig {
    let mut cfg = test_config(artifact_dir);
    cfg.server.health_listen = health.to_string();
    cfg.server.flight_listen = flight.to_string();
    cfg
}

fn htsat_clap_model_id() -> String {
    format!("local:{}", cookbook_fixture("htsat_clap_tiny").display())
}

/// Local text encoder for the eval leg — the same `tiny_bert` fixture the
/// `grpc_eval` suite embeds the patents corpus with.
fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

/// Canonical golden-set source name for the `golden_relevance.csv` fixture,
/// matching the form the eval engine resolves (`<source>.public.<table>`).
const GOLDEN_SOURCE: &str = "golden_rel.public.golden_relevance";

/// Build a minimal 16-bit PCM mono WAV (16 kHz) holding a sine tone at
/// `freq`. Mirrors the embedding-suite corpus generator so the Search path
/// exercises the same realistic decode → log-mel input.
fn sine_wav_bytes(freq: f32) -> Vec<u8> {
    let sample_rate: u32 = 16_000;
    let n = (sample_rate as f32 * 0.2) as usize;
    let samples: Vec<i16> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (0.5 * (2.0 * std::f32::consts::PI * freq * t).sin() * i16::MAX as f32) as i16
        })
        .collect();
    let data_len = (samples.len() * 2) as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

/// Write a three-row (clip_id, audio) corpus parquet and return its path.
fn write_corpus_parquet(dir: &TempDir) -> std::path::PathBuf {
    let clips: Vec<Vec<u8>> = [220.0_f32, 440.0, 880.0]
        .iter()
        .map(|&f| sine_wav_bytes(f))
        .collect();
    let parquet_path = dir.path().join("clips.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("clip_id", DataType::Utf8, false),
        Field::new("audio", DataType::Binary, false),
    ]));
    let ids = Arc::new(StringArray::from(vec!["clip_0", "clip_1", "clip_2"])) as ArrayRef;
    let audio: Vec<&[u8]> = clips.iter().map(|v| v.as_slice()).collect();
    let audio_array = Arc::new(BinaryArray::from(audio)) as ArrayRef;
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, audio_array]).unwrap();
    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    parquet_path
}

/// Scrape the `/metrics` text snapshot from the HTTP side-channel.
async fn scrape_metrics(health_addr: &std::net::SocketAddr) -> String {
    reqwest::get(format!("http://{health_addr}/metrics"))
        .await
        .expect("metrics GET")
        .text()
        .await
        .expect("metrics body")
}

/// Read the value of a Prometheus sample line (`<name> <value>`) from a
/// scraped exposition. A missing sample reads as `0.0` — a freshly-built
/// `IntCounter` exposes a `0` sample, so this only returns the default when
/// the metric is genuinely absent.
fn sample_value(metrics: &str, name: &str) -> f64 {
    metrics
        .lines()
        .find(|line| {
            let mut parts = line.split_whitespace();
            parts.next() == Some(name)
        })
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oss_server_serves_healthz_and_drives_live_metrics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (health_addr, flight_addr) = pick_two_ports().await;
    let cfg = test_oss_config(dir.path(), &health_addr, &flight_addr);

    let server = OssServer::new(cfg).await.expect("OssServer::new");
    let server_health_addr = server.health_addr();
    let server_flight_addr = server.flight_addr();

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server_task = tokio::spawn(async move {
        server
            .run_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });

    // Wait for both listeners to bind. The HTTP side-channel and the
    // Tonic chain bind in parallel; 200ms is comfortably above the
    // measured worst-case bind latency on CI hardware.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // /healthz round-trip.
    let healthz = reqwest::get(format!("http://{server_health_addr}/healthz"))
        .await
        .expect("healthz GET")
        .json::<serde_json::Value>()
        .await
        .expect("healthz json");
    assert_eq!(healthz["status"], "ok");

    // /readyz round-trip — the catalog probe runs against the live
    // SQLite catalog the test config provisioned.
    let readyz = reqwest::get(format!("http://{server_health_addr}/readyz"))
        .await
        .expect("readyz GET");
    assert_eq!(
        readyz.status(),
        200,
        "readyz must be 200 against live SQLite"
    );

    // Baseline scrape: capture each metric before any traffic so the
    // assertions check a real *delta*, not mere presence.
    let before = scrape_metrics(&server_health_addr).await;
    let grpc_before = sample_value(&before, "jammi_grpc_requests_total");
    let flight_before = sample_value(&before, "jammi_flight_queries_total");
    let search_count_before = sample_value(&before, "jammi_search_latency_seconds_count");
    let eval_before = sample_value(&before, "jammi_eval_invocations_total");

    // Drive a gRPC call + a Search over the embedding service. AddSource +
    // GenerateEmbeddings + Search all ride `/jammi.v1.*`, so they move
    // `grpc_requests`; the Search additionally records `search_latency`.
    let grpc_channel = Channel::from_shared(format!("http://{server_flight_addr}"))
        .expect("grpc channel uri")
        .connect()
        .await
        .expect("grpc connect");
    let mut catalog = CatalogServiceClient::new(grpc_channel.clone());
    let mut embedding = EmbeddingServiceClient::new(grpc_channel);

    let parquet_path = write_corpus_parquet(&dir);
    catalog
        .add_source(AddSourceRequest {
            source_id: "clips".into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: format!("file://{}", parquet_path.display()),
                format: FileFormat::Parquet as i32,
            }),
        })
        .await
        .expect("add_source");

    embedding
        .generate_embeddings(GenerateEmbeddingsRequest {
            source_id: "clips".into(),
            model_id: htsat_clap_model_id(),
            columns: vec!["audio".into()],
            key_column: "clip_id".into(),
            modality: Modality::Audio as i32,
        })
        .await
        .expect("generate_embeddings");

    let search = embedding
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::RowKey("clip_1".into())),
            k: 3,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
        })
        .await
        .expect("search")
        .into_inner();
    assert_eq!(search.hits.len(), 3, "k=3 over a three-row corpus");

    // Flight SQL round-trip — `execute` issues GetFlightInfo, `do_get`
    // issues DoGet, which anchors `flight_queries`.
    let channel = Channel::from_shared(format!("http://{server_flight_addr}"))
        .expect("channel uri")
        .connect()
        .await
        .expect("connect");
    let mut client = FlightSqlServiceClient::new(channel);
    let info = client
        .execute("SELECT 1 AS one".to_string(), None)
        .await
        .expect("flight execute");
    let endpoint = info
        .endpoint
        .first()
        .cloned()
        .expect("flight endpoint must be present");
    let ticket = endpoint.ticket.expect("endpoint ticket");
    let mut stream = client.do_get(ticket).await.expect("do_get");
    let mut total_rows: usize = 0;
    while let Some(batch) = stream.try_next().await.expect("flight stream") {
        total_rows += batch.num_rows();
    }
    assert!(
        total_rows > 0,
        "flight SQL roundtrip must return at least one row"
    );

    // Drive a real EvalService RPC. Register the patents text corpus + the
    // golden relevance set, embed `abstract` with the local tiny_bert encoder,
    // then run `EvalEmbeddings` over that table — mirroring `grpc_eval.rs`. This
    // is the only metric whose live path rides `/jammi.v1.eval.EvalService/*`,
    // so a wrong prefix in the layer makes the delta below stay zero.
    catalog
        .add_source(AddSourceRequest {
            source_id: "patents".into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: format!("file://{}", fixture("patents.parquet").display()),
                format: FileFormat::Parquet as i32,
            }),
        })
        .await
        .expect("add patents source");
    catalog
        .add_source(AddSourceRequest {
            source_id: "golden_rel".into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: format!("file://{}", fixture("golden_relevance.csv").display()),
                format: FileFormat::Csv as i32,
            }),
        })
        .await
        .expect("add golden source");
    let eval_table = embedding
        .generate_embeddings(GenerateEmbeddingsRequest {
            source_id: "patents".into(),
            model_id: tiny_bert_model_id(),
            columns: vec!["abstract".into()],
            key_column: "id".into(),
            modality: Modality::Text as i32,
        })
        .await
        .expect("generate_embeddings for eval")
        .into_inner()
        .table_name;
    let mut eval = EvalServiceClient::new(
        Channel::from_shared(format!("http://{server_flight_addr}"))
            .expect("eval channel uri")
            .connect()
            .await
            .expect("eval connect"),
    );
    let report = eval
        .eval_embeddings(EvalEmbeddingsRequest {
            source_id: "patents".into(),
            embedding_table: eval_table,
            golden_source: GOLDEN_SOURCE.into(),
            k: 10,
            cohorts: std::collections::HashMap::new(),
            tenant_id: String::new(),
        })
        .await
        .expect("eval_embeddings")
        .into_inner();
    assert!(
        !report.eval_run_id.is_empty(),
        "eval_embeddings must record a run id"
    );

    // Re-scrape and assert each substrate metric *moved*. This is the
    // dead-metric trap: a metric that is registered but never incremented
    // passes a name-presence check yet fails here.
    let after = scrape_metrics(&server_health_addr).await;
    let grpc_after = sample_value(&after, "jammi_grpc_requests_total");
    let flight_after = sample_value(&after, "jammi_flight_queries_total");
    let search_count_after = sample_value(&after, "jammi_search_latency_seconds_count");
    let eval_after = sample_value(&after, "jammi_eval_invocations_total");

    assert!(
        grpc_after > grpc_before,
        "gRPC requests must increment over the EmbeddingService calls: {grpc_before} -> {grpc_after}\n{after}"
    );
    assert!(
        flight_after > flight_before,
        "Flight queries must increment over the DoGet: {flight_before} -> {flight_after}\n{after}"
    );
    assert!(
        search_count_after > search_count_before,
        "search-latency histogram must observe the Search: count {search_count_before} -> {search_count_after}\n{after}"
    );
    assert!(
        eval_after > eval_before,
        "eval invocations must increment over the EvalService/EvalEmbeddings call: {eval_before} -> {eval_after}\n{after}"
    );

    // Trigger graceful shutdown and wait for the server task to drain.
    shutdown_tx.send(()).expect("shutdown send");
    let result = tokio::time::timeout(Duration::from_secs(5), server_task)
        .await
        .expect("server shutdown within 5s")
        .expect("server task panic");
    assert!(result.is_ok(), "server exit status: {result:?}");
}
