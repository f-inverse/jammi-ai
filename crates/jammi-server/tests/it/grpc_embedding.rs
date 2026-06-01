//! `EmbeddingService` end-to-end over the wire.
//!
//! An in-process Tonic server hosts the gRPC chain including the new
//! `EmbeddingService`. A client registers a synthetic audio corpus, calls
//! `GenerateAudioEmbeddings`, and calls `EncodeAudioQuery` — all over a real
//! HTTP/2 channel — and asserts the engine returns embeddings. Hermetic: the
//! corpus is three synthetic WAV tones built in-process and the encoder is
//! the `tiny_clap` random-weight cookbook fixture (no network, no download).
//!
//! This pins the wire adapter's contract: the verbs delegate to the
//! `InferenceSession` audio-embedding path and round-trip its results to a
//! remote client.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_ai::session::InferenceSession;
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::search_request::Query as SearchQuery;
use jammi_server::grpc::proto::embedding::{
    AddSourceRequest, EncodeAudioQueryRequest, FileFormat, GenerateAudioEmbeddingsRequest,
    QueryVector, SearchRequest, SourceConnection, SourceKind,
};
use jammi_server::grpc::session::SessionStore;
use jammi_test_utils::{cookbook_fixture, test_config};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use super::common::grpc::channel;

fn tiny_clap_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_clap").display())
}

/// Build a minimal 16-bit PCM mono WAV (16 kHz) holding a sine tone at
/// `freq`. Matches the cookbook smoke test's synthetic corpus generator so
/// the wire path exercises the same realistic decode → log-mel input.
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

/// Spin up an in-process gRPC server hosting the chain *with* the embedding
/// service (the only fixture in the suite that constructs an
/// `InferenceSession`, which the embedding verbs delegate to). Returns the
/// bound address plus the guards that keep the server + catalog alive.
async fn start_embedding_server() -> (
    SocketAddr,
    oneshot::Sender<()>,
    TempDir,
    tokio::task::JoinHandle<()>,
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let session = Arc::new(InferenceSession::new(cfg).await.expect("session"));

    let store = SessionStore::new();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let flight_ctx = session.context().clone();
    let binding = session.tenant_binding_arc();
    let handle = tokio::spawn(async move {
        jammi_server::runtime::serve_grpc_chain(
            addr,
            flight_ctx,
            binding,
            store,
            None,
            Some(session),
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
        .expect("grpc server");
    });

    // Give the server a moment to bind.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, shutdown_tx, dir, handle)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embedding_service_generates_and_encodes_audio_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let parquet_path = write_corpus_parquet(&dir);
    let model_id = tiny_clap_model_id();

    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    // AddSource — register the synthetic corpus parquet.
    client
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

    // GenerateAudioEmbeddings — one vector per row, persisted server-side.
    let table = client
        .generate_audio_embeddings(GenerateAudioEmbeddingsRequest {
            source_id: "clips".into(),
            model_id: model_id.clone(),
            audio_column: "audio".into(),
            key_column: "clip_id".into(),
        })
        .await
        .expect("generate_audio_embeddings")
        .into_inner();

    assert_eq!(table.status, "ready", "embedding table must be ready");
    assert_eq!(table.row_count, 3, "three clips embedded");
    assert_eq!(table.source_id, "clips");
    assert!(
        table.dimensions > 0,
        "tiny CLAP records an embedding dimensionality; got {}",
        table.dimensions
    );

    // EncodeAudioQuery — a single clip → one L2-normalized vector.
    let query_wav = sine_wav_bytes(440.0);
    let resp = client
        .encode_audio_query(EncodeAudioQueryRequest {
            model_id,
            audio_bytes: query_wav,
        })
        .await
        .expect("encode_audio_query")
        .into_inner();

    assert_eq!(
        resp.embedding.len() as i32,
        table.dimensions,
        "query embedding dim must match the corpus embedding dim"
    );
    let norm: f32 = resp.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "query vector must be L2-normalized, got norm={norm}"
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}

/// Register the synthetic corpus and embed it, returning the embedding-table
/// dimensionality. Shared setup for the `Search` wire tests.
async fn embed_corpus(
    client: &mut EmbeddingServiceClient<tonic::transport::Channel>,
    dir: &TempDir,
) {
    let parquet_path = write_corpus_parquet(dir);
    client
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
    client
        .generate_audio_embeddings(GenerateAudioEmbeddingsRequest {
            source_id: "clips".into(),
            model_id: tiny_clap_model_id(),
            audio_column: "audio".into(),
            key_column: "clip_id".into(),
        })
        .await
        .expect("generate_audio_embeddings");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_by_query_vector_ranks_self_match_first_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);
    embed_corpus(&mut client, &dir).await;

    // Encode clip_1's tone and search by that vector: clip_1 must rank first
    // (a clip is its own nearest neighbor under cosine similarity).
    let query_clip = sine_wav_bytes(440.0);
    let query_vec = client
        .encode_audio_query(EncodeAudioQueryRequest {
            model_id: tiny_clap_model_id(),
            audio_bytes: query_clip,
        })
        .await
        .expect("encode_audio_query")
        .into_inner()
        .embedding;

    let resp = client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::QueryVector(QueryVector { values: query_vec })),
            k: 3,
            filter: None,
            select: Vec::new(),
        })
        .await
        .expect("search by vector")
        .into_inner();

    assert_eq!(resp.hits.len(), 3, "k=3 over a three-row corpus");
    assert_eq!(resp.hits[0].key, "clip_1", "self-match ranks first");
    assert!(
        resp.hits[0].score >= resp.hits[1].score && resp.hits[1].score >= resp.hits[2].score,
        "hits must be ordered by descending score, got {:?}",
        resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
    );
    assert!(
        resp.hits.iter().all(|h| h.columns.is_empty()),
        "empty select returns key + score only, no columns"
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_by_row_key_ranks_that_row_first_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);
    embed_corpus(&mut client, &dir).await;

    // Query-by-example: the engine resolves clip_2's stored vector internally;
    // the vector never crosses the wire. clip_2 is its own top neighbor.
    let resp = client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::RowKey("clip_2".into())),
            k: 3,
            filter: None,
            select: Vec::new(),
        })
        .await
        .expect("search by row_key")
        .into_inner();

    assert_eq!(resp.hits.len(), 3, "k=3 over a three-row corpus");
    assert_eq!(resp.hits[0].key, "clip_2", "the query row ranks first");

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_applies_filter_and_select_projection_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);
    embed_corpus(&mut client, &dir).await;

    // Filter pushes a predicate over the hydrated source columns; select
    // projects `clip_id` into each hit's columns map.
    let resp = client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::RowKey("clip_0".into())),
            k: 3,
            filter: Some("clip_id != 'clip_0'".into()),
            select: vec!["clip_id".into()],
        })
        .await
        .expect("search with filter + select")
        .into_inner();

    assert!(
        !resp.hits.is_empty() && resp.hits.len() <= 2,
        "filter excludes clip_0, leaving at most the two other rows"
    );
    for hit in &resp.hits {
        assert_ne!(hit.key, "clip_0", "filtered row must not appear");
        assert_eq!(
            hit.columns.get("clip_id").map(String::as_str),
            Some(hit.key.as_str()),
            "projected clip_id must equal the hit key"
        );
    }

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_requires_a_query_over_the_wire() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    let err = client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: None,
            k: 3,
            filter: None,
            select: Vec::new(),
        })
        .await
        .expect_err("a search with no query must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embedding_service_rejects_unspecified_source_kind() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    let err = client
        .add_source(AddSourceRequest {
            source_id: "clips".into(),
            source_kind: SourceKind::Unspecified as i32,
            connection: Some(SourceConnection {
                url: "file:///tmp/whatever.parquet".into(),
                format: FileFormat::Parquet as i32,
            }),
        })
        .await
        .expect_err("unspecified source kind must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encode_audio_query_rejects_empty_bytes() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    let err = client
        .encode_audio_query(EncodeAudioQueryRequest {
            model_id: tiny_clap_model_id(),
            audio_bytes: Vec::new(),
        })
        .await
        .expect_err("empty audio_bytes must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}
