//! `EmbeddingService` end-to-end over the wire.
//!
//! An in-process Tonic server hosts the gRPC chain including the
//! `EmbeddingService`. A client registers a source, calls the unified
//! `GenerateEmbeddings` / `EncodeQuery` verbs keyed by a `Modality`, and
//! searches — all over a real HTTP/2 channel — asserting the engine returns
//! embeddings. The unification is exercised across two modalities:
//!
//! * `AUDIO` over a synthetic three-tone WAV corpus encoded by the
//!   `htsat_clap_tiny` real-key cookbook fixture.
//! * `TEXT` over the `patents.parquet` fixture's `abstract` column encoded by
//!   the `tiny_bert` cookbook fixture.
//!
//! Both are hermetic: the audio corpus is built in-process and both encoders
//! are local fixtures (no network, no download). This pins the wire adapter's
//! contract: the verbs route through the `Session` abstraction
//! and round-trip its results to a remote client. `RemoveSource` and the
//! modality/input validation at the wire edge are covered too.

use std::net::SocketAddr;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_ai::session::InferenceSession;
use jammi_server::grpc::proto::catalog::{
    AddSourceRequest, FileFormat, RemoveSourceRequest, SourceConnection, SourceKind,
};
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::encode_query_request::Input as EncodeInput;
use jammi_server::grpc::proto::embedding::search_request::Query as SearchQuery;
use jammi_server::grpc::proto::embedding::{
    EncodeQueryRequest, GenerateEmbeddingsRequest, ImportEmbeddingsRequest, Modality, QueryVector,
    SearchRequest,
};
use jammi_server::grpc::session::SessionStore;
use jammi_test_utils::{cookbook_fixture, fixture, test_config};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;
use tokio::sync::oneshot;

use super::common::grpc::{catalog_client, channel};

fn htsat_clap_model_id() -> String {
    format!("local:{}", cookbook_fixture("htsat_clap_tiny").display())
}

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_url() -> String {
    format!("file://{}", fixture("patents.parquet").display())
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

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let chain = jammi_server::runtime::GrpcChain {
        addr: super::common::grpc::ephemeral_addr(),
        flight_ctx: session.context().clone(),
        flight_binding: session.tenant_binding_arc(),
        store: store.clone(),
        trigger: None,
        engine: Some(session),
        tiers: jammi_server::tiers::TierSet::all_compiled(),
        metrics: Arc::new(jammi_server::routes::health::MetricsRegistry::new().unwrap()),
        tenant_resolver: jammi_server::grpc::session::SessionIdTenantResolver::arc(store),
    };
    let (addr, handle) = super::common::grpc::spawn_bound_chain(chain, shutdown_rx).await;

    (addr, shutdown_tx, dir, handle)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_and_encode_audio_modality_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let parquet_path = write_corpus_parquet(&dir);
    let model_id = htsat_clap_model_id();

    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    // AddSource — register the synthetic corpus parquet.
    catalog_client(addr)
        .await
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

    // GenerateEmbeddings(AUDIO) — one vector per row, persisted server-side.
    let table = client
        .generate_embeddings(GenerateEmbeddingsRequest {
            source_id: "clips".into(),
            model_id: model_id.clone(),
            columns: vec!["audio".into()],
            key_column: "clip_id".into(),
            modality: Modality::Audio as i32,
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect("generate_embeddings")
        .into_inner();

    assert_eq!(table.status, "ready", "embedding table must be ready");
    assert_eq!(table.row_count, 3, "three clips embedded");
    assert_eq!(table.source_id, "clips");
    assert!(
        table.dimensions > 0,
        "tiny CLAP records an embedding dimensionality; got {}",
        table.dimensions
    );

    // EncodeQuery(AUDIO) — a single clip → one L2-normalized vector.
    let query_wav = sine_wav_bytes(440.0);
    let resp = client
        .encode_query(EncodeQueryRequest {
            model_id,
            modality: Modality::Audio as i32,
            input: Some(EncodeInput::Data(query_wav)),
        })
        .await
        .expect("encode_query")
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_and_encode_text_modality_over_the_wire() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let model_id = tiny_bert_model_id();

    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    // AddSource — the patents fixture, embedded over its `abstract` column.
    catalog_client(addr)
        .await
        .add_source(AddSourceRequest {
            source_id: "patents".into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: patents_url(),
                format: FileFormat::Parquet as i32,
            }),
        })
        .await
        .expect("add_source");

    // GenerateEmbeddings(TEXT) — the same unified verb, a different tower.
    let table = client
        .generate_embeddings(GenerateEmbeddingsRequest {
            source_id: "patents".into(),
            model_id: model_id.clone(),
            columns: vec!["abstract".into()],
            key_column: "id".into(),
            modality: Modality::Text as i32,
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect("generate_embeddings")
        .into_inner();

    assert_eq!(table.status, "ready", "embedding table must be ready");
    assert!(table.row_count > 0, "patents corpus embeds some rows");
    assert_eq!(table.source_id, "patents");
    assert!(
        table.dimensions > 0,
        "tiny BERT records an embedding dimensionality; got {}",
        table.dimensions
    );

    // EncodeQuery(TEXT) — a text string → one L2-normalized vector whose
    // dimensionality matches the corpus the same model produced.
    let resp = client
        .encode_query(EncodeQueryRequest {
            model_id,
            modality: Modality::Text as i32,
            input: Some(EncodeInput::Text("quantum computing applications".into())),
        })
        .await
        .expect("encode_query")
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

/// Register the synthetic corpus and embed it (AUDIO modality). Shared setup
/// for the `Search` wire tests.
async fn embed_corpus(
    addr: SocketAddr,
    client: &mut EmbeddingServiceClient<tonic::transport::Channel>,
    dir: &TempDir,
) {
    let parquet_path = write_corpus_parquet(dir);
    catalog_client(addr)
        .await
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
        .generate_embeddings(GenerateEmbeddingsRequest {
            source_id: "clips".into(),
            model_id: htsat_clap_model_id(),
            columns: vec!["audio".into()],
            key_column: "clip_id".into(),
            modality: Modality::Audio as i32,
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect("generate_embeddings");
}

/// Encode `clip`'s tone into a query vector over the wire (AUDIO modality).
async fn encode_audio_query(
    client: &mut EmbeddingServiceClient<tonic::transport::Channel>,
    clip: Vec<u8>,
) -> Vec<f32> {
    client
        .encode_query(EncodeQueryRequest {
            model_id: htsat_clap_model_id(),
            modality: Modality::Audio as i32,
            input: Some(EncodeInput::Data(clip)),
        })
        .await
        .expect("encode_query")
        .into_inner()
        .embedding
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_by_query_vector_ranks_self_match_first_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);
    embed_corpus(addr, &mut client, &dir).await;

    // Encode clip_1's tone and search by that vector: clip_1 must rank first
    // (a clip is its own nearest neighbor under cosine similarity).
    let query_vec = encode_audio_query(&mut client, sine_wav_bytes(440.0)).await;

    let resp = client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::QueryVector(QueryVector { values: query_vec })),
            k: 3,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            oversample: None,
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
    embed_corpus(addr, &mut client, &dir).await;

    // Query-by-example: the engine resolves clip_2's stored vector internally;
    // the vector never crosses the wire. clip_2 is its own top neighbor.
    let resp = client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::RowKey("clip_2".into())),
            k: 3,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            oversample: None,
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
    embed_corpus(addr, &mut client, &dir).await;

    // Filter pushes a predicate over the hydrated source columns; select
    // projects `clip_id` into each hit's columns map.
    let resp = client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::RowKey("clip_0".into())),
            k: 3,
            embedding_table: None,
            filter: Some("clip_id != 'clip_0'".into()),
            select: vec!["clip_id".into()],
            oversample: None,
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
async fn remove_source_drops_the_source_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);
    embed_corpus(addr, &mut client, &dir).await;

    // RemoveSource drops the registered source; a search against it then fails
    // because the source no longer resolves.
    catalog_client(addr)
        .await
        .remove_source(RemoveSourceRequest {
            source_id: "clips".into(),
        })
        .await
        .expect("remove_source");

    client
        .search(SearchRequest {
            source_id: "clips".into(),
            query: Some(SearchQuery::RowKey("clip_0".into())),
            k: 3,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            oversample: None,
        })
        .await
        .expect_err("search against a removed source must fail: the source no longer resolves");

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
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            oversample: None,
        })
        .await
        .expect_err("a search with no query must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_embeddings_rejects_unspecified_modality() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    let err = client
        .generate_embeddings(GenerateEmbeddingsRequest {
            source_id: "clips".into(),
            model_id: htsat_clap_model_id(),
            columns: vec!["audio".into()],
            key_column: "clip_id".into(),
            modality: Modality::Unspecified as i32,
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect_err("unspecified modality must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encode_query_rejects_input_modality_mismatch() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    // TEXT modality with bytes input is a wire-edge mismatch.
    let err = client
        .encode_query(EncodeQueryRequest {
            model_id: tiny_bert_model_id(),
            modality: Modality::Text as i32,
            input: Some(EncodeInput::Data(vec![1, 2, 3])),
        })
        .await
        .expect_err("text modality with bytes input must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    // AUDIO modality with text input is the symmetric mismatch.
    let err = client
        .encode_query(EncodeQueryRequest {
            model_id: htsat_clap_model_id(),
            modality: Modality::Audio as i32,
            input: Some(EncodeInput::Text("not audio".into())),
        })
        .await
        .expect_err("audio modality with text input must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_service_rejects_unspecified_source_kind() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;

    let err = catalog_client(addr)
        .await
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

/// The `_row_id` keys the imported vectors are keyed by — the values a
/// registered source's key column must carry for search hydration to join.
const IMPORT_DOC_IDS: [&str; 3] = ["doc-0", "doc-1", "doc-2"];

/// Width of the precomputed import vectors.
const IMPORT_DIMS: usize = 4;

/// Write a `(doc_id: Utf8, body: Utf8)` source parquet whose `doc_id` values are
/// [`IMPORT_DOC_IDS`] — the source the imported embeddings are keyed into, so a
/// hydrating search can join `_row_id` back to `doc_id` and project `body`.
fn write_docs_source_parquet(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("docs.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let ids = Arc::new(StringArray::from_iter_values(IMPORT_DOC_IDS)) as ArrayRef;
    let bodies = Arc::new(StringArray::from_iter_values(
        IMPORT_DOC_IDS.iter().map(|id| format!("body of {id}")),
    )) as ArrayRef;
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, bodies]).unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

/// Write a `(_row_id: Utf8, vector: FixedSizeList<Float32>[IMPORT_DIMS])` parquet
/// of precomputed vectors keyed by [`IMPORT_DOC_IDS`]. Each doc gets a distinct
/// unit basis direction (doc-i gets the i-th standard basis vector), so a
/// nearest-neighbor query by one doc's direction resolves that doc
/// unambiguously. The vectors are non-zero, so
/// import L2-normalizes them without rejecting any.
fn write_import_vectors_parquet(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("precomputed.parquet");
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(item.clone(), IMPORT_DIMS as i32),
            false,
        ),
    ]));
    let ids = Arc::new(StringArray::from_iter_values(IMPORT_DOC_IDS)) as ArrayRef;
    // doc-i → basis vector e_i (scaled by 3 so it is clearly non-normalized on
    // input; import normalizes it).
    let flat: Vec<f32> = (0..IMPORT_DOC_IDS.len())
        .flat_map(|i| (0..IMPORT_DIMS).map(move |d| if d == i { 3.0 } else { 0.0 }))
        .collect();
    let vectors = Arc::new(
        FixedSizeListArray::try_new(
            item,
            IMPORT_DIMS as i32,
            Arc::new(Float32Array::from(flat)),
            None,
        )
        .unwrap(),
    ) as ArrayRef;
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_embeddings_registers_a_ready_searchable_table_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    // A registered source whose key column carries the ids the imports are keyed
    // by — the realistic "precompute offline, register the vectors for an
    // existing source" flow.
    let source_parquet = write_docs_source_parquet(&dir);
    catalog_client(addr)
        .await
        .add_source(AddSourceRequest {
            source_id: "docs".into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: format!("file://{}", source_parquet.display()),
                format: FileFormat::Parquet as i32,
            }),
        })
        .await
        .expect("add_source");

    // ImportEmbeddings — promote the precomputed vectors into a ready table
    // WITHOUT re-running any model (the model id is never loaded).
    let vectors_parquet = write_import_vectors_parquet(&dir);
    let table = client
        .import_embeddings(ImportEmbeddingsRequest {
            source_id: "docs".into(),
            model_id: "import-model".into(),
            vectors_url: format!("file://{}", vectors_parquet.display()),
            key_column: "doc_id".into(),
            text_columns: vec!["body".into()],
            dimensions: IMPORT_DIMS as u32,
        })
        .await
        .expect("import_embeddings")
        .into_inner();

    assert_eq!(table.status, "ready", "imported table must be ready");
    assert_eq!(table.source_id, "docs");
    assert_eq!(
        table.model_id, "import-model",
        "model id recorded canonical"
    );
    assert_eq!(
        table.row_count,
        IMPORT_DOC_IDS.len() as u64,
        "every precomputed row is imported"
    );
    assert_eq!(
        table.dimensions, IMPORT_DIMS as i32,
        "the imported table records its embedding width"
    );

    // Search by doc-1's direction (its standard basis vector): the imported
    // doc-1 row is its own nearest neighbor under cosine, so it ranks
    // first. `select` projects the
    // hydrated `body` column, proving the imported table joins back to its
    // source exactly like a generated one.
    let mut query = vec![0.0_f32; IMPORT_DIMS];
    query[1] = 1.0;
    let resp = client
        .search(SearchRequest {
            source_id: "docs".into(),
            query: Some(SearchQuery::QueryVector(QueryVector { values: query })),
            k: IMPORT_DOC_IDS.len() as u32,
            embedding_table: None,
            filter: None,
            select: vec!["body".into()],
            oversample: None,
        })
        .await
        .expect("search over the imported table")
        .into_inner();

    assert_eq!(
        resp.hits.len(),
        IMPORT_DOC_IDS.len(),
        "k over the imported corpus returns every row"
    );
    assert_eq!(
        resp.hits[0].key, "doc-1",
        "the query row is its own nearest neighbor"
    );
    assert!(
        resp.hits[0].score >= resp.hits[1].score && resp.hits[1].score >= resp.hits[2].score,
        "hits ordered by descending score, got {:?}",
        resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
    );
    assert_eq!(
        resp.hits[0].columns.get("body").map(String::as_str),
        Some("body of doc-1"),
        "the imported table hydrates its source's projected columns"
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_embeddings_rejects_dimension_mismatch_over_the_wire() {
    let (addr, shutdown, dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    // The vectors are width IMPORT_DIMS; claiming a different width is a caller
    // error rejected before any table is materialized.
    let vectors_parquet = write_import_vectors_parquet(&dir);
    let err = client
        .import_embeddings(ImportEmbeddingsRequest {
            source_id: "docs".into(),
            model_id: "import-model".into(),
            vectors_url: format!("file://{}", vectors_parquet.display()),
            key_column: "doc_id".into(),
            text_columns: Vec::new(),
            dimensions: (IMPORT_DIMS + 1) as u32,
        })
        .await
        .expect_err("a width mismatch must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_embeddings_requires_core_fields_over_the_wire() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    // A zero `dimensions` is rejected at the wire edge (as are the empty
    // required-string fields), before any storage read.
    let err = client
        .import_embeddings(ImportEmbeddingsRequest {
            source_id: "docs".into(),
            model_id: "import-model".into(),
            vectors_url: "file:///tmp/whatever.parquet".into(),
            key_column: "doc_id".into(),
            text_columns: Vec::new(),
            dimensions: 0,
        })
        .await
        .expect_err("dimensions must be > 0");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encode_query_rejects_empty_data() {
    let (addr, shutdown, _dir, handle) = start_embedding_server().await;
    let mut client = EmbeddingServiceClient::new(channel(addr).await);

    let err = client
        .encode_query(EncodeQueryRequest {
            model_id: htsat_clap_model_id(),
            modality: Modality::Audio as i32,
            input: Some(EncodeInput::Data(Vec::new())),
        })
        .await
        .expect_err("empty data must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = shutdown.send(());
    let _ = handle.await;
}
