//! `EmbeddingService` end-to-end over the wire, on a real CUDA device — the
//! client/server-topology GPU proof.
//!
//! Jammi's recommended shape is client/server + GPU: a CUDA `jammi-server`
//! served over gRPC / Flight SQL, driven by a remote client. That shape is
//! otherwise never exercised on a GPU anywhere — the `jammi-ai` crate's
//! `tests/gpu_capability` integration suite proves the engine core
//! *in-process*, and every other
//! lane is CPU. Here an in-process Tonic chain hosts a GPU-pinned
//! `InferenceSession` (`gpu.device = 0`, `require_gpu = true`) and a client
//! registers a source, runs the unified `GenerateEmbeddings` / `EncodeQuery`
//! verbs, and searches — all over a real HTTP/2 channel — asserting the served
//! GPU path returns well-formed, L2-normalized embeddings whose query dimension
//! matches the corpus the same model produced.
//!
//! This is the release gate for the CUDA server artifacts: a GPU build whose
//! *served* topology is unproven on a GPU must not ship — the class of failure
//! in issue #277 (a CUDA artifact that fails to load / serve on a device) is
//! invisible until a user hits it.
//!
//! ## Gating
//!
//! The module is compiled only under the `live-gpu-tests` cargo feature (its
//! `mod` line in `main.rs` is `#[cfg(feature = "live-gpu-tests")]`), and a
//! meaningful run also needs the `cuda` feature and a visible GPU. The GPU
//! session pins `require_gpu = true`, so on a CUDA host a test that reached the
//! wire calls *did* run on the GPU. Without a usable GPU the session fails to
//! construct, so the test skips with a `tracing::warn` (never a failure) and the
//! CPU / GPU-less lane runs it as a no-op. Live run:
//!
//! ```text
//! cargo test -p jammi-server --features cuda,live-gpu-tests --test it \
//!   grpc_embedding_gpu -- --nocapture --test-threads=1
//! ```

use std::net::SocketAddr;
use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_server::grpc::proto::catalog::{
    AddSourceRequest, FileFormat, SourceConnection, SourceKind,
};
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::encode_query_request::Input as EncodeInput;
use jammi_server::grpc::proto::embedding::search_request::Query as SearchQuery;
use jammi_server::grpc::proto::embedding::{
    EncodeQueryRequest, GenerateEmbeddingsRequest, Modality, QueryVector, SearchRequest,
};
use jammi_server::grpc::session::SessionStore;
use jammi_test_utils::{cookbook_fixture, fixture, test_config};
use tempfile::TempDir;
use tokio::sync::oneshot;

use super::common::grpc::{catalog_client, channel};

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_url() -> String {
    format!("file://{}", fixture("patents.parquet").display())
}

/// Spin up an in-process gRPC server whose `InferenceSession` is pinned to the
/// first CUDA device (`gpu.device = 0`, `require_gpu = true`). Returns `None`
/// — a clean skip — when no usable GPU opens (a CPU build, or a GPU-less host),
/// so the suite is a no-op off a CUDA host rather than a failure. A returned
/// `Some` guarantees the session was constructed on the GPU, so every wire call
/// against it runs the real CUDA served path.
async fn start_gpu_embedding_server() -> Option<(
    SocketAddr,
    oneshot::Sender<()>,
    TempDir,
    tokio::task::JoinHandle<()>,
)> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = test_config(dir.path());
    cfg.gpu.device = 0;
    cfg.gpu.require_gpu = true;

    let session = match InferenceSession::new(cfg).await {
        Ok(session) => Arc::new(session),
        Err(err) => {
            tracing::warn!(
                "SKIP grpc_embedding_gpu: no usable CUDA device — build with \
                 `--features cuda,live-gpu-tests` on a GPU host to run it ({err})"
            );
            return None;
        }
    };

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

    Some((addr, shutdown_tx, dir, handle))
}

/// The served GPU path end-to-end: register a corpus, embed it (TEXT tower on
/// the GPU), encode a text query on the GPU, and search by that query vector —
/// all over a real gRPC channel — asserting well-formed, L2-normalized,
/// dimensionally-consistent results. This is the recommended-topology proof:
/// the same wire verbs a `grpc://` client calls, running on real silicon.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_embeddings_served_over_the_wire_on_gpu() {
    let Some((addr, shutdown, _dir, handle)) = start_gpu_embedding_server().await else {
        return;
    };
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

    // GenerateEmbeddings(TEXT) — the TEXT tower runs on the GPU, one vector per
    // row, persisted server-side.
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

    // EncodeQuery(TEXT) — a text string → one L2-normalized vector on the GPU
    // whose dimensionality matches the corpus the same model produced.
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
    assert!(
        resp.embedding.iter().all(|v| v.is_finite()),
        "GPU query embedding must be finite — a NaN/Inf lane is a real kernel bug"
    );
    let norm: f32 = resp.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "query vector must be L2-normalized, got norm={norm}"
    );

    // Search by the GPU-encoded query vector: the served ANN path returns ranked
    // hits, ordered by descending score. This exercises encode → search over the
    // wire against the GPU-embedded corpus.
    let resp = client
        .search(SearchRequest {
            source_id: "patents".into(),
            query: Some(SearchQuery::QueryVector(QueryVector {
                values: resp.embedding,
            })),
            k: 5,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            oversample: None,
        })
        .await
        .expect("search by GPU-encoded query vector")
        .into_inner();

    assert!(
        !resp.hits.is_empty() && resp.hits.len() <= 5,
        "k=5 search returns between 1 and 5 hits, got {}",
        resp.hits.len()
    );
    assert!(
        resp.hits.windows(2).all(|w| w[0].score >= w[1].score),
        "hits must be ordered by descending score, got {:?}",
        resp.hits.iter().map(|h| h.score).collect::<Vec<_>>()
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}
