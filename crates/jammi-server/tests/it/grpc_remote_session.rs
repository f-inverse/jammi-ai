//! `RemoteSession` over the wire, proven interchangeable with `LocalSession`.
//!
//! An in-process gRPC chain (`runtime::serve_grpc_chain`) hosts a real
//! `InferenceSession`. A `jammi_ai::RemoteSession` connects to it over a real
//! HTTP/2 channel and a `jammi_ai::LocalSession` wraps the *same* engine `Arc`.
//! The three properties Stage 3b-1 must establish are each pinned here:
//!
//! * **Round-trip parity** — `generate_embeddings` → `search` → `remove_source`
//!   plus `encode_query` through the remote transport return the same results a
//!   local session returns against the same engine (realistic `tiny_bert` text
//!   embeddings over the `patents` corpus, never dummy vectors).
//! * **Error parity (the #1 proof)** — a real failure on this path (searching a
//!   source with no embedding table) returns the *same `JammiError` variant*
//!   from `RemoteSession` as from `LocalSession`, not merely the same gRPC code.
//!   This is what the typed-error wire detail buys; a heuristic reverse-map
//!   could not satisfy it.
//! * **Tenant** — `bind_tenant` (async) over the wire is observed by a later
//!   `tenant()` read; the binding is keyed by the client's session id.
//!
//! Hermetic: the encoder is the local `tiny_bert` cookbook fixture and the
//! corpus is the bundled `patents.parquet`; no live network, no download.

use std::sync::Arc;

use jammi_ai::{
    LocalSession, Modality, QueryInput, RemoteSession, SearchQuery, SearchRequest, Session,
};
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_test_utils::{cookbook_fixture, fixture};
use tonic::transport::Endpoint;

use super::common::grpc::{start_engine_server, tenant_a, EngineServer};

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_connection() -> SourceConnection {
    SourceConnection {
        url: Some(format!("file://{}", fixture("patents.parquet").display())),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }
}

/// Connect a `RemoteSession` to the in-process server.
async fn remote(server: &EngineServer) -> RemoteSession {
    let endpoint = Endpoint::from_shared(format!("http://{}", server.addr)).expect("endpoint");
    RemoteSession::connect(endpoint)
        .await
        .expect("remote session connect")
}

/// Wrap the server's engine `Arc` in a local session — the same engine the
/// remote calls reach, so any divergence is the transport's fault, not the
/// engine's.
fn local(server: &EngineServer) -> Session {
    Session::Local(LocalSession::new(Arc::clone(&server.engine)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_round_trips_embeddings_and_search_like_local() {
    let server = start_engine_server().await;
    let remote = Session::Remote(remote(&server).await);
    let model_id = tiny_bert_model_id();

    // The local session registers + embeds the corpus directly on the shared
    // engine (AddSource is a 3b-2 verb on the remote arm); both sessions then
    // search the same persisted embedding table.
    let local = local(&server);
    local
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect("add_source");

    let remote_table = remote
        .generate_embeddings(
            "patents",
            &model_id,
            &["abstract".to_string()],
            "id",
            Modality::Text,
        )
        .await
        .expect("remote generate_embeddings");

    assert_eq!(remote_table.status, "ready");
    assert!(remote_table.row_count > 0, "patents corpus embeds rows");
    assert!(remote_table.dimensions.is_some(), "dimensions recorded");
    assert_eq!(remote_table.source_id, "patents");
    // The remote arm reconstructs `task` from the requested modality (the wire
    // omits it as server-internal bookkeeping); it must match the tower.
    assert_eq!(remote_table.task, jammi_db::ModelTask::TextEmbedding);

    // encode_query parity: identical query, identical model → identical vector.
    let query = "quantum computing applications";
    let remote_vec = remote
        .encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        )
        .await
        .expect("remote encode_query");
    let local_vec = local
        .encode_query(
            &model_id,
            QueryInput::Text(query.to_string()),
            Modality::Text,
        )
        .await
        .expect("local encode_query");
    assert_eq!(
        remote_vec.len(),
        local_vec.len(),
        "remote/local query vectors share dimensionality"
    );
    assert_eq!(
        remote_vec, local_vec,
        "the same query through either transport encodes to the same vector"
    );

    // search parity: same request, same top-k keys + scores. The remote arm
    // rebuilds the result batch from the wire hits; compare on the
    // client-observable key/score the search verb surfaces.
    let request = |select: Vec<String>| SearchRequest {
        source_id: "patents".to_string(),
        query: SearchQuery::Vector(remote_vec.clone()),
        k: 5,
        filter: None,
        select,
    };
    let remote_hits = keys_and_scores(
        remote
            .search(request(Vec::new()))
            .await
            .expect("remote search"),
    );
    let local_hits = keys_and_scores(
        local
            .search(request(Vec::new()))
            .await
            .expect("local search"),
    );
    assert!(!remote_hits.is_empty(), "search returns hits");
    assert_eq!(
        remote_hits, local_hits,
        "remote and local search rank the same rows with the same scores"
    );

    // remove_source parity: after the remote drops the source, a search fails
    // on both transports (the source no longer resolves).
    remote
        .remove_source("patents")
        .await
        .expect("remote remove_source");
    let remote_err = remote
        .search(request(Vec::new()))
        .await
        .expect_err("search after remove must fail");
    let local_err = local
        .search(request(Vec::new()))
        .await
        .expect_err("local search after remove must fail too");
    assert_eq!(
        std::mem::discriminant(&remote_err),
        std::mem::discriminant(&local_err),
        "remote and local agree on the failure variant after removal"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// THE error-parity proof. A search against a source that has no ready
/// embedding table fails inside the engine with `JammiError::Catalog`. The
/// remote transport must reconstruct that *exact* variant from the typed wire
/// detail — not just report `invalid_argument`. A heuristic reverse-map from
/// the gRPC code could not distinguish `Catalog` from `Source` / `Config` /
/// `Eval` / `Tenant`, all of which the server maps onto `invalid_argument`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_reconstructs_the_exact_error_variant_local_returns() {
    let server = start_engine_server().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);

    // Register the corpus but never generate embeddings: a search-by-row-key
    // then fails resolving the (absent) embedding table — `JammiError::Catalog`.
    local
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect("add_source");

    let request = || SearchRequest {
        source_id: "patents".to_string(),
        query: SearchQuery::RowKey("any-key".to_string()),
        k: 3,
        filter: None,
        select: Vec::new(),
    };

    let local_err = local
        .search(request())
        .await
        .expect_err("local search must fail");
    let remote_err = remote
        .search(request())
        .await
        .expect_err("remote search must fail");

    // Same variant AND same payload — faithful reconstruction, not a category.
    assert!(
        matches!(local_err, JammiError::Catalog(_)),
        "local search on a source with no embedding table is a Catalog error, got {local_err:?}"
    );
    match (&local_err, &remote_err) {
        (JammiError::Catalog(local_msg), JammiError::Catalog(remote_msg)) => {
            assert_eq!(
                local_msg, remote_msg,
                "the remote transport carries the same Catalog message the engine produced"
            );
        }
        other => panic!("remote did not reconstruct the Catalog variant: {other:?}"),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Error-parity, second distinct variant on THIS surface. `encode_query` runs
/// model inference: resolving a `local:` model whose directory does not exist
/// fails inside the engine with `JammiError::Model { model_id, message }` — a
/// struct variant distinct from the `Catalog` case above, and one the typed wire
/// detail must reconstruct field-for-field. A heuristic reverse-map from the
/// gRPC code (`invalid_argument`) could not recover `model_id` or distinguish
/// `Model` from `Source` / `Config` / `Tenant`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_reconstructs_a_model_error_from_an_inference_failure() {
    let server = start_engine_server().await;
    let remote = Session::Remote(remote(&server).await);
    let local = local(&server);

    // A syntactically valid `local:` model id pointing at a directory that does
    // not exist. The query input matches the modality, so the request clears
    // wire validation and reaches the engine's model resolver, which fails.
    let missing_model = "local:/nonexistent/jammi-test-model";
    let query = || {
        (
            missing_model.to_string(),
            QueryInput::Text("quantum computing".to_string()),
            Modality::Text,
        )
    };

    let (m, i, md) = query();
    let local_err = local
        .encode_query(&m, i, md)
        .await
        .expect_err("local encode_query on a missing model must fail");
    let (m, i, md) = query();
    let remote_err = remote
        .encode_query(&m, i, md)
        .await
        .expect_err("remote encode_query on a missing model must fail");

    // Same variant AND same fields — the struct variant crosses the wire intact.
    match (&local_err, &remote_err) {
        (
            JammiError::Model {
                model_id: local_id,
                message: local_msg,
            },
            JammiError::Model {
                model_id: remote_id,
                message: remote_msg,
            },
        ) => {
            assert_eq!(
                local_id, remote_id,
                "the remote transport carries the same model_id the engine produced"
            );
            assert_eq!(
                local_msg, remote_msg,
                "the remote transport carries the same model message the engine produced"
            );
        }
        other => panic!("expected a faithful Model error from both transports, got {other:?}"),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Tenant trio over the wire: a `bind_tenant` (now async) followed by a
/// `tenant()` read observes the bound tenant; the binding is keyed by the
/// client's own session id, so a second client sees nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_binds_and_reads_tenant_over_the_wire() {
    let server = start_engine_server().await;
    let bound = Session::Remote(remote(&server).await);
    let other = Session::Remote(remote(&server).await);

    assert_eq!(bound.tenant().await.expect("get tenant"), None);

    bound.bind_tenant(tenant_a()).await.expect("bind tenant");
    assert_eq!(
        bound.tenant().await.expect("get tenant"),
        Some(tenant_a()),
        "the bound client observes its tenant"
    );
    assert_eq!(
        other.tenant().await.expect("get tenant"),
        None,
        "a different session id sees no binding"
    );

    bound.unbind_tenant().await.expect("unbind tenant");
    assert_eq!(
        bound.tenant().await.expect("get tenant"),
        None,
        "after unbind the tenant is cleared"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

/// Extract the (key, score) of each hit from a search result, the
/// client-observable identity of a hit, for cross-transport comparison.
fn keys_and_scores(batches: Vec<arrow::record_batch::RecordBatch>) -> Vec<(String, f32)> {
    use arrow::array::{Float32Array, StringArray};
    let mut out = Vec::new();
    for batch in &batches {
        let keys = batch
            .column_by_name("_row_id")
            .expect("_row_id column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_row_id is Utf8");
        let scores = batch
            .column_by_name("similarity")
            .expect("similarity column")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("similarity is Float32");
        for row in 0..batch.num_rows() {
            out.push((keys.value(row).to_string(), scores.value(row)));
        }
    }
    out
}
