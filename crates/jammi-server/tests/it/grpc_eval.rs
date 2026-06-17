//! `EvalService` end-to-end over the wire.
//!
//! An in-process Tonic server hosts the gRPC chain including the `EvalService`.
//! A client registers the `patents.parquet` corpus and the
//! `golden_relevance.csv` golden set, generates embeddings with the local
//! `tiny_bert` encoder (all through the engine-backed services on one shared
//! session), then drives the eval verbs:
//!
//! * `EvalEmbeddings` → an `EmbeddingEvalReport` with a recorded run id, an
//!   aggregate, and one per-query record per golden query.
//! * `EvalPerQuery` → the persisted per-query rows for that run.
//! * `EvalCompare` → a baseline + one delta entry for a self-comparison (every
//!   delta zero).
//!
//! Hermetic: the encoder is a local fixture and the corpus + golden set are
//! shipped fixtures. A tenant-scoped `EvalEmbeddings` is covered too.

use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::{
    AddSourceRequest, FileFormat, SourceConnection, SourceKind,
};
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::{GenerateEmbeddingsRequest, Modality};
use jammi_server::grpc::proto::eval::eval_service_client::EvalServiceClient;
use jammi_server::grpc::proto::eval::{
    EvalCompareRequest, EvalEmbeddingsRequest, EvalPerQueryRequest,
};
use jammi_test_utils::{cookbook_fixture, fixture};
use std::collections::HashMap;
use tonic::codegen::Body;

use super::common::grpc::{channel, start_engine_server, with_session, EngineServer, TENANT_A};

const GOLDEN_SOURCE: &str = "golden_rel.public.golden_relevance";

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_url() -> String {
    format!("file://{}", fixture("patents.parquet").display())
}

fn golden_url() -> String {
    format!("file://{}", fixture("golden_relevance.csv").display())
}

/// Register the patents corpus + golden relevance CSV (control plane) and
/// generate one embedding table over `abstract` (data plane). Returns the
/// generated table name. Both clients share one transport (and, for the
/// tenant-scoped test, one session-header interceptor), so the source
/// registration and the embedding compute run under the same tenant scope.
/// Generic over the client transport so the plain-channel tests and the
/// interceptor-wrapped tenant test share one body.
async fn embed_patents_and_golden<T>(
    mut catalog: CatalogServiceClient<T>,
    mut embedding: EmbeddingServiceClient<T>,
) -> String
where
    T: tonic::client::GrpcService<tonic::body::Body> + Clone,
    T::Error: Into<tonic::codegen::StdError>,
    T::ResponseBody: Body<Data = tonic::codegen::Bytes> + std::marker::Send + 'static,
    <T::ResponseBody as Body>::Error: Into<tonic::codegen::StdError> + std::marker::Send,
{
    catalog
        .add_source(AddSourceRequest {
            source_id: "patents".into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: patents_url(),
                format: FileFormat::Parquet as i32,
            }),
        })
        .await
        .expect("add patents");
    catalog
        .add_source(AddSourceRequest {
            source_id: "golden_rel".into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: golden_url(),
                format: FileFormat::Csv as i32,
            }),
        })
        .await
        .expect("add golden");
    embedding
        .generate_embeddings(GenerateEmbeddingsRequest {
            source_id: "patents".into(),
            model_id: tiny_bert_model_id(),
            columns: vec!["abstract".into()],
            key_column: "id".into(),
            modality: Modality::Text as i32,
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect("generate_embeddings")
        .into_inner()
        .table_name
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eval_embeddings_and_per_query_over_the_wire() {
    let server: EngineServer = start_engine_server().await;
    let table = embed_patents_and_golden(
        CatalogServiceClient::new(channel(server.addr).await),
        EmbeddingServiceClient::new(channel(server.addr).await),
    )
    .await;

    let mut client = EvalServiceClient::new(channel(server.addr).await);
    let report = client
        .eval_embeddings(EvalEmbeddingsRequest {
            source_id: "patents".into(),
            embedding_table: table,
            golden_source: GOLDEN_SOURCE.into(),
            k: 10,
            cohorts: HashMap::new(),
            tenant_id: String::new(),
        })
        .await
        .expect("eval_embeddings")
        .into_inner();

    // The report carries a recorded run id, an aggregate, and per-query rows.
    assert!(!report.eval_run_id.is_empty(), "run id is recorded");
    let aggregate = report.aggregate.expect("aggregate present");
    for (name, v) in [
        ("recall_at_k", aggregate.recall_at_k),
        ("precision_at_k", aggregate.precision_at_k),
        ("mrr", aggregate.mrr),
        ("ndcg", aggregate.ndcg),
    ] {
        assert!((0.0..=1.0).contains(&v), "{name}={v} outside [0,1]");
    }
    assert!(
        !report.per_query.is_empty(),
        "one per-query record per golden query"
    );
    let first = &report.per_query[0];
    assert!(
        !first.query_id.is_empty(),
        "per-query record carries query_id"
    );
    assert!(
        first.metrics.is_some(),
        "per-query record carries typed metrics"
    );

    // EvalPerQuery reads back the persisted rows for the same run.
    let persisted = client
        .eval_per_query(EvalPerQueryRequest {
            eval_run_id: report.eval_run_id.clone(),
            tenant_id: String::new(),
        })
        .await
        .expect("eval_per_query")
        .into_inner();
    assert_eq!(
        persisted.records.len(),
        report.per_query.len(),
        "one persisted row per per-query record"
    );
    for rec in &persisted.records {
        assert_eq!(rec.eval_run_id, report.eval_run_id);
        // metrics/cohorts columns are JSON-object strings by storage shape.
        let m: serde_json::Value = serde_json::from_str(&rec.metrics_json).expect("metrics json");
        assert!(m.is_object(), "metrics_json is a JSON object");
        let c: serde_json::Value = serde_json::from_str(&rec.cohorts_json).expect("cohorts json");
        assert!(c.is_object(), "cohorts_json is a JSON object");
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eval_compare_self_comparison_has_zero_deltas_over_the_wire() {
    let server = start_engine_server().await;
    let table = embed_patents_and_golden(
        CatalogServiceClient::new(channel(server.addr).await),
        EmbeddingServiceClient::new(channel(server.addr).await),
    )
    .await;

    let mut client = EvalServiceClient::new(channel(server.addr).await);
    let report = client
        .eval_compare(EvalCompareRequest {
            embedding_tables: vec![table.clone(), table],
            source_id: "patents".into(),
            golden_source: GOLDEN_SOURCE.into(),
            k: 10,
            tenant_id: String::new(),
        })
        .await
        .expect("eval_compare")
        .into_inner();

    assert_eq!(report.per_table.len(), 2, "two compared tables");
    assert!(
        report.per_table[0].delta.is_none(),
        "baseline carries no delta"
    );
    let delta = report.per_table[1]
        .delta
        .as_ref()
        .expect("non-baseline carries a delta");
    for (name, v) in [
        ("recall_at_k", delta.recall_at_k.as_ref().unwrap().absolute),
        (
            "precision_at_k",
            delta.precision_at_k.as_ref().unwrap().absolute,
        ),
        ("mrr", delta.mrr.as_ref().unwrap().absolute),
        ("ndcg", delta.ndcg.as_ref().unwrap().absolute),
    ] {
        assert!(
            v.abs() < 1e-9,
            "self-comparison {name} delta must be 0, got {v}"
        );
    }
    // The baseline entry also carries a full embedding eval report.
    assert!(
        report.per_table[0].embedding_eval.is_some(),
        "each compared table carries its embedding eval report"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eval_embeddings_under_a_tenant_scope_over_the_wire() {
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};

    let server = start_engine_server().await;
    let session_iface = with_session("eval-tenant-a");

    // Bind the session to TENANT_A, then do all setup + the eval under the
    // same session id so the interceptor scopes every call.
    CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone())
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant");

    let scoped_catalog =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    let scoped_embedding =
        EmbeddingServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    let table = embed_patents_and_golden(scoped_catalog, scoped_embedding).await;

    let mut client = EvalServiceClient::with_interceptor(channel(server.addr).await, session_iface);
    let report = client
        .eval_embeddings(EvalEmbeddingsRequest {
            source_id: "patents".into(),
            embedding_table: table,
            golden_source: GOLDEN_SOURCE.into(),
            k: 10,
            cohorts: HashMap::new(),
            tenant_id: String::new(),
        })
        .await
        .expect("tenant-scoped eval_embeddings")
        .into_inner();

    assert!(!report.eval_run_id.is_empty());
    assert!(report.aggregate.is_some());

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
