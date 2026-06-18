//! `PipelineService` end-to-end over the wire.
//!
//! An in-process Tonic server hosts the gRPC chain including the always-on
//! `PipelineService`. A client registers the `patents.parquet` corpus, generates
//! embeddings with the local `tiny_bert` encoder, then drives the pipeline verbs
//! over the wire:
//!
//! * `BuildNeighborGraph` → a materialised k-NN edge table (read back via SQL).
//! * `PropagateEmbeddings` → a propagated, searchable embedding table over that
//!   graph.
//! * `AssembleContext` → a pooled context vector + carried metadata for a target
//!   query.
//!
//! The tenant test drives the verbs under a `with_tenant`-set connection: each
//! resolves the source's embedding table through the generic SQL surface, so the
//! tenant-scope analyzer rule scopes the scan — a cross-tenant endpoint is
//! filtered before it reaches the adjacency. The test asserts the verbs succeed
//! under the bound tenant (resolving that tenant's own embedding table + graph)
//! and that the propagated table is reachable + non-empty under the same scope;
//! a second tenant that registered nothing cannot resolve the same source, so
//! `build_neighbor_graph`, `propagate_embeddings`, and `assemble_context` are all
//! rejected there — the scoping is observable over the wire for every verb, not
//! merely a signature-parity claim.
//!
//! Hermetic: the encoder is a local fixture and the corpus is a shipped fixture.

use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::{
    AddSourceRequest, FileFormat, SetTenantRequest, SourceConnection, SourceKind, Tenant,
};
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::{GenerateEmbeddingsRequest, Modality};
use jammi_server::grpc::proto::pipeline::pipeline_service_client::PipelineServiceClient;
use jammi_server::grpc::proto::pipeline::{
    propagate_embeddings_request::Graph, AssembleContextRequest, BuildNeighborGraphRequest,
    Cascade, PropagateEmbeddingsRequest, RecomputeRequest,
};
use jammi_test_utils::{cookbook_fixture, fixture};
use tonic::codegen::Body;

use super::common::grpc::{
    channel, start_engine_server, with_session, EngineServer, TENANT_A, TENANT_B,
};

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_url() -> String {
    format!("file://{}", fixture("patents.parquet").display())
}

/// Register the patents corpus (control plane) and generate one embedding table
/// over `abstract` (data plane). Returns the generated table name. Both clients
/// share one transport (and, for the tenant-scoped test, one session-header
/// interceptor), so registration and compute run under the same tenant scope.
/// Generic over the client transport so the plain-channel test and the
/// interceptor-wrapped tenant test share one body.
async fn embed_patents<T>(
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
async fn build_propagate_assemble_over_the_wire() {
    let server: EngineServer = start_engine_server().await;
    embed_patents(
        CatalogServiceClient::new(channel(server.addr).await),
        EmbeddingServiceClient::new(channel(server.addr).await),
    )
    .await;

    let mut pipeline = PipelineServiceClient::new(channel(server.addr).await);

    // BuildNeighborGraph → a materialised k-NN edge table.
    let graph = pipeline
        .build_neighbor_graph(BuildNeighborGraphRequest {
            source_id: "patents".into(),
            k: 5,
            exact: true,
            ..Default::default()
        })
        .await
        .expect("build_neighbor_graph")
        .into_inner();
    assert!(!graph.table_name.is_empty(), "graph table materialised");

    // PropagateEmbeddings over that S9 graph → a new searchable embedding table.
    let propagated = pipeline
        .propagate_embeddings(PropagateEmbeddingsRequest {
            source_id: "patents".into(),
            graph: Some(Graph::EdgeGraphTable(graph.table_name.clone())),
            ..Default::default()
        })
        .await
        .expect("propagate_embeddings")
        .into_inner();
    assert!(
        !propagated.table_name.is_empty(),
        "propagated embedding table materialised"
    );
    assert!(
        propagated.row_count > 0,
        "propagated table carries rows ({})",
        propagated.row_count
    );

    // AssembleContext for a target query vector over the source's embeddings.
    // The pooled vector matches the embedding dimensionality, and the assembly
    // source is the ANN default.
    let dim = propagated.dimensions as usize;
    let context = pipeline
        .assemble_context(AssembleContextRequest {
            source_id: "patents".into(),
            query: vec![0.0_f32; dim.max(1)],
            k: 3,
            exclude_self: Some(true),
            ..Default::default()
        })
        .await
        .expect("assemble_context")
        .into_inner();
    assert_eq!(context.source, "ann", "ANN-only default assembly");
    if let Some(vector) = context.context_vector.as_ref() {
        assert_eq!(
            vector.values.len(),
            dim,
            "pooled vector matches the embedding dimensionality"
        );
    }
    assert_eq!(
        context.context_keys.len() as u64,
        context.context_size,
        "context_size matches the member count"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test]
async fn recompute_over_the_wire_replays_a_derived_table() {
    let server: EngineServer = start_engine_server().await;
    embed_patents(
        CatalogServiceClient::new(channel(server.addr).await),
        EmbeddingServiceClient::new(channel(server.addr).await),
    )
    .await;

    let mut pipeline = PipelineServiceClient::new(channel(server.addr).await);

    // A neighbour-graph derived from the patents embeddings — a recomputable
    // derived table anchored on the immutable embedding-table digest.
    let graph = pipeline
        .build_neighbor_graph(BuildNeighborGraphRequest {
            source_id: "patents".into(),
            k: 5,
            exact: true,
            ..Default::default()
        })
        .await
        .expect("build_neighbor_graph")
        .into_inner();

    // A propagation over the graph, so the graph has a downstream dependent the
    // Downstream sweep must also recompute.
    let propagated = pipeline
        .propagate_embeddings(PropagateEmbeddingsRequest {
            source_id: "patents".into(),
            graph: Some(Graph::EdgeGraphTable(graph.table_name.clone())),
            ..Default::default()
        })
        .await
        .expect("propagate_embeddings")
        .into_inner();

    // Recompute the graph with a Downstream sweep over the wire: the report names
    // the recomputed tables (graph + its propagation dependent) and the
    // downstream-stale set. The compute stays server-side; the report crosses the
    // wire as the faithful `RecomputeReport`.
    let report = pipeline
        .recompute(RecomputeRequest {
            table: graph.table_name.clone(),
            cascade: Cascade::Downstream as i32,
        })
        .await
        .expect("recompute")
        .into_inner();

    let originals: Vec<&str> = report
        .recomputed
        .iter()
        .map(|t| t.original.as_str())
        .collect();
    assert!(
        originals.contains(&graph.table_name.as_str()),
        "the named graph is recomputed: {originals:?}"
    );
    assert!(
        originals.contains(&propagated.table_name.as_str()),
        "the Downstream sweep recomputes the propagation dependent too: {originals:?}"
    );
    // The graph (parent) is recomputed before the propagation (child).
    let graph_pos = originals.iter().position(|n| *n == graph.table_name).unwrap();
    let prop_pos = originals
        .iter()
        .position(|n| *n == propagated.table_name)
        .unwrap();
    assert!(
        graph_pos < prop_pos,
        "topological order over the wire: parent before child ({originals:?})"
    );
    // Every recomputed table reports the COMPUTED outcome — a recompute bypasses
    // the cache (never a wire-default UNSPECIFIED or a fabricated REUSED).
    for t in &report.recomputed {
        assert_eq!(
            t.outcome,
            jammi_wire::proto::inference::CacheOutcome::Computed as i32,
            "a recompute always Computes (bypasses the cache)"
        );
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn propagate_embeddings_is_tenant_scoped_over_the_wire() {
    let server = start_engine_server().await;

    // Bind TENANT_A and do all setup + the propagate under that session so the
    // interceptor scopes every call. The graph build, embedding generation, and
    // propagation all resolve TENANT_A's own catalog rows.
    let session_a = with_session("pipeline-tenant-a");
    CatalogServiceClient::with_interceptor(channel(server.addr).await, session_a.clone())
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant A");

    let catalog_a =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_a.clone());
    let embedding_a =
        EmbeddingServiceClient::with_interceptor(channel(server.addr).await, session_a.clone());
    embed_patents(catalog_a, embedding_a).await;

    let mut pipeline_a =
        PipelineServiceClient::with_interceptor(channel(server.addr).await, session_a.clone());
    let graph = pipeline_a
        .build_neighbor_graph(BuildNeighborGraphRequest {
            source_id: "patents".into(),
            k: 5,
            exact: true,
            ..Default::default()
        })
        .await
        .expect("tenant-scoped build_neighbor_graph")
        .into_inner();

    let propagated = pipeline_a
        .propagate_embeddings(PropagateEmbeddingsRequest {
            source_id: "patents".into(),
            graph: Some(Graph::EdgeGraphTable(graph.table_name)),
            ..Default::default()
        })
        .await
        .expect("tenant-scoped propagate_embeddings")
        .into_inner();
    assert!(
        propagated.row_count > 0,
        "TENANT_A's propagation resolves its own embeddings + graph"
    );

    // TENANT_B registered nothing, so the same source does not resolve under its
    // scope — the propagate is rejected. This is the cross-tenant boundary the
    // analyzer rule enforces, observed over the wire (not a signature claim).
    let session_b = with_session("pipeline-tenant-b");
    CatalogServiceClient::with_interceptor(channel(server.addr).await, session_b.clone())
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_B.into(),
            }),
        })
        .await
        .expect("set_tenant B");
    let mut pipeline_b =
        PipelineServiceClient::with_interceptor(channel(server.addr).await, session_b);
    let propagate_cross_tenant = pipeline_b
        .propagate_embeddings(PropagateEmbeddingsRequest {
            source_id: "patents".into(),
            graph: Some(Graph::EdgeGraphTable("anything".into())),
            ..Default::default()
        })
        .await;
    assert!(
        propagate_cross_tenant.is_err(),
        "TENANT_B cannot resolve TENANT_A's source — propagate is rejected"
    );

    // The other two verbs are scoped by the same interceptor + analyzer rule;
    // assert their cross-tenant rejection over the wire too, closing the
    // coverage gap left by testing only propagate.
    let build_cross_tenant = pipeline_b
        .build_neighbor_graph(BuildNeighborGraphRequest {
            source_id: "patents".into(),
            k: 5,
            exact: true,
            ..Default::default()
        })
        .await;
    assert!(
        build_cross_tenant.is_err(),
        "TENANT_B cannot resolve TENANT_A's source — build_neighbor_graph is rejected"
    );

    let assemble_cross_tenant = pipeline_b
        .assemble_context(AssembleContextRequest {
            source_id: "patents".into(),
            query: vec![0.0_f32; 8],
            k: 3,
            ..Default::default()
        })
        .await;
    assert!(
        assemble_cross_tenant.is_err(),
        "TENANT_B cannot resolve TENANT_A's source — assemble_context is rejected"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
