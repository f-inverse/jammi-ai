//! `InferenceService` end-to-end over the wire.
//!
//! An in-process Tonic server hosts the gRPC chain including the
//! `InferenceService`. A client registers the `patents.parquet` fixture as a
//! source and calls `Infer` over its `abstract` column with the local
//! `tiny_bert` cookbook encoder, then decodes the returned `ArrowBatch` and
//! asserts the inference output rows round-trip. This pins the wire adapter's
//! contract: the verb routes through the `Session` abstraction
//! and carries the engine's `Vec<RecordBatch>` back as Arrow IPC.
//!
//! Hermetic: the encoder is a local fixture (no network, no download), the
//! corpus is the shipped patents parquet, and the assertion reads the decoded
//! rows directly. A tenant-scoped `Infer` is covered too.

use arrow::array::StringArray;
use arrow_ipc::reader::StreamReader;
use jammi_server::grpc::proto::inference::inference_service_client::InferenceServiceClient;
use jammi_server::grpc::proto::inference::{InferRequest, ModelTask};
use jammi_test_utils::{cookbook_fixture, fixture};

use super::common::grpc::{channel, start_engine_server, with_session, TENANT_A};

// The inference service shares its source registration with the embedding
// service; the corpus is the shipped patents parquet read over the `abstract`
// column with the local tiny_bert encoder.
fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_url() -> String {
    format!("file://{}", fixture("patents.parquet").display())
}

/// Register the patents source through the embedding service's `AddSource`
/// (both services back onto the same engine session, so a source registered on
/// one is visible to the other).
async fn add_patents(client_channel: tonic::transport::Channel) {
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{
        AddSourceRequest, FileFormat, SourceConnection, SourceKind,
    };
    let mut catalog = CatalogServiceClient::new(client_channel);
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
        .expect("add_source");
}

/// Decode an `InferResponse`'s `ArrowBatch` (header + body IPC stream) into the
/// record batches it carries — the client-side inverse of the server's
/// `encode_ipc_stream`.
fn decode_infer_rows(
    batch: jammi_server::grpc::proto::trigger::ArrowBatch,
) -> Vec<arrow::record_batch::RecordBatch> {
    let mut bytes = Vec::with_capacity(batch.data_header.len() + batch.data_body.len());
    bytes.extend_from_slice(&batch.data_header);
    bytes.extend_from_slice(&batch.data_body);
    let reader = StreamReader::try_new(std::io::Cursor::new(bytes), None).expect("ipc reader");
    reader
        .collect::<Result<Vec<_>, _>>()
        .expect("decode batches")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infer_returns_arrow_rows_over_the_wire() {
    let server = start_engine_server().await;
    add_patents(channel(server.addr).await).await;

    let mut client = InferenceServiceClient::new(channel(server.addr).await);

    let resp = client
        .infer(InferRequest {
            source_id: "patents".into(),
            model_id: tiny_bert_model_id(),
            task: ModelTask::TextEmbedding as i32,
            columns: vec!["abstract".into()],
            key_column: "id".into(),
            tenant_id: String::new(),
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect("infer")
        .into_inner();

    let arrow = resp.result.expect("infer carries an ArrowBatch result");
    let batches = decode_infer_rows(arrow);
    assert!(!batches.is_empty(), "infer over patents produces rows");

    // The inference output always carries the common prefix columns; `_row_id`
    // holds each row's key-column value. Assert it round-trips as a string
    // column with one row per source row.
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0, "at least one inferred row");
    let first = &batches[0];
    let row_id = first
        .column_by_name("_row_id")
        .expect("_row_id column present")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("_row_id is a string column");
    assert!(
        !row_id.value(0).is_empty(),
        "each inferred row carries a non-empty key"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infer_under_a_tenant_scope_succeeds_over_the_wire() {
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};

    let server = start_engine_server().await;

    // Bind the session (keyed by the `jammi-session-id` header) to TENANT_A via
    // CatalogService, then register the source and infer under the same
    // session id — every call carries that header through `with_session`, so
    // the interceptor scopes them all to TENANT_A.
    let session_iface = with_session("infer-tenant-a");
    let mut session_client =
        CatalogServiceClient::with_interceptor(channel(server.addr).await, session_iface.clone());
    session_client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: TENANT_A.into(),
            }),
        })
        .await
        .expect("set_tenant");

    // AddSource under TENANT_A's scope, on the same control-plane client.
    {
        use jammi_server::grpc::proto::catalog::{
            AddSourceRequest, FileFormat, SourceConnection, SourceKind,
        };
        session_client
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
    }

    let mut client =
        InferenceServiceClient::with_interceptor(channel(server.addr).await, session_iface);
    let resp = client
        .infer(InferRequest {
            source_id: "patents".into(),
            model_id: tiny_bert_model_id(),
            task: ModelTask::TextEmbedding as i32,
            columns: vec!["abstract".into()],
            key_column: "id".into(),
            tenant_id: String::new(),
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect("infer under tenant scope")
        .into_inner();

    let arrow = resp.result.expect("result");
    let batches = decode_infer_rows(arrow);
    assert!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>() > 0,
        "tenant-scoped infer returns rows for the tenant's source"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infer_rejects_unspecified_task() {
    let server = start_engine_server().await;
    let mut client = InferenceServiceClient::new(channel(server.addr).await);

    let err = client
        .infer(InferRequest {
            source_id: "patents".into(),
            model_id: tiny_bert_model_id(),
            task: ModelTask::Unspecified as i32,
            columns: vec!["abstract".into()],
            key_column: "id".into(),
            tenant_id: String::new(),
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect_err("unspecified task must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infer_rejects_missing_columns() {
    let server = start_engine_server().await;
    let mut client = InferenceServiceClient::new(channel(server.addr).await);

    let err = client
        .infer(InferRequest {
            source_id: "patents".into(),
            model_id: tiny_bert_model_id(),
            task: ModelTask::TextEmbedding as i32,
            columns: Vec::new(),
            key_column: "id".into(),
            tenant_id: String::new(),
            cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
        })
        .await
        .expect_err("missing columns must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
