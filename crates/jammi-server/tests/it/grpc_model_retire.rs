//! `RetireModel` over the catalog gRPC service.
//!
//! Proves the retire verb crosses the wire with the tenant header: a tenant
//! retires its own model and the model then drops out of the remote
//! `list_models` listing, while a second tenant retiring the same model id is
//! rejected NotFound (the strict per-tenant retire scope holds across the wire).

use jammi_admin::CatalogClient;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::model_task::ModelTask;
use tonic::transport::Endpoint;

use super::common::grpc::{start_engine_server, tenant_a, tenant_b, EngineServer};

async fn remote(server: &EngineServer) -> CatalogClient {
    let endpoint = Endpoint::from_shared(format!("http://{}", server.addr)).expect("endpoint");
    CatalogClient::connect(endpoint)
        .await
        .expect("catalog client connect")
}

/// Register a tenant-A-scoped model directly on the shared engine catalog, so
/// the wire-side retire has a real, tenant-owned row to act on.
async fn register_tenant_a_model(server: &EngineServer, model_id: &str) {
    server
        .engine
        .catalog()
        .pinned_to_tenant(Some(tenant_a()))
        .register_model(RegisterModelParams {
            model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .expect("register_model");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_retire_model_hides_from_listing() {
    let server = start_engine_server().await;
    register_tenant_a_model(&server, "acme/embed-mini").await;

    let client = remote(&server).await;
    client.bind_tenant(tenant_a()).await.expect("bind tenant A");

    // Present before retire.
    assert!(
        client
            .list_models()
            .await
            .expect("list before")
            .iter()
            .any(|m| m.model_id == "acme/embed-mini"),
        "the registered model must be in the remote listing before retire"
    );

    client
        .retire_model("acme/embed-mini", None)
        .await
        .expect("remote retire over the wire");

    // Gone from the listing after retire.
    assert!(
        !client
            .list_models()
            .await
            .expect("list after")
            .iter()
            .any(|m| m.model_id == "acme/embed-mini"),
        "the retired model must be hidden from the remote listing"
    );

    server.shutdown.send(()).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_second_tenant_cannot_retire() {
    let server = start_engine_server().await;
    register_tenant_a_model(&server, "acme/embed-mini").await;

    let client = remote(&server).await;
    client.bind_tenant(tenant_b()).await.expect("bind tenant B");

    let err = client
        .retire_model("acme/embed-mini", None)
        .await
        .expect_err("tenant B must not retire tenant A's model over the wire");
    // The "no such model for this tenant" fault maps to NotFound at the wire.
    assert!(
        format!("{err:?}").to_lowercase().contains("not found")
            || format!("{err}").to_lowercase().contains("not found"),
        "cross-tenant retire must surface NotFound, got: {err:?}"
    );

    server.shutdown.send(()).ok();
}
