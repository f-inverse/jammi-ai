//! esc-096: `[models]` -> [`HubSource`] — the shared Hugging Face Hub client
//! every jammi-ai call site now builds through, instead of each independently
//! reaching for `hf_hub::api::sync::Api::new()`/`ApiBuilder::from_env()`.
//!
//! RED (before this unit): `ModelResolver` ignored `[models]` entirely — it
//! built its own `Api` via `Api::new()`, which never read `HF_TOKEN` (hf-hub
//! 0.5 does not), applied `HF_HOME`/`HF_ENDPOINT` inconsistently across the
//! resolver and the fine-tune worker's separate `Api::new()` call, and
//! panicked outright (`Cache::default()`'s `dirs::home_dir().expect(..)`)
//! when `HOME` was unset. `ModelResolver::new` took no `HubSource` argument
//! at all, so `cargo build -p jammi-ai --features local` at the pre-fix
//! revision fails to compile this file with:
//!
//! ```text
//! error[E0061]: this function takes 3 arguments but 2 arguments were supplied
//!   --> crates/jammi-ai/tests/it/hub_source.rs
//!    |
//!    |     let resolver = ModelResolver::new(catalog, artifact_store).unwrap();
//!    |                     ^^^^^^^^^^^^^^^^^^ -------  -------------- argument #3 of type `HubSource` is missing
//! ```
//!
//! (the exact verbatim compiler text this crate's RED capture recorded is in
//! the commit history for this file — `ModelResolver::new` grew its third
//! `hub: HubSource` parameter in this same commit, so the type this file
//! imports, `jammi_ai::model::hub::HubSource`, did not exist at all at the
//! pre-fix revision either.)

use std::sync::Arc;

use jammi_ai::model::hub::HubSource;
use jammi_ai::model::resolver::ModelResolver;
use jammi_ai::model::{BackendType, ModelSource, ModelTask};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::config::{ModelsConfig, SecretSource};
use jammi_db::error::JammiError;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const REPO_ID: &str = "acme/tiny-hub-model";
const FILENAME: &str = "config.json";
const BODY: &[u8] = b"{\"hidden_size\":4,\"model_type\":\"bert\"}";

/// Mount ONE mock matching every `GET` to `{repo_id}/resolve/main/{filename}`
/// — hf-hub 0.5's sync API issues exactly two such requests per fresh
/// download (`Api::metadata`'s `Range: bytes=0-0` HEAD-shaped probe, then the
/// real `Range: bytes=0-` body fetch — see `hf_hub::api::sync::Api::metadata`/
/// `download_from`), both against the identical URL, so one mock serves
/// both. Every header hf-hub's `metadata()` requires is present: `etag`,
/// `x-repo-commit`, and `content-range` (whose `/`-suffix it parses as the
/// file size).
async fn mount_repo_file(server: &MockServer, repo_id: &str, filename: &str, body: &'static [u8]) {
    Mock::given(method("GET"))
        .and(path(format!("/{repo_id}/resolve/main/{filename}")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("etag", "\"esc096-etag\"")
                .insert_header("x-repo-commit", "esc096commit")
                .insert_header("content-range", format!("bytes 0-0/{}", body.len()))
                .set_body_bytes(body),
        )
        .mount(server)
        .await;
}

fn write_minimal_safetensors(path: &std::path::Path) {
    let header = b"{}";
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    std::fs::write(path, buf).unwrap();
}

// --- esc-096 core oracle: bearer token + cache-root precedence over a live GET ---

#[tokio::test(flavor = "multi_thread")]
async fn config_token_reaches_the_mock_and_file_lands_under_root_hub() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let root = tempfile::tempdir().unwrap();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: Some(SecretSource::Inline("tok".into())),
        offline: false,
    };
    let hub = HubSource::from_config(&config, &|_: &str| None).unwrap();

    let downloaded = tokio::task::spawn_blocking({
        let hub = hub.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();

    assert!(
        downloaded.starts_with(root.path().join("hub")),
        "expected the downloaded file to land under {{root}}/hub, got {downloaded:?}"
    );
    assert_eq!(std::fs::read(&downloaded).unwrap(), BODY);

    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty(), "the mock never received a request");
    for req in &requests {
        assert_eq!(
            req.headers
                .get("authorization")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer tok"),
            "every request against the mocked hub must carry the configured bearer token"
        );
    }
}

/// Control: `hub_token` unset, `HF_TOKEN` present in the passed env map ->
/// the bearer comes from the env fallback.
#[tokio::test(flavor = "multi_thread")]
async fn env_token_used_when_config_token_absent() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let root = tempfile::tempdir().unwrap();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: false,
    };
    let env = |k: &str| (k == "HF_TOKEN").then(|| "env-tok".to_string());
    let hub = HubSource::from_config(&config, &env).unwrap();

    tokio::task::spawn_blocking({
        let hub = hub.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty());
    for req in &requests {
        assert_eq!(
            req.headers
                .get("authorization")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer env-tok")
        );
    }
}

/// Control: neither `hub_token` nor `HF_TOKEN` (nor a cache `token` file,
/// since `root` is a fresh tempdir) -> no `Authorization` header at all.
#[tokio::test(flavor = "multi_thread")]
async fn no_token_no_authorization_header() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let root = tempfile::tempdir().unwrap();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: false,
    };
    let hub = HubSource::from_config(&config, &|_: &str| None).unwrap();

    tokio::task::spawn_blocking({
        let hub = hub.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty());
    for req in &requests {
        assert!(
            req.headers.get("authorization").is_none(),
            "expected no Authorization header, got {:?}",
            req.headers.get("authorization")
        );
    }
}

// --- offline: hit / miss / warm-cache-miss, decided by the catalog alone ---

/// Hit: a `HuggingFace`-shaped model id whose catalog row already carries a
/// present, on-disk `artifact_path` (as if resolved online previously) keeps
/// loading under `offline = true` — the catalog lookup in `ModelResolver::resolve`
/// returns before the offline check is ever reached.
#[tokio::test]
async fn offline_hit_serves_from_catalog_without_hub_access() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("hit_model");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();
    write_minimal_safetensors(&model_dir.join("model.safetensors"));

    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    catalog
        .register_model(RegisterModelParams {
            model_id: "acme/hit-repo",
            version: 1,
            model_type: "huggingface",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: Some(model_dir.to_str().unwrap()),
            config_json: None,
        })
        .await
        .unwrap();

    let offline_hub = HubSource::from_config(
        &ModelsConfig {
            offline: true,
            ..offline_test_models_config()
        },
        &|_: &str| None,
    )
    .unwrap();
    let resolver = ModelResolver::new(
        Arc::clone(&catalog),
        crate::common::test_artifact_store(),
        offline_hub,
    )
    .unwrap();

    let resolved = resolver
        .resolve(
            &ModelSource::hf("acme/hit-repo"),
            ModelTask::TextEmbedding,
            Some(BackendType::Candle),
        )
        .await
        .unwrap();
    assert_eq!(resolved.model_id.0, "acme/hit-repo");
}

/// Miss: no catalog row at all for this repo id -> `offline = true` refuses
/// by name, never attempting a Hub network call.
#[tokio::test]
async fn offline_miss_refuses_by_name_with_no_catalog_row() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let offline_hub = HubSource::from_config(
        &ModelsConfig {
            offline: true,
            ..offline_test_models_config()
        },
        &|_: &str| None,
    )
    .unwrap();
    let resolver =
        ModelResolver::new(catalog, crate::common::test_artifact_store(), offline_hub).unwrap();

    let err = match resolver
        .resolve(
            &ModelSource::hf("acme/never-resolved"),
            ModelTask::TextEmbedding,
            None,
        )
        .await
    {
        Ok(_) => panic!("expected an offline refusal, got a resolved model"),
        Err(e) => e,
    };
    match err {
        JammiError::Model { model_id, message } => {
            assert_eq!(model_id, "acme/never-resolved");
            assert!(
                message.contains("offline") && message.contains("acme/never-resolved"),
                "expected the offline refusal to name the repo id, got: {message}"
            );
        }
        other => panic!("expected JammiError::Model, got {other:?}"),
    }
}

/// Warm-cache-miss: the Hub cache directory holds real, previously-downloaded
/// bytes for this exact repo (warmed through a genuine, non-offline
/// `HubSource` against the mock below) but the catalog carries no row for
/// it. `offline = true` still refuses — the on-disk cache is never consulted
/// as a substitute source of truth for "was this resolved online".
#[tokio::test(flavor = "multi_thread")]
async fn offline_warm_cache_without_catalog_row_still_refuses() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let root = tempfile::tempdir().unwrap();
    let online_config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: false,
    };
    let online_hub = HubSource::from_config(&online_config, &|_: &str| None).unwrap();
    tokio::task::spawn_blocking({
        let hub = online_hub.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();

    let offline_config = ModelsConfig {
        offline: true,
        ..online_config
    };
    let offline_hub = HubSource::from_config(&offline_config, &|_: &str| None).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver =
        ModelResolver::new(catalog, crate::common::test_artifact_store(), offline_hub).unwrap();

    let err = match resolver
        .resolve(&ModelSource::hf(REPO_ID), ModelTask::TextEmbedding, None)
        .await
    {
        Ok(_) => panic!("expected an offline refusal, got a resolved model"),
        Err(e) => e,
    };
    match err {
        JammiError::Model { message, .. } => assert!(
            message.contains("offline"),
            "expected the offline refusal, got: {message}"
        ),
        other => panic!("expected JammiError::Model, got {other:?}"),
    }
}

/// A hermetic `[models]` baseline for the offline tests above — a fresh
/// tempdir cache root so none of them can accidentally touch a real
/// developer/CI-runner Hub cache.
fn offline_test_models_config() -> ModelsConfig {
    ModelsConfig {
        hub_cache_dir: Some(tempfile::tempdir().unwrap().keep()),
        ..Default::default()
    }
}
