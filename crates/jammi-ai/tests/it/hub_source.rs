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

/// A minimal, valid safetensors file: an 8-byte little-endian header length
/// (`2`, for the two bytes that follow) then the empty JSON header `{}` — no
/// tensors. `estimate_safetensors_residency` accepts this shape; it is the
/// exact byte layout `write_minimal_safetensors` below writes to disk for
/// the catalog-hit tests, reused here as a mock response body so a full
/// `ModelResolver::resolve_hf_hub` can reach `Ok` over the wire.
const MINIMAL_SAFETENSORS: &[u8] = &[2, 0, 0, 0, 0, 0, 0, 0, b'{', b'}'];

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
        offline: Some(false),
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

/// #481 acceptance bullet 2(a): drive `HF_HOME` through the injected `env`
/// closure (never `std::env::set_var`, which would leak across the whole
/// process/test binary) with `[models] hub_cache_dir` left `None`, so
/// `resolve_root` falls through to the `HF_HOME` env tier. The downloaded
/// file must land under `{HF_HOME}/hub/…` — the `hub/` subdirectory
/// `HubSource::from_config` appends before handing the root to
/// `hf_hub::Cache::new`.
#[tokio::test(flavor = "multi_thread")]
async fn hf_home_env_drives_the_cache_root_end_to_end() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let hf_home = tempfile::tempdir().unwrap();
    let hf_home_path = hf_home.path().to_path_buf();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: None,
        hub_token: None,
        offline: None,
    };
    let env = move |k: &str| (k == "HF_HOME").then(|| hf_home_path.to_str().unwrap().to_string());
    let hub = HubSource::from_config(&config, &env).unwrap();

    let downloaded = tokio::task::spawn_blocking({
        let hub = hub.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();

    assert!(
        downloaded.starts_with(hf_home.path().join("hub")),
        "expected the downloaded file to land under {{HF_HOME}}/hub, got {downloaded:?}"
    );
    assert_eq!(std::fs::read(&downloaded).unwrap(), BODY);
}

/// #481 acceptance bullet 2(b), the warm-cache oracle: a SECOND `HubSource`
/// built over the exact same `hub_cache_dir` (a fresh process reusing a
/// mounted cache volume after a restart, standing in for the real scenario)
/// must serve the same file ENTIRELY from the warm on-disk cache — zero new
/// requests against the mock server. Proves a restart with a mounted volume
/// never re-downloads, not merely that the file "exists somewhere".
#[tokio::test(flavor = "multi_thread")]
async fn warm_cache_across_a_second_hub_source_issues_no_requests() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let root = tempfile::tempdir().unwrap();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: None,
    };

    // Cold cache: the first HubSource genuinely downloads.
    let first = HubSource::from_config(&config, &|_: &str| None).unwrap();
    tokio::task::spawn_blocking({
        let hub = first.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();
    let requests_after_first = server.received_requests().await.unwrap().len();
    assert!(
        requests_after_first > 0,
        "the first, cold-cache fetch must reach the mock at least once"
    );

    // Warm cache: a second, independently-constructed HubSource over the
    // SAME hub_cache_dir — models a process restart with the cache directory
    // mounted from a persistent volume.
    let second = HubSource::from_config(&config, &|_: &str| None).unwrap();
    let downloaded_again = tokio::task::spawn_blocking({
        let hub = second.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();
    assert_eq!(std::fs::read(&downloaded_again).unwrap(), BODY);

    let requests_after_second = server.received_requests().await.unwrap().len();
    assert_eq!(
        requests_after_second, requests_after_first,
        "a second HubSource over the same hub_cache_dir must issue ZERO new requests -- \
         a restart with a mounted cache volume must never re-download (got {requests_after_second} \
         total requests, expected exactly {requests_after_first})"
    );
}

/// #481 acceptance bullet 3, first control: `HF_HUB_OFFLINE=1` in the
/// injected env with NO `[models] offline` override at all — the SAME
/// "offline refusal by name" as `offline_miss_refuses_by_name_with_no_catalog_row`
/// above, but driven entirely from the environment fallback.
#[tokio::test]
async fn hf_hub_offline_env_refuses_by_name_with_no_config_override() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let env = |k: &str| (k == "HF_HUB_OFFLINE").then(|| "1".to_string());
    let offline_hub = HubSource::from_config(&offline_test_models_config(), &env).unwrap();
    assert!(
        offline_hub.offline(),
        "HF_HUB_OFFLINE=1 must be honoured when [models] offline is unset"
    );
    let resolver =
        ModelResolver::new(catalog, crate::common::test_artifact_store(), offline_hub).unwrap();

    let err = match resolver
        .resolve(
            &ModelSource::hf("acme/env-only-offline"),
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
            assert_eq!(model_id, "acme/env-only-offline");
            assert!(
                message.contains("offline") && message.contains("acme/env-only-offline"),
                "expected the offline refusal to name the repo id, got: {message}"
            );
        }
        other => panic!("expected JammiError::Model, got {other:?}"),
    }
}

/// #481 fix round 2, BLOCK reproducer: `HF_HUB_OFFLINE=ON` — a truthy value
/// per `huggingface_hub`'s own `ENV_VARS_TRUE_VALUES = {"1","ON","YES","TRUE"}`
/// (`huggingface_hub/constants.py:12`, `_is_true` at `:16-19`) — must refuse
/// offline by name, exactly like `HF_HUB_OFFLINE=1` above. Before this fix,
/// `is_hf_hub_offline_truthy` accepted only `"1"` and a case-insensitive
/// `"true"`, so `HF_HUB_OFFLINE=ON` silently resolved `offline=false` and the
/// resolver proceeded to a live fetch — fail-open on the air-gap knob.
///
/// RED at 6519633d (this test's own first assertion, `offline_hub.offline()`,
/// was the failure -- resolution never even reached the resolver):
/// ```text
/// thread '...::hf_hub_offline_on_refuses_by_name_with_no_config_override' panicked at crates/jammi-ai/tests/it/hub_source.rs:...:
/// HF_HUB_OFFLINE=ON must be honoured when [models] offline is unset -- widen is_hf_hub_offline_truthy to accept huggingface_hub's ENV_VARS_TRUE_VALUES
/// ```
#[tokio::test]
async fn hf_hub_offline_on_refuses_by_name_with_no_config_override() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let env = |k: &str| (k == "HF_HUB_OFFLINE").then(|| "ON".to_string());
    let offline_hub = HubSource::from_config(&offline_test_models_config(), &env).unwrap();
    assert!(
        offline_hub.offline(),
        "HF_HUB_OFFLINE=ON must be honoured when [models] offline is unset -- widen \
         is_hf_hub_offline_truthy to accept huggingface_hub's ENV_VARS_TRUE_VALUES"
    );
    let resolver =
        ModelResolver::new(catalog, crate::common::test_artifact_store(), offline_hub).unwrap();

    let err = match resolver
        .resolve(
            &ModelSource::hf("acme/env-only-offline-on"),
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
            assert_eq!(model_id, "acme/env-only-offline-on");
            assert!(
                message.contains("offline") && message.contains("acme/env-only-offline-on"),
                "expected the offline refusal to name the repo id, got: {message}"
            );
        }
        other => panic!("expected JammiError::Model, got {other:?}"),
    }
}

/// #481 fix round 2: `TRANSFORMERS_OFFLINE=1`, with `HF_HUB_OFFLINE` and
/// `[models] offline` both unset, must be honoured as the same fallback
/// `huggingface_hub` itself applies for this variable.
#[tokio::test]
async fn transformers_offline_env_refuses_by_name_with_no_config_override() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let env = |k: &str| (k == "TRANSFORMERS_OFFLINE").then(|| "1".to_string());
    let offline_hub = HubSource::from_config(&offline_test_models_config(), &env).unwrap();
    assert!(
        offline_hub.offline(),
        "TRANSFORMERS_OFFLINE=1 must be honoured when HF_HUB_OFFLINE and [models] offline \
         are both unset"
    );
    let resolver =
        ModelResolver::new(catalog, crate::common::test_artifact_store(), offline_hub).unwrap();

    let err = match resolver
        .resolve(
            &ModelSource::hf("acme/transformers-offline-only"),
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
            assert_eq!(model_id, "acme/transformers-offline-only");
            assert!(
                message.contains("offline") && message.contains("acme/transformers-offline-only"),
                "expected the offline refusal to name the repo id, got: {message}"
            );
        }
        other => panic!("expected JammiError::Model, got {other:?}"),
    }
}

/// #481 fix round 2, advisory A4: a fetch driven entirely by `HF_HUB_CACHE`
/// (no `[models] hub_cache_dir`, no `HF_HOME`) must land the downloaded file
/// directly under that directory (`models--…` immediately inside it) — no
/// `hub/` subdirectory appended, matching `huggingface_hub`'s own
/// `HF_HUB_CACHE` convention.
#[tokio::test(flavor = "multi_thread")]
async fn hf_hub_cache_env_drives_the_cache_root_directly_no_hub_subdir_appended() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let hf_hub_cache = tempfile::tempdir().unwrap();
    let hf_hub_cache_path = hf_hub_cache.path().to_path_buf();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: None,
        hub_token: None,
        offline: None,
    };
    let env = move |k: &str| {
        (k == "HF_HUB_CACHE").then(|| hf_hub_cache_path.to_str().unwrap().to_string())
    };
    let hub = HubSource::from_config(&config, &env).unwrap();

    let downloaded = tokio::task::spawn_blocking({
        let hub = hub.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();

    let expected_repo_dir = hf_hub_cache
        .path()
        .join(format!("models--{}", REPO_ID.replace('/', "--")));
    assert!(
        downloaded.starts_with(&expected_repo_dir),
        "expected the downloaded file to land directly under {{HF_HUB_CACHE}}/models--… (no \
         hub/ subdirectory), got {downloaded:?}, expected under {expected_repo_dir:?}"
    );
    assert!(
        !downloaded.starts_with(hf_hub_cache.path().join("hub")),
        "HF_HUB_CACHE must NOT get a hub/ subdirectory appended -- got {downloaded:?}"
    );
    assert_eq!(std::fs::read(&downloaded).unwrap(), BODY);
}

/// #481 acceptance bullet 3, second control (the direction): `[models]
/// offline = false` EXPLICIT wins over `HF_HUB_OFFLINE=1` in the environment
/// — config wins, matching `resolve_root`/`resolve_endpoint`/`resolve_token`'s
/// own "config beats env" precedence.
///
/// Proven by an OBSERVED completed network fetch, not by an error's shape.
/// An acceptance-verifier BLOCK on the prior version of this test found that
/// oracle vacuous: at the true pre-esc-096 base, `HF_HUB_OFFLINE` was never
/// read at all, so "the error message doesn't say offline" passed for the
/// wrong reason — it cannot distinguish "config correctly overrode the env"
/// from "the env is simply unimplemented and this genuinely 404s for an
/// unrelated reason". This version mounts the repo's `config.json` and
/// `model.safetensors` on the wiremock server exactly like the passing
/// fetch tests above (`mount_repo_file`, the file's shared mounting
/// helper), so a real config-wins outcome is observable two ways that a
/// vacuous "env unimplemented" run cannot fake: (i) `resolve` returns `Ok`
/// (an unmounted 404 or an actual offline refusal both return `Err`), and
/// (ii) the mock's `received_requests` is non-empty (the resolve did not
/// short-circuit before ever reaching the network — the offline refusal
/// path in `ModelResolver::resolve` returns before building the Hub repo
/// client at all). The `!hub.offline()` unit-level assertion stays, so this
/// test still pins the `HubSource`-level precedence directly, not only its
/// downstream effect.
#[tokio::test(flavor = "multi_thread")]
async fn config_offline_false_wins_over_hf_hub_offline_env() {
    let server = MockServer::start().await;
    let repo_id = "acme/config-wins-repo";
    mount_repo_file(&server, repo_id, "config.json", BODY).await;
    mount_repo_file(&server, repo_id, "model.safetensors", MINIMAL_SAFETENSORS).await;

    let root = tempfile::tempdir().unwrap();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: Some(false),
    };
    let env = |k: &str| (k == "HF_HUB_OFFLINE").then(|| "1".to_string());
    let hub = HubSource::from_config(&config, &env).unwrap();
    assert!(
        !hub.offline(),
        "explicit `offline = false` must win over HF_HUB_OFFLINE=1"
    );

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store(), hub).unwrap();

    let resolved = resolver
        .resolve(&ModelSource::hf(repo_id), ModelTask::TextEmbedding, None)
        .await
        .expect(
            "config `offline = false` must win over HF_HUB_OFFLINE=1 -- expected the \
             resolve to reach the mocked Hub and succeed, not refuse offline",
        );
    assert_eq!(resolved.model_id.0, repo_id);

    let requests = server.received_requests().await.unwrap();
    assert!(
        !requests.is_empty(),
        "config `offline = false` must win over HF_HUB_OFFLINE=1 -- the mock never \
         received a request, so this run cannot distinguish \"config correctly \
         overrode the env\" from \"HF_HUB_OFFLINE is simply unimplemented\""
    );
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
        offline: Some(false),
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
        offline: Some(false),
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
            offline: Some(true),
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
            offline: Some(true),
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
        offline: Some(false),
    };
    let online_hub = HubSource::from_config(&online_config, &|_: &str| None).unwrap();
    tokio::task::spawn_blocking({
        let hub = online_hub.clone();
        move || hub.api().model(REPO_ID.to_string()).get(FILENAME).unwrap()
    })
    .await
    .unwrap();

    let offline_config = ModelsConfig {
        offline: Some(true),
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
