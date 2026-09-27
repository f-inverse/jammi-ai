//! `[models]` -> [`HubSource`] — the shared Hugging Face Hub client every
//! jammi-ai call site builds through: every precedence tier (cache root,
//! endpoint, token, offline) reaches the wire through one `HubSource`, a warm
//! cache issues no request, and a transfer that stops sending fails typed
//! and bounded instead of waiting forever.

use std::sync::Arc;

use jammi_ai::model::hub::HubSource;
use jammi_ai::model::resolver::ModelResolver;
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;
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
/// — `HubRepo::get` issues exactly two such requests per fresh download (the
/// `Range: bytes=0-0` metadata probe, then the body fetch), both against the
/// identical URL, so one mock serves both. Every header the probe requires
/// is present: `etag`, `x-repo-commit`, and `content-range` (whose
/// `/`-suffix is the file size).
async fn mount_repo_file(server: &MockServer, repo_id: &str, filename: &str, body: &'static [u8]) {
    // An etag names a file's content — the cache stores each blob under it —
    // so two different files never share one.
    let etag = format!("\"etag-{}\"", filename.replace('/', "-"));
    Mock::given(method("GET"))
        .and(path(format!("/{repo_id}/resolve/main/{filename}")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("etag", etag.as_str())
                .insert_header("x-repo-commit", "hubsourcecommit")
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

// --- Core oracle: bearer token + cache-root precedence over a live GET ---

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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let hub = HubSource::from_config(&config, &|_: &str| None).unwrap();

    let downloaded = hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

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

/// Drive `HF_HOME` through the injected `env`
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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let env = move |k: &str| (k == "HF_HOME").then(|| hf_home_path.to_str().unwrap().to_string());
    let hub = HubSource::from_config(&config, &env).unwrap();

    let downloaded = hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

    assert!(
        downloaded.starts_with(hf_home.path().join("hub")),
        "expected the downloaded file to land under {{HF_HOME}}/hub, got {downloaded:?}"
    );
    assert_eq!(std::fs::read(&downloaded).unwrap(), BODY);
}

/// The warm-cache oracle: a SECOND `HubSource`
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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };

    // Cold cache: the first HubSource genuinely downloads.
    let first = HubSource::from_config(&config, &|_: &str| None).unwrap();
    first.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();
    let requests_after_first = server.received_requests().await.unwrap().len();
    assert!(
        requests_after_first > 0,
        "the first, cold-cache fetch must reach the mock at least once"
    );

    // Warm cache: a second, independently-constructed HubSource over the
    // SAME hub_cache_dir — models a process restart with the cache directory
    // mounted from a persistent volume.
    let second = HubSource::from_config(&config, &|_: &str| None).unwrap();
    let downloaded_again = second.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();
    assert_eq!(std::fs::read(&downloaded_again).unwrap(), BODY);

    let requests_after_second = server.received_requests().await.unwrap().len();
    assert_eq!(
        requests_after_second, requests_after_first,
        "a second HubSource over the same hub_cache_dir must issue ZERO new requests -- \
         a restart with a mounted cache volume must never re-download (got {requests_after_second} \
         total requests, expected exactly {requests_after_first})"
    );
}

/// `HF_HUB_OFFLINE=1` in the
/// injected env with NO `[models] offline` override at all — the SAME
/// "offline refusal by name" as `offline_miss_refuses_by_name_with_no_catalog_row`
/// above, but driven entirely from the environment fallback.
///
/// A `MockServer` is mounted and wired in as `hub_endpoint` (never actually
/// hit if the refusal is real) so this is a MEASURED refusal-before-request,
/// not merely an error message that happens to say "offline" for the wrong
/// reason: `received_requests()` must come back empty.
#[tokio::test(flavor = "multi_thread")]
async fn hf_hub_offline_env_refuses_by_name_with_no_config_override() {
    let server = MockServer::start().await;
    mount_repo_file(&server, "acme/env-only-offline", "config.json", BODY).await;

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let env = |k: &str| (k == "HF_HUB_OFFLINE").then(|| "1".to_string());
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        ..offline_test_models_config()
    };
    let offline_hub = HubSource::from_config(&config, &env).unwrap();
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
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "an offline refusal must never reach the network -- the mock received a request"
    );
}

/// `HF_HUB_OFFLINE=ON` — a truthy value per `huggingface_hub`'s own
/// `ENV_VARS_TRUE_VALUES = {"1","ON","YES","TRUE"}`
/// (`huggingface_hub/constants.py:12`, `_is_true` at `:16-19`) — must refuse
/// offline by name, exactly like `HF_HUB_OFFLINE=1` above: accepting only
/// `"1"` and a case-insensitive `"true"` would silently resolve
/// `offline=false` for `HF_HUB_OFFLINE=ON` and let the resolver proceed to a
/// live fetch — fail-open on the air-gap knob.
#[tokio::test(flavor = "multi_thread")]
async fn hf_hub_offline_on_refuses_by_name_with_no_config_override() {
    let server = MockServer::start().await;
    mount_repo_file(&server, "acme/env-only-offline-on", "config.json", BODY).await;

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let env = |k: &str| (k == "HF_HUB_OFFLINE").then(|| "ON".to_string());
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        ..offline_test_models_config()
    };
    let offline_hub = HubSource::from_config(&config, &env).unwrap();
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
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "an offline refusal must never reach the network -- the mock received a request"
    );
}

/// `TRANSFORMERS_OFFLINE=1`, with `HF_HUB_OFFLINE` and `[models] offline`
/// both unset, must be honoured as the same fallback `huggingface_hub`
/// itself applies for this variable.
#[tokio::test(flavor = "multi_thread")]
async fn transformers_offline_env_refuses_by_name_with_no_config_override() {
    let server = MockServer::start().await;
    mount_repo_file(
        &server,
        "acme/transformers-offline-only",
        "config.json",
        BODY,
    )
    .await;

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let env = |k: &str| (k == "TRANSFORMERS_OFFLINE").then(|| "1".to_string());
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        ..offline_test_models_config()
    };
    let offline_hub = HubSource::from_config(&config, &env).unwrap();
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
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "an offline refusal must never reach the network -- the mock received a request"
    );
}

/// A present-but-empty `HF_HUB_OFFLINE` (`HF_HUB_OFFLINE=` in a Compose/K8s
/// env block, which the
/// production closure `std::env::var(k).ok()` — `crate::session` —
/// resolves to `Some("")`) must not shadow `TRANSFORMERS_OFFLINE=1`: the
/// alias must still be honoured, offline by name, with the mock never
/// touched.
#[tokio::test(flavor = "multi_thread")]
async fn hf_hub_offline_empty_falls_through_to_transformers_offline_refuses_by_name() {
    let server = MockServer::start().await;
    mount_repo_file(&server, "acme/empty-offline-alias", "config.json", BODY).await;

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let env = |k: &str| match k {
        "HF_HUB_OFFLINE" => Some(String::new()),
        "TRANSFORMERS_OFFLINE" => Some("1".to_string()),
        _ => None,
    };
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        ..offline_test_models_config()
    };
    let offline_hub = HubSource::from_config(&config, &env).unwrap();
    assert!(
        offline_hub.offline(),
        "an empty HF_HUB_OFFLINE must be treated as unset, falling through to \
         TRANSFORMERS_OFFLINE=1"
    );
    let resolver =
        ModelResolver::new(catalog, crate::common::test_artifact_store(), offline_hub).unwrap();

    let err = match resolver
        .resolve(
            &ModelSource::hf("acme/empty-offline-alias"),
            ModelTask::TextEmbedding,
        )
        .await
    {
        Ok(_) => panic!("expected an offline refusal, got a resolved model"),
        Err(e) => e,
    };
    match err {
        JammiError::Model { model_id, message } => {
            assert_eq!(model_id, "acme/empty-offline-alias");
            assert!(
                message.contains("offline") && message.contains("acme/empty-offline-alias"),
                "expected the offline refusal to name the repo id, got: {message}"
            );
        }
        other => panic!("expected JammiError::Model, got {other:?}"),
    }
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "an offline refusal must never reach the network -- the mock received a request"
    );
}

/// A fetch driven entirely by `HF_HUB_CACHE`
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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let env = move |k: &str| {
        (k == "HF_HUB_CACHE").then(|| hf_hub_cache_path.to_str().unwrap().to_string())
    };
    let hub = HubSource::from_config(&config, &env).unwrap();

    let downloaded = hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

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

/// `[models] offline = false` EXPLICIT wins over `HF_HUB_OFFLINE=1` in the
/// environment — config wins, matching
/// `resolve_root`/`resolve_endpoint`/`resolve_token`'s own "config beats
/// env" precedence.
///
/// Proven by an OBSERVED completed network fetch, not by an error's shape:
/// an error-shape assertion alone is a vacuous oracle here, because it
/// cannot distinguish "config correctly overrode the env" from "the env is
/// simply unimplemented and this genuinely 404s for an unrelated reason" —
/// both present as "the error message doesn't say offline". This test
/// mounts the repo's `config.json` and `model.safetensors` on the wiremock
/// server exactly like the passing fetch tests above (`mount_repo_file`,
/// the file's shared mounting helper), so a real config-wins outcome is
/// observable two ways that a vacuous "env unimplemented" run cannot fake:
/// (i) `resolve` returns `Ok` (an unmounted 404 or an actual offline
/// refusal both return `Err`), and (ii) the mock's `received_requests` is
/// non-empty (the resolve did not short-circuit before ever reaching the
/// network — the offline refusal path in `ModelResolver::resolve` returns
/// before building the Hub repo client at all). The `!hub.offline()`
/// unit-level assertion stays, so this test still pins the
/// `HubSource`-level precedence directly, not only its downstream effect.
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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
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
        .resolve(&ModelSource::hf(repo_id), ModelTask::TextEmbedding)
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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let env = |k: &str| (k == "HF_TOKEN").then(|| "env-tok".to_string());
    let hub = HubSource::from_config(&config, &env).unwrap();

    hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let hub = HubSource::from_config(&config, &|_: &str| None).unwrap();

    hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

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

/// A user with `HF_HOME` set
/// (and a `huggingface-cli login` token file there) who ALSO sets
/// `HF_HUB_CACHE` -- a DIFFERENT directory with no `hub` component to pop
/// -- must still authenticate: the token comes from `<HF_HOME>/token`,
/// independent of the cache root, while the downloaded artifact still lands
/// under the `HF_HUB_CACHE` root the cache-root precedence actually
/// resolved.
#[tokio::test(flavor = "multi_thread")]
async fn hf_home_token_file_used_with_hf_hub_cache_set_file_lands_under_hf_hub_cache() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let hf_home = tempfile::tempdir().unwrap();
    std::fs::write(hf_home.path().join("token"), "home-token\n").unwrap();
    let hf_hub_cache = tempfile::tempdir().unwrap();

    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: None,
        hub_token: None,
        offline: None,
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let hf_home_path = hf_home.path().to_str().unwrap().to_string();
    let hf_hub_cache_path = hf_hub_cache.path().to_str().unwrap().to_string();
    let env = move |k: &str| match k {
        "HF_HOME" => Some(hf_home_path.clone()),
        "HF_HUB_CACHE" => Some(hf_hub_cache_path.clone()),
        _ => None,
    };
    let hub = HubSource::from_config(&config, &env).unwrap();

    let downloaded = hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

    assert!(
        downloaded.starts_with(hf_hub_cache.path()),
        "expected the downloaded file to land under {{HF_HUB_CACHE}}, got {downloaded:?}"
    );
    assert_eq!(std::fs::read(&downloaded).unwrap(), BODY);

    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty(), "the mock never received a request");
    for req in &requests {
        assert_eq!(
            req.headers
                .get("authorization")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer home-token"),
            "the token must come from <HF_HOME>/token even though HF_HUB_CACHE is also set"
        );
    }
}

/// A present-but-empty `HF_TOKEN` must fall through to the `<HF_HOME>/token`
/// file rather than resolving to an empty bearer token (or none at all).
#[tokio::test(flavor = "multi_thread")]
async fn empty_hf_token_falls_through_to_home_token_file() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let hf_home = tempfile::tempdir().unwrap();
    std::fs::write(hf_home.path().join("token"), "home-token\n").unwrap();
    let root = tempfile::tempdir().unwrap();

    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: Some(false),
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let hf_home_path = hf_home.path().to_str().unwrap().to_string();
    let env = move |k: &str| match k {
        "HF_TOKEN" => Some(String::new()),
        "HF_HOME" => Some(hf_home_path.clone()),
        _ => None,
    };
    let hub = HubSource::from_config(&config, &env).unwrap();

    hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty());
    for req in &requests {
        assert_eq!(
            req.headers
                .get("authorization")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer home-token"),
            "an empty HF_TOKEN must be treated as absent, falling through to the token file"
        );
    }
}

/// `HF_TOKEN_PATH` names the token file DIRECTLY -- independent of
/// `HF_HOME`, which here resolves to a directory with NO `token` file
/// inside it at all. Matches `huggingface_hub`'s own `HF_TOKEN_PATH`
/// (`constants.py:247-254`); ignoring `HF_TOKEN_PATH` here would carry no
/// `Authorization` header on this request -- a silent 401 on a gated repo
/// where `huggingface_hub` itself authenticates.
#[tokio::test(flavor = "multi_thread")]
async fn hf_token_path_env_used_when_hf_home_has_no_token_file() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let root = tempfile::tempdir().unwrap();
    let hf_home = tempfile::tempdir().unwrap(); // deliberately no `token` file inside
    let token_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(token_file.path(), "path-token\n").unwrap();

    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: Some(false),
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let hf_home_path = hf_home.path().to_str().unwrap().to_string();
    let token_path = token_file.path().to_str().unwrap().to_string();
    let env = move |k: &str| match k {
        "HF_HOME" => Some(hf_home_path.clone()),
        "HF_TOKEN_PATH" => Some(token_path.clone()),
        _ => None,
    };
    let hub = HubSource::from_config(&config, &env).unwrap();

    hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty(), "the mock never received a request");
    for req in &requests {
        assert_eq!(
            req.headers
                .get("authorization")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer path-token"),
            "HF_TOKEN_PATH must be used directly as the token file, independent of HF_HOME"
        );
    }
}

/// `HUGGING_FACE_HUB_TOKEN` -- `huggingface_hub`'s own LIVE legacy alias
/// for `HF_TOKEN` (`utils/_auth.py:145-147`, not deprecated-and-ignored) --
/// is honoured when `HF_TOKEN` itself is absent (an env tier reading only
/// `HF_TOKEN` would send no `Authorization` header).
#[tokio::test(flavor = "multi_thread")]
async fn legacy_hugging_face_hub_token_env_used_when_hf_token_absent() {
    let server = MockServer::start().await;
    mount_repo_file(&server, REPO_ID, FILENAME, BODY).await;

    let root = tempfile::tempdir().unwrap();
    let config = ModelsConfig {
        hub_endpoint: Some(server.uri()),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_token: None,
        offline: Some(false),
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let env = |k: &str| (k == "HUGGING_FACE_HUB_TOKEN").then(|| "legacy-tok".to_string());
    let hub = HubSource::from_config(&config, &env).unwrap();

    hub.model(REPO_ID).get(FILENAME).await.unwrap().unwrap();

    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty(), "the mock never received a request");
    for req in &requests {
        assert_eq!(
            req.headers
                .get("authorization")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer legacy-tok")
        );
    }
}

// --- offline: hit / miss / warm-cache-miss, decided by the catalog alone ---

/// Hit: a `HuggingFace`-shaped model id whose catalog row already carries a
/// present, on-disk location (as if already resolved online) keeps
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
            backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            external_location: Some(model_dir.to_str().unwrap()),
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
        .resolve(&ModelSource::hf("acme/hit-repo"), ModelTask::TextEmbedding)
        .await
        .unwrap();
    assert_eq!(resolved.model_id.0, "acme/hit-repo");
}

/// Miss: no catalog row at all for this repo id -> `offline = true` refuses
/// by name, never attempting a Hub network call. A mounted `MockServer`
/// wired in as `hub_endpoint` turns "never attempting" into a MEASURED
/// property (`received_requests()` empty), not an implication from the
/// error message's wording alone.
#[tokio::test(flavor = "multi_thread")]
async fn offline_miss_refuses_by_name_with_no_catalog_row() {
    let server = MockServer::start().await;
    mount_repo_file(&server, "acme/never-resolved", "config.json", BODY).await;

    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let offline_hub = HubSource::from_config(
        &ModelsConfig {
            offline: Some(true),
            hub_endpoint: Some(server.uri()),
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
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "an offline refusal must never reach the network -- the mock received a request"
    );
}

/// Warm-cache-miss: the Hub cache directory holds real, already-downloaded
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
        hub_idle_timeout_secs: None,
        remote: Default::default(),
    };
    let online_hub = HubSource::from_config(&online_config, &|_: &str| None).unwrap();
    online_hub
        .model(REPO_ID)
        .get(FILENAME)
        .await
        .unwrap()
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
        .resolve(&ModelSource::hf(REPO_ID), ModelTask::TextEmbedding)
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

// --- A stalled transfer fails typed and bounded, never waits forever ---

/// Where a stalled Hub endpoint stops sending.
#[derive(Clone, Copy)]
enum Stall {
    /// It accepts the request and never answers.
    BeforeAnswer,
    /// It answers the metadata probe, then sends the body's headers and a
    /// few bytes of it, and nothing more — the connection stays open.
    MidBody,
}

/// A Hub stand-in that stops sending as `stall` says and never closes the
/// connection — the shape a server or middlebox that goes silent takes.
/// Returns its endpoint and the task serving it (aborted when dropped).
async fn stalled_hub(stall: Stall) -> (String, tokio::task::JoinHandle<()>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let serving = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                loop {
                    let mut request = Vec::new();
                    let mut byte = [0u8; 1];
                    while !request.ends_with(b"\r\n\r\n") {
                        if socket.read(&mut byte).await.unwrap_or(0) == 0 {
                            return;
                        }
                        request.push(byte[0]);
                    }
                    let probe = String::from_utf8_lossy(&request)
                        .to_ascii_lowercase()
                        .contains("range: bytes=0-0");
                    match (stall, probe) {
                        (Stall::MidBody, true) => {
                            socket
                                .write_all(
                                    b"HTTP/1.1 206 Partial Content\r\nx-repo-commit: stalledcommit\r\n\
                                      etag: \"stalled-etag\"\r\ncontent-range: bytes 0-0/1000\r\n\
                                      content-length: 1\r\n\r\nx",
                                )
                                .await
                                .unwrap();
                        }
                        (Stall::MidBody, false) => {
                            socket
                                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 1000\r\n\r\nfirst")
                                .await
                                .unwrap();
                            std::future::pending::<()>().await;
                        }
                        (Stall::BeforeAnswer, _) => std::future::pending::<()>().await,
                    }
                }
            });
        }
    });
    (endpoint, serving)
}

/// A `HubSource` over `endpoint` with a one-second idle timeout and a fresh
/// cache root.
fn one_second_hub(endpoint: String) -> (HubSource, tempfile::TempDir) {
    let root = tempfile::tempdir().unwrap();
    let config = ModelsConfig {
        hub_endpoint: Some(endpoint),
        hub_cache_dir: Some(root.path().to_path_buf()),
        hub_idle_timeout_secs: Some(1),
        ..Default::default()
    };
    (
        HubSource::from_config(&config, &|_: &str| None).unwrap(),
        root,
    )
}

/// `err` is the retryable `Unavailable` naming `repo/file` as a stall.
fn assert_stalled(err: JammiError, repo: &str, file: &str) {
    match err {
        JammiError::Unavailable { resource, reason } => {
            assert_eq!(resource, format!("hf://{repo}/{file}"));
            assert!(
                reason.contains("stalled"),
                "the reason names the stall: {reason}"
            );
        }
        other => panic!("expected the retryable Unavailable, got {other:?}"),
    }
}

/// A transfer that stops mid-body fails within a bounded time as the typed,
/// retryable `Unavailable` naming the repo and the file — the incident this
/// guards against was a verb blocked for hours on exactly this — and leaves
/// no partial file or blob in the cache.
#[tokio::test(flavor = "multi_thread")]
async fn a_transfer_that_stops_mid_body_fails_typed_within_the_idle_timeout() {
    let (endpoint, serving) = stalled_hub(Stall::MidBody).await;
    let (hub, root) = one_second_hub(endpoint);

    let started = std::time::Instant::now();
    let err = hub
        .model("acme/stalled-model")
        .get("model.safetensors")
        .await
        .unwrap_err();
    let waited = started.elapsed();
    serving.abort();

    assert_stalled(err, "acme/stalled-model", "model.safetensors");
    assert!(
        waited < std::time::Duration::from_secs(10),
        "a one-second idle timeout bounds the wait, got {waited:?}"
    );
    let blobs = root.path().join("hub/models--acme--stalled-model/blobs");
    let left: Vec<_> = std::fs::read_dir(&blobs)
        .map(|entries| entries.map(|e| e.unwrap().file_name()).collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "no partial file or blob is left: {left:?}");
}

/// A Hub that accepts the request and never answers fails the same way.
#[tokio::test(flavor = "multi_thread")]
async fn a_hub_that_never_answers_fails_typed_within_the_idle_timeout() {
    let (endpoint, serving) = stalled_hub(Stall::BeforeAnswer).await;
    let (hub, _root) = one_second_hub(endpoint);

    let err = hub
        .model("acme/silent-model")
        .get("config.json")
        .await
        .unwrap_err();
    serving.abort();
    assert_stalled(err, "acme/silent-model", "config.json");
}

/// Through the verb path the incident took: a model resolution whose Hub
/// stops mid-transfer returns the typed stall, rather than never returning.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_resolution_over_a_stalled_hub_returns_the_stall() {
    let (endpoint, serving) = stalled_hub(Stall::MidBody).await;
    let (hub, _root) = one_second_hub(endpoint);
    let dir = tempfile::tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store(), hub).unwrap();

    let err = match resolver
        .resolve(
            &ModelSource::hf("acme/stalled-model"),
            ModelTask::TextEmbedding,
        )
        .await
    {
        Ok(_) => panic!("a stalled Hub cannot resolve a model"),
        Err(e) => e,
    };
    serving.abort();
    assert_stalled(err, "acme/stalled-model", "config.json");
}
