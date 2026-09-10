//! The one place `[models]` (`jammi_db::config::ModelsConfig`) turns into a
//! live Hugging Face Hub client (esc-096).
//!
//! Before this module, every `hf_hub::api::sync::Api::new()` call site (the
//! resolver, the fine-tune worker's HF fallback) built its own `Api` off
//! `ApiBuilder::new()`/`Cache::default()`, which reads `HF_HOME` inconsistently,
//! never reads `HF_TOKEN` at all (hf-hub 0.5 does not), and — worse —
//! `Cache::default()` panics outright when `HOME` is unset (`dirs::home_dir()
//! .expect(..)`, hf-hub `lib.rs:202-209`). `HubSource` is built exactly ONCE
//! per session, at the `jammi-ai` choke point
//! (`crate::session::InferenceSession::wrap`), and every call site downstream
//! (the resolver, the fine-tune worker) shares that one client.
//!
//! # Precedence
//!
//! **Cache root** (a `hub/` subdirectory is appended before it becomes the
//! `hf_hub::Cache`):
//! 1. `[models] hub_cache_dir`
//! 2. the `HF_HOME` environment variable
//! 3. `directories::BaseDirs::home_dir()/.cache/huggingface`
//!
//! If none of the three resolves (no config, no `HF_HOME`, and no home
//! directory — `HOME`/`USERPROFILE` unset) this is a typed
//! [`JammiError::Config`], **never** the panic `hf_hub::Cache::default()`
//! raises in the same situation.
//!
//! **Endpoint:**
//! 1. `[models] hub_endpoint`
//! 2. the `HF_ENDPOINT` environment variable
//! 3. hf-hub's own default (`https://huggingface.co`)
//!
//! **Token** (only set on the client when one resolves — an absent token
//! means no `Authorization` header, exactly like an anonymous
//! `huggingface-cli` session):
//! 1. `[models] hub_token` (a [`jammi_db::config::SecretSource`] — inline or
//!    file-backed)
//! 2. the `HF_TOKEN` environment variable
//! 3. the cache's own `token` file (`{root}/token`, `huggingface-cli login`'s
//!    file — [`hf_hub::Cache::token`])
//!
//! The `HF_HOME`/`HF_ENDPOINT`/`HF_TOKEN` fallbacks read the process
//! environment **here**, through the `env` closure the caller passes — never
//! inside `jammi_db::config::JammiConfig::load_from` (which stays
//! process-env-free by construction). Production passes
//! `&|k: &str| std::env::var(k).ok()`; a test passes a placeholder map.
//!
//! The client is always built with `ApiBuilder::from_cache(..)`, never
//! `ApiBuilder::from_env()`/`ApiBuilder::new()` — both of those re-derive the
//! cache root from `Cache::from_env()`/`Cache::default()` a second time,
//! independently of the precedence above, and `Cache::default()` is the exact
//! panic esc-096 exists to remove.
//!
//! # The `offline` promise is Hub-only
//!
//! `ModelsConfig::offline` refuses every *Hub network* fetch — every call
//! site that reaches `self.api()` checks it first. That is two call sites,
//! both after their own catalog lookup and both refusing by name when no
//! catalog row resolved the model (a warm Hub cache directory with no
//! catalog row is still a miss: the catalog, not the on-disk cache, is
//! offline's source of truth): `ModelResolver::resolve`'s `HuggingFace`
//! arm, and the fine-tune worker's `build_encoder_adapters` HF fallback
//! (reached when a fine-tune job's BASE model has a catalog row but no
//! `artifact_path` yet — i.e. it has never been resolved before). It does
//! **not** reach the fine-tune worker's ADAPTER fetch for an
//! already-trained model: that path always reads the adapter bundle
//! through the artifact store (object storage), never the Hub, offline or
//! not.
use std::path::PathBuf;

use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::Cache;
use jammi_db::config::ModelsConfig;
use jammi_db::error::{JammiError, Result};

/// A Hugging Face Hub client built once from `[models]`, shared by every
/// jammi-ai call site that talks to the Hub. See the module docs for the
/// precedence chain and the `offline` promise.
///
/// `Debug` is hand-written, NOT derived: `hf_hub::api::sync::Api`'s own
/// (derived) `Debug` walks down into its header map, which holds the
/// resolved bearer token as a plaintext `Authorization` header value once
/// [`HubSource::from_config`] has set one — the same class of leak
/// `jammi_db::config::secret::Secret` exists to prevent everywhere else in
/// this config. A `{:?}` of a `HubSource` must never reach it.
#[derive(Clone)]
pub struct HubSource {
    api: Api,
    offline: bool,
}

impl std::fmt::Debug for HubSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubSource")
            .field("api", &"<redacted: may hold a Hub bearer token>")
            .field("offline", &self.offline)
            .finish()
    }
}

impl HubSource {
    /// Build a [`HubSource`] from `[models]`, resolving the cache root,
    /// endpoint, and token per the module docs' precedence. `env` stands in
    /// for the process environment (`HF_HOME`/`HF_ENDPOINT`/`HF_TOKEN`) —
    /// production passes `&|k: &str| std::env::var(k).ok()`, a test passes a
    /// placeholder closure, so this function's own behaviour is
    /// deterministic and hermetic.
    ///
    /// Never panics: an unresolvable cache root (no `hub_cache_dir`, no
    /// `HF_HOME`, no home directory) is a typed [`JammiError::Config`], and a
    /// malformed [`jammi_db::config::SecretSource`] (a `hub_token = { file =
    /// .. }` naming an unreadable file) surfaces as its own typed error.
    pub fn from_config(
        config: &ModelsConfig,
        env: &impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let env: &dyn Fn(&str) -> Option<String> = env;
        let root = resolve_root(config, env)?;
        let cache = Cache::new(root.join("hub"));

        let mut builder = ApiBuilder::from_cache(cache.clone());
        if let Some(endpoint) = resolve_endpoint(config, env) {
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(token) = resolve_token(config, env, &cache)? {
            builder = builder.with_token(Some(token));
        }

        let api = builder
            .build()
            .map_err(|e| JammiError::Config(format!("Hugging Face Hub client init failed: {e}")))?;

        Ok(Self {
            api,
            offline: config.offline,
        })
    }

    /// The underlying blocking Hub client every resolver/worker call site
    /// shares.
    pub fn api(&self) -> &Api {
        &self.api
    }

    /// Whether `[models] offline` is set — see the module docs' "offline
    /// promise is Hub-only" section for exactly what this refuses.
    pub fn offline(&self) -> bool {
        self.offline
    }
}

/// Resolve the cache root per the module docs' precedence: `hub_cache_dir` >
/// `HF_HOME` > the platform home directory's `.cache/huggingface`. `None` at
/// every step is a typed [`JammiError::Config`], never
/// [`hf_hub::Cache::default`]'s panic.
fn resolve_root(config: &ModelsConfig, env: &dyn Fn(&str) -> Option<String>) -> Result<PathBuf> {
    resolve_root_with(config, env, default_home_dir)
}

/// The platform home directory, via `directories::BaseDirs` — a thin,
/// swappable wrapper so [`resolve_root_with`]'s "no home directory found"
/// branch is deterministically reachable in a unit test. `directories`
/// itself falls back through several platform mechanisms (`$HOME`, then
/// `getpwuid_r` on Unix) before giving up, so — unlike removing the `HOME`
/// environment variable, which a properly provisioned dev machine or CI
/// account's password-database entry silently absorbs — this function's
/// `None` case is realistically only reached in a minimal/distroless
/// container whose numeric UID has no `/etc/passwd` entry at all (esc-096's
/// own motivating case: the `nonroot`/`65532` runtime users this repo's
/// Dockerfile provisions).
fn default_home_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
}

/// [`resolve_root`]'s implementation, parameterized over the home-directory
/// lookup so tests can force the "nothing resolves" branch without depending
/// on a platform-specific, not-reliably-forceable `HOME`/passwd-database
/// interaction.
fn resolve_root_with(
    config: &ModelsConfig,
    env: &dyn Fn(&str) -> Option<String>,
    home_dir: impl FnOnce() -> Option<PathBuf>,
) -> Result<PathBuf> {
    if let Some(dir) = &config.hub_cache_dir {
        return Ok(dir.clone());
    }
    if let Some(home) = env("HF_HOME") {
        return Ok(PathBuf::from(home));
    }
    home_dir()
        .map(|home| home.join(".cache").join("huggingface"))
        .ok_or_else(|| {
            JammiError::Config(
                "cannot resolve a Hugging Face Hub cache root: `[models] hub_cache_dir` is \
                 unset, `HF_HOME` is unset, and no home directory could be determined \
                 (HOME/USERPROFILE unset, and no password-database entry for this user) — \
                 set `[models] hub_cache_dir` or `HF_HOME` explicitly"
                    .into(),
            )
        })
}

/// Resolve the endpoint per the module docs' precedence: `hub_endpoint` >
/// `HF_ENDPOINT` > `None` (hf-hub's own default, `https://huggingface.co`,
/// applies when the builder is never told otherwise).
fn resolve_endpoint(config: &ModelsConfig, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    config.hub_endpoint.clone().or_else(|| env("HF_ENDPOINT"))
}

/// Resolve the bearer token per the module docs' precedence: `hub_token` >
/// `HF_TOKEN` > the cache's own `token` file. `Ok(None)` means "send no
/// `Authorization` header", not a failure.
fn resolve_token(
    config: &ModelsConfig,
    env: &dyn Fn(&str) -> Option<String>,
    cache: &Cache,
) -> Result<Option<String>> {
    if let Some(source) = &config.hub_token {
        return Ok(Some(source.resolve()?.expose().to_string()));
    }
    if let Some(token) = env("HF_TOKEN") {
        return Ok(Some(token));
    }
    Ok(cache.token())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jammi_db::config::SecretSource;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    // --- precedence: cache root (config > env > default) ---

    #[test]
    fn root_prefers_config_over_env() {
        let config = ModelsConfig {
            hub_cache_dir: Some(PathBuf::from("/configured/root")),
            ..Default::default()
        };
        let env = |k: &str| (k == "HF_HOME").then(|| "/env/root".to_string());
        assert_eq!(
            resolve_root(&config, &env).unwrap(),
            PathBuf::from("/configured/root")
        );
    }

    #[test]
    fn root_falls_back_to_hf_home_env() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_HOME").then(|| "/env/root".to_string());
        assert_eq!(
            resolve_root(&config, &env).unwrap(),
            PathBuf::from("/env/root")
        );
    }

    #[test]
    fn root_falls_back_to_home_dir_when_config_and_env_absent() {
        let config = ModelsConfig::default();
        let resolved = resolve_root(&config, &no_env).unwrap();
        assert!(
            resolved.ends_with(".cache/huggingface"),
            "expected the platform home-dir fallback, got {resolved:?}"
        );
    }

    // --- precedence: endpoint (config > env > hf-hub default) ---

    #[test]
    fn endpoint_prefers_config_over_env() {
        let config = ModelsConfig {
            hub_endpoint: Some("https://configured.example".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "HF_ENDPOINT").then(|| "https://env.example".to_string());
        assert_eq!(
            resolve_endpoint(&config, &env),
            Some("https://configured.example".into())
        );
    }

    #[test]
    fn endpoint_falls_back_to_env() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_ENDPOINT").then(|| "https://env.example".to_string());
        assert_eq!(
            resolve_endpoint(&config, &env),
            Some("https://env.example".into())
        );
    }

    #[test]
    fn endpoint_none_when_config_and_env_absent() {
        let config = ModelsConfig::default();
        assert_eq!(resolve_endpoint(&config, &no_env), None);
    }

    // --- precedence: token (config > env > cache token file) ---

    #[test]
    fn token_prefers_config_over_env_and_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("token"), "file-token\n").unwrap();
        let cache = Cache::new(dir.path().join("hub"));
        let config = ModelsConfig {
            hub_token: Some(SecretSource::Inline("config-token".into())),
            ..Default::default()
        };
        let env = |k: &str| (k == "HF_TOKEN").then(|| "env-token".to_string());
        assert_eq!(
            resolve_token(&config, &env, &cache).unwrap(),
            Some("config-token".into())
        );
    }

    #[test]
    fn token_falls_back_to_env_over_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("token"), "file-token\n").unwrap();
        let cache = Cache::new(dir.path().join("hub"));
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_TOKEN").then(|| "env-token".to_string());
        assert_eq!(
            resolve_token(&config, &env, &cache).unwrap(),
            Some("env-token".into())
        );
    }

    #[test]
    fn token_falls_back_to_cache_file_when_config_and_env_absent() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("token"), "file-token\n").unwrap();
        let cache = Cache::new(dir.path().join("hub"));
        let config = ModelsConfig::default();
        assert_eq!(
            resolve_token(&config, &no_env, &cache).unwrap(),
            Some("file-token".into())
        );
    }

    #[test]
    fn token_none_when_nothing_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("hub"));
        let config = ModelsConfig::default();
        assert_eq!(resolve_token(&config, &no_env, &cache).unwrap(), None);
    }

    // --- Debug never prints the Hub token ---

    /// `hf_hub::api::sync::Api`'s derived `Debug` walks its header map,
    /// which carries the resolved bearer token in plaintext once a token
    /// resolves. `HubSource`'s own hand-written `Debug` must redact the
    /// whole `api` field rather than let that leak through.
    #[test]
    fn debug_never_prints_the_resolved_hub_token() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelsConfig {
            hub_cache_dir: Some(dir.path().to_path_buf()),
            hub_token: Some(SecretSource::Inline("secret-hub-token-xyz".into())),
            ..Default::default()
        };
        let hub = HubSource::from_config(&config, &no_env).unwrap();
        let rendered = format!("{hub:?}");
        assert!(
            !rendered.contains("secret-hub-token-xyz"),
            "HubSource::Debug leaked the Hub token: {rendered}"
        );
        let pretty = format!("{hub:#?}");
        assert!(
            !pretty.contains("secret-hub-token-xyz"),
            "HubSource::Debug (pretty) leaked the Hub token: {pretty}"
        );
    }

    // --- never a panic: unresolvable root is a typed error ---

    /// esc-096: `hf_hub::Cache::default()` (what every pre-fix `Api::new()`
    /// call site reached through `ApiBuilder::new()`) panics outright —
    /// `dirs::home_dir().expect(..)` — when no home directory resolves.
    /// `resolve_root`/`HubSource::from_config` must refuse the SAME
    /// situation with a typed `JammiError::Config` instead. Forces the
    /// "nothing resolves" branch through [`resolve_root_with`]'s injected
    /// `home_dir` closure rather than mutating the real process `HOME` —
    /// `directories::BaseDirs` falls back through the password-database
    /// entry for the running user on Unix (see `default_home_dir`'s doc), so
    /// removing `HOME` alone does not reliably reproduce this branch on a
    /// normally provisioned dev machine or CI account.
    #[test]
    fn unresolvable_root_is_a_typed_config_error_not_a_panic() {
        let config = ModelsConfig::default();
        let err = resolve_root_with(&config, &no_env, || None).unwrap_err();
        match err {
            JammiError::Config(message) => {
                assert!(
                    message.contains("Hugging Face Hub cache root"),
                    "expected the cache-root refusal, got: {message}"
                );
            }
            other => panic!("expected JammiError::Config, got {other:?}"),
        }

        // Same refusal end to end through the public constructor — proves
        // `from_config` propagates `resolve_root`'s error rather than
        // panicking or swallowing it. Uses the real (non-injected)
        // `resolve_root`, so this assertion only pins "never panics, always
        // a `Result`" — it cannot force the unresolvable branch on this
        // machine's account, but `catch_unwind` proves the panic-avoidance
        // property regardless of which branch this host's `directories`
        // resolution actually takes.
        let outcome = std::panic::catch_unwind(|| HubSource::from_config(&config, &no_env));
        assert!(
            outcome.is_ok(),
            "HubSource::from_config must never panic, even when nothing resolves the cache root"
        );
    }

    /// No home directory resolvable, but `hub_cache_dir` configured — the
    /// panic-prone home-directory fallback is never even consulted when the
    /// config already answers the question.
    #[test]
    fn hub_cache_dir_configured_skips_the_home_dir_fallback_entirely() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelsConfig {
            hub_cache_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let resolved = resolve_root_with(&config, &no_env, || {
            panic!("home_dir() must not be called when hub_cache_dir is configured")
        })
        .unwrap();
        assert_eq!(resolved, dir.path());
    }
}
