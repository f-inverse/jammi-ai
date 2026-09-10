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
//! **Cache root:**
//! 1. `[models] hub_cache_dir` — a `hub/` subdirectory is appended
//! 2. the `HF_HUB_CACHE` environment variable — used AS the `hf_hub::Cache`
//!    root directly, with nothing appended, matching `huggingface_hub`'s own
//!    `HF_HUB_CACHE` convention (it already names the hub cache dir itself,
//!    not a parent `HF_HOME`-style directory the client appends `hub/` to)
//! 3. the `HF_HOME` environment variable — a `hub/` subdirectory is appended
//! 4. `directories::BaseDirs::home_dir()/.cache/huggingface` — a `hub/`
//!    subdirectory is appended
//!
//! If none of the four resolves (no config, no `HF_HUB_CACHE`, no `HF_HOME`,
//! and no home directory — `HOME`/`USERPROFILE` unset) this is a typed
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
//! 2. the `HF_TOKEN` environment variable, when non-empty (see "empty
//!    values are absent" below)
//! 3. `<HF_HOME>/token`, read directly and trimmed — `HF_HOME` here is
//!    resolved on its own (non-empty env value, else the platform home
//!    directory's `.cache/huggingface`), **independently of whichever tier
//!    won the cache-root precedence above**. This is deliberately NOT
//!    `hf_hub::Cache::token`/`token_path`, which derives the token file by
//!    popping the CACHE ROOT's last path component (hf-hub 0.5's own
//!    comment: "Remove `\"hub\"`") — an arithmetic that only recovers
//!    `HF_HOME` when the cache root actually was built as `<HF_HOME>/hub`.
//!    With `HF_HUB_CACHE` set, the cache root has no `hub` component to pop
//!    at all, so `Cache::token_path` would silently look in
//!    `parent(HF_HUB_CACHE)/token` instead — this module never makes that
//!    mistake, matching `huggingface_hub`'s own `HF_TOKEN_PATH`
//!    (`utils/_auth.py`), which never varies with `HF_HUB_CACHE` either.
//!
//! **Offline** (whether `HubSource::offline` refuses a Hub fetch — see
//! "the `offline` promise is Hub-only" below for exactly what this refuses):
//! 1. `[models] offline`, when `Some(_)` — wins outright, in EITHER
//!    direction: a literal `offline = false` in the TOML forces online even
//!    when `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` is set in the
//!    environment, exactly as a literal `offline = true` forces offline even
//!    when neither is set
//! 2. the `HF_HUB_OFFLINE` environment variable, when `[models] offline` is
//!    omitted (`None`) AND `HF_HUB_OFFLINE` is non-empty; if `HF_HUB_OFFLINE`
//!    is itself unset OR present-but-empty, the `TRANSFORMERS_OFFLINE`
//!    environment variable, under the identical truthy rule —
//!    `huggingface_hub` reads `TRANSFORMERS_OFFLINE` as an alias precisely
//!    when `HF_HUB_OFFLINE` is unset (Python's `os.environ.get("HF_HUB_OFFLINE")
//!    or os.environ.get("TRANSFORMERS_OFFLINE")`, `constants.py:192`, where
//!    `or` also skips a `""` left side), and this module mirrors that.
//!    Accepted truthy values, for either variable, are `huggingface_hub`'s
//!    own `ENV_VARS_TRUE_VALUES` set — `{"1", "ON", "YES", "TRUE"}`, matched
//!    case-insensitively with surrounding whitespace trimmed first (trimming
//!    is a strict superset of `huggingface_hub`'s own exact-string match, so
//!    it can only ever push an edge case TOWARD offline, never away from it
//!    — see
//!    <https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhuboffline>;
//!    hf-hub, the Rust crate this module wraps, does not read either
//!    variable at all — see this module's private `resolve_offline`/
//!    `is_hf_hub_offline_truthy`)
//! 3. `false`
//!
//! # Empty values are absent
//!
//! Every one of this module's six env reads (`HF_HUB_OFFLINE`,
//! `TRANSFORMERS_OFFLINE`, `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`,
//! `HF_TOKEN`) goes through one helper, `env_nonempty`: a value that is
//! *present but empty* (after trimming ASCII whitespace) is treated exactly
//! like an ABSENT variable, never like a real value that happens to be
//! `""`. A Compose/K8s env block naming a variable with no value
//! (`HF_HUB_OFFLINE:`) or an operator's shell `export HF_TOKEN=` both
//! produce exactly this shape — the production closure
//! (`&|k: &str| std::env::var(k).ok()`) yields `Some("")`, not `None`, for
//! either, and every tier downstream of `env_nonempty` falls back exactly
//! as if the variable had never been set at all.
//!
//! This is a strict SUPERSET of `huggingface_hub`'s own handling for two of
//! the six: the offline alias's `os.environ.get(A) or os.environ.get(B)`
//! and `HF_TOKEN`'s own `_clean_token` (`huggingface_hub/utils/_auth.py`,
//! which maps `""` to `None`) already treat an empty value as absent, same
//! as here. For `HF_HOME`/`HF_ENDPOINT`, upstream's own `os.environ.get(KEY,
//! default)` does NOT — an empty string is present there, so it wins over
//! the default, and upstream itself would resolve a CWD-relative cache root
//! or an empty-string endpoint in that case. This module chooses the safer
//! reading everywhere instead of replicating that upstream inconsistency:
//! every one of the six variables falls back to its safe default when empty
//! (home cache, default endpoint, no token, offline resolved from the
//! non-empty variable only) — never toward the network, and never toward a
//! nonsense relative path.
//!
//! The `HF_HOME`/`HF_HUB_CACHE`/`HF_ENDPOINT`/`HF_TOKEN`/`HF_HUB_OFFLINE`/
//! `TRANSFORMERS_OFFLINE` fallbacks read the process environment **here**,
//! through the `env` closure the caller passes — never inside
//! `jammi_db::config::JammiConfig::load_from` (which stays process-env-free
//! by construction). Production passes `&|k: &str| std::env::var(k).ok()`; a
//! test passes a placeholder map.
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
        let cache = Cache::new(root);

        let mut builder = ApiBuilder::from_cache(cache.clone());
        if let Some(endpoint) = resolve_endpoint(config, env) {
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(token) = resolve_token(config, env)? {
            builder = builder.with_token(Some(token));
        }

        let api = builder
            .build()
            .map_err(|e| JammiError::Config(format!("Hugging Face Hub client init failed: {e}")))?;

        Ok(Self {
            api,
            offline: resolve_offline(config, env),
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
/// `HF_HUB_CACHE` > `HF_HOME` > the platform home directory's
/// `.cache/huggingface`. Returns the final [`hf_hub::Cache`] root directly —
/// `hub_cache_dir`, `HF_HOME`, and the platform-home fallback each get a
/// `hub/` subdirectory appended; `HF_HUB_CACHE` is used AS the cache root,
/// with nothing appended, matching `huggingface_hub`'s own convention for
/// that variable. `None` at every step is a typed [`JammiError::Config`],
/// never [`hf_hub::Cache::default`]'s panic.
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
        return Ok(dir.join("hub"));
    }
    if let Some(cache_dir) = env_nonempty(env, "HF_HUB_CACHE") {
        return Ok(PathBuf::from(cache_dir));
    }
    if let Some(home) = env_nonempty(env, "HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    home_dir()
        .map(|home| home.join(".cache").join("huggingface").join("hub"))
        .ok_or_else(|| {
            JammiError::Config(
                "cannot resolve a Hugging Face Hub cache root: `[models] hub_cache_dir` is \
                 unset, `HF_HUB_CACHE` is unset, `HF_HOME` is unset, and no home directory \
                 could be determined (HOME/USERPROFILE unset, and no password-database entry \
                 for this user) — set `[models] hub_cache_dir`, `HF_HUB_CACHE`, or `HF_HOME` \
                 explicitly"
                    .into(),
            )
        })
}

/// Resolve the endpoint per the module docs' precedence: `hub_endpoint` >
/// `HF_ENDPOINT` (non-empty) > `None` (hf-hub's own default,
/// `https://huggingface.co`, applies when the builder is never told
/// otherwise).
fn resolve_endpoint(config: &ModelsConfig, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    config
        .hub_endpoint
        .clone()
        .or_else(|| env_nonempty(env, "HF_ENDPOINT"))
}

/// Resolve the bearer token per the module docs' precedence: `hub_token` >
/// `HF_TOKEN` (non-empty) > `<HF_HOME>/token`. `Ok(None)` means "send no
/// `Authorization` header", not a failure.
///
/// The token FILE is resolved independently of the Hub cache root — see
/// [`token_file_path`]'s doc for why this must never go through
/// `hf_hub::Cache::token`/`token_path`.
fn resolve_token(
    config: &ModelsConfig,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<String>> {
    resolve_token_with(config, env, default_home_dir)
}

/// [`resolve_token`]'s implementation, parameterized over the home-directory
/// lookup for the same reason [`resolve_root_with`] is — so a unit test can
/// force the "no config, no `HF_HOME`, no platform home directory" branch
/// deterministically.
fn resolve_token_with(
    config: &ModelsConfig,
    env: &dyn Fn(&str) -> Option<String>,
    home_dir: impl FnOnce() -> Option<PathBuf>,
) -> Result<Option<String>> {
    if let Some(source) = &config.hub_token {
        return Ok(Some(source.resolve()?.expose().to_string()));
    }
    if let Some(token) = env_nonempty(env, "HF_TOKEN") {
        return Ok(Some(token));
    }
    Ok(token_file_path(env, home_dir).and_then(|path| read_token_file(&path)))
}

/// `<HF_HOME>/token` — matching `huggingface_hub`'s own `HF_TOKEN_PATH`
/// (`utils/_auth.py`) — where `HF_HOME` is resolved on its own: a non-empty
/// `HF_HOME` env value, else the platform home directory's
/// `.cache/huggingface`. Deliberately computed WITHOUT reference to the Hub
/// cache root [`resolve_root`] resolved: `hf_hub::Cache::token_path` derives
/// the token file by popping the CACHE ROOT's last path component (hf-hub
/// 0.5's own comment: "Remove `\"hub\"`"), which is only a correct recovery
/// of `HF_HOME` when the cache root was actually built as `<HF_HOME>/hub`.
/// When `HF_HUB_CACHE` wins the cache-root precedence instead, the cache
/// root has no `hub` component to pop at all, so `Cache::token_path` would
/// silently look in `parent(HF_HUB_CACHE)/token` — a user with `HF_HOME` set
/// (and a `huggingface-cli login` token there) who ALSO sets `HF_HUB_CACHE`
/// would then silently lose authentication. This function never takes that
/// path: it reads `HF_HOME` itself, independently of whichever tier won the
/// cache-root precedence.
fn token_file_path(
    env: &dyn Fn(&str) -> Option<String>,
    home_dir: impl FnOnce() -> Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(home) = env_nonempty(env, "HF_HOME") {
        return Some(PathBuf::from(home).join("token"));
    }
    home_dir().map(|home| home.join(".cache").join("huggingface").join("token"))
}

/// Reads a Hub token file (`huggingface-cli login`'s file, or
/// `<HF_HOME>/token`), trimming surrounding whitespace. An unreadable file,
/// or one that is empty after trimming, resolves to `None` rather than an
/// empty bearer token — matching `hf_hub::Cache::token`'s own convention for
/// the same file shape.
fn read_token_file(path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    let trimmed = contents.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Resolve `[models] offline` per the module docs' "Offline" precedence:
/// `[models] offline`, when `Some(_)`, wins outright in EITHER direction —
/// never OR'd with the environment, so a literal `offline = false` silences
/// `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` exactly as a literal `offline =
/// true` forces it on regardless of the environment. Only an OMITTED config
/// value (`None`) falls back to `HF_HUB_OFFLINE`; if THAT is itself unset OR
/// present-but-empty (see the module docs' "empty values are absent" —
/// `env_nonempty` treats the two identically, matching Python's `or` in
/// `huggingface_hub`'s own `os.environ.get("HF_HUB_OFFLINE") or
/// os.environ.get("TRANSFORMERS_OFFLINE")`), it falls back to
/// `TRANSFORMERS_OFFLINE` under the identical truthy rule, matching
/// `huggingface_hub`'s own alias; then to `false`.
fn resolve_offline(config: &ModelsConfig, env: &dyn Fn(&str) -> Option<String>) -> bool {
    if let Some(offline) = config.offline {
        return offline;
    }
    env_nonempty(env, "HF_HUB_OFFLINE")
        .or_else(|| env_nonempty(env, "TRANSFORMERS_OFFLINE"))
        .map(|value| is_hf_hub_offline_truthy(&value))
        .unwrap_or(false)
}

/// Every `HubSource` env read (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
/// `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`) goes through this
/// one helper — see the module docs' "empty values are absent" section for
/// why and for the exact upstream comparison. A present-but-empty value
/// (after trimming ASCII whitespace) is treated as ABSENT, not as a real
/// value that happens to be `""`.
fn env_nonempty(env: &dyn Fn(&str) -> Option<String>, key: &str) -> Option<String> {
    env(key).filter(|value| !value.trim().is_empty())
}

/// `true` for any of `huggingface_hub`'s own `ENV_VARS_TRUE_VALUES` —
/// `"1"`, `"on"`, `"yes"`, `"true"` — matched case-insensitively with
/// surrounding whitespace trimmed first; anything else — including empty,
/// `"0"`, `"false"`, `"off"`, `"no"`, or any other value `huggingface_hub`
/// would also treat as false — is not offline.
///
/// Trimming surrounding whitespace is a strict superset of
/// `huggingface_hub`'s own exact-string `value.upper() in
/// ENV_VARS_TRUE_VALUES` match (`huggingface_hub/constants.py`'s `_is_true`)
/// — it can only ever turn a value `huggingface_hub` itself would reject
/// (e.g. `" 1 "`) into truthy, never the reverse, so this function only
/// ever fails TOWARD offline (the safe direction for an air-gap knob), never
/// away from it.
///
/// This mirrors `huggingface_hub`'s own `HF_HUB_OFFLINE`/
/// `TRANSFORMERS_OFFLINE` truthy convention (see the module docs' "Offline"
/// precedence for the citation); hf-hub 0.5, the Rust crate `HubSource`
/// wraps, does not read either variable at all, so this crate reads and
/// parses them directly rather than leaving them silently ignored.
fn is_hf_hub_offline_truthy(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed == "1"
        || trimmed.eq_ignore_ascii_case("on")
        || trimmed.eq_ignore_ascii_case("yes")
        || trimmed.eq_ignore_ascii_case("true")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jammi_db::config::SecretSource;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    // --- env_nonempty: present-but-empty is absent (#481 fix round 3) ---

    #[test]
    fn env_nonempty_treats_present_but_empty_as_absent() {
        let env = |k: &str| match k {
            "EMPTY" => Some(String::new()),
            "WHITESPACE" => Some("   ".to_string()),
            "VALUE" => Some("v".to_string()),
            _ => None,
        };
        assert_eq!(
            env_nonempty(&env, "EMPTY"),
            None,
            "a present-but-empty value must resolve to None"
        );
        assert_eq!(
            env_nonempty(&env, "WHITESPACE"),
            None,
            "a whitespace-only value must resolve to None"
        );
        assert_eq!(env_nonempty(&env, "MISSING"), None);
        assert_eq!(env_nonempty(&env, "VALUE"), Some("v".to_string()));
    }

    // --- precedence: cache root (config > HF_HUB_CACHE > HF_HOME > default) ---

    #[test]
    fn root_prefers_config_over_hf_hub_cache_and_hf_home() {
        let config = ModelsConfig {
            hub_cache_dir: Some(PathBuf::from("/configured/root")),
            ..Default::default()
        };
        let env = |k: &str| match k {
            "HF_HUB_CACHE" => Some("/env/hub-cache".to_string()),
            "HF_HOME" => Some("/env/home".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_root(&config, &env).unwrap(),
            PathBuf::from("/configured/root/hub"),
            "hub_cache_dir must win over both HF_HUB_CACHE and HF_HOME, with hub/ appended"
        );
    }

    /// #481 fix round 2, advisory A4: `HF_HUB_CACHE` (used directly, no
    /// `hub/` appended) beats `HF_HOME` when `[models] hub_cache_dir` is
    /// unset — `huggingface_hub` honours `HF_HUB_CACHE` above `HF_HOME`, and
    /// this branch's own `jammi-encoders` live-hub harness already reads it
    /// at that precedence.
    #[test]
    fn root_falls_back_to_hf_hub_cache_over_hf_home_no_hub_suffix_appended() {
        let config = ModelsConfig::default();
        let env = |k: &str| match k {
            "HF_HUB_CACHE" => Some("/env/hub-cache".to_string()),
            "HF_HOME" => Some("/env/home".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_root(&config, &env).unwrap(),
            PathBuf::from("/env/hub-cache"),
            "HF_HUB_CACHE must be used directly as the cache root, with nothing appended"
        );
    }

    #[test]
    fn root_falls_back_to_hf_home_env_when_hf_hub_cache_absent() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_HOME").then(|| "/env/root".to_string());
        assert_eq!(
            resolve_root(&config, &env).unwrap(),
            PathBuf::from("/env/root/hub")
        );
    }

    #[test]
    fn root_falls_back_to_home_dir_when_config_and_env_absent() {
        let config = ModelsConfig::default();
        let resolved = resolve_root(&config, &no_env).unwrap();
        assert!(
            resolved.ends_with(".cache/huggingface/hub"),
            "expected the platform home-dir fallback with hub/ appended, got {resolved:?}"
        );
    }

    /// #481 fix round 3: a present-but-empty `HF_HUB_CACHE` must fall
    /// through to `HF_HOME`, not be used as a literal empty/CWD-relative
    /// cache root.
    #[test]
    fn root_empty_hf_hub_cache_falls_through_to_hf_home() {
        let config = ModelsConfig::default();
        let env = |k: &str| match k {
            "HF_HUB_CACHE" => Some(String::new()),
            "HF_HOME" => Some("/env/root".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_root(&config, &env).unwrap(),
            PathBuf::from("/env/root/hub"),
            "an empty HF_HUB_CACHE must be treated as unset, falling through to HF_HOME"
        );
    }

    /// #481 fix round 3: a present-but-empty `HF_HOME`, with `HF_HUB_CACHE`
    /// also absent, must fall through to the platform home-dir default
    /// rather than resolving to a `""`/CWD-relative root.
    #[test]
    fn root_empty_hf_home_falls_through_to_platform_home_dir() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_HOME").then(String::new);
        let resolved = resolve_root(&config, &env).unwrap();
        assert!(
            resolved.ends_with(".cache/huggingface/hub"),
            "an empty HF_HOME must be treated as unset, falling through to the platform \
             home-dir default, got {resolved:?}"
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

    /// #481 fix round 3: a present-but-empty `HF_ENDPOINT` must resolve to
    /// `None` (hf-hub's own default), never to a `""` endpoint URL.
    #[test]
    fn endpoint_empty_env_falls_back_to_none() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_ENDPOINT").then(String::new);
        assert_eq!(resolve_endpoint(&config, &env), None);
    }

    // --- precedence: token (config > env > <HF_HOME>/token file) ---

    #[test]
    fn token_prefers_config_over_env_and_file() {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join("token"), "file-token\n").unwrap();
        let home_path = home.path().to_str().unwrap().to_string();
        let config = ModelsConfig {
            hub_token: Some(SecretSource::Inline("config-token".into())),
            ..Default::default()
        };
        let env = move |k: &str| match k {
            "HF_TOKEN" => Some("env-token".to_string()),
            "HF_HOME" => Some(home_path.clone()),
            _ => None,
        };
        assert_eq!(
            resolve_token(&config, &env).unwrap(),
            Some("config-token".into())
        );
    }

    #[test]
    fn token_falls_back_to_env_over_file() {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join("token"), "file-token\n").unwrap();
        let home_path = home.path().to_str().unwrap().to_string();
        let config = ModelsConfig::default();
        let env = move |k: &str| match k {
            "HF_TOKEN" => Some("env-token".to_string()),
            "HF_HOME" => Some(home_path.clone()),
            _ => None,
        };
        assert_eq!(
            resolve_token(&config, &env).unwrap(),
            Some("env-token".into())
        );
    }

    #[test]
    fn token_falls_back_to_home_file_when_config_and_env_absent() {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join("token"), "file-token\n").unwrap();
        let home_path = home.path().to_str().unwrap().to_string();
        let config = ModelsConfig::default();
        let env = move |k: &str| (k == "HF_HOME").then(|| home_path.clone());
        assert_eq!(
            resolve_token(&config, &env).unwrap(),
            Some("file-token".into())
        );
    }

    #[test]
    fn token_none_when_nothing_resolves() {
        let home = tempfile::tempdir().unwrap();
        let home_path = home.path().to_str().unwrap().to_string();
        let config = ModelsConfig::default();
        let env = move |k: &str| (k == "HF_HOME").then(|| home_path.clone());
        assert_eq!(resolve_token(&config, &env).unwrap(), None);
    }

    /// #481 fix round 3, BLOCK 2: the token file resolves from `<HF_HOME>/token`
    /// -- completely INDEPENDENT of whichever tier won the cache-root
    /// precedence. `HF_HUB_CACHE` here names a wholly different directory
    /// holding no token file at all; `hf_hub::Cache::token_path`'s own "pop
    /// the cache root's last path component" arithmetic (hf-hub 0.5's
    /// comment: "Remove `\"hub\"`") would derive `parent(HF_HUB_CACHE)/token`
    /// instead and silently miss the real file -- a user with `HF_HOME` +
    /// a `huggingface-cli login` token who also sets `HF_HUB_CACHE` must
    /// still authenticate.
    #[test]
    fn token_file_resolves_from_hf_home_independent_of_hf_hub_cache() {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join("token"), "home-token\n").unwrap();
        let hub_cache = tempfile::tempdir().unwrap();
        let config = ModelsConfig::default();
        let home_path = home.path().to_str().unwrap().to_string();
        let hub_cache_path = hub_cache.path().to_str().unwrap().to_string();
        let env = move |k: &str| match k {
            "HF_HOME" => Some(home_path.clone()),
            "HF_HUB_CACHE" => Some(hub_cache_path.clone()),
            _ => None,
        };
        assert_eq!(
            resolve_token(&config, &env).unwrap(),
            Some("home-token".into()),
            "the token file must resolve from HF_HOME/token even when HF_HUB_CACHE names an \
             unrelated directory with no hub/ suffix to pop"
        );
    }

    /// #481 fix round 3, BLOCK 1 (advisory: `HF_TOKEN` mirror): a
    /// present-but-empty `HF_TOKEN` must fall through to the token file,
    /// never resolve to an empty bearer token.
    #[test]
    fn token_empty_hf_token_falls_through_to_home_file() {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join("token"), "file-token\n").unwrap();
        let home_path = home.path().to_str().unwrap().to_string();
        let config = ModelsConfig::default();
        let env = move |k: &str| match k {
            "HF_TOKEN" => Some(String::new()),
            "HF_HOME" => Some(home_path.clone()),
            _ => None,
        };
        assert_eq!(
            resolve_token(&config, &env).unwrap(),
            Some("file-token".into()),
            "an empty HF_TOKEN must be treated as absent, falling through to the token file"
        );
    }

    #[test]
    fn token_file_path_empty_hf_home_falls_back_to_injected_home_dir() {
        let env = |k: &str| (k == "HF_HOME").then(String::new);
        let path = token_file_path(&env, || Some(PathBuf::from("/injected/home")));
        assert_eq!(
            path,
            Some(PathBuf::from("/injected/home/.cache/huggingface/token"))
        );
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
        assert_eq!(resolved, dir.path().join("hub"));
    }

    // --- precedence: offline (config Some(_) wins outright > HF_HUB_OFFLINE > TRANSFORMERS_OFFLINE > false) ---

    #[test]
    fn offline_falls_back_to_hf_hub_offline_env_when_config_is_none() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_HUB_OFFLINE").then(|| "1".to_string());
        assert!(resolve_offline(&config, &env));
    }

    #[test]
    fn offline_is_false_when_neither_config_nor_env_says_so() {
        let config = ModelsConfig::default();
        assert!(!resolve_offline(&config, &no_env));
    }

    /// Direction: explicit `Some(false)` wins over `HF_HUB_OFFLINE=1` — never
    /// OR'd with the environment. Matches the other three chains' "config
    /// wins" precedence.
    #[test]
    fn offline_config_some_false_wins_over_hf_hub_offline_env() {
        let config = ModelsConfig {
            offline: Some(false),
            ..Default::default()
        };
        let env = |k: &str| (k == "HF_HUB_OFFLINE").then(|| "1".to_string());
        assert!(!resolve_offline(&config, &env));
    }

    /// Symmetric direction: explicit `Some(true)` wins even when the
    /// environment says nothing (or would say false) — `HF_HUB_OFFLINE` is
    /// consulted only when the config is `None`.
    #[test]
    fn offline_config_some_true_wins_regardless_of_env() {
        let config = ModelsConfig {
            offline: Some(true),
            ..Default::default()
        };
        assert!(resolve_offline(&config, &no_env));
    }

    /// #481 fix round 2, advisory: `TRANSFORMERS_OFFLINE` is honoured only
    /// when `HF_HUB_OFFLINE` is itself unset from the environment — matching
    /// `huggingface_hub`'s own alias behaviour.
    #[test]
    fn offline_falls_back_to_transformers_offline_when_hf_hub_offline_unset() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "TRANSFORMERS_OFFLINE").then(|| "1".to_string());
        assert!(resolve_offline(&config, &env));
    }

    #[test]
    fn offline_hf_hub_offline_wins_over_transformers_offline_when_both_set() {
        let config = ModelsConfig::default();
        let env = |k: &str| match k {
            "HF_HUB_OFFLINE" => Some("0".to_string()),
            "TRANSFORMERS_OFFLINE" => Some("1".to_string()),
            _ => None,
        };
        assert!(
            !resolve_offline(&config, &env),
            "HF_HUB_OFFLINE, even falsy, must win over TRANSFORMERS_OFFLINE -- the alias is \
             consulted only when HF_HUB_OFFLINE is absent from the environment entirely"
        );
    }

    /// #481 fix round 3, BLOCK 1: a present-but-empty `HF_HUB_OFFLINE`
    /// (`Some("")`, exactly what `std::env::var("HF_HUB_OFFLINE").ok()`
    /// yields for `HF_HUB_OFFLINE=` in a Compose/K8s env block) must NOT
    /// shadow the `TRANSFORMERS_OFFLINE` alias the way `Some("0")` correctly
    /// does above -- `huggingface_hub` itself resolves this exact shape
    /// offline (`os.environ.get("HF_HUB_OFFLINE") or
    /// os.environ.get("TRANSFORMERS_OFFLINE")`, where Python's `or` skips a
    /// `""` left side), so jammi fail-open here was a real divergence from
    /// upstream, not merely a stricter reading of it.
    ///
    /// RED at 7c581c89 (pre-`env_nonempty`): `env("HF_HUB_OFFLINE")` yields
    /// `Some(String::new())`, `Option::or_else` never fires because the
    /// `Option` is already `Some`, and `is_hf_hub_offline_truthy("")` is
    /// `false` -- `resolve_offline` returns `false` (online) instead of the
    /// `true` a present `TRANSFORMERS_OFFLINE=1` demands.
    #[test]
    fn offline_empty_hf_hub_offline_falls_through_to_transformers_offline() {
        let config = ModelsConfig::default();
        let env = |k: &str| match k {
            "HF_HUB_OFFLINE" => Some(String::new()),
            "TRANSFORMERS_OFFLINE" => Some("1".to_string()),
            _ => None,
        };
        assert!(
            resolve_offline(&config, &env),
            "an empty HF_HUB_OFFLINE must be treated as unset, falling through to \
             TRANSFORMERS_OFFLINE=1"
        );
    }

    /// #481 fix round 2, BLOCK fix: the accepted truthy set widens from
    /// `{"1", "true"}` to `huggingface_hub`'s own `ENV_VARS_TRUE_VALUES` —
    /// `{"1", "ON", "YES", "TRUE"}`, matched case-insensitively.
    #[test]
    fn hf_hub_offline_truthy_accepts_the_huggingface_hub_true_value_set() {
        assert!(is_hf_hub_offline_truthy("1"));
        assert!(is_hf_hub_offline_truthy("on"));
        assert!(is_hf_hub_offline_truthy("ON"));
        assert!(is_hf_hub_offline_truthy("On"));
        assert!(is_hf_hub_offline_truthy("yes"));
        assert!(is_hf_hub_offline_truthy("YES"));
        assert!(is_hf_hub_offline_truthy("Yes"));
        assert!(is_hf_hub_offline_truthy("true"));
        assert!(is_hf_hub_offline_truthy("TRUE"));
        assert!(is_hf_hub_offline_truthy("True"));
        assert!(is_hf_hub_offline_truthy("  true  "));
        assert!(is_hf_hub_offline_truthy("  ON  "));
    }

    #[test]
    fn hf_hub_offline_truthy_rejects_everything_else() {
        assert!(!is_hf_hub_offline_truthy("0"));
        assert!(!is_hf_hub_offline_truthy("false"));
        assert!(!is_hf_hub_offline_truthy("FALSE"));
        assert!(!is_hf_hub_offline_truthy("off"));
        assert!(!is_hf_hub_offline_truthy("OFF"));
        assert!(!is_hf_hub_offline_truthy("no"));
        assert!(!is_hf_hub_offline_truthy("NO"));
        assert!(!is_hf_hub_offline_truthy(""));
        assert!(!is_hf_hub_offline_truthy("garbage"));
        assert!(!is_hf_hub_offline_truthy("2"));
    }
}
