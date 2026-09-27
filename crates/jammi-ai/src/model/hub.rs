//! The one place `[models]` (`jammi_db::config::ModelsConfig`) turns into a
//! live Hugging Face Hub client.
//!
//! `HubSource` is built exactly ONCE per session, at the `jammi-ai` choke
//! point (`crate::session::InferenceSession::wrap`), and every call site
//! downstream (the resolver, the fine-tune worker) shares that one client.
//!
//! # The transfer is this module's own
//!
//! `hf_hub` supplies the on-disk cache layout (`hf_hub::Cache`: `blobs/`,
//! `snapshots/`, `refs/`), and nothing else: its clients build their HTTP
//! agents with no timeouts and expose no knob to set one (hf-hub 0.5
//! `api/sync.rs` `ApiBuilder::build`, `api/tokio.rs` likewise), so a server
//! or middlebox that stops sending without closing the connection would
//! block a fetch forever. `HubSource` owns the transfer instead — the
//! Hub's `resolve` endpoint, read with a client whose connect and per-read
//! timeouts are `[models] hub_idle_timeout_secs` — and writes the same
//! layout, so a warm cache is shared with any other `hf_hub` reader and a
//! cache hit issues no request at all. A transfer that goes that long
//! without a byte fails as a retryable
//! [`jammi_db::error::JammiError::Unavailable`] naming the repo and the
//! file; one that keeps making progress is never cut off. The fetch is
//! async, so a waiting transfer holds no thread: dropping it cancels it.
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
//! [`jammi_db::error::JammiError::Config`], **never** the panic `hf_hub::Cache::default()`
//! raises in the same situation.
//!
//! **Endpoint:**
//! 1. `[models] hub_endpoint`
//! 2. the `HF_ENDPOINT` environment variable
//! 3. the Hub's own endpoint, `https://huggingface.co`
//!
//! **Token** (only set on the client when one resolves — an absent token
//! means no `Authorization` header, exactly like an anonymous
//! `huggingface-cli` session). This is `huggingface_hub`'s own COMPLETE
//! determinant set for "which token is sent" (`utils/_auth.py`,
//! `constants.py:247-254`), not a jammi-specific subset of it:
//! 1. `[models] hub_token` (a [`jammi_db::config::SecretSource`] — inline or
//!    file-backed)
//! 2. the `HF_TOKEN` environment variable, when non-empty (see "empty
//!    values are absent" below)
//! 3. the `HUGGING_FACE_HUB_TOKEN` environment variable, when non-empty and
//!    `HF_TOKEN` is itself absent — `huggingface_hub`'s own LIVE legacy
//!    alias for `HF_TOKEN` (`utils/_auth.py:145-147`; not deprecated-and-
//!    ignored, upstream still reads it today), which hf-hub 0.5, the Rust
//!    crate this module wraps, does not read at all
//! 4. the token FILE: the `HF_TOKEN_PATH` environment variable, when
//!    non-empty, names the file DIRECTLY (matching `huggingface_hub`'s own
//!    `HF_TOKEN_PATH`, `constants.py:247-254`); otherwise `<HF_HOME>/token`,
//!    read directly and trimmed — `HF_HOME` here is resolved on its own
//!    (non-empty env value, else the platform home directory's
//!    `.cache/huggingface`), **independently of whichever tier won the
//!    cache-root precedence above**. This is deliberately NOT
//!    `hf_hub::Cache::token`/`token_path`, which derives the token file by
//!    popping the CACHE ROOT's last path component (hf-hub 0.5's own
//!    comment: "Remove `\"hub\"`") — an arithmetic that only recovers
//!    `HF_HOME` when the cache root actually was built as `<HF_HOME>/hub`.
//!    With `HF_HUB_CACHE` set, the cache root has no `hub` component to pop
//!    at all, so `Cache::token_path` would silently look in
//!    `parent(HF_HUB_CACHE)/token` instead — this module never makes that
//!    mistake, matching `huggingface_hub`'s own `HF_TOKEN_PATH` precedence
//!    (`utils/_auth.py`), which never varies with `HF_HUB_CACHE` either.
//!    (Ignoring `HF_TOKEN_PATH` and always reading `<HF_HOME>/token` instead
//!    would be a real divergence, not a stricter reading of upstream: with
//!    `HF_TOKEN_PATH` set naming a token file elsewhere and no token under
//!    `HF_HOME`, that would send no `Authorization` header where
//!    `huggingface_hub` authenticates — a silent 401 on a gated repo.)
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
//!    hf-hub, the Rust crate whose cache layout this module writes, does not
//!    read either variable at all — see this module's private `resolve_offline`/
//!    `is_hf_hub_offline_truthy`)
//! 3. `false`
//!
//! # Empty values are absent, and every value is trimmed
//!
//! Every one of this module's eight env reads (`HF_HUB_OFFLINE`,
//! `TRANSFORMERS_OFFLINE`, `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`,
//! `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH`) goes through one
//! helper, `env_nonempty`: a value that is *present but empty* (after
//! trimming ASCII whitespace) is treated exactly like an ABSENT variable,
//! never like a real value that happens to be `""`. A Compose/K8s env block
//! naming a variable with no value (`HF_HUB_OFFLINE:`) or an operator's
//! shell `export HF_TOKEN=` both produce exactly this shape — the
//! production closure (`&|k: &str| std::env::var(k).ok()`) yields
//! `Some("")`, not `None`, for either, and every tier downstream of
//! `env_nonempty` falls back exactly as if the variable had never been set
//! at all. `env_nonempty` also returns the TRIMMED value, not the raw one —
//! `HF_HOME=" /data/hf"` resolves to the cache root `/data/hf/hub`, never
//! the untrimmed `" /data/hf/hub"`, whose leading space would make
//! [`std::path::Path::is_absolute`] false and silently root the whole cache
//! (and, independently, the token file) under the current working directory
//! instead. The same trimming reaches `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`
//! (a padded bearer token would otherwise fail the Hub's own header
//! validation) and `HF_ENDPOINT`/`HF_TOKEN_PATH` identically — one helper,
//! not a per-tier judgment call about which env var "needs" trimming.
//!
//! This is a strict SUPERSET of `huggingface_hub`'s own handling for most of
//! the eight: the offline alias's `os.environ.get(A) or os.environ.get(B)`
//! and `HF_TOKEN`'s own `_clean_token` (`huggingface_hub/utils/_auth.py`,
//! which strips `\r`/`\n`/spaces — a narrower set than this module's
//! `str::trim`, itself a superset — and maps the emptied result to `None`)
//! already treat an empty value as absent, same as here. For
//! `HF_HOME`/`HF_ENDPOINT`, upstream's own `os.environ.get(KEY, default)`
//! does NOT — an empty string is present there, so it wins over the
//! default, and upstream itself would resolve a CWD-relative cache root or
//! an empty-string endpoint in that case. This module chooses the safer
//! reading everywhere instead of replicating that upstream inconsistency:
//! every one of the eight variables falls back to its safe default when
//! empty (home cache, default endpoint, no token, offline resolved from the
//! non-empty variable only) — never toward the network, and never toward a
//! nonsense relative path.
//!
//! One case is a genuine DIVERGENCE, not merely a stricter superset, and it
//! fails in OPPOSITE directions: a *whitespace-only* (not empty) value —
//! `HF_HUB_OFFLINE=" "` — is present-but-blank after trimming, so this
//! module's `env_nonempty` treats it as absent and falls through to
//! `TRANSFORMERS_OFFLINE`. Upstream's `os.environ.get("HF_HUB_OFFLINE") or
//! os.environ.get("TRANSFORMERS_OFFLINE")` does not: Python's `bool(" ")` is
//! `True` (a whitespace-only string is non-empty, so `or` never evaluates
//! its right side at all), so upstream STOPS at `" "`, and `_is_true(" ")`
//! is `False` — upstream resolves ONLINE regardless of whatever
//! `TRANSFORMERS_OFFLINE` says. This module's arm fails TOWARD offline (the
//! safe direction for an air-gap knob — see `is_hf_hub_offline_truthy`'s own
//! doc for the identical direction argument about its trimming);
//! upstream's arm fails TOWARD online. Neither reading is a bug in
//! isolation — this module simply never reproduces upstream's specific
//! blank-string exception to its own alias rule.
//!
//! The `HF_HOME`/`HF_HUB_CACHE`/`HF_ENDPOINT`/`HF_TOKEN`/
//! `HUGGING_FACE_HUB_TOKEN`/`HF_TOKEN_PATH`/`HF_HUB_OFFLINE`/
//! `TRANSFORMERS_OFFLINE` fallbacks read the process environment **here**,
//! through the `env` closure the caller passes — never inside
//! `jammi_db::config::JammiConfig::load_from` (which stays process-env-free
//! by construction). Production passes `&|k: &str| std::env::var(k).ok()`; a
//! test passes a placeholder map.
//!
//! The cache is always `hf_hub::Cache::new` over the root resolved above,
//! never `Cache::from_env()`/`Cache::default()` — those re-derive the root
//! a second time, independently of the precedence above, and
//! `Cache::default()` panics when no home directory resolves.
//!
//! # The `offline` promise is Hub-only
//!
//! `ModelsConfig::offline` refuses every *Hub network* fetch — every call
//! site that reaches `HubSource::model` checks it first. That is two call sites,
//! both after their own catalog lookup and both refusing by name when no
//! catalog row resolved the model (a warm Hub cache directory with no
//! catalog row is still a miss: the catalog, not the on-disk cache, is
//! offline's source of truth): `ModelResolver::resolve`'s `HuggingFace`
//! arm, and the fine-tune worker's `build_encoder_adapters` HF fallback
//! (reached when a fine-tune job's BASE model has a catalog row but no
//! location yet — i.e. it has never been resolved before). It does
//! **not** reach the fine-tune worker's ADAPTER fetch for an
//! already-trained model: that path always reads the adapter bundle
//! through the artifact store (object storage), never the Hub, offline or
//! not.
use std::path::{Path, PathBuf};
use std::time::Duration;

use hf_hub::{Cache, CacheRepo, Repo};
use jammi_db::config::ModelsConfig;
use jammi_db::error::{JammiError, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_RANGE, LOCATION, RANGE};
use reqwest::{redirect, Client, RequestBuilder, Response, StatusCode, Url};
use tokio::io::AsyncWriteExt;

/// The Hub's own endpoint, used when neither `[models] hub_endpoint` nor
/// `HF_ENDPOINT` names another.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// How long a Hub transfer may go without receiving a byte when `[models]
/// hub_idle_timeout_secs` is unset. It bounds silence, not duration: a
/// multi-gigabyte weights file that keeps arriving is never cut off, while a
/// connection that stops sending fails within this interval.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// How many relative redirects a metadata probe follows (the Hub answers a
/// renamed repo with one) before it refuses the chain as a loop.
const MAX_RELATIVE_REDIRECTS: usize = 10;

/// A Hugging Face Hub client built once from `[models]`, shared by every
/// jammi-ai call site that talks to the Hub. See the module docs for the
/// precedence chain, the `offline` promise, and why the transfer is this
/// module's own.
///
/// `Debug` is hand-written, NOT derived: each client's default headers hold
/// the resolved bearer token as an `Authorization` value once
/// [`HubSource::from_config`] has set one — the same class of leak
/// `jammi_db::config::secret::Secret` exists to prevent everywhere else in
/// this config. A `{:?}` of a `HubSource` must never reach it.
#[derive(Clone)]
pub struct HubSource {
    cache: Cache,
    endpoint: String,
    /// Follows redirects: a file body, which the Hub hands to its CDN.
    client: Client,
    /// Follows none: the metadata probe reads the commit and etag headers
    /// off the Hub's own answer, before a redirect to the CDN drops them.
    probe: Client,
    idle: Duration,
    offline: bool,
}

impl std::fmt::Debug for HubSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubSource")
            .field("endpoint", &self.endpoint)
            .field("client", &"<redacted: may hold a Hub bearer token>")
            .field("idle", &self.idle)
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
        let cache = Cache::new(resolve_root(config, env)?);
        let endpoint = resolve_endpoint(config, env)
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string())
            .trim_end_matches('/')
            .to_string();
        let token = resolve_token(config, env)?;
        let idle = config
            .hub_idle_timeout_secs
            .map_or(DEFAULT_IDLE_TIMEOUT, Duration::from_secs);
        Ok(Self {
            cache,
            endpoint,
            client: build_client(token.as_deref(), idle, redirect::Policy::default())?,
            probe: build_client(token.as_deref(), idle, redirect::Policy::none())?,
            idle,
            offline: resolve_offline(config, env),
        })
    }

    /// Whether `[models] offline` is set — see the module docs' "offline
    /// promise is Hub-only" section for exactly what this refuses.
    pub fn offline(&self) -> bool {
        self.offline
    }

    /// The model repository `repo_id` (`owner/name`) at its `main` revision.
    pub fn model(&self, repo_id: &str) -> HubRepo<'_> {
        HubRepo {
            hub: self,
            repo: Repo::model(repo_id.to_string()),
        }
    }
}

/// A client whose every request carries `token` as a bearer token (when one
/// resolved) and fails after `idle` without progress: to connect, and on
/// each read — reqwest resets its read timeout after every successful read,
/// so it bounds silence, never a transfer that keeps arriving
/// (<https://docs.rs/reqwest/0.12/reqwest/struct.ClientBuilder.html#method.read_timeout>).
/// reqwest drops the `Authorization` header on a redirect to another host,
/// so the token never reaches the CDN a file body is served from.
fn build_client(
    token: Option<&str>,
    idle: Duration,
    redirects: redirect::Policy,
) -> Result<Client> {
    let mut headers = HeaderMap::new();
    if let Some(token) = token {
        let mut bearer = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|e| {
            JammiError::Config(format!(
                "the Hugging Face Hub token is not a valid header: {e}"
            ))
        })?;
        bearer.set_sensitive(true);
        headers.insert(AUTHORIZATION, bearer);
    }
    Client::builder()
        .default_headers(headers)
        .user_agent(concat!("jammi/", env!("CARGO_PKG_VERSION")))
        .redirect(redirects)
        .connect_timeout(idle)
        .read_timeout(idle)
        .build()
        .map_err(|e| JammiError::Config(format!("Hugging Face Hub client init failed: {e}")))
}

/// One Hub repository, read through its [`HubSource`]: its files into the
/// shared on-disk cache, and its listing.
pub struct HubRepo<'a> {
    hub: &'a HubSource,
    repo: Repo,
}

/// What a file's metadata probe learnt: the commit the revision resolves
/// to, the blob's etag (its cache key), and its size in bytes.
struct FileMetadata {
    commit: String,
    etag: String,
    size: u64,
}

/// The repo listing's shape: every file the repo holds, by its path.
#[derive(serde::Deserialize)]
struct Listing {
    siblings: Vec<Sibling>,
}

#[derive(serde::Deserialize)]
struct Sibling {
    rfilename: String,
}

impl HubRepo<'_> {
    /// The local path of `filename`: straight from the cache when it is
    /// there (no request at all), else downloaded into it. `Ok(None)` when
    /// the Hub has no such file; a Hub that cannot be reached, or stops
    /// sending, is [`JammiError::Unavailable`] naming the repo and the file.
    pub async fn get(&self, filename: &str) -> Result<Option<PathBuf>> {
        match self.cache().get(filename) {
            Some(path) => Ok(Some(path)),
            None => self.download(filename).await,
        }
    }

    /// Every file the repo holds, by its path in the repo.
    pub async fn files(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/{}", self.hub.endpoint, self.repo.api_url());
        let response = self.send(self.hub.client.get(&url), "<listing>").await?;
        let response = self.expect_success(response, "<listing>")?;
        let listing: Listing = response
            .json()
            .await
            .map_err(|e| self.unreachable("<listing>", &e))?;
        Ok(listing.siblings.into_iter().map(|s| s.rfilename).collect())
    }

    fn cache(&self) -> CacheRepo {
        self.hub.cache.repo(self.repo.clone())
    }

    /// Download `filename` into the cache layout `hf_hub::Cache` reads —
    /// the body under `blobs/<etag>`, a relative symlink to it under
    /// `snapshots/<commit>/<filename>`, and `refs/<revision>` naming the
    /// commit — so a file this client fetched is a cache hit for any other
    /// reader of the same cache, and the reverse.
    async fn download(&self, filename: &str) -> Result<Option<PathBuf>> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.hub.endpoint,
            self.repo.url(),
            self.repo.url_revision(),
            filename
        );
        let Some(metadata) = self.metadata(&url, filename).await? else {
            return Ok(None);
        };
        let cache = self.cache();
        let blob = cache.blob_path(&metadata.etag);
        if let Some(blobs) = blob.parent() {
            tokio::fs::create_dir_all(blobs).await?;
        }
        if !tokio::fs::try_exists(&blob).await? {
            self.fetch(&url, filename, &blob, metadata.size).await?;
        }
        let mut pointer = cache.pointer_path(&metadata.commit);
        pointer.push(filename);
        if let Some(snapshot) = pointer.parent() {
            tokio::fs::create_dir_all(snapshot).await?;
        }
        link_pointer(&metadata.etag, &pointer, filename)?;
        cache.create_ref(&metadata.commit)?;
        Ok(Some(pointer))
    }

    /// Probe `url` for the file's commit, etag and size without its body: a
    /// one-byte range, following only relative redirects (a renamed repo),
    /// so the commit and etag headers come off the Hub's own answer. An
    /// absolute redirect hands the body to the CDN, whose one-byte answer
    /// carries the size. `Ok(None)` is the Hub saying there is no such file.
    async fn metadata(&self, url: &str, filename: &str) -> Result<Option<FileMetadata>> {
        let mut url = Url::parse(url).map_err(|e| self.malformed(filename, e))?;
        let mut hops = 0;
        let response = loop {
            let response = self
                .send(
                    self.hub.probe.get(url.clone()).header(RANGE, "bytes=0-0"),
                    filename,
                )
                .await?;
            let next = match location(&response) {
                Some(next) if response.status().is_redirection() && Url::parse(next).is_err() => {
                    next.to_string()
                }
                _ => break response,
            };
            hops += 1;
            if hops > MAX_RELATIVE_REDIRECTS {
                return Err(self.unavailable(
                    filename,
                    format!("more than {MAX_RELATIVE_REDIRECTS} relative redirects"),
                ));
            }
            url = url.join(&next).map_err(|e| self.malformed(filename, e))?;
        };
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let commit = header(&response, "x-repo-commit")
            .ok_or_else(|| self.missing(filename, "x-repo-commit"))?;
        let etag = header(&response, "x-linked-etag")
            .or_else(|| header(&response, "etag"))
            .ok_or_else(|| self.missing(filename, "etag"))?
            .replace('"', "");
        let sized = if response.status().is_redirection() {
            let cdn = location(&response).ok_or_else(|| self.missing(filename, "location"))?;
            self.send(
                self.hub.client.get(cdn).header(RANGE, "bytes=0-0"),
                filename,
            )
            .await?
        } else {
            response
        };
        let sized = self.expect_success(sized, filename)?;
        let size = header(&sized, CONTENT_RANGE.as_str())
            .and_then(|range| range.rsplit('/').next().and_then(|s| s.parse().ok()))
            .ok_or_else(|| self.missing(filename, "content-range"))?;
        Ok(Some(FileMetadata { commit, etag, size }))
    }

    /// Stream `url`'s body into `blob`, through a partial file renamed into
    /// place only once all `size` bytes arrived, so a reader never sees a
    /// torn blob. A partial file left by a failed transfer is removed.
    async fn fetch(&self, url: &str, filename: &str, blob: &Path, size: u64) -> Result<()> {
        let partial = blob.with_extension(format!("{}.part", uuid::Uuid::new_v4().simple()));
        let fetched = self.write_body(url, filename, &partial, size).await;
        match fetched {
            Ok(()) => Ok(tokio::fs::rename(&partial, blob).await?),
            Err(failed) => {
                if let Err(e) = tokio::fs::remove_file(&partial).await {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(path = %partial.display(), error = %e, "Hub: a failed transfer's partial file was not removed");
                    }
                }
                Err(failed)
            }
        }
    }

    async fn write_body(&self, url: &str, filename: &str, partial: &Path, size: u64) -> Result<()> {
        let response = self.send(self.hub.client.get(url), filename).await?;
        let mut response = self.expect_success(response, filename)?;
        let mut file = tokio::fs::File::create(partial).await?;
        let mut written: u64 = 0;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| self.unreachable(filename, &e))?
        {
            file.write_all(&chunk).await?;
            written += chunk.len() as u64;
        }
        file.flush().await?;
        if written == size {
            Ok(())
        } else {
            Err(self.unavailable(
                filename,
                format!("the transfer ended after {written} of {size} bytes"),
            ))
        }
    }

    async fn send(&self, request: RequestBuilder, filename: &str) -> Result<Response> {
        request
            .send()
            .await
            .map_err(|e| self.unreachable(filename, &e))
    }

    /// `response` when it succeeded; otherwise the refusal it is. A server
    /// fault or rate limit is retryable [`JammiError::Unavailable`]; any
    /// other answer (a gated repo refusing the token, a malformed request)
    /// is a [`JammiError::Model`] naming the file.
    fn expect_success(&self, response: Response, filename: &str) -> Result<Response> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let reason = format!("the Hub answered {status}");
        if status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS {
            Err(self.unavailable(filename, reason))
        } else {
            Err(JammiError::Model {
                model_id: self.repo_id(),
                message: format!("{filename}: {reason}"),
            })
        }
    }

    /// A transfer that could not be made or stopped making progress.
    fn unreachable(&self, filename: &str, error: &reqwest::Error) -> JammiError {
        let reason = if error.is_timeout() {
            format!(
                "no progress for {}s (`[models] hub_idle_timeout_secs`): the transfer stalled",
                self.hub.idle.as_secs()
            )
        } else {
            error.to_string()
        };
        self.unavailable(filename, reason)
    }

    fn unavailable(&self, filename: &str, reason: String) -> JammiError {
        JammiError::Unavailable {
            resource: format!("hf://{}/{filename}", self.repo_id()),
            reason,
        }
    }

    fn missing(&self, filename: &str, header: &str) -> JammiError {
        JammiError::Model {
            model_id: self.repo_id(),
            message: format!("{filename}: the Hub's answer carries no `{header}` header"),
        }
    }

    fn malformed(&self, filename: &str, error: impl std::fmt::Display) -> JammiError {
        JammiError::Model {
            model_id: self.repo_id(),
            message: format!("{filename}: not a valid Hub URL: {error}"),
        }
    }

    fn repo_id(&self) -> String {
        self.repo.url()
    }
}

/// The `Location` a redirect names, when it is readable.
fn location(response: &Response) -> Option<&str> {
    response.headers().get(LOCATION)?.to_str().ok()
}

/// A header's value, when present and readable.
fn header(response: &Response, name: &str) -> Option<String> {
    Some(response.headers().get(name)?.to_str().ok()?.to_string())
}

/// Point `pointer` (`snapshots/<commit>/<filename>`) at `blob`
/// (`blobs/<etag>`) by a relative symlink, as `hf_hub` does, so the cache
/// stays valid when its root moves (the engine builds for Unix targets
/// only). Another writer having linked it first is the same outcome.
fn link_pointer(etag: &str, pointer: &Path, filename: &str) -> Result<()> {
    if pointer.exists() {
        return Ok(());
    }
    // Up out of the file's own subdirectories and `snapshots/<commit>`, to
    // the repo directory `blobs/` sits in.
    let depth = Path::new(filename).components().count() + 1;
    let mut target: PathBuf = std::iter::repeat_n("..", depth).collect();
    target.push("blobs");
    target.push(etag);
    match std::os::unix::fs::symlink(&target, pointer) {
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => Err(e.into()),
        _ => Ok(()),
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
/// container whose numeric UID has no `/etc/passwd` entry at all (e.g. the
/// `nonroot`/`65532` runtime users this repo's Dockerfile provisions).
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
/// `HF_ENDPOINT` (non-empty) > `None` (the Hub's own endpoint,
/// [`DEFAULT_ENDPOINT`], applies).
fn resolve_endpoint(config: &ModelsConfig, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    config
        .hub_endpoint
        .clone()
        .or_else(|| env_nonempty(env, "HF_ENDPOINT"))
}

/// Resolve the bearer token per the module docs' precedence — `hub_token` >
/// `HF_TOKEN` (non-empty, trimmed) > `HUGGING_FACE_HUB_TOKEN` (non-empty,
/// trimmed, only when `HF_TOKEN` is itself absent — `huggingface_hub`'s own
/// live legacy alias, `utils/_auth.py:145-147`) > the token FILE
/// ([`token_file_path`]: `HF_TOKEN_PATH` when set, else `<HF_HOME>/token`).
/// `Ok(None)` means "send no `Authorization` header", not a failure. This is
/// `huggingface_hub`'s own complete determinant set for "which token is
/// sent" (see the module docs' "Precedence" section for the citation), not
/// a jammi-specific subset of it.
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
    if let Some(token) = env_nonempty(env, "HUGGING_FACE_HUB_TOKEN") {
        return Ok(Some(token));
    }
    Ok(token_file_path(env, home_dir).and_then(|path| read_token_file(&path)))
}

/// The Hub token FILE: the `HF_TOKEN_PATH` environment variable, when
/// non-empty, names the file DIRECTLY — matching `huggingface_hub`'s own
/// `HF_TOKEN_PATH` (`constants.py:247-254`); otherwise `<HF_HOME>/token`,
/// where `HF_HOME` is resolved on its own: a non-empty `HF_HOME` env value,
/// else the platform home directory's `.cache/huggingface`. Both tiers are
/// deliberately computed WITHOUT reference to the Hub cache root
/// [`resolve_root`] resolved: `hf_hub::Cache::token_path` derives the token
/// file by popping the CACHE ROOT's last path component (hf-hub 0.5's own
/// comment: "Remove `\"hub\"`"), which is only a correct recovery of
/// `HF_HOME` when the cache root was actually built as `<HF_HOME>/hub`.
/// When `HF_HUB_CACHE` wins the cache-root precedence instead, the cache
/// root has no `hub` component to pop at all, so `Cache::token_path` would
/// silently look in `parent(HF_HUB_CACHE)/token` — a user with `HF_HOME` set
/// (and a `huggingface-cli login` token there) who ALSO sets `HF_HUB_CACHE`
/// would then silently lose authentication. This function never takes that
/// path: it reads `HF_TOKEN_PATH`/`HF_HOME` itself, independently of
/// whichever tier won the cache-root precedence. Ignoring `HF_TOKEN_PATH`
/// entirely is the identical class of bug in a second home: a user who
/// points `HF_TOKEN_PATH` at a token file outside `HF_HOME` (or with no
/// `HF_HOME`/`hub_cache_dir` token file at all) would silently authenticate
/// with `huggingface_hub` but not with `jammi-ai` — a silent 401 on a gated
/// repo, not a loud one.
fn token_file_path(
    env: &dyn Fn(&str) -> Option<String>,
    home_dir: impl FnOnce() -> Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(path) = env_nonempty(env, "HF_TOKEN_PATH") {
        return Some(PathBuf::from(path));
    }
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
/// `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`,
/// `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH`) goes through this one helper —
/// see the module docs' "empty values are absent, and every value is
/// trimmed" section for why and for the exact upstream comparison. A
/// present-but-empty value (after trimming ASCII whitespace) is treated as
/// ABSENT, not as a real value that happens to be `""`. The returned value
/// is the TRIMMED string, not the raw one — `HF_HOME=" /data/hf"` resolves
/// the cache root to `/data/hf/hub`, never the untrimmed
/// `" /data/hf/hub"`, whose leading space would fail
/// [`std::path::Path::is_absolute`] and silently root the cache (and the
/// token file) under the current working directory instead.
fn env_nonempty(env: &dyn Fn(&str) -> Option<String>, key: &str) -> Option<String> {
    env(key)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
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
/// precedence for the citation); hf-hub 0.5, whose cache layout `HubSource`
/// writes, does not read either variable at all, so this crate reads and
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

    // --- env_nonempty: present-but-empty is absent ---

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

    /// `HF_HUB_CACHE` (used directly, no `hub/` appended) beats `HF_HOME`
    /// when `[models] hub_cache_dir` is unset — `huggingface_hub` honours
    /// `HF_HUB_CACHE` above `HF_HOME`, and this branch's own
    /// `jammi-encoders` live-hub harness reads it at that precedence too.
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

    /// A present-but-empty `HF_HUB_CACHE` must fall through to `HF_HOME`,
    /// not be used as a literal empty/CWD-relative cache root.
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

    /// A present-but-empty `HF_HOME`, with `HF_HUB_CACHE` also absent, must
    /// fall through to the platform home-dir default rather than resolving
    /// to a `""`/CWD-relative root.
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

    /// A present-but-empty `HF_ENDPOINT` must resolve to `None` (hf-hub's
    /// own default), never to a `""` endpoint URL.
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

    /// The token file resolves from `<HF_HOME>/token` -- completely
    /// INDEPENDENT of whichever tier won the cache-root precedence.
    /// `HF_HUB_CACHE` here names a wholly different directory
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

    /// A present-but-empty `HF_TOKEN` must fall through to the token file,
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

    // --- HF_TOKEN_PATH / HUGGING_FACE_HUB_TOKEN / trimming ---

    /// A token resolves from the file `HF_TOKEN_PATH` names directly, even
    /// though `HF_HOME` here resolves to a dir with NO `token` file inside
    /// it at all -- `huggingface_hub` authenticates in this exact shape
    /// (`constants.py:247-254`); ignoring `HF_TOKEN_PATH` would send no
    /// `Authorization` header, a silent 401 on a gated repo.
    #[test]
    fn token_resolves_from_hf_token_path_even_with_token_file_less_hf_home() {
        let hf_home = tempfile::tempdir().unwrap(); // deliberately no `token` file inside
        let token_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(token_file.path(), "path-token\n").unwrap();
        let config = ModelsConfig::default();
        let hf_home_path = hf_home.path().to_str().unwrap().to_string();
        let token_path = token_file.path().to_str().unwrap().to_string();
        let env = move |k: &str| match k {
            "HF_HOME" => Some(hf_home_path.clone()),
            "HF_TOKEN_PATH" => Some(token_path.clone()),
            _ => None,
        };
        assert_eq!(
            resolve_token(&config, &env).unwrap(),
            Some("path-token".into()),
            "HF_TOKEN_PATH must name the token file directly, independent of HF_HOME"
        );
    }

    /// `HF_TOKEN_PATH` wins over `<HF_HOME>/token` even when the latter ALSO
    /// carries a (different) token -- `HF_TOKEN_PATH` is the higher tier in
    /// the file-resolution precedence, not merely a fallback for when
    /// `HF_HOME` has nothing.
    #[test]
    fn token_file_path_hf_token_path_wins_over_hf_home_token_file() {
        let hf_home = tempfile::tempdir().unwrap();
        std::fs::write(hf_home.path().join("token"), "home-token\n").unwrap();
        let token_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(token_file.path(), "path-token\n").unwrap();
        let hf_home_path = hf_home.path().to_str().unwrap().to_string();
        let token_path = token_file.path().to_str().unwrap().to_string();
        let env = move |k: &str| match k {
            "HF_HOME" => Some(hf_home_path.clone()),
            "HF_TOKEN_PATH" => Some(token_path.clone()),
            _ => None,
        };
        assert_eq!(
            token_file_path(&env, || None),
            Some(token_file.path().to_path_buf())
        );
    }

    /// An empty `HF_TOKEN_PATH` is absent, same as every other env read here
    /// -- falls through to `<HF_HOME>/token`, never a literal empty path.
    #[test]
    fn token_file_path_empty_hf_token_path_falls_through_to_hf_home() {
        let env = |k: &str| match k {
            "HF_TOKEN_PATH" => Some(String::new()),
            "HF_HOME" => Some("/env/home".to_string()),
            _ => None,
        };
        assert_eq!(
            token_file_path(&env, || None),
            Some(PathBuf::from("/env/home/token"))
        );
    }

    /// `HUGGING_FACE_HUB_TOKEN` -- `huggingface_hub`'s own LIVE legacy alias
    /// (`utils/_auth.py:145-147`, not deprecated-and-ignored) -- is honoured
    /// when `HF_TOKEN` is absent.
    #[test]
    fn token_falls_back_to_legacy_hugging_face_hub_token_env() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HUGGING_FACE_HUB_TOKEN").then(|| "legacy".to_string());
        assert_eq!(resolve_token(&config, &env).unwrap(), Some("legacy".into()));
    }

    /// `HF_TOKEN` wins over `HUGGING_FACE_HUB_TOKEN` when both are set --
    /// the legacy alias is consulted only when the current name is absent.
    #[test]
    fn token_hf_token_wins_over_legacy_hugging_face_hub_token() {
        let config = ModelsConfig::default();
        let env = |k: &str| match k {
            "HF_TOKEN" => Some("current".to_string()),
            "HUGGING_FACE_HUB_TOKEN" => Some("legacy".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_token(&config, &env).unwrap(),
            Some("current".into())
        );
    }

    /// `env_nonempty` returns the TRIMMED value, not the raw one -- a padded
    /// `HF_TOKEN` must resolve to the bearer the Hub actually expects, never
    /// a value with leading/trailing whitespace or a trailing newline baked
    /// in. Filtering on `value.trim()` but returning the raw `value` would
    /// yield `Some("  padded\n")` here.
    #[test]
    fn token_hf_token_env_is_trimmed() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_TOKEN").then(|| "  padded\n".to_string());
        assert_eq!(resolve_token(&config, &env).unwrap(), Some("padded".into()));
    }

    /// A padded `HF_HOME` must resolve to a TRIMMED, absolute
    /// cache root -- the untrimmed raw value's leading space would fail
    /// `Path::is_absolute` and silently root the cache under the current
    /// working directory instead.
    #[test]
    fn root_hf_home_env_is_trimmed_never_cwd_relative() {
        let config = ModelsConfig::default();
        let env = |k: &str| (k == "HF_HOME").then(|| " /data/hf".to_string());
        let resolved = resolve_root(&config, &env).unwrap();
        assert_eq!(resolved, PathBuf::from("/data/hf/hub"));
        assert!(
            resolved.is_absolute(),
            "a padded HF_HOME must not resolve to a CWD-relative root, got {resolved:?}"
        );
    }

    /// `env_nonempty` itself: the unit-level oracle for the trimming
    /// property every tier above relies on.
    #[test]
    fn env_nonempty_returns_the_trimmed_value_not_the_raw_one() {
        let env = |k: &str| (k == "PADDED").then(|| "  padded-value  \n".to_string());
        assert_eq!(
            env_nonempty(&env, "PADDED"),
            Some("padded-value".to_string())
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

    /// `hf_hub::Cache::default()` (what `ApiBuilder::new()` reaches) panics
    /// outright —
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

    /// `TRANSFORMERS_OFFLINE` is honoured only when `HF_HUB_OFFLINE` is
    /// itself unset from the environment — matching `huggingface_hub`'s own
    /// alias behaviour.
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

    /// A present-but-empty `HF_HUB_OFFLINE` (`Some("")`, exactly what
    /// `std::env::var("HF_HUB_OFFLINE").ok()` yields for `HF_HUB_OFFLINE=`
    /// in a Compose/K8s env block) must NOT shadow the
    /// `TRANSFORMERS_OFFLINE` alias the way `Some("0")` correctly does
    /// above -- `huggingface_hub` itself resolves this exact shape offline
    /// (`os.environ.get("HF_HUB_OFFLINE") or
    /// os.environ.get("TRANSFORMERS_OFFLINE")`, where Python's `or` skips a
    /// `""` left side); failing open here would be a real divergence from
    /// upstream, not merely a stricter reading of it.
    ///
    /// Reading the raw `env("HF_HUB_OFFLINE")` instead of `env_nonempty`
    /// yields `Some(String::new())`, so `Option::or_else` never fires and
    /// `is_hf_hub_offline_truthy("")` is `false` -- `resolve_offline` would
    /// return `false` (online) instead of the `true` a present
    /// `TRANSFORMERS_OFFLINE=1` demands.
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

    /// The accepted truthy set is `huggingface_hub`'s own
    /// `ENV_VARS_TRUE_VALUES` — `{"1", "ON", "YES", "TRUE"}`, matched
    /// case-insensitively.
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
