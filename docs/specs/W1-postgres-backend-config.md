# W1 — Postgres backend selection in `JammiConfig`

**Status:** spec — pending review
**Owner:** TBD
**Estimated effort:** 1 week
**Workstream dependencies:** W00 (uses renamed `jammi-db` crate)
**Workstreams blocked by this:** W8 (SaaS deploy needs Postgres)

## Motivation

`PostgresBackend` impl already exists at `crates/jammi-db/src/catalog/backend_postgres.rs:31` (post-W00 rename). The session already has `with_backend` constructors (`crates/jammi-db/src/session.rs:97, 110`) that accept a caller-supplied `BackendImpl`. The `MutableBackend` already branches on `BackendKind::Sqlite | BackendKind::Postgres` (session.rs:168-175). Everything for Postgres operation exists at the substrate level.

The gap is **config-driven backend selection**: `JammiConfig` has no field declaring which backend to use, and `InferenceSession::new(config)` always constructs SQLite via `Catalog::open_with_tenant(&config.artifact_dir, ...)`. SaaS deploys need Postgres but cannot reach it via the default constructor — they have to use the lower-level `with_backend` path and construct the backend manually. That works for tests but is the wrong shape for production deploy code.

W1 closes this gap so a deployer can pick Postgres via config and `InferenceSession::new(config)` does the right thing.

## Current state (verified at spec time)

- `crates/jammi-engine/src/config.rs` — `JammiConfig` has fields: `artifact_dir`, `engine`, `gpu`, `inference`, `embedding`, `fine_tuning`, `cache`, `server`, `logging`. **No `catalog` field.**
- `crates/jammi-engine/src/session.rs:58` — `pub async fn new(config: JammiConfig) -> Result<Self>` constructs SQLite via `Catalog::open_with_tenant(&config.artifact_dir, ...)`. Wired to the default in-process broker.
- `crates/jammi-engine/src/session.rs:97` — `pub async fn with_backend(config, backend)` — accepts caller-supplied backend; the path tests use today
- `crates/jammi-engine/src/session.rs:110` — `pub async fn with_backend_and_broker(...)` — composable
- `crates/jammi-engine/src/catalog/backend.rs` — `BackendKind { Sqlite, Postgres }` enum + `BackendImpl` enum wrapping both
- `crates/jammi-engine/src/catalog/backend_postgres.rs:31` — `PostgresBackend` impl exists; built from `sqlx::PgPool`
- `crates/jammi-enterprise/crates/jammi-enterprise-server/src/main.rs:33` — `InferenceSession::new(JammiConfig::default())` — hardcoded default; no config loading

## Change

### 1. Add `CatalogConfig` enum to `JammiConfig`

```rust
// crates/jammi-db/src/config.rs

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct JammiConfig {
    pub artifact_dir: PathBuf,
    pub engine: EngineConfig,
    pub gpu: GpuConfig,
    pub inference: InferenceConfig,
    pub embedding: EmbeddingConfig,
    pub fine_tuning: FineTuningConfig,
    pub cache: CacheConfig,
    pub server: ServerConfig,
    pub logging: LoggingConfig,
    /// Catalog backend selection. Default: Sqlite under artifact_dir.
    pub catalog: CatalogConfig,
}

/// Catalog backend selection. The substrate's mutable companion tables
/// and the catalog itself share this backend.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogConfig {
    /// SQLite under artifact_dir. The laptop/dev default.
    Sqlite {
        /// Override the default path; defaults to `{artifact_dir}/catalog.db`.
        #[serde(default)]
        path: Option<PathBuf>,
    },
    /// Postgres (or compatible) catalog. Used for SaaS deployments and
    /// self-hosted production.
    Postgres {
        /// Connection URL, e.g. `postgres://user:pass@host:5432/jammi`.
        url: String,
        /// Pool size. Default: 8.
        #[serde(default = "default_pool_size")]
        pool_size: u32,
        /// Optional max connection lifetime in seconds.
        #[serde(default)]
        max_lifetime_secs: Option<u32>,
    },
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self::Sqlite { path: None }
    }
}

fn default_pool_size() -> u32 { 8 }
```

### 2. Wire `InferenceSession::new` to build backend from config

`crates/jammi-db/src/session.rs:58` — `InferenceSession::new(config)` becomes:

```rust
pub async fn new(config: JammiConfig) -> Result<Self> {
    let backend = match &config.catalog {
        CatalogConfig::Sqlite { path } => {
            let p = path.clone()
                .unwrap_or_else(|| config.artifact_dir.join("catalog.db"));
            BackendImpl::sqlite_from_path(&p).await?
        }
        CatalogConfig::Postgres { url, pool_size, max_lifetime_secs } => {
            BackendImpl::postgres_from_url(url, *pool_size, *max_lifetime_secs).await?
        }
    };
    Self::with_backend(config, backend).await
}
```

Existing `with_backend` and `with_backend_and_broker` are unchanged (they remain available for tests / custom deploys composing their own backend).

`BackendImpl::sqlite_from_path` and `BackendImpl::postgres_from_url` are new factory methods on the existing `BackendImpl` enum. They open the underlying connection (`rusqlite` or `sqlx::PgPool::connect`) and wrap into the enum.

### 3. Trigger broker config (concurrent change in the same crate)

Today `InferenceSession::new` uses the default in-process broker. SaaS deploys need a clustered broker (NATS JetStream).

Add `BrokerConfig` to `JammiConfig` similarly:

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrokerConfig {
    InMemory,
    JetStream {
        url: String,
        #[serde(default)]
        credentials_path: Option<PathBuf>,
    },
}

impl Default for BrokerConfig {
    fn default() -> Self { Self::InMemory }
}
```

`InferenceSession::new` constructs the broker similarly:

```rust
let broker: Arc<dyn TriggerBroker> = match &config.broker {
    BrokerConfig::InMemory => Arc::new(InMemoryBroker::new()),
    BrokerConfig::JetStream { url, credentials_path } => {
        Arc::new(JetStreamBroker::connect(url, credentials_path.as_deref()).await?)
    }
};
```

The `jetstream-broker` cargo feature gates the dep on `async-nats`; deploys that don't need JetStream skip the dep.

### 4. Update enterprise server bootstrap

`jammi-enterprise/crates/jammi-enterprise-server/src/main.rs:33` — replace hardcoded `JammiConfig::default()` with config loading:

```rust
let config_path = std::env::var("JAMMI_CONFIG")
    .map(PathBuf::from)
    .ok();
let config = match config_path {
    Some(p) => JammiConfig::load(&p)?,
    None => JammiConfig::default(),  // dev fallback: SQLite under cwd
};
let session = Arc::new(InferenceSession::new(config).await?);
```

`JammiConfig::load` is a `toml::from_str` over the file contents (existing pattern from other Jammi config files). Document the env var in the server's README.

Sample production config file `jammi-enterprise/deploy/jammi.toml`:

```toml
artifact_dir = "/var/lib/jammi"

[catalog]
kind = "postgres"
url = "${POSTGRES_URL}"     # interpolated from env at config-load time
pool_size = 16
max_lifetime_secs = 1800

[broker]
kind = "jet_stream"
url = "nats://${NATS_HOST}:4222"
credentials_path = "/var/run/secrets/nats.creds"

[gpu]
device = -1                  # CPU only on our SaaS cloud; tenant compute has GPUs
```

Env-var interpolation: a thin layer in `JammiConfig::load` substitutes `${VAR}` patterns from `std::env` before TOML parsing. Standard pattern.

## Files modified

### `crates/jammi-db/`
- `src/config.rs` — `CatalogConfig`, `BrokerConfig` enums; `JammiConfig` gains `catalog` and `broker` fields; `Default` impls
- `src/session.rs:58-95` — `InferenceSession::new` constructs backend + broker from config
- `src/catalog/backend.rs` — `BackendImpl::sqlite_from_path`, `BackendImpl::postgres_from_url` factory methods
- `src/trigger/jetstream.rs` — `JetStreamBroker::connect(url, credentials_path)` factory method (if not already present)
- `src/config.rs` — env-var interpolation in `JammiConfig::load`

### `crates/jammi-db/tests/`
- Parameterize existing integration tests over both backends via `#[rstest]` matrix where they don't already

### `crates/jammi-ai/`
- No changes (consumes the engine through `InferenceSession::new`; config plumbing is transparent)

### `crates/jammi-server/`
- `src/main.rs` — also accepts `JAMMI_CONFIG` env var for parity with enterprise server

### `jammi-enterprise/crates/jammi-enterprise-server/`
- `src/main.rs:33` — load config from env/file (post-W00 dep on jammi-db)
- `deploy/jammi.toml` (new, in W8) — reference production config

### `crates/jammi-cli/`
- `src/commands/serve.rs` and similar — accept `--config` flag pointing at TOML file; default to artifact-dir-only SQLite as today

## CI integration tests against Postgres

CI matrix gains a Postgres lane:
- Spin up `postgres:16` via `services:` block in GitHub Actions (or `services` in dev container)
- Set `JAMMI_TEST_PG_URL=postgres://...`
- Run `cargo test --workspace --features live-postgres-tests`

The `live-postgres-tests` feature already exists in `jammi-db/Cargo.toml`; existing Postgres-gated tests use it. W1 extends this to run the FULL workspace integration suite against Postgres, not just kernel-level Postgres tests.

## Success criteria

1. `cargo build --workspace` succeeds with `--features postgres` and without
2. `cargo test --workspace --exclude jammi-python` passes with the default SQLite backend
3. `JAMMI_TEST_PG_URL=postgres://... cargo test --workspace --features live-postgres-tests` passes against a live Postgres
4. `cargo clippy --workspace -- -D warnings` clean
5. Sample `deploy/jammi.toml` config loads successfully and produces a working `InferenceSession`
6. `JAMMI_CONFIG=/path/to/jammi.toml ./target/release/jammi-enterprise-server` boots against Postgres
7. No `JammiConfig::default()` remains in the enterprise server's main.rs

## Out of scope

- MySQL / MariaDB backends — defer; Postgres covers SaaS need
- Multi-region Postgres replicas — deferred to post-MVP per the plan
- Per-tenant Postgres schemas (vs single shared schema with row-level scoping) — current row-level scoping is the design; per-schema would be a future deployment shape
- Catalog migration tooling for moving an existing SQLite catalog to Postgres — defer (we have no production users to migrate)

## CLAUDE.md self-check

- [x] Clean separation — config selects backend; backend impls live behind a trait
- [x] DRY — single `match` in `InferenceSession::new` replaces ad-hoc backend construction
- [x] No backwards compatibility — `JammiConfig::default()` still returns SQLite; existing callers unaffected; SaaS callers explicitly opt into Postgres
- [x] Type-driven — `CatalogConfig` and `BrokerConfig` are tagged enums; no stringly-typed `backend = "postgres"` strings
- [x] No band-aids — env-var interpolation is principled, not ad-hoc string replacement
- [x] Atomic across workspace — engine PR ships with downstream enterprise bootstrap update in the same release cycle
