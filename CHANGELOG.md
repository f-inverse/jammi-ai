# Changelog

All notable changes to the Jammi AI workspace are recorded here. The
workspace ships every publishable crate at the same
`workspace.package.version`; PyPI `jammi-ai` mirrors that version.

## v0.8.0 — 2026-05-26

### Added

- `JammiConfig::catalog` and `JammiConfig::broker` fields, both tagged enums:
  - `CatalogConfig::Sqlite { path: Option<PathBuf> }` (default; uses
    `{artifact_dir}/catalog.db` when `path` is `None`) and
    `CatalogConfig::Postgres { url, pool_size, max_lifetime_secs }`.
    `pool_size` defaults to 8; `max_lifetime_secs` is optional and, when
    `Some`, sets `sqlx::PgPool::max_lifetime` to limit per-connection
    lifetime behind connection-pooling proxies (PgBouncer, RDS Proxy).
  - `BrokerConfig::InMemory` (default) and
    `BrokerConfig::JetStream { url, retention_seconds, credentials_path }`.
    `retention_seconds` defaults to 7 days; `credentials_path` is optional
    and selects authenticated vs anonymous NATS connection.
- `JammiConfig::load` runs `${VAR}` env-var interpolation on the raw TOML
  source before parsing. A missing variable is a typed
  `JammiError::Config` (no silent empty substitution); `$$` escapes a
  literal `$`; unterminated `${` is a typed error.
- `CatalogBackend::ping(&self) -> Result<(), BackendError>` plus
  per-backend implementations and a `Catalog::ping` thin wrapper. The
  primitive runs `SELECT 1` against the connection pool and classifies
  pool failures as `BackendError::Unavailable`. Cost is microseconds
  against a warm pool. Consumed by the OSS server's `/readyz` route.
- `BackendImpl::sqlite_from_path` and
  `BackendImpl::postgres_from_url(url, pool_size, max_lifetime_secs)`
  factories. The session resolver (`JammiSession::new`,
  `JammiSession::with_broker`, `JammiSession::with_backend`) reaches for
  these so a caller that overrides one dimension keeps the other
  config-driven.
- `JetStreamBroker::connect_with_credentials(url, retention_seconds, &Path)`
  for SaaS deployments where the broker rejects anonymous connections.
  Internally a `from_client` helper DRYs the two constructors so they
  agree on the schemas-cache and `JetStreamContext` derivation.
- `crates/jammi-db/examples/sample-postgres.toml` demonstrating a
  Postgres + JetStream production config.
- `docs/guide/src/catalog-and-broker.md` covering the TOML schema, the
  env-var interpolation rules, the SQLite/Postgres trade-off matrix,
  and the broker selection rationale.
- New integration test file `crates/jammi-db/tests/it/catalog_ping.rs`
  exercising `Catalog::ping` for SQLite (happy path, idempotency, arc
  lifetime) plus a Postgres lane behind `live-postgres-tests` with a
  happy-path and an unreachable-URL negative test.
- **`jammi-server` OSS binary.** The `jammi-server` crate gains a
  `[[bin]]` target. The binary loads `JammiConfig` from
  `--config`/`JAMMI_CONFIG`/the platform default, initialises tracing
  per the resolved logging configuration, and hands control to
  `jammi_server::runtime::OssServer`. The orchestration is one module
  (`src/runtime.rs`) — the Axum side-channel and the Tonic chain are
  wired together with one `tokio::sync::broadcast` channel for graceful
  shutdown.
- **Container image.** A `Dockerfile` at the workspace root builds a
  stripped distroless image (`gcr.io/distroless/cc-debian12`) from the
  CI base toolchain. The image runs as the nonroot user (uid `65532`),
  exposes `:8080` (HTTP side-channel) and `:8081` (gRPC + Flight SQL),
  and declares `/var/lib/jammi` as a volume. CI publishes to
  `ghcr.io/f-inverse/jammi-ai-server` on every `v*` tag via
  `.github/workflows/server-image.yml`.
- **Health endpoints.** The HTTP side-channel exposes `/healthz`
  (liveness; returns `{"status":"ok","version":"<crate version>"}`),
  `/readyz` (readiness; pings the catalog backend via
  `Catalog::ping().await` and returns 503 on failure), and `/metrics`
  (Prometheus text-format snapshot of the substrate counters: gRPC
  requests, Flight SQL queries, eval invocations, and a search-latency
  histogram).
- **`runtime::serve_grpc_chain` test-fixture entry-point.** The same
  chain builder the binary uses, exposed for integration tests that
  need to wire pre-seeded sessions to a unified Flight SQL + gRPC
  server on one port.
- **`InferenceSession::tenant_binding_arc`** accessor — required by the
  OSS server so the Flight SQL `TenantBoundProvider` can bind tenants
  for the duration of each query.
- **`cookbook/` tree at the repo root** — OSS source of truth for runnable
  Python recipes against `jammi-ai`. Layout:
  - `cookbook/README.md` — index
  - `cookbook/quickstart/` — 5-minute walkthrough (`README.md`,
    `01_install.md` .. `04_vector_search.md`, runnable `quickstart.py`)
  - `cookbook/recipes/{mutable_tables, trigger_streams, eval_embeddings,
    eval_inference, fine_tune, flight_sql}/{README.md, example.py}`
  - `cookbook/fixtures/` — deterministic `tiny_corpus.parquet`,
    `tiny_golden.json`, `tiny_labels.csv`, `tiny_pairs.csv` plus
    `generate.py`; `tiny_bert/` and `tiny_modernbert_classifier/` model
    fixtures moved here from `tests/fixtures/`. Total tree < 250 KB.
  Every recipe runs against the local fixture model so CI is hermetic;
  every recipe's README has a "When to use this pattern" callout.
- `tests/cookbook_smoke.py` — smoke runner that times every recipe,
  fails the build if `quickstart.py` exceeds 60s wall-clock, excludes
  `fine_tune` and `flight_sql` by default, surfaces them behind
  `JAMMI_COOKBOOK_SLOW=1`.
- `.github/workflows/cookbook.yml` — per-PR fast lane plus nightly
  cron that sets `JAMMI_COOKBOOK_SLOW=1` and builds
  `target/release/jammi` for the Flight SQL recipe.
- `docs/guide/src/cookbook-recipes.md` — mdBook entry that
  `{{#include}}`s every recipe README so the rendered guide and the
  OSS recipes never drift apart.
- `jammi_test_utils::{cookbook_fixture, cookbook_fixture_url,
  cookbook_fixtures_dir}` — first-class helper for the cookbook
  fixtures path. Every integration test that consumed `tiny_bert` /
  `tiny_modernbert_classifier` now reads from `cookbook/fixtures/`.

### Changed

- The CI `test-pg` job now uses `postgres:16` (was `postgres:15`),
  matching the spec's pinned base image.
- `README.md` quickstart section collapsed to a 10-line inline example
  with a link to `cookbook/quickstart/` for the full walkthrough — the
  cookbook tree is the single source of truth.
- `docs/guide/src/quickstart-python.md` rewritten as a stub that
  `{{#include}}`s the cookbook quickstart README.

### Removed

- `tests/cookbook_smoke_test.py` (legacy file from before the cookbook
  tree existed; used a non-existent `add_source(path=...)` kwarg and was
  never wired into CI). Superseded by `tests/cookbook_smoke.py`.
- `tests/fixtures/tiny_bert/` — relocated to `cookbook/fixtures/tiny_bert/`.
- `tests/fixtures/tiny_modernbert_classifier/` — relocated to
  `cookbook/fixtures/tiny_modernbert_classifier/`.

### Fixed

- `ResultStore::create_table` now sanitises `.` characters in the model id
  alongside `/`, `:`, and spaces. A dot in the embedded model-id path
  (e.g. `local:/foo/.cache/model`) survived the previous sanitiser and
  produced a result-table name like `foo__model_.cache__...`, which
  `Path::with_extension("")` then mis-parsed when the sidecar layout
  derived the on-disk stem — the trailing `.cache__...` component was
  treated as an extension and stripped, so the `.usearch` / `.rowmap` /
  `.manifest.json` siblings were written under a truncated name.
  Affected any deployment whose model-id path contained a dot.
- `docs/guide/src/installation.md` — Python install snippet was still
  `pip install jammi`; corrected to `pip install jammi-ai` (post-S1
  rename).
- `docs/guide/src/introduction.md` — three-ways-to-use table row for
  Python was still `pip install jammi`; corrected to
  `pip install jammi-ai`.

### Breaking

- `CatalogBackend` trait grew a new required method `ping`. Any
  out-of-tree implementor must add it; the workspace has no such
  callers.
- `PostgresBackend::open(url)` is renamed to
  `PostgresBackend::open_with_options(url, pool_size, max_lifetime_secs)`.
  The previous signature hardcoded `max_connections = 8` and did not
  expose connection lifetime; both knobs are now caller-supplied. No
  shim is provided.
- **`/health` renamed to `/healthz`.** The HTTP side-channel's liveness
  endpoint moves to the Kubernetes convention. No shim is provided —
  callers update in lockstep.
- **`jammi_server::serve_grpc_with_shutdown` removed.** The function is
  superseded by `jammi_server::runtime::OssServer::run` (binary
  entry-point) and `jammi_server::runtime::serve_grpc_chain` (test
  fixture). Migrate via the `serve_grpc_chain` helper which takes the
  same arguments plus the Flight SQL `SessionContext` and
  `TenantBinding`.

### Migration

```rust
// before
use jammi_db::catalog::backend_postgres::PostgresBackend;
let pg = PostgresBackend::open("postgres://...").await?;
// after
let pg = PostgresBackend::open_with_options("postgres://...", 8, None).await?;
```

```toml
# before — JammiConfig::default() implicitly chose SQLite + InMemory
[artifact_dir]
# ...

# after — both selections are explicit (and a missing `[catalog]` / `[broker]`
# stanza still defaults to SQLite + InMemory)
[catalog]
kind = "postgres"
url = "${POSTGRES_URL}"
pool_size = 16

[broker]
kind = "jet_stream"
url = "nats://${NATS_HOST}:4222"
credentials_path = "/var/run/secrets/nats.creds"
```

Out-of-tree `CatalogBackend` impls must add the new method:

```rust
fn ping(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
    Box::pin(async move {
        sqlx::query("SELECT 1").execute(&self.pool).await.map_err(classify)?;
        Ok(())
    })
}
```

```rust
// before
jammi_server::serve_grpc_with_shutdown(addr, store, Some(trigger), shutdown).await?;
// after
jammi_server::runtime::serve_grpc_chain(
    addr,
    session.context().clone(),
    session.tenant_binding_arc(),
    store,
    Some(trigger),
    shutdown,
)
.await?;
```

```bash
# Liveness probe URL.
# before
curl http://localhost:8080/health
# after
curl http://localhost:8080/healthz
```

