# Catalog Backend and Trigger Broker

> Coordinator to relocate to the docs site (C3) when scaffold lands.

Jammi's catalog (models, sources, eval runs, mutable companion tables) and
trigger broker (provenance channels, evidence streams) are selected through
two fields on `JammiConfig`: `catalog` and `broker`. The dev-laptop default
is SQLite + an in-process broker; production deployments swap one or both
for Postgres + JetStream.

## TOML schema

The catalog stanza is a tagged enum keyed by `kind`:

```toml
[catalog]
kind = "sqlite"
# path = "/var/lib/jammi/catalog.db"   # optional; defaults to {artifact_dir}/catalog.db
```

```toml
[catalog]
kind = "postgres"
url = "postgres://user:pass@host:5432/jammi"
pool_size = 16
max_lifetime_secs = 1800
```

The broker stanza follows the same shape:

```toml
[broker]
kind = "in_memory"
```

```toml
[broker]
kind = "jet_stream"
url = "nats://nats.svc:4222"
retention_seconds = 604800
credentials_path = "/var/run/secrets/nats.creds"
```

`broker.kind = "jet_stream"` requires the `jetstream-broker` cargo feature
on `jammi-db`; selecting it without the feature returns
`JammiError::Config` rather than panicking at session construction time.

## Environment variable interpolation

`JammiConfig::load` substitutes `${NAME}` patterns from the process
environment before TOML parsing. The rules:

- `${NAME}` is replaced by the value of `std::env::var("NAME")`.
- A missing variable is an error. The loader never silently substitutes an
  empty string — that is a common source of "deployed config has an empty
  Postgres URL" outages.
- `$$` escapes a literal `$`.
- A bare `$` not followed by `$` or `{` is preserved verbatim, so passwords
  containing a single `$` slip through unchanged.
- An unterminated `${` returns `JammiError::Config`.
- Interpolation is one-pass and not recursive: `${X}`'s value is not
  re-scanned.

Combined with the tagged-enum shape:

```toml
artifact_dir = "/var/lib/jammi"

[catalog]
kind = "postgres"
url = "${POSTGRES_URL}"
pool_size = 16
max_lifetime_secs = 1800

[broker]
kind = "jet_stream"
url = "nats://${NATS_HOST}:4222"
retention_seconds = 604800
credentials_path = "/var/run/secrets/nats.creds"
```

A working copy of this file ships at
`crates/jammi-db/examples/sample-postgres.toml`.

## SQLite vs Postgres trade-offs

| Concern | SQLite | Postgres |
| --- | --- | --- |
| Operational footprint | One file under `artifact_dir`. No daemon. | Externally-managed Postgres cluster. |
| Concurrent writers | One; WAL mode lets many readers run alongside one writer. | Many. |
| Multi-process deployment | Single-process only — sharing the file across `jammi-server` replicas corrupts WAL. | Multi-replica safe. |
| Failure recovery | File restore from backup. | Standard Postgres point-in-time-recovery. |
| Pool tuning | None — opens one pool of 8 connections. | `pool_size` + `max_lifetime_secs` honour `sqlx::PgPool` knobs. |

For laptop / single-tenant deployments, SQLite is the right answer; the
trade-off table tilts to Postgres the moment a second `jammi-server`
replica enters the picture.

## In-memory vs JetStream broker

| Concern | InMemory | JetStream |
| --- | --- | --- |
| Persistence | In-process only; lost on restart. | NATS server retains streams per `retention_seconds`. |
| Cross-process delivery | None — a publish in process A is invisible to a subscriber in process B. | All subscribers (any process, any host) see every published batch within the retention window. |
| Auth | None. | Anonymous or NATS `.creds` file via `credentials_path`. |
| Operational footprint | None. | One NATS server (or cluster). |

In-memory is fine for tests, local development, and single-process server
deployments where every consumer lives in the same `jammi-server` process.
JetStream is required for any deployment that wants replay across
restarts or fan-out across multiple `jammi-server` replicas.

## Health probe

`CatalogBackend::ping` runs `SELECT 1` against the underlying pool and
classifies pool failures as `BackendError::Unavailable`. The
`/readyz` endpoint on `jammi-server` (when wired) reaches this via
`session.catalog().ping().await`. The primitive is cheap — microseconds
against a warm pool — and never opens a transaction.
