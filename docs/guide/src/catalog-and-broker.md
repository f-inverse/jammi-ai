# Catalog Backend and Trigger Broker

Jammi's catalog (models, sources, eval runs, mutable companion tables) and
trigger broker (provenance channels, evidence streams) are selected through
two fields on `JammiConfig`: `catalog` and `broker`. The dev-laptop default
is SQLite + an in-process broker; production deployments swap one or both
for Postgres (catalog and/or broker) or NATS JetStream (broker only).

## TOML schema

The catalog stanza is an externally tagged enum: the variant name is its own
TOML table (or a bare string for a variant with no required fields):

```toml
[catalog.sqlite]
# path = "/var/lib/jammi/catalog.db"   # optional; defaults to {artifact_dir}/catalog.db
```

```toml
[catalog.postgres]
url = "postgres://user:pass@host:5432/jammi?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
pool_size = 16
max_lifetime_secs = 1800
```

`url` should carry `?sslmode=verify-full` for any connection that leaves a
trusted network: `sslmode=require` upgrades the connection to TLS but never
verifies the server's certificate (it defeats a MITM only when the network
path is already trusted), and `sqlx` verifies against the **webpki** root
store rather than the OS trust store, so a private CA needs its own
`sslrootcert=` path even on a host that already trusts it system-wide.

The broker stanza follows the same shape:

```toml
broker = "in_memory"
```

```toml
[broker.jet_stream]
url = "nats://nats.svc:4222"
retention_seconds = 604800
credentials = { file = "/var/run/secrets/nats.creds" }
```

`[broker.jet_stream]` requires the `jetstream-broker` cargo feature
on `jammi-db`; selecting it without the feature returns
`JammiError::Config` rather than panicking at session construction time.

```toml
[broker.postgres]
# url = "postgres://user:pass@host:5432/jammi"   # optional; defaults to
#                                                 # `catalog.postgres.url`
idle_poll_secs = 5
```

`[broker.postgres]` carries no cargo feature: `sqlx`'s `postgres` feature is
unconditional in the workspace, so this variant always compiles in. It is a
`LISTEN`/`NOTIFY` **wake-up transport**, not a second log — the topic's own
mutable backing table is the durable, authoritative log, and this driver
only tells a subscriber "topic T may have advanced; go check". Every replica
MUST point `url` at the SAME Postgres database — `NOTIFY` is scoped to one
instance, and a replica listening elsewhere is not detectable by config; it
silently degrades to `idle_poll`-only delivery (never data loss, since the
backing table is still replayed on the next tick). `url` defaults to
`catalog.postgres.url` when unset and the catalog itself is Postgres; a
SQLite catalog with no explicit `url` here is a load-time `JammiError::Config`
naming both keys. `idle_poll_secs` (default 5) must be `>= 1`.

## Environment variable interpolation

`JammiConfig::load` substitutes `${NAME}` patterns from the process
environment before TOML parsing (`load_from`/`parse_from` take the same
lookup as an explicit map instead — see [Configuration](./configuration.md)
— so a test never touches real process env). The rules:

- `${NAME}` is replaced by the looked-up value of `NAME` (`load`: the
  process environment via `std::env::var`).
- A missing variable is an error. The loader never silently substitutes an
  empty string — that is a common source of "deployed config has an empty
  Postgres URL" outages.
- `$$` escapes a literal `$`.
- A bare `$` not followed by `$` or `{` is preserved verbatim, so passwords
  containing a single `$` slip through unchanged.
- An unterminated `${` returns `JammiError::Config`.
- Interpolation is one-pass and not recursive: `${X}`'s value is not
  re-scanned.

Combined with the externally tagged shape:

```toml
artifact_dir = "/var/lib/jammi"

[catalog.postgres]
url = "${POSTGRES_URL}?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
pool_size = 16
max_lifetime_secs = 1800

[broker.jet_stream]
url = "nats://${NATS_HOST}:4222"
retention_seconds = 604800
credentials = { file = "/var/run/secrets/nats.creds" }
```

A working copy of this file ships at
`crates/jammi-db/examples/sample-postgres.toml`.

## SQLite vs Postgres trade-offs

| Concern | SQLite | Postgres |
| --- | --- | --- |
| Operational footprint | One file under `artifact_dir`. No daemon. | Externally-managed Postgres cluster. |
| Concurrent writers | One; WAL mode lets many readers run alongside one writer. | Many. |
| Multi-process deployment | Single-process only, and enforced on unix: the catalog opens through SQLite's `unix-excl` VFS, which holds a process-scoped exclusive lock on the file. A second process opening the same `artifact_dir` is refused with a typed `backend unavailable` error naming this contract after the 5 s busy timeout — it never corrupts the WAL and never hangs. Handing the directory to another process is an awaited event (`Catalog::close().await` / `JammiSession::close().await`), not a drop. | Multi-replica safe. |
| Failure recovery | File restore from backup. | Standard Postgres point-in-time-recovery. |
| Pool tuning | None — opens one pool of 8 connections. | `pool_size` + `max_lifetime_secs` honour `sqlx::PgPool` knobs. |

The single-process guarantee stops at the process boundary. A second SQLite
*library instance* inside the same process — the shape a Python caller reaches
by using the stdlib `sqlite3` module against a live engine's catalog — shares
this process's `fcntl` locks and so cannot be arbitrated: it can read a stale
image or corrupt the file. Close the engine first (`close()`), then touch the
file. On Windows the contract is documentation-only: `unix-excl` does not
exist there.

One operator knob exists and is diagnostic only: `JAMMI_SQLITE_VFS=default`
restores the platform default VFS on every target, re-arming exactly the
corruption the seam removes, so that the fix can be falsified by re-running
the escape's oracle against it and observing the RED; engaging it logs a
`WARN`, and it is never set in production.

For laptop / single-tenant deployments, SQLite is the right answer; the
trade-off table tilts to Postgres the moment a second `jammi-server`
replica enters the picture.

## In-memory vs Postgres vs JetStream broker

| Concern | InMemory | Postgres | JetStream |
| --- | --- | --- | --- |
| Persistence | In-process only; lost on restart. | None of its own — the topic's mutable backing table (already durable) is the log; the driver carries no bytes. | NATS server retains streams per `retention_seconds`. |
| Cross-process delivery | None — a publish in process A is invisible to a subscriber in process B. | All subscribers (any process, any host) see a wake within `idle_poll_secs` of a publish; the actual rows come from a replay of the shared backing table. | All subscribers (any process, any host) see every published batch within the retention window. |
| Auth | None. | Whatever `[broker.postgres] url` (or the catalog's) already authenticates with. | Anonymous or NATS `.creds` file contents via `credentials`. |
| Operational footprint | None. | **None beyond the catalog** — up to three extra connections to the SAME Postgres instance the catalog already uses (or points at); no extra service. | One NATS server (or cluster). |

In-memory is fine for tests, local development, and single-process server
deployments where every consumer lives in the same `jammi-server` process.
For any deployment that wants replay across restarts or fan-out across
multiple `jammi-server` replicas (Shapes B and C), **Postgres is the
recommended broker** whenever the catalog is already Postgres: it adds no
extra service, only a handful of connections to the database already in the
topology. Reach for JetStream when the deployment wants a dedicated,
broker-scoped retention window independent of the catalog's own lifecycle,
or when the catalog itself stays SQLite (single-process) while the broker
still needs to fan out beyond one process — a shape Postgres-as-broker
cannot serve without a Postgres catalog to default its `url` from.

## Health probe

`CatalogBackend::ping` runs `SELECT 1` against the underlying pool and
classifies pool failures as `BackendError::Unavailable`. The
`/readyz` endpoint on `jammi-server` (when wired) reaches this via
`session.catalog().ping().await`. The primitive is cheap — microseconds
against a warm pool — and never opens a transaction.
