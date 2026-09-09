# Backup and Restore

How to back up and restore a Jammi deployment safely, for each of the
[deployment shapes](./philosophy.md#how-it-deploys-one-binary-pluggable-backends)
the engine ships. The catalog (models, sources, eval runs, mutable companion
tables, result-table rows) and the object store (result-table Parquet, ANN
sidecars, model artifacts) are two independently-backed-up systems that must
stay consistent with each other; everything below is about preserving that
consistency across a backup/restore cycle. Everything described here ships
today.

## `cache/` is excludable from every backup

Both the local ANN-segment cache and the model-artifact fetch cache
(`{local_cache_dir}/index` and `{local_cache_dir}/artifact` — see
[Store Sources and Results in Cloud Object Storage](./cloud-storage.md) for
`with_root`'s `local_cache_dir` parameter; for the default embedded shape
they sit under `{artifact_dir}/cache/`) are **content-addressed derived
data**: every object in them is rebuilt on demand from the object store, and
never the sole copy of anything. It is always safe to:

- exclude `cache/` from a backup entirely;
- restore a backup whose `cache/` directory is **stale** relative to the
  restored catalog and storage — the store simply misses cache and refetches;
- restore a backup with **no** `cache/` directory at all — the store creates
  it on first use.

## Shape A / B: SQLite catalog

The SQLite catalog and the local (or object-store) result-table root are
backed up together, and the catalog file must be captured through a
consistent snapshot — a bare `cp` of a live `catalog.db` while its WAL is
being written is not that. **Close first:**

```rust,no_run
# extern crate jammi_db;
# use jammi_db::session::JammiSession;
# async fn ex(session: JammiSession) {
// Stop accepting new work on this session first, at the application layer;
// close() itself only waits for outstanding checkouts, not new admissions.
session.close().await;
# }
```

```python
db.close()
```

`close()` (`crates/jammi-db/src/session.rs`, `JammiSession::close`) awaits
every outstanding pool checkout and, for SQLite, releases the process-scoped
`unix-excl` file lock and lets the `-wal` sidecar quiesce. Only once it
returns is the on-disk image safe to copy:

```bash
# 1. Stop the process (or await close()) so the WAL is checkpointed and no
#    writer holds the file.
# 2. Copy the catalog file, its WAL, and the result-table root together —
#    they must be from the SAME instant, since a result-table row and its
#    bytes are two halves of one fact.
cp catalog.db catalog.db-wal /backup/2026-09-09/          # -wal may be absent
                                                            # if fully checkpointed
cp -r jammi_db/ /backup/2026-09-09/jammi_db/               # exclude jammi_db/cache/
```

**Restoring** is the reverse: with no `jammi-server` process holding the
directory, replace `catalog.db` (+ `-wal`) and the result-table root from the
same backup instant, then reopen:

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# use jammi_ai::Jammi;
# use jammi_db::config::JammiConfig;
# async fn ex() -> jammi_db::error::Result<()> {
let config = JammiConfig::load(None)?;
let session = Jammi::open(jammi_ai::Target::Local(config)).await?;
// ResultStore::recover() runs automatically at open — see "Ordering" below.
# let _ = session; Ok(()) }
```

A **hot copy is unsafe** under the `unix-excl` VFS contract described in
[Catalog Backend and Trigger Broker](./catalog-and-broker.md): copying the
file while the engine still holds it can capture a torn WAL image, and
handing the directory to a second reader before `close()` returns races the
first engine's in-memory WAL index. Always stop-or-close before copying.

## Shape B / C: Postgres catalog

Use Postgres's own backup primitives for the catalog — `pg_dump` for a
point-in-time logical snapshot, or continuous archiving / PITR for a
production deployment — and your object-store provider's own snapshot or
versioning feature (S3 versioning, GCS object versioning, a bucket-level
snapshot) for the result-table root. These are two independent systems with
independent backup tooling; nothing about restoring one requires stopping
`jammi-server` the way SQLite's file-level copy does, since Postgres and the
remote object store both serve concurrent readers/writers safely through
their own transaction/consistency models.

## Ordering rule

**Restore storage first, then the catalog — or run `reconcile` after.** A
catalog row is only meaningful if the bytes it names exist; restoring the
catalog to a point *after* the storage snapshot can leave rows pointing at
objects the storage restore does not have, and restoring the catalog to a
point *before* the storage snapshot leaves storage objects with no
referencing row (which `jammi reconcile` — see below — treats as orphans,
never as data loss, since a row-less object was never queryable in the
first place). Concretely:

1. Restore (or roll storage forward/back to) the storage snapshot first.
2. Restore the catalog to a snapshot from the **same or a later** instant.
3. If the two snapshots cannot be instant-matched exactly (a Postgres PITR
   target a few seconds off a bucket versioning rollback point, say), run
   `jammi reconcile` (`--apply` once you have reviewed a dry run) to
   reconcile the two: it flips a `ready` row whose required objects are
   missing to `failed`, and reports (or, on `--apply`, reclaims) any
   storage object no live row references. See
   [Catalog Backend and Trigger Broker → Multi-writer safety](./catalog-and-broker.md#multi-writer-safety)
   for what `reconcile` checks and its deletion arms, and the maintainer
   guide (`docs/maintainer/MAINTAINER-GUIDE.md`) for the full allowlist and
   referenced/required-object rules.

## When to run `reconcile`

Beyond a mismatched restore, run `jammi reconcile` (dry run first — `--apply`
defaults to `false`):

- after any restore where the catalog and storage snapshots were not taken
  atomically together;
- as a periodic housekeeping pass in a long-lived deployment, to reclaim
  orphaned objects left by a writer that crashed and was never reaped by a
  later session's startup recovery sweep (recovery only reaps a `building`
  row still present in the catalog; an object written but never inserted as
  a row at all is `reconcile`'s job, not recovery's);
- before decommissioning a tenant's data, as a dry run to confirm what a
  cross-tenant `--all` pass (gated behind an `AdminAuthorizer` — see
  [Security Posture](./security.md)) would report.

`reconcile` never mutates a training job and never reclaims a `pending`
(too-young) orphan, so a dry run is always safe to run against a live,
healthy deployment.
