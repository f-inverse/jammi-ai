//! Test-only checkpoints for the mutable-table lifecycle.
//!
//! Compiled only under `feature = "test-hooks"`. The `mutable_crash_recovery.rs`
//! integration test spawns a child process with [`READY_FILE_ENV`] set plus one
//! of two checkpoint selectors, then `SIGKILL`s the child mid-operation to
//! prove crash-consistency:
//!
//! - [`CHECKPOINT_AFTER_ENV`] keys the *insert* hook: [`maybe_signal`] fires
//!   once the per-write-call row counter crosses the threshold, proving a
//!   partial multi-row `INSERT` rolls back.
//! - [`LIFECYCLE_CHECKPOINT_ENV`] keys the *lifecycle* hook:
//!   [`maybe_signal_lifecycle`] fires just before a register/drop op's single
//!   transaction commits — all DDL + catalog SQL has been issued but nothing is
//!   durable — proving the op is all-or-nothing under a crash.
//!
//! In every case the hook writes [`READY_FILE_ENV`] so the parent knows the
//! child is mid-transaction, then awaits an unsignalled notifier so the child
//! parks until the parent sends `SIGKILL`. The transaction never commits; RAII
//! rollback delivers the all-or-nothing guarantee.
//!
//! When the env vars are unset (the default for every other test and every
//! production build), each hook is a single early-returning `if`. No production
//! code path observes this module.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use tokio::sync::Notify;

use crate::catalog::backend::BackendKind;

/// Path the child writes once a checkpoint fires. The parent polls
/// `try_exists` on this path and `SIGKILL`s the child as soon as it appears.
pub const READY_FILE_ENV: &str = "JAMMI_TEST_CHECKPOINT_READY_FILE";

/// Row threshold the child must cross before the *insert* hook signals. The
/// child increments `rows_so_far` once per `insert_batch` call; a test that
/// wants to fire after the 50th row passes a 50-row batch and sets this to 50.
pub const CHECKPOINT_AFTER_ENV: &str = "JAMMI_TEST_CHECKPOINT_AFTER";

/// Names the lifecycle commit boundary the child parks at: one of `register`,
/// `register_topic`, `drop_table`, or `drop_topic`. When the in-flight op's
/// label matches this value, [`maybe_signal_lifecycle`] fires just before the
/// op's single transaction commits — all DDL + catalog SQL has been issued on
/// the transaction but nothing is durable, so a `SIGKILL` here proves the op is
/// all-or-nothing. Independent of [`CHECKPOINT_AFTER_ENV`], which keys the
/// row-counting insert hook.
pub const LIFECYCLE_CHECKPOINT_ENV: &str = "JAMMI_TEST_LIFECYCLE_CHECKPOINT";

/// Park forever once a signal fires. Park-and-die is the contract: the caller
/// has SIGKILL teed up.
static PARK: OnceLock<Notify> = OnceLock::new();

/// One-shot guard so an op that reaches a checkpoint more than once only
/// signals once. Without this, the second call would re-write the ready file
/// and re-park (the first park has already returned), wasting wall-clock.
static SIGNALLED: OnceLock<()> = OnceLock::new();

/// Write the ready file and park forever. Shared by both hooks; the
/// [`SIGNALLED`] guard makes the first caller win and every later one a no-op.
async fn signal_and_park(ready_file: PathBuf) {
    if SIGNALLED.set(()).is_err() {
        return;
    }
    if let Some(parent) = ready_file.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    tokio::fs::write(&ready_file, b"ready")
        .await
        .expect("checkpoint ready-file write");
    let park = PARK.get_or_init(Notify::new);
    park.notified().await;
}

/// Signal-and-park if `rows_so_far` has crossed the [`CHECKPOINT_AFTER_ENV`]
/// threshold. First call past the threshold writes the ready file and parks
/// forever on a notifier that no one signals. Subsequent calls are no-ops.
pub async fn maybe_signal(rows_so_far: u64) {
    let ready_file = match std::env::var(READY_FILE_ENV).ok() {
        Some(p) => PathBuf::from(p),
        None => return,
    };
    let after = match std::env::var(CHECKPOINT_AFTER_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        Some(a) => a,
        None => return,
    };
    if rows_so_far < after {
        return;
    }
    signal_and_park(ready_file).await;
}

/// Signal-and-park if `op` matches the [`LIFECYCLE_CHECKPOINT_ENV`] selector.
/// Called at the commit boundary of a register/drop op — every statement has
/// been issued on the transaction but commit has not yet run — so a `SIGKILL`
/// while parked here leaves the op all-or-nothing. Subsequent calls are no-ops.
pub async fn maybe_signal_lifecycle(op: &str) {
    let ready_file = match std::env::var(READY_FILE_ENV).ok() {
        Some(p) => PathBuf::from(p),
        None => return,
    };
    let selector = match std::env::var(LIFECYCLE_CHECKPOINT_ENV).ok() {
        Some(s) => s,
        None => return,
    };
    if selector != op {
        return;
    }
    signal_and_park(ready_file).await;
}

/// Names the materialization commit boundary the child parks at. When set, the
/// result-store fires [`maybe_signal_materialization`] *after* the Parquet (and
/// ANN sidecar) bytes are durable but *before* the `.materialization.json`
/// sidecar is written and the `building -> ready` flip commits — the
/// crash window the contract must survive: a valid Parquet with no manifest. A
/// `SIGKILL` while parked here leaves a `building` row whose Parquet landed but
/// whose manifest never did, which recovery must reconcile to `failed` (it
/// cannot reconstruct the producing descriptor), never promote manifest-less.
pub const MATERIALIZATION_CHECKPOINT_ENV: &str = "JAMMI_TEST_MATERIALIZATION_CHECKPOINT";

/// Signal-and-park if [`MATERIALIZATION_CHECKPOINT_ENV`] is set. Called inside
/// the result-store's `building -> ready` boundary, after the Parquet bytes are
/// durable and before the manifest sidecar is written. Subsequent calls are
/// no-ops.
pub async fn maybe_signal_materialization() {
    let ready_file = match std::env::var(READY_FILE_ENV).ok() {
        Some(p) => PathBuf::from(p),
        None => return,
    };
    if std::env::var(MATERIALIZATION_CHECKPOINT_ENV).is_err() {
        return;
    }
    signal_and_park(ready_file).await;
}

/// Arms the migration-runner rendezvous. Value `<backend>:<parties>` with
/// `<backend>` one of `postgres` / `sqlite` and `<parties>` at least 2: the
/// first `<parties>` runners on that backend are each held by
/// [`maybe_signal_migration_ledger_read`] -- inside the open migration
/// transaction, ledger read, no DDL issued yet -- until all have arrived or
/// [`MIGRATION_LEDGER_BARRIER_TIMEOUT`] elapses, whichever is first. That pins
/// the interleaving the Postgres advisory lock in `catalog::migrations::run`
/// must make impossible: two runners that both read an empty ledger and both
/// go on to `CREATE TABLE`.
///
/// The backend selector keeps the arming from leaking into a sibling test on
/// the other dialect in the same process (a SQLite runner parked here would sit
/// inside its `BEGIN IMMEDIATE` for the timeout, against a 5 s `busy_timeout`
/// on the other pool). The wait is bounded rather than a strict barrier
/// because in the fixed world the second runner never reaches this point while
/// the first holds the lock (a strict barrier would deadlock the winner against
/// the loser it blocks), and on a fresh database even the unfixed world parks
/// the second runner on the ledger `CREATE TABLE`'s catalog lock. Either way
/// the first caller proceeds after the timeout and the test's assertions
/// decide.
pub const MIGRATION_LEDGER_BARRIER_ENV: &str = "JAMMI_TEST_MIGRATION_LEDGER_BARRIER";

/// Longest one party waits at the migration rendezvous for the others.
pub const MIGRATION_LEDGER_BARRIER_TIMEOUT: Duration = Duration::from_secs(5);

/// Arrivals at the migration rendezvous, process-wide. The rendezvous is
/// one-shot: once `n` callers have arrived every later caller passes straight
/// through, so a sibling test in the same binary can never be parked by a
/// stale arming.
static MIGRATION_LEDGER_ARRIVALS: AtomicUsize = AtomicUsize::new(0);

/// Release notifier for the migration rendezvous; the `n`-th arrival fires it.
static MIGRATION_LEDGER_RELEASE: OnceLock<Notify> = OnceLock::new();

/// Bounded rendezvous inside `catalog::migrations::run`, after the
/// `applied_migrations` ledger has been read and before the first migration
/// DDL statement. A no-op unless [`MIGRATION_LEDGER_BARRIER_ENV`] names
/// `kind` with a party count of at least two; see there for the interleaving
/// it pins.
pub async fn maybe_signal_migration_ledger_read(kind: BackendKind) {
    let Some(armed) = std::env::var(MIGRATION_LEDGER_BARRIER_ENV).ok() else {
        return;
    };
    let Some((backend, parties)) = armed.split_once(':') else {
        return;
    };
    let selected = match kind {
        BackendKind::Postgres => "postgres",
        BackendKind::Sqlite => "sqlite",
    };
    let parties = match parties.parse::<usize>() {
        Ok(n) if n >= 2 && backend == selected => n,
        _ => return,
    };
    let release = MIGRATION_LEDGER_RELEASE.get_or_init(Notify::new);
    // Register interest before counting the arrival so the releasing
    // `notify_waiters` from a party that arrives between the two steps is
    // not lost.
    let notified = release.notified();
    tokio::pin!(notified);
    notified.as_mut().enable();
    let arrived = MIGRATION_LEDGER_ARRIVALS.fetch_add(1, Ordering::SeqCst) + 1;
    if arrived >= parties {
        release.notify_waiters();
        return;
    }
    // Timeout is the documented bound, not a failure: the caller proceeds.
    let _ = tokio::time::timeout(MIGRATION_LEDGER_BARRIER_TIMEOUT, notified).await;
}
