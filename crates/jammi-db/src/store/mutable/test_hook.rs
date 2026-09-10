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
//! The result-table lifecycle points ([`MaterializationPoint`]) add an
//! in-process arming API ([`arm`] / [`Armed`]) beside the env-var selector, for
//! the same-process two-writer tests: the test parks one writer at a named
//! point, runs a peer against the same catalog, then releases the writer.
//!
//! When nothing is armed and the env vars are unset (the default for every
//! other test and every production build), each hook is a single
//! early-returning check. No production code path observes this module.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
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

/// Selects the result-table lifecycle point the process parks at. Read by
/// [`maybe_signal_point`] together with [`READY_FILE_ENV`]: the value names a
/// [`MaterializationPoint`] (`table_created` / `materialization`); the legacy
/// presence value `1` means `materialization`, the point the SIGKILL harness
/// has always armed. Any other value arms nothing.
///
/// `materialization` fires inside `BuildingTable::finish` *after* the lease
/// renew and the Parquet (and ANN sidecar) bytes are durable but *before* the
/// `.materialization.json` sidecar is written and the `building -> ready` flip
/// commits — the crash window the contract must survive: a valid Parquet with
/// no manifest. `table_created` fires inside `ResultStore::create_table`
/// after the `building` row's INSERT commits and its heartbeat is running,
/// before any bytes exist. A `SIGKILL` while parked at either leaves a
/// `building` row a dead writer's lease still names for a while; recovery
/// reconciles it once the lease expires (to `failed` — it cannot reconstruct
/// the producing descriptor — never a manifest-less promotion).
pub const MATERIALIZATION_CHECKPOINT_ENV: &str = "JAMMI_TEST_MATERIALIZATION_CHECKPOINT";

/// The named result-table lifecycle points a test can park a writer at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationPoint {
    /// After `create_table`'s INSERT committed and the heartbeat started; no
    /// bytes yet (the W1 window of esc-094).
    TableCreated,
    /// Inside `finish`, after the lease renew and before the manifest sidecar
    /// write (the W2 window of esc-094 and the SIGKILL harness's window).
    Materialization,
}

impl MaterializationPoint {
    /// The [`MATERIALIZATION_CHECKPOINT_ENV`] value naming this point.
    pub fn env_name(self) -> &'static str {
        match self {
            Self::TableCreated => "table_created",
            Self::Materialization => "materialization",
        }
    }

    fn from_env_value(value: &str) -> Option<Self> {
        match value {
            "table_created" => Some(Self::TableCreated),
            // Presence semantics: the SIGKILL harness arms `1`, which has
            // always meant the materialization window.
            "materialization" | "1" => Some(Self::Materialization),
            _ => None,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::TableCreated => 0,
            Self::Materialization => 1,
        }
    }
}

/// Longest a writer parked by an in-process [`arm`] waits for
/// [`Armed::release`] before proceeding on its own — the bound that keeps a
/// test that forgot to release (or panicked first) from hanging the writer
/// forever. The env-var (SIGKILL) path has no bound: park-and-die.
pub const IN_PROCESS_PARK_TIMEOUT: Duration = Duration::from_secs(30);

/// Longest [`Armed::wait_parked`] waits for the writer to reach its point.
pub const WAIT_PARKED_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-point in-process arming state, keyed by the writer it is armed for.
/// One-shot per point: the first call by THAT writer takes the arm; every other
/// writer in the process (a sibling test's, a recovery claim's) passes straight
/// through.
struct ArmState {
    writer_id: String,
    parked: Arc<AtomicBool>,
    parked_notify: Arc<Notify>,
    release: Arc<Notify>,
    released: Arc<AtomicBool>,
}

static ARMS: [Mutex<Option<ArmState>>; 2] = [Mutex::new(None), Mutex::new(None)];

/// The test's handle on an in-process arm: wait for the writer to park, then
/// release it. Dropping the handle releases the writer (if it is parked) and
/// disarms the point, so a panicking test never leaves a writer waiting out
/// the park timeout.
pub struct Armed {
    point: MaterializationPoint,
    parked: Arc<AtomicBool>,
    parked_notify: Arc<Notify>,
    release: Arc<Notify>,
    released: Arc<AtomicBool>,
}

/// The writer never reached the armed point within [`WAIT_PARKED_TIMEOUT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitParkedTimeout {
    /// The point that was armed.
    pub point: MaterializationPoint,
}

impl std::fmt::Display for WaitParkedTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no writer reached the armed `{}` point within {:?}",
            self.point.env_name(),
            WAIT_PARKED_TIMEOUT
        )
    }
}

impl std::error::Error for WaitParkedTimeout {}

/// Arm `point` for the same-process two-writer tests: the next time the store
/// whose id is `writer_id` reaches it, that writer parks (bounded by
/// [`IN_PROCESS_PARK_TIMEOUT`]) until [`Armed::release`] — or the returned
/// handle is dropped. Keyed by writer so sibling tests running in the same
/// binary never take each other's arm. Replaces any previous arm on the point.
pub fn arm(point: MaterializationPoint, writer_id: &str) -> Armed {
    let state = ArmState {
        writer_id: writer_id.to_string(),
        parked: Arc::new(AtomicBool::new(false)),
        parked_notify: Arc::new(Notify::new()),
        release: Arc::new(Notify::new()),
        released: Arc::new(AtomicBool::new(false)),
    };
    let handle = Armed {
        point,
        parked: Arc::clone(&state.parked),
        parked_notify: Arc::clone(&state.parked_notify),
        release: Arc::clone(&state.release),
        released: Arc::clone(&state.released),
    };
    *ARMS[point.index()].lock().expect("test-hook arm lock") = Some(state);
    handle
}

impl Armed {
    /// Wait until a writer has parked at the armed point. Errors after
    /// [`WAIT_PARKED_TIMEOUT`] — a test must fail loudly when the writer never
    /// reached the window, rather than pass vacuously.
    pub async fn wait_parked(&self) -> std::result::Result<(), WaitParkedTimeout> {
        let deadline = tokio::time::Instant::now() + WAIT_PARKED_TIMEOUT;
        loop {
            // Register interest before checking the flag so a park that lands
            // between the two steps is not lost.
            let notified = self.parked_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.parked.load(Ordering::SeqCst) {
                return Ok(());
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                if self.parked.load(Ordering::SeqCst) {
                    return Ok(());
                }
                return Err(WaitParkedTimeout { point: self.point });
            }
        }
    }

    /// Whether a writer is currently parked at this arm.
    pub fn is_parked(&self) -> bool {
        self.parked.load(Ordering::SeqCst)
    }

    /// Release the parked writer (idempotent; a release before the park lets
    /// the writer pass straight through).
    pub fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release.notify_waiters();
        self.release.notify_one();
    }
}

impl Drop for Armed {
    fn drop(&mut self) {
        self.release();
        if let Ok(mut slot) = ARMS[self.point.index()].lock() {
            // Only disarm our own arm, not a newer one on the same point.
            if slot
                .as_ref()
                .is_some_and(|s| Arc::ptr_eq(&s.parked, &self.parked))
            {
                *slot = None;
            }
        }
    }
}

/// Park at `point` if it is armed for `writer_id`: by an in-process [`arm`]
/// (bounded park, released by the test), else by
/// [`MATERIALIZATION_CHECKPOINT_ENV`] naming the point together with
/// [`READY_FILE_ENV`] (write the ready file and park forever — the SIGKILL
/// harness; not writer-keyed). Otherwise a single early-returning check.
pub async fn maybe_signal_point(point: MaterializationPoint, writer_id: &str) {
    // In-process arm: one-shot — take it only for the writer it was armed for.
    let taken = {
        let mut slot = ARMS[point.index()].lock().expect("test-hook arm lock");
        if slot.as_ref().is_some_and(|s| s.writer_id == writer_id) {
            slot.take()
        } else {
            None
        }
    };
    if let Some(state) = taken {
        state.parked.store(true, Ordering::SeqCst);
        state.parked_notify.notify_waiters();
        state.parked_notify.notify_one();
        if !state.released.load(Ordering::SeqCst) {
            let released = state.release.notified();
            tokio::pin!(released);
            released.as_mut().enable();
            if !state.released.load(Ordering::SeqCst) {
                // Timeout is the documented bound, not a failure: the writer
                // proceeds and the test's own assertions decide.
                let _ = tokio::time::timeout(IN_PROCESS_PARK_TIMEOUT, released).await;
            }
        }
        return;
    }

    let ready_file = match std::env::var(READY_FILE_ENV).ok() {
        Some(p) => PathBuf::from(p),
        None => return,
    };
    let selected = std::env::var(MATERIALIZATION_CHECKPOINT_ENV)
        .ok()
        .and_then(|v| MaterializationPoint::from_env_value(&v));
    if selected != Some(point) {
        return;
    }
    signal_and_park(ready_file).await;
}

/// [`maybe_signal_point`] at [`MaterializationPoint::Materialization`].
pub async fn maybe_signal_materialization(writer_id: &str) {
    maybe_signal_point(MaterializationPoint::Materialization, writer_id).await;
}

/// [`maybe_signal_point`] at [`MaterializationPoint::TableCreated`].
pub async fn maybe_signal_table_created(writer_id: &str) {
    maybe_signal_point(MaterializationPoint::TableCreated, writer_id).await;
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
