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
use std::sync::OnceLock;

use tokio::sync::Notify;

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
