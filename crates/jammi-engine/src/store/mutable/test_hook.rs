//! Test-only checkpoint for the mutable-table sink.
//!
//! Compiled only under `feature = "test-hooks"`. The `mutable_crash_recovery.rs`
//! integration test spawns a child process with [`READY_FILE_ENV`] and
//! [`CHECKPOINT_AFTER_ENV`] set; the child inserts a known number of rows
//! through the sink, this hook fires once the per-write-call row counter
//! crosses the threshold, writes [`READY_FILE_ENV`] so the parent knows the
//! child is mid-transaction, then awaits an unsignalled notifier so the child
//! parks until the parent sends `SIGKILL`. The transaction never commits;
//! RAII rollback delivers the spec's zero-row guarantee.
//!
//! When the env vars are unset (the default for every other test and every
//! production build), [`maybe_signal`] is a single `if` returning early. No
//! production code path observes this module.

use std::path::PathBuf;
use std::sync::OnceLock;

use tokio::sync::Notify;

/// Path the child writes once `rows_so_far >= CHECKPOINT_AFTER_ENV`. The
/// parent polls `try_exists` on this path and `SIGKILL`s the child as soon
/// as it appears.
pub const READY_FILE_ENV: &str = "JAMMI_TEST_CHECKPOINT_READY_FILE";

/// Row threshold the child must cross before signalling. The child increments
/// `rows_so_far` once per `insert_batch` call; a test that wants to fire after
/// the 50th row passes a 50-row batch and sets this to 50.
pub const CHECKPOINT_AFTER_ENV: &str = "JAMMI_TEST_CHECKPOINT_AFTER";

/// Park forever once the signal fires. Park-and-die is the contract: the
/// caller has SIGKILL teed up.
static PARK: OnceLock<Notify> = OnceLock::new();

/// One-shot guard so a test that calls `insert_batch` multiple times only
/// signals once. Without this, the second call would re-write the ready file
/// and re-park (the first park has already returned), wasting wall-clock.
static SIGNALLED: OnceLock<()> = OnceLock::new();

/// Resolve env-driven checkpoint configuration. `Some` only when both vars
/// are set and the threshold is a parseable `u64`.
struct CheckpointConfig {
    ready_file: PathBuf,
    after: u64,
}

fn config() -> Option<CheckpointConfig> {
    let ready_file = std::env::var(READY_FILE_ENV).ok()?;
    let after = std::env::var(CHECKPOINT_AFTER_ENV)
        .ok()?
        .parse::<u64>()
        .ok()?;
    Some(CheckpointConfig {
        ready_file: PathBuf::from(ready_file),
        after,
    })
}

/// Signal-and-park if `rows_so_far` has crossed the configured threshold.
/// First call past the threshold writes the ready file and parks forever on
/// a notifier that no one signals. Subsequent calls are no-ops.
pub async fn maybe_signal(rows_so_far: u64) {
    let cfg = match config() {
        Some(c) => c,
        None => return,
    };
    if rows_so_far < cfg.after {
        return;
    }
    if SIGNALLED.set(()).is_err() {
        return;
    }
    if let Some(parent) = cfg.ready_file.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    tokio::fs::write(&cfg.ready_file, b"ready")
        .await
        .expect("checkpoint ready-file write");
    let park = PARK.get_or_init(Notify::new);
    park.notified().await;
}
