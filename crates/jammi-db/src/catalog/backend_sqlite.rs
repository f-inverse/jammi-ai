//! SQLite implementation of [`CatalogBackend`] backed by `sqlx::SqlitePool`.
//!
//! # The single-process contract, mechanically enforced
//!
//! The SQLite catalog is **single-process only**: exactly one OS process may
//! have `catalog.db` open at a time (`docs/guide/src/catalog-and-broker.md`,
//! the SQLite row of the backend table). Sharing the file across
//! `jammi-server` replicas, or across an engine process and any other SQLite
//! client, is out of contract. Multiple *connections*, *pools*, *sessions* and
//! *threads* inside ONE process are fully supported and are the normal
//! topology.
//!
//! Documenting that contract is not enforcing it. POSIX `fcntl` advisory locks
//! — the mechanism the default unix VFS coordinates with — are scoped to a
//! *process*, not to a file descriptor or to a library instance
//! (<https://sqlite.org/howtocorrupt.html> §2.2–2.3). Two SQLite **library
//! instances** in one process (the shape `jammi-python` reaches: the extension
//! statically bundles its own amalgamation while CPython's `sqlite3` module
//! links the platform `libsqlite3`) therefore cannot see each other's locks at
//! all. Under the default VFS the consequence is not a refusal but a *signal*:
//! the foreign instance's `sqlite3_close` believes it holds the last
//! connection, checkpoints, then truncates and unlinks `catalog.db-wal` /
//! `catalog.db-shm`, and the engine instance's next touch of the `-shm` page
//! it still has **mmapped** faults past EOF with `SIGBUS` — a process-fatal
//! crash for out-of-contract input, which is never an acceptable failure shape
//! (esc-073).
//!
//! ## The seam: the `unix-excl` VFS
//!
//! On unix the pool opens through SQLite's `unix-excl` VFS
//! (<https://sqlite.org/vfs.html>) instead of the default `unix` VFS. Two
//! properties of that VFS carry the contract, and both are load-bearing here:
//!
//! 1. **Process-scoped exclusive lock.** The first lock any connection in this
//!    process takes on the database file is promoted to an exclusive
//!    `F_WRLCK` over the whole shared range, and it is never released while
//!    any connection in this process holds the file open (`os_unix.c`,
//!    `unixFileLock`: "the only lock ever obtained is an exclusive lock … all
//!    subsequent system locking operations become no-ops. Locking operations
//!    still happen internally, in order to coordinate access between separate
//!    database connections within this process"). Any OTHER process opening
//!    the same file gets `SQLITE_BUSY` after its busy-timeout — a typed
//!    refusal, which is exactly the single-process contract turned into a
//!    mechanism. Connections *within* this process keep coordinating through
//!    the shared in-process inode structure, so the supported multi-pool
//!    topology is unaffected.
//! 2. **Heap wal-index.** Because that exclusive lock means no other process
//!    can participate, the WAL index lives in heap memory and **no `-shm`
//!    file is created or mapped** (`os_unix.c`, `unixOpenSharedMemory`: the
//!    `-shm` is opened only `if( pInode->bProcessLock==0 )`). The engine
//!    therefore never mmaps a file any other party could truncate, and with
//!    `SQLITE_DEFAULT_MMAP_SIZE == 0` in the bundled amalgamation it mmaps
//!    nothing else either — so no engine-side `SIGBUS` is reachable, whatever
//!    a foreign library instance does to the sidecar files.
//!
//! WAL journal mode is unchanged and still in effect; only the *location* of
//! the wal-index moves (file → heap).
//!
//! ## Residual, deliberately not closed
//!
//! A foreign SQLite **library instance inside this same process** is still not
//! arbitrated: its `fcntl` locks and ours are indistinguishable to the kernel,
//! so it can still write the `-wal` behind the engine's back and corrupt the
//! catalog *silently*. No engine-side seam can arbitrate locks it cannot see.
//! The seam converts that topology's worst outcome from "process-fatal signal"
//! to "typed error or wrong data", and the topology itself stays excluded at
//! the call sites that could reach it. Cross-*process* sharing, by contrast,
//! is now fully refused.
//!
//! ## Releasing the file is an awaited event, not a drop
//!
//! Because the exclusive lock is what refuses the second process, "I am
//! finished with this catalog" has to be observable. Dropping a
//! [`super::Catalog`] is not: `sqlx` closes a returned connection from a
//! background task, so the lock and the `catalog.db-wal` sidecar outlive the
//! drop by an unbounded interval. [`CatalogBackend::close`] (surfaced as
//! [`super::Catalog::close`] and [`crate::session::JammiSession::close`]) is
//! the release point — after awaiting it, `catalog.db-wal` is gone and another
//! process opens the directory immediately. It is a *bounded* release rather
//! than an instantaneous barrier: `sqlx` offers no way to await the
//! connection-return task it spawns on drop, so a straggler close can
//! re-create the `-wal` for a few milliseconds. The close absorbs that with a
//! settle window and the oracle bounds what is left.
//!
//! On non-unix targets the platform default VFS is kept: Windows uses
//! mandatory `LockFile` ranges rather than POSIX advisory locks, `unix-excl`
//! does not exist there, and the residual is that the single-process contract
//! remains documentation-only on those targets.

use std::borrow::Cow;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};
use sqlx::ConnectOptions;

use super::backend::{classify, BackendError, BackendKind, CatalogBackend, Transaction, TxOptions};

/// Diagnostic escape hatch naming the SQLite VFS the catalog pool opens with.
///
/// Unset (the only supported configuration) selects the `unix-excl` VFS on
/// unix and the platform default elsewhere. Set to `default` it restores the
/// platform default VFS on every target, which **re-arms the esc-073 `SIGBUS`
/// and the cross-process WAL corruption the module docs describe**. Any other
/// value is passed to SQLite verbatim as a VFS name.
///
/// It exists so the fix can be falsified: a verifier re-runs the esc-073
/// oracle with `JAMMI_SQLITE_VFS=default` and must observe the pre-fix RED.
/// Engaging it logs a `WARN`. It is a test/diagnostic knob and is never set in
/// production.
pub const SQLITE_VFS_ENV: &str = "JAMMI_SQLITE_VFS";

/// The VFS name the next [`SqliteBackend::open`] will use, or `None` for the
/// platform default.
///
/// Returns `Some("unix-excl")` on unix and `None` on other targets, unless
/// [`SQLITE_VFS_ENV`] overrides it. Exposed so tests and diagnostics can
/// assert which seam is engaged without inferring it from side effects.
pub fn catalog_vfs() -> Option<Cow<'static, str>> {
    match std::env::var(SQLITE_VFS_ENV) {
        Ok(name) if name == "default" => {
            tracing::warn!(
                env = SQLITE_VFS_ENV,
                "SQLite catalog opening with the PLATFORM DEFAULT VFS: the single-process \
                 contract is no longer mechanically enforced and a foreign SQLite library \
                 instance can crash this process (esc-073). Diagnostic use only."
            );
            None
        }
        Ok(name) if !name.is_empty() => {
            tracing::warn!(
                env = SQLITE_VFS_ENV,
                vfs = %name,
                "SQLite catalog opening with an operator-supplied VFS instead of the \
                 single-process seam. Diagnostic use only."
            );
            Some(Cow::Owned(name))
        }
        // Unset, empty, or non-UTF-8: the supported configuration.
        _ => {
            if cfg!(unix) {
                Some(Cow::Borrowed("unix-excl"))
            } else {
                None
            }
        }
    }
}

/// Consecutive clean sidecar observations (one wait step apart, ~1 ms each)
/// that [`CatalogBackend::close`] requires before it calls the file released.
///
/// A single clean observation is not enough: a straggler connection closing
/// after the pool reports itself empty runs a checkpoint that re-creates a
/// `-wal` for a few milliseconds. This settle window is the measured cost of
/// covering that; it is paid once, on a shutdown path.
const CLOSE_SETTLE_STEPS: usize = 25;

/// Ceiling on the post-drain wait for SQLite's own release evidence.
///
/// Much tighter than the pool drain's ceiling, because this half is
/// best-effort: SQLite deletes the `-wal` only when its close-time PASSIVE
/// checkpoint succeeds, and a checkpoint it declines to complete leaves the
/// file behind legitimately. Waiting minutes for a file that is never coming
/// would turn a shutdown into a stall; two seconds is three orders of
/// magnitude above the measured straggler window and still imperceptible.
///
/// This ceiling is also the *whole* cost of closing one pool while a second
/// pool in this process still holds the same file: the `-wal` belongs to the
/// file, not to a pool, so it cannot disappear while the survivor is live and
/// the settle loop necessarily runs to the deadline. Bounded, once, on a
/// shutdown path — see [`CatalogBackend::close`]'s "evidence about the FILE"
/// section.
const CLOSE_SIDECAR_CEILING: Duration = Duration::from_secs(2);

/// SQLite's primary result code `SQLITE_BUSY`. `sqlx` surfaces the *extended*
/// code, whose low byte is the primary one, so every `SQLITE_BUSY_*` extended
/// code (`_RECOVERY` 261, `_SNAPSHOT` 517, `_TIMEOUT` 773) reduces to this.
const SQLITE_BUSY_PRIMARY: i32 = 5;

/// True when `err` is an `SQLITE_BUSY`-class database error — the shape a
/// second *process* gets from the `unix-excl` seam once this process holds the
/// file.
fn is_busy(err: &sqlx::Error) -> bool {
    let sqlx::Error::Database(db_err) = err else {
        return false;
    };
    db_err
        .code()
        .and_then(|c| c.parse::<i32>().ok())
        .is_some_and(|code| code & 0xff == SQLITE_BUSY_PRIMARY)
}

/// SQLite-backed catalog. Wraps a connection pool with WAL mode + 5 s busy
/// timeout, matching the original `r2d2_sqlite`-based behaviour.
///
/// The pool is opened through the process-exclusive `unix-excl` VFS on unix;
/// see the module documentation for the single-process contract this enforces
/// and the residual it leaves.
pub struct SqliteBackend {
    pool: SqlitePool,
    /// The database file this pool is open on. Kept so [`CatalogBackend::close`]
    /// can wait on SQLite's own release evidence — the disappearance of the
    /// sidecars — rather than only on the pool's connection accounting.
    path: std::path::PathBuf,
}

impl SqliteBackend {
    /// Open (or create) the catalog database at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] naming the single-process
    /// contract when another **process** already holds `path`: on unix the
    /// `unix-excl` VFS holds a process-scoped exclusive lock for as long as
    /// any connection here has the file open, so a second process's open
    /// waits out the 5 s busy timeout and is then refused with
    /// `SQLITE_BUSY`. That refusal is bounded and typed — never a hang, never
    /// a signal. Other failures classify through
    /// [`classify`].
    pub async fn open(path: &Path) -> Result<Arc<Self>, BackendError> {
        let mut opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .log_statements(tracing::log::LevelFilter::Trace);
        if let Some(vfs) = catalog_vfs() {
            opts = opts.vfs(vfs);
        }

        // Pool reaping is left at `sqlx`'s defaults (10 min idle, 30 min
        // lifetime) ON PURPOSE, even though the `unix-excl` process lock lives
        // only as long as some connection in this process holds the file open.
        // Pinning it by disabling both reapers was tried and reverted:
        // `sqlx-sqlite` runs every connection on its own OS thread, so a
        // process with many catalogs would accumulate threads for its whole
        // lifetime. The residual is bounded and typed rather than silent — if
        // a catalog goes fully idle long enough for its last connection to be
        // reaped, another process can take the file, and this process's next
        // query is refused with the `SQLITE_BUSY` error `open` documents,
        // never a corrupted WAL.
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(opts)
            .await
            .map_err(|err| {
                if is_busy(&err) {
                    BackendError::Unavailable(format!(
                        "SQLite catalog {} is locked and could not be opened within the 5 s busy \
                         timeout. The SQLite catalog is single-process only (one jammi process \
                         per catalog directory) and that contract is enforced by a \
                         process-exclusive lock, so the expected cause is another process holding \
                         this directory: stop it, or move to the Postgres backend to share one \
                         catalog across processes. Underlying error: {err}",
                        path.display()
                    ))
                } else {
                    classify(err)
                }
            })?;

        Ok(Arc::new(Self {
            pool,
            path: path.to_path_buf(),
        }))
    }

    /// Path of a SQLite sidecar for this backend's database file
    /// (`suffix` is `"-wal"` or `"-shm"`).
    fn sidecar(&self, suffix: &str) -> std::path::PathBuf {
        let mut name = self.path.clone().into_os_string();
        name.push(suffix);
        std::path::PathBuf::from(name)
    }
}

impl CatalogBackend for SqliteBackend {
    /// Open a transaction and run `f` within it.
    ///
    /// **Invariant: this future must be driven by a live tokio runtime — never
    /// blocked on from a runtime worker thread.** The uncancellable `BEGIN`
    /// (see the body) runs on a detached `tokio::spawn(...).await`, so a
    /// runtime must be free to poll that spawned task while this future awaits
    /// its join handle. Awaiting this future normally (on either a
    /// multi-thread or a single-thread runtime) is fine: the executor
    /// interleaves the spawned begin with this await. What deadlocks is
    /// `Handle::current().block_on(transaction(..))` *from inside* a runtime
    /// worker — that pins the worker on the join handle and the spawned begin
    /// never gets polled. The Postgres backend does not spawn and carries no
    /// such constraint.
    fn transaction<'a, F, R>(
        &'a self,
        opts: TxOptions,
        f: F,
    ) -> Pin<Box<dyn Future<Output = Result<R, BackendError>> + Send + 'a>>
    where
        F: for<'tx> FnOnce(
                &'tx mut Transaction<'tx>,
            )
                -> Pin<Box<dyn Future<Output = Result<R, BackendError>> + Send + 'tx>>
            + Send
            + 'a,
        R: Send + 'a,
    {
        Box::pin(async move {
            // SQLite has no SET TRANSACTION ISOLATION LEVEL; isolation is fixed
            // by the journal mode (WAL gives snapshot reads). The write/read
            // distinction is carried entirely by the BEGIN mode, which `sqlx`'s
            // default `Pool::begin` (always DEFERRED) cannot express — so we
            // open the transaction through `Pool::begin_with`, which runs our
            // custom BEGIN yet still yields a sqlx `Transaction` that rolls back
            // on drop/cancel.
            //
            // A write transaction MUST take the database write lock at BEGIN
            // time (`BEGIN IMMEDIATE`): under WAL, two DEFERRED transactions
            // that each read then upgrade to a write deadlock with
            // SQLITE_BUSY_SNAPSHOT, which `busy_timeout` cannot break (waiting
            // never resolves a snapshot-upgrade conflict). IMMEDIATE makes
            // concurrent writers serialise on `busy_timeout` instead. A
            // read-only transaction stays DEFERRED so reads take a snapshot
            // without serialising against each other or against writers.
            let begin = if opts.read_only {
                "BEGIN DEFERRED"
            } else {
                "BEGIN IMMEDIATE"
            };
            let _ = (opts.isolation, opts.read_only);

            // The BEGIN itself must be uncancellable. `Pool::begin_with` issues
            // the `BEGIN` statement and only then constructs the sqlx
            // `Transaction` whose drop guard rolls back. If the caller's future
            // is dropped *while that begin is in flight* — after the worker has
            // run `BEGIN IMMEDIATE` and bumped its per-connection transaction
            // depth, but before the `Transaction` exists — there is no guard to
            // roll it back: the pooled connection returns to the pool still
            // inside a transaction, holding the WAL write lock. Its next checkout
            // then fails (`InvalidSavePointStatement`, because a custom `BEGIN`
            // is illegal at depth > 0), and every other writer starves on the
            // leaked write lock (`database is locked`). Running the begin on a
            // detached task closes the window: a cancelled caller drops only the
            // `JoinHandle`, the task still drives the begin to a fully-formed
            // `Transaction`, and that `Transaction` then drops through its own
            // guard — rolling back and returning the connection clean.
            let pool = self.pool.clone();
            let mut tx = tokio::spawn(async move { pool.begin_with(begin).await })
                .await
                .map_err(|join| {
                    BackendError::Unavailable(format!("transaction begin task failed: {join}"))
                })?
                .map_err(classify)?;

            // Scope wrapper so its borrow of `tx` ends before we move `tx`
            // into commit/rollback. The HRTB on `f` borrows wrapper for its
            // entire lifetime, so wrapper must drop before tx moves.
            let outcome = {
                let mut wrapper = Transaction::new_sqlite(&mut tx);
                f(&mut wrapper).await
            };

            match outcome {
                Ok(value) => {
                    tx.commit().await.map_err(classify)?;
                    Ok(value)
                }
                Err(err) => {
                    let _ = tx.rollback().await;
                    Err(err)
                }
            }
        })
    }

    fn migrate(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
        Box::pin(async move { super::migrations::run(self).await })
    }

    fn ping(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
        Box::pin(async move {
            sqlx::query("SELECT 1")
                .execute(&self.pool)
                .await
                .map_err(classify)?;
            Ok(())
        })
    }

    /// Close the pool and wait for every connection to shut down.
    ///
    /// This is what releases the `unix-excl` process lock and lets SQLite
    /// delete `catalog.db-wal` on the last close: the lock lives as long as
    /// SOME connection in this process holds the file, and `sqlx` closes a
    /// returned connection from a background task, so a plain `drop` releases
    /// nothing at any bounded moment. `sqlx`'s own `Pool::close` is not enough
    /// either — see the crate-private `close_pool_and_drain` helper in
    /// `catalog::backend`.
    ///
    /// Draining the pool's connection accounting is still not the whole
    /// barrier: `Pool::size()` reaching zero was measured to precede the last
    /// `sqlite3_close` by a few milliseconds on roughly one cycle in seven, and
    /// it is that call — not the pool's bookkeeping — that drops the
    /// process-exclusive lock and deletes the sidecars. So this additionally
    /// waits on SQLite's own evidence of release: the disappearance of
    /// `<db>-wal`, which the last `sqlite3_close` deletes (no
    /// `SQLITE_FCNTL_PERSIST_WAL` is set here). A database that never entered
    /// WAL has none, so the wait is empty. It is bounded and warns rather than
    /// hanging if it expires — SQLite deletes the `-wal` only when its
    /// close-time checkpoint completes, so the wait is best-effort by
    /// construction.
    ///
    /// # The `-wal` is evidence about the FILE, not about this pool
    ///
    /// `-wal` disappearance is evidence that the LAST connection to this
    /// database in this process closed — not that *these* connections did. The
    /// seam is single-*process*, so a second pool on the same file inside this
    /// process is legal and supported (two [`super::Catalog`] handles on one
    /// directory), and while that second pool is live the `-wal` cannot go
    /// away. Closing the first pool therefore observes the `-wal` for the whole
    /// `CLOSE_SIDECAR_CEILING` and then warns, even though that pool's own
    /// connections were released promptly and nothing is wrong.
    ///
    /// That is a cost and a misleading log line, not a correctness defect: the
    /// wait is bounded by construction, the surviving pool keeps working, and
    /// the caller's connections are already gone when the settle loop starts
    /// (`close_pool_and_drain` has returned). Callers that close one of several
    /// live pools should expect this close to take up to the ceiling. A
    /// per-pool release signal would need evidence SQLite does not expose at
    /// this layer, so the mechanism is deliberately unchanged; the warning
    /// below names this as an expected cause so an operator reading it is not
    /// sent hunting for a leak that is not there.
    fn close(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            super::backend::close_pool_and_drain(&self.pool).await;

            // Then wait for SQLite's own evidence of release, and require it
            // to be STABLE rather than merely observed once. `sqlx` returns a
            // dropped connection to the pool from a spawned task, so a
            // straggler can still be closing after the pool's accounting reads
            // empty — and that straggler's close runs a checkpoint which
            // briefly RE-CREATES a `-wal` a previous close had already deleted
            // (measured: absent, then present, then absent again a few ms
            // later). So the census must hold clean across a settle window,
            // not merely at one instant.
            //
            // Only the `-wal` is watched. Its deletion coincides with the last
            // `sqlite3_close` under both VFSes, whereas a `-shm` on disk need
            // not be ours at all: under the seam this process never creates
            // one, so any `-shm` present is a leftover from a pre-seam crash
            // or belongs to a foreign library instance — and waiting for a
            // file nobody here will ever delete would turn every close on an
            // upgraded catalog directory into a full timeout.
            let wal = self.sidecar("-wal");
            let clean = || !wal.exists();
            let deadline = std::time::Instant::now() + CLOSE_SIDECAR_CEILING;
            let mut consecutive_clean = 0usize;
            loop {
                if clean() {
                    consecutive_clean += 1;
                    if consecutive_clean >= CLOSE_SETTLE_STEPS {
                        return;
                    }
                } else {
                    consecutive_clean = 0;
                }
                if std::time::Instant::now() >= deadline {
                    tracing::warn!(
                        path = %self.path.display(),
                        ceiling_secs = CLOSE_SIDECAR_CEILING.as_secs(),
                        "SQLite catalog close: this pool's connections are released, but `-wal` \
                         is still present after the bounded settle wait. Expected when ANOTHER \
                         live pool in this process still holds the same file (the seam is \
                         single-PROCESS, so that is legal) or when SQLite's close-time PASSIVE \
                         checkpoint declined to complete. The wait is bounded and the close is \
                         done; this is not a hang and not a lost write."
                    );
                    return;
                }
                super::backend::shutdown_wait_step().await;
            }
        })
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Sqlite
    }
}
