//! esc-073 RED oracle (`closes_escape: esc-073`) — a foreign SQLite **library
//! instance** coexisting with a live engine pool on one catalog file must never
//! kill the process.
//!
//! Contract under test: a foreign in-process SQLite connection writing to
//! `catalog.db` while an engine session's pool is live is OUT OF CONTRACT
//! (`docs/guide/src/catalog-and-broker.md:91`, single-process / engine-owned) —
//! but out-of-contract input must fail with a typed refusal or an
//! `SQLITE_BUSY`-class error, never a process-fatal signal. No topology
//! reachable from a supported or test-harness seam may `SIGBUS` the process.
//!
//! ## The library-instance fact this harness rests on
//!
//! `jammi-python`'s native extension links jammi-db, which enables
//! `sqlx/sqlite` → `sqlx-sqlite/bundled` → `libsqlite3-sys/bundled`
//! (`cargo tree -p jammi-python -i libsqlite3-sys -e features`): the extension
//! **statically bundles its own SQLite amalgamation** (3.46.0 in
//! `libsqlite3-sys 0.30.1`). CPython's `sqlite3` module is a separate shared
//! object linked against the platform `libsqlite3` (3.53.4 here). So
//! `test_conformance.py`'s raw-`sqlite3` seed and the engine's `sqlx` pool are
//! two DIFFERENT SQLite library instances inside ONE process.
//!
//! That is the whole hazard, and it does not need Python to reproduce: this
//! test binary already carries the bundled SQLite statically, so `dlopen`-ing
//! the platform `libsqlite3` gives the identical two-instances-one-process
//! topology with no extension build, no interpreter, and no test-collection
//! ordering. The foreign connection here performs exactly the sequence
//! `test_conformance.py:968-978`'s `_set_metrics` performs: open → `UPDATE` →
//! commit → close.
//!
//! ## Hypothesis under test (to confirm or refute, not to assume)
//!
//! POSIX `fcntl` advisory locks are per-PROCESS: two library instances in one
//! process cannot see each other's locks, and closing any descriptor for the
//! file drops that process's locks on it
//! (<https://sqlite.org/howtocorrupt.html> §2.2). A closing connection that
//! believes it is the LAST connection to a WAL database checkpoints and then
//! truncates + unlinks the `-wal`/`-shm` files. The other instance still has
//! the `-shm` **mmapped**, so its next touch of a page past the new EOF is a
//! `SIGBUS`.
//!
//! The [`keeper`](Role::Keeper) arm is the differential that decides this: it
//! is byte-identical to the colliding arm except that ONE extra foreign
//! connection stays open for the whole run, so the foreign library instance
//! never believes a closing connection is the last one and never runs the
//! close-time checkpoint/truncate. If the crash needs that truncate, the
//! keeper arm survives; if it crashes too, the hypothesis is refuted.
//!
//! ## Shape
//!
//! Every arm runs in a CHILD PROCESS (this same test binary re-executed with
//! `--exact` and a role env var), so a fatal signal is captured as a wait
//! status instead of taking the test runner down with it. The parent loops
//! until it reproduces or exhausts its attempt budget and reports the rate and
//! the exact signal.
//!
//! FAILURE, per the row's control: ANY process-fatal signal (`SIGBUS`,
//! `SIGSEGV`, `SIGABRT`, …) from either side. An `SQLITE_BUSY` / typed error
//! surfaced to either connection is the acceptable refusal shape and is
//! reported, not failed.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{SqlValue, TxOptions};
use jammi_db::catalog::Catalog;

/// Env var carrying the child's role. Absent ⇒ this process is the parent.
const ROLE_ENV: &str = "JAMMI_ESC073_ROLE";

/// Attempt budget per arm. The row records a historical ~2-in-4 reproduction
/// rate for the pytest shape; the parent stops at the first reproduction, so a
/// deterministic mechanism costs one attempt and a rare one still gets 40.
const ATTEMPTS: usize = 40;

/// Foreign open/write/close cycles per child. Each cycle is one full
/// `_set_metrics`-shaped visit: open → `UPDATE` → commit → close.
const FOREIGN_CYCLES: usize = 40;

/// Engine-side load tasks hammering the same file while the foreign cycles run.
/// They exist to keep the `-shm` mapping hot, which is what makes a truncation
/// underneath it observable rather than merely possible.
const ENGINE_LOAD_TASKS: usize = 4;

/// Ceiling on one child. A child that hangs is a FAILURE of the same weight as
/// a crash — nothing is retried after it fires.
const CHILD_CEILING: Duration = Duration::from_secs(120);

/// Platform `libsqlite3` candidates, most-specific first. The first entry is
/// the exact library CPython's `_sqlite3` extension links here (`otool -L`), so
/// the harness collides against the same instance the conformance suite does.
const FOREIGN_LIB_CANDIDATES: &[&str] = &[
    "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib",
    "/usr/lib/libsqlite3.dylib",
    "libsqlite3.so.0",
    "libsqlite3.so",
];

/// What a child does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Engine pool live; foreign connections opened, written, and CLOSED in a
    /// loop — the `_set_metrics` shape. This is the arm the row's control
    /// judges.
    Collide,
    /// Identical to [`Role::Collide`] except one foreign connection stays open
    /// for the whole run, so no foreign close is ever the last connection.
    /// The mechanism differential.
    Keeper,
    /// The reverse direction: a foreign connection stays open and keeps
    /// committing while ENGINE pools are opened and dropped underneath it, so
    /// the engine's close-time checkpoint is the one that can truncate a
    /// mapping the foreign instance holds.
    EngineChurn,
    /// esc-071's SYMPTOM in esc-073's topology, with no concurrency at all:
    /// engine commits → foreign `_set_metrics` cycle → engine reads back,
    /// strictly sequential. Answers whether the two-library topology alone
    /// (not pooling, not races) is what makes a committed value invisible.
    StaleRead,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Collide => "collide",
            Role::Keeper => "keeper",
            Role::EngineChurn => "engine-churn",
            Role::StaleRead => "stale-read",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "collide" => Some(Role::Collide),
            "keeper" => Some(Role::Keeper),
            "engine-churn" => Some(Role::EngineChurn),
            "stale-read" => Some(Role::StaleRead),
            _ => None,
        }
    }
}

/// Exit code a child uses for "this arm's own observation tripped" — a typed
/// error or a vanished table, as opposed to a fatal signal or a harness fault.
/// Kept distinct so the parent can report WHICH kind of RED it got.
const EXIT_TRIPPED: i32 = 66;

/// Exit code for the residual outcome: one instance read a *stale but
/// well-formed* image of the database — no signal, no corruption, just an
/// invisible commit. Out of contract and, per the seam's module docs,
/// unclosable from the engine side.
const EXIT_STALE: i32 = 67;

/// Exit code for the pre-fix outcome the seam is required to have removed: the
/// database file itself came back malformed (`SQLITE_CORRUPT`). Reported as
/// its own class because a corrupt catalog is a materially worse failure than
/// a stale read, and the fix's claim is precisely that this class is gone.
const EXIT_CORRUPT: i32 = 68;

/// Exit code for "no platform `libsqlite3` to collide with". The PARENT
/// decides what that means: a loud local skip, a hard failure under `CI`.
const EXIT_NO_FOREIGN_LIB: i32 = 77;

/// Report an observation failure from the child and exit with
/// [`EXIT_TRIPPED`]. Never a panic: a panic's exit code is indistinguishable
/// from a harness fault.
fn tripped(msg: String) -> ! {
    tripped_as(EXIT_TRIPPED, msg)
}

/// Report an observation failure classified into `code`
/// ([`EXIT_TRIPPED`] / [`EXIT_STALE`] / [`EXIT_CORRUPT`]).
fn tripped_as(code: i32, msg: String) -> ! {
    eprintln!("[child] TRIPPED({code}): {msg}");
    std::process::exit(code);
}

/// Classify an engine-side error string into [`EXIT_CORRUPT`] (the file itself
/// is damaged) or [`EXIT_TRIPPED`] (a typed refusal of any other shape).
///
/// SQLite reports a damaged file as `SQLITE_CORRUPT` (primary code 11,
/// "database disk image is malformed") or as `SQLITE_NOTADB` (26). Matching on
/// both the code and the message keeps the classification from silently
/// degrading if `sqlx` changes how it renders one of them.
fn classify_engine_error(err: &str) -> i32 {
    let lower = err.to_ascii_lowercase();
    if lower.contains("(code: 11)")
        || lower.contains("(code: 26)")
        || lower.contains("malformed")
        || lower.contains("not a database")
    {
        EXIT_CORRUPT
    } else {
        EXIT_TRIPPED
    }
}

/// Sequential rounds for [`Role::StaleRead`]. Values are pairwise distinct, so
/// a stale read can never coincidentally match the expected value.
const STALE_ROUNDS: usize = 8;

// ── The foreign SQLite library instance ─────────────────────────────────────

/// A `dlopen`-ed platform `libsqlite3` — a SECOND SQLite library instance in
/// this process, alongside the one `libsqlite3-sys/bundled` linked statically.
struct ForeignSqlite {
    handle: *mut c_void,
    open_v2: unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, *const c_char) -> c_int,
    exec: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *mut c_void,
        *mut c_void,
        *mut *mut c_char,
    ) -> c_int,
    close: unsafe extern "C" fn(*mut c_void) -> c_int,
    libversion: unsafe extern "C" fn() -> *const c_char,
    changes: unsafe extern "C" fn(*mut c_void) -> c_int,
}

/// SQLite's `SQLITE_OPEN_READWRITE` — no `CREATE`: the engine owns creation, so
/// a foreign open that would have to create the file is a harness bug, not a
/// collision.
const SQLITE_OPEN_READWRITE: c_int = 0x0000_0002;

impl ForeignSqlite {
    /// Load the first available platform `libsqlite3`, or `None` when the
    /// platform ships none at a known path (reported loudly by the caller —
    /// never a silent pass).
    fn load() -> Option<Self> {
        for candidate in FOREIGN_LIB_CANDIDATES {
            let name = CString::new(*candidate).unwrap();
            // SAFETY: `name` is a valid NUL-terminated path; `dlopen` returns
            // null on failure, which is checked.
            let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if handle.is_null() {
                continue;
            }
            // SAFETY: `handle` is a live `dlopen` handle. Each symbol is
            // transmuted to the signature `sqlite3.h` declares for it; a
            // missing symbol yields null, which aborts the load below.
            unsafe {
                let sym = |n: &str| {
                    let c = CString::new(n).unwrap();
                    libc::dlsym(handle, c.as_ptr())
                };
                let (o, e, c, v, n) = (
                    sym("sqlite3_open_v2"),
                    sym("sqlite3_exec"),
                    sym("sqlite3_close"),
                    sym("sqlite3_libversion"),
                    sym("sqlite3_changes"),
                );
                if o.is_null() || e.is_null() || c.is_null() || v.is_null() || n.is_null() {
                    libc::dlclose(handle);
                    continue;
                }
                return Some(Self {
                    handle,
                    open_v2: std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(
                            *const c_char,
                            *mut *mut c_void,
                            c_int,
                            *const c_char,
                        ) -> c_int,
                    >(o),
                    exec: std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(
                            *mut c_void,
                            *const c_char,
                            *mut c_void,
                            *mut c_void,
                            *mut *mut c_char,
                        ) -> c_int,
                    >(e),
                    close: std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut c_void) -> c_int,
                    >(c),
                    libversion: std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn() -> *const c_char,
                    >(v),
                    changes: std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut c_void) -> c_int,
                    >(n),
                });
            }
        }
        None
    }

    fn version(&self) -> String {
        // SAFETY: `sqlite3_libversion` returns a static NUL-terminated string.
        unsafe { CStr::from_ptr((self.libversion)()) }
            .to_string_lossy()
            .into_owned()
    }

    /// Open a connection on `path`. Returns the raw handle and the return code;
    /// a non-zero code is reported (an `SQLITE_BUSY`-class refusal is the
    /// ACCEPTABLE outcome for this out-of-contract topology).
    fn open(&self, path: &Path) -> Result<ForeignConn<'_>, c_int> {
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut db: *mut c_void = std::ptr::null_mut();
        // SAFETY: `cpath` is a valid NUL-terminated path and `db` is a valid
        // out-pointer; the null VFS argument selects the default VFS.
        let rc = unsafe {
            (self.open_v2)(
                cpath.as_ptr(),
                &mut db,
                SQLITE_OPEN_READWRITE,
                std::ptr::null(),
            )
        };
        if rc != 0 || db.is_null() {
            if !db.is_null() {
                // SAFETY: a failed open still yields a handle that must be closed.
                unsafe { (self.close)(db) };
            }
            return Err(rc);
        }
        Ok(ForeignConn { lib: self, db })
    }
}

impl Drop for ForeignSqlite {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `dlopen` and is closed exactly once.
        unsafe { libc::dlclose(self.handle) };
    }
}

/// One connection on the foreign library instance. Closed on drop — the same
/// `open → write → commit → close` cycle `_set_metrics` performs.
struct ForeignConn<'a> {
    lib: &'a ForeignSqlite,
    db: *mut c_void,
}

impl ForeignConn<'_> {
    /// Run `sql`, returning the SQLite result code and any error message.
    fn exec(&self, sql: &str) -> (c_int, String) {
        let csql = CString::new(sql).unwrap();
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: `self.db` is a live connection from `sqlite3_open_v2`, `csql`
        // is NUL-terminated, and the callback/arg pair is null (no row
        // callback). `err`, when set, is a `sqlite3_malloc` string; it is only
        // read here and deliberately not freed (the process is short-lived and
        // `sqlite3_free` is not among the symbols this harness binds).
        let rc = unsafe {
            (self.lib.exec)(
                self.db,
                csql.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut err,
            )
        };
        let message = if err.is_null() {
            String::new()
        } else {
            // SAFETY: non-null `err` is a NUL-terminated string from SQLite.
            unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned()
        };
        (rc, message)
    }

    /// Rows changed by the most recent statement on this connection. Used as
    /// the foreign side's READ oracle: a guarded `UPDATE … WHERE v = <expected>`
    /// that changes one row proves this instance observed `<expected>`, and one
    /// that changes zero rows proves it did not — no row-callback FFI needed.
    fn changes(&self) -> c_int {
        // SAFETY: `self.db` is a live connection from `sqlite3_open_v2`.
        unsafe { (self.lib.changes)(self.db) }
    }
}

impl Drop for ForeignConn<'_> {
    fn drop(&mut self) {
        // SAFETY: `db` is a live connection, closed exactly once. This close is
        // the operation under test: for a connection the foreign instance
        // believes is the last one on a WAL database, it checkpoints and then
        // truncates + unlinks `-wal`/`-shm`.
        unsafe { (self.lib.close)(self.db) };
    }
}

// ── Child ───────────────────────────────────────────────────────────────────

/// Byte size of `path`, or `-1` when it does not exist — the `-shm`/`-wal`
/// truncation/unlink evidence, sampled around each foreign close.
fn file_len(path: &Path) -> i64 {
    std::fs::metadata(path)
        .map(|m| m.len() as i64)
        .unwrap_or(-1)
}

/// Open an engine catalog and create the probe row both sides write. Used for
/// the FIRST open of a fresh directory, where a failure is a harness fault.
fn open_engine(rt: &tokio::runtime::Runtime, dir: &Path) -> Arc<Catalog> {
    Arc::new(rt.block_on(Catalog::open(dir)).expect("engine catalog"))
}

/// Re-open an engine catalog on a directory a foreign library instance has
/// been writing. A failure here is an ACCEPTABLE refusal, not a harness fault:
/// the foreign instance's writes are not arbitrated with this one's, so the
/// image the engine finds may genuinely be unusable. The row's control is that
/// the process stays alive and says so.
fn try_open_engine(rt: &tokio::runtime::Runtime, dir: &Path) -> Result<Arc<Catalog>, String> {
    rt.block_on(Catalog::open(dir))
        .map(Arc::new)
        .map_err(|e| e.to_string())
}

fn seed_probe(rt: &tokio::runtime::Runtime, catalog: &Catalog) {
    let backend = catalog.backend_arc();
    rt.block_on(backend.transaction(TxOptions::default(), |tx| {
        Box::pin(async move {
            tx.execute(
                "CREATE TABLE IF NOT EXISTS esc073_probe (id INTEGER PRIMARY KEY, v TEXT)",
                &[],
            )
            .await?;
            tx.execute(
                "INSERT OR REPLACE INTO esc073_probe (id, v) VALUES (1, 'seed')",
                &[],
            )
            .await?;
            Ok(())
        })
    }))
    .expect("seed probe row");
}

/// Engine-side write of the probe row, through the catalog backend.
fn engine_write(
    rt: &tokio::runtime::Runtime,
    catalog: &Catalog,
    value: &str,
) -> Result<(), String> {
    let backend = catalog.backend_arc();
    let value = value.to_string();
    rt.block_on(backend.transaction(TxOptions::default(), move |tx| {
        Box::pin(async move {
            tx.execute(
                "UPDATE esc073_probe SET v = $1 WHERE id = 1",
                &[SqlValue::TextOwned(value)],
            )
            .await?;
            Ok(())
        })
    }))
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// Engine-side read of the probe row, through the catalog backend.
fn engine_read(rt: &tokio::runtime::Runtime, catalog: &Catalog) -> Result<Option<String>, String> {
    let backend = catalog.backend_arc();
    rt.block_on(backend.transaction(
        TxOptions {
            read_only: true,
            ..Default::default()
        },
        |tx| {
            Box::pin(async move {
                tx.query_opt("SELECT v FROM esc073_probe WHERE id = 1", &[], |r| {
                    r.get::<String>("v")
                })
                .await
            })
        },
    ))
    .map_err(|e| e.to_string())
}

/// Engine-side load: read-then-write transactions in a tight loop, which keeps
/// this instance's WAL index (`-shm`) mapped and actively touched.
fn spawn_engine_load(
    rt: &tokio::runtime::Runtime,
    catalog: &Arc<Catalog>,
    stop: &Arc<AtomicBool>,
    errors: &Arc<AtomicU64>,
) -> Vec<tokio::task::JoinHandle<()>> {
    (0..ENGINE_LOAD_TASKS)
        .map(|n| {
            let catalog = Arc::clone(catalog);
            let stop = Arc::clone(stop);
            let errors = Arc::clone(errors);
            rt.spawn(async move {
                let mut i: i64 = 0;
                while !stop.load(Ordering::Relaxed) {
                    let backend = catalog.backend_arc();
                    let value = format!("engine-{n}-{i}");
                    let res = backend
                        .transaction(TxOptions::default(), move |tx| {
                            Box::pin(async move {
                                let _ = tx
                                    .query_opt(
                                        "SELECT v FROM esc073_probe WHERE id = 1",
                                        &[],
                                        |r| r.get::<String>("v"),
                                    )
                                    .await?;
                                tx.execute(
                                    "UPDATE esc073_probe SET v = $1 WHERE id = 1",
                                    &[SqlValue::TextOwned(value)],
                                )
                                .await?;
                                Ok(())
                            })
                        })
                        .await;
                    if let Err(err) = res {
                        // An error is the ACCEPTABLE refusal shape for this
                        // out-of-contract topology — counted and reported, not
                        // fatal. Only a signal is fatal.
                        errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("[child] engine error (acceptable refusal shape): {err}");
                    }
                    i += 1;
                    tokio::task::yield_now().await;
                }
            })
        })
        .collect()
}

/// The child body. Never returns — always `exit`s, so a clean run is
/// distinguishable from a signal by wait status alone.
fn child_main(role: Role) -> ! {
    let dir = tempfile::tempdir().expect("child tempdir");
    let db_path: PathBuf = dir.path().join("catalog.db");
    let shm = dir.path().join("catalog.db-shm");
    let wal = dir.path().join("catalog.db-wal");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("child runtime");

    let catalog = open_engine(&rt, dir.path());
    seed_probe(&rt, &catalog);

    let Some(lib) = ForeignSqlite::load() else {
        eprintln!("[child] SKIP: no platform libsqlite3 at any of {FOREIGN_LIB_CANDIDATES:?}");
        std::process::exit(EXIT_NO_FOREIGN_LIB);
    };
    let bundled = rt
        .block_on(catalog.backend_arc().transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt("SELECT sqlite_version() AS v", &[], |r| {
                        r.get::<String>("v")
                    })
                    .await
                })
            },
        ))
        .expect("bundled version")
        .unwrap_or_default();
    eprintln!(
        "[child role={}] bundled(sqlx, static)={bundled} foreign(dlopen)={}",
        role.as_str(),
        lib.version()
    );
    assert_ne!(
        bundled,
        lib.version(),
        "[child] the two SQLite instances report the same version — the harness may have \
         bound one instance twice"
    );

    let stop = Arc::new(AtomicBool::new(false));
    let engine_errors = Arc::new(AtomicU64::new(0));
    // The stale-read arm is deliberately SEQUENTIAL: no background load, so a
    // stale observation there cannot be blamed on a race.
    let load = if role == Role::StaleRead {
        Vec::new()
    } else {
        spawn_engine_load(&rt, &catalog, &stop, &engine_errors)
    };

    // Give the engine load a moment to actually map the `-shm`.
    std::thread::sleep(Duration::from_millis(50));
    eprintln!("[child] shm={} wal={}", file_len(&shm), file_len(&wal));

    let mut refusals = 0usize;
    match role {
        Role::Collide | Role::Keeper => {
            // The keeper arm holds ONE foreign connection open for the whole
            // run; the colliding arm holds none.
            //
            // The keeper must actually TOUCH the database: `sqlite3_open_v2`
            // is lazy — it opens no file and joins no WAL index until the
            // first statement runs — so an unused keeper would leave the
            // foreign instance's shm refcount at zero and the differential
            // would test nothing. The read below forces the foreign instance
            // to open and map the `-shm`, which is what a sibling close then
            // has to observe.
            let keeper = if role == Role::Keeper {
                let k = lib.open(&db_path).expect("keeper foreign connection");
                let (rc, msg) = k.exec("SELECT count(*) FROM esc073_probe");
                if rc != 0 {
                    tripped(format!(
                        "the foreign instance's FIRST read could not see the table the engine \
                         had already committed: rc={rc} {msg:?}"
                    ));
                }
                eprintln!(
                    "[child] keeper attached; shm={} wal={}",
                    file_len(&shm),
                    file_len(&wal)
                );
                Some(k)
            } else {
                None
            };

            for i in 0..FOREIGN_CYCLES {
                let conn = match lib.open(&db_path) {
                    Ok(c) => c,
                    Err(rc) => {
                        refusals += 1;
                        eprintln!("[child] foreign open refused rc={rc} (acceptable)");
                        continue;
                    }
                };
                let _ = conn.exec("PRAGMA busy_timeout = 5000");
                let (rc, msg) = conn.exec(&format!(
                    "UPDATE esc073_probe SET v = 'foreign-{i}' WHERE id = 1"
                ));
                if rc != 0 {
                    refusals += 1;
                    eprintln!("[child] foreign write refused rc={rc} msg={msg:?} (acceptable)");
                }
                let before = (file_len(&shm), file_len(&wal));
                drop(conn); // ← sqlite3_close: the checkpoint/truncate under test.
                let after = (file_len(&shm), file_len(&wal));
                if before != after {
                    eprintln!(
                        "[child] cycle {i}: foreign close changed (shm,wal) {before:?} -> {after:?}"
                    );
                }
            }
            drop(keeper);
        }
        Role::StaleRead => {
            // Strictly sequential, no concurrency, no sleep, no retry:
            //   engine commits `engine-i`
            //   → foreign opens and runs a guarded UPDATE whose row count IS
            //     its read oracle (one row changed ⟺ it observed `engine-i`)
            //   → foreign commits `foreign-i` and closes
            //   → engine reads back and must observe `foreign-i`.
            // Values are pairwise distinct across rounds, so neither side can
            // coincidentally match while reading a stale image.
            for i in 0..STALE_ROUNDS {
                let engine_value = format!("engine-{i}");
                let foreign_value = format!("foreign-{i}");
                if let Err(err) = engine_write(&rt, &catalog, &engine_value) {
                    tripped_as(
                        classify_engine_error(&err),
                        format!("round {i}: engine write failed: {err}"),
                    );
                }

                let conn = match lib.open(&db_path) {
                    Ok(c) => c,
                    Err(rc) => tripped(format!("round {i}: foreign open failed rc={rc}")),
                };
                let _ = conn.exec("PRAGMA busy_timeout = 5000");
                let (rc, msg) = conn.exec(&format!(
                    "UPDATE esc073_probe SET v = '{foreign_value}' \
                     WHERE id = 1 AND v = '{engine_value}'"
                ));
                if rc != 0 {
                    tripped_as(
                        classify_engine_error(&format!("(code: {rc}) {msg}")),
                        format!("round {i}: foreign guarded write failed rc={rc} msg={msg:?}"),
                    );
                }
                let changed = conn.changes();
                if changed != 1 {
                    tripped_as(
                        EXIT_STALE,
                        format!(
                            "round {i}: the FOREIGN library instance did not observe the engine's \
                             committed value {engine_value:?} (guarded UPDATE changed {changed} \
                             row(s)) — it read a stale database image"
                        ),
                    );
                }
                drop(conn); // close: checkpoint/truncate window.

                match engine_read(&rt, &catalog) {
                    Ok(Some(v)) if v == foreign_value => {}
                    Ok(Some(v)) => tripped_as(
                        EXIT_STALE,
                        format!(
                            "round {i}: the ENGINE observed {v:?} after the foreign instance \
                             committed {foreign_value:?} — a stale read"
                        ),
                    ),
                    Ok(None) => tripped_as(
                        EXIT_STALE,
                        format!("round {i}: the engine's probe row vanished"),
                    ),
                    Err(err) => tripped_as(
                        classify_engine_error(&err),
                        format!("round {i}: engine read failed: {err}"),
                    ),
                }
            }
        }
        Role::EngineChurn => {
            // Reverse direction: the foreign connection stays open and keeps
            // committing while ENGINE pools open and close underneath it.
            let conn = lib.open(&db_path).expect("foreign connection");
            let _ = conn.exec("PRAGMA busy_timeout = 5000");
            for i in 0..FOREIGN_CYCLES {
                let churned = match try_open_engine(&rt, dir.path()) {
                    Ok(c) => c,
                    Err(err) => {
                        // The foreign instance has been writing this file with
                        // a wal-index this one cannot see, so the engine may
                        // find an image it cannot use. Typed, reported, and
                        // not fatal — see this arm's oracle.
                        refusals += 1;
                        eprintln!("[child] churn {i}: engine re-open refused (acceptable): {err}");
                        continue;
                    }
                };
                let (rc, msg) = conn.exec(&format!(
                    "UPDATE esc073_probe SET v = 'foreign-churn-{i}' WHERE id = 1"
                ));
                if rc != 0 {
                    refusals += 1;
                    eprintln!("[child] foreign write refused rc={rc} msg={msg:?} (acceptable)");
                }
                let before = (file_len(&shm), file_len(&wal));
                drop(churned); // ← engine-side close/checkpoint.
                let after = (file_len(&shm), file_len(&wal));
                if before != after {
                    eprintln!(
                        "[child] churn {i}: engine close changed (shm,wal) {before:?} -> {after:?}"
                    );
                }
                let (rc, msg) = conn.exec("SELECT count(*) FROM esc073_probe");
                if rc != 0 {
                    refusals += 1;
                    eprintln!("[child] foreign read refused rc={rc} msg={msg:?} (acceptable)");
                }
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    for h in load {
        let _ = rt.block_on(h);
    }
    eprintln!(
        "[child role={}] survived: {refusals} refusal(s), {} engine error(s)",
        role.as_str(),
        engine_errors.load(Ordering::Relaxed)
    );
    std::process::exit(0);
}

// ── Parent ──────────────────────────────────────────────────────────────────

/// Outcome of one child run.
enum Attempt {
    Survived,
    Skipped,
    /// Process-fatal signal — the row's headline failure.
    Signal(i32),
    /// The arm's own observation tripped with a TYPED error. Not a signal.
    Tripped,
    /// One instance read a stale-but-well-formed image. The documented
    /// residual of the two-library-instances topology.
    Stale,
    /// The database file came back malformed. The pre-fix class the seam is
    /// required to have removed.
    Corrupt,
    ExitCode(i32),
    Hung,
}

/// What a whole arm observed across its attempts.
struct Summary {
    signals: Vec<i32>,
    tripped: usize,
    stale: usize,
    corrupt: usize,
    survived: usize,
    log: String,
}

impl Summary {
    /// Every attempt completed with no failure of any class.
    fn is_clean(&self) -> bool {
        self.signals.is_empty() && self.tripped == 0 && self.stale == 0 && self.corrupt == 0
    }

    /// The contract this row actually carries for an out-of-contract topology:
    /// no process-fatal signal, and — post-seam — no damaged database file
    /// either. A typed error or a stale read is the acceptable residual and is
    /// recorded, not failed.
    fn is_signal_free_and_uncorrupted(&self) -> bool {
        self.signals.is_empty() && self.corrupt == 0
    }

    /// One-line census of what the arm observed, for the record.
    fn census(&self) -> String {
        format!(
            "{} survived, {} typed-error, {} stale-read, {} corrupt-file, signals {:?}",
            self.survived, self.tripped, self.stale, self.corrupt, self.signals
        )
    }
}

/// Re-execute this test binary as a child in `role`, running only `test_name`.
fn run_child(role: Role, test_name: &str) -> (Attempt, String) {
    let exe = std::env::current_exe().expect("current test binary");
    let mut child = Command::new(exe)
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, role.as_str())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn child");

    let deadline = std::time::Instant::now() + CHILD_CEILING;
    loop {
        match child.try_wait().expect("try_wait") {
            Some(_) => break,
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return (Attempt::Hung, String::new());
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let out = child.wait_with_output().expect("child output");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let attempt = match (out.status.signal(), out.status.code()) {
        (Some(sig), _) => Attempt::Signal(sig),
        (None, Some(0)) => Attempt::Survived,
        (None, Some(EXIT_NO_FOREIGN_LIB)) => Attempt::Skipped,
        (None, Some(EXIT_TRIPPED)) => Attempt::Tripped,
        (None, Some(EXIT_STALE)) => Attempt::Stale,
        (None, Some(EXIT_CORRUPT)) => Attempt::Corrupt,
        (None, Some(code)) => Attempt::ExitCode(code),
        (None, None) => Attempt::ExitCode(-1),
    };
    (attempt, log)
}

/// Drive `role` for up to [`ATTEMPTS`] child runs, stopping at the first
/// failure of any class.
fn drive(role: Role, test_name: &str) -> Summary {
    drive_with(role, test_name, true)
}

/// Drive `role` for all [`ATTEMPTS`] runs regardless of outcome, so the census
/// records a rate rather than a first hit. Used by the deterministic
/// [`Role::StaleRead`] arm, whose value is the distribution.
fn drive_all(role: Role, test_name: &str) -> Summary {
    drive_with(role, test_name, false)
}

fn drive_with(role: Role, test_name: &str, stop_at_first_failure: bool) -> Summary {
    let mut summary = Summary {
        signals: Vec::new(),
        tripped: 0,
        stale: 0,
        corrupt: 0,
        survived: 0,
        log: String::new(),
    };
    for attempt in 1..=ATTEMPTS {
        let (outcome, log) = run_child(role, test_name);
        summary.log = log;
        let failed = match outcome {
            Attempt::Survived => {
                summary.survived += 1;
                false
            }
            Attempt::Skipped => {
                // Fail closed under CI: a harness that silently evaporates on
                // the machine that gates merges proves nothing. Locally it is
                // a loud skip.
                assert!(
                    std::env::var_os("CI").is_none(),
                    "esc-073 [{}]: no platform libsqlite3 at any of {FOREIGN_LIB_CANDIDATES:?}, \
                     so this harness is VACUOUS — and `CI` is set, where a vacuous escape oracle \
                     is a failure. Install a platform libsqlite3 in the CI image or add its path \
                     to FOREIGN_LIB_CANDIDATES.\n{}",
                    role.as_str(),
                    summary.log
                );
                eprintln!(
                    "esc-073 [{}]: SKIPPED — no platform libsqlite3 to collide with; this \
                     harness is vacuous on this machine:\n{}",
                    role.as_str(),
                    summary.log
                );
                return summary;
            }
            Attempt::Signal(sig) => {
                eprintln!(
                    "esc-073 [{}]: attempt {attempt}/{ATTEMPTS} died with SIGNAL {sig}",
                    role.as_str()
                );
                summary.signals.push(sig);
                true
            }
            Attempt::Tripped => {
                eprintln!(
                    "esc-073 [{}]: attempt {attempt}/{ATTEMPTS} tripped with a TYPED error",
                    role.as_str()
                );
                summary.tripped += 1;
                true
            }
            Attempt::Stale => {
                eprintln!(
                    "esc-073 [{}]: attempt {attempt}/{ATTEMPTS} observed a STALE READ",
                    role.as_str()
                );
                summary.stale += 1;
                true
            }
            Attempt::Corrupt => {
                eprintln!(
                    "esc-073 [{}]: attempt {attempt}/{ATTEMPTS} found the database file MALFORMED",
                    role.as_str()
                );
                summary.corrupt += 1;
                true
            }
            Attempt::ExitCode(code) => panic!(
                "esc-073 [{}]: attempt {attempt}/{ATTEMPTS} exited {code} (harness fault, not a \
                 crash):\n{}",
                role.as_str(),
                summary.log
            ),
            Attempt::Hung => panic!(
                "esc-073 [{}]: attempt {attempt}/{ATTEMPTS} hung past {CHILD_CEILING:?}",
                role.as_str()
            ),
        };
        if failed && stop_at_first_failure {
            break;
        }
    }
    summary
}

/// **The oracle.** A foreign SQLite library instance performing the
/// `_set_metrics` cycle (open → `UPDATE` → commit → close) against a catalog
/// file an engine pool is live on must never take the process down. An
/// `SQLITE_BUSY` / typed error on either side is acceptable and is reported; a
/// fatal signal is not.
#[test]
fn foreign_library_write_cycle_never_kills_the_process() {
    let name =
        "esc_073_foreign_sqlite_library::foreign_library_write_cycle_never_kills_the_process";
    if let Some(role) = std::env::var(ROLE_ENV).ok().and_then(|r| Role::parse(&r)) {
        child_main(role);
    }
    let s = drive(Role::Collide, name);
    assert!(
        s.is_clean(),
        "esc-073: a foreign SQLite library instance's open/write/close cycle broke the process \
         ({}) — an out-of-contract topology must refuse, never die by signal. Child log:\n{}",
        s.census(),
        s.log
    );
}

/// **Mechanism differential.** Identical to the oracle except one foreign
/// connection stays open for the whole run, so the foreign library instance
/// never treats a closing connection as the last one and never runs the
/// close-time checkpoint + `-wal`/`-shm` truncate.
///
/// Read this arm against the oracle: oracle RED + this arm GREEN confirms the
/// close-time truncate-under-a-live-mapping hypothesis; both RED refutes it and
/// points at plain concurrent-WAL-write incompatibility instead.
#[test]
fn keeper_connection_suppresses_the_close_time_checkpoint() {
    let name =
        "esc_073_foreign_sqlite_library::keeper_connection_suppresses_the_close_time_checkpoint";
    if let Some(role) = std::env::var(ROLE_ENV).ok().and_then(|r| Role::parse(&r)) {
        child_main(role);
    }
    let s = drive(Role::Keeper, name);
    assert!(
        s.is_clean(),
        "esc-073 keeper arm: {}. Child log:\n{}",
        s.census(),
        s.log
    );
}

/// **Reverse direction.** The foreign connection stays open and keeps
/// committing while ENGINE pools open and close underneath it, so the
/// engine-side close/checkpoint is the candidate truncator. Same control: a
/// signal is a failure, a typed refusal is not.
///
/// This is the harshest arm for the residual: the foreign instance holds the
/// file open across every engine re-open, so its WAL writes and the engine's
/// heap wal-index are guaranteed to diverge, and an engine re-open can
/// legitimately find an image it refuses (`SQLITE_CORRUPT`). That refusal is
/// counted and printed by the child. What is asserted here is only what an
/// out-of-contract topology is entitled to: no process-fatal signal, and no
/// hang.
#[test]
fn engine_pool_churn_under_a_live_foreign_connection_never_kills_the_process() {
    let name = "esc_073_foreign_sqlite_library::\
                engine_pool_churn_under_a_live_foreign_connection_never_kills_the_process";
    if let Some(role) = std::env::var(ROLE_ENV).ok().and_then(|r| Role::parse(&r)) {
        child_main(role);
    }
    let s = drive(Role::EngineChurn, name);
    assert!(
        s.is_clean(),
        "esc-073 reverse direction: an engine pool closing under a live foreign connection \
         broke the process ({}). Child log:\n{}",
        s.census(),
        s.log
    );
}

/// **esc-071's symptom, in esc-073's topology.** Strictly sequential — no
/// background load, no concurrency, no sleep, no retry: engine commits, the
/// foreign library instance performs one `_set_metrics` cycle, and each side
/// looks for what the other just committed.
///
/// This is the arm that connects the two rows: a lone foreign
/// open/write/close makes a committed value invisible with no race in sight,
/// so the read-visibility lag the python wave attributed to engine pooling is
/// a property of the two-library-instances-one-file topology, not of the pool.
///
/// # What this arm asserts, and what it deliberately does not
///
/// Mutual visibility between two SQLite *library instances* on one file is not
/// achievable from the engine side and is not asserted here. Their `fcntl`
/// locks are the same process's locks, so neither instance can see the other's
/// WAL-index state; `jammi_db::catalog::backend_sqlite`'s module docs record
/// that residual. What IS asserted is the pair of classes the fix removed:
///
/// * **no process-fatal signal** — the row's headline failure (`SIGBUS` from a
///   truncated, mmapped `-shm`), and
/// * **no malformed database file** — the pre-fix `SQLITE_CORRUPT` this arm
///   returned 17 times in 20 on the default VFS.
///
/// A stale read or a typed error is the acceptable residual; the census
/// records which, so a regression from "stale" back to "corrupt" is visible in
/// the assertion message rather than inferred. All [`ATTEMPTS`] runs are
/// driven (not stopped at the first hit) so the census is a rate.
#[test]
fn the_stale_read_topology_neither_signals_nor_corrupts_the_file() {
    let name = "esc_073_foreign_sqlite_library::\
                the_stale_read_topology_neither_signals_nor_corrupts_the_file";
    if let Some(role) = std::env::var(ROLE_ENV).ok().and_then(|r| Role::parse(&r)) {
        child_main(role);
    }
    let s = drive_all(Role::StaleRead, name);
    assert!(
        s.is_signal_free_and_uncorrupted(),
        "esc-073 stale-read arm: {} over {ATTEMPTS} attempt(s). A signal, or a malformed \
         database file, is a failure; a stale read or a typed error is the documented residual. \
         Child log:\n{}",
        s.census(),
        s.log
    );
    eprintln!(
        "esc-073 stale-read arm census over {ATTEMPTS} attempt(s): {}",
        s.census()
    );
}
