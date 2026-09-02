//! esc-073 FIX oracle (`closes_escape: esc-073`) — the SQLite catalog's
//! single-process contract is enforced by a mechanism, not by prose.
//!
//! `docs/guide/src/catalog-and-broker.md`'s SQLite row states the contract:
//! one process per catalog file; sharing it corrupts the WAL. Two facts follow
//! that the RED harness (`esc_073_foreign_sqlite_library.rs`) proved were not
//! true of the pre-fix engine:
//!
//! * a second *process* could open the file and race the engine into a
//!   corrupt WAL, and
//! * a second SQLite *library instance* in the same process could truncate the
//!   `-shm` file the engine had mmapped, taking the process down with `SIGBUS`
//!   — a process-fatal signal for out-of-contract input.
//!
//! The fix opens the pool through SQLite's `unix-excl` VFS
//! (`jammi_db::catalog::backend_sqlite`, module docs). This file proves the
//! properties that seam is bought for, each with a control that fails when the
//! seam is off:
//!
//! 1. [`wal_index_is_heap_resident_and_wal_mode_is_still_engaged`] — no `-shm`
//!    file exists while a live pool has written, and WAL is still the journal
//!    mode. Its control child re-runs the same probe with the seam disabled
//!    (`JAMMI_SQLITE_VFS=default`) and requires the `-shm` to be present, so a
//!    green probe can never mean "the file was missing for some other reason".
//! 2. [`a_second_process_is_refused_with_a_typed_error`] and
//!    [`the_first_process_to_open_wins_whichever_it_is`] — a second process
//!    opening the same catalog is refused with a typed
//!    [`BackendError::Unavailable`] naming the contract, bounded by the 5 s
//!    busy timeout, in both orderings. Never a hang, never a signal.
//! 3. [`closing_the_catalog_releases_the_file_and_its_sidecars`] — the seam
//!    would be a trap without an awaitable release point, because a `drop` of
//!    a `Catalog` releases nothing at any bounded moment (`sqlx` closes
//!    returned connections from a background task). `Catalog::close().await`
//!    is that point: after it a successor process opens the directory
//!    immediately and reads what this one committed, repeated. It also
//!    settles what a consumer polling for sidecar absence as a "released"
//!    barrier can rely on: `catalog.db-shm` never exists under the seam, and
//!    `catalog.db-wal` is best-effort — SQLite removes it only when the last
//!    close's PASSIVE checkpoint completes, so it can outlive a fully
//!    released file. `close().await` is the barrier; the files are not.
//! 4. [`closing_one_of_two_live_pools_is_bounded_and_leaves_the_other_working`]
//!    — the same point from the other direction. The seam is single-*process*,
//!    so two pools on one file inside this process are supported, and the
//!    `-wal` the settle wait watches belongs to the FILE: the survivor keeps it
//!    alive, so the first `close()` runs its settle loop to the ceiling and
//!    warns. That is a bounded cost and a log line, not a correctness loss —
//!    the close returns, and the survivor reads and writes across it.
//!
//! ## Re-demonstrating the pre-fix RED
//!
//! Every arm of `esc_073_foreign_sqlite_library.rs` and this file's mechanism
//! probe are restored to their pre-fix behaviour by setting
//! `JAMMI_SQLITE_VFS=default`, which puts the pool back on the platform
//! default VFS without touching the source:
//!
//! ```text
//! JAMMI_SQLITE_VFS=default cargo test -p jammi-db --test it esc_073 -- --nocapture
//! ```
//!
//! [`BackendError::Unavailable`]: jammi_db::catalog::backend::BackendError::Unavailable

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use jammi_db::catalog::backend::{BackendError, SqlValue, TxOptions};
use jammi_db::catalog::backend_sqlite::{catalog_vfs, SQLITE_VFS_ENV};
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_test_utils::child::{render, Capture, DrainedChild, Epoch, DEFAULT_HEAD_CAP};

/// Env var carrying a child's role. Absent ⇒ this process is the parent.
const ROLE_ENV: &str = "JAMMI_SEAM_ROLE";
/// Env var carrying the catalog directory a child must operate on.
const DIR_ENV: &str = "JAMMI_SEAM_DIR";

/// Child exit code: the open was refused with the contracted typed error.
const EXIT_REFUSED_AS_CONTRACTED: i32 = 70;
/// Child exit code: the open SUCCEEDED while another process held the file —
/// the contract is not enforced.
const EXIT_OPENED_ANYWAY: i32 = 71;
/// Child exit code: the open failed, but not with the contracted error. A
/// refusal whose shape the operator cannot act on is its own failure.
const EXIT_WRONG_ERROR: i32 = 72;
/// Child exit code: the control probe did not observe what the disabled seam
/// must produce.
const EXIT_CONTROL_FAILED: i32 = 73;
/// Child exit code: the open SUCCEEDED and the seeded row was readable — the
/// success half of the handoff.
const EXIT_OPENED_AND_READ: i32 = 74;
/// Child exit code: the open succeeded but the seeded row was not there.
const EXIT_OPENED_BUT_EMPTY: i32 = 75;

/// Ceiling on any child. Generously above the 5 s busy timeout: anything near
/// it is a hang, which fails with the same weight as a crash.
const CHILD_CEILING: Duration = Duration::from_secs(60);

/// Ceiling on one *refused* open. The busy timeout is 5 s; a debug-profile
/// process start and migration pass fit comfortably under this.
const REFUSAL_CEILING: Duration = Duration::from_secs(30);

/// Close/handoff cycles the release handshake is repeated over. Each spawns a
/// successor process, so the count trades coverage against wall time: the
/// hazard is a race inside the pool's connection return, and at the ~1-in-7
/// rate measured before the release was made an awaited event, this many
/// cycles catch a regression with probability ~0.7 per run — and the module
/// runs on every suite invocation.
const RELEASE_ITERATIONS: usize = 8;

/// Substring every cross-process refusal must carry, so the operator reads the
/// contract and the remedy off the error rather than off a `(code: 5)`.
const CONTRACT_PHRASE: &str = "single-process only";

/// Mirror of `backend_sqlite::CLOSE_SIDECAR_CEILING` (private to the crate):
/// the bound on `close()`'s post-drain wait for `-wal` disappearance.
const CLOSE_SIDECAR_CEILING: Duration = Duration::from_secs(2);

/// What a close that cannot see its evidence is allowed to cost: the settle
/// ceiling plus slack for the pool drain and a loaded CI box. The point of the
/// assertion is BOUNDEDNESS — that a second live pool makes the first close
/// slow, never hung — so the slack is deliberately generous; a regression that
/// turned the wait unbounded would blow past this by orders of magnitude.
const CLOSE_CEILING_WITH_SLACK: Duration = Duration::from_secs(20);

// ── Shared probe bodies ─────────────────────────────────────────────────────

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime")
}

/// Open a catalog on `dir` and commit one row, so the pool has certainly taken
/// the database lock and certainly written WAL frames.
fn open_and_write(rt: &tokio::runtime::Runtime, dir: &Path) -> Catalog {
    let catalog = rt.block_on(Catalog::open(dir)).expect("open catalog");
    let backend = catalog.backend_arc();
    rt.block_on(backend.transaction(TxOptions::default(), |tx| {
        Box::pin(async move {
            tx.execute(
                "CREATE TABLE IF NOT EXISTS seam_probe (id INTEGER PRIMARY KEY, v TEXT)",
                &[],
            )
            .await?;
            tx.execute(
                "INSERT OR REPLACE INTO seam_probe (id, v) VALUES (1, $1)",
                &[SqlValue::Text("written")],
            )
            .await?;
            Ok(())
        })
    }))
    .expect("probe write");
    catalog
}

/// The journal mode SQLite reports for this pool's connection.
fn journal_mode(rt: &tokio::runtime::Runtime, catalog: &Catalog) -> String {
    let backend = catalog.backend_arc();
    rt.block_on(backend.transaction(
        TxOptions {
            read_only: true,
            ..Default::default()
        },
        |tx| {
            Box::pin(async move {
                tx.query_opt(
                    "SELECT journal_mode AS journal_mode FROM pragma_journal_mode()",
                    &[],
                    |r| r.get::<String>("journal_mode"),
                )
                .await
            })
        },
    ))
    .expect("read journal_mode")
    .expect("journal_mode row")
    .to_ascii_lowercase()
}

/// Sidecar-file census taken while `catalog` is alive: `(shm exists, wal exists)`.
fn sidecars(dir: &Path) -> (bool, bool) {
    (
        dir.join("catalog.db-shm").exists(),
        dir.join("catalog.db-wal").exists(),
    )
}

// ── Children ────────────────────────────────────────────────────────────────

/// Render both of a [`Capture`]'s streams (stdout then stderr) for a
/// panic/failure message.
fn rendered_log(cap: &Capture) -> String {
    format!(
        "{}{}",
        render(&cap.stdout, cap.stdout_truncated, DEFAULT_HEAD_CAP),
        render(&cap.stderr, cap.stderr_truncated, DEFAULT_HEAD_CAP),
    )
}

/// Re-execute this binary running only `test_name`, in `role`, with `envs`
/// applied, via [`DrainedChild`] (both streams drained on background threads
/// while the child runs, so evidence survives both a chatty child and a
/// killed one). Returns `(exit code or None-if-signalled, signal, elapsed,
/// log)`.
///
/// `elapsed` is [`Capture::elapsed`] — measured POST-SETTLE (after the reader
/// threads observe EOF), not at exit DETECTION the way this harness's
/// pre-drain `started.elapsed()` was. That is up to ~1 s later, which is safe
/// against every ceiling this value is checked against here
/// ([`REFUSAL_CEILING`], 30 s): the settle bound is a small fraction of that
/// margin.
fn run_child(
    role: &str,
    test_name: &str,
    envs: &[(&str, String)],
) -> (Option<i32>, Option<i32>, Duration, String) {
    let exe = std::env::current_exe().expect("current test binary");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, role);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let child = DrainedChild::spawn(&mut cmd).expect("spawn child");
    let cap = child.wait_bounded(CHILD_CEILING, Epoch::Spawn);
    if cap.hung {
        panic!(
            "esc-073 seam: child role={role} hung past {CHILD_CEILING:?} — a refusal must be \
             bounded by the busy timeout, never a hang. Both streams:\n{}",
            rendered_log(&cap)
        );
    }
    let code = cap.status.and_then(|s| s.code());
    let signal = cap.status.and_then(|s| s.signal());
    let elapsed = cap.elapsed;
    let log = rendered_log(&cap);
    (code, signal, elapsed, log)
}

/// Report a child observation and exit with `code`. Never a panic: a panic's
/// exit code is indistinguishable from a harness fault.
fn child_exit(code: i32, msg: String) -> ! {
    eprintln!("[child] {msg}");
    std::process::exit(code);
}

/// Control child for the mechanism probe: the seam is OFF for this process
/// (`JAMMI_SQLITE_VFS=default`), so the very same probe must observe a `-shm`
/// file. If it does not, the probe is not measuring what it claims and the
/// GREEN in the parent is vacuous.
fn child_shm_control() -> ! {
    let rt = runtime();
    let dir = tempfile::tempdir().expect("child tempdir");
    if catalog_vfs().is_some() {
        child_exit(
            EXIT_CONTROL_FAILED,
            format!(
                "control child expected the platform default VFS, got {:?}",
                catalog_vfs()
            ),
        );
    }
    let catalog = open_and_write(&rt, dir.path());
    let (shm, wal) = sidecars(dir.path());
    let mode = journal_mode(&rt, &catalog);
    eprintln!("[child control] shm={shm} wal={wal} journal_mode={mode}");
    if mode != "wal" {
        child_exit(
            EXIT_CONTROL_FAILED,
            format!("control child is not in WAL mode ({mode}); the probe proves nothing"),
        );
    }
    if !shm {
        child_exit(
            EXIT_CONTROL_FAILED,
            "with the seam DISABLED a live WAL pool must map a `-shm` file, and none exists — \
             the `-shm` census cannot discriminate the seam"
                .to_string(),
        );
    }
    drop(catalog);
    std::process::exit(0);
}

/// Child that opens the catalog at [`DIR_ENV`] while the parent holds it, and
/// reports which shape the refusal took.
fn child_open_expect_refusal() -> ! {
    let rt = runtime();
    let dir = PathBuf::from(std::env::var(DIR_ENV).expect("dir env"));
    let started = Instant::now();
    match rt.block_on(Catalog::open(&dir)) {
        Ok(_) => child_exit(
            EXIT_OPENED_ANYWAY,
            format!(
                "opened {} while another process holds it, after {:?} — the single-process \
                 contract is NOT enforced",
                dir.display(),
                started.elapsed()
            ),
        ),
        Err(JammiError::BackendDriver(BackendError::Unavailable(msg)))
            if msg.contains(CONTRACT_PHRASE) =>
        {
            child_exit(
                EXIT_REFUSED_AS_CONTRACTED,
                format!("refused after {:?} with: {msg}", started.elapsed()),
            )
        }
        Err(other) => child_exit(
            EXIT_WRONG_ERROR,
            format!(
                "refused after {:?}, but not with the contracted \
                 BackendError::Unavailable/{CONTRACT_PHRASE}: {other}",
                started.elapsed()
            ),
        ),
    }
}

/// Child that opens the catalog at [`DIR_ENV`] FIRST and holds it until the
/// parent drops a `stop` file — the reverse ordering.
fn child_hold() -> ! {
    let rt = runtime();
    let dir = PathBuf::from(std::env::var(DIR_ENV).expect("dir env"));
    let catalog = open_and_write(&rt, &dir);
    std::fs::write(dir.join("ready"), b"1").expect("ready flag");
    eprintln!("[child hold] holding {}", dir.display());
    let deadline = Instant::now() + CHILD_CEILING;
    while !dir.join("stop").exists() {
        if Instant::now() >= deadline {
            child_exit(
                EXIT_CONTROL_FAILED,
                "holder timed out waiting for the parent's stop flag".to_string(),
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(catalog);
    std::process::exit(0);
}

/// Child that opens the catalog at [`DIR_ENV`] after the parent has released
/// it, and reads back the row the parent seeded — the seed-before-spawn
/// handoff, and the inverse of [`child_open_expect_refusal`].
fn child_open_expect_success() -> ! {
    let rt = runtime();
    let dir = PathBuf::from(std::env::var(DIR_ENV).expect("dir env"));
    let started = Instant::now();
    let catalog = match rt.block_on(Catalog::open(&dir)) {
        Ok(c) => c,
        Err(err) => child_exit(
            EXIT_WRONG_ERROR,
            format!(
                "could not open {} after the seeding process closed it, in {:?}: {err}",
                dir.display(),
                started.elapsed()
            ),
        ),
    };
    let backend = catalog.backend_arc();
    let seen = rt
        .block_on(backend.transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt("SELECT v FROM seam_probe WHERE id = 1", &[], |r| {
                        r.get::<String>("v")
                    })
                    .await
                })
            },
        ))
        .expect("read seeded row");
    if seen.as_deref() != Some("written") {
        child_exit(
            EXIT_OPENED_BUT_EMPTY,
            format!("opened the handed-off catalog but read {seen:?}, not Some(\"written\")"),
        );
    }
    child_exit(
        EXIT_OPENED_AND_READ,
        format!(
            "opened the handed-off catalog in {:?} and read the seeded row",
            started.elapsed()
        ),
    );
}

/// Child that writes a catalog at [`DIR_ENV`] through the PLATFORM DEFAULT
/// VFS (the pre-seam engine) and then abandons it: `std::process::exit` runs
/// no destructors, so the pool never closes, SQLite never checkpoints, and the
/// `-wal` + `-shm` it created are left on disk exactly as a killed pre-seam
/// process would leave them.
fn child_seed_default_vfs_and_abandon() -> ! {
    let rt = runtime();
    let dir = PathBuf::from(std::env::var(DIR_ENV).expect("dir env"));
    if catalog_vfs().is_some() {
        child_exit(
            EXIT_CONTROL_FAILED,
            format!(
                "seeding child expected the platform default VFS, got {:?}",
                catalog_vfs()
            ),
        );
    }
    let catalog = open_and_write(&rt, &dir);
    let (shm, wal) = sidecars(&dir);
    eprintln!("[child seed-default] shm={shm} wal={wal}; abandoning without close");
    if !(shm && wal) {
        child_exit(
            EXIT_CONTROL_FAILED,
            format!("default VFS did not produce both sidecars: shm={shm} wal={wal}"),
        );
    }
    std::mem::forget(catalog);
    std::mem::forget(rt);
    std::process::exit(0);
}

/// Dispatch to the child body for `role`, if this process is a child.
fn dispatch_child() {
    let Ok(role) = std::env::var(ROLE_ENV) else {
        return;
    };
    match role.as_str() {
        "shm-control" => child_shm_control(),
        "open-expect-refusal" => child_open_expect_refusal(),
        "open-expect-success" => child_open_expect_success(),
        "seed-default-vfs-and-abandon" => child_seed_default_vfs_and_abandon(),
        "hold" => child_hold(),
        other => panic!("unknown child role {other:?}"),
    }
}

// ── Oracles ─────────────────────────────────────────────────────────────────

/// **Mechanism.** While a live pool has written, the WAL index is heap
/// resident: no `catalog.db-shm` file exists for anyone to truncate — which is
/// what makes an engine-side `SIGBUS` unreachable, since the `-shm` was the
/// only file the engine mmapped (`SQLITE_DEFAULT_MMAP_SIZE == 0` in the
/// bundled amalgamation). WAL itself is unchanged: the journal mode is still
/// `wal` and the `-wal` file is still there.
///
/// Deterministic in both directions. The control child runs the identical
/// probe with the seam disabled and is *required* to find a `-shm`, so this
/// test fails if the seam regresses AND fails if the census stops
/// discriminating.
#[test]
fn wal_index_is_heap_resident_and_wal_mode_is_still_engaged() {
    dispatch_child();
    let name =
        "sqlite_single_process_seam::wal_index_is_heap_resident_and_wal_mode_is_still_engaged";

    assert_eq!(
        catalog_vfs().as_deref(),
        Some("unix-excl"),
        "the catalog pool must open through the process-exclusive VFS on unix"
    );

    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let catalog = open_and_write(&rt, dir.path());
    let (shm, wal) = sidecars(dir.path());
    let mode = journal_mode(&rt, &catalog);

    assert_eq!(
        mode, "wal",
        "the seam must not silently drop WAL: journal_mode is {mode:?}"
    );
    assert!(
        wal,
        "WAL mode is reported but no `catalog.db-wal` exists — the probe is not observing a live \
         WAL database"
    );
    assert!(
        !shm,
        "`{}` exists while the pool is live: the wal-index is file-backed and mmapped, so a \
         foreign SQLite library instance can still truncate it under the engine (esc-073 SIGBUS)",
        dir.path().join("catalog.db-shm").display()
    );
    drop(catalog);

    // Control: the same census with the seam OFF must find the `-shm`.
    let (code, signal, elapsed, log) = run_child(
        "shm-control",
        name,
        &[(SQLITE_VFS_ENV, "default".to_string())],
    );
    assert_eq!(
        signal, None,
        "control child died with a signal: {signal:?}\n{log}"
    );
    assert_eq!(
        code,
        Some(0),
        "control child (seam disabled) exited {code:?} after {elapsed:?} — the `-shm` census does \
         not discriminate the seam, so the GREEN above is vacuous:\n{log}"
    );
}

/// **Cross-process refusal, child second.** The parent holds a live pool; a
/// second process opening the same catalog directory is refused with a typed
/// [`BackendError::Unavailable`] naming the single-process contract, inside
/// the busy timeout. Never a hang, never a signal, never a success.
///
/// [`BackendError::Unavailable`]: jammi_db::catalog::backend::BackendError::Unavailable
#[test]
fn a_second_process_is_refused_with_a_typed_error() {
    dispatch_child();
    let name = "sqlite_single_process_seam::a_second_process_is_refused_with_a_typed_error";

    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let held = open_and_write(&rt, dir.path());

    let (code, signal, elapsed, log) = run_child(
        "open-expect-refusal",
        name,
        &[(DIR_ENV, dir.path().display().to_string())],
    );
    assert_eq!(
        signal, None,
        "the second process died with a signal:\n{log}"
    );
    assert_eq!(
        code,
        Some(EXIT_REFUSED_AS_CONTRACTED),
        "a second process opening a held catalog exited {code:?} (expected \
         {EXIT_REFUSED_AS_CONTRACTED}: refused with BackendError::Unavailable naming \
         {CONTRACT_PHRASE:?}); {EXIT_OPENED_ANYWAY} means it opened anyway, {EXIT_WRONG_ERROR} \
         means the refusal was untyped:\n{log}"
    );
    assert!(
        elapsed < REFUSAL_CEILING,
        "the refusal took {elapsed:?}, past the {REFUSAL_CEILING:?} ceiling — a busy-timeout \
         refusal must be bounded:\n{log}"
    );
    eprintln!("[esc-073 seam] second process refused in {elapsed:?}:\n{log}");

    // The holder is still usable afterwards: refusing the intruder must not
    // have cost the incumbent its own catalog.
    let mode = journal_mode(&rt, &held);
    assert_eq!(mode, "wal", "holder lost WAL after refusing an intruder");
    drop(held);
}

/// **Cross-process refusal, reverse ordering.** A child opens the catalog
/// first and holds it; the parent's later open is the one refused. The seam is
/// first-come-first-served and symmetric — it is not an artifact of who
/// created the file.
#[test]
fn the_first_process_to_open_wins_whichever_it_is() {
    dispatch_child();
    let name = "sqlite_single_process_seam::the_first_process_to_open_wins_whichever_it_is";

    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_path_buf();

    let exe = std::env::current_exe().expect("current test binary");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, "hold")
        .env(DIR_ENV, dir_path.display().to_string());
    let mut holder = DrainedChild::spawn(&mut cmd).expect("spawn holder");

    // Wait for the holder to actually own the file.
    let ready_by = Instant::now() + CHILD_CEILING;
    loop {
        if dir_path.join("ready").exists() {
            break;
        }
        if let Some(status) = holder.try_wait().expect("try_wait") {
            // The holder already exited: settle (it's already down, so this
            // returns promptly) and snapshot rather than reading via
            // `wait_with_output` directly, so the panic below carries the
            // SAME drained log shape every other diagnostic in this file
            // does.
            let cap = holder.wait_bounded(Duration::from_secs(1), Epoch::Call);
            panic!(
                "holder exited ({status:?}) before signalling ready. Both streams:\n{}",
                rendered_log(&cap)
            );
        }
        if Instant::now() >= ready_by {
            // `wait_bounded` checks its deadline BEFORE sleeping, so a zero
            // ceiling is well-defined: it kills the holder immediately,
            // settles, and snapshots — never a second, unbounded wait.
            let cap = holder.wait_bounded(Duration::ZERO, Epoch::Call);
            panic!(
                "holder never signalled ready within {CHILD_CEILING:?}. Both streams:\n{}",
                rendered_log(&cap)
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let started = Instant::now();
    let outcome = rt.block_on(Catalog::open(&dir_path));
    let elapsed = started.elapsed();

    // Release the holder before asserting, so a failed assertion cannot leave
    // an orphan process behind. Bounded from THIS call (`Epoch::Call`), not
    // from spawn: a legal refusal of up to `REFUSAL_CEILING` plus the
    // holder's own open could otherwise exhaust a from-spawn budget and
    // SIGKILL a healthy holder — this bounds only the post-`stop` drain.
    std::fs::write(dir_path.join("stop"), b"1").expect("stop flag");
    let holder_cap = holder.wait_bounded(CHILD_CEILING, Epoch::Call);
    // `hung` is checked BEFORE the signal/code asserts below: a SIGKILLed
    // holder must be reported as hung, not misread as "died with a signal".
    if holder_cap.hung {
        panic!(
            "esc-073 seam: the holder hung past {CHILD_CEILING:?} after the parent released it — \
             releasing a catalog must be bounded, never a hang. Both streams:\n{}",
            rendered_log(&holder_cap)
        );
    }
    let holder_log = rendered_log(&holder_cap);

    match outcome {
        Ok(_) => panic!(
            "this process opened a catalog another process already holds, after {elapsed:?} — the \
             single-process contract is NOT enforced in the child-first ordering. Holder log:\n\
             {holder_log}"
        ),
        Err(JammiError::BackendDriver(BackendError::Unavailable(msg))) => {
            assert!(
                msg.contains(CONTRACT_PHRASE),
                "refused, but the message does not name the contract ({CONTRACT_PHRASE:?}): {msg}"
            );
            eprintln!("[esc-073 seam] holder-first: this process refused in {elapsed:?}: {msg}");
        }
        Err(other) => panic!(
            "refused after {elapsed:?}, but not with the contracted \
             BackendError::Unavailable/{CONTRACT_PHRASE}: {other}. Holder log:\n{holder_log}"
        ),
    }
    assert!(
        elapsed < REFUSAL_CEILING,
        "the refusal took {elapsed:?}, past the {REFUSAL_CEILING:?} ceiling"
    );
    let holder_code = holder_cap.status.and_then(|s| s.code());
    let holder_signal = holder_cap.status.and_then(|s| s.signal());
    assert_eq!(
        holder_signal, None,
        "the holder died with a signal:\n{holder_log}"
    );
    assert_eq!(
        holder_code,
        Some(0),
        "the holder exited {holder_code:?} — holding a catalog while another process is refused \
         must be uneventful for the incumbent:\n{holder_log}"
    );
}

/// **The release handshake.** `Catalog::close().await` is the release point:
/// after it returns, this process's exclusive hold on the catalog file is
/// gone and a *second process* opens the same directory immediately.
///
/// This is the inverse of [`a_second_process_is_refused_with_a_typed_error`]
/// and the shape a seed-before-spawn handoff needs: under the `unix-excl` seam
/// the incumbent's lock is what refuses the newcomer, so "I am done with this
/// directory" has to be an awaitable event rather than a `drop`.
///
/// That refusal test is also the control that makes this one non-vacuous: the
/// *same* child role, run against a catalog that was NOT closed, is refused
/// with the typed error. Here, after the close, it must succeed — and must
/// read back the row the parent seeded, so a pass cannot mean "it opened a
/// fresh empty database somewhere else". It is repeated, because the hazard
/// is a race rather than a state: `sqlx` returns a dropped connection to the
/// pool from a spawned task, so a close that merely *usually* waits for the
/// last `sqlite3_close` passes once and leaks the lock on the cycle that
/// matters.
///
/// # Why the successor open, and not a sidecar census
///
/// A successor process is the only faithful probe. `catalog.db-shm` never
/// exists under the seam, and `catalog.db-wal` is *not* a reliable
/// released/held signal: SQLite deletes it in the last `sqlite3_close` only
/// when that close's PASSIVE checkpoint completes, and a checkpoint it
/// declines leaves the file behind on a perfectly clean close (observed once
/// in 40 runs of this module — the file was still there 2 s later, while the
/// lock had been released the whole time). The census is therefore reported
/// here, not asserted; a consumer polling for `-wal` absence as a "released"
/// barrier is polling for something that may never happen and should call
/// `close().await` instead.
#[test]
fn closing_the_catalog_releases_the_file_and_its_sidecars() {
    dispatch_child();
    let name = "sqlite_single_process_seam::closing_the_catalog_releases_the_file_and_its_sidecars";

    let rt = runtime();

    // Each cycle's directory is kept alive for the whole test rather than
    // dropped at the end of its cycle, so a released inode cannot be recycled
    // under the next cycle's `catalog.db` (SQLite keys its per-process lock
    // bookkeeping on `(dev, ino)`).
    let mut dirs = Vec::with_capacity(RELEASE_ITERATIONS);
    let mut wal_left_behind = 0usize;
    let mut slowest = Duration::ZERO;
    for i in 0..RELEASE_ITERATIONS {
        let dir = tempfile::tempdir().unwrap();
        let catalog = open_and_write(&rt, dir.path());

        // While live: `-wal` present (WAL is real), `-shm` absent (heap
        // wal-index).
        let (shm_live, wal_live) = sidecars(dir.path());
        assert!(
            wal_live && !shm_live,
            "cycle {i}: before close, expected (shm=false, wal=true) under the seam; got \
             (shm={shm_live}, wal={wal_live})"
        );

        rt.block_on(catalog.close());

        let (shm_after, wal_after) = sidecars(dir.path());
        assert!(
            !shm_after,
            "cycle {i}: a `-shm` file appeared; under the seam the wal-index is heap resident and \
             this process never creates one"
        );
        if wal_after {
            wal_left_behind += 1;
        }
        assert!(
            dir.path().join("catalog.db").exists(),
            "cycle {i}: closing must not remove the database itself"
        );

        // The real probe: immediately — no sleep, no retry — a second process
        // takes the file and reads what this one committed.
        let (code, signal, elapsed, log) = run_child(
            "open-expect-success",
            name,
            &[(DIR_ENV, dir.path().display().to_string())],
        );
        assert_eq!(
            signal, None,
            "cycle {i}: the successor process died with a signal:\n{log}"
        );
        assert_eq!(
            code,
            Some(EXIT_OPENED_AND_READ),
            "cycle {i}: after `close().await` a second process must open the handed-off catalog \
             and read the seeded row; it exited {code:?} after {elapsed:?} \
             ({EXIT_OPENED_BUT_EMPTY} = opened but the row was missing, {EXIT_WRONG_ERROR} = \
             still refused, so the close did not release):\n{log}"
        );
        assert!(
            elapsed < REFUSAL_CEILING,
            "cycle {i}: the successor took {elapsed:?} — a released file must be available at \
             once, not after a busy timeout:\n{log}"
        );
        slowest = slowest.max(elapsed);
        dirs.push(dir);
    }
    eprintln!(
        "[esc-073 seam] release census over {RELEASE_ITERATIONS} close/handoff cycles: successor \
         process opened every time, slowest {slowest:?}; `-wal` still present after close on \
         {wal_left_behind} cycle(s) (SQLite's own best-effort checkpoint, not a held lock)"
    );
    drop(dirs);
}

/// The seam is single-**process**, so TWO live pools on one catalog file
/// inside this process are legal and supported — and `close()`'s settle wait
/// watches `catalog.db-wal`, which is evidence about the FILE, not about the
/// pool being closed. While the survivor is live the `-wal` cannot disappear,
/// so the first `close()` necessarily runs its settle loop to the ceiling and
/// logs the "may still be held" warning.
///
/// This pins that as a bounded COST, not a correctness loss:
///
///   * the first `close()` RETURNS (within the settle ceiling plus slack) —
///     the wait is bounded by construction, never a hang; and
///   * the second pool keeps working across it — it reads what it wrote before
///     the first close, writes again after it, and reads that back. Closing
///     one pool must not disturb another pool's connections.
///
/// The elapsed time is reported rather than asserted tight: a close that finds
/// the `-wal` gone early is just as correct as one that burns the ceiling, and
/// asserting the slow shape would pin the current mechanism rather than the
/// property.
#[test]
fn closing_one_of_two_live_pools_is_bounded_and_leaves_the_other_working() {
    dispatch_child();
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();

    // Two independent pools on the SAME catalog file, in this one process.
    let first = open_and_write(&rt, dir.path());
    let second = rt
        .block_on(Catalog::open(dir.path()))
        .expect("a second pool on the same file is legal in ONE process — the seam is per-process");

    // The survivor writes its own row before the first close, so the post-close
    // read has something only it could have put there.
    let backend = second.backend_arc();
    rt.block_on(backend.transaction(TxOptions::default(), |tx| {
        Box::pin(async move {
            tx.execute(
                "INSERT OR REPLACE INTO seam_probe (id, v) VALUES (2, $1)",
                &[SqlValue::Text("second-before")],
            )
            .await?;
            Ok(())
        })
    }))
    .expect("the survivor writes while both pools are live");

    let (_, wal_live) = sidecars(dir.path());
    assert!(
        wal_live,
        "precondition: a `-wal` exists while both pools have written"
    );

    let started = Instant::now();
    rt.block_on(first.close());
    let elapsed = started.elapsed();

    assert!(
        elapsed < CLOSE_CEILING_WITH_SLACK,
        "closing one of two live pools took {elapsed:?}; the settle wait must be BOUNDED (the \
         survivor's `-wal` can never disappear, so the loop runs to its {CLOSE_SIDECAR_CEILING:?} \
         ceiling and returns — it must not hang)"
    );

    // The survivor is untouched: it reads what it wrote, writes again, and
    // reads that back.
    let backend = second.backend_arc();
    let before: Option<String> = rt
        .block_on(backend.transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt("SELECT v AS v FROM seam_probe WHERE id = 2", &[], |r| {
                        r.get::<String>("v")
                    })
                    .await
                })
            },
        ))
        .expect("the survivor still serves reads after the other pool closed");
    assert_eq!(
        before.as_deref(),
        Some("second-before"),
        "closing the first pool must not disturb the survivor's committed data"
    );

    let backend = second.backend_arc();
    rt.block_on(backend.transaction(TxOptions::default(), |tx| {
        Box::pin(async move {
            tx.execute(
                "INSERT OR REPLACE INTO seam_probe (id, v) VALUES (3, $1)",
                &[SqlValue::Text("second-after")],
            )
            .await?;
            Ok(())
        })
    }))
    .expect("the survivor still serves WRITES after the other pool closed");

    let backend = second.backend_arc();
    let after: Option<String> = rt
        .block_on(backend.transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt("SELECT v AS v FROM seam_probe WHERE id = 3", &[], |r| {
                        r.get::<String>("v")
                    })
                    .await
                })
            },
        ))
        .expect("read back the survivor's post-close write");
    assert_eq!(
        after.as_deref(),
        Some("second-after"),
        "the survivor's post-close write must be durable and readable"
    );

    eprintln!(
        "[esc-073 seam] closing 1 of 2 live pools on one file returned in {elapsed:?} (settle \
         ceiling {CLOSE_SIDECAR_CEILING:?}); the survivor read and wrote across it"
    );

    // Closing the survivor is the last close, so this one CAN see the `-wal` go.
    rt.block_on(second.close());
    drop(dir);
}

/// **Upgrade path / degenerate input.** A catalog directory left behind by a
/// pre-seam engine — `catalog.db` plus a hot `catalog.db-wal` AND a
/// `catalog.db-shm` written by the platform default VFS, abandoned without a
/// clean close — must open under the seam, recover the committed WAL frames,
/// and still be in WAL mode.
///
/// This is the boundary the seam could plausibly break: `unix-excl` ignores
/// the `-shm` file entirely and rebuilds the wal-index in heap memory, so if
/// recovery from an on-disk `-wal` did not work the first upgraded process
/// would silently lose every commit that had not been checkpointed. The child
/// exits via `std::process::exit`, which runs no destructors, so the sidecars
/// really are abandoned rather than closed.
#[test]
fn a_catalog_abandoned_by_the_pre_seam_vfs_recovers_under_the_seam() {
    dispatch_child();
    let name = "sqlite_single_process_seam::\
                a_catalog_abandoned_by_the_pre_seam_vfs_recovers_under_the_seam";

    let dir = tempfile::tempdir().unwrap();
    let (code, signal, _elapsed, log) = run_child(
        "seed-default-vfs-and-abandon",
        name,
        &[
            (SQLITE_VFS_ENV, "default".to_string()),
            (DIR_ENV, dir.path().display().to_string()),
        ],
    );
    assert_eq!(signal, None, "the seeding child died with a signal:\n{log}");
    assert_eq!(code, Some(0), "the seeding child exited {code:?}:\n{log}");

    // The pre-seam artifacts really are on disk and really were abandoned.
    let (shm, wal) = sidecars(dir.path());
    assert!(
        shm && wal,
        "the seeding child was supposed to leave a default-VFS `-shm` AND a hot `-wal` behind; \
         got (shm={shm}, wal={wal}). Without them this test proves nothing:\n{log}"
    );

    // Now open it the way an upgraded engine would.
    let rt = runtime();
    let catalog = rt
        .block_on(Catalog::open(dir.path()))
        .expect("an upgraded engine must open a catalog abandoned by the previous VFS");
    let backend = catalog.backend_arc();
    let seen = rt
        .block_on(backend.transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt("SELECT v FROM seam_probe WHERE id = 1", &[], |r| {
                        r.get::<String>("v")
                    })
                    .await
                })
            },
        ))
        .expect("read the abandoned commit");
    assert_eq!(
        seen.as_deref(),
        Some("written"),
        "the commit the abandoned `-wal` carried was lost across the VFS change — WAL recovery \
         into the heap wal-index did not happen"
    );
    assert_eq!(
        journal_mode(&rt, &catalog),
        "wal",
        "the recovered catalog is no longer in WAL mode"
    );

    rt.block_on(catalog.close());

    // The pre-seam `-shm` is LEFT BEHIND, and that is the behaviour to pin:
    // this process never opened it (the wal-index is heap resident), so it is
    // not this process's file to delete. It is inert litter, and `close()`
    // deliberately does not wait on it — waiting for a file nobody here will
    // ever remove would turn every close on an upgraded catalog directory into
    // a full timeout.
    //
    // The `-wal`'s disappearance is deliberately NOT asserted here. SQLite
    // deletes it only when its close-time PASSIVE checkpoint completes, and on
    // a directory an abandoned foreign-VFS process left mid-flight that
    // checkpoint may legitimately decline; the release handshake is measured
    // on a clean catalog by
    // [`closing_the_catalog_releases_the_file_and_its_sidecars`]. What this
    // oracle owns is the recovery claim above.
    let (shm_after, _wal_after) = sidecars(dir.path());
    assert!(
        shm_after,
        "the pre-seam `-shm` was removed; if the engine has started touching `-shm` files again \
         the heap-wal-index claim needs re-checking"
    );
}
