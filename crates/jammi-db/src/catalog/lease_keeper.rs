//! One lease-renewal thread per process (N3): a dedicated OS thread running
//! its own `current_thread` tokio runtime and its OWN catalog connection,
//! renewing every lease the process holds from a hold list.
//!
//! **Why a dedicated thread, not a `tokio::spawn`'d task.** A lease
//! heartbeat that runs as a task on the process's MAIN runtime competes for
//! the same worker threads an inline compute job's CPU-bound (blocking)
//! work occupies. N+1 such jobs on an N-thread runtime can starve every
//! async task on it — including the heartbeat — for the whole duration of
//! the blocking work, which is exactly the shape that let a live holder's
//! lease expire out from under it and be reclaimed by a peer. A dedicated
//! `std::thread` with its own runtime and its own connection can never be
//! starved by contention on the main runtime's worker pool.
//!
//! **Why the thread opens its OWN connection, never borrows the caller's.**
//! A `sqlx` connection pool's background bookkeeping (Postgres's async
//! socket reactor registration in particular) is tied to the tokio runtime
//! it was created on. Reusing a pool built on the (possibly starved) main
//! runtime from this thread's own runtime would reintroduce the exact
//! starvation this primitive exists to route around. `catalog_connect` is
//! therefore a factory the CALLER supplies — it alone knows how to reopen a
//! fresh connection to whatever backend is in play — and is invoked only
//! from inside the thread's own runtime (once per connect attempt; see
//! [`LeaseKeeper::start`] for the bounded retry).
//!
//! **Why a hold reads the keeper's liveness, not only its own renew
//! outcome.** A `lost` flag that is flipped ONLY by a renew that matched
//! zero rows is silent in every failure mode where no renew runs at all: a
//! keeper thread that never connected, one that panicked, one whose
//! backend hung. In each of those the row's lease genuinely expires (a
//! peer's reclaim is free to take it) while every holder in this process
//! keeps reading "live" forever. So [`LeaseKeeper::start`] refuses to
//! return a keeper that has not connected and completed its first renewal
//! pass, and [`LeaseHold::lost`] folds in two more arms beyond its own
//! renew outcome: the keeper thread has exited (for any reason — the
//! thread's exit guard also flips every registered hold's flag), or this
//! hold's last successful renewal is older than the lease window (the
//! keeper is alive but stalled, or every renewal has been erroring for that
//! long — either way the row's lease has already lapsed).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use tracing::{error, warn};

use super::lease::LeaseIntervals;
use super::result_repo::ResultTableCas;
use super::Catalog;
use crate::error::{JammiError, Result};

/// What one [`LeaseHold`] renews.
#[derive(Debug, Clone)]
pub enum LeaseTarget {
    /// This process's own `instances` row — renewed via
    /// [`Catalog::touch_instance`].
    Instance(String),
    /// A claimed `jobs` row — renewed via [`Catalog::heartbeat_job`], the
    /// same full attempt guard (`claimed_by`/`status`/`attempts`) every
    /// other lease-guarded job write carries.
    Job {
        job_id: String,
        instance_id: String,
        attempts: u32,
    },
    /// A `building` `result_tables` row — renewed via
    /// [`Catalog::renew_lease`] under [`ResultTableCas::writer_any_tenant`].
    ResultTable { table: String, writer_id: String },
    /// A `building` `result_table_versions` row (an in-flight refresh or
    /// compaction of a ready table) — renewed via
    /// [`Catalog::renew_version_lease`] under the writer's own CAS with no
    /// tenant arm, for the same reason `ResultTable` uses
    /// `writer_any_tenant`.
    ResultTableVersion {
        table: String,
        version: i64,
        writer_id: String,
    },
}

/// The keeper thread's liveness, shared by the keeper handle and every
/// hold it issues: whether the thread is running, when it last completed a
/// renewal pass, and the lease window a hold's own last renewal is judged
/// against. Times are milliseconds since `epoch` so they fit an atomic.
struct Liveness {
    epoch: Instant,
    alive: AtomicBool,
    last_pass_ms: AtomicU64,
    lease: Duration,
}

impl Liveness {
    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn stamp(&self, cell: &AtomicU64) {
        cell.store(self.now_ms(), Ordering::SeqCst);
    }

    /// `true` when `cell`'s stamp is older than the lease window.
    fn lapsed(&self, cell: &AtomicU64) -> bool {
        let age = self.now_ms().saturating_sub(cell.load(Ordering::SeqCst));
        age > self.lease.as_millis() as u64
    }

    fn instant_of(&self, cell: &AtomicU64) -> Instant {
        self.epoch + Duration::from_millis(cell.load(Ordering::SeqCst))
    }
}

/// A live hold on a lease target with a [`LeaseKeeper`]. Renewed every
/// `heartbeat` interval until this handle is dropped, at which point the
/// target is removed from the keeper's list and renewed no more — "release
/// on drop".
pub struct LeaseHold {
    id: u64,
    holds: Arc<Mutex<HashMap<u64, HeldState>>>,
    lost: Arc<AtomicBool>,
    last_renewed_ms: Arc<AtomicU64>,
    liveness: Arc<Liveness>,
}

impl LeaseHold {
    /// `true` once this hold cannot be trusted to own its row, on any
    /// of three arms: a renew for this hold matched zero rows (the row was
    /// claimed by a peer, or went terminal, underneath this holder); the
    /// keeper thread has exited (a panic, a failed reconnect, shutdown) so
    /// nothing renews anything any more; or this hold's last SUCCESSFUL
    /// renewal is older than the lease window (the keeper is alive but has
    /// not managed to renew this row within the time a peer's reclaim needs
    /// to take it). A holder observing `true` must treat its claim as gone:
    /// it does not own the row and must not act as though it does.
    pub fn lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
            || !self.liveness.alive.load(Ordering::SeqCst)
            || self.liveness.lapsed(&self.last_renewed_ms)
    }

    /// When this hold's renewal last matched its row — the hold's creation
    /// instant until the keeper's first successful renewal of it.
    pub fn last_renewed_at(&self) -> Instant {
        self.liveness.instant_of(&self.last_renewed_ms)
    }

    /// A clone of this hold's own `lost` flag — for a caller that already
    /// threads an `Arc<AtomicBool>` cancellation flag through a deep call
    /// chain (a training loop's epoch-boundary check) and wants the
    /// keeper's renewal outcome to set that SAME flag directly, with no
    /// separate polling task: the flag this returns is flipped by the
    /// keeper's own dedicated thread on the very next renewal that misses,
    /// and by the thread's exit guard when the keeper dies for any reason,
    /// exactly as [`Self::lost`] observes those two arms, just reachable by
    /// identity rather than by asking this handle. The third arm
    /// [`Self::lost`] folds in — a keeper that is alive but has not renewed
    /// this hold within the lease window — has no thread able to flip a
    /// flag, so it is visible only through [`Self::lost`] itself. The hold
    /// must still be kept alive (not dropped) for as long as the lease
    /// should keep renewing — dropping it releases the hold, but does not
    /// retroactively un-flip a flag a caller is still holding a clone of.
    pub fn lost_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.lost)
    }
}

impl Drop for LeaseHold {
    fn drop(&mut self) {
        self.holds
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.id);
    }
}

struct HeldState {
    target: LeaseTarget,
    lost: Arc<AtomicBool>,
    last_renewed_ms: Arc<AtomicU64>,
    /// Set by [`LeaseKeeper::release_job_holds`] once this hold's row was
    /// RELEASED (`Catalog::release_job_lease` returned `Ok(true)`) — only
    /// ever on a [`LeaseTarget::Job`] hold. `renew_all` skips a released
    /// hold; an optimisation only — the SQL `lease_expires_at IS NOT NULL`
    /// arm on `heartbeat_job` is the guarantee that a released lease is
    /// never re-armed, and a hold registered AFTER the release (which has
    /// no flag) misses through that same arm and flips `lost`.
    released: bool,
}

/// The one-shot handshake behind [`LeaseKeeper::release_job_holds`]: the
/// caller sets `requested` and wakes the thread; the thread runs the release
/// pass on its own connection, publishes the count, and notifies `done`.
struct ReleaseRequest {
    requested: AtomicBool,
    released: AtomicUsize,
    done: tokio::sync::Notify,
}

/// A future-returning catalog-connection factory: called from inside the
/// keeper thread's own runtime, once per connect attempt, to open the
/// connection every renewal on that thread uses.
type CatalogConnect =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = Result<Catalog>> + Send>> + Send + 'static>;

/// Flips the keeper's liveness off — and every registered hold's `lost`
/// flag on — when the keeper thread exits, however it exits: a clean
/// shutdown, a `return` after a failed connect, or an unwinding panic.
struct ExitGuard {
    liveness: Arc<Liveness>,
    holds: Arc<Mutex<HashMap<u64, HeldState>>>,
}

impl Drop for ExitGuard {
    fn drop(&mut self) {
        self.liveness.alive.store(false, Ordering::SeqCst);
        for held in self
            .holds
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
        {
            held.lost.store(true, Ordering::SeqCst);
        }
    }
}

/// Shortest and longest pause between two connect attempts inside the
/// keeper thread — doubling from the first to the second, and never
/// sleeping past the lease-window deadline.
const CONNECT_BACKOFF_MIN: Duration = Duration::from_millis(50);
const CONNECT_BACKOFF_MAX: Duration = Duration::from_secs(1);

/// One lease-renewal thread per process. See the module docs for why a
/// dedicated thread and a dedicated connection, rather than a `tokio::spawn`
/// task on the caller's own runtime.
pub struct LeaseKeeper {
    holds: Arc<Mutex<HashMap<u64, HeldState>>>,
    next_id: AtomicU64,
    shutdown: Arc<AtomicBool>,
    /// Wakes the thread's sleep the instant [`Self::shutdown_and_join`] (or
    /// `Drop`) signals `shutdown`, rather than leaving it to notice on the
    /// next heartbeat tick (which, at the engine's default 10 s heartbeat,
    /// would make a caller waiting to release the catalog file wait up to
    /// 10 s for nothing).
    wake: Arc<tokio::sync::Notify>,
    liveness: Arc<Liveness>,
    release: Arc<ReleaseRequest>,
    #[cfg(feature = "test-hooks")]
    kill: Arc<AtomicBool>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl LeaseKeeper {
    /// Start the keeper thread and return a shared handle to it once the
    /// thread is genuinely able to renew: it has built its runtime, opened
    /// its own catalog connection through `catalog_connect`, and completed
    /// its first renewal pass. `intervals` is the deployment's lease timing
    /// — the SAME [`LeaseIntervals`] every other leased row family renews
    /// under, so a job/table lease this keeper renews always targets the
    /// identical deadline the reclaim path compares against.
    ///
    /// A connect that fails is retried inside the thread with a bounded,
    /// doubling backoff (50 ms up to 1 s between attempts) until the lease
    /// window has elapsed; a [`JammiError::Config`] from the factory is
    /// taken as permanent and ends the retry at once. When no attempt
    /// succeeds within the window this returns the typed error (a
    /// [`JammiError::Catalog`] carrying the last connect error, or the
    /// `Config` error itself) rather than a handle whose holds would all
    /// read "live" while nothing renews them. A runtime that cannot be built
    /// is a [`JammiError::Config`].
    pub async fn start<F, Fut>(catalog_connect: F, intervals: LeaseIntervals) -> Result<Arc<Self>>
    where
        F: Fn() -> Fut + Send + 'static,
        Fut: Future<Output = Result<Catalog>> + Send + 'static,
    {
        let connect: CatalogConnect = Box::new(move || Box::pin(catalog_connect()));
        let holds: Arc<Mutex<HashMap<u64, HeldState>>> = Arc::new(Mutex::new(HashMap::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let wake = Arc::new(tokio::sync::Notify::new());
        let liveness = Arc::new(Liveness {
            epoch: Instant::now(),
            alive: AtomicBool::new(false),
            last_pass_ms: AtomicU64::new(0),
            lease: intervals.lease(),
        });
        let release = Arc::new(ReleaseRequest {
            requested: AtomicBool::new(false),
            released: AtomicUsize::new(0),
            done: tokio::sync::Notify::new(),
        });
        #[cfg(feature = "test-hooks")]
        let kill = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<()>>();

        let thread_holds = Arc::clone(&holds);
        let thread_shutdown = Arc::clone(&shutdown);
        let thread_wake = Arc::clone(&wake);
        let thread_liveness = Arc::clone(&liveness);
        let thread_release = Arc::clone(&release);
        #[cfg(feature = "test-hooks")]
        let thread_kill = Arc::clone(&kill);
        let thread = std::thread::Builder::new()
            .name("jammi-lease-keeper".into())
            .spawn(move || {
                let _exit = ExitGuard {
                    liveness: Arc::clone(&thread_liveness),
                    holds: Arc::clone(&thread_holds),
                };
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        error!(error = %e, "lease keeper: failed to build its dedicated runtime");
                        let _ = ready_tx.send(Err(JammiError::Config(format!(
                            "lease keeper: failed to build its dedicated runtime: {e}"
                        ))));
                        return;
                    }
                };
                rt.block_on(async move {
                    let catalog = match connect_with_backoff(&connect, intervals.lease()).await {
                        Ok(c) => c,
                        Err(e) => {
                            error!(error = %e, "lease keeper: gave up opening its own catalog connection");
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    // The first renewal pass: trivially empty at this point
                    // (no hold can be registered before `start` returns),
                    // but it is the pass every hold's staleness arm is
                    // dated from, so it runs before readiness is reported.
                    renew_all(&catalog, &thread_holds, &thread_liveness, intervals).await;
                    thread_liveness.alive.store(true, Ordering::SeqCst);
                    let _ = ready_tx.send(Ok(()));
                    loop {
                        // Wake on whichever comes first: the next heartbeat
                        // tick, or a shutdown signal — so
                        // `shutdown_and_join`'s wait is bounded by a
                        // scheduling quantum, never a full heartbeat
                        // interval (the engine's default is 10 s).
                        tokio::select! {
                            _ = tokio::time::sleep(intervals.heartbeat()) => {},
                            _ = thread_wake.notified() => {},
                        }
                        if thread_shutdown.load(Ordering::SeqCst) {
                            // The keeper's own catalog connection is closed
                            // HERE, on the thread that owns it, before the
                            // thread exits — the one bounded release point
                            // `shutdown_and_join` waits for. Never rely on
                            // this connection's `Drop`: closing a `Catalog`
                            // without awaiting `close()` returns nothing to
                            // the pool for an unbounded time (see
                            // `Catalog::close`'s doc), which for the SQLite
                            // backend is exactly the connection the
                            // process-exclusive `unix-excl` lock is held by.
                            catalog.close().await;
                            return;
                        }
                        #[cfg(feature = "test-hooks")]
                        if thread_kill.load(Ordering::SeqCst) {
                            panic!("lease keeper: thread killed by test hook");
                        }
                        // A RELEASE request runs here, on this thread and
                        // this connection, serialised with the renewal pass
                        // that follows it — so no renewal of a hold this
                        // pass releases can be in flight beside it.
                        if thread_release.requested.swap(false, Ordering::SeqCst) {
                            let n = release_job_holds_on_thread(&catalog, &thread_holds).await;
                            thread_release.released.store(n, Ordering::SeqCst);
                            thread_release.done.notify_waiters();
                        }
                        renew_all(&catalog, &thread_holds, &thread_liveness, intervals).await;
                    }
                });
            })
            .map_err(|e| {
                JammiError::Config(format!("lease keeper: failed to spawn its thread: {e}"))
            })?;

        match ready_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(JammiError::Catalog(
                    "lease keeper: thread exited before reporting readiness".into(),
                ))
            }
        }

        Ok(Arc::new(Self {
            holds,
            next_id: AtomicU64::new(0),
            shutdown,
            wake,
            liveness,
            release,
            #[cfg(feature = "test-hooks")]
            kill,
            thread: Mutex::new(Some(thread)),
        }))
    }

    /// Hold a lease target open, renewed on this keeper's thread. Renewal
    /// starts on the NEXT tick (at most one `heartbeat` interval away, never
    /// blocking the caller); stops when the returned [`LeaseHold`] is
    /// dropped. The hold's own "last renewed" stamp starts at this call —
    /// the caller has just claimed the row, so its lease is fresh from the
    /// claim.
    pub fn hold(&self, target: LeaseTarget) -> LeaseHold {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let lost = Arc::new(AtomicBool::new(false));
        let last_renewed_ms = Arc::new(AtomicU64::new(0));
        self.liveness.stamp(&last_renewed_ms);
        self.holds
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(
                id,
                HeldState {
                    target,
                    lost: Arc::clone(&lost),
                    last_renewed_ms: Arc::clone(&last_renewed_ms),
                    released: false,
                },
            );
        LeaseHold {
            id,
            holds: Arc::clone(&self.holds),
            lost,
            last_renewed_ms,
            liveness: Arc::clone(&self.liveness),
        }
    }

    /// RELEASE every [`LeaseTarget::Job`] hold this keeper currently holds
    /// — the per-hold half of a two-mode shutdown's RELEASE arm — and wait,
    /// bounded by `bound` (one heartbeat), for the pass to complete.
    /// Returns how many holds were released.
    ///
    /// The pass runs ON THE KEEPER THREAD, on its own connection, serialised
    /// with `renew_all`: for every registered `Job` hold not already
    /// released it issues `Catalog::release_job_lease(job_id, instance_id,
    /// attempts)`; `Ok(true)` flips the hold's `lost` flag (so a training
    /// loop polling it bails at its next epoch boundary without writing a
    /// bundle) and marks the hold released; `Ok(false)` — an INLINE row
    /// (`execution = 'inline'` never matches), a row a peer already took,
    /// or a lease already released — leaves the hold untouched, as does an
    /// `Err` (logged; the expiry path covers it). Runs BEFORE any loop task
    /// is aborted, so every live hold's `lost` flips while the hold still
    /// exists.
    ///
    /// ONLY the `Job` class. A [`LeaseTarget::ResultTable`] hold — the
    /// second lease class a compute job holds — is never touched here:
    /// every `BuildingTable` on a session is adopted under the store's one
    /// `writer_id`, so from this map a loop-claimed materialization's table,
    /// an inline `run_now`'s and a library materialization's are
    /// indistinguishable, and a per-target release would NULL the wrong
    /// lease. That class is released solely by the jobs-linked sweep
    /// (`Catalog::release_building_tables_of_claimant`), after which the
    /// loop's own `ResultTable` hold flips `lost` through the guarded
    /// renewal (`Catalog::renew_lease`'s lease-present arm) within one
    /// heartbeat.
    ///
    /// # Errors
    ///
    /// [`JammiError::Catalog`] when the keeper thread is dead (nothing can
    /// run the pass — every hold already reads lost) or when the pass has
    /// not completed within `bound` — never a hang.
    pub async fn release_job_holds(&self, bound: Duration) -> Result<usize> {
        if !self.is_alive() {
            return Err(JammiError::Catalog(
                "lease keeper: cannot release job holds, its thread is dead".into(),
            ));
        }
        // Register interest BEFORE requesting, so a `notify_waiters` that
        // lands between the two cannot be missed (`Notify::notify_waiters`
        // stores no permit).
        let done = self.release.done.notified();
        tokio::pin!(done);
        done.as_mut().enable();
        self.release.requested.store(true, Ordering::SeqCst);
        self.wake.notify_one();
        match tokio::time::timeout(bound, done).await {
            Ok(()) => Ok(self.release.released.load(Ordering::SeqCst)),
            Err(_elapsed) => Err(JammiError::Catalog(format!(
                "lease keeper: release_job_holds did not complete within {bound:?}"
            ))),
        }
    }

    /// `true` while the keeper thread is running its renewal loop. `false`
    /// once it has exited for any reason — and every [`LeaseHold`] issued by
    /// this keeper reports [`LeaseHold::lost`] from that moment.
    pub fn is_alive(&self) -> bool {
        self.liveness.alive.load(Ordering::SeqCst)
    }

    /// When the keeper thread last completed a renewal pass over every
    /// hold (the pass that ran before [`Self::start`] returned, until the
    /// first heartbeat tick).
    pub fn last_renewed_at(&self) -> Instant {
        self.liveness.instant_of(&self.liveness.last_pass_ms)
    }

    /// Test hook: make the keeper thread panic at its next heartbeat tick,
    /// the way a defect inside the renewal loop would kill it — so a test
    /// can prove every hold observes the death rather than reading "live"
    /// forever.
    #[cfg(feature = "test-hooks")]
    pub fn kill_thread_for_test(&self) {
        self.kill.store(true, Ordering::SeqCst);
    }

    /// Signal the keeper thread to stop, wait — bounded by `timeout` — for
    /// it to actually exit, and reap its `JoinHandle`.
    ///
    /// Unlike `Drop` ("best-effort … not joined synchronously"), this is
    /// the ONE path that guarantees the keeper's own catalog connection is
    /// gone by the time it returns `Ok(())`: the thread closes that
    /// connection itself (see the `shutdown` branch inside [`Self::start`]'s
    /// loop) before it exits, so observing `Ok(())` here is the caller's
    /// evidence that whatever process-exclusive lock that connection held
    /// (the SQLite `unix-excl` VFS, in particular) has actually been
    /// released — never merely that a flag was set.
    ///
    /// The thread is woken immediately via a `Notify` rather than left to
    /// notice on its next heartbeat tick, so in practice this returns within
    /// a scheduling quantum, not the full heartbeat interval (10 s by
    /// default). The actual `std::thread::JoinHandle::join` is a blocking
    /// OS call, so it runs on tokio's blocking pool (`spawn_blocking`)
    /// rather than on this async fn's own executor thread — and NOT via
    /// `JoinHandle::is_finished` polling, which was measured on this
    /// platform to lag the thread's real exit by seconds under load (the
    /// std docs make no bounded-latency promise for it; a genuine blocking
    /// `join` does not have that hazard).
    ///
    /// Idempotent: a second call — after the first already took the
    /// handle (whether it went on to succeed, time out, or observe a
    /// panicked thread) — finds no handle left and returns `Ok(())` at
    /// once, without waiting on whatever the first call's `join` is still
    /// doing.
    ///
    /// # Errors
    ///
    /// [`JammiError::Catalog`] when the thread has not exited within
    /// `timeout` — never a hang — or when it exited via a panic (its
    /// shutdown branch may never have run, so this call cannot promise the
    /// connection is closed). Either way `shutdown` stays set, so the
    /// thread's own progress (if it is still running) continues toward
    /// closing its connection regardless of this call having given up on
    /// waiting for it.
    pub async fn shutdown_and_join(&self, timeout: Duration) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        self.wake.notify_one();
        let handle = self
            .thread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let Some(handle) = handle else {
            // Already taken by an earlier call (or a concurrent one) — the
            // thread has exited or is being waited on elsewhere.
            return Ok(());
        };
        match tokio::time::timeout(timeout, tokio::task::spawn_blocking(move || handle.join()))
            .await
        {
            Ok(Ok(Ok(()))) => Ok(()),
            Ok(Ok(Err(panic))) => {
                // The thread panicked (e.g. the `test-hooks` kill switch, or
                // a genuine defect) rather than reaching its own `shutdown`
                // branch — its catalog connection may still be open. The
                // payload is `Box<dyn Any + Send>`, which carries neither
                // `Debug` nor `Display`; extract the message the same way
                // `std`'s own default panic hook does, for the two payload
                // shapes `panic!`/`assert!` actually produce.
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                error!(
                    panic = %message,
                    "lease keeper: thread panicked during shutdown; its catalog connection may \
                     still be open"
                );
                Err(JammiError::Catalog(format!(
                    "lease keeper: thread panicked before it could close its own catalog \
                     connection: {message}"
                )))
            }
            Ok(Err(join_err)) => Err(JammiError::Catalog(format!(
                "lease keeper: internal error joining its thread: {join_err}"
            ))),
            Err(_elapsed) => Err(JammiError::Catalog(format!(
                "lease keeper: thread did not exit within the {timeout:?} shutdown window; \
                 its own catalog connection may still be open"
            ))),
        }
    }
}

impl Drop for LeaseKeeper {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.wake.notify_one();
        if let Some(handle) = self
            .thread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            // Best-effort: the thread wakes (immediately, via `wake`, rather
            // than waiting for its next heartbeat tick) and exits, closing
            // its own catalog connection on its own time. Not joined
            // synchronously here — `Drop` runs on whatever thread drops the
            // last `Arc<LeaseKeeper>`, which may itself be an async task
            // that must not block on a `std::thread::join`, so a caller that
            // needs the connection actually gone by a bounded point must
            // call `shutdown_and_join` — dropping this handle is NOT that
            // release point.
            drop(handle);
        }
    }
}

/// Open the keeper's connection, retrying with a doubling backoff until
/// `window` has elapsed since the first attempt. A [`JammiError::Config`]
/// is permanent (a bad URL, a missing driver) and ends the retry at once;
/// every other error is retried while time remains. Each attempt is itself
/// bounded by the time left in the window, so a connect that never resolves
/// cannot hold `start` past the deadline either.
async fn connect_with_backoff(connect: &CatalogConnect, window: Duration) -> Result<Catalog> {
    let deadline = Instant::now() + window;
    let mut backoff = CONNECT_BACKOFF_MIN;
    let mut attempts: u32 = 0;
    loop {
        attempts += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        let outcome =
            match tokio::time::timeout(remaining.max(Duration::from_millis(1)), connect()).await {
                Ok(outcome) => outcome,
                Err(_elapsed) => Err(JammiError::Catalog(format!(
                    "connect attempt {attempts} did not complete within the lease window"
                ))),
            };
        match outcome {
            Ok(catalog) => return Ok(catalog),
            Err(e @ JammiError::Config(_)) => return Err(e),
            Err(e) => {
                let now = Instant::now();
                if now >= deadline {
                    return Err(JammiError::Catalog(format!(
                        "lease keeper: could not open its own catalog connection within the \
                         {:?} lease window ({attempts} attempts; last error: {e})",
                        window
                    )));
                }
                warn!(attempt = attempts, error = %e, "lease keeper: catalog connect failed; retrying");
                let pause = backoff.min(deadline - now);
                tokio::time::sleep(pause).await;
                backoff = (backoff * 2).min(CONNECT_BACKOFF_MAX);
            }
        }
    }
}

/// One renewal pass over every current hold. A hold whose renew misses
/// (matches zero rows) has its `lost` flag set; a hold whose renew lands
/// has its "last renewed" stamp refreshed; every other outcome (a
/// transient backend error, logged and retried next tick) leaves both
/// alone — so a hold that keeps erroring past the lease window reports
/// lost through [`LeaseHold::lost`]'s staleness arm, never silently
/// "live". Holds added or removed mid-pass are not raced against: the
/// snapshot is taken once at the top of the pass, and a [`LeaseHold`]'s
/// `Drop` removing it from the map does not affect a renewal already in
/// flight for it — the renewal simply writes to a row no one observes the
/// outcome of.
async fn renew_all(
    catalog: &Catalog,
    holds: &Arc<Mutex<HashMap<u64, HeldState>>>,
    liveness: &Liveness,
    intervals: LeaseIntervals,
) {
    let snapshot: Vec<(LeaseTarget, Arc<AtomicBool>, Arc<AtomicU64>)> = {
        let regs = holds.lock().unwrap_or_else(PoisonError::into_inner);
        regs.values()
            // A released Job hold is never renewed again (its row's lease is
            // NULL and `heartbeat_job`'s `IS NOT NULL` arm would miss anyway).
            .filter(|r| !r.released)
            .map(|r| {
                (
                    r.target.clone(),
                    Arc::clone(&r.lost),
                    Arc::clone(&r.last_renewed_ms),
                )
            })
            .collect()
    };
    for (target, lost, last_renewed_ms) in snapshot {
        let renewed = match &target {
            LeaseTarget::Instance(instance_id) => match catalog.touch_instance(instance_id).await {
                Ok(matched) => Some(matched),
                Err(e) => {
                    warn!(instance_id, error = %e, "lease keeper: instance heartbeat failed");
                    None
                }
            },
            LeaseTarget::Job {
                job_id,
                instance_id,
                attempts,
            } => match catalog
                .heartbeat_job(job_id, instance_id, *attempts, intervals.lease())
                .await
            {
                Ok(matched) => Some(matched),
                Err(e) => {
                    warn!(job_id, error = %e, "lease keeper: job heartbeat failed");
                    None
                }
            },
            LeaseTarget::ResultTable { table, writer_id } => {
                let cas = ResultTableCas::writer_any_tenant(table, writer_id);
                match catalog.renew_lease(&cas, intervals.lease()).await {
                    Ok(()) => Some(true),
                    Err(
                        JammiError::RowGone { .. }
                        | JammiError::TenantMismatch { .. }
                        | JammiError::LeaseLost { .. }
                        | JammiError::CasFailed { .. },
                    ) => Some(false),
                    Err(e) => {
                        warn!(table, error = %e, "lease keeper: result-table heartbeat failed");
                        None
                    }
                }
            }
            LeaseTarget::ResultTableVersion {
                table,
                version,
                writer_id,
            } => {
                let cas = crate::catalog::version_repo::VersionCas::writer_any_tenant(
                    table, *version, writer_id,
                );
                match catalog.renew_version_lease(&cas, intervals.lease()).await {
                    Ok(()) => Some(true),
                    Err(
                        JammiError::RowGone { .. }
                        | JammiError::TenantMismatch { .. }
                        | JammiError::LeaseLost { .. }
                        | JammiError::CasFailed { .. },
                    ) => Some(false),
                    Err(e) => {
                        warn!(table, version, error = %e, "lease keeper: version heartbeat failed");
                        None
                    }
                }
            }
        };
        match renewed {
            Some(true) => liveness.stamp(&last_renewed_ms),
            Some(false) => lost.store(true, Ordering::SeqCst),
            None => {}
        }
    }
    liveness.stamp(&liveness.last_pass_ms);
}

/// The RELEASE pass [`LeaseKeeper::release_job_holds`] runs on the keeper
/// thread: release every not-yet-released [`LeaseTarget::Job`] hold's row
/// through `Catalog::release_job_lease`, flipping `lost` and marking the
/// hold released on `Ok(true)`. Returns the count released. The snapshot is
/// taken once; a hold dropped mid-pass simply has its row released to no
/// observer.
async fn release_job_holds_on_thread(
    catalog: &Catalog,
    holds: &Arc<Mutex<HashMap<u64, HeldState>>>,
) -> usize {
    let snapshot: Vec<(u64, String, String, u32, Arc<AtomicBool>)> = {
        let regs = holds.lock().unwrap_or_else(PoisonError::into_inner);
        regs.iter()
            .filter(|(_, r)| !r.released)
            .filter_map(|(id, r)| match &r.target {
                LeaseTarget::Job {
                    job_id,
                    instance_id,
                    attempts,
                } => Some((
                    *id,
                    job_id.clone(),
                    instance_id.clone(),
                    *attempts,
                    Arc::clone(&r.lost),
                )),
                LeaseTarget::Instance(_) | LeaseTarget::ResultTable { .. } => None,
            })
            .collect()
    };
    let mut released = 0;
    for (id, job_id, instance_id, attempts, lost) in snapshot {
        match catalog
            .release_job_lease(&job_id, &instance_id, attempts)
            .await
        {
            Ok(true) => {
                lost.store(true, Ordering::SeqCst);
                if let Some(held) = holds
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get_mut(&id)
                {
                    held.released = true;
                }
                released += 1;
            }
            Ok(false) => {}
            Err(e) => {
                warn!(job_id, error = %e, "lease keeper: job lease release failed; left to expiry");
            }
        }
    }
    released
}
