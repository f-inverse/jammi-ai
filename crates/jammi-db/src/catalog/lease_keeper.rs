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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    /// `true` once this hold can no longer be trusted to own its row, on any
    /// of three arms: a renew for this hold matched zero rows (the row was
    /// claimed by a peer, or went terminal, underneath this holder); the
    /// keeper thread has exited (a panic, a failed reconnect, shutdown) so
    /// nothing renews anything any more; or this hold's last SUCCESSFUL
    /// renewal is older than the lease window (the keeper is alive but has
    /// not managed to renew this row within the time a peer's reclaim needs
    /// to take it). A holder observing `true` must treat its claim as gone:
    /// it no longer owns the row and must not act as though it does.
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
    liveness: Arc<Liveness>,
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
        let liveness = Arc::new(Liveness {
            epoch: Instant::now(),
            alive: AtomicBool::new(false),
            last_pass_ms: AtomicU64::new(0),
            lease: intervals.lease(),
        });
        #[cfg(feature = "test-hooks")]
        let kill = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<()>>();

        let thread_holds = Arc::clone(&holds);
        let thread_shutdown = Arc::clone(&shutdown);
        let thread_liveness = Arc::clone(&liveness);
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
                        tokio::time::sleep(intervals.heartbeat()).await;
                        if thread_shutdown.load(Ordering::SeqCst) {
                            return;
                        }
                        #[cfg(feature = "test-hooks")]
                        if thread_kill.load(Ordering::SeqCst) {
                            panic!("lease keeper: thread killed by test hook");
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
            liveness,
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
}

impl Drop for LeaseKeeper {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self
            .thread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            // Best-effort: the thread wakes at the next heartbeat tick and
            // exits. Not joined synchronously here — `Drop` runs on whatever
            // thread drops the last `Arc<LeaseKeeper>`, which may itself be
            // an async task that must not block on a `std::thread::join`.
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
        };
        match renewed {
            Some(true) => liveness.stamp(&last_renewed_ms),
            Some(false) => lost.store(true, Ordering::SeqCst),
            None => {}
        }
    }
    liveness.stamp(&liveness.last_pass_ms);
}
