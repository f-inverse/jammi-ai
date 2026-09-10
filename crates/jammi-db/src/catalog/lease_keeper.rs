//! One lease-renewal thread per process (N3): a dedicated OS thread running
//! its own `current_thread` tokio runtime and its OWN catalog connection,
//! renewing every lease the process holds from a registration list.
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
//! fresh connection to whatever backend is in play — and is invoked exactly
//! once, from inside the thread's own runtime.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tracing::{error, warn};

use super::lease::LeaseIntervals;
use super::result_repo::ResultTableCas;
use super::Catalog;
use crate::error::{JammiError, Result};

/// What one [`Registration`] renews.
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

/// A live registration with a [`LeaseKeeper`]. Renewed every `heartbeat`
/// interval until this handle is dropped, at which point the target is
/// removed from the keeper's list and renewed no more — "unregister on
/// drop".
pub struct Registration {
    id: u64,
    registrations: Arc<Mutex<HashMap<u64, RegistrationState>>>,
    lost: Arc<AtomicBool>,
}

impl Registration {
    /// `true` once a renew for this registration matched zero rows — the row
    /// was claimed by a peer, or went terminal, underneath this holder. A
    /// holder observing `true` must treat its claim as gone: it no longer
    /// owns the row and must not act as though it does.
    pub fn lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// A clone of this registration's own `lost` flag — for a caller that
    /// already threads an `Arc<AtomicBool>` cancellation flag through a deep
    /// call chain (a training loop's epoch-boundary check) and wants the
    /// keeper's renewal outcome to set that SAME flag directly, with no
    /// separate polling task: the flag this returns is flipped by the
    /// keeper's own dedicated thread on the very next renewal that misses,
    /// exactly as [`Self::lost`] observes it, just reachable by identity
    /// rather than by asking this handle. The registration itself must
    /// still be kept alive (not dropped) for as long as the lease should
    /// keep renewing — dropping it unregisters, but does not retroactively
    /// un-flip a flag a caller is still holding a clone of.
    pub fn lost_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.lost)
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.registrations.lock().unwrap().remove(&self.id);
    }
}

struct RegistrationState {
    target: LeaseTarget,
    lost: Arc<AtomicBool>,
}

/// A future-returning catalog-connection factory: called exactly once, from
/// inside the keeper thread's own runtime, to open the connection every
/// renewal on that thread uses. `Fn` (not `FnOnce`) only because closures
/// capture by reference by default; [`LeaseKeeper::start`] calls it once.
type CatalogConnect =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = Result<Catalog>> + Send>> + Send + 'static>;

/// One lease-renewal thread per process. See the module docs for why a
/// dedicated thread and a dedicated connection, rather than a `tokio::spawn`
/// task on the caller's own runtime.
pub struct LeaseKeeper {
    registrations: Arc<Mutex<HashMap<u64, RegistrationState>>>,
    next_id: AtomicU64,
    shutdown: Arc<AtomicBool>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl LeaseKeeper {
    /// Start the keeper thread and return a shared handle to it.
    /// `catalog_connect` is invoked exactly once, inside the thread's own
    /// `current_thread` runtime, to open the connection every renewal on
    /// this thread reuses; `intervals` is the deployment's lease timing — the
    /// SAME [`LeaseIntervals`] every other leased row family renews under, so
    /// a job/table lease this keeper renews always targets the identical
    /// deadline the reclaim path compares against.
    pub fn start<F, Fut>(catalog_connect: F, intervals: LeaseIntervals) -> Arc<Self>
    where
        F: Fn() -> Fut + Send + 'static,
        Fut: Future<Output = Result<Catalog>> + Send + 'static,
    {
        let connect: CatalogConnect = Box::new(move || Box::pin(catalog_connect()));
        let registrations: Arc<Mutex<HashMap<u64, RegistrationState>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let thread_registrations = Arc::clone(&registrations);
        let thread_shutdown = Arc::clone(&shutdown);
        let thread = std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    error!(error = %e, "lease keeper: failed to build its dedicated runtime");
                    return;
                }
            };
            rt.block_on(async move {
                let catalog = match connect().await {
                    Ok(c) => c,
                    Err(e) => {
                        error!(error = %e, "lease keeper: failed to open its own catalog connection");
                        return;
                    }
                };
                loop {
                    tokio::time::sleep(intervals.heartbeat()).await;
                    if thread_shutdown.load(Ordering::SeqCst) {
                        return;
                    }
                    renew_all(&catalog, &thread_registrations, intervals).await;
                }
            });
        });

        Arc::new(Self {
            registrations,
            next_id: AtomicU64::new(0),
            shutdown,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Register a new lease target for renewal on this keeper's thread.
    /// Renewal starts on the NEXT tick (at most one `heartbeat` interval
    /// away, never blocking the caller); stops when the returned
    /// [`Registration`] is dropped.
    pub fn register(&self, target: LeaseTarget) -> Registration {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let lost = Arc::new(AtomicBool::new(false));
        self.registrations.lock().unwrap().insert(
            id,
            RegistrationState {
                target,
                lost: Arc::clone(&lost),
            },
        );
        Registration {
            id,
            registrations: Arc::clone(&self.registrations),
            lost,
        }
    }
}

impl Drop for LeaseKeeper {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.lock().unwrap().take() {
            // Best-effort: the thread wakes at the next heartbeat tick and
            // exits. Not joined synchronously here — `Drop` runs on whatever
            // thread drops the last `Arc<LeaseKeeper>`, which may itself be
            // an async task that must not block on a `std::thread::join`.
            drop(handle);
        }
    }
}

/// One renewal pass over every current registration. A registration whose
/// renew misses (matches zero rows) has its `lost` flag set; every other
/// outcome (including a transient backend error, logged and retried next
/// tick) leaves it unset. Registrations added or removed mid-pass are not
/// raced against: the snapshot is taken once at the top of the pass, and a
/// [`Registration`]'s `Drop` removing it from the map does not affect a
/// renewal already in flight for it — the renewal simply writes to a row no
/// one observes the outcome of.
async fn renew_all(
    catalog: &Catalog,
    registrations: &Arc<Mutex<HashMap<u64, RegistrationState>>>,
    intervals: LeaseIntervals,
) {
    let snapshot: Vec<(LeaseTarget, Arc<AtomicBool>)> = {
        let regs = registrations.lock().unwrap();
        regs.values()
            .map(|r| (r.target.clone(), Arc::clone(&r.lost)))
            .collect()
    };
    for (target, lost) in snapshot {
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
        if renewed == Some(false) {
            lost.store(true, Ordering::SeqCst);
        }
    }
}
