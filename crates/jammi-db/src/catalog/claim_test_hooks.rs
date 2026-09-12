//! Test-only rendezvous inside [`super::Catalog::claim_next`]'s transaction
//! (`feature = "test-hooks"`; mirrors `crate::store::reconcile_test_hooks`):
//! a test arms [`ParkPoint::ClaimBeforeCommit`] for the instance whose claim
//! it wants to hold open, and the next claim that instance lands parks —
//! after the `UPDATE … RETURNING` ran, before the transaction's COMMIT —
//! until the test releases it. This is the ONE way to manufacture "a claim
//! is in flight while a RELEASE runs" deterministically: on SQLite the parked
//! transaction holds the database write lock (`BEGIN IMMEDIATE`), so every
//! write the RELEASE arm issues meanwhile waits out its own `busy_timeout`
//! and logs `database is locked` — the oracle tolerates those and asserts
//! only the row after the unpark. No production path observes anything here
//! beyond the `maybe_park` call, which returns at once when nothing is
//! armed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use tokio::sync::Notify;

/// Where the claim parks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParkPoint {
    /// Inside `claim_next`'s transaction closure, after the claiming
    /// `UPDATE … RETURNING` matched a row and before the closure returns
    /// (and the backend commits).
    ClaimBeforeCommit,
}

struct Armed {
    instance_id: String,
    point: ParkPoint,
    parked: Arc<AtomicBool>,
    parked_notify: Arc<Notify>,
    released: Arc<AtomicBool>,
    release_notify: Arc<Notify>,
}

fn armed() -> &'static Mutex<Vec<Armed>> {
    static ARMED: OnceLock<Mutex<Vec<Armed>>> = OnceLock::new();
    ARMED.get_or_init(|| Mutex::new(Vec::new()))
}

/// The test's side of one armed park: wait for the claim to arrive, then
/// let it continue. Dropping the handle without releasing leaves the claim
/// parked (and, on SQLite, the write lock held) — release it explicitly.
pub struct ParkHandle {
    parked: Arc<AtomicBool>,
    parked_notify: Arc<Notify>,
    released: Arc<AtomicBool>,
    release_notify: Arc<Notify>,
}

impl ParkHandle {
    /// Resolve once a claim has reached the park point.
    pub async fn wait_parked(&self) {
        while !self.parked.load(Ordering::SeqCst) {
            self.parked_notify.notified().await;
        }
    }

    /// Let the parked claim continue to COMMIT.
    pub fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release_notify.notify_one();
    }
}

/// Arm one park for the next claim `instance_id` lands at `point`.
/// One-shot: the park disarms as soon as a claim takes it. Keyed by the
/// claimant so sibling tests in one binary never take each other's arm.
pub fn arm(instance_id: &str, point: ParkPoint) -> ParkHandle {
    let parked = Arc::new(AtomicBool::new(false));
    let parked_notify = Arc::new(Notify::new());
    let released = Arc::new(AtomicBool::new(false));
    let release_notify = Arc::new(Notify::new());
    armed()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(Armed {
            instance_id: instance_id.to_string(),
            point,
            parked: Arc::clone(&parked),
            parked_notify: Arc::clone(&parked_notify),
            released: Arc::clone(&released),
            release_notify: Arc::clone(&release_notify),
        });
    ParkHandle {
        parked,
        parked_notify,
        released,
        release_notify,
    }
}

/// Park if `instance_id` has `point` armed; return at once otherwise.
pub async fn maybe_park(instance_id: &str, point: ParkPoint) {
    let taken = {
        let mut list = armed().lock().unwrap_or_else(PoisonError::into_inner);
        list.iter()
            .position(|a| a.instance_id == instance_id && a.point == point)
            .map(|i| list.remove(i))
    };
    let Some(armed) = taken else {
        return;
    };
    armed.parked.store(true, Ordering::SeqCst);
    armed.parked_notify.notify_one();
    while !armed.released.load(Ordering::SeqCst) {
        armed.release_notify.notified().await;
    }
}
