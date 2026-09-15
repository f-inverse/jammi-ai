//! Test-only failure injection for [`super::Catalog::upsert_worker`]
//! (`feature = "test-hooks"`; mirrors [`super::claim_test_hooks`]'s shape):
//! a test arms a one-shot failure for a specific `instance_id`, and the
//! NEXT [`super::Catalog::upsert_worker`] call for that instance returns a
//! typed error instead of writing the row — the ONE way to manufacture "the
//! first `workers` upsert a claim loop issues fails" deterministically,
//! without touching the real backend (dropping the `workers` table, killing
//! the connection, …), which would also break every OTHER catalog call in
//! the same test. No production path observes anything here beyond the
//! `take_armed` check at the top of `upsert_worker`, which is `false` (a
//! no-op) when nothing is armed.

use std::sync::{Mutex, OnceLock, PoisonError};

fn armed() -> &'static Mutex<Vec<String>> {
    static ARMED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    ARMED.get_or_init(|| Mutex::new(Vec::new()))
}

/// Arm ONE failing `upsert_worker` call for `instance_id`. One-shot: the
/// arm is consumed (and disarms) the moment `upsert_worker` for this
/// `instance_id` is next called, whether or not the test ever awaits
/// anything — sibling tests in one binary never take each other's arm.
pub fn arm_upsert_worker_failure(instance_id: &str) {
    armed()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(instance_id.to_string());
}

/// `upsert_worker`'s own check: `true` (and disarmed) iff a failure was
/// armed for `instance_id`.
pub(super) fn take_armed(instance_id: &str) -> bool {
    let mut guard = armed().lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(pos) = guard.iter().position(|id| id == instance_id) {
        guard.remove(pos);
        true
    } else {
        false
    }
}
