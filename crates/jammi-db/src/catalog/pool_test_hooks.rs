//! Test-only park on a catalog pool's connection RETURN (`feature =
//! "test-hooks"`; mirrors [`super::claim_test_hooks`]): a test arms a delay
//! on one backend's [`ReturnPark`], and every connection that backend's pool
//! takes back from a dropped `PoolConnection` spends that long inside
//! `sqlx`'s `after_release` callback — after the return task has read the
//! pool's closed flag, before its liveness ping and its push onto the idle
//! queue. That is the window `catalog::backend::close_pool_and_drain`'s
//! documentation names: one ping round trip wide in production, so the only
//! way to hold "a return is in flight while the pool closes" open long enough
//! to observe deterministically is to widen it here. Disarmed, the callback
//! returns at once; a build without the feature installs no callback.
//!
//! Per pool, never global: a server's session pool and its lease keeper's own
//! pool are separate backends, and a test arms exactly the one whose close it
//! measures — sibling tests in one binary never take each other's park.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The armed delay on one pool's returns. `0` is disarmed.
#[derive(Debug, Default)]
pub struct ReturnPark {
    delay_ms: AtomicU64,
}

impl ReturnPark {
    /// Every return this pool takes back from now on parks for `delay`
    /// before it reaches the idle queue.
    pub fn park_for(&self, delay: Duration) {
        self.delay_ms
            .store(delay.as_millis() as u64, Ordering::SeqCst);
    }

    /// Returns pass straight through again.
    pub fn clear(&self) {
        self.delay_ms.store(0, Ordering::SeqCst);
    }

    fn delay(&self) -> Option<Duration> {
        match self.delay_ms.load(Ordering::SeqCst) {
            0 => None,
            ms => Some(Duration::from_millis(ms)),
        }
    }
}

/// Install the park as `options`' `after_release` callback. Called by both
/// backends' pool builders under the feature.
pub(crate) fn install<DB: sqlx::Database>(
    options: sqlx::pool::PoolOptions<DB>,
    park: &Arc<ReturnPark>,
) -> sqlx::pool::PoolOptions<DB> {
    let park = Arc::clone(park);
    options.after_release(move |_conn, _meta| {
        let park = Arc::clone(&park);
        Box::pin(async move {
            if let Some(delay) = park.delay() {
                tokio::time::sleep(delay).await;
            }
            Ok(true)
        })
    })
}
