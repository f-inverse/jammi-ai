use std::hint;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jammi_engine::error::{JammiError, Result};

/// Memory-budget GPU scheduler with priority levels.
///
/// `new_unlimited()` passes every permit — useful for tests and CPU-only
/// deployments. `new()` enforces memory-budget admission via CAS.
pub struct GpuScheduler {
    total_gpu_memory: usize,
    reserved_memory: AtomicUsize,
    headroom_fraction: f64,
    unlimited: bool,
    pub(crate) notify: tokio::sync::Notify,
}

/// Priority level for GPU work. Higher values wait longer under contention.
///
/// In v1, priority is a label carried on the API for forward compatibility.
/// `acquire()` wakes waiters in arbitrary order. Real priority scheduling
/// (priority queue of oneshot senders) is a v2 concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuPriority {
    /// User-facing queries (search, infer) — lowest latency tolerance.
    Interactive = 0,
    /// Eval, batch embedding generation — can tolerate short waits.
    Background = 1,
    /// Fine-tuning — long-running, can wait for memory.
    Training = 2,
}

/// RAII memory reservation. Released on drop.
pub struct GpuPermit {
    reserved_bytes: usize,
    scheduler: Arc<GpuScheduler>,
}

impl Drop for GpuPermit {
    fn drop(&mut self) {
        if !self.scheduler.unlimited {
            self.scheduler
                .reserved_memory
                .fetch_sub(self.reserved_bytes, Ordering::Release);
            self.scheduler.notify.notify_waiters();
        }
    }
}

impl GpuScheduler {
    /// Memory-budget constructor. Validates headroom_fraction is in [0.0, 1.0].
    pub fn new(total_gpu_memory: usize, headroom_fraction: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&headroom_fraction),
            "headroom_fraction must be between 0.0 and 1.0, got {headroom_fraction}"
        );
        Self {
            total_gpu_memory,
            reserved_memory: AtomicUsize::new(0),
            headroom_fraction,
            unlimited: false,
            notify: tokio::sync::Notify::new(),
        }
    }

    /// Unlimited pass-through — always grants permits immediately.
    /// Retained for tests and CPU-only deployments.
    pub fn new_unlimited() -> Self {
        Self {
            total_gpu_memory: usize::MAX,
            reserved_memory: AtomicUsize::new(0),
            headroom_fraction: 0.0,
            unlimited: true,
            notify: tokio::sync::Notify::new(),
        }
    }

    /// Query available GPU memory via CUDA.
    /// Returns `(free_bytes, total_bytes)`. Returns `Err` on CPU-only machines.
    pub fn detect_gpu_memory(_device_id: usize) -> Result<(usize, usize)> {
        // Real CUDA detection requires the cudarc crate, which is only useful
        // on GPU machines. Return Err on CPU-only builds. When CUDA support is
        // needed, gate behind a `cuda` feature and use cudarc::driver.
        Err(JammiError::Gpu(
            "GPU memory detection not available (no CUDA runtime)".into(),
        ))
    }

    /// Usable GPU memory after headroom reservation.
    fn usable(&self) -> usize {
        (self.total_gpu_memory as f64 * (1.0 - self.headroom_fraction)) as usize
    }

    /// Unreserved GPU bytes currently available.
    pub fn available(&self) -> usize {
        if self.unlimited {
            return usize::MAX;
        }
        self.usable()
            .saturating_sub(self.reserved_memory.load(Ordering::Acquire))
    }

    /// Non-blocking acquisition attempt. Returns `None` if insufficient memory.
    ///
    /// CAS loop with `spin_loop()` hint on contention — the retry window is
    /// a single atomic compare-exchange, so spinning is cheaper than yielding.
    pub fn try_acquire(self: &Arc<Self>, bytes: usize) -> Option<GpuPermit> {
        if self.unlimited {
            return Some(GpuPermit {
                reserved_bytes: bytes,
                scheduler: Arc::clone(self),
            });
        }
        let usable = self.usable();
        loop {
            let current = self.reserved_memory.load(Ordering::Acquire);
            if current + bytes > usable {
                return None;
            }
            match self.reserved_memory.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(GpuPermit {
                        reserved_bytes: bytes,
                        scheduler: Arc::clone(self),
                    });
                }
                Err(_) => {
                    hint::spin_loop();
                }
            }
        }
    }

    /// Acquire GPU memory asynchronously. Blocks until `estimated_bytes` fits.
    ///
    /// Uses `tokio::sync::Notify` for async waiting. The `Notified` future is
    /// registered via `enable()` BEFORE the `try_acquire` check to prevent
    /// lost-wakeup races.
    pub async fn acquire(
        self: &Arc<Self>,
        estimated_bytes: usize,
        _priority: GpuPriority,
    ) -> Result<GpuPermit> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            if let Some(permit) = self.try_acquire(estimated_bytes) {
                return Ok(permit);
            }

            notified.await;
        }
    }
}
