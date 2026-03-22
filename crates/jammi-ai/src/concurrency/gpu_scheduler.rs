use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// GPU memory scheduler. All GPU-sensitive components acquire permits
/// through this interface. Phase 03 provides `new_unlimited()` (always
/// grants permits immediately). Phase 09 adds `new()` with CAS-based
/// memory-budget admission.
pub struct GpuScheduler {
    total_gpu_memory: usize,
    reserved_memory: AtomicUsize,
    headroom_fraction: f64,
    pub(crate) notify: tokio::sync::Notify,
    unlimited: bool,
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
    /// Unlimited scheduler — always grants permits immediately.
    pub fn new_unlimited() -> Self {
        Self {
            total_gpu_memory: usize::MAX,
            reserved_memory: AtomicUsize::new(0),
            headroom_fraction: 0.0,
            notify: tokio::sync::Notify::new(),
            unlimited: true,
        }
    }

    /// Attempt to reserve `bytes` of GPU memory. Return `None` if insufficient.
    pub fn try_acquire(self: &Arc<Self>, bytes: usize) -> Option<GpuPermit> {
        if self.unlimited {
            return Some(GpuPermit {
                reserved_bytes: bytes,
                scheduler: Arc::clone(self),
            });
        }
        // CAS loop — Phase 09 provides the real implementation
        todo!()
    }

    /// Return the number of unreserved GPU bytes.
    pub fn available(&self) -> usize {
        if self.unlimited {
            return usize::MAX;
        }
        let usable = (self.total_gpu_memory as f64 * (1.0 - self.headroom_fraction)) as usize;
        usable.saturating_sub(self.reserved_memory.load(Ordering::Acquire))
    }
}
