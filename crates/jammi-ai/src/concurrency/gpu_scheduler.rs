use std::hint;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jammi_db::error::{JammiError, Result};

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

    /// Query the device's memory via CUDA, returning `(free_bytes, total_bytes)`.
    ///
    /// Retains the device's primary context (shared, refcounted — candle later
    /// retains the same one) and reads `cuMemGetInfo`. `Err` on a CPU-only build
    /// or when the device cannot be reached, which the caller
    /// ([`Self::for_device`]) treats as "no budget, run unlimited".
    #[cfg(feature = "cuda")]
    pub fn detect_gpu_memory(device_id: usize) -> Result<(usize, usize)> {
        use candle_core::cuda::cudarc::driver::CudaContext;
        let ctx = CudaContext::new(device_id)
            .map_err(|e| JammiError::Gpu(format!("CUDA device {device_id} init failed: {e}")))?;
        ctx.mem_get_info().map_err(|e| {
            JammiError::Gpu(format!(
                "cuMemGetInfo failed on CUDA device {device_id}: {e}"
            ))
        })
    }

    /// CPU-only build: there is no CUDA runtime to probe.
    #[cfg(not(feature = "cuda"))]
    pub fn detect_gpu_memory(_device_id: usize) -> Result<(usize, usize)> {
        Err(JammiError::Gpu(
            "GPU memory detection not available (CPU-only build)".into(),
        ))
    }

    /// The scheduler appropriate for the configured device.
    ///
    /// On a CUDA device whose memory can be probed, this is a real memory-budget
    /// scheduler sized to the card: `memory_fraction` of total VRAM is usable and
    /// the remainder is headroom for activations / workspace the coarse
    /// per-model estimate does not count. Otherwise — a negative ordinal (CPU),
    /// a CPU-only build, or a device that could not be probed — it is an
    /// unlimited pass-through, since admission control without a real budget
    /// would gate nothing meaningfully.
    pub fn for_device(gpu_device: i32, memory_fraction: f64) -> Self {
        if gpu_device < 0 {
            return Self::new_unlimited();
        }
        Self::from_probe(
            gpu_device,
            memory_fraction,
            Self::detect_gpu_memory(gpu_device as usize),
        )
    }

    /// Build the scheduler from an already-obtained [`Self::detect_gpu_memory`]
    /// result, so a caller that needs to know the probe's own outcome (a test
    /// asserting `for_device` reflects it) probes exactly once. A second,
    /// separate `detect_gpu_memory` call to "check what the probe said" would
    /// be a TOCTOU: on a live pod the card's free memory — and thus whether
    /// the probe itself succeeds — can change between two calls.
    fn from_probe(gpu_device: i32, memory_fraction: f64, probe: Result<(usize, usize)>) -> Self {
        match probe {
            Ok((_free, total)) => {
                let headroom_fraction = (1.0 - memory_fraction).clamp(0.0, 1.0);
                tracing::info!(
                    gpu_device,
                    total_bytes = total,
                    memory_fraction,
                    "GPU memory admission enabled"
                );
                Self::new(total, headroom_fraction)
            }
            Err(e) => {
                // On a CUDA build a probe failure is a real anomaly worth a warn.
                // On a CPU-only build it is the normal case (the default ordinal
                // is 0 even with no GPU); the model-load path already reports the
                // CPU fallback, so staying quiet here avoids a redundant warn on
                // every session construction.
                #[cfg(feature = "cuda")]
                tracing::warn!(
                    gpu_device,
                    "GPU memory probe failed ({e}); GPU memory admission disabled (unlimited)"
                );
                #[cfg(not(feature = "cuda"))]
                let _ = &e;
                Self::new_unlimited()
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A negative ordinal is CPU: no budget, admit everything.
    #[test]
    fn for_device_cpu_is_unlimited() {
        let sched = GpuScheduler::for_device(-1, 0.9);
        assert!(sched.unlimited);
        assert_eq!(sched.available(), usize::MAX);
    }

    /// `for_device` must reflect whether the probe actually succeeds, not
    /// whether the `cuda` feature happens to be compiled in.
    ///
    /// Two arms, chosen by [`GpuScheduler::detect_gpu_memory`] itself (the
    /// exact probe `for_device` calls — probed exactly once here and fed into
    /// [`GpuScheduler::from_probe`], since a second, separate
    /// `detect_gpu_memory` call to decide which arm applies would be a
    /// TOCTOU on a live pod), not by `#[cfg(feature = "cuda")]`: there is no
    /// CI lane that builds `--features cuda` on a GPU-less box and runs this
    /// suite — the only `--features cuda` builds are the from-source release
    /// lanes (`cargo build --bin jammi-server`, never `cargo test`) and the
    /// GPU-prove lane, which always runs on a real rented A100. So a
    /// feature-gated test would never exercise its "device absent" branch in
    /// CI at all.
    /// - **No device** (hermetic CPU build, or a `cuda`-feature build with no
    ///   hardware/driver): the probe errs, so `for_device` must degrade to
    ///   unlimited rather than fabricating a budget.
    /// - **Device present** (a real GPU pod): the probe succeeds, so
    ///   `for_device` must enable real memory-budget admission, not the
    ///   unlimited pass-through.
    #[test]
    fn for_device_reflects_probe_presence() {
        let probe = GpuScheduler::detect_gpu_memory(0);
        let probe_ok = probe.is_ok();
        let sched = GpuScheduler::from_probe(0, 0.9, probe);
        if probe_ok {
            assert!(
                !sched.unlimited,
                "a real, probed GPU must enable memory-budget admission, not unlimited pass-through"
            );
        } else {
            assert!(sched.unlimited);
        }
    }

    /// The CPU-only-build stub is a fixed contract, not merely "whatever the
    /// `cuda` feature happens to leave uncalled": under
    /// `#[cfg(not(feature = "cuda"))]`, `detect_gpu_memory` must always
    /// report `Err`, for every `device_id` — there is no CUDA runtime to
    /// probe in this build, full stop. `for_device_reflects_probe_presence`
    /// above only checks that `for_device`'s `unlimited` flag tracks whether
    /// the probe succeeded or failed; it never checks the probe's own
    /// Err/Ok verdict independently, so a stub mutated to fabricate
    /// `Ok((0, 0))` slips through undetected there (confirmed by `cargo
    /// mutants --file gpu_scheduler.rs`, which missed exactly this).
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn detect_gpu_memory_cpu_only_build_always_errs() {
        assert!(GpuScheduler::detect_gpu_memory(0).is_err());
        assert!(GpuScheduler::detect_gpu_memory(7).is_err());
    }

    /// `try_acquire`'s admission check is a SUM against the budget
    /// (`current + bytes > usable`), not a product. With the round numbers
    /// `budget_reflects_memory_fraction` (below) uses, `+` and `*` happen to
    /// agree on every assertion there (`cargo mutants` confirmed `replace +
    /// with *` survives it), so this picks small values where they visibly
    /// disagree: after a 10-byte reservation, `10 + 50 = 60` fits a 100-byte
    /// budget but `10 * 50 = 500` would not.
    #[test]
    fn try_acquire_is_additive_not_multiplicative() {
        let sched = Arc::new(GpuScheduler::new(100, 0.0));
        let _first = sched.try_acquire(10).expect("10 <= 100 admits");
        assert!(
            sched.try_acquire(50).is_some(),
            "10 (already reserved) + 50 (requested) = 60 must fit a 100-byte budget"
        );
    }

    /// `memory_fraction` maps to usable budget: with a 1 GiB card and 0.9,
    /// 900 MiB is admittable and the rest is headroom.
    #[test]
    fn budget_reflects_memory_fraction() {
        let total = 1024 * 1024 * 1024;
        let sched = Arc::new(GpuScheduler::new(total, 1.0 - 0.9));
        let usable = sched.available();
        // 90% of the card, within rounding.
        assert!(
            (usable as f64 - total as f64 * 0.9).abs() < 2.0,
            "usable {usable} should be ~90% of {total}"
        );
        // A request under budget is admitted and reserves; over-budget is refused.
        let permit = sched.try_acquire(usable / 2).expect("half-budget admits");
        assert!(
            sched.try_acquire(usable).is_none(),
            "over remaining refused"
        );
        drop(permit);
        assert!(
            sched.try_acquire(usable).is_some(),
            "release frees the budget"
        );
    }
}
