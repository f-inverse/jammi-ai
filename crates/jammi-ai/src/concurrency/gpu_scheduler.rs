use std::hint;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jammi_db::error::{JammiError, Result};

/// Memory-budget GPU scheduler with priority levels.
///
/// `new_unlimited()` passes every permit — useful for tests and CPU-only
/// deployments. `new()` enforces memory-budget admission via CAS.
#[derive(Debug)]
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
        match Self::detect_gpu_memory(gpu_device as usize) {
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

    /// The total usable GPU budget in bytes — `Self::usable` made public,
    /// or `usize::MAX` for an unlimited scheduler. Lets a caller distinguish
    /// "this request can never be admitted, no matter how much frees up"
    /// (`bytes > usable_capacity()`) from "temporarily contended, and
    /// waiting for an outstanding release will eventually satisfy it"
    /// (`bytes <= usable_capacity()`, just not available right now). Used by
    /// `ModelCache::do_load`'s admission loop to decide between a hard error
    /// and continuing to wait on its own dual-notify admission loop (a
    /// permit-release notify plus a guard-idle notify, registered together
    /// before each `try_acquire`/`evict_one` pass) once `evict_one` has
    /// nothing left to evict — see `ModelCache::do_load`'s admission loop
    /// for the full wake-set enumeration and why `Self::acquire`'s
    /// single-notify wait is not enough on its own.
    pub fn usable_capacity(&self) -> usize {
        if self.unlimited {
            return usize::MAX;
        }
        self.usable()
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

/// One [`GpuScheduler`] per configured device.
///
/// A memory budget is a property of a DEVICE, not of a process: two ranks on
/// two cards have two budgets, and admitting against a single shared counter
/// would either over-admit on one card or starve the other. This is the map
/// from ordinal to that device's scheduler, plus the primary — the device a
/// caller that names none gets.
///
/// Built from the resolved `[gpu] devices` list, so an entry exists for every
/// device the deployment declared and for no other: [`Self::get`] returning
/// `None` means "this deployment never declared that device", which is a
/// typed refusal at the caller rather than a silently fabricated budget.
#[derive(Debug)]
pub struct DeviceSchedulers {
    primary: i32,
    by_device: Vec<(i32, Arc<GpuScheduler>)>,
}

impl DeviceSchedulers {
    /// A scheduler per entry of `devices`, each sized to that device by
    /// [`GpuScheduler::for_device`] (so a CPU ordinal, a CPU-only build, or a
    /// device that cannot be probed is an unlimited pass-through, exactly as
    /// for a single-device session). `devices[0]` is the primary.
    ///
    /// An empty list is refused: a session with no device has nowhere to
    /// place a model, and the alternative to refusing here is a `None` from
    /// every later lookup with no statement of why.
    pub fn for_devices(devices: &[i32], memory_fraction: f64) -> Result<Self> {
        let Some(&primary) = devices.first() else {
            return Err(JammiError::Config(
                "[gpu] devices resolved to an empty list: a session needs at least one device"
                    .into(),
            ));
        };
        let mut by_device: Vec<(i32, Arc<GpuScheduler>)> = Vec::with_capacity(devices.len());
        for &device in devices {
            if by_device.iter().any(|(d, _)| *d == device) {
                // Two entries for one ordinal would be two budgets over one
                // card's memory — each admitting as if it owned all of it.
                return Err(JammiError::Config(format!(
                    "[gpu] devices repeats device {device}: one budget per device"
                )));
            }
            by_device.push((
                device,
                Arc::new(GpuScheduler::for_device(device, memory_fraction)),
            ));
        }
        Ok(Self { primary, by_device })
    }

    /// Unlimited pass-through schedulers for `devices` — the CPU-only and
    /// test shape of [`Self::for_devices`].
    pub fn unlimited(devices: &[i32]) -> Result<Self> {
        let mut schedulers = Self::for_devices(devices, 1.0)?;
        schedulers.by_device = schedulers
            .by_device
            .into_iter()
            .map(|(d, _)| (d, Arc::new(GpuScheduler::new_unlimited())))
            .collect();
        Ok(schedulers)
    }

    /// A one-device set over `scheduler` — the single-device deployment,
    /// where the caller already owns the scheduler.
    pub fn single(device: i32, scheduler: Arc<GpuScheduler>) -> Self {
        Self {
            primary: device,
            by_device: vec![(device, scheduler)],
        }
    }

    /// The device a caller that names none gets: the first configured one.
    pub fn primary(&self) -> i32 {
        self.primary
    }

    /// Every configured device, in order.
    pub fn devices(&self) -> impl Iterator<Item = i32> + '_ {
        self.by_device.iter().map(|(d, _)| *d)
    }

    /// This device's scheduler, or `None` when the deployment never declared
    /// it.
    pub fn get(&self, device: i32) -> Option<&Arc<GpuScheduler>> {
        self.by_device
            .iter()
            .find(|(d, _)| *d == device)
            .map(|(_, s)| s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scheduler per declared device, the first one primary, and a device
    /// the deployment never declared has none — the lookup states the
    /// absence rather than handing back the primary's budget.
    #[test]
    fn device_schedulers_cover_exactly_the_declared_devices() {
        let schedulers = DeviceSchedulers::for_devices(&[0, 1, 2], 0.9).expect("schedulers");
        assert_eq!(schedulers.primary(), 0);
        assert_eq!(schedulers.devices().collect::<Vec<_>>(), vec![0, 1, 2]);
        for device in [0, 1, 2] {
            assert!(schedulers.get(device).is_some(), "device {device}");
        }
        assert!(
            schedulers.get(3).is_none(),
            "an undeclared device has no budget of its own"
        );
        // Two ranks on two devices hold two independent budgets: a
        // reservation on one is invisible to the other.
        let first = Arc::clone(schedulers.get(0).expect("device 0"));
        let second = Arc::clone(schedulers.get(1).expect("device 1"));
        assert!(!Arc::ptr_eq(&first, &second));
    }

    /// An empty list and a repeated ordinal are domain errors at
    /// construction, not conditions a later admission discovers.
    #[test]
    fn device_schedulers_refuse_an_empty_or_repeating_list() {
        DeviceSchedulers::for_devices(&[], 0.9).expect_err("a session needs a device");
        DeviceSchedulers::for_devices(&[0, 0], 0.9)
            .expect_err("two budgets over one card each admit as if they owned all of it");
    }

    /// Two devices' budgets are accounted separately: filling one admits
    /// nothing more on it and everything still fits on the other.
    #[test]
    fn a_reservation_on_one_device_does_not_consume_another_device_budget() {
        let schedulers = DeviceSchedulers {
            primary: 0,
            by_device: vec![
                (0, Arc::new(GpuScheduler::new(1_000, 0.0))),
                (1, Arc::new(GpuScheduler::new(1_000, 0.0))),
            ],
        };
        let first = Arc::clone(schedulers.get(0).expect("device 0"));
        let second = Arc::clone(schedulers.get(1).expect("device 1"));
        let _permit = first
            .try_acquire(1_000)
            .expect("device 0 admits its budget");
        assert!(first.try_acquire(1).is_none(), "device 0's budget is spent");
        assert!(
            second.try_acquire(1_000).is_some(),
            "device 1's budget is its own"
        );
    }

    /// A negative ordinal is CPU: no budget, admit everything.
    #[test]
    fn for_device_cpu_is_unlimited() {
        let sched = GpuScheduler::for_device(-1, 0.9);
        assert!(sched.unlimited);
        assert_eq!(sched.available(), usize::MAX);
    }

    /// When the device cannot be probed — which is always true in the hermetic
    /// CPU test build, where `detect_gpu_memory` reports no CUDA runtime —
    /// `for_device` degrades to unlimited rather than fabricating a budget.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn for_device_falls_back_to_unlimited_when_probe_fails() {
        let sched = GpuScheduler::for_device(0, 0.9);
        assert!(sched.unlimited);
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
