use std::hint;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jammi_db::config::MemoryLimit;
use jammi_db::error::{JammiError, Result};

/// One device's admission: a memory budget for the models resident on it,
/// and a bound on the model forwards running on it at once.
///
/// `new_unlimited()` passes every memory permit — useful for tests and
/// CPU-only deployments. `new()` enforces a byte budget via CAS.
///
/// Forward admission belongs to the DEVICE, never to a plan node: every
/// `InferenceExec` whose model is resident here — every partition of one, two
/// plans running side by side, a task decoded from the wire into a process
/// already running others — admits its forwards through this one bound, so
/// the device runs no more at once than it can. A budgeted device (an
/// accelerator) admits one forward at a time; an unbudgeted one (the CPU)
/// admits the engine's CPU parallelism budget.
#[derive(Debug)]
pub struct GpuScheduler {
    budget: usize,
    reserved_memory: AtomicUsize,
    unlimited: bool,
    pub(crate) notify: tokio::sync::Notify,
    forward_slots: Arc<tokio::sync::Semaphore>,
}

/// RAII admission of one model forward on a device. Released on drop.
#[derive(Debug)]
pub struct ForwardPermit {
    _slot: tokio::sync::OwnedSemaphorePermit,
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

impl GpuPermit {
    /// The device this reservation is held on.
    pub(crate) fn device(&self) -> &Arc<GpuScheduler> {
        &self.scheduler
    }
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
    /// A device that admits models while their reservations fit `budget`
    /// bytes.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            reserved_memory: AtomicUsize::new(0),
            unlimited: false,
            notify: tokio::sync::Notify::new(),
            forward_slots: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    /// Unlimited pass-through — every memory permit and every forward is
    /// granted at once. It reads nothing from the host, so a test that uses it
    /// behaves the same on every machine.
    pub fn new_unlimited() -> Self {
        Self::unbudgeted(tokio::sync::Semaphore::MAX_PERMITS)
    }

    /// The CPU as a device: no memory budget, `threads` forwards at once —
    /// the engine's CPU parallelism budget (`[engine] execution_threads`).
    pub fn cpu(threads: NonZeroUsize) -> Self {
        Self::unbudgeted(threads.get())
    }

    /// A remote endpoint as a device: it holds no memory this process
    /// budgets, and takes `max_in_flight` forwards at once.
    pub fn endpoint(max_in_flight: NonZeroUsize) -> Self {
        Self::unbudgeted(max_in_flight.get())
    }

    /// No memory budget, `forward_slots` forwards at once.
    fn unbudgeted(forward_slots: usize) -> Self {
        Self {
            budget: usize::MAX,
            reserved_memory: AtomicUsize::new(0),
            unlimited: true,
            notify: tokio::sync::Notify::new(),
            forward_slots: Arc::new(tokio::sync::Semaphore::new(forward_slots)),
        }
    }

    /// Admit one model forward on this device, waiting for a slot. The permit
    /// is held for the forward call and released on drop.
    pub async fn admit_forward(&self) -> Result<ForwardPermit> {
        let slot = Arc::clone(&self.forward_slots)
            .acquire_owned()
            .await
            .map_err(|_| JammiError::Gpu("the device's forward admission is closed".into()))?;
        Ok(ForwardPermit { _slot: slot })
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
    /// scheduler sized to the card by `memory_limit` ([`Self::budget_for`]).
    /// Otherwise — a negative ordinal (CPU),
    /// a CPU-only build, or a device that could not be probed — it is an
    /// unlimited pass-through, since admission control without a real budget
    /// would gate nothing meaningfully.
    ///
    /// Forwards: a probed card admits one at a time. An ordinal that cannot be
    /// probed is a Metal device on a Metal build (one at a time), and otherwise
    /// a request the loader serves on the CPU (`cpu_threads` at a time).
    pub fn for_device(
        gpu_device: i32,
        memory_limit: MemoryLimit,
        cpu_threads: NonZeroUsize,
    ) -> Result<Self> {
        if gpu_device < 0 {
            return Ok(Self::cpu(cpu_threads));
        }
        match Self::detect_gpu_memory(gpu_device as usize) {
            Ok((_free, total)) => {
                let budget = Self::budget_for(gpu_device, total, memory_limit)?;
                tracing::info!(
                    gpu_device,
                    total_bytes = total,
                    budget_bytes = budget,
                    %memory_limit,
                    "GPU memory admission enabled"
                );
                Ok(Self::new(budget))
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
                Ok(if cfg!(feature = "metal") {
                    Self::unbudgeted(1)
                } else {
                    Self::cpu(cpu_threads)
                })
            }
        }
    }

    /// The residency budget `[gpu] memory_limit` allows on a device of
    /// `total` bytes: a share of the card, or an absolute size that fits it.
    /// An absolute size larger than the card is refused, naming the device —
    /// a budget the card cannot hold would admit models that then fail to
    /// allocate, which is the failure admission exists to prevent.
    pub fn budget_for(gpu_device: i32, total: usize, memory_limit: MemoryLimit) -> Result<usize> {
        let total_bytes = total as u64;
        let budget = memory_limit.resolve(|| Ok::<_, JammiError>(total_bytes))?;
        if budget > total_bytes {
            return Err(JammiError::Config(format!(
                "[gpu] memory_limit = \"{memory_limit}\" is more than device {gpu_device}'s \
                 {total_bytes} bytes: give an absolute limit that fits the card, or a share"
            )));
        }
        // `budget <= total`, and `total` came from a `usize`.
        Ok(budget as usize)
    }

    /// Unreserved GPU bytes currently available.
    pub fn available(&self) -> usize {
        if self.unlimited {
            return usize::MAX;
        }
        self.budget
            .saturating_sub(self.reserved_memory.load(Ordering::Acquire))
    }

    /// The device's budget in bytes, or `usize::MAX` for an unlimited
    /// scheduler. Lets a caller distinguish
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
        self.budget
    }

    /// A reservation of no memory — what a model that holds no memory on this
    /// device (a remote model, whose device is its endpoint) is admitted
    /// with. Always granted: it takes nothing from the budget and releases
    /// nothing when it drops.
    pub(crate) fn reserve_nothing(self: &Arc<Self>) -> GpuPermit {
        GpuPermit {
            reserved_bytes: 0,
            scheduler: Arc::clone(self),
        }
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
        let usable = self.budget;
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
    pub fn for_devices(
        devices: &[i32],
        memory_limit: MemoryLimit,
        cpu_threads: NonZeroUsize,
    ) -> Result<Self> {
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
                Arc::new(GpuScheduler::for_device(device, memory_limit, cpu_threads)?),
            ));
        }
        Ok(Self { primary, by_device })
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

    /// A budgeted device admits one forward at a time and re-admits as soon
    /// as the permit drops; the CPU admits exactly its configured budget.
    #[tokio::test]
    async fn forward_admission_is_sized_by_the_device() {
        let accelerator = GpuScheduler::new(1 << 30);
        let first = accelerator.admit_forward().await.expect("admitted");
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(50),
                accelerator.admit_forward()
            )
            .await
            .is_err(),
            "a second forward waits while the first is admitted"
        );
        drop(first);
        accelerator
            .admit_forward()
            .await
            .expect("admitted once the first forward released");

        let budget = NonZeroUsize::new(3).expect("non-zero");
        let cpu = GpuScheduler::cpu(budget);
        let mut held = Vec::new();
        for _ in 0..budget.get() {
            held.push(cpu.admit_forward().await.expect("admitted"));
        }
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), cpu.admit_forward())
                .await
                .is_err(),
            "the CPU admits exactly its budget of forwards"
        );
    }

    /// A scheduler per declared device, the first one primary, and a device
    /// the deployment never declared has none — the lookup states the
    /// absence rather than handing back the primary's budget.
    #[test]
    fn device_schedulers_cover_exactly_the_declared_devices() {
        let schedulers = DeviceSchedulers::for_devices(&[0, 1, 2], share(90), NonZeroUsize::MIN)
            .expect("schedulers");
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
        DeviceSchedulers::for_devices(&[], share(90), NonZeroUsize::MIN)
            .expect_err("a session needs a device");
        DeviceSchedulers::for_devices(&[0, 0], share(90), NonZeroUsize::MIN)
            .expect_err("two budgets over one card each admit as if they owned all of it");
    }

    /// Two devices' budgets are accounted separately: filling one admits
    /// nothing more on it and everything still fits on the other.
    #[test]
    fn a_reservation_on_one_device_does_not_consume_another_device_budget() {
        let schedulers = DeviceSchedulers {
            primary: 0,
            by_device: vec![
                (0, Arc::new(GpuScheduler::new(1_000))),
                (1, Arc::new(GpuScheduler::new(1_000))),
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
        let sched = GpuScheduler::for_device(-1, share(90), NonZeroUsize::MIN).expect("the CPU");
        assert!(sched.unlimited);
        assert_eq!(sched.available(), usize::MAX);
    }

    /// When the device cannot be probed — which is always true in the hermetic
    /// CPU test build, where `detect_gpu_memory` reports no CUDA runtime —
    /// `for_device` degrades to unlimited rather than fabricating a budget.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn for_device_falls_back_to_unlimited_when_probe_fails() {
        let sched = GpuScheduler::for_device(0, share(90), NonZeroUsize::MIN).expect("unprobed");
        assert!(sched.unlimited);
    }

    /// `[gpu] memory_limit` sizes the budget on a probed card: a share of
    /// it, or an absolute size that fits it, and admission holds to that
    /// budget exactly.
    #[test]
    fn memory_limit_sizes_the_device_budget() {
        let card = 1 << 30;
        assert_eq!(
            GpuScheduler::budget_for(0, card, share(90)).expect("a share fits"),
            966_367_641,
            "90% of 1 GiB, rounded down"
        );
        let absolute = "512MB".parse().expect("a limit");
        let budget = GpuScheduler::budget_for(0, card, absolute).expect("512 MiB fits 1 GiB");
        assert_eq!(budget, 512 << 20);

        let sched = Arc::new(GpuScheduler::new(budget));
        let permit = sched.try_acquire(budget / 2).expect("half-budget admits");
        assert!(
            sched.try_acquire(budget).is_none(),
            "over remaining refused"
        );
        drop(permit);
        assert!(
            sched.try_acquire(budget).is_some(),
            "release frees the budget"
        );
    }

    /// An absolute limit larger than the card is refused by device, never
    /// silently clipped to the card or admitted past it.
    #[test]
    fn an_absolute_limit_larger_than_the_card_is_refused() {
        let err = GpuScheduler::budget_for(3, 1 << 30, "2GB".parse().expect("a limit"))
            .expect_err("2 GiB does not fit a 1 GiB card");
        let msg = err.to_string();
        assert!(msg.contains("memory_limit"), "{msg}");
        assert!(msg.contains("device 3"), "{msg}");
    }

    /// A reservation of nothing is granted even on a spent budget, and gives
    /// nothing back when it drops.
    #[test]
    fn a_reservation_of_nothing_is_granted_on_a_spent_budget() {
        let sched = Arc::new(GpuScheduler::new(1_000));
        let full = sched.try_acquire(1_000).expect("the whole budget");
        let nothing = sched.reserve_nothing();
        assert_eq!(sched.available(), 0);
        drop(nothing);
        assert_eq!(sched.available(), 0, "dropping it released nothing");
        drop(full);
        assert_eq!(sched.available(), 1_000);
    }

    fn share(percent: u8) -> MemoryLimit {
        MemoryLimit::Share(jammi_db::config::Percent::new(percent).expect("a percentage"))
    }
}
