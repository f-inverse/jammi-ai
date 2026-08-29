use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::error::{JammiError, Result};
use tokio::sync::RwLock;

use super::backend::candle::CandleBackend;
use super::backend::ort::OrtBackend;
use super::backend::{DeviceConfig, ModelBackend};
use super::resolver::ModelResolver;
use super::{BackendType, LoadedModel, ModelGuard, ModelId, ModelSource, ModelTask};
use crate::concurrency::{GpuPermit, GpuScheduler};

/// Where a cached model currently resides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelResidency {
    /// Weights are loaded on GPU memory.
    Gpu,
    /// Weights are loaded in CPU memory.
    Cpu,
    /// Model has been evicted and is no longer in memory.
    Unloaded,
}

struct CacheEntry {
    model: Arc<LoadedModel>,
    ref_count: Arc<AtomicUsize>,
    memory_bytes: usize,
    _residency: ModelResidency,
    /// Audit round 62, F-3: shared with every outstanding `ModelGuard` handed
    /// out for this entry (see `ModelGuard::gpu_permit`'s doc). Removing this
    /// `CacheEntry` from the cache (stale-fingerprint eviction, `evict_one`)
    /// drops only THIS clone — the reservation is released by `GpuPermit`'s
    /// `Drop` only once every clone (this one plus any live guard's) is gone,
    /// so `reserved_memory` is never decremented while a guard still holds
    /// this model's device tensors resident across a forward pass.
    gpu_permit: Arc<GpuPermit>,
}

struct CacheInner {
    entries: HashMap<ModelId, CacheEntry>,
    lru_order: VecDeque<ModelId>,
    in_flight: HashMap<ModelId, Arc<tokio::sync::Notify>>,
}

struct Backends {
    candle: CandleBackend,
    ort: OrtBackend,
}

/// Audit round 62, F-3': test-only deterministic interleaving seam for
/// `get_or_load`'s fast path. A unit test installs one of these (via
/// [`ModelCache::install_probe_pause`]) to pause a warm-hit task at the
/// EXACT window F-3' exploited — between releasing the snapshot's READ lock
/// and calling `probe_freshness`, i.e. before `ref_count` is incremented and
/// before the `gpu_permit` clone (deferred to the re-validate branch by the
/// fix) — so a concurrent, budget-pressure-triggered `evict_one` can be
/// driven deterministically into that window without a `sleep`-based race.
/// Every pre-existing cache test either sizes the GPU budget to never evict,
/// or uses [`GpuScheduler::new_unlimited`], so none of them ever interleaved
/// a real `evict_one` call with a fast-path probe in flight — this seam
/// closes that coverage hole. `None` (production, and every test that never
/// installs it) is a complete no-op.
#[cfg(test)]
pub(crate) struct ProbePauseHandle {
    /// Signalled once the paused task has reached the pause point, so the
    /// test knows it is now safe to drive a concurrent `evict_one`.
    pub(crate) arrived: Arc<tokio::sync::Notify>,
    /// The test calls `.notify_one()` on this to resume the paused task.
    pub(crate) release: Arc<tokio::sync::Notify>,
}

/// LRU cache of loaded models with GPU memory tracking and single-flight loading.
pub struct ModelCache {
    inner: Arc<RwLock<CacheInner>>,
    resolver: ModelResolver,
    backends: Backends,
    device_config: DeviceConfig,
    gpu_scheduler: Arc<GpuScheduler>,
    /// Unit 62, closure-audit BLOCK 1 (admission-wake liveness hole): the
    /// cache-level admission wake source. See [`ModelGuard`]'s
    /// `admission_notify` field doc for why `GpuScheduler`'s own release
    /// notify is not sufficient on its own, and `do_load`'s admission loop
    /// for the full wake-set enumeration this notify is one half of.
    admission_notify: Arc<tokio::sync::Notify>,
    /// See [`ProbePauseHandle`]. Consumed (taken) the first time
    /// `get_or_load`'s fast path reaches the pause point, so it only ever
    /// pauses once per installation.
    #[cfg(test)]
    probe_pause: std::sync::Mutex<Option<ProbePauseHandle>>,
    /// The single-flight-wait peer of `probe_pause` (audit round 62
    /// advisory) — same [`ProbePauseHandle`] shape, a different pause point
    /// (`get_or_load`'s single-flight wait branch, after `enable()`/`drop`,
    /// before `.await`).
    #[cfg(test)]
    single_flight_pause: std::sync::Mutex<Option<ProbePauseHandle>>,
}

impl ModelCache {
    /// Create a cache backed by the given resolver, device config, and GPU scheduler.
    pub fn new(
        resolver: ModelResolver,
        device_config: DeviceConfig,
        gpu_scheduler: Arc<GpuScheduler>,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CacheInner {
                entries: HashMap::new(),
                lru_order: VecDeque::new(),
                in_flight: HashMap::new(),
            })),
            resolver,
            backends: Backends {
                candle: CandleBackend,
                ort: OrtBackend,
            },
            device_config,
            gpu_scheduler,
            admission_notify: Arc::new(tokio::sync::Notify::new()),
            #[cfg(test)]
            probe_pause: std::sync::Mutex::new(None),
            #[cfg(test)]
            single_flight_pause: std::sync::Mutex::new(None),
        }
    }

    /// Audit round 62, F-3' test seam: install a fresh [`ProbePauseHandle`]
    /// pair on this cache and return the caller's half. See
    /// [`ProbePauseHandle`]'s doc for exactly what it pauses.
    #[cfg(test)]
    pub(crate) fn install_probe_pause(&self) -> ProbePauseHandle {
        let arrived = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        *self.probe_pause.lock().unwrap() = Some(ProbePauseHandle {
            arrived: Arc::clone(&arrived),
            release: Arc::clone(&release),
        });
        ProbePauseHandle { arrived, release }
    }

    /// Take (consume) the installed pause, if any, signal the test that this
    /// task has arrived, and block until the test releases it. A no-op
    /// (including in every production build, where this method does not
    /// even exist) once already consumed or never installed.
    #[cfg(test)]
    async fn pause_before_probe_for_test(&self) {
        let handle = self.probe_pause.lock().unwrap().take();
        if let Some(handle) = handle {
            handle.arrived.notify_one();
            handle.release.notified().await;
        }
    }

    /// Audit round 62 advisory test seam: the single-flight-wait peer of
    /// [`ModelCache::install_probe_pause`].
    #[cfg(test)]
    pub(crate) fn install_single_flight_pause(&self) -> ProbePauseHandle {
        let arrived = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        *self.single_flight_pause.lock().unwrap() = Some(ProbePauseHandle {
            arrived: Arc::clone(&arrived),
            release: Arc::clone(&release),
        });
        ProbePauseHandle { arrived, release }
    }

    #[cfg(test)]
    async fn pause_before_single_flight_wait_for_test(&self) {
        let handle = self.single_flight_pause.lock().unwrap().take();
        if let Some(handle) = handle {
            handle.arrived.notify_one();
            handle.release.notified().await;
        }
    }

    /// Remove `id`'s `CacheEntry` if — and only if — it still holds the SAME
    /// `Arc<LoadedModel>` this call's snapshot probed (`Arc::ptr_eq`); a
    /// concurrent task may have already evicted/reloaded it, in which case
    /// this is a no-op and whatever is there now is left alone. Shared by
    /// `get_or_load`'s `Ok(false)` (stale) and `Err` (unit-62 design
    /// pressure-test, item 3: wedge elimination) arms — the same removal
    /// discipline applies to both: serving stale bytes, or re-probing a
    /// permanently dead entry forever, are both correctness bugs the
    /// idle-only `evict_one` (memory-pressure eviction) does not address.
    async fn evict_if_current(&self, id: &ModelId, model: &Arc<LoadedModel>) {
        let mut cache = self.inner.write().await;
        if cache
            .entries
            .get(id)
            .is_some_and(|e| Arc::ptr_eq(&e.model, model))
        {
            cache.entries.remove(id);
            cache.lru_order.retain(|x| x != id);
        }
    }

    /// Get or load a model. Returns a guard that keeps the model alive.
    ///
    /// **Staleness contract (unit-62 design pressure-test, PINNED — see
    /// `backend::candle::ModelFingerprint`'s doc for the full accounting):
    /// this cache's warm-hit staleness detection (esc-058) is NARROW.** It
    /// re-`stat`s the FILES the resolver selected at load time and reloads
    /// on in-place mutation, deletion, or appearance among them; it does
    /// NOT re-verify catalog `artifact_path`/`backend` rewrites (a
    /// fine-tuned retrain's new adapter goes unnoticed by a warm entry until
    /// process restart), catalog-vs-local precedence, `task`/`backend_hint`
    /// cache keying (entries are keyed by `ModelId` alone), HF revision
    /// moves, or remote sibling listings — those are unit 65's scope
    /// (`docs/plans/65-resolve-witness`). The guarantee this DOES provide is
    /// BOUNDED STALENESS, never per-hit freshness: the returned
    /// [`ModelGuard`] was fresh at some instant before this call began, but
    /// is never revalidated again — a TOCTOU window between that instant and
    /// the guard's actual use is inherent, not a defect.
    pub async fn get_or_load(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ModelGuard> {
        let id = ModelId::from(source);

        loop {
            // Fast-path snapshot: clone the currently cached entry's shared
            // handles under a short-held READ lock (tokio's `RwLock` admits
            // concurrent readers), then drop the lock BEFORE the esc-058
            // staleness probe below. `probe_freshness` runs one blocking
            // `stat` per fingerprinted candidate (config, weights, tokenizer,
            // pooling, preprocessor, adapter pair) — audit round 62 advisory:
            // running that sequence under the cache's single write lock (as
            // the pre-fix code did) would block every OTHER model's
            // concurrent `get_or_load` for the duration. The snapshot +
            // `Arc::ptr_eq` re-validate pattern below accepts a narrow race
            // instead: if another task evicts/reloads this id while we
            // probe, we simply retry from the top against whatever is there
            // now — single-flight below still ensures at most one loader per
            // id.
            // Audit round 62, F-3': the probe snapshot carries ONLY the
            // `Arc<LoadedModel>` + `ref_count` handle — never a
            // `gpu_permit` clone. `evict_one` (below) treats `ref_count ==
            // 0` as "idle, safe to evict, and my caller may count the
            // eviction as real progress toward its memory budget" — but a
            // permit clone made HERE, before `ref_count` is incremented,
            // would be an outstanding `Arc<GpuPermit>` that `evict_one`
            // cannot see: it removes the `CacheEntry` (dropping only ITS
            // clone), returns `true` (progress), yet the reservation is
            // NOT actually released because this snapshot's clone is still
            // live — `do_load`'s admission loop trusts that `true`
            // unconditionally and never retries, so the next
            // `try_acquire` fails again with the memory still
            // double-booked, cascading into evicting a second model or a
            // hard "nothing to evict" error on a load that would otherwise
            // have succeeded. Deferring the permit clone to the
            // re-validate branch below — which already holds the WRITE
            // lock and increments `ref_count` in the same critical section
            // — makes the two atomic: `evict_one` (also write-lock-gated)
            // never observes `ref_count == 0` with an outstanding permit
            // clone made by THIS acquire-side snapshot path.
            //
            // That closes only the acquire side (F-3'). Audit round 62,
            // F-A: the release side had a matching hole — `ModelGuard`'s
            // `Drop` used to decrement `ref_count` in its body while its own
            // `_gpu_permit` clone dropped only afterward (Rust drops struct
            // fields after the `Drop` impl body returns), so `ref_count ==
            // 0` could become visible to a concurrent `evict_one` while that
            // guard's permit clone was still alive. `ModelGuard::drop` now
            // releases its permit clone (via `Option::take`) BEFORE the
            // `fetch_sub`, but `evict_one` no longer trusts `ref_count == 0`
            // as sufficient PROOF of "no permit clone outstanding" either
            // way — it is NOT vacuously true just because both known races
            // are closed; a future caller could reintroduce a third one.
            // `evict_one` instead checks `Arc::strong_count(&entry.gpu_permit)
            // == 1` directly at removal time, which is the actual quantity
            // that determines whether dropping the entry releases the
            // reservation. That check, not the ordering above, is the real
            // guarantee behind `evict_one`'s `true`.
            let snapshot = {
                let cache = self.inner.read().await;
                cache
                    .entries
                    .get(&id)
                    .map(|entry| (Arc::clone(&entry.model), Arc::clone(&entry.ref_count)))
            };

            if let Some((model, ref_count)) = snapshot {
                // F-3' test seam: a no-op unless a test installed a pause
                // (`ProbePauseHandle`'s doc) — pauses exactly HERE, after
                // the snapshot but before the probe, which is the window a
                // concurrent `evict_one` needs to interleave into to
                // reproduce F-3's race.
                #[cfg(test)]
                self.pause_before_probe_for_test().await;

                match model.probe_freshness() {
                    Ok(true) => {
                        let mut cache = self.inner.write().await;
                        let still_current = cache
                            .entries
                            .get(&id)
                            .is_some_and(|e| Arc::ptr_eq(&e.model, &model));
                        if still_current {
                            // Atomic with the clone below: `evict_one`
                            // cannot interleave between these two lines
                            // (both require this same write lock), so it
                            // never sees `ref_count == 0` while this clone
                            // is outstanding (F-3').
                            ref_count.fetch_add(1, Ordering::Acquire);
                            let gpu_permit = Arc::clone(
                                &cache
                                    .entries
                                    .get(&id)
                                    .expect("just matched still_current above")
                                    .gpu_permit,
                            );
                            cache.touch_lru(&id);
                            return Ok(ModelGuard::new(
                                model,
                                ref_count,
                                gpu_permit,
                                Arc::clone(&self.admission_notify),
                            ));
                        }
                        // The entry changed under us (a concurrent
                        // evict/reload won the race) — retry against the
                        // current state.
                        continue;
                    }
                    Ok(false) => {
                        // Stale: at least one fingerprinted candidate's
                        // (len, mtime) diverged from load time, or a
                        // candidate absent at load time now exists (F-4b).
                        // Evict — but only the SAME entry we probed; if it
                        // already changed (another task raced us), leave
                        // whatever is there now alone and retry.
                        //
                        // Deliberately unconditional on `ref_count`
                        // (unlike `evict_one`, which only evicts an idle
                        // `ref_count == 0` entry for memory-pressure
                        // reasons): serving stale bytes is a correctness
                        // bug, not a capacity one, so it overrides the
                        // idle-only discipline. Removing the `CacheEntry`
                        // drops only ITS `Arc<GpuPermit>` clone (F-3) — the
                        // reservation is not released while any live
                        // `ModelGuard`'s clone (e.g. one still forwarding
                        // through the pre-mutation model) is outstanding,
                        // so the accounting never double-books the
                        // pre-mutation model's still-resident memory. This
                        // snapshot itself never held a permit clone (F-3'
                        // above): it never incremented `ref_count`, so
                        // there is nothing of ITS OWN to release here.
                        self.evict_if_current(&id, &model).await;
                        // Fall through to single-flight/load below, which
                        // re-resolves and re-hashes the CURRENT bytes.
                    }
                    Err(e) => {
                        // Unit-62 design pressure-test, item 3 (wedge
                        // elimination): a fingerprinted candidate vanished
                        // or became unreadable between load and this probe —
                        // still a typed refusal (K2), never a silent "treat
                        // as fresh". But leaving the stale `CacheEntry`
                        // cached here (the pre-fix behavior) meant every
                        // LATER call re-probed the identical dead entry and
                        // re-`Err`ed forever — a permanent wedge, even for a
                        // cause that was only transient (the catalog-shadow
                        // fallthrough case unit 65 scopes) or that a
                        // subsequent cold resolve would in fact route around
                        // via an alternate the probe's OWN slot didn't see
                        // fail. Evict here too, exactly like the `Ok(false)`
                        // arm above, so the entry is gone before we return:
                        // the NEXT call takes the full cold path instead of
                        // the identical stale entry. Under the narrow
                        // staleness contract (see `ModelFingerprint`'s doc)
                        // this is cold-equivalence, not a silent recovery —
                        // if the cause is still present, the next call hits
                        // the loader's own typed error (a different message
                        // than this probe's, but the identical OBSERVABLE
                        // outcome: refusal), never a wedge; if the cause was
                        // transient, the system self-heals instead.
                        self.evict_if_current(&id, &model).await;
                        return Err(e);
                    }
                }
            }

            let mut cache = self.inner.write().await;
            // Re-check under the write lock: another task may have
            // inserted this id (via `do_load` below) between our snapshot
            // (which saw no entry, or a now-evicted one) and here.
            if cache.entries.contains_key(&id) {
                drop(cache);
                continue;
            }

            // Single-flight: wait if another task is loading this model.
            //
            // Audit round 62 advisory: `Notify::notify_waiters` only wakes
            // futures that are ALREADY registered as waiting at the moment
            // it is called — it does not persist a wakeup for a `Notified`
            // future created afterward. The pre-fix code created the
            // `Notified` future (`notify.notified()`) only AFTER dropping
            // this write lock; if the loader task finished, took the write
            // lock, removed itself from `in_flight`, dropped ITS lock, and
            // called `notify_waiters()` in that gap — before this task's
            // `.await` below ever polled — the wakeup was lost forever
            // (there is no timeout, so the waiter would hang until some
            // UNRELATED future load of the same id happened to call
            // `notify_waiters` again). Following the identical idiom
            // `GpuScheduler::acquire` already uses (see
            // `concurrency/gpu_scheduler.rs`): build the `Notified` future
            // and `enable()` it — which registers the waiter synchronously,
            // no `.await` needed — WHILE STILL HOLDING this write lock.
            // The loader task cannot acquire this same write lock (needed
            // to remove itself from `in_flight` before it may call
            // `notify_waiters`) until this task has dropped it below, so
            // registration always happens-before any `notify_waiters` call
            // that could apply to this wait — the lost-wakeup window is
            // closed structurally, not by a timeout.
            if let Some(notify) = cache.in_flight.get(&id) {
                let notify = Arc::clone(notify);
                let notified = notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                drop(cache);
                // Advisory test seam: a no-op unless a test installed a
                // pause. Placed AFTER `enable()` (the registration point,
                // now already unconditionally reached before `drop(cache)`
                // above) so this only ever exercises whether an already-
                // REGISTERED waiter still wakes when the loader completes
                // and calls `notify_waiters` during the pause — never a
                // reproduction of the pre-fix gap itself (which no longer
                // structurally exists in this function's control flow).
                #[cfg(test)]
                self.pause_before_single_flight_wait_for_test().await;
                notified.await;
                continue;
            }

            // We are the loader
            let notify = Arc::new(tokio::sync::Notify::new());
            cache.in_flight.insert(id.clone(), Arc::clone(&notify));
            drop(cache);

            let result = self.do_load(&id, source, task, backend_hint).await;

            let mut cache = self.inner.write().await;
            cache.in_flight.remove(&id);

            match result {
                Ok(guard) => {
                    drop(cache);
                    notify.notify_waiters();
                    return Ok(guard);
                }
                Err(e) => {
                    drop(cache);
                    notify.notify_waiters();
                    return Err(e);
                }
            }
        }
    }

    /// TEST-ONLY: resolve and load a fresh, UNSHARED [`LoadedModel`] off the
    /// resolver + backend, bypassing the shared LRU cache entirely. The shared
    /// cache hands out `Arc<LoadedModel>` (no `&mut`), so a test that needs to
    /// mutate a model — e.g. the regression non-vacuity guard zeroing the trained
    /// distribution head via
    /// [`LoadedModel::zero_distribution_head_for_test`] — must own it. This goes
    /// through the same resolve + `backend.load` path serving uses, so the owned
    /// model is byte-identical to what `get_or_load` would cache. Not used by any
    /// production path.
    #[doc(hidden)]
    pub async fn load_owned_for_test(
        &self,
        source: &ModelSource,
        task: ModelTask,
    ) -> Result<LoadedModel> {
        let resolved = self.resolver.resolve(source, task, None).await?;
        let backend: &dyn ModelBackend = match resolved.backend {
            BackendType::Candle => &self.backends.candle,
            BackendType::Ort => &self.backends.ort,
            other => {
                return Err(JammiError::Model {
                    model_id: source.to_string(),
                    message: format!("Backend {other:?} not available"),
                })
            }
        };
        backend.load(&resolved, &self.device_config)
    }

    async fn do_load(
        &self,
        id: &ModelId,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ModelGuard> {
        let resolved = self.resolver.resolve(source, task, backend_hint).await?;
        let source_str = source.to_string();
        let backend: &dyn ModelBackend = match resolved.backend {
            BackendType::Candle => &self.backends.candle,
            BackendType::Ort => &self.backends.ort,
            other => {
                return Err(JammiError::Model {
                    model_id: source_str,
                    message: format!("Backend {other:?} not available"),
                })
            }
        };
        let memory_bytes = backend.estimate_memory(&resolved);

        // Unit-62 design pressure-test, item 2 (block): a stale-fingerprint
        // reload can transiently need this model's budget TWICE — the
        // caller in `get_or_load`'s `Ok(false)` arm already removed the
        // stale `CacheEntry` from `cache.entries` (so `evict_one` here can
        // never find it again), but its `Arc<GpuPermit>` clone stays
        // outstanding for as long as ANY live `ModelGuard` from before the
        // mutation is still held — the reservation is not released until
        // that guard drops (F-3's Arc-shared-permit accounting, unchanged).
        // Under a budget realistically sized to one resident copy of this
        // model, `evict_one` therefore finds nothing evictable even though
        // the request is perfectly satisfiable — just not yet. Distinguish
        // that from a genuinely unsatisfiable request (more bytes than the
        // scheduler could EVER admit, evictions or waiting or not) via
        // `GpuScheduler::usable_capacity` — the one case a hard error is
        // still honest, since no amount of waiting would ever succeed.
        //
        // Unit 62, closure-audit BLOCK 1 (admission-wake liveness hole,
        // FIXES the doc this replaces — the old text here claimed
        // `GpuScheduler::acquire`'s wait "wakes on ANY permit release (or a
        // later `evict_one` finding a newly-idle entry)"; the second half was
        // FALSE. `GpuScheduler::acquire` only ever waits on
        // `GpuScheduler`'s own release notify — it has no way to observe
        // `evict_one`'s eligibility condition at all, let alone wake because
        // of it. Consider: budget sized to one resident copy; A holds M1's
        // guard; B's `do_load` (loading M2) fails `try_acquire`, finds M1
        // NOT evictable (`ref_count == 1`), and would fall back to
        // `GpuScheduler::acquire`'s wait. A then drops its guard: per
        // `ModelGuard::drop`'s ordering, the permit clone releases BEFORE
        // the `ref_count` decrement — but M1's `CacheEntry` is still present
        // in the cache and retains its OWN clone of the SAME `Arc<GpuPermit>`
        // (F-3's sharing discipline), so dropping A's clone only lowers the
        // `Arc`'s strong count from 2 to 1 — it does NOT reach zero, so
        // `GpuPermit::drop`'s body (and its `notify_waiters()` call) never
        // runs. `ref_count` reaching zero — the transition that makes M1
        // newly eligible for `evict_one` — is therefore INVISIBLE to
        // `GpuScheduler`'s notify. A waiter parked purely on
        // `GpuScheduler::acquire` would hang forever, holding
        // `cache.in_flight[M2]`, wedging every later M2 caller behind it
        // (the single-flight branch never runs `evict_one`, so nothing ever
        // rescues it).
        //
        // Wake-set enumeration (the two, and only two, transitions that can
        // ever make a previously-failed admission attempt newly succeed —
        // this loop's `select!` below covers BOTH, and re-runs the ENTIRE
        // admission sequence — `try_acquire`, `evict_one`, the unsatisfiable
        // check — from the top on either):
        //
        //   1. A `GpuPermit`'s LAST `Arc` clone drops for real, directly
        //      decrementing `GpuScheduler::reserved_memory` and calling
        //      `scheduler.notify.notify_waiters()` (`GpuPermit::drop`, the
        //      sole call site). This covers every already-cache-evicted
        //      entry's outgoing guard being the final clone (the
        //      `evict_if_current`/stale-reload case — the pre-existing,
        //      still-green wait test) and `evict_one`'s own removal of an
        //      idle entry (also a last-clone drop, since `evict_one` only
        //      ever removes an entry whose `strong_count == 1`).
        //   2. Any `ModelGuard::drop`, unconditionally — signals
        //      `ModelCache::admission_notify` AFTER its `ref_count`
        //      decrement (see `ModelGuard`'s `admission_notify` field doc).
        //      This covers the hole above: a STILL-CACHED entry's
        //      `ref_count` reaching zero, which makes `evict_one` newly
        //      eligible to reclaim it WITHOUT any `Arc<GpuPermit>` clone
        //      ever actually dropping (the `CacheEntry` keeps its own clone
        //      alive the whole time) — a transition (1) alone cannot see.
        //
        // Both `Notified` futures are registered (`enable()`d) BEFORE the
        // `try_acquire`/`evict_one`/unsatisfiable checks below, following the
        // identical lost-wakeup-safe idiom `GpuScheduler::acquire` itself
        // uses (register, then check, so a notify that fires in the gap is
        // never missed) — never a timeout: the wake-set above is complete
        // for every way this loop's admission state can change, so an
        // unbounded wait is the honest contract, not a masked liveness bug.
        let gpu_permit = loop {
            let permit_released = self.gpu_scheduler.notify.notified();
            tokio::pin!(permit_released);
            permit_released.as_mut().enable();
            let entry_became_idle = self.admission_notify.notified();
            tokio::pin!(entry_became_idle);
            entry_became_idle.as_mut().enable();

            if let Some(permit) = self.gpu_scheduler.try_acquire(memory_bytes) {
                break permit;
            }
            let evicted = {
                let mut cache = self.inner.write().await;
                cache.evict_one()
            };
            if evicted {
                continue;
            }
            if memory_bytes > self.gpu_scheduler.usable_capacity() {
                return Err(JammiError::Model {
                    model_id: source_str,
                    message: format!(
                        "Cannot acquire GPU memory: {memory_bytes} bytes requested exceeds \
                         the total usable GPU budget of {} bytes — no amount of eviction or \
                         waiting could ever satisfy this request",
                        self.gpu_scheduler.usable_capacity()
                    ),
                });
            }
            tokio::select! {
                _ = permit_released => {}
                _ = entry_became_idle => {}
            }
        };

        let loaded = backend.load(&resolved, &self.device_config)?;

        // Register model in catalog (idempotent — ignores if already registered).
        // Store the parent directory of the first weights file so that
        // `build_encoder_adapters` can locate config.json and tokenizer.json.
        let backend_str = format!("{:?}", resolved.backend).to_lowercase();
        let model_type = match source {
            ModelSource::HuggingFace(_) => "huggingface",
            ModelSource::Local(_) => "local",
        };
        let artifact_dir_str: Option<String> = resolved
            .weights_paths
            .first()
            .and_then(|p| p.parent())
            .and_then(|p| p.to_str())
            .map(|s| s.to_owned());
        if let Err(e) = self
            .resolver
            .catalog()
            .register_model(RegisterModelParams {
                model_id: &source_str,
                version: 1,
                model_type,
                backend: &backend_str,
                task,
                base_model_id: None,
                artifact_path: artifact_dir_str.as_deref(),
                config_json: None,
            })
            .await
        {
            tracing::warn!(model_id = %source_str, "Failed to register model in catalog: {e}");
        }

        let mut cache = self.inner.write().await;
        let ref_count = Arc::new(AtomicUsize::new(1));
        let model = Arc::new(loaded);
        // F-3: the permit is `Arc`-shared between this `CacheEntry` and the
        // `ModelGuard` returned below (and every subsequent warm-hit guard) —
        // see `ModelGuard::gpu_permit`'s doc for why.
        let gpu_permit = Arc::new(gpu_permit);
        cache.entries.insert(
            id.clone(),
            CacheEntry {
                model: Arc::clone(&model),
                ref_count: Arc::clone(&ref_count),
                memory_bytes,
                _residency: ModelResidency::Gpu,
                gpu_permit: Arc::clone(&gpu_permit),
            },
        );
        cache.lru_order.push_back(id.clone());

        Ok(ModelGuard::new(
            model,
            ref_count,
            gpu_permit,
            Arc::clone(&self.admission_notify),
        ))
    }

    /// Preload a model without running inference.
    pub async fn preload(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<()> {
        let guard = self.get_or_load(source, task, backend_hint).await?;
        drop(guard);
        Ok(())
    }
}

impl CacheInner {
    fn touch_lru(&mut self, id: &ModelId) {
        if let Some(pos) = self.lru_order.iter().position(|x| x == id) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_back(id.clone());
    }

    /// Evict the oldest idle entry and report whether real progress was
    /// made (i.e. a `GpuPermit` reservation was actually released).
    ///
    /// Audit round 62, F-A: `ref_count == 0` alone is NOT sufficient to
    /// promise a caller (`do_load`'s admission loop) that removing this
    /// entry frees GPU budget. A `ModelGuard`'s `Drop` releases its permit
    /// clone before decrementing `ref_count` (see `ModelGuard::drop`'s
    /// doc), so by the time `ref_count` reaches 0 the LAST outstanding
    /// clone has, in the common case, already gone — but a snapshot taken
    /// by a concurrent fast-path `get_or_load` between the write-lock
    /// section that increments `ref_count` and clones the permit (see
    /// `get_or_load`'s "atomic with the clone below" comment) can, in
    /// principle, still hold a permit clone this scan cannot see reflected
    /// in `ref_count` alone. We therefore gate progress on the actual,
    /// checkable invariant: at removal time, THIS `CacheEntry`'s
    /// `gpu_permit` clone must be the only clone left
    /// (`Arc::strong_count == 1`) — only then does dropping it truly
    /// decrement `GpuScheduler::reserved_memory`. An entry that is
    /// `ref_count == 0` but whose permit still has outstanding clones is
    /// not idle in the accounting sense: we skip it (leave it in the cache)
    /// and keep scanning for another candidate, rather than removing it and
    /// lying about progress.
    fn evict_one(&mut self) -> bool {
        let evict_id = self
            .lru_order
            .iter()
            .find(|id| {
                self.entries.get(*id).is_some_and(|e| {
                    e.ref_count.load(Ordering::Relaxed) == 0
                        && Arc::strong_count(&e.gpu_permit) == 1
                })
            })
            .cloned();

        if let Some(id) = evict_id {
            if let Some(entry) = self.entries.remove(&id) {
                self.lru_order.retain(|x| x != &id);
                debug_assert_eq!(
                    Arc::strong_count(&entry.gpu_permit),
                    1,
                    "evict_one only removes entries whose permit clone is the last one \
                     outstanding — dropping `entry` here must be what actually releases \
                     the GpuScheduler reservation"
                );
                tracing::info!(
                    model_id = %id.0,
                    bytes = entry.memory_bytes,
                    "Evicted model from cache"
                );
            }
            true
        } else {
            false
        }
    }
}

// ── F-3' (audit round 62, adversarial round 3): `evict_one`'s "true means \
//    real progress" contract must hold even while a concurrent fast-path \
//    probe is racing it ──

#[cfg(test)]
mod f3_prime_tests {
    use super::*;
    use std::sync::Arc;

    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn test_artifact_store() -> Arc<ArtifactStore> {
        let cache_dir = tempfile::tempdir().unwrap().keep();
        Arc::new(
            ArtifactStore::with_root(
                StorageUrl::memory("f3-prime-test-artifacts"),
                StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        )
    }

    /// Copy the hermetic `tiny_bert` fixture into a fresh directory under
    /// `root/name` and return a `ModelSource::local` pointing at it. Two
    /// distinct `name`s under the same `root` are two distinct `ModelId`s
    /// with byte-identical weights (same source file size), which is all
    /// this test needs: a real, admittable, real-budget-consuming model.
    fn tiny_bert_source(root: &std::path::Path, name: &str) -> (ModelSource, usize) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(file), dir.join(file)).unwrap();
        }
        let weights_len = std::fs::metadata(dir.join("model.safetensors"))
            .unwrap()
            .len() as usize;
        (ModelSource::local(&dir), weights_len)
    }

    /// F-3' (block): reproduces the auditor's exact scenario. A GPU budget
    /// that fits exactly ONE `tiny_bert`-sized model forces `do_load`'s
    /// admission loop to call `evict_one` for a second, distinct model
    /// while a fast-path `get_or_load` on the FIRST (idle, `ref_count ==
    /// 0`) model is deterministically paused between its snapshot and its
    /// `probe_freshness` call — the exact pre-fix window where a premature
    /// `gpu_permit` clone left `evict_one`'s `true` lying about real
    /// progress.
    ///
    /// Pre-fix: the paused task's snapshot already holds a SECOND
    /// `Arc<GpuPermit>` clone of model A's permit (cloned before `ref_count`
    /// was incremented). `evict_one` (driven by model B's admission loop)
    /// removes A's `CacheEntry` and drops only ITS clone — the permit's
    /// `Arc` strong count is still ≥ 1 (the paused task's clone), so
    /// `GpuScheduler::reserved_memory` is NOT actually decremented despite
    /// `evict_one` returning `true`. B's admission loop re-tries
    /// `try_acquire`, fails again (no real memory freed), calls `evict_one`
    /// a second time — finds nothing else idle — and B's load hard-errors
    /// "Cannot acquire GPU memory: nothing to evict", even though A was, in
    /// fact, "evicted." This assertion is RED pre-fix: B's `get_or_load`
    /// fails.
    ///
    /// Post-fix: the paused task's snapshot holds NO permit clone (deferred
    /// to the write-lock-protected re-validate branch), so `evict_one`'s
    /// removal of A genuinely drops A's only permit clone, genuinely
    /// decrementing `reserved_memory` — B's `try_acquire` then succeeds on
    /// the very next attempt, and B's `get_or_load` returns `Ok`.
    #[tokio::test]
    async fn evict_one_true_is_always_real_progress_under_a_racing_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog), test_artifact_store()).unwrap();

        let (source_a, weights_len) = tiny_bert_source(tmp.path(), "model_a");
        let (source_b, weights_len_b) = tiny_bert_source(tmp.path(), "model_b");
        assert_eq!(
            weights_len, weights_len_b,
            "both fixtures are copies of the same tiny_bert weights file"
        );

        // Budget fits EXACTLY one model — B's admission loop MUST evict A
        // to succeed; there is no slack that could mask the bug by simply
        // admitting B without ever calling `evict_one`.
        let scheduler = Arc::new(GpuScheduler::new(weights_len, 0.0));
        let cache = Arc::new(ModelCache::new(resolver, device_config(), scheduler));

        // (1) Load A, then drop the guard: A is warm, resident, and IDLE
        // (`ref_count == 0` AND its permit's only remaining clone is the
        // `CacheEntry`'s own, i.e. `Arc::strong_count(&gpu_permit) == 1`) —
        // `evict_one`'s eligibility condition (F-A, round 4).
        let guard_a = cache
            .get_or_load(&source_a, ModelTask::TextEmbedding, None)
            .await
            .unwrap();
        drop(guard_a);

        // (2) Install the F-3' pause and immediately start a warm-hit
        // `get_or_load` on A: it snapshots A's entry (ref_count == 0, no
        // permit clone), then pauses HERE — before `probe_freshness`,
        // before the `ref_count` increment, before any permit clone.
        let pause = cache.install_probe_pause();
        let cache_for_a = Arc::clone(&cache);
        let source_a_for_task = source_a.clone();
        let task_a = tokio::spawn(async move {
            cache_for_a
                .get_or_load(&source_a_for_task, ModelTask::TextEmbedding, None)
                .await
        });

        // Deterministic rendezvous: wait until task_a has actually reached
        // the pause point (no sleep-based guess).
        pause.arrived.notified().await;

        // (3) With A's fast-path task paused in the pre-ref_count-increment
        // window, drive B's load — under this budget it MUST evict
        // something, and A is the only idle entry.
        let guard_b = cache
            .get_or_load(&source_b, ModelTask::TextEmbedding, None)
            .await;
        let guard_b = match guard_b {
            Ok(g) => g,
            Err(e) => panic!(
                "B's load must succeed: evict_one's `true` for evicting idle model \
                 A must correspond to REAL freed memory, even while A's fast-path \
                 probe is paused mid-flight holding no premature permit clone \
                 (F-3') — got Err({e})"
            ),
        };
        // B is genuinely resident and the sole occupant of the 1-model
        // budget — no room left.
        assert_eq!(
            cache.gpu_scheduler.available(),
            0,
            "B occupies the entire 1-model budget after evicting idle A"
        );
        drop(guard_b);

        // (4) Release A's paused task: it resumes, probes (A's on-disk
        // files are untouched, so `probe_freshness` reports fresh), finds
        // its entry gone (evicted by B in step 3), retries from the top,
        // and reloads A fresh — evicting the now-idle B in turn.
        pause.release.notify_one();
        let result_a = task_a.await.unwrap();
        if let Err(e) = result_a {
            panic!(
                "A's paused fast-path task must eventually complete successfully \
                 once released, reloading fresh after its stale snapshot lost the \
                 race to B's eviction — got Err({e})"
            );
        }
    }

    /// F-A (audit round 62, adversarial round 4, block): `evict_one` must
    /// not claim progress for a `ref_count == 0` entry whose `gpu_permit`
    /// still has an outstanding clone — the exact window
    /// `ModelGuard::drop`'s pre-fix field-declaration-order drop could
    /// produce (the body's `fetch_sub` made `ref_count == 0` visible before
    /// the struct's `_gpu_permit` field itself dropped). Rather than trying
    /// to reproduce that narrow ordering race through the async cache API,
    /// this test drives `CacheInner::evict_one` DIRECTLY against a
    /// hand-built `CacheEntry` whose permit has a second, test-held clone —
    /// deterministically constructing the exact state the race could leave
    /// behind, with no timing dependency at all.
    ///
    /// Pre-fix (`evict_one` gated only on `ref_count == 0`): this assertion
    /// is RED — `evict_one` removes the misleading entry, returns `true`,
    /// and the outstanding clone (`outstanding_clone`) means
    /// `GpuScheduler::reserved_memory` is NOT actually decremented by that
    /// removal, exactly reproducing the double-booking `do_load`'s
    /// admission loop would trust.
    ///
    /// Post-fix (`evict_one` also gates on
    /// `Arc::strong_count(&entry.gpu_permit) == 1`): GREEN — the misleading
    /// entry is skipped, the genuinely idle one is evicted instead, and once
    /// the outstanding clone is finally dropped the (now genuinely idle)
    /// remaining entry is evicted too.
    ///
    /// **Advisory (audit round 62, adversarial round 6, folded)**: this test
    /// previously ran against [`GpuScheduler::new_unlimited`], whose
    /// `GpuPermit::drop` is a documented no-op — `reserved_memory` (and
    /// therefore [`GpuScheduler::available`]) NEVER moves under that
    /// scheduler, real progress or not, so the doc prose above ("would not
    /// decrement `GpuScheduler::reserved_memory`") narrated an accounting
    /// property this test never actually exercised; only the `CacheEntry`
    /// containment assertions were load-bearing. It now runs against a real
    /// BUDGETED [`GpuScheduler::new`] sized to exactly fit the real load
    /// plus X's and Y's one-byte permits, with no slack — and asserts
    /// [`GpuScheduler::available`] directly at each step: unmoved while
    /// `evict_one` (correctly) claims no progress for the misleading entry,
    /// and moved by exactly the freed amount when it evicts a genuinely
    /// idle one. The doc now claims exactly what the test proves.
    #[tokio::test]
    async fn evict_one_does_not_claim_progress_for_an_entry_whose_permit_has_an_outstanding_clone()
    {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog), test_artifact_store()).unwrap();
        let (source, weights_len) = tiny_bert_source(tmp.path(), "model_x");

        // A real, loaded `Arc<LoadedModel>` — only used as a valid handle
        // for the hand-built `CacheEntry`s below; the model's own state is
        // irrelevant to this test. Budgeted with NO slack beyond the real
        // load's own weight (`weights_len`) plus X's and Y's one-byte
        // synthetic permits below, so every `available()` assertion is
        // exact, not merely directionally suggestive.
        let scheduler = Arc::new(GpuScheduler::new(weights_len + 2, 0.0));
        let cache = ModelCache::new(resolver, device_config(), Arc::clone(&scheduler));
        let guard = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await
            .unwrap();
        let model = Arc::clone(&guard.model);
        drop(guard);

        // Entry X: `ref_count == 0` (the naive "idle" signal) but its
        // `gpu_permit` has a SECOND outstanding clone — the F-A window.
        // `evict_one` must not remove this entry and must not count it as
        // progress.
        let permit_x = Arc::new(scheduler.try_acquire(1).unwrap());
        let outstanding_clone = Arc::clone(&permit_x);
        let id_x = ModelId("model_x".into());

        // Entry Y: genuinely idle — `ref_count == 0` AND its permit's only
        // clone is this entry's own.
        let permit_y = Arc::new(scheduler.try_acquire(1).unwrap());
        let id_y = ModelId("model_y".into());

        let mut inner = CacheInner {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            in_flight: HashMap::new(),
        };
        inner.entries.insert(
            id_x.clone(),
            CacheEntry {
                model: Arc::clone(&model),
                ref_count: Arc::new(AtomicUsize::new(0)),
                memory_bytes: 1,
                _residency: ModelResidency::Gpu,
                gpu_permit: permit_x,
            },
        );
        inner.entries.insert(
            id_y.clone(),
            CacheEntry {
                model,
                ref_count: Arc::new(AtomicUsize::new(0)),
                memory_bytes: 1,
                _residency: ModelResidency::Gpu,
                gpu_permit: permit_y,
            },
        );
        // X is scanned before Y — the scan must SKIP X and land on Y rather
        // than stopping at the first misleading candidate.
        inner.lru_order.push_back(id_x.clone());
        inner.lru_order.push_back(id_y.clone());

        // The budget has NO slack: the real load (`weights_len`) plus X's
        // and Y's one-byte permits exactly exhaust it.
        assert_eq!(
            scheduler.available(),
            0,
            "sanity: the budget is fully reserved before any eviction"
        );

        assert!(
            inner.evict_one(),
            "Y is genuinely idle (no outstanding permit clone) — evict_one \
             must find and remove it, skipping past the misleading X"
        );
        assert!(
            inner.entries.contains_key(&id_x),
            "X must NOT have been removed: its permit still has an \
             outstanding clone, so evicting it would not have released \
             real memory (F-A)"
        );
        assert!(
            !inner.entries.contains_key(&id_y),
            "Y — the genuinely idle entry — must be the one actually evicted"
        );
        assert_eq!(
            scheduler.available(),
            1,
            "evicting the genuinely idle Y must ACTUALLY release its one \
             reserved byte — GpuScheduler::available() must move, not just \
             the CacheEntry map"
        );

        // Now only the misleading X remains. evict_one must report NO
        // progress rather than removing X and lying about it.
        assert!(
            !inner.evict_one(),
            "evict_one claimed progress for the sole remaining entry even \
             though its permit clone is still outstanding — this is the F-A \
             bug: removing X here would not decrement \
             GpuScheduler::reserved_memory because `outstanding_clone` is \
             still alive"
        );
        assert!(inner.entries.contains_key(&id_x));
        assert_eq!(
            scheduler.available(),
            1,
            "evict_one's false claim of no progress must correspond to \
             GpuScheduler::available() genuinely NOT moving — the exact \
             accounting property the pre-fix code lied about"
        );

        drop(outstanding_clone);
        assert_eq!(
            scheduler.available(),
            1,
            "dropping only the TEST's outstanding clone (not the CacheEntry's \
             own) must not yet release the reservation — X's permit still \
             has one live clone (the entry's own)"
        );
        assert!(
            inner.evict_one(),
            "once the outstanding clone is gone, X is genuinely idle and \
             evict_one must now claim (and deliver) real progress"
        );
        assert!(!inner.entries.contains_key(&id_x));
        assert_eq!(
            scheduler.available(),
            2,
            "the final evict_one — now genuinely idle — must ACTUALLY \
             release X's reserved byte too, matching its claimed progress"
        );
    }
}

// ── Advisory (audit round 62, adversarial round 3): single-flight \
//    lost-wakeup ──

#[cfg(test)]
mod single_flight_advisory_tests {
    use super::*;
    use std::time::Duration;

    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn test_artifact_store() -> Arc<ArtifactStore> {
        let cache_dir = tempfile::tempdir().unwrap().keep();
        Arc::new(
            ArtifactStore::with_root(
                StorageUrl::memory("single-flight-advisory-test-artifacts"),
                StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        )
    }

    fn tiny_bert_source(root: &std::path::Path, name: &str) -> ModelSource {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(file), dir.join(file)).unwrap();
        }
        ModelSource::local(&dir)
    }

    /// Advisory (audit round 62): a waiter that has genuinely REGISTERED
    /// (its `Notified` future `enable()`d) before the loader removes its
    /// `in_flight` entry and calls `notify_waiters` must always wake — even
    /// when the loader's completion (simulated directly here, bypassing
    /// `do_load`, for full determinism) lands exactly inside the pause
    /// window between this task's registration and its `.await`.
    ///
    /// Pre-fix, the equivalent window sat BEFORE `notify.notified()` was
    /// even constructed (`drop(cache); notify.notified().await;`) — a
    /// `notify_waiters()` landing there is unconditionally lost, and the
    /// waiter hangs forever (no timeout). Bounded here with a generous
    /// timeout so a REGRESSION of the lost-wakeup bug fails this test with
    /// a clear "timed out" message rather than hanging CI.
    #[tokio::test]
    async fn registered_waiter_always_wakes_even_if_notify_races_the_pause() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog), test_artifact_store()).unwrap();
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let cache = Arc::new(ModelCache::new(resolver, device_config(), scheduler));

        let source = tiny_bert_source(tmp.path(), "single_flight_model");
        let id = ModelId::from(&source);

        // Simulate "another task is already loading this id" directly,
        // bypassing `do_load` entirely — the ONLY state `get_or_load`'s
        // single-flight branch actually observes.
        let loader_notify = Arc::new(tokio::sync::Notify::new());
        {
            let mut inner = cache.inner.write().await;
            inner
                .in_flight
                .insert(id.clone(), Arc::clone(&loader_notify));
        }

        let pause = cache.install_single_flight_pause();
        let cache_for_waiter = Arc::clone(&cache);
        let source_for_waiter = source.clone();
        let waiter = tokio::spawn(async move {
            cache_for_waiter
                .get_or_load(&source_for_waiter, ModelTask::TextEmbedding, None)
                .await
        });

        // Deterministic rendezvous: the waiter has registered and reached
        // the pause.
        pause.arrived.notified().await;

        // Simulate the loader completing WHILE the waiter is paused —
        // exactly the lost-wakeup window this advisory closes.
        {
            let mut inner = cache.inner.write().await;
            inner.in_flight.remove(&id);
        }
        loader_notify.notify_waiters();

        // Release the waiter: it must wake immediately (already
        // registered), see `in_flight` empty, and become the loader
        // itself, completing the real load.
        pause.release.notify_one();

        let joined = tokio::time::timeout(Duration::from_secs(10), waiter).await;
        let result = match joined {
            Ok(joined) => joined.unwrap(),
            Err(_) => panic!(
                "the waiter never woke within 10s — a lost wakeup (the advisory's \
                 pre-fix defect) would hang here forever; this bound turns a \
                 regression into a fast, clear test failure instead of a hang"
            ),
        };
        if let Err(e) = result {
            panic!("the waiter must complete its (now solo) load successfully after waking, got Err({e})");
        }
    }
}

// ── R5-F1 (audit round 62, adversarial round 6): a deleted tokenizer.json \
//    must never permanently wedge `get_or_load` ──

#[cfg(test)]
mod r5_f1_tokenizer_tests {
    use super::*;

    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn test_artifact_store() -> Arc<ArtifactStore> {
        let cache_dir = tempfile::tempdir().unwrap().keep();
        Arc::new(
            ArtifactStore::with_root(
                StorageUrl::memory("r5-f1-tokenizer-test-artifacts"),
                StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        )
    }

    fn tiny_bert_source(root: &std::path::Path, name: &str) -> ModelSource {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(file), dir.join(file)).unwrap();
        }
        ModelSource::local(&dir)
    }

    /// R5-F1 (block), end-to-end at the `ModelCache::get_or_load` level: a
    /// warm model whose `tokenizer.json` is deleted from its live directory
    /// must stale-reload, never wedge.
    ///
    /// RED pre-fix: the tokenizer digest candidate was unconditionally
    /// `optional: false` ("the loader has no fallback" — false for this
    /// slot, since every resolver path re-derives `tokenizer: None` on
    /// absence and `CandleBackend::load` accepts it). `ModelFingerprint::probe`
    /// therefore fell into arm (c) and returned `Err`, and — critically —
    /// `get_or_load`'s `Err` branch returns immediately WITHOUT removing the
    /// stale `CacheEntry` from the cache. So the SECOND `get_or_load` call
    /// (after the same deletion) hits the identical still-cached, still-stale
    /// entry, re-probes, and re-`Err`s — wedged forever, exactly like a cold
    /// process would NOT be (a cold process loading this same
    /// tokenizer-less directory loads fine, with `tokenizer: None`).
    ///
    /// GREEN post-fix: the tokenizer candidate is `optional: true` (arm
    /// (b)) — the first post-deletion `get_or_load` evicts the stale entry
    /// and reloads (succeeding, with `tokenizer: None`, mirroring
    /// cold-process semantics), and the second call is an ordinary warm hit
    /// against the freshly-reloaded (tokenizer-less) entry. Neither call
    /// wedges.
    #[tokio::test]
    async fn tokenizer_deleted_after_load_stale_reloads_never_wedges() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog), test_artifact_store()).unwrap();
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let cache = ModelCache::new(resolver, device_config(), scheduler);

        let source = tiny_bert_source(tmp.path(), "tokenizer_wedge_model");

        // (1) Cold load: tokenizer.json present, resolves to `Some(..)`.
        let guard = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await
            .expect("initial load with tokenizer.json present must succeed");
        drop(guard);

        // Delete tokenizer.json from the LIVE model directory — the exact
        // scenario the auditor named.
        let model_dir = match &source {
            ModelSource::Local(p) => p.clone(),
            other => panic!("expected a Local source, got {other:?}"),
        };
        std::fs::remove_file(model_dir.join("tokenizer.json")).unwrap();

        // (2) First post-deletion call: must stale-reload (evict the
        // now-invalid fingerprint entry and reload with `tokenizer: None`),
        // never return the typed refusal that would come from mis-classifying
        // the tokenizer as REQUIRED.
        let first = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await;
        match &first {
            Ok(_) => {}
            Err(e) => panic!(
                "the first get_or_load after tokenizer.json's deletion must \
                 stale-reload and succeed (R5-F1) — a cold process loading \
                 this same directory would serve it fine — got Err({e})"
            ),
        }
        drop(first);

        // (3) Second, CONSECUTIVE post-deletion call: this is the wedge
        // check. Pre-fix, the first call above already returned `Err`
        // without evicting, so this second call would hit the identical
        // stale entry and `Err` again — "permanently wedged". Post-fix, this
        // is an ordinary warm hit against the freshly-reloaded entry from
        // step (2).
        let second = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await;
        if let Err(e) = second {
            panic!(
                "the second, consecutive get_or_load call after \
                 tokenizer.json's deletion must also succeed — two \
                 consecutive calls behaving like cold-process loads, never a \
                 permanent wedge (R5-F1). Got Err({e})"
            );
        }
    }

    /// R7-F1 (block), the full restore cycle end-to-end at
    /// `ModelCache::get_or_load`: delete `tokenizer.json` -> warm reload
    /// (tokenizer-less, embedding refuses) -> RESTORE `tokenizer.json` ->
    /// the NEXT `get_or_load` must detect the staleness that restoration
    /// creates and reload WITH the tokenizer — embedding serves again.
    ///
    /// RED pre-fix: `all_candidate_paths` pushed the tokenizer candidate
    /// ONLY when `resolved.tokenizer.is_some()` (R5-F1's own fix, which
    /// correctly stopped the tokenizer-deleted case from wedging, but left
    /// this hole). After the round-6 stale-reload (this test's steps 1-3,
    /// identical to `tokenizer_deleted_after_load_stale_reloads_never_wedges`
    /// above), the freshly-reloaded entry's fingerprint was computed from a
    /// `resolved.tokenizer == None`, so the tokenizer candidate was dropped
    /// from the candidate set ENTIRELY — nothing was ever fingerprinted for
    /// it, present or absent. Restoring `tokenizer.json` on disk (step 4)
    /// therefore changed nothing any fingerprinted `(rel, path, snapshot)`
    /// tuple tracked: `probe` reported fresh forever, `get_or_load`'s fast
    /// path kept serving the tokenizer-less entry, and every embedding call
    /// kept failing on a directory a cold process would serve fine.
    ///
    /// GREEN post-fix: the tokenizer candidate is pushed UNCONDITIONALLY
    /// (both `tokenizer.json` and `bpe_simple_vocab_16e6.txt.gz`, mirroring
    /// `1_Pooling`/`preprocessor`), so the tokenizer-less reload's
    /// fingerprint still records an ABSENT snapshot for `tokenizer.json`.
    /// Restoring the file flips that candidate from `NotFound` to `Ok`,
    /// which `ModelFingerprint::probe`'s F-4b arm unconditionally reports as
    /// stale (`Ok(false)`) regardless of `optional` — the next `get_or_load`
    /// evicts and reloads, this time resolving `tokenizer: Some(..)` again,
    /// and the reloaded entry serves embeddings.
    #[tokio::test]
    async fn tokenizer_restored_after_stale_reload_is_detected_and_reloads_with_tokenizer() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog), test_artifact_store()).unwrap();
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let cache = ModelCache::new(resolver, device_config(), scheduler);

        let source = tiny_bert_source(tmp.path(), "tokenizer_restore_model");
        let model_dir = match &source {
            ModelSource::Local(p) => p.clone(),
            other => panic!("expected a Local source, got {other:?}"),
        };
        let text_content: Vec<arrow::array::ArrayRef> =
            vec![std::sync::Arc::new(arrow::array::StringArray::from(vec![
                "a sentence to embed",
            ]))];

        // (1) Cold load: `tokenizer.json` present, resolves to `Some(..)`.
        let guard = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await
            .expect("initial load with tokenizer.json present must succeed");
        guard
            .model
            .forward(&text_content, ModelTask::TextEmbedding)
            .expect("the cold-loaded, tokenizer-bearing entry must serve embeddings");
        drop(guard);

        // (2) Delete `tokenizer.json` from the LIVE model directory.
        std::fs::remove_file(model_dir.join("tokenizer.json")).unwrap();

        // (3) Stale-reload (R5-F1's own fix): succeeds, `tokenizer: None`.
        // Its OWN forward call must now refuse — a tokenizer-less entry
        // cannot serve embeddings — with the typed "no tokenizer" error,
        // never a panic or a silently-wrong output.
        let tokenizer_less = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await
            .expect("stale-reload after tokenizer.json's deletion must succeed (R5-F1)");
        match tokenizer_less
            .model
            .forward(&text_content, ModelTask::TextEmbedding)
        {
            Err(e) => {
                // Advisory (audit round 62, adversarial round 10 fold): pin
                // the refusal to the SPECIFIC typed message
                // (`CandleModel::forward_embedding`'s tokenizer guard), not
                // merely "any Err" — proves this refuses for the reason this
                // test names ("no tokenizer loaded"), not some incidental
                // unrelated failure that would make the assertion vacuous.
                let msg = e.to_string();
                assert!(
                    msg.contains("No tokenizer loaded"),
                    "expected the typed 'No tokenizer loaded' refusal for a \
                     tokenizer-less warm entry's embedding call, got: {msg}"
                );
            }
            Ok(_) => panic!(
                "a tokenizer-less warm entry must refuse an embedding call, not silently serve"
            ),
        }
        drop(tokenizer_less);

        // (4) RESTORE `tokenizer.json` — the exact scenario R7-F1 names: the
        // file a cold process would happily use again is back on disk.
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        std::fs::copy(
            fixture.join("tokenizer.json"),
            model_dir.join("tokenizer.json"),
        )
        .unwrap();

        // (5) The NEXT get_or_load must detect the restoration as staleness
        // (not report fresh — the R7-F1 bug) and reload WITH the tokenizer.
        let restored = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await
            .expect("get_or_load after tokenizer.json's restoration must succeed");

        // (6) Final serving state: the restored entry serves embeddings
        // again — the full cycle's actual observable outcome, not merely
        // "the fingerprint changed".
        restored
            .model
            .forward(&text_content, ModelTask::TextEmbedding)
            .expect(
                "the entry reloaded after tokenizer.json's restoration must serve embeddings \
                 again (R7-F1) — a cold process loading this same, now-restored directory would \
                 serve it fine",
            );
    }
}

// ── Unit 62, closure-audit BLOCK 1 (block): the admission-wait liveness \
//    hole — a plain LRU-budget-pressure eviction (NOT a stale reload) must \
//    wake once the blocking entry's ref_count reaches zero, even though no \
//    `Arc<GpuPermit>` clone ever actually drops for that transition ──

#[cfg(test)]
mod admission_wake_tests {
    use super::*;
    use std::time::Duration;

    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn test_artifact_store() -> Arc<ArtifactStore> {
        let cache_dir = tempfile::tempdir().unwrap().keep();
        Arc::new(
            ArtifactStore::with_root(
                StorageUrl::memory("admission-wake-test-artifacts"),
                StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        )
    }

    /// Copy the hermetic `tiny_bert` fixture into a fresh directory under
    /// `root/name` and return a `ModelSource::local` pointing at it, plus
    /// the weights file's byte length (the scheduler budget unit this
    /// module sizes against).
    fn tiny_bert_source(root: &std::path::Path, name: &str) -> (ModelSource, usize) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(file), dir.join(file)).unwrap();
        }
        let weights_len = std::fs::metadata(dir.join("model.safetensors"))
            .unwrap()
            .len() as usize;
        (ModelSource::local(&dir), weights_len)
    }

    /// The missing lattice cell (auditor-verified mechanism chain): budget
    /// sized to ONE resident copy; A holds M1's guard; B spawns
    /// `get_or_load(M2)` — under this budget `do_load` cannot admit M2
    /// without evicting M1, and M1 is NOT evictable while A's guard is live
    /// (`ref_count == 1`), so B must park in the admission loop's wait. A
    /// then drops M1's guard WITHOUT the entry ever being removed from the
    /// cache (a plain idle-LRU eviction target, unlike every pre-existing
    /// admission-wait test, which is a STALE RELOAD of the SAME model —
    /// there the outgoing guard holds the permit's LAST clone, since the
    /// stale entry was already removed from `cache.entries`, so
    /// `GpuPermit::drop`'s own `notify_waiters()` fires directly). Here the
    /// `CacheEntry` for M1 is still present and keeps its own clone of the
    /// SAME `Arc<GpuPermit>` alive the whole time (F-3's sharing
    /// discipline) — A's guard drop lowers the `Arc`'s strong count from 2
    /// to 1, never to 0, so `GpuPermit::drop`'s body — and its
    /// `notify_waiters()` — never runs. The ONLY transition that fires here
    /// is `ModelGuard::drop`'s unconditional `admission_notify.notify_waiters()`
    /// signal (see `ModelGuard`'s `admission_notify` field doc) — a waiter
    /// parked purely on `GpuScheduler`'s own release notify (the pre-fix
    /// shape: `do_load` fell back to `GpuScheduler::acquire`, which only
    /// ever waits on that one notify) would never wake, hanging forever
    /// while permanently holding `cache.in_flight[M2]`.
    ///
    /// RED proof (see this test's own doc for how to reproduce): reverting
    /// `do_load`'s admission loop to plain `self.gpu_scheduler.acquire(...)`
    /// (dropping the `admission_notify`/`select!` wait entirely) makes B's
    /// task hang — never observing M1's `ref_count -> 0` transition — and
    /// this test's final `tokio::time::timeout` bound turns that hang into
    /// a clear, fast assertion failure instead of wedging CI. This mirrors
    /// the single-flight lost-wakeup advisory test's own precedent
    /// (`single_flight_advisory_tests::registered_waiter_always_wakes_even_if_notify_races_the_pause`)
    /// for bounding a liveness regression with a timeout rather than
    /// relying on an indefinite hang to "prove" the wait.
    #[tokio::test]
    async fn plain_lru_eviction_wakes_once_the_blocking_guard_drops_even_with_no_permit_release() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(Arc::clone(&catalog), test_artifact_store()).unwrap();

        let (source_a, weights_len) = tiny_bert_source(tmp.path(), "admission_wake_model_a");
        let (source_b, weights_len_b) = tiny_bert_source(tmp.path(), "admission_wake_model_b");
        assert_eq!(
            weights_len, weights_len_b,
            "both fixtures are copies of the same tiny_bert weights file"
        );

        // Budget fits EXACTLY one model — B cannot be admitted until A's
        // guard drops and M1 is evicted; there is no slack that would let
        // B's `try_acquire` succeed on its own.
        let scheduler = Arc::new(GpuScheduler::new(weights_len, 0.0));
        let cache = Arc::new(ModelCache::new(resolver, device_config(), scheduler));

        // (1) A loads M1 and KEEPS THE GUARD — the entire budget is
        // reserved, and M1 is NOT idle (`ref_count == 1`).
        let guard_a = cache
            .get_or_load(&source_a, ModelTask::TextEmbedding, None)
            .await
            .unwrap();

        // (2) Spawn B's load of the DISTINCT model M2 while A's guard is
        // still held. This must park in `do_load`'s admission wait: M1 is
        // not evictable (ref_count != 0), the budget cannot admit a second
        // resident copy, and the request is within `usable_capacity`
        // (satisfiable once A releases), so no hard error is legitimate.
        let cache_for_b = Arc::clone(&cache);
        let source_b_for_task = source_b.clone();
        let mut task_b = tokio::spawn(async move {
            cache_for_b
                .get_or_load(&source_b_for_task, ModelTask::TextEmbedding, None)
                .await
        });

        // (3) B must NOT complete yet — a generous-but-bounded probe turns
        // "still parked" into a fast, clear assertion rather than an
        // indefinite block that merely LOOKS like proof of parking.
        let still_pending = tokio::time::timeout(Duration::from_millis(500), &mut task_b).await;
        match still_pending {
            Err(_elapsed) => {
                // Timed out waiting for completion — genuinely still
                // parked, exactly what "waiting on admission" looks like.
            }
            Ok(joined) => match joined.unwrap() {
                Ok(_guard) => panic!(
                    "B's load completed BEFORE A's guard was dropped, while the \
                     1x budget could not possibly admit a second resident copy \
                     alongside A's still-live guard — structurally impossible \
                     either way"
                ),
                Err(e) => panic!(
                    "B's load returned Err BEFORE A's guard was dropped — the \
                     admission loop must WAIT here (M1 is genuinely reclaimable \
                     once A releases), never hard-error on a request that is \
                     perfectly satisfiable once the blocking guard drops — got \
                     Err({e})"
                ),
            },
        }

        // (4) Release A's guard: M1's `CacheEntry` is STILL IN THE CACHE
        // (never removed) — this is the plain idle-LRU shape, not a stale
        // reload, so no `Arc<GpuPermit>` clone reaches zero here. Only the
        // `ModelGuard::drop`-signalled admission notify can wake B.
        drop(guard_a);

        // (5) B must now wake, evict the now-idle M1 via `evict_one`, load
        // M2, and complete — within a generous bound. A hang here (bounded
        // by the timeout, never an indefinite `.await`) is exactly the
        // liveness hole this test targets.
        let joined = tokio::time::timeout(Duration::from_secs(10), task_b).await;
        let result = match joined {
            Ok(joined) => joined.unwrap(),
            Err(_) => panic!(
                "B's load never completed within 10s after A's guard was dropped \
                 — the admission loop never woke because no GpuPermit clone ever \
                 actually released (M1's CacheEntry kept its own clone alive the \
                 whole time); this is the admission-wake liveness hole (unit 62, \
                 closure-audit BLOCK 1) — B is left parked forever, permanently \
                 holding cache.in_flight[M2]"
            ),
        };
        let guard_b = match result {
            Ok(guard) => guard,
            Err(e) => panic!(
                "B's load must succeed once A's guard is dropped and M1 becomes \
                 evictable — got Err({e})"
            ),
        };

        // (6) Sanity: B genuinely occupies the entire 1-model budget — M1
        // was actually evicted (real progress), not merely "unblocked" by
        // some accounting fluke.
        assert_eq!(
            cache.gpu_scheduler.available(),
            0,
            "B occupies the entire 1-model budget after evicting the now-idle M1"
        );
        drop(guard_b);
    }
}
