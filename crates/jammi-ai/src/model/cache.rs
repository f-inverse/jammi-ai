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

    /// Get or load a model. Returns a guard that keeps the model alive.
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
            // can only ever observe `ref_count == 0` with NO permit clone
            // outstanding, so its `true` is always real progress.
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
                            return Ok(ModelGuard {
                                model,
                                ref_count,
                                _gpu_permit: gpu_permit,
                            });
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
                        let mut cache = self.inner.write().await;
                        if cache
                            .entries
                            .get(&id)
                            .is_some_and(|e| Arc::ptr_eq(&e.model, &model))
                        {
                            cache.entries.remove(&id);
                            cache.lru_order.retain(|x| x != &id);
                        }
                        drop(cache);
                        // Fall through to single-flight/load below, which
                        // re-resolves and re-hashes the CURRENT bytes.
                    }
                    Err(e) => {
                        // A fingerprinted candidate vanished or became
                        // unreadable between load and this probe: a typed
                        // refusal (K2), never a silent "treat as fresh" or
                        // "treat as stale".
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

        let gpu_permit = loop {
            if let Some(permit) = self.gpu_scheduler.try_acquire(memory_bytes) {
                break permit;
            }
            let mut cache = self.inner.write().await;
            if !cache.evict_one() {
                return Err(JammiError::Model {
                    model_id: source_str,
                    message: "Cannot acquire GPU memory: nothing to evict".into(),
                });
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

        Ok(ModelGuard {
            model,
            ref_count,
            _gpu_permit: gpu_permit,
        })
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

    fn evict_one(&mut self) -> bool {
        let evict_id = self
            .lru_order
            .iter()
            .find(|id| {
                self.entries
                    .get(*id)
                    .is_some_and(|e| e.ref_count.load(Ordering::Relaxed) == 0)
            })
            .cloned();

        if let Some(id) = evict_id {
            if let Some(entry) = self.entries.remove(&id) {
                self.lru_order.retain(|x| x != &id);
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
        // (`ref_count == 0`) — `evict_one`'s only eligibility condition.
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
