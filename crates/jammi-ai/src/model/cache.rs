use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jammi_db::catalog::model_repo::{ModelBackendKind, RegisterModelParams};
use jammi_db::config::CacheBounds;
use jammi_db::error::{JammiError, Result};

use super::backend::candle::CandleBackend;
use super::backend::remote::RemoteModel;
use super::backend::{DeviceConfig, ModelBackend};
use super::memo::{Memo, MemoEntry};
use super::resolver::ModelResolver;
use super::{LoadedModel, ModelDescription, ModelGuard, ModelId, ResolvedModel};
use crate::concurrency::{DeviceSchedulers, GpuPermit, GpuScheduler};
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;

/// A loaded model the cache holds: the model, the count of guards handed
/// out for it, and its device reservation.
pub(crate) struct CacheEntry {
    model: Arc<LoadedModel>,
    ref_count: Arc<AtomicUsize>,
    memory_bytes: usize,
    /// Shared with every outstanding `ModelGuard` handed
    /// out for this entry (see `ModelGuard::gpu_permit`'s doc). Removing this
    /// `CacheEntry` from the cache (stale-fingerprint eviction, `evict_one`)
    /// drops only THIS clone — the reservation is released by `GpuPermit`'s
    /// `Drop` only once every clone (this one plus any live guard's) is gone,
    /// so `reserved_memory` is never decremented while a guard still holds
    /// this model's device tensors resident across a forward pass.
    gpu_permit: Arc<GpuPermit>,
    /// Signalled by every guard's drop — see [`ModelGuard`]'s
    /// `admission_notify` field doc.
    admission_notify: Arc<tokio::sync::Notify>,
}

/// A loaded model is probed through its description's fingerprint, and
/// handed out as a [`ModelGuard`].
///
/// The probe handle is the `Arc<LoadedModel>` alone — never a `gpu_permit`
/// clone. A permit clone made before `ref_count` is incremented would be an
/// outstanding `Arc<GpuPermit>` invisible to eviction: the entry would be
/// removed and reported as progress while the reservation stays held, and
/// `do_load`'s admission loop, which trusts that report, would evict a
/// second model or fail a load that fits. The permit clone is taken in
/// [`MemoEntry::hand_out`], under the memo's write lock and in the same
/// critical section as the `ref_count` increment.
///
/// `ModelGuard::drop` likewise releases its permit clone before
/// decrementing `ref_count`. Neither ordering is what eviction relies on:
/// [`MemoEntry::is_idle`] checks `Arc::strong_count(&gpu_permit) == 1`, the
/// quantity that actually decides whether dropping the entry releases the
/// reservation. `ref_count == 0` alone is not enough — an entry whose permit
/// still has a clone outstanding is skipped, never removed as progress.
///
/// A stale entry is evicted whether or not it is in use (serving stale
/// bytes is a correctness bug, not a capacity one). Removing it drops only
/// ITS permit clone, so the reservation is not released while a guard
/// still forwarding through the pre-mutation model holds another — the
/// accounting never double-books that model's still-resident memory.
impl MemoEntry for CacheEntry {
    type Probe = Arc<LoadedModel>;
    type Handle = ModelGuard;

    fn probe_handle(&self) -> Arc<LoadedModel> {
        Arc::clone(&self.model)
    }

    fn is(&self, probe: &Arc<LoadedModel>) -> bool {
        Arc::ptr_eq(&self.model, probe)
    }

    fn probe_freshness(probe: &Arc<LoadedModel>) -> Result<bool> {
        probe.description().probe_freshness()
    }

    fn hand_out(&self) -> ModelGuard {
        self.ref_count.fetch_add(1, Ordering::Acquire);
        ModelGuard::new(
            Arc::clone(&self.model),
            Arc::clone(&self.ref_count),
            Arc::clone(&self.gpu_permit),
            Arc::clone(&self.admission_notify),
        )
    }

    fn is_idle(&self) -> bool {
        self.ref_count.load(Ordering::Relaxed) == 0 && Arc::strong_count(&self.gpu_permit) == 1
    }
}

/// A description is probed through its own fingerprint and handed out as
/// the shared `Arc`. It holds no reservation, so it is always idle.
impl MemoEntry for Arc<ModelDescription> {
    type Probe = Arc<ModelDescription>;
    type Handle = Arc<ModelDescription>;

    fn probe_handle(&self) -> Arc<ModelDescription> {
        Arc::clone(self)
    }

    fn is(&self, probe: &Arc<ModelDescription>) -> bool {
        Arc::ptr_eq(self, probe)
    }

    fn probe_freshness(probe: &Arc<ModelDescription>) -> Result<bool> {
        probe.probe_freshness()
    }

    fn hand_out(&self) -> Arc<ModelDescription> {
        Arc::clone(self)
    }

    fn is_idle(&self) -> bool {
        true
    }
}

#[cfg(test)]
impl CacheEntry {
    /// An idle entry for an inner-state test: `model` holding `gpu_permit`,
    /// no guard handed out.
    pub(crate) fn for_test(
        model: &Arc<LoadedModel>,
        gpu_permit: Arc<GpuPermit>,
        memory_bytes: usize,
    ) -> Self {
        Self {
            model: Arc::clone(model),
            ref_count: Arc::new(AtomicUsize::new(0)),
            memory_bytes,
            gpu_permit,
            admission_notify: Arc::new(tokio::sync::Notify::new()),
        }
    }
}

/// What identifies one cached model.
///
/// A model id alone is not an identity: the same weights loaded on two
/// devices are two resident copies with two budgets, and the same
/// checkpoint loaded for two tasks is two different loaded objects. Keying
/// on the id alone made the second load a warm HIT on the first — a model
/// served from the wrong device, or with the wrong head.
///
/// A `None` task is a DISTINCT KEY VALUE, never a wildcard: it neither
/// matches nor is matched by a key that names a task. A wildcard would
/// reintroduce exactly the collision this key exists to remove.
///
/// Both memos key by it, entries and single-flight alike: a single-flight
/// that keyed more coarsely than the entries would make one loader stand in
/// for a load of a different thing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// The resolved model id.
    pub model_id: ModelId,
    /// The device ordinal this copy is resident on (-1 for the host).
    pub device: i32,
    /// The task the model was loaded for, when the caller named one.
    pub task: Option<ModelTask>,
}

#[cfg(test)]
impl CacheKey {
    /// A key for an inner-state test: one model id on one device, with no
    /// task named. `None` here is a DISTINCT key value, never a wildcard.
    pub(crate) fn for_test(model_id: &str, device: i32) -> Self {
        Self {
            model_id: ModelId(model_id.to_string()),
            device,
            task: None,
        }
    }
}

/// LRU cache of loaded models with GPU memory tracking and single-flight
/// loading, and the memo of the descriptions they are materialized from.
pub struct ModelCache {
    models: Memo<CacheKey, CacheEntry>,
    descriptions: Memo<CacheKey, Arc<ModelDescription>>,
    resolver: ModelResolver,
    backend: CandleBackend,
    device_config: DeviceConfig,
    /// One admission budget per configured device — see [`DeviceSchedulers`].
    gpu_schedulers: DeviceSchedulers,
    /// The cache-level admission wake source. See [`ModelGuard`]'s
    /// `admission_notify` field doc for why `GpuScheduler`'s own release
    /// notify is not sufficient on its own, and `do_load`'s admission loop
    /// for the full wake-set enumeration this notify is one half of.
    admission_notify: Arc<tokio::sync::Notify>,
    /// The remote models this deployment declares, by name.
    remote: BTreeMap<String, RemoteEndpoint>,
}

/// A remote model this deployment declares, and the admission its forwards
/// share: the endpoint is the device they run on.
struct RemoteEndpoint {
    model: Arc<RemoteModel>,
    admission: Arc<GpuScheduler>,
}

impl ModelCache {
    /// Create a cache over the device config's PRIMARY device, backed by the
    /// given resolver and that device's scheduler.
    ///
    /// The single-device shape. A deployment that declares several devices
    /// builds the cache with [`Self::with_device_schedulers`] instead, so
    /// every declared device gets its own budget.
    pub fn new(
        resolver: ModelResolver,
        device_config: DeviceConfig,
        gpu_scheduler: Arc<GpuScheduler>,
    ) -> Self {
        let primary = device_config.gpu_device;
        Self::with_device_schedulers(
            resolver,
            device_config,
            DeviceSchedulers::single(primary, gpu_scheduler),
        )
    }

    /// Create a cache over EVERY device the deployment declared, each with
    /// its own admission budget.
    ///
    /// The cache is one map keyed by [`CacheKey`] rather than one cache per
    /// device: a model resident on two devices is two entries of one LRU, so
    /// eviction still reasons over the whole process's residency, while
    /// admission is charged to the device the copy actually occupies.
    ///
    /// Bounded by the `[inference]` defaults; a deployment's own bounds are
    /// set with [`Self::bounded`].
    pub fn with_device_schedulers(
        resolver: ModelResolver,
        device_config: DeviceConfig,
        gpu_schedulers: DeviceSchedulers,
    ) -> Self {
        let bounds = jammi_db::config::InferenceConfig::default().cache_bounds();
        Self {
            models: Memo::new(bounds.loaded_models),
            descriptions: Memo::new(bounds.described_models),
            resolver,
            backend: CandleBackend,
            device_config,
            gpu_schedulers,
            admission_notify: Arc::new(tokio::sync::Notify::new()),
            remote: BTreeMap::new(),
        }
    }

    /// This cache, serving the remote models `declared` — each with its own
    /// admission of `max_in_flight` forwards. A declaration whose
    /// credentials cannot be read is refused here, when the session opens.
    pub fn with_remote_models(
        self,
        declared: &BTreeMap<String, jammi_db::config::RemoteModelConfig>,
    ) -> Result<Self> {
        let remote = declared
            .iter()
            .map(|(name, config)| {
                Ok((
                    name.clone(),
                    RemoteEndpoint {
                        model: Arc::new(RemoteModel::from_config(name, config)?),
                        admission: Arc::new(GpuScheduler::endpoint(config.max_in_flight)),
                    },
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self { remote, ..self })
    }

    /// The declared remote model `name`, or a refusal naming the missing
    /// declaration.
    fn remote_endpoint(&self, name: &str) -> Result<&RemoteEndpoint> {
        self.remote.get(name).ok_or_else(|| JammiError::Model {
            model_id: ModelSource::remote(name).to_string(),
            message: format!(
                "no remote model '{name}' is declared: add [models.remote.{name}] to this \
                 deployment's configuration"
            ),
        })
    }

    /// A remote model's description: its declaration, for a task its
    /// protocol carries.
    fn describe_remote(
        &self,
        source: &ModelSource,
        endpoint: &RemoteEndpoint,
        task: ModelTask,
    ) -> Result<Arc<ModelDescription>> {
        endpoint.model.check_task(task)?;
        Ok(Arc::new(ModelDescription {
            model_id: source.to_string(),
            backing: super::DescribedBacking::Remote(endpoint.model.run().clone()),
        }))
    }

    /// This cache, keeping at most `bounds` idle entries.
    pub fn bounded(self, bounds: CacheBounds) -> Self {
        Self {
            models: Memo::new(bounds.loaded_models),
            descriptions: Memo::new(bounds.described_models),
            ..self
        }
    }

    /// The per-device admission budgets this cache admits loads against.
    pub fn schedulers(&self) -> &DeviceSchedulers {
        &self.gpu_schedulers
    }

    /// Test seam: pause the next warm lookup of a loaded model between its
    /// snapshot and its freshness probe — the window a concurrent eviction
    /// must interleave into.
    #[cfg(test)]
    pub(crate) fn install_probe_pause(&self) -> super::memo::PauseHandle {
        self.models.install_probe_pause()
    }

    /// Test seam: pause the next caller that waits on another's load, after
    /// it registered as a waiter and before it awaits.
    #[cfg(test)]
    pub(crate) fn install_single_flight_pause(&self) -> super::memo::PauseHandle {
        self.models.install_wait_pause()
    }

    /// Get or load a model. Returns a guard that keeps the model alive.
    ///
    /// **Staleness contract (see `backend::candle::ModelFingerprint`'s doc
    /// for the full accounting): this cache's warm-hit staleness detection
    /// is NARROW.** It re-`stat`s the FILES the resolver selected at load
    /// time and reloads on in-place mutation, deletion, or appearance among
    /// them; it does NOT re-verify catalog location
    /// rewrites (a fine-tuned retrain's new adapter goes unnoticed by a warm
    /// entry until process restart), catalog-vs-local precedence, HF
    /// revision moves, or remote sibling listings. The guarantee this DOES
    /// provide is
    /// BOUNDED STALENESS, never per-hit freshness: the returned
    /// [`ModelGuard`] was fresh at some instant before this call began, but
    /// is never revalidated again — a TOCTOU window between that instant and
    /// the guard's actual use is inherent, not a defect.
    pub async fn get_or_load(&self, source: &ModelSource, task: ModelTask) -> Result<ModelGuard> {
        self.get_or_load_on(self.gpu_schedulers.primary(), source, task)
            .await
    }

    /// [`Self::get_or_load`] on a NAMED device of this deployment.
    ///
    /// The device-plural entry point: rank `r` of a gang asks for its own
    /// device, and gets a copy resident there rather than the primary's.
    /// Two devices therefore hold two entries for one model id — they are
    /// two resident copies, charged to two budgets.
    ///
    /// A device the deployment never declared is refused with a typed error:
    /// there is no budget for it, and loading onto it anyway would put a
    /// model somewhere the operator did not place this process.
    pub async fn get_or_load_on(
        &self,
        device: i32,
        source: &ModelSource,
        task: ModelTask,
    ) -> Result<ModelGuard> {
        // The device is validated here — a warm hit must not be able to
        // return a copy from a device this deployment never declared — but
        // the per-device `DeviceConfig` is built in `do_load`, on the cold
        // path only: it owns a `Vec`, and cloning one per warm hit would put
        // an allocation on the path that exists to avoid work.
        let scheduler = Arc::clone(self.gpu_schedulers.get(device).ok_or_else(|| {
            JammiError::Config(format!(
                "device {device} has no admission budget in this session: it is not one of \
                 the configured [gpu] devices {:?}",
                self.device_config.devices
            ))
        })?);
        let id = CacheKey {
            model_id: ModelId::from(source),
            device,
            task: Some(task),
        };

        self.models
            .get_or_compute(&id, || async {
                match source {
                    ModelSource::Remote(name) => self.load_remote(source, name, task).await,
                    ModelSource::HuggingFace(_) | ModelSource::Local(_) => {
                        self.do_load(&id, &scheduler, source, task).await
                    }
                }
            })
            .await
    }

    /// Describe a model without materializing it: everything planning its
    /// run needs — the identity a materialization records, its output
    /// width, its regression head's form — read from the resolved files
    /// and configuration, for this deployment's primary device. A submitter
    /// that places a plan elsewhere reads this and never holds the
    /// weights; the description it plans against is the one a later load
    /// on any process materializes from, so the two cannot disagree.
    ///
    /// Memoized per [`CacheKey`] with the same bounded-staleness contract
    /// as [`Self::get_or_load`]: the files are re-`stat`ed on every call
    /// and the description recomputed when they changed.
    pub async fn describe(
        &self,
        source: &ModelSource,
        task: ModelTask,
    ) -> Result<Arc<ModelDescription>> {
        self.describe_on(self.gpu_schedulers.primary(), source, task)
            .await
    }

    /// [`Self::describe`] for a NAMED device of this deployment — the
    /// precision the configuration resolves is that device's. A device the
    /// deployment never declared is refused.
    pub async fn describe_on(
        &self,
        device: i32,
        source: &ModelSource,
        task: ModelTask,
    ) -> Result<Arc<ModelDescription>> {
        let device_config = self.device_config.for_device(device)?;
        if let ModelSource::Remote(name) = source {
            return self.describe_remote(source, self.remote_endpoint(name)?, task);
        }
        let id = CacheKey {
            model_id: ModelId::from(source),
            device,
            task: Some(task),
        };
        let resolved = self.resolver.resolve(source, task).await?;
        self.describe_resolved(&id, &resolved, &device_config).await
    }

    /// The memoized description of `id`: `resolved` described for
    /// `device_config`, computed once for every concurrent caller of the
    /// same key and reused by [`Self::do_load`], so the content digest is
    /// hashed once per resolved directory.
    async fn describe_resolved(
        &self,
        id: &CacheKey,
        resolved: &ResolvedModel,
        device_config: &DeviceConfig,
    ) -> Result<Arc<ModelDescription>> {
        self.descriptions
            .get_or_compute(id, || async {
                self.backend.describe(resolved, device_config).map(Arc::new)
            })
            .await
    }

    /// TEST-ONLY: the models resident in this process right now, one id per
    /// model this cache holds weights for, on any device — what a proof that
    /// planning materialized nothing reads. Not used by any production path.
    #[doc(hidden)]
    pub async fn resident_models_for_test(&self) -> Vec<ModelId> {
        let models = self.models.read().await;
        let mut ids: Vec<ModelId> = models.keys().map(|k| k.model_id.clone()).collect();
        ids.sort_by(|a, b| a.0.cmp(&b.0));
        ids.dedup();
        ids
    }

    /// TEST-ONLY: resolve and load a fresh, UNSHARED [`LoadedModel`] off the
    /// resolver + backend, bypassing the shared LRU cache entirely. The shared
    /// cache hands out `Arc<LoadedModel>` (no `&mut`), so a test that needs to
    /// mutate a model — e.g. the regression non-vacuity guard zeroing the trained
    /// distribution head via
    /// [`LoadedModel::zero_distribution_head_for_test`] — must own it. This goes
    /// through the same resolve + describe + materialize path serving uses, so
    /// the owned model is byte-identical to what `get_or_load` would cache. Not
    /// used by any production path.
    #[doc(hidden)]
    pub async fn load_owned_for_test(
        &self,
        source: &ModelSource,
        task: ModelTask,
    ) -> Result<LoadedModel> {
        let resolved = self.resolver.resolve(source, task).await?;
        self.backend.load(&resolved, &self.device_config)
    }

    /// Complete a generic (plain local/HuggingFace, or `"embedding"`
    /// FK-placeholder) catalog row's registration after a successful load.
    /// Store the parent directory of the first weights file so that
    /// `build_encoder_adapters` can locate config.json and tokenizer.json.
    ///
    /// Split out of [`Self::do_load`] so this catalog-only read/write
    /// mechanism can be driven directly by a test — independent of an
    /// actual model resolve/load — for both the type-gate and the
    /// read-failure fail-closed behaviour below.
    ///
    /// This bookkeeping write must never touch a catalog row some other
    /// producer owns. `source_str` can also name a fine-tuned model, an epoch
    /// checkpoint, or a context-predictor — `ModelSource::parse`'s fallback
    /// maps any string without a `local:`/`file://` prefix to `HuggingFace`,
    /// so a fine-tuned id like `jammi:fine-tuned:{uuid}` parses exactly like
    /// a real HF Hub repo id would. The catalog itself refuses to re-register
    /// a row that references an artifact (`Catalog::register_model`), but a
    /// directly-registered row of another kind has no such backstop:
    /// `model_type` and `base_model_id` are unconditional in
    /// `register_model`'s `ON CONFLICT` clause, and a non-null
    /// `external_location` wins the `COALESCE`, so registering over it with
    /// `model_type: "huggingface"`, `base_model_id: None`, and this resolve's
    /// underlying BASE weights directory would lose its type, its lineage
    /// and its own location.
    ///
    /// `model_type` is an open TEXT domain: `"fine-tuned"`,
    /// `"context-predictor"`, `"bert"`, `"distilbert"`, `"modernbert"`,
    /// `"open_clip"`, `"clap_audio_model"` and any future architecture id
    /// `EncoderFamily::adapter_model_type` mints all live in this same
    /// column. A DENYLIST of the terminal types to protect fails open on
    /// every one of those: an unenumerated type (or a typo, or a type this
    /// crate has not been taught about yet) falls through and gets
    /// rewritten. So this is an ALLOWLIST of the generic, non-terminal
    /// kinds this call exists to COMPLETE, never a denylist of the
    /// terminal ones to protect: only a plain `"local"`/`"huggingface"` row
    /// (this call's own prior write, safe to refresh idempotently) or an
    /// `"embedding"` placeholder row (the FK-satisfying pre-registration
    /// `Session::submit_fine_tune_spec`/`ContextPredictor` write before the
    /// base model is ever loaded, always with no location, meant to be
    /// completed by this exact call) — or no row at all yet — may be
    /// written here. Every other type, enumerated or not, is left
    /// untouched; this call's own params already never carry a
    /// `base_model_id` or a location other than the ones it
    /// produces itself, so completing one of these rows can never clobber
    /// a value some other producer wrote.
    async fn complete_generic_registration(
        &self,
        source: &ModelSource,
        source_str: &str,
        backend: ModelBackendKind,
        location: Option<&str>,
        task: ModelTask,
    ) {
        const GENERIC_COMPLETABLE_TYPES: &[&str] = &["local", "huggingface", "remote", "embedding"];
        // A catalog READ error is not "no row" — collapsing it to `None`
        // would fall through to the write below and could clobber a row this
        // call never actually inspected. This bookkeeping is best-effort (a `register_model`
        // failure already only `warn!`s and keeps serving), so a read failure fails closed:
        // skip the write entirely rather than guess the row is absent.
        match self
            .resolver
            .catalog()
            .get_model_version(source_str, 1)
            .await
        {
            Err(e) => {
                tracing::warn!(
                    model_id = %source_str,
                    "Failed to read catalog row before load bookkeeping ({e}); skipping \
                     best-effort registration rather than writing over a row this call \
                     could not inspect"
                );
            }
            Ok(existing) => {
                let can_complete = existing
                    .as_ref()
                    .is_none_or(|r| GENERIC_COMPLETABLE_TYPES.contains(&r.model_type.as_str()));
                if !can_complete {
                    tracing::debug!(
                        model_id = %source_str,
                        model_type = existing.as_ref().map(|r| r.model_type.as_str()).unwrap_or(""),
                        "skipping generic load-bookkeeping registration: this id is already a \
                         catalog-managed record of a different kind"
                    );
                } else {
                    let model_type = match source {
                        ModelSource::HuggingFace(_) => "huggingface",
                        ModelSource::Local(_) => "local",
                        ModelSource::Remote(_) => "remote",
                    };
                    if let Err(e) = self
                        .resolver
                        .catalog()
                        .register_model(RegisterModelParams {
                            model_id: source_str,
                            version: 1,
                            model_type,
                            backend,
                            task,
                            base_model_id: None,
                            external_location: location,
                            config_json: None,
                        })
                        .await
                    {
                        tracing::warn!(
                            model_id = %source_str,
                            "Failed to register model in catalog: {e}"
                        );
                    }
                }
            }
        }
    }

    /// Resolve, describe, admit and materialize `id`: the entry the model
    /// memo inserts and hands the first guard out of.
    async fn do_load(
        &self,
        id: &CacheKey,
        gpu_scheduler: &Arc<GpuScheduler>,
        source: &ModelSource,
        task: ModelTask,
    ) -> Result<CacheEntry> {
        // This load's device, as the backend sees it. Built here rather than
        // at the (warm-hit) entry point: it owns a `Vec`, and the cold path
        // is the only one that needs it.
        let device_config = self.device_config.for_device(id.device)?;
        let resolved = self.resolver.resolve(source, task).await?;
        let source_str = source.to_string();
        // Describe before admitting: the description is what the load
        // materializes from, and a submitter that already described this
        // model has it memoized — the digest is never hashed twice.
        let description = self
            .describe_resolved(id, &resolved, &device_config)
            .await?;
        let memory_bytes = self.backend.estimate_memory(&resolved);

        // A stale-fingerprint reload can transiently need this model's
        // budget TWICE — the memo already removed the stale `CacheEntry`
        // (so `evict_one` here can never find it again), but its
        // `Arc<GpuPermit>` clone stays outstanding for as long as ANY live
        // `ModelGuard` from before the mutation is still held — the
        // reservation is not released until that guard drops. Under a budget realistically sized to
        // one resident copy of this model, `evict_one` therefore finds nothing evictable even
        // though the request is perfectly satisfiable — just not yet. Distinguish
        // that from a genuinely unsatisfiable request (more bytes than the
        // scheduler could EVER admit, evictions or waiting or not) via
        // `GpuScheduler::usable_capacity` — the one case a hard error is
        // still honest, since no amount of waiting would ever succeed.
        //
        // Waiting on `GpuScheduler::acquire` alone is not enough: it only
        // ever waits on `GpuScheduler`'s own release notify and cannot
        // observe `evict_one`'s eligibility condition. Consider: budget
        // sized to one resident copy; A holds M1's
        // guard; B's `do_load` (loading M2) fails `try_acquire`, finds M1
        // NOT evictable (`ref_count == 1`), and would fall back to
        // `GpuScheduler::acquire`'s wait. A then drops its guard: per
        // `ModelGuard::drop`'s ordering, the permit clone releases BEFORE
        // the `ref_count` decrement — but M1's `CacheEntry` is still present
        // in the cache and retains its OWN clone of the SAME `Arc<GpuPermit>`,
        // so dropping A's clone only lowers the
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
        //      `evict_if_current`/stale-reload case) and `evict_one`'s own
        //      removal of an
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
            let permit_released = gpu_scheduler.notify.notified();
            tokio::pin!(permit_released);
            permit_released.as_mut().enable();
            let entry_became_idle = self.admission_notify.notified();
            tokio::pin!(entry_became_idle);
            entry_became_idle.as_mut().enable();

            if let Some(permit) = gpu_scheduler.try_acquire(memory_bytes) {
                break permit;
            }
            // The budget this loop is waiting on is `id.device`'s, so only a
            // copy resident on `id.device` can release it.
            if let Some((evicted, entry)) = self
                .models
                .evict_one(|resident| resident.device == id.device)
                .await
            {
                tracing::info!(
                    model_id = %evicted.model_id.0,
                    device = evicted.device,
                    bytes = entry.memory_bytes,
                    "Evicted model from cache"
                );
                continue;
            }
            if memory_bytes > gpu_scheduler.usable_capacity() {
                return Err(JammiError::Model {
                    model_id: source_str,
                    message: format!(
                        "Cannot acquire GPU memory: {memory_bytes} bytes requested exceeds \
                         the total usable GPU budget of {} bytes — no amount of eviction or \
                         waiting could ever satisfy this request",
                        gpu_scheduler.usable_capacity()
                    ),
                });
            }
            tokio::select! {
                _ = permit_released => {}
                _ = entry_became_idle => {}
            }
        };

        let loaded = self
            .backend
            .materialize(&resolved, description, &device_config)?;

        // Register model in catalog (idempotent — ignores if already registered).
        // See `Self::complete_generic_registration`'s own doc for which rows
        // this call may write.
        self.complete_generic_registration(
            source,
            &source_str,
            ModelBackendKind::Candle,
            resolved.weights_dir(),
            task,
        )
        .await;

        // The permit is `Arc`-shared between this `CacheEntry` and every
        // guard handed out of it — see `ModelGuard::gpu_permit`'s doc for why.
        Ok(CacheEntry {
            model: Arc::new(loaded),
            ref_count: Arc::new(AtomicUsize::new(0)),
            memory_bytes,
            gpu_permit: Arc::new(gpu_permit),
            admission_notify: Arc::clone(&self.admission_notify),
        })
    }

    /// Describe and admit a declared remote model: the entry holds no
    /// device memory, and its guard's forwards share the endpoint's
    /// admission.
    async fn load_remote(
        &self,
        source: &ModelSource,
        name: &str,
        task: ModelTask,
    ) -> Result<CacheEntry> {
        let endpoint = self.remote_endpoint(name)?;
        let description = self.describe_remote(source, endpoint, task)?;
        let gpu_permit = endpoint.admission.reserve_nothing();
        self.complete_generic_registration(
            source,
            &source.to_string(),
            ModelBackendKind::Remote,
            None,
            task,
        )
        .await;
        Ok(CacheEntry {
            model: Arc::new(LoadedModel::remote(
                Arc::clone(&endpoint.model),
                description,
            )),
            ref_count: Arc::new(AtomicUsize::new(0)),
            memory_bytes: 0,
            gpu_permit: Arc::new(gpu_permit),
            admission_notify: Arc::clone(&self.admission_notify),
        })
    }

    /// Preload a model without running inference — the server's
    /// warm-before-ready step (`[server] preload_models`) and the Python
    /// `preload_model` verb.
    pub async fn preload(&self, source: &ModelSource, task: ModelTask) -> Result<()> {
        #[cfg(feature = "test-hooks")]
        preload_test_hooks::maybe_park(
            &source.to_string(),
            preload_test_hooks::ParkPoint::BeforeLoad,
        )
        .await;
        let guard = self.get_or_load(source, task).await?;
        drop(guard);
        Ok(())
    }
}

// ── The cache key: two devices are two resident copies, not one ────────────

#[cfg(test)]
mod cache_key_tests {
    use std::collections::HashMap;

    use super::super::memo::MemoState;
    use super::*;

    /// A load in flight for device 0 is not a load in flight for device 1:
    /// keyed by the id alone, the second caller would wait on the first's
    /// load and then be handed a copy resident on the wrong card.
    ///
    /// Hermetic mechanism: two CPU "devices" are indistinguishable at
    /// execution, so the oracle drives the `device` COMPONENT of the key
    /// with two distinct values, which is exactly the quantity a two-card
    /// deployment varies. Asserted on the PRODUCTION single-flight map, the
    /// one `get_or_load` reads and writes.
    #[test]
    fn two_devices_hold_two_single_flight_slots_for_one_model_id() {
        let first = CacheKey::for_test("tiny-bert", 0);
        let second = CacheKey::for_test("tiny-bert", 1);
        assert_eq!(
            first.model_id, second.model_id,
            "the control: it is ONE model id, so an id-keyed map would hold one slot"
        );
        assert_ne!(first, second);

        let mut state = MemoState::<CacheKey, CacheEntry>::default();
        let in_flight = state.in_flight_for_test();
        in_flight.insert(first.clone(), Arc::new(tokio::sync::Notify::new()));
        in_flight.insert(second.clone(), Arc::new(tokio::sync::Notify::new()));
        assert_eq!(
            in_flight.len(),
            2,
            "the single-flight map holds one slot per device, not one per model id"
        );
    }

    /// The PRODUCTION entries map, and its recency order, hold one entry per
    /// device for one model id.
    ///
    /// Asserted on the model memo's own state, the map `get_or_load` reads and
    /// writes, and not on a throwaway `HashMap` built beside it: a mirror
    /// keyed by [`CacheKey`] would report two entries however the real map
    /// were keyed, so it measures this test's own construction rather than
    /// the cache's.
    ///
    /// A `CacheEntry` needs a really-loaded model, so this oracle loads one
    /// and shares the handle between the two entries — the model object is
    /// not what is under test; the key is. The two entries are told apart by
    /// their `memory_bytes`, which is per-entry state a collision would
    /// destroy.
    #[tokio::test]
    async fn the_entries_map_holds_one_entry_per_device_for_one_model_id() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(
            jammi_db::catalog::Catalog::open(catalog_dir.path())
                .await
                .unwrap(),
        );
        let cache_dir = tempfile::tempdir().unwrap().keep();
        let store = Arc::new(
            jammi_db::store::ArtifactStore::with_root(
                jammi_db::storage::StorageUrl::memory("cache-key-test-artifacts"),
                jammi_db::storage::StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        );
        let hub = crate::model::hub::HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(tempfile::tempdir().unwrap().keep()),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap();
        let resolver = ModelResolver::new(Arc::clone(&catalog), store, hub).unwrap();

        let dir = tmp.path().join("tiny_bert");
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(file), dir.join(file)).unwrap();
        }
        let source = ModelSource::local(&dir);

        let device_config = DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        };
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let loader = ModelCache::new(resolver, device_config, Arc::clone(&scheduler));
        let guard = loader
            .get_or_load(&source, ModelTask::TextEmbedding)
            .await
            .unwrap();
        let model = Arc::clone(&guard.model);
        drop(guard);

        let first = CacheKey::for_test("tiny-bert", 0);
        let second = CacheKey::for_test("tiny-bert", 1);
        assert_eq!(
            first.model_id, second.model_id,
            "the control: it is ONE model id, so an id-keyed map would hold one entry"
        );

        let mut state = MemoState::<CacheKey, CacheEntry>::default();
        for (id, memory_bytes) in [(&first, 11usize), (&second, 22)] {
            state.insert_for_test(
                id.clone(),
                CacheEntry::for_test(
                    &model,
                    Arc::new(scheduler.try_acquire(memory_bytes).unwrap()),
                    memory_bytes,
                ),
            );
        }

        assert_eq!(
            state.len(),
            2,
            "two devices are two resident copies of one model id, in the cache's own map"
        );
        assert_eq!(
            state.get(&first).map(|e| e.memory_bytes),
            Some(11),
            "device 0's entry must still be device 0's"
        );
        assert_eq!(
            state.get(&second).map(|e| e.memory_bytes),
            Some(22),
            "device 1's entry must not have overwritten device 0's"
        );

        // The recency order tracks both copies, and touching one does not
        // move the other.
        state.touch_for_test(&first);
        assert_eq!(
            state.keys().cloned().collect::<Vec<_>>(),
            vec![second.clone(), first.clone()],
            "touching device 0's copy must not move device 1's"
        );
    }

    /// `None` is a distinct key VALUE, never a wildcard: a key that names no
    /// task neither matches nor is matched by one that does, and two named
    /// tasks are two keys.
    #[test]
    fn an_unnamed_task_is_its_own_key_never_a_wildcard() {
        let unnamed = CacheKey::for_test("tiny-bert", 0);

        let with_task = CacheKey {
            task: Some(ModelTask::TextEmbedding),
            ..unnamed.clone()
        };
        let other_task = CacheKey {
            task: Some(ModelTask::Classification),
            ..unnamed.clone()
        };

        let mut map: HashMap<CacheKey, &str> = HashMap::new();
        for (key, label) in [
            (unnamed.clone(), "named none"),
            (with_task.clone(), "named a task"),
            (other_task.clone(), "named another task"),
        ] {
            map.insert(key, label);
        }
        assert_eq!(map.len(), 3, "three distinct keys, three entries");
        assert_eq!(map.get(&unnamed), Some(&"named none"));
        assert_eq!(map.get(&with_task), Some(&"named a task"));
        assert_eq!(map.get(&other_task), Some(&"named another task"));
    }

    /// A device the deployment never declared has no budget and no cache
    /// slot: naming it is a typed refusal, not a load onto the primary.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_device_this_deployment_never_declared_is_refused() {
        let config = DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        };
        assert!(config.for_device(-1).is_ok());
        let error = config
            .for_device(3)
            .expect_err("device 3 was never declared by this deployment");
        assert!(
            error.to_string().contains("not one of the configured"),
            "unexpected message: {error}"
        );
    }
}

// ── `evict_one`'s "true means real progress" contract holds even while a \
//    concurrent fast-path probe is racing it ──

#[cfg(test)]
mod f3_prime_tests {
    use super::super::memo::MemoState;
    use super::*;
    use std::sync::Arc;

    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    use crate::model::hub::HubSource;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
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

    fn test_hub_source() -> HubSource {
        let root = tempfile::tempdir().unwrap().keep();
        HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(root),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap()
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

    /// A GPU budget that fits exactly ONE `tiny_bert`-sized model forces
    /// `do_load`'s admission loop to call `evict_one` for a second, distinct
    /// model while a fast-path `get_or_load` on the FIRST (idle,
    /// `ref_count == 0`) model is deterministically paused between its
    /// snapshot and its `probe_freshness` call.
    ///
    /// The paused snapshot holds no permit clone (that is taken only in the
    /// write-lock-protected re-validate branch), so `evict_one`'s removal of
    /// A drops A's only permit clone and really decrements
    /// `reserved_memory` — B's `try_acquire` succeeds on the next attempt.
    /// A snapshot that held its own clone would leave the reservation
    /// booked behind `evict_one`'s `true`, and B's load would fail with
    /// "nothing to evict".
    #[tokio::test]
    async fn evict_one_true_is_always_real_progress_under_a_racing_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();

        let (source_a, weights_len) = tiny_bert_source(tmp.path(), "model_a");
        let (source_b, weights_len_b) = tiny_bert_source(tmp.path(), "model_b");
        assert_eq!(
            weights_len, weights_len_b,
            "both fixtures are copies of the same tiny_bert weights file"
        );

        // Budget fits EXACTLY one model — B's admission loop MUST evict A
        // to succeed; there is no slack that could mask the bug by simply
        // admitting B without ever calling `evict_one`.
        let scheduler = Arc::new(GpuScheduler::new(weights_len));
        let cache = Arc::new(ModelCache::new(resolver, device_config(), scheduler));

        // (1) Load A, then drop the guard: A is warm, resident, and IDLE
        // (`ref_count == 0` AND its permit's only remaining clone is the
        // `CacheEntry`'s own, i.e. `Arc::strong_count(&gpu_permit) == 1`) —
        // `evict_one`'s eligibility condition.
        let guard_a = cache
            .get_or_load(&source_a, ModelTask::TextEmbedding)
            .await
            .unwrap();
        drop(guard_a);

        // (2) Install the probe pause and immediately start a warm-hit
        // `get_or_load` on A: it snapshots A's entry (ref_count == 0, no
        // permit clone), then pauses HERE — before `probe_freshness`,
        // before the `ref_count` increment, before any permit clone.
        let pause = cache.install_probe_pause();
        let cache_for_a = Arc::clone(&cache);
        let source_a_for_task = source_a.clone();
        let task_a = tokio::spawn(async move {
            cache_for_a
                .get_or_load(&source_a_for_task, ModelTask::TextEmbedding)
                .await
        });

        // Deterministic rendezvous: wait until task_a has actually reached
        // the pause point (no sleep-based guess).
        pause.arrived.notified().await;

        // (3) With A's fast-path task paused in the pre-ref_count-increment
        // window, drive B's load — under this budget it MUST evict
        // something, and A is the only idle entry.
        let guard_b = cache.get_or_load(&source_b, ModelTask::TextEmbedding).await;
        let guard_b = match guard_b {
            Ok(g) => g,
            Err(e) => panic!(
                "B's load must succeed: evict_one's `true` for evicting idle model \
                 A must correspond to REAL freed memory, even while A's fast-path \
                 probe is paused mid-flight holding no premature permit clone \
                 — got Err({e})"
            ),
        };
        // B is genuinely resident and the sole occupant of the 1-model
        // budget — no room left.
        assert_eq!(
            cache
                .schedulers()
                .get(cache.device_config.gpu_device)
                .expect("the primary device has a budget")
                .available(),
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

    /// `evict_one` must not claim progress for a `ref_count == 0` entry
    /// whose `gpu_permit` still has an outstanding clone. Rather than
    /// reproducing an ordering race through the async cache API, this test
    /// drives `MemoState::evict_one` DIRECTLY against a hand-built
    /// `CacheEntry` whose permit has a second, test-held clone —
    /// deterministically constructing the state such a race would leave
    /// behind, with no timing dependency.
    ///
    /// Because `evict_one` also gates on
    /// `Arc::strong_count(&entry.gpu_permit) == 1`, the misleading entry is
    /// skipped, the genuinely idle one is evicted instead, and once the
    /// outstanding clone is dropped the remaining entry is evicted too.
    ///
    /// It runs against a BUDGETED [`GpuScheduler::new`] sized to exactly fit
    /// the real load plus X's and Y's one-byte permits, with no slack — not
    /// [`GpuScheduler::new_unlimited`], whose `GpuPermit::drop` is a no-op
    /// and whose [`GpuScheduler::available`] never moves — and asserts
    /// [`GpuScheduler::available`] directly at each step: unmoved while
    /// `evict_one` claims no progress for the misleading entry, and moved by
    /// exactly the freed amount when it evicts a genuinely idle one.
    #[tokio::test]
    async fn evict_one_does_not_claim_progress_for_an_entry_whose_permit_has_an_outstanding_clone()
    {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();
        let (source, weights_len) = tiny_bert_source(tmp.path(), "model_x");

        // A real, loaded `Arc<LoadedModel>` — only used as a valid handle
        // for the hand-built `CacheEntry`s below; the model's own state is
        // irrelevant to this test. Budgeted with NO slack beyond the real
        // load's own weight (`weights_len`) plus X's and Y's one-byte
        // synthetic permits below, so every `available()` assertion is
        // exact, not merely directionally suggestive.
        let scheduler = Arc::new(GpuScheduler::new(weights_len + 2));
        let cache = ModelCache::new(resolver, device_config(), Arc::clone(&scheduler));
        let guard = cache
            .get_or_load(&source, ModelTask::TextEmbedding)
            .await
            .unwrap();
        let model = Arc::clone(&guard.model);
        drop(guard);

        // Entry X: `ref_count == 0` (the naive "idle" signal) but its
        // `gpu_permit` has a SECOND outstanding clone.
        // `evict_one` must not remove this entry and must not count it as
        // progress.
        let permit_x = Arc::new(scheduler.try_acquire(1).unwrap());
        let outstanding_clone = Arc::clone(&permit_x);
        let id_x = CacheKey::for_test("model_x", -1);

        // Entry Y: genuinely idle — `ref_count == 0` AND its permit's only
        // clone is this entry's own.
        let permit_y = Arc::new(scheduler.try_acquire(1).unwrap());
        let id_y = CacheKey::for_test("model_y", -1);

        // X is scanned before Y — the scan must SKIP X and land on Y rather
        // than stopping at the first misleading candidate.
        let mut inner = MemoState::<CacheKey, CacheEntry>::default();
        inner.insert_for_test(id_x.clone(), CacheEntry::for_test(&model, permit_x, 1));
        inner.insert_for_test(id_y.clone(), CacheEntry::for_test(&model, permit_y, 1));

        // The budget has NO slack: the real load (`weights_len`) plus X's
        // and Y's one-byte permits exactly exhaust it.
        assert_eq!(
            scheduler.available(),
            0,
            "sanity: the budget is fully reserved before any eviction"
        );

        assert!(
            inner.evict_one(|k| k.device == -1).is_some(),
            "Y is genuinely idle (no outstanding permit clone) — evict_one \
             must find and remove it, skipping past the misleading X"
        );
        assert!(
            inner.get(&id_x).is_some(),
            "X must NOT have been removed: its permit still has an \
             outstanding clone, so evicting it would not have released \
             real memory"
        );
        assert!(
            inner.get(&id_y).is_none(),
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
            inner.evict_one(|k| k.device == -1).is_none(),
            "evict_one claimed progress for the sole remaining entry even \
             though its permit clone is still outstanding — removing X here would not decrement \
             GpuScheduler::reserved_memory because `outstanding_clone` is \
             still alive"
        );
        assert!(inner.get(&id_x).is_some());
        assert_eq!(
            scheduler.available(),
            1,
            "evict_one's false claim of no progress must correspond to \
             GpuScheduler::available() genuinely NOT moving"
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
            inner.evict_one(|k| k.device == -1).is_some(),
            "once the outstanding clone is gone, X is genuinely idle and \
             evict_one must now claim (and deliver) real progress"
        );
        assert!(inner.get(&id_x).is_none());
        assert_eq!(
            scheduler.available(),
            2,
            "the final evict_one — now genuinely idle — must ACTUALLY \
             release X's reserved byte too, matching its claimed progress"
        );
    }

    /// `[inference] max_loaded_models` bounds the models a cache keeps: past
    /// it the least recently used idle model is evicted, and a model in use
    /// is kept however far past the bound that takes the count.
    #[tokio::test]
    async fn max_loaded_models_evicts_the_oldest_idle_model_and_never_one_in_use() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();
        let (source_a, _) = tiny_bert_source(tmp.path(), "model_a");
        let (source_b, _) = tiny_bert_source(tmp.path(), "model_b");
        let (source_c, _) = tiny_bert_source(tmp.path(), "model_c");

        let cache = ModelCache::new(
            resolver,
            device_config(),
            Arc::new(GpuScheduler::new_unlimited()),
        )
        .bounded(CacheBounds {
            loaded_models: std::num::NonZeroUsize::new(1),
            described_models: None,
        });
        let resident = || async {
            cache
                .resident_models_for_test()
                .await
                .into_iter()
                .map(|id| id.0)
                .collect::<Vec<_>>()
        };

        let guard_a = cache
            .get_or_load(&source_a, ModelTask::TextEmbedding)
            .await
            .unwrap();
        let guard_b = cache
            .get_or_load(&source_b, ModelTask::TextEmbedding)
            .await
            .unwrap();
        assert_eq!(
            resident().await.len(),
            2,
            "both models are in use, so neither is evicted past the bound"
        );

        drop(guard_a);
        drop(guard_b);
        drop(
            cache
                .get_or_load(&source_c, ModelTask::TextEmbedding)
                .await
                .unwrap(),
        );
        assert_eq!(
            resident().await,
            vec![source_c.to_string()],
            "once idle, the older models are evicted down to the bound"
        );
    }

    /// Eviction is scoped to the device whose admission is under pressure.
    ///
    /// Admission is per device — `DeviceSchedulers` holds one budget per
    /// card and a reservation on one is invisible to the other — so removing
    /// a resident copy from device 1 releases device 1's budget and not one
    /// byte of device 0's. A device-blind LRU scan answers device 0's
    /// pressure with device 1's oldest idle entry, reports `true`, sends
    /// `do_load`'s admission loop round again against an unchanged budget,
    /// and repeats: device 1's cache is emptied and device 0 is exactly
    /// where it started.
    ///
    /// The LRU deliberately leads with device 1's entries, so a scan that
    /// merely happened to reach device 0's first would not pass.
    #[tokio::test]
    async fn evict_one_frees_the_admitting_device_and_never_another() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();
        let (source, _weights_len) = tiny_bert_source(tmp.path(), "model_a");

        // A real `Arc<LoadedModel>`, used only as a valid handle for the
        // hand-built entries below; the cache it comes from is unbudgeted
        // and plays no part in the accounting under test.
        let loader = ModelCache::new(
            resolver,
            device_config(),
            Arc::new(GpuScheduler::new_unlimited()),
        );
        let guard = loader
            .get_or_load(&source, ModelTask::TextEmbedding)
            .await
            .unwrap();
        let model = Arc::clone(&guard.model);
        drop(guard);

        // One budget per device, each with exactly one byte of slack beyond
        // what its resident copies hold.
        let budget_0 = Arc::new(GpuScheduler::new(1));
        let budget_1 = Arc::new(GpuScheduler::new(2));

        let on_device_0 = CacheKey::for_test("model_a", 0);
        let older_on_device_1 = CacheKey::for_test("model_b", 1);
        let newer_on_device_1 = CacheKey::for_test("model_c", 1);

        let mut inner = MemoState::<CacheKey, CacheEntry>::default();
        for (id, scheduler) in [
            (&older_on_device_1, &budget_1),
            (&newer_on_device_1, &budget_1),
            (&on_device_0, &budget_0),
        ] {
            // Device 1's two copies are the OLDEST entries in the LRU.
            inner.insert_for_test(
                id.clone(),
                CacheEntry::for_test(&model, Arc::new(scheduler.try_acquire(1).unwrap()), 1),
            );
        }
        assert_eq!(budget_0.available(), 0, "sanity: device 0 is full");
        assert_eq!(budget_1.available(), 0, "sanity: device 1 is full");

        // Device 0 is the one under pressure.
        assert!(
            inner.evict_one(|k| k.device == 0).is_some(),
            "device 0 holds an idle entry, so there is real progress to make"
        );
        assert!(
            inner.get(&on_device_0).is_none(),
            "the entry evicted for device 0's pressure must be device 0's own"
        );
        assert!(
            inner.get(&older_on_device_1).is_some() && inner.get(&newer_on_device_1).is_some(),
            "device 1's resident copies are not device 0's to spend"
        );
        assert_eq!(
            budget_0.available(),
            1,
            "the eviction must release the budget of the device that was short of it"
        );
        assert_eq!(
            budget_1.available(),
            0,
            "device 1's budget must be exactly where it started"
        );

        // Device 0 now has nothing idle. The honest answer is "no progress"
        // — NOT device 1's oldest entry, which would free nothing for
        // device 0 while emptying another card's cache.
        assert!(
            inner.evict_one(|k| k.device == 0).is_none(),
            "nothing device 0 holds can be evicted, and nothing another device holds would help"
        );
        assert_eq!(inner.len(), 2, "device 1's entries are untouched");
        assert!(
            inner.keys().all(|id| id.device == 1),
            "the LRU still tracks exactly device 1's two copies"
        );

        // Device 1's own pressure evicts device 1's oldest, and only it.
        assert!(inner.evict_one(|k| k.device == 1).is_some());
        assert!(
            inner.get(&older_on_device_1).is_none() && inner.get(&newer_on_device_1).is_some(),
            "within a device the scan is still oldest-first"
        );
        assert_eq!(budget_1.available(), 1);
    }
}

// ── Single-flight: a registered waiter never loses its wakeup ──

#[cfg(test)]
mod single_flight_tests {
    use super::*;
    use std::time::Duration;

    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn test_artifact_store() -> Arc<ArtifactStore> {
        let cache_dir = tempfile::tempdir().unwrap().keep();
        Arc::new(
            ArtifactStore::with_root(
                StorageUrl::memory("single-flight-test-artifacts"),
                StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        )
    }

    fn test_hub_source() -> crate::model::hub::HubSource {
        let root = tempfile::tempdir().unwrap().keep();
        crate::model::hub::HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(root),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap()
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

    /// A waiter that has genuinely REGISTERED (its `Notified` future
    /// `enable()`d) before the loader removes its `in_flight` entry and
    /// calls `notify_waiters` must always wake — even when the loader's
    /// completion (simulated directly here, bypassing `do_load`, for full
    /// determinism) lands exactly inside the pause window between this
    /// task's registration and its `.await`.
    ///
    /// A lost wakeup hangs the waiter forever (there is no timeout in
    /// `get_or_load`), so the test bounds the wait with a generous timeout
    /// to fail with a clear message rather than hang CI.
    #[tokio::test]
    async fn registered_waiter_always_wakes_even_if_notify_races_the_pause() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let cache = Arc::new(ModelCache::new(resolver, device_config(), scheduler));

        let source = tiny_bert_source(tmp.path(), "single_flight_model");
        let id = CacheKey {
            model_id: ModelId::from(&source),
            device: cache.device_config.gpu_device,
            task: Some(ModelTask::TextEmbedding),
        };

        // Simulate "another task is already loading this id" directly,
        // bypassing `do_load` entirely — the ONLY state `get_or_load`'s
        // single-flight branch actually observes.
        let loader_notify = Arc::new(tokio::sync::Notify::new());
        cache
            .models
            .write_for_test()
            .await
            .in_flight_for_test()
            .insert(id.clone(), Arc::clone(&loader_notify));

        let pause = cache.install_single_flight_pause();
        let cache_for_waiter = Arc::clone(&cache);
        let source_for_waiter = source.clone();
        let waiter = tokio::spawn(async move {
            cache_for_waiter
                .get_or_load(&source_for_waiter, ModelTask::TextEmbedding)
                .await
        });

        // Deterministic rendezvous: the waiter has registered and reached
        // the pause.
        pause.arrived.notified().await;

        // Simulate the loader completing WHILE the waiter is paused —
        // exactly the lost-wakeup window.
        cache
            .models
            .write_for_test()
            .await
            .in_flight_for_test()
            .remove(&id);
        loader_notify.notify_waiters();

        // Release the waiter: it must wake immediately (already
        // registered), see `in_flight` empty, and become the loader
        // itself, completing the real load.
        pause.release.notify_one();

        let joined = tokio::time::timeout(Duration::from_secs(10), waiter).await;
        let result = match joined {
            Ok(joined) => joined.unwrap(),
            Err(_) => panic!(
                "the waiter never woke within 10s — a lost wakeup would hang \
                 here forever; this bound turns it into a fast, clear test \
                 failure instead of a hang"
            ),
        };
        if let Err(e) = result {
            panic!("the waiter must complete its (now solo) load successfully after waking, got Err({e})");
        }
    }
}

// ── A deleted or restored tokenizer.json never permanently wedges \
//    `get_or_load` ──

#[cfg(test)]
mod r5_f1_tokenizer_tests {
    use super::*;

    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
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

    fn test_hub_source() -> crate::model::hub::HubSource {
        let root = tempfile::tempdir().unwrap().keep();
        crate::model::hub::HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(root),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap()
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

    /// End-to-end at the `ModelCache::get_or_load` level: a warm model whose
    /// `tokenizer.json` is deleted from its live directory must
    /// stale-reload, never wedge.
    ///
    /// The tokenizer candidate is `optional: true` (every resolver path
    /// re-derives `tokenizer: None` on absence and `CandleBackend::load`
    /// accepts it), so the first post-deletion `get_or_load` evicts the
    /// stale entry and reloads with `tokenizer: None` — exactly what a cold
    /// process loading the same directory does — and the second call is an
    /// ordinary warm hit against the reloaded entry. Were the tokenizer
    /// required, the probe would `Err` on every call and the entry would
    /// never serve again.
    #[tokio::test]
    async fn tokenizer_deleted_after_load_stale_reloads_never_wedges() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();
        let scheduler = Arc::new(GpuScheduler::new_unlimited());
        let cache = ModelCache::new(resolver, device_config(), scheduler);

        let source = tiny_bert_source(tmp.path(), "tokenizer_wedge_model");

        // (1) Cold load: tokenizer.json present, resolves to `Some(..)`.
        let guard = cache
            .get_or_load(&source, ModelTask::TextEmbedding)
            .await
            .expect("initial load with tokenizer.json present must succeed");
        drop(guard);

        // Delete tokenizer.json from the LIVE model directory.
        let model_dir = match &source {
            ModelSource::Local(p) => p.clone(),
            other => panic!("expected a Local source, got {other:?}"),
        };
        std::fs::remove_file(model_dir.join("tokenizer.json")).unwrap();

        // (2) First post-deletion call: must stale-reload (evict the
        // now-invalid fingerprint entry and reload with `tokenizer: None`),
        // never return the typed refusal that would come from mis-classifying
        // the tokenizer as REQUIRED.
        let first = cache.get_or_load(&source, ModelTask::TextEmbedding).await;
        match &first {
            Ok(_) => {}
            Err(e) => panic!(
                "the first get_or_load after tokenizer.json's deletion must \
                 stale-reload and succeed — a cold process loading \
                 this same directory would serve it fine — got Err({e})"
            ),
        }
        drop(first);

        // (3) Second, CONSECUTIVE post-deletion call: the wedge check. This
        // is an ordinary warm hit against the freshly-reloaded entry from
        // step (2); a still-cached stale entry would `Err` here again.
        let second = cache.get_or_load(&source, ModelTask::TextEmbedding).await;
        if let Err(e) = second {
            panic!(
                "the second, consecutive get_or_load call after \
                 tokenizer.json's deletion must also succeed — two \
                 consecutive calls behaving like cold-process loads, never a \
                 permanent wedge. Got Err({e})"
            );
        }
    }

    /// The full restore cycle end-to-end at `ModelCache::get_or_load`:
    /// delete `tokenizer.json` -> warm reload (tokenizer-less, embedding
    /// refuses) -> RESTORE `tokenizer.json` -> the NEXT `get_or_load` must
    /// detect the staleness that restoration creates and reload WITH the
    /// tokenizer — embedding serves again.
    ///
    /// The tokenizer candidate is fingerprinted UNCONDITIONALLY (both
    /// `tokenizer.json` and `bpe_simple_vocab_16e6.txt.gz`, mirroring
    /// `1_Pooling`/`preprocessor`), so the tokenizer-less reload's
    /// fingerprint still records an ABSENT snapshot for `tokenizer.json`.
    /// Restoring the file flips that candidate from `NotFound` to `Ok`,
    /// which `ModelFingerprint::probe`'s appearance arm reports as stale
    /// (`Ok(false)`) regardless of `optional`. Were the candidate tracked
    /// only when a tokenizer resolved, nothing fingerprinted would change on
    /// restoration and the tokenizer-less entry would be served forever.
    #[tokio::test]
    async fn tokenizer_restored_after_stale_reload_is_detected_and_reloads_with_tokenizer() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();
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
            .get_or_load(&source, ModelTask::TextEmbedding)
            .await
            .expect("initial load with tokenizer.json present must succeed");
        guard
            .model
            .forward(&text_content, ModelTask::TextEmbedding)
            .await
            .expect("the cold-loaded, tokenizer-bearing entry must serve embeddings");
        drop(guard);

        // (2) Delete `tokenizer.json` from the LIVE model directory.
        std::fs::remove_file(model_dir.join("tokenizer.json")).unwrap();

        // (3) Stale-reload: succeeds, `tokenizer: None`.
        // Its OWN forward call must now refuse — a tokenizer-less entry
        // cannot serve embeddings — with the typed "no tokenizer" error,
        // never a panic or a silently-wrong output.
        let tokenizer_less = cache
            .get_or_load(&source, ModelTask::TextEmbedding)
            .await
            .expect("stale-reload after tokenizer.json's deletion must succeed");
        match tokenizer_less
            .model
            .forward(&text_content, ModelTask::TextEmbedding)
            .await
        {
            Err(e) => {
                // Pin the refusal to the SPECIFIC typed message
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

        // (4) RESTORE `tokenizer.json`: the file a cold process would
        // happily use again is back on disk.
        let fixture = jammi_test_utils::cookbook_fixture("tiny_bert");
        std::fs::copy(
            fixture.join("tokenizer.json"),
            model_dir.join("tokenizer.json"),
        )
        .unwrap();

        // (5) The NEXT get_or_load must detect the restoration as staleness
        // (not report fresh) and reload WITH the tokenizer.
        let restored = cache
            .get_or_load(&source, ModelTask::TextEmbedding)
            .await
            .expect("get_or_load after tokenizer.json's restoration must succeed");

        // (6) Final serving state: the restored entry serves embeddings
        // again — the full cycle's actual observable outcome, not merely
        // "the fingerprint changed".
        restored
            .model
            .forward(&text_content, ModelTask::TextEmbedding)
            .await
            .expect(
                "the entry reloaded after tokenizer.json's restoration must serve embeddings \
                 again — a cold process loading this same, now-restored directory would \
                 serve it fine",
            );
    }
}

// ── Admission-wait liveness: a plain LRU-budget-pressure eviction (NOT a stale reload) must \
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
            devices: vec![-1],
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

    fn test_hub_source() -> crate::model::hub::HubSource {
        let root = tempfile::tempdir().unwrap().keep();
        crate::model::hub::HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(root),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap()
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

    /// Budget sized to ONE resident copy; A holds M1's guard; B spawns
    /// `get_or_load(M2)` — under this budget `do_load` cannot admit M2
    /// without evicting M1, and M1 is NOT evictable while A's guard is live
    /// (`ref_count == 1`), so B must park in the admission loop's wait. A
    /// then drops M1's guard WITHOUT the entry ever being removed from the
    /// cache — a plain idle-LRU eviction target, unlike a STALE RELOAD of
    /// the same model, where the outgoing guard holds the permit's LAST
    /// clone and `GpuPermit::drop`'s own `notify_waiters()` fires directly.
    /// Here the `CacheEntry` for M1 keeps its own clone of the SAME
    /// `Arc<GpuPermit>` alive the whole time, so A's guard drop lowers the
    /// strong count from 2 to 1, never to 0, and `GpuPermit::drop` never
    /// runs. The ONLY transition that fires is `ModelGuard::drop`'s
    /// unconditional `admission_notify.notify_waiters()` (see `ModelGuard`'s
    /// `admission_notify` field doc); a waiter parked purely on
    /// `GpuScheduler`'s own release notify would hang forever while holding
    /// `cache.in_flight[M2]`.
    ///
    /// The final `tokio::time::timeout` turns such a hang into a clear, fast
    /// assertion failure instead of wedging CI, the same bound the
    /// single-flight lost-wakeup test
    /// (`single_flight_tests::registered_waiter_always_wakes_even_if_notify_races_the_pause`)
    /// uses.
    #[tokio::test]
    async fn plain_lru_eviction_wakes_once_the_blocking_guard_drops_even_with_no_permit_release() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            Arc::clone(&catalog),
            test_artifact_store(),
            test_hub_source(),
        )
        .unwrap();

        let (source_a, weights_len) = tiny_bert_source(tmp.path(), "admission_wake_model_a");
        let (source_b, weights_len_b) = tiny_bert_source(tmp.path(), "admission_wake_model_b");
        assert_eq!(
            weights_len, weights_len_b,
            "both fixtures are copies of the same tiny_bert weights file"
        );

        // Budget fits EXACTLY one model — B cannot be admitted until A's
        // guard drops and M1 is evicted; there is no slack that would let
        // B's `try_acquire` succeed on its own.
        let scheduler = Arc::new(GpuScheduler::new(weights_len));
        let cache = Arc::new(ModelCache::new(resolver, device_config(), scheduler));

        // (1) A loads M1 and KEEPS THE GUARD — the entire budget is
        // reserved, and M1 is NOT idle (`ref_count == 1`).
        let guard_a = cache
            .get_or_load(&source_a, ModelTask::TextEmbedding)
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
                .get_or_load(&source_b_for_task, ModelTask::TextEmbedding)
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
                 whole time) — B is left parked forever, permanently \
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
            cache
                .schedulers()
                .get(cache.device_config.gpu_device)
                .expect("the primary device has a budget")
                .available(),
            0,
            "B occupies the entire 1-model budget after evicting the now-idle M1"
        );
        drop(guard_b);
    }
}

// The load-bookkeeping write is an ALLOWLIST of the generic rows it may complete, fails CLOSED
// on a catalog read error, and never overwrites a row a terminal producer already owns.

#[cfg(test)]
mod load_bookkeeping_tests {
    use super::*;

    use jammi_db::catalog::model_repo::{ModelLocation, RegisterModelParams};
    use jammi_db::catalog::Catalog;
    use jammi_db::storage::{StorageRegistry, StorageUrl};
    use jammi_db::store::ArtifactStore;

    fn device_config() -> DeviceConfig {
        DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }

    fn test_artifact_store() -> Arc<ArtifactStore> {
        let cache_dir = tempfile::tempdir().unwrap().keep();
        Arc::new(
            ArtifactStore::with_root(
                StorageUrl::memory("load-bookkeeping-test-artifacts"),
                StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        )
    }

    fn test_hub_source() -> crate::model::hub::HubSource {
        let root = tempfile::tempdir().unwrap().keep();
        crate::model::hub::HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(root),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap()
    }

    /// A minimal, fabricated `ResolvedModel` — `complete_generic_registration`
    /// only reads `resolved.backend` and `resolved.weights_paths`, so this
    /// never needs a real resolve/load to drive its mechanism directly.
    fn fake_resolved(model_id: &str, weights_dir: &std::path::Path) -> ResolvedModel {
        ResolvedModel {
            model_id: ModelId(model_id.to_string()),
            weights_format: super::super::WeightsFormat::Safetensors,
            task: ModelTask::TextEmbedding,
            config_path: weights_dir.join("config.json"),
            weights_paths: vec![weights_dir.join("model.safetensors")],
            tokenizer: None,
            model_config: serde_json::json!({}),
            preprocessor_config: None,
            pooling_config: None,
            base_model_id: None,
            adapter_path: None,
            estimated_memory: 1,
        }
    }

    fn new_cache(catalog: Arc<Catalog>) -> ModelCache {
        let resolver =
            ModelResolver::new(catalog, test_artifact_store(), test_hub_source()).unwrap();
        ModelCache::new(
            resolver,
            device_config(),
            Arc::new(GpuScheduler::new_unlimited()),
        )
    }

    /// A pre-registered `"open_clip"` row (a real, live `model_type` —
    /// `EncoderFamily::adapter_model_type`) carrying its OWN pointers must
    /// survive `complete_generic_registration` byte-for-byte. A denylist of
    /// terminal types (`"fine-tuned"`, `"context-predictor"`, `"checkpoint"`)
    /// would not name `"open_clip"`, and would rewrite `model_type` to
    /// `"local"`, clobbering `base_model_id` and its location.
    #[tokio::test]
    async fn open_clip_row_survives_generic_bookkeeping_byte_for_byte() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let model_id = "open-clip-probe-model";

        catalog
            .register_model(RegisterModelParams {
                model_id,
                version: 1,
                model_type: "open_clip",
                backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
                task: ModelTask::ImageEmbedding,
                base_model_id: Some("producer-owned-base"),
                external_location: Some("/producer/owned/weights"),
                config_json: Some("{\"producer_owned\":true}"),
            })
            .await
            .unwrap();
        let before = catalog.get_model(model_id).await.unwrap().unwrap();

        let cache = new_cache(Arc::clone(&catalog));
        let resolved = fake_resolved(model_id, tmp.path());
        cache
            .complete_generic_registration(
                &ModelSource::hf(model_id),
                model_id,
                ModelBackendKind::Candle,
                resolved.weights_dir(),
                ModelTask::TextEmbedding,
            )
            .await;

        let after = catalog.get_model(model_id).await.unwrap().unwrap();
        assert_eq!(
            after.model_type, before.model_type,
            "model_type must survive untouched"
        );
        assert_eq!(
            after.base_model_id, before.base_model_id,
            "base_model_id lineage must survive untouched"
        );
        assert_eq!(
            after.location, before.location,
            "the location must survive untouched"
        );
        assert_eq!(
            after.config_json, before.config_json,
            "config_json must survive untouched"
        );
        assert_eq!(
            after.backend, before.backend,
            "backend must survive untouched"
        );
        assert_eq!(after.task, before.task, "task must survive untouched");
    }

    /// General case: an entirely UNENUMERATED `model_type` — not a known
    /// architecture id, just some future or unrecognised string — must ALSO
    /// survive. This is why the gate is an allowlist rather than a denylist
    /// with more names added: a denylist protects only what someone thought
    /// to enumerate, so a truly novel type would fail open on it.
    #[tokio::test]
    async fn wholly_unenumerated_model_type_survives_generic_bookkeeping() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let model_id = "unknown-type-probe-model";

        catalog
            .register_model(RegisterModelParams {
                model_id,
                version: 1,
                model_type: "some-future-architecture-nobody-enumerated-yet",
                backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
                task: ModelTask::TextEmbedding,
                base_model_id: Some("some-base"),
                external_location: Some("/some/owned/weights"),
                config_json: None,
            })
            .await
            .unwrap();
        let before = catalog.get_model(model_id).await.unwrap().unwrap();

        let cache = new_cache(Arc::clone(&catalog));
        let resolved = fake_resolved(model_id, tmp.path());
        cache
            .complete_generic_registration(
                &ModelSource::hf(model_id),
                model_id,
                ModelBackendKind::Candle,
                resolved.weights_dir(),
                ModelTask::TextEmbedding,
            )
            .await;

        let after = catalog.get_model(model_id).await.unwrap().unwrap();
        assert_eq!(after.model_type, before.model_type);
        assert_eq!(after.base_model_id, before.base_model_id);
        assert_eq!(after.location, before.location);
    }

    /// Positive case: a plain `"local"` row (this call's own prior
    /// write) IS completed — the allowlist must not become so conservative
    /// that it stops doing the one thing this bookkeeping exists for.
    #[tokio::test]
    async fn local_row_is_completed() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let model_id = "local-completion-probe-model";

        catalog
            .register_model(RegisterModelParams {
                model_id,
                version: 1,
                model_type: "local",
                backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: None,
                config_json: None,
            })
            .await
            .unwrap();

        let cache = new_cache(Arc::clone(&catalog));
        let resolved = fake_resolved(model_id, tmp.path());
        cache
            .complete_generic_registration(
                &ModelSource::local(tmp.path()),
                model_id,
                ModelBackendKind::Candle,
                resolved.weights_dir(),
                ModelTask::TextEmbedding,
            )
            .await;

        let after = catalog.get_model(model_id).await.unwrap().unwrap();
        assert_eq!(after.model_type, "local");
        assert_eq!(
            after.location,
            Some(ModelLocation::External(
                tmp.path().to_str().unwrap().to_string()
            )),
            "a plain local row must be completed with the resolved weights directory"
        );
    }

    /// Positive case: the `"embedding"` FK placeholder
    /// (`Session::submit_fine_tune_spec`'s pre-registration, always
    /// no location before the base model is ever loaded) IS
    /// completed by this call.
    #[tokio::test]
    async fn embedding_placeholder_is_completed() {
        let tmp = tempfile::tempdir().unwrap();
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let model_id = "embedding-placeholder-probe-model";

        catalog
            .register_model(RegisterModelParams {
                model_id,
                version: 1,
                model_type: "embedding",
                backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: None,
                config_json: None,
            })
            .await
            .unwrap();

        let cache = new_cache(Arc::clone(&catalog));
        let resolved = fake_resolved(model_id, tmp.path());
        cache
            .complete_generic_registration(
                &ModelSource::local(tmp.path()),
                model_id,
                ModelBackendKind::Candle,
                resolved.weights_dir(),
                ModelTask::TextEmbedding,
            )
            .await;

        let after = catalog.get_model(model_id).await.unwrap().unwrap();
        assert_eq!(
            after.model_type, "local",
            "the placeholder must be completed to the source's own generic type"
        );
        assert_eq!(
            after.location,
            Some(ModelLocation::External(
                tmp.path().to_str().unwrap().to_string()
            )),
            "the placeholder's location must be completed, not left None"
        );
    }

    /// A catalog READ error must skip the write entirely — never collapse
    /// to "no row" and clobber a row this call never actually inspected.
    /// Both the read AND a subsequent write attempt fail on the SAME closed
    /// pool, so the final DB state alone cannot distinguish "skipped" from
    /// "attempted and also failed" — the oracle instead captures which
    /// `tracing::warn!` fires: the read-failure message, and no call to
    /// `register_model` (whose own failure, on the same closed pool, would
    /// log the register-failure message). A read error swallowed as "no
    /// row" flips `saw_write_attempt` to `true` and `saw_read_failure_log`
    /// to `false`.
    ///
    /// Fault injection: two `Catalog` handles share the SAME backend `Arc`
    /// (`Catalog::pinned_to_tenant`); closing one closes the shared
    /// connection pool out from under the other, which is the closed/dropped
    /// connection this crate's own `Catalog::close` doc describes as making
    /// every sibling handle's next query fail.
    #[test]
    fn catalog_read_error_skips_bookkeeping_write() {
        use std::io;
        use std::sync::Mutex;
        use tracing_subscriber::fmt::MakeWriter;

        #[derive(Clone, Default)]
        struct BufferWriter(Arc<Mutex<Vec<u8>>>);
        impl io::Write for BufferWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        impl<'w> MakeWriter<'w> for BufferWriter {
            type Writer = BufferWriter;
            fn make_writer(&'w self) -> Self::Writer {
                self.clone()
            }
        }

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(BufferWriter(buffer.clone()))
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        runtime.block_on(async {
            let tmp = tempfile::tempdir().unwrap();
            let catalog_dir = tempfile::tempdir().unwrap();
            let owner = Catalog::open(catalog_dir.path()).await.unwrap();
            let model_id = "read-error-probe-model";

            // Seed a pre-existing "local" row (a completable type) through
            // the live handle, BEFORE the pool is closed, so a
            // wrongly-proceeding write would have something real to clobber.
            owner
                .register_model(RegisterModelParams {
                    model_id,
                    version: 1,
                    model_type: "local",
                    backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
                    task: ModelTask::TextEmbedding,
                    base_model_id: None,
                    external_location: None,
                    config_json: None,
                })
                .await
                .unwrap();

            let shared = owner.pinned_to_tenant(None);
            owner.close().await;

            // Confirm the fault actually landed: the shared handle's own
            // read must now be an `Err`, not a `None` — otherwise this test
            // would not be exercising the read-error path at all.
            let probe_err = shared.get_model_version(model_id, 1).await;
            assert!(
                probe_err.is_err(),
                "fault injection failed to land: expected the closed pool to make a \
                 read error, got {probe_err:?}"
            );

            let cache = new_cache(Arc::new(shared));
            let resolved = fake_resolved(model_id, tmp.path());
            cache
                .complete_generic_registration(
                    &ModelSource::local(tmp.path()),
                    model_id,
                    ModelBackendKind::Candle,
                    resolved.weights_dir(),
                    ModelTask::TextEmbedding,
                )
                .await;
        });

        let logs = String::from_utf8(buffer.lock().unwrap().clone()).expect("utf-8 logs");
        let saw_read_failure_log =
            logs.contains("Failed to read catalog row before load bookkeeping");
        let saw_write_attempt = logs.contains("Failed to register model in catalog");
        assert!(
            saw_read_failure_log,
            "expected the read-error path to log its own skip warning; captured logs:\n{logs}"
        );
        assert!(
            !saw_write_attempt,
            "a catalog read error must skip the write entirely, never fall through to \
             attempting (and separately failing) a `register_model` call; captured logs:\n{logs}"
        );
    }
}

/// Test-only rendezvous inside [`ModelCache::preload`] (`feature =
/// "test-hooks"`; mirrors `crate::jobs::compute_test_hooks`): a test arms
/// [`ParkPoint::BeforeLoad`] for a model source's canonical string and the
/// next preload of that source parks — before any bytes load — until
/// released, so a server's warm-before-ready window (`/readyz` 503
/// "preloading", the claim loop parked at `warming`) is observable at a
/// documented point rather than raced. The cache's own pause handle is
/// `#[cfg(test)]`, unreachable from another crate's tests. No production
/// path observes anything here beyond the `maybe_park` call.
#[cfg(feature = "test-hooks")]
pub mod preload_test_hooks {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use tokio::sync::Notify;

    /// Where a preload parks.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ParkPoint {
        /// Inside `ModelCache::preload`, before the load.
        BeforeLoad,
    }

    struct Armed {
        key: String,
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

    /// The test's side of one armed park.
    pub struct ParkHandle {
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    impl ParkHandle {
        /// Resolve once the preload has reached the park point.
        pub async fn wait_parked(&self) {
            while !self.parked.load(Ordering::SeqCst) {
                self.parked_notify.notified().await;
            }
        }

        /// Let the parked preload continue.
        pub fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.release_notify.notify_one();
        }
    }

    /// Arm one park for the next preload whose source's canonical string
    /// (`ModelSource`'s `Display`) is `key`. One-shot.
    pub fn arm(key: &str, point: ParkPoint) -> ParkHandle {
        let parked = Arc::new(AtomicBool::new(false));
        let parked_notify = Arc::new(Notify::new());
        let released = Arc::new(AtomicBool::new(false));
        let release_notify = Arc::new(Notify::new());
        armed()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Armed {
                key: key.to_string(),
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

    pub(super) async fn maybe_park(key: &str, point: ParkPoint) {
        let taken = {
            let mut list = armed().lock().unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|a| a.key == key && a.point == point)
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
}
