use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::JammiConfig;
use jammi_db::error::{JammiError, Result};
use jammi_db::index::SearchMethod;
use jammi_db::session::{JammiSession, QueryContext, QueryFunction};
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::{ArtifactStore, PinnedSource, ResultStore};

use crate::eval::runner::EvalRunner;
use crate::fine_tune::spec::{TrainingCommon, TrainingSpec};
use crate::fine_tune::training_job::{fine_tuned_model_id, resolve_model_id, TrainingJob};
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::model::backend::DeviceConfig;
use crate::model::cache::ModelCache;
use crate::model::hub::HubSource;
use crate::model::resolver::ModelResolver;
use crate::pipeline::embedding::EmbeddingPipeline;
use crate::query::QueryBuilder;
use jammi_datafusion::inference::observer::InferenceObserver;
use jammi_datafusion::BackendOutput;
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;
use jammi_datafusion::RowOrder;
use jammi_datafusion::{plan_inference, InferenceFanOut, InferenceRuntime, InferenceSpec};
use jammi_db::cache::ann_cache::AnnCache;

/// An inference-capable session that wraps `JammiSession` with model loading
/// and inference execution. This is the primary entry point for CP2+.
pub struct InferenceSession {
    inner: Arc<JammiSession>,
    model_cache: Arc<ModelCache>,
    result_store: Arc<ResultStore>,
    artifact_store: Arc<ArtifactStore>,
    observer: Option<Arc<dyn InferenceObserver>>,
    ann_cache: Arc<AnnCache>,
    device_config: DeviceConfig,
    /// The one Hugging Face Hub client this session's resolver and fine-tune
    /// worker share — built once, below, from `[models]`.
    hub: HubSource,
    /// Registry of open ephemeral sessions, shared with the timeout scanner.
    ephemeral_sessions: jammi_db::ephemeral::ActiveSessions,
    /// This process's identity in the `instances`/`jobs.claimed_by`
    /// vocabulary: a UUID minted once at construction
    /// ([`crate::fine_tune::worker::mint_instance_id`]) and never taken
    /// from the environment — `JAMMI_WORKER_ID` is only the row's `label`
    /// ([`crate::fine_tune::worker::worker_label`]), so two processes
    /// sharing an operator label (a restart, a sibling replica) are two
    /// `instances` rows and a dead one's inline jobs are reclaimed rather
    /// than kept alive by its namesake's heartbeat. Every claimant on this
    /// process (a [`crate::fine_tune::worker::JobWorker`]'s poll loop, a
    /// [`Self::run_now`] inline claim) shares this one `claimed_by`
    /// identity, and `catalog::lease_keeper::LeaseTarget::Instance`
    /// registers the SAME id `jobs.claimed_by` carries.
    instance_id: String,
    /// This host's admission state ([`crate::fine_tune::worker::HostAdmission`]):
    /// the shutdown phase, the single job-slot holder, and this process's
    /// `InstanceRegistration` — the ONE carrier [`Self::instance_id`]'s row
    /// is written from
    /// ([`jammi_db::catalog::instance::InstanceRegistration::from_config`]),
    /// shared with the lease keeper's [`jammi_db::catalog::lease_keeper::
    /// LeaseTarget::Instance`] hold below. [`crate::fine_tune::worker::
    /// JobWorker`] and [`crate::fine_tune::worker::EmbeddedWorker`] are the
    /// SOLE owners of the registration's worker half (see
    /// [`Self::instance_registration`]); the gang admission handler and the
    /// claim loop contend for the holder; DRAIN/RELEASE flip the phase.
    host_admission: Arc<crate::fine_tune::worker::HostAdmission>,
    /// The process's one lease-renewal thread — every claimed lease
    /// this session (or a job/table it owns) holds is held open here instead
    /// of spawning its own `tokio::spawn` heartbeat task, so a CPU-bound
    /// inline compute job on the main runtime can never starve a renewal.
    lease_keeper: Arc<jammi_db::catalog::lease_keeper::LeaseKeeper>,
    /// This session's `instances` row hold — held for its `Drop` (releases
    /// the keeper renewal when the session drops), never read.
    _instance_hold: jammi_db::catalog::lease_keeper::LeaseHold,
    /// The worker gate (warm-before-claim): a claim loop spawned over this
    /// session waits for `true` before its first claim. Open by default —
    /// fixtures and library callers never touch it; a server closes it
    /// ([`Self::close_worker_gate`]) before binding while it preloads models
    /// and opens it ([`Self::open_worker_gate`]) once warm.
    worker_gate: tokio::sync::watch::Sender<bool>,
}

/// The model-side links a training kind's `jobs` row is submitted with —
/// see [`InferenceSession::training_job_links`].
pub(crate) struct TrainingJobLinks {
    /// `jobs.model_ref`: the base model's catalog PK.
    pub(crate) model_ref: String,
    /// `jobs.output_model_id`: the NAME the finished model registers under.
    pub(crate) output_model_id: String,
}

/// Refuse, at session open, a `[worker] collective` this BUILD cannot honour.
///
/// The predicate half lives in `jammi_db` (`CollectiveSelection::
/// requires_cuda`), which cannot see another crate's cargo features; this is
/// the half that knows its own build. `nccl` on a build without CUDA is
/// refused rather than degraded: the deployment asked for the device
/// interconnect, and silently reducing on the host at a fraction of the
/// throughput would be a different deployment reported as the one asked for.
/// `auto` degrades by definition and `cpu` wants no device, so neither is
/// affected.
///
/// At OPEN, not at submit: a process that cannot honour its own configuration
/// should not come up and then refuse every job it is handed.
fn refuse_unreachable_collective(topology: &jammi_db::config::WorkerTopology) -> Result<()> {
    if topology.collective().requires_cuda() && !cfg!(feature = "cuda") {
        return Err(JammiError::Config(format!(
            "[worker] collective = \"{}\" needs a build with the `cuda` feature; this binary \
             has none, so the requested collective cannot be reached (set collective = \
             \"auto\" to use the host reduction, or run a CUDA build)",
            topology.collective()
        )));
    }
    Ok(())
}

impl InferenceSession {
    /// Create a new session with model loading and inference capabilities.
    pub async fn new(config: JammiConfig) -> Result<Self> {
        Self::with_observer(config, None).await
    }

    /// Build a session behind an `Arc` with the compound-query SQL functions
    /// registered (`annotate`, …). This is the canonical constructor for any
    /// long-lived shared session — the embedded `Database`, the OSS server, and
    /// the `Jammi::open` local arm — so the in-process `sql` surface and the
    /// Flight SQL lane both expose the same SQL functions. Sessions that never
    /// run compound SQL (short-lived CLI commands) can still use the plain
    /// [`Self::new`].
    pub async fn open(config: JammiConfig) -> Result<Arc<Self>> {
        let session = Arc::new(Self::new(config).await?);
        session.install_query_functions();
        Ok(session)
    }

    /// [`Self::open`] with an explicit segment placement — which process owns
    /// which ANN index segment, read by the result store at every online
    /// search resolve. `open` builds a placement from `[server] placement`
    /// itself ([`jammi_db::index::AllLocal`] by default, or
    /// [`jammi_db::index::RendezvousPlacement`] when the config names
    /// `"rendezvous"`); THIS constructor is the explicit override a library
    /// process that knows its own topology uses instead — a
    /// [`jammi_db::index::StaticPlacement`], or any other
    /// `SegmentPlacement` — becoming a full coordinator over the gRPC peer
    /// transport (`jammi_wire::peer::GrpcPeerTransport`, wired by the store
    /// builder) regardless of `[server] placement`. Precondition for any
    /// non-local placement: `storage.result_root` (or a shared local
    /// `artifact_dir`) is a root every replica can read, spelled IDENTICALLY
    /// on every replica. When `[server] peer_advertise` is set, this
    /// shared-root topology is exactly what
    /// [`jammi_db::config::JammiConfig::resolved_result_root`] yields
    /// verbatim into `instances.result_root` — carried AND consulted:
    /// [`jammi_db::catalog::Catalog::list_gang_members`] admits on address,
    /// kinds, state, freshness AND root-identity equality, and
    /// [`jammi_db::index::RendezvousPlacement`]'s ring evaluates the SAME
    /// root-identity predicate. Identical spelling stays necessary, never
    /// sufficient, for shared storage (see
    /// [`jammi_db::catalog::instance::InstanceRegistration::from_config`]).
    pub async fn open_with_placement(
        config: JammiConfig,
        placement: Arc<dyn jammi_db::index::SegmentPlacement>,
    ) -> Result<Arc<Self>> {
        let inner = JammiSession::new(config).await?;
        let session = Arc::new(Self::wrap_with(inner, None, Some(placement)).await?);
        session.install_query_functions();
        Ok(session)
    }

    /// Create a new session with an optional inference observer.
    pub async fn with_observer(
        config: JammiConfig,
        observer: Option<Arc<dyn InferenceObserver>>,
    ) -> Result<Self> {
        let inner = JammiSession::new(config).await?;
        Self::wrap(inner, observer).await
    }

    /// Create a session whose trigger-stream surface is bound to a
    /// caller-supplied broker. Forwarded to
    /// [`jammi_db::session::JammiSession::with_broker`]; the
    /// `InferenceSession` adds model-loading, eval, and inference layers on
    /// top. Used by tests that need a broker with controlled behaviour, e.g.
    /// an [`jammi_db::trigger::InMemoryBroker`] armed with
    /// `trigger_failure_for_next_publish` to deterministically exercise
    /// publisher-failure paths.
    pub async fn with_broker(
        config: JammiConfig,
        trigger_broker: Arc<dyn jammi_db::trigger::TriggerBroker>,
    ) -> Result<Self> {
        let inner = JammiSession::with_broker(config, trigger_broker).await?;
        Self::wrap(inner, None).await
    }

    /// Create a session whose audit sign/verify path routes through a
    /// caller-supplied [`jammi_db::audit::SigningKeyStore`]. The catalog
    /// backend and trigger broker stay config-driven (the counterpart to
    /// [`Self::with_broker`]); forwarded to
    /// [`jammi_db::session::JammiSession::with_signing_key_store`]. Deployments
    /// whose audit master key lives behind a secrets adapter inject it here so
    /// both the engine's audit *sign* path (`scope.audit().log`) and any
    /// out-of-band verify path share one store.
    pub async fn with_signing_key_store(
        config: JammiConfig,
        signing_key_store: Arc<dyn jammi_db::audit::SigningKeyStore>,
    ) -> Result<Self> {
        let inner = JammiSession::with_signing_key_store(config, signing_key_store).await?;
        Self::wrap(inner, None).await
    }

    async fn wrap(
        inner: JammiSession,
        observer: Option<Arc<dyn InferenceObserver>>,
    ) -> Result<Self> {
        Self::wrap_with(inner, observer, None).await
    }

    /// `placement_override: None` means "derive the placement from `[server]
    /// placement`" (the DEFAULT path, [`Self::open`]/[`Self::new`]/every
    /// other constructor); `Some(p)` is [`Self::open_with_placement`]'s
    /// explicit override, which wins regardless of what `[server] placement`
    /// says.
    async fn wrap_with(
        inner: JammiSession,
        observer: Option<Arc<dyn InferenceObserver>>,
        placement_override: Option<Arc<dyn jammi_db::index::SegmentPlacement>>,
    ) -> Result<Self> {
        let inner = Arc::new(inner);
        let catalog = Arc::clone(inner.catalog());

        // The ONE choke point, run FIRST — before the lease keeper starts, before the result store
        // creates a single directory, before any other side effect:
        // `peer_advertise` unset yields a non-member registration
        // (`peer_addr`/`member_root` both `None`) with no filesystem/config
        // check at all; `peer_advertise` set parses it as a `PeerAddr`,
        // requires `peer_bind`, and carries `resolved_result_root()`
        // VERBATIM as the member root — no filesystem access, no
        // interpretation of the root at all. `wrap_with` is the universal
        // funnel every `InferenceSession` constructor reaches, so a
        // hand-built config (never routed through `JammiConfig::load_from`)
        // is still covered here.
        // `JammiConfig::load_from` calls `InferenceConfig::validate`, but a
        // hand-built `JammiConfig` handed straight to an `InferenceSession`
        // constructor never runs `load_from` at all. `wrap_with` is the same
        // universal funnel that covers `MembershipConfig::validate` for the
        // identical reason (see the comment above), so the `[inference]`
        // fan-out and batch size are validated here too.
        inner.config().inference.validate()?;
        // A query this session plans through the optimizer keeps the
        // inference fan-out its plan was built with.
        inner.add_physical_optimizer_rule(Arc::new(InferenceFanOut));

        let instance_id = crate::fine_tune::worker::mint_instance_id();
        let label = crate::fine_tune::worker::worker_label();
        let registration = Arc::new(
            jammi_db::catalog::instance::InstanceRegistration::from_config(
                inner.config(),
                instance_id.clone(),
                label.as_deref(),
                None,
            )?,
        );

        // One lease-renewal thread per process, started before anything
        // holds a lease with it (the result store's `BuildingTable`
        // adoptions below, this session's own `instances` row, and every
        // job/table lease a `JobWorker`/`run_now` claim holds later).
        // `catalog_connect` opens a FRESH backend connection from inside the
        // keeper's own dedicated runtime — never this session's `catalog`
        // handle — so a CPU-bound inline job saturating the main runtime can
        // never starve the renewal (see `lease_keeper`'s module docs).
        let lease_intervals = inner.config().lease.intervals()?;
        let config_for_keeper = inner.config().clone();
        // `start` is fallible: it returns only once the keeper thread has
        // connected and completed its first renewal pass, so no hold this
        // session later registers can read "live" against a keeper that
        // never ran (a typed `Catalog`/`Config` error surfaces here instead).
        let lease_keeper = jammi_db::catalog::lease_keeper::LeaseKeeper::start(
            move || {
                let config = config_for_keeper.clone();
                Box::pin(async move { jammi_db::session::open_catalog_from_config(&config).await })
            },
            lease_intervals,
        )
        .await?;

        // `[server] placement` resolved AFTER `instance_id` is minted (a
        // `RendezvousPlacement` names itself by it) and AFTER `lease_intervals`
        // is known (its ring-liveness margin derives from the SAME lease
        // window every other leased row family uses) — an explicit override
        // from `Self::open_with_placement` wins outright, regardless of what
        // `[server] placement` says.
        let placement: Arc<dyn jammi_db::index::SegmentPlacement> = match placement_override {
            Some(placement) => placement,
            None => match inner.config().server.placement {
                jammi_db::config::PlacementMode::Local => Arc::new(jammi_db::index::AllLocal),
                jammi_db::config::PlacementMode::Rendezvous => {
                    Arc::new(jammi_db::index::RendezvousPlacement::new(
                        Arc::clone(&catalog),
                        instance_id.clone(),
                        jammi_db::catalog::lease::instance_liveness_margin(lease_intervals.lease()),
                    ))
                }
            },
        };

        // The result store is built first: it owns the session's
        // `ArtifactStore` internally (rooted at `{result_store_root}/models`,
        // one storage knob serving both), so the resolver reads that SAME
        // handle rather than a second store independently constructed at a
        // root that could disagree with it. A fine-tuned model's catalog
        // row references an object-store artifact, fetched into a local
        // cache before candle loads it, so an adapter trained on one host
        // serves on another. It registers every `BuildingTable` it adopts
        // with the keeper above rather than spawning its own heartbeat task.
        let result_store = Arc::new(
            build_result_store(&inner, Arc::clone(&catalog), placement)?
                .with_lease_keeper(Arc::clone(&lease_keeper)),
        );
        let artifact_store = result_store.artifact_store();
        // The Hub choke point: `[models]` -> `HubSource`, exactly
        // once per session. Every downstream Hub call (the resolver's
        // HuggingFace arm, the fine-tune worker's HF fallback) shares this
        // one client rather than each re-deriving its own from
        // `hf_hub::api::sync::Api::new()`/`ApiBuilder::from_env()`. Process
        // env is read HERE (the `HF_HOME`/`HF_ENDPOINT`/`HF_TOKEN`
        // fallbacks) — never inside `JammiConfig::load_from`, which stays
        // process-env-free.
        let hub = HubSource::from_config(&inner.config().models, &|k: &str| std::env::var(k).ok())?;
        let resolver =
            ModelResolver::new(catalog.clone(), Arc::clone(&artifact_store), hub.clone())?;
        // The validated rank topology, resolved ONCE per session: how many
        // ranks this deployment runs, which device each of them gets, and
        // which collective they reduce over. `JammiConfig::load_from` has
        // already run these bounds, so a loaded configuration re-derives the
        // same verdict here; a programmatically built one gets it for the
        // first time.
        let topology = inner.config().worker.topology(&inner.config().gpu)?;
        refuse_unreachable_collective(&topology)?;

        let device_config = DeviceConfig::from_config(inner.config());
        // One admission budget PER DEVICE, over the resolved `[gpu] devices`
        // list. A budget is a property of a card: a single counter shared by
        // two devices would either over-admit on one or starve the other.
        // A single-device deployment resolves to one entry and is exactly
        // what it was before the list existed.
        let schedulers = crate::concurrency::DeviceSchedulers::for_devices(
            &device_config.devices,
            inner.config().gpu.memory_limit,
            inner.config().engine.execution_threads,
        )?;
        let model_cache = Arc::new(
            ModelCache::with_device_schedulers(resolver, device_config.clone(), schedulers)
                .bounded(inner.config().inference.cache_bounds())
                .with_remote_models(&inner.config().models.remote)?,
        );
        // What a sink this process runs records as having produced its
        // bytes — its device and the models the plan ran — wherever the
        // plan was submitted from.
        result_store.install_producing_environment(Arc::new(
            crate::inference::environment::InferenceEnvironment {
                device: crate::model::backend::candle::effective_compute_device(&device_config),
                model_cache: Arc::clone(&model_cache),
            },
        ));
        // Install the tenant-gating result-table schema as the context's
        // default schema before any table is loaded, so bare `jammi.{name}`
        // resolutions honour the catalog owner on every read lane and source
        // removal can find the provider to clear.
        result_store.install_result_schema(inner.context())?;
        // Every model-facing source scan projects `jammi_content_hash(...)`
        // (`build_source_query`), so the UDF is part of the session's base
        // context — not of the opt-in compound-query set
        // (`install_query_functions`), which a plain `new` never installs.
        inner.install_functions([QueryFunction::Scalar(crate::query::content_hash_udf())]);
        result_store.recover().await?;
        result_store.load_existing_tables(inner.context()).await?;

        // Item 5's construction sweep: `recover()` (PR-A, above) plus the
        // jobs-side reclaim and the two retention prunes, unconditionally —
        // a process that never claims still benefits from reclaiming a dead
        // peer's expired leases and does no harm running the sweep.
        catalog
            .reclaim_expired_jobs(
                lease_intervals.lease(),
                crate::fine_tune::worker::MAX_ATTEMPTS,
            )
            .await?;
        // Two windows, two knobs: an `instances` row is stale once it has
        // missed the same `2 × lease` liveness margin the inline-job
        // reclaim arm judges it by (a process that has not heartbeated for
        // two lease windows is dead to every reader), while terminal job
        // rows live for `[jobs] retention_days` — the retention knob never
        // decides process liveness.
        catalog
            .prune_instances(jammi_db::catalog::lease::instance_prune_window(
                lease_intervals.lease(),
            ))
            .await?;
        // `Catalog::prune_jobs` is tenant-scoped: every
        // `JobService::PruneJobs` RPC call runs it under the CALLER's own
        // `scoped(...)` tenant, like every other job RPC. This construction
        // sweep is not an RPC — it runs once per process boot, before any
        // request is scoped — and stays global BY DESIGN (a process that
        // never claims still benefits from reclaiming every tenant's stale
        // terminal rows), so it explicitly drops the tenant predicate via
        // `with_admin_scope` rather than silently sweeping only the `NULL`
        // (unscoped) tenant's rows.
        {
            let retention = inner.config().jobs.retention();
            let catalog = Arc::clone(&catalog);
            inner
                .with_admin_scope(|_admin| async move { catalog.prune_jobs(retention).await })
                .await?;
        }

        // This process's `instances` row + keeper hold — every session
        // upserts and heartbeats one, whether or not it runs a claim loop
        // (only `workers` membership is gated on `[worker] enabled`,
        // upserted by `EmbeddedWorker::spawn`/`JobWorker`). `registration`
        // was already validated above, before any side effect; this is its
        // first actual write.
        catalog.upsert_instance(&registration).await?;
        let instance_hold = lease_keeper.hold(
            jammi_db::catalog::lease_keeper::LeaseTarget::Instance(Arc::clone(&registration)),
        );

        let ann_cache_size = inner.config().cache.ann_cache_max_entries as u64;
        let ann_cache = Arc::new(AnnCache::new(ann_cache_size));

        let (worker_gate, _) = tokio::sync::watch::channel(true);

        Ok(Self {
            inner,
            model_cache,
            result_store,
            artifact_store,
            observer,
            ann_cache,
            device_config,
            hub,
            ephemeral_sessions: jammi_db::ephemeral::ActiveSessions::new(),
            instance_id,
            host_admission: crate::fine_tune::worker::HostAdmission::new(registration),
            lease_keeper,
            _instance_hold: instance_hold,
            worker_gate,
        })
    }

    /// Close the worker gate: a claim loop over this session (spawned
    /// before or after this call) parks before its first claim, with its
    /// `workers` row reading `warming`, until [`Self::open_worker_gate`].
    /// A server closes it while `preload_models` loads, so `/readyz` never
    /// says "preloading" while this process is already claiming.
    pub fn close_worker_gate(&self) {
        self.worker_gate.send_replace(false);
    }

    /// Open the worker gate (the default state): a parked loop passes its
    /// wait, flips its row to `claiming` and claims.
    pub fn open_worker_gate(&self) {
        self.worker_gate.send_replace(true);
    }

    /// A receiver on the worker gate — what a claim loop `wait_for(|open|
    /// *open)`s before its first claim.
    pub fn worker_gate_receiver(&self) -> tokio::sync::watch::Receiver<bool> {
        self.worker_gate.subscribe()
    }

    /// RELEASE this session's job leases without a loop to stop — the
    /// library's `release_and_stop` when no worker was spawned, and the
    /// server's when `[worker] enabled = false`: the phase flips to
    /// `Releasing` first ([`crate::fine_tune::worker::HostAdmission::begin_release`],
    /// so every gang rank held on this host ends with the `Drain` reason),
    /// then 2b (the keeper releases
    /// every `LeaseTarget::Job` hold it holds — inline holds return
    /// `Ok(false)` and are left alone) then 2c (the jobs sweep and the
    /// jobs-linked building sweep). With no loop-claimed row on this
    /// instance every statement matches nothing by construction, so on a
    /// worker-less process this is a no-op that keeps the surface uniform.
    /// Returns `(2b's outcome, sweep counts)` — see
    /// [`crate::fine_tune::worker::HoldReleaseOutcome`] for why a bare count
    /// cannot stand in for 2b's own result: `Unobserved` (the keeper's pass
    /// itself could not be confirmed to run) is a different fact from "zero
    /// holds were held".
    pub async fn release_job_leases(
        &self,
    ) -> Result<(
        crate::fine_tune::worker::HoldReleaseOutcome,
        crate::fine_tune::worker::ReleaseSweep,
    )> {
        self.host_admission.begin_release().await;
        let heartbeat = self.worker_intervals()?.heartbeat;
        let holds = match self.lease_keeper.release_job_holds(heartbeat).await {
            Ok(hr) => crate::fine_tune::worker::HoldReleaseOutcome::Observed(hr),
            Err(e) => {
                tracing::warn!(error = %e, "release_job_leases: the keeper's per-hold pass failed");
                crate::fine_tune::worker::HoldReleaseOutcome::Unobserved
            }
        };
        let sweep = crate::fine_tune::worker::release_sweep(
            self.catalog(),
            &self.instance_id,
            self.result_store.writer_id(),
        )
        .await;
        Ok((holds, sweep))
    }

    /// This process's `instances`/`jobs.claimed_by` identity: a UUID
    /// minted at construction, never `JAMMI_WORKER_ID` (which is only the
    /// row's label). Shared by every claimant on this session — a
    /// [`crate::fine_tune::worker::JobWorker`]'s poll loop and
    /// [`Self::run_now`]'s inline claim alike.
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// This process's [`jammi_db::catalog::instance::InstanceRegistration`]
    /// — the SAME value [`Self::instance_id`]'s row was written from and the
    /// lease keeper's `LeaseTarget::Instance` hold renews. A
    /// [`crate::fine_tune::worker::JobWorker`] (and the
    /// [`crate::fine_tune::worker::EmbeddedWorker`] guard spawned over it)
    /// is the sole owner of its `worker` half: it sets the cell only AFTER
    /// every row write as ONE fact with the row (`write_worker_facts` in
    /// `fine_tune::worker`): the cell is
    /// set to the facts about to be UPSERTED and reverted if that upsert
    /// fails — after a failed first write it is `None` again — and cleared
    /// before every `delete_worker` call. A keeper reregister therefore
    /// re-upserts only facts a row write of this process succeeded with, or
    /// the facts an in-flight upsert is about to write.
    pub(crate) fn instance_registration(
        &self,
    ) -> &Arc<jammi_db::catalog::instance::InstanceRegistration> {
        self.host_admission.registry()
    }

    /// This host's admission state: the shutdown phase every claim loop and
    /// every held gang rank on this process reads, and the one job-slot
    /// holder they contend for ([`crate::fine_tune::worker::HostAdmission`]).
    pub fn host_admission(&self) -> &Arc<crate::fine_tune::worker::HostAdmission> {
        &self.host_admission
    }

    /// This process's one lease-renewal thread. A
    /// [`crate::fine_tune::worker::JobWorker`] and [`Self::run_now`] both
    /// hold their claimed job leases here rather than spawning their own
    /// heartbeat task.
    pub fn lease_keeper(&self) -> &Arc<jammi_db::catalog::lease_keeper::LeaseKeeper> {
        &self.lease_keeper
    }

    /// The validated lease/heartbeat/idle-poll timing this session's
    /// `[lease]`/`[worker]` configuration resolves to.
    pub fn worker_intervals(&self) -> Result<jammi_db::config::WorkerIntervals> {
        self.inner
            .config()
            .worker
            .worker_intervals(self.inner.config().lease.intervals()?)
    }

    /// Release every catalog connection this session holds — its own
    /// shared backend pool AND the lease keeper's dedicated
    /// connection — so a successor process can open the SAME catalog
    /// directory immediately.
    ///
    /// **Dropping the session is not a release point**, for the same
    /// reason [`jammi_db::session::JammiSession::close`] documents for the
    /// shared pool, plus one this session adds on top: the lease keeper
    /// opens its OWN connection on its own dedicated OS thread ([`Self::new`]
    /// wires it before anything else can hold a lease), entirely
    /// independent of the shared pool `JammiSession::close` releases.
    /// Closing only that shared pool while the keeper's thread stays up
    /// leaves the SQLite `unix-excl` VFS's process-scoped exclusive
    /// lock held — the keeper's own connection is still open — so a
    /// successor process opening the same directory is refused within the
    /// busy timeout even after every OTHER handle has let go.
    ///
    /// Order: shut the keeper down and wait — bounded, see
    /// [`jammi_db::catalog::lease_keeper::LeaseKeeper::shutdown_and_join`] —
    /// for its thread to close its own connection, THEN close the shared
    /// pool, so nothing renews a lease against a catalog this call is in
    /// the middle of tearing down. A keeper that does not exit within the
    /// shutdown window is logged and does NOT block the shared-pool close
    /// that follows — this call must never hang, even at the cost of
    /// leaving the keeper's connection's fate unresolved in that
    /// (unexpected) case. The window is generous (30 s): the cost of a
    /// caller's own release call taking that long in the genuinely rare
    /// case the keeper's thread is slow to exit is far smaller than the
    /// cost of giving up early and handing back a directory a successor
    /// process cannot then open.
    ///
    /// Idempotent: [`jammi_db::session::JammiSession::close`] is
    /// idempotent, and shutting down an already-stopped keeper finds no
    /// thread left to join.
    pub async fn close(&self) {
        const KEEPER_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
        if let Err(e) = self
            .lease_keeper
            .shutdown_and_join(KEEPER_SHUTDOWN_TIMEOUT)
            .await
        {
            tracing::error!(
                error = %e,
                "InferenceSession::close: the lease keeper did not exit within its shutdown \
                 window; its own catalog connection may still hold the process-exclusive lock"
            );
        }
        self.inner.close().await;
    }

    /// Row-scoped on-read reclaim: a caller that just read `record`
    /// (e.g. `JobService`'s `JobStatus`/`WaitJob`/`ListJobs`) offers it here so
    /// an expired lease is reaped inline with the read, without waiting for the
    /// worker loop's or the construction sweep's next pass. Returns the
    /// reclaimed row when this call performed a requeue-or-fail transition;
    /// `None` when `record` needed no reclaim (not `running`, or a live
    /// lease) — the caller keeps using its own `record` in that case. Shares
    /// `crate::fine_tune::worker::MAX_ATTEMPTS` with the worker loop's own
    /// `reclaim_expired_jobs` call so both reclaim paths apply the identical
    /// attempts cap.
    pub async fn reclaim_job_on_read(
        &self,
        record: &jammi_db::catalog::jobs_repo::JobRecord,
    ) -> Result<Option<jammi_db::catalog::jobs_repo::JobRecord>> {
        let lease = self.worker_intervals()?.lease;
        self.catalog()
            .reclaim_job_on_read(record, lease, crate::fine_tune::worker::MAX_ATTEMPTS)
            .await
    }

    /// Install the engine's compound-query SQL functions on this session
    /// ([`JammiSession::install_functions`]), so SQL — in-process (`sql`) and
    /// over the Flight SQL lane alike — can call them.
    ///
    /// This installs the `annotate(model, task, relation, key, col…)` table
    /// function (model inference as a relation) and the vector-aggregation
    /// UDAFs (`vector_mean`/`vector_sum`/`vector_max`, element-wise reduction
    /// over a group of fixed-width vectors). It must be called once per session,
    /// after the session is behind an `Arc`, because `annotate` holds a
    /// [`std::sync::Weak`] back-reference to the session it serves — weak to
    /// avoid the cycle the strong handle would form (the session owns the
    /// context the function is installed on). The Flight SQL request path
    /// clones this context's state, so installing here makes every function
    /// reachable on every Flight SQL session too.
    pub fn install_query_functions(self: &Arc<Self>) {
        let annotate = QueryFunction::Table {
            name: crate::query::AnnotateTableFunction::NAME.to_string(),
            function: Arc::new(crate::query::AnnotateTableFunction::new(Arc::downgrade(
                self,
            ))),
        };
        self.inner.install_functions(
            std::iter::once(annotate)
                .chain(crate::query::vector_agg_udafs().map(QueryFunction::Aggregate)),
        );
    }

    /// Register a data source.
    pub async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        self.inner
            .add_source(source_id, source_type, connection)
            .await
    }

    /// Remove a source and all associated state (result tables, disk files,
    /// ANN cache, DataFusion registration). Eval runs are preserved.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        self.inner.remove_source(source_id).await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(())
    }

    /// Execute a SQL query.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        self.inner.sql(query).await
    }

    /// [`Self::sql`]'s streamed twin: plan `query` and return the
    /// `SendableRecordBatchStream` rather than collecting it. Forwarded to
    /// [`jammi_db::session::JammiSession::sql_stream`] — see that method's
    /// doc for the tenant-scoping and residency-release guarantees a caller
    /// (the per-rank training-set stream, `fine_tune::stream`) relies on.
    pub async fn sql_stream(
        &self,
        query: &str,
    ) -> Result<datafusion::execution::SendableRecordBatchStream> {
        self.inner.sql_stream(query).await
    }

    /// This session's `[engine] memory_limit`-bounded memory pool. Forwarded
    /// to [`jammi_db::session::JammiSession::memory_pool`] — the route a
    /// caller registers its own [`datafusion::execution::memory_pool::
    /// MemoryConsumer`] through (a training-set stream's per-rank
    /// reservation, an eager read's collected-batch reservation) so it is
    /// bounded by the SAME knob a plan exhausting DataFusion's own operators
    /// would surface [`jammi_db::error::JammiError::ResourcesExhausted`]
    /// against.
    pub fn memory_pool(
        &self,
    ) -> std::sync::Arc<dyn datafusion::execution::memory_pool::MemoryPool> {
        self.inner.memory_pool()
    }

    /// Every ANN index segment of `table_name`, ordered by `segment_id`.
    /// Forwarded to
    /// [`jammi_db::session::JammiSession::list_index_segments`], which owns the
    /// tenant gate (the table is resolved through the tenant-filtered
    /// result-table read first, so an unresolvable table lists nothing).
    pub async fn list_index_segments(
        &self,
        table_name: &str,
    ) -> Result<Vec<jammi_db::catalog::segment_repo::IndexSegment>> {
        self.inner.list_index_segments(table_name).await
    }

    /// This session's loaded configuration — the deployment's own statement
    /// of its devices, its worker knobs and its storage roots.
    pub fn jammi_config(&self) -> &jammi_db::config::JammiConfig {
        self.inner.config()
    }

    /// Access the catalog.
    pub fn catalog(&self) -> &jammi_db::catalog::Catalog {
        self.inner.catalog()
    }

    /// The shared catalog handle behind an `Arc` — the form a [`TrainingJob`]
    /// handle clones to poll its job after the submitting call returns, and
    /// the form `jammi-ballista`'s `CatalogClusterState`/`CatalogJobState`/
    /// `DevicePlacement` need to hold
    /// their own long-lived handle rather than borrowing this session's.
    pub fn catalog_arc(&self) -> &Arc<jammi_db::catalog::Catalog> {
        self.inner.catalog()
    }

    /// Access the topic-catalog repo (used by trigger-stream callers that
    /// do not want to go through Flight SQL DDL).
    pub fn topic_repo(&self) -> Arc<jammi_db::catalog::topic_repo::TopicRepo> {
        self.inner.topic_repo()
    }

    /// Access the trigger-stream publisher.
    pub fn publisher(&self) -> Arc<jammi_db::trigger::Publisher> {
        self.inner.publisher()
    }

    /// Access the trigger-stream subscriber.
    pub fn subscriber(&self) -> Arc<jammi_db::trigger::Subscriber> {
        self.inner.subscriber()
    }

    /// Access the trigger broker the session was constructed with.
    pub fn trigger_broker(&self) -> Arc<dyn jammi_db::trigger::TriggerBroker> {
        self.inner.trigger_broker()
    }

    /// Register a mutable companion table. After this returns the table is
    /// queryable as `mutable.public.<id>` in the same SQL surface that
    /// federates Parquet result tables and external sources.
    pub async fn create_mutable_table(
        &self,
        def: jammi_db::store::mutable::MutableTableDefinition,
    ) -> Result<jammi_db::store::mutable::MutableTableId> {
        self.inner.create_mutable_table(def).await
    }

    /// Drop a mutable companion table.
    pub async fn drop_mutable_table(
        &self,
        id: &jammi_db::store::mutable::MutableTableId,
    ) -> Result<()> {
        self.inner.drop_mutable_table(id).await
    }

    /// Reference to the mutable-table registry.
    pub fn mutable_tables(&self) -> &jammi_db::source::mutable::MutableTableRegistry {
        self.inner.mutable_tables()
    }

    /// List every mutable companion table registered to the session's tenant.
    /// Registry introspection, not a SQL query.
    pub async fn list_mutable_tables(
        &self,
    ) -> Result<Vec<jammi_db::store::mutable::MutableTableDefinition>> {
        Ok(self
            .inner
            .mutable_tables()
            .list(self.inner.tenant())
            .await?)
    }

    /// Bind a tenant scope to this session. Subsequent reads/writes filter
    /// to `tenant_id = t OR tenant_id IS NULL`; writes record `tenant_id = t`.
    ///
    /// This is the sticky form: it mutates session-shared state. For
    /// concurrent gRPC request handlers on a shared `Arc<InferenceSession>`,
    /// prefer [`Self::with_tenant_scoped`].
    pub fn bind_tenant(&self, t: jammi_db::TenantId) {
        self.inner.bind_tenant(t);
    }

    /// Clear the bound tenant.
    pub fn unbind_tenant(&self) {
        self.inner.unbind_tenant();
    }

    /// Return the tenant currently bound, if any.
    pub fn tenant(&self) -> Option<jammi_db::TenantId> {
        self.inner.tenant()
    }

    /// Typed handle to the per-query audit primitive, scoped to this session's
    /// tenant. Delegates to [`jammi_db::session::JammiSession::audit`].
    pub fn audit(&self) -> jammi_db::AuditHandle<'_> {
        self.inner.audit()
    }

    /// Open an ephemeral, session-scoped storage context bound to the tenant
    /// currently set on this connection.
    ///
    /// Tables created through the returned [`jammi_db::EphemeralSession`] are
    /// auto-deleted when it ends (explicit `close`, `Drop`, or timeout), and
    /// every transition publishes to `jammi.audit.session_lifecycle.v1`. The
    /// session shares this connection's `JammiSession` (tenant binding, trigger
    /// broker, catalog), and registers with the connection's timeout-scanner
    /// registry — call [`Self::spawn_ephemeral_timeout_scanner`] once to enforce
    /// timeouts in-process.
    ///
    /// Returns [`jammi_db::EphemeralError::NoTenantBinding`] if no tenant is
    /// bound.
    pub async fn ephemeral_session(
        &self,
        timeout: std::time::Duration,
    ) -> std::result::Result<jammi_db::EphemeralSession, jammi_db::EphemeralError> {
        jammi_db::EphemeralSession::open(
            Arc::clone(&self.inner),
            timeout,
            self.ephemeral_sessions.clone(),
        )
        .await
    }

    /// Shared handle to the ephemeral-session registry the timeout scanner reads.
    pub fn ephemeral_sessions(&self) -> jammi_db::ephemeral::ActiveSessions {
        self.ephemeral_sessions.clone()
    }

    /// Spawn the in-process timeout scanner that force-closes ephemeral sessions
    /// past their deadline. Returns the task handle; the scanner runs until the
    /// handle is aborted or the runtime shuts down. Call once per connection.
    pub fn spawn_ephemeral_timeout_scanner(
        &self,
        interval: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        jammi_db::ephemeral::spawn_timeout_scanner(
            Arc::clone(&self.inner),
            self.ephemeral_sessions.clone(),
            interval,
        )
    }

    /// Run `f` with `tenant` bound for the duration of the closure's future.
    ///
    /// The binding is installed as a Tokio task-local that shadows the
    /// session's sticky shared binding for the executing task only.
    /// Concurrent invocations from different tasks each see their own
    /// `tenant`; no race exists on the shared
    /// `Arc<RwLock<TenantContext>>` because no shared write happens in the
    /// scoped path.
    ///
    /// Delegates to [`jammi_db::session::JammiSession::with_tenant_scoped`];
    /// see that method for the design rationale (Option β: task-local
    /// override rather than per-call session rebuild).
    pub async fn with_tenant_scoped<'a, F, Fut, T>(&'a self, tenant: jammi_db::TenantId, f: F) -> T
    where
        F: FnOnce(jammi_db::TenantScope<'a>) -> Fut,
        Fut: std::future::Future<Output = T> + 'a,
    {
        self.inner.with_tenant_scoped(tenant, f).await
    }

    /// Run `f` with the tenant analyzer rule disabled for the duration of
    /// the closure's future.
    ///
    /// Cross-tenant administrative reads (server-startup recovery scans,
    /// background audit jobs) live here. The closure receives an
    /// [`jammi_db::AdminScope`] handle whose [`jammi_db::AdminScope::sql`] returns
    /// fully materialised batches; once the closure resolves, subsequent
    /// reads on the same session are tenant-filtered again.
    ///
    /// This surface is **not** exposed on the gRPC wire — `jammi-server`
    /// must invoke it only from in-process administrative code paths, not
    /// from a request handler.
    ///
    /// Delegates to [`jammi_db::session::JammiSession::with_admin_scope`];
    /// see that method for the safety contract.
    pub async fn with_admin_scope<'a, F, Fut, T>(&'a self, f: F) -> T
    where
        F: FnOnce(jammi_db::AdminScope<'a>) -> Fut,
        Fut: std::future::Future<Output = T> + 'a,
    {
        self.inner.with_admin_scope(f).await
    }

    /// Access the model cache.
    pub fn model_cache(&self) -> &Arc<ModelCache> {
        &self.model_cache
    }

    /// The handles an `InferenceExec` bound in this process runs against:
    /// this session's model cache and observer.
    pub fn inference_runtime(&self) -> InferenceRuntime {
        InferenceRuntime {
            model: Arc::clone(&self.model_cache) as Arc<dyn jammi_datafusion::ModelRuntime>,
            observer: self.observer.clone(),
        }
    }

    /// Access the result store.
    pub fn result_store(&self) -> Arc<ResultStore> {
        Arc::clone(&self.result_store)
    }

    /// Access the artifact store — the object-store surface model artifacts
    /// (fine-tune adapters, context-predictor weights) are written to and
    /// reloaded through, so a cross-host worker fleet shares trained models.
    pub fn artifact_store(&self) -> Arc<ArtifactStore> {
        Arc::clone(&self.artifact_store)
    }

    /// The one Hugging Face Hub client this session's resolver was built
    /// with — the fine-tune worker's HF fallback path threads this
    /// through `RunFineTuneParams` rather than building its own.
    pub(crate) fn hub(&self) -> &HubSource {
        &self.hub
    }

    /// The device configuration the session resolves candle tensors onto — the
    /// GPU ordinal / CPU fallback every in-process training path builds its
    /// `VarMap` against.
    pub(crate) fn device_config(&self) -> &DeviceConfig {
        &self.device_config
    }

    /// The [`ComputeDevice`](jammi_db::store::manifest::ComputeDevice) this
    /// session effectively runs models on — the device-identity the
    /// materialization contract folds into every result table's definition hash,
    /// so a CPU and a CUDA run of the same model are not falsely reported as a
    /// `Match`.
    pub fn compute_device(&self) -> jammi_db::store::manifest::ComputeDevice {
        crate::model::backend::candle::effective_compute_device(&self.device_config)
    }

    /// The read-only view of the session's DataFusion context, forwarded
    /// from [`JammiSession::context`].
    pub fn context(&self) -> &QueryContext {
        self.inner.context()
    }

    /// The compute-plane slot, forwarded from
    /// [`JammiSession::compute_plane`]: where a compute-plane role installs
    /// the plane this process's materializations are submitted to.
    pub fn compute_plane(&self) -> &Arc<jammi_db::compute_plane::ComputePlaneSlot> {
        self.inner.compute_plane()
    }

    /// The device kind a plan this session builds requires of whoever holds
    /// it: the kind the installed compute plane places onto when the
    /// deployment names one, this session's own otherwise. A plan's
    /// admission is the only reader — where the plan actually runs, and so
    /// what its table records, is the holder's own device.
    pub fn required_device_kind(&self) -> jammi_datafusion::ComputeDeviceKind {
        self.compute_plane()
            .plane()
            .and_then(|plane| plane.device_kind())
            .unwrap_or_else(|| self.compute_device().kind())
    }

    /// Access the engine configuration.
    pub fn inner_config(&self) -> &jammi_db::config::JammiConfig {
        self.inner.config()
    }

    /// Shared handle to the engine's tenant binding. The OSS server's
    /// Flight SQL `TenantBoundProvider` updates this for the duration of
    /// each query so the analyzer rule scopes rows to the bound tenant.
    pub fn tenant_binding_arc(&self) -> jammi_db::tenant_scope::TenantBinding {
        self.inner.tenant_binding_arc()
    }

    /// Access the ANN cache.
    pub fn ann_cache(&self) -> &Arc<AnnCache> {
        &self.ann_cache
    }

    /// Start a vector-search-seeded compound query over an embedding table.
    ///
    /// Returns the fluent [`QueryBuilder`]: the first node is the ANN search,
    /// onto which `join` / `annotate` / `filter` / `select` / `sort` / `limit`
    /// compose. The bounded typed `search` verb (on [`crate::Session`]) is a
    /// thin wrapper over this — vector-search then optional `filter`/`select`
    /// then `run`.
    ///
    /// `method` chooses an approximate search through the table's ANN index
    /// (with an optional per-call oversample) or an exact one.
    pub async fn search(
        self: &Arc<Self>,
        source_id: &str,
        query: Vec<f32>,
        k: usize,
        embedding_table: Option<&str>,
        method: SearchMethod,
    ) -> Result<QueryBuilder> {
        QueryBuilder::new(
            Arc::clone(self),
            source_id,
            query,
            k,
            embedding_table,
            method,
            jammi_db::index::QuerySource::Caller,
        )
        .await
    }

    /// Start a search ranked by an existing row (query-by-example).
    ///
    /// Resolves `row_key`'s stored vector from the source's embedding table
    /// **inside the engine** and delegates to [`Self::search`]. The vector
    /// never crosses the API boundary — this is consistent with the engine's
    /// "no raw-vector reads" line while exposing the standard vector-search
    /// primitive ("rows like this row").
    ///
    /// `embedding_table` selects which table both supplies the example vector
    /// and is searched — the example and its neighbours come from the same
    /// table. `None` selects the source's most-recent ready table.
    pub async fn search_by_id(
        self: &Arc<Self>,
        source_id: &str,
        row_key: &str,
        k: usize,
        embedding_table: Option<&str>,
        method: SearchMethod,
    ) -> Result<QueryBuilder> {
        let table = self
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;
        let pin = self.result_store.pin_current_version(table).await?;
        let query = self
            .result_store
            .read_vector_by_key(self.context(), &pin, row_key)
            .await?;
        // The vector was READ BACK from the table: its provenance is
        // `Stored`, so a non-finite component is a corrupt artifact named by
        // the table (gRPC `Internal`), never the caller's fault.
        QueryBuilder::new(
            Arc::clone(self),
            source_id,
            query,
            k,
            embedding_table,
            method,
            jammi_db::index::QuerySource::Stored {
                table: pin.table_name().to_string(),
            },
        )
        .await
    }

    /// Run a model over `columns` of an arbitrary input plan, appending the
    /// task's inference columns (the `annotate` operation).
    ///
    /// This is the single inference-over-a-relation operator. Both the fluent
    /// [`crate::query::QueryBuilder::annotate`] and the Flight-SQL `annotate`
    /// table function descend through it, so the in-process and remote compound
    /// surfaces run the *same* plan node rather than two reimplementations of
    /// "run inference over these columns".
    ///
    /// The output schema is the inference prefix (`_row_id`, `_source`,
    /// `_model`, `_status`, `_error`, `_latency_ms`) followed by the task's
    /// columns (e.g. a `vector` FixedSizeList for an embedding task). `key_column`
    /// names the input column carried through as `_row_id`.
    pub async fn annotate_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        model: &ModelSource,
        task: ModelTask,
        columns: &[String],
        key_column: &str,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // The output schema is the described model's — its width and its
        // regression head's form — so planning holds no weights.
        let description = self.model_cache.describe(model, task).await?;

        // `input` is caller-supplied — the fluent chain's plan, or the
        // `annotate` table function's scan — with no key order to impose, so
        // its rows are numbered as they arrive.
        let inference = &self.inner.config().inference;
        let spec = InferenceSpec {
            source: model.clone(),
            task,
            content_columns: columns.to_vec(),
            key_column: key_column.to_string(),
            source_id: String::new(),
            chunk: inference.chunk_budget()?,
            embedding_dim: Some(description.embedding_dim()),
            regression_form: description.regression_form().cloned(),
            passthrough: Vec::new(),
            device_kind: self.required_device_kind(),
            partitions: inference.fan_out()?,
        };
        let plan = plan_inference(input, RowOrder::Arrival, spec, self.inference_runtime())?;

        Ok(plan)
    }

    /// Encode a single text query into a vector using the given model.
    pub async fn encode_text_query(&self, model_id: &str, text: &str) -> Result<Vec<f32>> {
        let model_source = ModelSource::parse(model_id);

        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::TextEmbedding)
            .await?;

        // Build a single-row input with the text
        let text_array = Arc::new(arrow::array::StringArray::from(vec![text.to_string()]))
            as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[text_array], ModelTask::TextEmbedding)
            .await
            .map_err(|e| JammiError::Inference(format!("encode_query forward: {e}")))?;

        // A single-row query has no other row to fall back on, so an empty or
        // otherwise refused text row must surface as `Err`, never as the
        // all-zero placeholder a per-row backend substitutes for a
        // decode/preprocess failure (mirroring `encode_image_query` /
        // `encode_audio_query` below).
        Ok(output.single_row_or_err(0)?.to_vec())
    }

    /// Generate embeddings for a source with the given model and modality —
    /// the thin [`Self::run_now`] wrapper unifying the three
    /// modality-specific materializers
    /// ([`Self::generate_text_embeddings`], [`Self::generate_image_embeddings`],
    /// [`Self::generate_audio_embeddings`]) behind one
    /// [`crate::jobs::ComputeSpec::Embedding`]. Submits and executes inline
    /// under [`Self::run_now`]'s claim/lease/finish path, then returns the
    /// terminal [`ResultTableRecord`] + [`CacheOutcome`](jammi_db::store::CacheOutcome)
    /// — so a direct call and a queued-and-claimed `embedding` job of the
    /// same spec run identical code.
    pub async fn generate_embeddings(
        self: &Arc<Self>,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
        modality: jammi_wire::request::Modality,
        cache: jammi_db::store::CachePolicy,
    ) -> Result<(ResultTableRecord, jammi_db::store::CacheOutcome)> {
        let spec = crate::jobs::ComputeSpec::Embedding {
            source_id: source_id.to_string(),
            model_id: model_id.to_string(),
            columns: columns.to_vec(),
            key_column: key_column.to_string(),
            modality,
            cache,
        };
        match self.run_now(spec).await? {
            crate::jobs::JobResult::Table {
                table,
                cache_outcome,
            } => {
                let record = self
                    .catalog()
                    .get_result_table(&table)
                    .await?
                    .ok_or_else(|| {
                        JammiError::Catalog(format!(
                            "generate_embeddings: run_now's own table '{table}' vanished \
                             before it could be read back"
                        ))
                    })?;
                Ok((record, cache_outcome))
            }
            crate::jobs::JobResult::Model { .. } => Err(JammiError::Inference(
                "generate_embeddings: run_now returned a training JobResult for a compute spec"
                    .into(),
            )),
        }
    }

    /// Generate embeddings for a source and persist to Jammi DB.
    /// Invalidates the ANN cache for this source after completion.
    ///
    /// One of [`Self::generate_embeddings`]'s three modality-specific
    /// materializers — see that method's doc for the `job_attempt`
    /// convention every `*_materialize`-shaped method shares (this one is
    /// not itself suffixed `_materialize` since it is already
    /// modality-specific, never a public verb name on its own).
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_text_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
        cache: jammi_db::store::CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(ResultTableRecord, jammi_db::store::CacheOutcome)> {
        let result = EmbeddingPipeline::new(self, &self.result_store, ModelTask::TextEmbedding)
            .run(source_id, model_id, columns, key_column, cache, job_attempt)
            .await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(result)
    }

    /// Read the `vector` column of a pinned embedding result table into one
    /// `Vec<f32>` per row — [`ResultStore::read_vectors`] on this session's
    /// context. Takes the [`PinnedSource`] rather than a bare record: the
    /// rows come from the version the caller pinned, never from whatever
    /// this session happens to have bound.
    pub async fn read_vectors(&self, pin: &PinnedSource) -> Result<Vec<Vec<f32>>> {
        self.result_store.read_vectors(self.context(), pin).await
    }

    /// Read the paired `(_row_id, vector)` rows of a precomputed-embedding
    /// Parquet object at `url`, resolved through the session's storage registry.
    /// The read leg of [`Self::import_embeddings`]; delegates to
    /// [`jammi_db::session::JammiSession::read_keyed_vectors`].
    pub(crate) async fn read_keyed_vectors(
        &self,
        url: &jammi_db::storage::StorageUrl,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        self.inner.read_keyed_vectors(url).await
    }

    /// Register precomputed per-row vectors at `vectors_url` as a ready
    /// `(source_id, TextEmbedding, model_id)` embedding table without re-running
    /// any encoder — a thin promotion of precomputed vectors through the single
    /// materialization funnel.
    ///
    /// Each incoming vector is L2-normalized (every embedding table holds
    /// normalized vectors — the cosine ANN sidecar assumes it); a zero-norm
    /// vector is rejected. `model_id` is validated as a well-formed encoder
    /// reference and canonicalized, never loaded, so import runs GPU-free.
    /// `key_column` / `text_columns` are recorded as catalog provenance; the
    /// physical key stays `_row_id`. The resulting table is recompute-inert (its
    /// `External` descriptor refuses replay). Reads the input object fully into
    /// memory.
    pub async fn import_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        vectors_url: &jammi_db::storage::StorageUrl,
        key_column: &str,
        text_columns: &[String],
        dimensions: usize,
    ) -> Result<ResultTableRecord> {
        let record = crate::pipeline::import::ImportPipeline::new(self, &self.result_store)
            .run(
                source_id,
                model_id,
                vectors_url,
                key_column,
                text_columns,
                dimensions,
            )
            .await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(record)
    }

    /// Generate image embeddings for a source and persist to Jammi DB. See
    /// [`Self::generate_text_embeddings`]'s doc for the `job_attempt`
    /// convention.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_image_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        image_column: &str,
        key_column: &str,
        cache: jammi_db::store::CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(ResultTableRecord, jammi_db::store::CacheOutcome)> {
        let result = EmbeddingPipeline::new(self, &self.result_store, ModelTask::ImageEmbedding)
            .run(
                source_id,
                model_id,
                &[image_column.to_string()],
                key_column,
                cache,
                job_attempt,
            )
            .await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(result)
    }

    /// Encode a single image into a vector using the given vision model.
    pub async fn encode_image_query(&self, model_id: &str, image_bytes: &[u8]) -> Result<Vec<f32>> {
        let model_source = ModelSource::parse(model_id);

        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::ImageEmbedding)
            .await?;

        let binary_array =
            Arc::new(arrow::array::BinaryArray::from(vec![image_bytes])) as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[binary_array], ModelTask::ImageEmbedding)
            .await
            .map_err(|e| JammiError::Inference(format!("encode_image_query forward: {e}")))?;

        // A single-row query has no other row to fall back on, so a corrupt
        // image must surface as `Err`, never as the all-zero placeholder row
        // the backend writes for a decode/preprocess failure.
        Ok(output.single_row_or_err(0)?.to_vec())
    }

    /// Generate audio embeddings for a source and persist to Jammi DB.
    ///
    /// Peer of [`Self::generate_image_embeddings`]: scans `audio_column` (raw
    /// encoded audio bytes or file paths), decodes → resamples → log-mel →
    /// CLAP audio tower, and writes one L2-normalized vector per row. Reuses
    /// the modality-agnostic [`EmbeddingPipeline`] unchanged.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate_audio_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        audio_column: &str,
        key_column: &str,
        cache: jammi_db::store::CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(ResultTableRecord, jammi_db::store::CacheOutcome)> {
        let result = EmbeddingPipeline::new(self, &self.result_store, ModelTask::AudioEmbedding)
            .run(
                source_id,
                model_id,
                &[audio_column.to_string()],
                key_column,
                cache,
                job_attempt,
            )
            .await?;
        self.ann_cache.invalidate_source(source_id)?;
        Ok(result)
    }

    /// Encode a single audio clip into a vector using the given audio model.
    ///
    /// Peer of [`Self::encode_image_query`]: `audio_bytes` is an encoded clip
    /// (WAV/FLAC/MP3/Ogg); the backend owns decode → resample → log-mel →
    /// forward, returning the L2-normalized shared-latent embedding.
    pub async fn encode_audio_query(&self, model_id: &str, audio_bytes: &[u8]) -> Result<Vec<f32>> {
        let model_source = ModelSource::parse(model_id);

        let guard = self
            .model_cache
            .get_or_load(&model_source, ModelTask::AudioEmbedding)
            .await?;

        let binary_array =
            Arc::new(arrow::array::BinaryArray::from(vec![audio_bytes])) as arrow::array::ArrayRef;
        let output = guard
            .model
            .forward(&[binary_array], ModelTask::AudioEmbedding)
            .await
            .map_err(|e| JammiError::Inference(format!("encode_audio_query forward: {e}")))?;

        // A single-row query has no other row to fall back on, so a corrupt
        // clip must surface as `Err`, never as the all-zero placeholder row
        // the backend writes for a decode/preprocess failure.
        Ok(output.single_row_or_err(0)?.to_vec())
    }

    /// TEST-ONLY non-vacuity seam for the regression surface. Loads a fresh,
    /// unshared copy of `model` (off the same resolve + backend load path serving
    /// uses), optionally zeroes its trained distribution head, runs a regression
    /// forward pass over `texts`, and returns the de-standardised served value of
    /// distribution column 0 (the Gaussian mean / the lowest quantile) per row.
    ///
    /// With `zero_head = false` this returns exactly what the head learned, so a
    /// test can confirm two text groups separate. With `zero_head = true` the head
    /// is collapsed to its zero-initialised base, which emits the scaler offset
    /// `μ_y` for every input — so the SAME separation assertion must FAIL,
    /// proving the trained test is non-vacuous (it measures learning, not the
    /// scaler centring at μ). Production never calls this; it mutates only the
    /// per-call owned model.
    #[doc(hidden)]
    pub async fn served_regression_col0_for_test(
        &self,
        model: &ModelSource,
        texts: &[String],
        zero_head: bool,
    ) -> Result<Vec<f32>> {
        self.served_regression_col_for_test(model, texts, zero_head, 0)
            .await
    }

    /// TEST-ONLY non-vacuity seam for the regression surface, generalized
    /// from [`Self::served_regression_col0_for_test`] to any distribution
    /// column (`col_idx < head_width` — a Gaussian head has `head_width ==
    /// 2`, a quantile head `head_width == quantile_levels.len()`), so a
    /// power-limited seed sweep (`untrained_quantile_head_collapses_
    /// to_mu_no_separation` in `tests/it/regression_surface.rs`) can pool
    /// EVERY already-trained quantile level's separation, not only column 0,
    /// for `3x` the independent samples per seed at ZERO extra training cost
    /// (same already-trained model, re-served at a different column) —
    /// `served_regression_col0_for_test` keeps its own signature (six
    /// existing call sites) and delegates here at `col_idx = 0`.
    #[doc(hidden)]
    pub async fn served_regression_col_for_test(
        &self,
        model: &ModelSource,
        texts: &[String],
        zero_head: bool,
        col_idx: usize,
    ) -> Result<Vec<f32>> {
        use arrow::array::StringArray;

        let mut loaded = self
            .model_cache
            .load_owned_for_test(model, ModelTask::Regression)
            .await?;
        if zero_head {
            loaded.zero_distribution_head_for_test();
        }
        let col: arrow::array::ArrayRef = Arc::new(StringArray::from(texts.to_vec()));
        let output = loaded.forward(&[col], ModelTask::Regression).await?;
        extract_test_column(&output, col_idx)
    }

    /// Run inference on a registered source using a model — the thin
    /// [`Self::run_now`] wrapper every embedded caller reaches:
    /// submits a [`crate::jobs::ComputeSpec::Infer`], executes it inline
    /// under the same claim/lease/finish path a queued `infer` kind runs,
    /// and returns the terminal rows read back through the SAME ordered SQL
    /// `Self::infer_materialize`'s own cache-hit arm uses
    /// (`ORDER BY _row_id, _ordinal`) — so a `run_now` call and a
    /// queued-and-claimed `infer` job of the same spec produce byte-identical
    /// rows in byte-identical order.
    #[allow(clippy::too_many_arguments)]
    pub async fn infer(
        self: &Arc<Self>,
        source_id: &str,
        source: &ModelSource,
        task: ModelTask,
        content_columns: &[String],
        key_column: &str,
        cache: jammi_db::store::CachePolicy,
    ) -> Result<(Vec<RecordBatch>, jammi_db::store::CacheOutcome)> {
        let spec = crate::jobs::ComputeSpec::Infer {
            source_id: source_id.to_string(),
            model_id: source.to_string(),
            task,
            content_columns: content_columns.to_vec(),
            key_column: key_column.to_string(),
            cache,
        };
        match self.run_now(spec).await? {
            crate::jobs::JobResult::Table {
                table,
                cache_outcome,
            } => {
                let batches = self.sql(&infer_ordered_read_back_sql(&table)).await?;
                let batches = normalize_view_batches(batches)?;
                Ok((batches, cache_outcome))
            }
            crate::jobs::JobResult::Model { .. } => Err(JammiError::Inference(
                "infer: run_now returned a training JobResult for a compute spec".into(),
            )),
        }
    }

    /// `infer`'s actual materializer — dispatched to by
    /// [`crate::jobs::execute_compute`] (from [`Self::run_now`] or a claimed
    /// `infer` job) with the claim's [`JobAttempt`](jammi_db::catalog::result_repo::JobAttempt)
    /// so the result table's `partial_result` CAS lands under the correct
    /// attempt; every OTHER internal caller that materializes directly
    /// without a job of record ([`crate::pipeline::recompute`]'s replay,
    /// [`crate::eval::runner::EvalRunner`]) passes `None`.
    ///
    /// Scans the source, feeds `content_columns` through the model, and
    /// returns RecordBatches with prefix + task-specific columns. The
    /// per-row `_status`/`_error` prefix columns are reserved for **pre-forward
    /// input validation** (an empty/null content row is annotated
    /// `_status = "error"` before the model ever sees it) — a `model.forward`
    /// failure itself is always systemic (a broken kernel, a
    /// contiguity/PTX/dtype mismatch, or a model incapable of the requested
    /// task), so it fails this call loudly as an `Err` rather than being
    /// annotated as an all-`_status = "error"` relation.
    ///
    /// An inference ALWAYS creates its (possibly empty) result table — even a
    /// zero-row scan is a real, queryable artifact — and every read-back
    /// (this materialize path's own fresh-compute return, and its cache-hit
    /// short-circuit above) reads it back through the identical
    /// `ORDER BY _row_id, _ordinal` query
    /// ([`infer_ordered_read_back_sql`]), so a caller sees the SAME row order
    /// regardless of which arm ran and regardless of the order the
    /// underlying scan or model batches actually arrived in.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn infer_materialize(
        &self,
        source_id: &str,
        source: &ModelSource,
        task: ModelTask,
        content_columns: &[String],
        key_column: &str,
        cache: jammi_db::store::CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(String, Vec<RecordBatch>, jammi_db::store::CacheOutcome)> {
        // Validate content columns are not empty
        if content_columns.is_empty() {
            return Err(JammiError::Inference(
                "At least one content column is required".into(),
            ));
        }

        let table_name = self.find_table_name(source_id).await?;
        let query = self.build_source_query(source_id, &table_name, key_column, content_columns);

        let df = self.inner.context().sql(&query).await.map_err(|e| {
            JammiError::Inference(format!("Failed to scan source '{source_id}': {e}"))
        })?;

        let input_plan = df
            .create_physical_plan()
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to create scan plan: {e}")))?;
        // Describe the model for the schema and the definition: its width,
        // its regression head's form, and the identity the environment
        // records. Nothing is loaded here — the executing process
        // materializes the weights when the plan runs.
        let description = self.model_cache.describe(source, task).await?;
        let identity = description.identity();

        // The materialization contract is knowable here (model described,
        // source named), so the cache probe keys on the identical definition + anchors
        // the funnel records at finalize. The sole input is the raw source with
        // no version surface → `UnpinnedAtInstant`, so a `Use` request is
        // honestly always a miss; the probe still runs for surface uniformity.
        let descriptor = jammi_db::store::manifest::ProducingDescriptor::Inference {
            model_id: source.to_string(),
            task,
            source_id: source_id.to_string(),
            content_columns: content_columns.to_vec(),
            key_column: key_column.to_string(),
        };
        let env = jammi_db::store::manifest::MaterializationEnv::of_models(
            self.compute_device(),
            vec![identity],
        );
        let inputs = vec![jammi_db::store::manifest::InputAnchor::unpinned_at_instant(
            source_id,
            chrono::Utc::now().to_rfc3339(),
        )];

        if cache == jammi_db::store::CachePolicy::Use {
            let def_hash = jammi_db::store::manifest::MaterializationManifest::definition_of(
                &descriptor,
                &env,
            )
            .map_err(jammi_db::store::manifest_to_jammi)?;
            if let Some(reused) = self
                .result_store
                .probe_cache_record(&def_hash, &inputs)
                .await?
            {
                // A sound hit: return the cached table's rows (not a fresh
                // compute) and report the reuse. Inference anchors unpinned, so
                // this never fires today — but the path is correct the moment a
                // versioned source makes inference cacheable. Read back through
                // the SAME ordered query the fresh-compute arm below uses, so a
                // cache hit and a cache miss return byte-identical row order.
                let batches = self
                    .sql(&infer_ordered_read_back_sql(&reused.table_name))
                    .await?;
                let batches = normalize_view_batches(batches)?;
                let outcome = jammi_db::store::CacheOutcome::Reused(
                    jammi_db::store::ReusedArtifact::Table(reused.name()),
                );
                return Ok((reused.table_name, batches, outcome));
            }
        }

        // The same plan the embedding pipeline runs: the keyed input refuses
        // a null key typed before any model call, and the total order makes
        // the output deterministic across `execution_threads`.
        let inference = &self.inner.config().inference;
        let spec = InferenceSpec {
            source: source.clone(),
            task,
            content_columns: content_columns.to_vec(),
            key_column: key_column.to_string(),
            source_id: source_id.to_string(),
            chunk: inference.chunk_budget()?,
            embedding_dim: Some(description.embedding_dim()),
            regression_form: description.regression_form().cloned(),
            passthrough: Vec::new(),
            device_kind: self.required_device_kind(),
            partitions: inference.fan_out()?,
        };
        let inference_exec = plan_inference(
            input_plan,
            RowOrder::Keyed {
                key_column: key_column.to_string(),
                tie_breakers: vec![jammi_db::store::schema::CONTENT_HASH_COLUMN.to_string()],
            },
            spec,
            self.inference_runtime(),
        )?;

        // An inference always creates its (possibly empty) result table — a
        // zero-row scan is a real, queryable artifact too, never a case the
        // producer silently skips materializing. Every row the plan
        // produces is written through the sink where the compute plane says
        // (a typed refusal raised inside the plan, placed or not, reaches
        // the caller as that variant).
        let mut building = self
            .result_store
            .create_table(
                source_id,
                task,
                jammi_db::catalog::result_repo::ResultTableKind::Model,
                None,
                &source.to_string(),
                None,
                None,
                None,
                job_attempt,
            )
            .await?;
        let summary = self
            .result_store
            .write_result_table(
                &mut building,
                jammi_db::store::SinkKind::Rows,
                inference_exec,
                self.inner.context().task_ctx(),
            )
            .await?;
        let row_count = summary.rows as usize;

        // Finish with the descriptor and anchors built at the top and the
        // environment the process that ran the plan reports — the cache
        // probe keyed on this process's prediction of it. Every `?` above
        // unwinds through the handle's Drop (a best-effort `building ->
        // failed` CAS, no byte deletion); `finish` is the single `building
        // -> ready` funnel, renewing the writer's lease before it attests.
        let record = building
            .finish(
                self.inner.context(),
                row_count,
                jammi_db::store::manifest::Materialization::new(&descriptor, &summary.env, inputs),
            )
            .await?;

        // Read back through the ordered query — see this method's doc for
        // why the fresh-compute arm re-reads rather than returning the
        // in-memory `batches` directly: it is the only way the returned row
        // order is provably identical to the cache-hit arm's, regardless of
        // how the underlying scan or model batches actually arrived.
        let ordered = self
            .sql(&infer_ordered_read_back_sql(&record.table_name))
            .await?;
        let ordered = normalize_view_batches(ordered)?;
        Ok((
            record.table_name,
            ordered,
            jammi_db::store::CacheOutcome::Computed,
        ))
    }

    /// Build a SELECT query for the key + content columns from a source table.
    pub(crate) fn build_source_query(
        &self,
        source_id: &str,
        table_name: &str,
        key_column: &str,
        content_columns: &[String],
    ) -> String {
        let all_columns: Vec<&str> = std::iter::once(key_column)
            .chain(content_columns.iter().map(|s| s.as_str()))
            .collect();
        let select_list = all_columns
            .iter()
            .map(|c| quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ");
        // The per-row content hash over the RAW content columns, computed by
        // the `jammi_content_hash` UDF with the runner's own rendering (never
        // a SQL `CAST`), carried by every model-facing scan — the embedding
        // pipeline persists it as the table's `_content_hash`, `infer`
        // ignores it, and an incremental refresh classifies rows by it.
        let hash_args = content_columns
            .iter()
            .map(|c| quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "SELECT {select_list}, {}({hash_args}) AS {} FROM {}",
            crate::query::CONTENT_HASH_UDF_NAME,
            jammi_db::store::schema::CONTENT_HASH_COLUMN,
            source_relation(source_id, table_name)
        )
    }

    /// The first table a source serves, resolved through the catalog's
    /// `sources` row ([`JammiSession::source_table_names`]) — a source
    /// registered on any replica resolves here, and a missing one is
    /// [`JammiError::SourceNotFound`].
    pub(crate) async fn find_table_name(&self, source_id: &str) -> Result<String> {
        self.inner
            .source_table_names(source_id)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| {
                JammiError::Inference(format!("No tables found in source '{source_id}'"))
            })
    }

    /// Materialize the k-nearest-neighbour graph of a source's embedding table
    /// as a queryable edge `result_table` — the thin [`Self::run_now`] wrapper:
    /// submits a [`crate::jobs::ComputeSpec::NeighborGraph`] and
    /// returns the terminal [`ResultTableRecord`] + [`CacheOutcome`](jammi_db::store::CacheOutcome)
    /// [`Self::run_now`] produced, so a direct call and a queued-and-claimed
    /// `neighbor_graph` job of the same spec run identical code
    /// (`Self::build_neighbor_graph_materialize`, through
    /// [`crate::jobs::execute_compute`]).
    ///
    /// This is for *global-structure* work — clustering, near-duplicate
    /// detection, connected components, graph-aware training-data generation —
    /// where the whole edge set is consumed as a durable artifact. For
    /// "neighbours of *these* rows", compose [`Self::search`] instead.
    ///
    /// `cache` opts the build into memoization: under
    /// [`CachePolicy::Use`](jammi_db::store::CachePolicy::Use) an exact prior
    /// materialisation (same definition over the same source-table digest) is
    /// reused instead of rebuilt, and the returned
    /// [`CacheOutcome`](jammi_db::store::CacheOutcome) reports which path ran. The
    /// default [`Bypass`](jammi_db::store::CachePolicy::Bypass) always rebuilds.
    pub async fn build_neighbor_graph(
        self: &Arc<Self>,
        source_id: &str,
        embedding_table: Option<&str>,
        params: &crate::pipeline::neighbor_graph::BuildNeighborGraph,
        cache: jammi_db::store::CachePolicy,
    ) -> Result<(ResultTableRecord, jammi_db::store::CacheOutcome)> {
        let spec = crate::jobs::ComputeSpec::NeighborGraph {
            source_id: source_id.to_string(),
            embedding_table: embedding_table.map(str::to_string),
            params: params.clone(),
            cache,
        };
        match self.run_now(spec).await? {
            crate::jobs::JobResult::Table {
                table,
                cache_outcome,
            } => {
                let record = self
                    .catalog()
                    .get_result_table(&table)
                    .await?
                    .ok_or_else(|| {
                        JammiError::Catalog(format!(
                            "build_neighbor_graph: run_now's own table '{table}' vanished \
                             before it could be read back"
                        ))
                    })?;
                Ok((record, cache_outcome))
            }
            crate::jobs::JobResult::Model { .. } => Err(JammiError::Inference(
                "build_neighbor_graph: run_now returned a training JobResult for a compute spec"
                    .into(),
            )),
        }
    }

    /// `build_neighbor_graph`'s actual materializer — see
    /// `Self::infer_materialize`'s doc for the `job_attempt` convention
    /// every `*_materialize` method shares.
    ///
    /// The build resolves the input embedding table through the same
    /// tenant-scoped catalog path `search` uses: when a tenant is bound it runs
    /// inside that tenant's scope, so a caller cannot point the build at another
    /// tenant's table. The returned edge table is `kind = neighbor_graph`,
    /// derived from the resolved embedding table, with `src`/`dst` endpoints
    /// that join directly to source data on the key.
    ///
    /// The default driver is index-assisted and produces an *approximate*,
    /// *non-deterministic* graph; set `BuildNeighborGraph::exact` for a
    /// deterministic, complete one (gated by a row-count ceiling).
    pub(crate) async fn build_neighbor_graph_materialize(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        params: &crate::pipeline::neighbor_graph::BuildNeighborGraph,
        cache: jammi_db::store::CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(ResultTableRecord, jammi_db::store::CacheOutcome)> {
        match self.tenant() {
            // A bound tenant runs the build inside its scope, so the catalog
            // resolves only that tenant's embedding table — a caller cannot
            // point the build at another tenant's table.
            Some(tenant) => {
                self.with_tenant_scoped(tenant, |_scope| async move {
                    crate::pipeline::neighbor_graph::NeighborGraphPipeline::new(
                        self,
                        self.result_store.as_ref(),
                    )
                    .run(source_id, embedding_table, params, cache, job_attempt)
                    .await
                })
                .await
            }
            None => {
                crate::pipeline::neighbor_graph::NeighborGraphPipeline::new(
                    self,
                    self.result_store.as_ref(),
                )
                .run(source_id, embedding_table, params, cache, job_attempt)
                .await
            }
        }
    }

    /// Assemble a point-in-time-correct table — the thin [`Self::run_now`]
    /// wrapper: submits a [`crate::jobs::ComputeSpec::AsofJoin`]
    /// and returns the terminal [`ResultTableRecord`] [`Self::run_now`]
    /// produced, so a direct call and a queued-and-claimed `asof_join` job of
    /// the same spec run identical code
    /// (`Self::asof_join_materialize`, through
    /// [`crate::jobs::execute_compute`]).
    ///
    /// `spine` and `facts` are registered source ids. The
    /// [`AsofJoinSpec`](crate::pipeline::asof::AsofJoinSpec) carries the four
    /// pinned knobs (direction, boundary, tolerance, tie-break) and the
    /// equality/temporal key roles. An as-of join always recomputes — it has
    /// no cache-opt-in surface (every input is honestly `UnpinnedAtInstant`).
    pub async fn asof_join(
        self: &Arc<Self>,
        spine: &str,
        facts: &str,
        spec: &crate::pipeline::asof::AsofJoinSpec,
    ) -> Result<ResultTableRecord> {
        let job_spec = crate::jobs::ComputeSpec::AsofJoin {
            spine: spine.to_string(),
            facts: facts.to_string(),
            spec: spec.clone(),
        };
        match self.run_now(job_spec).await? {
            crate::jobs::JobResult::Table { table, .. } => self
                .catalog()
                .get_result_table(&table)
                .await?
                .ok_or_else(|| {
                    JammiError::Catalog(format!(
                        "asof_join: run_now's own table '{table}' vanished before it could be \
                         read back"
                    ))
                }),
            crate::jobs::JobResult::Model { .. } => Err(JammiError::Inference(
                "asof_join: run_now returned a training JobResult for a compute spec".into(),
            )),
        }
    }

    /// `asof_join`'s actual materializer — see `Self::infer_materialize`'s
    /// doc for the `job_attempt` convention every `*_materialize` method
    /// shares.
    ///
    /// Writes a result table (carrying the materialization manifest) and
    /// returns its record. Left rows are always preserved; unmatched fact
    /// columns are null. Both relations are resolved through the session's
    /// tenant-scoped catalog: when a tenant is bound the join runs inside that
    /// tenant's scope, so a caller cannot point either side at another
    /// tenant's relation.
    pub(crate) async fn asof_join_materialize(
        &self,
        spine: &str,
        facts: &str,
        spec: &crate::pipeline::asof::AsofJoinSpec,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<ResultTableRecord> {
        match self.tenant() {
            Some(tenant) => {
                self.with_tenant_scoped(tenant, |_scope| async move {
                    crate::pipeline::asof::verb::run(self, spine, facts, spec, job_attempt).await
                })
                .await
            }
            None => crate::pipeline::asof::verb::run(self, spine, facts, spec, job_attempt).await,
        }
    }

    // =====================================================================
    // Fine-tuning
    // =====================================================================

    /// Submit a LoRA fine-tuning job on a registered source.
    ///
    /// Persists a self-describing [`TrainingSpec::FineTune`] into a `queued`
    /// catalog job and returns a [`TrainingJob`] handle immediately — the
    /// training runs later under a [`crate::fine_tune::worker::JobWorker`]
    /// that claims the job under a lease, reconstructs the data loader from the
    /// persisted source + columns, and trains while heartbeating. Call
    /// `job.wait().await` to block until a worker drives the job to a terminal
    /// state.
    pub async fn fine_tune(
        &self,
        source: &str,
        base_model: &str,
        columns: &[String],
        method: FineTuneMethod,
        task: ModelTask,
        config: Option<FineTuneConfig>,
    ) -> Result<TrainingJob> {
        let config = config.unwrap_or_default();
        config.validate()?;

        let spec = TrainingSpec::FineTune {
            source: source.to_string(),
            columns: columns.to_vec(),
            method,
            task,
            common: TrainingCommon {
                base_model: base_model.to_string(),
                config: config.clone(),
                world_size: crate::fine_tune::spec::DEFAULT_WORLD_SIZE,
                cache: jammi_db::store::CachePolicy::Bypass,
            },
            // This loose-argument entry point has no cache parameter to
            // carry a caller's choice; a caller that wants the model-level
            // reuse probe builds the spec directly and submits it through
            // `Self::run_training_spec`. Matches the pre-`cache` behaviour
            // byte-for-byte.
        };
        self.submit_fine_tune_spec(spec).await
    }

    /// Submit a column-source fine-tune from the flattened request shape the
    /// data-plane client also submits
    /// ([`jammi_wire::request::FineTuneRequest`]).
    ///
    /// [`Self::fine_tune`]'s loose parameter list cannot carry the rank count
    /// without growing an argument every caller that does not choose one must
    /// still pass, so the count arrives on a request struct — the same one
    /// `jammi_client::DataClient::submit_fine_tune` takes. One request shape
    /// on both surfaces is what makes the remote-equals-embedded parity
    /// oracle a comparison of the two paths rather than of two request
    /// vocabularies.
    ///
    /// `world_size: None` is UNSET and submits the single-rank job
    /// [`Self::fine_tune`] submits;
    /// [`jammi_wire::request::FineTuneRequest::world_size`]'s `NonZeroU32`
    /// keeps a zero-rank request unrepresentable at this edge. A count this
    /// deployment cannot serve is refused with a typed error and enqueues
    /// nothing.
    pub async fn submit_fine_tune(
        &self,
        request: jammi_wire::request::FineTuneRequest,
    ) -> Result<TrainingJob> {
        let jammi_wire::request::FineTuneRequest {
            source,
            base_model,
            columns,
            method,
            task,
            config,
            world_size,
            cache,
        } = request;
        let config = config.unwrap_or_default();
        config.validate()?;

        let spec = TrainingSpec::FineTune {
            source,
            columns,
            method,
            task,
            common: TrainingCommon {
                base_model,
                config,
                world_size: world_size
                    .map_or(crate::fine_tune::spec::DEFAULT_WORLD_SIZE, |w| w.get()),
                cache,
            },
        };
        self.submit_fine_tune_spec(spec).await
    }

    /// Submit a job carrying one of the two LoRA fine-tune specs. Shared by the
    /// column-source [`Self::fine_tune`] and the graph [`Self::fine_tune_graph`]
    /// paths — the only thing that differs upstream is which spec variant is
    /// built. No data is read and no model is loaded here; the worker does both
    /// from the persisted spec. The row's `model_ref`/`output_model_id` come
    /// from [`Self::training_job_links`], the one derivation `enqueue` and
    /// the context-predictor submit share.
    async fn submit_fine_tune_spec(&self, spec: TrainingSpec) -> Result<TrainingJob> {
        self.submit_fine_tune_spec_deduped(spec, None).await
    }

    /// [`Self::submit_fine_tune_spec`], additionally deduped by an optional
    /// per-tenant `idempotency_key` (migration 030) via
    /// [`jammi_db::catalog::Catalog::submit_job_deduped`] — see that
    /// method's doc for the atomicity guarantee. `idempotency_key` of `None`
    /// always inserts fresh, matching [`Self::submit_fine_tune_spec`]
    /// byte-for-byte (every embedded caller of that method routes through
    /// here with `None`). `Some(key)` that collides with a still-known prior
    /// submission returns THAT job's handle — re-read from its catalog row,
    /// never the handle this call would have minted — so a caller cannot
    /// observe two different in-memory handles for what the catalog now
    /// records as one job.
    async fn submit_fine_tune_spec_deduped(
        &self,
        spec: TrainingSpec,
        idempotency_key: Option<&str>,
    ) -> Result<TrainingJob> {
        // The ONE admission (per-kind validation, rank admission) is
        // applied HERE, before anything durable
        // exists: this method is one of the durable submit edges for a
        // training spec, so every entry path that reaches it —
        // [`Self::fine_tune`], [`Self::fine_tune_graph`],
        // [`Self::submit_fine_tune`] and [`Self::run_training_spec_deduped`]
        // (which the gRPC handler and the Python binding both drive) — is
        // admitted by this call. [`InferenceSession::enqueue`] and
        // `train_context_predictor_deduped` are the other two durable
        // edges; all three call
        // [`crate::fine_tune::spec::admit_training_spec`]. A refusal leaves
        // no row behind, because no row has been written yet. Consuming
        // `spec` here and reading it back only through `admitted.spec()` is
        // the witness enforcement: there is no path below this line that
        // could serialize/submit the pre-admission `spec` value, because
        // that binding is consumed. This function builds no
        // `SubmitJobParams` of its own:
        // [`crate::fine_tune::spec::submit_admitted_training`] is the one
        // place that construction happens.
        let admitted = crate::fine_tune::spec::admit_training_spec(self.inner.config(), spec)?;
        let job_id = uuid::Uuid::new_v4().to_string();
        let links = self.training_job_links(admitted.spec(), &job_id).await?;
        let submitted = crate::fine_tune::spec::submit_admitted_training(
            self.inner.catalog(),
            &admitted,
            &job_id,
            &links.model_ref,
            &links.output_model_id,
            0,
            idempotency_key,
        )
        .await?;
        let recorded_job_id = submitted.recorded_job_id;

        if recorded_job_id == job_id {
            Ok(TrainingJob::new(
                job_id,
                "queued".into(),
                links.output_model_id,
                Arc::clone(self.inner.catalog()),
            ))
        } else {
            // A concurrent or prior call already holds this idempotency key
            // — the durable row of record, not this call's own (unused)
            // job_id. Re-read its current state rather than guessing
            // `"queued"`: a racing caller may observe it already claimed.
            let record = self.inner.catalog().get_job(&recorded_job_id).await?;
            let model_id = resolve_model_id(&recorded_job_id, &record)?;
            Ok(TrainingJob::new(
                recorded_job_id,
                record.status,
                model_id,
                Arc::clone(self.inner.catalog()),
            ))
        }
    }

    /// The two model-side links every training kind's `jobs` row carries,
    /// derived ONCE for every submitter — the dedicated entry points
    /// ([`Self::fine_tune`], [`Self::fine_tune_graph`],
    /// [`Self::train_context_predictor`]) and the generic
    /// [`Self::enqueue`] alike — so a row cannot be linked differently
    /// depending on which door it came through. `model_ref` is the base
    /// model's catalog PK (the row is registered first when absent);
    /// `output_model_id` is the NAME the finish CAS mints the output under
    /// (`fine_tuned_model_id(job_id)` for the two LoRA kinds, the spec's
    /// own `model_id` for a context predictor).
    pub(crate) async fn training_job_links(
        &self,
        spec: &TrainingSpec,
        job_id: &str,
    ) -> Result<TrainingJobLinks> {
        match spec {
            TrainingSpec::FineTune { task, common, .. } => Ok(TrainingJobLinks {
                model_ref: self.ensure_base_model_pk(&common.base_model, *task).await?,
                output_model_id: fine_tuned_model_id(job_id),
            }),
            // A graph fine-tune trains a text-embedding metric over the node
            // source's text; the edges only supervise the pairing.
            TrainingSpec::GraphFineTune { common, .. } => Ok(TrainingJobLinks {
                model_ref: self
                    .ensure_base_model_pk(&common.base_model, ModelTask::TextEmbedding)
                    .await?,
                output_model_id: fine_tuned_model_id(job_id),
            }),
            TrainingSpec::ContextPredictor {
                source,
                predictor_spec,
            } => Ok(TrainingJobLinks {
                model_ref: self
                    .context_predictor_base_model_pk(source, predictor_spec)
                    .await?,
                output_model_id: predictor_spec.model_id.clone(),
            }),
        }
    }

    /// Resolve `base_model` to its catalog PK, registering the row first
    /// when the catalog has none (the worker resolves the same row when it
    /// loads weights). The `jobs.model_ref` FK must bind to the RESOLVED
    /// row's PK, not a reconstructed `name::version`: a tenant fine-tuning
    /// a global base model references the global (unqualified) PK, and one
    /// fine-tuning its own model references its tenant-qualified PK — the
    /// resolved record's `catalog_pk` carries whichever applies.
    async fn ensure_base_model_pk(&self, base_model: &str, task: ModelTask) -> Result<String> {
        // Parse model source to get the canonical name (what ModelCache uses for
        // registration).
        let canonical_name = ModelSource::parse(base_model).to_string();
        if self.catalog().get_model(&canonical_name).await?.is_none() {
            self.catalog()
                .register_shared_model(jammi_db::catalog::model_repo::RegisterModelParams {
                    model_id: &canonical_name,
                    version: 1,
                    model_type: "embedding",
                    backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
                    task,
                    base_model_id: None,
                    external_location: None,
                    config_json: None,
                })
                .await?;
        }
        Ok(self
            .catalog()
            .get_model(&canonical_name)
            .await?
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Base model '{canonical_name}' not registered in catalog"
                ))
            })?
            .catalog_pk)
    }

    /// Graph-supervised fine-tune: learn an embedding metric that encodes
    /// a graph's structure. Reads a node-text source and an edge source, samples
    /// the graph into `(anchor, positive, [hard_negative])` text pairs via biased
    /// random walks (node2vec), and drives the existing in-batch-negative
    /// (MNRL) or triplet objective — **no new loss**.
    ///
    /// `node_source` supplies the text the encoder embeds, keyed by `id_column`,
    /// with the text in `text_column`. `edge_source` supplies directed edges
    /// (`src_column` → `dst_column`); endpoints join to `id_column`. Every
    /// endpoint must resolve to a node (the text-bearing precondition) or this is
    /// a typed error.
    ///
    /// `provenance` declares whether the edges are external/declared structure or
    /// similarity edges — **the load-bearing distinction**: training on
    /// similarity edges largely re-learns the base metric (a degenerate feedback
    /// loop), so genuine gain comes from declared edges. Similarity-only edges
    /// are a weak bootstrap, never the sole supervision.
    pub async fn fine_tune_graph(
        &self,
        sources: &crate::fine_tune::graph_sampler::GraphFineTuneSources,
        base_model: &str,
        sample_config: crate::fine_tune::graph_sampler::GraphSampleConfig,
        config: Option<FineTuneConfig>,
    ) -> Result<TrainingJob> {
        let config = config.unwrap_or_default();
        config.validate()?;
        sample_config.validate_for_training()?;

        // The job record's `source` field records the node source — the model is
        // fine-tuned on that source's text, the edges only supervise the pairing.
        // The graph is read and re-sampled by the worker from the persisted
        // sources + seeded sample_config (deterministic), never from in-memory
        // batches carried across the submit boundary.
        let spec = TrainingSpec::GraphFineTune {
            sources: sources.clone(),
            sample_config,
            common: TrainingCommon {
                base_model: base_model.to_string(),
                config: config.clone(),
                world_size: crate::fine_tune::spec::DEFAULT_WORLD_SIZE,
                // This loose-argument entry point has no cache parameter to
                // carry a caller's choice; a caller that wants the model-level
                // reuse probe builds the spec directly and submits it through
                // `Self::run_training_spec`.
                cache: jammi_db::store::CachePolicy::Bypass,
            },
        };
        self.submit_fine_tune_spec(spec).await
    }

    /// Run a decoded [`TrainingSpec`] on this session, dispatching each variant
    /// to its training entry point.
    ///
    /// The single spec→session seam: the gRPC `StartTraining` handler and the
    /// embedded binding both decode a `StartTrainingRequest` into a
    /// [`TrainingSpec`] and submit it here, so an identical decode yields an
    /// identical job on either transport. The dispatch lives once, beside the
    /// entry points it calls, rather than being re-written per transport.
    pub async fn run_training_spec(self: &Arc<Self>, spec: TrainingSpec) -> Result<TrainingJob> {
        self.run_training_spec_deduped(spec, None).await
    }

    /// [`Self::run_training_spec`], additionally deduped by an optional
    /// per-tenant `idempotency_key` (migration 030) — the seam
    /// `JobService::submit_job`'s gRPC handler drives (the only caller that
    /// has a wire-supplied key; [`Self::run_training_spec`] and every other
    /// caller — the Python binding included — routes through here with
    /// `None`, byte-identical to before this method existed).
    ///
    /// Dispatches directly through the deduped catalog seam — `spec` is
    /// already fully formed here (decoded off the wire or handed in by a
    /// caller), so there is no need to re-destructure it through the loose-
    /// argument entry points' own constructors only to rebuild the
    /// identical spec. Neither dispatch arm below skips admission: each is
    /// itself one of the three durable submit edges for a `TrainingSpec`
    /// and applies `admit_training_spec` before writing a row.
    pub async fn run_training_spec_deduped(
        self: &Arc<Self>,
        spec: TrainingSpec,
        idempotency_key: Option<&str>,
    ) -> Result<TrainingJob> {
        match spec {
            TrainingSpec::ContextPredictor {
                source,
                predictor_spec,
            } => {
                self.train_context_predictor_deduped(&source, &predictor_spec, idempotency_key)
                    .await
            }
            other => {
                self.submit_fine_tune_spec_deduped(other, idempotency_key)
                    .await
            }
        }
    }

    // =====================================================================
    // Evaluation
    // =====================================================================

    /// Evaluate embedding quality against golden relevance judgments.
    ///
    /// `cohorts` maps a golden-set `query_id` to an opaque `{key: value}`
    /// segment map persisted alongside that query's per-query metrics
    /// (`_jammi_eval_per_query`, spec J9). Pass an empty map for no tags.
    pub async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
        cohorts: &std::collections::HashMap<String, std::collections::BTreeMap<String, String>>,
    ) -> Result<crate::eval::EmbeddingEvalReport> {
        EvalRunner { session: self }
            .eval_embeddings(source_id, embedding_table, golden_source, k, cohorts)
            .await
    }

    /// Read back the persisted per-query eval records for a run, scoped to the
    /// session tenant (spec J9). Returns Recall@{1,3,5,10}, MRR, nDCG,
    /// distance, and any cohort tags stored at eval time.
    pub async fn eval_per_query(
        &self,
        eval_run_id: &str,
    ) -> Result<Vec<jammi_db::catalog::eval_repo::PerQueryEvalRecord>> {
        self.catalog().get_eval_per_query(eval_run_id).await
    }

    /// Evaluate inference quality against golden labels.
    pub async fn eval_inference(
        &self,
        model_id: &str,
        source_id: &str,
        columns: &[String],
        task: crate::eval::EvalTask,
        golden_source: &str,
        label_column: &str,
    ) -> Result<crate::eval::InferenceEvalReport> {
        EvalRunner { session: self }
            .eval_inference(
                model_id,
                source_id,
                columns,
                task,
                golden_source,
                label_column,
            )
            .await
    }

    /// Compare multiple embedding tables side-by-side.
    pub async fn eval_compare(
        &self,
        embedding_tables: &[String],
        source_id: &str,
        golden_source: &str,
        k: usize,
    ) -> Result<crate::eval::CompareEvalReport> {
        EvalRunner { session: self }
            .eval_compare(embedding_tables, source_id, golden_source, k)
            .await
    }

    /// Evaluate whether a predictor's uncertainty is honest.
    ///
    /// `golden_source` is a held-out set pairing a predictive distribution with
    /// its realised `outcome`; `shape` selects the predictor's output family
    /// (parametric Gaussian or ensemble) and the columns read. `cohorts` maps a
    /// `record_id` to an opaque `{key: value}` segment map persisted alongside
    /// that record's per-record scores, the same way `eval_embeddings` cohorts
    /// work. Returns a report headlining a strictly proper score (CRPS) with the
    /// PIT-calibration diagnostic, sharpness, coverage, and per-cohort slices.
    pub async fn eval_calibration(
        &self,
        source_id: &str,
        golden_source: &str,
        shape: crate::eval::EvalCalibrationShape,
        cohorts: &std::collections::HashMap<String, std::collections::BTreeMap<String, String>>,
    ) -> Result<crate::eval::CalibrationEvalReport> {
        EvalRunner { session: self }
            .eval_calibration(source_id, golden_source, shape, cohorts)
            .await
    }
}

/// The ONE query string both `infer` arms (the fresh-compute finish and the
/// exact-cache-hit short-circuit) issue to read an inference result table
/// back — a shared function so the two arms cannot drift into two
/// independently-typed strings that happen to agree today. Orders by
/// `(_row_id, _ordinal)`: `_row_id` alone is not enough (a source can key
/// multiple rows under one id), so `_ordinal` — the stream-scoped monotonic
/// counter [`jammi_datafusion::inference::schema::common_prefix_fields`] documents —
/// breaks every tie in the order the model actually emitted the rows,
/// regardless of how the underlying Parquet scan or model batches arrive on
/// a later read.
fn infer_ordered_read_back_sql(table: &str) -> String {
    format!(
        "SELECT * FROM {} ORDER BY _row_id, _ordinal",
        jammi_db::store::result_table_relation(table)
    )
}

/// Normalize every `Utf8View`/`BinaryView` column of `batches` back to the
/// plain `Utf8`/`Binary` arrow-rs types they were built from —
/// the registered result-table scan
/// ([`jammi_db::store::ResultStore::register_table`]'s doc: "the resolved
/// Arrow schema (Utf8View under the Arrow parquet-reader default) matches")
/// widens string columns to the View encoding on every read-back, which
/// [`InferenceSession::infer`]'s ordered read-back takes on BOTH arms — so
/// without this normalization a caller downcasting `_status`/`_error`/a
/// string task column to `StringArray` would see a different shape purely
/// as a side effect of routing through SQL rather than returning the
/// in-memory compute batch. Every OTHER column (including `_ordinal`
/// itself) passes through unchanged.
fn normalize_view_columns(batch: &RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();
    let mut changed = false;
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns: Vec<arrow::array::ArrayRef> = Vec::with_capacity(schema.fields().len());
    for (field, col) in schema.fields().iter().zip(batch.columns()) {
        let target = match field.data_type() {
            arrow::datatypes::DataType::Utf8View => Some(arrow::datatypes::DataType::Utf8),
            arrow::datatypes::DataType::BinaryView => Some(arrow::datatypes::DataType::Binary),
            _ => None,
        };
        match target {
            Some(target_ty) => {
                changed = true;
                let cast = arrow::compute::cast(col, &target_ty).map_err(|e| {
                    JammiError::Inference(format!(
                        "infer read-back: normalizing column '{}' from {:?} to {target_ty:?}: {e}",
                        field.name(),
                        field.data_type()
                    ))
                })?;
                fields.push(std::sync::Arc::new(arrow::datatypes::Field::new(
                    field.name(),
                    target_ty,
                    field.is_nullable(),
                )));
                columns.push(cast);
            }
            None => {
                fields.push(std::sync::Arc::clone(field));
                columns.push(std::sync::Arc::clone(col));
            }
        }
    }
    if !changed {
        return Ok(batch.clone());
    }
    let new_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(fields));
    RecordBatch::try_new(new_schema, columns)
        .map_err(|e| JammiError::Inference(format!("infer read-back: rebuild batch: {e}")))
}

/// [`normalize_view_columns`] applied over a whole read-back result set.
fn normalize_view_batches(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    batches.iter().map(normalize_view_columns).collect()
}

/// The validation core of [`InferenceSession::served_regression_col_for_test`]:
/// pulls distribution column `col_idx` out of a regression `BackendOutput`,
/// refusing (rather than indexing blindly) whenever the output's shape,
/// float head, or row-count bookkeeping is inconsistent. Extracted to a free
/// function of `&BackendOutput` (a `BackendOutput` literal, not a live model
/// forward pass) so each fail-closed arm has a direct, deterministic oracle.
fn extract_test_column(output: &BackendOutput, col_idx: usize) -> Result<Vec<f32>> {
    let (num_rows, head_width) = *output.shapes.first().ok_or_else(|| {
        JammiError::Inference(
            "served_regression_col_for_test: backend emitted no output-head shape".into(),
        )
    })?;
    if col_idx >= head_width {
        return Err(JammiError::Inference(format!(
            "served_regression_col_for_test: col_idx {col_idx} out of range for head_width \
             {head_width}"
        )));
    }
    let flat = output.float_outputs.first().ok_or_else(|| {
        JammiError::Inference(
            "served_regression_col_for_test: backend emitted no float head".into(),
        )
    })?;
    // Checked multiply (mirrors `BackendOutput::checked_rows`'s
    // `rows.checked_mul(dim)`): a raw `num_rows * head_width` could silently
    // overflow on an adversarial shape.
    let expected = num_rows.checked_mul(head_width).ok_or_else(|| {
        JammiError::Inference(format!(
            "served_regression_col_for_test: num_rows*head_width overflows (num_rows={num_rows}, \
             head_width={head_width})"
        ))
    })?;
    if flat.len() != expected || output.row_status.len() != num_rows {
        return Err(JammiError::Inference(format!(
            "served_regression_col_for_test: backend output is inconsistent (float_outputs[0] \
             has {} value(s), row_status has {} entries, expected rows({num_rows}) * \
             head_width({head_width}) and one row_status entry per row)",
            flat.len(),
            output.row_status.len()
        )));
    }
    let mut col = Vec::with_capacity(num_rows);
    for row in 0..num_rows {
        if output.row_status[row] {
            col.push(flat[row * head_width + col_idx]);
        }
    }
    Ok(col)
}

/// Construct the session's [`ResultStore`], honouring `config.storage`.
///
/// When `storage.result_root` is set, result tables (Parquet + USearch
/// sidecars) are rooted there via [`ResultStore::with_root`], sharing the
/// session's [`jammi_db::storage::StorageRegistry`] so the deploy-wide
/// `[storage.cloud]` credentials resolve the root's driver. When it is unset,
/// result tables live on local disk under `{artifact_dir}/jammi_db/` —
/// today's behaviour. The catalog backend is independent of this choice.
fn build_result_store(
    inner: &JammiSession,
    catalog: Arc<jammi_db::catalog::Catalog>,
    placement: Arc<dyn jammi_db::index::SegmentPlacement>,
) -> Result<ResultStore> {
    let ann = inner.config().embedding.ann;
    // The one lease timing every leased row shares: the store's building
    // tables are held under the same `[lease]` the training worker uses.
    let lease = inner.config().lease.intervals()?;
    // `resolved_result_root()` is the ONE `{artifact_dir}/jammi_db` (or
    // explicit `storage.result_root`) derivation — the SAME string a gang
    // member's `instances.result_root` row carries verbatim
    // (`InstanceRegistration::from_config`) — so this session's store is
    // rooted at EXACTLY the root the member row records (carried, not
    // consulted by the membership predicate), never a second, independently
    // re-derived path.
    let root = jammi_db::storage::StorageUrl::parse(&inner.config().resolved_result_root()?)?;
    // `local_cache_dir` is the PARENT of the two local caches
    // `ResultStore::with_root` derives (`{local_cache_dir}/index` — the ANN
    // segment cache, always local since USearch reads the local filesystem
    // even when `root` is a cloud scheme — and `{local_cache_dir}/artifact`,
    // the model-artifact fetch cache its own internal `ArtifactStore` uses).
    // Rooted under the local artifact dir's `cache/` sub-prefix: relocated
    // OUT of the `jammi_db/` result-table root so a `reconcile`/backup pass
    // over that root never walks scratch cache state.
    let local_cache_dir = inner.config().artifact_dir.join("cache");
    let store = ResultStore::with_root(
        root,
        inner.storage_registry(),
        catalog,
        ann,
        local_cache_dir,
    )?;
    Ok(store
        .with_lease_intervals(lease)
        // The placed-search seams: who owns which segment (`AllLocal` unless
        // the session was opened with a placement), and the gRPC transport a
        // remote segment is searched through — the same client whether this
        // process is a server replica or a library coordinator.
        .with_placement(placement)
        .with_peer_transport(Arc::new(jammi_wire::peer::GrpcPeerTransport::new()))
        // `[server] peer_local_load_bytes`: the marginal-load admission
        // budget of the placed-search failure ladder. Read by the store
        // (the only reader), from the same config a wire deployment and an
        // in-process one share.
        .with_peer_local_load_bytes(inner.config().server.peer_local_load_bytes))
}

#[cfg(test)]
mod extract_test_column_tests {
    use super::*;

    fn two_rows_width2(status: Vec<bool>) -> BackendOutput {
        BackendOutput {
            float_outputs: vec![vec![1.0, 2.0, 3.0, 4.0]],
            string_outputs: vec![],
            row_status: status,
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 2)],
        }
    }

    #[test]
    fn extracts_the_requested_column_skipping_failed_rows() {
        let out = two_rows_width2(vec![true, false]);
        let col = extract_test_column(&out, 0).unwrap();
        assert_eq!(col, vec![1.0]);
    }

    /// A `BackendOutput` with no shape entry at all must be a named refusal,
    /// not an index into an absent element 0 (an unchecked
    /// `output.shapes[0]` would panic instead of returning the `Err` asserted
    /// below).
    #[test]
    fn refuses_when_no_output_head_shape_is_present() {
        let out = BackendOutput {
            float_outputs: vec![vec![1.0, 2.0]],
            string_outputs: vec![],
            row_status: vec![true],
            row_errors: vec![String::new()],
            shapes: vec![],
        };
        let err = extract_test_column(&out, 0).unwrap_err();
        assert!(err.to_string().contains("output-head shape"), "{err}");
    }

    /// A `BackendOutput` with no float head at all must be a named refusal,
    /// not an index into an absent element 0 (an unchecked
    /// `output.float_outputs[0]` would panic instead of returning the `Err`
    /// asserted below).
    #[test]
    fn refuses_when_no_float_head_is_present() {
        let out = BackendOutput {
            float_outputs: vec![],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 2)],
        };
        let err = extract_test_column(&out, 0).unwrap_err();
        assert!(err.to_string().contains("no float head"), "{err}");
    }

    /// A flat buffer / `row_status` that disagree with `num_rows * head_width`
    /// must be a named refusal, not a read past (or short of) the intended
    /// rows (without the consistency check, an out-of-bounds index panics
    /// instead of returning the `Err` asserted below).
    #[test]
    fn refuses_when_flat_and_row_status_disagree_with_shape() {
        let out = BackendOutput {
            // shapes say 2 rows of width 2 (4 values); the flat buffer is one
            // short, and row_status is short too.
            float_outputs: vec![vec![1.0, 2.0, 3.0]],
            string_outputs: vec![],
            row_status: vec![true],
            row_errors: vec![String::new()],
            shapes: vec![(2, 2)],
        };
        let err = extract_test_column(&out, 0).unwrap_err();
        assert!(err.to_string().contains("inconsistent"), "{err}");
    }

    /// `num_rows * head_width` computed with a raw multiply could silently
    /// overflow on an adversarial shape; the checked multiply must refuse by
    /// name instead (a raw multiply panics in debug, or wraps to a small
    /// `expected` in release, rather than returning the `Err` asserted
    /// below).
    #[test]
    fn refuses_an_overflowing_num_rows_times_head_width_without_panicking() {
        let out = BackendOutput {
            float_outputs: vec![vec![]],
            string_outputs: vec![],
            row_status: vec![],
            row_errors: vec![],
            shapes: vec![(usize::MAX, 2)],
        };
        let err = extract_test_column(&out, 0).unwrap_err();
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    #[test]
    fn refuses_a_col_idx_out_of_range_for_head_width() {
        let out = two_rows_width2(vec![true, true]);
        let err = extract_test_column(&out, 2).unwrap_err();
        assert!(err.to_string().contains("col_idx"), "{err}");
    }
}
