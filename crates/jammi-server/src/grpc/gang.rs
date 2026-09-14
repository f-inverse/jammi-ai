//! `GangService` — the coordinator-to-member admission wire for a multi-host
//! gang run (see `docs/rigor/contracts/feat_500-C-U5a-1.md` for the
//! committed mechanism contract this module implements).
//!
//! This module freezes the wire (`jammi.v1.gang`, see `gang.proto`), the
//! wire-level K2 edges (`world == 0`, `rank >= world`) decided before any row
//! is ever read, and this handler's own full I-GANG decision for the
//! `world_size == 1` lattice this unit ships: `get_job_for_rank` for the row
//! predicate and `fresh_instance` for the coordinator's own liveness. Every
//! determinant collapses to the SAME `FailedPrecondition` status with a
//! FIXED message (non-disclosure) — the listener discloses neither a job's
//! existence, claimant, nor attempt. Having decided every determinant and
//! found no reason to refuse, this handler still has no `HostAdmission`
//! session to hand the call to, so it ends `Unimplemented`: a call
//! satisfying EVERY I-GANG determinant still reaches the handler ... and,
//! having no `HostAdmission` session to hand the call to, returns
//! `Unimplemented`.
//! `HostAdmission` itself (the admit-and-hold session, drain,
//! re-verification) is built on top of this handler once it exists
//! (docs/plans/67-distributed-training/UNITS.md § U5a-2).
//!
//! **This unit ships the `world_size == 1` lattice only.** `row.world_size`
//! (`jammi_db::catalog::jobs_repo::WorldSizeFact`) is a ROW FACT, decoded
//! from the SAME `spec` JSON the claiming worker reconstructs its run from —
//! never the caller's own `Assign.world`. A `spec` column that does not
//! decode a `world_size` at all is likewise a row fact, never a fault of the
//! read that found it (see [`GangRefusalReason::SpecUndecodable`]).
//! `assign.world != row.world_size` is itself an I-GANG refusal (the SAME
//! fixed message as every other one — see [`RankAdmissionRow::world_size`]'s
//! own doc for why a caller-keyed gate is unsound); separately, a row whose
//! own `world_size` disagrees with `1` refuses the SAME way EVEN WHEN the
//! caller's `Assign.world` agrees with it (see
//! [`GangRefusalReason::MultiHostUnsupported`]) — the training-set pair
//! conjunct and its sidecar verify that would admit a genuine multi-host row
//! are `HostAdmission`'s to build (docs/plans/67-distributed-training/UNITS.md
//! § U5a-2); this handler never attempts them.
//!
//! **A catalog fault during admission is `Unavailable`, never
//! `FailedPrecondition`**: see `admission_catalog_fault`.
//!
//! **I-GANG derives tenant from the row, never ambient admin scope**
//! (§I1(a)) — this handler refuses outright, before any row is even read,
//! whenever [`TenantBinding::is_admin_scope`] is ambient: see
//! [`GangRefusalReason::AdminScope`].
//!
//! **`test-hooks` non-disclosure introspection**: behind
//! `#[cfg(feature = "test-hooks")]`, `GangServer::last_refusal_reason`
//! exposes which [`GangRefusalReason`] variant the most recent call refused
//! for — same-process, test-only, never reaching the wire — so the
//! `test-hooks` lane can assert every I-GANG determinant was actually
//! DECIDED (not merely "some earlier check happened to also refuse")
//! without leaking that distinction into the response the plain lane's own
//! non-disclosure oracle asserts is uniform.
//!
//! Served only on the internal `[server] peer_bind` listener, mounted beside
//! `PeerService` (`OssServer::bind`) — never on the public listener, never
//! wrapped by the tenant-binding layer. Tenant is derived from the verified
//! `jobs` row, never the caller (I-GANG) — this handler never reads a
//! `SessionTenant` extension the way the tenant-wrapped services do.

use std::pin::Pin;
use std::sync::Arc;
#[cfg(feature = "test-hooks")]
use std::sync::Mutex;
use std::time::Duration;

use futures::Stream;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::{RankAdmissionRow, WorldSizeFact};
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::status::{JobStatus, ResultTableStatus};
use jammi_db::error::JammiError;
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use tonic::{Request, Response, Status};

use tokio_stream::StreamExt;

use crate::grpc::proto::gang::gang_service_server::GangService;
use crate::grpc::proto::gang::{rank_control, RankControl, RankEvent};

/// §I1 Non-disclosure: every I-GANG refusal — ambient admin scope, row
/// absent, wrong status, wrong claimant, wrong attempt, lease not live, an
/// undecodable `world_size`, the caller's `Assign.world` not matching the
/// row's own `world_size`, a row whose own `world_size` names more than one
/// rank, or the coordinator's own `instances` row not fresh — collapses to
/// this ONE status with this ONE fixed message. Never interpolate a job id,
/// a claimant, or a reason into it: that is exactly the disclosure this
/// property forbids.
const I_GANG_REFUSAL_MESSAGE: &str = "gang admission refused";

fn i_gang_refused() -> Status {
    Status::failed_precondition(I_GANG_REFUSAL_MESSAGE)
}

/// Every I-GANG determinant a `RunRank` call can refuse for, kept
/// distinguishable for the non-disclosure oracle's `test-hooks` seam. Every
/// variant here refuses the wire with the exact SAME
/// `I_GANG_REFUSAL_MESSAGE` — this enum exists so the `test-hooks` lane's
/// `GangServer::last_refusal_reason` can distinguish them same-process,
/// never so the wire can. Adding a determinant this handler decides without
/// a matching variant here reopens exactly the coverage gap this enum
/// closes: the pairwise non-disclosure oracle only proves what it can
/// distinguish. Defined unconditionally (a plain
/// enum costs nothing) so `GangServer::record_refusal`'s call sites never
/// need their own `#[cfg]`; only the STORAGE ([`GangServer`]'s field) and the
/// GETTER (`GangServer::last_refusal_reason`) are `test-hooks`-gated, so a
/// plain build carries no additional state and this seam is provably inert
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GangRefusalReason {
    /// [`TenantBinding::is_admin_scope`] was ambient when this call reached
    /// the handler — I-GANG derives tenant from the row, never ambient admin
    /// scope, so this is decided before any row is even read.
    AdminScope,
    /// No row exists for `assign.job_id`.
    NotFound,
    /// The row was not `status = 'running'`.
    NotRunning,
    /// `claimed_by` did not match `assign.coordinator_instance_id`.
    WrongClaimant,
    /// `attempts` did not match `assign.attempt`.
    WrongAttempt,
    /// The lease was not live (`NOT(lease_expired_clause)` was false).
    LeaseDead,
    /// The row's `spec` column did not decode a `world_size`
    /// (`WorldSizeFact::Undecodable`) — a row fact about the content this
    /// claimant wrote, never a fault of the read that found it.
    SpecUndecodable,
    /// `assign.world != row.world_size` (the row's own `world_size`
    /// decoded successfully).
    WorldMismatch,
    /// `assign.world == row.world_size`, but that shared value is not `1`
    /// — this unit ships the `world_size == 1` lattice only; the
    /// training-set pair conjunct and its sidecar verify that would admit a
    /// genuine multi-host row are `HostAdmission`'s to build
    /// (docs/plans/67-distributed-training/UNITS.md § U5a-2).
    MultiHostUnsupported,
    /// The coordinator's own `instances` row was absent or stale
    /// (`Catalog::fresh_instance` returned `false`).
    CoordinatorNotFresh,
}

/// The fixed bound for the first inbound `Assign` frame: a silent client is
/// dropped at that bound. A handler-local constant, not a
/// config knob, sized to one round trip — not `[lease] heartbeat_secs` or
/// any deployment-tunable value.
const FIRST_ASSIGN_BOUND: Duration = Duration::from_secs(10);

/// Server-side handler for the gang admission surface. Holds the shared
/// engine session for its catalog + result store — the same handle
/// [`crate::grpc::peer::PeerServer`] holds for the owner side of the
/// distributed data plane — and the `[lease]` window this deployment runs
/// with, needed for [`jammi_db::catalog::Catalog::fresh_instance`]'s own
/// `instance_liveness_margin` computation (§I1) without reaching back into
/// `InferenceSession` for a config accessor this crate does not own.
pub struct GangServer {
    session: Arc<InferenceSession>,
    lease: Duration,
    /// The test-only introspection state the non-disclosure oracle's
    /// `test-hooks` seam reads
    /// ([`Self::last_refusal_reason`], [`Self::refusal_reason_handle`]).
    /// Absent entirely from a plain build — this field, and every write to
    /// it, compiles away, so a published `jammi-server` binary carries no
    /// additional state for this. `Arc<Mutex<..>>`, not a bare `Mutex`, so
    /// [`Self::refusal_reason_handle`] can clone out a handle onto this SAME
    /// state before `self` is moved into tonic's generated service wrapper
    /// (`GangServiceServer::new` takes it by value) — the mounted, actually
    /// serving instance and the handle a test harness holds observe the
    /// identical state.
    #[cfg(feature = "test-hooks")]
    last_refusal: Arc<Mutex<Option<GangRefusalReason>>>,
}

impl GangServer {
    pub fn new(session: Arc<InferenceSession>, lease: Duration) -> Self {
        Self {
            session,
            lease,
            #[cfg(feature = "test-hooks")]
            last_refusal: Arc::new(Mutex::new(None)),
        }
    }

    /// Records which [`GangRefusalReason`] the call in progress refused for.
    /// A no-op under a plain build (no `last_refusal` field exists to write)
    /// — called unconditionally at every refusal site regardless of feature,
    /// so no call site needs its own `#[cfg]`. Last-write-wins: this
    /// `GangServer` serves one call at a time in every test that reads it
    /// back (a single in-process client, sequential `run_rank` calls); a
    /// concurrent caller reading `last_refusal_reason()` mid-call would see
    /// an interleaved answer, which is exactly why this seam is `test-hooks`
    /// only, never a production observability surface.
    #[cfg_attr(not(feature = "test-hooks"), allow(unused_variables))]
    fn record_refusal(&self, reason: GangRefusalReason) {
        #[cfg(feature = "test-hooks")]
        {
            *self.last_refusal.lock().unwrap() = Some(reason);
        }
    }

    /// `test-hooks` only: which [`GangRefusalReason`] the most recent
    /// `run_rank` call on this `GangServer` refused for, if any. `None`
    /// until the first refusal, or after a call that reached
    /// `Unimplemented` (reaching the not-yet-implemented terminal state is
    /// not itself a refusal — no reason is recorded
    /// for it, so a prior refusal's reason survives an admitting call,
    /// intentionally: this seam names the last determinant that actually
    /// refused, not "whether the most recent call was refused").
    #[cfg(feature = "test-hooks")]
    pub fn last_refusal_reason(&self) -> Option<GangRefusalReason> {
        *self.last_refusal.lock().unwrap()
    }

    /// `test-hooks` only: a cheaply cloneable [`GangRefusalHandle`] onto this
    /// `GangServer`'s own refusal-reason state. `GangServiceServer::new`
    /// (tonic's generated wrapper, `runtime.rs`'s `OssServer::bind`) takes
    /// `GangServer` BY VALUE, so nothing outside this module ever keeps a
    /// reference to the mounted instance itself — a test harness driving
    /// `RunRank` over the real `peer_bind` listener calls this BEFORE
    /// mounting to keep observing the SAME state the served instance
    /// mutates.
    #[cfg(feature = "test-hooks")]
    pub fn refusal_reason_handle(&self) -> GangRefusalHandle {
        GangRefusalHandle(Arc::clone(&self.last_refusal))
    }
}

/// `test-hooks` only: see [`GangServer::refusal_reason_handle`].
#[cfg(feature = "test-hooks")]
#[derive(Clone)]
pub struct GangRefusalHandle(Arc<Mutex<Option<GangRefusalReason>>>);

#[cfg(feature = "test-hooks")]
impl GangRefusalHandle {
    /// Which [`GangRefusalReason`] the most recent `run_rank` call on the
    /// `GangServer` this handle was cloned from refused for.
    pub fn get(&self) -> Option<GangRefusalReason> {
        *self.0.lock().unwrap()
    }
}

/// §I3/§I4's transient class: a genuine catalog fault reached DURING
/// admission — `Catalog::get_job_for_rank`'s own read or
/// `Catalog::fresh_instance`'s own read ERRORING rather than simply finding
/// no row / no fresh instance — is `Unavailable`, never `map_engine_error`'s
/// generic mapping. A raw catalog-backend fault surfaces as
/// `JammiError::BackendDriver`, an arm `map_engine_error` has no case for —
/// it would fall through to that function's own `other => Internal`
/// catch-all and never tell a retrying caller this was transient. Every
/// admission-time catalog read on the `RunRank` path uses this SAME
/// classification; `map_engine_error` is never called on this path. This is
/// the ADMISSION-time classification; the mid-stream three-way split between
/// `Refuted` / `Unavailable` / `StoreUnavailable` at RE-VERIFICATION is built
/// with `HostAdmission` (docs/plans/67-distributed-training/UNITS.md §
/// U5a-2) once an admitted session exists to re-verify inside. This unit's
/// handler never reaches `Catalog::get_result_table_for_tenant` or
/// `ResultStore::read_materialization_manifest` (the training-set sidecar
/// lookup `resolve_training_set_identity_classified` still uses this same
/// classification for, as a standalone, directly-tested primitive —
/// `HostAdmission`'s to call once it exists, docs/plans/67-distributed-training/UNITS.md
/// § U5a-2).
fn admission_catalog_fault(err: JammiError) -> Status {
    tracing::warn!(
        error = %err,
        "gang admission: a catalog read faulted rather than returning no row"
    );
    Status::unavailable("gang admission: catalog temporarily unavailable")
}

#[tonic::async_trait]
impl GangService for GangServer {
    type RunRankStream = Pin<Box<dyn Stream<Item = Result<RankEvent, Status>> + Send + 'static>>;

    async fn run_rank(
        &self,
        request: Request<tonic::Streaming<RankControl>>,
    ) -> Result<Response<Self::RunRankStream>, Status> {
        let mut inbound = request.into_inner();

        // Await the first inbound frame inline, bounded. A
        // silent client (nothing within the bound) is dropped without a
        // status frame — there is no admitted session yet for one to belong
        // to.
        let first = match tokio::time::timeout(FIRST_ASSIGN_BOUND, inbound.next()).await {
            Ok(Some(Ok(frame))) => frame,
            Ok(Some(Err(status))) => return Err(status),
            Ok(None) => {
                return Err(Status::invalid_argument(
                    "RunRank stream closed before Assign",
                ))
            }
            Err(_) => {
                return Err(Status::invalid_argument(
                    "no Assign received within the admission bound",
                ))
            }
        };
        let assign = match first.control {
            Some(rank_control::Control::Assign(assign)) => assign,
            _ => {
                return Err(Status::invalid_argument(
                    "RunRank must open with an Assign frame",
                ))
            }
        };

        // §W1 K2, decided before I-GANG runs: `world == 0` and `rank >=
        // world` are refused `InvalidArgument`, one case each.
        if assign.world == 0 {
            return Err(Status::invalid_argument("world must be greater than zero"));
        }
        if assign.rank >= assign.world {
            return Err(Status::invalid_argument("rank must be less than world"));
        }

        // I-GANG derives tenant from the row, never ambient admin scope
        // (§I1(a)) — this holds for the WHOLE handler, not merely a
        // tenant-scoped catalog read it might one day perform: a call
        // reaching this handler while wrapped in admin scope is refused
        // outright, before any row is even read, the same fixed way every
        // other determinant refuses.
        if TenantBinding::is_admin_scope() {
            self.record_refusal(GangRefusalReason::AdminScope);
            return Err(i_gang_refused());
        }

        // The row predicate: primary-key-only, no tenant predicate —
        // `Catalog::get_job_for_rank` never reads `assign.job_id`'s tenant
        // from the caller (I-GANG: tenant is derived from the row itself,
        // below).
        let catalog = self.session.catalog();
        let row: RankAdmissionRow = match catalog.get_job_for_rank(&assign.job_id).await {
            Ok(Some(row)) => row,
            Ok(None) => {
                self.record_refusal(GangRefusalReason::NotFound);
                return Err(i_gang_refused());
            }
            // The row genuinely erroring (a catalog fault), never conflated
            // with `Ok(None)` (no such row) — `Unavailable`, not the
            // generic `map_engine_error` mapping.
            Err(e) => return Err(admission_catalog_fault(e)),
        };

        if row.status != JobStatus::Running.to_string() {
            self.record_refusal(GangRefusalReason::NotRunning);
            return Err(i_gang_refused());
        }
        if row.claimed_by.as_deref() != Some(assign.coordinator_instance_id.as_str()) {
            self.record_refusal(GangRefusalReason::WrongClaimant);
            return Err(i_gang_refused());
        }
        if i64::from(row.attempts) != assign.attempt {
            self.record_refusal(GangRefusalReason::WrongAttempt);
            return Err(i_gang_refused());
        }
        if !row.lease_live {
            self.record_refusal(GangRefusalReason::LeaseDead);
            return Err(i_gang_refused());
        }

        // The row's own `world_size` is a ROW FACT
        // (`jammi_db::catalog::jobs_repo::WorldSizeFact`), never a fault of
        // the read that found it: a `spec` column that does not decode a
        // `world_size` at all refuses the SAME fixed way every other
        // determinant does, counted against the attempt budget like every
        // refusal — never `admission_catalog_fault`, which is reserved for
        // the read ITSELF faulting (`get_job_for_rank` returning `Err`).
        let world_size = match row.world_size {
            WorldSizeFact::Undecodable => {
                self.record_refusal(GangRefusalReason::SpecUndecodable);
                return Err(i_gang_refused());
            }
            WorldSizeFact::Decoded(n) => n,
        };

        // The lattice is keyed on the ROW's own `world_size`, never the
        // caller's `assign.world` — a caller naming a `world` the row does
        // not agree with is itself a refusal, with the SAME fixed message
        // every other I-GANG determinant refuses with. This runs BEFORE the
        // `world_size != 1` gate below so a multi-host job assigned at a
        // mismatched `world` (in either direction) is distinguished from one
        // whose caller-supplied `world` genuinely agrees with the row.
        if assign.world != world_size {
            self.record_refusal(GangRefusalReason::WorldMismatch);
            return Err(i_gang_refused());
        }

        // This unit ships the `world_size == 1` lattice only: the
        // training-set pair conjunct and its sidecar verify (§I1(b)) that
        // would admit a genuine multi-host row are `HostAdmission`'s to
        // build (docs/plans/67-distributed-training/UNITS.md § U5a-2). A row
        // whose OWN `world_size` (now known to equal `assign.world`, the
        // conjunct above) names more than one rank refuses the SAME fixed
        // way — never admitted, never silently treated as `world_size == 1`.
        if world_size != 1 {
            self.record_refusal(GangRefusalReason::MultiHostUnsupported);
            return Err(i_gang_refused());
        }

        // §I1: the coordinator's own `instances` row must be fresh. A
        // genuine catalog fault reading this row is `Unavailable`
        // (`admission_catalog_fault`), the SAME admission-time
        // classification every other catalog read on this path uses — never
        // `map_engine_error`'s generic mapping, which has no case for a raw
        // backend fault and would fall through to `Internal`.
        match catalog
            .fresh_instance(&assign.coordinator_instance_id, self.lease)
            .await
        {
            Ok(true) => {}
            Ok(false) => {
                self.record_refusal(GangRefusalReason::CoordinatorNotFresh);
                return Err(i_gang_refused());
            }
            Err(e) => return Err(admission_catalog_fault(e)),
        }

        // Every I-GANG determinant is satisfied. This handler has no
        // `HostAdmission` session to hand the call to yet
        // (docs/plans/67-distributed-training/UNITS.md § U5a-2 builds it).
        Err(Status::unimplemented(
            "gang admission is not implemented on this build",
        ))
    }
}

/// Verifies (never locates) the result table
/// `training_set_location` names, for the tenant `get_job_for_rank`
/// resolves the calling job under — the ONE tenant-pinned lookup a rank
/// performs, no listing, no candidate search.
///
/// Two properties bind this function:
///
/// - **Admin scope is guarded explicitly at this call site, not left to the
///   verb below.** [`TenantBinding::is_admin_scope`] reads ambient
///   task-local state this call site does not control by construction, so
///   this function refuses immediately, before ever calling the strict
///   resolver, whenever admin scope is active — regardless of which tenant
///   or table it was asked to resolve. This never relies on
///   `jammi_db::catalog::Catalog::get_result_table_for_tenant`'s OWN
///   admin-scope behaviour (unchanged by this function: called directly,
///   outside this guard, it still resolves any tenant's table under admin
///   scope, matching the rest of that repo's verbs).
/// - **The lookup is the strict-predicate verb, never the relaxed read.**
///   The lookup below is exactly `Catalog::get_result_table_for_tenant`,
///   never `Catalog::get_result_table` — the strict predicate `tenant_id =
///   $t OR (tenant_id IS NULL AND $t IS NULL)`, never the relaxed `OR
///   tenant_id IS NULL` a *read* resolver uses to also see a global row.
///
/// The admission-time classification collapses every unresolvable /
/// unverifiable outcome — name absent, the strict resolver returns `None`,
/// the row not `ready`, the sidecar absent (`Ok(None)`), a digest mismatch,
/// or [`ResultStore::read_materialization_manifest`] itself erroring (a
/// network/backend fault reaching this host's own store) — into the ONE
/// member-scoped `FailedPrecondition` already names for this class; the
/// three-way split that distinguishes a `StoreUnavailable` re-verification
/// end from a `Refuted` one is mid-stream, once an admitted session
/// (`HostAdmission`, docs/plans/67-distributed-training/UNITS.md § U5a-2)
/// exists to be mid-stream in.
///
/// This is the ONLY call site this program has for
/// `Catalog::get_result_table_for_tenant` outside its own crate's tests
/// (`impossibility_claims`, below) — it sits inside
/// `resolve_training_set_identity_classified`, which has NO production
/// caller in this unit: `GangServer::run_rank` ships the `world_size == 1`
/// lattice only and never reaches a tenant-scoped catalog read (see the
/// module-level doc). This function, and the classification it wraps, are
/// retained as the tested primitive the multi-host admission gate builds on
/// once `HostAdmission` exists to admit a `world_size > 1` row into
/// (docs/plans/67-distributed-training/UNITS.md § U5a-2) — its only callers
/// today are the two direct unit tests below, which exercise it in isolation
/// to demonstrate the tenant-isolation and admin-scope hazards it guards
/// against.
pub async fn resolve_training_set_identity(
    store: &ResultStore,
    tenant: Option<TenantId>,
    training_set_ref: &str,
    training_set_location: &str,
) -> Result<(), Status> {
    match resolve_training_set_identity_classified(
        store,
        tenant,
        training_set_ref,
        training_set_location,
    )
    .await?
    {
        TrainingSetOutcome::Verified => Ok(()),
        TrainingSetOutcome::AdminScopeRefused => Err(Status::failed_precondition(
            "training-set resolution is refused under admin scope",
        )),
        TrainingSetOutcome::OtherTenant
        | TrainingSetOutcome::NotReady
        | TrainingSetOutcome::DigestMismatch => {
            Err(Status::failed_precondition("training set unresolved"))
        }
    }
}

/// The classification [`resolve_training_set_identity`] collapses to
/// `FailedPrecondition` for the wire (non-disclosure; the admission-time
/// collapse) — kept distinguishable here (never on the wire) for a future
/// caller (`HostAdmission`, docs/plans/67-distributed-training/UNITS.md §
/// U5a-2) to name its own exact refusal reason from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainingSetOutcome {
    /// The row resolved for this job's own tenant, is `ready`, and its
    /// sidecar manifest verifies `training_set_ref`.
    Verified,
    /// [`TenantBinding::is_admin_scope`] was ambient; refused before the
    /// strict resolver ever ran (never an I-GANG determinant `run_rank`
    /// records a [`GangRefusalReason`] for — not reachable through the
    /// tenant-derived `run_rank` path in production, only through
    /// `resolve_training_set_identity`'s own direct callers under
    /// `with_admin_scope`).
    AdminScopeRefused,
    /// The strict tenant-pinned resolver found no `result_tables` row —
    /// whether none exists at all, or it belongs to another tenant (the
    /// strict predicate does not distinguish the two).
    OtherTenant,
    /// A row was found for this job's own tenant, but its `status` was not
    /// `ready`.
    NotReady,
    /// The row was `ready`, but its sidecar manifest did not verify
    /// `training_set_ref` — the URL failed to parse, the sidecar read
    /// erred, no sidecar existed, or the digest genuinely mismatched — this
    /// verify's own admission-time collapse: none of these are
    /// distinguished here either (only RE-VERIFICATION, §I3, needs the
    /// three-way `Refuted`/`StoreUnavailable` split, once an admitted
    /// session exists to re-verify inside).
    DigestMismatch,
}

/// Classifies §I1(b)'s sidecar verify (see [`resolve_training_set_identity`]
/// for the two properties this implements: the explicit admin-scope guard,
/// and the strict-predicate lookup). A genuine catalog fault reading
/// `Catalog::get_result_table_for_tenant` (not merely finding no row) is
/// `Err(Unavailable)` (`admission_catalog_fault`) — the ONLY
/// way this function returns `Err`; every other outcome, including every
/// unresolvable/unverifiable one `ResultStore::read_materialization_manifest`
/// itself erroring produces, is classified into a `TrainingSetOutcome`
/// variant and returned `Ok`.
pub(crate) async fn resolve_training_set_identity_classified(
    store: &ResultStore,
    tenant: Option<TenantId>,
    training_set_ref: &str,
    training_set_location: &str,
) -> Result<TrainingSetOutcome, Status> {
    if TenantBinding::is_admin_scope() {
        return Ok(TrainingSetOutcome::AdminScopeRefused);
    }

    let lookup = store
        .catalog()
        .get_result_table_for_tenant(training_set_location, tenant)
        .await
        .map_err(admission_catalog_fault)?;
    let record: ResultTableRecord = match lookup {
        None => return Ok(TrainingSetOutcome::OtherTenant),
        Some(record) if record.status != ResultTableStatus::Ready.to_string() => {
            return Ok(TrainingSetOutcome::NotReady)
        }
        Some(record) => record,
    };

    let url = match StorageUrl::parse(&record.parquet_path) {
        Ok(url) => url,
        Err(_) => return Ok(TrainingSetOutcome::DigestMismatch),
    };

    // Admission-time classification: every unresolvable /
    // unverifiable outcome — `Ok(None)` (no sidecar), a digest mismatch, or
    // the read itself erroring — collapses to the same `DigestMismatch`
    // classification (itself the same member-scoped `FailedPrecondition` on
    // the wire). The three-way split into `Refuted` / `StoreUnavailable` is
    // §I3's re-verification classification, which needs an admitted
    // (`HostAdmission`) session to re-verify inside — not this
    // admission-time call.
    match store.read_materialization_manifest(&url).await {
        Ok(Some(manifest)) if manifest.artifact.0 == training_set_ref => {
            Ok(TrainingSetOutcome::Verified)
        }
        Ok(_) | Err(_) => Ok(TrainingSetOutcome::DigestMismatch),
    }
}
