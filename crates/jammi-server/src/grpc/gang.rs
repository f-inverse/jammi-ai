//! `GangService` — the coordinator-to-member admission wire for a multi-host
//! gang run. This module is the `HostAdmission` half: admit-and-hold, the
//! holder lattice, drain, re-verification.
//!
//! `GangServer::run_rank` decides, in this order, before a single stream
//! event is emitted: the wire-level K2 edges (`world == 0`, `rank >= world`)
//! before any row is read; ambient admin scope, refused outright; the
//! I-GANG row predicate through `Catalog::get_job_for_rank` (`running`,
//! claimed by the caller's coordinator, at the caller's attempt, lease
//! live); the ROW's own `world_size` (`WorldSizeFact`, decoded from the
//! same `spec` JSON the claiming worker reconstructs its run from — never
//! the caller's `Assign.world`, which must merely agree with it); and, for
//! `row.world_size > 1` ONLY, the world>1 conjunct (#566 R2): (a) the
//! training-set identity pair is filled on the row, and (b) the row's own
//! `tenant_id` pins a strict tenant-scoped resolution of a `ready`
//! `result_tables` row named by `training_set_location`
//! ([`jammi_db::catalog::Catalog::get_result_table_for_tenant`], never the
//! relaxed read) whose sidecar manifest verifies `artifact ==
//! training_set_ref` (K4's verify-at-read instance; a sidecar written
//! before the leaf inventory reads as absent and refuses). Then the
//! coordinator's own liveness (`Catalog::fresh_instance`). Every one of
//! those determinants collapses to the SAME `FailedPrecondition` with ONE
//! fixed message (non-disclosure, #566 R3): the listener discloses neither
//! a job's existence, its claimant, its attempt, its tenant, nor another
//! tenant's table.
//!
//! Only once every determinant holds does the handler contend for this
//! host's single job slot — `HostAdmission::admit_rank`'s
//! compare-and-set on the holder cell (`Free` admits; a `ClaimProbe` is
//! waited on for at most one heartbeat; `JobRun` or another `Rank` refuse
//! `Unavailable` at once — transient, no assembly budget consumed) — and
//! only once the CAS succeeded is `Admitted` emitted. A `world_size > 1`
//! session then hands its `MemberLink` to the RANK BODY
//! (`jammi_ai::fine_tune::worker::run_member_rank`, spawned here: rank
//! `assign.rank` of the gang, trained over the session's own stream as
//! `RunnerRole::Rank`); a `world_size == 1` session has no body to run.
//! Either way the stream is HELD by a spawned loop owning the [`RankHold`]
//! guard, with exactly five arms, of which a session takes four: the
//! inbound stream (`Cancel` ends the session cooperatively; a second
//! `Assign` is a protocol violation, `InvalidArgument` — K2; a round frame
//! — `RoundInbox::is_round_frame` — is delivered to the session's round
//! inbox and the session stays held; an empty frame is a protocol
//! violation), the host's phase watch (a DRAIN or RELEASE ends every held
//! rank with `Drain`, the only host-initiated cut), the re-verification
//! tick (one per heartbeat: the row predicate, the training-set identity,
//! and the coordinator's liveness are re-decided, ending the session
//! `Refuted` / `Unavailable` / `StoreUnavailable` — see [`ReverifyEnd`] for
//! why those three are pairwise distinct in scope and in whether they
//! count), and — one or the other, never both — the rank body's end (a
//! body-bearing session ends with the body's ONE `RankEvent::Outcome`:
//! `Trained{artifact_digest}` or `Failed{reason}`; or its prologue's
//! `Aborted{reason}`) or the park bound (a body-less session ends `NoBody`
//! one lease window after admission). A session that ends for any reason
//! but its own body's end tells the body to stop (its cancel flag; the
//! round inbox severed, so the body's next collective faults) and never
//! forwards a later frame of it. The peer writes NOTHING terminal on
//! behalf of a rank: every end is a stream event, the job row untouched —
//! a source-scan oracle over this file enumerates the catalog's `jobs`
//! writers and asserts none is named here; the body itself runs as a
//! `RunnerRole::Rank`, a type that holds no `LeaseHolder` to write as.
//!
//! **A catalog fault at admission is `Unavailable`, never
//! `FailedPrecondition`**: see `admission_catalog_fault`; every
//! admission-time catalog read on this path maps through it, never
//! `map_engine_error` (a source-scan oracle pins this over `run_rank`'s own
//! body).
//!
//! **Tenant is derived, never accepted (I-GANG).** No tenant value is read
//! from the caller: `Assign` carries none, no `SessionTenant` extension is
//! read, and ambient admin scope is refused before any row is read AND
//! again at the resolution site before the strict resolver is ever called
//! (`resolve_training_set_identity`). The ONLY tenant this handler uses is
//! the `jobs` row's own `tenant_id`, read by `get_job_for_rank` as raw text
//! and parsed here (a value that does not parse is a row fact, refused the
//! same fixed way).
//!
//! **Two observables, split at admission.** BEFORE admission every
//! determinant is the call's own result: `Err(Status)` — the ONE fixed
//! `FailedPrecondition` for every I-GANG determinant (R3), `Unavailable`
//! for a catalog fault or a busy slot, `InvalidArgument` for the wire K2
//! edges — and no stream ever exists. AFTER admission the call has
//! returned `Ok(stream)`, so every later outcome is delivered IN the
//! stream: `Admitted`, then exactly one terminal event — `Aborted{reason}`,
//! or a body-bearing session's `Outcome` — or, for a protocol violation on
//! the admitted stream (a second `Assign`, K2), a status TRAILER ending the
//! stream, never a second initial result. An admitted session's ends may name their reason
//! (`Refuted`/`Unavailable`/`StoreUnavailable`/`Drain`/`Cancelled`/
//! `NoBody`): the caller already holds the job's own coordinates and was
//! admitted on them, so a reason discloses nothing a pre-admission refusal
//! withholds — pre-admission refusals never name one.
//!
//! **`test-hooks` non-disclosure introspection**: behind
//! `#[cfg(feature = "test-hooks")]`, `GangServer::last_refusal_reason`
//! exposes which [`GangRefusalReason`] variant the most recent call refused
//! for — same-process, test-only, never reaching the wire — so the
//! `test-hooks` lane can assert every determinant was actually DECIDED.
//!
//! Served only on the internal `[server] peer_bind` listener, mounted beside
//! `PeerService` (`OssServer::bind`) — never on the public listener, never
//! wrapped by the tenant-binding layer.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(feature = "test-hooks")]
use std::sync::Mutex;
use std::time::Duration;

use futures::Stream;
use jammi_ai::fine_tune::collective::MemberLink;
use jammi_ai::fine_tune::worker::{
    run_member_rank, HolderBusy, MemberAssignment, RankHold, RankOutcome, WorkerPhase,
};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::{RankAdmissionRow, WorldSizeFact};
use jammi_db::catalog::lease::LeaseFact;
use jammi_db::catalog::status::{JobStatus, ResultTableStatus};
use jammi_db::error::JammiError;
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};

use crate::grpc::gang_rounds::{member_link, RoundInbox};
use crate::grpc::proto::gang::gang_service_server::GangService;
use crate::grpc::proto::gang::{
    outcome, rank_control, rank_event, AbortReason, Aborted, Admitted, Assign, FailedOutcome,
    Outcome, RankControl, RankEvent, TrainedOutcome,
};

/// Non-disclosure: every I-GANG refusal — ambient admin scope, row absent, wrong
/// status, wrong claimant, wrong attempt, lease not live, an undecodable
/// `world_size`, the caller's `Assign.world` not matching the row's own
/// `world_size`, the world>1 conjunct's every determinant (pair missing,
/// tenant undecodable, unresolved under the job's tenant, not ready, sidecar
/// absent, digest mismatch, this host's store faulting), or the
/// coordinator's own `instances` row not fresh — collapses to this ONE
/// status with this ONE fixed message. Never interpolate a job id, a
/// claimant, a tenant, a table name or a reason into it: that is exactly
/// the disclosure this property forbids.
const I_GANG_REFUSAL_MESSAGE: &str = "gang admission refused";

fn i_gang_refused() -> Status {
    Status::failed_precondition(I_GANG_REFUSAL_MESSAGE)
}

/// The holder-contention refusal: every determinant held, but this host's
/// single job slot is busy ([`HolderBusy`]). `Unavailable` — TRANSIENT, no
/// assembly budget consumed; the coordinator retries after at least one
/// heartbeat or picks another member. One fixed message for every holder
/// kind: which job or rank this host is busy with is not the caller's to
/// learn.
const SLOT_BUSY_MESSAGE: &str = "gang admission: this host's job slot is busy";

fn slot_busy() -> Status {
    Status::unavailable(SLOT_BUSY_MESSAGE)
}

/// Every I-GANG determinant a `RunRank` call can refuse for, kept
/// distinguishable for the non-disclosure oracle's `test-hooks` seam. Every
/// variant here refuses the wire with the exact SAME
/// `I_GANG_REFUSAL_MESSAGE` — this enum exists so the `test-hooks` lane's
/// `GangServer::last_refusal_reason` can distinguish them same-process,
/// never so the wire can. Adding a determinant this handler decides without
/// a matching variant here reopens exactly the coverage gap this enum
/// closes: the pairwise non-disclosure oracle only proves what it can
/// distinguish. Defined unconditionally (a plain enum costs nothing) so
/// `GangServer::record_refusal`'s call sites never need their own `#[cfg]`;
/// only the STORAGE ([`GangServer`]'s field) and the GETTER
/// (`GangServer::last_refusal_reason`) are `test-hooks`-gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GangRefusalReason {
    /// [`TenantBinding::is_admin_scope`] was ambient — refused before any
    /// row is read, and again at the training-set resolution site before
    /// the strict resolver is ever called (`resolve_training_set_identity`).
    AdminScope,
    /// No row exists for `assign.job_id`.
    NotFound,
    /// The row was not `status = 'running'`.
    NotRunning,
    /// `claimed_by` did not match `assign.coordinator_instance_id`.
    WrongClaimant,
    /// `attempts` did not match `assign.attempt`.
    WrongAttempt,
    /// The lease was not live: [`jammi_db::catalog::lease::LeaseFact::Dead`]
    /// (a `NULL` column, or a deadline at or before now).
    LeaseDead,
    /// The row's `lease_expires_at` column held text that did not parse as a
    /// timestamp for this backend
    /// ([`jammi_db::catalog::lease::LeaseFact::Undecodable`]) — a row fact
    /// about this claimant's own row, never a fault of the read that found
    /// it (<https://github.com/f-inverse/jammi-ai/issues/574>).
    LeaseUndecodable,
    /// The row's `spec` column did not decode a `world_size`
    /// (`WorldSizeFact::Undecodable`) — a row fact about the content this
    /// claimant wrote, never a fault of the read that found it.
    SpecUndecodable,
    /// `assign.world != row.world_size` (the row's own `world_size`
    /// decoded successfully).
    WorldMismatch,
    /// `row.world_size > 1` and the training-set identity pair
    /// (`training_set_ref`, `training_set_location`) is not filled on the
    /// row — the coordinator has not materialized (or recorded) the
    /// training set this rank would read.
    TrainingSetPairMissing,
    /// `row.world_size > 1`, the pair is filled, but the row's own
    /// `tenant_id` text does not parse as a tenant — a row fact, refused
    /// before any tenant-pinned read.
    TenantUndecodable,
    /// The strict tenant-pinned resolver found no `result_tables` row named
    /// `training_set_location` under the job's OWN tenant — whether none
    /// exists at all, or one exists under another tenant or under no tenant
    /// (the strict predicate does not distinguish the three; a NULL-tenant
    /// row never resolves for a real tenant).
    TrainingSetUnresolved,
    /// A row was found for the job's own tenant, but its `status` was not
    /// `ready`.
    TrainingSetNotReady,
    /// The row was `ready`, but no sidecar manifest exists for it — or the
    /// one that exists predates the leaf inventory and reads as absent
    /// (`ResultStore::read_materialization_manifest`'s `Ok(None)`).
    TrainingSetSidecarAbsent,
    /// The sidecar exists but does not verify `training_set_ref`: its
    /// `artifact` digest differs, its body does not decode, or the row's
    /// `parquet_path` is not a storage URL at all — the artifact's own
    /// facts, not this host's.
    TrainingSetDigestMismatch,
    /// This host's own object store faulted reading the sidecar
    /// (`JammiError::Storage`/`Io` — the driver could not be built or the
    /// read itself errored): refused the same fixed way at admission; the
    /// member-scoped `StoreUnavailable` distinction is re-verification's.
    TrainingSetStoreFault,
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
/// engine session (its catalog, its result store, and its
/// [`jammi_ai::fine_tune::worker::HostAdmission`]) and the deployment's
/// `[lease]` timing: `lease` is [`jammi_db::catalog::Catalog::fresh_instance`]'s
/// margin input and the held session's park bound; `heartbeat` is the
/// re-verification cadence and the longest a `ClaimProbe` is waited on.
pub struct GangServer {
    session: Arc<InferenceSession>,
    lease: Duration,
    heartbeat: Duration,
    /// The test-only introspection state the non-disclosure oracle's
    /// `test-hooks` seam reads
    /// ([`Self::last_refusal_reason`], [`Self::refusal_reason_handle`]).
    /// Absent entirely from a plain build — this field, and every write to
    /// it, compiles away. `Arc<Mutex<..>>`, not a bare `Mutex`, so
    /// [`Self::refusal_reason_handle`] can clone out a handle onto this SAME
    /// state before `self` is moved into tonic's generated service wrapper.
    #[cfg(feature = "test-hooks")]
    last_refusal: Arc<Mutex<Option<GangRefusalReason>>>,
    /// `test-hooks` only: the taker an admitted session hands its
    /// [`MemberLink`] to instead of keeping it — see
    /// [`Self::take_member_links`]. Absent from a plain build.
    #[cfg(feature = "test-hooks")]
    member_link_tap: Arc<Mutex<Option<mpsc::UnboundedSender<MemberLink>>>>,
}

impl GangServer {
    pub fn new(session: Arc<InferenceSession>, lease: Duration, heartbeat: Duration) -> Self {
        Self {
            session,
            lease,
            heartbeat,
            #[cfg(feature = "test-hooks")]
            last_refusal: Arc::new(Mutex::new(None)),
            #[cfg(feature = "test-hooks")]
            member_link_tap: Arc::new(Mutex::new(None)),
        }
    }

    /// The owner a BODY-LESS (`world_size == 1`) session's [`MemberLink`]
    /// goes to: the session itself (`Some` — it keeps the link for its
    /// whole life, see [`HeldSession::member`]), or, under `test-hooks`
    /// with a taker registered by [`Self::take_member_links`], that taker
    /// (`None`). A taker that has gone away leaves the link with the
    /// session. A body-bearing session never offers its link: the rank
    /// body owns it.
    fn offer_member_link(&self, member: MemberLink) -> Option<MemberLink> {
        #[cfg(feature = "test-hooks")]
        {
            if let Some(tap) = self.member_link_tap.lock().unwrap().as_ref() {
                return match tap.send(member) {
                    Ok(()) => None,
                    Err(mpsc::error::SendError(member)) => Some(member),
                };
            }
        }
        Some(member)
    }

    /// `test-hooks` only: every BODY-LESS (`world_size == 1`) session this
    /// `GangServer` admits from now on hands its [`MemberLink`] to the
    /// returned receiver instead of keeping it — the seam an oracle drops
    /// a held session's link through to reach the "round frame after the
    /// link closed" trailer. A `world_size > 1` session's link is the rank
    /// body's and is never offered. Call BEFORE mounting (tonic's wrapper
    /// takes the instance by value).
    #[cfg(feature = "test-hooks")]
    pub fn take_member_links(&self) -> mpsc::UnboundedReceiver<MemberLink> {
        let (tx, rx) = mpsc::unbounded_channel();
        *self.member_link_tap.lock().unwrap() = Some(tx);
        rx
    }

    /// Records which [`GangRefusalReason`] the call in progress refused for.
    /// A no-op under a plain build (no `last_refusal` field exists to write)
    /// — called unconditionally at every refusal site regardless of feature,
    /// so no call site needs its own `#[cfg]`. Last-write-wins: this
    /// `GangServer` serves one call at a time in every test that reads it
    /// back; a concurrent caller reading `last_refusal_reason()` mid-call
    /// would see an interleaved answer, which is exactly why this seam is
    /// `test-hooks` only, never a production observability surface.
    #[cfg_attr(not(feature = "test-hooks"), allow(unused_variables))]
    fn record_refusal(&self, reason: GangRefusalReason) {
        #[cfg(feature = "test-hooks")]
        {
            *self.last_refusal.lock().unwrap() = Some(reason);
        }
    }

    /// `test-hooks` only: which [`GangRefusalReason`] the most recent
    /// `run_rank` call on this `GangServer` refused for, if any. `None`
    /// until the first refusal; an admitting call (or one refused
    /// `Unavailable` for a busy slot, which is not an I-GANG determinant)
    /// records nothing, so a prior refusal's reason survives it —
    /// intentionally: this seam names the last determinant that actually
    /// refused, not "whether the most recent call was refused".
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

/// The transient class this admission path uses: a genuine catalog fault
/// reached DURING admission — `Catalog::get_job_for_rank`'s own read,
/// `Catalog::get_result_table_for_tenant`'s own read, or
/// `Catalog::fresh_instance`'s own read ERRORING rather than simply finding
/// no row / no fresh instance — is `Unavailable`, never `map_engine_error`'s
/// generic mapping. A raw catalog-backend fault surfaces as
/// `JammiError::BackendDriver`, an arm `map_engine_error` has no case for —
/// it would fall through to that function's own `other => Internal`
/// catch-all and never tell a retrying caller this was transient. Every
/// admission-time catalog read on the `RunRank` path uses this SAME
/// classification; `map_engine_error` is never called on this path. This is
/// the ADMISSION-time classification; a held session's re-verification
/// classifies the same fault as [`ReverifyEnd::Unavailable`].
fn admission_catalog_fault(err: JammiError) -> Status {
    tracing::warn!(
        error = %err,
        "gang admission: a catalog read faulted rather than returning no row"
    );
    Status::unavailable("gang admission: catalog temporarily unavailable")
}

/// The classification of the world>1 conjunct's resolution + verify
/// ([`resolve_training_set_identity`]), collapsed by `run_rank` to the ONE
/// fixed `FailedPrecondition` for the wire (non-disclosure) and kept
/// distinguishable here so `run_rank` can name the exact
/// [`GangRefusalReason`] the `test-hooks` lane records, and so a held
/// session's re-verification can split [`Self::StoreFault`] (member-scoped)
/// from every other non-`Verified` arm (the artifact's or the row's fact).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingSetOutcome {
    /// The row resolved for this job's own tenant, is `ready`, and its
    /// sidecar manifest's `artifact` equals `training_set_ref`.
    Verified,
    /// [`TenantBinding::is_admin_scope`] was ambient at the resolution
    /// site; refused before the strict resolver ever ran.
    AdminScopeRefused,
    /// No `result_tables` row named `training_set_location` under the job's
    /// own tenant (none at all, another tenant's, or a NULL-tenant row).
    Unresolved,
    /// Found for the job's tenant, but not `ready`.
    NotReady,
    /// `ready`, but no sidecar manifest (or one predating the leaf
    /// inventory, which reads as absent).
    SidecarAbsent,
    /// The sidecar does not verify `training_set_ref` (digest differs, body
    /// does not decode, or `parquet_path` is not a storage URL).
    DigestMismatch,
    /// This host's own object store faulted reading the sidecar.
    StoreFault,
}

/// The world>1 conjunct's resolution + verify (#566 R2(b)): the ONE
/// tenant-pinned lookup a rank performs for the training set its job row
/// names — no listing, no candidate search — and the sidecar verify against
/// the row's recorded digest (K4, verify-at-read).
///
/// Two properties bind this function:
///
/// - **Admin scope is guarded explicitly at this call site, not left to the
///   verb.** [`TenantBinding::is_admin_scope`] is ambient task-local state
///   this call site does not control by construction, so this function
///   refuses immediately, before ever calling the strict resolver, whenever
///   admin scope is active — regardless of which tenant or table it was
///   asked to resolve. The strict verb itself ignores ambient scope too
///   (its own tests prove it); this guard is the resolution site's own line.
/// - **The lookup is the strict-predicate verb, never the relaxed read.**
///   Exactly [`jammi_db::catalog::Catalog::get_result_table_for_tenant`]
///   (`tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)`), never
///   `get_result_table` (whose `OR tenant_id IS NULL` would also hand a
///   real tenant every GLOBAL row of the same name).
///
/// `Err` is the ONE way this returns a fault: the catalog read itself
/// erroring (never "no row"). Every store-side outcome — the sidecar
/// absent, mismatching, undecodable, or the read erroring — is CLASSIFIED
/// into a [`TrainingSetOutcome`] and returned `Ok`, so an admission-time
/// caller collapses it to the fixed refusal and a re-verification caller
/// splits [`TrainingSetOutcome::StoreFault`] (this host's `Storage`/`Io`
/// error) from the artifact's own facts.
///
/// This is the ONLY production caller of `get_result_table_for_tenant`
/// (`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs` enumerates
/// it); its sole production caller in turn is [`GangServer::run_rank`] and
/// the hold loop's re-verification, threading the job's own `tenant_id` and
/// pair straight from the row `get_job_for_rank` resolved.
pub async fn resolve_training_set_identity(
    store: &ResultStore,
    tenant: Option<TenantId>,
    training_set_ref: &str,
    training_set_location: &str,
) -> Result<TrainingSetOutcome, JammiError> {
    if TenantBinding::is_admin_scope() {
        return Ok(TrainingSetOutcome::AdminScopeRefused);
    }

    let record = match store
        .catalog()
        .get_result_table_for_tenant(training_set_location, tenant)
        .await?
    {
        None => return Ok(TrainingSetOutcome::Unresolved),
        Some(record) if record.status != ResultTableStatus::Ready.to_string() => {
            return Ok(TrainingSetOutcome::NotReady)
        }
        Some(record) => record,
    };

    let url = match StorageUrl::parse(&record.parquet_path) {
        Ok(url) => url,
        Err(_) => return Ok(TrainingSetOutcome::DigestMismatch),
    };

    Ok(match store.read_materialization_manifest(&url).await {
        Ok(Some(manifest)) if manifest.artifact.0 == training_set_ref => {
            TrainingSetOutcome::Verified
        }
        Ok(Some(_)) => TrainingSetOutcome::DigestMismatch,
        Ok(None) => TrainingSetOutcome::SidecarAbsent,
        // This host's store: the driver could not be built for the row's
        // URL, or the object read itself errored — member-scoped.
        Err(JammiError::Storage(_)) | Err(JammiError::Io(_)) => TrainingSetOutcome::StoreFault,
        // A sidecar body that does not decode (or names a format this
        // engine does not read) is the ARTIFACT's fact: it verifies nothing.
        Err(_) => TrainingSetOutcome::DigestMismatch,
    })
}

/// The training-set identity a `world_size > 1` session was admitted
/// against, re-verified on every tick exactly as it was decided at
/// admission: the row's OWN tenant (derived, never accepted) and the pair
/// as the row carried it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TrainingSetIdentity {
    tenant: Option<TenantId>,
    training_set_ref: String,
    training_set_location: String,
}

/// How a held session ends at re-verification — three outcomes, pairwise
/// distinct in their scope and in whether they count toward the assembly's
/// attempt budget (`[worker] assembly_attempts`, the coordinator's to
/// consume — this unit ships the classification the coordinator reads off
/// the wire, never the counter):
///
/// | end | wire reason | scope | counts |
/// |---|---|---|---|
/// | `Refuted` | `REFUTED` | assembly | yes — a row fact refuted admission (status, claimant, attempt, lease, the training-set identity, or the coordinator's liveness no longer holds) |
/// | `Unavailable` | `UNAVAILABLE` | assembly | never — the catalog did not answer; transient |
/// | `StoreUnavailable` | `STORE_UNAVAILABLE` | member | never — THIS host's object store faulted; another member may verify fine |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverifyEnd {
    Refuted,
    Unavailable,
    StoreUnavailable,
}

/// Whose failure a [`ReverifyEnd`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverifyScope {
    /// The assembly's: every member sees the same fact.
    Assembly,
    /// This member's alone.
    Member,
}

impl ReverifyEnd {
    /// The `Aborted.reason` this end is emitted as.
    pub fn abort_reason(self) -> AbortReason {
        match self {
            Self::Refuted => AbortReason::Refuted,
            Self::Unavailable => AbortReason::Unavailable,
            Self::StoreUnavailable => AbortReason::StoreUnavailable,
        }
    }

    /// Whose failure this is.
    pub fn scope(self) -> ReverifyScope {
        match self {
            Self::Refuted | Self::Unavailable => ReverifyScope::Assembly,
            Self::StoreUnavailable => ReverifyScope::Member,
        }
    }

    /// Whether this end consumes one of the assembly's attempts. Only a
    /// refutation does — a fact about the run, not about the weather.
    pub fn counts_toward_assembly_attempts(self) -> bool {
        matches!(self, Self::Refuted)
    }
}

/// One re-verification tick over an admitted session: the SAME determinants
/// admission decided, re-read against the live row — the I-GANG row
/// predicate, the training-set identity (`world_size > 1` sessions only),
/// and the coordinator's liveness — classified into a [`ReverifyEnd`] when
/// any no longer holds. A row fact that no longer holds is `Refuted`; a
/// catalog read that errors is `Unavailable`; this host's own store erroring
/// on the sidecar is `StoreUnavailable`.
async fn reverify(
    session: &InferenceSession,
    assign: &Assign,
    identity: Option<&TrainingSetIdentity>,
    lease: Duration,
) -> Result<(), ReverifyEnd> {
    let catalog = session.catalog();
    let row: RankAdmissionRow = match catalog.get_job_for_rank(&assign.job_id).await {
        Ok(Some(row)) => row,
        Ok(None) => return Err(ReverifyEnd::Refuted),
        Err(_) => return Err(ReverifyEnd::Unavailable),
    };
    let row_holds = row.status == JobStatus::Running.to_string()
        && row.claimed_by.as_deref() == Some(assign.coordinator_instance_id.as_str())
        && i64::from(row.attempts) == assign.attempt
        && matches!(row.lease, LeaseFact::Live { .. });
    if !row_holds {
        return Err(ReverifyEnd::Refuted);
    }
    if let Some(identity) = identity {
        // The pair is write-once (the CAS + the schema `CHECK`), so a row
        // whose pair no longer equals what was admitted is a different
        // fact about this job than the one admitted against.
        if row.training_set_ref.as_deref() != Some(identity.training_set_ref.as_str())
            || row.training_set_location.as_deref() != Some(identity.training_set_location.as_str())
        {
            return Err(ReverifyEnd::Refuted);
        }
        match resolve_training_set_identity(
            session.result_store().as_ref(),
            identity.tenant,
            &identity.training_set_ref,
            &identity.training_set_location,
        )
        .await
        {
            Ok(TrainingSetOutcome::Verified) => {}
            Ok(TrainingSetOutcome::StoreFault) => return Err(ReverifyEnd::StoreUnavailable),
            // `AdminScopeRefused` cannot occur on a spawned hold task (no
            // task-local scope is inherited); classified as a refutation
            // if it ever did — never as an admission.
            Ok(_) => return Err(ReverifyEnd::Refuted),
            Err(_) => return Err(ReverifyEnd::Unavailable),
        }
    }
    match catalog
        .fresh_instance(&assign.coordinator_instance_id, lease)
        .await
    {
        Ok(true) => Ok(()),
        Ok(false) => Err(ReverifyEnd::Refuted),
        Err(_) => Err(ReverifyEnd::Unavailable),
    }
}

fn admitted_event() -> RankEvent {
    RankEvent {
        event: Some(rank_event::Event::Admitted(Admitted {})),
    }
}

fn aborted_event(reason: AbortReason) -> RankEvent {
    RankEvent {
        event: Some(rank_event::Event::Aborted(Aborted {
            reason: reason as i32,
        })),
    }
}

/// The rank body's natural end as the session's one terminal event
/// (`gang.proto`'s `Outcome`, frozen by U5a-1): a completed run's
/// `Trained{artifact_digest}`, a typed failure's `Failed{reason}`.
fn outcome_event(result: outcome::Result) -> RankEvent {
    RankEvent {
        event: Some(rank_event::Event::Outcome(Outcome {
            result: Some(result),
        })),
    }
}

/// An admitted session: everything the spawned hold loop owns. The
/// [`RankHold`] is dropped with it, on every exit path, freeing the slot.
struct HeldSession {
    session: Arc<InferenceSession>,
    hold: RankHold,
    assign: Assign,
    identity: Option<TrainingSetIdentity>,
    lease: Duration,
    heartbeat: Duration,
    inbound: tonic::Streaming<RankControl>,
    events: mpsc::Sender<Result<RankEvent, Status>>,
    /// Where the hold loop delivers every round frame it reads off the
    /// inbound stream (`RoundInbox::deliver`), and reports a transport error
    /// on it (`RoundInbox::fail`). Dropped with the session.
    inbox: RoundInbox,
    /// A BODY-LESS session's end of the round protocol — the link built at
    /// admission (`gang_rounds::member_link`), which a `world_size == 1`
    /// session keeps for its whole life (or, under `test-hooks`, handed to
    /// the taker registered by [`GangServer::take_member_links`]). `None`
    /// for a body-bearing session: its link is the rank body's, moved into
    /// [`Self::body`]'s task at admission.
    member: Option<MemberLink>,
    /// The rank body (`run_member_rank`), for a `world_size > 1` session:
    /// the task training rank `assign.rank` over this session's link. Its
    /// end is the session's end (the fifth arm); `None` for a body-less
    /// session, which parks instead.
    body: Option<tokio::task::JoinHandle<RankOutcome>>,
    /// The body's cooperative stop flag — the trainer's epoch-boundary
    /// check reads it exactly as rank 0 reads its lease flag. Set when the
    /// session ends for any reason but the body's own end.
    body_cancel: Arc<AtomicBool>,
}

/// How a held session ends: the one stream event (or one trailer) the hold
/// loop emits before the stream closes.
enum SessionEnd {
    /// `Aborted{reason}` — a session arm's end (cancel, drain, a refuted or
    /// unavailable re-verification, the park bound), or the rank body's own
    /// pre-collective refusal.
    Aborted(AbortReason),
    /// The rank body's natural end: `Outcome{Trained | Failed}`.
    Outcome(outcome::Result),
    /// A protocol violation on the admitted stream: a status trailer.
    Violation(Status),
}

/// What one inbound control frame does to an admitted session.
enum FrameOutcome {
    /// A round frame, delivered to the member's inbox: the session stays
    /// held.
    Held,
    /// The session ends — one `Aborted{reason}` event, or one status
    /// trailer for a protocol violation.
    End(Result<AbortReason, Status>),
}

impl HeldSession {
    /// The inbound arm's decision for one control frame on an ADMITTED
    /// stream: `Cancel` ends the session cooperatively
    /// (`Aborted{Cancelled}`); a second `Assign` ends it as the K2 protocol
    /// violation (a status trailer, never a second admission); every OTHER
    /// frame goes through [`Self::dispatch_round_frame`], the one site the
    /// round protocol is wired at — a round frame keeps the session held.
    async fn on_control_frame(&mut self, frame: RankControl) -> FrameOutcome {
        match &frame.control {
            Some(rank_control::Control::Cancel(_)) => FrameOutcome::End(Ok(AbortReason::Cancelled)),
            Some(rank_control::Control::Assign(_)) => {
                FrameOutcome::End(Err(Status::invalid_argument(
                    "a second Assign on an admitted RunRank stream is a protocol violation",
                )))
            }
            _ => self.dispatch_round_frame(frame).await,
        }
    }

    /// THE dispatch point for every inbound control frame on an admitted
    /// stream that is neither `Assign` nor `Cancel`. A round frame
    /// (`RoundInbox::is_round_frame`: `RoundResult`, `RoundChunk`,
    /// `RoundCommit`, `RoundFault`) is delivered to the session's inbox —
    /// waiting for inbox room, which is the member's `Peer` reading at its
    /// own pace — and the session stays held; a delivery the inbox refuses
    /// (the member's link is gone: its owner dropped it, so no round can be
    /// in progress and none can start) ends the session with a
    /// `FailedPrecondition` trailer rather than buffering a round nobody
    /// will read. Everything else is a protocol violation ending the
    /// session with an `InvalidArgument` trailer: today that is exactly the
    /// EMPTY frame (`control: None`), and the match below is exhaustive on
    /// `RankControl`'s oneof so a NEW arm is a compile error here, never a
    /// silent delivery or refusal.
    async fn dispatch_round_frame(&mut self, frame: RankControl) -> FrameOutcome {
        if RoundInbox::is_round_frame(&frame) {
            return if self.inbox.deliver(frame).await {
                FrameOutcome::Held
            } else {
                FrameOutcome::End(Err(Status::failed_precondition(
                    "a round frame on an admitted RunRank stream whose member link has closed",
                )))
            };
        }
        FrameOutcome::End(Err(match frame.control {
            None => Status::invalid_argument(
                "an empty RankControl frame on an admitted RunRank stream is a protocol violation",
            ),
            Some(rank_control::Control::Assign(_) | rank_control::Control::Cancel(_)) => {
                unreachable!("Assign and Cancel are decided by on_control_frame, never dispatched")
            }
            Some(
                rank_control::Control::RoundResult(_)
                | rank_control::Control::RoundChunk(_)
                | rank_control::Control::RoundCommit(_)
                | rank_control::Control::RoundFault(_),
            ) => unreachable!(
                "every round arm is recognised by RoundInbox::is_round_frame and delivered above"
            ),
        }))
    }

    /// The HOLD loop: exactly five arms, of which a session takes four —
    /// the inbound stream, the host's phase (drain) watch, the
    /// re-verification tick, and EITHER the rank body's end (a
    /// body-bearing session) OR the park bound (a body-less one). Nothing
    /// else ends a held session. Every end is ONE stream event (or, for a
    /// protocol violation, one status) followed by the stream closing; the
    /// job row is never written here. A session that ends for any reason
    /// but its body's own end tells the body to stop (`body_cancel`) and
    /// severs the round inbox (`RoundInbox::sever`: the body's next
    /// collective ends `Disconnected`, and no frame of the body's reaches
    /// the stream after the terminal event); the body's task runs on to
    /// that fault on its own and its result is discarded.
    async fn hold(mut self) {
        let admission = Arc::clone(self.session.host_admission());
        let mut phase = admission.phase_receiver();
        let mut tick =
            tokio::time::interval_at(tokio::time::Instant::now() + self.heartbeat, self.heartbeat);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let park = tokio::time::sleep(self.lease);
        tokio::pin!(park);
        // The body, taken out of `self` so the arm below can await it
        // mutably beside the other arms' borrows.
        let mut body = self.body.take();
        // The client half-closing its send side (it sent its one `Assign`
        // and has nothing more to say) is NOT an end: the session stays
        // held and the arm is simply disabled. A transport error on the
        // inbound side means the client is gone — there is nobody to send
        // a reason to, so the loop ends silently (the body, if any, is told
        // to stop exactly as on every other foreign end).
        let mut inbound_open = true;

        let end: SessionEnd = loop {
            tokio::select! {
                frame = self.inbound.next(), if inbound_open => match frame {
                    Some(Ok(frame)) => match self.on_control_frame(frame).await {
                        FrameOutcome::Held => {}
                        FrameOutcome::End(Ok(reason)) => break SessionEnd::Aborted(reason),
                        FrameOutcome::End(Err(status)) => break SessionEnd::Violation(status),
                    },
                    Some(Err(status)) => {
                        // The member's link learns the inbound stream FAILED
                        // (its round in progress ends naming the reason,
                        // never a bare close) before the session goes.
                        self.inbox.fail(&status).await;
                        tracing::debug!(
                            job_id = %self.assign.job_id,
                            %status,
                            "gang hold: the inbound stream errored; ending the session"
                        );
                        self.body_cancel.store(true, Ordering::SeqCst);
                        self.inbox.sever();
                        return;
                    }
                    None => {
                        inbound_open = false;
                    }
                },
                // The rank body's end — a body-bearing session's own end.
                // A task that did not return an outcome (a panic in the
                // body's async prologue) is that body's failure.
                joined = async { body.as_mut().expect("guarded by the arm's condition").await },
                    if body.is_some() =>
                {
                    body = None;
                    break match joined {
                        Ok(RankOutcome::Trained { artifact_digest }) => SessionEnd::Outcome(
                            outcome::Result::Trained(TrainedOutcome { artifact_digest }),
                        ),
                        Ok(RankOutcome::Failed { reason }) => {
                            SessionEnd::Outcome(outcome::Result::Failed(FailedOutcome { reason }))
                        }
                        Ok(RankOutcome::Aborted(reason)) => SessionEnd::Aborted(reason),
                        Err(e) => SessionEnd::Outcome(outcome::Result::Failed(FailedOutcome {
                            reason: format!("rank {}: the rank body's task ended: {e}", self.assign.rank),
                        })),
                    };
                }
                // The `watch::Ref` `wait_for` yields is consumed inside this
                // arm's own future (never held in the `select!`'s output
                // across another arm's await), so the hold loop stays
                // `Send` for `tokio::spawn`. The sender dropping (the
                // session is gone) reads as a drain too: this host is going
                // away either way.
                () = async {
                    let _ = phase.wait_for(|p| *p != WorkerPhase::Running).await;
                } => break SessionEnd::Aborted(AbortReason::Drain),
                _ = tick.tick() => {
                    if let Err(end) =
                        reverify(&self.session, &self.assign, self.identity.as_ref(), self.lease)
                            .await
                    {
                        break SessionEnd::Aborted(end.abort_reason());
                    }
                }
                // The park bound: a body-less session only. A body-bearing
                // session's bound is its body's — the gang deadline on its
                // every round wait, and the re-verification tick above.
                _ = &mut park, if body.is_none() => break SessionEnd::Aborted(AbortReason::NoBody),
            }
        };

        // A foreign end while the body runs: stop it cooperatively. Its
        // task is detached (dropped with `body` below), never joined — it
        // ends at its next collective, which the severed inbox fails.
        if body.is_some() {
            self.body_cancel.store(true, Ordering::SeqCst);
        }
        let item = match end {
            SessionEnd::Aborted(reason) => Ok(aborted_event(reason)),
            SessionEnd::Outcome(result) => Ok(outcome_event(result)),
            SessionEnd::Violation(status) => Err(status),
        };
        // A receiver already gone (the client dropped the response stream)
        // has nobody to tell; the hold is freed regardless, below.
        let _ = self.events.send(item).await;
        drop(self.events);
        // The round protocol's ends go with the session: the inbox severed
        // first (no further delivery either way — the forwarder held the
        // last clone of the event sender, so the response stream closes
        // now), then the body-less session's link, then the slot.
        self.inbox.sever();
        drop(self.member);
        drop(body);
        drop(self.hold);
    }
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

        // Wire-level K2, decided before I-GANG runs: `world == 0` and
        // `rank >= world` are refused `InvalidArgument`, one case each.
        if assign.world == 0 {
            return Err(Status::invalid_argument("world must be greater than zero"));
        }
        if assign.rank >= assign.world {
            return Err(Status::invalid_argument("rank must be less than world"));
        }

        // I-GANG refuses ambient admin scope — for the WHOLE handler: a call
        // reaching this handler while wrapped in admin scope is refused
        // outright, before any row is even read, the same fixed way every
        // other determinant refuses.
        if TenantBinding::is_admin_scope() {
            self.record_refusal(GangRefusalReason::AdminScope);
            return Err(i_gang_refused());
        }

        // The row predicate: primary-key-only, no tenant predicate. The
        // row's OWN tenant comes back as raw text for the world>1 conjunct
        // below to derive and pin (never the caller's).
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
        // The row's own `lease_expires_at` is a ROW FACT
        // (`jammi_db::catalog::lease::LeaseFact`), decoded in Rust from the
        // raw stored text on EITHER backend — never a fault of the read
        // that found it (issue #574): a value that does not parse for this
        // backend refuses the SAME fixed way `LeaseDead` does, under its own
        // `test-hooks`-distinguishable variant, never conflated with it.
        match row.lease {
            LeaseFact::Live { .. } => {}
            LeaseFact::Dead => {
                self.record_refusal(GangRefusalReason::LeaseDead);
                return Err(i_gang_refused());
            }
            LeaseFact::Undecodable => {
                self.record_refusal(GangRefusalReason::LeaseUndecodable);
                return Err(i_gang_refused());
            }
        }

        // The row's own `world_size` is a ROW FACT
        // (`jammi_db::catalog::jobs_repo::WorldSizeFact`), never a fault of
        // the read that found it: a `spec` column that does not decode a
        // `world_size` at all refuses the SAME fixed way every other
        // determinant does — never `admission_catalog_fault`, which is
        // reserved for the read ITSELF faulting.
        let world_size = match row.world_size {
            WorldSizeFact::Undecodable => {
                self.record_refusal(GangRefusalReason::SpecUndecodable);
                return Err(i_gang_refused());
            }
            WorldSizeFact::Decoded(n) => n,
        };

        // The lattice is keyed on the ROW's own `world_size`, never the
        // caller's `assign.world` — a caller naming a `world` the row does
        // not agree with is itself a refusal. Keying the conjunct below on
        // `assign.world` instead would let a `world_size > 1` job admit
        // under a caller-supplied `world = 1`, skipping the pair conjunct
        // and the sidecar verify entirely (#566 R2).
        if assign.world != world_size {
            self.record_refusal(GangRefusalReason::WorldMismatch);
            return Err(i_gang_refused());
        }

        // The world>1 conjunct (#566 R2), on the ROW's decoded fact: (a)
        // the training-set identity pair is filled; (b) the row's own
        // tenant pins a strict resolution of a `ready` table whose sidecar
        // verifies the recorded digest. A `world_size == 1` row reads no
        // tenant value and no pair at all.
        let identity = if world_size > 1 {
            let (Some(training_set_ref), Some(training_set_location)) = (
                row.training_set_ref.as_deref(),
                row.training_set_location.as_deref(),
            ) else {
                self.record_refusal(GangRefusalReason::TrainingSetPairMissing);
                return Err(i_gang_refused());
            };
            let tenant: Option<TenantId> = match row.tenant_id.as_deref() {
                None => None,
                Some(text) => match text.parse::<TenantId>() {
                    Ok(tenant) => Some(tenant),
                    Err(_) => {
                        self.record_refusal(GangRefusalReason::TenantUndecodable);
                        return Err(i_gang_refused());
                    }
                },
            };
            // The strict resolver's own catalog read erroring is a catalog
            // fault — `Unavailable` through the SAME classification every
            // other admission-time catalog read on this path uses.
            let outcome = resolve_training_set_identity(
                self.session.result_store().as_ref(),
                tenant,
                training_set_ref,
                training_set_location,
            )
            .await
            .map_err(admission_catalog_fault)?;
            let reason = match outcome {
                TrainingSetOutcome::Verified => None,
                TrainingSetOutcome::AdminScopeRefused => Some(GangRefusalReason::AdminScope),
                TrainingSetOutcome::Unresolved => Some(GangRefusalReason::TrainingSetUnresolved),
                TrainingSetOutcome::NotReady => Some(GangRefusalReason::TrainingSetNotReady),
                TrainingSetOutcome::SidecarAbsent => {
                    Some(GangRefusalReason::TrainingSetSidecarAbsent)
                }
                TrainingSetOutcome::DigestMismatch => {
                    Some(GangRefusalReason::TrainingSetDigestMismatch)
                }
                TrainingSetOutcome::StoreFault => Some(GangRefusalReason::TrainingSetStoreFault),
            };
            if let Some(reason) = reason {
                self.record_refusal(reason);
                return Err(i_gang_refused());
            }
            Some(TrainingSetIdentity {
                tenant,
                training_set_ref: training_set_ref.to_string(),
                training_set_location: training_set_location.to_string(),
            })
        } else {
            None
        };

        // Coordinator freshness: the coordinator's own `instances` row must
        // be fresh. A genuine catalog fault reading this row is `Unavailable`
        // (`admission_catalog_fault`).
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

        // Every I-GANG determinant is satisfied. ONLY NOW the slot: the
        // holder CAS runs after the decision, never before it, so a refused
        // call never touches this host's holder and an admitted one is
        // held under a guard that frees the slot on every exit path.
        let hold = match self
            .session
            .host_admission()
            .admit_rank(&assign.job_id, row.attempts, self.heartbeat)
            .await
        {
            Ok(hold) => hold,
            Err(busy) => {
                tracing::debug!(
                    job_id = %assign.job_id,
                    ?busy,
                    "gang admission: every determinant held but this host's slot is busy"
                );
                let _: HolderBusy = busy;
                return Err(slot_busy());
            }
        };

        // `Admitted` is emitted only after the CAS succeeded; the guard
        // moves into the spawned HOLD loop with the inbound stream.
        let (events, rx) = mpsc::channel::<Result<RankEvent, Status>>(4);
        // The round protocol's member end, over this session's OWN event
        // sender: the hold loop delivers round frames to `inbox`, and the
        // link's events ride the same response stream as `Admitted` and the
        // session's end. Built before `Admitted` is queued, so an admitted
        // stream always has its inbox. `member_link` fails only outside a
        // runtime context, which a tonic handler never is.
        let (inbox, member) = member_link(events.clone()).map_err(|e| {
            Status::internal(format!("gang admission: building the member link: {e}"))
        })?;
        // The rank body: a `world_size > 1` session (the identity is the
        // world>1 conjunct's product) hands its link to the body, which
        // trains rank `assign.rank` over it as `RunnerRole::Rank`; a
        // `world_size == 1` session keeps (or, under `test-hooks`, offers)
        // the link and parks. The body is spawned BEFORE `Admitted` is
        // queued, so an admitted stream always has its body reading its
        // link before the coordinator's first round frame arrives.
        let body_cancel = Arc::new(AtomicBool::new(false));
        let (member, body) = match &identity {
            Some(identity) => {
                let assignment = MemberAssignment {
                    job_id: assign.job_id.clone(),
                    attempt: row.attempts,
                    rank: assign.rank,
                    world: assign.world,
                    coordinator_instance_id: assign.coordinator_instance_id.clone(),
                    tenant: identity.tenant,
                    training_set_ref: identity.training_set_ref.clone(),
                    training_set_location: identity.training_set_location.clone(),
                    spec_json: row.spec.clone(),
                };
                let body = tokio::spawn(run_member_rank(
                    Arc::clone(&self.session),
                    assignment,
                    member,
                    Arc::clone(&body_cancel),
                ));
                (None, Some(body))
            }
            None => (self.offer_member_link(member), None),
        };
        if events.try_send(Ok(admitted_event())).is_err() {
            // A fresh channel with room for four frames cannot refuse the
            // first; stated rather than unwrapped.
            return Err(Status::internal("gang admission: could not emit Admitted"));
        }
        let held = HeldSession {
            session: Arc::clone(&self.session),
            hold,
            assign,
            identity,
            lease: self.lease,
            heartbeat: self.heartbeat,
            inbound,
            events,
            inbox,
            member,
            body,
            body_cancel,
        };
        tokio::spawn(held.hold());
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// i2' at the classification: the three re-verification ends are
    /// pairwise distinct on the wire reason, on their scope, and on the
    /// count rule — never two ends that differ only in name.
    #[test]
    fn reverify_ends_are_pairwise_distinguishable_in_reason_scope_and_count() {
        let ends = [
            ReverifyEnd::Refuted,
            ReverifyEnd::Unavailable,
            ReverifyEnd::StoreUnavailable,
        ];
        for (i, a) in ends.iter().enumerate() {
            for b in &ends[i + 1..] {
                assert_ne!(a.abort_reason(), b.abort_reason(), "{a:?} vs {b:?}");
                assert!(
                    a.scope() != b.scope()
                        || a.counts_toward_assembly_attempts()
                            != b.counts_toward_assembly_attempts(),
                    "{a:?} and {b:?} must differ in scope or in the count rule, not only in name"
                );
            }
        }
        assert!(ReverifyEnd::Refuted.counts_toward_assembly_attempts());
        assert!(!ReverifyEnd::Unavailable.counts_toward_assembly_attempts());
        assert!(!ReverifyEnd::StoreUnavailable.counts_toward_assembly_attempts());
        assert_eq!(ReverifyEnd::StoreUnavailable.scope(), ReverifyScope::Member);
        assert_eq!(ReverifyEnd::Refuted.scope(), ReverifyScope::Assembly);
        assert_eq!(ReverifyEnd::Unavailable.scope(), ReverifyScope::Assembly);
    }
}
