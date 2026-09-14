//! `GangService` — the coordinator-to-member admission wire for a multi-host
//! gang run (`CONTRACT-U5a.md`, U5a-1).
//!
//! U5a-1 freezes the wire (`jammi.v1.gang`, see `gang.proto`), the wire-level
//! K2 edges (`world == 0`, `rank >= world`) decided before any row is ever
//! read, and this handler's own full I-GANG decision (§I1): `get_job_for_rank`
//! for the row predicate, `resolve_training_set_identity` for the
//! `world_size > 1` sidecar verify, and `fresh_instance` for the
//! coordinator's own liveness. Every determinant collapses to the SAME
//! `FailedPrecondition` status with a FIXED message (§I1 Non-disclosure) —
//! the listener discloses neither a job's existence, claimant, nor attempt.
//! Having decided every determinant and found no reason to refuse, this
//! handler still has no `HostAdmission` session to hand the call to (U5a-2
//! builds that), so it ends `Unimplemented` — f1': "a call satisfying EVERY
//! I-GANG determinant still reaches the handler ... and, having no
//! `HostAdmission` session to hand the call to, returns `Unimplemented`."
//! `HostAdmission` itself (the admit-and-hold session, drain,
//! re-verification) is U5a-2's, built on top of this handler once it exists.
//!
//! Served only on the internal `[server] peer_bind` listener, mounted beside
//! `PeerService` (`OssServer::bind`) — never on the public listener, never
//! wrapped by the tenant-binding layer. Tenant is derived from the verified
//! `jobs` row, never the caller (I-GANG) — this handler never reads a
//! `SessionTenant` extension the way the tenant-wrapped services do.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::RankAdmissionRow;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::status::{JobStatus, ResultTableStatus};
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use tonic::{Request, Response, Status};

use tokio_stream::StreamExt;

use crate::grpc::proto::gang::gang_service_server::GangService;
use crate::grpc::proto::gang::{rank_control, RankControl, RankEvent};
use crate::grpc::wire::map_engine_error;

/// §I1 Non-disclosure: every I-GANG refusal — row absent, wrong status,
/// wrong claimant, wrong attempt, lease not live, the training-set pair
/// missing/unresolved for `world_size > 1`, or the coordinator's own
/// `instances` row not fresh — collapses to this ONE status with this ONE
/// fixed message. Never interpolate a job id, a claimant, or a reason into
/// it: that is exactly the disclosure this property forbids.
const I_GANG_REFUSAL_MESSAGE: &str = "gang admission refused";

fn i_gang_refused() -> Status {
    Status::failed_precondition(I_GANG_REFUSAL_MESSAGE)
}

/// The fixed bound for the first inbound `Assign` frame (§H3 step 1: "a
/// silent client is dropped at that bound"). A handler-local constant, not a
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
}

impl GangServer {
    pub fn new(session: Arc<InferenceSession>, lease: Duration) -> Self {
        Self { session, lease }
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

        // §H3 step 1: await the first inbound frame inline, bounded. A
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

        // §I1(a): the row predicate. Primary-key-only, no tenant predicate —
        // `Catalog::get_job_for_rank` never reads `assign.job_id`'s tenant
        // from the caller (I-GANG: tenant is derived from the row itself,
        // below).
        let catalog = self.session.catalog();
        let row: RankAdmissionRow = match catalog.get_job_for_rank(&assign.job_id).await {
            Ok(Some(row)) => row,
            Ok(None) => return Err(i_gang_refused()),
            Err(e) => return Err(map_engine_error(e)),
        };

        let running = row.status == JobStatus::Running.to_string();
        let claimant_matches =
            row.claimed_by.as_deref() == Some(assign.coordinator_instance_id.as_str());
        let attempt_matches = i64::from(row.attempts) == assign.attempt;
        if !running || !claimant_matches || !attempt_matches || !row.lease_live {
            return Err(i_gang_refused());
        }

        // §I1(b), only for `world_size > 1`: the pair must be filled, and
        // its sidecar must verify — a separate, host-local step, never
        // folded into `get_job_for_rank`'s own statement (§I1).
        if assign.world > 1 {
            let (Some(training_set_ref), Some(training_set_location)) = (
                row.training_set_ref.as_deref(),
                row.training_set_location.as_deref(),
            ) else {
                return Err(i_gang_refused());
            };
            if resolve_training_set_identity(
                self.session.result_store().as_ref(),
                row.tenant_id,
                training_set_ref,
                training_set_location,
            )
            .await
            .is_err()
            {
                return Err(i_gang_refused());
            }
        }

        // §I1: the coordinator's own `instances` row must be fresh.
        match catalog
            .fresh_instance(&assign.coordinator_instance_id, self.lease)
            .await
        {
            Ok(true) => {}
            Ok(false) => return Err(i_gang_refused()),
            Err(e) => return Err(map_engine_error(e)),
        }

        // Every I-GANG determinant is satisfied. This unit has no
        // `HostAdmission` session to hand the call to (U5a-2 builds that) —
        // f1'.
        Err(Status::unimplemented(
            "gang admission is not implemented on this build",
        ))
    }
}

/// §W2 Resolution: verify (never locate) the result table
/// `training_set_location` names, for the tenant `get_job_for_rank`
/// resolves the calling job under — the ONE tenant-pinned lookup a rank
/// performs, no listing, no candidate search.
///
/// Two properties bind this function, both round-11 folds:
///
/// - **Ruling 4 (admin-scope is ambient, guard it explicitly here).**
///   [`TenantBinding::is_admin_scope`] reads ambient task-local state this
///   call site does not control by construction, so this function refuses
///   immediately, before ever calling the strict resolver, whenever admin
///   scope is active — regardless of which tenant or table it was asked to
///   resolve. This never relies on
///   `jammi_db::catalog::Catalog::get_result_table_for_tenant`'s OWN
///   admin-scope behaviour (which this contract does not change: called
///   directly, outside this guard, it still resolves any tenant's table
///   under admin scope, matching the rest of that repo's verbs).
/// - **Ruling 5 (a strict-predicate verb, never the relaxed read).** The
///   lookup below is exactly `Catalog::get_result_table_for_tenant`, never
///   `Catalog::get_result_table` — the strict predicate `tenant_id = $t OR
///   (tenant_id IS NULL AND $t IS NULL)`, never the relaxed `OR tenant_id IS
///   NULL` a *read* resolver uses to also see a global row.
///
/// The admission-time classification collapses every unresolvable /
/// unverifiable outcome — name absent, the strict resolver returns `None`,
/// the row not `ready`, the sidecar absent (`Ok(None)`), a digest mismatch,
/// or [`ResultStore::read_materialization_manifest`] itself erroring (a
/// network/backend fault reaching this host's own store) — into the ONE
/// member-scoped `FailedPrecondition` §I4 already names for this class; the
/// three-way split that distinguishes a `StoreUnavailable` re-verification
/// end from a `Refuted` one is §I3's, mid-stream, once an admitted session
/// (`HostAdmission`, U5a-2) exists to be mid-stream in.
///
/// This is the ONLY call site this program has for
/// `Catalog::get_result_table_for_tenant` outside its own crate's tests
/// (`impossibility_claims`, below) — the gang `RunRank` handler is meant to
/// be its sole caller, though nothing yet threads a job's derived tenant and
/// `training_set_ref`/`training_set_location` pair into it from
/// [`GangServer::run_rank`] (that wiring needs `get_job_for_rank`, this
/// unit's next step).
pub async fn resolve_training_set_identity(
    store: &ResultStore,
    tenant: Option<TenantId>,
    training_set_ref: &str,
    training_set_location: &str,
) -> Result<(), Status> {
    if TenantBinding::is_admin_scope() {
        return Err(Status::failed_precondition(
            "training-set resolution is refused under admin scope",
        ));
    }

    let record: ResultTableRecord = match store
        .catalog()
        .get_result_table_for_tenant(training_set_location, tenant)
        .await
        .map_err(map_engine_error)?
    {
        Some(record) if record.status == ResultTableStatus::Ready.to_string() => record,
        _ => return Err(Status::failed_precondition("training set unresolved")),
    };

    let url = match StorageUrl::parse(&record.parquet_path) {
        Ok(url) => url,
        Err(_) => return Err(Status::failed_precondition("training set unresolved")),
    };

    // Admission-time classification (§W2 Resolution): every unresolvable /
    // unverifiable outcome — `Ok(None)` (no sidecar), a digest mismatch, or
    // the read itself erroring — collapses to the same member-scoped
    // `FailedPrecondition`. The three-way split into `Refuted` /
    // `StoreUnavailable` is §I3's re-verification classification, which
    // needs an admitted (`HostAdmission`) session to re-verify inside — not
    // this admission-time call.
    match store.read_materialization_manifest(&url).await {
        Ok(Some(manifest)) if manifest.artifact.0 == training_set_ref => Ok(()),
        Ok(_) | Err(_) => Err(Status::failed_precondition("training set unresolved")),
    }
}
