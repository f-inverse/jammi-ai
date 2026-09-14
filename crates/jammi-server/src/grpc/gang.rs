//! `GangService` — the coordinator-to-member admission wire for a multi-host
//! gang run (`CONTRACT-U5a.md`, U5a-1).
//!
//! U5a-1 freezes the wire (`jammi.v1.gang`, see `gang.proto`) and this
//! handler's own decidable-without-a-database-verb surface: reading the
//! first inbound `Assign` under a fixed bound and refusing the wire-level K2
//! edges (`world == 0`, `rank >= world`) before any row is ever read. §I1's
//! own row predicate (`get_job_for_rank`) and the write-once CAS pair
//! (§W2 Fill) are a later step's within this same unit (the migration adding
//! `jobs.training_set_ref`/`training_set_location` does not exist yet on
//! this branch) — until that step wires them in here, every wire-valid call
//! is refused `Unimplemented`, matching f1': "having no `HostAdmission`
//! session to hand the call to, returns `Unimplemented`". `HostAdmission`
//! itself (the admit-and-hold session, drain, re-verification) is U5a-2's,
//! built on top of this handler once it exists.
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
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::storage::StorageUrl;
use jammi_db::store::ResultStore;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use tonic::{Request, Response, Status};

use tokio_stream::StreamExt;

use crate::grpc::proto::gang::gang_service_server::GangService;
use crate::grpc::proto::gang::{rank_control, RankControl, RankEvent};
use crate::grpc::wire::map_engine_error;

/// The fixed bound for the first inbound `Assign` frame (§H3 step 1: "a
/// silent client is dropped at that bound"). A handler-local constant, not a
/// config knob, sized to one round trip — not `[lease] heartbeat_secs` or
/// any deployment-tunable value.
const FIRST_ASSIGN_BOUND: Duration = Duration::from_secs(10);

/// Server-side handler for the gang admission surface. Holds the shared
/// engine session for its catalog + result store — the same handle
/// [`crate::grpc::peer::PeerServer`] holds for the owner side of the
/// distributed data plane.
///
/// The `session` field is unread by [`GangServer::run_rank`] on THIS branch:
/// §I1's row predicate (`get_job_for_rank`) does not exist yet (the migration
/// adding `jobs.training_set_ref`/`training_set_location` is this unit's
/// still-pending db step), so nothing here yet derives a job's tenant to
/// pass into [`resolve_training_set_identity`]. Held now (never constructed
/// lazily later) so `run_rank`'s signature and this struct's shape do not
/// change again once that wiring lands — the same reason
/// [`crate::grpc::peer::PeerServer`] holds its session even for the requests
/// its own tenant-free handler never reads a tenant from.
#[allow(dead_code)]
pub struct GangServer {
    session: Arc<InferenceSession>,
}

impl GangServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
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

        // §I1's row predicate (`get_job_for_rank`) and §W2's write-once pair
        // need the `jobs.training_set_ref`/`training_set_location` columns,
        // which this branch's migration does not add — that is this unit's
        // NEXT step. Nothing calls `resolve_training_set_identity` from here
        // yet; the seam it will be wired through (once a job row and its
        // derived tenant are in hand) is that function, below.
        let _ = assign;
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
