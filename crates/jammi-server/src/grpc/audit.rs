//! `AuditService` gRPC implementation.
//!
//! Three verbs land on the wire: `AuditLog`, `AuditFetchByQueryId`, and
//! `AuditFetchRecent`. Each is a thin adapter over the transport-agnostic
//! [`Session`] abstraction (never raw [`InferenceSession`]
//! calls): proto in, one flat `Session::audit_*` call, proto out. The service
//! reimplements no signing, storage, or query logic.
//!
//! The [`pb::PerQueryAudit`] message mirrors the Rust [`jammi_db::PerQueryAudit`]
//! field for field, including its server-computed `signature`. On `AuditLog` the
//! caller leaves `tenant_id` and `signature` empty — the engine stamps the
//! session tenant and signs each record; both fields are populated on every
//! record a fetch verb returns. `executed_at` rides as epoch microseconds (the
//! audit table's storage form).
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface. The audit primitive requires a bound tenant — an unscoped call
//! surfaces as `failed_precondition`.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::{AuditError, PerQueryAudit, Session};
use jammi_wire::{attach_audit_detail, parse_query_id};
use tonic::{Code, Request, Response, Status};

use crate::grpc::proto::audit as pb;
use crate::grpc::proto::audit::audit_service_server::AuditService;
use crate::grpc::wire::{scoped, session_tenant};

/// Server-side handler for the audit gRPC surface. Holds a shared engine session
/// it wraps in a [`Session`] per call to reach the unified transport
/// surface.
pub struct AuditServer {
    session: Arc<InferenceSession>,
}

impl AuditServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// A [`Session`] over the shared engine; see [`crate::grpc::inference`] for
    /// the tenant-scope wiring rationale.
    fn local(&self) -> Session {
        Session::new(Arc::clone(&self.session))
    }
}

#[tonic::async_trait]
impl AuditService for AuditServer {
    async fn audit_log(
        &self,
        request: Request<pb::AuditLogRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let records = req
            .records
            .into_iter()
            .map(record_from_proto)
            .collect::<Result<Vec<_>, Status>>()?;
        let session = self.local();

        scoped(&self.session, tenant, || session.audit_log(records))
            .await
            .map_err(map_audit_error)?;

        Ok(Response::new(()))
    }

    async fn audit_fetch_by_query_id(
        &self,
        request: Request<pb::AuditFetchByQueryIdRequest>,
    ) -> Result<Response<pb::AuditFetchByQueryIdResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let query_id = parse_query_id(&req.query_id)?;
        let session = self.local();

        let record = scoped(&self.session, tenant, || {
            session.audit_fetch_by_query_id(query_id)
        })
        .await
        .map_err(map_audit_error)?;

        Ok(Response::new(pb::AuditFetchByQueryIdResponse {
            record: record.map(Into::into),
        }))
    }

    async fn audit_fetch_recent(
        &self,
        request: Request<pb::AuditFetchRecentRequest>,
    ) -> Result<Response<pb::AuditFetchRecentResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let session = self.local();

        let records = scoped(&self.session, tenant, || {
            session.audit_fetch_recent(req.limit as usize)
        })
        .await
        .map_err(map_audit_error)?;

        Ok(Response::new(pb::AuditFetchRecentResponse {
            records: records.into_iter().map(Into::into).collect(),
        }))
    }
}

/// Build an engine [`PerQueryAudit`] from the wire message. This is the
/// receive-side decode (not a pure wire conversion): it shares the query-id
/// parse with the wire module but constructs the record through
/// [`PerQueryAudit::new`] — enforcing the length-agreement invariant between
/// `top_k_result_ids` and `retrieval_scores` at the boundary — and maps the
/// resulting [`AuditError`] through the server's [`map_audit_error`]. The
/// caller-supplied `tenant_id`/`signature`/`executed_at` are ignored — the
/// engine stamps the tenant, signs, and timestamps on write.
fn record_from_proto(p: pb::PerQueryAudit) -> Result<PerQueryAudit, Status> {
    let query_id = parse_query_id(&p.query_id)?;
    let query_lineage: serde_json::Value = if p.query_lineage.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str(&p.query_lineage)
            .map_err(|e| Status::invalid_argument(format!("query_lineage is not JSON: {e}")))?
    };
    PerQueryAudit::new(
        query_id,
        p.model_id,
        p.model_version,
        query_lineage,
        p.top_k_result_ids,
        p.retrieval_scores,
    )
    .map_err(map_audit_error)
}

/// Map an [`AuditError`] onto a gRPC [`Status`], preserving the failure kind so
/// a client can tell a bad request from an internal fault. A missing tenant
/// binding is a precondition failure (the caller must scope the session first);
/// a signature mismatch is data loss; size/length violations are bad arguments.
///
/// The `code` + `message` are the idiomatic gRPC surface; on top of that every
/// status carries a faithful [`jammi_wire`] audit-error detail so a remote
/// the data-plane client reconstructs the *exact* [`AuditError`] the in-process path
/// returns. The detail is built centrally here so the faithful path covers the
/// whole `AuditError` enum from one place — the audit analogue of
/// `map_engine_error`.
fn map_audit_error(err: AuditError) -> Status {
    let code = match &err {
        AuditError::LengthMismatch { .. } | AuditError::LineageTooLarge { .. } => {
            Code::InvalidArgument
        }
        AuditError::NoTenantBinding | AuditError::MasterKey(_) => Code::FailedPrecondition,
        AuditError::SignatureMismatch(_) => Code::DataLoss,
        AuditError::Serde(_) | AuditError::Storage(_) | AuditError::Broker(_) => Code::Internal,
    };
    attach_audit_detail(code, err.to_string(), &err)
}
