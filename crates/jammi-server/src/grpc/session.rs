//! `SessionService` gRPC implementation + `TenantInterceptor`.
//!
//! Per ADR-01 §3.2, the wire surface for tenant binding is a typed gRPC
//! service. The service updates a shared [`SessionStore`] keyed by an opaque
//! `SessionId` (carried in the `jammi-session-id` request header). Every
//! downstream request runs through the [`TenantInterceptor`], which reads
//! the header, looks up the bound tenant in the store, and attaches it to
//! the request's extensions as a [`SessionTenant`] for downstream services
//! (Flight SQL handlers and the engine's tenant-scope analyzer rule) to
//! consume.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, RwLock};

use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::{Request, Response, Status};

use jammi_engine::TenantId;

use super::proto::session::session_service_server::SessionService;
use super::proto::session::{GetTenantResponse, SetTenantRequest, Tenant};

/// Opaque session identifier carried in the `jammi-session-id` request
/// header. Clients mint a UUID for each connection / session; the server
/// uses it as the lookup key for the tenant binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Per-session tenant attached to a gRPC / Flight SQL request by the
/// [`TenantInterceptor`]. Downstream handlers read it via
/// `request.extensions().get::<SessionTenant>()`.
#[derive(Debug, Clone, Copy)]
pub struct SessionTenant(pub Option<TenantId>);

/// In-process map of `SessionId` → bound tenant. Shared between the
/// `SessionService` impl (writers) and the `TenantInterceptor` (readers).
#[derive(Debug, Default, Clone)]
pub struct SessionStore {
    inner: Arc<RwLock<HashMap<SessionId, Option<TenantId>>>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, session: SessionId, tenant: Option<TenantId>) {
        self.inner
            .write()
            .expect("session store lock poisoned")
            .insert(session, tenant);
    }

    pub fn get(&self, session: &SessionId) -> Option<TenantId> {
        self.inner
            .read()
            .expect("session store lock poisoned")
            .get(session)
            .copied()
            .flatten()
    }

    pub fn clear(&self, session: &SessionId) {
        self.inner
            .write()
            .expect("session store lock poisoned")
            .remove(session);
    }
}

/// Header name carrying the session identifier. Clients must include this
/// on every request that needs tenant-scoped semantics.
pub const SESSION_HEADER: &str = "jammi-session-id";

/// Tonic gRPC service impl backed by a shared [`SessionStore`].
#[derive(Debug, Clone)]
pub struct SessionServer {
    store: SessionStore,
}

impl SessionServer {
    pub fn new(store: SessionStore) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl SessionService for SessionServer {
    async fn set_tenant(&self, request: Request<SetTenantRequest>) -> Result<Response<()>, Status> {
        let session = session_id_from_request(&request)?;
        let body = request.into_inner();
        let tenant = body
            .tenant
            .ok_or_else(|| Status::invalid_argument("tenant field is required"))?;
        let parsed = parse_tenant(&tenant)?;
        self.store.set(session, parsed);
        Ok(Response::new(()))
    }

    async fn get_tenant(
        &self,
        request: Request<()>,
    ) -> Result<Response<GetTenantResponse>, Status> {
        let session = session_id_from_request(&request)?;
        let tenant = self.store.get(&session);
        Ok(Response::new(GetTenantResponse {
            tenant: Some(Tenant {
                id: tenant.map(|t| t.to_string()).unwrap_or_default(),
            }),
        }))
    }

    async fn clear_tenant(&self, request: Request<()>) -> Result<Response<()>, Status> {
        let session = session_id_from_request(&request)?;
        self.store.clear(&session);
        Ok(Response::new(()))
    }
}

/// Tonic interceptor that resolves the request's session and attaches the
/// bound tenant to the request extensions. Apply to every Tonic service
/// (Flight SQL + SessionService) so downstream handlers can read the bound
/// tenant from the extension.
#[derive(Clone)]
pub struct TenantInterceptor {
    store: SessionStore,
}

impl TenantInterceptor {
    pub fn new(store: SessionStore) -> Self {
        Self { store }
    }
}

impl Interceptor for TenantInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        // Resolving the session is optional: a request without a session
        // header simply runs unscoped (no tenant attached).
        let tenant = match read_session_header(&request) {
            Some(session) => self.store.get(&session),
            None => None,
        };
        request.extensions_mut().insert(SessionTenant(tenant));
        Ok(request)
    }
}

/// Extract the `jammi-session-id` header from a request.
fn read_session_header<T>(request: &Request<T>) -> Option<SessionId> {
    request
        .metadata()
        .get(SESSION_HEADER)
        .and_then(|v: &MetadataValue<_>| v.to_str().ok())
        .map(SessionId::new)
}

/// Extract the session id and return a typed `InvalidArgument` error if it's
/// missing. Used by methods that mutate or read session state directly.
fn session_id_from_request<T>(request: &Request<T>) -> Result<SessionId, Status> {
    read_session_header(request).ok_or_else(|| {
        Status::invalid_argument(format!(
            "request missing required `{SESSION_HEADER}` metadata header"
        ))
    })
}

/// Parse a wire-format [`Tenant`] into the engine's [`TenantId`] newtype.
/// Empty string is interpreted as "no tenant" → `Ok(None)`. Any other value
/// must parse as a non-nil UUID per ADR-00.
fn parse_tenant(t: &Tenant) -> Result<Option<TenantId>, Status> {
    if t.id.is_empty() {
        return Ok(None);
    }
    TenantId::from_str(&t.id)
        .map(Some)
        .map_err(|e| Status::invalid_argument(format!("invalid tenant id: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t_a() -> TenantId {
        TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap()
    }

    fn t_b() -> TenantId {
        TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap()
    }

    #[test]
    fn store_get_set_clear_roundtrip() {
        let store = SessionStore::new();
        let sid = SessionId::new("conn-1");
        assert!(store.get(&sid).is_none());

        store.set(sid.clone(), Some(t_a()));
        assert_eq!(store.get(&sid), Some(t_a()));

        store.set(sid.clone(), Some(t_b()));
        assert_eq!(store.get(&sid), Some(t_b()));

        store.set(sid.clone(), None);
        assert!(store.get(&sid).is_none());

        store.set(sid.clone(), Some(t_a()));
        store.clear(&sid);
        assert!(store.get(&sid).is_none());
    }

    #[test]
    fn store_isolates_sessions() {
        let store = SessionStore::new();
        let s1 = SessionId::new("conn-1");
        let s2 = SessionId::new("conn-2");
        store.set(s1.clone(), Some(t_a()));
        store.set(s2.clone(), Some(t_b()));
        assert_eq!(store.get(&s1), Some(t_a()));
        assert_eq!(store.get(&s2), Some(t_b()));
    }

    #[test]
    fn parse_tenant_empty_means_unscoped() {
        assert_eq!(parse_tenant(&Tenant { id: String::new() }).unwrap(), None);
    }

    #[test]
    fn parse_tenant_rejects_nil_and_garbage() {
        assert!(parse_tenant(&Tenant {
            id: "00000000-0000-0000-0000-000000000000".into()
        })
        .is_err());
        assert!(parse_tenant(&Tenant {
            id: "not-a-uuid".into()
        })
        .is_err());
    }

    #[test]
    fn parse_tenant_round_trip() {
        let parsed = parse_tenant(&Tenant {
            id: t_a().to_string(),
        })
        .unwrap();
        assert_eq!(parsed, Some(t_a()));
    }

    #[tokio::test]
    async fn interceptor_attaches_unscoped_when_no_header() {
        let mut interceptor = TenantInterceptor::new(SessionStore::new());
        let req = Request::new(());
        let req = interceptor.call(req).unwrap();
        let tenant = req.extensions().get::<SessionTenant>().unwrap();
        assert!(tenant.0.is_none());
    }

    #[tokio::test]
    async fn interceptor_attaches_bound_tenant_when_header_present() {
        let store = SessionStore::new();
        store.set(SessionId::new("conn-1"), Some(t_a()));
        let mut interceptor = TenantInterceptor::new(store);
        let mut req = Request::new(());
        req.metadata_mut()
            .insert(SESSION_HEADER, "conn-1".parse().unwrap());
        let req = interceptor.call(req).unwrap();
        let tenant = req.extensions().get::<SessionTenant>().unwrap();
        assert_eq!(tenant.0, Some(t_a()));
    }

    #[tokio::test]
    async fn set_tenant_requires_session_header() {
        let server = SessionServer::new(SessionStore::new());
        let req = Request::new(SetTenantRequest {
            tenant: Some(Tenant {
                id: t_a().to_string(),
            }),
        });
        let err = server.set_tenant(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn set_then_get_then_clear_round_trip() {
        let store = SessionStore::new();
        let server = SessionServer::new(store.clone());
        let session_id = "conn-1";

        // Set
        let mut req = Request::new(SetTenantRequest {
            tenant: Some(Tenant {
                id: t_a().to_string(),
            }),
        });
        req.metadata_mut()
            .insert(SESSION_HEADER, session_id.parse().unwrap());
        server.set_tenant(req).await.unwrap();
        assert_eq!(store.get(&SessionId::new(session_id)), Some(t_a()));

        // Get
        let mut req = Request::new(());
        req.metadata_mut()
            .insert(SESSION_HEADER, session_id.parse().unwrap());
        let resp = server.get_tenant(req).await.unwrap().into_inner();
        assert_eq!(resp.tenant.unwrap().id, t_a().to_string());

        // Clear
        let mut req = Request::new(());
        req.metadata_mut()
            .insert(SESSION_HEADER, session_id.parse().unwrap());
        server.clear_tenant(req).await.unwrap();
        assert!(store.get(&SessionId::new(session_id)).is_none());

        // GetTenant after clear returns empty string.
        let mut req = Request::new(());
        req.metadata_mut()
            .insert(SESSION_HEADER, session_id.parse().unwrap());
        let resp = server.get_tenant(req).await.unwrap().into_inner();
        assert!(resp.tenant.unwrap().id.is_empty());
    }
}
