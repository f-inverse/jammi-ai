//! Session state shared between the gRPC services and the `TenantInterceptor`.
//!
//! Per ADR-01 §3.2, the wire surface for tenant binding is a typed gRPC verb
//! (the tenant trio on [`CatalogService`](crate::grpc::catalog)). The verb
//! updates a shared [`SessionStore`] keyed by an opaque `SessionId` (carried in
//! the `jammi-session-id` request header). Every downstream request runs through
//! the [`TenantInterceptor`], which reads the header, looks up the bound tenant
//! in the store, and attaches it to the request's extensions as a
//! [`SessionTenant`] for downstream services (Flight SQL handlers, the engine's
//! tenant-scope analyzer rule, and the engine-backed gRPC verbs) to consume.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::{Request, Status};

use jammi_db::TenantId;

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

/// In-process map of `SessionId` → bound tenant. Shared between the tenant trio
/// on [`CatalogService`](crate::grpc::catalog) (writers) and the
/// [`TenantInterceptor`] (readers).
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

/// Header name carrying the session identifier. Clients must include this on
/// every request that needs tenant-scoped semantics. Defined once in
/// [`jammi_wire`] (shared with the client crates) and re-exported here so
/// server-side callers and the integration tests keep their existing path.
pub use jammi_wire::SESSION_HEADER;

/// Tonic interceptor that resolves the request's session and attaches the
/// bound tenant to the request extensions. Apply to every Tonic service
/// (Flight SQL + the typed gRPC services) so downstream handlers can read the
/// bound tenant from the extension.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

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
}
