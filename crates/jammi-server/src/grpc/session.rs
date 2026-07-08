//! Session state and the tenant-binding seam shared by every gRPC service and
//! the Flight SQL provider.
//!
//! Tenant binding is uniformly resolver-driven: one [`TenantResolver`] resolves
//! each request's transport metadata to a [`TenantScope`], and the async
//! tenant-binding tower layer ([`crate::tenant_resolver_layer`]) maps that scope
//! onto the request's [`SessionTenant`] extension that every downstream handler
//! reads (Flight SQL handlers, the engine's tenant-scope analyzer rule, and the
//! engine-backed gRPC verbs). There is exactly ONE binder — no legacy
//! interceptor path.
//!
//! The engine ships [`SessionIdTenantResolver`], the OSS-cooperative default:
//! per ADR-01 §3.2 a client binds its tenant with a typed gRPC verb (the tenant
//! trio on [`CatalogService`](crate::grpc::catalog)) that updates a shared
//! [`SessionStore`] keyed by an opaque `SessionId` (the `jammi-session-id`
//! header); the resolver reads that header, looks the tenant up, and returns
//! [`TenantScope::Tenant`] when bound or [`TenantScope::Global`] when absent —
//! the EXPLICIT unscoped choice, never a silent default. A downstream composing
//! the seam supplies its own authenticating resolver instead (returning
//! `Tenant`/`Err`, never `Global`), unifying the composability seam with the
//! bring-your-own-auth seam.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use tonic::metadata::{MetadataMap, MetadataValue};
use tonic::Status;

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
/// tenant-binding tower layer, which maps the resolved [`TenantScope`] onto it
/// ([`TenantScope::Tenant`] → `Some`, [`TenantScope::Global`] → `None`).
/// Downstream handlers read it via
/// `request.extensions().get::<SessionTenant>()`.
#[derive(Debug, Clone, Copy)]
pub struct SessionTenant(pub Option<TenantId>);

/// In-process map of `SessionId` → bound tenant. Shared between the tenant trio
/// on [`CatalogService`](crate::grpc::catalog) (writers) and the
/// [`SessionIdTenantResolver`] (reader).
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

/// The scope a request runs under, as decided by a [`TenantResolver`].
///
/// [`TenantScope::Global`] is the EXPLICIT unscoped choice — the honest value
/// an OSS-cooperative or single-tenant deployment returns when no tenant is
/// bound. It is never a silent default a rejection falls through to: a resolver
/// that authenticates (the platform / BYO-auth path) returns `Tenant`/`Err` and
/// NEVER `Global`; the "reject-don't-default" contract lives per-resolver, not
/// in this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TenantScope {
    /// Bound to a specific tenant; the request sees `tenant_id = t` ∪ global rows.
    Tenant(TenantId),
    /// Explicitly unscoped; the request sees only global (`tenant_id IS NULL`)
    /// rows. The deliberate no-tenant answer, not a fallback.
    Global,
}

/// The single tenant-binding seam. One resolver resolves each request's
/// transport metadata to a [`TenantScope`]; the async tenant-binding tower layer
/// ([`crate::tenant_resolver_layer`]) applies it to every engine service, and
/// the Flight SQL provider applies it to the `db.sql` lane. There is exactly one
/// binder — no separate interceptor.
///
/// The engine default is [`SessionIdTenantResolver`] (OSS-cooperative:
/// `jammi-session-id` → [`SessionStore`]). A downstream composing the seam
/// supplies its own authenticating resolver instead, unifying the composability
/// seam with the bring-your-own-auth seam. Engine-generic: `&MetadataMap` in,
/// engine [`TenantScope`] out — the seam names no consumer, scheme, or identity
/// provider (a JWT from an OIDC gateway, a signed bearer, a service-to-service
/// token all reach for this one shape).
#[async_trait::async_trait]
pub trait TenantResolver: Send + Sync + 'static {
    /// Resolve a request's transport metadata to the scope it runs under, or
    /// reject it. An authenticating resolver returns [`TenantScope::Tenant`] for
    /// a valid credential and `Err(Status)` (`UNAUTHENTICATED`) for a
    /// missing/invalid one — it never returns [`TenantScope::Global`], so a
    /// rejected caller can never fall through to an unscoped read. The
    /// OSS-cooperative default resolver legitimately returns `Global` when no
    /// session tenant is bound (the explicit unscoped choice).
    async fn resolve(&self, metadata: &MetadataMap) -> Result<TenantScope, Status>;
}

/// Header name carrying the session identifier. Clients must include this on
/// every request that needs tenant-scoped semantics. Defined once in
/// [`jammi_wire`] (shared with the client crates) and re-exported here so
/// server-side callers and the integration tests keep their existing path.
pub use jammi_wire::SESSION_HEADER;

/// The engine's default [`TenantResolver`] — the OSS-cooperative,
/// single-listener multitenancy behavior made explicit. It reads the
/// `jammi-session-id` header, looks the session's bound tenant up in the shared
/// [`SessionStore`] (written by the `CatalogService` tenant trio), and returns
/// [`TenantScope::Tenant`] when bound or [`TenantScope::Global`] when the header
/// is absent or the session carries no binding.
///
/// This resolver NEVER rejects: an unauthenticated `jammi-session-id` header is
/// a client-minted correlation id on a trusted network, not a credential, so a
/// missing binding is the explicit `Global` scope, not an error. A deployment
/// that needs authentication supplies an authenticating resolver instead (which
/// returns `Tenant`/`Err`, never `Global`).
#[derive(Clone)]
pub struct SessionIdTenantResolver {
    store: SessionStore,
}

impl SessionIdTenantResolver {
    pub fn new(store: SessionStore) -> Self {
        Self { store }
    }

    /// Construct the engine default resolver as the boxed trait object the
    /// [`crate::runtime::GrpcChain`] field and the Flight provider hold.
    pub fn arc(store: SessionStore) -> Arc<dyn TenantResolver> {
        Arc::new(Self::new(store))
    }
}

#[async_trait::async_trait]
impl TenantResolver for SessionIdTenantResolver {
    async fn resolve(&self, metadata: &MetadataMap) -> Result<TenantScope, Status> {
        let tenant = read_session_header(metadata).and_then(|sid| self.store.get(&sid));
        Ok(match tenant {
            Some(t) => TenantScope::Tenant(t),
            None => TenantScope::Global,
        })
    }
}

/// Extract the `jammi-session-id` header from request metadata.
fn read_session_header(metadata: &MetadataMap) -> Option<SessionId> {
    metadata
        .get(SESSION_HEADER)
        .and_then(|v: &MetadataValue<_>| v.to_str().ok())
        .map(SessionId::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use tonic::Request;

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
    async fn session_id_resolver_is_global_when_no_header() {
        let resolver = SessionIdTenantResolver::new(SessionStore::new());
        let scope = resolver.resolve(Request::new(()).metadata()).await.unwrap();
        assert_eq!(
            scope,
            TenantScope::Global,
            "no session header is the explicit Global scope, not an error"
        );
    }

    #[tokio::test]
    async fn session_id_resolver_binds_tenant_when_header_present() {
        let store = SessionStore::new();
        store.set(SessionId::new("conn-1"), Some(t_a()));
        let resolver = SessionIdTenantResolver::new(store);
        let mut req = Request::new(());
        req.metadata_mut()
            .insert(SESSION_HEADER, "conn-1".parse().unwrap());
        let scope = resolver.resolve(req.metadata()).await.unwrap();
        assert_eq!(scope, TenantScope::Tenant(t_a()));
    }

    #[tokio::test]
    async fn session_id_resolver_is_global_when_session_unbound() {
        // A header present but with no binding in the store resolves to Global,
        // not an error — the OSS-cooperative default never rejects.
        let resolver = SessionIdTenantResolver::new(SessionStore::new());
        let mut req = Request::new(());
        req.metadata_mut()
            .insert(SESSION_HEADER, "unknown".parse().unwrap());
        let scope = resolver.resolve(req.metadata()).await.unwrap();
        assert_eq!(scope, TenantScope::Global);
    }
}
