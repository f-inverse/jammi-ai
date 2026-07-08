//! The async tenant-binding tower layer — the SINGLE binder for every engine
//! gRPC service.
//!
//! [`crate::runtime::assemble_grpc_chain`] wraps each engine `*ServiceServer`
//! with this layer, driven by the one [`TenantResolver`] the
//! [`crate::runtime::GrpcChain`] carries (the engine default
//! [`SessionIdTenantResolver`](crate::grpc::session::SessionIdTenantResolver) or
//! a downstream's authenticating resolver). There is no separate interceptor
//! path. Per request the wrapped service:
//!
//! 1. `await`s [`TenantResolver::resolve`] against the request's metadata;
//! 2. on `Ok(scope)` maps the [`TenantScope`] onto the [`SessionTenant`]
//!    extension every handler reads ([`TenantScope::Tenant`] → `Some`,
//!    [`TenantScope::Global`] → `None`) and calls the inner service — so
//!    `session_tenant()` / `with_tenant_scoped` downstream are unchanged;
//! 3. on `Err(status)` returns that `Status` as the response WITHOUT calling the
//!    inner service — the handler never runs, so an authenticating resolver's
//!    rejection cannot fall through to an unscoped read.
//!
//! [`TenantResolved`] forwards [`tonic::server::NamedService`] from the inner
//! service, so tonic routing keys it under the inner service's proto path — a
//! wrapper that dropped `NAME` would silently disappear from the router.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tonic::codegen::http::{Request, Response};
use tonic::codegen::Service;
use tonic::metadata::MetadataMap;
use tower::Layer;

use crate::grpc::session::{SessionTenant, TenantResolver, TenantScope};

/// [`Layer`] that installs [`TenantResolved`] over one engine `*ServiceServer`.
/// Holds the resolver as `Arc<dyn TenantResolver>` so it is cheap to clone onto
/// every wrapped service.
#[derive(Clone)]
pub struct TenantResolverLayer {
    resolver: Arc<dyn TenantResolver>,
}

impl TenantResolverLayer {
    pub fn new(resolver: Arc<dyn TenantResolver>) -> Self {
        Self { resolver }
    }
}

impl<S> Layer<S> for TenantResolverLayer {
    type Service = TenantResolved<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TenantResolved {
            inner,
            resolver: Arc::clone(&self.resolver),
        }
    }
}

/// Service that resolves + binds the request's tenant before delegating to the
/// wrapped engine service. See the module docs for the per-request contract.
#[derive(Clone)]
pub struct TenantResolved<S> {
    inner: S,
    resolver: Arc<dyn TenantResolver>,
}

/// Forward the inner service's proto path so tonic routes requests to the
/// wrapped service — MUST-FIX 5: without this the wrapped service drops off the
/// router.
impl<S> tonic::server::NamedService for TenantResolved<S>
where
    S: tonic::server::NamedService,
{
    const NAME: &'static str = S::NAME;
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for TenantResolved<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Default,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<ReqBody>) -> Self::Future {
        let resolver = Arc::clone(&self.resolver);
        // Readiness-preserving clone: the spawned future drives the clone we just
        // polled ready (`inner`); `self.inner` keeps the fresh clone for the next
        // `poll_ready`. Swapping (rather than cloning into the future) avoids
        // calling a not-yet-ready service — the tower discipline for a boxed,
        // `async move` future.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        Box::pin(async move {
            let metadata = MetadataMap::from_headers(req.headers().clone());
            match resolver.resolve(&metadata).await {
                Ok(scope) => {
                    // Map the resolved scope onto the extension every handler
                    // reads: an explicit `Global` binds `None` (unscoped), a
                    // `Tenant` binds `Some(t)`.
                    let tenant = match scope {
                        TenantScope::Tenant(t) => Some(t),
                        TenantScope::Global => None,
                    };
                    req.extensions_mut().insert(SessionTenant(tenant));
                    inner.call(req).await
                }
                // Reject WITHOUT calling the inner service: the handler never
                // runs, so an authenticating resolver's rejection cannot fall
                // through to an unscoped read. The gRPC `Status` becomes the
                // error response body directly.
                Err(status) => Ok(status.into_http()),
            }
        })
    }
}
