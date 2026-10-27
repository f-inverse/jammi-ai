//! The async tenant-binding tower layer — the SINGLE binder for every engine
//! service, gRPC and Flight SQL alike.
//!
//! [`crate::runtime::assemble_grpc_chain`] wraps each engine `*ServiceServer`
//! — the Flight SQL service among them — with this layer, driven by the one
//! [`TenantResolver`] the [`crate::runtime::GrpcChain`] carries (the engine
//! default [`SessionIdTenantResolver`](crate::grpc::session::SessionIdTenantResolver)
//! or a downstream's authenticating resolver). There is no separate
//! interceptor path. Per request the wrapped service:
//!
//! 1. `await`s [`TenantResolver::resolve`] against the request's metadata;
//! 2. on `Ok(scope)` maps the [`TenantScope`] onto the [`SessionTenant`]
//!    extension every handler reads ([`TenantScope::Tenant`] → `Some`,
//!    [`TenantScope::Global`] → `None`) and calls the inner service inside
//!    that scope: the handler runs with the scope installed as the engine's
//!    task-local tenant override, and so does every poll of the response body
//!    it returns ([`TenantScopedBody`]). A streaming response — a Flight SQL
//!    statement's rows — is produced after the handler returned, by whichever
//!    task polls the body, so the scope must travel with the body: a statement
//!    whose work happens while its rows stream (an `INSERT`'s sink) runs as the
//!    request's tenant. The scope is the request's own — never a value
//!    concurrent requests share — and a `Global` request runs under an explicit
//!    unscoped scope, never none;
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

use http_body::{Body, Frame, SizeHint};
use jammi_db::tenant::TenantContext;
use jammi_db::tenant_scope::TenantBinding;
use pin_project::pin_project;
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
    type Response = Response<TenantScopedBody<ResBody>>;
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
                    // Run the handler, and every poll of the body it returns,
                    // inside the request's own scope.
                    let scope = TenantContext::from_option(tenant);
                    let response = TenantBinding::scope_with(scope, inner.call(req)).await?;
                    Ok(response.map(|body| TenantScopedBody::new(scope, body)))
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

/// A response body every frame of which is produced inside its request's
/// tenant scope — the [`TenantResolved`] response body.
///
/// A handler returns a streaming body before its frames exist; whichever
/// task polls the body afterwards produces them, outside the scope the
/// handler ran in. Each poll re-installs the request's scope as the engine's
/// task-local tenant override, so the work a stream does while it is polled
/// runs as the request's tenant.
#[pin_project]
pub struct TenantScopedBody<B> {
    tenant: TenantContext,
    #[pin]
    inner: B,
}

impl<B> TenantScopedBody<B> {
    fn new(tenant: TenantContext, inner: B) -> Self {
        Self { tenant, inner }
    }
}

/// The early-refusal body (a resolver's rejection, [`tonic::Status::into_http`])
/// carries no frames, so the scope it would poll under is immaterial.
impl<B: Default> Default for TenantScopedBody<B> {
    fn default() -> Self {
        Self::new(TenantContext::Unscoped, B::default())
    }
}

impl<B: Body> Body for TenantScopedBody<B> {
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        let inner = this.inner;
        TenantBinding::sync_scope_with(*this.tenant, || inner.poll_frame(cx))
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::str::FromStr;
    use std::sync::Mutex;

    use bytes::Bytes;
    use jammi_db::TenantId;
    use tonic::Status;
    use tower::ServiceExt;

    use super::*;

    /// A resolver that resolves every request to one fixed scope.
    struct Fixed(TenantScope);

    #[async_trait::async_trait]
    impl TenantResolver for Fixed {
        async fn resolve(&self, _metadata: &MetadataMap) -> Result<TenantScope, Status> {
            Ok(self.0)
        }
    }

    /// What a probe saw of the engine's task-local tenant override.
    type Seen = Arc<Mutex<Option<Option<TenantContext>>>>;

    /// A one-frame body that records the tenant override its frame is
    /// produced under.
    #[derive(Default)]
    struct ProbeBody {
        seen: Seen,
        done: bool,
    }

    impl Body for ProbeBody {
        type Data = Bytes;
        type Error = Infallible;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
            if self.done {
                return Poll::Ready(None);
            }
            *self.seen.lock().unwrap() = Some(TenantBinding::scoped_context());
            self.done = true;
            Poll::Ready(Some(Ok(Frame::data(Bytes::new()))))
        }
    }

    /// The handler, and every poll of the body it returns, run inside the
    /// request's resolved scope — the body even when it is polled on a task
    /// of its own, where nothing of the handler's scope is left. A `Global`
    /// request runs under an explicit unscoped override, never none.
    #[tokio::test]
    async fn the_request_scope_reaches_the_handler_and_every_poll_of_its_body() {
        let tenant = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
        for (resolved, expected) in [
            (TenantScope::Tenant(tenant), TenantContext::Scoped(tenant)),
            (TenantScope::Global, TenantContext::Unscoped),
        ] {
            let (in_handler, in_body) = (Seen::default(), Seen::default());
            let (handler_seen, body_seen) = (Arc::clone(&in_handler), Arc::clone(&in_body));
            let inner = tower::service_fn(move |_req: Request<()>| {
                let (handler_seen, body_seen) = (Arc::clone(&handler_seen), Arc::clone(&body_seen));
                async move {
                    *handler_seen.lock().unwrap() = Some(TenantBinding::scoped_context());
                    Ok::<_, Infallible>(Response::new(ProbeBody {
                        seen: body_seen,
                        done: false,
                    }))
                }
            });
            let mut service = TenantResolverLayer::new(Arc::new(Fixed(resolved))).layer(inner);
            let response = service
                .ready()
                .await
                .unwrap()
                .call(Request::new(()))
                .await
                .unwrap();
            let mut body = response.into_body();
            tokio::spawn(async move {
                while std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
                    .await
                    .is_some()
                {}
            })
            .await
            .unwrap();

            assert_eq!(*in_handler.lock().unwrap(), Some(Some(expected)));
            assert_eq!(*in_body.lock().unwrap(), Some(Some(expected)));
        }
    }
}
