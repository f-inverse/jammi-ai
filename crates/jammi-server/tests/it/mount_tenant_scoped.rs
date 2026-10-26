//! `AssembledChain::mount_tenant_scoped` — a downstream service opted into the
//! engine's single tenant-binding layer, wired beside the plain
//! [`AssembledChain::mount`] (un-gated) counterpart.
//!
//! `jammi_test_utils::echo`'s `EchoService` is the "downstream mounts its own
//! service" fixture, implemented here to echo the `SessionTenant` extension it
//! observed (or its absence) into `EchoReply.tenant`, so a call can
//! distinguish "wrapped, `Tenant`", "wrapped, `Global`", and "not wrapped at
//! all" (`mount`).
//!
//! A single generic resolver ([`ProbeResolver`]) drives all three outcomes off
//! one metadata header, so one fixture proves: (i) `mount_tenant_scoped` binds
//! `SessionTenant` from the resolver (`Tenant` → `Some`, `Global` → `None`);
//! (ii) `mount` stays un-gated — the same service observes no `SessionTenant`
//! at all; (iii) a rejecting resolver stops the wrapped service's handler from
//! running at all, while the plain-`mount`ed service still runs; (iv) the
//! mounted ledger records the wrapped service under its `NAME` — the wrap does
//! not drop the route.

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

use jammi_db::session::JammiSession;
use jammi_db::TenantId;
use jammi_server::grpc::session::{SessionStore, SessionTenant, TenantResolver, TenantScope};
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{assemble_grpc_chain, GrpcChain};
use jammi_server::tiers::TierSet;
use jammi_test_utils::echo::echo_service_client::EchoServiceClient;
use jammi_test_utils::echo::echo_service_server::{EchoService, EchoServiceServer};
use jammi_test_utils::echo::{EchoReply, EchoRequest};
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

/// Header a probe client attaches; absent means "no credential offered".
const PROBE_HEADER: &str = "x-probe-cred";
/// The one credential value [`ProbeResolver`] recognizes as authorizing a
/// tenant. Any other non-empty value is rejected.
const PROBE_CRED_TENANT: &str = "known-tenant";

/// The tenant [`ProbeResolver`] binds for [`PROBE_CRED_TENANT`].
fn probe_tenant() -> TenantId {
    TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9c").expect("valid uuid")
}

/// A generic resolver mapping ONE metadata header to all three
/// [`TenantResolver`] outcomes, so one fixture exercises Tenant / Global /
/// rejected without juggling multiple resolver types: absent header →
/// [`TenantScope::Global`] (the explicit unscoped choice); [`PROBE_CRED_TENANT`]
/// → [`TenantScope::Tenant`]; any other value → `Err(UNAUTHENTICATED)`.
struct ProbeResolver;

#[tonic::async_trait]
impl TenantResolver for ProbeResolver {
    async fn resolve(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<TenantScope, Status> {
        match metadata.get(PROBE_HEADER).and_then(|v| v.to_str().ok()) {
            None => Ok(TenantScope::Global),
            Some(PROBE_CRED_TENANT) => Ok(TenantScope::Tenant(probe_tenant())),
            Some(_) => Err(Status::unauthenticated("unknown probe credential")),
        }
    }
}

/// The downstream service: echoes the `SessionTenant` extension it observed
/// into `tenant` — absent (never bound, a plain [`mount`]) → `"unbound"`;
/// bound `SessionTenant(None)` (resolver said `Global`) → `"global"`; bound
/// `SessionTenant(Some(t))` → `"tenant:<t>"`.
///
/// [`mount`]: jammi_server::runtime::AssembledChain::mount
#[derive(Clone, Default)]
struct TenantEcho;

#[tonic::async_trait]
impl EchoService for TenantEcho {
    async fn echo(&self, request: Request<EchoRequest>) -> Result<Response<EchoReply>, Status> {
        let tenant = match request.extensions().get::<SessionTenant>() {
            None => "unbound".to_string(),
            Some(SessionTenant(None)) => "global".to_string(),
            Some(SessionTenant(Some(t))) => format!("tenant:{t}"),
        };
        Ok(Response::new(EchoReply {
            payload: request.into_inner().payload,
            authorization: String::new(),
            session_id: String::new(),
            tenant,
        }))
    }
}

/// Build a transport-only [`GrpcChain`] (no engine — this seam needs only the
/// always-present `CatalogService` core to prove the tenant-binding layer)
/// driven by `resolver`. Returns the chain plus the guards that keep the
/// backing session alive.
async fn probe_chain(addr: SocketAddr, resolver: Arc<dyn TenantResolver>) -> (GrpcChain, TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = Arc::new(
        JammiSession::new(test_config(dir.path()))
            .await
            .expect("session"),
    );
    let flight_ctx = session.context().clone();
    let flight_binding = session.tenant_binding_arc();
    let chain = GrpcChain {
        addr,
        flight_ctx,
        flight_binding,
        store: SessionStore::new(),
        trigger: None,
        engine: None,
        tiers: TierSet::resolve(std::iter::empty()),
        metrics: Arc::new(MetricsRegistry::new().unwrap()),
        tenant_resolver: resolver,
        admin_authorizer: None,
        limits: jammi_db::config::LimitsConfig::default(),
    };
    (chain, dir)
}

/// Call `EchoService.Echo` against `addr`, attaching [`PROBE_HEADER`] when
/// `cred` is `Some`.
async fn probe_echo(addr: SocketAddr, cred: Option<&'static str>) -> Result<EchoReply, Status> {
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    let header: Option<tonic::metadata::MetadataValue<_>> =
        cred.map(|c| c.parse().expect("cred is ascii"));
    let attach = move |mut req: Request<()>| {
        if let Some(h) = header.clone() {
            req.metadata_mut().insert(PROBE_HEADER, h);
        }
        Ok(req)
    };
    let mut client = EchoServiceClient::with_interceptor(channel, attach);
    Ok(client
        .echo(EchoRequest {
            payload: "probe".into(),
        })
        .await?
        .into_inner())
}

/// `mount_tenant_scoped` binds `SessionTenant` from the resolver exactly like
/// an engine service does — `Tenant` → `Some`, `Global` → `None` — driven
/// through a real request over the wire, and the mounted ledger still records
/// the wrapped service under its `NAME` (the wrap does not drop the route).
#[tokio::test]
async fn mount_tenant_scoped_binds_session_tenant_from_the_resolver() {
    let (chain, _dir) = probe_chain(
        super::common::grpc::ephemeral_addr(),
        Arc::new(ProbeResolver),
    )
    .await;

    let assembled = assemble_grpc_chain(chain)
        .expect("assemble")
        .mount_tenant_scoped(EchoServiceServer::new(TenantEcho));

    assert!(
        assembled
            .mounted()
            .iter()
            .any(|m| m == "jammi.test.echo.EchoService"),
        "mount_tenant_scoped must record the wrapped service under its NAME: {:?}",
        assembled.mounted()
    );

    let bound = assembled.bind().await.expect("bind");
    let addr = bound.addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        bound
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve");
    });

    // No credential: the resolver's EXPLICIT Global scope — the wrapper still
    // binds `SessionTenant(None)`, distinct from never having bound at all.
    let global = probe_echo(addr, None).await.expect("global echo");
    assert_eq!(
        global.tenant, "global",
        "a resolved Global scope must bind SessionTenant(None), not leave it unbound"
    );

    // A known credential: `SessionTenant(Some(tenant))`.
    let tenant = probe_echo(addr, Some(PROBE_CRED_TENANT))
        .await
        .expect("tenant echo");
    assert_eq!(
        tenant.tenant,
        format!("tenant:{}", probe_tenant()),
        "a resolved Tenant scope must bind SessionTenant(Some(tenant))"
    );

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}

/// Rejection path: a `mount_tenant_scoped` service's handler is NOT called
/// when the resolver rejects (reject-before-inner) — the same contract engine
/// services get. The SAME service mounted via plain `mount` on a separate
/// listener stays un-gated: it still runs on the identical rejecting
/// credential and observes no `SessionTenant` at all, proving `mount` is
/// unaffected by opting a *different* mount into the resolver.
#[tokio::test]
async fn mount_tenant_scoped_rejects_before_inner_while_plain_mount_stays_ungated() {
    // The `mount_tenant_scoped` listener: a credential the resolver does not
    // recognize must reject before the handler runs.
    let (chain_scoped, _dir_scoped) = probe_chain(
        super::common::grpc::ephemeral_addr(),
        Arc::new(ProbeResolver),
    )
    .await;
    let scoped = assemble_grpc_chain(chain_scoped)
        .expect("assemble")
        .mount_tenant_scoped(EchoServiceServer::new(TenantEcho));
    let bound_scoped = scoped.bind().await.expect("bind");
    let addr_scoped = bound_scoped.addr();
    let (tx_scoped, rx_scoped) = oneshot::channel::<()>();
    let h_scoped = tokio::spawn(async move {
        bound_scoped
            .serve_with_shutdown(async move {
                let _ = rx_scoped.await;
            })
            .await
            .expect("serve");
    });

    let err = probe_echo(addr_scoped, Some("bogus-credential"))
        .await
        .expect_err("a resolver-rejected credential must reject before the handler runs");
    assert_eq!(
        err.code(),
        tonic::Code::Unauthenticated,
        "the rejection must surface the resolver's status, got {err:?}"
    );

    let _ = tx_scoped.send(());
    let _ = h_scoped.await;

    // The plain-`mount` listener: the IDENTICAL rejecting credential must NOT
    // stop the handler — `mount` is un-gated by the resolver entirely.
    let (chain_plain, _dir_plain) = probe_chain(
        super::common::grpc::ephemeral_addr(),
        Arc::new(ProbeResolver),
    )
    .await;
    let plain = assemble_grpc_chain(chain_plain)
        .expect("assemble")
        .mount(EchoServiceServer::new(TenantEcho));
    let bound_plain = plain.bind().await.expect("bind");
    let addr_plain = bound_plain.addr();
    let (tx_plain, rx_plain) = oneshot::channel::<()>();
    let h_plain = tokio::spawn(async move {
        bound_plain
            .serve_with_shutdown(async move {
                let _ = rx_plain.await;
            })
            .await
            .expect("serve");
    });

    let ok = probe_echo(addr_plain, Some("bogus-credential"))
        .await
        .expect("a plain-mounted service must still run on a resolver-rejecting credential");
    assert_eq!(
        ok.tenant, "unbound",
        "a plain-mounted service must never observe a SessionTenant extension"
    );

    let _ = tx_plain.send(());
    let _ = h_plain.await;
}
