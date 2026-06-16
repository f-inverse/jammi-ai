//! Bring-your-own-auth seam — the worked example.
//!
//! Jammi authenticates nothing on its own: a deployment runs the gRPC surface
//! on a trusted network and a consumer in front of it decides *who* a caller is
//! and *which tenant* they may act as. The `jammi-session-id` header the stock
//! [`TenantInterceptor`](jammi_server::grpc::session::TenantInterceptor) reads is
//! a client-minted, opaque transport correlation id — not a credential. Anyone
//! who presents another session's id assumes that session's tenant. It is a
//! convenience for the trusted-network case, never a trust boundary.
//!
//! This file proves the seam a real consumer plugs into: a **custom tonic
//! [`Interceptor`]** that runs *in front of* every engine-backed verb,
//! authenticates the caller's credential, derives the tenant from the *verified*
//! claim, and binds it by inserting the
//! [`SessionTenant`](jammi_server::grpc::session::SessionTenant) extension every
//! handler resolves on. Authentication and tenant authorization therefore sit
//! ahead of session resolution — the principal is established before any tenant
//! is bound, and the bound tenant is the one the credential authorizes, not one
//! the caller asserts in a header.
//!
//! The credential here is a **generic signed bearer token** — an opaque subject
//! plus a tenant claim, HMAC-SHA256 over both under a key only the consumer's
//! issuer and gateway share. It stands in for whatever a consumer's identity
//! system mints (a JWT from an OIDC provider, a session cookie a gateway
//! exchanges, a service-to-service token); the seam does not care, it only needs
//! the interceptor to turn a verified claim into a [`TenantId`]. No identity
//! provider, scheme, or consumer is named — an unrelated consumer (a feature
//! store, an ad-attribution chain) would reach for exactly this shape.
//!
//! What it pins:
//!
//! * **Isolation through the custom interceptor** — two callers presenting valid
//!   tokens for two distinct tenants each see only their own tenant's sources,
//!   end to end through the authenticating interceptor.
//! * **A missing credential is rejected** — no token → `unauthenticated`, the
//!   request never reaches a handler, so it cannot read any tenant's data.
//! * **An invalid credential is rejected** — a tampered / wrong-key token →
//!   `unauthenticated`; a forged tenant claim buys nothing because the signature
//!   covers the claim.
//! * **A rejected caller does not fall through to another tenant** — the
//!   interceptor *fails the request* rather than binding `None`, so there is no
//!   path by which an unauthenticated caller silently reads another tenant's
//!   rows (the defect the framing rule guards against).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use hmac::{Hmac, Mac};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::TenantId;
use jammi_server::grpc::catalog::CatalogServer;
use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::catalog_service_server::CatalogServiceServer;
use jammi_server::grpc::proto::catalog::ListSourcesRequest;
use jammi_server::grpc::session::SessionTenant;
use jammi_test_utils::test_config;
use sha2::Sha256;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Server};
use tonic::{Request, Status};

use super::common::grpc::{tenant_a, tenant_b, TENANT_A, TENANT_B};

type HmacSha256 = Hmac<Sha256>;

// --- the consumer's credential (generic, lives in consumer territory) --------

/// Header the consumer's clients present their bearer token on. A consumer
/// names this; Jammi has no opinion on it — the engine only ever sees the
/// `SessionTenant` extension the interceptor derives from it.
const AUTH_HEADER: &str = "authorization";

/// The shared secret a consumer's token issuer and its gateway hold. In a real
/// deployment this is a rotated signing key, never compiled in; here it is a
/// fixed test value so the fixture can mint tokens the interceptor verifies.
const SIGNING_KEY: &[u8] = b"consumer-issuer-signing-key-not-jammi";

/// Mint a generic signed bearer token: `"<subject>.<tenant>.<hex-mac>"`, where
/// the MAC is HMAC-SHA256 over `"<subject>.<tenant>"` under [`SIGNING_KEY`].
/// This stands in for whatever a consumer's identity system issues; the tenant
/// claim is *inside* the signed payload, so it cannot be forged without the key.
fn mint_token(subject: &str, tenant: &str) -> String {
    let claim = format!("{subject}.{tenant}");
    let mut mac = <HmacSha256 as Mac>::new_from_slice(SIGNING_KEY).expect("key length");
    mac.update(claim.as_bytes());
    let sig = hex::encode(mac.finalize().into_bytes());
    format!("Bearer {claim}.{sig}")
}

/// Verify a bearer token and return its tenant claim, or `None` if the token is
/// missing a part, carries a malformed claim, or fails the signature check. This
/// is the consumer's authentication step: a verified claim is the *only* thing
/// that yields a tenant.
fn verify_token(token: &str) -> Option<TenantId> {
    let raw = token.strip_prefix("Bearer ")?;
    // "<subject>.<tenant>.<mac>"
    let (claim, sig_hex) = raw.rsplit_once('.')?;
    let (subject, tenant) = claim.split_once('.')?;
    if subject.is_empty() {
        return None;
    }

    let expected = hex::decode(sig_hex).ok()?;
    let mut mac = <HmacSha256 as Mac>::new_from_slice(SIGNING_KEY).expect("key length");
    mac.update(claim.as_bytes());
    // Constant-time verification: the MAC type compares, not us.
    mac.verify_slice(&expected).ok()?;

    // Only a verified claim derives a tenant. A malformed tenant in a validly
    // signed token is still a rejection — the issuer never mints those.
    tenant.parse::<TenantId>().ok()
}

/// The consumer's custom tonic interceptor — the BYO-auth plug. It runs ahead
/// of every engine verb: it authenticates the bearer token, derives the tenant
/// from the *verified* claim, and binds it by inserting the `SessionTenant`
/// extension the engine handlers resolve on. A missing or invalid credential
/// fails the request with `unauthenticated` — the call never reaches a handler,
/// so it can never read another tenant's data. The interceptor never trusts the
/// `jammi-session-id` header for identity; it does not read it at all.
#[derive(Clone)]
struct BearerAuthInterceptor;

impl Interceptor for BearerAuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        let token = request
            .metadata()
            .get(AUTH_HEADER)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("missing bearer token"))?;

        let tenant =
            verify_token(token).ok_or_else(|| Status::unauthenticated("invalid bearer token"))?;

        // Bind the *verified* tenant for every downstream handler. This is the
        // same extension the stock interceptor sets from an (unauthenticated)
        // session lookup — here it is set from an authenticated claim instead,
        // which is the whole point of the seam.
        request.extensions_mut().insert(SessionTenant(Some(tenant)));
        Ok(request)
    }
}

// --- the engine behind the seam ----------------------------------------------

struct AuthServer {
    addr: SocketAddr,
    shutdown: oneshot::Sender<()>,
    _dir: TempDir,
    handle: tokio::task::JoinHandle<()>,
}

/// Stand up an in-process `CatalogService` fronted by the consumer's
/// [`BearerAuthInterceptor`] (in place of the stock `TenantInterceptor`). The
/// fixture pre-registers one source per tenant — each under its own tenant scope
/// via the engine's per-request `with_tenant_scoped`, exactly as a tenant-bound
/// `add_source` verb would — so a later `list_sources` returns disjoint results
/// per authenticated caller.
async fn start_auth_server() -> AuthServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let session = Arc::new(InferenceSession::new(cfg).await.expect("session"));

    // Register a tenant-scoped source for each tenant. The binding at
    // registration time stamps the catalog row's `tenant_id`, so `list_sources`
    // filters them apart downstream.
    register_source_for(&session, tenant_a(), "a_source", &dir).await;
    register_source_for(&session, tenant_b(), "b_source", &dir).await;

    let store = jammi_server::grpc::session::SessionStore::new();
    let catalog = CatalogServer::new(
        store,
        jammi_server::tiers::TierSet::resolve([]).expect("core tier resolves"),
        Some(Arc::clone(&session)),
    );
    let catalog_svc = CatalogServiceServer::with_interceptor(catalog, BearerAuthInterceptor);

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        Server::builder()
            .add_service(catalog_svc)
            .serve_with_shutdown(addr, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("grpc server");
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    AuthServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle,
    }
}

/// Register a file source scoped to `tenant`, mirroring what a tenant-bound
/// `add_source` verb does on the engine: the registration runs inside
/// `with_tenant_scoped`, so the catalog row carries `tenant`'s id.
async fn register_source_for(
    session: &Arc<InferenceSession>,
    tenant: TenantId,
    source_id: &str,
    dir: &TempDir,
) {
    let pq = dir.path().join(format!("{source_id}.parquet"));
    write_empty_parquet(&pq);
    let conn = SourceConnection {
        url: Some(format!("file://{}", pq.display())),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    };
    let local = jammi_ai::Session::new(Arc::clone(session));
    let id = source_id.to_string();
    session
        .with_tenant_scoped(tenant, |_scope| {
            local.add_source(&id, SourceType::File, conn)
        })
        .await
        .expect("tenant-scoped add_source");
}

/// Write a one-row, one-column Parquet file so the source registers against a
/// real, readable file (registration validates the path).
fn write_empty_parquet(path: &std::path::Path) {
    use arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef],
    )
    .expect("batch");
    let file = std::fs::File::create(path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

/// A catalog client that presents `token` on every request via a closure
/// interceptor — the data-plane analogue of a consumer's gateway attaching the
/// caller's credential. `None` attaches no token at all.
async fn client_with_token(
    addr: SocketAddr,
    token: Option<String>,
) -> CatalogServiceClient<tonic::service::interceptor::InterceptedService<Channel, impl Interceptor>>
{
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    let header: Option<MetadataValue<_>> = token.map(|t| t.parse().expect("token is ascii"));
    let attach = move |mut req: Request<()>| {
        if let Some(h) = header.clone() {
            req.metadata_mut().insert(AUTH_HEADER, h);
        }
        Ok(req)
    };
    CatalogServiceClient::with_interceptor(channel, attach)
}

/// Pull the source ids a caller can see.
async fn list_source_ids(
    client: &mut CatalogServiceClient<
        tonic::service::interceptor::InterceptedService<Channel, impl Interceptor>,
    >,
) -> Result<Vec<String>, Status> {
    let resp = client.list_sources(ListSourcesRequest {}).await?;
    Ok(resp
        .into_inner()
        .sources
        .into_iter()
        .map(|s| s.source_id)
        .collect())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_authenticated_tenants_see_isolated_sources() {
    let server = start_auth_server().await;

    // Two callers, two valid tokens for two distinct tenants. The tenant is
    // *inside* the signed claim — neither caller asserts it in a plain header.
    let mut client_a =
        client_with_token(server.addr, Some(mint_token("user-alice", TENANT_A))).await;
    let mut client_b = client_with_token(server.addr, Some(mint_token("user-bob", TENANT_B))).await;

    let sources_a = list_source_ids(&mut client_a).await.expect("a lists");
    let sources_b = list_source_ids(&mut client_b).await.expect("b lists");

    assert_eq!(
        sources_a,
        vec!["a_source".to_string()],
        "tenant A's authenticated caller sees only A's source; got {sources_a:?}"
    );
    assert_eq!(
        sources_b,
        vec!["b_source".to_string()],
        "tenant B's authenticated caller sees only B's source; got {sources_b:?}"
    );
    assert!(
        !sources_a.contains(&"b_source".to_string()),
        "tenant A must never see B's source through the seam"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_credential_is_rejected_not_run_unscoped() {
    let server = start_auth_server().await;

    // No token at all. The interceptor fails the request *before* any handler,
    // so the caller reads nothing — it does not fall through to an unscoped read
    // that could surface another tenant's rows.
    let mut anon = client_with_token(server.addr, None).await;
    let err = list_source_ids(&mut anon)
        .await
        .expect_err("a request with no credential must be rejected");
    assert_eq!(
        err.code(),
        tonic::Code::Unauthenticated,
        "a missing credential is Unauthenticated, got {:?}: {}",
        err.code(),
        err.message()
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_credential_is_rejected() {
    let server = start_auth_server().await;

    // A token whose signature does not cover its claim — a forged tenant. The
    // MAC check fails, so the caller is rejected and binds no tenant.
    let forged = format!("Bearer user-mallory.{TENANT_A}.deadbeef");
    let mut mallory = client_with_token(server.addr, Some(forged)).await;
    let err = list_source_ids(&mut mallory)
        .await
        .expect_err("a forged token must be rejected");
    assert_eq!(
        err.code(),
        tonic::Code::Unauthenticated,
        "a forged credential is Unauthenticated, got {:?}: {}",
        err.code(),
        err.message()
    );

    // And it bought nothing: a valid token for the *same* tenant the forgery
    // claimed does resolve, proving the rejection was the signature, not the
    // tenant — the seam authenticates the claim, it does not blocklist a tenant.
    let mut legit = client_with_token(server.addr, Some(mint_token("user-alice", TENANT_A))).await;
    let sources = list_source_ids(&mut legit).await.expect("legit lists");
    assert_eq!(sources, vec!["a_source".to_string()]);

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
