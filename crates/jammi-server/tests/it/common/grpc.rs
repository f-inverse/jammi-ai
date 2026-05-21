//! gRPC test helpers shared between `grpc_session`, `grpc_trigger`, and
//! `flight_tenant`. Each of those files previously carried its own copy of
//! the `with_session` interceptor closure, the `channel(addr)` constructor,
//! and the two well-known tenant UUIDs we use as test fixtures — three
//! near-identical copies that violated CLAUDE.md §DRY. Centralising them
//! here keeps the three test surfaces in lockstep and gives new tests one
//! obvious place to plug into.

use std::net::SocketAddr;
use std::str::FromStr;

use jammi_engine::TenantId;
use jammi_server::grpc::session::SESSION_HEADER;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::Request;

/// Well-known tenant UUIDs used as fixtures across the gRPC integration
/// tests. These are generic UUIDs not coupled to any downstream tenant
/// (jammi is the substrate; accurisk/lace/etc. live in product crates).
pub const TENANT_A: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
pub const TENANT_B: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b";

/// Parse [`TENANT_A`] into a typed [`TenantId`]. Panics on programmer
/// error (the constant is a valid UUID by construction).
pub fn tenant_a() -> TenantId {
    TenantId::from_str(TENANT_A).expect("TENANT_A is a valid UUID")
}

/// Parse [`TENANT_B`] into a typed [`TenantId`].
pub fn tenant_b() -> TenantId {
    TenantId::from_str(TENANT_B).expect("TENANT_B is a valid UUID")
}

/// Build an HTTP/2 channel to an in-process Tonic server on `addr`. Used by
/// every gRPC test that needs to attach a client — the address is supplied
/// by the per-test fixture (typically backed by a `TcpListener` bound to
/// `127.0.0.1:0`).
pub async fn channel(addr: SocketAddr) -> Channel {
    Channel::from_shared(format!("http://{addr}"))
        .expect("channel uri")
        .connect()
        .await
        .expect("channel connect")
}

/// Build a request-extending interceptor closure that injects the
/// `jammi-session-id` header on every outgoing request. This is the test
/// counterpart to [`jammi_server::grpc::session::TenantInterceptor`]: the
/// server reads the header and binds the tenant; the test passes the same
/// session id on every call so the binding is observable.
pub fn with_session(
    session_id: &str,
) -> impl Fn(Request<()>) -> Result<Request<()>, tonic::Status> + Clone {
    let id: MetadataValue<_> = session_id.parse().expect("session-id ascii");
    move |mut req: Request<()>| {
        req.metadata_mut().insert(SESSION_HEADER, id.clone());
        Ok(req)
    }
}
