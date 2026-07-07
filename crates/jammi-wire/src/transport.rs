//! The shared session transport: one gRPC [`Channel`] plus the
//! [`SessionHeader`] interceptor every typed client stamps.
//!
//! A [`SessionTransport`] owns the tonic channel and the opaque per-session id
//! the server keys its tenant binding against. Both the control-plane client
//! (`jammi-admin`) and the data-plane client (`jammi-client`) build their
//! per-service stubs over the *same* transport, so a tenant bound through
//! `CatalogService.SetTenant` on one is observed by every verb on the other —
//! the bind-then-query single-session invariant the strict client relies on.

use tonic::codegen::InterceptedService;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use jammi_db::error::{JammiError, Result};

/// Header name carrying the opaque session identifier. Clients mint a
/// per-connection id and include it on every request that needs tenant-scoped
/// semantics; the server's session store keys the tenant binding by it. One
/// definition shared by both sides so the header name cannot drift.
pub const SESSION_HEADER: &str = "jammi-session-id";

/// The standard `authorization` header an authenticated client stamps its
/// bearer under (via [`SessionTransport::with_bearer`]). Named once so the
/// header cannot drift.
pub const AUTHORIZATION_HEADER: &str = "authorization";

/// Injects the [`SESSION_HEADER`] carrying this client's opaque session id on
/// every outbound request, so the server's tenant interceptor resolves the same
/// binding the tenant trio set. One per [`SessionTransport`]; cheap to clone (a
/// pre-parsed metadata value).
///
/// When the transport was built with [`SessionTransport::with_bearer`], it also
/// carries a pre-parsed `authorization: Bearer <token>` value and stamps it
/// alongside the session id. The two headers are orthogonal — both are stamped
/// with no precedence — and the bearer is opaque: it is carried, never
/// interpreted.
#[derive(Clone)]
pub struct SessionHeader {
    id: MetadataValue<tonic::metadata::Ascii>,
    /// Pre-parsed `authorization` value (`Bearer <token>`), stamped on every
    /// request when present. `None` for an unauthenticated transport.
    bearer: Option<MetadataValue<tonic::metadata::Ascii>>,
}

impl Interceptor for SessionHeader {
    fn call(&mut self, mut request: Request<()>) -> std::result::Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert(SESSION_HEADER, self.id.clone());
        if let Some(bearer) = &self.bearer {
            request
                .metadata_mut()
                .insert(AUTHORIZATION_HEADER, bearer.clone());
        }
        Ok(request)
    }
}

/// A connected gRPC transport scoped to one client session.
///
/// Cheap to clone: the [`Channel`] is a reference-counted multiplexed
/// connection and the [`SessionHeader`] wraps a pre-parsed metadata value, so a
/// per-service stub is built by cloning the channel + header rather than
/// dialling again.
#[derive(Clone)]
pub struct SessionTransport {
    channel: Channel,
    header: SessionHeader,
    /// The opaque session id this client minted, kept so [`Self::session_id`]
    /// can report the key the server binds tenant state against; the
    /// interceptor carries the parsed copy.
    session_id: String,
}

impl SessionTransport {
    /// Connect to a `jammi.v1` gRPC endpoint and mint a fresh session id.
    ///
    /// `endpoint` is any value an [`Endpoint`] accepts (e.g. an
    /// `"http://host:port"` string). The transport is native tonic here; the
    /// session id is a v4 UUID so two clients against one server never collide
    /// on a tenant binding.
    pub async fn connect(endpoint: impl Into<Endpoint>) -> Result<Self> {
        let channel = endpoint
            .into()
            .connect()
            .await
            .map_err(|e| JammiError::Config(format!("connect to jammi endpoint: {e}")))?;
        let session_id = uuid::Uuid::new_v4().to_string();
        let id: MetadataValue<tonic::metadata::Ascii> = session_id
            .parse()
            .map_err(|e| JammiError::Config(format!("session id metadata: {e}")))?;
        Ok(Self {
            channel,
            header: SessionHeader { id, bearer: None },
            session_id,
        })
    }

    /// Return this transport with an `authorization: Bearer <bearer>` header
    /// stamped on every request, alongside the session id.
    ///
    /// The bearer is an opaque credential the transport carries and never
    /// interprets — no signature check, no expiry math. Any authenticated
    /// client (a CLI, a downstream service, a test fixture) reuses this; the
    /// wire substrate names no consumer and encodes no auth policy.
    ///
    /// Fallible exactly like [`Self::connect`]: the bearer must parse as an
    /// ASCII gRPC metadata value, and a value that does not surfaces as
    /// [`JammiError::Config`] — never a panic, never a silently dropped header.
    pub fn with_bearer(mut self, bearer: impl AsRef<str>) -> Result<Self> {
        let value: MetadataValue<tonic::metadata::Ascii> = format!("Bearer {}", bearer.as_ref())
            .parse()
            .map_err(|e| JammiError::Config(format!("bearer metadata: {e}")))?;
        self.header.bearer = Some(value);
        Ok(self)
    }

    /// The opaque session id this client minted. The server keys the tenant
    /// binding by it; two transports over the same channel are isolated because
    /// each mints its own.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The underlying multiplexed gRPC channel. Used by the Flight SQL client,
    /// which opens its own service over the same connection.
    pub fn channel(&self) -> Channel {
        self.channel.clone()
    }

    /// Build a per-service stub over this transport's channel + session header.
    /// The `make` closure is a generated `with_interceptor` constructor; the
    /// channel and header are cloned (cheap) so every service shares one
    /// connection and one session id.
    pub fn service<S>(&self, make: impl FnOnce(Channel, SessionHeader) -> S) -> S {
        make(self.channel.clone(), self.header.clone())
    }
}

/// The intercepted-channel type every generated `with_interceptor` stub takes.
/// Named once here so the client crates name one type, not the full tonic
/// generic each time.
pub type SessionChannel = InterceptedService<Channel, SessionHeader>;
