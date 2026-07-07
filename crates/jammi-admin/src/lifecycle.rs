//! The lifecycle/auth contract client.
//!
//! [`LifecycleClient`] wraps the generated `LifecycleServiceClient` over a
//! shared [`jammi_wire::SessionTransport`] and exposes the four operator/CLI
//! boundary verbs — `apply_license`, `bootstrap`, `status`, `login`. It mirrors
//! [`CatalogClient`](crate::CatalogClient)'s shape (connect / over the shared
//! transport) and is candle-free: it speaks the typed `jammi.v1.lifecycle` wire
//! only and pulls no implementation.
//!
//! The OSS engine implements NONE of these verbs — an OSS `jammi-server` answers
//! `UNIMPLEMENTED` for all four. The server that answers them (a platform
//! server) owns their meaning; this client is a thin caller. So each verb
//! surfaces the answering server's [`tonic::Status`] verbatim on failure —
//! preserving its gRPC code (e.g. `UNIMPLEMENTED` against an OSS server) rather
//! than folding it into the engine's `JammiError` taxonomy, which these
//! platform-answered verbs are not part of.

use jammi_wire::proto::lifecycle::lifecycle_service_client::LifecycleServiceClient;
use jammi_wire::proto::lifecycle::{ApplyLicenseRequest, BootstrapRequest, LoginRequest};
use jammi_wire::{SessionChannel, SessionTransport};
use tonic::transport::Endpoint;
use tonic::Status;

/// What `ApplyLicense` returns. `detail` is the human-readable server verdict —
/// a caller prints it on rejection.
#[derive(Debug, Clone)]
pub struct LicenseApplied {
    pub accepted: bool,
    pub detail: String,
}

/// What `Bootstrap` returns: where the dashboard lives plus the one-time
/// first-login credential the operator changes on first login.
#[derive(Debug, Clone)]
pub struct Bootstrapped {
    pub dashboard_url: String,
    pub first_login_credential: String,
}

/// What `Status` returns: the answering server's liveness / licensed /
/// bootstrapped state. `license_state` is a server-defined string.
#[derive(Debug, Clone)]
pub struct PlatformStatus {
    pub bootstrapped: bool,
    pub licensed: bool,
    pub license_state: String,
    pub platform_version: String,
}

/// What `Login` returns: the opaque bearer plus its expiry. `expires_at_micros`
/// of `0` means non-expiring — the issuing server decides expiry policy.
#[derive(Debug, Clone)]
pub struct Bearer {
    pub bearer: String,
    pub expires_at_micros: i64,
}

/// A lifecycle/auth contract client over a shared [`SessionTransport`].
///
/// Cheap to clone: it holds the cloneable transport. Reuses the same transport a
/// [`CatalogClient`](crate::CatalogClient) speaks over, so a CLI that already
/// connected can open lifecycle verbs on the same session id.
#[derive(Clone)]
pub struct LifecycleClient {
    transport: SessionTransport,
}

impl LifecycleClient {
    /// Connect to a `jammi.v1` gRPC endpoint and mint a fresh session id.
    pub async fn connect(
        endpoint: impl Into<Endpoint>,
    ) -> Result<Self, jammi_db::error::JammiError> {
        Ok(Self {
            transport: SessionTransport::connect(endpoint).await?,
        })
    }

    /// Build a lifecycle client over an existing transport, reusing its session
    /// id and any bearer it carries.
    pub fn over(transport: SessionTransport) -> Self {
        Self { transport }
    }

    /// The transport this client speaks over.
    pub fn transport(&self) -> &SessionTransport {
        &self.transport
    }

    fn client(&self) -> LifecycleServiceClient<SessionChannel> {
        self.transport
            .service(LifecycleServiceClient::with_interceptor)
    }

    /// Apply a signed offline entitlement file. The bytes are opaque; the
    /// answering server verifies and decides acceptance.
    pub async fn apply_license(&self, signed: Vec<u8>) -> Result<LicenseApplied, Status> {
        let resp = self
            .client()
            .apply_license(ApplyLicenseRequest {
                signed_entitlement: signed,
            })
            .await?
            .into_inner();
        Ok(LicenseApplied {
            accepted: resp.accepted,
            detail: resp.detail,
        })
    }

    /// Seed the first administrator. An empty `display_name` / `bootstrap_token`
    /// maps to the proto3 default (field absent on the wire) — callers without a
    /// value pass `""`.
    pub async fn bootstrap(
        &self,
        email: &str,
        password: &str,
        display_name: &str,
        bootstrap_token: &str,
    ) -> Result<Bootstrapped, Status> {
        let resp = self
            .client()
            .bootstrap(BootstrapRequest {
                admin_email: email.to_string(),
                admin_password: password.to_string(),
                admin_name: display_name.to_string(),
                bootstrap_token: bootstrap_token.to_string(),
            })
            .await?
            .into_inner();
        Ok(Bootstrapped {
            dashboard_url: resp.dashboard_url,
            first_login_credential: resp.first_login_credential,
        })
    }

    /// Report the answering server's platform state. Against an OSS server this
    /// returns `UNIMPLEMENTED` (the engine holds no platform state).
    pub async fn status(&self) -> Result<PlatformStatus, Status> {
        let resp = self.client().status(()).await?.into_inner();
        Ok(PlatformStatus {
            bootstrapped: resp.bootstrapped,
            licensed: resp.licensed,
            license_state: resp.license_state,
            platform_version: resp.platform_version,
        })
    }

    /// Exchange an email/password for a bearer.
    pub async fn login(&self, email: &str, password: &str) -> Result<Bearer, Status> {
        let resp = self
            .client()
            .login(LoginRequest {
                email: email.to_string(),
                password: password.to_string(),
            })
            .await?
            .into_inner();
        Ok(Bearer {
            bearer: resp.bearer,
            expires_at_micros: resp.expires_at_micros,
        })
    }
}
