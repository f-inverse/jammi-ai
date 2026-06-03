//! The SDK front door: [`Jammi::open`] turns a [`Target`] into a [`Session`].
//!
//! A consumer that wants "use the SDK, run any shape" makes one call. The
//! [`Target`] it passes selects the transport, and the returned [`Session`] is
//! the same transport-agnostic surface either way — so the choice of embedded
//! vs remote is made once, at open time, and never leaks into the call site of
//! any verb.
//!
//! Why one `open(Target)` rather than two `connect_local` / `connect_remote`
//! constructors: the transport split already lives in the [`Session`] enum, and
//! [`Target`] is its constructor-side mirror — one closed choice dispatched by a
//! `match`, the same abstraction at both layers rather than a second parallel
//! surface. It also makes the invalid state unrepresentable for free: the remote
//! arm is `#[cfg(feature = "wire")]`, so a default / embedded build cannot even
//! *name* a remote target — there is no dead arm, no `unreachable!`, no runtime
//! "remote not compiled in" error. A `wire` build gains the `Remote` variant and
//! the `match` arm together.
//!
//! This is purely a constructor over existing pieces: the local arm threads
//! [`InferenceSession::new`] → [`LocalSession::new`] → [`Session::Local`]; the
//! remote arm (under `wire`) is `RemoteSession::connect` → `Session::Remote`. No
//! construction logic is duplicated here.

#[cfg(feature = "local")]
use std::sync::Arc;

#[cfg(feature = "local")]
use jammi_db::config::JammiConfig;
use jammi_db::error::Result;

#[cfg(feature = "local")]
use crate::local_session::LocalSession;
use crate::local_session::Session;
#[cfg(feature = "local")]
use crate::session::InferenceSession;

/// Where a [`Session`] should run. The transport is chosen once, here, when the
/// caller opens the SDK; every verb on the resulting [`Session`] is then
/// transport-agnostic.
pub enum Target {
    /// An embedded, in-process engine built from this [`JammiConfig`]. Present
    /// only under the default-on `local` feature; a thin remote-only (`wire`-only)
    /// build cannot name an embedded target — there is no dead arm, no
    /// `unreachable!`, just the one `Remote` constructor.
    #[cfg(feature = "local")]
    Local(JammiConfig),
    /// A remote engine reached over the `jammi.v1` gRPC wire at this endpoint.
    /// Present only under the `wire` feature; a default / embedded build has no
    /// remote transport and so cannot name this variant.
    #[cfg(feature = "wire")]
    Remote(tonic::transport::Endpoint),
}

/// The SDK front door. [`Jammi::open`] is the single ergonomic entry point that
/// opens a [`Session`] against a [`Target`], selecting the embedded or remote
/// transport. It carries no state of its own — it is the constructor seam the
/// language binding (PyO3) will later expose as the SDK's `open`.
pub struct Jammi;

impl Jammi {
    /// Open a [`Session`] against `target`, selecting its transport.
    ///
    /// * [`Target::Local`] builds an in-process [`InferenceSession`] from the
    ///   config and wraps it as [`Session::Local`].
    /// * `Target::Remote` connects a `RemoteSession` to the endpoint and wraps
    ///   it as `Session::Remote` (only under the `wire` feature).
    pub async fn open(target: Target) -> Result<Session> {
        match target {
            #[cfg(feature = "local")]
            Target::Local(config) => {
                let engine = Arc::new(InferenceSession::new(config).await?);
                Ok(Session::Local(LocalSession::new(engine)))
            }
            #[cfg(feature = "wire")]
            Target::Remote(endpoint) => {
                let remote = crate::remote_session::RemoteSession::connect(endpoint).await?;
                Ok(Session::Remote(remote))
            }
        }
    }
}
