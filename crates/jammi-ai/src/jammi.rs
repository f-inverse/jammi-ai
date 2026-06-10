//! The SDK front door: [`Jammi::open`] turns a [`Target`] into a [`Session`].
//!
//! A consumer that wants "use the embedded SDK" makes one call. The [`Target`]
//! it passes carries the engine config, and the returned [`Session`] is the
//! in-process consumer surface — every verb on it runs against the embedded
//! engine the config built.
//!
//! This is purely a constructor over existing pieces: it threads
//! [`InferenceSession::open`] → [`Session::with_embedded_worker`]. No
//! construction logic is duplicated here. The remote transport is a separate
//! crate (`jammi-client`'s `DataClient` / `jammi-admin`'s `CatalogClient`),
//! which speaks the same request/result vocabulary over gRPC; an embedded
//! consumer reaches it there, not through this front door.

use jammi_db::config::JammiConfig;
use jammi_db::error::Result;

use crate::local_session::Session;
use crate::session::InferenceSession;

/// Where an embedded [`Session`] should run: an in-process engine built from
/// this [`JammiConfig`]. The transport is chosen once, here, when the caller
/// opens the SDK.
pub enum Target {
    /// An embedded, in-process engine built from this [`JammiConfig`].
    Local(JammiConfig),
}

/// The SDK front door. [`Jammi::open`] is the single ergonomic entry point that
/// opens a [`Session`] against a [`Target`]. It carries no state of its own — it
/// is the constructor seam the language binding (PyO3) exposes as the SDK's
/// `open`.
pub struct Jammi;

impl Jammi {
    /// Open a [`Session`] against `target`.
    ///
    /// [`Target::Local`] builds an in-process [`InferenceSession`] from the
    /// config and wraps it as an embedded [`Session`]. The front-door embedded
    /// session owns the training worker (RAII): it both submits training jobs
    /// and runs them, and the worker stops when the session drops.
    pub async fn open(target: Target) -> Result<Session> {
        match target {
            Target::Local(config) => {
                let engine = InferenceSession::open(config).await?;
                Session::with_embedded_worker(engine)
            }
        }
    }
}
