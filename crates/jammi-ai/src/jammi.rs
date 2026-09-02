//! The SDK front door: [`Jammi::open`] turns a [`Target`] into a [`Session`].
//!
//! A consumer that wants "use the embedded SDK" makes one call. The [`Target`]
//! it passes carries the engine config, and the returned [`Session`] is the
//! in-process consumer surface — every verb on it runs against the embedded
//! engine the config built.
//!
//! This is purely a constructor over existing pieces: it threads
//! [`InferenceSession::open`] → [`Session::with_configured_worker`]. No
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
    /// config and wraps it as an embedded [`Session`].
    ///
    /// # `training.run_worker` — whether this process claims
    ///
    /// The front-door embedded session owns the training worker (RAII) **when
    /// the config asks this process to claim**. That is one key,
    /// [`jammi_db::config::TrainingConfig::run_worker`] (default `true`), read
    /// off the `JammiConfig` the caller passed in — the SAME key the server
    /// binary's `train` tier and the Python `Database` read, so a deployment
    /// setting `JAMMI_TRAINING__RUN_WORKER=false` (or the `[training]` TOML key)
    /// reaches the Rust SDK arm exactly as it reaches the other two:
    ///
    /// * `true` (the default) — the session both submits training jobs and runs
    ///   them; the worker stops when the session drops.
    /// * `false` — no worker is spawned. The session still submits training jobs
    ///   and serves their status; it just never claims one, so on a
    ///   single-process catalog a submitted job stays `queued` until a process
    ///   configured to claim opens the directory.
    ///
    /// See [`Session::with_configured_worker`] for the constructor this threads
    /// to, and [`Session::with_embedded_worker`] for the explicit
    /// spawn-regardless form.
    pub async fn open(target: Target) -> Result<Session> {
        match target {
            Target::Local(config) => {
                let engine = InferenceSession::open(config).await?;
                Session::with_configured_worker(engine)
            }
        }
    }
}
