mod audit;
mod convert;
mod database;
mod ephemeral;
mod error;
mod job;
pub mod model_task;

/// Drive `future` to completion on `runtime` with the GIL released — the one
/// way a Python verb waits on the engine. The engine's work (a query, an
/// embedding plan, a request to a remote model) runs while every other Python
/// thread in the process keeps running, including one the engine itself is
/// waiting on, such as an in-process HTTP endpoint.
pub(crate) fn released<F>(runtime: &tokio::runtime::Runtime, future: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    pyo3::Python::attach(|py| py.detach(|| runtime.block_on(future)))
}

use std::io::IsTerminal;

use pyo3::prelude::*;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use jammi_db::config::JammiConfig;

use crate::error::to_pyerr;
use crate::job::PyJob;
use crate::model_task::PyModelTask;

/// The embedded engine's log level when neither `[logging] level` nor
/// `RUST_LOG` names one: a library inside the caller's process reports what
/// needs their attention and is otherwise quiet.
const LOG_DEFAULT: tracing::level_filters::LevelFilter = tracing::level_filters::LevelFilter::WARN;

/// The `_NativeDatabase` pyclass: the low-level embedded engine handle. The
/// user-facing surface is the thin Python wrapper (`jammi._embedded.EmbeddedBackend`)
/// that holds one of these. Re-exported so native Rust consumers (such as
/// downstream crates that layer their own bindings on top of this one) can
/// hold and drive the same instance the Python interpreter sees, and call
/// [`PyDatabase::session_arc`] to share its underlying session.
pub use crate::database::PyDatabase;

/// Re-export of the underlying inference session type. External Rust
/// consumers need to name this to receive the `Arc<InferenceSession>`
/// returned by [`PyDatabase::session_arc`] and share schema-upgrade lock,
/// trigger broker, catalog cache, and tenant binding with the OSS
/// `jammi.EmbeddedBackend`.
pub use jammi_ai::session::InferenceSession;

/// Module entry point for the top-level `jammi_native` extension module.
///
/// `jammi_native` exposes only the LOCAL, in-process engine. There is no remote
/// transport here: the embed wheel links no tonic/proto, and its remote arm is
/// the bundled pure-Python `jammi`, dispatched in `jammi/__init__.py`.
/// This is the runtime shape of the Rust build's `#[cfg(feature = "local")]`
/// gate — the wheel cannot even name a remote transport.
#[pymodule]
fn jammi_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_local, m)?)?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<crate::database::PyTenantScope>()?;
    m.add_class::<PyJob>()?;
    m.add_class::<PyModelTask>()?;
    m.add_class::<crate::audit::PyPerQueryAudit>()?;
    m.add_class::<crate::audit::PyAuditHandle>()?;
    m.add_class::<crate::ephemeral::PyEphemeralSession>()?;
    Ok(())
}

/// Open the embedded, in-process engine and return a [`PyDatabase`].
///
/// The local arm of the unified `connect(target)` — `jammi.connect` calls
/// this for a `file://` target and delegates a remote target to the pure-Python
/// `jammi` remote client. All parameters are optional keyword-only arguments that
/// override the default or file-based configuration.
///
/// # One configuration surface, both deployments
///
/// The `JammiConfig` is built by [`JammiConfig::load`] — the SAME call
/// `jammi-server`'s `main` makes, with the same precedence: the explicit
/// `config` path when given, else `JAMMI_CONFIG`, else `./jammi.toml`, else the
/// platform config dir, and then the `JAMMI_*` environment overrides layered on
/// top (and the load-time validation of the training timing that comes with
/// it). An embedded process therefore answers every deployment key — notably
/// `worker.enabled` (`JAMMI_WORKER__ENABLED`), which decides whether
/// THIS process runs the training claim loop — exactly the way a server process
/// does. A key only one arm honoured would be a server-only feature, i.e. a
/// deployment-shaped divergence rather than a setting.
///
/// The explicit kwargs are applied AFTER the load and still win, so the
/// `artifact_dir` `jammi.connect("file://…")` derives from the target is never
/// overridden by a config file or by `JAMMI_ARTIFACT_DIR`.
#[pyfunction]
#[pyo3(signature = (*, config=None, artifact_dir=None, gpu_device=None, inference_batch_size=None))]
fn open_local(
    config: Option<String>,
    artifact_dir: Option<String>,
    gpu_device: Option<i32>,
    inference_batch_size: Option<usize>,
) -> PyResult<PyDatabase> {
    // One call, whether or not a path was given (`load` falls back to
    // `JAMMI_CONFIG` / `./jammi.toml` / the platform config dir and then applies
    // the `JAMMI_*` overrides) — the server's own resolution, not a private
    // embedded variant that would silently ignore the operator's environment.
    let mut cfg =
        JammiConfig::load(config.as_deref().map(std::path::Path::new)).map_err(to_pyerr)?;

    if let Some(dir) = artifact_dir {
        cfg.artifact_dir = dir.into();
    }
    if let Some(dev) = gpu_device {
        cfg.gpu.device = Some(dev);
    }
    if let Some(bs) = inference_batch_size {
        cfg.inference.batch_size = bs;
    }

    // Build the runtime `PyDatabase::open_with_runtime` will drive the
    // session on — BEFORE installing tracing, not after: the OTLP export
    // layer's tonic `Channel` (built inside `jammi_ai::telemetry::layers`) needs an
    // active Tokio reactor merely to construct, and needs to keep living
    // for as long as the connection does, so it must be built on the SAME
    // runtime the session itself will run on, entered here, not a
    // throwaway one that would be dropped before the channel is ever used.
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| to_pyerr(jammi_db::error::JammiError::from(e)))?;
    let runtime = std::sync::Arc::new(runtime);

    // Install a stderr tracing subscriber (+ the OTLP export layer per
    // `[observability]`), filtered by `RUST_LOG`, else `[logging] level`,
    // else `LOG_DEFAULT`. An endpoint this build cannot honour surfaces as
    // a Python exception on every connect, since the layers are built (and
    // the refusal checked) before any install is attempted. A subscriber
    // already installed — an earlier connect()'s, or the host
    // application's own — is kept.
    let layers = {
        let _enter = runtime.enter();
        jammi_ai::telemetry::layers(
            &cfg,
            LOG_DEFAULT,
            std::io::stderr,
            std::io::stderr().is_terminal(),
        )
        .map_err(to_pyerr)?
    };
    if let Err(installed) = tracing_subscriber::registry().with(layers).try_init() {
        tracing::debug!(%installed, "keeping the tracing subscriber already installed");
    }

    PyDatabase::open_with_runtime(cfg, runtime).map_err(to_pyerr)
}
