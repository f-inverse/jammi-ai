mod audit;
mod convert;
mod database;
mod ephemeral;
mod error;
mod job;
pub mod model_task;
mod remote;
mod search;

use pyo3::prelude::*;
use tracing_subscriber::EnvFilter;

use jammi_db::config::JammiConfig;

use crate::error::to_pyerr;
use crate::job::PyFineTuneJob;
use crate::model_task::PyModelTask;
use crate::search::PySearchBuilder;

/// The `Database` pyclass. Re-exported so native Rust consumers (such as
/// downstream crates that layer their own bindings on top of this one) can
/// hold and drive the same instance the Python interpreter sees, and call
/// [`PyDatabase::session_arc`] to share its underlying session.
pub use crate::database::PyDatabase;

/// The remote-transport `RemoteDatabase` pyclass and its `connect_remote`
/// constructor. Re-exported so the Rust integration test (and any downstream
/// Rust consumer layering its own bindings) can drive a remote Python session
/// against an in-process gRPC server.
pub use crate::remote::{connect_remote, PyRemoteDatabase};

/// Re-export of the underlying inference session type. External Rust
/// consumers need to name this to receive the `Arc<InferenceSession>`
/// returned by [`PyDatabase::session_arc`] and share schema-upgrade lock,
/// trigger broker, catalog cache, and tenant binding with the OSS
/// `jammi_ai.Database`.
pub use jammi_ai::session::InferenceSession;

/// Module entry point for `jammi_ai._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_function(wrap_pyfunction!(connect_remote, m)?)?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyRemoteDatabase>()?;
    m.add_class::<PySearchBuilder>()?;
    m.add_class::<PyFineTuneJob>()?;
    m.add_class::<PyModelTask>()?;
    m.add_class::<crate::audit::PyPerQueryAudit>()?;
    m.add_class::<crate::audit::PyAuditHandle>()?;
    m.add_class::<crate::ephemeral::PyEphemeralSession>()?;
    Ok(())
}

/// Create a connected Database instance.
///
/// All parameters are optional keyword-only arguments that override
/// the default or file-based configuration.
#[pyfunction]
#[pyo3(signature = (*, config=None, artifact_dir=None, gpu_device=None, inference_batch_size=None))]
fn connect(
    config: Option<String>,
    artifact_dir: Option<String>,
    gpu_device: Option<i32>,
    inference_batch_size: Option<usize>,
) -> PyResult<PyDatabase> {
    // Install a stderr tracing subscriber the first time connect() is called.
    // Reads RUST_LOG; falls back to showing INFO from jammi crates only.
    // try_init() is a no-op if a subscriber was already installed.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("jammi_ai=info,jammi_db=info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();

    let mut cfg = match config {
        Some(path) => JammiConfig::load(Some(std::path::Path::new(&path))).map_err(to_pyerr)?,
        None => JammiConfig::default(),
    };

    if let Some(dir) = artifact_dir {
        cfg.artifact_dir = dir.into();
    }
    if let Some(dev) = gpu_device {
        cfg.gpu.device = dev;
    }
    if let Some(bs) = inference_batch_size {
        cfg.inference.batch_size = bs;
    }

    PyDatabase::open(cfg).map_err(to_pyerr)
}
