mod convert;
mod database;
mod error;
mod job;
mod search;

use std::sync::Arc;

use pyo3::prelude::*;
use tracing_subscriber::EnvFilter;

use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;

use crate::database::PyDatabase;
use crate::error::to_pyerr;
use crate::job::PyFineTuneJob;
use crate::search::PySearchBuilder;

/// Module entry point for `jammi._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PySearchBuilder>()?;
    m.add_class::<PyFineTuneJob>()?;
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
        .unwrap_or_else(|_| EnvFilter::new("jammi_ai=info,jammi_engine=info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tokio init: {e}")))?;
    let rt = Arc::new(runtime);

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

    let session = rt.block_on(InferenceSession::new(cfg)).map_err(to_pyerr)?;
    Ok(PyDatabase {
        session: Arc::new(session),
        runtime: rt,
    })
}
