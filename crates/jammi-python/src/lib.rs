mod audit;
mod convert;
mod database;
mod ephemeral;
mod error;
mod job;
pub mod model_task;

use pyo3::prelude::*;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};

use jammi_db::config::JammiConfig;
use jammi_db::error::Result as JammiResult;

use crate::error::to_pyerr;
use crate::job::PyTrainingJob;
use crate::model_task::PyModelTask;

/// Keeps the OTLP tracer-provider's background flush thread and gRPC
/// channel alive for the process, once [`build_tracing_layers`] configures
/// one — mirrors `jammi-server`'s `telemetry::OTLP_PROVIDER_HANDLE`.
/// `open_local` may run many times in one process (a script that calls
/// `jammi.connect` more than once); only the FIRST call's handle is kept
/// (`OnceLock::set` on a later call is a no-op) — the exporter it drives
/// stays alive regardless, and every `otlp_layer` built afterwards from an
/// identically-configured endpoint would behave the same either way.
static OTLP_PROVIDER_HANDLE: std::sync::OnceLock<jammi_ai::telemetry::OtlpProviderHandle> =
    std::sync::OnceLock::new();

/// Build the tracing layers `open_local` installs: the `fmt` formatter (to
/// stderr, filtered by `RUST_LOG` or a `jammi_ai=info,jammi_db=info`
/// default) plus — when `[observability] otlp_endpoint` is configured — the
/// OTLP export layer (#486), via the SAME `jammi_ai::telemetry::otlp_layer`
/// factory `jammi-server`'s `telemetry::install` uses (B4).
///
/// Factored out of `open_local` (private -- a `#[pyfunction]`, not a `pub`
/// Rust item) so a test can build the identical
/// composition over a scoped subscriber
/// (`tracing::subscriber::set_default`) rather than the process-global
/// `try_init()`, which only ever succeeds once per process and so cannot be
/// exercised repeatably from a test. `pub` (rather than `pub(crate)`)
/// specifically so `tests/it.rs` — an external integration test, per this
/// crate's `rlib` crate-type — can drive it directly.
pub fn build_tracing_layers(
    config: &JammiConfig,
) -> JammiResult<Vec<Box<dyn Layer<Registry> + Send + Sync>>> {
    // K2: refuse a configured endpoint this build cannot honour, before
    // building anything else.
    jammi_ai::telemetry::refuse_if_endpoint_without_feature(&config.observability)?;

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("jammi_ai=info,jammi_db=info"));
    let fmt_layer: Box<dyn Layer<Registry> + Send + Sync> = Box::new(
        tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_filter(filter),
    );
    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = vec![fmt_layer];

    if let Some(otlp) = jammi_ai::telemetry::otlp_layer(&config.observability)? {
        let _ = OTLP_PROVIDER_HANDLE.set(otlp.provider_handle());
        layers.push(Box::new(otlp.layer));
    }

    Ok(layers)
}

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
    m.add_class::<PyTrainingJob>()?;
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
        cfg.gpu.device = dev;
    }
    if let Some(bs) = inference_batch_size {
        cfg.inference.batch_size = bs;
    }

    // Build the runtime `PyDatabase::open_with_runtime` will drive the
    // session on — BEFORE installing tracing, not after: the OTLP export
    // layer's tonic `Channel` (built inside `build_tracing_layers`) needs an
    // active Tokio reactor merely to construct, and needs to keep living
    // for as long as the connection does, so it must be built on the SAME
    // runtime the session itself will run on, entered here, not a
    // throwaway one that would be dropped before the channel is ever used.
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| to_pyerr(jammi_db::error::JammiError::from(e)))?;
    let runtime = std::sync::Arc::new(runtime);

    // Install a stderr tracing subscriber (+ the OTLP export layer per
    // `[observability]`, #486) the first time connect() is called. Reads
    // RUST_LOG; falls back to showing INFO from jammi crates only.
    // try_init() is a no-op if a subscriber was already installed — an
    // endpoint misconfiguration this build cannot honour (K2) still
    // surfaces as a Python exception either way, since
    // `build_tracing_layers` runs its refusal check before `try_init`.
    let layers = {
        let _enter = runtime.enter();
        build_tracing_layers(&cfg).map_err(to_pyerr)?
    };
    let _ = tracing_subscriber::registry().with(layers).try_init();

    PyDatabase::open_with_runtime(cfg, runtime).map_err(to_pyerr)
}
