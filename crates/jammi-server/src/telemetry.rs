//! Tracing initialisation shared by every server entry-point.
//!
//! Both the standalone `jammi-server` binary and the `jammi serve`
//! subcommand install the *same* global subscriber so that a server's
//! logs behave identically however it is launched. Logging is wired to
//! the resolved [`LoggingConfig`](jammi_db::config::LoggingConfig): the
//! level/filter comes from the config (with `RUST_LOG` as an optional
//! override) and the formatter is JSON or text per `logging.format`.
//!
//! Output always goes to stdout regardless of whether stdout is a
//! terminal — a server runs non-interactively (containers, redirected
//! output, systemd) by design, so records must never be gated on a TTY.
//! Only ANSI colouring is TTY-aware: colour codes are emitted solely
//! when stdout is a terminal, keeping redirected log files clean.
//!
//! # OTLP (#486)
//!
//! The global subscriber is a [`tracing_subscriber::Registry`] layered with
//! the `fmt` formatter above, PLUS — when `[observability] otlp_endpoint`
//! is configured — [`jammi_ai::telemetry::otlp_layer`]'s export layer. A
//! configured endpoint this build cannot honour (compiled without
//! `jammi-ai`'s `telemetry-otlp` feature) is a typed startup refusal (K2),
//! checked BEFORE anything else in [`init_tracing`] — never a silently
//! dropped span. The tracer-provider handle the exporter depends on is kept
//! alive for the process in `OTLP_PROVIDER_HANDLE` (private, below): dropping
//! it would tear down the batch processor's background thread and stop every
//! future span from ever being sent.

use std::io::{self, IsTerminal};

use jammi_db::config::{JammiConfig, LogFormat};
use jammi_db::error::Result;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};

/// Keeps the OTLP tracer-provider's background flush thread and gRPC
/// channel alive for the process, once [`init_tracing`] configures one.
/// `OnceLock` rather than an owned return value from [`init_tracing`]
/// because both server entry points (the standalone binary and `jammi
/// serve`) call it the same way, before either constructs anything that
/// would otherwise hold it.
#[cfg(feature = "telemetry-otlp")]
static OTLP_PROVIDER_HANDLE: std::sync::OnceLock<jammi_ai::telemetry::OtlpProviderHandle> =
    std::sync::OnceLock::new();

/// Install the global tracing subscriber from the engine config.
///
/// Honours `RUST_LOG` when set; otherwise falls back to the config's
/// `logging.level`. Emits JSON when `logging.format = "json"`, otherwise
/// a human-readable text layer. Writes to stdout unconditionally; ANSI
/// colour is enabled only when stdout is a terminal. Also installs the
/// OTLP export layer per `[observability]` — see the module docs.
///
/// Returns a typed [`jammi_db::error::JammiError::Config`] when
/// `otlp_endpoint` is configured but this build cannot honour it (K2), or
/// when the endpoint/header configuration the exporter itself rejects is
/// malformed. The caller (`main.rs`) treats this exactly like a config-load
/// failure: print to stderr and exit before anything else starts, since
/// tracing is not initialised yet either way.
pub fn init_tracing(config: &JammiConfig) -> Result<()> {
    install(config, io::stdout, io::stdout().is_terminal())
}

/// Build and install the global subscriber over an arbitrary writer.
///
/// Factored out from [`init_tracing`] so the writer and TTY decision are
/// injectable: the binaries pass stdout, and the regression tests pass an
/// in-memory buffer with `ansi = false` to assert that records are emitted
/// to a non-terminal sink.
fn install<W>(config: &JammiConfig, writer: W, ansi: bool) -> Result<()>
where
    W: for<'w> MakeWriter<'w> + Send + Sync + 'static,
{
    // Refuse a configuration this build cannot honour BEFORE building
    // anything else — present in every build of `jammi-ai`, feature or not.
    jammi_ai::telemetry::refuse_if_endpoint_without_feature(&config.observability)?;

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.logging.level.clone()));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(writer)
        .with_ansi(ansi);
    let fmt_layer: Box<dyn Layer<Registry> + Send + Sync> = match config.logging.format {
        LogFormat::Json => Box::new(fmt_layer.json().with_filter(filter)),
        LogFormat::Text => Box::new(fmt_layer.with_filter(filter)),
    };

    // A `Vec<Box<dyn Layer<Registry>>>` is itself one `Layer<Registry>` —
    // tracing_subscriber's blanket impl applies each element to the SAME
    // base `Registry` as siblings, exactly like two sequential `.with()`
    // calls would, but without each successive box needing to satisfy
    // `Layer<Layered<PriorBox, Registry>>` (a DIFFERENT, ever-growing type
    // per layer added, which a `Box<dyn Layer<Registry>>` cannot name).
    // `mut` is only exercised (`.push`ed into) under `telemetry-otlp` — a
    // `--no-default-features` build never pushes a second layer, so the
    // binding would otherwise warn as unused in that one configuration.
    #[cfg_attr(not(feature = "telemetry-otlp"), allow(unused_mut))]
    let mut layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = vec![fmt_layer];

    #[cfg(feature = "telemetry-otlp")]
    if let Some(otlp) = jammi_ai::telemetry::otlp_layer(&config.observability)? {
        // Keep the tracer-provider handle alive for the process BEFORE
        // `.layer` moves into `layers` below — see `OTLP_PROVIDER_HANDLE`'s
        // docs. `set` returning `Err` (a second `init_tracing` call in the
        // same process, e.g. a test harness) is not this function's
        // problem to report; the SAME handle from the first call keeps the
        // exporter alive regardless.
        let _ = OTLP_PROVIDER_HANDLE.set(otlp.provider_handle());
        layers.push(Box::new(otlp.layer));
    }

    tracing_subscriber::registry().with(layers).init();

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};

    use jammi_db::config::{JammiConfig, LogFormat};
    use tracing::subscriber::DefaultGuard;
    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    /// A `MakeWriter` that captures everything written into a shared
    /// buffer, standing in for a redirected (non-TTY) log file.
    #[derive(Clone)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'w> MakeWriter<'w> for BufferWriter {
        type Writer = BufferWriter;
        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    /// Build a subscriber over the buffer with the same layering as
    /// [`install`] but scoped to the current thread, so the assertion
    /// does not race with the process-global subscriber.
    fn capture(format: LogFormat, emit: impl FnOnce()) -> String {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = BufferWriter(buffer.clone());

        let mut config = JammiConfig::default();
        config.logging.level = "info".into();
        config.logging.format = format;

        let filter = EnvFilter::new(config.logging.level.clone());
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(writer)
            // A redirected sink is never a terminal.
            .with_ansi(false);
        let fmt_layer: Box<dyn Layer<Registry> + Send + Sync> = match config.logging.format {
            LogFormat::Json => Box::new(fmt_layer.json().with_filter(filter)),
            LogFormat::Text => Box::new(fmt_layer.with_filter(filter)),
        };

        let _guard: DefaultGuard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(fmt_layer));

        emit();

        let bytes = buffer.lock().unwrap().clone();
        String::from_utf8(bytes).expect("log output is valid UTF-8")
    }

    #[test]
    fn emits_records_to_a_non_tty_sink_in_text_format() {
        let output = capture(LogFormat::Text, || {
            tracing::info!(answer = 42, "device selected");
        });

        assert!(
            !output.is_empty(),
            "text logging produced no output to a non-TTY sink"
        );
        assert!(output.contains("device selected"));
        assert!(output.contains("answer"));
    }

    #[test]
    fn emits_records_to_a_non_tty_sink_in_json_format() {
        let output = capture(LogFormat::Json, || {
            tracing::info!(answer = 42, "device selected");
        });

        assert!(
            !output.is_empty(),
            "json logging produced no output to a non-TTY sink"
        );
        assert!(output.contains("\"message\":\"device selected\""));
        assert!(output.contains("\"answer\":42"));
        assert!(output.contains("\"level\":\"INFO\""));
    }

    /// K2: a configured `otlp_endpoint` this build cannot honour (compiled
    /// without `telemetry-otlp`) is a typed refusal naming the feature —
    /// never a silently dropped span. Runs only under a build WITHOUT the
    /// feature; the default (feature-on) build has something real to
    /// build instead (proven in `jammi-ai`'s own `telemetry` tests).
    #[test]
    #[cfg(not(feature = "telemetry-otlp"))]
    fn refuses_a_configured_endpoint_without_the_feature() {
        let mut config = JammiConfig::default();
        config.observability.otlp_endpoint = Some("http://localhost:4317".to_string());
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = BufferWriter(buffer);

        let err = install(&config, writer, false)
            .expect_err("a configured endpoint without the feature must be refused");
        assert!(err.to_string().contains("telemetry-otlp"));
    }
}
