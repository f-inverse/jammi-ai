//! Tracing initialisation shared by every server entry-point.
//!
//! Both the standalone `jammi-server` binary and the `jammi serve`
//! subcommand install the *same* global subscriber so that a server's
//! logs behave identically however it is launched. Logging is wired to
//! the resolved [`LoggingConfig`](jammi_db::config::LoggingConfig) through
//! [`jammi_ai::telemetry::fmt_layer`]: the filter comes from the config (with
//! `RUST_LOG` as an optional override, and `info` when neither is set) and
//! the formatter is JSON or text per `logging.format`.
//!
//! Output always goes to stdout regardless of whether stdout is a
//! terminal — a server runs non-interactively (containers, redirected
//! output, systemd) by design, so records must never be gated on a TTY.
//! Only ANSI colouring is TTY-aware: colour codes are emitted solely
//! when stdout is a terminal, keeping redirected log files clean.
//!
//! # OTLP
//!
//! The global subscriber is a [`tracing_subscriber::Registry`] layered with
//! [`jammi_ai::telemetry::layers`]: the `fmt` formatter above, PLUS — when
//! `[observability] otlp_endpoint` is configured — the OTLP export layer. A
//! configured endpoint this build cannot honour (compiled without
//! `jammi-ai`'s `telemetry-otlp` feature) is a typed startup refusal,
//! checked BEFORE anything else — never a silently dropped span. The
//! library keeps the exporter's tracer provider alive for the process, and
//! both shutdown arms stop it with [`jammi_ai::telemetry::flush_otlp`].

use std::io::{self, IsTerminal};

use jammi_db::config::JammiConfig;
use jammi_db::error::Result;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// A server's log level when neither `[logging] level` nor `RUST_LOG` names
/// one: a daemon logs its operations.
const LOG_DEFAULT: LevelFilter = LevelFilter::INFO;

/// Install the global tracing subscriber from the engine config.
///
/// Honours `RUST_LOG` when set; otherwise the config's `logging.level`, else
/// `info`. Emits JSON when `logging.format = "json"`, otherwise
/// a human-readable text layer. Writes to stdout unconditionally; ANSI
/// colour is enabled only when stdout is a terminal. Also installs the
/// OTLP export layer per `[observability]` — see the module docs.
///
/// Returns a typed [`jammi_db::error::JammiError::Config`] when
/// `otlp_endpoint` is configured but this build cannot honour it, or
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
    let layers = jammi_ai::telemetry::layers(config, LOG_DEFAULT, writer, ansi)?;
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
        config.logging.format = format;
        // A redirected sink is never a terminal.
        let fmt_layer = jammi_ai::telemetry::fmt_layer(&config.logging, LOG_DEFAULT, writer, false);

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

    /// A configured `otlp_endpoint` this build cannot honour (compiled
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
