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

use std::io::{self, IsTerminal};

use jammi_db::config::{JammiConfig, LogFormat};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

/// Install the global tracing subscriber from the engine config.
///
/// Honours `RUST_LOG` when set; otherwise falls back to the config's
/// `logging.level`. Emits JSON when `logging.format = "json"`, otherwise
/// a human-readable text layer. Writes to stdout unconditionally; ANSI
/// colour is enabled only when stdout is a terminal.
pub fn init_tracing(config: &JammiConfig) {
    install(config, io::stdout, io::stdout().is_terminal());
}

/// Build and install the global subscriber over an arbitrary writer.
///
/// Factored out from [`init_tracing`] so the writer and TTY decision are
/// injectable: the binaries pass stdout, and the regression test passes
/// an in-memory buffer with `ansi = false` to assert that records are
/// emitted to a non-terminal sink.
fn install<W>(config: &JammiConfig, writer: W, ansi: bool)
where
    W: for<'w> MakeWriter<'w> + Send + Sync + 'static,
{
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.logging.level.clone()));

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(writer)
        .with_ansi(ansi);

    match config.logging.format {
        LogFormat::Json => builder.json().init(),
        LogFormat::Text => builder.init(),
    }
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
        let builder = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            // A redirected sink is never a terminal.
            .with_ansi(false);

        let _guard: DefaultGuard = match config.logging.format {
            LogFormat::Json => tracing::subscriber::set_default(builder.json().finish()),
            LogFormat::Text => tracing::subscriber::set_default(builder.finish()),
        };

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
}
