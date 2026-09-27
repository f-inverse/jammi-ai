//! The log formatter under the process-global subscriber a host installs.
//!
//! Its own binary because the behaviour lives in the GLOBAL dispatcher's
//! callsite-interest cache: with one global subscriber and no scoped ones, an
//! enabled callsite's interest is `always` and its event skips the
//! dispatcher's `enabled()` query — the path a scoped test subscriber never
//! takes. `init()` succeeds once per process, so this binary holds one test.

use std::io;
use std::sync::{Arc, Mutex};

use jammi_ai::telemetry::fmt_layer;
use jammi_db::config::LoggingConfig;
use tracing::callsite::Callsite as _;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Layer, Registry};

/// A redirected log file: everything the formatter writes, in one buffer.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl io::Write for Captured {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Captured {
    type Writer = Captured;
    fn make_writer(&'a self) -> Captured {
        self.clone()
    }
}

/// A caller that asks the dispatcher whether a record the filter rejects is
/// enabled and then emits nothing — the `log` bridge's `Log::enabled`, which
/// sqlx's statement logger queries on every statement — never costs the next
/// event on that thread its line.
#[test]
fn an_enabled_query_that_emits_nothing_never_drops_the_next_event() {
    let captured = Captured::default();
    let logging = LoggingConfig {
        level: Some("info".into()),
        ..LoggingConfig::default()
    };
    let sink = captured.clone();
    // The host's composition: `jammi-server`'s `telemetry::install` and
    // `jammi-python`'s `build_tracing_layers` both hand the registry a
    // `Vec<Box<dyn Layer<Registry>>>` holding this formatter.
    let layers: Vec<Box<dyn Layer<Registry> + Send + Sync>> = vec![fmt_layer(
        &logging,
        LevelFilter::INFO,
        move || sink.clone(),
        false,
    )];
    tracing_subscriber::registry().with(layers).init();

    let rejected = tracing::callsite! {
        name: "a dependency's statement log",
        kind: tracing::metadata::Kind::EVENT,
        target: "a_dependency::query",
        level: tracing::Level::DEBUG,
        fields: []
    };
    for round in 0..3 {
        tracing::dispatcher::get_default(|dispatch| dispatch.enabled(rejected.metadata()));
        tracing::info!(round, "the line after the query");
    }

    let log = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
    assert_eq!(
        log.matches("the line after the query").count(),
        3,
        "every event after an enabled() query that emitted nothing is written:\n{log}"
    );
}
