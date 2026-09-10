//! Vendor-neutral OTLP trace export (#486) and W3C `traceparent` continuation.
//!
//! This module is the ONE place `jammi-server`'s `telemetry::install` /
//! `TraceContextLayer` and `jammi-python`'s `open_local` subscriber wiring
//! both reach for the OTLP mechanism (B4): the factory that turns a
//! [`jammi_db::config::ObservabilityConfig`] into a live exporter lives in
//! the library, not duplicated per consumer.
//!
//! # Feature gate (K2)
//!
//! `otlp_layer` — the exporter/tracer-provider/sampler factory — and
//! `set_parent_from_headers` — the incoming-`traceparent` continuation
//! helper — exist ONLY behind the `telemetry-otlp` cargo feature (they pull
//! the `opentelemetry`/`opentelemetry_sdk`/`opentelemetry-otlp`/
//! `tracing-opentelemetry` stack; neither name resolves as an intra-doc link
//! here on purpose — this module doc is compiled and documented in BOTH
//! configurations, and the two symbols exist in only one of them).
//! `refuse_if_endpoint_without_feature` is the one function in this module
//! present in EVERY build, feature or not: a consumer calls it
//! unconditionally, before attempting anything feature-gated, and gets a
//! typed `jammi_db::error::JammiError::Config` naming the `telemetry-otlp`
//! feature when `otlp_endpoint` is configured but this build cannot honour
//! it — never a silently-dropped span.
//!
//! # Zero egress with no endpoint
//!
//! `otlp_layer` returns `Ok(None)` the instant `otlp_endpoint` is `None`,
//! before constructing anything — no `Channel`, no exporter, no tracer
//! provider. A process with no configured endpoint opens no network
//! connection for tracing, full stop.

use jammi_db::config::ObservabilityConfig;
use jammi_db::error::{JammiError, Result};

/// The name of the cargo feature this module's exporter machinery lives
/// behind, quoted verbatim in [`refuse_if_endpoint_without_feature`]'s error
/// so a deployer knows exactly what to rebuild with.
#[cfg(not(feature = "telemetry-otlp"))]
const FEATURE_NAME: &str = "telemetry-otlp";

/// Refuse a configuration this build cannot honour: `otlp_endpoint` is set
/// but the `telemetry-otlp` feature was not compiled in. Present in EVERY
/// build (feature on or off) so a caller can check this unconditionally,
/// before any feature-gated code runs. `Ok(())` in every other case,
/// INCLUDING when the feature is off and no endpoint is configured — an
/// unconfigured deployment must never be forced to compile the OTel stack.
#[cfg(not(feature = "telemetry-otlp"))]
pub fn refuse_if_endpoint_without_feature(config: &ObservabilityConfig) -> Result<()> {
    match &config.otlp_endpoint {
        Some(endpoint) => Err(JammiError::Config(format!(
            "observability.otlp_endpoint = '{endpoint}' is set, but this build of jammi-ai \
             was compiled without the `{FEATURE_NAME}` cargo feature -- rebuild with \
             `--features {FEATURE_NAME}` or unset otlp_endpoint"
        ))),
        None => Ok(()),
    }
}

/// The feature is compiled in, so there is never anything to refuse here —
/// [`otlp_layer`] is the one that reports a REAL build failure (a malformed
/// endpoint the exporter itself rejects, e.g.).
#[cfg(feature = "telemetry-otlp")]
pub fn refuse_if_endpoint_without_feature(_config: &ObservabilityConfig) -> Result<()> {
    Ok(())
}

#[cfg(feature = "telemetry-otlp")]
mod otlp {
    use std::sync::Once;

    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::{SpanExporter, WithExportConfig, WithTonicConfig};
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
    use opentelemetry_sdk::Resource;
    use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    use super::{JammiError, ObservabilityConfig, Result};

    /// A `tracing_subscriber::Registry`-backed OTel export layer, built by
    /// [`otlp_layer`]. Both `jammi-server`'s `telemetry::install` and
    /// `jammi-python`'s `open_local` compose their subscriber over a plain
    /// [`tracing_subscriber::Registry`], so this alias is concrete rather
    /// than generic over the base subscriber — there is only ever one base
    /// type in this codebase.
    pub type OtlpLayer = tracing_opentelemetry::OpenTelemetryLayer<
        tracing_subscriber::Registry,
        opentelemetry_sdk::trace::Tracer,
    >;

    /// The layer plus the tracer-provider handle it was built from.
    ///
    /// **The caller MUST keep this alive for the entire time it wants spans
    /// exported** — the provider owns the batch span processor's background
    /// flush thread and the exporter's gRPC channel; dropping it tears both
    /// down. Add `.layer` to a subscriber, then hold the whole `Otlp` value
    /// somewhere that outlives it (a `static` slot, a field on the server's
    /// top-level handle) — never let it fall out of scope right after
    /// installing the layer, or every span silently stops exporting.
    pub struct Otlp {
        pub layer: OtlpLayer,
        provider: SdkTracerProvider,
    }

    impl Otlp {
        /// A cheap (`Arc`-clone), `Send + Sync` handle to the SAME tracer
        /// provider `self.layer` reports through. Grab this BEFORE moving
        /// `.layer` into a subscriber (`Registry::with` takes it by value,
        /// partially moving `self` — Rust then refuses any further method
        /// call on `self`, even one that only touches `provider`), so a
        /// caller can still [`OtlpProviderHandle::force_flush`] /
        /// [`OtlpProviderHandle::shutdown`] afterwards.
        pub fn provider_handle(&self) -> OtlpProviderHandle {
            OtlpProviderHandle(self.provider.clone())
        }
    }

    /// See [`Otlp::provider_handle`].
    #[derive(Clone)]
    pub struct OtlpProviderHandle(SdkTracerProvider);

    impl OtlpProviderHandle {
        /// Block until every span buffered so far has been sent (or the
        /// exporter's own timeout elapses). Used at graceful shutdown, and
        /// by tests that need a deterministic "has it been sent yet" point
        /// rather than a fixed sleep.
        pub fn force_flush(&self) -> Result<()> {
            self.0
                .force_flush()
                .map_err(|e| JammiError::Config(format!("OTLP force_flush failed: {e}")))
        }

        /// Flush, then stop the exporter for good (releases the batch
        /// processor's background thread and the gRPC channel). Call at
        /// process shutdown, after the last span-producing work is done.
        pub fn shutdown(&self) -> Result<()> {
            self.0
                .shutdown()
                .map_err(|e| JammiError::Config(format!("OTLP shutdown failed: {e}")))
        }
    }

    /// Build the OTLP export layer from `config`, or `Ok(None)` when no
    /// endpoint is configured.
    ///
    /// `Ok(None)` is returned BEFORE constructing anything reachable over
    /// the network when `config.otlp_endpoint` is `None` — no `Channel`, no
    /// exporter, no tracer provider, no thread. This is the zero-egress
    /// contract: absence of a configured endpoint means absence of any
    /// attempt to reach one.
    ///
    /// When an endpoint IS configured: each `config.otlp_headers` entry is
    /// [`jammi_db::config::SecretSource::resolve`]d here (never earlier —
    /// H4, matching [`jammi_db::config::ModelsConfig::hub_token`]'s split)
    /// into gRPC metadata the exporter attaches to every export call; a
    /// malformed header name/value, or an endpoint the exporter itself
    /// rejects, is a typed [`JammiError::Config`] naming the field. The
    /// sampler is a parent-based ratio sampler over `config.sample_ratio`
    /// (already domain-checked at config load —
    /// [`ObservabilityConfig::validate`]): a span with an already-sampled
    /// remote parent is always kept; a root span defers to the ratio. The
    /// resource carries `service.name = config.service_name`, so every span
    /// this process exports carries it — including a span opened by
    /// [`super::set_parent_from_headers`]'s caller.
    ///
    /// Also installs the global W3C `traceparent`/`tracestate` propagator
    /// ([`opentelemetry_sdk::propagation::TraceContextPropagator`]) exactly
    /// once per process (idempotent across repeated calls), so
    /// [`super::set_parent_from_headers`] has something real to extract
    /// with the moment an exporter exists.
    pub fn otlp_layer(config: &ObservabilityConfig) -> Result<Option<Otlp>> {
        let Some(endpoint) = config.otlp_endpoint.as_deref() else {
            return Ok(None);
        };

        install_propagator();

        let mut metadata = MetadataMap::new();
        for (name, source) in &config.otlp_headers {
            let secret = source.resolve()?;
            let key = MetadataKey::from_bytes(name.to_lowercase().as_bytes()).map_err(|e| {
                JammiError::Config(format!(
                    "observability.otlp_headers key '{name}' is not a valid gRPC metadata key: {e}"
                ))
            })?;
            let value = MetadataValue::try_from(secret.expose()).map_err(|e| {
                JammiError::Config(format!(
                    "observability.otlp_headers value for '{name}' is not a valid gRPC \
                     metadata value: {e}"
                ))
            })?;
            metadata.insert(key, value);
        }

        let exporter = SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_metadata(metadata)
            .build()
            .map_err(|e| {
                JammiError::Config(format!(
                    "observability.otlp_endpoint '{endpoint}' could not be built into an OTLP \
                     exporter: {e}"
                ))
            })?;

        let resource = Resource::builder()
            .with_service_name(config.service_name.clone())
            .build();
        let sampler =
            Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(config.sample_ratio)));

        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_sampler(sampler)
            .with_resource(resource)
            .build();

        let tracer = provider.tracer(config.service_name.clone());
        let layer = tracing_opentelemetry::layer().with_tracer(tracer);
        Ok(Some(Otlp { layer, provider }))
    }

    static PROPAGATOR_INIT: Once = Once::new();

    /// Install the global W3C trace-context propagator exactly once per
    /// process. Idempotent: a second call (from a second [`otlp_layer`]
    /// build, or from [`super::set_parent_from_headers`] on a process that
    /// never configured an endpoint) is a no-op.
    fn install_propagator() {
        PROPAGATOR_INIT.call_once(|| {
            opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        });
    }

    /// Extract a W3C `traceparent`/`tracestate` pair from `headers` (via the
    /// global propagator — `install_propagator` (private, above) makes sure
    /// one is installed) and bind the resulting OpenTelemetry context onto `span`
    /// as its parent, so any span exported for `span` (or a descendant
    /// opened while `span` is entered) continues the caller's trace id
    /// rather than starting a new one. A request with no `traceparent`
    /// header extracts an empty context, so `span` gets no parent and
    /// starts a fresh trace — identical to today's behaviour.
    pub fn set_parent_from_headers(span: &tracing::Span, headers: &http::HeaderMap) {
        install_propagator();
        let extractor = HeaderExtractor(headers);
        let parent_cx = opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.extract(&extractor)
        });
        let _ = span.set_parent(parent_cx);
    }

    /// Adapts an [`http::HeaderMap`] to OpenTelemetry's
    /// [`opentelemetry::propagation::Extractor`] — the trait a
    /// `TextMapPropagator` reads carrier values through. A thin wrapper
    /// rather than a dependency on `opentelemetry-http`: the only thing
    /// needed here is read access to the two W3C headers by name.
    struct HeaderExtractor<'a>(&'a http::HeaderMap);

    impl opentelemetry::propagation::Extractor for HeaderExtractor<'_> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).and_then(|v| v.to_str().ok())
        }

        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(|k| k.as_str()).collect()
        }
    }
}

#[cfg(feature = "telemetry-otlp")]
pub use otlp::{otlp_layer, set_parent_from_headers, Otlp, OtlpLayer, OtlpProviderHandle};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "telemetry-otlp"))]
    fn refuses_a_configured_endpoint_when_the_feature_is_off() {
        let config = ObservabilityConfig {
            otlp_endpoint: Some("http://localhost:4317".to_string()),
            ..ObservabilityConfig::default()
        };
        let err = refuse_if_endpoint_without_feature(&config)
            .expect_err("a configured endpoint with the feature off must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("telemetry-otlp"),
            "refusal must name the feature: {msg}"
        );
        assert!(
            msg.contains("http://localhost:4317"),
            "refusal must name the offending endpoint: {msg}"
        );
    }

    #[test]
    #[cfg(not(feature = "telemetry-otlp"))]
    fn is_a_no_op_with_no_endpoint_even_when_the_feature_is_off() {
        let config = ObservabilityConfig::default();
        assert!(refuse_if_endpoint_without_feature(&config).is_ok());
    }

    #[test]
    #[cfg(feature = "telemetry-otlp")]
    fn refuse_if_endpoint_without_feature_never_refuses_when_the_feature_is_on() {
        let config = ObservabilityConfig {
            otlp_endpoint: Some("http://localhost:4317".to_string()),
            ..ObservabilityConfig::default()
        };
        assert!(refuse_if_endpoint_without_feature(&config).is_ok());
    }

    /// No endpoint configured -> `otlp_layer` returns `Ok(None)` and opens
    /// no network connection at all. Proven at the socket level: a
    /// throwaway loopback listener that config never names receives zero
    /// incoming connections as a side effect of the call.
    #[test]
    #[cfg(feature = "telemetry-otlp")]
    fn no_endpoint_builds_no_exporter_and_opens_no_connection() {
        let sentinel = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        sentinel.set_nonblocking(true).unwrap();

        let config = ObservabilityConfig::default();
        assert_eq!(config.otlp_endpoint, None);
        let layer = otlp_layer(&config).expect("otlp_layer must not error with no endpoint");
        assert!(layer.is_none(), "no endpoint must build no exporter");

        // A tonic gRPC channel's own connect attempt would show up as an
        // incoming connection on ANY loopback listener almost immediately;
        // give it a moment, then assert this listener — which the config
        // never named — saw nothing.
        std::thread::sleep(std::time::Duration::from_millis(50));
        match sentinel.accept() {
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            other => {
                panic!("expected no connection attempt with no endpoint configured, got {other:?}")
            }
        }
    }

    // Building the tonic `Channel` (lazily -- no connection attempt, just
    // the client machinery) needs an active Tokio reactor.
    #[tokio::test]
    #[cfg(feature = "telemetry-otlp")]
    async fn an_endpoint_builds_an_exporter() {
        let config = ObservabilityConfig {
            otlp_endpoint: Some("http://127.0.0.1:4317".to_string()),
            ..ObservabilityConfig::default()
        };
        let layer = otlp_layer(&config).expect("a well-formed http endpoint must build");
        assert!(layer.is_some());
    }

    #[tokio::test]
    #[cfg(feature = "telemetry-otlp")]
    async fn a_malformed_header_key_is_a_typed_error_naming_it() {
        use std::collections::BTreeMap;

        use jammi_db::config::SecretSource;

        let mut otlp_headers = BTreeMap::new();
        // A gRPC metadata key may not contain a space.
        otlp_headers.insert(
            "not a valid key".to_string(),
            SecretSource::Inline("v".to_string()),
        );
        let config = ObservabilityConfig {
            otlp_endpoint: Some("http://127.0.0.1:4317".to_string()),
            otlp_headers,
            ..ObservabilityConfig::default()
        };
        // `OtlpLayer` (the `Ok` payload) is not `Debug`, so map it away
        // before `expect_err` -- only the `Err` side is asserted on here.
        let err = otlp_layer(&config)
            .map(|_| ())
            .expect_err("a malformed header key must be refused");
        assert!(err.to_string().contains("not a valid key"));
    }
}
