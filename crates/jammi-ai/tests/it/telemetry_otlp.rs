//! RED, end-to-end proof of #486's OTLP mechanism: a real (in-process,
//! loopback) tonic stub implementing
//! `opentelemetry.proto.collector.trace.v1.TraceService` receives exactly
//! one exported span per opened `tracing::Span`, carrying the configured
//! `service.name` resource attribute and the INCOMING W3C `traceparent`'s
//! trace id — the two properties `jammi-server`'s `telemetry::install` +
//! `TraceContextLayer` and `jammi-python`'s `open_local` both depend on
//! `jammi_ai::telemetry` for.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use jammi_db::config::ObservabilityConfig;
use opentelemetry_proto::tonic::collector::trace::v1::trace_service_server::{
    TraceService, TraceServiceServer,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use tokio::net::TcpListener;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tracing_subscriber::layer::SubscriberExt;

/// The stub collector: records every `ExportTraceServiceRequest` it receives
/// into a shared buffer and answers with an empty success.
#[derive(Clone, Default)]
struct StubCollector {
    received: Arc<Mutex<Vec<ExportTraceServiceRequest>>>,
}

#[tonic::async_trait]
impl TraceService for StubCollector {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        self.received.lock().unwrap().push(request.into_inner());
        Ok(Response::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

/// Start the stub collector on an ephemeral loopback port and return its
/// address plus the shared buffer it writes into. The server task is
/// detached (`tokio::spawn`, never joined) — it lives for the process/test
/// binary, same as any other test-local fixture server in this suite.
async fn spawn_stub_collector() -> (SocketAddr, Arc<Mutex<Vec<ExportTraceServiceRequest>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpIncoming::from(listener);
    let stub = StubCollector::default();
    let received = Arc::clone(&stub.received);

    tokio::spawn(async move {
        Server::builder()
            .add_service(TraceServiceServer::new(stub))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    (addr, received)
}

/// The 32-hex-digit trace id of a synthetic, well-formed W3C `traceparent`
/// header — the value the OTel spec's own example uses.
const INCOMING_TRACE_ID_HEX: &str = "4bf92f3577b34da6a3ce929d0e0e4736";

fn incoming_traceparent_headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        "traceparent",
        format!("00-{INCOMING_TRACE_ID_HEX}-00f067aa0ba902b7-01")
            .parse()
            .unwrap(),
    );
    headers
}

// `multi_thread`: `force_flush` blocks the calling task synchronously
// waiting on the batch processor's background export to complete, which
// itself needs a Tokio worker to drive the tonic export call. A
// single-threaded (default) runtime would deadlock -- the one worker thread
// is the one doing the blocking wait.
#[tokio::test(flavor = "multi_thread")]
async fn stub_collector_receives_one_span_with_service_name_and_the_incoming_trace_id() {
    let (addr, received) = spawn_stub_collector().await;

    let config = ObservabilityConfig {
        otlp_endpoint: Some(format!("http://{addr}")),
        service_name: "jammi-otlp-test".to_string(),
        sample_ratio: 1.0,
        ..ObservabilityConfig::default()
    };
    let otlp = jammi_ai::telemetry::otlp_layer(&config)
        .expect("otlp_layer must not error")
        .expect("an endpoint is configured, so a layer must build");
    // Grab the provider handle BEFORE `.layer` moves into the subscriber
    // below (a partial move of `otlp` would otherwise block any further
    // method call on it).
    let provider_handle = otlp.provider_handle();

    let subscriber = tracing_subscriber::registry().with(otlp.layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    // Simulate what `TraceContextLayer` does per RPC: open one span, extract
    // the incoming W3C context from the request's headers, bind it as this
    // span's OTel parent, and do the "request work" inside it.
    let span = tracing::info_span!("test_rpc");
    jammi_ai::telemetry::set_parent_from_headers(&span, &incoming_traceparent_headers());
    span.in_scope(|| {
        tracing::info!("handling request");
    });
    drop(span);

    provider_handle
        .force_flush()
        .expect("force_flush must succeed");

    let received = received.lock().unwrap();
    assert_eq!(
        received.len(),
        1,
        "expected exactly one ExportTraceServiceRequest, got {}",
        received.len()
    );
    let resource_spans = &received[0].resource_spans;
    assert_eq!(
        resource_spans.len(),
        1,
        "expected exactly one ResourceSpans"
    );

    let resource = resource_spans[0]
        .resource
        .as_ref()
        .expect("ResourceSpans must carry a Resource");
    let service_name = resource
        .attributes
        .iter()
        .find(|kv| kv.key == "service.name")
        .and_then(|kv| kv.value.as_ref())
        .and_then(|v| v.value.as_ref())
        .map(|v| match v {
            opentelemetry_proto::tonic::common::v1::any_value::Value::StringValue(s) => s.clone(),
            other => panic!("service.name was not a string value: {other:?}"),
        })
        .expect("resource must carry a service.name attribute");
    assert_eq!(service_name, "jammi-otlp-test");

    let scope_spans = &resource_spans[0].scope_spans;
    assert_eq!(scope_spans.len(), 1);
    let spans = &scope_spans[0].spans;
    assert_eq!(spans.len(), 1, "expected exactly one span per RPC");

    let trace_id_hex = hex::encode(&spans[0].trace_id);
    assert_eq!(
        trace_id_hex, INCOMING_TRACE_ID_HEX,
        "the exported span must continue the INCOMING traceparent's trace id"
    );
}
