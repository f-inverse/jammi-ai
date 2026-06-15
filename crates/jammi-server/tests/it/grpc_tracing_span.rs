//! Tracing span oracle for the gRPC handler surface.
//!
//! Each tenant-aware gRPC handler opens a `#[tracing::instrument]` span named
//! for the request kind (the handler method) and stamps its resolved `tenant_id`
//! onto the span once `session_tenant_traced` reads it from the request — a tower
//! layer cannot, because the per-service `TenantInterceptor` deposits the
//! `SessionTenant` extension *post*-routing, after a pre-routing layer has
//! already run. This drives one instrumented handler directly on the test task
//! (NOT through the real TCP server, whose spawned per-connection tasks a
//! thread-local subscriber cannot capture) under a test-local subscriber and
//! asserts a span actually carrying the request tenant was recorded.
//!
//! The capture is deterministic, independent of fmt formatting, event-flush
//! timing, and span-close timing: a custom `Layer` runs a field visitor on
//! `on_record` — the instant the handler calls `Span::record("tenant_id", …)`,
//! synchronously inside the handler body before its `.await` returns — and writes
//! the captured value straight into an `Arc`-shared buffer keyed by the span's
//! registry `Id`. The capture does NOT wait for `on_close`: sqlx and tower keep a
//! clone of the request span alive and close it on a pool worker thread well after
//! the call returns, so a close-time capture races the assertion and intermittently
//! reads an empty buffer — the original flake. Recording-time capture removes that
//! race entirely. The assertion is non-vacuous: with the handler's `#[instrument]`
//! gone, no `training_status` span opens, nothing records `tenant_id`, and the
//! buffer holds no matching snapshot.

use std::str::FromStr;
use std::sync::{Arc, Mutex};

use jammi_ai::session::InferenceSession;
use jammi_db::TenantId;
use jammi_server::grpc::proto::training as pb;
use jammi_server::grpc::proto::training::training_service_server::TrainingService;
use jammi_server::grpc::session::SessionTenant;
use jammi_server::grpc::training::TrainingServer;
use jammi_test_utils::test_config;
use tonic::Request;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

/// A field visitor that records `name -> debug-rendered value` for every field a
/// span carries, whether stamped at span-open (`on_new_span`) or recorded later
/// (`on_record`). The `?`/Debug rendering matches how the handler records
/// `tenant_id` (`tracing::field::debug(&tenant)`), so an `Option<TenantId>`
/// captures as `Some(<uuid>)`.
#[derive(Default)]
struct FieldSnapshot {
    fields: Vec<(String, String)>,
}

impl FieldSnapshot {
    fn get(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn set(&mut self, name: &str, value: String) {
        match self.fields.iter_mut().find(|(k, _)| k == name) {
            Some((_, v)) => *v = value,
            None => self.fields.push((name.to_string(), value)),
        }
    }
}

impl Visit for FieldSnapshot {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.set(field.name(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.set(field.name(), value.to_string());
    }
}

/// One captured span: its name and the fields seen so far, keyed in the shared
/// buffer by the span's registry `Id`.
struct CapturedSpan {
    name: &'static str,
    fields: FieldSnapshot,
}

/// A test-only `Layer` that snapshots span fields into a shared buffer the
/// instant they are recorded — never on a deferred `on_close`.
///
/// `on_new_span` seeds an entry keyed by the span's `Id` with the fields stamped
/// at open; `on_record` merges each later-recorded field into that same entry.
/// Both callbacks run synchronously inside the registry's dispatch: `on_record`
/// fires the moment the handler calls `Span::record("tenant_id", …)`, *during*
/// the handler body and before its `.await` returns. The captured value is
/// therefore visible to the assertion regardless of when — or on which thread —
/// the span finally closes (sqlx and tower may keep a span clone alive and close
/// it on a pool worker long after the call returns; an `on_close`-based capture
/// would race that and intermittently read an empty buffer).
struct CaptureLayer {
    captured: Arc<Mutex<Vec<(u64, CapturedSpan)>>>,
}

impl CaptureLayer {
    /// Merge `record` into the entry for `id`, seeding `name` if it is new. The
    /// entry is keyed by the registry `Id` so repeated records for the same span
    /// accumulate in place rather than appending duplicates.
    fn merge(&self, id: &Id, name: &'static str, record: impl FnOnce(&mut FieldSnapshot)) {
        let key = id.into_u64();
        let mut captured = self.captured.lock().unwrap();
        let entry = match captured.iter_mut().find(|(k, _)| *k == key) {
            Some((_, span)) => &mut span.fields,
            None => {
                captured.push((
                    key,
                    CapturedSpan {
                        name,
                        fields: FieldSnapshot::default(),
                    },
                ));
                &mut captured.last_mut().expect("just pushed").1.fields
            }
        };
        record(entry);
    }
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let name = ctx
            .span(id)
            .expect("new span exists in the registry")
            .name();
        self.merge(id, name, |snapshot| attrs.record(snapshot));
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let name = ctx
            .span(id)
            .expect("recorded span exists in the registry")
            .name();
        self.merge(id, name, |snapshot| values.record(snapshot));
    }
}

#[test]
fn handler_span_carries_tenant() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = InferenceSession::open(test_config(dir.path()))
            .await
            .expect("session");
        let server = TrainingServer::new(Arc::clone(&session));

        // A request carrying the `SessionTenant` extension exactly as the
        // per-service `TenantInterceptor` deposits it post-routing — the only
        // place the tenant is in scope for the handler to read and record.
        let tenant =
            TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a").expect("valid tenant uuid");
        let mut request = Request::new(pb::TrainingStatusRequest {
            job_id: "job-tracing-oracle".to_string(),
        });
        request.extensions_mut().insert(SessionTenant(Some(tenant)));

        // Install a test-local subscriber whose capture layer snapshots span
        // fields deterministically (prod `telemetry::install` is untouched — no
        // prod log-volume change), then call the instrumented handler directly on
        // this task.
        let captured = Arc::new(Mutex::new(Vec::new()));
        let layer = CaptureLayer {
            captured: Arc::clone(&captured),
        };
        let _guard = tracing_subscriber::registry().with(layer).set_default();

        // The handler records the tenant on its span before any engine work; the
        // job does not exist, so the call returns an error — but the span has
        // already opened and recorded the tenant, which is exactly what this
        // oracle asserts on. The error outcome is the expected path here (no such
        // job for this tenant), asserted explicitly so the return value is
        // handled, not silently discarded.
        let outcome = server.training_status(request).await;
        assert!(
            outcome.is_err(),
            "no such job: the handler errors after recording the span tenant"
        );

        // The span name is the handler method (`training_status`). The `tenant_id`
        // field is recorded with `?` (Debug), so the `Option<TenantId>` snapshots
        // as `Some(...)` wrapping the uuid. A captured span with that name carrying
        // exactly this tenant proves the span was emitted with it.
        let spans = captured.lock().unwrap();
        let span = spans.iter().find(|(_, span)| {
            span.name == "training_status"
                && span.fields.get("tenant_id").is_some_and(|value| {
                    value.starts_with("Some(") && value.contains(&tenant.to_string())
                })
        });
        assert!(
            span.is_some(),
            "expected a captured training_status span carrying \
             tenant_id=Some(..{tenant}..); captured spans: {:?}",
            spans
                .iter()
                .map(|(_, s)| (s.name, s.fields.get("tenant_id").map(str::to_string)))
                .collect::<Vec<_>>()
        );
    });
}
