//! Integration test for the Rust-facing surface of `jammi-python`.
//!
//! `PyDatabase::session_arc` is what lets a downstream Rust crate
//! (e.g. a downstream Python-bindings layer) share the OSS database's
//! `Arc<InferenceSession>` — and therefore its schema-upgrade lock,
//! trigger broker, catalog cache, and tenant binding — instead of opening
//! a parallel session against the same artifact directory. Without that
//! sharing, two sessions race on schema migrations and observe one
//! another's tenant binding inconsistently.
//!
//! This test asserts both halves of the contract:
//!   1. The returned `Arc` aliases the database's session (proven by the
//!      strong count growing on every call).
//!   2. State mutated through one alias is visible through the other
//!      (proven by binding a tenant via the freshly-cloned `Arc` and
//!      reading it back through a second clone).
//!
//! The remote transport is NOT exercised here: the embed wheel is local-only,
//! and its remote arm is the bundled pure-Python `jammi`. The remote
//! wire surface is proven by the `jammi-server` crate's own data-plane client tests
//! (under `--features wire`) plus the `jammi` conformance test.

use std::str::FromStr;
use std::sync::Arc;

use jammi_db::config::JammiConfig;
use jammi_db::TenantId;
use jammi_native::PyDatabase;
use tempfile::tempdir;

fn test_config(artifact_dir: &std::path::Path) -> JammiConfig {
    JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: jammi_db::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        inference: jammi_db::config::InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn session_arc_shares_session_state_with_pydatabase() {
    let dir = tempdir().expect("tempdir");
    let db = PyDatabase::open(test_config(dir.path())).expect("open PyDatabase");

    // Baseline strong count for the session inside the database. Every
    // `session_arc()` call must increment it — that is what proves the
    // returned `Arc` aliases the database's session rather than a freshly
    // constructed parallel one.
    let first = db.session_arc();
    let count_after_first = Arc::strong_count(&first);

    let second = db.session_arc();
    let count_after_second = Arc::strong_count(&second);

    assert_eq!(
        count_after_second,
        count_after_first + 1,
        "session_arc() must clone the same Arc — strong count should grow \
         by exactly one per call (saw {count_after_first} then {count_after_second})",
    );
    assert!(
        Arc::ptr_eq(&first, &second),
        "both clones must point at the same InferenceSession allocation",
    );

    // Cross-handle state visibility: bind a tenant through `first`, read
    // it back through `second`. A parallel session would not observe the
    // write.
    let tenant =
        TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").expect("valid tenant uuid");
    first.bind_tenant(tenant);
    assert_eq!(
        second.tenant(),
        Some(tenant),
        "binding through one Arc must be visible through the other",
    );

    // And visible through the database's own Python-facing surface.
    let third = db.session_arc();
    assert_eq!(
        third.tenant(),
        Some(tenant),
        "binding must also be visible to subsequently-issued Arcs",
    );

    first.unbind_tenant();
    assert_eq!(
        second.tenant(),
        None,
        "unbinding through one Arc must clear the shared state",
    );
}

// ── OTLP wiring (#486): `open_local`'s subscriber composition ──────────────
//
// A Python-level test would need to observe an in-process wheel install and
// its own tracing subscriber install (`open_local`'s `try_init()` can only
// ever succeed once per process), which the `pytest` harness cannot exercise
// repeatably. This Rust-side test drives the SAME composition function
// `open_local` calls (`jammi_native::build_tracing_layers`) directly, scoped
// to the current thread via `tracing::subscriber::set_default` rather than
// the process-global `try_init()`.

#[test]
fn build_tracing_layers_with_no_endpoint_is_fmt_only() {
    use tracing_subscriber::layer::SubscriberExt;

    let config = test_config(tempdir().expect("tempdir").path());
    let layers = jammi_native::build_tracing_layers(&config).expect("no endpoint must not error");
    assert_eq!(layers.len(), 1, "no otlp_endpoint -> the fmt layer alone");

    // Scoped install (not global): proves the composed subscriber actually
    // accepts spans/events without panicking, without touching the
    // process-global default any other test in this binary might rely on.
    let subscriber = tracing_subscriber::registry().with(layers);
    let _guard = tracing::subscriber::set_default(subscriber);
    tracing::info!("build_tracing_layers_with_no_endpoint_is_fmt_only smoke event");
}

// Building the tonic `Channel` (lazily -- no connection attempt, just the
// client machinery) needs an active Tokio reactor.
#[tokio::test]
async fn build_tracing_layers_with_an_endpoint_adds_the_otlp_layer() {
    use tracing_subscriber::layer::SubscriberExt;

    let mut config = test_config(tempdir().expect("tempdir").path());
    config.observability.otlp_endpoint = Some("http://127.0.0.1:4317".to_string());
    let layers =
        jammi_native::build_tracing_layers(&config).expect("a well-formed endpoint must build");
    assert_eq!(
        layers.len(),
        2,
        "a configured otlp_endpoint -> fmt layer + otlp layer"
    );

    let subscriber = tracing_subscriber::registry().with(layers);
    let _guard = tracing::subscriber::set_default(subscriber);
    tracing::info!("build_tracing_layers_with_an_endpoint_adds_the_otlp_layer smoke event");
}
