//! Integration test for the Rust-facing surface of `jammi-python`.
//!
//! `PyDatabase::session_arc` is what lets a downstream Rust crate
//! (e.g. an enterprise Python-bindings layer) share the OSS database's
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

use std::str::FromStr;
use std::sync::Arc;

use _native::PyDatabase;
use jammi_engine::config::JammiConfig;
use jammi_engine::TenantId;
use tempfile::tempdir;

fn test_config(artifact_dir: &std::path::Path) -> JammiConfig {
    JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: jammi_engine::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        inference: jammi_engine::config::InferenceConfig {
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
