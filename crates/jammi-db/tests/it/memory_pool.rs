//! `[engine] memory_limit` becomes the session's memory pool: a plan that reserves past the pool
//! surfaces the typed [`jammi_db::error::JammiError::ResourcesExhausted`] — never a panic, never a
//! silent spill past the configured bound — and the pool is reachable through
//! [`jammi_db::session::JammiSession::memory_pool`] for a caller that registers its own reservation
//! against the same bound.

use datafusion::execution::memory_pool::MemoryConsumer;
use jammi_db::error::JammiError;
use jammi_test_utils::test_config;
use tempfile::tempdir;

/// A session whose pool sits exactly at [`EngineConfig::MEMORY_LIMIT_FLOOR_BYTES`]
/// (the smallest value the grammar accepts) — the floor is deliberately the
/// probe point: if a plan reserving well past 64 MiB is refused THERE, it is
/// refused at every larger, more realistic deployment value too.
async fn floor_pool_session(dir: &std::path::Path) -> jammi_db::session::JammiSession {
    let mut config = test_config(dir);
    config.engine.memory_limit = "64MB".parse().unwrap();
    jammi_db::session::JammiSession::new(config)
        .await
        .expect("sqlite-backed session at the memory_limit floor")
}

/// A query whose blocking sort must buffer well past the 64 MiB pool
/// surfaces the typed [`JammiError::ResourcesExhausted`] — no registration,
/// no fixture file: `generate_series` plus `repeat` synthesizes ~200 MB of
/// row data purely in SQL, sorted by an unordered `ListingTable`-free
/// in-memory relation, which DataFusion's own `SortExec`/`ExternalSorter`
/// must hold (at least in significant part) before it can emit a row.
///
/// The sort key is `value` (the sequence itself, descending — a genuine
/// reorder), never the wide `repeat(...)` column: DataFusion's own planner
/// optimizes `ORDER BY` on a constant expression into a no-op (the wide
/// column is a compile-time constant, so sorting BY it is trivially already
/// satisfied), which would plan no `SortExec` at all and this oracle would
/// prove nothing.
#[tokio::test]
async fn a_plan_that_reserves_past_the_pool_surfaces_resources_exhausted() {
    let dir = tempdir().unwrap();
    let session = floor_pool_session(dir.path()).await;

    // 10,000 rows x 20,000 bytes = ~200 MB of string data -- well past the
    // 64 MiB (67,108,864-byte) pool.
    let err = session
        .sql(
            "SELECT value, repeat('x', 20000) AS s FROM generate_series(1, 10000) \
             ORDER BY value DESC",
        )
        .await
        .unwrap_err();
    match err {
        JammiError::ResourcesExhausted { detail, .. } => {
            assert!(!detail.is_empty(), "detail must carry the raising message");
        }
        other => panic!("expected ResourcesExhausted, got {other:?}"),
    }
}

/// [`jammi_db::session::JammiSession::memory_pool`] is reachable and IS the
/// pool the session's own queries reserve against: a
/// [`MemoryConsumer`] registered directly on it can grow up to the
/// configured limit and is refused past it — typed, never a panic, and
/// never silently granted past the bound.
#[tokio::test]
async fn memory_pool_is_reachable_and_refuses_a_consumer_past_the_limit() {
    let dir = tempdir().unwrap();
    let session = floor_pool_session(dir.path()).await;
    let pool = session.memory_pool();

    let consumer = MemoryConsumer::new("test-consumer");
    let reservation = consumer.register(&pool);

    // Under the limit: granted.
    reservation
        .try_grow(1024)
        .expect("a reservation well under the 64 MiB floor must be granted");

    // Asking for the WHOLE floor again (on top of the 1 KiB already held)
    // pushes the total past the pool's exact size -- refused, typed by
    // DataFusion's own `MemoryPool::try_grow` (this is the raw DataFusion
    // API, not `JammiSession::sql`, so the error stays a `DataFusionError`
    // here; `JammiError::ResourcesExhausted` is what the SAME failure looks
    // like once it crosses `?` into a `jammi_db::error::Result`).
    let over = usize::try_from(jammi_db::config::EngineConfig::MEMORY_LIMIT_FLOOR_BYTES).unwrap();
    let result = reservation.try_grow(over);
    assert!(
        result.is_err(),
        "a reservation past the pool's configured limit must be refused, not silently granted"
    );
}
