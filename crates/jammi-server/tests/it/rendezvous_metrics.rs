//! `jammi_placement_ring_empty_total` is documented as a scraped operator
//! observable (`operability.md`, `reference-topologies.md`, the CHANGELOG,
//! `RendezvousMetrics`'s own rustdoc) — this test proves it is ACTUALLY
//! registered into the server's metrics registry and exported on
//! `/metrics`, over the real `OssServer` path (`start_engine_server_from_config`,
//! never a hand-built registry).
//!
//! A real single-node server is opened with `[server] placement =
//! "rendezvous"` (+ `peer_bind`/`peer_advertise`, which that mode requires).
//! On boot, `InferenceSession::wrap_with` upserts this process's OWN
//! `instances` row before returning, so the ring is never naturally empty —
//! forcing the ring-empty arm needs deleting that row directly (out of band,
//! the same technique `gang_membership.rs`'s `force_delete_instance` uses),
//! then issuing one placed search, which finds no root for `self_instance_id`
//! at all (the self-referencing subquery in `Catalog::list_ring_members`
//! returns nothing) and falls back to all-local, counting the fallback.

use jammi_datafusion::ModelTask;
use jammi_db::catalog::backend::{SqlValue, TxOptions};
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::config::{PlacementMode, StoragePrecision};

use crate::common::grpc::{peer_bind_config, start_engine_server_from_config};
use crate::peer_service::{built_index, ROWS};

/// Delete `instance_id`'s own `instances` row out of band — the state a lost
/// row (a construction race, or — as here — a test forcing the ring-empty
/// arm) leaves behind. Mirrors `gang_membership.rs`'s `force_delete_instance`.
async fn force_delete_self_row(catalog: &jammi_db::catalog::Catalog, instance_id: &str) {
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "DELETE FROM instances WHERE instance_id = $1",
                    &[SqlValue::TextOwned(instance_id)],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// Scrape `health_addr`'s `/metrics` and return the exact numeric value of
/// the FIRST sample line for `metric` (no labels — `jammi_placement_ring_
/// empty_total` is a bare counter), or `None` if the metric is absent
/// entirely (dropping the registry-installation call makes this line
/// vanish).
async fn scrape_bare_counter(health_addr: std::net::SocketAddr, metric: &str) -> Option<f64> {
    let body = reqwest::get(format!("http://{health_addr}/metrics"))
        .await
        .expect("GET /metrics")
        .text()
        .await
        .expect("metrics body is text");
    body.lines()
        .find(|line| line.starts_with(metric) && line.as_bytes().get(metric.len()) == Some(&b' '))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|v| v.trim().parse::<f64>().ok())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ring_empty_fallback_is_registered_and_scraped_on_metrics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = peer_bind_config(dir.path());
    cfg.server.placement = PlacementMode::Rendezvous;
    // `peer_advertise` is required by `placement = "rendezvous"` (RV4); this
    // process never needs to be DIALED in this test (the ring is forced
    // empty before any peer call would be attempted), so an arbitrary valid
    // `host:port` is enough — it only ever appears as this row's own
    // `peer_addr` column.
    cfg.server.peer_advertise = Some("127.0.0.1:19321".to_string());
    let server = start_engine_server_from_config(cfg, Some(dir)).await;

    // One segment is enough to reach `RendezvousPlacement::plan` at all
    // (`resolve_search_mode` returns `None` before ever calling `plan` for a
    // table with zero segments).
    let store = server.engine.result_store();
    let table = store
        .create_table(
            "rendezvous-ring-empty",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "model",
            Some(4),
            Some("_row_id"),
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        table
            .append_segment(&built_index(&ROWS, StoragePrecision::default()))
            .await
            .unwrap()
            .0,
        0
    );
    let record = store
        .catalog()
        .get_result_table(table.table_name())
        .await
        .unwrap()
        .unwrap();

    let before = scrape_bare_counter(server.health_addr, "jammi_placement_ring_empty_total")
        .await
        .expect(
            "jammi_placement_ring_empty_total must already be present on /metrics before the \
             fallback ever fires — a Prometheus counter that only appears after its first \
             increment is indistinguishable from one that was never registered",
        );
    assert_eq!(before, 0.0, "no fallback has fired yet");

    force_delete_self_row(store.catalog(), server.engine.instance_id()).await;

    // The placed search itself: `resolve_search_mode` -> `RendezvousPlacement
    // ::plan` finds an empty ring (self's own row is gone) and falls back to
    // all-local for the one segment — the search still succeeds (it is
    // still this process's own data), the fallback is what this test proves
    // gets counted.
    let placed = store
        .resolve_search_mode(&record)
        .await
        .unwrap()
        .expect("one segment exists");
    let query = jammi_test_utils::vq(&[1.0, 0.0, 0.0, 0.0]);
    let _ = placed
        .search_final_placed(&query, 1, 1)
        .await
        .expect("an all-local fallback plan still searches this process's own segment");

    let after = scrape_bare_counter(server.health_addr, "jammi_placement_ring_empty_total")
        .await
        .expect("still present after the fallback");
    assert_eq!(
        after, 1.0,
        "the ring-empty fallback must be visible on /metrics, not just in-process"
    );

    table.abort().await.unwrap();
    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
