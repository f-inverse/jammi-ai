//! Placed search over REAL, separate `jammi-server` processes — RENDEZVOUS
//! RV7, the multi-process proof the in-process K4 analogue
//! (`crates/jammi-server/tests/it/peer_placement.rs`) cannot give: a genuine
//! network round trip, a genuine `RendezvousPlacement` ring read from the
//! shared catalog (never a `StaticPlacement` fixture), and a genuine SIGKILL
//! of an owner process.
//!
//! Three `jammi-server` workers, all `[server] placement = "rendezvous"`
//! (`Fleet::spawn_with_placement(.., PlacementKnob::Rendezvous)`), share the
//! harness's Postgres catalog and MinIO `result_root`. The harness itself
//! builds a 24-segment f32-precision embedding table directly through its own
//! `ResultStore` (`create_table` + `append_segment`, `key_column = None` so
//! `Search`'s hydration join never runs — no `sources` registration needed);
//! 24 segments over 3 ring members means the ranking (a domain-separated hash
//! of `(instance_id, table, segment_id)`, unknowable to this test ahead of
//! its own session's randomly minted `instance_id`) almost certainly spreads
//! ownership across every worker AND puts `worker-1` (the fixed coordinator
//! every query below dials) at every one of the three ranks (winner / retry /
//! excluded) for at least one segment — `(2/3)^24 ≈ 2e-5` for any one rank
//! never occurring.
//!
//! - **K4 byte parity + owner observability (baseline, all three alive).** A
//!   `Search` client fixture (a raw `EmbeddingServiceClient` dialing
//!   worker-1's public flight port) returns hits byte-identical (`key` order)
//!   to an independently computed brute-force cosine ranking; the COMBINED
//!   `jammi_peer_requests_total{rpc="SegmentSearch"}` scraped from worker-2's
//!   and worker-3's own `/metrics` increases (an existing counter — RV7 adds
//!   no new one).
//! - **The failure ladder's retry_ok arm (one process killed).** SIGKILL
//!   worker-2 (within
//!   the liveness margin, so its `instances` row still reads live and the
//!   ring still names it a candidate — the failure ladder, never a
//!   healthy-ring reroute, is what this arm proves). A search still succeeds
//!   with the SAME byte-identical brute-force parity; worker-1's own
//!   `jammi_peer_search_failures_total{reason="unreachable"}` AND
//!   `{reason="retry_ok"}` both increase.
//! - **The failure ladder's UNAVAILABLE arm (two processes killed).** SIGKILL worker-3 too
//!   (only worker-1, the coordinator, survives). `peer_local_load_bytes = 1`
//!   on every worker (K2: `0` is refused) refuses the local-load rung, so any
//!   segment whose top-two ranks exclude worker-1 entirely now exhausts the
//!   ladder: the `Search` RPC fails `Code::Unavailable`, the wire detail
//!   round-trips to `JammiError::Unavailable`, and worker-1's own
//!   `{reason="unavailable"}` counter increases.
//!
//! Stated limits of this oracle (accepted, matching
//! `peer_placement.rs`'s own "Stated limits" discipline): the specific
//! segment-to-worker assignment is HASH-DETERMINED, not test-controlled — the
//! assertions above are the ones that hold regardless of which worker wins
//! which segment, at 24 segments over 3 members; "the OTHER worker" in RV7's
//! own wording is generalised here to "the combined OTHER workers" for the
//! baseline observability check, since which specific worker owns which
//! segment is exactly what this test does not (and structurally cannot,
//! without duplicating the placement algorithm) predict ahead of time.
//! Third, the retry_ok arm carries a real-time coupling: worker-2's
//! `instances` row must still read live (within the 2*lease staleness
//! cutoff) for the WHOLE query loop that runs between the two kills, or the
//! ring would silently stop naming worker-2 a candidate partway through and
//! this test would end up proving a healthy-ring reroute instead of the
//! failure ladder — the loop asserts its own elapsed wall time against that
//! cutoff, with headroom, immediately before the second kill, rather than
//! trusting that a handful of small vector searches always finishes fast
//! enough.

use std::sync::Arc;
use std::time::{Duration, Instant};

use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::store::BuildingTable;
use jammi_numerics::distance::cosine_distance;
use jammi_wire::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_wire::proto::embedding::search_request::Query as WireQuery;
use jammi_wire::proto::embedding::{QueryVector, SearchRequest as WireSearchRequest};
use tonic::transport::Channel;
use tonic::Code;

use crate::harness::{harness_session, Backends, Fleet, PlacementKnob, WorkerPorts, LEASE_SECS};

const DIMS: usize = 4;
const SEGMENTS: usize = 24;
const ROWS_PER_SEGMENT: usize = 3;

/// A small, deterministic, distinct-per-`(segment, row)` vector — no real
/// randomness (a `sin`-based pseudo-random fill, seeded by `(segment, row,
/// dim)`), so the brute-force reference and the wire response are comparing
/// the exact same corpus every run, AND no two rows are a coordinate
/// permutation of each other (a permutation-symmetric fixture — e.g. one
/// dominant component whose POSITION alone varies by segment — makes a
/// symmetric query like `[c, c, c, c]` tie every row with the same "row
/// role" across every segment, an artifact of the fixture, not of placement;
/// this construction has no such symmetry).
fn vector_for(segment: usize, row: usize) -> [f32; DIMS] {
    let mut v = [0f32; DIMS];
    for (d, slot) in v.iter_mut().enumerate() {
        let seed = (segment * 131 + row * 17 + d * 7 + 1) as f32;
        *slot = 0.1 + (seed * 12.9898).sin().abs() * 0.9;
    }
    v
}

/// Build a `SEGMENTS`-segment f32-precision embedding table directly through the
/// harness's own `ResultStore` — the same `create_table` + `append_segment`
/// recipe `peer_placement.rs`'s `two_segment_table` uses, scaled up and with
/// `key_column = None` so `Search`'s hydration join never runs (no `sources`
/// registration needed: the wire request's own `source_id` is never
/// consulted once `embedding_table` is named, and hydration reads the
/// CATALOG row's `key_column`, not the request).
async fn build_placed_table(
    harness: &Arc<InferenceSession>,
    source_id: &str,
) -> (BuildingTable, ResultTableRecord, Vec<(String, [f32; DIMS])>) {
    let store = harness.result_store();
    let table = store
        .create_table(
            source_id,
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "model",
            Some(DIMS as i32),
            None,
            None,
            None,
        )
        .await
        .expect("create the placed-search table");
    let precision = table.storage_precision();
    // `StoragePrecision::default()` IS the crate's `#[default]` variant
    // (32-bit float) — compared by its own default rather than a second,
    // independently-spelled copy of "which precision this fixture uses",
    // so the two can never drift apart.
    assert_eq!(
        precision,
        StoragePrecision::default(),
        "this lane scopes the crate's default (32-bit float) precision only"
    );

    let mut all_rows = Vec::with_capacity(SEGMENTS * ROWS_PER_SEGMENT);
    for seg in 0..SEGMENTS {
        let mut idx = SidecarIndex::new(DIMS, &AnnIndexConfig::default(), precision)
            .expect("build a small sidecar index");
        let mut seg_rows = Vec::with_capacity(ROWS_PER_SEGMENT);
        for row in 0..ROWS_PER_SEGMENT {
            let id = format!("s{seg}-r{row}");
            let v = vector_for(seg, row);
            idx.add(&id, &v).expect("add a row to the sidecar index");
            seg_rows.push((id, v));
        }
        idx.build().expect("build the sidecar index");
        let segment_id = table
            .append_segment(&idx)
            .await
            .expect("append a segment to the placed-search table");
        assert_eq!(segment_id.0 as usize, seg, "segments append in order");
        all_rows.extend(seg_rows);
    }

    let record = store
        .catalog()
        .get_result_table(table.table_name())
        .await
        .expect("read back the table record")
        .expect("the table exists");
    (table, record, all_rows)
}

/// The independent brute-force reference: every row's cosine distance to
/// `query`, ascending (nearest first), ties broken by id — the SAME order a
/// correct placed search must return `key`s in.
fn brute_force_keys(rows: &[(String, [f32; DIMS])], query: &[f32], k: usize) -> Vec<String> {
    let vq = jammi_test_utils::vq(query);
    let mut scored: Vec<(String, f32)> = rows
        .iter()
        .map(|(id, v)| (id.clone(), cosine_distance(&vq, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

async fn channel(addr: &str) -> Channel {
    Channel::from_shared(format!("http://{addr}"))
        .expect("channel uri")
        .connect()
        .await
        .unwrap_or_else(|e| panic!("connect to {addr}: {e}"))
}

async fn search_via(
    client: &mut EmbeddingServiceClient<Channel>,
    table_name: &str,
    query: &[f32],
    k: usize,
) -> Result<Vec<String>, tonic::Status> {
    let response = client
        .search(WireSearchRequest {
            source_id: "rendezvous-placed-search".into(),
            query: Some(WireQuery::QueryVector(QueryVector {
                values: query.to_vec(),
            })),
            k: k as u32,
            filter: None,
            select: Vec::new(),
            embedding_table: Some(table_name.to_string()),
            oversample: None,
        })
        .await?;
    Ok(response
        .into_inner()
        .hits
        .into_iter()
        .map(|h| h.key)
        .collect())
}

/// Poll `ports`'s `/readyz` until it answers 200 (catalog connect + migrate +
/// tier mount) — bounded at 30 s, generous next to a cold CI runner, never a
/// fixed sleep.
async fn wait_ready(ports: &WorkerPorts) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let url = format!("{}/readyz", ports.health_url());
    loop {
        if let Ok(resp) = reqwest::get(&url).await {
            if resp.status() == reqwest::StatusCode::OK {
                return;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("worker at {url} never became ready within 30 s");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Scrape `health_url`'s `/metrics` and sum every sample line of `metric`
/// whose label text contains `label_substr` — a minimal Prometheus text-format
/// reader (no crate dependency): a matching line looks like
/// `metric_name{label="value"} 3`.
async fn scrape_counter(health_url: &str, metric: &str, label_substr: &str) -> u64 {
    let body = reqwest::get(format!("{health_url}/metrics"))
        .await
        .unwrap_or_else(|e| panic!("GET {health_url}/metrics: {e}"))
        .text()
        .await
        .expect("metrics body is text");
    body.lines()
        .filter(|line| line.starts_with(metric) && line.contains(label_substr))
        .filter_map(|line| line.rsplit(' ').next())
        .filter_map(|value| value.trim().parse::<f64>().ok())
        .map(|value| value as u64)
        .sum()
}

/// The lane's shared backends, or a skip that CI can never take silently:
/// with `JAMMI_REQUIRE_DISTRIBUTED` set, unconfigured backends are a hard
/// failure rather than a hollow green (the same per-file require-gate idiom
/// `artifact_crash_window.rs`/`exactly_one_claim.rs` carry). The nested,
/// un-collapsed `if`s are the registry verifier's canonical shape.
#[allow(clippy::collapsible_if)]
fn required_backends(test: &str) -> Option<Backends> {
    let backends = Backends::from_env_or_skip(test);
    if backends.is_none() {
        if std::env::var_os("JAMMI_REQUIRE_DISTRIBUTED").is_some() {
            panic!(
                "{test}: JAMMI_REQUIRE_DISTRIBUTED is set but the distributed lane's shared \
                 backends are unconfigured — a silent skip is not acceptable here"
            );
        }
    }
    backends
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rendezvous_placed_search_over_real_worker_processes() {
    let Some(backends) = required_backends("rendezvous_placed_search") else {
        return;
    };
    let result_root = backends.unique_result_root("rendezvous-placed-search");
    let (harness, _harness_dir) = harness_session(&backends, &result_root).await;

    let (_table, record, rows) = build_placed_table(&harness, "rendezvous-placed-search").await;

    let mut fleet =
        Fleet::spawn_with_placement(&backends, &result_root, 3, PlacementKnob::Rendezvous);

    let coordinator_ports = fleet
        .worker_ports("worker-1")
        .expect("worker-1 was spawned");
    let owner2_ports = fleet
        .worker_ports("worker-2")
        .expect("worker-2 was spawned");
    let owner3_ports = fleet
        .worker_ports("worker-3")
        .expect("worker-3 was spawned");

    // Wait for every worker's `/readyz` (catalog connect + migrate + tier
    // mount) rather than a fixed sleep — bounded, but generous next to a cold
    // CI runner.
    for ports in [&coordinator_ports, &owner2_ports, &owner3_ports] {
        wait_ready(ports).await;
    }

    let mut client = EmbeddingServiceClient::new(channel(&coordinator_ports.flight_addr()).await);

    let queries: Vec<Vec<f32>> = (0..SEGMENTS)
        .map(|seg| vector_for(seg, 0).to_vec())
        .chain([vec![0.3, 0.3, 0.3, 0.3], vec![1.0, 0.0, 0.0, 0.0]])
        .collect();
    let k = 5;

    // ---------------------------------------------------------------------
    // Baseline: all three alive. K4 byte parity + owner observability.
    // ---------------------------------------------------------------------
    let ss_before = scrape_counter(
        &owner2_ports.health_url(),
        "jammi_peer_requests_total",
        "SegmentSearch",
    )
    .await
        + scrape_counter(
            &owner3_ports.health_url(),
            "jammi_peer_requests_total",
            "SegmentSearch",
        )
        .await;

    for q in &queries {
        let got = search_via(&mut client, &record.table_name, q, k)
            .await
            .unwrap_or_else(|status| panic!("baseline search must succeed: {status:?}"));
        assert_eq!(
            got,
            brute_force_keys(&rows, q, k),
            "K4: placed search over real processes must byte-match brute force for query {q:?}"
        );
    }

    let ss_after = scrape_counter(
        &owner2_ports.health_url(),
        "jammi_peer_requests_total",
        "SegmentSearch",
    )
    .await
        + scrape_counter(
            &owner3_ports.health_url(),
            "jammi_peer_requests_total",
            "SegmentSearch",
        )
        .await;
    assert!(
        ss_after > ss_before,
        "at least one of the OTHER two workers must have served >= 1 SegmentSearch \
         (before={ss_before}, after={ss_after})"
    );

    // ---------------------------------------------------------------------
    // The failure ladder's retry_ok arm: kill worker-2, act immediately
    // (well inside the 2*lease=6s liveness margin, so the ring still names
    // it — the FAILURE LADDER is what this proves, never a healthy-ring
    // reroute).
    // ---------------------------------------------------------------------
    let unreachable_before = scrape_counter(
        &coordinator_ports.health_url(),
        "jammi_peer_search_failures_total",
        "unreachable",
    )
    .await;
    let retry_ok_before = scrape_counter(
        &coordinator_ports.health_url(),
        "jammi_peer_search_failures_total",
        "retry_ok",
    )
    .await;
    assert!(
        fleet.kill9("worker-2"),
        "worker-2 must be a live fleet member to kill"
    );
    let worker2_killed_at = Instant::now();

    for q in &queries {
        let got = search_via(&mut client, &record.table_name, q, k)
            .await
            .unwrap_or_else(|status| {
                panic!("search with worker-2 dead (worker-3 still live) must still succeed: {status:?}")
            });
        assert_eq!(
            got,
            brute_force_keys(&rows, q, k),
            "K4 must hold even under the retry ladder for query {q:?}"
        );
    }
    let unreachable_after = scrape_counter(
        &coordinator_ports.health_url(),
        "jammi_peer_search_failures_total",
        "unreachable",
    )
    .await;
    let retry_ok_after = scrape_counter(
        &coordinator_ports.health_url(),
        "jammi_peer_search_failures_total",
        "retry_ok",
    )
    .await;
    assert!(
        unreachable_after > unreachable_before,
        "worker-1 must have observed >= 1 Unreachable call to the killed worker-2 \
         (before={unreachable_before}, after={unreachable_after})"
    );
    assert!(
        retry_ok_after > retry_ok_before,
        "worker-1 must have observed >= 1 successful retry at the second candidate \
         (before={retry_ok_before}, after={retry_ok_after})"
    );

    // Ring-membership timing guard (this file's "Stated limits"): the whole
    // query loop above must have finished comfortably inside the 2*lease
    // staleness cutoff worker-2's now-dead `instances` row would otherwise
    // cross — past that cutoff the ring stops naming worker-2 a candidate at
    // all, and the `unreachable_after`/`retry_ok_after` assertions above
    // would have been proving a healthy-ring reroute rather than the failure
    // ladder. Asserted with headroom (half the cutoff) rather than at the
    // cutoff itself, so a merely slow-but-honest run fails loudly here
    // instead of passing on an accidentally-already-healthy ring.
    //
    // `elapsed_since_kill` is measured from the SIGKILL instant, not from
    // worker-2's actual last heartbeat WRITE, which lands at most one
    // heartbeat interval (~1 s) EARLIER — so the row's real time-since-
    // last-seen is up to ~1 s LARGER than this measurement. That is the
    // conservative side of the two ways this could be wrong: this half-
    // cutoff headroom (3 s out of the full 6 s cutoff at the validated
    // 3 s lease) already exceeds that ~1 s worst-case understatement by a
    // wide margin, so the guard stays sound without needing to read
    // worker-2's own last-heartbeat timestamp back out of the catalog.
    let staleness_cutoff = Duration::from_secs(2 * LEASE_SECS);
    let elapsed_since_kill = worker2_killed_at.elapsed();
    assert!(
        elapsed_since_kill < staleness_cutoff / 2,
        "the retry_ok query loop took {elapsed_since_kill:?}, more than half of the \
         {staleness_cutoff:?} staleness cutoff — worker-2's dead row may have gone \
         stale mid-loop, invalidating the failure-ladder assertions above"
    );

    // ---------------------------------------------------------------------
    // The failure ladder's UNAVAILABLE arm: kill worker-3 too. Only
    // worker-1 (the coordinator) survives; `peer_local_load_bytes = 1`
    // refuses the local-load rung, so any segment whose top-two exclude
    // worker-1 exhausts the ladder.
    // ---------------------------------------------------------------------
    let unavailable_before = scrape_counter(
        &coordinator_ports.health_url(),
        "jammi_peer_search_failures_total",
        "unavailable",
    )
    .await;
    assert!(
        fleet.kill9("worker-3"),
        "worker-3 must be a live fleet member to kill"
    );

    let mut saw_unavailable = false;
    for q in &queries {
        match search_via(&mut client, &record.table_name, q, k).await {
            Ok(got) => {
                // A query every one of whose segments happens to rank
                // worker-1 first or second still succeeds — accepted (see
                // this file's "stated limits"); it must still byte-match
                // brute force.
                assert_eq!(got, brute_force_keys(&rows, q, k), "K4 for query {q:?}");
            }
            Err(status) => {
                saw_unavailable = true;
                assert_eq!(
                    status.code(),
                    Code::Unavailable,
                    "a ladder-exhausted segment must surface Code::Unavailable: {status:?}"
                );
                match jammi_wire::error_from_status(&status) {
                    jammi_db::error::JammiError::Unavailable { resource, .. } => {
                        assert!(
                            resource.contains(&record.table_name),
                            "the Unavailable detail must name the table: {resource}"
                        );
                    }
                    other => panic!("wire detail must round-trip to Unavailable: {other:?}"),
                }
            }
        }
    }
    assert!(
        saw_unavailable,
        "with both peers dead and peer_local_load_bytes=1, at least one of {} segments over 3 \
         ring members must exhaust the ladder ((2/3)^{} chance of none doing so)",
        SEGMENTS, SEGMENTS
    );
    let unavailable_after = scrape_counter(
        &coordinator_ports.health_url(),
        "jammi_peer_search_failures_total",
        "unavailable",
    )
    .await;
    assert!(
        unavailable_after > unavailable_before,
        "worker-1 must have counted >= 1 Unavailable outcome \
         (before={unavailable_before}, after={unavailable_after})"
    );
}
