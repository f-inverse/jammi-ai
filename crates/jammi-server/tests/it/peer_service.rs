//! `PeerService` on `[server] peer_bind` — the owner side of the distributed
//! data plane, over the real third listener (`OssServer::bind`).
//!
//! - A4: `peer_bind` unset → no third listener; set → bound, ephemeral.
//! - A5 / commit-2 (c): `SegmentSearch` over `peer_bind` byte-equals the
//!   in-process `search_unit` on the same segment; an id outside the table's
//!   segment list, a query of the wrong width, and an unknown row id are all
//!   own-data — `FAILED_PRECONDITION` (the whole request refused, never an
//!   empty unit; this owner's own data disagreeing with the coordinator's,
//!   never the caller's fault, so it ladders); a precision that mismatches
//!   the bundle is refused the same way; `ExactRescore` equals the
//!   in-process `rescore`; a duplicated row id, a duplicated or empty
//!   segment list, and a non-finite component are genuine request
//!   malformation — `INVALID_ARGUMENT`, terminal at the coordinator
//!   (`owner_refuses_non_conforming_requests`).
//! - A6 / commit-2 (d): the PUBLIC listener of the same server answers
//!   `UNIMPLEMENTED` for `PeerService/*`.
//! - Observability: `jammi_peer_requests_total{rpc}` counts the served
//!   calls; `/metrics` exposes it and `jammi_peer_search_failures_total`.
//!
//! The tables are built through the `BuildingTable` recipe (`create_table` +
//! `append_segment` over 4-d synthetic vectors) — the same catalog + bundle
//! path an embedded table takes, without a model load.

use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::segment::{rescore, search_unit};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::{SegmentSearchPhase, VectorIndex};
use jammi_db::model_task::ModelTask;
use jammi_db::storage::StorageUrl;
use jammi_db::store::{BuildingTable, ResultStore};
use jammi_test_utils::vq;
use jammi_wire::proto::peer::peer_service_client::PeerServiceClient;
use jammi_wire::proto::peer::{
    self as pb, ExactRescoreRequest, SegmentRowIds, SegmentSearchRequest,
};
use tonic::Code;

use crate::common::grpc::{
    channel, peer_bind_config, start_engine_server_from_config, start_engine_server_with_peer_bind,
    PeerEngineServer,
};

/// Build a fully-built one-segment [`SidecarIndex`] over `rows` at `precision`.
pub fn built_index(rows: &[(&str, [f32; 4])], precision: StoragePrecision) -> SidecarIndex {
    let mut idx = SidecarIndex::new(4, &AnnIndexConfig::default(), precision).unwrap();
    for (id, v) in rows {
        idx.add(id, v).unwrap();
    }
    idx.build().unwrap();
    idx
}

/// Register a `building` embedding table on `store` and return the writer's
/// handle (the lease-owned row every segment append is a CAS against).
pub async fn building_table(store: &ResultStore, source_id: &str) -> BuildingTable {
    store
        .create_table(
            source_id,
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
        .unwrap()
}

pub const ROWS: [(&str, [f32; 4]); 4] = [
    ("a", [1.0, 0.0, 0.0, 0.1]),
    ("b", [0.0, 1.0, 0.0, 0.2]),
    ("c", [0.0, 0.0, 1.0, 0.3]),
    ("d", [0.9, 0.1, 0.0, 0.0]),
];

/// Load segment `segment_id` of `table` in-process through the owner's own
/// segment cache — the reference the wire answer is compared against.
pub async fn load_segment(
    store: &ResultStore,
    table: &str,
    segment_id: i64,
    precision: StoragePrecision,
) -> SidecarIndex {
    let segs = store.catalog().list_index_segments(table).await.unwrap();
    let seg = segs
        .iter()
        .find(|s| s.segment_id == segment_id)
        .expect("segment exists");
    let url = StorageUrl::parse(&seg.index_path).unwrap();
    store
        .segment_cache()
        .load_segment(&url, store.ann_config(), precision)
        .await
        .unwrap()
}

fn pairs(hits: &[pb::Hit]) -> Vec<(String, f32)> {
    hits.iter()
        .map(|h| (h.row_id.clone(), h.distance))
        .collect()
}

fn served(server: &PeerEngineServer, rpc: &str) -> u64 {
    server.metrics.peer_requests.with_label_values(&[rpc]).get()
}

// A4 — `peer_bind` unset → no third listener.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn peer_bind_unset_means_no_third_listener() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.server.health_listen = "127.0.0.1:0".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    assert_eq!(cfg.server.peer_bind, None);
    let server = jammi_server::runtime::OssServer::new(cfg).await.unwrap();
    let engine = server.session();
    let bound = server.bind().await.unwrap();
    assert_eq!(bound.peer_addr(), None, "unset = not mounted = single node");
    drop(bound);
    engine.close().await;
}

// A5 / (c) / (d) — SegmentSearch over peer_bind == in-process search_unit;
// unknown id refused; precision mismatch refused; public listener UNIMPLEMENTED.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn segment_search_over_peer_bind_equals_in_process_search_unit() {
    let server = start_engine_server_with_peer_bind().await;
    let store = server.engine.result_store();
    let table = building_table(&store, "src_f32").await;
    let seg0 = table
        .append_segment(&built_index(&ROWS, StoragePrecision::F32))
        .await
        .unwrap();
    assert_eq!(seg0.0, 0);
    let table_name = table.table_name().to_string();
    let q = [1.0f32, 0.0, 0.0, 0.0];

    // In-process reference on the very same bundle.
    let index = load_segment(&store, &table_name, 0, StoragePrecision::F32).await;
    let want = search_unit(
        jammi_db::index::SegmentId(0),
        &index,
        &vq(&q),
        3,
        SegmentSearchPhase::Final,
        &|id| index.get_exact(id),
    )
    .unwrap();
    assert_eq!(want.len(), 3);

    let mut client = PeerServiceClient::new(channel(server.peer_addr).await);
    let before = served(&server, "SegmentSearch");
    let resp = client
        .segment_search(SegmentSearchRequest {
            table_name: table_name.clone(),
            segment_ids: vec![0],
            storage_precision: pb::StoragePrecision::F32 as i32,
            query: q.to_vec(),
            width: 3,
            phase: pb::SegmentSearchPhase::Final as i32,
        })
        .await
        .expect("SegmentSearch over peer_bind")
        .into_inner();
    assert_eq!(resp.units.len(), 1);
    assert_eq!(resp.units[0].segment_id, 0);
    assert_eq!(
        pairs(&resp.units[0].hits),
        want,
        "the owner's answer over the wire is the in-process search_unit's bytes"
    );
    assert_eq!(served(&server, "SegmentSearch"), before + 1);

    // An id outside the table's segment list refuses the WHOLE request.
    let err = client
        .segment_search(SegmentSearchRequest {
            table_name: table_name.clone(),
            segment_ids: vec![0, 7],
            storage_precision: pb::StoragePrecision::F32 as i32,
            query: q.to_vec(),
            width: 3,
            phase: pb::SegmentSearchPhase::Final as i32,
        })
        .await
        .expect_err("segment 7 is not a segment of the table");
    // Own-data: the owner's OWN segment list disagrees with what the
    // coordinator named — never the caller's fault. `FAILED_PRECONDITION`,
    // which ladders (a retry, then a local load), unlike a genuine request
    // fault.
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    assert!(
        err.message().contains('7') && err.message().contains(&table_name),
        "{err:?}"
    );

    // A precision the bundle is not stamped with is refused by the strict load.
    let err = client
        .segment_search(SegmentSearchRequest {
            table_name: table_name.clone(),
            segment_ids: vec![0],
            storage_precision: pb::StoragePrecision::Int8 as i32,
            query: q.to_vec(),
            width: 3,
            phase: pb::SegmentSearchPhase::Final as i32,
        })
        .await
        .expect_err("an F32 bundle is not an Int8 bundle");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");

    // Unspecified enums (raw 0) are request malformation — refused, never
    // defaulted.
    let err = client
        .segment_search(SegmentSearchRequest {
            table_name: table_name.clone(),
            segment_ids: vec![0],
            storage_precision: 0,
            query: q.to_vec(),
            width: 3,
            phase: pb::SegmentSearchPhase::Final as i32,
        })
        .await
        .expect_err("unspecified precision");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");

    // An UNRECOGNISED non-zero raw enum value is NOT the same fault: it is a
    // value a newer coordinator knows and this owner's build does not —
    // rolling-upgrade version skew, own-data, `FAILED_PRECONDITION` (ladders).
    // `0` and an unrecognised non-zero value must never collapse.
    let err = client
        .segment_search(SegmentSearchRequest {
            table_name: table_name.clone(),
            segment_ids: vec![0],
            storage_precision: 99,
            query: q.to_vec(),
            width: 3,
            phase: pb::SegmentSearchPhase::Final as i32,
        })
        .await
        .expect_err("an unrecognised non-zero precision value");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    let err = client
        .segment_search(SegmentSearchRequest {
            table_name: table_name.clone(),
            segment_ids: vec![0],
            storage_precision: pb::StoragePrecision::F32 as i32,
            query: q.to_vec(),
            width: 3,
            phase: 99,
        })
        .await
        .expect_err("an unrecognised non-zero phase value");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");

    // (d) The SAME server's PUBLIC listener answers UNIMPLEMENTED.
    let mut public = PeerServiceClient::new(channel(server.public_addr).await);
    let err = public
        .segment_search(SegmentSearchRequest {
            table_name: table_name.clone(),
            segment_ids: vec![0],
            storage_precision: pb::StoragePrecision::F32 as i32,
            query: q.to_vec(),
            width: 3,
            phase: pb::SegmentSearchPhase::Final as i32,
        })
        .await
        .expect_err("the public listener must not serve PeerService");
    assert_eq!(err.code(), Code::Unimplemented, "{err:?}");

    // /metrics carries both peer families.
    let body = reqwest::get(format!("http://{}/metrics", server.health_addr))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        body.contains("jammi_peer_requests_total{rpc=\"SegmentSearch\"}"),
        "{body}"
    );
    assert!(
        body.contains("jammi_peer_search_failures_total{reason=\"unreachable\"} 0"),
        "{body}"
    );

    table.abort().await.unwrap();
    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

// A5 — ExactRescore over peer_bind == in-process rescore on an Int8 table;
// a candidate with no exact vector is DATA_LOSS for the whole request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_rescore_over_peer_bind_equals_in_process_rescore() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = peer_bind_config(dir.path());
    cfg.embedding.ann.storage_precision = StoragePrecision::Int8;
    let server = start_engine_server_from_config(cfg, Some(dir)).await;
    let store = server.engine.result_store();
    let table = building_table(&store, "src_int8").await;
    table
        .append_segment(&built_index(&ROWS, StoragePrecision::Int8))
        .await
        .unwrap();
    let table_name = table.table_name().to_string();
    let q = [0.7f32, 0.7, 0.0, 0.0];

    let index = load_segment(&store, &table_name, 0, StoragePrecision::Int8).await;
    let want = rescore(
        jammi_db::index::SegmentId(0),
        vec![
            ("d".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
        ],
        &|id| index.get_exact(id),
        &vq(&q),
    )
    .unwrap();

    let mut client = PeerServiceClient::new(channel(server.peer_addr).await);
    let before = served(&server, "ExactRescore");
    let resp = client
        .exact_rescore(ExactRescoreRequest {
            table_name: table_name.clone(),
            storage_precision: pb::StoragePrecision::Int8 as i32,
            query: q.to_vec(),
            row_ids_by_segment: vec![SegmentRowIds {
                segment_id: 0,
                row_ids: vec!["d".into(), "a".into(), "b".into()],
            }],
        })
        .await
        .expect("ExactRescore over peer_bind")
        .into_inner();
    assert_eq!(pairs(&resp.hits), want);
    assert_eq!(served(&server, "ExactRescore"), before + 1);

    let err = client
        .exact_rescore(ExactRescoreRequest {
            table_name: table_name.clone(),
            storage_precision: pb::StoragePrecision::Int8 as i32,
            query: q.to_vec(),
            row_ids_by_segment: vec![SegmentRowIds {
                segment_id: 0,
                row_ids: vec!["a".into(), "ghost".into()],
            }],
        })
        .await
        .expect_err("a row id the segment does not index is own-data, not a caller fault");
    // Own-data, NOT `DATA_LOSS` and not the caller's fault: this owner
    // reloads its segment per RPC, so a rebuild between phases can move ids
    // out from under it — `FAILED_PRECONDITION`, which ladders, never drives
    // the coordinator's local-load rung with the wrong reason a `DATA_LOSS`
    // "torn" classification would.
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");

    let err = client
        .exact_rescore(ExactRescoreRequest {
            table_name: table_name.clone(),
            storage_precision: pb::StoragePrecision::Int8 as i32,
            query: q.to_vec(),
            row_ids_by_segment: vec![SegmentRowIds {
                segment_id: 3,
                row_ids: vec!["a".into()],
            }],
        })
        .await
        .expect_err("segment 3 is not a segment of the table");
    // Own-data: the same class as the SegmentSearch case above.
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");

    table.abort().await.unwrap();
    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

// Input-edge reconciliation at the OWNER: every value that arrives over the
// peer seam is checked against what the owner knows before any kernel runs
// — never a panic, never a silent prefix-scored answer, never `DATA_LOSS`
// (which would drive the coordinator's local-load rung for a fault the
// coordinator itself caused). The class splits on WHOSE fault it is: a
// genuinely malformed request (an empty or duplicated segment id, a
// duplicated row id, a non-finite component) is `INVALID_ARGUMENT`,
// TERMINAL at the coordinator; a width or row-id mismatch against THIS
// OWNER's own loaded segment is own-data — `FAILED_PRECONDITION`, which
// ladders — because the coordinator authoritatively enforces width against
// its own index or the catalog before any fan-out, so an owner-reported
// width mismatch can only be this owner's segment drifting from that
// authority, never the caller's fault.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn owner_refuses_non_conforming_requests() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = peer_bind_config(dir.path());
    cfg.embedding.ann.storage_precision = StoragePrecision::Int8;
    let server = start_engine_server_from_config(cfg, Some(dir)).await;
    let store = server.engine.result_store();
    let table = building_table(&store, "src_edge").await;
    table
        .append_segment(&built_index(&ROWS, StoragePrecision::Int8))
        .await
        .unwrap();
    let table_name = table.table_name().to_string();
    let mut client = PeerServiceClient::new(channel(server.peer_addr).await);
    let search = |query: Vec<f32>, segment_ids: Vec<i64>| SegmentSearchRequest {
        table_name: table_name.clone(),
        segment_ids,
        storage_precision: pb::StoragePrecision::Int8 as i32,
        query,
        width: 3,
        phase: pb::SegmentSearchPhase::Approximate as i32,
    };
    let rescore_req = |query: Vec<f32>, groups: Vec<SegmentRowIds>| ExactRescoreRequest {
        table_name: table_name.clone(),
        storage_precision: pb::StoragePrecision::Int8 as i32,
        query,
        row_ids_by_segment: groups,
    };
    let rows = |ids: &[&str]| SegmentRowIds {
        segment_id: 0,
        row_ids: ids.iter().map(|s| s.to_string()).collect(),
    };

    // A LONGER query than the segment's dimensions (5 vs 4): SegmentSearch.
    // Own-data: the coordinator authoritatively enforces width before any
    // fan-out, so an owner-reported mismatch can only be THIS owner's
    // segment drifting from that authority.
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![0]))
        .await
        .expect_err("a 5-wide query against a 4-wide segment is own-data, not a caller fault");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    assert!(
        err.message().contains('5') && err.message().contains('4'),
        "{err:?}"
    );
    // … and ExactRescore (this one indexes past the stored vector in `cosine_distance`).
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            vec![rows(&["a"])],
        ))
        .await
        .expect_err("a longer query must be refused, never panic");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    // A SHORTER query (3 vs 4): refused, never silently scored over a prefix.
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0], vec![0]))
        .await
        .expect_err("a 3-wide query is own-data, not a caller fault");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    let err = client
        .exact_rescore(rescore_req(vec![1.0, 0.0, 0.0], vec![rows(&["a"])]))
        .await
        .expect_err("a 3-wide query is own-data, not a caller fault");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    // An EMPTY query is the same fault.
    let err = client
        .segment_search(search(vec![], vec![0]))
        .await
        .expect_err("an empty query is own-data, not a caller fault");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");

    // A4 — a NON-FINITE component is a CALLER fault at the owner's edge:
    // `INVALID_ARGUMENT` on both rpcs, never `DATA_LOSS` (which would count
    // as `torn` at the coordinator and drive its local-load rung).
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = client
            .segment_search(search(vec![poison, 0.0, 0.0, 0.0], vec![0]))
            .await
            .expect_err("a non-finite query component is a caller fault");
        assert_eq!(err.code(), Code::InvalidArgument, "{poison:?}: {err:?}");
        let err = client
            .exact_rescore(rescore_req(vec![poison, 0.0, 0.0, 0.0], vec![rows(&["a"])]))
            .await
            .expect_err("a non-finite query component is a caller fault");
        assert_eq!(err.code(), Code::InvalidArgument, "{poison:?}: {err:?}");
    }

    // A row id the segment does not index is own-data (this owner reloads
    // its segment per RPC; a rebuild between phases can move ids out from
    // under it), never the caller's fault and never a torn bundle.
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a", "ghost"])],
        ))
        .await
        .expect_err("an unknown row id is own-data, not a caller fault");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    assert!(err.message().contains("ghost"), "{err:?}");
    // A row id named twice in one group is refused too.
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a", "a"])],
        ))
        .await
        .expect_err("a duplicated row id is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");
    // A segment named twice in one request (either RPC) is refused.
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0, 0.0], vec![0, 0]))
        .await
        .expect_err("a duplicated segment id is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a"]), rows(&["b"])],
        ))
        .await
        .expect_err("a duplicated segment group is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");
    // An empty segment list is refused (a unit-less answer would be a silent shrink).
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0, 0.0], vec![]))
        .await
        .expect_err("an empty segment list is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");

    // The conforming request still serves — the edge refuses, it does not
    // shadow the happy path.
    let ok = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a", "d"])],
        ))
        .await
        .expect("a conforming ExactRescore serves")
        .into_inner();
    assert_eq!(ok.hits.len(), 2);

    table.abort().await.unwrap();
    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

// Input-edge reconciliation at the OWNER: every value that arrives over the
// peer seam is checked against what the owner knows before any kernel runs
// — never a panic, never a silent prefix-scored answer, never DATA_LOSS
// (which would drive the coordinator's local-load rung for a fault the
// coordinator itself caused). The class splits on WHOSE fault it is: a
// width mismatch against THIS OWNER's own loaded segment, and an unknown
// row id THIS OWNER's segment does not index, are own-data —
// FAILED_PRECONDITION, which ladders (this owner's segment can drift from
// the coordinator's authority, or be reloaded between phases); a
// duplicated row id, a duplicated or empty segment list, are genuine
// request malformation — INVALID_ARGUMENT, terminal at the coordinator.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn owner_refuses_non_conforming_requests_with_invalid_argument() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = peer_bind_config(dir.path());
    cfg.embedding.ann.storage_precision = StoragePrecision::Int8;
    let server = start_engine_server_from_config(cfg, Some(dir)).await;
    let store = server.engine.result_store();
    let table = building_table(&store, "src_edge").await;
    table
        .append_segment(&built_index(&ROWS, StoragePrecision::Int8))
        .await
        .unwrap();
    let table_name = table.table_name().to_string();
    let mut client = PeerServiceClient::new(channel(server.peer_addr).await);
    let search = |query: Vec<f32>, segment_ids: Vec<i64>| SegmentSearchRequest {
        table_name: table_name.clone(),
        segment_ids,
        storage_precision: pb::StoragePrecision::Int8 as i32,
        query,
        width: 3,
        phase: pb::SegmentSearchPhase::Approximate as i32,
    };
    let rescore_req = |query: Vec<f32>, groups: Vec<SegmentRowIds>| ExactRescoreRequest {
        table_name: table_name.clone(),
        storage_precision: pb::StoragePrecision::Int8 as i32,
        query,
        row_ids_by_segment: groups,
    };
    let rows = |ids: &[&str]| SegmentRowIds {
        segment_id: 0,
        row_ids: ids.iter().map(|s| s.to_string()).collect(),
    };

    // A LONGER query than the segment's dimensions (5 vs 4): own-data — this
    // owner's loaded segment disagrees with the coordinator's authority.
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![0]))
        .await
        .expect_err("a 5-wide query against a 4-wide segment is own-data");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    assert!(
        err.message().contains('5') && err.message().contains('4'),
        "{err:?}"
    );
    // … and ExactRescore (this one indexes past the stored vector in `cosine_distance`).
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            vec![rows(&["a"])],
        ))
        .await
        .expect_err("a longer query must be refused, never panic");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    // A SHORTER query (3 vs 4): refused, never silently scored over a prefix.
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0], vec![0]))
        .await
        .expect_err("a 3-wide query is own-data");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    let err = client
        .exact_rescore(rescore_req(vec![1.0, 0.0, 0.0], vec![rows(&["a"])]))
        .await
        .expect_err("a 3-wide query is own-data");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    // An EMPTY query is the same fault.
    let err = client
        .segment_search(search(vec![], vec![0]))
        .await
        .expect_err("an empty query is own-data");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");

    // A row id the segment does not index is own-data (this owner reloads
    // its segment per RPC, so a rebuild between phases can move ids out
    // from under it) — never mistaken for a torn bundle.
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a", "ghost"])],
        ))
        .await
        .expect_err("an unknown row id is own-data");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err:?}");
    assert!(err.message().contains("ghost"), "{err:?}");
    // A row id named twice in one group is refused too.
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a", "a"])],
        ))
        .await
        .expect_err("a duplicated row id is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");
    // A segment named twice in one request (either RPC) is refused.
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0, 0.0], vec![0, 0]))
        .await
        .expect_err("a duplicated segment id is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");
    let err = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a"]), rows(&["b"])],
        ))
        .await
        .expect_err("a duplicated segment group is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");
    // An empty segment list is refused (a unit-less answer would be a silent shrink).
    let err = client
        .segment_search(search(vec![1.0, 0.0, 0.0, 0.0], vec![]))
        .await
        .expect_err("an empty segment list is a caller fault");
    assert_eq!(err.code(), Code::InvalidArgument, "{err:?}");

    // The conforming request still serves — the edge refuses, it does not
    // shadow the happy path.
    let ok = client
        .exact_rescore(rescore_req(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![rows(&["a", "d"])],
        ))
        .await
        .expect("a conforming ExactRescore serves")
        .into_inner();
    assert_eq!(ok.hits.len(), 2);

    table.abort().await.unwrap();
    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
