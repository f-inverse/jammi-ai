//! Placed search end to end — the K4 analogue over the peer tower.
//!
//! Two engine instances in ONE test process over one SQLite catalog file and
//! one shared local `artifact_dir` (a valid stand-in for a shared
//! `result_root`: both instances read the same `file://` bundles in place).
//! Instance B is a `[server] peer_bind` server started through the production
//! `OssServer` path; instance A is `InferenceSession::open_with_placement` with
//! a placement that maps chosen segments to B (or to a dead port), over the
//! `GrpcPeerTransport` the session's store builder wires.
//!
//! - A8: for F32, Int8 and Binary, a two-segment table's
//!   `search_final_placed` (segment 1 owned by B) byte-equals the all-local
//!   `search_final` over both segments and the brute-force ids; B served ≥ 1
//!   `SegmentSearch` (and, for Int8, ≥ 1 `ExactRescore`); A's `local_load`
//!   counter did not move.
//! - A9: the ladder — `[dead, B]` leaves bytes unchanged (`unreachable == 1`,
//!   `retry_ok == 1`); `[dead, dead]` with `peer_local_load_bytes = Some(1)`
//!   is `Unavailable` naming `table/1` (and `Code::Unavailable` over the public
//!   `Search` verb, the detail round-tripping); `dimensions = None` is
//!   `Unavailable`; budget unset loads locally (`local_load == 1`, bytes
//!   unchanged). Readiness is 200 throughout.
//! - A7: a coordinator scoped to tenant B cannot resolve tenant A's table —
//!   it fails before any fan-out (B's serve count does not move).
//! - A13: the force-local entries (the neighbor-graph build's
//!   `resolve_search_mode_local`, the eval runner's `search_vectors_local`)
//!   complete with every peer counter delta 0 while the placed entries on the
//!   SAME store return `Unavailable`.
//!
//! Stated limits (accepted): the second SQLite pool's close is slow and noisy;
//! both sessions run the boot recovery sweep over the same root (B opens
//! before A creates anything); this proves transport + merge + ladder only —
//! not object-store fetch, process isolation, or network partition.

use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, RwLock};

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::config::{ServerConfig, StoragePrecision};
use jammi_db::error::JammiError;
use jammi_db::index::peer::{PeerAddr, SegmentPlacement};
use jammi_db::index::SegmentId;
use jammi_db::model_task::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::StorageUrl;
use jammi_db::store::{BuildingTable, ResultStore};
use jammi_db::TenantId;
use jammi_numerics::distance::cosine_distance;
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::search_request::Query as WireSearchQuery;
use jammi_server::grpc::proto::embedding::{QueryVector, SearchRequest as WireSearchRequest};
use jammi_wire::request::{SearchQuery, SearchRequest};
use parquet::arrow::ArrowWriter;
use tonic::Code;

use crate::common::grpc::{
    channel, start_engine_server_over_session, start_engine_server_with_peer_bind, PeerEngineServer,
};
use crate::peer_service::{built_index, ROWS};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

type Owners = BTreeMap<(String, SegmentId), Vec<PeerAddr>>;

/// A placement the oracles mutate between arms: `(table, segment)` → owners.
/// A pair absent from the map is local (the `StaticPlacement` semantics).
#[derive(Default, Clone)]
struct TestPlacement(Arc<RwLock<Owners>>);

impl TestPlacement {
    fn set(&self, table: &str, segment: i64, owners: Vec<PeerAddr>) {
        self.0
            .write()
            .unwrap()
            .insert((table.to_string(), SegmentId(segment)), owners);
    }
}

#[tonic::async_trait]
impl SegmentPlacement for TestPlacement {
    async fn owners(&self, table: &str, segment: SegmentId) -> Vec<PeerAddr> {
        self.0
            .read()
            .unwrap()
            .get(&(table.to_string(), segment))
            .cloned()
            .unwrap_or_default()
    }
}

/// A loopback port nothing listens on — connection refused, the
/// `Unreachable` rung.
fn dead_addr() -> PeerAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    PeerAddr(addr.to_string())
}

/// Instance A: a library coordinator over the SAME artifact dir (catalog +
/// root) as B, at `precision`, with `placement` and the given local-load
/// budget.
async fn open_a(
    dir: &Path,
    precision: StoragePrecision,
    budget: Option<u64>,
    placement: TestPlacement,
) -> Arc<InferenceSession> {
    let mut cfg = jammi_test_utils::test_config(dir);
    cfg.embedding.ann.storage_precision = precision;
    cfg.server.peer_local_load_bytes = budget;
    InferenceSession::open_with_placement(cfg, Arc::new(placement))
        .await
        .expect("instance A opens over the shared dir")
}

fn dir_of(b: &PeerEngineServer) -> std::path::PathBuf {
    b._dir
        .as_ref()
        .expect("B owns its dir")
        .path()
        .to_path_buf()
}

fn served(b: &PeerEngineServer, rpc: &str) -> u64 {
    b.metrics.peer_requests.with_label_values(&[rpc]).get()
}

fn snapshot(store: &ResultStore) -> Vec<(&'static str, u64)> {
    store.peer_failures().snapshot()
}

fn delta(before: &[(&'static str, u64)], after: &[(&'static str, u64)], label: &str) -> u64 {
    let b = before.iter().find(|(l, _)| *l == label).unwrap().1;
    let a = after.iter().find(|(l, _)| *l == label).unwrap().1;
    a - b
}

pub const ROWS_B: [(&str, [f32; 4]); 4] = [
    ("e", [0.0, 0.0, 0.0, 1.0]),
    ("f", [0.1, 0.9, 0.0, 0.0]),
    ("g", [0.0, 0.1, 0.9, 0.0]),
    ("h", [0.5, 0.5, 0.0, 0.0]),
];

fn brute_force_ids(query: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<(String, f32)> = ROWS
        .iter()
        .chain(ROWS_B.iter())
        .map(|(id, v)| (id.to_string(), cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn ids(hits: &[(String, f32)]) -> Vec<String> {
    hits.iter().map(|(id, _)| id.clone()).collect()
}

/// A `building` embedding table with `dimensions` recorded (or not) and two
/// appended segments (`ROWS` → segment 0, `ROWS_B` → segment 1) at the store's
/// precision.
async fn two_segment_table(
    store: &ResultStore,
    source_id: &str,
    dimensions: Option<i32>,
) -> (BuildingTable, ResultTableRecord) {
    let table = store
        .create_table(
            source_id,
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "model",
            dimensions,
            Some("_row_id"),
            None,
            None,
        )
        .await
        .unwrap();
    let precision = table.storage_precision();
    assert_eq!(
        table
            .append_segment(&built_index(&ROWS, precision))
            .await
            .unwrap()
            .0,
        0
    );
    assert_eq!(
        table
            .append_segment(&built_index(&ROWS_B, precision))
            .await
            .unwrap()
            .0,
        1
    );
    let record = store
        .catalog()
        .get_result_table(table.table_name())
        .await
        .unwrap()
        .unwrap();
    (table, record)
}

const DOC_IDS: [&str; 3] = ["doc-0", "doc-1", "doc-2"];
const DOC_DIMS: usize = 4;

/// A ready, searchable one-segment F32 embedding table WITHOUT a model: a
/// registered source plus `import_embeddings` of precomputed basis vectors
/// (`doc-i` → `e_i`). The table the `Search` verb resolves for `source_id`.
async fn import_table(a: &Arc<InferenceSession>, dir: &Path, source_id: &str) -> ResultTableRecord {
    let source = dir.join(format!("{source_id}.parquet"));
    let vectors = dir.join(format!("{source_id}_vectors.parquet"));
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new("body", DataType::Utf8, false),
        ]));
        let ids = Arc::new(StringArray::from_iter_values(DOC_IDS)) as ArrayRef;
        let bodies = Arc::new(StringArray::from_iter_values(
            DOC_IDS.iter().map(|id| format!("body of {id}")),
        )) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, bodies]).unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&source).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let schema = Arc::new(Schema::new(vec![
            Field::new("_row_id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(item.clone(), DOC_DIMS as i32),
                false,
            ),
        ]));
        let ids = Arc::new(StringArray::from_iter_values(DOC_IDS)) as ArrayRef;
        let flat: Vec<f32> = (0..DOC_IDS.len())
            .flat_map(|i| (0..DOC_DIMS).map(move |d| if d == i { 3.0 } else { 0.0 }))
            .collect();
        let vectors_col = Arc::new(
            FixedSizeListArray::try_new(
                item,
                DOC_DIMS as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors_col]).unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&vectors).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    a.add_source(
        source_id,
        SourceType::File,
        SourceConnection {
            url: Some(format!("file://{}", source.display())),
            format: Some(FileFormat::Parquet),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let record = a
        .import_embeddings(
            source_id,
            "import-model",
            &StorageUrl::parse(&format!("file://{}", vectors.display())).unwrap(),
            "doc_id",
            &["body".to_string()],
            DOC_DIMS,
        )
        .await
        .unwrap();
    assert_eq!(record.status, "ready");
    let segments = a
        .catalog()
        .list_index_segments(&record.table_name)
        .await
        .unwrap();
    assert_eq!(segments.len(), 1, "an imported table is one segment");
    record
}

fn e(i: usize) -> Vec<f32> {
    (0..DOC_DIMS)
        .map(|d| if d == i { 1.0 } else { 0.0 })
        .collect()
}

fn search_request(source_id: &str, query: Vec<f32>, k: usize) -> SearchRequest {
    SearchRequest {
        source_id: source_id.to_string(),
        query: SearchQuery::Vector(query),
        k,
        embedding_table: None,
        filter: None,
        select: Vec::new(),
        oversample: None,
    }
}

async fn readyz_is_200(b: &PeerEngineServer) {
    let status = reqwest::get(format!("http://{}/readyz", b.health_addr))
        .await
        .unwrap()
        .status();
    assert_eq!(status, 200, "readiness stays the catalog ping");
}

// ---------------------------------------------------------------------------
// A8 — two instances, one process: placed == all-local == brute force
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn placed_search_over_two_instances_equals_all_local_and_brute_force() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner = PeerAddr(b.peer_addr.to_string());
    let queries: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.6, 0.6, 0.0, 0.1],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    for precision in [
        StoragePrecision::F32,
        StoragePrecision::Int8,
        StoragePrecision::Binary,
    ] {
        let placement = TestPlacement::default();
        let a = open_a(&dir, precision, None, placement.clone()).await;
        let store = a.result_store();
        let (table, record) =
            two_segment_table(&store, &format!("src_{precision:?}"), Some(4)).await;
        assert_eq!(record.storage_precision, Some(precision));
        placement.set(&record.table_name, 1, vec![owner.clone()]);

        let before = snapshot(&store);
        let ss_before = served(&b, "SegmentSearch");
        let er_before = served(&b, "ExactRescore");

        for q in &queries {
            for (k, oversample) in [(1usize, 1usize), (3, 4), (5, 32)] {
                let placed = store.resolve_search_mode(&record).await.unwrap().unwrap();
                assert!(placed.has_remote(), "segment 1 is B's");
                assert_eq!(placed.len(), 8);
                let got = placed.search_final_placed(q, k, oversample).await.unwrap();
                let all_local = store
                    .resolve_search_mode_local(&record)
                    .await
                    .unwrap()
                    .unwrap()
                    .search_final(q, k, oversample)
                    .unwrap();
                assert_eq!(
                    got, all_local,
                    "{precision:?} k={k} oversample={oversample}: the placed entry over B \
                     must return the all-local merge's bytes"
                );
                assert_eq!(
                    ids(&got),
                    brute_force_ids(q, k),
                    "{precision:?} k={k} oversample={oversample}: and the brute-force ids"
                );
            }
        }

        let after = snapshot(&store);
        assert!(
            served(&b, "SegmentSearch") > ss_before,
            "{precision:?}: B served SegmentSearch"
        );
        if precision == StoragePrecision::Int8 {
            assert!(
                served(&b, "ExactRescore") > er_before,
                "Int8: B served ExactRescore"
            );
        } else {
            assert_eq!(
                served(&b, "ExactRescore"),
                er_before,
                "{precision:?}: one Final phase"
            );
        }
        assert_eq!(
            delta(&before, &after, "local_load"),
            0,
            "{precision:?}: nothing loaded locally"
        );
        assert_eq!(delta(&before, &after, "unavailable"), 0);
        assert_eq!(delta(&before, &after, "unreachable"), 0);

        table.abort().await.unwrap();
        a.close().await;
    }
    readyz_is_200(&b).await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}

// ---------------------------------------------------------------------------
// A caller's width fault never traverses the ladder
// ---------------------------------------------------------------------------

/// A wrong-width query is the CALLER's fault, and the coordinator knows it
/// before it dials anyone. Two halves, both asserting ZERO `SegmentSearch`
/// served by B — the observable that no fan-out happened — and neither
/// panicking:
///
///  * the placed entry on a 4-wide BINARY table, the case that used to PANIC
///    at the coordinator: `pack_threshold_bits` takes `ceil(len/8)` bytes, so
///    an over-long query passes usearch untouched, and the fault surfaced
///    only inside `cosine_distance` — after the owner had refused it
///    (`Refused`), the retry had failed, and the local-load rung had pulled
///    the whole remote segment down;
///  * the public `Search` verb, refused at `QueryBuilder::new` — the first
///    point at which any width is known at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_caller_width_fault_is_refused_before_any_fan_out() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner = PeerAddr(b.peer_addr.to_string());
    let placement = TestPlacement::default();
    let a = open_a(&dir, StoragePrecision::Binary, None, placement.clone()).await;
    let store = a.result_store();

    // --- the placed entry, 4-wide Binary, segment 1 owned by B ---
    let (table, record) = two_segment_table(&store, "src_width", Some(4)).await;
    placement.set(&record.table_name, 1, vec![owner.clone()]);
    let served_before = served(&b, "SegmentSearch");
    let counters_before = snapshot(&store);

    let placed = store.resolve_search_mode(&record).await.unwrap().unwrap();
    let err = placed
        .search_final_placed(&[1.0, 0.0, 0.0, 0.0, 0.0], 3, 4)
        .await
        .expect_err("a 5-wide query on a 4-wide Binary table must be a typed refusal");
    let text = err.to_string();
    assert!(
        text.contains("5 dimensions") && text.contains("4 dimensions"),
        "the refusal names both widths: {text}"
    );
    assert_eq!(
        served(&b, "SegmentSearch"),
        served_before,
        "a caller fault must not reach an owner"
    );
    let counters_after = snapshot(&store);
    for (label, _) in &counters_before {
        assert_eq!(
            delta(&counters_before, &counters_after, label),
            0,
            "a caller fault must not move the ladder counter {label}"
        );
    }

    // --- the public Search verb, refused at QueryBuilder::new ---
    let imported = import_table(&a, &dir, "docs_width").await;
    placement.set(&imported.table_name, 0, vec![owner.clone()]);
    let served_before = served(&b, "SegmentSearch");
    let err = Session::new(Arc::clone(&a))
        .search(search_request(
            "docs_width",
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            1,
        ))
        .await
        .expect_err("a 5-wide query on a 4-wide table must be refused at the entry");
    let text = err.to_string();
    assert!(
        text.contains(&imported.table_name) && text.contains("5 dimensions"),
        "the refusal names the table and the width: {text}"
    );
    assert!(
        !matches!(err, JammiError::Unavailable { .. }),
        "a caller fault is not a peer outage: {err:?}"
    );
    assert_eq!(
        served(&b, "SegmentSearch"),
        served_before,
        "the caller fault never traversed the ladder"
    );
    // The honest query on the same table still serves, through the same owner.
    let hits = Session::new(Arc::clone(&a))
        .search(search_request("docs_width", e(1), 1))
        .await
        .expect("a conforming query still fans out");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert!(
        served(&b, "SegmentSearch") > served_before,
        "…and reached B"
    );

    table.abort().await.unwrap();
    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}

// ---------------------------------------------------------------------------
// A9 — the ladder
// ---------------------------------------------------------------------------// ---------------------------------------------------------------------------
// A9 — the ladder
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ladder_retries_then_loads_locally_or_refuses_unavailable() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner = PeerAddr(b.peer_addr.to_string());
    let dead = dead_addr();
    let q = vec![0.0f32, 0.0, 1.0, 0.0];
    let (k, oversample) = (3usize, 4usize);

    // Budget unset: rung 3 is admitted.
    let placement = TestPlacement::default();
    let a = open_a(&dir, StoragePrecision::F32, None, placement.clone()).await;
    let store = a.result_store();
    let (table, record) = two_segment_table(&store, "src_ladder", Some(4)).await;
    let expected = store
        .resolve_search_mode_local(&record)
        .await
        .unwrap()
        .unwrap()
        .search_final(&q, k, oversample)
        .unwrap();
    readyz_is_200(&b).await;

    // Rung 2: one retry at the next rendezvous candidate, then success.
    placement.set(&record.table_name, 1, vec![dead.clone(), owner.clone()]);
    let c0 = snapshot(&store);
    let got = store
        .resolve_search_mode(&record)
        .await
        .unwrap()
        .unwrap()
        .search_final_placed(&q, k, oversample)
        .await
        .unwrap();
    assert_eq!(got, expected, "one retry leaves the bytes unchanged");
    let c1 = snapshot(&store);
    assert_eq!(delta(&c0, &c1, "unreachable"), 1);
    assert_eq!(delta(&c0, &c1, "retry_ok"), 1);
    assert_eq!(delta(&c0, &c1, "local_load"), 0);
    assert_eq!(delta(&c0, &c1, "unavailable"), 0);

    // Rung 3, admitted (budget unset): both owners dead → local load.
    placement.set(&record.table_name, 1, vec![dead.clone(), dead.clone()]);
    let got = store
        .resolve_search_mode(&record)
        .await
        .unwrap()
        .unwrap()
        .search_final_placed(&q, k, oversample)
        .await
        .unwrap();
    assert_eq!(got, expected, "the local load leaves the bytes unchanged");
    let c2 = snapshot(&store);
    assert_eq!(delta(&c1, &c2, "unreachable"), 2);
    assert_eq!(delta(&c1, &c2, "retry_ok"), 0);
    assert_eq!(delta(&c1, &c2, "local_load"), 1);
    assert_eq!(delta(&c1, &c2, "unavailable"), 0);
    readyz_is_200(&b).await;

    // Rung 4: `peer_local_load_bytes = Some(1)` refuses the load → Unavailable
    // naming `table/1`. A second coordinator over the same dir with the budget.
    let a2 = open_a(&dir, StoragePrecision::F32, Some(1), placement.clone()).await;
    let store2 = a2.result_store();
    let d0 = snapshot(&store2);
    let err = store2
        .resolve_search_mode(&record)
        .await
        .unwrap()
        .unwrap()
        .search_final_placed(&q, k, oversample)
        .await
        .unwrap_err();
    match &err {
        JammiError::Unavailable { resource, reason } => {
            assert_eq!(*resource, format!("segment {}/1", record.table_name));
            assert!(reason.contains("peer_local_load_bytes = 1"), "{reason}");
        }
        other => panic!("expected Unavailable, got {other:?}"),
    }
    let d1 = snapshot(&store2);
    assert_eq!(delta(&d0, &d1, "unreachable"), 2);
    assert_eq!(delta(&d0, &d1, "unavailable"), 1);
    assert_eq!(delta(&d0, &d1, "local_load"), 0);

    // `dimensions = None` on the record skips rung 3 → Unavailable.
    let (table_nodims, record_nodims) = two_segment_table(&store2, "src_nodims", None).await;
    assert_eq!(record_nodims.dimensions, None);
    placement.set(
        &record_nodims.table_name,
        1,
        vec![dead.clone(), dead.clone()],
    );
    let err = store2
        .resolve_search_mode(&record_nodims)
        .await
        .unwrap()
        .unwrap()
        .search_final_placed(&q, k, oversample)
        .await
        .unwrap_err();
    match &err {
        JammiError::Unavailable { resource, reason } => {
            assert_eq!(*resource, format!("segment {}/1", record_nodims.table_name));
            assert!(reason.contains("no dimensions"), "{reason}");
        }
        other => panic!("expected Unavailable, got {other:?}"),
    }

    // `Some(0)` is refused by validate (K2).
    let zero = ServerConfig {
        peer_local_load_bytes: Some(0),
        ..Default::default()
    };
    assert!(zero.validate().is_err());

    // The same refusal over the PUBLIC `Search` verb: `Code::Unavailable`, and
    // the wire detail reconstructs the exact variant.
    let imported = import_table(&a2, &dir, "docs_dead").await;
    placement.set(&imported.table_name, 0, vec![dead.clone(), dead.clone()]);
    let (public_addr, shutdown, handle) = start_engine_server_over_session(Arc::clone(&a2)).await;
    let mut client = EmbeddingServiceClient::new(channel(public_addr).await);
    let status = client
        .search(WireSearchRequest {
            source_id: "docs_dead".into(),
            query: Some(WireSearchQuery::QueryVector(QueryVector { values: e(1) })),
            k: 1,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            oversample: None,
        })
        .await
        .expect_err("the placed table's owners are dead and the budget refuses the load");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    match jammi_wire::error_from_status(&status) {
        JammiError::Unavailable { resource, .. } => {
            assert_eq!(resource, format!("segment {}/0", imported.table_name));
        }
        other => panic!("the detail must round-trip as Unavailable, got {other:?}"),
    }
    readyz_is_200(&b).await;

    let _ = shutdown.send(());
    let _ = handle.await;
    table_nodims.abort().await.unwrap();
    table.abort().await.unwrap();
    a2.close().await;
    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}

// ---------------------------------------------------------------------------
// A7 — tenant scope is enforced at the coordinator, before any fan-out
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn coordinator_under_another_tenant_fails_before_fan_out() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner = PeerAddr(b.peer_addr.to_string());
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();

    let placement = TestPlacement::default();
    let a = open_a(&dir, StoragePrecision::F32, None, placement.clone()).await;
    // Tenant A owns the source and its imported table.
    let record = a
        .with_tenant_scoped(tenant_a, |_| import_table(&a, &dir, "docs_tenant"))
        .await;
    assert_eq!(
        record.tenant_id.as_deref(),
        Some(tenant_a.to_string().as_str())
    );
    placement.set(&record.table_name, 0, vec![owner.clone()]);
    let session = Session::new(Arc::clone(&a));

    // Tenant B: the table does not resolve — no peer call is made.
    let before = served(&b, "SegmentSearch");
    let err = a
        .with_tenant_scoped(tenant_b, |_| {
            session.search(search_request("docs_tenant", e(1), 1))
        })
        .await
        .expect_err("tenant B cannot name tenant A's table");
    assert!(
        !matches!(err, JammiError::Unavailable { .. }),
        "the refusal is the tenant-scoped resolve, not the peer ladder: {err:?}"
    );
    assert_eq!(
        served(&b, "SegmentSearch"),
        before,
        "B's handler was never reached"
    );

    // Tenant A: the same request resolves and fans out to B.
    let batches = a
        .with_tenant_scoped(tenant_a, |_| {
            session.search(search_request("docs_tenant", e(1), 1))
        })
        .await
        .expect("tenant A searches its own table");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 1);
    assert_eq!(
        served(&b, "SegmentSearch"),
        before + 1,
        "the fan-out reached B"
    );

    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}

// ---------------------------------------------------------------------------
// A13 — force-local routing vs. placed routing on ONE store
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn force_local_entries_ignore_placement_while_placed_entries_refuse() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let dead = dead_addr();

    // Budget `Some(1)`: the placed ladder's local-load rung is refused, so a
    // placed entry over an unreachable owner can only be `Unavailable`.
    let placement = TestPlacement::default();
    let a = open_a(&dir, StoragePrecision::F32, Some(1), placement.clone()).await;
    let store = a.result_store();
    let record = import_table(&a, &dir, "docs_local").await;
    placement.set(&record.table_name, 0, vec![dead.clone(), dead.clone()]);

    // Force-local half: the neighbor-graph build (`resolve_strategy` →
    // `resolve_search_mode_local`) and the eval runner's store entry
    // (`search_vectors_local`) complete; every counter delta is 0.
    let c0 = snapshot(&store);
    let (graph, _outcome) = a
        .build_neighbor_graph(
            "docs_local",
            None,
            &jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph {
                k: 2,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect("a batch build never fans out");
    assert_eq!(graph.status, "ready");
    let hits = store
        .search_vectors_local(a.context(), &record, &e(2), 2)
        .await
        .expect("the eval entry is force-local");
    assert_eq!(hits.len(), 2);
    let c1 = snapshot(&store);
    for (label, _) in &c0 {
        assert_eq!(
            delta(&c0, &c1, label),
            0,
            "force-local touched counter {label}"
        );
    }

    // Placed half on the SAME store: the context-set entry and the Search leaf
    // both return `Unavailable`.
    let err = store
        .search_vectors(a.context(), &record, &e(2), 2)
        .await
        .unwrap_err();
    assert!(matches!(err, JammiError::Unavailable { .. }), "{err:?}");
    let err = Session::new(Arc::clone(&a))
        .search(search_request("docs_local", e(2), 2))
        .await
        .unwrap_err();
    assert!(matches!(err, JammiError::Unavailable { .. }), "{err:?}");
    let c2 = snapshot(&store);
    assert!(delta(&c1, &c2, "unreachable") >= 1);
    assert!(delta(&c1, &c2, "unavailable") >= 1);
    assert_eq!(delta(&c1, &c2, "local_load"), 0);

    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}
