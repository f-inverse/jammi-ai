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
use jammi_db::config::{AnnIndexConfig, ServerConfig, StoragePrecision};
use jammi_db::error::JammiError;
use jammi_db::index::peer::{
    ExactRescoreRequest as DomainExactRescoreRequest, PeerAddr, PeerError, PeerTransport,
    SegmentPlacement, SegmentSearchRequest as DomainSegmentSearchRequest, SegmentUnit,
};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::{validate_query, QuerySource, SegmentId, VectorIndex};
use jammi_db::model_task::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{
    ComputeDevice, ComputePrecision, Materialization, MaterializationEnv, ModelContentDigest,
    ModelIdentity, ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{BuildingTable, ResultStore};
use jammi_db::TenantId;
use jammi_numerics::distance::cosine_distance;
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::search_request::Query as WireSearchQuery;
use jammi_server::grpc::proto::embedding::{QueryVector, SearchRequest as WireSearchRequest};
use jammi_server::grpc::wire::map_engine_error;
use jammi_test_utils::vq;
use jammi_wire::peer::GrpcPeerTransport;
use jammi_wire::request::{SearchQuery, SearchRequest};
use parquet::arrow::ArrowWriter;
use std::time::Duration;
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

fn dir_of_session(a: &Arc<InferenceSession>) -> std::path::PathBuf {
    a.inner_config().artifact_dir.clone()
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
        .map(|(id, v)| (id.to_string(), cosine_distance(&vq(query), v)))
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
                let got = placed
                    .search_final_placed(&vq(q), k, oversample)
                    .await
                    .unwrap();
                let all_local = store
                    .resolve_search_mode_local(&record)
                    .await
                    .unwrap()
                    .unwrap()
                    .search_final(&vq(q), k, oversample)
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
        .search_final_placed(&vq(&[1.0, 0.0, 0.0, 0.0, 0.0]), 3, 4)
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
    // A CALLER fault is the schema class (`InvalidArgument` on the wire) and
    // names both widths; only a STORED fault names the table.
    assert!(
        matches!(&err, JammiError::Schema { .. })
            && text.contains("5 dimensions")
            && text.contains("4 dimensions"),
        "the refusal is typed and names both widths: {text}"
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
// A2 — an ALL-REMOTE placement never fans a caller fault out
// ---------------------------------------------------------------------------

/// The shape with no local segment: nothing at the coordinator holds an index,
/// so every width and finiteness check must happen at the ENTRY, from the
/// catalog. A NaN component through the public `Search` verb, a wrong-width
/// query at the placed entry, and a table with NO width on record are each a
/// typed refusal with ZERO `SegmentSearch` served and every ladder counter at
/// its previous value — `retry_ok`, `local_load`, `unavailable`, `torn`
/// included.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_remote_placement_refuses_a_caller_fault_before_any_fan_out() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner = PeerAddr(b.peer_addr.to_string());
    let placement = TestPlacement::default();
    let a = open_a(&dir, StoragePrecision::F32, None, placement.clone()).await;
    let store = a.result_store();
    let counters_before = snapshot(&store);
    let served_before = served(&b, "SegmentSearch");

    // (i) The public verb, NaN component, the one imported segment owned by B.
    let imported = import_table(&a, &dir, "docs_nan").await;
    placement.set(&imported.table_name, 0, vec![owner.clone()]);
    let mut q = e(1);
    q[2] = f32::NAN;
    let err = Session::new(Arc::clone(&a))
        .search(search_request("docs_nan", q, 1))
        .await
        .expect_err("a NaN component is refused at the entry");
    assert!(matches!(&err, JammiError::Schema { .. }), "{err:?}");
    assert!(!matches!(&err, JammiError::Unavailable { .. }));

    // (ii) The placed entry, every segment remote, no width on record.
    let (table_nodims, record_nodims) = two_segment_table(&store, "src_remote_nodims", None).await;
    placement.set(&record_nodims.table_name, 0, vec![owner.clone()]);
    placement.set(&record_nodims.table_name, 1, vec![owner.clone()]);
    let err = store
        .resolve_search_mode(&record_nodims)
        .await
        .unwrap()
        .unwrap()
        .search_final_placed(&vq(&[1.0, 0.0, 0.0, 0.0]), 3, 4)
        .await
        .expect_err("no width on record and no local segment → refuse, never fan out");
    assert!(
        matches!(&err, JammiError::Schema { column, .. } if column == "dimensions"),
        "{err:?}"
    );

    // (iii) The placed entry, every segment remote, width on record, 5-wide query.
    let (table_w, record_w) = two_segment_table(&store, "src_remote_width", Some(4)).await;
    placement.set(&record_w.table_name, 0, vec![owner.clone()]);
    placement.set(&record_w.table_name, 1, vec![owner.clone()]);
    let err = store
        .resolve_search_mode(&record_w)
        .await
        .unwrap()
        .unwrap()
        .search_final_placed(&vq(&[1.0, 0.0, 0.0, 0.0, 0.0]), 3, 4)
        .await
        .expect_err("a 5-wide query against a recorded 4 is refused before fan-out");
    assert!(matches!(&err, JammiError::Schema { .. }), "{err:?}");

    assert_eq!(
        served(&b, "SegmentSearch"),
        served_before,
        "nothing was fanned out"
    );
    let counters_after = snapshot(&store);
    for (label, _) in &counters_before {
        assert_eq!(
            delta(&counters_before, &counters_after, label),
            0,
            "counter {label} moved"
        );
    }
    // The honest all-remote search on the same table still fans out.
    let hits = store
        .resolve_search_mode(&record_w)
        .await
        .unwrap()
        .unwrap()
        .search_final_placed(&vq(&[1.0, 0.0, 0.0, 0.0]), 3, 4)
        .await
        .expect("a conforming all-remote search serves");
    assert_eq!(hits.len(), 3);
    assert!(served(&b, "SegmentSearch") > served_before);

    table_w.abort().await.unwrap();
    table_nodims.abort().await.unwrap();
    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}

// ---------------------------------------------------------------------------
// A5 — a STORED vector with a non-finite component is a corrupt artifact
// ---------------------------------------------------------------------------

/// Plant a `ready` 4-wide embedding table for `source_id` whose row `row-1`
/// carries `poison` in its first component — written straight to the
/// building table's parquet and promoted through the real `finish` (the
/// import verb refuses a non-finite norm, so this is the only way such a row
/// can exist: a corrupted artifact, not an engine write).
async fn ready_table_with_poisoned_row(
    a: &Arc<InferenceSession>,
    source_id: &str,
    poison: Option<f32>,
) -> ResultTableRecord {
    // The SOURCE the result table hydrates against (keyed by `_row_id`).
    let source_path = dir_of_session(a).join(format!("{source_id}.parquet"));
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_row_id", DataType::Utf8, false),
            Field::new("body", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["row-0", "row-1", "row-2"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["b0", "b1", "b2"])) as ArrayRef,
            ],
        )
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&source_path).unwrap(), schema, None)
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    a.add_source(
        source_id,
        SourceType::File,
        SourceConnection {
            url: Some(format!("file://{}", source_path.display())),
            format: Some(FileFormat::Parquet),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let store = a.result_store();
    let building = store
        .create_table(
            source_id,
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(4),
            Some("_row_id"),
            Some("body"),
            None,
        )
        .await
        .unwrap();
    let schema = embedding_table_schema(4);
    let rows: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [poison.unwrap_or(0.5), 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ];
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vectors =
        FixedSizeListArray::try_new(item, 4, Arc::new(Float32Array::from(flat)), None).unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["row-0", "row-1", "row-2"])) as ArrayRef,
            Arc::new(StringArray::from(vec![source_id; 3])),
            Arc::new(StringArray::from(vec!["test-model"; 3])),
            Arc::new(vectors),
            jammi_db::store::content_hash::null_hash_column(3),
        ],
    )
    .unwrap();
    let mut writer = store
        .open_writer(building.parquet_url(), schema)
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    let n = writer.close().await.unwrap();
    let descriptor = ProducingDescriptor::Embedding {
        model_id: "test-model".into(),
        task: ModelTask::TextEmbedding,
        source_id: source_id.into(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: 4,
    };
    let env = MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "test-model".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("it-fixture-digest".into()),
            quantization: None,
        }],
    );
    building
        .finish(
            a.context(),
            n,
            Materialization::new(&descriptor, &env, vec![]),
        )
        .await
        .unwrap()
}

/// A5 — `search_by_id` reads its query back from the table, so a non-finite
/// component there is a corrupt ARTIFACT named by the table (gRPC
/// `Internal`), never the caller's `InvalidArgument`. It is refused at the
/// entry: no fan-out, no counter moves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_by_id_on_a_poisoned_stored_vector_is_a_corrupt_artifact_named_by_the_table() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner = PeerAddr(b.peer_addr.to_string());
    let placement = TestPlacement::default();
    let a = open_a(&dir, StoragePrecision::F32, None, placement.clone()).await;
    let served_before = served(&b, "SegmentSearch");
    for (i, poison) in [f32::NAN, f32::INFINITY].into_iter().enumerate() {
        let source_id = format!("docs_stored_poison_{i}");
        let record = ready_table_with_poisoned_row(&a, &source_id, Some(poison)).await;
        assert_eq!(record.status, "ready");
        // Whatever segment placement says, the refusal must come first.
        placement.set(&record.table_name, 0, vec![owner.clone()]);
        let err = match a.search_by_id(&source_id, "row-1", 1, None, None).await {
            Err(err) => err,
            Ok(_) => panic!("{poison:?}: a stored non-finite vector must be refused at the entry"),
        };
        match &err {
            JammiError::IncompatibleFormat { artifact, .. } => {
                assert!(
                    artifact.contains(&record.table_name),
                    "names the table: {artifact}"
                )
            }
            other => panic!("{poison:?}: expected the corrupt-artifact variant, got {other:?}"),
        }
        let status = jammi_server::grpc::wire::map_engine_error(err);
        assert_eq!(status.code(), Code::Internal, "{status:?}");
        // An HONEST self-query on the SAME table is refused too — the exact
        // path scans the poisoned row and sink B-c refuses the table as a
        // corrupt artifact (typed, table-named, recovered through the plan
        // boundary), never a top-k over a silently dropped row.
        let err = a
            .search_by_id(&source_id, "row-0", 1, None, None)
            .await
            .expect("an honest stored vector validates as a self-query")
            .run()
            .await
            .expect_err("a table with a corrupt row is refused at the sink");
        match &err {
            JammiError::IncompatibleFormat {
                artifact, found, ..
            } => {
                assert!(artifact.contains(&record.table_name), "{artifact}");
                assert!(found.contains("row-1"), "names the corrupt row: {found}");
            }
            other => panic!("{poison:?}: sink B-c must surface typed, got {other:?}"),
        }
    }
    // …and on a CLEAN table the honest self-query serves: `Stored` provenance
    // refuses corruption, not ordinary stored vectors.
    let clean = ready_table_with_poisoned_row(&a, "docs_stored_clean", None).await;
    placement.set(&clean.table_name, 0, vec![owner.clone()]);
    let hits = a
        .search_by_id("docs_stored_clean", "row-0", 1, None, None)
        .await
        .expect("an honest stored vector is a valid self-query")
        .run()
        .await
        .expect("the self-query runs");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert_eq!(
        served(&b, "SegmentSearch"),
        served_before,
        "a corrupt stored vector never fans out"
    );
    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}

// ---------------------------------------------------------------------------
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
        .search_final(&vq(&q), k, oversample)
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
        .search_final_placed(&vq(&q), k, oversample)
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
        .search_final_placed(&vq(&q), k, oversample)
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
        .search_final_placed(&vq(&q), k, oversample)
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
        .search_final_placed(&vq(&q), k, oversample)
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
        .search_vectors_local(a.context(), &record, &vq(&e(2)), 2)
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
        .search_vectors(a.context(), &record, &vq(&e(2)), 2)
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

// ---------------------------------------------------------------------------
// D1 — a REAL owner's INVALID_ARGUMENT is TERMINAL, classified from a REAL
// `Status` by the REAL `classify_status`, never a synthetic enum injected
// into a fake owner.
// ---------------------------------------------------------------------------

/// Wraps a real transport and clears `segment_ids` before forwarding a
/// `SegmentSearch` — the one request shape this coordinator's own
/// construction never produces (an owner group is never built empty),
/// modelling an engine invariant violation. Exercises the REAL wire: the
/// real [`GrpcPeerTransport`] really encodes this request, the real
/// `PeerServer` really refuses it at its own input edge, and
/// `classify_status` really derives `CallerFault` from the real `Status`
/// that comes back — never an enum handed straight to a fake owner.
struct CorruptRequestTransport(GrpcPeerTransport);

#[tonic::async_trait]
impl PeerTransport for CorruptRequestTransport {
    async fn segment_search(
        &self,
        owner: &PeerAddr,
        req: &DomainSegmentSearchRequest,
        deadline: Duration,
    ) -> std::result::Result<Vec<SegmentUnit>, PeerError> {
        let mut corrupted = req.clone();
        corrupted.segment_ids.clear();
        self.0.segment_search(owner, &corrupted, deadline).await
    }

    async fn exact_rescore(
        &self,
        owner: &PeerAddr,
        req: &DomainExactRescoreRequest,
        deadline: Duration,
    ) -> std::result::Result<Vec<(String, f32)>, PeerError> {
        self.0.exact_rescore(owner, req, deadline).await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_owner_caller_fault_is_terminal_and_classified_from_a_real_status() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner_addr = PeerAddr(b.peer_addr.to_string());
    let placement = TestPlacement::default();
    let a = open_a(&dir, StoragePrecision::F32, None, placement.clone()).await;
    let store = a.result_store();
    let (table, record) = two_segment_table(&store, "src_real_caller_fault", Some(4)).await;
    placement.set(&record.table_name, 1, vec![owner_addr.clone()]);

    // The SAME catalog/segment cache/placement, a DIFFERENT transport: the
    // only seam this test swaps.
    let corrupting_store = (*store)
        .clone()
        .with_peer_transport(Arc::new(CorruptRequestTransport(GrpcPeerTransport::new())));

    let served_before = served(&b, "SegmentSearch");
    let counters_before = snapshot(&store);

    let placed = corrupting_store
        .resolve_search_mode(&record)
        .await
        .unwrap()
        .unwrap();
    let err = placed
        .search_final_placed(&vq(&[1.0, 0.0, 0.0, 0.0]), 3, 1)
        .await
        .expect_err(
            "a corrupted (empty segment list) request a real owner refuses is a terminal \
             caller fault",
        );
    match &err {
        JammiError::Other(msg) => {
            assert!(msg.contains(&owner_addr.0), "names the owner: {msg}");
            assert!(msg.contains(&record.table_name), "names the table: {msg}");
            // The corrupted request names no segment (that IS the
            // malformation), so `PeerError`'s "first requested id"
            // convention has nothing to report and falls back to `-1` —
            // still a typed, present field, never an absent one.
            assert!(msg.contains("segment -1"), "names the segment: {msg}");
        }
        other => panic!("expected JammiError::Other naming the owner and segment, got {other:?}"),
    }
    assert_eq!(
        map_engine_error(err).code(),
        Code::Internal,
        "a coordinator-built malformed request is an engine fault, never billed to the caller \
         as InvalidArgument"
    );

    // Exactly one call reached the real owner — no retry at the next
    // candidate, no local load, no `Unavailable`.
    assert_eq!(served(&b, "SegmentSearch"), served_before + 1);
    let counters_after = snapshot(&store);
    assert_eq!(delta(&counters_before, &counters_after, "caller_fault"), 1);
    for label in [
        "retry_ok",
        "local_load",
        "unavailable",
        "torn",
        "unreachable",
        "refused",
    ] {
        assert_eq!(
            delta(&counters_before, &counters_after, label),
            0,
            "{label} must not move"
        );
    }

    table.abort().await.unwrap();
    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}

// ---------------------------------------------------------------------------
// O3 — a query fans out to a WIDTH-DRIFTED remote segment: the real owner
// answers FAILED_PRECONDITION (own-data — a segment whose width has drifted
// from the table's recorded `dimensions`), the ladder exhausts (one owner,
// no retry candidate) and COMPLETES at rung 3 (a generous budget), where a
// local load reproduces the SAME provenance-aware width refusal the
// coordinator's local kernels apply everywhere else: `Stored` names the
// table (`Internal`); `Caller` blames the query (`InvalidArgument`). Pins
// both arms of F2's remedy with zero provenance threading through the
// terminal `caller_fault` arm — the class arrives via the ladder, not the
// terminal path.
// ---------------------------------------------------------------------------

/// A one-segment [`SidecarIndex`] built at `width` dimensions — plants a
/// segment whose width has drifted from the table's recorded `dimensions`: a
/// real artifact defect (e.g. a schema change an append path failed to
/// reconcile), never a synthetic error.
fn built_index_at_width(
    rows: &[(&str, Vec<f32>)],
    width: usize,
    precision: StoragePrecision,
) -> SidecarIndex {
    let mut idx = SidecarIndex::new(width, &AnnIndexConfig::default(), precision).unwrap();
    for (id, v) in rows {
        idx.add(id, v).unwrap();
    }
    idx.build().unwrap();
    idx
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stored_width_drift_answered_by_an_owner_ladders_to_a_named_refusal() {
    let b = start_engine_server_with_peer_bind().await;
    let dir = dir_of(&b);
    let owner = PeerAddr(b.peer_addr.to_string());
    let placement = TestPlacement::default();
    // No budget: the ladder must be permitted to COMPLETE at rung 3 — the
    // class now arrives through the local load, not the terminal arm.
    let a = open_a(&dir, StoragePrecision::F32, None, placement.clone()).await;
    let store = a.result_store();

    // Segment 0: local, 4-wide — matches the table's recorded `dimensions`.
    // Segment 1: REMOTE, drifted to 5-wide — a real artifact defect the
    // coordinator has no local index to catch before fan-out (that check
    // only exists in the all-remote shape).
    let table = store
        .create_table(
            "src_o3_drift",
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
    let precision = table.storage_precision();
    assert_eq!(precision, StoragePrecision::F32);
    assert_eq!(
        table
            .append_segment(&built_index(&ROWS, precision))
            .await
            .unwrap()
            .0,
        0
    );
    let drifted: [(&str, Vec<f32>); 2] = [
        ("z0", vec![0.2, 0.2, 0.2, 0.2, 0.2]),
        ("z1", vec![0.1, 0.1, 0.1, 0.1, 0.1]),
    ];
    assert_eq!(
        table
            .append_segment(&built_index_at_width(&drifted, 5, precision))
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
    placement.set(&record.table_name, 1, vec![owner.clone()]);

    let placed = store.resolve_search_mode(&record).await.unwrap().unwrap();

    // --- Stored provenance: the query was READ BACK from storage (the same
    //     provenance `InferenceSession::search_by_id` builds), so a fault is
    //     a corrupt artifact named by the table — `Internal`.
    let served_before = served(&b, "SegmentSearch");
    let counters_before = snapshot(&store);
    let stored_query = validate_query(
        vec![1.0, 0.0, 0.0, 0.0],
        None,
        QuerySource::Stored {
            table: record.table_name.clone(),
        },
    )
    .unwrap();
    let err = placed
        .search_final_placed(&stored_query, 3, 4)
        .await
        .expect_err(
            "a stored query fanned out to a width-drifted owner must ladder to a named refusal",
        );
    match &err {
        JammiError::IncompatibleFormat { artifact, .. } => {
            assert!(
                artifact.contains(&record.table_name),
                "names the table: {artifact}"
            )
        }
        other => panic!("expected IncompatibleFormat (Stored provenance), got {other:?}"),
    }
    assert_eq!(map_engine_error(err).code(), Code::Internal);
    // The real owner WAS reached (and refused own-data) before the ladder
    // fell to the local load — never the terminal caller-fault arm.
    assert!(
        served(&b, "SegmentSearch") > served_before,
        "the owner answered before the local load"
    );
    let counters_after = snapshot(&store);
    assert_eq!(
        delta(&counters_before, &counters_after, "refused"),
        1,
        "the owner's FAILED_PRECONDITION ladders as Refused, not terminal"
    );
    assert_eq!(
        delta(&counters_before, &counters_after, "caller_fault"),
        0,
        "never terminal"
    );

    // --- Caller-provenance twin, the SAME drifted table: the SAME
    //     mechanism (the local rung-3 width check) classifies a caller's
    //     own query `InvalidArgument` — the class is decided by provenance
    //     at the point of failure, with zero threading through the
    //     terminal `caller_fault` arm.
    let served_before = served(&b, "SegmentSearch");
    let caller_query = vq(&[1.0, 0.0, 0.0, 0.0]);
    let err = placed
        .search_final_placed(&caller_query, 3, 4)
        .await
        .expect_err(
            "a caller query fanned out to a width-drifted owner must ladder to a named refusal",
        );
    assert!(
        matches!(&err, JammiError::Schema { column, .. } if column == "query"),
        "{err:?}"
    );
    assert_eq!(map_engine_error(err).code(), Code::InvalidArgument);
    assert!(served(&b, "SegmentSearch") > served_before);

    table.abort().await.unwrap();
    a.close().await;
    let _ = b.shutdown.send(());
    let _ = b.handle.await;
}
