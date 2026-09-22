//! `InferenceSession::generate_structure_embeddings`: an embedding table from
//! an edge relation alone.
//!
//! Hermetic: a tempdir session carries a registered edge source
//! (`src`/`dst`) and, where a test hydrates, a node source keyed by
//! `account` — an id-only account graph, the shape the verb exists for. The
//! graphs are generated in the test from a seeded generator: a
//! planted-partition graph for quality, a ring for determinism, a complete
//! bipartite graph and a declared isolated node for the operator's
//! invariants.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use jammi_ai::pipeline::graph_neighbourhood::EdgeSourceRef;
use jammi_ai::pipeline::graph_structure::{StructureRequest, DEFAULT_STRUCTURE_HOP_CAP};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::JammiConfig;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

use crate::common;

/// A small multiplicative generator — enough to plant communities
/// reproducibly, and dependency-free.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// An undirected planted-partition graph: `communities × per_community`
/// nodes, an edge inside a community with probability `p_in`, across with
/// `p_out`. Returns the edges and each node's community.
fn planted_partition(
    communities: usize,
    per_community: usize,
    p_in: f64,
    p_out: f64,
    seed: u64,
) -> (Vec<(String, String)>, HashMap<String, usize>) {
    let mut rng = Lcg(seed);
    let n = communities * per_community;
    let name = |i: usize| format!("acct-{i:04}");
    let community: HashMap<String, usize> = (0..n).map(|i| (name(i), i / per_community)).collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let p = if i / per_community == j / per_community {
                p_in
            } else {
                p_out
            };
            if rng.next_f64() < p {
                edges.push((name(i), name(j)));
            }
        }
    }
    (edges, community)
}

fn ring(n: usize) -> Vec<(String, String)> {
    (0..n)
        .map(|i| (format!("acct-{i:04}"), format!("acct-{:04}", (i + 1) % n)))
        .collect()
}

fn write_parquet(dir: &TempDir, name: &str, schema: Arc<Schema>, batch: RecordBatch) -> String {
    let path = dir.path().join(name);
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    format!("file://{}", path.to_str().unwrap())
}

fn utf8(values: &[String]) -> ArrayRef {
    Arc::new(StringArray::from(
        values.iter().map(String::as_str).collect::<Vec<_>>(),
    ))
}

/// A session over the `edges` source (`src`, `dst`), at `target_partitions`
/// execution threads, plus a `nodes` source (`account`, `segment`) carrying
/// every endpoint and its `community` label.
async fn graph_session(
    edges: &[(String, String)],
    community: &HashMap<String, usize>,
    target_partitions: usize,
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let mut config: JammiConfig = common::test_config(dir.path());
    config.engine.execution_threads = std::num::NonZeroUsize::new(target_partitions).unwrap();
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session.install_query_functions();

    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
    ]));
    let (src, dst): (Vec<String>, Vec<String>) = edges.iter().cloned().unzip();
    let edge_batch =
        RecordBatch::try_new(Arc::clone(&edge_schema), vec![utf8(&src), utf8(&dst)]).unwrap();
    let edge_url = write_parquet(&dir, "edges.parquet", edge_schema, edge_batch);
    add_file_source(&session, "edges", edge_url).await;

    let mut accounts: Vec<String> = edges
        .iter()
        .flat_map(|(s, d)| [s.clone(), d.clone()])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    accounts.sort();
    let segments: Vec<i64> = accounts
        .iter()
        .map(|a| community.get(a).copied().unwrap_or(0) as i64)
        .collect();
    let node_schema = Arc::new(Schema::new(vec![
        Field::new("account", DataType::Utf8, false),
        Field::new("segment", DataType::Int64, false),
    ]));
    let node_batch = RecordBatch::try_new(
        Arc::clone(&node_schema),
        vec![utf8(&accounts), Arc::new(Int64Array::from(segments))],
    )
    .unwrap();
    let node_url = write_parquet(&dir, "nodes.parquet", node_schema, node_batch);
    add_file_source(&session, "nodes", node_url).await;
    (session, dir)
}

async fn add_file_source(session: &InferenceSession, id: &str, url: String) {
    session
        .add_source(
            id,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

fn edges_ref() -> EdgeSourceRef {
    EdgeSourceRef::Registered {
        source_id: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        type_column: None,
        weight_column: None,
        as_of_column: None,
    }
}

/// The default request, on the `nodes` source keyed by `account`.
fn request() -> StructureRequest {
    StructureRequest::new("nodes", edges_ref()).with_key_column("account")
}

async fn encode(session: &Arc<InferenceSession>, request: &StructureRequest) -> ResultTableRecord {
    session
        .generate_structure_embeddings(request, CachePolicy::Bypass)
        .await
        .unwrap()
        .0
}

async fn read_table_vectors(
    session: &Arc<InferenceSession>,
    table: &ResultTableRecord,
) -> HashMap<String, Vec<f32>> {
    let batches = session
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{}\"",
            table.table_name
        ))
        .await
        .unwrap();
    let mut out = HashMap::new();
    for batch in &batches {
        let ids = arrow::compute::cast(batch.column(0), &DataType::Utf8).unwrap();
        let ids = ids.as_any().downcast_ref::<StringArray>().unwrap();
        let list = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            let cell = list.value(i);
            let floats = cell.as_any().downcast_ref::<Float32Array>().unwrap();
            out.insert(ids.value(i).to_string(), floats.values().to_vec());
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// The share of each node's `k` nearest neighbours (by cosine, over the
/// table) that carry its label, averaged over the nodes.
fn nn_same_label_rate(
    vectors: &HashMap<String, Vec<f32>>,
    label: impl Fn(&str) -> usize,
    k: usize,
) -> f64 {
    let keys: Vec<&String> = vectors.keys().collect();
    let mut total = 0.0;
    for key in &keys {
        let mut scored: Vec<(f64, &str)> = keys
            .iter()
            .filter(|other| *other != key)
            .map(|other| (cosine(&vectors[*key], &vectors[*other]), other.as_str()))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let same = scored
            .iter()
            .take(k)
            .filter(|(_, other)| label(other) == label(key))
            .count();
        total += same as f64 / k as f64;
    }
    total / keys.len() as f64
}

#[tokio::test]
async fn planted_communities_are_recovered_well_above_the_seed_and_the_base_rate() {
    // Four communities of sixty: an expected in-community degree of ~9 and a
    // cross-community degree of ~2 — a sparse graph with a clear plant.
    let (edges, community) = planted_partition(4, 60, 0.15, 0.01, 11);
    let (session, _dir) = graph_session(&edges, &community, 2).await;
    let label = |key: &str| community[key];
    let k = 5;
    let base_rate = 59.0 / 239.0;

    let structure = read_table_vectors(&session, &encode(&session, &request()).await).await;
    let seed_only = read_table_vectors(
        &session,
        &encode(&session, &request().with_weights([1.0])).await,
    )
    .await;
    assert_eq!(structure.len(), 240);
    let structure_rate = nn_same_label_rate(&structure, label, k);
    let seed_rate = nn_same_label_rate(&seed_only, label, k);
    println!(
        "planted partition: structure nn@{k} same-community {structure_rate:.3}, \
         seed-only {seed_rate:.3}, base rate {base_rate:.3}"
    );
    // The floors sit well under the measured 0.992 / 0.223 / 0.247: the
    // plant is recovered, the un-propagated seed is chance.
    assert!(
        structure_rate > 0.85,
        "structure recovers the plant: {structure_rate}"
    );
    assert!(seed_rate < 0.35, "the seed alone is chance: {seed_rate}");
    assert!(
        structure_rate > seed_rate + 0.5,
        "propagation is what recovers it: {structure_rate} vs {seed_rate}"
    );
}

#[tokio::test]
async fn output_is_byte_identical_across_partitions_and_edge_row_orders() {
    let (edges, community) = planted_partition(3, 40, 0.2, 0.02, 5);
    let (session_one, _d1) = graph_session(&edges, &community, 1).await;
    let reference = read_table_vectors(&session_one, &encode(&session_one, &request()).await).await;

    let mut reversed = edges.clone();
    reversed.reverse();
    // Every edge declared the other way round, too: the same undirected graph.
    let flipped: Vec<(String, String)> =
        edges.iter().map(|(s, d)| (d.clone(), s.clone())).collect();
    for (label, edges, partitions) in [
        ("four partitions", &edges, 4),
        ("reversed rows", &reversed, 3),
        ("flipped endpoints", &flipped, 4),
    ] {
        let (session, _dir) = graph_session(edges, &community, partitions).await;
        let vectors = read_table_vectors(&session, &encode(&session, &request()).await).await;
        assert_eq!(vectors.len(), reference.len(), "{label}");
        for (key, vector) in &reference {
            assert_eq!(
                bits(vector),
                bits(&vectors[key]),
                "{label}: node {key} differs"
            );
        }
    }
}

#[tokio::test]
async fn adding_a_node_leaves_seed_rows_and_far_nodes_bit_identical() {
    let n = 40;
    let community = HashMap::new();
    let before = ring(n);
    let mut after = before.clone();
    after.push(("acct-0000".into(), "acct-new".into()));
    let two_hops = request().with_weights([0.0, 1.0, 1.0]);
    let seed_only = request().with_weights([1.0]);
    let (s_before, _d1) = graph_session(&before, &community, 2).await;
    let (s_after, _d2) = graph_session(&after, &community, 2).await;

    // The seed rows: every existing node's is unchanged — a row is a function
    // of its key, and at β = 0 of nothing else.
    let seeds_before = read_table_vectors(&s_before, &encode(&s_before, &seed_only).await).await;
    let seeds_after = read_table_vectors(&s_after, &encode(&s_after, &seed_only).await).await;
    for (key, row) in &seeds_before {
        assert_eq!(
            bits(row),
            bits(&seeds_after[key]),
            "seed row of {key} moved"
        );
    }
    assert!(seeds_after.contains_key("acct-new"));

    // The propagated rows: only nodes within K hops of the attachment can
    // move, and the attachment point does. At β = 0 the change reaches a
    // node only through the degree of the one it touched, so the measured
    // radius is K − 1: three rows, the farthest at distance 1.
    let out_before = read_table_vectors(&s_before, &encode(&s_before, &two_hops).await).await;
    let out_after = read_table_vectors(&s_after, &encode(&s_after, &two_hops).await).await;
    let distance = |key: &str| -> usize {
        let i: usize = key["acct-".len()..].parse().unwrap();
        i.min(n - i)
    };
    let moved: Vec<&String> = out_before
        .iter()
        .filter(|(key, row)| bits(row) != bits(&out_after[*key]))
        .map(|(key, _)| key)
        .collect();
    let farthest = moved.iter().map(|key| distance(key)).max().unwrap_or(0);
    println!(
        "node addition under K = 2: {} rows moved, farthest at distance {farthest}",
        moved.len()
    );
    assert!(
        moved.iter().any(|key| key.as_str() == "acct-0000"),
        "the attachment point moved"
    );
    assert!(
        farthest <= 2,
        "a node farther than K hops moved: distance {farthest}"
    );
    assert!(
        out_before.len() - moved.len() > 30,
        "the far ring is untouched ({} of {} moved)",
        moved.len(),
        out_before.len()
    );
}

#[tokio::test]
async fn an_isolated_node_keeps_its_seed_direction() {
    let mut edges = ring(12);
    edges.push(("acct-iso".into(), "acct-iso".into()));
    let (session, _dir) = graph_session(&edges, &HashMap::new(), 2).await;
    let full = read_table_vectors(&session, &encode(&session, &request()).await).await;
    let seed_only = read_table_vectors(
        &session,
        &encode(&session, &request().with_weights([1.0])).await,
    )
    .await;
    assert_eq!(full.len(), 13, "the self-edge declares the node");
    let similarity = cosine(&full["acct-iso"], &seed_only["acct-iso"]);
    assert!(
        similarity > 1.0 - 1e-6,
        "every block of an isolated node is its seed row: cosine {similarity}"
    );
}

#[tokio::test]
async fn a_bipartite_graph_clusters_by_side_under_odd_and_even_blocks_alike() {
    // K₁₅,₁₅: under the paper's operator the walk alternates sides forever,
    // so odd blocks and even blocks disagree about where a node sits. The
    // self-loop makes the walk lazy: both parities place a node with its
    // side.
    let left: Vec<String> = (0..15).map(|i| format!("l-{i:02}")).collect();
    let right: Vec<String> = (0..15).map(|i| format!("r-{i:02}")).collect();
    let edges: Vec<(String, String)> = left
        .iter()
        .flat_map(|l| right.iter().map(move |r| (l.clone(), r.clone())))
        .collect();
    let (session, _dir) = graph_session(&edges, &HashMap::new(), 2).await;
    let side = |key: &str| usize::from(key.starts_with('r'));
    let base_rate = 14.0 / 29.0;
    for (label, weights) in [
        ("odd block", [0.0, 1.0, 0.0]),
        ("even block", [0.0, 0.0, 1.0]),
    ] {
        let vectors = read_table_vectors(
            &session,
            &encode(&session, &request().with_weights(weights)).await,
        )
        .await;
        let rate = nn_same_label_rate(&vectors, side, 5);
        println!("bipartite {label}: nn@5 same-side {rate:.3} (base {base_rate:.3})");
        assert!(rate > 0.95, "{label}: same-side rate {rate}");
    }
}

#[tokio::test]
async fn the_seed_the_exponent_and_the_weights_each_change_the_table() {
    let (edges, community) = planted_partition(2, 30, 0.2, 0.02, 3);
    let (session, _dir) = graph_session(&edges, &community, 2).await;
    let reference = read_table_vectors(&session, &encode(&session, &request()).await).await;
    for (label, request) in [
        ("seed", request().with_seed(1)),
        ("beta", request().with_beta(-0.5)),
        ("weights", request().with_weights([0.0, 0.0, 1.0, 1.0, 2.0])),
        ("sparsity", request().with_sparsity(8.0)),
    ] {
        let vectors = read_table_vectors(&session, &encode(&session, &request).await).await;
        assert!(
            reference
                .iter()
                .any(|(key, row)| bits(row) != bits(&vectors[key])),
            "{label} left every row unchanged"
        );
    }
}

#[tokio::test]
async fn search_by_row_key_ranks_the_node_s_community_and_hydrates_its_source() {
    let (edges, community) = planted_partition(3, 40, 0.2, 0.01, 17);
    let (session, _dir) = graph_session(&edges, &community, 2).await;
    let table = encode(&session, &request()).await;
    assert_eq!(table.key_column.as_deref(), Some("account"));
    assert_eq!(table.model_id, "graph_structure");

    let batches = session
        .search_by_id("nodes", "acct-0007", 5, Some(&table.table_name), None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let schema = batches[0].schema();
    let segment_index = schema
        .index_of("segment")
        .expect("hydrated from the nodes source");
    let mut same = 0;
    let mut total = 0;
    for batch in &batches {
        let segments = batch
            .column(segment_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            total += 1;
            same += usize::from(segments.value(i) == community["acct-0007"] as i64);
        }
    }
    assert!(total >= 4, "a neighbourhood of {total}");
    assert!(
        same * 5 >= total * 4,
        "{same} of {total} neighbours share the community"
    );

    // A vector of another width is refused by the index: there is no encoder
    // for this space, and a foreign vector is not a query in it.
    let refused = session
        .search("nodes", vec![0.1; 7], 5, Some(&table.table_name), None)
        .await;
    let refused = match refused {
        Ok(builder) => builder.run().await.err(),
        Err(e) => Some(e),
    };
    assert!(refused.is_some(), "a 7-wide query against a 256-wide table");
}

#[tokio::test]
async fn invalid_requests_are_typed_refusals() {
    let (edges, community) = planted_partition(2, 10, 0.3, 0.05, 1);
    let (session, _dir) = graph_session(&edges, &community, 1).await;
    let refuse = |request: StructureRequest| {
        let session = Arc::clone(&session);
        async move {
            session
                .generate_structure_embeddings(&request, CachePolicy::Bypass)
                .await
                .unwrap_err()
        }
    };
    for (label, request) in [
        ("zero dimensions", request().with_dimensions(0)),
        ("no weights", request().with_weights(Vec::<f64>::new())),
        ("all-zero weights", request().with_weights([0.0, 0.0])),
        ("a NaN weight", request().with_weights([1.0, f64::NAN])),
        (
            "a depth past the cap",
            request().with_weights(vec![1.0; DEFAULT_STRUCTURE_HOP_CAP + 2]),
        ),
        ("a NaN exponent", request().with_beta(f64::NAN)),
        ("a sparsity below one", request().with_sparsity(0.5)),
    ] {
        let err = refuse(request).await;
        assert!(matches!(err, JammiError::Config(_)), "{label}: {err}");
    }

    let err = refuse(request().with_key_column("customer")).await;
    assert!(
        matches!(&err, JammiError::Schema { column, .. } if column == "customer"),
        "an unknown key column is named: {err}"
    );
    let err = refuse(StructureRequest::new(
        "nodes",
        EdgeSourceRef::Registered {
            source_id: "edges".into(),
            src_column: "from".into(),
            dst_column: "dst".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        },
    ))
    .await;
    assert!(
        matches!(&err, JammiError::Schema { column, .. } if column == "from"),
        "an unknown edge column is named: {err}"
    );
    let err = refuse(StructureRequest::new(
        "nodes",
        EdgeSourceRef::Registered {
            source_id: "ledger".into(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        },
    ))
    .await;
    assert!(
        matches!(err, JammiError::SourceNotFound { .. }),
        "an unknown edge source: {err}"
    );
}

#[tokio::test]
async fn an_empty_graph_is_a_typed_refusal() {
    let (session, _dir) = graph_session(&[], &HashMap::new(), 1).await;
    let err = session
        .generate_structure_embeddings(
            &StructureRequest::new("edges", edges_ref()),
            CachePolicy::Bypass,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, JammiError::Config(_)),
        "no node to embed: {err}"
    );
}

// ─── The adjacency snapshot ──────────────────────────────────────────────────

use jammi_ai::jobs::compute_test_hooks::{self, ParkPoint};
use jammi_db::catalog::result_repo::{ResultTableCas, ResultTableKind};
use jammi_db::catalog::status::ResultTableStatus;

/// Register the session's `edges.parquet` again under `source` — a name one
/// test owns, so the park it arms (parks are keyed by source id, process-wide)
/// is taken by its own run and no other's.
async fn own_edge_source(
    session: &InferenceSession,
    dir: &TempDir,
    source: &str,
) -> StructureRequest {
    let url = format!("file://{}", dir.path().join("edges.parquet").display());
    add_file_source(session, source, url).await;
    StructureRequest::new(
        source,
        EdgeSourceRef::Registered {
            source_id: source.into(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        },
    )
}

fn overwrite_edges(dir: &TempDir, edges: &[(String, String)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
    ]));
    let (src, dst): (Vec<String>, Vec<String>) = edges.iter().cloned().unzip();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![utf8(&src), utf8(&dst)]).unwrap();
    write_parquet(dir, "edges.parquet", schema, batch);
}

/// Every working table the catalog knows, by status.
async fn adjacency_rows(
    session: &InferenceSession,
    status: ResultTableStatus,
) -> Vec<ResultTableRecord> {
    session
        .catalog()
        .list_result_tables_by_status(status)
        .await
        .unwrap()
        .into_iter()
        .filter(|row| row.kind == ResultTableKind::Working)
        .collect()
}

fn bytes_exist(row: &ResultTableRecord) -> bool {
    std::path::Path::new(row.parquet_path.trim_start_matches("file://")).exists()
}

/// No adjacency table is held or promoted, and none that ended left bytes.
async fn assert_no_snapshot_left(session: &InferenceSession, ending: &str) {
    for status in [ResultTableStatus::Building, ResultTableStatus::Ready] {
        let rows = adjacency_rows(session, status).await;
        assert!(
            rows.is_empty(),
            "{ending}: {} snapshot(s) still {status}",
            rows.len()
        );
    }
    for row in adjacency_rows(session, ResultTableStatus::Failed).await {
        assert!(
            !bytes_exist(&row),
            "{ending}: '{}' left its bytes",
            row.table_name
        );
    }
}

#[tokio::test]
async fn an_edge_source_that_moves_mid_run_changes_nothing() {
    let (original, community) = planted_partition(3, 30, 0.2, 0.02, 23);
    // The same accounts, wired differently.
    let (moved, _) = planted_partition(3, 30, 0.05, 0.2, 99);

    let (reference_session, reference_dir) = graph_session(&original, &community, 2).await;
    let request = own_edge_source(&reference_session, &reference_dir, "ledger_reference").await;
    let reference = read_table_vectors(
        &reference_session,
        &encode(&reference_session, &request).await,
    )
    .await;

    // The run parks once its snapshot is written; the source is rewritten
    // under it; every hop — and the degrees — read after that.
    let (session, dir) = graph_session(&original, &community, 2).await;
    let request = own_edge_source(&session, &dir, "ledger_moving").await;
    let parked = compute_test_hooks::arm("ledger_moving", ParkPoint::AfterAdjacencySnapshot);
    let run = {
        let (session, request) = (Arc::clone(&session), request.clone());
        tokio::spawn(async move {
            session
                .generate_structure_embeddings(&request, CachePolicy::Bypass)
                .await
        })
    };
    parked.wait_parked().await;
    overwrite_edges(&dir, &moved);
    parked.release();
    let (table, _) = run.await.unwrap().unwrap();
    let vectors = read_table_vectors(&session, &table).await;
    assert_eq!(vectors.len(), reference.len());
    for (key, row) in &reference {
        assert_eq!(
            bits(row),
            bits(&vectors[key]),
            "node {key} saw the moved source"
        );
    }

    // The control: the moved graph, read from the start, is a different table
    // — so the equality above is the snapshot's doing, not the mutation's
    // invisibility.
    let (moved_session, moved_dir) = graph_session(&moved, &community, 2).await;
    let request = own_edge_source(&moved_session, &moved_dir, "ledger_moved").await;
    let moved_vectors =
        read_table_vectors(&moved_session, &encode(&moved_session, &request).await).await;
    assert!(
        reference
            .iter()
            .any(|(key, row)| bits(row) != bits(&moved_vectors[key])),
        "the moved graph encodes differently"
    );
}

#[tokio::test]
async fn the_adjacency_snapshot_is_gone_after_every_ending() {
    let (edges, community) = planted_partition(2, 20, 0.3, 0.05, 7);

    // Success.
    let (session, dir) = graph_session(&edges, &community, 2).await;
    let request = own_edge_source(&session, &dir, "ledger_success").await;
    encode(&session, &request).await;
    assert_no_snapshot_left(&session, "success").await;

    // Failure: an empty graph is refused after its (empty) snapshot is written.
    let (empty_session, empty_dir) = graph_session(&[], &HashMap::new(), 2).await;
    let request = own_edge_source(&empty_session, &empty_dir, "ledger_failure").await;
    empty_session
        .generate_structure_embeddings(&request, CachePolicy::Bypass)
        .await
        .unwrap_err();
    assert_no_snapshot_left(&empty_session, "failure").await;

    // Cancel: the run is dropped while it holds the snapshot.
    let request = own_edge_source(&session, &dir, "ledger_cancel").await;
    let parked = compute_test_hooks::arm("ledger_cancel", ParkPoint::AfterAdjacencySnapshot);
    let run = {
        let (session, request) = (Arc::clone(&session), request.clone());
        tokio::spawn(async move {
            session
                .generate_structure_embeddings(&request, CachePolicy::Bypass)
                .await
        })
    };
    parked.wait_parked().await;
    let held = adjacency_rows(&session, ResultTableStatus::Building).await;
    assert_eq!(held.len(), 1, "the parked run holds one snapshot");
    assert!(bytes_exist(&held[0]), "and its bytes are written");
    run.abort();
    assert!(run.await.unwrap_err().is_cancelled());
    // The dropped run's snapshot aborts on the runtime that was driving it.
    for _ in 0..500 {
        let held = adjacency_rows(&session, ResultTableStatus::Building).await;
        let failed = adjacency_rows(&session, ResultTableStatus::Failed).await;
        if held.is_empty() && !failed.iter().any(bytes_exist) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_no_snapshot_left(&session, "cancel").await;

    // A process that is gone: its lease runs out under it, and a sweep —
    // any replica's — fails the row and reaps it while the writer is still
    // parked. The run that wakes to a reclaimed snapshot fails; it promotes
    // nothing.
    let request = own_edge_source(&session, &dir, "ledger_lost").await;
    let parked = compute_test_hooks::arm("ledger_lost", ParkPoint::AfterAdjacencySnapshot);
    let run = {
        let (session, request) = (Arc::clone(&session), request.clone());
        tokio::spawn(async move {
            session
                .generate_structure_embeddings(&request, CachePolicy::Bypass)
                .await
        })
    };
    parked.wait_parked().await;
    let held = adjacency_rows(&session, ResultTableStatus::Building).await;
    assert_eq!(held.len(), 1);
    session
        .catalog()
        .expire_lease_for_test(&ResultTableCas::writer(
            &held[0].table_name,
            session.result_store().writer_id(),
            None,
        ))
        .await
        .unwrap();
    session.result_store().recover().await.unwrap();
    assert_no_snapshot_left(&session, "a lost process").await;
    parked.release();
    run.await
        .unwrap()
        .expect_err("a run whose snapshot was reclaimed under it lands nothing");
    assert_no_snapshot_left(&session, "a lost process, after its run woke").await;
}
