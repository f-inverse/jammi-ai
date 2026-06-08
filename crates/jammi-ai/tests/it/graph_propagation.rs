//! `InferenceSession::propagate_embeddings` (spec S12): the decoupled-GNN
//! forward pass over a declared graph.
//!
//! Hermetic: a tempdir session carries a synthetic node embedding table (a
//! distinct vector per node, keyed by `_row_id`) and a registered external edge
//! source (`src`/`dst`[/`weight`][/`tenant_id`]). The node ids are the edge
//! endpoints and the embedding keys, so a propagated row aggregates through the
//! same shared vector-aggregation UDAF an ANN context pools through. No consumer
//! vocabulary appears — the fixtures are a neutral citation-/co-purchase-style
//! graph.
//!
//! The tests assert the contracts the spec bakes in: self-loop correctness
//! (isolated node → `X⁽⁰⁾`), the homophily gain / heterophily-loss directional
//! pair, oversmoothing control via the `α`-restart, determinism across two
//! `target_partitions` settings, the edge-similarity clamp, the hop cap, the
//! load-bearing cross-tenant exclusion, the Jumping-Knowledge output shape, and
//! `derived_from` lineage + a re-graphable / R1-evaluable output.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use jammi_ai::pipeline::graph_neighbourhood::EdgeDirection;
use jammi_ai::pipeline::graph_propagation::{
    PropagateRequest, PropagationOutput, PropagationWeighting,
};
use jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::TenantId;

use crate::common;

const DIM: usize = 16;

/// A node fixture: id (the embedding key + edge endpoint) and a class index that
/// drives both its base feature vector and the (homophilous / heterophilous)
/// edge wiring.
struct Node {
    id: String,
    class: usize,
}

/// A declared edge: endpoints, an optional weight, an optional tenant tag (for
/// the cross-tenant leak fixture).
struct Edge {
    src: String,
    dst: String,
    weight: Option<f64>,
    tenant: Option<TenantId>,
}

impl Edge {
    fn plain(src: &str, dst: &str) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            weight: None,
            tenant: None,
        }
    }
    fn weighted(src: &str, dst: &str, w: f64) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            weight: Some(w),
            tenant: None,
        }
    }
}

/// A deterministic per-class centroid plus substantial per-node noise, so that
/// raw features are NOT trivially class-separable (averaging within a class
/// *denoises* toward the centroid — homophily gain — and averaging across
/// classes *collapses* the separation — heterophily loss). Both the centroid and
/// the noise are smooth functions of the inputs, so the whole fixture is
/// reproducible.
fn node_vector(id: &str, class: usize) -> Vec<f32> {
    let id_hash = id
        .bytes()
        .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    (0..DIM)
        .map(|i| {
            // A spread-out class centroid (not one-hot), so denoising/collapse
            // both move the whole vector, not a single lane.
            let centroid = (((class as f32 + 1.0) * (i as f32 + 1.0)) * 0.7).sin();
            // Per-node noise at the same scale as the inter-class centroid gap,
            // so raw nearest-neighbour class agreement is high-but-imperfect and
            // measurably moves under propagation.
            let noise = (((id_hash.wrapping_add((i as u32).wrapping_mul(2_654_435_761))) % 1000)
                as f32
                / 1000.0
                - 0.5)
                * 0.9;
            centroid + noise
        })
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

/// Stand up a session with a configured `target_partitions` (DataFusion
/// execution-thread count) so the determinism test can vary the partitioning.
async fn graph_session_with_partitions(
    nodes: &[Node],
    edges: &[Edge],
    tenant: Option<TenantId>,
    target_partitions: usize,
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let mut config: JammiConfig = common::test_config(dir.path());
    config.engine.execution_threads = target_partitions;
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session.register_query_functions();
    if let Some(t) = tenant {
        session.bind_tenant(t);
    }

    // nodes parquet: _row_id, class.
    let node_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("class", DataType::Int64, false),
    ]));
    let node_batch = RecordBatch::try_new(
        Arc::clone(&node_schema),
        vec![
            Arc::new(StringArray::from(
                nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(arrow::array::Int64Array::from(
                nodes.iter().map(|n| n.class as i64).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let node_url = write_parquet(&dir, "nodes.parquet", node_schema, node_batch);
    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Embedding table keyed by _row_id, vector = node_vector(id, class).
    let pairs: Vec<(String, Vec<f32>)> = nodes
        .iter()
        .map(|n| (n.id.clone(), node_vector(&n.id, n.class)))
        .collect();
    session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            "nodes",
            "synthetic-embed",
            None,
            &pairs,
            DIM,
        )
        .await
        .unwrap();

    // edges parquet: src, dst, weight, tenant_id.
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
        Field::new("weight", DataType::Float64, true),
        Field::new("tenant_id", DataType::Utf8, true),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(StringArray::from(
                edges.iter().map(|e| e.src.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                edges.iter().map(|e| e.dst.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                edges.iter().map(|e| e.weight).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                edges
                    .iter()
                    .map(|e| e.tenant.map(|t| t.to_string()))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let edge_url = write_parquet(&dir, "edges.parquet", edge_schema, edge_batch);
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    (session, dir)
}

/// Single-thread default for tests that do not vary the partitioning.
async fn graph_session(
    nodes: &[Node],
    edges: &[Edge],
    tenant: Option<TenantId>,
) -> (Arc<InferenceSession>, TempDir) {
    graph_session_with_partitions(nodes, edges, tenant, 1).await
}

/// A registered, undirected edge gather over the `edges` source, pinned to the
/// `nodes` source's *original* embedding table.
///
/// The pin is load-bearing for any test that propagates more than once on a
/// session: a propagated table is itself a `kind=Model` embedding table for the
/// `nodes` source, so `resolve_embedding_table("nodes", None)` would otherwise
/// resolve a later propagation's INPUT to an earlier propagation's OUTPUT.
/// Undirected so a symmetric `Â` is meaningful (APPNP/SGC assume undirected
/// adjacency).
async fn registered_request(session: &Arc<InferenceSession>) -> PropagateRequest {
    // Pin to the ORIGINAL synthetic-embed table (by its model id), never the
    // latest — the latest may be an earlier propagation's output.
    let source_table = session
        .catalog()
        .find_result_tables("nodes", None, Some("synthetic-embed"))
        .await
        .unwrap()
        .into_iter()
        .next()
        .expect("the synthetic source embedding table");
    PropagateRequest::new(
        "nodes",
        jammi_ai::pipeline::graph_neighbourhood::EdgeSourceRef::Registered {
            source_id: "edges".into(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            type_column: None,
            weight_column: Some("weight".into()),
            as_of_column: None,
        },
    )
    .with_embedding_table(source_table.table_name)
    .with_direction(EdgeDirection::Undirected)
}

/// Read a materialised embedding table's `(_row_id, vector)` rows back into a map.
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
            out.insert(
                ids.value(i).to_string(),
                (0..floats.len()).map(|j| floats.value(j)).collect(),
            );
        }
    }
    out
}

/// Two equal-size classes, fully wired within class (homophilous) — a clean
/// signal-sharing graph the propagated mean should denoise.
fn two_class_homophilous(per_class: usize) -> (Vec<Node>, Vec<Edge>) {
    let mut nodes = Vec::new();
    for class in 0..2 {
        for i in 0..per_class {
            nodes.push(Node {
                id: format!("c{class}_{i}"),
                class,
            });
        }
    }
    let mut edges = Vec::new();
    for class in 0..2 {
        for i in 0..per_class {
            for j in (i + 1)..per_class {
                edges.push(Edge::plain(
                    &format!("c{class}_{i}"),
                    &format!("c{class}_{j}"),
                ));
            }
        }
    }
    (nodes, edges)
}

/// The rate at which each node's nearest neighbour (by cosine, over the given
/// vectors, excluding itself) shares its class label. The structure-respect
/// metric `eval_embeddings` would compute against a golden set, here measured
/// directly and deterministically.
fn nn_same_class_rate(
    vectors: &HashMap<String, Vec<f32>>,
    class_of: &HashMap<String, usize>,
) -> f64 {
    let entries: Vec<(&String, &Vec<f32>)> = vectors.iter().collect();
    let mut agree = 0usize;
    for (id, v) in &entries {
        let mut best: Option<(&String, f32)> = None;
        for (other_id, ov) in &entries {
            if other_id == id {
                continue;
            }
            let d = jammi_numerics::distance::cosine_distance(v, ov);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((other_id, d));
            }
        }
        if let Some((nn, _)) = best {
            if class_of[*id] == class_of[nn] {
                agree += 1;
            }
        }
    }
    agree as f64 / entries.len() as f64
}

/// Class separation: mean inter-class cosine distance minus mean intra-class
/// cosine distance over the vectors. Larger = better-separated classes.
/// Homophilous propagation (within-class averaging) denoises and *increases* it;
/// heterophilous propagation (cross-class averaging) collapses it toward zero or
/// negative.
fn class_separation(vectors: &HashMap<String, Vec<f32>>, class_of: &HashMap<String, usize>) -> f64 {
    let entries: Vec<(&String, &Vec<f32>)> = vectors.iter().collect();
    let (mut intra_sum, mut intra_n) = (0.0f64, 0u64);
    let (mut inter_sum, mut inter_n) = (0.0f64, 0u64);
    for a in 0..entries.len() {
        for b in (a + 1)..entries.len() {
            let d = jammi_numerics::distance::cosine_distance(entries[a].1, entries[b].1) as f64;
            if class_of[entries[a].0] == class_of[entries[b].0] {
                intra_sum += d;
                intra_n += 1;
            } else {
                inter_sum += d;
                inter_n += 1;
            }
        }
    }
    inter_sum / inter_n.max(1) as f64 - intra_sum / intra_n.max(1) as f64
}

/// Effective rank via singular-value (spectral) entropy: `exp(H)` where
/// `H = −Σ pᵢ ln pᵢ` over the normalised eigenvalues `pᵢ` of the centered
/// feature covariance matrix (the squared singular values of the data matrix).
/// A collapsed (over-smoothed) representation concentrates its energy in one
/// direction → low effective rank. The covariance is `DIM×DIM`; its eigenvalues
/// come from a symmetric Jacobi sweep.
fn effective_rank(vectors: &HashMap<String, Vec<f32>>) -> f64 {
    let rows: Vec<&Vec<f32>> = vectors.values().collect();
    let n = rows.len() as f64;
    let dim = rows[0].len();
    // Column means.
    let mean: Vec<f64> = (0..dim)
        .map(|j| rows.iter().map(|r| r[j] as f64).sum::<f64>() / n)
        .collect();
    // Covariance C[i][j] = mean_k (x_ki - mean_i)(x_kj - mean_j).
    let mut cov = vec![vec![0.0f64; dim]; dim];
    for r in &rows {
        for i in 0..dim {
            for j in 0..dim {
                cov[i][j] += (r[i] as f64 - mean[i]) * (r[j] as f64 - mean[j]);
            }
        }
    }
    for row in cov.iter_mut() {
        for c in row.iter_mut() {
            *c /= n;
        }
    }
    let eigs = jacobi_eigenvalues(cov);
    let total: f64 = eigs.iter().filter(|&&e| e > 0.0).sum();
    if total <= 0.0 {
        return 1.0;
    }
    let entropy: f64 = eigs
        .iter()
        .map(|&e| e / total)
        .filter(|&p| p > 1e-12)
        .map(|p| -p * p.ln())
        .sum();
    entropy.exp()
}

/// Eigenvalues of a symmetric matrix via the cyclic Jacobi method. Small
/// (`DIM×DIM`) so a fixed sweep count converges; returns the diagonal after
/// the off-diagonal mass is rotated away.
fn jacobi_eigenvalues(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..100 {
        // Largest off-diagonal magnitude (the pivot `(p, q)` to rotate away).
        let (mut p, mut q, mut max) = (0usize, 1usize, 0.0);
        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate().skip(i + 1) {
                if val.abs() > max {
                    max = val.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-12 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        // Rotate columns p, q (each row's two pivot entries).
        for row in a.iter_mut() {
            let (akp, akq) = (row[p], row[q]);
            row[p] = c * akp - s * akq;
            row[q] = s * akp + c * akq;
        }
        // Rotate rows p, q (the two pivot rows' entries).
        for k in 0..n {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
    }
    (0..n).map(|i| a[i][i]).collect()
}

#[tokio::test]
async fn declared_self_edge_does_not_inflate_augmented_degree() {
    // `Ã = A + I` injects exactly one canonical self-loop per node, giving the
    // augmented degree d̃ = deg + 1. A user who also declares an explicit `(a, a)`
    // edge must NOT push a's degree to deg + 2 (or deg + 3, doubled under
    // `Undirected`) — the declared self-edge is deduped against the injected one.
    //
    // Proof by equivalence: propagating `a—b` is byte-identical to propagating
    // `a—b` plus a declared `(a, a)`. If the declared self-edge were double-
    // counted, a's degree fold and self-contribution would change and the two
    // outputs would diverge.
    let nodes = vec![
        Node {
            id: "a".into(),
            class: 0,
        },
        Node {
            id: "b".into(),
            class: 1,
        },
    ];

    let (plain_session, _d1) = graph_session(&nodes, &[Edge::plain("a", "b")], None).await;
    let plain_table = plain_session
        .propagate_embeddings(&registered_request(&plain_session).await)
        .await
        .unwrap();
    let plain_out = read_table_vectors(&plain_session, &plain_table).await;

    let (self_session, _d2) = graph_session(
        &nodes,
        // Same graph plus a redundant explicit self-edge on `a`.
        &[Edge::plain("a", "b"), Edge::plain("a", "a")],
        None,
    )
    .await;
    let self_table = self_session
        .propagate_embeddings(&registered_request(&self_session).await)
        .await
        .unwrap();
    let self_out = read_table_vectors(&self_session, &self_table).await;

    let plain_a = &plain_out["a"];
    let self_a = &self_out["a"];
    assert_eq!(
        plain_a.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        self_a.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "a declared self-edge must dedupe against the injected I self-loop \
         (d̃ stays deg+1): {plain_a:?} vs {self_a:?}"
    );
}

#[tokio::test]
async fn isolated_node_propagates_to_its_own_x0() {
    // a—b connected; `lonely` has no edge. With the self-loop Ã = A + I, the
    // isolated node's only neighbour is itself, so it must propagate to exactly
    // X⁽⁰⁾ and remain present with its own _row_id.
    let nodes = vec![
        Node {
            id: "a".into(),
            class: 0,
        },
        Node {
            id: "b".into(),
            class: 0,
        },
        Node {
            id: "lonely".into(),
            class: 1,
        },
    ];
    let edges = vec![Edge::plain("a", "b")];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let raw = node_vector("lonely", 1);
    let table = session
        .propagate_embeddings(&registered_request(&session).await)
        .await
        .unwrap();
    let out = read_table_vectors(&session, &table).await;

    let got = out
        .get("lonely")
        .expect("isolated node present with its _row_id");
    for (g, r) in got.iter().zip(&raw) {
        assert!(
            (g - r).abs() < 1e-5,
            "isolated node propagates to X⁽⁰⁾ exactly: {g} vs {r}"
        );
    }
}

#[tokio::test]
async fn homophily_propagation_beats_raw() {
    // Within-class fully-wired graph: propagation denoises toward the class
    // centroid, so the propagated vectors' nearest-neighbour-same-class rate must
    // beat the raw features'.
    let (nodes, edges) = two_class_homophilous(6);
    let class_of: HashMap<String, usize> = nodes.iter().map(|n| (n.id.clone(), n.class)).collect();
    let raw: HashMap<String, Vec<f32>> = nodes
        .iter()
        .map(|n| (n.id.clone(), node_vector(&n.id, n.class)))
        .collect();
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let table = session
        .propagate_embeddings(&registered_request(&session).await)
        .await
        .unwrap();
    let propagated = read_table_vectors(&session, &table).await;

    // Structure respect both ways: the nearest-neighbour class-agreement rate
    // does not drop, and the class separation strictly improves (denoising).
    let raw_rate = nn_same_class_rate(&raw, &class_of);
    let prop_rate = nn_same_class_rate(&propagated, &class_of);
    let raw_sep = class_separation(&raw, &class_of);
    let prop_sep = class_separation(&propagated, &class_of);
    assert!(
        prop_rate >= raw_rate,
        "homophilous propagation must not hurt NN class agreement: raw {raw_rate}, propagated {prop_rate}"
    );
    assert!(
        prop_sep > raw_sep,
        "homophilous propagation must improve class separation (denoising): \
         raw {raw_sep}, propagated {prop_sep}"
    );
}

#[tokio::test]
async fn heterophily_propagation_is_worse_than_raw() {
    // Adversarial bipartite-by-class wiring: every edge crosses the class
    // boundary, so neighbour-averaging mixes opposing signal. Propagation must be
    // SIGNIFICANTLY worse than raw (directional, seeded margin) — the evidence
    // that gates the learned-attention answer (S13).
    let per_class = 6;
    let mut nodes = Vec::new();
    for class in 0..2 {
        for i in 0..per_class {
            nodes.push(Node {
                id: format!("c{class}_{i}"),
                class,
            });
        }
    }
    // Bipartite: every class-0 node wired to every class-1 node.
    let mut edges = Vec::new();
    for i in 0..per_class {
        for j in 0..per_class {
            edges.push(Edge::plain(&format!("c0_{i}"), &format!("c1_{j}")));
        }
    }
    let class_of: HashMap<String, usize> = nodes.iter().map(|n| (n.id.clone(), n.class)).collect();
    let raw: HashMap<String, Vec<f32>> = nodes
        .iter()
        .map(|n| (n.id.clone(), node_vector(&n.id, n.class)))
        .collect();
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    // No teleport restart so the heterophilous mixing is not masked by the
    // α-anchor — this is the homophily-failure contract, demonstrated.
    let table = session
        .propagate_embeddings(&registered_request(&session).await.with_alpha(0.0))
        .await
        .unwrap();
    let propagated = read_table_vectors(&session, &table).await;

    // Cross-class averaging collapses the class separation: propagated
    // separation must be SIGNIFICANTLY below raw (a seeded directional margin) —
    // the evidence that gates the learned-attention answer (S13).
    let raw_sep = class_separation(&raw, &class_of);
    let prop_sep = class_separation(&propagated, &class_of);
    assert!(
        prop_sep < raw_sep - 0.05,
        "heterophilous propagation must significantly collapse class separation \
         (S13-gating): raw {raw_sep}, propagated {prop_sep}"
    );
}

/// A single connected community: `n` nodes (all class 0) on a connected path,
/// so deep uniform mean drives every node to the one global mean (rank collapse)
/// while the APPNP restart keeps each node's own X⁽⁰⁾ signal (rank preserved).
fn one_community_path(n: usize) -> (Vec<Node>, Vec<Edge>) {
    let nodes: Vec<Node> = (0..n)
        .map(|i| Node {
            id: format!("p{i}"),
            class: 0,
        })
        .collect();
    let edges: Vec<Edge> = (0..n - 1)
        .map(|i| Edge::plain(&format!("p{i}"), &format!("p{}", i + 1)))
        .collect();
    (nodes, edges)
}

#[tokio::test]
async fn alpha_restart_controls_oversmoothing() {
    // One connected community, deep propagation. Uniform mean with no teleport
    // drives every node to the single global mean (rank collapse → effective rank
    // near 1); the APPNP α-restart re-anchors each node to its own X⁽⁰⁾, keeping
    // the representation spread (higher effective rank). Assert uniform-deep erank
    // ≤ ratio · appnp-deep erank, and that uniform deepening converges
    // geometrically.
    // A larger community so the global mean genuinely averages many distinct
    // nodes — uniform deepening then drives every node onto it.
    let (nodes, edges) = one_community_path(20);
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    // The hop cap is 3; deepen to the cap. Uniform-no-teleport collapses; the
    // APPNP restart re-anchors each node every hop.
    let uniform_deep = session
        .propagate_embeddings(
            &registered_request(&session)
                .await
                .with_weighting(PropagationWeighting::Uniform)
                .with_alpha(0.0)
                .with_hops(3),
        )
        .await
        .unwrap();
    let appnp_deep = session
        .propagate_embeddings(
            &registered_request(&session)
                .await
                .with_weighting(PropagationWeighting::DegreeNormalized)
                .with_alpha(0.3)
                .with_hops(3),
        )
        .await
        .unwrap();

    let uniform_vecs = read_table_vectors(&session, &uniform_deep).await;
    let appnp_vecs = read_table_vectors(&session, &appnp_deep).await;

    let uniform_erank = effective_rank(&uniform_vecs);
    let appnp_erank = effective_rank(&appnp_vecs);
    // Uniform deepening collapses toward the single global mean (effective rank
    // near 1); the restart keeps strictly more effective rank.
    assert!(
        uniform_erank < appnp_erank - 0.05,
        "the α-restart must keep strictly more effective rank than no-teleport \
         mean: uniform {uniform_erank}, appnp {appnp_erank}"
    );

    // Geometric convergence: ‖X⁽ᵏ⁾ − X⁽ᵏ⁻¹⁾‖ shrinks as uniform mean deepens.
    let one_hop = read_table_vectors(
        &session,
        &session
            .propagate_embeddings(
                &registered_request(&session)
                    .await
                    .with_weighting(PropagationWeighting::Uniform)
                    .with_alpha(0.0)
                    .with_hops(1),
            )
            .await
            .unwrap(),
    )
    .await;
    let two_hop = read_table_vectors(
        &session,
        &session
            .propagate_embeddings(
                &registered_request(&session)
                    .await
                    .with_weighting(PropagationWeighting::Uniform)
                    .with_alpha(0.0)
                    .with_hops(2),
            )
            .await
            .unwrap(),
    )
    .await;
    let raw: HashMap<String, Vec<f32>> = nodes
        .iter()
        .map(|n| (n.id.clone(), node_vector(&n.id, n.class)))
        .collect();
    let step1 = total_drift(&raw, &one_hop);
    let step2 = total_drift(&one_hop, &two_hop);
    assert!(
        step2 < step1,
        "successive hops must converge: ‖X¹−X⁰‖ {step1}, ‖X²−X¹‖ {step2}"
    );
}

/// Sum of L2 distances between matched per-key vectors.
fn total_drift(a: &HashMap<String, Vec<f32>>, b: &HashMap<String, Vec<f32>>) -> f64 {
    a.iter()
        .map(|(k, va)| {
            let vb = &b[k];
            va.iter()
                .zip(vb)
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum()
}

#[tokio::test]
async fn deterministic_across_target_partitions() {
    // The determinism contract: byte-identical (f32::to_bits) output across two
    // different target_partitions settings — the partition-order test, not just
    // two runs at one config.
    let (nodes, edges) = two_class_homophilous(8);

    let (session1, _d1) = graph_session_with_partitions(&nodes, &edges, None, 1).await;
    let req1 = registered_request(&session1).await;
    let table1 = session1.propagate_embeddings(&req1).await.unwrap();
    let out1 = read_table_vectors(&session1, &table1).await;

    let (session4, _d4) = graph_session_with_partitions(&nodes, &edges, None, 4).await;
    let req4 = registered_request(&session4).await;
    let table4 = session4.propagate_embeddings(&req4).await.unwrap();
    let out4 = read_table_vectors(&session4, &table4).await;

    assert_eq!(out1.len(), out4.len());
    for (key, v1) in &out1 {
        let v4 = out4.get(key).expect("same keys across partitionings");
        let bits1: Vec<u32> = v1.iter().map(|f| f.to_bits()).collect();
        let bits4: Vec<u32> = v4.iter().map(|f| f.to_bits()).collect();
        assert_eq!(
            bits1, bits4,
            "node '{key}' differs across target_partitions"
        );
    }
}

#[tokio::test]
async fn edge_similarity_clamps_negative_and_falls_back_on_zero_weight() {
    // a—b carries a strong positive weight; a—c carries a negative similarity
    // (anti-signal) that must clamp to zero. With the self-loop weight 1, a's
    // edge-similarity mean is over {a (w=1), b (w=2)}, never c.
    let nodes = vec![
        Node {
            id: "a".into(),
            class: 0,
        },
        Node {
            id: "b".into(),
            class: 0,
        },
        Node {
            id: "c".into(),
            class: 1,
        },
    ];
    let edges = vec![
        Edge::weighted("a", "b", 2.0),
        Edge::weighted("a", "c", -0.9),
    ];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let table = session
        .propagate_embeddings(
            &registered_request(&session)
                .await
                .with_weighting(PropagationWeighting::EdgeSimilarity)
                .with_alpha(0.0)
                .with_hops(1),
        )
        .await
        .unwrap();
    let out = read_table_vectors(&session, &table).await;

    // Expected: a = (1·X_a + 2·X_b) / 3 — c excluded by the clamp.
    let xa = node_vector("a", 0);
    let xb = node_vector("b", 0);
    let got = &out["a"];
    for lane in 0..DIM {
        let want = (xa[lane] + 2.0 * xb[lane]) / 3.0;
        assert!(
            (got[lane] - want).abs() < 1e-4,
            "edge-similarity clamps the negative edge: lane {lane} got {} want {want}",
            got[lane]
        );
    }
}

#[tokio::test]
async fn edge_similarity_isolated_node_falls_back_to_x0() {
    // `lonely` has no edge; with the self-loop weight 1 its Σw > 0 and it equals
    // X⁽⁰⁾ — the defined behaviour for a node with no real neighbour.
    let nodes = vec![
        Node {
            id: "a".into(),
            class: 0,
        },
        Node {
            id: "lonely".into(),
            class: 1,
        },
    ];
    let edges = vec![Edge::weighted("a", "a", 1.0)]; // self only; a—a not a real edge
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let table = session
        .propagate_embeddings(
            &registered_request(&session)
                .await
                .with_weighting(PropagationWeighting::EdgeSimilarity),
        )
        .await
        .unwrap();
    let out = read_table_vectors(&session, &table).await;
    let raw = node_vector("lonely", 1);
    for (g, r) in out["lonely"].iter().zip(&raw) {
        assert!((g - r).abs() < 1e-5, "edge-similarity isolated node → X⁽⁰⁾");
    }
}

#[tokio::test]
async fn hop_cap_clamps_request() {
    // Requesting 10 hops runs the same as an explicit 3-hop (the cap), and
    // differs from a 1-hop — the cap is effective, not ignored. A path graph is
    // used so depth actually matters (a 3-hop frontier reaches further than a
    // 1-hop one, unlike a fully-connected community where 1 hop already mixes
    // everything). No teleport so successive hops keep moving.
    let (nodes, edges) = one_community_path(12);
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let capped = read_table_vectors(
        &session,
        &session
            .propagate_embeddings(
                &registered_request(&session)
                    .await
                    .with_alpha(0.0)
                    .with_hops(10),
            )
            .await
            .unwrap(),
    )
    .await;
    let three = read_table_vectors(
        &session,
        &session
            .propagate_embeddings(
                &registered_request(&session)
                    .await
                    .with_alpha(0.0)
                    .with_hops(3),
            )
            .await
            .unwrap(),
    )
    .await;
    let one = read_table_vectors(
        &session,
        &session
            .propagate_embeddings(
                &registered_request(&session)
                    .await
                    .with_alpha(0.0)
                    .with_hops(1),
            )
            .await
            .unwrap(),
    )
    .await;

    for (key, v_capped) in &capped {
        let bits_capped: Vec<u32> = v_capped.iter().map(|f| f.to_bits()).collect();
        let bits_three: Vec<u32> = three[key].iter().map(|f| f.to_bits()).collect();
        assert_eq!(
            bits_capped, bits_three,
            "10 hops clamps to the 3-hop cap for '{key}'"
        );
    }
    assert!(
        capped.keys().any(|k| {
            capped[k].iter().map(|f| f.to_bits()).collect::<Vec<_>>()
                != one[k].iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        }),
        "the cap (3 hops) differs from 1 hop — depth is actually applied"
    );
}

#[tokio::test]
async fn jumping_knowledge_concats_every_hop_normalizes_blocks_and_is_searchable() {
    let (nodes, edges) = two_class_homophilous(5);
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    // 2 hops → K+1 = 3 blocks (X⁰, X¹, X²) → output dim (K+1)·d.
    let hops = 2;
    let blocks = hops + 1;
    let table = session
        .propagate_embeddings(
            &registered_request(&session)
                .await
                .with_hops(hops)
                .with_output(PropagationOutput::JumpingKnowledge),
        )
        .await
        .unwrap();
    assert_eq!(
        table.dimensions,
        Some((DIM * blocks) as i32),
        "JK output dim = (K+1)·d — one block per hop plus X⁰"
    );
    let out = read_table_vectors(&session, &table).await;
    for (key, v) in &out {
        assert_eq!(
            v.len(),
            DIM * blocks,
            "row '{key}' carries every per-hop block"
        );
        // Each d-wide block is independently L2-normalised before concat.
        for (b, chunk) in v.chunks(DIM).enumerate() {
            let norm = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "block {b} of row '{key}' is L2-normalised"
            );
        }
    }

    // Searchable in its own space: a JK vector queried against the JK table
    // returns its own row first.
    let probe_key = "c0_0";
    let probe = out[probe_key].clone();
    let hits = session
        .result_store()
        .search_vectors(session.context(), &table, &probe, 1)
        .await
        .unwrap();
    assert_eq!(
        hits[0].0, probe_key,
        "the JK table is a searchable embedding table"
    );
}

#[tokio::test]
async fn output_is_model_kind_with_lineage_and_is_regraphable() {
    let (nodes, edges) = two_class_homophilous(5);
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let source_table = session
        .catalog()
        .resolve_embedding_table("nodes", None)
        .await
        .unwrap();
    let table = session
        .propagate_embeddings(&registered_request(&session).await)
        .await
        .unwrap();

    assert_eq!(
        table.kind,
        jammi_db::catalog::result_repo::ResultTableKind::Model,
        "a propagated table is a normal Model embedding table"
    );
    assert_eq!(
        table.derived_from.as_deref(),
        Some(source_table.table_name.as_str()),
        "derived_from records the source embedding table (FK lineage)"
    );
    assert!(
        table.index_path.is_some(),
        "a Model embedding table gets a sidecar index"
    );

    // Re-graphable: build a neighbor graph over the propagated table.
    let graph = session
        .build_neighbor_graph(
            "nodes",
            Some(&table.table_name),
            &BuildNeighborGraph {
                k: 2,
                exact: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(graph.row_count > 0, "the propagated table is re-graphable");
}

#[tokio::test]
async fn evaluable_through_r1_eval_embeddings() {
    // R1 hook: a propagated table is a normal embedding table the eval runner
    // (`EvalRunner::eval_embeddings`) resolves by name and reads through its
    // per-query `search_vectors` loop. Exercise that resolve + search path
    // directly (no live encoder), which is exactly the R1 read seam.
    let (nodes, edges) = two_class_homophilous(5);
    let (session, _dir) = graph_session(&nodes, &edges, None).await;
    let table = session
        .propagate_embeddings(&registered_request(&session).await)
        .await
        .unwrap();

    let resolved = session
        .catalog()
        .resolve_embedding_table("nodes", Some(&table.table_name))
        .await
        .unwrap();
    assert_eq!(resolved.key_column.as_deref(), Some("_row_id"));
    assert_eq!(resolved.dimensions, Some(DIM as i32));

    // The eval runner's per-query loop runs `search_vectors` over the resolved
    // table — exercise exactly that read path (the R1 hook) without a live
    // encoder: a query vector against the propagated table returns ranked hits.
    let probe = read_table_vectors(&session, &table).await["c0_0"].clone();
    let hits = session
        .result_store()
        .search_vectors(session.context(), &resolved, &probe, 3)
        .await
        .unwrap();
    assert!(
        !hits.is_empty(),
        "the propagated table serves the R1 eval search path"
    );
}

#[tokio::test]
async fn weighting_variants_hand_checked() {
    // A tiny path a—b—c, undirected, 1 hop, no teleport. Hand-check Uniform vs
    // DegreeNormalized on node b (degree 2 + self = 3) and node a (degree 1 +
    // self = 2).
    let nodes = vec![
        Node {
            id: "a".into(),
            class: 0,
        },
        Node {
            id: "b".into(),
            class: 0,
        },
        Node {
            id: "c".into(),
            class: 0,
        },
    ];
    let edges = vec![Edge::plain("a", "b"), Edge::plain("b", "c")];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let xa = node_vector("a", 0);
    let xb = node_vector("b", 0);
    let xc = node_vector("c", 0);

    // Uniform = D̃⁻¹Ã = mean over self-loop neighbourhood.
    let uniform = read_table_vectors(
        &session,
        &session
            .propagate_embeddings(
                &registered_request(&session)
                    .await
                    .with_weighting(PropagationWeighting::Uniform)
                    .with_alpha(0.0)
                    .with_hops(1),
            )
            .await
            .unwrap(),
    )
    .await;
    // b averages {a, b, c}.
    for lane in 0..DIM {
        let want = (xa[lane] + xb[lane] + xc[lane]) / 3.0;
        assert!(
            (uniform["b"][lane] - want).abs() < 1e-4,
            "uniform b lane {lane}"
        );
    }

    // DegreeNormalized = Σ_{v∈Ñ(u)} X(v)/(√d̃_u·√d̃_v). For b: neighbours a,b,c
    // with d̃_a=2, d̃_b=3, d̃_c=2 → b = (X_a/√2 + X_b/√3 + X_c/√2)/√3.
    let degnorm = read_table_vectors(
        &session,
        &session
            .propagate_embeddings(
                &registered_request(&session)
                    .await
                    .with_weighting(PropagationWeighting::DegreeNormalized)
                    .with_alpha(0.0)
                    .with_hops(1),
            )
            .await
            .unwrap(),
    )
    .await;
    let s2 = 2.0_f32.sqrt();
    let s3 = 3.0_f32.sqrt();
    for lane in 0..DIM {
        let want = (xa[lane] / s2 + xb[lane] / s3 + xc[lane] / s2) / s3;
        assert!(
            (degnorm["b"][lane] - want).abs() < 1e-4,
            "degree-normalized b lane {lane}: got {} want {want}",
            degnorm["b"][lane]
        );
    }
}

#[tokio::test]
async fn cross_tenant_edge_endpoint_is_never_propagated() {
    // THE load-bearing tenancy contract. Alice binds her tenant; the edge source
    // carries her own edge (a—b) and an edge into a foreign tenant's node
    // (a—secret, tagged with Bob's tenant). The propagation's edge scan runs
    // inside the analyzer scope, so the Bob-tagged row is filtered before it
    // reaches the adjacency — `secret` can never influence Alice's a.
    let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e01").unwrap();
    let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e02").unwrap();

    let nodes = vec![
        Node {
            id: "a".into(),
            class: 0,
        },
        Node {
            id: "b".into(),
            class: 0,
        },
    ];
    let edges = vec![
        Edge {
            src: "a".into(),
            dst: "b".into(),
            weight: None,
            tenant: Some(alice),
        },
        Edge {
            src: "a".into(),
            dst: "secret".into(),
            weight: None,
            tenant: Some(bob),
        },
    ];
    let (session, _dir) = graph_session(&nodes, &edges, Some(alice)).await;

    // With only the a—b edge visible, a (degree 1 + self = 2) and b symmetric.
    // If `secret` leaked, a's degree would be 2 + self = 3, changing the result.
    let table = session
        .propagate_embeddings(
            &registered_request(&session)
                .await
                .with_weighting(PropagationWeighting::Uniform)
                .with_alpha(0.0)
                .with_hops(1),
        )
        .await
        .unwrap();
    let out = read_table_vectors(&session, &table).await;

    assert!(
        !out.contains_key("secret"),
        "a foreign-tenant node never enters the output"
    );
    let xa = node_vector("a", 0);
    let xb = node_vector("b", 0);
    for lane in 0..DIM {
        // a = mean{a, b} only — never {a, b, secret}.
        let want = (xa[lane] + xb[lane]) / 2.0;
        assert!(
            (out["a"][lane] - want).abs() < 1e-4,
            "a aggregates only Alice's edge: lane {lane}"
        );
    }
}

#[tokio::test]
async fn edge_set_over_ceiling_is_refused() {
    let (nodes, edges) = two_class_homophilous(4);
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let mut request = registered_request(&session).await;
    request.max_rows = 1; // the fixture has many edges (counting both directions)
    let err = session.propagate_embeddings(&request).await.unwrap_err();
    assert!(
        err.to_string().contains("exceeds the ceiling"),
        "an over-ceiling edge set is refused loudly, not silently OOM: {err}"
    );
}
