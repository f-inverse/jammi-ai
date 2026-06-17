//! `InferenceSession::assemble_context` over **declared edges** (spec S16-G):
//! the bounded, target-anchored graph-neighbourhood context, its homophily
//! diagnostic, and — load-bearing — that the gather runs inside the tenant
//! scope so a cross-tenant edge endpoint is never materialised.
//!
//! Hermetic: a tempdir session carries a synthetic node embedding table (a
//! distinct vector per node, keyed by `_row_id`) and a registered external edge
//! source (`src`/`dst`[/`type`][/`tenant_id`]). The node ids are the edge
//! endpoints and the embedding keys, so a gathered neighbour pools through the
//! same shared vector-aggregation UDAF an ANN context does. No consumer
//! vocabulary appears — the fixtures are a neutral citation-/co-purchase-style
//! graph.

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use jammi_ai::pipeline::context_set::{
    ContextRequest, ContextSource, ContextSourceKind, HybridMerge,
};
use jammi_ai::pipeline::graph_neighbourhood::{EdgeGather, EdgeSourceRef};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::TenantId;

use crate::common;

const DIM: usize = 8;

/// A node: its id (the embedding key and edge endpoint), a categorical `label`
/// (for the homophily diagnostic), and a `component` key (for the graph-locality
/// split). Vectors are derived from the id so pooling is deterministic.
struct Node {
    id: &'static str,
    label: &'static str,
    component: &'static str,
}

/// A declared edge: endpoints, an optional type, an optional weight, and an
/// optional tenant tag (for the cross-tenant leak fixture).
struct Edge {
    src: &'static str,
    dst: &'static str,
    edge_type: Option<&'static str>,
    weight: Option<f64>,
    tenant: Option<TenantId>,
}

impl Edge {
    fn plain(src: &'static str, dst: &'static str) -> Self {
        Self {
            src,
            dst,
            edge_type: None,
            weight: None,
            tenant: None,
        }
    }
    fn typed(src: &'static str, dst: &'static str, t: &'static str) -> Self {
        Self {
            src,
            dst,
            edge_type: Some(t),
            weight: None,
            tenant: None,
        }
    }
    fn weighted(src: &'static str, dst: &'static str, w: f64) -> Self {
        Self {
            src,
            dst,
            edge_type: None,
            weight: Some(w),
            tenant: None,
        }
    }
}

/// A distinct, deterministic vector for a node id — a smooth function of the id
/// bytes so different nodes pool to different vectors.
fn node_vector(id: &str) -> Vec<f32> {
    let base = id.bytes().map(|b| b as f32).sum::<f32>();
    (0..DIM)
        .map(|i| ((base + i as f32) * 0.013).sin())
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

/// Stand up a session: register a `nodes` source (`_row_id`/`label`/`component`),
/// materialise its embedding table, and register an `edges` source
/// (`src`/`dst`/`type`/`tenant_id`). When `tenant` is set, the build runs under
/// it (so the embedding table and sources are that tenant's).
async fn graph_session(
    nodes: &[Node],
    edges: &[Edge],
    tenant: Option<TenantId>,
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session.register_query_functions();
    if let Some(t) = tenant {
        session.bind_tenant(t);
    }

    // nodes parquet: _row_id, label, component.
    let node_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("component", DataType::Utf8, false),
    ]));
    let node_batch = RecordBatch::try_new(
        Arc::clone(&node_schema),
        vec![
            Arc::new(StringArray::from(
                nodes.iter().map(|n| n.id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                nodes.iter().map(|n| n.label).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                nodes.iter().map(|n| n.component).collect::<Vec<_>>(),
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

    // The embedding table keyed by _row_id, vector = node_vector(id).
    let pairs: Vec<(String, Vec<f32>)> = nodes
        .iter()
        .map(|n| (n.id.to_string(), node_vector(n.id)))
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

    // edges parquet: src, dst, type, weight, tenant_id.
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
        Field::new("type", DataType::Utf8, true),
        Field::new("weight", DataType::Float64, true),
        Field::new("tenant_id", DataType::Utf8, true),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(StringArray::from(
                edges.iter().map(|e| e.src).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                edges.iter().map(|e| e.dst).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                edges.iter().map(|e| e.edge_type).collect::<Vec<_>>(),
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

/// A registered-edge gather over the `edges` source's `src`/`dst` columns.
fn edge_gather() -> EdgeGather {
    EdgeGather::new(EdgeSourceRef::Registered {
        source_id: "edges".to_string(),
        src_column: "src".to_string(),
        dst_column: "dst".to_string(),
        type_column: Some("type".to_string()),
        weight_column: None,
        as_of_column: None,
    })
}

/// An `Edges` context request anchored at `target`.
fn edges_request(target: &str, gather: EdgeGather) -> ContextRequest {
    let mut request = ContextRequest::new("nodes", node_vector(target), 0);
    request.source = ContextSource::Edges(gather);
    request.exclude_key = Some(target.to_string());
    request
}

fn keyset(keys: &[String]) -> std::collections::HashSet<String> {
    keys.iter().cloned().collect()
}

/// A small homophilous path `a → b → c → d`, plus a star out of `a`.
fn path_nodes() -> Vec<Node> {
    ["a", "b", "c", "d", "e"]
        .into_iter()
        .map(|id| Node {
            id,
            label: "x",
            component: "one",
        })
        .collect()
}

#[tokio::test]
async fn edges_gather_one_hop_and_two_hop() {
    let nodes = path_nodes();
    let edges = [
        Edge::plain("a", "b"),
        Edge::plain("a", "c"),
        Edge::plain("b", "d"),
    ];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    // 1 hop from a → {b, c}.
    let rep = session
        .assemble_context(&edges_request("a", edge_gather()))
        .await
        .unwrap();
    assert_eq!(rep.source, ContextSourceKind::Edges, "source fact is Edges");
    assert_eq!(
        keyset(&rep.context_keys),
        keyset(&["b".to_string(), "c".to_string()]),
        "1-hop out-neighbours of a"
    );
    assert!(
        rep.context_vector.is_some(),
        "a non-empty graph context pools to a vector"
    );

    // 2 hops from a → {b, c, d}.
    let mut g = edge_gather();
    g.hops = 2;
    let rep2 = session
        .assemble_context(&edges_request("a", g))
        .await
        .unwrap();
    assert_eq!(
        keyset(&rep2.context_keys),
        keyset(&["b".to_string(), "c".to_string(), "d".to_string()]),
        "2-hop reaches d via b"
    );
}

#[tokio::test]
async fn hop_depth_is_hard_capped() {
    let nodes = path_nodes();
    let edges = [
        Edge::plain("a", "b"),
        Edge::plain("b", "c"),
        Edge::plain("c", "d"),
        Edge::plain("d", "e"),
    ];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    // hops far past the cap (default 3) reaches at most the 3-hop set {b,c,d},
    // never e (4 hops) — the cap is structural, not advisory.
    let mut g = edge_gather();
    g.hops = 99;
    let rep = session
        .assemble_context(&edges_request("a", g))
        .await
        .unwrap();
    assert_eq!(
        keyset(&rep.context_keys),
        keyset(&["b".to_string(), "c".to_string(), "d".to_string()]),
        "depth is hard-capped at 3 hops; e (hop 4) is unreachable"
    );
}

#[tokio::test]
async fn edge_types_filter_the_walk() {
    let nodes = path_nodes();
    let edges = [
        Edge::typed("a", "b", "cites"),
        Edge::typed("a", "c", "mentions"),
    ];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let mut g = edge_gather();
    g.edge_types = Some(vec!["cites".to_string()]);
    let rep = session
        .assemble_context(&edges_request("a", g))
        .await
        .unwrap();
    assert_eq!(
        rep.context_keys,
        vec!["b".to_string()],
        "only the `cites` edge traverses; `mentions` is filtered"
    );
}

#[tokio::test]
async fn min_weight_filters_weak_edges() {
    let nodes = path_nodes();
    let edges = [
        Edge::weighted("a", "b", 0.9), // strong — kept
        Edge::weighted("a", "c", 0.1), // weak — filtered
    ];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let mut g = EdgeGather::new(EdgeSourceRef::Registered {
        source_id: "edges".to_string(),
        src_column: "src".to_string(),
        dst_column: "dst".to_string(),
        type_column: None,
        weight_column: Some("weight".to_string()),
        as_of_column: None,
    });
    g.min_weight = Some(0.5);
    let rep = session
        .assemble_context(&edges_request("a", g))
        .await
        .unwrap();
    assert_eq!(
        rep.context_keys,
        vec!["b".to_string()],
        "only the edge at or above min_weight (0.9) traverses; the 0.1 edge is filtered"
    );
}

#[tokio::test]
async fn gather_is_deterministic_under_fanout() {
    // A high-degree node with fan-out: the seeded sample reproduces.
    let nodes: Vec<Node> = ('a'..='z')
        .map(|c| {
            let id: &'static str = Box::leak(c.to_string().into_boxed_str());
            Node {
                id,
                label: "x",
                component: "one",
            }
        })
        .collect();
    let center = nodes[0].id;
    let edges: Vec<Edge> = nodes[1..]
        .iter()
        .map(|n| Edge::plain(center, n.id))
        .collect();
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let mut g = edge_gather();
    g.fanout = Some(5);
    let first = session
        .assemble_context(&edges_request(center, g.clone()))
        .await
        .unwrap();
    let again = session
        .assemble_context(&edges_request(center, g))
        .await
        .unwrap();
    assert_eq!(first.context_keys.len(), 5, "fan-out caps the gather at 5");
    assert_eq!(
        first.context_keys, again.context_keys,
        "the seeded fan-out sample reproduces byte-identically for a target"
    );
    assert_eq!(
        first.context_vector, again.context_vector,
        "the pooled vector reproduces (the determinism contract)"
    );
}

#[tokio::test]
async fn hybrid_unions_ann_and_declared_edges() {
    // a's ANN neighbours (by vector similarity) and its declared edge to a
    // distant node both enter the hybrid context.
    let nodes = path_nodes();
    let edges = [Edge::plain("a", "e")];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let mut request = ContextRequest::new("nodes", node_vector("a"), 2);
    request.source = ContextSource::Hybrid {
        ann_k: 2,
        edges: edge_gather(),
        merge: HybridMerge::Union,
    };
    request.exclude_key = Some("a".to_string());
    let rep = session.assemble_context(&request).await.unwrap();
    assert_eq!(rep.source, ContextSourceKind::Hybrid);
    assert!(
        rep.context_keys.contains(&"e".to_string()),
        "the declared edge a→e is unioned into the hybrid context: {:?}",
        rep.context_keys
    );
    assert!(
        rep.context_keys.len() >= 3,
        "hybrid = ANN (2) ∪ the edge neighbour, deduped"
    );
}

#[tokio::test]
async fn disconnected_target_yields_empty_context() {
    let nodes = path_nodes();
    let edges = [Edge::plain("a", "b")];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    // `e` has no edges → empty graph context, defined (not a crash, not a
    // one-neighbour average).
    let rep = session
        .assemble_context(&edges_request("e", edge_gather()))
        .await
        .unwrap();
    assert!(rep.is_empty(), "a disconnected target has an empty context");
    assert!(rep.context_vector.is_none());
    assert_eq!(rep.source, ContextSourceKind::Edges);
}

#[tokio::test]
async fn source_fact_distinguishes_ann_edges_hybrid() {
    let nodes = path_nodes();
    let edges = [Edge::plain("a", "b")];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let ann = ContextRequest::new("nodes", node_vector("a"), 2);
    assert_eq!(
        session.assemble_context(&ann).await.unwrap().source,
        ContextSourceKind::Ann
    );

    let edges_rep = session
        .assemble_context(&edges_request("a", edge_gather()))
        .await
        .unwrap();
    assert_eq!(edges_rep.source, ContextSourceKind::Edges);
}

#[tokio::test]
async fn graph_locality_split_keeps_adjacency_on_one_side() {
    // Adjacency correlates with the `component` key: a,b in component "one",
    // c in component "two", with a cross-component edge a→c. A split scoped by
    // component keeps the gathered context inside the target's component, so a
    // graph-adjacent cross-component row never leaks across the train/eval line
    // — the leakage a row-random split would admit.
    let nodes = vec![
        Node {
            id: "a",
            label: "x",
            component: "one",
        },
        Node {
            id: "b",
            label: "x",
            component: "one",
        },
        Node {
            id: "c",
            label: "x",
            component: "two",
        },
    ];
    let edges = [Edge::plain("a", "b"), Edge::plain("a", "c")];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    // No split: both neighbours gather (the would-be leak across components).
    let unscoped = session
        .assemble_context(&edges_request("a", edge_gather()))
        .await
        .unwrap();
    assert_eq!(
        keyset(&unscoped.context_keys),
        keyset(&["b".to_string(), "c".to_string()]),
        "without a locality split, the cross-component neighbour is admitted"
    );

    // Locality split by component: c (component "two") is held out.
    let mut request = edges_request("a", edge_gather());
    request.split = Some("component = 'one'".to_string());
    let scoped = session.assemble_context(&request).await.unwrap();
    assert_eq!(
        scoped.context_keys,
        vec!["b".to_string()],
        "the locality split keeps the context inside the target's component"
    );
}

#[tokio::test]
async fn homophily_diagnostic_flags_heterophily() {
    // `same`-typed edges connect equal labels (homophilous); `diff`-typed edges
    // connect opposite labels (heterophilous). The diagnostic surfaces the
    // per-type agreement so a caller can see which edge type helps.
    let nodes = vec![
        Node {
            id: "a",
            label: "red",
            component: "one",
        },
        Node {
            id: "b",
            label: "red",
            component: "one",
        },
        Node {
            id: "c",
            label: "blue",
            component: "one",
        },
        Node {
            id: "d",
            label: "blue",
            component: "one",
        },
    ];
    let edges = [
        Edge::typed("a", "b", "same"), // red–red
        Edge::typed("c", "d", "same"), // blue–blue
        Edge::typed("a", "c", "diff"), // red–blue
        Edge::typed("b", "d", "diff"), // red–blue
    ];
    let (session, _dir) = graph_session(&nodes, &edges, None).await;

    let homophily = session
        .homophily_by_edge_type(&edge_gather(), "nodes", "_row_id", "label")
        .await
        .unwrap();
    assert_eq!(
        homophily.get("same"),
        Some(&1.0),
        "the `same` type is perfectly homophilous"
    );
    assert_eq!(
        homophily.get("diff"),
        Some(&0.0),
        "the `diff` type is heterophilous — flagged by a near-zero agreement"
    );
}

#[tokio::test]
async fn cross_tenant_edge_endpoint_is_never_gathered() {
    // THE load-bearing tenancy contract. Alice binds her tenant; the edge source
    // carries one of her edges (a→b) and one edge into a foreign tenant's node
    // (a→secret, tagged with Bob's tenant). The gather runs inside the analyzer
    // scope, so the Bob-tagged row is filtered before it reaches the adjacency —
    // `secret` can never enter Alice's context.
    let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e01").unwrap();
    let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e02").unwrap();

    let nodes = vec![
        Node {
            id: "a",
            label: "x",
            component: "one",
        },
        Node {
            id: "b",
            label: "x",
            component: "one",
        },
    ];
    let edges = [
        Edge {
            src: "a",
            dst: "b",
            edge_type: None,
            weight: None,
            tenant: Some(alice),
        },
        Edge {
            src: "a",
            dst: "secret",
            edge_type: None,
            weight: None,
            tenant: Some(bob),
        },
    ];
    let (session, _dir) = graph_session(&nodes, &edges, Some(alice)).await;

    let rep = session
        .assemble_context(&edges_request("a", edge_gather()))
        .await
        .unwrap();
    assert_eq!(
        rep.context_keys,
        vec!["b".to_string()],
        "only Alice's own edge gathers"
    );
    assert!(
        !rep.context_keys.contains(&"secret".to_string()),
        "a cross-tenant edge endpoint is unmaterialisable under tenant scope"
    );
}
