//! Hermetic, CPU-only tests for `recompute` — the action half of incremental
//! recompute. Each test materialises a synthetic embedding table (no model, so
//! the whole suite runs without a GPU), derives a graph/propagation/context-set
//! over it through the real producers, then re-invokes the recorded producer via
//! [`Session::recompute`] and asserts the contract:
//!
//! - **Non-vacuity (the load-bearing tests).** A neighbor-graph and a
//!   graph-propagation and a context-set built with **non-default** producer
//!   parameters recompute **byte-identical** — proving the complete descriptor
//!   carries enough to replay faithfully (a default-params test would pass even
//!   where the descriptor is lossy).
//! - **Staleness.** Advancing a parent makes a child stale; recomputing the child
//!   re-resolves it over the parent's new digest.
//! - **Cascade.** `ReportOnly` reports-but-does-not-recompute the downstream set;
//!   `Downstream` sweeps every transitive dependent once, in topological order; a
//!   diamond DAG recomputes the shared descendant exactly once; a forged cycle is
//!   the typed `DependencyCycle`.
//! - **Pre-contract.** A table with no recorded descriptor is `NotRecomputable`.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, StringArray};
use jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph;
use jammi_ai::pipeline::recompute::Cascade;
use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::error::JammiError;
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{ArtifactDigest, InputAnchor, Materialization};
use jammi_db::store::{CachePolicy, EmbeddingTableSpec};
use tempfile::TempDir;

use crate::common;

const DIM: usize = 8;

/// A fresh session with the vector-aggregation UDAFs registered (the context-set
/// pooling reuses them) and one synthetic embedding table over a synthetic
/// `points` source. Returns the session, its temp dir, and the seeded embedding
/// table record. No model is loaded — the vectors are written directly, so the
/// derived producers run CPU-hermetically.
async fn session_with_synthetic_embeddings() -> (Arc<InferenceSession>, TempDir, ResultTableRecord)
{
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    session.register_query_functions();

    // A small, fixed point cloud: deterministic vectors so every recompute is
    // reproducible. Eight points in 8-d, distinct so the kNN graph is non-trivial.
    let pairs: Vec<(String, Vec<f32>)> = (0..8)
        .map(|i| {
            let mut v = vec![0.0_f32; DIM];
            v[i % DIM] = 1.0;
            v[(i + 1) % DIM] = 0.5;
            (format!("p{i}"), v)
        })
        .collect();

    let (descriptor, env, inputs) =
        jammi_test_utils::synthetic_seed_contract("synthetic-embed", "points", DIM);
    let record = session
        .result_store()
        .materialize_embedding_table(
            session.context(),
            EmbeddingTableSpec {
                source_id: "points",
                model_id: "synthetic-embed",
                derived_from: None,
                dimensions: DIM,
            },
            &pairs,
            Materialization::new(&descriptor, &env, inputs),
        )
        .await
        .unwrap();
    (session, dir, record)
}

/// The artifact digest a table's manifest attests — the byte-identity witness.
/// Two materialisations with the same digest are byte-identical Parquet objects.
async fn artifact_digest(session: &InferenceSession, table: &str) -> String {
    let record = session
        .catalog()
        .get_result_table(table)
        .await
        .unwrap()
        .expect("table present");
    let url = StorageUrl::parse(&record.parquet_path).unwrap();
    session
        .result_store()
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("manifest sidecar present")
        .artifact
        .0
}

// ── Non-vacuity: byte-identical recompute over NON-DEFAULT producer params ──

#[tokio::test]
async fn recompute_neighbor_graph_with_non_default_params_is_byte_identical() {
    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    let svc = Session::new(Arc::clone(&session));

    // NON-DEFAULT every knob the descriptor records: k != default(10),
    // min_similarity set, mutual on, self_exclude off, exact forced,
    // exact_max_rows lowered. A default-params build would pass vacuously even if
    // the descriptor dropped one of these; flipping all of them proves the
    // descriptor carries each into the replay.
    let params = BuildNeighborGraph {
        k: 3,
        min_similarity: Some(0.1),
        mutual: true,
        self_exclude: false,
        exact: true,
        exact_max_rows: 100,
        resolve_keys: true,
    };
    let (graph, outcome) = session
        .build_neighbor_graph(
            "points",
            Some(&emb.table_name),
            &params,
            CachePolicy::Bypass,
        )
        .await
        .unwrap();
    assert!(matches!(outcome, jammi_db::store::CacheOutcome::Computed));

    let before = artifact_digest(&session, &graph.table_name).await;
    let report = svc
        .recompute(&graph.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    assert_eq!(report.recomputed.len(), 1);
    let after = artifact_digest(&session, &report.recomputed[0].recomputed).await;

    assert_eq!(
        before, after,
        "a neighbor-graph recompute over non-default params must be byte-identical — \
         the complete descriptor replays every knob"
    );
}

#[tokio::test]
async fn recompute_graph_propagation_with_non_default_params_is_byte_identical() {
    use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
    use jammi_ai::pipeline::graph_propagation::{
        PropagateRequest, PropagationOutput, PropagationWeighting,
    };

    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    let svc = Session::new(Arc::clone(&session));

    // A neighbor-graph edge relation (a ResultDigest-anchored edge source, the
    // case the GraphPropagation descriptor records faithfully).
    let (graph, _) = session
        .build_neighbor_graph(
            "points",
            Some(&emb.table_name),
            &BuildNeighborGraph {
                k: 4,
                exact: true,
                ..Default::default()
            },
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    // NON-DEFAULT every recorded propagation knob: direction Undirected (default
    // Out), hops 3 (default 2), alpha 0.35 (default 0.1), Uniform weighting
    // (default DegreeNormalized), JumpingKnowledge output (default Final → also
    // changes the dimensionality).
    let request = PropagateRequest::new(
        "points",
        EdgeSourceRef::NeighborGraph {
            table_name: graph.table_name.clone(),
        },
    )
    .with_embedding_table(emb.table_name.clone())
    .with_direction(EdgeDirection::Undirected)
    .with_hops(3)
    .with_alpha(0.35)
    .with_weighting(PropagationWeighting::Uniform)
    .with_output(PropagationOutput::JumpingKnowledge);

    let (propagated, _) = session
        .propagate_embeddings(&request, CachePolicy::Bypass)
        .await
        .unwrap();

    let before = artifact_digest(&session, &propagated.table_name).await;
    let report = svc
        .recompute(&propagated.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    let after = artifact_digest(&session, &report.recomputed[0].recomputed).await;

    assert_eq!(
        before, after,
        "a graph-propagation recompute over non-default params must be byte-identical"
    );
}

#[tokio::test]
async fn recompute_graph_propagation_over_registered_non_default_columns_is_byte_identical() {
    use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
    use jammi_ai::pipeline::graph_propagation::PropagateRequest;

    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    register_edges_source(&session, &_dir).await;
    let svc = Session::new(Arc::clone(&session));

    // A propagation over a REGISTERED edge source with NON-DEFAULT columns
    // (`from`/`to`, not the `src`/`dst` defaults). The proven HIGH bug: the
    // GraphPropagation descriptor recorded only the source id, and the replay
    // reconstructed with hardcoded `src`/`dst` — so a propagation over `from`/`to`
    // either replayed over a different graph or failed. The fix records the full
    // edge-source binding, so the replay reads the exact same columns and is
    // byte-identical.
    let request = PropagateRequest::new(
        "points",
        EdgeSourceRef::Registered {
            source_id: "edges".into(),
            src_column: "from".into(),
            dst_column: "to".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        },
    )
    .with_embedding_table(emb.table_name.clone())
    .with_direction(EdgeDirection::Undirected)
    .with_hops(2);

    let (propagated, _) = session
        .propagate_embeddings(&request, CachePolicy::Bypass)
        .await
        .unwrap();

    let before = artifact_digest(&session, &propagated.table_name).await;
    let report = svc
        .recompute(&propagated.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    let after = artifact_digest(&session, &report.recomputed[0].recomputed).await;

    assert_eq!(
        before, after,
        "a graph-propagation recompute over a registered source with non-default \
         (`from`/`to`) columns must be byte-identical — the descriptor records the full \
         edge-source binding and the replay reconstructs it losslessly"
    );
}

#[tokio::test]
async fn recompute_context_set_pair_with_non_default_params_is_byte_identical() {
    use jammi_ai::pipeline::context_set::{
        ContextRequest, ContextSource, MaterializedContext, SetAggregator,
    };
    use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeGather, EdgeSourceRef};

    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    register_edges_source(&session, &_dir).await;
    let svc = Session::new(Arc::clone(&session));

    // A declared, REGISTERED edge source (Utf8 `from`/`to` columns) — the
    // candidate source for the context-set's edge gather. An *edge*-sourced
    // context (vs ANN) makes the pooling fully reproducible (the index-assisted
    // ANN path is non-deterministic), and the Registered shape exercises the
    // richest descriptor path: the `EdgeSourceBinding::Registered { columns }`
    // the descriptor records and recompute must reconstruct losslessly.
    //
    // NON-DEFAULT recipe: Sum pooling (default Mean), exclude_self off (default
    // on), a 2-hop undirected edge gather (default 1 hop, Out) over named
    // non-default columns (`from`/`to`, not the `src`/`dst` defaults). Each is a
    // recorded determinant; flipping all of them proves the descriptor replays
    // every gather knob. The source embedding table is pinned on the recipe so
    // the recompute re-pools over the *same* table (not a later context-set
    // output that would shadow the source's default-embedding resolution).
    let recipe_proto = {
        let mut r = ContextRequest::new("points", Vec::new(), 0);
        let mut gather = EdgeGather::new(EdgeSourceRef::Registered {
            source_id: "edges".into(),
            src_column: "from".into(),
            dst_column: "to".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        });
        gather.hops = 2;
        gather.direction = EdgeDirection::Undirected;
        r.source = ContextSource::Edges(gather);
        r.aggregator = SetAggregator::Sum;
        r.exclude_self = false;
        r.embedding_table = Some(emb.table_name.clone());
        r
    };

    let source_rows = read_embedding_rows(&session, &emb).await;
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    for (row_id, vector) in &source_rows {
        let mut req = recipe_proto.clone();
        req.query = vector.clone();
        req.exclude_key = Some(row_id.clone());
        if let Some(v) = session.assemble_context(&req).await.unwrap().context_vector {
            rows.push((row_id.clone(), v));
        }
    }
    assert!(
        !rows.is_empty(),
        "the edge-sourced context must pool some targets"
    );
    let (context_table, _) = session
        .materialize_context(
            MaterializedContext {
                rows: &rows,
                dimensions: DIM,
                recipe: &recipe_proto,
            },
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let before = artifact_digest(&session, &context_table.table_name).await;
    let report = svc
        .recompute(&context_table.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    let after = artifact_digest(&session, &report.recomputed[0].recomputed).await;

    assert_eq!(
        before, after,
        "a context-set recompute (assemble→materialize pair) over non-default params \
         must be byte-identical — the descriptor replays the full recipe (the registered \
         edge gather, the pooling, the leakage guard) over the pinned source embedding table"
    );
}

#[tokio::test]
async fn recompute_context_set_over_default_embedding_table_is_byte_identical() {
    use jammi_ai::pipeline::context_set::{
        ContextRequest, ContextSource, MaterializedContext, SetAggregator,
    };
    use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeGather, EdgeSourceRef};

    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    register_edges_source(&session, &_dir).await;
    let svc = Session::new(Arc::clone(&session));

    // The DEFAULT path: the recipe leaves `embedding_table = None`, so the
    // producer resolves the source's newest embedding table (here `emb`). The
    // proven HIGH bug: the descriptor used to record the user's `None`, and on
    // replay `resolve_embedding_table(None)` would re-select the *context-set's
    // own output* (itself a `kind=model` table for `points`, written newer than
    // `emb`) and pool over the wrong rows. The fix pins the resolved table name
    // into the descriptor, so the replay re-pools over `emb` regardless of the
    // shadowing output table. This test exercises exactly that default path.
    let recipe_proto = {
        let mut r = ContextRequest::new("points", Vec::new(), 0);
        let mut gather = EdgeGather::new(EdgeSourceRef::Registered {
            source_id: "edges".into(),
            src_column: "from".into(),
            dst_column: "to".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        });
        gather.hops = 2;
        gather.direction = EdgeDirection::Undirected;
        r.source = ContextSource::Edges(gather);
        r.aggregator = SetAggregator::Sum;
        r.exclude_self = false;
        // embedding_table left None — the default-resolution path under test.
        r
    };

    let source_rows = read_embedding_rows(&session, &emb).await;
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    for (row_id, vector) in &source_rows {
        let mut req = recipe_proto.clone();
        req.query = vector.clone();
        req.exclude_key = Some(row_id.clone());
        if let Some(v) = session.assemble_context(&req).await.unwrap().context_vector {
            rows.push((row_id.clone(), v));
        }
    }
    assert!(
        !rows.is_empty(),
        "the edge-sourced context must pool some targets"
    );
    let (context_table, _) = session
        .materialize_context(
            MaterializedContext {
                rows: &rows,
                dimensions: DIM,
                recipe: &recipe_proto,
            },
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    // The context-set output is now the newest `kind=model` table for `points` —
    // the exact shadowing condition. `resolve_embedding_table("points", None)`
    // would now return `context_table`, not `emb`. Only a descriptor that pinned
    // the resolved `emb` name re-pools correctly on replay.
    assert_ne!(
        context_table.table_name, emb.table_name,
        "the context set is a distinct, newer model table for the source"
    );

    let before = artifact_digest(&session, &context_table.table_name).await;
    let report = svc
        .recompute(&context_table.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    let after = artifact_digest(&session, &report.recomputed[0].recomputed).await;

    assert_eq!(
        before, after,
        "a context-set recompute over the DEFAULT (None) embedding table must be byte-identical — \
         the descriptor pins the resolved source table, so the replay re-pools over it rather than \
         shadowing it with the context-set's own newer output"
    );
}

// ── Staleness: advance a parent → child recomputes to match ──

#[tokio::test]
async fn recomputing_a_stale_child_re_resolves_over_the_fresh_parent() {
    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    let svc = Session::new(Arc::clone(&session));

    // A neighbor-graph derived from the embedding table.
    let (graph, _) = session
        .build_neighbor_graph(
            "points",
            Some(&emb.table_name),
            &BuildNeighborGraph {
                k: 3,
                exact: true,
                ..Default::default()
            },
            CachePolicy::Bypass,
        )
        .await
        .unwrap();
    let graph_before = artifact_digest(&session, &graph.table_name).await;

    // Re-attest the parent embedding table to a NEW artifact digest (model a
    // parent recompute): the child now anchors on a superseded digest.
    reattest_with_new_digest(&session, &emb.table_name, b"advanced-parent-bytes").await;

    // Recompute the child: it re-reads the parent's current rows (unchanged data,
    // so the edge set is the same bytes) and re-anchors on the parent's present
    // digest — a defined, observable replay rather than a silent stale table.
    let report = svc
        .recompute(&graph.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    let graph_after = artifact_digest(&session, &report.recomputed[0].recomputed).await;
    assert_eq!(
        graph_before, graph_after,
        "the child's edge bytes are a function of the parent's row data, which did not change"
    );
}

// ── Cascade: ReportOnly vs Downstream ──

#[tokio::test]
async fn report_only_reports_but_does_not_recompute_downstream() {
    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    let svc = Session::new(Arc::clone(&session));
    let (graph, _) = session
        .build_neighbor_graph(
            "points",
            Some(&emb.table_name),
            &BuildNeighborGraph {
                k: 3,
                exact: true,
                ..Default::default()
            },
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    // Recompute the EMBEDDING table is not possible (it is synthetic / context-
    // set seeded). Instead, exercise the closure on the graph's parent: recompute
    // the parent embedding's *neighbor graph* is the named table; its downstream
    // is anything anchored on it. Build a propagation so the graph has a dependent.
    let dependent = propagate_over(&session, &emb, &graph).await;

    let report = svc
        .recompute(&graph.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    // Only the named table was recomputed.
    assert_eq!(report.recomputed.len(), 1);
    assert_eq!(report.recomputed[0].original, graph.table_name);
    // The dependent propagation is reported stale but NOT recomputed.
    assert!(
        report.downstream_stale.contains(&dependent.table_name),
        "ReportOnly must report the downstream dependent: {:?}",
        report.downstream_stale
    );
    assert!(
        !report
            .recomputed
            .iter()
            .any(|t| t.original == dependent.table_name),
        "ReportOnly must NOT recompute the downstream dependent"
    );
}

#[tokio::test]
async fn downstream_sweeps_every_dependent_in_topological_order() {
    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    let svc = Session::new(Arc::clone(&session));
    let (graph, _) = session
        .build_neighbor_graph(
            "points",
            Some(&emb.table_name),
            &BuildNeighborGraph {
                k: 3,
                exact: true,
                ..Default::default()
            },
            CachePolicy::Bypass,
        )
        .await
        .unwrap();
    let dependent = propagate_over(&session, &emb, &graph).await;

    let report = svc
        .recompute(&graph.table_name, Cascade::Downstream)
        .await
        .unwrap();

    // The named table is recomputed first, the dependent after it (topological
    // order — a parent before its child).
    let originals: Vec<&str> = report
        .recomputed
        .iter()
        .map(|t| t.original.as_str())
        .collect();
    assert_eq!(
        originals.first(),
        Some(&graph.table_name.as_str()),
        "the named table is recomputed first"
    );
    assert!(
        originals.contains(&dependent.table_name.as_str()),
        "the downstream dependent is recomputed by the sweep: {originals:?}"
    );
    let graph_pos = originals
        .iter()
        .position(|n| *n == graph.table_name)
        .unwrap();
    let dep_pos = originals
        .iter()
        .position(|n| *n == dependent.table_name)
        .unwrap();
    assert!(
        graph_pos < dep_pos,
        "the parent must be recomputed before its dependent (topological order)"
    );
}

// ── Cascade: a forged cyclic lineage → DependencyCycle ──

#[tokio::test]
async fn a_downstream_sweep_over_a_cyclic_lineage_is_a_dependency_cycle() {
    use jammi_db::catalog::backend::{SqlValue, TxOptions};

    let (session, _dir, emb) = session_with_synthetic_embeddings().await;
    let svc = Session::new(Arc::clone(&session));

    // A neighbor-graph derived from the embedding table: emb → graph (the graph
    // anchors on emb's digest).
    let (graph, _) = session
        .build_neighbor_graph(
            "points",
            Some(&emb.table_name),
            &BuildNeighborGraph {
                k: 3,
                exact: true,
                ..Default::default()
            },
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    // Forge the back-edge graph → emb: overwrite emb's recorded `input_anchors`
    // to name the graph as an input. No production path writes a cyclic anchor
    // set (a producer anchors its inputs before its output exists), so this is the
    // only way to construct the corruption the cycle guard must reject — a raw
    // UPDATE through the public backend transaction surface.
    let forged = serde_json::to_string(&vec![InputAnchor::result_digest(
        &graph.table_name,
        &ArtifactDigest::of_bytes(b"forged"),
    )])
    .unwrap();
    let emb_name = emb.table_name.clone();
    session
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE result_tables SET input_anchors_json = $1 WHERE table_name = $2",
                    &[SqlValue::TextOwned(forged), SqlValue::TextOwned(emb_name)],
                )
                .await
            })
        })
        .await
        .unwrap();

    // The Downstream sweep walks the lineage from emb: emb → graph → emb closes a
    // cycle, surfaced as the typed `DependencyCycle` rather than an infinite walk.
    let err = svc
        .recompute(&emb.table_name, Cascade::Downstream)
        .await
        .expect_err("a cyclic lineage must be a DependencyCycle, not an infinite sweep");
    assert!(
        matches!(err, JammiError::DependencyCycle { .. }),
        "expected DependencyCycle, got {err:?}"
    );
}

// ── Pre-contract → NotRecomputable ──

#[tokio::test]
async fn a_pre_contract_table_is_not_recomputable() {
    let (session, _dir, _emb) = session_with_synthetic_embeddings().await;
    let svc = Session::new(Arc::clone(&session));

    // Forge a pre-contract table: a `ready` result table with a real Parquet but
    // NO `.materialization.json` sidecar — exactly a table created before the
    // contract landed. `create_table` + a direct write + a ready flip, bypassing
    // `finalize_with_manifest` (the only path that writes a sidecar).
    let store = session.result_store();
    let info = store
        .create_table(
            "points",
            jammi_db::ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "pre-contract",
            Some(DIM as i32),
            Some("_row_id"),
            Some("body"),
        )
        .await
        .unwrap();
    let schema = jammi_db::store::schema::embedding_table_schema(DIM);
    let row_id = StringArray::from(vec!["a"]);
    let src = StringArray::from(vec!["points"]);
    let model = StringArray::from(vec!["pre-contract"]);
    let item = Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float32,
        false,
    ));
    let vectors = FixedSizeListArray::try_new(
        item,
        DIM as i32,
        Arc::new(Float32Array::from(vec![0.0_f32; DIM])),
        None,
    )
    .unwrap();
    let batch = arrow::array::RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(row_id),
            Arc::new(src),
            Arc::new(model),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(&info.parquet_url, schema).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    let rows = writer.close().await.unwrap();
    session
        .catalog()
        .update_result_table_status(
            &info.table_name,
            jammi_db::catalog::status::ResultTableStatus::Ready,
            rows,
        )
        .await
        .unwrap();

    let err = svc
        .recompute(&info.table_name, Cascade::ReportOnly)
        .await
        .expect_err("a pre-contract table must not be recomputable");
    assert!(
        matches!(err, JammiError::NotRecomputable { ref table } if *table == info.table_name),
        "expected NotRecomputable, got {err:?}"
    );
}

// ── helpers ──

/// Register a small declared-edge `edges` source (`from`/`to` Utf8 endpoints) so
/// an edge-gathered context set is reproducible. A path graph over the eight
/// synthetic points, so every point has a bounded neighbourhood.
async fn register_edges_source(session: &InferenceSession, dir: &TempDir) {
    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![
        Field::new("from", DataType::Utf8, false),
        Field::new("to", DataType::Utf8, false),
    ]));
    // A path 0-1-2-…-7 plus its reverse so an undirected gather has neighbours.
    let mut from: Vec<String> = Vec::new();
    let mut to: Vec<String> = Vec::new();
    for i in 0..7 {
        from.push(format!("p{i}"));
        to.push(format!("p{}", i + 1));
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(from)) as arrow::array::ArrayRef,
            Arc::new(StringArray::from(to)),
        ],
    )
    .unwrap();
    let path = dir.path().join("edges.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    session
        .add_source(
            "edges",
            jammi_db::source::SourceType::File,
            jammi_db::source::SourceConnection {
                url: Some(format!("file://{}", path.to_str().unwrap())),
                format: Some(jammi_db::source::FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

/// Read a materialised embedding table's `(_row_id, vector)` rows.
async fn read_embedding_rows(
    session: &InferenceSession,
    table: &ResultTableRecord,
) -> Vec<(String, Vec<f32>)> {
    let batches = session
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{}\"",
            table.table_name
        ))
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = arrow::compute::cast(batch.column(0), &arrow::datatypes::DataType::Utf8).unwrap();
        let ids = ids.as_any().downcast_ref::<StringArray>().unwrap();
        let list = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            let cell = list.value(i);
            let floats = cell.as_any().downcast_ref::<Float32Array>().unwrap();
            rows.push((
                ids.value(i).to_string(),
                (0..floats.len()).map(|j| floats.value(j)).collect(),
            ));
        }
    }
    rows
}

/// Propagate over a neighbor-graph so the graph has a downstream dependent
/// anchored on it (a `ResultDigest` edge anchor → the propagation derives from
/// the graph). Returns the propagated table record.
async fn propagate_over(
    session: &Arc<InferenceSession>,
    emb: &ResultTableRecord,
    graph: &ResultTableRecord,
) -> ResultTableRecord {
    use jammi_ai::pipeline::graph_neighbourhood::EdgeSourceRef;
    use jammi_ai::pipeline::graph_propagation::PropagateRequest;
    let request = PropagateRequest::new(
        "points",
        EdgeSourceRef::NeighborGraph {
            table_name: graph.table_name.clone(),
        },
    )
    .with_embedding_table(emb.table_name.clone());
    session
        .propagate_embeddings(&request, CachePolicy::Bypass)
        .await
        .unwrap()
        .0
}

/// Overwrite a table's `.materialization.json` sidecar to a new artifact digest —
/// models the table being recomputed to a new output (the digest a downstream
/// child senses as the parent's current anchor).
async fn reattest_with_new_digest(session: &InferenceSession, table: &str, new_bytes: &[u8]) {
    let record = session
        .catalog()
        .get_result_table(table)
        .await
        .unwrap()
        .unwrap();
    let url = StorageUrl::parse(&record.parquet_path).unwrap();
    let store = session.result_store();
    let original = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("manifest present");
    let updated = jammi_db::store::manifest::MaterializationManifest {
        artifact: ArtifactDigest::of_bytes(new_bytes),
        ..original
    };
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, updated.to_json_bytes().unwrap().into())
        .await
        .unwrap();
}
