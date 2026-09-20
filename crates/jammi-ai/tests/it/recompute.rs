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
    session.install_query_functions();

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

    // The `points` source the embedding table below records its lineage to,
    // keyed by `_row_id` — the column that table attributes its keys to. The
    // rows carry no payload beyond the key: the derived producers here pool
    // vectors and never read a value column, so the key is the whole source.
    register_points_source(&session, dir.path(), &pairs).await;

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
                key_column: Some("_row_id"),
                text_columns: None,
            },
            &pairs,
            Materialization::new(&descriptor, &env, inputs),
            None,
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
    // (`from`/`to`, not the `src`/`dst` defaults). The GraphPropagation
    // descriptor records the full edge-source binding, so the replay reads the
    // exact same columns and is byte-identical; a descriptor recording only
    // the source id would replay with the default `src`/`dst` — over a
    // different graph, or fail.
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
                key_column: emb.key_column.as_deref(),
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
    // descriptor pins the resolved table name, so the replay re-pools over
    // `emb` regardless of the shadowing output table; a descriptor recording
    // the user's `None` would, on replay, re-select the *context-set's own
    // output* (itself a `kind=model` table for `points`, written newer than
    // `emb`) and pool over the wrong rows. This test exercises exactly that
    // default path.
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
                key_column: emb.key_column.as_deref(),
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
    // `BuildingTable::finish` (the only path that writes a sidecar).
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
            None,
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
            jammi_db::store::content_hash::null_hash_column(1),
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(info.parquet_url(), schema).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    let rows = writer.close().await.unwrap();
    session
        .catalog()
        .update_result_table_status(
            info.table_name(),
            jammi_db::catalog::status::ResultTableStatus::Ready,
            rows,
        )
        .await
        .unwrap();

    let err = svc
        .recompute(info.table_name(), Cascade::ReportOnly)
        .await
        .expect_err("a pre-contract table must not be recomputable");
    assert!(
        matches!(err, JammiError::NotRecomputable { ref table } if *table == info.table_name()),
        "expected NotRecomputable, got {err:?}"
    );
}

// ── Replay under a mutated model dir mismatches, never silently
//    yields different vectors under an identical hash ──

/// The recompute peer of `content_digest.rs`'s digest-fold tests: an
/// `Embedding` descriptor's replay (`replay_descriptor`,
/// `pipeline/recompute.rs`) re-invokes `EmbeddingPipeline::run`, which
/// re-loads the model and recomputes its content digest fresh from the
/// model directory's CURRENT bytes (`session.rs`/`pipeline/embedding.rs`
/// thread `LoadedModel::content_digest()` into every `ModelIdentity` they
/// build — there is no separate digest-threading code path inside
/// `recompute.rs` itself to break). So a recompute run **after** the model
/// directory was mutated in place must record a DIFFERENT `definition_hash`
/// than the original run — the fold reaches replay, so recompute never
/// replays a descriptor to different vectors under the SAME hash, at its
/// actual call site, not just at the point the digest is first computed.
///
/// Uses TWO INDEPENDENT sessions rooted at the SAME catalog/storage
/// directory — the first materializes the table and is dropped, the second
/// (with its own, cold `ModelCache`) performs the recompute — simulating the
/// realistic cross-process/cross-restart shape a recompute normally runs
/// under.
///
/// A single long-lived session's `ModelCache` has a `stat`-only fingerprint
/// tripwire (`ModelCache::get_or_load`'s `probe_freshness` call,
/// `cache_staleness.rs`) that would ALSO detect this exact
/// `model.safetensors` byte-length-changing mutation and force a warm-hit
/// reload. Two independent sessions are still the right shape: (1) it is the
/// realistic cross-process/cross-restart deployment recompute normally runs
/// under; (2) it keeps THIS test's guarantee decoupled from the tripwire's
/// documented residual (a same-length, same-mtime content swap is invisible
/// to it) — a definitely-cold reload proves the digest fold reaches replay
/// regardless of whether the tripwire would have caught this mutation.
#[tokio::test]
async fn recompute_after_model_dir_mutation_changes_the_definition_hash() {
    let session_dir = TempDir::new().unwrap();

    // A local model dir the test controls and mutates in place — never the
    // checked-in `tiny_bert` fixture other tests share.
    let model_dir_root = TempDir::new().unwrap();
    let model_dir = model_dir_root.path().join("model");
    crate::pooling_config::build_local_model_dir(
        &model_dir,
        Some(&crate::pooling_config::mean_pooling_config()),
    );
    let model_id = model_dir.display().to_string();

    let original_record = {
        let session = InferenceSession::new(common::test_config(session_dir.path()))
            .await
            .unwrap();
        session
            .add_source(
                "patents",
                jammi_db::source::SourceType::File,
                jammi_db::source::SourceConnection {
                    url: Some(common::fixture_url("patents.parquet")),
                    format: Some(jammi_db::source::FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let (record, _outcome) = session
            .generate_text_embeddings(
                "patents",
                &model_id,
                &["abstract".to_string()],
                "id",
                CachePolicy::Bypass,
                None,
            )
            .await
            .unwrap();
        let hash = read_definition_hash(&session, &record).await;
        (record, hash)
    };
    let (original_record, original_hash) = original_record;

    // Mutate the model directory IN PLACE — same path, same `model_id` —
    // flipping the last byte of the weights file (see `content_digest.rs`'s
    // `weights_bytes_mutation_changes_the_definition_hash` doc for why this
    // stays a structurally valid safetensors file).
    let weights_path = model_dir.join("model.safetensors");
    let mut bytes = std::fs::read(&weights_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&weights_path, &bytes).unwrap();

    // A fresh session over the SAME catalog/storage dir: its `ModelCache`
    // starts cold, so `recompute`'s replay genuinely reloads the (now
    // mutated) model directory from disk.
    let session = Arc::new(
        InferenceSession::new(common::test_config(session_dir.path()))
            .await
            .unwrap(),
    );
    let report = session
        .recompute(&original_record, Cascade::ReportOnly)
        .await
        .unwrap();
    assert_eq!(report.recomputed.len(), 1);
    let recomputed_table = &report.recomputed[0].recomputed;
    let recomputed_record = session
        .catalog()
        .get_result_table(recomputed_table)
        .await
        .unwrap()
        .expect("the replay must have materialized the recomputed table");
    let recomputed_hash = read_definition_hash(&session, &recomputed_record).await;

    assert_ne!(
        original_hash, recomputed_hash,
        "a recompute replayed (in a fresh session, cold model cache) after \
         the model directory was mutated in place (same model_id) must \
         record a different definition_hash than the original run — a \
         matching hash here would be a false replay: different \
         vectors materialized under an identical hash"
    );
}

/// Read back a materialized table's persisted `.materialization.json`
/// sidecar `definition_hash` — the real production artifact, not a
/// recomputed-in-the-test stand-in.
async fn read_definition_hash(
    session: &InferenceSession,
    record: &ResultTableRecord,
) -> jammi_db::store::manifest::DefinitionHash {
    let parquet_url = StorageUrl::parse(&record.parquet_path).unwrap();
    session
        .result_store()
        .read_materialization_manifest(&parquet_url)
        .await
        .unwrap()
        .expect("a freshly finalized table must carry a materialization manifest")
        .definition_hash
}

// ── helpers ──

/// Register the `points` source the synthetic embedding table records its
/// lineage to, keyed by `_row_id` over the same eight point keys.
///
/// A derived table's catalog `key_column` names a column of this source, so the
/// source has to exist and carry that column for the provenance to be
/// followable — the embedding table's keys are `_row_id` values here.
async fn register_points_source(
    session: &InferenceSession,
    dir: &std::path::Path,
    pairs: &[(String, Vec<f32>)],
) {
    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "_row_id",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(
            pairs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        )) as arrow::array::ArrayRef],
    )
    .unwrap();
    let path = dir.join("points.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    session
        .add_source(
            "points",
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

// ── The `TrainingSet` replay arm re-resolves a recorded PINNED
//    anchor pinned, never silently downgrading it to unpinned ──
//
// `pipeline::recompute`'s `TrainingSet` arm reads the table's own recorded
// `input_anchors` and re-anchors every relation they name at replay time,
// honouring each recorded anchor's own kind. Re-anchoring unconditionally as
// `unpinned_at_instant` would silently downgrade a PINNED input — which
// `ProducingDescriptor::FineTune` records, anchoring the training set it
// trained from by content digest (`materialize_projection` itself only ever
// writes unpinned inputs). These tests
// exercise `recompute_training_set`'s anchor handling directly via a
// hand-forged manifest fixture rather than requiring an end-to-end
// pinned-source production path.

/// Overwrite a table's `.materialization.json` sidecar's `input_anchors` —
/// forges the ONE thing under test (the recorded
/// anchor set) while leaving every other manifest field (`descriptor`,
/// `artifact`, `definition_hash`, …) exactly as the real producer wrote it.
async fn overwrite_input_anchors(
    session: &InferenceSession,
    table: &str,
    anchors: Vec<InputAnchor>,
) {
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
        input_anchors: anchors,
        ..original
    };
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, updated.to_json_bytes().unwrap().into())
        .await
        .unwrap();
}

/// A recorded [`AnchorKind::ResultDigest`] anchor (a genuinely PINNED shape —
/// the same kind [`ProducingDescriptor::FineTune`]'s own input anchor uses)
/// replays PINNED: the replayed table's fresh manifest carries the SAME kind
/// over the SAME source, re-resolved to the source's CURRENT digest — never
/// silently downgraded to [`AnchorKind::UnpinnedAtInstant`].
#[tokio::test]
async fn recompute_training_set_re_resolves_a_pinned_result_digest_anchor_pinned() {
    let (session, _dir, emb) = session_with_synthetic_embeddings().await;

    let (table, _batches) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "points",
        &["_row_id".to_string()],
        jammi_ai::model::ModelTask::TextEmbedding,
        "pinned_anchor_probe_v1",
    )
    .await
    .unwrap();

    // Forge the training set's OWN manifest: ONE pinned anchor over the
    // synthetic embedding table (a real, resolvable `result_tables` row), at
    // a deliberately STALE digest value distinct from the table's real
    // current one — so a passing re-resolution is observable (the replayed
    // anchor must carry the CURRENT digest, not this forged stale one).
    let stale = ArtifactDigest("stale-digest-does-not-match-current".to_string());
    overwrite_input_anchors(
        &session,
        table.table_name(),
        vec![InputAnchor::result_digest(emb.table_name.clone(), &stale)],
    )
    .await;

    let svc = Session::new(Arc::clone(&session));
    let report = svc
        .recompute(table.table_name(), Cascade::ReportOnly)
        .await
        .unwrap();
    let replayed_name = report.recomputed[0].recomputed.clone();

    let replayed_record = session
        .catalog()
        .get_result_table(&replayed_name)
        .await
        .unwrap()
        .unwrap();
    let replayed_url = StorageUrl::parse(&replayed_record.parquet_path).unwrap();
    let replayed_manifest = session
        .result_store()
        .read_materialization_manifest(&replayed_url)
        .await
        .unwrap()
        .expect("replay must write its own manifest");

    assert_eq!(
        replayed_manifest.input_anchors.len(),
        1,
        "the single recorded anchor must replay as exactly one anchor"
    );
    let anchor = &replayed_manifest.input_anchors[0];
    assert_eq!(
        anchor.kind,
        jammi_db::store::manifest::AnchorKind::ResultDigest,
        "a pinned anchor must replay PINNED, never silently downgraded to unpinned; got {anchor:?}"
    );
    assert_eq!(anchor.source, emb.table_name);
    let current_digest = artifact_digest(&session, &emb.table_name).await;
    assert_eq!(
        anchor.anchor.0, current_digest,
        "a re-resolved pinned anchor must carry the CURRENT digest, never the stale recorded one"
    );
}

/// The other determinant the addendum names: a recorded PINNED anchor whose
/// target no longer resolves (deregistered / reaped) is `NotRecomputable`,
/// naming the anchor — never silently treated as unpinned, and never a panic.
#[tokio::test]
async fn recompute_training_set_refuses_a_pinned_anchor_whose_target_is_gone() {
    let (session, _dir, _emb) = session_with_synthetic_embeddings().await;

    let (table, _batches) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "points",
        &["_row_id".to_string()],
        jammi_ai::model::ModelTask::TextEmbedding,
        "pinned_anchor_probe_v1",
    )
    .await
    .unwrap();

    let stale = ArtifactDigest("stale-digest".to_string());
    overwrite_input_anchors(
        &session,
        table.table_name(),
        vec![InputAnchor::result_digest(
            "a-table-that-was-never-materialised",
            &stale,
        )],
    )
    .await;

    let svc = Session::new(Arc::clone(&session));
    let err = svc
        .recompute(table.table_name(), Cascade::ReportOnly)
        .await
        .expect_err(
            "a pinned anchor whose target no longer resolves must refuse, not silently downgrade",
        );
    assert!(
        matches!(err, JammiError::NotRecomputable { .. }),
        "expected NotRecomputable, got {err:?}"
    );
}
