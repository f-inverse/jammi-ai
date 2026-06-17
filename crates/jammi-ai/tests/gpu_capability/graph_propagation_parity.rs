//! P1 — `propagate_embeddings` is device-independent (SGC and APPNP).
//!
//! Unlike the embedding / encode / predict verbs, graph propagation has **no
//! GPU kernel**: it is a fixed-order `f64` fold in Rust with a single final
//! `f32` cast (see `pipeline::graph_propagation`), deterministic and identical
//! regardless of the session's `gpu.device`. So this is not a GPU-kernel parity
//! proof (there is nothing on the GPU to diverge) — it is a **device-independence
//! + determinism guard**: the GPU-pinned session (which really holds CUDA, via
//! `require_gpu`) and the CPU-pinned session run the same propagation over the
//! same fixture and must produce **bit-identical** per-node vectors. A
//! regression that routed propagation through a device-sensitive path, or made
//! it non-deterministic, would break the exact equality.
//!
//! A synthetic homophilous two-class graph (a neutral citation-/co-purchase-style
//! fixture, mirroring the CPU `it::graph_propagation` setup) carries one feature
//! vector per node, propagated under both the SGC weighting (`α = 0`,
//! degree-normalised, multi-hop) and the APPNP weighting (`α`-restart).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
use jammi_ai::pipeline::graph_propagation::{PropagateRequest, PropagationWeighting};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

const DIM: usize = 16;

/// `(node_id, class)` — a node row of the synthetic graph.
type GraphNode = (String, usize);
/// `(src_id, dst_id)` — a declared edge of the synthetic graph.
type GraphEdge = (String, String);

/// A deterministic per-class centroid plus per-node noise, so propagation moves
/// the vectors measurably (the same generator the CPU suite uses).
fn node_vector(id: &str, class: usize) -> Vec<f32> {
    let id_hash = id
        .bytes()
        .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    (0..DIM)
        .map(|i| {
            let centroid = (((class as f32 + 1.0) * (i as f32 + 1.0)) * 0.7).sin();
            let noise = (((id_hash.wrapping_add((i as u32).wrapping_mul(2_654_435_761))) % 1000)
                as f32
                / 1000.0
                - 0.5)
                * 0.9;
            centroid + noise
        })
        .collect()
}

/// Two equal-size classes, fully wired within class — a homophilous graph whose
/// propagated mean denoises toward each class centroid.
fn two_class_homophilous(per_class: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    for class in 0..2 {
        for i in 0..per_class {
            nodes.push((format!("c{class}_n{i}"), class));
        }
    }
    let mut edges = Vec::new();
    for class in 0..2 {
        for i in 0..per_class {
            for j in (i + 1)..per_class {
                edges.push((format!("c{class}_n{i}"), format!("c{class}_n{j}")));
            }
        }
    }
    (nodes, edges)
}

fn write_parquet(dir: &TempDir, name: &str, schema: Arc<Schema>, batch: RecordBatch) -> String {
    let path = dir.path().join(name);
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    format!("file://{}", path.to_str().unwrap())
}

/// Stand up a graph session on `device` carrying the node embedding table and a
/// registered undirected edge source over the synthetic fixture.
async fn graph_session(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    device_cpu: bool,
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let session = if device_cpu {
        harness::cpu_session(dir.path()).await
    } else {
        harness::gpu_session(dir.path()).await
    };
    session.register_query_functions();

    // nodes parquet: _row_id, class.
    let node_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("class", DataType::Int64, false),
    ]));
    let node_batch = RecordBatch::try_new(
        Arc::clone(&node_schema),
        vec![
            Arc::new(StringArray::from(
                nodes.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(arrow::array::Int64Array::from(
                nodes.iter().map(|(_, c)| *c as i64).collect::<Vec<_>>(),
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

    let pairs: Vec<(String, Vec<f32>)> = nodes
        .iter()
        .map(|(id, c)| (id.clone(), node_vector(id, *c)))
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

    // edges parquet: src, dst, weight.
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
        Field::new("weight", DataType::Float64, true),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(StringArray::from(
                edges.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                edges.iter().map(|(_, d)| d.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                edges.iter().map(|_| None::<f64>).collect::<Vec<_>>(),
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

/// A registered undirected edge gather pinned to the original synthetic
/// embedding table.
async fn base_request(session: &Arc<InferenceSession>) -> PropagateRequest {
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
        EdgeSourceRef::Registered {
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

/// Read a propagated table's `(_row_id, vector)` rows into a map.
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

/// Run the same propagation request, built by `tune`, on a CPU-pinned and a
/// GPU-pinned session and assert the per-node vectors are **bit-identical**.
///
/// Propagation has no GPU kernel (a deterministic `f64` fold), so the invariant
/// is exact equality, not a tolerance: the GPU-pinned session — which genuinely
/// holds CUDA (`require_gpu`) — must route propagation through the same
/// device-independent CPU path and produce the same bytes as the CPU session.
async fn device_independence_for(label: &str, tune: impl Fn(PropagateRequest) -> PropagateRequest) {
    let (nodes, edges) = two_class_homophilous(6);

    let (cpu, _cd) = graph_session(&nodes, &edges, true).await;
    let cpu_table = cpu
        .propagate_embeddings(&tune(base_request(&cpu).await))
        .await
        .unwrap();
    let cpu_vecs = read_table_vectors(&cpu, &cpu_table).await;

    let (gpu, _gd) = graph_session(&nodes, &edges, false).await;
    let gpu_table = gpu
        .propagate_embeddings(&tune(base_request(&gpu).await))
        .await
        .unwrap();
    let gpu_vecs = read_table_vectors(&gpu, &gpu_table).await;

    assert_eq!(cpu_vecs.len(), nodes.len(), "all nodes present (CPU)");
    assert_eq!(gpu_vecs.len(), nodes.len(), "all nodes present (GPU)");

    for (id, cpu_v) in &cpu_vecs {
        let gpu_v = gpu_vecs.get(id).expect("matching node on GPU");
        assert_eq!(
            cpu_v, gpu_v,
            "{label}[{id}]: propagation is device-independent, so the GPU-pinned \
             and CPU-pinned sessions must produce bit-identical vectors"
        );
    }
    tracing::info!(
        label,
        nodes = nodes.len(),
        "propagate device-independence (Δ = 0)"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn propagate_sgc_is_device_independent() {
    skip_without_gpu!();
    harness::loss_capture::install();
    // SGC: no restart, degree-normalised Â, three hops.
    device_independence_for("propagate_sgc", |r| {
        r.with_weighting(PropagationWeighting::DegreeNormalized)
            .with_alpha(0.0)
            .with_hops(3)
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn propagate_appnp_is_device_independent() {
    skip_without_gpu!();
    harness::loss_capture::install();
    // APPNP: personalised-PageRank restart (α = 0.3), degree-normalised, three hops.
    device_independence_for("propagate_appnp", |r| {
        r.with_weighting(PropagationWeighting::DegreeNormalized)
            .with_alpha(0.3)
            .with_hops(3)
    })
    .await;
}
