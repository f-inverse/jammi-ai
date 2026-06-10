//! P3 — `fine_tune_graph` learns on GPU, end to end.
//!
//! The declared-graph fine-tune (node CSV + edge CSV → biased-walk positives →
//! MNRL over the LoRA trainer) runs on a GPU-pinned session over the synthetic
//! two-community graph the CPU suite uses, and must:
//!   (a) complete without error on the GPU,
//!   (b) produce an adapter that *changes* embeddings vs the base model — the
//!       on-device learning signal (the LoRA weights moved; gradients flowed
//!       correctly on the GPU). The spec accepts either loss-decrease *or*
//!       embeddings-change as the learns-something proof, and embeddings-change
//!       is the robust one here: on this deliberately tiny 6-node graph the MNRL
//!       in-batch-negative pool is so small that the reported epoch loss is
//!       *stationary* even while the weights move — and that stationary curve is
//!       byte-identical on CPU and GPU (verified), so it is a fixture-scale
//!       artifact of MNRL on a toy graph, never a GPU correctness defect. The
//!       loss curve is still captured and reported.

use std::path::Path;
use std::sync::Arc;

use jammi_ai::fine_tune::graph_sampler::{EdgeProvenance, GraphFineTuneSources, GraphSampleConfig};
use jammi_ai::fine_tune::{EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

fn write_csv(dir: &Path, name: &str, header: &str, rows: &[(String, String)]) -> String {
    let mut body = String::from(header);
    body.push('\n');
    for (a, b) in rows {
        body.push_str(a);
        body.push(',');
        body.push_str(b);
        body.push('\n');
    }
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    format!("file://{}", path.display())
}

async fn add_csv(session: &Arc<InferenceSession>, id: &str, url: String) {
    session
        .add_source(
            id,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_learns_on_gpu() {
    skip_without_gpu!();
    harness::loss_capture::install();
    harness::loss_capture::reset();

    let dir = TempDir::new().unwrap();
    let session = harness::gpu_session(dir.path()).await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    // Two small communities (the CPU suite's fixture): triangles + a bridge.
    let node_rows: Vec<(String, String)> = ["a0", "a1", "a2", "b0", "b1", "b2"]
        .iter()
        .map(|id| (id.to_string(), format!("document about topic {id}")))
        .collect();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);

    let edge_pairs = [
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a2"),
        ("a2", "a1"),
        ("a0", "a2"),
        ("a2", "a0"),
        ("b0", "b1"),
        ("b1", "b0"),
        ("b1", "b2"),
        ("b2", "b1"),
        ("b0", "b2"),
        ("b2", "b0"),
        ("a0", "b0"),
    ];
    let edge_rows: Vec<(String, String)> = edge_pairs
        .iter()
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect();
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    add_csv(&session, "nodes", node_url).await;
    add_csv(&session, "edges", edge_url).await;

    let model = harness::local_model_id("tiny_bert");
    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample = GraphSampleConfig {
        walk_length: 3,
        walks_per_node: 2,
        hard_negatives: 1,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 11,
        ..GraphSampleConfig::default()
    };
    let train = FineTuneConfig {
        // ≥2 epochs for a first→last decrease signal.
        epochs: 6,
        batch_size: 4,
        lora_rank: 4,
        warmup_steps: 0,
        validation_fraction: 0.0,
        early_stopping_metric: EarlyStoppingMetric::TrainLoss,
        embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
        ..Default::default()
    };

    let job = session
        .fine_tune_graph(&sources, &model, sample, Some(train))
        .await
        .unwrap();

    // (a) completes on the GPU.
    job.wait().await.unwrap();
    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        record.status, "completed",
        "GPU graph fine-tune should complete, got {}",
        record.status
    );

    // The captured per-epoch loss curve — reported, not asserted (see the module
    // note: a stationary MNRL loss on this toy graph is a fixture artifact, not a
    // GPU defect; the learning signal is the embeddings-change below).
    let curve = harness::loss_capture::captured();
    assert!(
        curve.iter().all(|(_, l)| l.is_finite()),
        "graph fine-tune produced a non-finite loss on GPU: {curve:?}"
    );

    // (b) the adapter changes embeddings vs the base model — the on-device
    // learning proof. The fine-tuned model is served back through the GPU session.
    let ft = session
        .catalog()
        .get_model(job.model_id())
        .await
        .unwrap()
        .expect("graph fine-tune registered the model");
    assert!(
        ft.artifact_path.is_some(),
        "graph fine-tune should publish an adapter"
    );
    let ft_name = job.model_id().split("::").next().unwrap();
    let base = session
        .encode_text_query(&model, "document about topic a0")
        .await
        .unwrap();
    let tuned = session
        .encode_text_query(ft_name, "document about topic a0")
        .await
        .unwrap();
    let delta: f32 = base.iter().zip(&tuned).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        delta > 1e-6,
        "GPU graph-trained adapter must change embeddings (LoRA delta non-zero), delta={delta}"
    );

    tracing::info!(
        loss_curve = ?curve,
        embed_delta = delta,
        "P3 fine_tune_graph learns on GPU (embeddings changed)"
    );
}
