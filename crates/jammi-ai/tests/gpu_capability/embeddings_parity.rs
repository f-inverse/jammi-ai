//! P1 — CPU↔GPU parity for `generate_text_embeddings` and `encode_text_query`.
//!
//! gpu-parity-cell: Bert × TextEmbedding
//!
//! The same `tiny_bert` encoder runs over the same `patents.parquet` subset on a
//! GPU-pinned and a CPU-pinned session; the resulting per-row embedding vectors
//! (keyed by `_row_id`, so the comparison is row-exact regardless of scan order)
//! and the same encoded query vector must match within the parity tolerance.

use jammi_db::store::CachePolicy;
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

#[tokio::test(flavor = "multi_thread")]
async fn generate_embeddings_cpu_gpu_parity() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let model = harness::local_model_id("tiny_bert");

    let cpu_dir = TempDir::new().unwrap();
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    harness::add_patents(&cpu).await;
    let (cpu_table, _) = cpu
        .generate_text_embeddings(
            "patents",
            &model,
            &["abstract".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .unwrap();
    let cpu_vecs = harness::keyed_result_vectors(&cpu, &cpu_table).await;

    let gpu_dir = TempDir::new().unwrap();
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    harness::add_patents(&gpu).await;
    let (gpu_table, _) = gpu
        .generate_text_embeddings(
            "patents",
            &model,
            &["abstract".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .unwrap();
    let gpu_vecs = harness::keyed_result_vectors(&gpu, &gpu_table).await;

    assert_eq!(
        cpu_table.dimensions, gpu_table.dimensions,
        "CPU and GPU embedding tables must share a dimension"
    );
    assert!(
        !cpu_vecs.is_empty() && cpu_vecs.len() == gpu_vecs.len(),
        "CPU ({}) and GPU ({}) embeddings must cover the same rows",
        cpu_vecs.len(),
        gpu_vecs.len()
    );

    let mut worst_cos = 1.0f64;
    let mut worst_abs = 0.0f64;
    for (id, cpu_v) in &cpu_vecs {
        let gpu_v = gpu_vecs.get(id).expect("matching _row_id on GPU");
        let (cos, abs) =
            harness::assert_parity(&format!("generate_embeddings[{id}]"), cpu_v, gpu_v);
        worst_cos = worst_cos.min(cos);
        worst_abs = worst_abs.max(abs);
    }
    tracing::info!(
        rows = cpu_vecs.len(),
        worst_cos,
        worst_abs,
        "generate_embeddings parity over patents subset"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn encode_query_cpu_gpu_parity() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let model = harness::local_model_id("tiny_bert");
    let query = "a method for quantum error correction in superconducting qubits";

    let cpu_dir = TempDir::new().unwrap();
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    let cpu_vec = cpu.encode_text_query(&model, query).await.unwrap();

    let gpu_dir = TempDir::new().unwrap();
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    let gpu_vec = gpu.encode_text_query(&model, query).await.unwrap();

    assert!(
        cpu_vec.iter().any(|&v| v != 0.0),
        "query vector must not be all-zero"
    );
    harness::assert_parity("encode_query", &cpu_vec, &gpu_vec);
}
