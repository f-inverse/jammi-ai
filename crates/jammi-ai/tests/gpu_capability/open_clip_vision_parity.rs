//! P1 — CPU↔GPU parity for `generate_image_embeddings` and
//! `encode_image_query` over the OpenCLIP **vision** tower
//! (`OpenClipVision`).
//!
//! gpu-parity-cell: OpenClipVision × ImageEmbedding
//!
//! The vision tower's patch-embed + transformer forward is a distinct kernel
//! path from every text-tower cell (`ClipText × TextEmbedding`,
//! `embeddings_parity`, `modernbert_embeddings_parity`) — a 2-D patch
//! convolution feeding the same transformer block shape, projected into the
//! shared OpenCLIP `embed_dim`. The same `tiny_open_clip` vision tower runs
//! over the same `tiny_image_corpus` fixture (packed into a `(figure_id,
//! image)` Parquet table, the shape `generate_image_embeddings` expects) on
//! a GPU-pinned and a CPU-pinned session; the resulting per-row embedding
//! vectors (keyed by `_row_id`, so the comparison is row-exact regardless of
//! scan order) and the same encoded image-query vector must match within the
//! parity tolerance.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

use crate::harness;
use crate::skip_without_gpu;

/// Pack every top-level `.png` under `cookbook/fixtures/tiny_image_corpus/`
/// (the per-clip files; the `queries/` subdirectory holds held-out query
/// images and is skipped by the `is_file()` filter since it is a
/// subdirectory) into a `(figure_id, image)` Parquet table at
/// `dir/image_corpus.parquet`, the Binary-column shape
/// `generate_image_embeddings` expects (mirrors `recipe_generate_image_embeddings`'s
/// `figures.parquet` shape).
fn write_image_corpus(dir: &Path) -> std::path::PathBuf {
    let corpus_dir = harness::cookbook_fixture("tiny_image_corpus");
    let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "tiny_image_corpus has no .png files");

    let ids: Vec<String> = entries
        .iter()
        .map(|p| p.file_stem().unwrap().to_str().unwrap().to_string())
        .collect();
    let images: Vec<Vec<u8>> = entries.iter().map(|p| std::fs::read(p).unwrap()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("figure_id", DataType::Utf8, false),
        Field::new("image", DataType::Binary, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(BinaryArray::from(
                images.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();

    let path = dir.join("image_corpus.parquet");
    let mut w = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

/// Register the packed image corpus as a source named `"figures"` on
/// `session`.
async fn add_image_corpus(session: &Arc<InferenceSession>, parquet_path: &Path) {
    session
        .add_source(
            "figures",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", parquet_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn open_clip_vision_generate_embeddings_cpu_gpu_parity() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let model = harness::local_fixture_model_id("tiny_open_clip");

    let cpu_dir = TempDir::new().unwrap();
    let cpu_corpus = write_image_corpus(cpu_dir.path());
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    add_image_corpus(&cpu, &cpu_corpus).await;
    let (cpu_table, _) = cpu
        .generate_image_embeddings("figures", &model, "image", "figure_id", CachePolicy::Bypass)
        .await
        .unwrap();
    let cpu_vecs = harness::keyed_result_vectors(&cpu, &cpu_table).await;

    let gpu_dir = TempDir::new().unwrap();
    let gpu_corpus = write_image_corpus(gpu_dir.path());
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    add_image_corpus(&gpu, &gpu_corpus).await;
    let (gpu_table, _) = gpu
        .generate_image_embeddings("figures", &model, "image", "figure_id", CachePolicy::Bypass)
        .await
        .unwrap();
    let gpu_vecs = harness::keyed_result_vectors(&gpu, &gpu_table).await;

    assert_eq!(
        cpu_table.dimensions, gpu_table.dimensions,
        "CPU and GPU embedding tables must share a dimension (OpenCLIP embed_dim)"
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
        let (cos, abs) = harness::assert_parity(
            &format!("open_clip_vision_generate_embeddings[{id}]"),
            cpu_v,
            gpu_v,
        );
        worst_cos = worst_cos.min(cos);
        worst_abs = worst_abs.max(abs);
    }
    tracing::info!(
        rows = cpu_vecs.len(),
        worst_cos,
        worst_abs,
        "OpenCLIP vision-tower generate_embeddings parity over tiny_image_corpus"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn open_clip_vision_encode_query_cpu_gpu_parity() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let model = harness::local_fixture_model_id("tiny_open_clip");
    let image_bytes =
        std::fs::read(harness::cookbook_fixture("tiny_image_corpus").join("img_circle_0.png"))
            .unwrap();

    let cpu_dir = TempDir::new().unwrap();
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    let cpu_vec = cpu.encode_image_query(&model, &image_bytes).await.unwrap();

    let gpu_dir = TempDir::new().unwrap();
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    let gpu_vec = gpu.encode_image_query(&model, &image_bytes).await.unwrap();

    assert!(
        cpu_vec.iter().any(|&v| v != 0.0),
        "query vector must not be all-zero"
    );
    harness::assert_parity("open_clip_vision_encode_query", &cpu_vec, &gpu_vec);
}
