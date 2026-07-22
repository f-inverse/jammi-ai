//! P1 — CPU↔GPU parity for `generate_audio_embeddings` and
//! `encode_audio_query` over the HTSAT-CLAP audio tower (`HtsatAudio`).
//!
//! gpu-parity-cell: HtsatAudio × AudioEmbedding
//!
//! HTSAT's patch-merge Swin-style transformer forward over a mel
//! spectrogram is a distinct kernel path from every text/vision cell in this
//! suite. The same `htsat_clap_tiny` audio tower runs over the same
//! `tiny_audio_corpus` fixture (packed into a `(clip_id, audio)` Parquet
//! table, the shape `generate_audio_embeddings` expects) on a GPU-pinned and
//! a CPU-pinned session; the resulting per-row embedding vectors (keyed by
//! `_row_id`, so the comparison is row-exact regardless of scan order) and
//! the same encoded audio-query vector must match within the parity
//! tolerance.

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

/// Pack every top-level `.wav` under `cookbook/fixtures/tiny_audio_corpus/`
/// (the per-clip files; the `queries/` subdirectory holds held-out query
/// clips and is skipped by the `is_file()` filter since it is a
/// subdirectory) into a `(clip_id, audio)` Parquet table at
/// `dir/audio_corpus.parquet`, the Binary-column shape
/// `generate_audio_embeddings` expects (mirrors the CPU `it` suite's
/// `write_audio_corpus_and_golden` corpus half).
fn write_audio_corpus(dir: &Path) -> std::path::PathBuf {
    let corpus_dir = harness::cookbook_fixture("tiny_audio_corpus");
    let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "tiny_audio_corpus has no .wav files");

    let ids: Vec<String> = entries
        .iter()
        .map(|p| p.file_stem().unwrap().to_str().unwrap().to_string())
        .collect();
    let clips: Vec<Vec<u8>> = entries.iter().map(|p| std::fs::read(p).unwrap()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("clip_id", DataType::Utf8, false),
        Field::new("audio", DataType::Binary, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(BinaryArray::from(
                clips.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();

    let path = dir.join("audio_corpus.parquet");
    let mut w = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

/// Register the packed audio corpus as a source named `"clips"` on
/// `session`.
async fn add_audio_corpus(session: &Arc<InferenceSession>, parquet_path: &Path) {
    session
        .add_source(
            "clips",
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
async fn htsat_audio_generate_embeddings_cpu_gpu_parity() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let model = harness::local_model_id("htsat_clap_tiny");

    let cpu_dir = TempDir::new().unwrap();
    let cpu_corpus = write_audio_corpus(cpu_dir.path());
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    add_audio_corpus(&cpu, &cpu_corpus).await;
    let (cpu_table, _) = cpu
        .generate_audio_embeddings("clips", &model, "audio", "clip_id", CachePolicy::Bypass)
        .await
        .unwrap();
    let cpu_vecs = harness::keyed_result_vectors(&cpu, &cpu_table).await;

    let gpu_dir = TempDir::new().unwrap();
    let gpu_corpus = write_audio_corpus(gpu_dir.path());
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    add_audio_corpus(&gpu, &gpu_corpus).await;
    let (gpu_table, _) = gpu
        .generate_audio_embeddings("clips", &model, "audio", "clip_id", CachePolicy::Bypass)
        .await
        .unwrap();
    let gpu_vecs = harness::keyed_result_vectors(&gpu, &gpu_table).await;

    assert_eq!(
        cpu_table.dimensions, gpu_table.dimensions,
        "CPU and GPU embedding tables must share a dimension (CLAP projection_dim)"
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
            &format!("htsat_audio_generate_embeddings[{id}]"),
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
        "HTSAT-CLAP generate_embeddings parity over tiny_audio_corpus"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn htsat_audio_encode_query_cpu_gpu_parity() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let model = harness::local_model_id("htsat_clap_tiny");
    let clip_bytes =
        std::fs::read(harness::cookbook_fixture("tiny_audio_corpus").join("clip_sine_0.wav"))
            .unwrap();

    let cpu_dir = TempDir::new().unwrap();
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    let cpu_vec = cpu.encode_audio_query(&model, &clip_bytes).await.unwrap();

    let gpu_dir = TempDir::new().unwrap();
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    let gpu_vec = gpu.encode_audio_query(&model, &clip_bytes).await.unwrap();

    assert!(
        cpu_vec.iter().any(|&v| v != 0.0),
        "query vector must not be all-zero"
    );
    harness::assert_parity("htsat_audio_encode_query", &cpu_vec, &gpu_vec);
}
