//! CPU↔GPU parity for `infer` (`Classification`) over a **ModernBERT** classifier.
//!
//! The embedding parity lane runs a plain-BERT encoder; this lane is the only
//! one exercising the ModernBERT backbone (rotary attention, GeGLU) on a real
//! device. That path silently failed on GPU — the score-matmul relied on RoPE
//! leaving Q/K contiguous, which holds on CPU but not CUDA, so every row errored
//! and `infer` returned empty scores (annotate semantics swallow the per-row
//! error). This test is the regression guard: it asserts the GPU actually
//! produced a score distribution for every row AND that it matches the CPU
//! distribution up to reduction noise.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, StringArray};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

/// The committed tiny ModernBERT classifier fixture (`local:` — no network).
const CLASSIFIER: &str = "tiny_modernbert_classifier";

/// Register the patents fixture as a source on `session`.
async fn add_patents(session: &Arc<InferenceSession>) {
    session
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(harness::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

/// Serve `infer` (Classification) over the patents `abstract` column and read
/// back each row's full score distribution, keyed by `_row_id` so a CPU and GPU
/// run compare row-exact regardless of scan order. Each value is the label→score
/// map parsed from `all_scores_json`.
async fn keyed_scores(session: &Arc<InferenceSession>) -> HashMap<String, HashMap<String, f64>> {
    let source = ModelSource::parse(&harness::local_model_id(CLASSIFIER));
    let (batches, _) = session
        .infer(
            "patents",
            &source,
            ModelTask::Classification,
            &["abstract".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let mut out = HashMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("_row_id")
            .expect("infer output has _row_id")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_row_id is Utf8");
        let scores = batch
            .column_by_name("all_scores_json")
            .expect("infer output has all_scores_json")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("all_scores_json is Utf8");
        for i in 0..batch.num_rows() {
            if ids.is_null(i) || scores.is_null(i) {
                continue;
            }
            let map: HashMap<String, f64> =
                serde_json::from_str(scores.value(i)).expect("all_scores_json parses");
            out.insert(ids.value(i).to_string(), map);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn classification_parity_cpu_vs_gpu_over_modernbert() {
    skip_without_gpu!();

    let cpu_dir = TempDir::new().unwrap();
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    add_patents(&cpu).await;
    let cpu_scores = keyed_scores(&cpu).await;

    let gpu_dir = TempDir::new().unwrap();
    let gpu = harness::gpu_session(gpu_dir.path()).await;
    add_patents(&gpu).await;
    let gpu_scores = keyed_scores(&gpu).await;

    // The decisive assertion the RoPE-contiguity fix restores: the GPU produced a
    // score distribution for *every* row the CPU did. Before the fix the GPU
    // ModernBERT attention matmul errored on non-contiguous Q/K, so `infer`
    // returned zero scored rows here.
    assert!(
        !cpu_scores.is_empty(),
        "CPU produced no scores — fixture broken"
    );
    assert_eq!(
        gpu_scores.len(),
        cpu_scores.len(),
        "GPU scored {} rows but CPU scored {} — the ModernBERT GPU forward dropped rows",
        gpu_scores.len(),
        cpu_scores.len(),
    );

    // And every row's distribution matches the CPU's up to reduction noise. The
    // scores are a softmax distribution; compare each label's probability with a
    // generous absolute tolerance (looser than embeddings' cosine floor because a
    // softmax compresses the low bits, tighter than any real kernel bug).
    for (key, cpu_map) in &cpu_scores {
        let gpu_map = gpu_scores
            .get(key)
            .unwrap_or_else(|| panic!("GPU has no scores for row {key}"));
        assert_eq!(
            cpu_map.len(),
            gpu_map.len(),
            "row {key}: CPU has {} labels, GPU {}",
            cpu_map.len(),
            gpu_map.len(),
        );
        for (label, &cpu_p) in cpu_map {
            let gpu_p = *gpu_map
                .get(label)
                .unwrap_or_else(|| panic!("row {key}: GPU missing label {label}"));
            assert!(
                (cpu_p - gpu_p).abs() < 1e-2,
                "row {key} label {label}: CPU {cpu_p} vs GPU {gpu_p} diverge beyond tolerance",
            );
        }
    }
}
