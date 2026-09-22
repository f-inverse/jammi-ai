//! Throwaway serving probe: rows/s of `generate_text_embeddings` over a
//! variable-length corpus parquet on a CUDA device, plus the served vectors
//! (for a cosine check against a reference) and the loaded model's kernel
//! admission ledger.
//!
//! args: <model_dir> <corpus.parquet> <cuda ordinal> <warmup> <iters> <out_dir>

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array};
use arrow::util::display::array_value_to_string;
use jammi_ai::model::ModelSource;
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, JammiConfig};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = &args[1];
    let corpus = &args[2];
    let cuda: i32 = args[3].parse()?;
    let warmup: usize = args[4].parse()?;
    let iters: usize = args[5].parse()?;
    let out_dir = std::path::PathBuf::from(&args[6]);
    std::fs::create_dir_all(&out_dir)?;

    let artifact_dir = tempfile::tempdir()?;
    let config = JammiConfig {
        artifact_dir: artifact_dir.path().to_path_buf(),
        gpu: GpuConfig {
            device: cuda,
            require_gpu: true,
            compute_precision: jammi_numerics::ComputePrecision::BF16,
            ..Default::default()
        },
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await?);
    session.install_query_functions();
    session
        .add_source(
            "corpus",
            jammi_db::source::SourceType::File,
            jammi_db::source::SourceConnection {
                url: Some(format!("file://{corpus}")),
                format: Some(jammi_db::source::FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;
    let model_id = format!("local:{model_dir}");

    let serve = || async {
        let t0 = std::time::Instant::now();
        let (record, _) = session
            .generate_text_embeddings(
                "corpus",
                &model_id,
                &["text".to_string()],
                "_row_id",
                jammi_db::store::CachePolicy::Bypass,
                None,
            )
            .await?;
        Ok::<_, Box<dyn std::error::Error>>((record, t0.elapsed().as_secs_f64()))
    };
    for _ in 0..warmup {
        let (record, s) = serve().await?;
        println!("warmup rows={} wall_s={s:.3}", record.row_count);
    }
    let mut samples = Vec::new();
    let mut last = None;
    for _ in 0..iters {
        let (record, s) = serve().await?;
        println!(
            "iter rows={} wall_s={s:.3} rows_per_s={:.1}",
            record.row_count,
            record.row_count as f64 / s
        );
        samples.push((record.row_count as f64, s));
        last = Some(record);
    }
    let mean_s = samples.iter().map(|(_, s)| s).sum::<f64>() / samples.len() as f64;
    let rows = samples[0].0;
    println!(
        "MEAN rows={rows} wall_s={mean_s:.3} rows_per_s={:.1}",
        rows / mean_s
    );

    // LEDGER-BEGIN
    let guard = session
        .model_cache()
        .get_or_load(
            &ModelSource::parse(&model_id),
            ModelTask::TextEmbedding,
            None,
        )
        .await?;
    let ledger = guard.model.kernel_admission();
    for (op, d) in &ledger.two_arm {
        println!("LEDGER two_arm {op} fused={} eager={}", d.fused, d.eager);
    }
    for (op, d) in &ledger.cascade {
        println!(
            "LEDGER cascade {op} fused={} eager={} declined={}",
            d.fused, d.eager, d.declined
        );
    }
    drop(guard);
    // LEDGER-END

    // Served vectors of the last serve, keyed by `_row_id`, for the cosine check.
    let record = last.expect("iters >= 1");
    let path = jammi_db::storage::StorageUrl::parse(&record.parquet_path)?;
    let ctx = datafusion::prelude::SessionContext::new();
    let df = ctx
        .read_parquet(
            path.path().to_string(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await?;
    let batches = df.collect().await?;
    let mut ids = String::new();
    let mut bytes: Vec<u8> = Vec::new();
    let mut dim = 0usize;
    for batch in &batches {
        let id_col = batch.column_by_name("_row_id").expect("_row_id");
        let vec_col = batch
            .column_by_name("vector")
            .expect("vector")
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("fixed size list");
        dim = vec_col.value_length() as usize;
        let values = vec_col
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("f32");
        for r in 0..batch.num_rows() {
            ids.push_str(&array_value_to_string(id_col, r)?);
            ids.push('\n');
            for k in 0..dim {
                bytes.extend_from_slice(&values.value(r * dim + k).to_le_bytes());
            }
        }
    }
    std::fs::write(out_dir.join("served_ids.txt"), ids)?;
    std::fs::write(out_dir.join("served_vectors.f32"), bytes)?;
    println!("WROTE dim={dim} to {}", out_dir.display());
    Ok(())
}
