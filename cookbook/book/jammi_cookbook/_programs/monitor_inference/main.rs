//! An inference observer watching an embedding run — the "Monitor Inference"
//! guide page, run: an `InferenceSession` opened with an `InferenceObserver`,
//! which the engine calls once per output batch of every model run.
//!
//! Usage: `monitor_inference <corpus-url> <model>`. The artifact directory and
//! every other setting come from `jammi.toml` or the `JAMMI_*` environment.
//! Prints one JSON line per observed batch: `rows`, `errors`, `model`,
//! `latency_ms`.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use arrow::array::{Array, RecordBatch, StringArray};
use jammi_ai::inference::InferenceObserver;
use jammi_ai::session::InferenceSession;
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

/// One observed batch: its rows, how many of them failed, the model that
/// produced it, and how long the batch took.
struct Observed {
    rows: usize,
    errors: usize,
    model: String,
    latency: Duration,
}

/// Records every batch it is shown.
#[derive(Default)]
struct Recorder {
    batches: Mutex<Vec<Observed>>,
}

impl InferenceObserver for Recorder {
    fn on_batch(&self, batch: &RecordBatch, model_id: &str, latency: Duration) {
        let errors = batch
            .column_by_name("_status")
            .and_then(|status| status.as_any().downcast_ref::<StringArray>())
            .map_or(0, |status| {
                status.iter().filter(|s| *s == Some("error")).count()
            });
        self.batches
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Observed {
                rows: batch.num_rows(),
                errors,
                model: model_id.to_string(),
                latency,
            });
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(corpus), Some(model)) = (args.next(), args.next()) else {
        return Err("usage: monitor_inference <corpus-url> <model>".into());
    };

    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn InferenceObserver> = recorder.clone();
    let session = InferenceSession::with_observer(JammiConfig::load(None)?, Some(observer)).await?;

    let connection = SourceConnection::parse(&corpus, FileFormat::Parquet)?;
    session
        .add_source("corpus", SourceType::File, connection)
        .await?;
    session
        .generate_text_embeddings(
            "corpus",
            &model,
            &["content".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await?;

    let batches = recorder
        .batches
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    for seen in batches.iter() {
        println!(
            r#"{{"rows": {}, "errors": {}, "model": "{}", "latency_ms": {:.3}}}"#,
            seen.rows,
            seen.errors,
            seen.model,
            seen.latency.as_secs_f64() * 1000.0
        );
    }
    Ok(())
}
