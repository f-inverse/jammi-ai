//! The forward-chunk composition measurement over a variable-length corpus:
//! padded versus real tokens, the distinct forward shapes, the plan's per-row
//! and fixed cost against a bare forward over the plan's own chunks, and the
//! peak resident set — printed under `--nocapture`, asserted only for
//! consistency. The row counts come from `JAMMI_CHUNK_MEASURE_ROWS`
//! (comma-separated), `16,1024` when unset.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::{collect, ExecutionPlan};
use tempfile::TempDir;

use jammi_ai::model::tokenizer::TokenizerWrapper;
use jammi_ai::model::LoadedModel;
use jammi_ai::session::InferenceSession;
use jammi_datafusion::inference::chunk::ChunkAssembler;
use jammi_datafusion::inference::runner::test_hooks;
use jammi_datafusion::ComputeDeviceKind;
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;
use jammi_datafusion::{plan_inference, InferenceSpec};
use jammi_datafusion::{NumberedInputExec, RowOrder};
use jammi_numerics::ChunkBudget;

use crate::common;

const BATCH_SIZE: usize = 32;
const BATCH_TOKENS: usize = 16384;
const MAX_SEQ: usize = 128;

fn in_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("text", DataType::Utf8, true),
        Field::new("_content_hash", DataType::Utf8, true),
    ]))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

/// `n` rows of 3..=120 single-character words (every word one token of the
/// tiny_bert vocabulary), lengths drawn by a fixed generator, keyed by a
/// zero-padded string in corpus order so key order is corpus order.
fn corpus(n: usize) -> RecordBatch {
    let mut rng = Lcg(0xc0ffee ^ n as u64);
    let texts: Vec<String> = (0..n)
        .map(|_| {
            let words = 3 + rng.next(118);
            (0..words)
                .map(|_| {
                    char::from_digit(rng.next(36) as u32, 36)
                        .unwrap()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    RecordBatch::try_new(
        in_schema(),
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|i| format!("{i:06}")),
            )),
            Arc::new(StringArray::from(texts)),
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|i| format!("hash-{i:06}")),
            )),
        ],
    )
    .unwrap()
}

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

fn spec(source_id: &str, partitions: usize) -> InferenceSpec {
    InferenceSpec {
        source: ModelSource::parse(&tiny_bert_model()),
        task: ModelTask::TextEmbedding,
        content_columns: vec!["text".to_string()],
        key_column: "id".to_string(),
        source_id: source_id.to_string(),
        chunk: ChunkBudget {
            rows: NonZeroUsize::new(BATCH_SIZE).unwrap(),
            tokens: NonZeroUsize::new(BATCH_TOKENS).unwrap(),
        },
        embedding_dim: Some(32),
        regression_form: None,
        passthrough: Vec::new(),
        device_kind: ComputeDeviceKind::Cpu,
        partitions: NonZeroUsize::new(partitions).unwrap(),
    }
}

/// Peak resident set in MiB from `/proc/self/status` `VmHWM`, where present.
fn peak_rss_mib() -> Option<f64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
}

/// The real token count of every row, tokenised alone (no batch padding).
fn real_tokens(texts: &StringArray) -> Vec<usize> {
    let tokenizer =
        TokenizerWrapper::from_file(&common::cookbook_fixture("tiny_bert").join("tokenizer.json"))
            .unwrap();
    texts
        .iter()
        .map(|t| {
            tokenizer
                .encode_batch(&[t.unwrap()], Some(MAX_SEQ))
                .unwrap()
                .seq_len
        })
        .collect()
}

struct Run {
    elapsed: Duration,
    forwards: usize,
}

/// The plan's own chunks: the numbered input alone, its rows regrouped by
/// `_chunk`.
async fn chunks_of(
    session: &InferenceSession,
    source_id: &str,
    batch: &RecordBatch,
) -> Vec<RecordBatch> {
    let source =
        MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], in_schema(), None).unwrap();
    let numbered = NumberedInputExec::try_new(
        source,
        RowOrder::Keyed {
            key_column: "id".into(),
            tie_breakers: vec![jammi_db::store::schema::CONTENT_HASH_COLUMN.to_string()],
        },
        spec(source_id, 1),
        session.inference_runtime(),
    )
    .unwrap();
    let mut assembler = ChunkAssembler::try_new(numbered.schema()).unwrap();
    let mut chunks: Vec<RecordBatch> = Vec::new();
    for b in collect(Arc::new(numbered), session.context().task_ctx())
        .await
        .unwrap()
    {
        chunks.extend(assembler.push(&b).unwrap());
    }
    chunks.extend(assembler.finish().unwrap());
    chunks
}

/// The padded tokens the model runs over `chunks` (each chunk's rows times
/// its longest row's ladder width) and the count of distinct widths.
fn padded_over(model: &LoadedModel, chunks: &[RecordBatch]) -> (usize, usize) {
    let ladder = model.shape_ladder(ModelTask::TextEmbedding).unwrap();
    let widths: Vec<usize> = chunks
        .iter()
        .map(|c| {
            let text: ArrayRef = Arc::clone(c.column(1));
            let longest = model
                .row_costs(&[text], ModelTask::TextEmbedding)
                .unwrap()
                .into_iter()
                .max()
                .unwrap_or(0);
            ladder.width(longest as usize)
        })
        .collect();
    let padded = chunks
        .iter()
        .zip(&widths)
        .map(|(c, w)| c.num_rows() * w)
        .sum();
    (padded, widths.iter().collect::<BTreeSet<_>>().len())
}

/// `source_id` keys the forward counter, so each measurement reads its own
/// forwards while the other runs beside it in the same binary.
async fn run_plan(
    session: &InferenceSession,
    source_id: &str,
    batch: &RecordBatch,
    partitions: usize,
) -> Run {
    let source =
        MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], in_schema(), None).unwrap();
    test_hooks::reset_forward_calls_for(source_id);
    let start = Instant::now();
    let plan: Arc<dyn ExecutionPlan> = plan_inference(
        source,
        RowOrder::Keyed {
            key_column: "id".into(),
            tie_breakers: vec![jammi_db::store::schema::CONTENT_HASH_COLUMN.to_string()],
        },
        spec(source_id, partitions),
        session.inference_runtime(),
    )
    .unwrap();
    let out = collect(plan, session.context().task_ctx()).await.unwrap();
    let elapsed = start.elapsed();
    let rows: usize = out.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, batch.num_rows());
    Run {
        elapsed,
        forwards: test_hooks::forward_calls_for(source_id) as usize,
    }
}

/// `chunks` forwarded bare — prepare then forward, no plan above: the
/// floor the plan's overhead is measured against.
async fn run_direct(model: &LoadedModel, chunks: &[RecordBatch]) -> Run {
    let start = Instant::now();
    let mut rows = 0;
    for chunk in chunks {
        let text: ArrayRef = Arc::clone(chunk.column(1));
        let prepared = model.prepare(&[text], ModelTask::TextEmbedding).unwrap();
        rows += model
            .forward_prepared(prepared)
            .await
            .unwrap()
            .row_status
            .len();
    }
    let elapsed = start.elapsed();
    assert_eq!(rows, chunks.iter().map(|c| c.num_rows()).sum::<usize>());
    Run {
        elapsed,
        forwards: chunks.len(),
    }
}

async fn measure(label: &str, sizes: &[usize]) {
    let dir = TempDir::new().unwrap();
    let cfg = common::test_config(dir.path());
    let session = InferenceSession::new(cfg).await.unwrap();
    let guard = session
        .model_cache()
        .get_or_load(
            &ModelSource::parse(&tiny_bert_model()),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();
    let model = &guard.model;
    // Warm: the first forward is paid once, outside every timing.
    let source_id = format!("chunk-measure-{label}");
    run_direct(model, &chunks_of(&session, &source_id, &corpus(16)).await).await;

    println!(
        "== chunk composition [{label}] batch_size={BATCH_SIZE} batch_tokens={BATCH_TOKENS} =="
    );
    println!(
        "{:>6} {:>10} {:>10} {:>8} {:>6} {:>7} {:>10} {:>10} {:>10} {:>8}",
        "rows",
        "real_tok",
        "padded",
        "ratio",
        "fwds",
        "widths",
        "direct_ms",
        "plan1_ms",
        "plan4_ms",
        "rss_mib"
    );
    let mut points: Vec<(usize, f64, f64, f64)> = Vec::new();
    for &n in sizes {
        let batch = corpus(n);
        let real: usize = real_tokens(batch.column(1).as_any().downcast_ref().unwrap())
            .iter()
            .sum();
        let chunks = chunks_of(&session, &source_id, &batch).await;
        let (padded, widths) = padded_over(model, &chunks);
        let direct = run_direct(model, &chunks).await;
        let plan1 = run_plan(&session, &source_id, &batch, 1).await;
        let plan4 = run_plan(&session, &source_id, &batch, 4).await;
        assert_eq!(
            plan1.forwards, direct.forwards,
            "the plan forwards the same chunks"
        );
        assert_eq!(
            plan4.forwards, direct.forwards,
            "the fan-out forwards the same chunks"
        );
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        println!(
            "{:>6} {:>10} {:>10} {:>8.2} {:>6} {:>7} {:>10.1} {:>10.1} {:>10.1} {:>8}",
            n,
            real,
            padded,
            padded as f64 / real as f64,
            direct.forwards,
            widths,
            ms(direct.elapsed),
            ms(plan1.elapsed),
            ms(plan4.elapsed),
            peak_rss_mib().map_or("n/a".to_string(), |m| format!("{m:.0}")),
        );
        points.push((n, ms(direct.elapsed), ms(plan1.elapsed), ms(plan4.elapsed)));
    }
    if let [(n0, d0, p0, q0), .., (n1, d1, p1, q1)] = points[..] {
        let fit = |a: f64, b: f64| {
            let per_row = (b - a) / (n1 - n0) as f64;
            (per_row, a - per_row * n0 as f64)
        };
        let (dr, df) = fit(d0, d1);
        let (pr, pf) = fit(p0, p1);
        let (qr, qf) = fit(q0, q1);
        println!(
            "fit over rows {n0}..{n1}: direct per_row={dr:.4} ms fixed={df:.2} ms | plan1 per_row={pr:.4} ms ({:.2}x) fixed={pf:.2} ms | plan4 per_row={qr:.4} ms ({:.2}x) fixed={qf:.2} ms",
            pr / dr,
            qr / dr
        );
    }
}

fn sizes() -> Vec<usize> {
    std::env::var("JAMMI_CHUNK_MEASURE_ROWS")
        .ok()
        .map(|v| v.split(',').map(|n| n.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![16, 1024])
}

#[tokio::test(flavor = "current_thread")]
async fn chunk_composition_single_threaded() {
    measure("current_thread", &sizes()).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn chunk_composition_default_threaded() {
    measure("multi_thread", &sizes()).await;
}
