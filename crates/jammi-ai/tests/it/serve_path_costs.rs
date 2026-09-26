//! Where a served embedding table's wall time and bytes go, measured on the
//! engine against itself and printed under `--nocapture`; asserted only for
//! consistency.
//!
//! * `serve_phases` — one `generate_embeddings` call decomposed into the
//!   engine's own phase spans (`job.*`, `embed.*`, `sink.*`, `table.*`,
//!   `catalog.transaction`), at fan-outs of one and four, with the residue no
//!   span claims.
//! * `table_bytes` — the written Parquet's bytes per column chunk, for an
//!   embedding table and an `infer` table, so a constant-valued column's
//!   on-disk cost is a number.
//! * `tokenize_step` — the tokenize step of a 64-row batch at a
//!   production-sized vocabulary.
//!
//! Row counts come from `JAMMI_SERVE_MEASURE_ROWS` (comma-separated), `16`
//! when unset; serves per cell from `JAMMI_SERVE_MEASURE_RUNS`, `3` when
//! unset.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use tempfile::TempDir;
use tracing::span::{Attributes, Id};
use tracing::{Instrument, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;

use crate::common;

/// Per phase: how many times it closed, and its summed open-to-close time.
/// A catalog transaction is attributed to the phase it ran under
/// (`catalog.transaction@<phase>`) as well as counted on its own; the sink's
/// own metrics event lands as `sink.<metric>`. The table is process-global,
/// so a `job.*` span is kept only under this harness's own [`MEASURE`] root
/// — a sibling test's job in the same binary is not this measurement's.
type Phases = BTreeMap<String, (u64, Duration)>;

/// The root span every measured serve runs under.
const MEASURE: &str = "serve.measure";

/// Sums every closed span's open-to-close time by name.
struct PhaseTimes(Arc<Mutex<Phases>>);

struct Opened(Instant);

impl PhaseTimes {
    fn add(&self, name: String, elapsed: Duration) {
        let mut phases = self.0.lock().unwrap();
        let entry = phases.entry(name).or_default();
        entry.0 += 1;
        entry.1 += elapsed;
    }
}

/// Folds the `*_ms` fields of the sink's metrics event.
struct MetricFields(Vec<(String, f64)>);

impl tracing::field::Visit for MetricFields {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        if let Some(name) = field.name().strip_suffix("_ms") {
            self.0.push((format!("sink.{name}"), value));
        }
    }
    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for PhaseTimes {
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Opened(Instant::now()));
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut fields = MetricFields(Vec::new());
        event.record(&mut fields);
        for (name, ms) in fields.0 {
            self.add(name, Duration::from_secs_f64(ms / 1000.0));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let Some(elapsed) = span.extensions().get::<Opened>().map(|o| o.0.elapsed()) else {
            return;
        };
        if span.name().starts_with("job.") && !span.scope().any(|s| s.name() == MEASURE) {
            return;
        }
        self.add(span.name().to_string(), elapsed);
        if span.name() == "catalog.transaction" {
            if let Some(parent) = span.parent() {
                self.add(format!("catalog.transaction@{}", parent.name()), elapsed);
            }
        }
    }
}

/// The process-wide phase table. The subscriber is global because a plan's
/// partitions run on whichever worker thread polls them; it records only
/// this engine's phase spans.
fn phases() -> Arc<Mutex<Phases>> {
    static PHASES: OnceLock<Arc<Mutex<Phases>>> = OnceLock::new();
    Arc::clone(PHASES.get_or_init(|| {
        let table = Arc::new(Mutex::new(Phases::new()));
        let names = tracing_subscriber::filter::filter_fn(|meta| {
            (meta.target().starts_with("jammi") || meta.name() == MEASURE)
                && (meta.is_span() || meta.fields().field("index_build_ms").is_some())
        });
        let subscriber =
            tracing_subscriber::registry().with(PhaseTimes(Arc::clone(&table)).with_filter(names));
        tracing::subscriber::set_global_default(subscriber)
            .expect("the phase subscriber is this binary's only global one");
        table
    }))
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').map(|n| n.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| default.to_vec())
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

/// One Parquet file of `n` rows: a zero-padded string key and a passage of
/// 3..=120 single-character words, lengths long-tailed toward the short end.
fn write_corpus(dir: &std::path::Path, n: usize) -> String {
    let mut rng = Lcg(0xfeed ^ n as u64);
    let texts: Vec<String> = (0..n)
        .map(|_| {
            let words = 3 + rng.next(118).min(rng.next(118));
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
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|i| format!("{i:08}")),
            )),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();
    std::fs::create_dir_all(dir).unwrap();
    let file = std::fs::File::create(dir.join("corpus.parquet")).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    format!("file://{}", dir.display())
}

fn model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session_over(
    dir: &std::path::Path,
    url: &str,
    partitions: usize,
) -> Arc<InferenceSession> {
    std::fs::create_dir_all(dir).unwrap();
    let mut cfg = common::test_config(dir);
    cfg.inference.partitions = partitions;
    cfg.inference.batch_size = 32;
    let session = Arc::new(InferenceSession::new(cfg).await.unwrap());
    session
        .add_source(
            "corpus",
            jammi_db::source::SourceType::File,
            jammi_db::source::SourceConnection {
                url: Some(url.to_string()),
                format: Some(jammi_db::source::FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
}

async fn serve(session: &Arc<InferenceSession>) -> ResultTableRecord {
    session
        .generate_embeddings(jammi_ai::local_session::EmbeddingRequest {
            source_id: "corpus".to_string(),
            model_id: model().to_string(),
            columns: vec!["text".to_string()],
            key_column: "id".to_string(),
            modality: jammi_wire::request::Modality::Text,
            dimensions: None,
            cache: jammi_db::store::CachePolicy::Bypass,
        })
        .await
        .unwrap()
        .0
}

/// The spans that tile a serve end to end, outermost first; every other
/// recorded span nests inside one of them.
const TILING: [&str; 4] = ["job.submit", "job.claim", "job.execute", "job.finish"];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serve_phases() {
    let table = phases();
    let runs = env_list("JAMMI_SERVE_MEASURE_RUNS", &[3])[0];
    let dir = TempDir::new().unwrap();
    for n in env_list("JAMMI_SERVE_MEASURE_ROWS", &[16]) {
        let url = write_corpus(&dir.path().join(format!("corpus-{n}")), n);
        for partitions in [1usize, 4] {
            let session = session_over(
                &dir.path().join(format!("s-{n}-{partitions}")),
                &url,
                partitions,
            )
            .await;
            // Warm: the model load and the first forward are paid outside
            // every timing.
            serve(&session).await;
            table.lock().unwrap().clear();
            let start = Instant::now();
            for _ in 0..runs {
                let served = serve(&session)
                    .instrument(tracing::info_span!(MEASURE))
                    .await;
                assert_eq!(served.row_count, n);
            }
            let wall = start.elapsed();
            let recorded = std::mem::take(&mut *table.lock().unwrap());
            let ms = |d: Duration| d.as_secs_f64() * 1000.0 / runs as f64;
            println!("== serve phases rows={n} partitions={partitions} runs={runs} ==");
            println!("{:<24} {:>8} {:>10}", "phase", "per_run", "ms_per_run");
            for (name, (count, total)) in &recorded {
                println!(
                    "{name:<24} {:>8.1} {:>10.3}",
                    *count as f64 / runs as f64,
                    ms(*total)
                );
            }
            let tiled: Duration = TILING
                .iter()
                .filter_map(|name| recorded.get(*name).map(|(_, d)| *d))
                .sum();
            println!(
                "{:<24} {:>8} {:>10.3}\n{:<24} {:>8} {:>10.3}",
                "wall",
                "",
                ms(wall),
                "outside every job span",
                "",
                ms(wall.saturating_sub(tiled))
            );
            assert!(tiled <= wall, "the tiling spans are disjoint");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn table_bytes() {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let dir = TempDir::new().unwrap();
    for n in env_list("JAMMI_SERVE_MEASURE_ROWS", &[16]) {
        let url = write_corpus(&dir.path().join(format!("corpus-{n}")), n);
        let session = session_over(&dir.path().join(format!("s-{n}")), &url, 1).await;
        let embedded = serve(&session).await;
        session
            .infer(
                "corpus",
                &jammi_datafusion::ModelSource::parse(&model()),
                jammi_datafusion::ModelTask::TextEmbedding,
                &["text".to_string()],
                "id",
                jammi_db::store::CachePolicy::Bypass,
            )
            .await
            .unwrap();
        // The one other ready table of this source: the infer's.
        let inferred = session
            .catalog()
            .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Ready)
            .await
            .unwrap()
            .into_iter()
            .find(|r| r.table_name != embedded.table_name)
            .expect("the infer table is registered");
        for (kind, record) in [("embedding", embedded), ("infer", inferred)] {
            let path = record
                .parquet_path
                .strip_prefix("file://")
                .expect("a local store");
            let file = std::fs::File::open(path).unwrap();
            let total = file.metadata().unwrap().len();
            let reader = SerializedFileReader::new(file).unwrap();
            let meta = reader.metadata();
            let mut by_column: BTreeMap<String, (i64, i64)> = BTreeMap::new();
            for group in meta.row_groups() {
                for column in group.columns() {
                    let entry = by_column.entry(column.column_path().string()).or_default();
                    entry.0 += column.compressed_size();
                    entry.1 += column.uncompressed_size();
                }
            }
            println!(
                "== table bytes kind={kind} rows={n} file={total} row_groups={} ==",
                meta.num_row_groups()
            );
            println!(
                "{:<28} {:>12} {:>12} {:>8}",
                "column", "compressed", "uncompressed", "share"
            );
            for (name, (compressed, uncompressed)) in &by_column {
                println!(
                    "{name:<28} {compressed:>12} {uncompressed:>12} {:>7.2}%",
                    *compressed as f64 * 100.0 / total as f64
                );
            }
            assert_eq!(meta.file_metadata().num_rows() as usize, n);
        }
    }
}

/// A BPE tokenizer of a production-sized vocabulary (`tokens` entries, each a
/// merge of two earlier ones), so a clone of it costs what a served model's
/// tokenizer clone costs: the fixtures' 256-entry vocabularies would not.
fn synthetic_bpe(tokens: usize) -> tokenizers::Tokenizer {
    use tokenizers::models::bpe::BPE;
    let letters: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
    let mut vocab: Vec<String> = letters.clone();
    let mut merges: Vec<(String, String)> = Vec::new();
    let mut rng = Lcg(7);
    while vocab.len() < tokens {
        let a = vocab[rng.next(vocab.len())].clone();
        let b = vocab[rng.next(vocab.len())].clone();
        let merged = format!("{a}{b}");
        if merged.len() > 12 || vocab.contains(&merged) {
            continue;
        }
        vocab.push(merged);
        merges.push((a, b));
    }
    let vocab: tokenizers::models::bpe::Vocab = vocab
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i as u32))
        .collect();
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    tokenizer
}

/// `batches` batches of `rows` texts of 8..=24 words of 1..=9 letters.
fn text_batches(batches: usize, rows: usize) -> Vec<Vec<String>> {
    let mut rng = Lcg(11);
    (0..batches)
        .map(|_| {
            (0..rows)
                .map(|_| {
                    (0..8 + rng.next(17))
                        .map(|_| {
                            (0..1 + rng.next(9))
                                .map(|_| (b'a' + rng.next(26) as u8) as char)
                                .collect::<String>()
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect()
        })
        .collect()
}

/// The tokenize step of a 64-row training batch — and of every forward's
/// preparation — at a production-sized vocabulary: a tokenizer cloned and
/// truncated per call (its word cache empty every time) against the one
/// held per truncation length, with the encoded ids required equal.
#[test]
fn tokenize_step() {
    use jammi_ai::model::tokenizer::TokenizerWrapper;
    let rows = 64;
    let batches = text_batches(50, rows);
    let max_length = 128;
    let base = synthetic_bpe(32_000);
    let wrapper = TokenizerWrapper::from_tokenizer(base.clone());

    fn refs(batch: &[String]) -> Vec<&str> {
        batch.iter().map(String::as_str).collect()
    }
    let mut cloned_ids = Vec::new();
    let start = Instant::now();
    for batch in &batches {
        let mut per_call = base.clone();
        per_call
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            }))
            .unwrap();
        let encodings = per_call.encode_batch(refs(batch), true).unwrap();
        cloned_ids.push(
            encodings
                .iter()
                .map(|e| e.get_ids().to_vec())
                .collect::<Vec<_>>(),
        );
    }
    let per_call = start.elapsed();

    let mut held_ids = Vec::new();
    let start = Instant::now();
    for batch in &batches {
        held_ids.push(
            wrapper
                .encode_batch(&refs(batch), Some(max_length))
                .unwrap()
                .input_ids,
        );
    }
    let held = start.elapsed();
    assert_eq!(cloned_ids, held_ids, "the same tokens either way");
    let ms = |d: Duration| d.as_secs_f64() * 1000.0 / batches.len() as f64;
    println!(
        "== tokenize step rows={rows} vocab=32000 max_length={max_length} batches={} ==",
        batches.len()
    );
    println!("{:<36} {:>10}", "shape", "ms_per_batch");
    println!("{:<36} {:>10.3}", "clone + truncate per call", ms(per_call));
    println!("{:<36} {:>10.3}", "held per truncation length", ms(held));
}
