//! `jammi-bench` — the scale and performance measurement harness for the Jammi
//! engine.
//!
//! A measurement *consumer* of the engine: it links `jammi-db`/`jammi-numerics`
//! and drives their public surfaces at scale, emitting one machine-readable JSON
//! report per run. It is `publish = false` and names no consumer — it measures
//! the engine's generic primitives (exact search, ANN-vs-exact recall, and later
//! embed throughput, ANN QPS, propagate latency, peak RSS), so it is kept out of
//! the published workspace to keep the engine a clean library while still being
//! compile-checked by the workspace gate.
//!
//! Invoke as `cargo run -p jammi-bench --release -- <subcommand>`. Two
//! subcommands are functional: `search-rss`, the bounded-RSS proof for streamed
//! exact search (the payoff of the streamed `exact_vector_search` rewrite), and
//! `arxiv`, the ANN-vs-exact recall curve over a committed corpus measured with
//! a HELD-OUT query set disjoint from the corpus (the exact oracle's top-k vs a
//! frozen sidecar's, set-intersected). The remaining perf metrics are scaffolded
//! as explicit `not yet measured` stubs so the report schema is stable from the
//! first emit.

mod corpus;
mod fixture;
mod rate_gate;
mod recall;
mod report;
mod rss;
mod search_rss;
mod sweep;
mod train_scale;

use clap::{Parser, Subcommand};

use std::path::PathBuf;

use report::{ArxivTier, Host, Report, Tiers};
use search_rss::Variant;
use train_scale::BackwardPath;

/// Workspace version this binary was built from, stamped into every report so a
/// downstream gate can reject a cross-version comparison.
const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "jammi-bench",
    about = "Scale and performance measurement harness for the Jammi engine.",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The bounded-RSS proof for streamed exact vector search: the streamed
    /// engine path holds a flat resident set as the corpus grows while a naive
    /// collect-all baseline (the negative control) grows linearly. Emits the
    /// JSON report and exits non-zero if the proof does not hold.
    SearchRss,
    /// The realistic quality tier over a committed corpus. Measures the
    /// ANN-vs-exact recall curve (recall@k for k∈{1,10,100}) over a HELD-OUT
    /// query set — a query parquet disjoint from the indexed corpus — so recall
    /// reflects how the frozen sidecar recovers the exact neighbours of unseen
    /// points, the exact oracle's top-k vs the sidecar's, set-intersected; the
    /// perf metrics (embed throughput, search QPS, propagate latency, peak RSS)
    /// ride along as explicit `not yet measured` markers until the perf lane
    /// lands.
    Arxiv,
    /// Internal: measure one search variant over a pre-materialized corpus in a
    /// fresh process and print `<peak_rss_mib> <result_digest>`. The `search-rss`
    /// parent spawns this per `(variant, size)` so each peak-RSS sample starts
    /// from a clean process high-water mark. Not intended for direct use.
    #[command(hide = true)]
    MeasureOnce {
        /// Which search path to exercise (`streamed` or `naive`).
        #[arg(long)]
        variant: String,
        /// The corpus size, for the child's own diagnostics and result check.
        #[arg(long)]
        rows: usize,
        /// Path to the pre-materialized corpus Parquet the parent wrote.
        #[arg(long)]
        corpus_path: PathBuf,
    },
    /// The recall-vs-cost sweep: how ANN recall and its build/query cost move as
    /// the HNSW knobs are swept, each point measured against the exact oracle
    /// over a held-out query set. Sweeps the build knobs (connectivity,
    /// build_expansion → build time + index size) and the query knob
    /// (search_expansion → recall vs QPS over one re-dialed graph), emitting the
    /// `recall_sweep` tier. An on-box emitter (it builds a graph per build-knob
    /// point): run with `RAYON_NUM_THREADS=1` and read the JSON; the committed
    /// curve is its output, not a CI step.
    RecallSweep {
        /// Corpus vectors parquet — built into each swept graph and scored by
        /// the exact oracle.
        #[arg(long)]
        corpus_src: PathBuf,
        /// Held-out query vectors parquet, disjoint from the corpus.
        #[arg(long)]
        query_src: PathBuf,
    },
    /// Internal: build the committed held-out recall fixture from the full scale
    /// cache. Reads the source corpus + held-out query parquets, takes the
    /// deterministic first-N / first-M sorted-`_row_id` subsets, freezes one
    /// sidecar over the corpus slice, and writes the fixture bundle + `floor.json`
    /// (floor = measured recall − margin). Run off-box once with
    /// `RAYON_NUM_THREADS=1`; the bundle is committed and CI only loads it. Not a
    /// CI step — the provenance-recording builder for the committed fixture.
    #[command(hide = true)]
    BuildScaleFixture {
        /// Source corpus vectors parquet (the full cache corpus).
        #[arg(long)]
        corpus_src: PathBuf,
        /// Source held-out query vectors parquet (disjoint from the corpus).
        #[arg(long)]
        query_src: PathBuf,
        /// Output directory for the fixture bundle (the committed `fixtures/scale/`).
        #[arg(long)]
        out_dir: PathBuf,
        /// How many corpus rows to keep (first N by sorted `_row_id`).
        #[arg(long)]
        corpus_rows: usize,
        /// How many held-out query rows to keep (first M by sorted `_row_id`).
        #[arg(long)]
        query_rows: usize,
    },
    /// The CPU-hermetic training tier: measures the engine's in-batch-negative
    /// fine-tune throughput (pairs/s) through one GradCache backward + AdamW step
    /// on `Device::Cpu`, and re-triggers the activation-memory negative control —
    /// the single-pass backward (every encoder graph alive at once) grows with
    /// the pair count while the bounded GradCache path stays flat. Emits the JSON
    /// report with the `training` tier set; the rate gate against the committed
    /// baseline runs in `cargo test`, not here.
    TrainScale,
    /// Internal: run one backward path over a synthetic CPU fine-tune at a given
    /// pair count in a fresh process and print `<peak_rss_mib>`. The `train-scale`
    /// OOM control spawns this per `(path, pairs)` so each peak-RSS sample starts
    /// from a clean process high-water mark. Not intended for direct use.
    #[command(hide = true)]
    TrainMeasureOnce {
        /// Which backward path to exercise (`gradcache` or `single-pass`).
        #[arg(long)]
        path: String,
        /// In-batch-negative pair count for this measurement.
        #[arg(long)]
        pairs: usize,
    },
    /// Internal: measure GradCache training throughput at a given pair count in a
    /// fresh process and print `<pairs_per_s> <wall_ms>`. The `cargo test` rate
    /// gate spawns this at a reduced pair count to drive the same throughput code
    /// path the committed baseline is set from. Not intended for direct use.
    #[command(hide = true)]
    TrainThroughputOnce {
        /// In-batch-negative pair count to time one GradCache backward + step over.
        #[arg(long)]
        pairs: usize,
    },
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::SearchRss => run_search_rss().await,
        Command::Arxiv => run_arxiv().await,
        Command::RecallSweep {
            corpus_src,
            query_src,
        } => run_recall_sweep(&corpus_src, &query_src).await,
        Command::MeasureOnce {
            variant,
            rows,
            corpus_path,
        } => run_measure_once(&variant, rows, &corpus_path).await,
        Command::BuildScaleFixture {
            corpus_src,
            query_src,
            out_dir,
            corpus_rows,
            query_rows,
        } => {
            run_build_scale_fixture(&corpus_src, &query_src, &out_dir, corpus_rows, query_rows)
                .await
        }
        Command::TrainScale => run_train_scale().await,
        Command::TrainMeasureOnce { path, pairs } => run_train_measure_once(&path, pairs),
        Command::TrainThroughputOnce { pairs } => run_train_throughput_once(pairs),
    }
}

/// The `train-throughput-once` child: time one GradCache backward + step over
/// `pairs` synthetic pairs in this fresh process and print its rate. A failure
/// exits non-zero so the parent gate surfaces it.
fn run_train_throughput_once(pairs: usize) -> std::process::ExitCode {
    match train_scale::run_throughput_at(pairs) {
        Ok(t) => {
            train_scale::emit_throughput_result(t.pairs_per_s, t.wall_ms);
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("train-throughput-once (@ {pairs} pairs) failed: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// The `train-measure-once` child: run one backward path over a synthetic CPU
/// fine-tune at `pairs` pairs, print its peak RSS, and exit. The parent reads the
/// single stdout line; a failure exits non-zero so the parent surfaces it.
fn run_train_measure_once(path: &str, pairs: usize) -> std::process::ExitCode {
    let path = match BackwardPath::parse(path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("train-measure-once: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match train_scale::measure_once(path, pairs) {
        Ok(rss_mib) => {
            train_scale::emit_child_result(rss_mib);
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!(
                "train-measure-once ({} @ {pairs} pairs) failed: {e}",
                path.as_str()
            );
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the CPU-hermetic training tier: measure GradCache throughput, run the
/// activation-memory negative control, emit the report with the `training` tier
/// set, and map the OOM verdict to the process exit code. A failed control
/// prints the full numbers and exits non-zero — the run never fakes a pass.
async fn run_train_scale() -> std::process::ExitCode {
    let baseline = match train_scale::Baseline::load() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("train-scale could not load the committed throughput baseline: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let throughput = match train_scale::run_throughput() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("train-scale throughput measurement failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let oom = match train_scale::run_oom_control().await {
        Ok(o) => o,
        Err(e) => {
            eprintln!("train-scale OOM control failed to run: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = train_scale::build_tier(throughput, baseline, oom);
    let oom_passed = tier.oom.assertion.passed;
    let rate_passed = tier.rate_gate.as_ref().is_none_or(|v| v.passed);
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "train-scale",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: Some(tier),
        },
    };
    emit(&report);
    if !oom_passed {
        eprintln!(
            "training OOM control FAILED — the single-pass backward is not growing past the \
             floor while the activation-graph separation dominates; see \
             tiers.training.oom.assertion for the numbers"
        );
    }
    if !rate_passed {
        eprintln!(
            "training throughput REGRESSED below the committed baseline floor; see \
             tiers.training.rate_gate for the numbers"
        );
    }
    if oom_passed && rate_passed {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}

/// The `measure-once` child: run one variant over the pre-materialized corpus,
/// print its peak RSS and result digest, and exit. The parent reads the single
/// stdout line; a failure exits non-zero so the parent surfaces it.
async fn run_measure_once(
    variant: &str,
    rows: usize,
    corpus_path: &std::path::Path,
) -> std::process::ExitCode {
    let variant = match Variant::parse(variant) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("measure-once: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match search_rss::measure_once(variant, rows, corpus_path).await {
        Ok((rss_mib, digest)) => {
            search_rss::emit_child_result(rss_mib, &digest);
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!(
                "measure-once ({} @ {rows} rows) failed: {e}",
                variant.as_str()
            );
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the bounded-RSS proof, emit its JSON, and map the assertion verdict to
/// the process exit code. A failed proof prints the full numbers and exits
/// non-zero — the run never fakes a pass.
async fn run_search_rss() -> std::process::ExitCode {
    let tier = match search_rss::run().await {
        Ok(tier) => tier,
        Err(e) => {
            eprintln!("search-rss proof failed to run: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = tier.assertion.passed;
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "search-rss",
        tiers: Tiers {
            arxiv: None,
            binding: Some(tier),
            recall_sweep: None,
            training: None,
        },
    };
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "bounded-RSS assertion FAILED — streamed path is not flat while naive grows; \
             see the report's tiers.binding.assertion for the numbers"
        );
        std::process::ExitCode::FAILURE
    }
}

/// Run the realistic-tier recall path and emit the tier.
///
/// Measures the ANN-vs-exact recall curve (recall@k for k∈{1,10,100}) over the
/// committed *held-out* recall fixture bundle — a corpus, a sidecar frozen over
/// it, and a SEPARATE disjoint query set — filling the `recall` slots with real
/// datapoints; the perf metrics (embed/search QPS, propagate latency, peak RSS)
/// stay explicit `not yet measured` markers — they are the perf lane, measured
/// in a later PR. The fixture bundle is committed under `fixtures/scale/`; until
/// it is present this subcommand fails loudly rather than emitting a faked
/// recall number.
async fn run_arxiv() -> std::process::ExitCode {
    let fixture_dir = arxiv_fixture_dir();
    let recall = match recall::recall_curve_held_out(&fixture_dir).await {
        Ok(curve) => curve,
        Err(e) => {
            eprintln!(
                "arxiv recall path failed over fixture {}: {e}",
                fixture_dir.display()
            );
            return std::process::ExitCode::FAILURE;
        }
    };
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "arxiv",
        tiers: Tiers {
            arxiv: Some(ArxivTier::with_recall(recall)),
            binding: None,
            recall_sweep: None,
            training: None,
        },
    };
    emit(&report);
    std::process::ExitCode::SUCCESS
}

/// Run the recall-vs-cost sweep over a corpus + held-out query parquet and emit
/// the `recall_sweep` tier.
///
/// The on-box emitter for the recall-vs-cost curve: it builds one graph per
/// build-knob point (so it is not a CI step — run it off-box with
/// `RAYON_NUM_THREADS=1`) and re-dials `search_expansion` over one frozen graph
/// for the query-cost axis. The committed curve is this command's JSON output.
async fn run_recall_sweep(
    corpus_src: &std::path::Path,
    query_src: &std::path::Path,
) -> std::process::ExitCode {
    // Resolve to absolute paths: the corpus is registered as a `file://` object
    // store URL, which a relative path cannot form. Canonicalize surfaces a
    // missing input as a clear error here rather than an opaque store-not-found
    // one deeper in DataFusion.
    let (corpus_src, query_src) = match (
        std::fs::canonicalize(corpus_src),
        std::fs::canonicalize(query_src),
    ) {
        (Ok(c), Ok(q)) => (c, q),
        (Err(e), _) => {
            eprintln!("recall-sweep: corpus {}: {e}", corpus_src.display());
            return std::process::ExitCode::FAILURE;
        }
        (_, Err(e)) => {
            eprintln!("recall-sweep: query {}: {e}", query_src.display());
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match sweep::run(&corpus_src, &query_src).await {
        Ok(tier) => tier,
        Err(e) => {
            eprintln!("recall-sweep failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "recall-sweep",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: Some(tier),
            training: None,
        },
    };
    emit(&report);
    std::process::ExitCode::SUCCESS
}

/// Build the committed held-out recall fixture from the full scale cache and
/// print the resulting floor record.
///
/// Off-box one-shot: subsets the source parquets, freezes the one sidecar, and
/// writes the fixture bundle plus `floor.json`. Prints the measured recall and
/// the derived floors so the operator sees the numbers being committed.
async fn run_build_scale_fixture(
    corpus_src: &std::path::Path,
    query_src: &std::path::Path,
    out_dir: &std::path::Path,
    corpus_rows: usize,
    query_rows: usize,
) -> std::process::ExitCode {
    match fixture::build_held_out_fixture(corpus_src, query_src, out_dir, corpus_rows, query_rows)
        .await
    {
        Ok(record) => match serde_json::to_string_pretty(&record) {
            Ok(json) => {
                println!("{json}");
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("failed to serialize floor record: {e}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("build-scale-fixture failed: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// The committed recall-fixture bundle directory, resolved against the crate
/// root so the path is stable regardless of the working directory the harness
/// is launched from.
fn arxiv_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("scale")
}

/// Write the report as pretty JSON to stdout.
fn emit(report: &Report) {
    match serde_json::to_string_pretty(report) {
        Ok(json) => println!("{json}"),
        Err(e) => eprintln!("failed to serialize report: {e}"),
    }
}
