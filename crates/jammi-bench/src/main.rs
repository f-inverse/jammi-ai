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

mod conformal;
mod context_predictor;
mod corpus;
mod eval;
mod fixture;
mod graph_train;
mod model_inference;
mod propagate;
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
    /// The CPU-hermetic conformal-coverage tier: re-folds the engine's split
    /// conformal calibration (LAC classification, absolute-residual and CQR
    /// regression) over a committed spec, measuring the marginal coverage as a
    /// PORTABLE FRACTION at each calibration-set size and gating it against a
    /// committed floor (`coverage_floor = measured − margin`, the recall-floor
    /// idiom). Emits the JSON report with the `conformal` tier set and exits
    /// non-zero if any coverage falls below its floor.
    ConformalScale,
    /// The CPU-hermetic eval-metric tier: re-folds the engine's retrieval
    /// (recall/MRR/nDCG) and classification (accuracy/F1) metric kernels and the
    /// order-invariant `eval_compare` bootstrap CI over a committed golden,
    /// gating each metric against its committed value within a tolerance and
    /// asserting the bootstrap's order-invariance (engine #173). Emits the JSON
    /// report with the `eval` tier set and exits non-zero on any drift.
    EvalScale,
    /// Internal: rebuild the committed conformal spec (`baselines/conformal.json`)
    /// from a fresh measurement — measures each family's coverage at each
    /// calibration size and writes `floor = measured − margin`. Run off-box once
    /// when the spec is established or the engine's conformal contract changes;
    /// CI only loads and re-folds it. Not a CI step — the provenance-recording
    /// rebuilder for the committed spec.
    #[command(hide = true)]
    RebuildConformalSpec,
    /// Internal: rebuild the committed eval spec (`baselines/eval.json`) from a
    /// fresh fold — folds each metric at each eval-set size and records it as the
    /// golden. Run off-box once when the spec is established or a metric kernel
    /// changes; CI only loads and re-folds it. Not a CI step — the
    /// provenance-recording rebuilder for the committed golden.
    #[command(hide = true)]
    RebuildEvalSpec,
    /// The CPU-hermetic propagation tier: re-folds the engine's
    /// `propagate_embeddings` (APPNP/SGC decoupled-GNN forward pass) over a
    /// committed synthetic graph+embedding fixture and gates the DETERMINISM
    /// contract — a committed digest of the propagated output vectors that any
    /// box re-derives — while measuring propagation wall-time at named graph
    /// sizes as an un-gated, machine-dependent reference. Emits the JSON report
    /// with the `propagate` tier set and exits non-zero if the digest drifts.
    PropagateScale,
    /// Internal: rebuild the committed propagation spec
    /// (`baselines/propagate.json`) from a fresh fold — folds the gated fixture
    /// through the engine and records its output digest. Run off-box once when
    /// the spec is established or the engine's propagation contract changes; CI
    /// only loads and re-folds it. Not a CI step — the provenance-recording
    /// rebuilder for the committed digest.
    #[command(hide = true)]
    RebuildPropagateSpec,
    /// The CPU-hermetic graph fine-tune tier: re-samples the engine's biased-walk
    /// graph sampler (`GraphSampler` — the data path `fine_tune_graph` threads
    /// through) over a committed synthetic graph, gates the sampled-pair set on a
    /// committed determinism digest any box re-derives, and gates the
    /// sampled-pairs-per-second throughput against a committed same-box baseline.
    /// Emits the JSON report with the `graph_train` tier set and exits non-zero if
    /// the digest drifts or the throughput regresses.
    GraphTrainScale,
    /// Internal: rebuild the committed graph fine-tune spec
    /// (`baselines/graph_train.json`) from a fresh sample — regenerates the graph,
    /// samples it through the engine, and records the sampled-pair digest and the
    /// same-box throughput. Run off-box once when the spec is established or the
    /// sampler contract changes; CI only loads and re-samples it. Not a CI step —
    /// the provenance-recording rebuilder for the committed digest + baseline.
    #[command(hide = true)]
    RebuildGraphTrainSpec,
    /// The CPU-hermetic context-predictor tier: measures the engine's
    /// `train_context_predictor` meta-training throughput (gated against a
    /// committed same-box baseline) and gates `predict_with_context_predictor` on
    /// a committed digest of the predicted distributions over a committed trained
    /// weight bundle (predict is byte-deterministic given the weights + targets),
    /// with predict wall-time as an un-gated reference. Emits the JSON report with
    /// the `context_predictor` tier set and exits non-zero if the digest drifts or
    /// the throughput regresses.
    ContextPredictorScale,
    /// Internal: rebuild the committed context-predictor spec
    /// (`baselines/context_predictor.json`) and its trained weight bundle
    /// (`baselines/context_predictor_weights/`) from a fresh train + predict —
    /// trains a predictor through the engine, commits the trained weights, and
    /// records the predict digest those weights produce plus the same-box training
    /// baseline. Run off-box once when the spec is established or the serve/predict
    /// contract changes; CI only loads the committed weights and re-predicts. Not a
    /// CI step — the provenance-recording rebuilder for the committed bundle.
    #[command(hide = true)]
    RebuildContextPredictorSpec,
    /// The CPU-hermetic model-inference tier: drives the engine's GPU-model
    /// serving verbs `generate_text_embeddings` (the `generate_embeddings` path)
    /// and `infer` (`Classification`) on `Device::Cpu` over tiny committed model
    /// bundles. Each verb gates a committed determinism digest of the served
    /// output (the portable cell anchor) and a coarse same-box serving rate. The
    /// rate is a code-path-regression net over the tiny model, NOT the full-scale
    /// scaling SLO — that representative number is captured off-box in the
    /// cookbook (the A/B split). Emits the JSON report with the `model_inference`
    /// tier set and exits non-zero if a digest drifts or a throughput regresses.
    ModelInferenceScale,
    /// Internal: rebuild the committed model-inference spec
    /// (`baselines/model_inference.json`) from a fresh serve — regenerates the
    /// corpus, serves both verbs over the committed tiny bundles
    /// (`baselines/embed_model/`, `baselines/classifier_model/`), and records both
    /// digests and both same-box serving baselines. Run off-box once when the spec
    /// is established or the serving contract changes; CI only loads and
    /// re-serves. Not a CI step — the provenance-recording rebuilder.
    #[command(hide = true)]
    RebuildModelInferenceSpec,
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
        Command::ConformalScale => run_conformal_scale(),
        Command::EvalScale => run_eval_scale(),
        Command::RebuildConformalSpec => run_rebuild_conformal_spec(),
        Command::RebuildEvalSpec => run_rebuild_eval_spec(),
        Command::PropagateScale => run_propagate_scale().await,
        Command::RebuildPropagateSpec => run_rebuild_propagate_spec().await,
        Command::GraphTrainScale => run_graph_train_scale(),
        Command::RebuildGraphTrainSpec => run_rebuild_graph_train_spec(),
        Command::ContextPredictorScale => run_context_predictor_scale().await,
        Command::RebuildContextPredictorSpec => run_rebuild_context_predictor_spec().await,
        Command::ModelInferenceScale => run_model_inference_scale().await,
        Command::RebuildModelInferenceSpec => run_rebuild_model_inference_spec().await,
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
            conformal: None,
            eval: None,
            propagate: None,
            graph_train: None,
            context_predictor: None,
            model_inference: None,
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

/// Run the CPU-hermetic conformal-coverage tier: load the committed spec, re-fold
/// every score family through the engine's real conformal calibration, emit the
/// report with the `conformal` tier set, and map the coverage-floor verdict to
/// the exit code. A coverage below its floor prints the numbers and exits
/// non-zero — the run never fakes a pass.
fn run_conformal_scale() -> std::process::ExitCode {
    let spec = match conformal::ConformalSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("conformal-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match conformal::run(&spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("conformal-scale coverage measurement failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = conformal::all_gates_passed(&tier);
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "conformal-scale",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            conformal: Some(tier),
            eval: None,
            propagate: None,
            graph_train: None,
            context_predictor: None,
            model_inference: None,
        },
    };
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "conformal coverage FELL BELOW a committed floor — see tiers.conformal.points[*] \
             for the family and size that regressed"
        );
        std::process::ExitCode::FAILURE
    }
}

/// Run the CPU-hermetic eval-metric tier: load the committed spec, re-fold every
/// metric through the engine's real metric kernels, assert the bootstrap CI's
/// order-invariance, emit the report with the `eval` tier set, and map the
/// tolerance verdict to the exit code. Any drift prints and exits non-zero.
fn run_eval_scale() -> std::process::ExitCode {
    let spec = match eval::EvalSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("eval-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match eval::run(&spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("eval-scale metric fold failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = eval::all_gates_passed(&tier);
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "eval-scale",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            conformal: None,
            eval: Some(tier),
            propagate: None,
            graph_train: None,
            context_predictor: None,
            model_inference: None,
        },
    };
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "an eval metric DRIFTED off its committed golden (or the eval_compare bootstrap CI \
             diverged across orderings) — see tiers.eval for the metric that regressed"
        );
        std::process::ExitCode::FAILURE
    }
}

/// Run the CPU-hermetic propagation tier: load the committed spec, re-fold the
/// gated digest through the engine's real `propagate_embeddings`, measure the
/// un-gated latency reference, emit the report with the `propagate` tier set, and
/// map the digest verdict to the exit code. A digest drift prints and exits
/// non-zero — the run never fakes a pass.
async fn run_propagate_scale() -> std::process::ExitCode {
    let spec = match propagate::PropagateSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("propagate-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match propagate::run(&spec).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("propagate-scale digest fold failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = propagate::gate_passed(&tier);
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "propagate-scale",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            conformal: None,
            eval: None,
            propagate: Some(tier),
            graph_train: None,
            context_predictor: None,
            model_inference: None,
        },
    };
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "the propagation digest DRIFTED off its committed value — the engine's \
             propagate_embeddings output changed; see tiers.propagate.digest for the bits"
        );
        std::process::ExitCode::FAILURE
    }
}

/// The calibration-set sizes the conformal-coverage curve is committed at: how
/// the finite-sample coverage tightens toward `1 − α` as `n` grows. The coverage
/// is the gate at each; the size is the curve.
const CONFORMAL_CAL_SIZES: [usize; 3] = [1_000, 10_000, 100_000];
/// The held-out test-set size every conformal coverage point is scored over —
/// large enough that the empirical coverage estimate is tight around the
/// guarantee at every calibration size.
const CONFORMAL_TEST_ROWS: usize = 20_000;
/// The nominal miscoverage level the committed conformal spec targets (a 90%
/// coverage guarantee).
const CONFORMAL_ALPHA: f64 = 0.1;
/// The class cardinality the synthetic LAC classification spec draws over.
const CONFORMAL_N_CLASSES: usize = 5;

/// Rebuild and write the committed conformal spec from a fresh measurement. The
/// off-box one-shot; prints the spec it wrote so the operator sees the numbers
/// being committed.
fn run_rebuild_conformal_spec() -> std::process::ExitCode {
    let spec = match conformal::rebuild_spec(
        CONFORMAL_ALPHA,
        CONFORMAL_N_CLASSES,
        CONFORMAL_TEST_ROWS,
        &CONFORMAL_CAL_SIZES,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rebuild-conformal-spec failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match serde_json::to_string_pretty(&spec) {
        Ok(json) => {
            if let Err(e) = std::fs::write(conformal::ConformalSpec::path(), format!("{json}\n")) {
                eprintln!("rebuild-conformal-spec could not write the spec: {e}");
                return std::process::ExitCode::FAILURE;
            }
            println!("{json}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to serialize conformal spec: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// The eval-set sizes the committed metric curve is folded at, as
/// `(retrieval_query_rows, classification_inference_rows)` pairs.
///
/// A geometric 3-point curve over sizes the hermetic `cargo test` gate re-folds
/// in seconds: the gate's job is to prove the metric kernels fold the same number
/// off the committed golden at each size (size-invariance of the *correctness*
/// numbers), which a tractable largest point demonstrates as faithfully as a huge
/// one — exactly the recall gate's "small committed projection in the engine
/// repo" precedent. The full {1k, 10k, 100k} *timing* curve is an off-box /
/// cookbook concern (re-run `rebuild-eval-spec` with larger sizes there), the
/// same split recall.rs documents between its committed slice and the 168k
/// cookbook gate.
const EVAL_SIZES: [(usize, usize); 3] = [(1_000, 1_000), (4_000, 4_000), (16_000, 16_000)];
/// The retrieval cutoff `k` the committed eval spec folds recall/MRR/nDCG at.
const EVAL_K: usize = 10;
/// The candidate-list length each synthetic query retrieves.
const EVAL_LIST_LEN: usize = 50;
/// The number of relevant documents seeded per synthetic query.
const EVAL_RELEVANT_PER_QUERY: usize = 5;
/// The class cardinality the synthetic classification golden draws over.
const EVAL_N_CLASSES: usize = 4;

/// Rebuild and write the committed eval spec from a fresh fold. The off-box
/// one-shot; prints the spec it wrote.
fn run_rebuild_eval_spec() -> std::process::ExitCode {
    let spec = eval::rebuild_spec(
        EVAL_K,
        EVAL_LIST_LEN,
        EVAL_RELEVANT_PER_QUERY,
        EVAL_N_CLASSES,
        &EVAL_SIZES,
    );
    match serde_json::to_string_pretty(&spec) {
        Ok(json) => {
            if let Err(e) = std::fs::write(eval::EvalSpec::path(), format!("{json}\n")) {
                eprintln!("rebuild-eval-spec could not write the spec: {e}");
                return std::process::ExitCode::FAILURE;
            }
            println!("{json}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to serialize eval spec: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// The embedding dimensionality the committed propagation fixture's `X⁽⁰⁾` and
/// the propagated output live in.
const PROPAGATE_DIM: usize = 16;
/// The number of classes the committed propagation graph wires within — a clique
/// per class, so two classes give a graph with structure to smooth over.
const PROPAGATE_N_CLASSES: usize = 4;
/// Nodes per class in the *gated* fixture (the tractable digest size the
/// hermetic `cargo test` gate re-folds in seconds). The gated node count is
/// `PROPAGATE_N_CLASSES · PROPAGATE_GATE_PER_CLASS`.
const PROPAGATE_GATE_PER_CLASS: usize = 8;
/// Bounded fan-out: each node wires to its next `PROPAGATE_FAN_OUT` class-mates
/// (a circulant graph per class), so the edge set is `O(nodes · fan_out)` and
/// stays under the engine's edge-set ceiling at the larger latency sizes.
const PROPAGATE_FAN_OUT: usize = 4;
/// The APPNP hop count the committed digest is folded at — the engine's
/// over-smoothing sweet spot.
const PROPAGATE_HOPS: usize = 2;
/// The APPNP teleport probability the committed digest is folded with — the
/// engine's default restart.
const PROPAGATE_ALPHA: f64 = 0.1;
/// The node counts the un-gated propagation latency reference is measured at — a
/// machine-dependent wall-time curve, ascending. The named sizes the
/// `propagate-scale` subcommand emits the reference at; they are NOT gated.
const PROPAGATE_LATENCY_NODES: [usize; 2] = [1_000, 10_000];

/// Rebuild and write the committed propagation spec from a fresh fold. The
/// off-box one-shot; prints the spec it wrote so the operator sees the digest
/// being committed.
async fn run_rebuild_propagate_spec() -> std::process::ExitCode {
    let spec = match propagate::rebuild_spec(
        PROPAGATE_DIM,
        PROPAGATE_N_CLASSES,
        PROPAGATE_GATE_PER_CLASS,
        PROPAGATE_FAN_OUT,
        PROPAGATE_HOPS,
        PROPAGATE_ALPHA,
        &PROPAGATE_LATENCY_NODES,
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rebuild-propagate-spec failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match serde_json::to_string_pretty(&spec) {
        Ok(json) => {
            if let Err(e) = std::fs::write(propagate::PropagateSpec::path(), format!("{json}\n")) {
                eprintln!("rebuild-propagate-spec could not write the spec: {e}");
                return std::process::ExitCode::FAILURE;
            }
            println!("{json}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to serialize propagate spec: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the CPU-hermetic graph fine-tune tier: load the committed spec, re-sample
/// the graph through the engine's real `GraphSampler`, gate the sampled-pair
/// digest and the throughput, emit the report with the `graph_train` tier set,
/// and map the verdict to the exit code. A digest drift or a throughput
/// regression prints and exits non-zero — the run never fakes a pass.
fn run_graph_train_scale() -> std::process::ExitCode {
    let spec = match graph_train::GraphTrainSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("graph-train-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match graph_train::run(&spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("graph-train-scale sample failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = graph_train::gates_passed(&tier);
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "graph-train-scale",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            conformal: None,
            eval: None,
            propagate: None,
            graph_train: Some(tier),
            context_predictor: None,
            model_inference: None,
        },
    };
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "graph fine-tune gate FAILED — the sampled-pair digest drifted off its committed \
             value, or the sample throughput regressed below the same-box floor; see \
             tiers.graph_train for the numbers"
        );
        std::process::ExitCode::FAILURE
    }
}

/// The committed graph fine-tune generation parameters — the synthetic-graph shape
/// and the sampler knobs the committed digest and same-box baseline are derived
/// from. A multi-community circulant with sparse bridges, sampled by a
/// higher-order biased walk with structure-aware negative mining.
const GRAPH_TRAIN_PARAMS: graph_train::GraphTrainParams = graph_train::GraphTrainParams {
    communities: 8,
    nodes_per: 64,
    intra_degree: 4,
    bridge_stride: 8,
    walk_length: 4,
    walks_per_node: 4,
    return_p: 1.0,
    in_out_q: 0.5,
    hard_negatives: 2,
    exclude_hops: 1,
    seed: 0x00C0_FFEE_0011,
};

/// Rebuild and write the committed graph fine-tune spec from a fresh sample. The
/// off-box one-shot; prints the spec it wrote so the operator sees the digest and
/// baseline being committed.
fn run_rebuild_graph_train_spec() -> std::process::ExitCode {
    let spec = match graph_train::rebuild_spec(GRAPH_TRAIN_PARAMS) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rebuild-graph-train-spec failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match serde_json::to_string_pretty(&spec) {
        Ok(json) => {
            if let Err(e) = std::fs::write(graph_train::GraphTrainSpec::path(), format!("{json}\n"))
            {
                eprintln!("rebuild-graph-train-spec could not write the spec: {e}");
                return std::process::ExitCode::FAILURE;
            }
            println!("{json}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to serialize graph-train spec: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the CPU-hermetic context-predictor tier: load the committed spec, measure
/// `train_context_predictor` throughput, re-fold the predict digest over the
/// committed weight bundle through `predict_with_context_predictor`, emit the
/// report with the `context_predictor` tier set, and map the verdict to the exit
/// code. A digest drift or a throughput regression prints and exits non-zero.
async fn run_context_predictor_scale() -> std::process::ExitCode {
    let spec = match context_predictor::ContextPredictorSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("context-predictor-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match context_predictor::run(&spec).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("context-predictor-scale run failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = context_predictor::gates_passed(&tier);
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "context-predictor-scale",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            conformal: None,
            eval: None,
            propagate: None,
            graph_train: None,
            context_predictor: Some(tier),
            model_inference: None,
        },
    };
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "context-predictor gate FAILED — the predicted-distribution digest drifted off its \
             committed value, or the training throughput regressed below the same-box floor; see \
             tiers.context_predictor for the numbers"
        );
        std::process::ExitCode::FAILURE
    }
}

/// The committed context-predictor generation parameters — the synthetic
/// meta-dataset shape, the predictor spec, and how many targets the predict digest
/// folds over. A CNP over a family of linear functions, the engine
/// context-predictor suite's own hermetic shape.
const CONTEXT_PREDICTOR_PARAMS: context_predictor::ContextPredictorParams =
    context_predictor::ContextPredictorParams {
        n_tasks: 8,
        rows_per_task: 18,
        dataset_seed: 321,
        architecture: "Cnp",
        context_k: 6,
        hidden_dim: 16,
        num_heads: 2,
        num_layers: 2,
        epochs: 30,
        learning_rate: 0.005,
        grad_clip: 1.0,
        test_task_fraction: 0.25,
        min_task_count: 4,
        spec_seed: 7,
        target_count: 5,
    };

/// Rebuild and write the committed context-predictor spec and its trained weight
/// bundle from a fresh train + predict. The off-box one-shot; prints the spec it
/// wrote so the operator sees the digest and baseline being committed.
async fn run_rebuild_context_predictor_spec() -> std::process::ExitCode {
    let spec = match context_predictor::rebuild_spec(CONTEXT_PREDICTOR_PARAMS).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rebuild-context-predictor-spec failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match serde_json::to_string_pretty(&spec) {
        Ok(json) => {
            if let Err(e) = std::fs::write(
                context_predictor::ContextPredictorSpec::path(),
                format!("{json}\n"),
            ) {
                eprintln!("rebuild-context-predictor-spec could not write the spec: {e}");
                return std::process::ExitCode::FAILURE;
            }
            println!("{json}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to serialize context-predictor spec: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the CPU-hermetic model-inference tier: load the committed spec, serve both
/// GPU-model verbs (`generate_text_embeddings` and `infer`) over the committed
/// tiny bundles on `Device::Cpu`, re-fold both digests, emit the report with the
/// `model_inference` tier set, and map the verdict to the exit code. A digest
/// drift or a serving-throughput regression prints and exits non-zero.
async fn run_model_inference_scale() -> std::process::ExitCode {
    let spec = match model_inference::ModelInferenceSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("model-inference-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match model_inference::run(&spec).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("model-inference-scale run failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = model_inference::gates_passed(&tier);
    let report = Report {
        engine_version: ENGINE_VERSION,
        host: Host::detect(),
        subcommand: "model-inference-scale",
        tiers: Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            conformal: None,
            eval: None,
            propagate: None,
            graph_train: None,
            context_predictor: None,
            model_inference: Some(tier),
        },
    };
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "model-inference gate FAILED — a served-output digest drifted off its committed value, \
             or a serving throughput regressed below the same-box floor; see tiers.model_inference \
             for the numbers"
        );
        std::process::ExitCode::FAILURE
    }
}

/// The committed model-inference generation parameters — the synthetic corpus
/// shape and how many targets the infer digest folds over.
const MODEL_INFERENCE_PARAMS: model_inference::ModelInferenceParams =
    model_inference::ModelInferenceParams {
        row_count: 16,
        corpus_seed: 11,
        target_count: 8,
    };

/// Rebuild and write the committed model-inference spec from a fresh serve over
/// the committed bundles. The off-box one-shot; prints the spec it wrote so the
/// operator sees the digests and baselines being committed.
async fn run_rebuild_model_inference_spec() -> std::process::ExitCode {
    let spec = match model_inference::rebuild_spec(MODEL_INFERENCE_PARAMS).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rebuild-model-inference-spec failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match serde_json::to_string_pretty(&spec) {
        Ok(json) => {
            if let Err(e) = std::fs::write(
                model_inference::ModelInferenceSpec::path(),
                format!("{json}\n"),
            ) {
                eprintln!("rebuild-model-inference-spec could not write the spec: {e}");
                return std::process::ExitCode::FAILURE;
            }
            println!("{json}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to serialize model-inference spec: {e}");
            std::process::ExitCode::FAILURE
        }
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
            conformal: None,
            eval: None,
            propagate: None,
            graph_train: None,
            context_predictor: None,
            model_inference: None,
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
            conformal: None,
            eval: None,
            propagate: None,
            graph_train: None,
            context_predictor: None,
            model_inference: None,
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
            conformal: None,
            eval: None,
            propagate: None,
            graph_train: None,
            context_predictor: None,
            model_inference: None,
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
