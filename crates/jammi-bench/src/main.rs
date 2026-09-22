//! `jammi-bench` — the scale and performance measurement harness for the Jammi
//! engine.
//!
//! A measurement *consumer* of the engine: it links `jammi-db`/`jammi-numerics`/
//! `jammi-ai` and drives their public surfaces at scale, emitting one
//! machine-readable JSON report per run. It is `publish = false` and names no
//! consumer — it measures the engine's generic primitives — so it is kept out of
//! the published workspace to keep the engine a clean library while still being
//! compile-checked by the workspace gate.
//!
//! Invoke as `cargo run -p jammi-bench --release -- <subcommand>`. Each tier has a
//! subcommand (and, where it commits a baseline/spec, a `rebuild-*` peer):
//! `search-rss` (bounded-RSS proof for streamed exact search), `arxiv` (ANN-vs-exact
//! recall curve over a committed corpus, held-out query set), `recall-sweep`,
//! `train-scale` (fine-tune throughput + live OOM negative-control), `conformal-scale`
//! (split-conformal coverage floor), `eval-scale` (retrieval/classification metric
//! goldens + bootstrap order-invariance),
//! and `encode-step` (the `encode` workload's rungs — the loaded model, the
//! serving plan at one and at N partitions — as legs for the ladder). Every committed number is a real re-derivable fold (a
//! `rebuild-*` subcommand reproduces it); an un-measured slot serializes as `null`,
//! never a faked zero.
//!
//! ## Breadth-grid cell (d): which verbs get a scale benchmark, and which are N/A
//!
//! A scale benchmark is meaningful only for verbs whose cost grows with input size
//! (the data/compute plane). Control-plane/metadata verbs have no scale dimension;
//! benchmarking them "at scale" would measure the catalog backend, not a Jammi
//! primitive — so they are documented N/A here rather than silently skipped.
//!
//! * SCALE-RELEVANT (benchmarked above): `search` + `build_neighbor_graph` (recall /
//!   RSS / sweep tiers), `fine_tune`, `fine_tune_graph`, `train_context_predictor`,
//!   `predict_with_context_predictor`, `generate_embeddings`, `infer`,
//!   `propagate_embeddings`, `conformalize{,_cqr,_interval}`, and the
//!   `eval_{embeddings,per_query,inference,compare}` family.
//! * N/A — control plane (O(1)/O(catalog), not O(corpus)): the `list_*` /
//!   `describe_*` / `get_server_info` / `set_tenant` / `tenant` / `tenant_scope` /
//!   `register_*` / `drop_*` / `publish_topic` / `subscribe_collect` /
//!   `create_mutable_table` / `drop_mutable_table` / `delete_model` / `preload_model` /
//!   `ephemeral_session` surfaces. (Trigger-stream delivery under load is a *guarantee*
//!   proven elsewhere, not a throughput cell.)
//! * N/A — bounded-by-k / covered-by-proxy: `rrf_fuse` and `assemble_context` are
//!   bounded by retrieval depth k (and `assemble_context`'s retrieval cost IS the
//!   `search` it composes); `encode_query` is single-item (its scale cousin is
//!   `generate_embeddings`); `sql`'s data-scale cost is DataFusion's, and the only
//!   Jammi-owned scale edge it reaches (vector scan) is `search`.
//!
//! The model-inference serving rates (`generate_embeddings`, `infer`) and the
//! training throughputs are inherently GPU-model numbers at scale; their PORTABLE
//! gates (output digests, coverage/metric floors) run here in CI over tiny CPU models
//! / same-box baselines, while the representative scaling RATE is captured off-box in
//! the cookbook (the A/B split). No automated CI lane invokes these `*-scale`
//! subcommands on a representative box and fails on the exit code.

mod cache_slo;
mod capture;
mod conformal;
mod context_predictor;
mod corpus;
mod encode_step;
mod eval;
mod finetune_run;
mod finetune_step;
mod fixture;
mod grad_oracle;
mod graph_sample;
mod kernel_arm;
mod ladder;
mod leg;
mod operator_mirror;
mod propagate;
mod rate_gate;
mod recall;
mod recompute_scale;
mod report;
mod rss;
mod search_rss;
mod sweep;
mod timing;
mod train_scale;
mod vram;

use clap::{Parser, Subcommand};

use std::path::PathBuf;

use report::{ArxivTier, Provenance, Report, Tiers};
use search_rss::Variant;
use train_scale::BackwardPath;

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

/// `finetune-run`'s flags, flattened into a `Box<Self>` `Command::FinetuneRun`
/// payload (`clippy::large_enum_variant`; see that variant's own doc).
#[derive(clap::Args)]
struct FinetuneRunArgs {
    /// Directory holding `config.json` + `model.safetensors` +
    /// `tokenizer.json`.
    #[arg(long)]
    model_dir: PathBuf,
    /// `fused` or `alloff` — CALLER-declared (see `finetune_run::Arm`'s
    /// doc); the caller is responsible for setting
    /// `JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused`
    /// itself before invoking this binary for the `alloff` arm.
    #[arg(long)]
    arm: String,
    /// Which TOWER of `--model-dir`'s checkpoint to fine-tune:
    /// `text_embedding` (the default), `image_embedding` (an
    /// OpenCLIP vision tower), or `audio_embedding` (an HF-CLAP HTSAT audio
    /// tower). The task also selects which ROW SHAPE `--train-jsonl` /
    /// `--heldout-jsonl` must carry — see those flags' docs and
    /// `finetune_run::Task`'s.
    #[arg(long, default_value = "text_embedding")]
    task: String,
    /// Training triplets — JSONL, one object per line.
    ///
    /// For `--task text_embedding`:
    /// `{"anchor_id","anchor_text","positive_id","positive_text","negative_id","negative_text"}`,
    /// in the committed train-split's fixed order.
    ///
    /// For a MEDIA task (`image_embedding`/`audio_embedding`):
    /// `{"anchor_id","anchor_path","positive_id","positive_path","negative_id","negative_path"}`,
    /// where each `*_path` is RELATIVE TO THIS FILE'S OWN DIRECTORY and
    /// names an encoded image/audio file this loader reads the bytes of
    /// (the shape `ci/scripts/perf/gen_fixed_shape_image_corpus.py` and
    /// `gen_fixed_length_audio_corpus.py` emit).
    #[arg(long)]
    train_jsonl: PathBuf,
    /// The held-out fixture's committed id list — TAB-separated
    /// `anchor_id\tpositive_id\tnegative_id`, one row per line, in
    /// COMMITTED order (this order is scoring identity).
    /// This file's bytes are what `heldout_ids_sha256` hashes.
    #[arg(long)]
    heldout_ids: PathBuf,
    /// The held-out fixture's text content — same JSONL shape as
    /// `--train-jsonl`, joined to `--heldout-ids`' rows BY id (row order
    /// need not match; `--heldout-ids` alone decides scoring order). This
    /// file's own bytes (the whole file, as read — not any per-row
    /// re-derivation) are what `heldout_pairs_sha256` hashes (the held-out
    /// TEXT is a total
    /// determinant of every per-example loss `d_i`, so it must be
    /// content-anchored exactly as `--heldout-ids` already is).
    #[arg(long)]
    heldout_jsonl: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Defaults to the tier's own protocol
    /// (`finetune_run::DEFAULT_EPOCHS`; see `DEFAULT_LEARNING_RATE`'s doc).
    #[arg(long, default_value_t = finetune_run::DEFAULT_EPOCHS)]
    epochs: usize,
    #[arg(long, default_value_t = finetune_run::DEFAULT_EVAL_CADENCE)]
    eval_cadence: usize,
    #[arg(long, default_value_t = 32)]
    batch: usize,
    /// Defaults to the tier's own protocol
    /// (`finetune_run::DEFAULT_LEARNING_RATE`, whose doc says why it is not
    /// the engine's `2e-4`).
    #[arg(long, default_value_t = finetune_run::DEFAULT_LEARNING_RATE)]
    lr: f64,
    /// Run this job as its own negative control: every optimizer step is
    /// applied at learning rate zero, so the whole loop runs and no trainable
    /// tensor moves. The leg reports `lr: 0.0`. `--lr` stays the job's real,
    /// positive rate (`--lr 0` is refused, as it is for any job) — see
    /// `finetune_run::FinetuneRunParams::applied_learning_rate`'s doc.
    #[arg(long, default_value_t = false)]
    zero_lr_control: bool,
    /// `constant`, `cosine_decay`, or `linear_decay`.
    #[arg(long, default_value = "constant")]
    schedule: String,
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,
    #[arg(long, default_value_t = 1)]
    grad_accum: usize,
    #[arg(long, default_value_t = 0.1)]
    validation_fraction: f64,
    /// MUST be `>= 10_000` (a patience that never stops the run early) —
    /// the run refuses a smaller value.
    #[arg(long, default_value_t = 10_000)]
    early_stopping_patience: usize,
    /// `train_loss` or `val_loss`.
    #[arg(long, default_value = "val_loss")]
    early_stopping_metric: String,
    /// `0.0` disables clipping.
    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f64,
    /// `triplet` or `mnrl` — which embedding objective to train over the
    /// SAME committed fixture (`mnrl`
    /// consumes the (anchor, positive) projection of the same rows in
    /// committed order, dropping the negative column). See
    /// `finetune_run::Objective`'s own doc.
    #[arg(long, default_value = "triplet")]
    objective: String,
    /// The Triplet objective's margin — read when `--objective triplet`.
    #[arg(long, default_value_t = 0.3)]
    margin: f64,
    /// MNRL's similarity-scale knob — read when `--objective mnrl`. `20.0`
    /// is the standard default (`jammi_wire::fine_tune::EmbeddingLoss::MultipleNegativesRanking`'s
    /// own doc).
    #[arg(long, default_value_t = 20.0)]
    temperature: f64,
    /// Comma-separated Matryoshka prefix dims; empty trains the full
    /// dimension only.
    #[arg(long, default_value = "")]
    matryoshka_dims: String,
    #[arg(long, default_value_t = 8)]
    lora_rank: usize,
    #[arg(long, default_value_t = 16.0)]
    lora_alpha: f64,
    #[arg(long, default_value_t = 0.05)]
    lora_dropout: f64,
    /// `zeros_b` (the default) or `gaussian` — which LoRA initialization
    /// this run's adapters are built under.
    /// `gaussian` exists for the BF16 pre-flight: under `zeros_b` every
    /// LoRA `A` has `dL/dA == 0` at step 1 (`B == 0` kills the gradient
    /// path), so a "every LoRA Var got a non-zero gradient" dtype check is
    /// VACUOUS in that mode. See `finetune_run::FinetuneRunParams::lora_init`'s
    /// doc.
    #[arg(long, default_value = "zeros_b")]
    lora_init: String,
    /// Comma-separated op key(s) this invocation INTENDS
    /// `JAMMI_KERNELS_DISABLE` to carry. When set, the run refuses (a) at
    /// START unless every named key is present in the process's real
    /// `JAMMI_KERNELS_DISABLE`, (b) at the END unless
    /// `jammi_kernels::admission::unmatched_disables()` is empty, and (c)
    /// at the END unless every named key's `fused` dispatch counter reads
    /// `0` — a leg failing any of the three is INVALID, never a datum. The
    /// eager-twin proof a forced-eager profile leg needs: without it, a
    /// dropped/mistyped/unforwarded env var reads identically to "nothing
    /// requested" and the leg silently measures the FUSED arm under an
    /// eager label. Same grammar as `JAMMI_KERNELS_DISABLE` itself, parsed
    /// by the SAME `jammi_kernels::admission::parse_disable_list`. See
    /// `finetune_run::FinetuneRunParams::expect_kernels_disabled`'s doc.
    #[arg(long)]
    expect_kernels_disabled: Option<String>,
    /// Comma-separated LoRA target selectors.
    #[arg(long, default_value = "Wqkv,Wo,Wi")]
    target_modules: String,
    /// Optional comma-separated layer indices LoRA injection is restricted
    /// to (`jammi_lora::should_apply_lora`'s own doc: a layer must appear
    /// in this list to receive an adapter). Empty (the default) means no
    /// restriction — every layer matching `--target-modules` gets a LoRA
    /// adapter.
    #[arg(long, default_value = "")]
    layers_to_transform: String,
    /// Backbone precision: f32, f16, or bf16.
    #[arg(long, default_value = "f32")]
    backbone_dtype: String,
    /// The tokenizer's truncation length. Defaults to the ENGINE's own
    /// default (`FineTuneConfig::max_seq_length`'s), read from the same
    /// constant: this tier measures the trainer users run, and a shorter
    /// bench-only default would silently measure a regime no job gets unless
    /// it asks for it. Recorded on the leg as the identity field
    /// `max_seq_length`.
    #[arg(long, default_value_t = jammi_ai::fine_tune::DEFAULT_MAX_SEQ_LENGTH)]
    max_seq_length: usize,
    /// CALLER-declared premise for the report's `admission_is_dense` field
    /// (default: `false`, matching the committed fixture's padded
    /// transport). This tier's real-text path never reaches
    /// `forward_with_lengths`'s dense-vs-padded fork, so there is no live
    /// signal to check this claim against — the value is recorded exactly
    /// as declared, for a downstream merger to check against the fixture's
    /// own known shape (see `finetune_run::FinetuneRunParams::expect_dense`'s
    /// doc).
    #[arg(long, default_value_t = false)]
    expect_dense: bool,
    /// CUDA ordinal; omit for CPU (the CPU-hermetic smoke path).
    #[arg(long)]
    cuda: Option<usize>,
    /// Scratch directory for this run's local catalog/artifact-store
    /// state. The run also writes `initial_adapter.safetensors` here — its
    /// untrained adapter, whose sha256 it reports as
    /// `initial_adapter_sha256` — before anything trains.
    #[arg(long)]
    work_dir: PathBuf,
    /// The mutant's own label (e.g.
    /// `"eps-0.10"` — no-producer: an illustrative example label, not a
    /// measurement) — OPTIONAL, and all-or-none with `--mutant-base-sha` /
    /// `--mutant-patch-sha256` (a partial mutant label is refused; see
    /// `finetune_run::run`'s own leading validation). Omitted entirely for
    /// an ordinary (non-mutant) leg — see
    /// `finetune_run::FinetuneRunParams::mutant_id`'s own doc for why these
    /// three are honest-labeling fields, never identity or provenance.
    #[arg(long)]
    mutant_id: Option<String>,
    /// The git commit sha this mutant's patch was cut against — see
    /// `mutant_id`'s doc.
    #[arg(long)]
    mutant_base_sha: Option<String>,
    /// sha256 (hex) of the mutant patch's own content — see `mutant_id`'s
    /// doc.
    #[arg(long)]
    mutant_patch_sha256: Option<String>,
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
    /// Internal: build the committed precision-recall fixture — freezes an
    /// `Int8` sidecar over the ALREADY-COMMITTED `fixtures/scale/` corpus (the
    /// same real embeddings the frozen `F32` bundle indexes) and merges a
    /// `"precision"` section (measured recall@k for the two-stage
    /// retrieve→rescore at the deployment default oversample, and for the
    /// naive `oversample = 1` no-rescore baseline; floor = measured − margin)
    /// into the existing `floor.json`. Run off-box once with
    /// `RAYON_NUM_THREADS=1` after `build-scale-fixture` has produced the `F32`
    /// fixture; the bundle is committed and CI only loads it. Not a CI step —
    /// the provenance-recording builder for the committed precision floor.
    #[command(hide = true)]
    BuildPrecisionRecallFixture,
    /// Internal: build the committed `Binary`-precision recall fixture —
    /// freezes a `Binary` (`B1`/Hamming) sidecar over the ALREADY-COMMITTED
    /// `fixtures/scale/` corpus (the same real embeddings the frozen
    /// `F32`/`Int8` bundles index) and merges `binary_*` keys (measured
    /// recall@k for the two-stage retrieve→rescore at `Binary`'s own default
    /// oversample, and for the naive `oversample = 1` no-rescore baseline;
    /// floor = bootstrap-CI-lower-bound − margin, NOT measured − margin — see
    /// `fixture.rs`'s `build_binary_recall_fixture`) into the existing
    /// `floor.json`. Run off-box once with `RAYON_NUM_THREADS=1` after
    /// `build-scale-fixture` and `build-precision-recall-fixture` have
    /// produced the `F32`/`Int8` fixtures; the bundle is committed and CI only
    /// loads it. Not a CI step — the provenance-recording builder for the
    /// committed `Binary` floor.
    #[command(hide = true)]
    BuildBinaryRecallFixture,
    /// Internal: build the committed segment-merge recall fixture — for each
    /// of the `Int8`/`Binary` precisions (the ones whose `search_final` runs a
    /// real retrieve→rescore stage), splits the ALREADY-COMMITTED
    /// `fixtures/scale/` corpus into two or more segments under each of
    /// several committed row→segment partitionings, freezes one sidecar per
    /// segment AT THAT PRECISION, assembles a `SegmentedIndex` per
    /// partitioning, measures its held-out `search_final` recall@k against the
    /// exact oracle, and merges a `"segment_merge"` section (per-precision,
    /// per-partitioning measured recall@k + floor, the live single-graph
    /// baseline at that precision, and the merge-vs-single-graph tracking
    /// margin — see `fixture.rs`'s `build_segment_recall_fixture`) into the
    /// existing `floor.json`. Run off-box once with `RAYON_NUM_THREADS=1`
    /// after `build-precision-recall-fixture` and
    /// `build-binary-recall-fixture` have produced the single-graph
    /// `Int8`/`Binary` fixtures; the segment bundles are committed and CI only
    /// ever loads them. Not a CI step — the provenance-recording builder for
    /// the committed segment-merge floor.
    #[command(hide = true)]
    BuildSegmentRecallFixture,
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
    /// asserting the bootstrap's order-invariance. Emits the JSON
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
    /// The `encode` workload's producer: the engine's serving path — rows in a
    /// table → one artifact per key — run through a RUNG (`--rung direct`:
    /// the loaded model on the rows, no plan; `plan`: the real verb at one
    /// partition; `plan-partitioned`: at `--partitions` N) for a task
    /// (`--task embed|infer`), over a `--rows` sweep, on CPU or `--cuda`, over
    /// a `--model-dir` checkpoint or the compiled-in fixture. Every (unit,
    /// take) is measured in a child process under the device-memory sampler;
    /// more than one `--rung` is served interleaved in that child. Emits one
    /// leg per rung per unit per take (`--legs-dir`, the ladder's leg
    /// contract) and prints a summary. Every number is RECORDED; the
    /// comparison is `jammi-bench ladder encode`'s.
    EncodeStep {
        /// What the rows are served for.
        #[arg(long, value_enum, default_value_t = encode_step::Task::Embed)]
        task: encode_step::Task,
        /// A rung of the leg session; repeat to serve several interleaved.
        #[arg(long = "rung", value_enum, required = true)]
        rungs: Vec<encode_step::Rung>,
        /// CUDA ordinal; omit for CPU.
        #[arg(long)]
        cuda: Option<usize>,
        /// A local checkpoint directory (`config.json`, `model.safetensors`,
        /// `tokenizer.json`, optionally `1_Pooling/config.json`); omit for
        /// the task's compiled-in fixture.
        #[arg(long)]
        model_dir: Option<std::path::PathBuf>,
        /// The corpus row count of each sweep unit, comma-separated.
        #[arg(long, value_delimiter = ',', default_values_t = [16, 256])]
        rows: Vec<usize>,
        /// Measured repeats of each unit, each in a process of its own.
        #[arg(long, default_value_t = 1)]
        takes: usize,
        /// The corpus generation seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// `[inference] batch_size` — the row cap of a forward chunk, every
        /// rung.
        #[arg(long, default_value_t = jammi_db::config::InferenceConfig::default().batch_size)]
        batch_size: usize,
        /// `[inference] batch_tokens` — the padded-token cap of a forward
        /// chunk, every rung.
        #[arg(long, default_value_t = jammi_db::config::InferenceConfig::default().batch_tokens)]
        batch_tokens: usize,
        /// `[inference] partitions` of the plan-partitioned rung.
        #[arg(long, default_value_t = 4)]
        partitions: usize,
        /// `[gpu] compute_precision` (`f32`, `f16`, `bf16`) — what the model
        /// loads at unless its own `config.json` declares one.
        #[arg(long, default_value = "f32")]
        compute_precision: jammi_numerics::ComputePrecision,
        /// Warm serves discarded before the measured ones, per rung.
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        /// Measured serves per rung (even, when rungs are interleaved).
        #[arg(long, default_value_t = 10)]
        iters: usize,
        /// Leave each unit's corpus (`corpus_<rows>.parquet`) and — without
        /// `--model-dir` — the fixture checkpoint (`model/`) here, for
        /// `reference/torch_encode.py` to read.
        #[arg(long)]
        exchange_dir: Option<std::path::PathBuf>,
        /// Write the legs here, one `<rung>__rows<N>__r<take>.json` each,
        /// a unit's first take with its vectors beside it.
        #[arg(long)]
        legs_dir: Option<std::path::PathBuf>,
    },
    /// Run a command under the device-memory sampler — the one external
    /// instrument every rung's leg, including a PyTorch reference's, reads
    /// its `peak_vram_bytes` from — and print one JSON object: the peak, the
    /// command's whole standard output and its exit code. Exits as the
    /// command did.
    SampleDevice {
        /// CUDA ordinal to sample; omit to sample nothing (a CPU leg).
        #[arg(long)]
        cuda: Option<usize>,
        /// The command and its arguments.
        #[arg(required = true, last = true)]
        command: Vec<String>,
    },
    /// Internal: one leg session — every `--rung` over one `--rows` unit at
    /// one `--take`, measured in THIS process — printing one report per
    /// rung as a JSON array. `encode-step` runs one per (unit, take) under
    /// the device-memory sampler.
    #[command(hide = true)]
    EncodeLeg {
        #[arg(long, value_enum)]
        task: encode_step::Task,
        #[arg(long = "rung", value_enum, required = true)]
        rungs: Vec<encode_step::Rung>,
        #[arg(long)]
        cuda: Option<usize>,
        #[arg(long)]
        model_dir: Option<std::path::PathBuf>,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        take: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        batch_size: usize,
        #[arg(long)]
        batch_tokens: usize,
        #[arg(long)]
        partitions: usize,
        #[arg(long)]
        compute_precision: jammi_numerics::ComputePrecision,
        #[arg(long)]
        warmup: usize,
        #[arg(long)]
        iters: usize,
        #[arg(long)]
        exchange_dir: Option<std::path::PathBuf>,
        #[arg(long)]
        legs_dir: Option<std::path::PathBuf>,
    },
    /// The encoder fine-tune step tier: time one real LoRA training step —
    /// three encoder forwards live on the tape at once, a cosine-margin triplet
    /// loss, one backward into the adapter tensors, an optional PRODUCTION
    /// gradient clip (`--max-grad-norm`, absent by default), and one AdamW
    /// step — over a ModernBERT checkpoint on disk. Runs on CPU by default and
    /// on a CUDA ordinal with `--cuda`.
    ///
    /// Every number is RECORDED, never gated: a step time is a property of
    /// `code x device x box`, so the only comparison a heterogeneous rented
    /// fleet supports is a ratio between two runs on the same box. Emits the
    /// JSON report with the `finetune_step` tier set.
    FinetuneStep {
        /// Directory holding `config.json` + `model.safetensors`.
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 128)]
        seq: usize,
        #[arg(long, default_value_t = 20)]
        steps: usize,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        #[arg(long, default_value_t = 8)]
        lora_rank: usize,
        #[arg(long, default_value_t = 16.0)]
        lora_alpha: f64,
        #[arg(long, default_value_t = 0.05)]
        lora_dropout: f32,
        /// Comma-separated LoRA target selectors.
        #[arg(long, default_value = "Wqkv,Wo,Wi")]
        target_modules: String,
        /// Backbone precision: f32, f16, or bf16.
        #[arg(long, default_value = "f32")]
        backbone_dtype: String,
        /// CUDA ordinal; omit for CPU.
        #[arg(long)]
        cuda: Option<usize>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Encode the three triplet groups in one forward (the trainer's
        /// behaviour). `--batched-forward false` measures the three-forward
        /// shape for an A/B on the same box.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        batched_forward: bool,
        /// Run the PRODUCTION gradient-clip (`clip_gradients`) after backward
        /// and before the optimizer step, at the point the trainer clips —
        /// the shipped trainer's default (`max_grad_norm = 1.0`) always
        /// clips, so omitting this flag measures a step the product does not
        /// run. Absent (the default) skips clipping entirely. Must be
        /// finite and > 0.0 when supplied.
        #[arg(long)]
        max_grad_norm: Option<f32>,
        /// Comma-separated op key(s) this invocation INTENDS
        /// `JAMMI_KERNELS_DISABLE` to carry. When set, the run hard-errors
        /// unless `jammi_kernels::admission::disabled_ops_requested()`
        /// reads back exactly this set — the binary controls its own argv,
        /// so a dropped/mistyped/unforwarded `JAMMI_KERNELS_DISABLE` env
        /// var (which otherwise reads identically to "nothing requested")
        /// becomes a hard failure on the SAME invocation instead of
        /// something a caller has to notice by eyeballing
        /// `kernels_disabled_requested` in the emitted JSON report
        /// afterward. See `finetune_step::FinetuneStepParams::expect_kernels_disabled`'s
        /// doc.
        #[arg(long)]
        expect_kernels_disabled: Option<String>,
        /// Comma-separated per-row REAL (non-pad) lengths for a genuinely
        /// right-padded batch -- one usize per row, `--batch` entries total,
        /// each in `1..=--seq`. Omit for this tier's dense behaviour (an
        /// all-ones mask). When supplied, every forward
        /// routes through `ModernBert::forward_with_lengths`'s trusted-
        /// lengths path, building the mask FROM
        /// these lengths (row `b`'s first `lengths[b]` positions `1`, the
        /// rest `0`) so the mask and the lengths can never disagree. See
        /// `finetune_step::FinetuneStepParams::row_lengths`'s doc.
        #[arg(long)]
        row_lengths: Option<String>,
    },
    /// The finetune-run tier: one full (seed, arm)
    /// fine-tune run driving the REAL `TrainingLoopBuilder` + the public
    /// `evaluate_held_out` seam over committed TRIPLET (anchor/positive/
    /// negative) text fixtures. See `finetune_run.rs`'s module doc for the
    /// resume-cycled per-epoch trajectory design and the arm-as-provenance
    /// convention. Emits the JSON report with the `finetune_run` tier set.
    ///
    /// Boxed (`clippy::large_enum_variant`): this variant's field count
    /// alone (every `FinetuneRunParams` knob is a CLI flag) makes it far
    /// the largest `Command` payload — boxing, not a blanket `#[allow]`,
    /// mirrors `jammi_ai::fine_tune::target::TrainingTarget::EncoderAdapters`'s
    /// own fix for the identical lint (see that variant's doc).
    FinetuneRun(Box<FinetuneRunArgs>),
    /// The jammi-vs-torch learning oracle: one forward+backward at
    /// identical LoRA weights (never an optimizer step), emitted as a
    /// `train-step` leg whose `gradients` carries every trainable tensor's
    /// gradient and weight. Filed under the `grads` take of the `train-step`
    /// ladder's `torch -> reference` edge, it is judged as gradient
    /// agreement — direction, never loss trajectory; see `grad_oracle.rs`'s
    /// module doc for why. `--lora-weights-out` writes the LoRA `A`/`B`
    /// values this call actually used (jammi's own internal safetensors
    /// naming); a later call's `--lora-weights-in` loads them back,
    /// overwriting the fresh seeded draw before the forward runs.
    GradOracle {
        /// Directory holding `config.json` + `model.safetensors`.
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 128)]
        seq: usize,
        #[arg(long, default_value_t = 16)]
        lora_rank: usize,
        #[arg(long, default_value_t = 32.0)]
        lora_alpha: f64,
        /// Comma-separated LoRA target selectors.
        #[arg(long, default_value = "Wqkv,Wo,Wi")]
        target_modules: String,
        /// Backbone precision: f32, f16, or bf16.
        #[arg(long, default_value = "f32")]
        backbone_dtype: String,
        /// CUDA ordinal; omit for CPU.
        #[arg(long)]
        cuda: Option<usize>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        batched_forward: bool,
        /// Load LoRA A/B from this safetensors file before the forward
        /// (jammi's own internal tensor naming — see `grad_oracle.rs`'s
        /// module doc). Omit to use a fresh seeded draw instead.
        #[arg(long)]
        lora_weights_in: Option<PathBuf>,
        /// Write the LoRA A/B values actually used (post-load, if any) to
        /// this safetensors file.
        #[arg(long)]
        lora_weights_out: Option<PathBuf>,
        /// Write the leg (a `grad-oracle` report with a `finetune_step`
        /// tier) here.
        #[arg(long)]
        out: PathBuf,
    },
    /// The CPU-hermetic cache-hit SLO tier: drives the engine's opt-in producer
    /// memoization (`CachePolicy::Use`) on a cacheable producer (the
    /// neighbour-graph, anchored on the immutable source-table `ResultDigest`).
    /// Times a cold `Use` build (nothing cached → the full compute) against a warm
    /// `Use` hit (the top-of-producer probe short-circuits the whole build) and
    /// gates that the hit cleared the committed minimum speed-up — the portable
    /// property of skipping the work, not a machine-dependent absolute. Emits the
    /// JSON report with the `cache_slo` tier set and exits non-zero if the warm hit
    /// did not short-circuit (wrong outcome) or did not clear the floor.
    CacheSloScale,
    /// The CPU-hermetic recompute tier: drives the engine's `recompute(Downstream)`
    /// bounded topological sweep over a synthetic derived-table DAG (an embedding
    /// table → a neighbour-graph → a graph propagation). Gates the sweep's
    /// CORRECTNESS — every DAG node recomputed exactly once, in topological
    /// (parent-before-child) order, a box-independent invariant — with the sweep
    /// wall-time at the named DAG size as an un-gated reference. Emits the JSON
    /// report with the `recompute` tier set and exits non-zero if the sweep
    /// dropped a node, double-counted one, or mis-ordered a parent/child.
    RecomputeScale,
    /// Internal: print this binary's own baked build-time identity
    /// (`build_sha`, `target`, `profile`, `build_features`,
    /// `report_schema_version`) as standalone JSON — the SAME object every
    /// other subcommand's report carries at `report.provenance`
    /// (`report::Provenance::baked`, filled by `build.rs` at COMPILE time,
    /// never read at run time). A shell producer (`finetune_step_ab.sh`,
    /// `proof_artifact.py`) runs this BEFORE a leg to cross-check
    /// `build_sha` against its own resolved sha, rather than discovering a
    /// stale binary only after paying for the measurement.
    #[command(hide = true)]
    Provenance,
    /// The parity ladder: compare a workload's rungs, edge by edge, over a
    /// directory of legs any producer emitted — this binary's own tiers for
    /// the engine's rungs, a reference script's JSON for the reference
    /// framework's. One operator judges every edge on speed, space and
    /// outcome; see `ladder`'s module doc. Emits one JSON verdict and a
    /// table, and exits non-zero on a refusal or a failed hard rule.
    Ladder(ladder::LadderArgs),
    /// The `graph-sample` workload's engine rung, `sampler`: sample each
    /// `--graph` through the engine's biased-walk sampler and file one leg per
    /// graph under `--legs-dir` (`sampler__edges<N>__r<take>.json`, the pair
    /// table beside it, the unit's law file `edges<N>.json` under `--law-dir`)
    /// — the warm per-iteration series, the peak resident set, the pair
    /// table's digest and the walks' second-order transition counts in the
    /// law's order. Several graphs are a size sweep, one process each. The
    /// comparison is `jammi-bench ladder graph-sample`'s.
    GraphSample(graph_sample::GraphSampleArgs),
    /// Write the synthetic multi-community graph at a given size as a graph
    /// directory — the input of a `graph-sample` size sweep.
    GraphFixture(graph_sample::GraphFixtureArgs),
    /// Sample a graph exactly as a `fine_tune_graph` job at the same sampler
    /// configuration does and write its training set as `pairs.jsonl`, in the
    /// job's `_ordinal` order and the row shape `finetune-run --train-jsonl`
    /// reads — so a resident fine-tune, the graph job and a PyTorch trainer all
    /// train on byte-identical input.
    GraphPairs(graph_sample::GraphPairsArgs),
    /// The `propagate` workload's engine rungs, `plan` (one partition) and
    /// `plan-partitioned` (`--partitions`): `propagate_embeddings` over the
    /// synthetic graph at each `--nodes` size, one leg and one process per
    /// point under `--legs-dir` — the warm per-iteration series, the peak
    /// resident set, the digest of the key-sorted propagated vectors and the
    /// vectors themselves — beside the unit's input files the PyTorch rungs
    /// read. The comparison is `jammi-bench ladder propagate`'s.
    Propagate(propagate::PropagateArgs),
    /// The `predictor-train-run` workload's engine rung, `in-process`, for the
    /// family member `--arch` names (`Cnp`, `AttnCnp`, `Tnp`): at each
    /// `--seeds` seed, sample the committed meta-dataset into episodes, write
    /// them and the seeded initial weights (the files a PyTorch twin loads),
    /// meta-train with the engine's own fit, and file one leg — every
    /// optimizer step's wall-clock, the peak resident set, the held-out loss at
    /// init and after every epoch, the train-side probe series, and the head's
    /// output on every held-out target. The comparison is
    /// `jammi-bench ladder predictor-train-run`'s.
    PredictorTrainRun(context_predictor::PredictorTrainArgs),
    /// The `JAMMI_KERNELS_DISABLE` value of a kernel arm on a checkpoint:
    /// the arm's families' keys, restricted to the keys one training step on
    /// the checkpoint consults.
    KernelArm(kernel_arm::KernelArmArgs),
    /// One pass of a checkpoint's admission census under this process's
    /// `JAMMI_KERNELS_DISABLE`, as JSON: `kernel-arm`'s child.
    KernelCensus(kernel_arm::KernelCensusArgs),
}

/// Map a leg subcommand's outcome to the process exit code, naming the
/// subcommand on failure.
fn leg_exit(
    subcommand: &str,
    outcome: Result<(), Box<dyn std::error::Error>>,
) -> std::process::ExitCode {
    match outcome {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{subcommand}: {e}");
            std::process::ExitCode::FAILURE
        }
    }
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
        Command::BuildPrecisionRecallFixture => run_build_precision_recall_fixture().await,
        Command::BuildBinaryRecallFixture => run_build_binary_recall_fixture().await,
        Command::BuildSegmentRecallFixture => run_build_segment_recall_fixture().await,
        Command::TrainScale => run_train_scale().await,
        Command::TrainMeasureOnce { path, pairs } => run_train_measure_once(&path, pairs),
        Command::TrainThroughputOnce { pairs } => run_train_throughput_once(pairs),
        Command::ConformalScale => run_conformal_scale(),
        Command::EvalScale => run_eval_scale(),
        Command::RebuildConformalSpec => run_rebuild_conformal_spec(),
        Command::RebuildEvalSpec => run_rebuild_eval_spec(),
        Command::RebuildContextPredictorSpec => run_rebuild_context_predictor_spec().await,
        Command::EncodeStep {
            task,
            rungs,
            cuda,
            model_dir,
            rows,
            takes,
            seed,
            batch_size,
            batch_tokens,
            partitions,
            compute_precision,
            warmup,
            iters,
            exchange_dir,
            legs_dir,
        } => run_encode_step(encode_step::EncodeStepParams {
            task,
            rungs,
            model_dir,
            rows,
            takes,
            seed,
            batch_size,
            batch_tokens,
            partitions,
            compute_precision,
            warmup,
            iters,
            gpu_device: cuda.map_or(encode_step::CPU_HERMETIC_DEVICE, |ordinal| ordinal as i32),
            exchange_dir,
            legs_dir,
        }),
        Command::SampleDevice { cuda, command } => run_sample_device(cuda, &command),
        Command::EncodeLeg {
            task,
            rungs,
            cuda,
            model_dir,
            rows,
            take,
            seed,
            batch_size,
            batch_tokens,
            partitions,
            compute_precision,
            warmup,
            iters,
            exchange_dir,
            legs_dir,
        } => {
            run_encode_leg(
                encode_step::EncodeStepParams {
                    task,
                    rungs,
                    model_dir,
                    rows: vec![rows],
                    takes: take,
                    seed,
                    batch_size,
                    batch_tokens,
                    partitions,
                    compute_precision,
                    warmup,
                    iters,
                    gpu_device: cuda
                        .map_or(encode_step::CPU_HERMETIC_DEVICE, |ordinal| ordinal as i32),
                    exchange_dir,
                    legs_dir,
                },
                rows,
                take,
            )
            .await
        }
        Command::FinetuneStep {
            model_dir,
            batch,
            seq,
            steps,
            warmup,
            lora_rank,
            lora_alpha,
            lora_dropout,
            target_modules,
            backbone_dtype,
            cuda,
            seed,
            batched_forward,
            max_grad_norm,
            expect_kernels_disabled,
            row_lengths,
        } => run_finetune_step(finetune_step::FinetuneStepParams {
            model_dir,
            batch,
            seq,
            steps,
            warmup,
            lora_rank,
            lora_alpha,
            lora_dropout,
            target_modules: target_modules
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
            backbone_dtype: match backbone_dtype.as_str() {
                "f32" => jammi_numerics::ComputePrecision::F32,
                "f16" => jammi_numerics::ComputePrecision::F16,
                "bf16" => jammi_numerics::ComputePrecision::BF16,
                other => {
                    eprintln!("unknown backbone_dtype {other:?}; expected f32, f16, or bf16");
                    return std::process::ExitCode::FAILURE;
                }
            },
            cuda_device: cuda,
            seed,
            batched_forward,
            max_grad_norm,
            // `--expect-kernels-disabled` and `JAMMI_KERNELS_DISABLE` are
            // the SAME grammar (a caller states the same disable list two
            // ways) — routed through the identical
            // `jammi_kernels::admission::parse_disable_list` a genuine
            // `JAMMI_KERNELS_DISABLE` read goes through, rather than a
            // second, hand-rolled parser that could (and did) diverge on
            // duplicate entries. Sorted into a `Vec` here (not left as a
            // `HashSet`) because `disabled_ops_requested()` — what this
            // gets compared against in `finetune_step::run` — is itself a
            // sorted, deduplicated `Vec`.
            expect_kernels_disabled: expect_kernels_disabled.map(|s| {
                let mut v: Vec<String> = jammi_kernels::admission::parse_disable_list(Some(&s))
                    .into_iter()
                    .collect();
                v.sort();
                v
            }),
            row_lengths: match row_lengths {
                None => None,
                Some(s) => match s
                    .split(',')
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(|t| t.parse::<usize>())
                    .collect::<Result<Vec<usize>, _>>()
                {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!(
                            "--row-lengths {s:?} is invalid: {e} (expected a comma-separated \
                             list of non-negative integers, one per row)"
                        );
                        return std::process::ExitCode::FAILURE;
                    }
                },
            },
        }),
        Command::FinetuneRun(args) => {
            let FinetuneRunArgs {
                model_dir,
                arm,
                task,
                train_jsonl,
                heldout_ids,
                heldout_jsonl,
                seed,
                epochs,
                eval_cadence,
                batch,
                lr,
                zero_lr_control,
                schedule,
                warmup_steps,
                weight_decay,
                grad_accum,
                validation_fraction,
                early_stopping_patience,
                early_stopping_metric,
                max_grad_norm,
                objective,
                margin,
                temperature,
                matryoshka_dims,
                lora_rank,
                lora_alpha,
                lora_dropout,
                lora_init,
                expect_kernels_disabled,
                target_modules,
                layers_to_transform,
                backbone_dtype,
                max_seq_length,
                expect_dense,
                cuda,
                work_dir,
                mutant_id,
                mutant_base_sha,
                mutant_patch_sha256,
            } = *args;
            let arm = match arm.parse::<finetune_run::Arm>() {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("finetune-run: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            let objective = match objective.parse::<finetune_run::Objective>() {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("finetune-run: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            let task = match task.parse::<finetune_run::Task>() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("finetune-run: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            // Parsed HERE, alongside `--arm`/`--objective`/`--task`, so an
            // invalid token refuses before this process reads a single
            // corpus file — a caller error, not something worth loading a
            // media corpus to discover.
            let lora_init = match finetune_run::parse_lora_init(&lora_init) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("finetune-run: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            // `--task` selects the ROW SHAPE, so exactly one of the two
            // loader pairs runs and the other's vectors stay empty (the
            // invariant `FinetuneRunParams::validate_rows_match_task`
            // re-checks on the tier side). A text leg takes byte-identically
            // the path it always has.
            let is_media = task != finetune_run::Task::Text;
            let mut train_pairs = Vec::new();
            let mut heldout_pairs = Vec::new();
            let mut train_media = Vec::new();
            let mut heldout_media = Vec::new();
            let train_pairs_file_sha256;
            let heldout_ids_sha256;
            let heldout_pairs_sha256;
            if is_media {
                match load_train_media_jsonl(&train_jsonl) {
                    Ok((rows, sha)) => {
                        train_media = rows;
                        train_pairs_file_sha256 = sha;
                    }
                    Err(e) => {
                        eprintln!("finetune-run: --train-jsonl {train_jsonl:?}: {e}");
                        return std::process::ExitCode::FAILURE;
                    }
                }
                match load_heldout_media_fixture(&heldout_ids, &heldout_jsonl) {
                    Ok((rows, ids_sha, jsonl_sha)) => {
                        heldout_media = rows;
                        heldout_ids_sha256 = ids_sha;
                        heldout_pairs_sha256 = jsonl_sha;
                    }
                    Err(e) => {
                        eprintln!(
                            "finetune-run: --heldout-ids {heldout_ids:?} / --heldout-jsonl \
                             {heldout_jsonl:?}: {e}"
                        );
                        return std::process::ExitCode::FAILURE;
                    }
                }
            } else {
                match load_train_jsonl(&train_jsonl) {
                    Ok((rows, sha)) => {
                        train_pairs = rows;
                        train_pairs_file_sha256 = sha;
                    }
                    Err(e) => {
                        eprintln!("finetune-run: --train-jsonl {train_jsonl:?}: {e}");
                        return std::process::ExitCode::FAILURE;
                    }
                }
                match load_heldout_fixture(&heldout_ids, &heldout_jsonl) {
                    Ok((rows, ids_sha, jsonl_sha)) => {
                        heldout_pairs = rows;
                        heldout_ids_sha256 = ids_sha;
                        heldout_pairs_sha256 = jsonl_sha;
                    }
                    Err(e) => {
                        eprintln!(
                            "finetune-run: --heldout-ids {heldout_ids:?} / --heldout-jsonl \
                             {heldout_jsonl:?}: {e}"
                        );
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
            let lr_schedule = match schedule.as_str() {
                "constant" => jammi_ai::fine_tune::LrSchedule::Constant,
                "cosine_decay" => jammi_ai::fine_tune::LrSchedule::CosineDecay,
                "linear_decay" => jammi_ai::fine_tune::LrSchedule::LinearDecay,
                other => {
                    eprintln!(
                        "finetune-run: unknown --schedule {other:?}; expected constant, \
                         cosine_decay, or linear_decay"
                    );
                    return std::process::ExitCode::FAILURE;
                }
            };
            let early_stopping_metric_v = match early_stopping_metric.as_str() {
                "train_loss" => jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
                "val_loss" => jammi_ai::fine_tune::EarlyStoppingMetric::ValLoss,
                other => {
                    eprintln!(
                        "finetune-run: unknown --early-stopping-metric {other:?}; expected \
                         train_loss or val_loss"
                    );
                    return std::process::ExitCode::FAILURE;
                }
            };
            let params = finetune_run::FinetuneRunParams {
                model_dir,
                arm,
                task,
                train_pairs,
                heldout_pairs,
                train_media,
                heldout_media,
                train_pairs_file_sha256,
                heldout_ids_sha256,
                heldout_pairs_sha256,
                seed,
                epochs,
                eval_cadence,
                batch_size: batch,
                learning_rate: lr,
                applied_learning_rate: if zero_lr_control {
                    jammi_ai::fine_tune::trainer::AppliedLearningRate::Zero
                } else {
                    jammi_ai::fine_tune::trainer::AppliedLearningRate::Scheduled
                },
                lr_schedule,
                warmup_steps,
                weight_decay,
                gradient_accumulation_steps: grad_accum,
                validation_fraction,
                early_stopping_patience,
                early_stopping_metric: early_stopping_metric_v,
                max_grad_norm,
                objective,
                margin,
                temperature,
                matryoshka_dims: matryoshka_dims
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<Vec<usize>, _>>()
                    .unwrap_or_else(|e| {
                        eprintln!("finetune-run: --matryoshka-dims is invalid: {e}");
                        std::process::exit(1);
                    }),
                lora_rank,
                lora_alpha,
                lora_dropout,
                lora_init,
                // `--expect-kernels-disabled` and `JAMMI_KERNELS_DISABLE` are
                // the SAME grammar (a caller states the same disable list two
                // ways) — routed through the identical
                // `jammi_kernels::admission::parse_disable_list` a genuine
                // `JAMMI_KERNELS_DISABLE` read goes through, never a second,
                // hand-rolled parser that could (and once did, on the
                // `finetune-step` twin) diverge on duplicate entries. Sorted
                // into a `Vec` here because `disabled_ops_requested()` — what
                // `finetune_run::run` compares this against — is itself a
                // sorted, deduplicated `Vec`.
                expect_kernels_disabled: expect_kernels_disabled.map(|s| {
                    let mut v: Vec<String> = jammi_kernels::admission::parse_disable_list(Some(&s))
                        .into_iter()
                        .collect();
                    v.sort();
                    v
                }),
                target_modules: target_modules
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect(),
                layers_to_transform: {
                    let selectors: Vec<&str> = layers_to_transform
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect();
                    if selectors.is_empty() {
                        None
                    } else {
                        match selectors
                            .iter()
                            .map(|s| s.parse::<usize>())
                            .collect::<Result<Vec<usize>, _>>()
                        {
                            Ok(v) => Some(v),
                            Err(e) => {
                                eprintln!("finetune-run: --layers-to-transform is invalid: {e}");
                                return std::process::ExitCode::FAILURE;
                            }
                        }
                    }
                },
                backbone_dtype: match backbone_dtype.as_str() {
                    "f32" => jammi_numerics::ComputePrecision::F32,
                    "f16" => jammi_numerics::ComputePrecision::F16,
                    "bf16" => jammi_numerics::ComputePrecision::BF16,
                    other => {
                        eprintln!("unknown backbone_dtype {other:?}; expected f32, f16, or bf16");
                        return std::process::ExitCode::FAILURE;
                    }
                },
                max_seq_length,
                expect_dense,
                cuda_device: cuda,
                work_dir,
                mutant_id,
                mutant_base_sha,
                mutant_patch_sha256,
            };
            run_finetune_run(params).await
        }
        Command::GradOracle {
            model_dir,
            batch,
            seq,
            lora_rank,
            lora_alpha,
            target_modules,
            backbone_dtype,
            cuda,
            seed,
            batched_forward,
            lora_weights_in,
            lora_weights_out,
            out,
        } => run_grad_oracle(
            grad_oracle::GradOracleParams {
                model_dir,
                batch,
                seq,
                lora_rank,
                lora_alpha,
                target_modules: target_modules
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect(),
                backbone_dtype: match backbone_dtype.as_str() {
                    "f32" => jammi_numerics::ComputePrecision::F32,
                    "f16" => jammi_numerics::ComputePrecision::F16,
                    "bf16" => jammi_numerics::ComputePrecision::BF16,
                    other => {
                        eprintln!("unknown backbone_dtype {other:?}; expected f32, f16, or bf16");
                        return std::process::ExitCode::FAILURE;
                    }
                },
                cuda_device: cuda,
                seed,
                batched_forward,
                lora_weights_in,
                lora_weights_out,
            },
            &out,
        ),
        Command::CacheSloScale => run_cache_slo_scale().await,
        Command::RecomputeScale => run_recompute_scale().await,
        Command::Provenance => run_provenance(),
        Command::Ladder(args) => ladder::run(&args),
        Command::GraphSample(args) => leg_exit("graph-sample", args.execute().await),
        Command::GraphFixture(args) => leg_exit("graph-fixture", args.execute()),
        Command::GraphPairs(args) => leg_exit("graph-pairs", args.execute()),
        Command::Propagate(args) => leg_exit("propagate", args.execute().await),
        Command::PredictorTrainRun(args) => leg_exit("predictor-train-run", args.execute().await),
        Command::KernelArm(args) => kernel_arm::run(&args),
        Command::KernelCensus(args) => kernel_arm::run_census(&args),
    }
}

/// The `provenance` subcommand: print this binary's baked build-time
/// identity as standalone JSON. Never fails — every field is a compile-time
/// literal or a linked-crate `const`, so there is nothing to read at run
/// time that could error (`provenance_baked.rs`'s
/// `runtime_env_and_cwd_are_inert` proves the run-time inertness this
/// relies on).
fn run_provenance() -> std::process::ExitCode {
    let provenance = Provenance::baked();
    println!(
        "{}",
        serde_json::to_string_pretty(&provenance).expect("serialize provenance")
    );
    std::process::ExitCode::SUCCESS
}

/// The `grad-oracle` subcommand: run one forward+backward (no optimizer
/// step) and write the leg to `out`, as a report with a `finetune_step`
/// tier. Records; does not gate — `jammi-bench ladder` judges it. Exits
/// non-zero only when the forward could not be taken at all.
fn run_grad_oracle(
    params: grad_oracle::GradOracleParams,
    out: &std::path::Path,
) -> std::process::ExitCode {
    let leg = match grad_oracle::run(&params) {
        Ok(leg) => leg,
        Err(e) => {
            eprintln!("grad-oracle failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let report = Report::new(
        "grad-oracle",
        Tiers {
            finetune_step: Some(leg),
            ..Default::default()
        },
    );
    let json = match serde_json::to_string_pretty(&report) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("grad-oracle: failed to serialize report: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(out, json) {
        eprintln!("grad-oracle: failed to write {out:?}: {e}");
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}

/// The `finetune-step` subcommand: run the tier and emit the report. Records;
/// does not gate. Exits non-zero only when the step could not be measured at
/// all — a missing checkpoint, a target-module set that matched no linear, a
/// device that could not be resolved, a `--max-grad-norm` that was supplied
/// but not finite and > 0.0, or `JAMMI_KERNELS_DISABLE`
/// naming an op key that never disabled a live dispatch this run
/// (`finetune_step::run`'s doc) — an INVALID run, reported as a failure
/// rather than as a JSON tier with a suspiciously-clean dispatch split.
fn run_finetune_step(params: finetune_step::FinetuneStepParams) -> std::process::ExitCode {
    let tier = match finetune_step::run(&params) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("finetune-step failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let report = Report::new(
        "finetune-step",
        Tiers {
            finetune_step: Some(tier),
            ..Default::default()
        },
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
    std::process::ExitCode::SUCCESS
}

/// One JSONL row shared by `--train-jsonl` and `--heldout-jsonl`: an
/// (anchor, positive, negative) TRIPLET, keyed by `anchor_id` — the SAME
/// field names the committed `finetune_heldout` fixture uses
/// in its own `heldout_pairs.jsonl`, so a producer script can point this
/// flag straight at that file (or a re-derivation of it) without a reshape.
#[derive(serde::Deserialize)]
struct TripletRow {
    anchor_id: String,
    anchor_text: String,
    #[serde(rename = "positive_id")]
    _positive_id: String,
    positive_text: String,
    #[serde(rename = "negative_id")]
    _negative_id: String,
    negative_text: String,
}

/// Load a JSONL file of [`TripletRow`]s, in file order, into
/// [`finetune_run::IdTriplet`]s keyed by `anchor_id` — the shape
/// `--train-jsonl` supplies. Returns the rows plus the sha256 (hex) of the
/// file's own bytes (this run's `train_pairs_file_sha256` — MEASURED off the
/// file this run actually read, never a caller-transcribed digest).
///
/// This is the RAW BYTES of one file this tier read directly, distinct in
/// KIND from the committed fixture manifest's own `dataset_sha256` (a
/// Merkle digest over PER-PAIR content hashes, built by a producer script
/// off-process — see `ci/scripts/perf/finetune_run_ab.sh`'s
/// `train_ids_sha256.json`): the two are different quantities computed by
/// different mechanisms over overlapping-but-not-identical inputs, so this
/// field earns its own name rather than colliding with that one under a
/// shared spelling. Content-
/// anchoring this run's train file against the committed
/// `train_ids_sha256.json` manifest is the PRODUCER's pre-run provisioning
/// check, not something this tier verifies for itself — it only records the
/// digest of the bytes it actually read.
fn load_train_jsonl(
    path: &std::path::Path,
) -> Result<(Vec<finetune_run::IdTriplet>, String), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let sha256 = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hex::encode(hasher.finalize())
    };
    let text = String::from_utf8(bytes)?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: TripletRow =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        rows.push(finetune_run::IdTriplet {
            id: row.anchor_id,
            anchor: row.anchor_text,
            positive: row.positive_text,
            negative: row.negative_text,
        });
    }
    Ok((rows, sha256))
}

/// [`TripletRow`]'s MEDIA twin, shared by `--train-jsonl` and
/// `--heldout-jsonl` when `--task` selects a media tower: an (anchor,
/// positive, negative) triplet of FILE PATHS, keyed by `anchor_id`.
///
/// Two explicit row structs selected by `--task`, deliberately NOT one
/// `#[serde(untagged)]` enum: an untagged enum silently falls through to the
/// second variant when the first fails to parse, so a text corpus fed to a
/// media leg (or a single row with a typo'd `anchor_text` key) would be
/// re-interpreted as the other shape instead of refused. The task says which
/// shape is expected and a row that is not that shape is a parse error
/// naming the line.
///
/// Every `*_path` is resolved RELATIVE TO THE JSONL'S OWN DIRECTORY (never
/// the process's cwd), so a corpus tree stays valid wherever it is unpacked
/// — the property the `ci/scripts/perf/gen_fixed_*_corpus.py` producers'
/// output depends on.
#[derive(serde::Deserialize)]
struct MediaTripletRow {
    anchor_id: String,
    anchor_path: String,
    #[serde(rename = "positive_id")]
    _positive_id: String,
    positive_path: String,
    #[serde(rename = "negative_id")]
    _negative_id: String,
    negative_path: String,
}

/// sha256 (hex) of `bytes` — the MEASURED content anchor this crate stamps
/// on every media member it reads (see [`finetune_run::MediaTriplet`]'s
/// doc).
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Read one media member: resolve `rel` against `base_dir` and return its
/// bytes plus their sha256. Refuses (naming the row, the field and the
/// resolved path) rather than substituting an empty buffer for a missing
/// file — a silently empty clip would decode to nothing and quietly train
/// on a degenerate batch.
fn read_media_member(
    base_dir: &std::path::Path,
    rel: &str,
    line_no: usize,
    field: &str,
) -> Result<(Vec<u8>, String), Box<dyn std::error::Error>> {
    let path = base_dir.join(rel);
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "line {line_no}: {field} {rel:?} resolved to {} — {e}",
            path.display()
        )
    })?;
    if bytes.is_empty() {
        return Err(format!(
            "line {line_no}: {field} {rel:?} ({}) is empty — an empty media file cannot be \
             decoded into a training example",
            path.display()
        )
        .into());
    }
    let sha = sha256_hex(&bytes);
    Ok((bytes, sha))
}

/// Load a JSONL file of [`MediaTripletRow`]s, in file order, into
/// [`finetune_run::MediaTriplet`]s keyed by `anchor_id` — the media shape
/// `--train-jsonl` supplies under `--task image_embedding`/`audio_embedding`.
/// Returns the rows plus the sha256 (hex) of the JSONL file's own bytes.
///
/// The returned digest carries the SAME meaning it does on the text path
/// ([`load_train_jsonl`]): the bytes of the manifest this run opened. It is
/// NOT a digest of the corpus — for media the manifest names files whose
/// content a caller could swap without touching the JSONL, so each row's
/// three members carry their OWN measured `*_sha256` (see
/// [`finetune_run::MediaTriplet`]), exactly the content-anchoring the
/// held-out text path has.
fn load_train_media_jsonl(
    path: &std::path::Path,
) -> Result<(Vec<finetune_run::MediaTriplet>, String), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let sha256 = sha256_hex(&bytes);
    let base_dir = path
        .parent()
        .ok_or("media --train-jsonl path has no parent directory to resolve *_path against")?
        .to_path_buf();
    let text = String::from_utf8(bytes)?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_no = i + 1;
        let row: MediaTripletRow =
            serde_json::from_str(line).map_err(|e| format!("line {line_no}: {e}"))?;
        let (anchor, anchor_sha256) =
            read_media_member(&base_dir, &row.anchor_path, line_no, "anchor_path")?;
        let (positive, positive_sha256) =
            read_media_member(&base_dir, &row.positive_path, line_no, "positive_path")?;
        let (negative, negative_sha256) =
            read_media_member(&base_dir, &row.negative_path, line_no, "negative_path")?;
        rows.push(finetune_run::MediaTriplet {
            id: row.anchor_id,
            anchor,
            positive,
            negative,
            anchor_sha256,
            positive_sha256,
            negative_sha256,
        });
    }
    Ok((rows, sha256))
}

/// The media twin of [`load_heldout_fixture`]: `heldout_ids` names the
/// COMMITTED scoring order (`anchor_id\tpositive_id\tnegative_id` per line),
/// `heldout_jsonl` supplies the media PATHS joined to it BY `anchor_id`.
/// Both files' own bytes are hashed here, off the files this run opened.
fn load_heldout_media_fixture(
    heldout_ids: &std::path::Path,
    heldout_jsonl: &std::path::Path,
) -> Result<(Vec<finetune_run::MediaTriplet>, String, String), Box<dyn std::error::Error>> {
    let ids_bytes = std::fs::read(heldout_ids)?;
    let heldout_ids_sha256 = sha256_hex(&ids_bytes);
    let ids_text = String::from_utf8(ids_bytes)?;

    let (rows, heldout_pairs_sha256) = load_train_media_jsonl(heldout_jsonl)?;
    let mut by_anchor: std::collections::HashMap<String, finetune_run::MediaTriplet> =
        std::collections::HashMap::new();
    for row in rows {
        by_anchor.insert(row.id.clone(), row);
    }

    let mut out = Vec::new();
    for (i, line) in ids_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let anchor_id = line
            .split('\t')
            .next()
            .ok_or_else(|| format!("heldout_ids line {}: missing anchor_id", i + 1))?
            .to_string();
        let row = by_anchor.get(&anchor_id).cloned().ok_or_else(|| {
            format!(
                "heldout_ids line {}: anchor_id {anchor_id:?} has no matching row in \
                 --heldout-jsonl",
                i + 1
            )
        })?;
        out.push(row);
    }
    Ok((out, heldout_ids_sha256, heldout_pairs_sha256))
}

/// Load the held-out fixture: `heldout_ids` names the COMMITTED order
/// (`anchor_id\tpositive_id\tnegative_id` per line — this file's bytes are
/// what `heldout_ids_sha256` hashes, MEASURED here, never transcribed);
/// `heldout_jsonl` supplies the TEXT, joined to each id row BY `anchor_id` —
/// its own bytes are what `heldout_pairs_sha256` hashes, likewise MEASURED
/// here off the file this run actually read (the held-out TEXT is a total
/// determinant of every per-example loss `d_i`; the id ORDER alone would
/// let a caller swap the anchor/positive/negative TEXT under a constant id
/// list without changing either committed digest).
fn load_heldout_fixture(
    heldout_ids: &std::path::Path,
    heldout_jsonl: &std::path::Path,
) -> Result<(Vec<finetune_run::IdTriplet>, String, String), Box<dyn std::error::Error>> {
    let ids_bytes = std::fs::read(heldout_ids)?;
    let heldout_ids_sha256 = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&ids_bytes);
        hex::encode(hasher.finalize())
    };
    let ids_text = String::from_utf8(ids_bytes)?;

    let jsonl_bytes = std::fs::read(heldout_jsonl)?;
    let heldout_pairs_sha256 = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&jsonl_bytes);
        hex::encode(hasher.finalize())
    };
    let jsonl_text = String::from_utf8(jsonl_bytes)?;
    let mut by_anchor: std::collections::HashMap<String, (String, String, String)> =
        std::collections::HashMap::new();
    for (i, line) in jsonl_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: TripletRow =
            serde_json::from_str(line).map_err(|e| format!("heldout jsonl line {}: {e}", i + 1))?;
        by_anchor.insert(
            row.anchor_id,
            (row.anchor_text, row.positive_text, row.negative_text),
        );
    }

    let mut rows = Vec::new();
    for (i, line) in ids_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let anchor_id = cols
            .next()
            .ok_or_else(|| format!("heldout_ids line {}: missing anchor_id", i + 1))?
            .to_string();
        let (anchor_text, positive_text, negative_text) =
            by_anchor.get(&anchor_id).cloned().ok_or_else(|| {
                format!(
                    "heldout_ids line {}: anchor_id {anchor_id:?} has no matching row in \
                     --heldout-jsonl",
                    i + 1
                )
            })?;
        rows.push(finetune_run::IdTriplet {
            id: anchor_id,
            anchor: anchor_text,
            positive: positive_text,
            negative: negative_text,
        });
    }
    Ok((rows, heldout_ids_sha256, heldout_pairs_sha256))
}

/// The `finetune-run` subcommand: run the tier (a resume-cycled multi-epoch
/// leg over the REAL `TrainingLoopBuilder` + the public `evaluate_held_out`
/// seam) and emit the report. `finetune_run::run` blocks on catalog/artifact-
/// store I/O via `Handle::current().block_on(..)` (mirroring
/// `fine_tune::worker::run_fine_tune_blocking`'s own posture — see that
/// function's doc), so it is driven from `spawn_blocking`, never called
/// directly on this async task (which would panic: "cannot start a runtime
/// from within a runtime").
async fn run_finetune_run(params: finetune_run::FinetuneRunParams) -> std::process::ExitCode {
    let tier = match jammi_ai::fine_tune::collective::BlockingCall::spawn_blocking(move |call| {
        finetune_run::run(&call, &params)
    })
    .await
    {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            eprintln!("finetune-run failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("finetune-run: task panicked: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let report = Report::new(
        "finetune-run",
        Tiers {
            finetune_run: Some(tier),
            ..Default::default()
        },
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
    std::process::ExitCode::SUCCESS
}

/// The `cache-slo-scale` subcommand: load the committed spec, run the tier (cold
/// vs warm `Use` neighbour-graph build), emit the report, and exit non-zero if
/// the warm hit did not clear the committed speed-up floor (or did not actually
/// short-circuit — `run` errors in that case).
async fn run_cache_slo_scale() -> std::process::ExitCode {
    let spec = match cache_slo::CacheSloSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cache-slo-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match cache_slo::run(&spec).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cache-slo-scale run failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = cache_slo::gate_passed(&tier);
    let report = Report::new(
        "cache-slo-scale",
        Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            finetune_step: None,
            finetune_run: None,
            conformal: None,
            eval: None,
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: Some(tier),
            recompute: None,
        },
    );
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "the cache hit did NOT clear the committed speed-up floor — the \
             top-of-producer probe did not short-circuit; see tiers.cache_slo.speedup"
        );
        std::process::ExitCode::FAILURE
    }
}

/// The `recompute-scale` tier: build the synthetic derived-table DAG, run the
/// `recompute(Downstream)` topological sweep, and gate that it recomputed every
/// node once in topological order. Emits the JSON report and exits non-zero if the
/// sweep's correctness invariant did not hold.
async fn run_recompute_scale() -> std::process::ExitCode {
    let spec = match recompute_scale::RecomputeSpec::load() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("recompute-scale could not load the committed spec: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tier = match recompute_scale::run(&spec).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("recompute-scale run failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let passed = recompute_scale::gate_passed(&tier);
    let report = Report::new(
        "recompute-scale",
        Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            finetune_step: None,
            finetune_run: None,
            conformal: None,
            eval: None,
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: None,
            recompute: Some(tier),
        },
    );
    emit(&report);
    if passed {
        std::process::ExitCode::SUCCESS
    } else {
        eprintln!(
            "the Downstream recompute sweep did NOT recompute every DAG node once in \
             topological order; see tiers.recompute"
        );
        std::process::ExitCode::FAILURE
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
    let report = Report::new(
        "train-scale",
        Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: Some(tier),
            finetune_step: None,
            finetune_run: None,
            conformal: None,
            eval: None,
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: None,
            recompute: None,
        },
    );
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
    let report = Report::new(
        "conformal-scale",
        Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            finetune_step: None,
            finetune_run: None,
            conformal: Some(tier),
            eval: None,
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: None,
            recompute: None,
        },
    );
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
    let report = Report::new(
        "eval-scale",
        Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: None,
            training: None,
            finetune_step: None,
            finetune_run: None,
            conformal: None,
            eval: Some(tier),
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: None,
            recompute: None,
        },
    );
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

/// Run the encode producer: every (unit, take) in a child under the sampler,
/// the legs written, the sweep summarised on stdout. Exits non-zero only when
/// a leg session failed — a checkpoint that does not load, a CUDA ordinal the
/// box does not have, a rung that was not deterministic or lost a row.
fn run_encode_step(params: encode_step::EncodeStepParams) -> std::process::ExitCode {
    match encode_step::run(&params) {
        Ok(sweep) => match serde_json::to_string_pretty(&sweep) {
            Ok(json) => {
                println!("{json}");
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("failed to serialize the encode sweep: {e}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("encode-step run failed: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run `command` under the sampler for `cuda` and print what it read.
fn run_sample_device(cuda: Option<usize>, command: &[String]) -> std::process::ExitCode {
    let (program, args) = match command.split_first() {
        Some(split) => split,
        None => {
            eprintln!("sample-device needs a command after --");
            return std::process::ExitCode::FAILURE;
        }
    };
    let mut child = std::process::Command::new(program);
    child.args(args);
    let sampled = match vram::run_sampled(&mut child, vram::device_memory_probe(cuda)) {
        Ok((output, peak_vram_bytes)) => vram::SampledRun {
            peak_vram_bytes,
            child_stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            exit_code: output.status.code(),
        },
        Err(e) => {
            eprintln!("sample-device could not run {program}: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match serde_json::to_string(&sampled) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("failed to serialize the sampled run: {e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    match sampled.exit_code {
        Some(0) => std::process::ExitCode::SUCCESS,
        Some(code) => std::process::ExitCode::from(code.clamp(1, 255) as u8),
        None => std::process::ExitCode::FAILURE,
    }
}

/// The `encode-leg` child: one leg session in this process, one report per
/// rung printed as a JSON array for `encode-step` to file.
async fn run_encode_leg(
    params: encode_step::EncodeStepParams,
    rows: usize,
    take: usize,
) -> std::process::ExitCode {
    let legs = match encode_step::measure_legs(&params, rows, take).await {
        Ok(legs) => legs,
        Err(e) => {
            eprintln!("encode-leg session failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match serde_json::to_string(&encode_step::leg_reports(legs)) {
        Ok(json) => {
            println!("{json}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to serialize the legs: {e}");
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
    let report = Report::new(
        "search-rss",
        Tiers {
            arxiv: None,
            binding: Some(tier),
            recall_sweep: None,
            training: None,
            finetune_step: None,
            finetune_run: None,
            conformal: None,
            eval: None,
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: None,
            recompute: None,
        },
    );
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
    let report = Report::new(
        "arxiv",
        Tiers {
            arxiv: Some(ArxivTier::with_recall(recall)),
            binding: None,
            recall_sweep: None,
            training: None,
            finetune_step: None,
            finetune_run: None,
            conformal: None,
            eval: None,
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: None,
            recompute: None,
        },
    );
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
    let report = Report::new(
        "recall-sweep",
        Tiers {
            arxiv: None,
            binding: None,
            recall_sweep: Some(tier),
            training: None,
            finetune_step: None,
            finetune_run: None,
            conformal: None,
            eval: None,
            encode_step: None,
            graph_sample: None,
            propagate: None,
            predictor_train_run: None,
            cache_slo: None,
            recompute: None,
        },
    );
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

/// Build the committed precision-recall fixture (the frozen `Int8` sidecar +
/// its `floor.json` `"precision"` section) over the already-committed
/// `fixtures/scale/` corpus, and print the resulting floor record.
///
/// Off-box one-shot: freezes the one `Int8` sidecar and merges the measured
/// recall@k (rescored and no-rescore) into the existing `floor.json`. Prints
/// the measured recall and the derived floors so the operator sees the
/// numbers being committed.
async fn run_build_precision_recall_fixture() -> std::process::ExitCode {
    match fixture::build_precision_recall_fixture(&arxiv_fixture_dir()).await {
        Ok(record) => match serde_json::to_string_pretty(&record) {
            Ok(json) => {
                println!("{json}");
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("failed to serialize precision floor record: {e}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("build-precision-recall-fixture failed: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Build the committed `Binary`-precision recall fixture (the frozen `Binary`
/// sidecar + its `floor.json` `binary_*` keys) over the already-committed
/// `fixtures/scale/` corpus, and print the resulting floor record.
///
/// Off-box one-shot: freezes the one `Binary` sidecar and merges the measured
/// recall@k (rescored at `Binary`'s own default oversample, and no-rescore)
/// into the existing `floor.json`. Prints the measured recall and the derived
/// floors so the operator sees the numbers being committed.
async fn run_build_binary_recall_fixture() -> std::process::ExitCode {
    match fixture::build_binary_recall_fixture(&arxiv_fixture_dir()).await {
        Ok(record) => match serde_json::to_string_pretty(&record) {
            Ok(json) => {
                println!("{json}");
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("failed to serialize binary precision floor record: {e}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("build-binary-recall-fixture failed: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Build the committed segment-merge recall fixture (the frozen per-segment
/// `Int8`/`Binary` sidecars + its `floor.json` `"segment_merge"` section) over
/// the already-committed `fixtures/scale/` corpus, and print the resulting
/// floor record.
///
/// Off-box one-shot: freezes the per-precision, per-partitioning segment
/// sidecars and merges the measured recall@k, the single-graph baseline at
/// each precision, and the tracking margin into the existing `floor.json`.
/// Prints the record so the operator sees the numbers being committed.
async fn run_build_segment_recall_fixture() -> std::process::ExitCode {
    match fixture::build_segment_recall_fixture(&arxiv_fixture_dir()).await {
        Ok(record) => match serde_json::to_string_pretty(&record) {
            Ok(json) => {
                println!("{json}");
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("failed to serialize segment recall floor record: {e}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("build-segment-recall-fixture failed: {e}");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical SHA-256 test vector for the message `"abc"` (NIST FIPS
    /// 180-4, Appendix B.1) — an INDEPENDENTLY KNOWN reference, not a value
    /// re-derived from this crate's own hasher. It is what binds
    /// [`sha256_hex`] (and through it every measured media digest) to the
    /// algorithm it claims to compute rather than to itself.
    const SHA256_OF_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn sha256_hex_matches_the_published_test_vector() {
        assert_eq!(sha256_hex(b"abc"), SHA256_OF_ABC);
        // Negative control: a one-bit-different message must not collide
        // with the vector (a stub that returned the constant would pass the
        // assertion above).
        assert_ne!(sha256_hex(b"abd"), SHA256_OF_ABC);
    }

    /// Write a media corpus tree: `files` (relative name -> bytes) plus a
    /// `triplets.jsonl` in the SAME directory whose `*_path` values are
    /// those relative names — the exact shape
    /// `ci/scripts/perf/gen_fixed_shape_image_corpus.py` and
    /// `gen_fixed_length_audio_corpus.py` emit.
    fn write_media_corpus(
        dir: &std::path::Path,
        files: &[(&str, &[u8])],
        rows: &[(&str, &str, &str, &str)],
    ) -> PathBuf {
        std::fs::create_dir_all(dir).expect("create corpus dir");
        for (name, bytes) in files {
            std::fs::write(dir.join(name), bytes).expect("write member");
        }
        let jsonl = dir.join("triplets.jsonl");
        let mut out = String::new();
        for (id, a, p, n) in rows {
            out.push_str(&format!(
                "{{\"anchor_id\":\"{id}\",\"anchor_path\":\"{a}\",\"positive_id\":\"{id}-p\",\
                 \"positive_path\":\"{p}\",\"negative_id\":\"{id}-n\",\"negative_path\":\"{n}\"}}\n"
            ));
        }
        std::fs::write(&jsonl, out).expect("write jsonl");
        jsonl
    }

    /// The media loader reads each row's three files RELATIVE TO THE JSONL'S
    /// OWN DIRECTORY and stamps each member's MEASURED sha256.
    ///
    /// The corpus lives in a tempdir that is not the process's cwd and holds
    /// names (`a.bin`, …) that do not exist beside the test binary, so a
    /// loader that resolved against the cwd would fail outright rather than
    /// pass for the wrong reason. One member is literally `b"abc"`, so its
    /// stamped digest is checked against the published vector above rather
    /// than against another call to the same hasher.
    #[test]
    fn media_loader_resolves_paths_against_the_jsonl_and_measures_digests() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus = tmp.path().join("nested").join("corpus");
        let jsonl = write_media_corpus(
            &corpus,
            &[
                ("a.bin", b"abc"),
                ("p.bin", b"positive"),
                ("n.bin", b"negative"),
            ],
            &[("row-0", "a.bin", "p.bin", "n.bin")],
        );
        let (rows, file_sha) = load_train_media_jsonl(&jsonl).expect("load media jsonl");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.id, "row-0");
        assert_eq!(row.anchor, b"abc");
        assert_eq!(row.positive, b"positive");
        assert_eq!(row.negative, b"negative");
        assert_eq!(
            row.anchor_sha256, SHA256_OF_ABC,
            "the stamped digest must be the sha256 of the bytes actually read"
        );
        assert_ne!(row.positive_sha256, row.negative_sha256);
        assert_eq!(
            file_sha,
            sha256_hex(&std::fs::read(&jsonl).expect("re-read jsonl")),
            "the returned manifest digest must be the JSONL file's own bytes"
        );
    }

    /// A `*_path` that does not resolve is a refusal naming the LINE, the
    /// FIELD and the resolved path — never an empty buffer that would train
    /// on nothing.
    #[test]
    fn media_loader_refuses_a_missing_member() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus = tmp.path().join("corpus");
        let jsonl = write_media_corpus(
            &corpus,
            &[("a.bin", b"abc"), ("p.bin", b"pp")],
            &[("row-0", "a.bin", "p.bin", "missing.bin")],
        );
        let err = load_train_media_jsonl(&jsonl)
            .expect_err("a missing member must refuse")
            .to_string();
        assert!(err.contains("line 1"), "{err}");
        assert!(err.contains("negative_path"), "{err}");
        assert!(err.contains("missing.bin"), "{err}");
    }

    /// An EMPTY member file is refused too — it resolves and reads fine, so
    /// only an explicit emptiness check catches it, and a zero-byte clip
    /// would otherwise decode to nothing and quietly train on a degenerate
    /// batch.
    #[test]
    fn media_loader_refuses_an_empty_member() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus = tmp.path().join("corpus");
        let jsonl = write_media_corpus(
            &corpus,
            &[("a.bin", b""), ("p.bin", b"pp"), ("n.bin", b"nn")],
            &[("row-0", "a.bin", "p.bin", "n.bin")],
        );
        let err = load_train_media_jsonl(&jsonl)
            .expect_err("an empty member must refuse")
            .to_string();
        assert!(err.contains("is empty"), "{err}");
        assert!(err.contains("anchor_path"), "{err}");
    }

    /// A TEXT row fed to the media loader is a parse error naming the line —
    /// not silently re-interpreted. This is why the two row shapes are two
    /// explicit structs rather than one `#[serde(untagged)]` enum (see
    /// [`MediaTripletRow`]'s doc): an untagged enum would fall through to
    /// the other variant instead of refusing.
    #[test]
    fn media_loader_refuses_a_text_shaped_row() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus = tmp.path().join("corpus");
        std::fs::create_dir_all(&corpus).expect("mkdir");
        let jsonl = corpus.join("triplets.jsonl");
        std::fs::write(
            &jsonl,
            "{\"anchor_id\":\"r\",\"anchor_text\":\"a\",\"positive_id\":\"p\",\
             \"positive_text\":\"b\",\"negative_id\":\"n\",\"negative_text\":\"c\"}\n",
        )
        .expect("write jsonl");
        let err = load_train_media_jsonl(&jsonl)
            .expect_err("a text row must not parse as a media row")
            .to_string();
        assert!(err.contains("line 1"), "{err}");
        assert!(
            err.contains("anchor_path"),
            "the error must name the missing field: {err}"
        );
    }

    /// The held-out media fixture is scored in the ORDER `--heldout-ids`
    /// commits, not the order the JSONL happens to list — the same identity
    /// rule the text path holds.
    #[test]
    fn media_heldout_fixture_follows_the_committed_id_order() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus = tmp.path().join("corpus");
        let jsonl = write_media_corpus(
            &corpus,
            &[
                ("a0.bin", b"a0"),
                ("a1.bin", b"a1"),
                ("p.bin", b"pp"),
                ("n.bin", b"nn"),
            ],
            &[
                ("row-0", "a0.bin", "p.bin", "n.bin"),
                ("row-1", "a1.bin", "p.bin", "n.bin"),
            ],
        );
        // Ids file lists row-1 FIRST — the reverse of the JSONL's order.
        let ids = corpus.join("heldout_ids.txt");
        std::fs::write(&ids, "row-1\trow-1-p\trow-1-n\nrow-0\trow-0-p\trow-0-n\n")
            .expect("write ids");
        let (rows, ids_sha, jsonl_sha) =
            load_heldout_media_fixture(&ids, &jsonl).expect("load media heldout");
        assert_eq!(
            rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["row-1", "row-0"],
            "scoring order is the ids file's, never the JSONL's"
        );
        assert_eq!(rows[0].anchor, b"a1");
        assert_eq!(
            ids_sha,
            sha256_hex(&std::fs::read(&ids).expect("re-read ids"))
        );
        assert_eq!(
            jsonl_sha,
            sha256_hex(&std::fs::read(&jsonl).expect("re-read jsonl"))
        );
    }

    /// An id in `--heldout-ids` with no matching row in the media JSONL is
    /// refused by name — never silently dropped, which would shorten the
    /// scored set without changing either committed digest.
    #[test]
    fn media_heldout_fixture_refuses_an_unjoined_id() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus = tmp.path().join("corpus");
        let jsonl = write_media_corpus(
            &corpus,
            &[("a.bin", b"aa"), ("p.bin", b"pp"), ("n.bin", b"nn")],
            &[("row-0", "a.bin", "p.bin", "n.bin")],
        );
        let ids = corpus.join("heldout_ids.txt");
        std::fs::write(&ids, "row-0\tx\ty\nrow-9\tx\ty\n").expect("write ids");
        let err = load_heldout_media_fixture(&ids, &jsonl)
            .expect_err("an unjoined id must refuse")
            .to_string();
        assert!(err.contains("row-9"), "{err}");
    }

    /// End-to-end schema binding: the committed Python producers' OUTPUT
    /// must load through this crate's media loader unchanged.
    ///
    /// This is the one place the two sides of the contract meet — the
    /// producers pin the six key names as a literal in their own suites, and
    /// this test proves the Rust struct deserializes exactly what they
    /// write, files and all. It runs whenever `python3` is on PATH (CI
    /// always has it — the whole `ci/scripts/perf/test_*.py` suite depends on
    /// it); on a machine without it the test says so loudly on stderr rather
    /// than reporting a pass it did not earn.
    #[test]
    fn the_python_producers_emit_exactly_the_media_row_shape_this_loader_reads() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut ran_any = false;
        for (script, extra) in [
            (
                "ci/scripts/perf/gen_fixed_shape_image_corpus.py",
                vec!["--size".to_string(), "8".to_string()],
            ),
            (
                "ci/scripts/perf/gen_fixed_length_audio_corpus.py",
                vec![
                    "--seconds".to_string(),
                    "0.05".to_string(),
                    "--sample-rate".to_string(),
                    "16000".to_string(),
                ],
            ),
        ] {
            let out_dir = tmp.path().join(
                std::path::Path::new(script)
                    .file_stem()
                    .expect("script stem"),
            );
            let output = match std::process::Command::new("python3")
                .current_dir(&repo_root)
                .arg(script)
                .arg("--rows")
                .arg("4")
                .arg("--seed")
                .arg("11")
                .arg("--out-dir")
                .arg(&out_dir)
                .args(&extra)
                .output()
            {
                Ok(o) => o,
                Err(e) => {
                    eprintln!(
                        "python3 is not runnable here ({e}); the producer/loader schema binding \
                         was NOT exercised for {script}"
                    );
                    continue;
                }
            };
            assert!(
                output.status.success(),
                "{script} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            ran_any = true;
            let jsonl = out_dir.join("triplets.jsonl");
            let (rows, _sha) = load_train_media_jsonl(&jsonl)
                .unwrap_or_else(|e| panic!("{script}'s output must load through this crate: {e}"));
            assert_eq!(rows.len(), 4, "{script}");
            for row in &rows {
                assert!(
                    !row.anchor.is_empty() && !row.positive.is_empty() && !row.negative.is_empty()
                );
                assert_ne!(
                    row.anchor_sha256, row.positive_sha256,
                    "{script} must emit a positive that is a DISTINCT file from its anchor"
                );
                assert_ne!(row.anchor_sha256, row.negative_sha256, "{script}");
                assert_ne!(row.positive_sha256, row.negative_sha256, "{script}");
            }
        }
        if !ran_any {
            eprintln!(
                "WARNING: no producer was exercised — python3 was unavailable. The \
                 producer/loader schema binding is UNVERIFIED in this run."
            );
        }
    }
}
