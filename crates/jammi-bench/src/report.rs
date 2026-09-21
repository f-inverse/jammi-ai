//! The machine-readable report schema the harness emits.
//!
//! One JSON document per invocation, shaped so a downstream perf-gate can diff
//! a fresh run against committed goldens without parsing free text. The shape
//! covers every planned measurement tier up front — embed throughput, search
//! QPS, recall@k, propagate latency, peak RSS — so the schema is stable from
//! the first emit; tiers not yet measured serialize an explicit
//! [`Measurement::not_yet_measured`] marker rather than a zero that a gate
//! could mistake for a real datapoint.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::leg::Leg;

use jammi_db::config::StoragePrecision;

/// Workspace version this binary was built from, stamped into every report so a
/// downstream gate can reject a cross-version comparison.
const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// One harness invocation's full output.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Workspace version this binary was built from — a gate rejects a
    /// cross-version comparison, so the version travels with every number.
    pub engine_version: &'static str,
    /// Host facts that bear on the numbers (core count, total RAM).
    pub host: Host,
    /// Which subcommand produced this report.
    pub subcommand: &'static str,
    /// This binary's own baked build-time identity — see [`Provenance`]'s
    /// own doc. Present on every report (never conditionally omitted),
    /// exactly like `engine_version`/`host`.
    pub provenance: Provenance,
    /// The measured tiers. Each tier is a named bag of measurements; a tier
    /// not exercised by this subcommand is omitted entirely (absent, not null).
    pub tiers: Tiers,
}

impl Report {
    /// Construct a report for `subcommand`, filling `engine_version`/`host`/
    /// `provenance` from this process's own identity — the ONE constructor,
    /// so the three shared fields are filled in one place, not at every
    /// subcommand where they could individually drift.
    pub fn new(subcommand: &'static str, tiers: Tiers) -> Self {
        let report = Self {
            engine_version: ENGINE_VERSION,
            host: Host::detect(),
            subcommand,
            provenance: Provenance::baked(),
            tiers,
        };
        // Identity completeness, enforced on every real run — not only in a test
        // that could itself drift from a future field rename (see
        // `assert_identity_fields_present`'s own doc).
        let value =
            serde_json::to_value(&report.provenance).expect("serialize provenance for self-check");
        assert_identity_fields_present(&value, REPORT_IDENTITY_FIELDS);
        report
    }
}

/// Verify every `(field, nullable)` entry in `fields` is present in `value`
/// (a serialized report/tier/provenance object), and non-null where
/// declared [`Nullable::NonNull`] — the RUNTIME enforcement half of the
/// identity-completeness consts ([`TrainStepPayload::IDENTITY_FIELDS`],
/// `crate::grad_oracle::GradOracleReport::IDENTITY_FIELDS`,
/// [`REPORT_IDENTITY_FIELDS`]). Called from `Report::new` (every emitted
/// report), `finetune_step::run`, and `grad_oracle::run` — a producer's own
/// declared identity contract is checked against its OWN emitted JSON on
/// every real invocation, not only inside a `#[test]` fixture that could
/// itself silently drift out of sync with a future field rename. Panics (a
/// bug in THIS crate, never a caller input — see `finetune_step_identity_
/// fields_are_emitted`/`grad_oracle_identity_fields_are_emitted` for the
/// same check exercised as a `#[test]`), mirroring the `.expect(..)`-heavy
/// posture every emit site already takes on its own serialization.
pub fn assert_identity_fields_present(value: &serde_json::Value, fields: &[(&str, Nullable)]) {
    let obj = value
        .as_object()
        .expect("identity-checked value must be a JSON object");
    for (field, nullable) in fields {
        let entry = obj
            .get(*field)
            .unwrap_or_else(|| panic!("IDENTITY_FIELDS names {field:?}, absent on this report"));
        if *nullable == Nullable::NonNull {
            assert!(
                !entry.is_null(),
                "{field:?} is declared NonNull but serialized as null"
            );
            // An empty STRING is not JSON `null`, so the check above alone
            // lets a NonNull field through with no real content (e.g. a
            // `TARGET`/`PROFILE` baked as `""`; `build.rs` falls back to
            // `"unknown"` at the source, and this check catches a
            // regression back to `""`). A non-string NonNull field (a number,
            // bool, array, object) is unaffected — this check only fires
            // on the string variant.
            if let Some(s) = entry.as_str() {
                assert!(
                    !s.is_empty(),
                    "{field:?} is declared NonNull but serialized as an empty string"
                );
            }
        }
    }
}

/// This exact binary's build-time identity, baked by `build.rs` and read
/// back here with `env!()` — a compile-time literal, NEVER a run-time
/// `std::env::var`/`git` read (`provenance_baked.rs`'s
/// `runtime_env_and_cwd_are_inert` is the mechanical proof: re-running the
/// same binary with a DIFFERENT `JAMMI_BUILD_SHA` in the environment, from a
/// fresh `cwd`, yields a byte-identical [`Provenance`]).
///
/// One instance per process, shared by every emitted [`Report`] (at
/// `report.provenance`) AND by the standalone `jammi-bench provenance`
/// subcommand (`main.rs`'s `run_provenance`), so a shell producer can read
/// this binary's identity BEFORE spending any wall-clock on a leg, rather
/// than discovering a stale binary only after the fact.
/// Every producer script cross-checks it against the sha it stamps before
/// any leg runs.
///
/// The three fields [`REPORT_IDENTITY_FIELDS`] names — `build_sha`,
/// `target`, `profile` — are what lets a downstream identity-completeness reader
/// tell "these two legs ran the same code on the same target" apart from
/// "these two legs merely both filled in the same COMPARISON tuple".
/// Consumers: `check_cuda_run_artifacts.py`'s v2-leg identity walk (rule
/// (i)) parses this const straight out of this source file
/// (`build_identity_tuples`, "never hand-typed") as the provenance half of
/// the identity-key roster a v2 leg must carry; the ladder compares a
/// payload's `IDENTITY_FIELDS` only, which does not include
/// `build_sha`/`target`/`profile`.
/// `build_features` is measurement/provenance context (what this binary
/// COULD dispatch), not part of that identity triple.
#[derive(Debug, Serialize)]
pub struct Provenance {
    /// `<sha>`, `<sha>-dirty`, or the literal `"unknown"` — see `build.rs`'s
    /// own module doc for the full precedence. NEVER asserted 40-hex by any
    /// reader in this tree: a dev tree is legitimately dirty, or entirely
    /// git-less (a packaged source tarball), and both are honest states,
    /// not failures.
    pub build_sha: &'static str,
    /// Cargo's own `$TARGET` for this build (e.g.
    /// `x86_64-unknown-linux-gnu`).
    pub target: &'static str,
    /// Cargo's own `$PROFILE` for this build (`debug` or `release`).
    pub profile: &'static str,
    /// Sorted, deduplicated linked-crate feature names this binary was
    /// compiled with — see [`build_features`]'s own doc for why these are
    /// cross-crate `const`s, never `CARGO_FEATURE_*`.
    pub build_features: Vec<&'static str>,
    /// The `Report` JSON shape version: `2`.
    pub report_schema_version: u32,
}

impl Provenance {
    /// Read this binary's baked identity. Every field is a compile-time
    /// literal (`env!()`) or a linked-crate `const` — nothing here touches
    /// the process environment or the filesystem at RUN time.
    pub fn baked() -> Self {
        Self {
            build_sha: env!("JAMMI_BUILD_SHA"),
            target: env!("JAMMI_BUILD_TARGET"),
            profile: env!("JAMMI_BUILD_PROFILE"),
            build_features: build_features(),
            report_schema_version: 2,
        }
    }
}

/// The linked-crate feature facts baked into [`Provenance::build_features`] —
/// sorted, deduplicated. Read as `const`s from the crate that actually OWNS
/// each feature's resolution (`jammi_kernels::admission::{CUDA_COMPILED,
/// FLASH_COMPILED}`), never `CARGO_FEATURE_*`: that family of env vars only
/// ever reflects THIS crate's (`jammi-bench`'s) own `[features]` graph, and
/// would silently miss `jammi-kernels`' actual cuda/flash-attn resolution —
/// `jammi_kernels::admission::FLASH_COMPILED`'s own doc blesses exactly
/// this cross-crate constant read for a process-identity report. `"bench-cuda"`
/// is jammi-bench's OWN `cuda` feature (`Cargo.toml`'s `cuda = ["jammi-ai/cuda"]`,
/// the on-GPU inference tier's own gate) — named distinctly from
/// `jammi-kernels`' `"cuda"` so the two are never conflated: a build can
/// have one on without the other (e.g. `jammi-bench` built with `--features
/// cuda` while `jammi-kernels` itself resolved `cuda` off is impossible in
/// practice given the dependency chain, but the two facts are still
/// independently-sourced constants, not one flag standing in for both).
///
/// `pub(crate)`: `finetune_step.rs` calls this SAME function to fill
/// `TrainStepPayload::build_features` (a tier-level echo of this exact
/// list, never a second, independently-drifting computation — see that
/// field's own doc for why a raw leg needs this locally, not only via
/// `report.provenance.build_features`).
pub(crate) fn build_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if jammi_kernels::admission::CUDA_COMPILED {
        features.push("cuda");
    }
    if jammi_kernels::admission::FLASH_COMPILED {
        features.push("flash-attn");
    }
    if cfg!(feature = "cuda") {
        features.push("bench-cuda");
    }
    features.sort_unstable();
    features.dedup();
    features
}

/// Whether an identity-completeness field may legitimately serialize as
/// JSON `null`, and what that null MEANS when it does — never a bare
/// "absent"/"producer predates this field" reading. A field with no
/// [`NullMeans`](Nullable::NullMeans) entry is
/// [`NonNull`](Nullable::NonNull): a null reading on it is itself a
/// finding, not a legitimate state a downstream leg-premise check should
/// silently accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nullable {
    /// Must always be present and non-null on a valid report.
    NonNull,
    /// May be present-and-null, and when it is, means exactly this — e.g.
    /// `max_grad_norm: null` means "no clip was applied", never "this
    /// producer predates the clip determinant". Constructed by
    /// `TrainStepPayload::IDENTITY_FIELDS`'s own `max_grad_norm` entry.
    NullMeans(&'static str),
}

/// The three [`Provenance`] fields every tier-level `IDENTITY_FIELDS` const
/// appends (the report-level half) — `build_sha`,
/// `target`, `profile`. Declared once here so
/// [`TrainStepPayload::IDENTITY_FIELDS`] and
/// [`crate::grad_oracle::GradOracleReport::IDENTITY_FIELDS`] both cite the
/// SAME three names rather than each spelling them out independently.
pub const REPORT_IDENTITY_FIELDS: &[(&str, Nullable)] = &[
    ("build_sha", Nullable::NonNull),
    ("target", Nullable::NonNull),
    ("profile", Nullable::NonNull),
];

/// Host facts that contextualize a measurement.
#[derive(Debug, Serialize)]
pub struct Host {
    /// Logical CPU count.
    pub logical_cpus: usize,
    /// Total system RAM in mebibytes, as reported by the OS.
    pub total_ram_mib: u64,
}

impl Host {
    /// Read host facts from the running process's view of the machine.
    pub fn detect() -> Self {
        Self {
            logical_cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            total_ram_mib: total_ram_mib(),
        }
    }
}

/// Total system RAM in MiB, parsed from `/proc/meminfo` `MemTotal`. Returns 0
/// when the field is unreadable — the number is contextual, not load-bearing
/// for any assertion, so an unreadable value degrades gracefully.
fn total_ram_mib() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("MemTotal:"))
                .and_then(|rest| rest.trim().strip_suffix("kB"))
                .and_then(|kb| kb.trim().parse::<u64>().ok())
        })
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

/// The measured tiers. Every field is optional so one report carries only the
/// tiers its subcommand produced.
#[derive(Debug, Default, Serialize)]
pub struct Tiers {
    /// The realistic quality tier (committed corpus): the ANN-vs-exact recall
    /// curve, plus the perf metrics (embed throughput, search QPS, propagate
    /// latency, peak RSS) still stubbed not-yet-measured. Populated by `arxiv`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arxiv: Option<ArxivTier>,
    /// The binding-memory tier: the streamed exact-search RSS proof and its
    /// negative control. Populated by `search-rss`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binding: Option<BindingTier>,
    /// The recall-vs-cost tier: how ANN recall and its build/query cost move as
    /// the HNSW knobs are swept. Populated by `recall-sweep`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_sweep: Option<RecallSweepTier>,
    /// The CPU-hermetic training tier: in-batch-negative fine-tune throughput
    /// (pairs/s) gated against a committed same-box baseline, plus the bounded
    /// (GradCache) vs unbounded (single-pass) activation-memory negative control.
    /// Populated by `train-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingTier>,
    /// One LoRA optimizer step over a synthetic batch: a leg of the
    /// `train-step` workload. Populated by `finetune-step`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finetune_step: Option<Leg<TrainStepPayload>>,
    /// A real training run over a committed pair table: a leg of the
    /// `train-run` workload. Populated by `finetune-run`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finetune_run: Option<Leg<TrainRunPayload>>,
    /// The CPU-hermetic conformal-coverage tier: the engine's split-conformal
    /// calibration drives a marginal coverage that is gated against a committed
    /// floor (`coverage_floor = measured − MARGIN`, the recall-floor idiom), one
    /// point per calibration-set size. Populated by `conformal-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conformal: Option<ConformalTier>,
    /// The CPU-hermetic eval-metric tier: the engine's retrieval / classification
    /// metric folds and the order-invariant bootstrap CI, each re-folded over a
    /// committed golden and gated against a committed value within a tolerance.
    /// Populated by `eval-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval: Option<EvalTier>,
    /// The CPU-hermetic propagation tier: the engine's `propagate_embeddings`
    /// (APPNP/SGC decoupled-GNN forward pass) over a committed synthetic
    /// graph+embedding fixture. Gated on the DETERMINISM contract — a committed
    /// digest of the propagated output vectors that is byte-identical across
    /// `target_partitions` ON THE SAME BOX (`f32` reduction order is not
    /// bit-identical across CPUs; this is a same-machine, not any-box,
    /// guarantee) — with propagation wall-time at named graph sizes riding
    /// along as an un-gated, machine-dependent reference. Populated by
    /// `propagate-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub propagate: Option<PropagateTier>,
    /// The CPU-hermetic graph fine-tune tier: the engine's biased-walk graph
    /// sampler (`GraphSampler`, the data path `fine_tune_graph` threads through)
    /// drives a sampled-pairs-per-second throughput gated against a committed
    /// same-box baseline by [`crate::rate_gate`], plus a committed-digest gate on
    /// the sampled pair set (the sampler is seeded, so the pairs are byte-stable —
    /// a regression in the walk bias, the negative mining, or the adjacency moves
    /// the digest). Populated by `graph-train-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_train: Option<GraphTrainTier>,
    /// The CPU-hermetic context-predictor tier: the engine's episodic
    /// meta-training (`sample_context_episodes` + `train_loop`) drives a
    /// training-throughput rate gated against a committed same-box baseline, and
    /// the serving path (`predict_with_context_predictor` over committed weights)
    /// carries a committed-digest gate on the predicted distribution with predict
    /// wall-time as an un-gated reference. Populated by `context-predictor-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_predictor: Option<ContextPredictorTier>,
    /// The CPU-hermetic model-inference tier: the engine's GPU-model serving
    /// verbs `generate_text_embeddings` (the `generate_embeddings` path) and
    /// `infer` (`Classification`), driven on `Device::Cpu` over tiny committed
    /// model bundles. Each lane gates a committed determinism DIGEST of the served
    /// output (the portable cell anchor) and a coarse same-box serving rate by
    /// [`crate::rate_gate`]. The rate is a code-path-regression net over the tiny
    /// model — NOT the full-scale scaling SLO, which is captured off-box in the
    /// cookbook (the A/B split). Populated by `model-inference-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_inference: Option<ModelInferenceTier>,
    /// The text-embedding serving path over a small corpus: a leg of the
    /// `encode` workload. Populated by `encode-step`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encode_step: Option<Leg<EncodePayload>>,
    /// The CPU-hermetic cache-hit SLO tier: the engine's opt-in producer
    /// memoization (`CachePolicy::Use`) on a cacheable producer (the
    /// neighbour-graph, anchored on an immutable `ResultDigest`). A cold `Use`
    /// build (nothing cached) is timed against a warm `Use` hit (the top-of-
    /// producer probe short-circuits the whole compute), and the gate asserts the
    /// hit clears a committed speed-up floor — the portable property of skipping
    /// the build, not the machine-dependent absolute wall-time. Populated by
    /// `cache-slo-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_slo: Option<CacheSloTier>,
    /// The CPU-hermetic recompute tier: the engine's `recompute(Downstream)`
    /// bounded topological sweep over a synthetic derived-table DAG (an embedding
    /// table → a neighbour-graph → a graph propagation). The gated property is the
    /// sweep's CORRECTNESS — every DAG node is recomputed exactly once, in
    /// topological (parent-before-child) order — a box-independent invariant; the
    /// sweep wall-time at the named DAG size rides along as an un-gated,
    /// machine-dependent reference. Populated by `recompute-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recompute: Option<RecomputeScaleTier>,
}

/// The k values the recall curve is reported at: recall@1, recall@10, recall@100.
///
/// A curve rather than a single scalar because ANN quality is k-dependent — a
/// graph index can nail the nearest neighbour (recall@1 high) yet thin out by
/// recall@100, or the reverse. One number hides that shape; the curve makes the
/// quality-vs-breadth trade visible and lets each k carry its own floor. These
/// are the k the committed ground-truth top-k is emitted at, so the recall path
/// can compute every point from one set of artifacts.
pub const RECALL_KS: [usize; 3] = [1, 10, 100];

/// The realistic quality tier — embed/search/recall/propagate over a committed
/// corpus. Every metric is a [`Measurement`] so an un-run metric is explicit.
#[derive(Debug, Serialize)]
pub struct ArxivTier {
    /// Embedding throughput, rows per second.
    pub embed_per_s: Measurement,
    /// Exact (brute-force) search throughput, queries per second.
    pub search_qps_exact: Measurement,
    /// ANN (sidecar-index) search throughput, queries per second.
    pub search_qps_ann: Measurement,
    /// Recall of ANN-over-frozen-index vs the exact ground truth, as a curve
    /// keyed by k. Each entry is the mean over the query set of
    /// `|ANN_topk ∩ EXACT_topk| / k` — a set-intersection fraction, so it is
    /// insensitive to within-top-k ordering. Keyed by k (1, 10, 100) so the
    /// quality-vs-breadth shape is explicit and each k can carry its own floor.
    /// A [`BTreeMap`] so the curve serializes in ascending k order.
    pub recall: BTreeMap<usize, Measurement>,
    /// Neighbor-graph propagation latency, milliseconds.
    pub propagate_latency_ms: Measurement,
    /// Process peak resident set, mebibytes.
    pub peak_rss_mib: Measurement,
}

impl ArxivTier {
    /// The tier with a measured recall curve and every perf metric still stubbed
    /// `not yet measured`.
    ///
    /// The recall lane is the portable, machine-independent gate (a fraction, not
    /// a rate), so it carries real datapoints from the first emit; the perf
    /// metrics (embed/search QPS, propagate latency, peak RSS) are rate/latency
    /// numbers measured on the emit box in a later PR, and stay explicit
    /// not-yet-measured markers until then rather than a zero a gate could
    /// mistake for a datapoint.
    pub fn with_recall(recall: BTreeMap<usize, Measurement>) -> Self {
        Self {
            embed_per_s: Measurement::not_yet_measured("rows_per_s"),
            search_qps_exact: Measurement::not_yet_measured("queries_per_s"),
            search_qps_ann: Measurement::not_yet_measured("queries_per_s"),
            recall,
            propagate_latency_ms: Measurement::not_yet_measured("ms"),
            peak_rss_mib: Measurement::not_yet_measured("mib"),
        }
    }
}

/// The recall-vs-cost tier: how ANN recall and its build/query cost move as the
/// HNSW knobs are swept, each point measured against the exact oracle over a
/// held-out query set.
///
/// Three axes, because the cost lifecycles differ. The **build** axis sweeps
/// the construction knobs (connectivity, build_expansion) — each point is a
/// *separately built* graph, so the cost is build time and on-disk size, and
/// recall here is an on-box reference (the swept graphs are not committed, so a
/// reader cannot re-derive it — it is not a portable gate). The **search** axis
/// sweeps `search_expansion` (ef_search) over ONE frozen graph re-dialed at query
/// time — recall rises and QPS falls as ef grows, and because it re-dials a
/// single committed index it *is* re-derivable, the portable recall-floor gate.
/// The **precision** axis sweeps `storage_precision` (and, for a quantized
/// precision, the retrieve→rescore `oversample`) — like the build axis, each
/// point is a separately built graph (quantization is baked in at construction,
/// unlike `search_expansion`), so it is an on-box reference; the portable,
/// COMMITTED recall floor for the shipped Int8 precision is asserted separately
/// against a frozen Int8 bundle (mirroring the search axis's frozen-graph
/// discipline) in `cargo test`.
#[derive(Debug, Serialize)]
pub struct RecallSweepTier {
    /// The USearch backend version the swept graphs were built/loaded with —
    /// recall and the graph format are backend-dependent, so the version travels
    /// with the curve and a reader rejects a cross-backend comparison.
    pub backend_version: &'static str,
    /// The corpus dimensionality every point was measured at.
    pub dim: usize,
    /// Corpus rows each graph was built over.
    pub corpus_rows: usize,
    /// Held-out queries each recall point averaged over.
    pub query_rows: usize,
    /// The build-knob axis (recall-vs-BUILD-cost): build time and index size rise
    /// with the knobs while recall holds above its floor. On-box reference.
    pub build_sweep: Vec<SweepPoint>,
    /// The search-knob axis (recall-vs-QUERY-cost): one frozen graph re-dialed at
    /// each `search_expansion`; recall rises and QPS falls as ef grows.
    pub search_sweep: Vec<SweepPoint>,
    /// The precision axis (recall-vs-STORAGE-cost): `F32` (exact, single-stage)
    /// vs `Int8` at the shipped default oversample (two-stage retrieve→rescore)
    /// vs `Int8` at `oversample = 1` (the naive quantized-graph-only baseline,
    /// no candidate widening for the rescore to recover from). The delta between
    /// the two `Int8` points is the recall the retrieve→rescore design recovers.
    pub precision_sweep: Vec<PrecisionSweepPoint>,
}

/// One swept point: the HNSW knobs and every cost/quality metric measured at
/// them. A metric that does not apply to a point's axis is an explicit
/// [`Measurement::not_yet_measured`] marker, never a zero.
#[derive(Debug, Serialize)]
pub struct SweepPoint {
    /// Max connections per graph node (HNSW *M*); `0` = backend default.
    pub connectivity: usize,
    /// Graph-construction candidate width (HNSW *ef_construction*); `0` = default.
    pub build_expansion: usize,
    /// Search candidate width (HNSW *ef_search*); `0` = backend default.
    pub search_expansion: usize,
    /// ANN-vs-exact recall@k over the held-out queries, set-intersection keyed by
    /// k — the same portable fraction the `arxiv` tier reports.
    pub recall: BTreeMap<usize, Measurement>,
    /// Wall-clock to build the graph over the corpus (build-knob axis only).
    pub build_time_ms: Measurement,
    /// Serialized graph size on disk (build-knob axis only).
    pub index_size_bytes: Measurement,
    /// ANN search throughput at k=10 over the held-out queries.
    pub search_qps: Measurement,
}

/// One precision-axis point: the storage precision (and, for a quantized
/// precision, the retrieve→rescore oversample) a graph was built at, and every
/// cost/quality metric measured over it.
#[derive(Debug, Serialize)]
pub struct PrecisionSweepPoint {
    /// The precision this point's graph stored its vectors at.
    pub precision: StoragePrecision,
    /// The retrieve→rescore oversample this point queried with. Irrelevant at
    /// `F32` (single-stage, no rescore) — carried anyway so the point is
    /// self-describing.
    pub oversample: usize,
    /// ANN-vs-exact recall@k over the held-out queries, set-intersection keyed
    /// by k — the same portable fraction the `arxiv` tier reports.
    pub recall: BTreeMap<usize, Measurement>,
    /// Wall-clock to build the graph over the corpus.
    pub build_time_ms: Measurement,
    /// Serialized graph size on disk (the `.usearch` file; a quantized
    /// precision's on-disk footprint also includes the `.rawf32` rescore
    /// companion, not counted here since it is not part of the in-RAM graph
    /// this axis measures).
    pub index_size_bytes: Measurement,
    /// ANN search throughput at k=10 over the held-out queries, through the
    /// same retrieve→rescore path the recall column measures.
    pub search_qps: Measurement,
}

/// The binding-memory tier: the bounded-RSS proof for streamed exact search.
///
/// Carries, at each measured corpus size, the streamed path's peak RSS and the
/// bench-only naive collect-all baseline's peak RSS (the negative control), and
/// the verdict of the flat-vs-linear assertion across the two sizes.
#[derive(Debug, Serialize)]
pub struct BindingTier {
    /// How peak RSS was sampled (allocator-internal stat vs process high-water).
    pub rss_source: RssSource,
    /// The corpus dimensionality every measurement used.
    pub dim: usize,
    /// One entry per corpus size, ascending in `rows`.
    pub points: Vec<RssPoint>,
    /// The verdict over `points`: streamed RSS flat, naive RSS grows ~linearly.
    pub assertion: RssAssertion,
}

/// How a peak-RSS number was obtained, recorded so a reader knows the
/// measurement's reliability.
///
/// This build does not register a jemalloc allocator, so the only available
/// source is the kernel's whole-process high-water mark. The enum names that
/// single source explicitly rather than leaving it implicit; were a future PR
/// to link `tikv-jemallocator`, it would add the allocator-resident variant and
/// the probe that selects it in the same change — the scaffold is not
/// pre-grown with a variant nothing can construct.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RssSource {
    /// `/proc/self/status` `VmHWM` — the whole-process high-water mark. The
    /// assertion relies on the *delta* between corpus sizes, which cancels the
    /// constant process baseline.
    ProcVmHwm,
}

/// One corpus size's measurements: the DataFusion scan-reader baseline, the
/// streamed search, and the naive baseline — each a fresh-process peak RSS.
///
/// The raw search peaks include the reader's own footprint; the *search
/// overhead* fields isolate each algorithm's accumulator as `search − scan`,
/// which is the quantity the bounded-memory claim is actually about.
#[derive(Debug, Serialize)]
pub struct RssPoint {
    /// Corpus size in rows.
    pub rows: usize,
    /// Peak RSS (MiB) of streaming the scan and dropping every batch unscored —
    /// DataFusion's parquet-reader baseline, owned by neither search.
    pub scan_only_rss_mib: f64,
    /// Peak RSS (MiB) of the streamed engine `exact_vector_search`.
    pub streamed_rss_mib: f64,
    /// Peak RSS (MiB) of the bench-only naive collect-all baseline.
    pub naive_rss_mib: f64,
    /// Streamed search's footprint above the scan baseline (`streamed − scan`) —
    /// the bounded accumulator's own resident cost.
    pub streamed_search_overhead_mib: f64,
    /// Naive baseline's footprint above the scan baseline (`naive − scan`) — the
    /// unbounded `O(N·d)` accumulator's own resident cost.
    pub naive_search_overhead_mib: f64,
    /// The k used for the search.
    pub k: usize,
}

/// The verdict of the bounded-RSS proof, over the *search overhead* (each
/// algorithm's footprint above the shared DataFusion scan baseline) between the
/// smallest and largest corpus.
#[derive(Debug, Serialize)]
pub struct RssAssertion {
    /// Whether the proof held: the streamed accumulator's overhead stayed flat
    /// as N grew AND the naive accumulator's overhead grew ~linearly in N.
    pub passed: bool,
    /// Streamed search-overhead delta (MiB) between the smallest and largest
    /// corpus — the bounded accumulator should not grow with N.
    pub streamed_overhead_delta_mib: f64,
    /// The ceiling (MiB) the streamed overhead delta had to stay under for
    /// "flat".
    pub streamed_flat_epsilon_mib: f64,
    /// Naive search-overhead delta (MiB) between the smallest and largest corpus
    /// — the unbounded accumulator should grow with N.
    pub naive_overhead_delta_mib: f64,
    /// The floor (MiB) the naive overhead delta had to exceed for "grows".
    pub naive_growth_floor_mib: f64,
    /// Naive overhead growth as a fraction of the model's predicted `N·d·4`
    /// growth — near 1.0 confirms the baseline scales with corpus size as theory
    /// says, which is what gives the negative control its teeth.
    pub naive_growth_vs_linear_ratio: f64,
    /// The scan baseline's own delta (MiB) between the two sizes, reported for
    /// transparency: it is the reader-side N-dependence the overhead subtraction
    /// removes from the search comparison.
    pub scan_baseline_delta_mib: f64,
    /// Human-readable summary of the verdict.
    pub detail: String,
}

pub use crate::leg::Measurement;

#[derive(Debug, Serialize)]
pub struct TrainingTier {
    /// The base-model hidden width the synthetic embeddings and projection head
    /// run at — the encoder activation per row scales with it, so it travels
    /// with the numbers.
    pub hidden_size: usize,
    /// The pair count the throughput was measured at (the largest in the OOM
    /// sweep, where the per-second rate is most stable).
    pub throughput_pairs: usize,
    /// In-batch-negative fine-tune throughput: pairs trained per second through
    /// one GradCache backward + optimizer step on `Device::Cpu`.
    pub pairs_per_s: Measurement,
    /// Wall-clock of the single measured GradCache epoch (one backward + step
    /// over `throughput_pairs` pairs), milliseconds.
    pub epoch_wall_ms: Measurement,
    /// The throughput rate-regression verdict: the measured `pairs_per_s` gated
    /// against the committed same-box baseline. Present only when the baseline
    /// was loaded; absent when the report is emitted without a baseline to gate
    /// against (the rate then rides as a bare measurement).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_gate: Option<RateVerdict>,
    /// The bounded-vs-growth activation-memory proof over ascending pair counts.
    pub oom: OomControl,
}

/// The throughput rate-regression verdict carried in the report: the measured
/// rate, the committed baseline it was gated against, the threshold applied, the
/// derived floor, and whether the gate held. Mirrors the in-code gate's fields
/// so the report records the full arithmetic, not a bare boolean.
#[derive(Debug, Serialize)]
pub struct RateVerdict {
    /// The measured rate the gate evaluated.
    pub measured_pairs_per_s: f64,
    /// The committed same-box baseline rate.
    pub baseline_pairs_per_s: f64,
    /// The relative-drop threshold applied.
    pub threshold: f64,
    /// The floor `baseline · (1 − threshold)` the measured rate had to clear.
    pub floor_pairs_per_s: f64,
    /// Whether the measured rate cleared the floor.
    pub passed: bool,
    /// Human-readable summary of the verdict with the full arithmetic.
    pub detail: String,
}

/// The activation-memory negative control: the GradCache (bounded) and
/// single-pass (unbounded) backward peak RSS at each ascending pair count, and
/// the verdict that the encoder activation graph GradCache removes is the
/// dominant growth term.
///
/// The single-pass backward holds every pair's encoder activation graph
/// (`O(n · depth · d)`) plus the `n × n` in-batch-negative similarity graph
/// alive until the one `backward()` returns; GradCache detaches each chunk's
/// representation and backprops one chunk's graph at a time, so it holds only
/// `O(chunk · depth · d)` of activations — but it still keeps the `O(n · d)`
/// representations and the `n × n` similarity graph, so its footprint is not
/// flat in `n`, just far smaller. The verdict is over the smallest-to-largest
/// delta: the single-pass delta must exceed the GradCache delta by a clear
/// separation margin (the activation graph GradCache removed) *and* itself clear
/// a growth floor.
#[derive(Debug, Serialize)]
pub struct OomControl {
    /// How peak RSS was sampled — the same whole-process high-water source the
    /// binding tier uses.
    pub rss_source: RssSource,
    /// One entry per pair count, ascending in `pairs`.
    pub points: Vec<OomPoint>,
    /// The verdict over `points`: GradCache RSS flat, single-pass RSS grows.
    pub assertion: OomAssertion,
}

/// One pair count's peak RSS for each backward path, each a fresh-process
/// high-water mark so an earlier larger run's `VmHWM` cannot contaminate it.
#[derive(Debug, Serialize)]
pub struct OomPoint {
    /// In-batch-negative pair count for this point.
    pub pairs: usize,
    /// Peak RSS (MiB) of the GradCache (chunked) backward + step over `pairs`
    /// pairs — the bounded path.
    pub gradcache_rss_mib: f64,
    /// Peak RSS (MiB) of the single-pass backward over `pairs` pairs — the
    /// unbounded negative control that keeps every encoder graph alive at once.
    pub single_pass_rss_mib: f64,
}

/// The verdict of the activation-memory proof, over the peak RSS delta between
/// the smallest and largest pair count.
#[derive(Debug, Serialize)]
pub struct OomAssertion {
    /// Whether the proof held: the single-pass RSS grew past the growth floor
    /// AND exceeded the GradCache RSS growth by the separation margin (the
    /// activation graph GradCache removed is the dominant growth term).
    pub passed: bool,
    /// GradCache peak-RSS delta (MiB) between the smallest and largest pair
    /// count — reps-plus-similarity growth, which GradCache does *not* remove.
    pub gradcache_delta_mib: f64,
    /// Single-pass peak-RSS delta (MiB) between the smallest and largest pair
    /// count — the full activation graph, which grows with `n`.
    pub single_pass_delta_mib: f64,
    /// The floor (MiB) the single-pass delta had to exceed for "grows".
    pub single_pass_growth_floor_mib: f64,
    /// The single-pass-minus-GradCache delta (MiB): the activation-graph growth
    /// GradCache's chunked re-encode kept off the resident set.
    pub activation_graph_separation_mib: f64,
    /// The margin (MiB) `activation_graph_separation_mib` had to exceed for the
    /// removed activation graph to count as the dominant growth.
    pub activation_graph_separation_floor_mib: f64,
    /// Human-readable summary of the verdict.
    pub detail: String,
}

/// The CPU-hermetic conformal-coverage tier: the marginal coverage the engine's
/// split-conformal calibration achieves over a committed calibration/test split,
/// gated against a committed floor.
///
/// Coverage is a *portable fraction* — the `⌈(n+1)(1-alpha)⌉` quantile and the
/// `1[score ≤ q̂]` coverage count are pure arithmetic over committed scores, so
/// any box re-derives the same number. It therefore carries a *real CI floor*
/// gate in the recall fraction's `measured ≥ floor` idiom (not the same-box rate
/// gate the GPU-bound tiers need): `coverage_floor = measured − MARGIN`, the
/// margin the headroom the guarantee has against a quantile-arithmetic drift
/// before the gate trips.
///
/// One point per calibration-set size: the set size is the *curve* (how the
/// finite-sample coverage tightens toward `1 − α` as `n` grows), the coverage is
/// the *gate* (each point must clear its committed floor, and the floor tracks
/// the `1 − α − ε` the guarantee promises).
#[derive(Debug, Serialize)]
pub struct ConformalTier {
    /// The nominal miscoverage level `α` the thresholds target — the guarantee is
    /// marginal coverage `≥ 1 − α`, so it travels with every point.
    pub alpha: f64,
    /// One coverage point per calibration-set size, ascending in `cal_rows`.
    pub points: Vec<ConformalPoint>,
}

/// One conformal coverage point: a calibration-set size and the marginal
/// coverage each score family achieved over the held-out test split at it, with
/// the committed floor each was gated against.
///
/// Three score families, one per verb the tier covers: LAC classification
/// (`conformalize`), absolute-residual regression (`conformalize_interval`), and
/// CQR regression (`conformalize_cqr`). Each is the engine's own
/// `ConformalModel` calibration scored by the engine's `coverage` /
/// `interval_coverage` over the *same* committed test split, so a regression in
/// any conformal code path moves the measured number here.
#[derive(Debug, Serialize)]
pub struct ConformalPoint {
    /// Calibration-set size this point calibrated the thresholds over.
    pub cal_rows: usize,
    /// Held-out test-set size the coverage was measured over.
    pub test_rows: usize,
    /// LAC-classification marginal coverage (`conformalize`): the fraction of
    /// test rows whose prediction set contained the true class.
    pub classification_coverage: CoverageGate,
    /// Absolute-residual marginal coverage (`conformalize_interval`): the
    /// fraction of test rows whose interval `[ŷ − q̂, ŷ + q̂]` contained `y`.
    pub absolute_residual_coverage: CoverageGate,
    /// CQR marginal coverage (`conformalize_cqr`): the fraction of test rows
    /// whose adaptive interval `[q_lo − q̂, q_hi + q̂]` contained `y`.
    pub cqr_coverage: CoverageGate,
}

/// One coverage measurement and the committed floor it was gated against — the
/// portable-fraction analogue of [`RateVerdict`], asserting `measured ≥ floor`
/// where `floor = committed_measured − MARGIN`.
#[derive(Debug, Serialize)]
pub struct CoverageGate {
    /// The marginal coverage measured this run, a fraction in `[0, 1]`.
    pub measured: Measurement,
    /// The committed floor `measured` must clear: the coverage measured on this
    /// same committed split minus the safety margin.
    pub floor: f64,
    /// Whether the gate held: `measured ≥ floor`.
    pub passed: bool,
}

/// The CPU-hermetic eval-metric tier: the engine's retrieval / classification
/// metric folds and the order-invariant bootstrap significance CI, each
/// re-folded over a committed golden and gated against a committed value within
/// a tolerance.
///
/// Every number is a *deterministic fold* of committed inputs through the
/// engine's own metric kernels — `RetrievalMetrics` (recall/MRR/nDCG),
/// `ClassificationMetrics` (accuracy/F1), and the seeded order-invariant
/// `bootstrap_ci` (the `eval_compare` significance interval). The committed
/// golden carries the value each fold produced when the golden was cut; the gate
/// asserts the re-fold lands within a tight tolerance of it (a fold is exact
/// arithmetic, so the tolerance is for f64 reassociation, not measurement noise)
/// — a regression in any metric kernel moves the re-folded number off the golden.
///
/// One point per eval-set size: the set size is the *curve*, the metrics are the
/// gated correctness numbers (they hold across sizes because the committed
/// golden is generated at each size).
#[derive(Debug, Serialize)]
pub struct EvalTier {
    /// The k cutoff the retrieval metrics were folded at.
    pub k: usize,
    /// One metric point per eval-set size, ascending in `query_rows`.
    pub points: Vec<EvalPoint>,
    /// The order-invariance verdict for the `eval_compare` bootstrap CI: the same
    /// per-query delta multiset in two different orders yields a byte-identical
    /// interval. Asserted once over the largest point's deltas.
    pub bootstrap_order_invariant: BootstrapDeterminism,
}

/// One eval point: an eval-set size and the metric folds the engine kernels
/// produced over the committed golden at it, each gated against its committed
/// value within a tolerance.
#[derive(Debug, Serialize)]
pub struct EvalPoint {
    /// Retrieval query count this point folded the recall/MRR/nDCG over
    /// (`eval_embeddings` / `eval_per_query`).
    pub query_rows: usize,
    /// Classification row count this point folded accuracy/F1 over
    /// (`eval_inference`).
    pub inference_rows: usize,
    /// Mean recall@k over the golden retrieval set.
    pub recall_at_k: MetricGate,
    /// Mean reciprocal rank over the golden retrieval set.
    pub mrr: MetricGate,
    /// Mean nDCG over the golden retrieval set.
    pub ndcg: MetricGate,
    /// Classification accuracy over the golden inference set.
    pub accuracy: MetricGate,
    /// Macro-F1 over the golden inference set.
    pub macro_f1: MetricGate,
}

/// One metric fold and the committed golden value it was gated against, asserting
/// `|measured − golden| ≤ tolerance`. Unlike a coverage floor (a one-sided `≥`),
/// a metric re-fold is exact arithmetic, so the gate is a two-sided tolerance
/// band catching any drift in either direction.
#[derive(Debug, Serialize)]
pub struct MetricGate {
    /// The metric value re-folded this run through the engine kernel.
    pub measured: Measurement,
    /// The committed golden value the fold must match within `tolerance`.
    pub golden: f64,
    /// The two-sided tolerance band: `|measured − golden| ≤ tolerance`.
    pub tolerance: f64,
    /// Whether the gate held.
    pub passed: bool,
}

/// The `eval_compare` bootstrap-CI determinism verdict: the seeded percentile
/// bootstrap is a function of the per-query delta *multiset*, not its order, so
/// the same deltas shuffled into a different order yield a byte-identical
/// interval. The verdict carries both intervals so a failure surfaces the
/// divergence, not just a boolean.
#[derive(Debug, Serialize)]
pub struct BootstrapDeterminism {
    /// Whether the two orderings produced a byte-identical `[lower, upper]`.
    pub passed: bool,
    /// The CI lower bound from the canonical-order resample.
    pub canonical_lower: f64,
    /// The CI upper bound from the canonical-order resample.
    pub canonical_upper: f64,
    /// The CI lower bound from the shuffled-order resample — equal to
    /// `canonical_lower` when the order-invariance holds.
    pub shuffled_lower: f64,
    /// The CI upper bound from the shuffled-order resample.
    pub shuffled_upper: f64,
    /// Human-readable summary of the verdict.
    pub detail: String,
}

/// The CPU-hermetic propagation tier: the engine's `propagate_embeddings`
/// (APPNP/SGC decoupled-GNN forward pass) folded over a committed synthetic
/// graph+embedding fixture, gated on the engine's documented *determinism
/// contract* and carrying propagation wall-time at named graph sizes as an
/// un-gated reference.
///
/// Two lanes, mirroring the harness's split between a portable gate and a
/// machine-dependent reference (the binding/training tiers' rate-vs-proof split,
/// and the recall tier's portable-fraction floor vs on-box cost):
///
/// * **The determinism gate** ([`digest`](PropagateTier::digest)) — the engine's
///   real contract is that `propagate_embeddings` is byte-identical across runs and
///   `target_partitions` *on a machine* (a fixed `(group, neighbour)` fold order in
///   `f64` with one final `f32` cast). The output is `f32`, so the exact bits are
///   NOT identical across CPUs (SIMD/FMA/BLAS reduction order differs), and a
///   committed cross-machine bit digest would be the wrong gate shape. So the
///   *portable* gate re-folds the committed fixture through the real engine twice on
///   the running box and asserts the two digests are equal to each other (a
///   [`DeterminismGate`]). A regression in the propagation math (the APPNP fold, the
///   `D̃^{-1/2}` degree normalisation, the hop count, the `α`-teleport) is caught by
///   the relative perturbation teeth in `cargo test`, which compare a perturbed fold
///   against the in-process baseline on the same box. The committed digest rides as
///   a documented same-box reference, never asserted for cross-machine equality.
/// * **The latency reference** ([`latencies`](PropagateTier::latencies)) —
///   propagation wall-time at named graph sizes. Machine-dependent, so it rides
///   as a [`Measurement`] reference only, NEVER a portable floor (the un-gated-rate
///   discipline: a wall-time is a property of the box, not the engine).
#[derive(Debug, Serialize)]
pub struct PropagateTier {
    /// The embedding dimensionality the fixture's `X⁽⁰⁾` and the propagated
    /// output live in — the digest is over `dim`-wide vectors, so it travels.
    pub dim: usize,
    /// Hops the gated fold ran (the APPNP depth the committed digest is over).
    pub hops: usize,
    /// The APPNP teleport probability `α` the gated fold ran with.
    pub alpha: f64,
    /// The neighbour-weighting the gated fold ran (the engine's
    /// `PropagationWeighting`, named so the digest's provenance is explicit).
    pub weighting: &'static str,
    /// The determinism gate: the digest of the propagated output vectors re-folded
    /// twice this run through the real engine on this box, asserting the two are
    /// equal (the same-machine byte-identity contract). The committed digest rides
    /// as a same-box reference, never asserted for cross-machine equality.
    pub digest: DeterminismGate,
    /// Propagation wall-time at each named graph size — an un-gated,
    /// machine-dependent reference curve, ascending in `nodes`.
    pub latencies: Vec<PropagateLatency>,
}

/// A committed-equality determinism gate: a stable checksum of an engine output,
/// re-folded this run and compared to the committed digest for byte-equality.
///
/// The digest is a real fold (a deterministic checksum over the output), never a
/// hand-written constant — the off-box rebuilder re-derives it from the same
/// fixture the gate re-folds. Asserting `measured == committed` is the analogue of
/// the eval tier's exact golden match, and is only the right relation when the
/// output is byte-identical *across machines* — i.e. the fold is over integer /
/// string bytes with no floating-point reduction (the graph-sampler pair set:
/// node ids and walk sequences over a seeded integer RNG). For `f32`-output folds,
/// cross-machine bit-identity does NOT hold (SIMD/FMA/BLAS reduction order varies
/// by CPU), so those tiers use [`DeterminismGate`] instead, which proves the real
/// (same-machine) contract.
#[derive(Debug, Serialize)]
pub struct DigestGate {
    /// The digest re-folded this run through the real engine path.
    pub measured: String,
    /// The committed digest the re-fold must equal — the same fold, recorded when
    /// the spec was cut.
    pub committed: String,
    /// Whether the gate held: `measured == committed`.
    pub passed: bool,
}

/// A same-machine determinism gate for an `f32`-output fold: the engine's real
/// contract is "byte-identical across runs and threads ON A MACHINE", not
/// cross-machine bit-identity (an `f32` reduction's exact bits depend on the CPU's
/// SIMD/FMA/BLAS reduction order, which varies by machine). So this gate proves the
/// real, portable property: it re-folds the same fixture through the same real
/// engine path twice on the running box and asserts the two digests are equal *to
/// each other*. That is true on any machine by construction, regardless of which
/// exact bits that machine produces.
///
/// The committed digest rides along as `committed_reference` — a documented
/// **same-box reference** (like the machine-dependent rate baselines), recorded
/// when the spec was cut on the rebuild box. It is reported so a human can compare
/// against the box the spec was cut on, but it is NEVER asserted for equality: a
/// different CI box producing different `f32` bits is expected, not a regression.
#[derive(Debug, Serialize)]
pub struct DeterminismGate {
    /// The digest the first same-machine fold produced this run.
    pub first: String,
    /// The digest the second same-machine fold produced this run. Equal to
    /// [`first`](DeterminismGate::first) iff the engine path is deterministic on
    /// this box — the gated property.
    pub second: String,
    /// The committed digest, recorded on the rebuild box when the spec was cut. A
    /// documented same-box reference only — reported for human comparison, never
    /// asserted for cross-machine equality.
    pub committed_reference: String,
    /// Whether the gate held: `first == second` (the same-machine determinism
    /// contract).
    pub passed: bool,
}

impl DeterminismGate {
    /// Build the gate from two same-machine folds and the committed reference.
    /// `passed` is `first == second` — the portable same-machine contract.
    pub fn new(first: String, second: String, committed_reference: String) -> Self {
        let passed = first == second;
        Self {
            first,
            second,
            committed_reference,
            passed,
        }
    }
}

/// One named graph size's propagation wall-time — an un-gated reference point.
///
/// Both the node count and the bounded fan-out travel with the time, because the
/// wall-time is a function of the edge set the engine folds, not the node count
/// alone. The latency is a [`Measurement`] so an un-run size is an explicit
/// not-yet-measured marker rather than a zero a reader could mistake for "instant".
#[derive(Debug, Serialize)]
pub struct PropagateLatency {
    /// Node count for this reference point.
    pub nodes: usize,
    /// Bounded fan-out (intra-class clique size minus one) each node wired at —
    /// the edge count, and so the fold cost, scales with it.
    pub fan_out: usize,
    /// Propagation wall-time over the whole `propagate_embeddings` call (load +
    /// fold + materialize), milliseconds. Machine-dependent reference, un-gated.
    pub propagate_ms: Measurement,
}

/// The CPU-hermetic graph fine-tune tier: the engine's biased-walk graph sampler
/// (`GraphSampler`, the data path `fine_tune_graph` threads through) measured for
/// throughput and gated for determinism.
///
/// Two lanes, the harness's portable-gate-vs-machine-dependent-rate split applied
/// to the graph-supervision data path:
///
/// * **Throughput** ([`pairs_per_s`](GraphTrainTier::pairs_per_s)) — the
///   `(anchor, positive, hard_negatives)` training rows the biased-walk sampler
///   draws per second over a committed synthetic graph. A *rate* (a property of
///   the box), so it is gated against a committed same-box baseline by
///   [`crate::rate_gate`], not a portable floor — the same discipline the
///   training tier's throughput follows.
/// * **The determinism digest** ([`digest`](GraphTrainTier::digest)) — the
///   sampler is seeded (a `SplitMix64` integer walk/negative stream), so the
///   sampled pair set is byte-stable across runs. The digest folds the sampled
///   rows' node ids / text bytes — integers and strings, NO floating-point
///   reduction — so it is byte-identical *across machines*, not merely same-box
///   (the biased-walk roulette is sequential scalar `f64` with no SIMD/FMA/BLAS, so
///   it too is IEEE-754 portable). This is why this tier gates the committed digest
///   for cross-machine equality (a [`DigestGate`]), unlike the `f32`-output tiers
///   (propagate, context-predictor, model-inference) whose bits float by CPU and so
///   gate same-machine determinism instead. A regression in the walk bias (`p`/`q`),
///   the structure-aware negative mining (the k-hop false-negative guard), or the
///   adjacency construction moves the rows and trips the gate. Licensed by the
///   sampler's documented seeded reproducibility (the engine's
///   `graph_spec_round_trip_resamples_identical_pairs` contract).
#[derive(Debug, Serialize)]
pub struct GraphTrainTier {
    /// The node count of the committed synthetic graph the throughput and the
    /// digest were measured over — the sampler cost scales with it, so it travels.
    pub nodes: usize,
    /// The edge count of the committed synthetic graph.
    pub edges: usize,
    /// The number of `(anchor, positive, [hard_negative])` rows the sampler drew
    /// from the committed graph — the digest is over these, and the throughput is
    /// this count divided by the sample wall-clock.
    pub sampled_pairs: usize,
    /// Sampled training rows drawn per second through one `GraphSampler::sample`
    /// over the committed graph, on the CPU. A *rate*, gated against a committed
    /// same-box baseline.
    pub pairs_per_s: Measurement,
    /// Wall-clock of the single measured `GraphSampler::sample` call,
    /// milliseconds.
    pub sample_wall_ms: Measurement,
    /// The throughput rate-regression verdict: the measured `pairs_per_s` gated
    /// against the committed same-box baseline. Present only when the baseline was
    /// loaded; absent when the rate rides as a bare measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_gate: Option<RateVerdict>,
    /// The determinism gate: the digest of the sampled pair set re-drawn this run
    /// through the real engine sampler, the committed digest, and the
    /// `re-drawn == committed` verdict.
    pub digest: DigestGate,
}

/// The CPU-hermetic context-predictor tier: the engine's episodic meta-training
/// and in-context serving, measured for throughput and gated for determinism.
///
/// Two lanes, the harness's split between a machine-dependent rate and a portable
/// gate, here spanning the *two* engine verbs the tier covers:
///
/// * **Training throughput** ([`train_pairs_per_s`](ContextPredictorTier::train_pairs_per_s))
///   — the meta-training episodes the engine's `sample_context_episodes` +
///   `train_loop` drive per second on the CPU. A *rate*, so it is gated against a
///   committed same-box baseline by [`crate::rate_gate`], the training tier's
///   discipline.
/// * **The predict determinism gate** ([`predict_digest`](ContextPredictorTier::predict_digest))
///   — `predict_with_context_predictor` is byte-deterministic given the served
///   weights and the target *on a machine* (the engine's inference-only no-gradient
///   contract). The predicted distribution is `f32`, so its exact bits are not
///   identical across CPUs; the gate re-predicts the committed targets over the
///   committed weight bundle twice on the running box and asserts the two digests
///   are equal to each other (a [`DeterminismGate`]). A regression in the
///   serve/predict path (context assembly, the in-context forward, the distribution
///   adapter, the de-standardisation) is caught by the relative perturbation teeth
///   in `cargo test` (a wrong `context_k` vs the in-process baseline). The committed
///   digest rides as a same-box reference, never asserted for cross-machine
///   equality. Predict wall-time rides as an un-gated, machine-dependent
///   [`Measurement`] reference — a latency is a property of the box, never a
///   portable floor.
#[derive(Debug, Serialize)]
pub struct ContextPredictorTier {
    /// The predictor architecture the committed weights were trained under
    /// (`Cnp` / `AttnCnp`), so the digest's provenance is explicit.
    pub architecture: &'static str,
    /// The context width `k` the committed predictor serves at — the digest is
    /// over a `k`-neighbour context, so a serve at a different `k` moves it.
    pub context_k: usize,
    /// The number of meta-training episodes the throughput was measured over.
    pub train_episodes: usize,
    /// Meta-training episode-steps per second through the engine's
    /// `train_context_predictor` (sample + `train_loop`) on the CPU. A *rate*,
    /// gated against a committed same-box baseline.
    pub train_pairs_per_s: Measurement,
    /// Wall-clock of the single measured meta-training run (the episodic
    /// `train_loop` over `train_episodes` episodes), milliseconds.
    pub train_wall_ms: Measurement,
    /// The training throughput rate-regression verdict against the committed
    /// same-box baseline. Present only when the baseline was loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_gate: Option<RateVerdict>,
    /// The predict determinism gate: the digest of the predicted distributions
    /// `predict_with_context_predictor` produced over the committed weight bundle
    /// and committed targets, re-folded twice this run on this box and asserted
    /// equal (the same-machine determinism contract). The committed digest rides as
    /// a same-box reference, never asserted for cross-machine equality.
    pub predict_digest: DeterminismGate,
    /// Predict wall-time over the committed target set — an un-gated,
    /// machine-dependent reference, never a portable floor.
    pub predict_latency_ms: Measurement,
}

/// The CPU-hermetic model-inference tier: the engine's GPU-model serving verbs
/// `generate_text_embeddings` (the `generate_embeddings` path) and `infer`
/// (`Classification`), driven on `Device::Cpu` over tiny committed model bundles,
/// measured for serving throughput and gated for determinism.
///
/// ## The A/B split this tier embodies
///
/// These are GPU-model inference rates. The representative full-scale rate — rows
/// per second through a production-size model on a GPU — is the scaling SLO and
/// is captured off-box in the cookbook **(A)**; it does NOT live here. This tier
/// is the CPU-hermetic gate **(B)**: it drives the *same engine verbs* over a tiny
/// committed bundle so the regression net runs in `cargo test` with no download.
///
/// Two lanes per verb, the harness's portable-gate-vs-machine-dependent-rate
/// split:
///
/// * **The determinism gate** ([`embed_digest`](ModelInferenceTier::embed_digest),
///   [`infer_digest`](ModelInferenceTier::infer_digest)) — the engine's serving
///   path is byte-deterministic on the CPU over a fixed model and fixed inputs *on
///   a machine*. The served output is `f32` (embedding vectors; score
///   distributions), so its exact bits are not identical across CPUs; each gate
///   re-serves twice on the running box and asserts the two digests are equal to
///   each other (a [`DeterminismGate`]). A regression in the resolve / tokenize /
///   forward / pool / adapt path is caught by the relative perturbation teeth in
///   `cargo test` (a different model / perturbed input vs the in-process baseline).
///   The committed digests ride as same-box references, never asserted for
///   cross-machine equality.
/// * **The serving throughput** ([`embed_rows_per_s`](ModelInferenceTier::embed_rows_per_s),
///   [`infer_rows_per_s`](ModelInferenceTier::infer_rows_per_s)) — rows/s the tiny
///   model serves through the real verb on this box, gated against a committed
///   same-box baseline by [`crate::rate_gate`]. This is a coarse
///   *code-path-regression* net (it catches lost batching, a per-row model
///   reload, a dropped fast path) — emphatically NOT the scaling SLO, which is the
///   cookbook (A) value over a real model on a real device.
#[derive(Debug, Serialize)]
pub struct ModelInferenceTier {
    /// The number of target rows the infer digest folded over — the embed digest
    /// folds the whole persisted vector column, so this is the infer fold width.
    pub targets: usize,
    /// Embed serving throughput: rows/s through the engine's real
    /// `generate_text_embeddings` over the tiny embed bundle on the CPU. A coarse
    /// same-box code-path net, NOT the scaling SLO.
    pub embed_rows_per_s: Measurement,
    /// Wall-clock of the single measured `generate_text_embeddings` call,
    /// milliseconds. Machine-dependent reference.
    pub embed_serve_ms: Measurement,
    /// The embed throughput rate-regression verdict against the committed same-box
    /// baseline. Present only when the baseline was loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_rate_gate: Option<RateVerdict>,
    /// The embed determinism gate: the digest of the persisted embedding vectors
    /// `generate_text_embeddings` produced over the corpus and the committed embed
    /// bundle, re-served twice this run on this box and asserted equal (the
    /// same-machine determinism contract). The committed digest rides as a same-box
    /// reference, never asserted for cross-machine equality.
    pub embed_digest: DeterminismGate,
    /// Infer serving throughput: rows/s through the engine's real `infer`
    /// (`Classification`) over the tiny classifier bundle on the CPU. A coarse
    /// same-box code-path net, NOT the scaling SLO.
    pub infer_rows_per_s: Measurement,
    /// Wall-clock of the single measured `infer` call, milliseconds.
    /// Machine-dependent reference.
    pub infer_serve_ms: Measurement,
    /// The infer throughput rate-regression verdict against the committed same-box
    /// baseline. Present only when the baseline was loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infer_rate_gate: Option<RateVerdict>,
    /// The infer determinism gate: the digest of the per-row score distributions
    /// `infer` produced over the committed targets and the committed classifier
    /// bundle, re-served twice this run on this box and asserted equal (the
    /// same-machine determinism contract). The committed digest rides as a same-box
    /// reference, never asserted for cross-machine equality.
    pub infer_digest: DeterminismGate,
}

use crate::leg::Payload;

/// One optimizer step over a synthetic batch: three encoder forwards on the
/// tape, a triplet loss, one backward into the adapter tensors, one AdamW
/// update. It measures the step's cost, never learning. The leg it sits in
/// carries the per-step series, the memory peaks and the dispatch counters
/// that prove its kernel arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStepPayload {
    /// The device the step resolved to (`cpu`, `cuda:0`).
    pub device: String,
    pub backbone_dtype: String,
    pub checkpoint_config_sha256: String,
    pub checkpoint_weights_sha256: String,
    pub checkpoint_weights_size_bytes: u64,
    pub seed: u64,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    pub margin: f64,
    pub target_modules: Vec<String>,
    pub batched_forward: bool,
    /// `None` is a step without gradient clipping — a value, never an
    /// unknown: every leg states which step it measured.
    pub max_grad_norm: Option<f32>,
    pub trainable_tensors: usize,
    pub warmup: usize,
    /// One real length per row; `[seq; batch]` for a dense batch.
    pub row_lengths: Vec<usize>,
    pub steps_measured: usize,
    /// The pre-update loss of each measured step's batch, in run order.
    pub losses: Vec<f32>,
    pub loss_first: f32,
    pub loss_last: f32,
    /// How many times the gradient clip ran: pre-step, warmup and measured.
    pub clip_invocations: u64,
    pub s_per_step_p50: Measurement,
    pub s_per_step_mean: Measurement,
    pub steps_per_s: Measurement,
    pub triplets_per_s: Measurement,
}

impl Payload for TrainStepPayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("seed", Nullable::NonNull),
        ("batch", Nullable::NonNull),
        ("seq", Nullable::NonNull),
        ("row_lengths", Nullable::NonNull),
        ("lora_rank", Nullable::NonNull),
        ("lora_alpha", Nullable::NonNull),
        ("lora_dropout", Nullable::NonNull),
        ("margin", Nullable::NonNull),
        ("target_modules", Nullable::NonNull),
        ("batched_forward", Nullable::NonNull),
        ("backbone_dtype", Nullable::NonNull),
        ("warmup", Nullable::NonNull),
        ("steps_measured", Nullable::NonNull),
        ("checkpoint_config_sha256", Nullable::NonNull),
        ("checkpoint_weights_sha256", Nullable::NonNull),
        ("checkpoint_weights_size_bytes", Nullable::NonNull),
        ("max_grad_norm", Nullable::NullMeans("no clip")),
    ];
}

/// A real training run: the trainer over a committed pair table for a fixed
/// number of epochs, evaluated on a committed held-out set. The leg it sits
/// in carries the held-out trajectory and final loss, the train-side probe
/// and the dispatch counters that prove its kernel arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRunPayload {
    pub seed: u64,
    pub task: String,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    pub lora_init: String,
    /// The triplet objective's margin; `None` under an objective without
    /// one.
    pub margin: Option<f64>,
    pub target_modules: Vec<String>,
    /// `None` adapts every layer.
    pub layers_to_transform: Option<Vec<usize>>,
    pub backbone_dtype: String,
    pub checkpoint_config_sha256: String,
    pub checkpoint_weights_sha256: String,
    pub checkpoint_weights_size_bytes: u64,
    /// `None` is a run without gradient clipping.
    pub max_grad_norm: Option<f64>,
    /// `None`: a real run has no warmup phase to discard.
    pub warmup: Option<usize>,
    /// `None`: rows are whatever the pair table holds.
    pub row_lengths: Option<Vec<usize>>,
    pub epochs: usize,
    pub lr: f64,
    pub schedule: String,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub grad_accum: usize,
    pub validation_fraction: f64,
    pub train_pairs_file_sha256: String,
    /// `None` on a text task.
    pub train_media_sha256: Option<String>,
    pub heldout_ids_sha256: String,
    pub heldout_pairs_sha256: String,
    /// `None` on a text task.
    pub heldout_media_sha256: Option<String>,
    /// Digest of the held-out set's batch partition: the loss is a mean
    /// over batches, so two runs must partition alike to be paired.
    pub heldout_batch_partition_sha256: String,
    pub embedding_loss: String,
    /// The contrastive objective's temperature; `None` under an objective
    /// without one.
    pub temperature: Option<f64>,
    pub matryoshka_dims: Vec<usize>,
    pub early_stopping_patience: usize,
    pub early_stopping_metric: String,
    pub eval_cadence: usize,
    /// The keys `--expect-kernels-disabled` named, sorted.
    pub kernels_disabled_expected: Vec<String>,
    /// How many times one training forward reaches each fusible seam,
    /// walked off the built tower: the `calls` of `fused + eager == calls ×
    /// forwards` for every counted family.
    pub fusible_site_census: jammi_encoders::FusibleSiteCensus,
    pub split_rule: String,
    pub batched_forward: bool,
    pub steps_measured: usize,
    pub rayon_pool_threads: usize,
    pub final_epoch: usize,
    pub held_out_count: usize,
    /// The trainer's own `final_loss` — a minimum over epochs, for
    /// comparison only; the leg's `held_out_example_mean` is the datum.
    pub final_loss_diagnostic: f64,
    /// Wall seconds inside the training loop, over every epoch.
    pub train_run_wall_s: f64,
    /// Wall seconds in the media front end; `None` on a text task.
    pub media_front_end_wall_s: Option<f64>,
}

impl Payload for TrainRunPayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("seed", Nullable::NonNull),
        ("task", Nullable::NonNull),
        ("batch", Nullable::NonNull),
        ("seq", Nullable::NonNull),
        ("lora_rank", Nullable::NonNull),
        ("lora_alpha", Nullable::NonNull),
        ("lora_dropout", Nullable::NonNull),
        ("lora_init", Nullable::NonNull),
        (
            "margin",
            Nullable::NullMeans("no triplet margin: another objective"),
        ),
        ("target_modules", Nullable::NonNull),
        (
            "layers_to_transform",
            Nullable::NullMeans("every layer is adapted"),
        ),
        ("backbone_dtype", Nullable::NonNull),
        ("checkpoint_config_sha256", Nullable::NonNull),
        ("checkpoint_weights_sha256", Nullable::NonNull),
        ("checkpoint_weights_size_bytes", Nullable::NonNull),
        ("max_grad_norm", Nullable::NullMeans("no clip")),
        ("warmup", Nullable::NullMeans("no warmup phase")),
        (
            "row_lengths",
            Nullable::NullMeans("rows as the pair table holds them"),
        ),
        ("epochs", Nullable::NonNull),
        ("lr", Nullable::NonNull),
        ("schedule", Nullable::NonNull),
        ("warmup_steps", Nullable::NonNull),
        ("weight_decay", Nullable::NonNull),
        ("grad_accum", Nullable::NonNull),
        ("validation_fraction", Nullable::NonNull),
        ("train_pairs_file_sha256", Nullable::NonNull),
        (
            "train_media_sha256",
            Nullable::NullMeans("text task: no media corpus"),
        ),
        ("heldout_ids_sha256", Nullable::NonNull),
        ("heldout_pairs_sha256", Nullable::NonNull),
        (
            "heldout_media_sha256",
            Nullable::NullMeans("text task: no media corpus"),
        ),
        ("heldout_batch_partition_sha256", Nullable::NonNull),
        ("embedding_loss", Nullable::NonNull),
        (
            "temperature",
            Nullable::NullMeans("no temperature: another objective"),
        ),
        ("matryoshka_dims", Nullable::NonNull),
        ("early_stopping_patience", Nullable::NonNull),
        ("early_stopping_metric", Nullable::NonNull),
        ("eval_cadence", Nullable::NonNull),
    ];
}

/// The engine's text-embedding serving path — resolve, tokenize, forward,
/// pool, normalize — over a small deterministic corpus and a committed
/// checkpoint. The leg it sits in carries the per-serve series, the memory
/// peaks and the digest of the vectors served.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodePayload {
    pub seed: u64,
    /// Rows served per call.
    pub batch: usize,
    /// The padded sequence length the tokenizer produced.
    pub seq: usize,
    /// One real length per row.
    pub row_lengths: Vec<usize>,
    pub compute_precision: String,
    pub checkpoint_config_sha256: String,
    pub checkpoint_weights_sha256: String,
    pub checkpoint_weights_size_bytes: u64,
    pub checkpoint_tokenizer_sha256: String,
    pub pooling: String,
    /// `None` when the checkpoint carries no pooling config of its own.
    pub checkpoint_pooling_sha256: Option<String>,
    pub normalize: bool,
    pub warmup: usize,
    pub iters_measured: usize,
    /// The device the caller asked for (`cpu`, `cuda:0`).
    pub device_requested: String,
    /// `None`: the serving path has no chunked-attention arm.
    pub chunk_size: Option<u64>,
    pub embed_rows_per_s: Measurement,
    pub embed_serve_ms: Measurement,
}

impl Payload for EncodePayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("seed", Nullable::NonNull),
        ("batch", Nullable::NonNull),
        ("seq", Nullable::NonNull),
        ("row_lengths", Nullable::NonNull),
        ("compute_precision", Nullable::NonNull),
        ("checkpoint_config_sha256", Nullable::NonNull),
        ("checkpoint_weights_sha256", Nullable::NonNull),
        ("checkpoint_weights_size_bytes", Nullable::NonNull),
        ("checkpoint_tokenizer_sha256", Nullable::NonNull),
        ("pooling", Nullable::NonNull),
        ("normalize", Nullable::NonNull),
        ("warmup", Nullable::NonNull),
        ("iters_measured", Nullable::NonNull),
        (
            "checkpoint_pooling_sha256",
            Nullable::NullMeans("no 1_Pooling/config.json in this model dir"),
        ),
        ("device_requested", Nullable::NonNull),
    ];
}

/// The CPU-hermetic cache-hit SLO tier: the engine's opt-in producer memoization
/// (`CachePolicy::Use`) measured on a genuinely cacheable producer — the
/// neighbour-graph, which anchors on the immutable source-table `ResultDigest`,
/// so the same build over the same parent is a sound reuse.
///
/// The gated property is the **speed-up**, not an absolute wall-time. A cold
/// `Use` build (nothing cached yet) does the whole compute; a warm `Use` over the
/// identical `(definition, source-digest)` short-circuits at the top-of-producer
/// probe — a catalog lookup plus an extant-bytes check — and skips the entire
/// build. The ratio `cold / warm` is a portable property of *skipping the work*
/// (a build that takes time vs a lookup that does not), so gating a committed
/// minimum speed-up is the right SLO: it has teeth (a probe that did not actually
/// short-circuit would fail it) yet does not pin a machine-dependent absolute.
#[derive(Debug, Serialize)]
pub struct CacheSloTier {
    /// Neighbourhood size `k` the gated graph is built at.
    pub k: usize,
    /// Cold `Use` build wall-time (nothing cached → the full compute),
    /// milliseconds. Machine-dependent reference; only its *ratio* to the hit is
    /// gated.
    pub cold_build_ms: Measurement,
    /// Warm `Use` hit wall-time (the top-of-producer probe short-circuits the
    /// whole build), milliseconds. Machine-dependent reference.
    pub warm_hit_ms: Measurement,
    /// The speed-up gate: the measured `cold / warm` ratio against the committed
    /// minimum, and whether it cleared.
    pub speedup: SpeedupGate,
}

/// The cache-hit speed-up verdict: the measured `cold / warm` ratio gated against
/// a committed minimum. Records the full arithmetic (both wall-times, the ratio,
/// the floor, the verdict), mirroring the rate gate's honesty — never a bare
/// boolean.
#[derive(Debug, Serialize)]
pub struct SpeedupGate {
    /// The cold `Use` build wall-time, milliseconds.
    pub cold_ms: f64,
    /// The warm `Use` hit wall-time, milliseconds.
    pub warm_ms: f64,
    /// The measured speed-up `cold_ms / warm_ms`.
    pub measured_speedup: f64,
    /// The committed minimum speed-up the hit had to clear.
    pub min_speedup: f64,
    /// Whether `measured_speedup >= min_speedup` — the cache hit really did
    /// short-circuit the build, by at least the committed margin.
    pub passed: bool,
}

impl SpeedupGate {
    /// Build the verdict from the two wall-times and the committed floor. A
    /// zero/negative warm time (a clock that did not advance) is treated as an
    /// unbounded speed-up — the hit was below the timer resolution, which clears
    /// any finite floor.
    pub fn new(cold_ms: f64, warm_ms: f64, min_speedup: f64) -> Self {
        let measured_speedup = if warm_ms > 0.0 {
            cold_ms / warm_ms
        } else {
            f64::INFINITY
        };
        Self {
            cold_ms,
            warm_ms,
            measured_speedup,
            min_speedup,
            passed: measured_speedup >= min_speedup,
        }
    }
}

/// The CPU-hermetic recompute tier: a `recompute(Downstream)` bounded topological
/// sweep over a synthetic derived-table DAG, measured at the committed node count.
///
/// The gated property is the sweep's **correctness**, not a wall-time: the
/// Downstream sweep must recompute every transitive dependent of the named table
/// exactly once, in topological order (a parent strictly before each child that
/// anchors on it). That invariant is box-independent — it is the engine's
/// contract — so gating it has teeth (a sweep that skipped a node, double-counted
/// a diamond descendant, or mis-ordered a parent/child would fail) without pinning
/// a machine-dependent absolute. The sweep wall-time rides along as an un-gated
/// reference, the discipline every scale tier's timing lane follows.
#[derive(Debug, Serialize)]
pub struct RecomputeScaleTier {
    /// Number of synthetic source nodes the DAG's embedding table holds.
    pub nodes: usize,
    /// The number of tables the Downstream sweep recomputed — the named table
    /// plus its transitive dependents.
    pub recomputed_count: usize,
    /// The number of tables the sweep was expected to recompute (the DAG's node
    /// count). The gate is `recomputed_count == expected_count`.
    pub expected_count: usize,
    /// Whether every recomputed table landed after all of its in-DAG parents — the
    /// topological-order invariant the sweep guarantees.
    pub topological_order_held: bool,
    /// Whether the correctness gate held: the right node count, each once, in
    /// topological order.
    pub passed: bool,
    /// The whole Downstream sweep's wall-time, milliseconds. Machine-dependent
    /// reference only — never gated.
    pub sweep_ms: Measurement,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leg::{Facts, Measured, Provenance};

    fn provenance() -> Provenance {
        Provenance {
            device_name: "cpu".into(),
            build_features: vec!["cuda".into()],
            flash_compiled: false,
            kernels_disabled_requested: vec![],
            kernels_disabled_fired: vec![],
            arm: "fused".into(),
            attention_arm: "fused".into(),
            mutant: Default::default(),
        }
    }

    fn train_step(max_grad_norm: Option<f32>) -> Leg<TrainStepPayload> {
        Leg::new(
            TrainStepPayload {
                device: "cpu".into(),
                backbone_dtype: "f32".into(),
                checkpoint_config_sha256: "c".into(),
                checkpoint_weights_sha256: "w".into(),
                checkpoint_weights_size_bytes: 10,
                seed: 42,
                batch: 2,
                seq: 6,
                lora_rank: 2,
                lora_alpha: 4.0,
                lora_dropout: 0.0,
                margin: 0.3,
                target_modules: vec!["Wqkv".into()],
                batched_forward: true,
                max_grad_norm,
                trainable_tensors: 2,
                warmup: 0,
                row_lengths: vec![6, 6],
                steps_measured: 1,
                losses: vec![0.5],
                loss_first: 0.5,
                loss_last: 0.5,
                clip_invocations: 0,
                s_per_step_p50: Measurement::measured(0.1, "s"),
                s_per_step_mean: Measurement::measured(0.1, "s"),
                steps_per_s: Measurement::measured(10.0, "steps/s"),
                triplets_per_s: Measurement::measured(20.0, "triplets/s"),
            },
            provenance(),
            Measured {
                iter_wall_s: Some(vec![0.1]),
                work: Some(2.0),
                ..Default::default()
            },
            Facts::default(),
        )
    }

    #[test]
    fn a_train_step_leg_states_its_clip_as_null_or_a_number_never_absent() {
        let off = train_step(None).to_value();
        assert!(off.get("max_grad_norm").is_some());
        assert!(off["max_grad_norm"].is_null());
        assert_eq!(train_step(Some(1.0)).to_value()["max_grad_norm"], 1.0);
    }

    #[test]
    fn a_train_step_leg_carries_identity_provenance_and_measurements_flat() {
        let value = train_step(None).to_value();
        for (field, _) in TrainStepPayload::IDENTITY_FIELDS {
            assert!(value.get(*field).is_some(), "{field}");
        }
        for field in [
            "device_name",
            "arm",
            "attention_arm",
            "flash_compiled",
            "iter_wall_s",
            "work",
        ] {
            assert!(value.get(field).is_some(), "{field}");
        }
    }

    #[test]
    fn a_train_run_legs_identity_is_the_thirty_seven_fields_of_the_run() {
        assert_eq!(TrainRunPayload::IDENTITY_FIELDS.len(), 37);
        let names: std::collections::BTreeSet<&str> = TrainRunPayload::IDENTITY_FIELDS
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(names.len(), 37);
        for provenance in ["arm", "device_name", "attention_arm", "mutant_id"] {
            assert!(
                !names.contains(provenance),
                "{provenance} is provenance, never identity"
            );
        }
    }

    #[test]
    fn report_identity_fields_are_emitted_under_provenance() {
        let value = serde_json::to_value(super::Provenance::baked()).unwrap();
        assert_identity_fields_present(&value, REPORT_IDENTITY_FIELDS);
    }

    #[test]
    #[should_panic(expected = "absent on this report")]
    fn an_absent_identity_field_panics_the_assertion() {
        assert_identity_fields_present(
            &serde_json::json!({"seed": 1}),
            &[("seed", Nullable::NonNull), ("batch", Nullable::NonNull)],
        );
    }

    #[test]
    #[should_panic(expected = "null")]
    fn a_null_non_null_identity_field_panics_the_assertion() {
        assert_identity_fields_present(
            &serde_json::json!({"seed": null}),
            &[("seed", Nullable::NonNull)],
        );
    }

    #[test]
    fn a_null_reading_is_accepted_where_null_means_something() {
        assert_identity_fields_present(
            &serde_json::json!({"margin": null}),
            &[("margin", Nullable::NullMeans("no clip"))],
        );
    }
}
