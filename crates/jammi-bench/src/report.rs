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

use serde::Serialize;

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
    /// `provenance` from this process's own identity — the ONE constructor
    /// unification contract C2.1 introduces to replace the 14 hand-written
    /// `Report { engine_version: ENGINE_VERSION, host: Host::detect(), .. }`
    /// literals `main.rs` carried before it (one place the three shared
    /// fields are filled, not fourteen that could individually drift).
    pub fn new(subcommand: &'static str, tiers: Tiers) -> Self {
        let report = Self {
            engine_version: ENGINE_VERSION,
            host: Host::detect(),
            subcommand,
            provenance: Provenance::baked(),
            tiers,
        };
        // K7-completeness, enforced on every real run — not only in a test
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
/// declared [`Nullable::NonNull`] — the RUNTIME enforcement half of the K7-
/// completeness identity consts ([`FinetuneStepTier::IDENTITY_FIELDS`],
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
            // Round-3 audit (advisory 5): an empty STRING is not JSON
            // `null`, so the check above alone let a NonNull field through
            // with no real content (`build.rs`'s own advisory A3 bug —
            // `TARGET`/`PROFILE` baking `""` — was closed at the SOURCE in
            // round 2, but nothing here would have caught a regression
            // back to that shape). A non-string NonNull field (a number,
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
/// `std::env::var`/`git` read (unification contract C1; `provenance_baked.rs`'s
/// `runtime_env_and_cwd_are_inert` is the mechanical proof: re-running the
/// same binary with a DIFFERENT `JAMMI_BUILD_SHA` in the environment, from a
/// fresh `cwd`, yields a byte-identical [`Provenance`]).
///
/// One instance per process, shared by every emitted [`Report`] (at
/// `report.provenance`) AND by the standalone `jammi-bench provenance`
/// subcommand (`main.rs`'s `run_provenance`), so a shell producer can read
/// this binary's identity BEFORE spending any wall-clock on a leg, rather
/// than discovering a stale binary only after the fact. Round-3 audit fix:
/// `stacked_sweep.sh`/`proof_artifact.py` do NOT read this today — the
/// producer-side cross-check (`provenance.build_sha == $SHA`, contract
/// C5.1) is phase 2, not built in this unit; this subcommand exists NOW so
/// that phase 2 has something to call.
///
/// The three fields [`REPORT_IDENTITY_FIELDS`] names — `build_sha`,
/// `target`, `profile` — are what lets a downstream K7-completeness reader
/// tell "these two legs ran the same code on the same target" apart from
/// "these two legs merely both filled in the same COMPARISON tuple".
/// Consumers: `check_cuda_run_artifacts.py`'s v2-leg identity walk (rule
/// (i), contract C6.3) parses this const straight out of this source file
/// (`build_identity_tuples`, "never hand-typed") as the provenance half of
/// the identity-key roster a v2 leg must carry; `ab_merge.py`'s leg-premise
/// check compares `FINETUNE_IDENTITY_FIELDS` only (contract C4.1's
/// comparison tuple, which does NOT include `build_sha`/`target`/`profile`
/// at all — those are Rust-only K7-completeness additions, not part of the
/// comparison tuple).
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
    /// The `Report` JSON shape version this struct's own introduction
    /// bumped to (unification contract C2.1): `2`.
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
/// `FinetuneStepTier::build_features` (a tier-level echo of this exact
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

/// Whether a K7-completeness identity field may legitimately serialize as
/// JSON `null`, and what that null MEANS when it does — never a bare
/// "absent"/"producer predates this field" reading (unification contract
/// P8/C3.3). A field with no [`NullMeans`](Nullable::NullMeans) entry is
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
    /// `FinetuneStepTier::IDENTITY_FIELDS`'s own `max_grad_norm` entry as of
    /// the merge with PR #381 (contract C3.4's own predicted rebase —
    /// "whichever of #381 and this phase lands second... adds the entry" —
    /// this is that moment).
    NullMeans(&'static str),
}

/// The three [`Provenance`] fields every tier-level `IDENTITY_FIELDS` const
/// appends (contract C3.1's "at the report level" half) — `build_sha`,
/// `target`, `profile`. Declared once here so
/// [`FinetuneStepTier::IDENTITY_FIELDS`] and
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
    /// The encoder fine-tune step tier: real LoRA step cost on the resolved
    /// device. Recorded, never gated. Populated by `finetune-step`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finetune_step: Option<FinetuneStepTier>,
    /// The finetune-run tier (unit 63, CONTRACT H4): one full (seed, arm)
    /// fine-tune run driving the REAL `TrainingLoopBuilder` and the public
    /// per-example held-out evaluation seam. Populated by `finetune-run`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finetune_run: Option<FinetuneRunTier>,
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
    /// digest of the propagated output vectors that any box re-derives — with
    /// propagation wall-time at named graph sizes riding along as an un-gated,
    /// machine-dependent reference. Populated by `propagate-scale`.
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
    /// The on-GPU throughput/latency observability tier: the embed
    /// (`generate_text_embeddings`) and classification (`infer`) serving verbs
    /// each measured on `gpu.device = 0`, tagged with the concrete device that
    /// served them. Records rows/s, p50/p99 tail latency, and cross-repeat
    /// determinism as measurements, not gates — an absolute rate on the ephemeral
    /// heterogeneous prove fleet is not a property of the code alone. The
    /// classification lane additionally hard-gates row conservation (a
    /// correctness property, not a perf one). The GPU peer of `model_inference`.
    /// Populated by `gpu-inference-scale` (behind the `cuda` feature /
    /// `live-gpu-tests` lane).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_inference: Option<GpuInferenceTier>,
    /// The identity-audited encode-step tier (unit 62, K7/E3): drives the
    /// engine's real `generate_text_embeddings` serving path over a small
    /// deterministic corpus and a fixture model dir with an EXPLICIT
    /// `1_Pooling/config.json`, folding the complete output-affecting
    /// parameter set into `EncodeStepTier::IDENTITY_FIELDS` so two legs are
    /// comparable only when every one of those fields agrees. See
    /// `EncodeStepTier`'s own doc for the full K7/esc-057 rationale.
    /// Populated by `encode-step`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encode_step: Option<EncodeStepTier>,
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

/// One measurement slot in the schema: a value carrying its unit, where a
/// `None` value is an explicit not-yet-measured marker.
///
/// The field is always present with its unit named, so the JSON shape is stable
/// from the first emit; a downstream gate reads `value: null` as "no datapoint"
/// and never mistakes an unrun metric for a zero. When a later PR measures the
/// metric it sets `value` — no schema change, no dead variant pre-grown.
#[derive(Debug, Serialize)]
pub struct Measurement {
    /// The measured value, or `null` when no run has produced it yet.
    pub value: Option<f64>,
    /// The unit the value is (or will be) expressed in.
    pub unit: &'static str,
}

impl Measurement {
    /// A not-yet-measured slot for a metric expressed in `unit`.
    pub fn not_yet_measured(unit: &'static str) -> Self {
        Self { value: None, unit }
    }

    /// A measured datapoint: `value` expressed in `unit`.
    pub fn measured(value: f64, unit: &'static str) -> Self {
        Self {
            value: Some(value),
            unit,
        }
    }
}

/// The CPU-hermetic training tier: how fast the engine's in-batch-negative
/// fine-tune primitive trains on this box, and the proof that the GradCache
/// (chunked) backward holds a bounded activation footprint while the single-pass
/// backward — which keeps every row's encoder graph alive at once — grows with
/// the pair count.
///
/// Two lanes, mirroring the binding tier's split between a measured rate and a
/// bounded-vs-growth proof:
///
/// * **Throughput** ([`pairs_per_s`](TrainingTier::pairs_per_s)) — pairs trained
///   per second through one GradCache backward + AdamW step over the largest
///   pair count, on `Device::Cpu`. A *rate*, so it is gated against a committed
///   same-box baseline by [`crate::rate_gate`], not a portable floor.
/// * **The OOM negative control** ([`oom`](TrainingTier::oom)) — the same
///   activation-memory cliff the binding tier's RSS proof has for search: the
///   single-pass backward's peak RSS grows with the pair count while GradCache's
///   stays flat. The verdict is observed live across ascending pair counts,
///   never asserted against a remembered constant.
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
    /// interval (engine #173). Asserted once over the largest point's deltas.
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

/// One verb's measured GPU lane: sustained throughput, tail latency, the
/// per-serve scored row count, and whether the serve was deterministic across
/// the measured repeats. Recorded observability, not a perf gate — see
/// [`GpuInferenceTier`] for why an absolute rate on this fleet cannot be a
/// committed baseline. `rows` is load-bearing for the one thing this lane
/// hard-gates: the classification lane asserts `rows` equals the corpus row
/// count on every serve (row conservation is correctness, not perf — a
/// per-row forward failure that `infer`'s annotate semantics silently drops
/// must not pass as a smaller-but-fine result).
#[derive(Debug, Serialize)]
pub struct GpuLane {
    /// Rows the serve scored — the same count every measured repeat produced
    /// (the embed lane's persisted-vector count; the infer lane's scored-row
    /// count, checked for conservation against the corpus size).
    pub rows: usize,
    /// Sustained serving throughput at the median serve, rows/s.
    pub rows_per_s: Measurement,
    /// Median per-serve wall latency, ms.
    pub p50_ms: Measurement,
    /// 99th-percentile per-serve wall latency, ms.
    pub p99_ms: Measurement,
    /// Whether every measured serve's digest matched the first — the on-device
    /// determinism contract across repeats.
    pub deterministic: bool,
}

/// The encoder fine-tune step tier: the cost of one real LoRA training step —
/// three encoder forwards on the tape at once, a triplet loss, one backward into
/// the adapter tensors, and one optimizer step.
///
/// Every field is **recorded**, never gated. A step time is a property of
/// `code x device x box`; the comparable quantity on a heterogeneous fleet is
/// the ratio between two runs on the *same* box, which is why the device and its
/// concrete sub-class are carried alongside every number.
#[derive(Debug, Serialize)]
pub struct FinetuneStepTier {
    /// The device the step ran on (`cpu` or `cuda:N`).
    pub device: String,
    /// The concrete device sub-class (e.g. `NVIDIA A100-SXM4-80GB`), so a
    /// recorded rate stays interpretable across a fleet that is not pinned.
    pub device_name: String,
    /// The precision the frozen backbone ran at.
    pub backbone_dtype: String,
    /// sha256 (hex) of `model_dir/config.json`'s raw bytes — round-4 audit
    /// fold-in on PR #372: the SAME base-checkpoint content-identity
    /// mechanism `grad_oracle.rs`'s `GradOracleReport` carries (see that
    /// module's doc's determinant table), added to THIS tier too so
    /// `ab_merge.py`'s leg-premise check can verify the jammi/torch legs of
    /// one A/B config loaded the byte-identical checkpoint, not merely a
    /// path string that happens to match. Computed via the SAME streaming
    /// `sha256_and_len` `grad_oracle.rs` reuses (never a second,
    /// independently-drifting hashing implementation).
    pub checkpoint_config_sha256: String,
    /// sha256 (hex) of `model_dir/model.safetensors`'s raw bytes.
    pub checkpoint_weights_sha256: String,
    /// `model_dir/model.safetensors`'s byte length — a cheap, redundant
    /// cross-check alongside the sha256 above.
    pub checkpoint_weights_size_bytes: u64,
    /// Drives the synthetic batch AND (when `--lora-init jammi`) the fresh
    /// LoRA draw — `ab_merge.py`'s own leg-premise check (the adjacent
    /// probe that folded this field in: `ci/scripts/perf/ab_merge.py` had
    /// NO premise-identity check at all before this round) reads this
    /// alongside `torch_finetune_step.py`'s `args.seed` to verify the
    /// jammi and torch legs of one A/B config actually ran the SAME batch,
    /// not merely that the sweep script PASSED them the same `--seed` flag
    /// (the difference matters the moment a leg is re-run by hand outside
    /// `finetune_ab.sh`'s own matched-flags convention).
    pub seed: u64,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    /// The LoRA scaling factor. Round-4 audit fold-in on PR #372: this input
    /// (`FinetuneStepParams::lora_alpha`) was ALREADY threaded through from
    /// the CLI but never actually emitted on this tier — `ab_merge.py`'s
    /// leg-premise check reads it alongside torch's `args.lora_alpha`
    /// (`torch_finetune_step.py`'s own report, one level up from this
    /// tier's own sub-block, same asymmetry `seed`'s own doc above
    /// describes).
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    /// The triplet-loss margin. jammi HARDCODES this to `0.3`
    /// (`finetune_step.rs`'s own `triplet_loss(&a, &p, &n, 0.3)` call site —
    /// there is no `--margin` CLI flag on this tier); torch's own
    /// `--margin` defaults to the SAME `0.3` but is independently
    /// overridable, so this field exists to let the leg-premise check
    /// catch an operator who overrode `--margin` on the torch leg only —
    /// the two legs would then be minimizing a DIFFERENT loss, not merely
    /// running different kernels.
    pub margin: f64,
    /// The LoRA target-module selectors, which decide how many linears carry an
    /// adapter — and therefore how much of the step is adapter work.
    pub target_modules: Vec<String>,
    /// Whether the three triplet groups were encoded in one forward (the
    /// trainer's behaviour) or three. On a dispatch-bound device this is the
    /// largest single term in the step, so it is recorded alongside every rate.
    pub batched_forward: bool,
    /// The `--max-grad-norm` this row ran with, or `null` when the flag was
    /// absent (no clipping — today's behaviour, bit-identical to before this
    /// field existed). Present so the step this row measured is unambiguous:
    /// the shipped trainer always calls
    /// [`jammi_ai::fine_tune::optimizer::clip_gradients`] at the default
    /// `max_grad_norm = 1.0`
    /// (`jammi_wire::fine_tune::FineTuneConfig::max_grad_norm`'s default), so
    /// a step measured with this field `null` is NOT the step the trainer
    /// runs — it is a distinct, useful reference point (the device-side
    /// clip's `4n + 4`-op cost isolated out), not an oversight. Deliberately
    /// NOT
    /// `#[serde(skip_serializing_if = "Option::is_none")]`: an omitted key
    /// reads as "this report predates the field", which is false — every
    /// `finetune-step` report from this build carries an opinion on
    /// clipping, so absence is always meaningful and always emitted as an
    /// explicit `null`, never folded away. See
    /// `finetune_step_tier_serializes_null_not_omitted_for_absent_max_grad_norm`
    /// below for the pinned schema shape.
    pub max_grad_norm: Option<f32>,
    /// Trainable tensor count. Zero would mean the selectors matched nothing and
    /// the measurement is of a frozen forward, so it is reported, not assumed.
    pub trainable_tensors: usize,
    /// Warmup iterations this run executed before the measured ones — an
    /// IDENTITY field (shared set, see `attention_arm`'s doc): `warmup`
    /// changes what [`clip_invocations`](Self::clip_invocations) counts
    /// (pre-step + warmup + measured), so two legs at different warmups
    /// are not comparable on that fact. torch emits it under `args`.
    pub warmup: usize,
    /// Per-row REAL (non-pad) lengths this leg fed the encoder -- an
    /// IDENTITY field (contract v4 §1 item 1, K7 audit: identity_fields.py's
    /// `FINETUNE_IDENTITY_FIELDS` grows 17 -> 18): two legs differing here
    /// ran the SAME `(batch, seq)` shape over a DIFFERENT padding structure
    /// -- a genuinely padded batch dispatches through
    /// `jammi_encoders::ModernBert::forward_with_lengths`'s trusted-lengths
    /// path P (the B3-padded transport, contract v4 §3.7), which a dense
    /// batch never reaches -- so the two rows' throughput/VRAM numbers are
    /// not comparable at all. `lengths.len() == batch`, each entry in
    /// `1..=seq`.
    ///
    /// DENSE-LEG VALUE (`FinetuneStepParams::row_lengths == None`, this
    /// tier's ORIGINAL behaviour, unchanged by this field's addition):
    /// `[seq; batch]` -- every row's real length equals `seq`, the SAME
    /// discriminator `jammi_encoders`' `CompactedBatch::is_dense` uses
    /// internally (`lengths.iter().all(|&l| l == seq)`), so a dense leg's
    /// `row_lengths` value is derivable from `batch`/`seq` alone and never
    /// disagrees with them.
    ///
    /// NEVER `null`: unlike [`max_grad_norm`](Self::max_grad_norm), this
    /// field carries no absent/off state -- both this producer and
    /// `torch_finetune_step.py` always emit a concrete vector (dense or
    /// padded), so it needs no `identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`
    /// entry.
    pub row_lengths: Vec<usize>,
    /// Measured steps after warmup.
    pub steps_measured: usize,
    /// Per-measured-step triplet loss, in step order, warmup EXCLUDED — one
    /// value per element of [`steps_measured`](Self::steps_measured), same
    /// length. Each entry is read once, from the same loss tensor
    /// `opt.step` for that iteration backpropagated through
    /// (`finetune_step.rs`'s existing post-`opt.step` `.to_scalar()` read,
    /// which exists to force the CUDA queue to completion before the clock
    /// stops — no second device-to-host read was added to get this field).
    /// Reading the tensor AFTER `opt.step` only decides when the host
    /// blocks; the loss value itself was computed by the forward BEFORE
    /// that step's optimizer update, so `losses[i]` is the PRE-update loss
    /// of measured step `i`'s batch, not a re-evaluation against the
    /// updated weights. `torch_finetune_step.py` reads its own loss at the
    /// mirror-image point for the identical reason, so the two stacks'
    /// trajectories share a placement convention — see that file's
    /// `_step_once` doc.
    ///
    /// This is cost-fixture data, not a quality result: the module doc's
    /// "Honesty about what is measured" section applies unchanged — token
    /// ids are synthetic and uniform, so a falling or rising trajectory
    /// here says nothing about learning quality, only that the forward /
    /// backward / optimizer path executed and produced finite numbers.
    /// Never quote this field as a quality result.
    ///
    /// PRECISION: this field is read in the BACKBONE dtype (`finetune_step.rs`'s
    /// `loss.to_dtype(DType::F32)` widens the STORAGE type for the D2H read,
    /// it never adds mantissa bits the upstream tensor did not have) —
    /// every real sweep leg runs `--backbone-dtype bf16`, whose 7 explicit
    /// mantissa bits give a ULP of `2^-9 ≈ 0.001953125` at a value near
    /// `0.30` (exponent bucket `[0.25, 0.5)`). Two adjacent recorded steps
    /// CAN legitimately repeat the exact same `f32` bit pattern even though
    /// the true (infinite-precision) loss moved, because the move landed
    /// inside that ULP — this is not a stuck optimizer, it is bf16
    /// quantization made visible. A caller printing more than ~3 decimal
    /// digits of a bf16-sourced entry here is displaying precision the
    /// dtype does not carry; see `ci/scripts/perf/ab_merge.py`'s `fmt_loss`
    /// for the table formatter that respects this.
    pub losses: Vec<f32>,
    /// `losses[0]`, carried as a scalar for table/summary use so a reader
    /// does not have to index into `losses` for the common case.
    pub loss_first: f32,
    /// `losses[losses.len() - 1]`.
    pub loss_last: f32,
    /// How many times `jammi_encoders`' bias-free training-mode LayerNorm
    /// actually dispatched the fused kernel (`jammi_kernels::ops::LayerNormFused`)
    /// during this run (warmup + measured steps) — a delta over the
    /// process-wide dispatch counters taken immediately before and after
    /// the step loop. This is the positive-proof channel a fused-vs-eager
    /// A/B needs: the step time alone cannot distinguish "the fused path
    /// ran and was fast" from "the fused path silently fell back and
    /// eager was fast anyway" (K2, scope decision 6 of the fused-kernels
    /// plan).
    pub ln_fused_dispatches: u64,
    /// How many times that same call site fell back to the eager
    /// (`slow()`) composition instead — outside the fused kernel's domain
    /// (dtype/contiguity/device/hidden), or because the admission
    /// predicate failed for any other stated reason. Non-zero here on a
    /// `ModernBert` bias-free training run is itself a signal worth
    /// reading, not just a complement of `ln_fused_dispatches`.
    pub ln_eager_dispatches: u64,
    /// How many times ModernBERT's training-mode fused RoPE (rotate-half)
    /// kernel (`jammi_kernels::ops::RopeFused`) actually dispatched during
    /// this run — the same positive-proof channel as `ln_fused_dispatches`,
    /// for the C3 fused-kernels commit (see `jammi_encoders::modernbert`'s
    /// `RotaryEmbedding` doc).
    pub rope_fused_dispatches: u64,
    /// How many times that same call site fell back to the eager
    /// (`RotaryEmbedding::apply`) composition instead — outside the fused
    /// kernel's domain (dtype/contiguity/device/head_dim), or because the
    /// admission predicate failed for any other stated reason.
    pub rope_eager_dispatches: u64,
    /// How many times ModernBERT's training-mode fused masked-softmax
    /// kernel (`jammi_kernels::ops::SoftmaxLastDimFused`) actually
    /// dispatched during this run — the same positive-proof channel as
    /// `ln_fused_dispatches` / `rope_fused_dispatches`, for the C4
    /// fused-kernels commit (see `jammi_encoders::modernbert`'s
    /// `softmax_apply_training` doc).
    pub softmax_fused_dispatches: u64,
    /// How many times that same call site fell back to the eager
    /// (`broadcast_add` + `candle_nn::ops::softmax`) composition instead —
    /// outside the fused kernel's domain (dtype/contiguity/device/rank/
    /// last-dim), or because the admission predicate failed for any other
    /// stated reason.
    pub softmax_eager_dispatches: u64,
    /// How many times ModernBERT's training-mode fused GeGLU kernel
    /// (`jammi_kernels::ops::GegluFused`) actually dispatched during this
    /// run — the same positive-proof channel as `ln_fused_dispatches` /
    /// `rope_fused_dispatches` / `softmax_fused_dispatches`, for the C5
    /// fused-kernels commit (see `jammi_encoders::modernbert`'s
    /// `geglu_apply_training` doc).
    pub geglu_fused_dispatches: u64,
    /// How many times that same call site fell back to the eager
    /// (`narrow`+`narrow`+`gelu_erf`+`mul`) composition instead — outside
    /// the fused kernel's domain (dtype/contiguity/device/even-last-dim),
    /// or because the admission predicate failed for any other stated
    /// reason.
    pub geglu_eager_dispatches: u64,
    /// How many times a LoRA-site fused epilogue
    /// (`jammi_kernels::ops::ScaledCastAdd`, `base_out + cast(lora_out *
    /// scaling)`) actually dispatched during this run — the same
    /// positive-proof channel as `ln_fused_dispatches` /
    /// `rope_fused_dispatches` / `softmax_fused_dispatches` /
    /// `geglu_fused_dispatches`, for the C6 fused-kernels commit (see
    /// `jammi_lora::LoraLinear::forward`'s doc). Read via
    /// `jammi_lora::lora_epilogue_dispatch_snapshot` rather than a
    /// `jammi_encoders`-side wrapper — the counters live in
    /// `jammi-kernels`' op-keyed registry, so any crate that knows the op
    /// name (`"lora_epilogue"`) reads the same table.
    pub lora_epilogue_fused_dispatches: u64,
    /// How many times that same call site fell back to the eager `[mul,
    /// cast, add]` composition instead — outside the fused kernel's
    /// domain (dtype/contiguity/device/shape), because `training ==
    /// false` (eval/serving never dispatches the fused kernel at all —
    /// see the doc above), or because the admission predicate failed for
    /// any other stated reason.
    pub lora_epilogue_eager_dispatches: u64,
    /// How many times the fused LoRA SITE
    /// (`jammi_kernels::ops::LowRankResidualLinear` — one `CustomOp3` covering
    /// the base matmul, dropout, both LoRA GEMMs, AND the epilogue, not
    /// just `ScaledCastAdd`'s standalone epilogue) actually dispatched
    /// during this run — the same positive-proof channel as
    /// `ln_fused_dispatches` / … / `lora_epilogue_fused_dispatches`. Read
    /// via `jammi_lora::lora_linear_fused_dispatch_snapshot`.
    /// `lora_epilogue_fused_dispatches`/`lora_epilogue_eager_dispatches`
    /// (above) are PERMANENTLY ZERO on a run where this field is nonzero:
    /// `LowRankResidualLinear` reuses `ScaledCastAdd`'s `cpu_fwd`/`cuda_fwd`
    /// directly as an internal step, never through the standalone
    /// epilogue's own `admit` call (see `jammi_lora::lora_epilogue_counters`'s
    /// doc) — that pairing going to `0` is the expected baseline, not a
    /// missing-dispatch regression.
    pub lora_linear_fused_dispatches: u64,
    /// How many times that same call site fell back to the eager
    /// composition (`[base matmul, dropout, A-matmul, B-matmul, mul,
    /// cast, add]`) instead — outside the fused kernel's domain
    /// (bias-carrying base, unsupported dtype/device, non-contiguous
    /// view, unsupported rank) DURING TRAINING. `training == false`
    /// (eval/serving) does NOT increment this field: `LoraLinear::forward`
    /// returns through its own always-eager composition BEFORE ever
    /// calling `admit` (the thing that increments either counter), so an
    /// eval-only run leaves BOTH `lora_linear_fused_dispatches` and this
    /// field at `0` — not evidence the eager arm ran, just evidence
    /// neither counter was ever touched.
    pub lora_linear_eager_dispatches: u64,
    /// How many times ModernBERT's training-mode fused whole-attention-block
    /// kernel (`jammi_kernels::ops::AttentionBlockFused`) actually
    /// dispatched during this run — the same positive-proof channel as
    /// `ln_fused_dispatches` / `rope_fused_dispatches` /
    /// `softmax_fused_dispatches` / `geglu_fused_dispatches` /
    /// `lora_epilogue_fused_dispatches`, for the fused attention-block
    /// commit (see `jammi_encoders::modernbert`'s
    /// `ModernBertAttention::forward_training_attention` doc). Read via
    /// `jammi_encoders::attention_block_dispatch_snapshot`, mirroring the
    /// sibling counters' read API exactly.
    pub attention_block_fused_dispatches: u64,
    /// How many times that same call site fell back to the eager
    /// (`apply1`/`apply2`/`apply3` composed masked-softmax attention)
    /// path instead — outside the fused kernel's domain
    /// (dtype/contiguity/device/`seq`/mask-shape), or because `head_dim`
    /// is not exactly the fused kernel's fixed domain (64 —
    /// `jammi_kernels::ops::ATTENTION_BLOCK_HEAD_DIM`). On a checkpoint
    /// whose `head_dim != 64` this is the only path admitted, so the pair
    /// reads `0` fused / `N` eager (`N` = the number of attention calls
    /// the step made) — that is the predicate refusing by domain, not a
    /// broken counter.
    pub attention_block_eager_dispatches: u64,
    /// How many times [`jammi_ai::fine_tune::adamw::AdamW::step`]'s
    /// per-`Var` dispatch actually admitted the fused multi-tensor kernel
    /// (`jammi_kernels::ops::adamw_step_fused_t`, three launches — two
    /// `InplaceOp2` calls (EMA the first/second moment) then one
    /// `InplaceOp3` call (bias-correct, decoupled weight decay, and the
    /// adaptive update), all in place, zero `Var::set`/memcpy) during this
    /// run — the same positive-proof channel as `ln_fused_dispatches` /
    /// … / `attention_block_fused_dispatches`, for the multi-tensor AdamW
    /// commit. Read via
    /// `jammi_kernels::admission::counters_for("adamw_step_fused")`,
    /// mirroring the sibling counters' read API exactly — `"adamw_step_fused"`
    /// is also the op key a caller names in `JAMMI_KERNELS_DISABLE` to force
    /// every `Var` this run onto the eager arm below.
    pub adamw_fused_dispatches: u64,
    /// How many times that same per-`Var` dispatch fell back to the eager
    /// candle-op chain instead — outside the fused kernel's domain
    /// (device/dtype/contiguity/shape agreement across
    /// `theta`/`m`/`v`/`grad`), or because `JAMMI_KERNELS_DISABLE` named
    /// `"adamw_step_fused"` for this process.
    pub adamw_eager_dispatches: u64,
    /// How many times this run invoked the PRODUCTION
    /// [`jammi_ai::fine_tune::optimizer::clip_gradients`] — a before/after
    /// delta over `finetune_step.rs`'s process-wide `CLIP_INVOCATIONS`
    /// counter taken around `run()`'s pre-step + warmup + measured loop, so
    /// it reads `warmup + steps + 1` on a clip-on row and exactly `0` on a
    /// clip-off one. The COUNTED fact behind [`max_grad_norm`](Self::max_grad_norm)
    /// (which only echoes what was REQUESTED), emitted next to the fused-
    /// dispatch deltas above for the same reason they exist: a row's claim
    /// about what ran is a number a merge stage can check, not a log line
    /// an operator trusts. `torch_finetune_step.py` emits its twin
    /// (`finetune_step.clip_invocations`, counting `clip_grad_norm_`
    /// calls over the identical window); `ci/scripts/perf/ab_merge.py`'s
    /// `clip_fact_violations` refuses a leg whose request and count
    /// disagree in kind.
    pub clip_invocations: u64,
    /// The attention REFERENCE CLASS the operator ASKED this run to measure
    /// — `"eager"` iff an attention base (`attention_block`,
    /// `attention_block_flash`, or the `"all"` wildcard) is in
    /// [`kernels_disabled_requested`](Self::kernels_disabled_requested),
    /// else `"fused"` (jammi has no `--attn` lever; `JAMMI_KERNELS_DISABLE`
    /// is the lever). A member of the SHARED jammi/torch identity set
    /// (`ci/scripts/perf/identity_fields.py`'s `FINETUNE_IDENTITY_FIELDS`,
    /// whose entry carries the full rationale): `torch_finetune_step.py`
    /// emits `"eager"` for a resolved `eager` implementation and `"fused"`
    /// for `sdpa` (and every other HF fused-kernel implementation), so
    /// `ab_merge.py`'s leg-premise check refuses a jammi-eager ↔ torch-sdpa
    /// pairing — the "two references, never mixed" rule as a CHECKED
    /// premise. Deliberately NOT derived from the
    /// `attention_block_*_dispatches` deltas above: those read eager on a
    /// by-design DOMAIN decline (`head_dim != 64`, `seq > 4096`, dtype /
    /// contiguity / mask arms — see `attention_block_eager_dispatches`'s
    /// own doc), which is a measurement about the checkpoint, not a
    /// premise; whether the fused arm actually ran is `fused_proof`'s and
    /// the counters' job. See `finetune_step.rs`'s `attention_arm`.
    pub attention_arm: String,
    /// How many times the FlashAttention-2 DENSE cascade
    /// (`attention_block_flash`, P6 Stage B B3-dense) actually dispatched
    /// `Fused` — a THIRD training-attention arm, separate from
    /// `attention_block_fused_dispatches` (the BLOCK arm's own counter):
    /// when this arm fires for a layer, the block arm's own `admit` call
    /// for that SAME layer is never reached at all (an early return, see
    /// `jammi_encoders::modernbert`'s
    /// `ModernBertAttention::forward_training_attention` doc) — so a run
    /// where flash fires on every layer reads `attention_block_fused_
    /// dispatches == 0` and `attention_block_flash_fused_dispatches ==
    /// num_hidden_layers * forwards_per_step * steps_measured`, not a
    /// contradiction. Read via
    /// `jammi_encoders::attention_block_flash_dispatch_snapshot`.
    pub attention_block_flash_fused_dispatches: u64,
    /// How many times that cascade DECLINED (a domain miss -- e.g. real
    /// padding, out of this unit's dense-only scope -- or a capability
    /// miss -- not CUDA, `flash-attn` not compiled, wrong arch, or named
    /// in `JAMMI_KERNELS_DISABLE`) instead of dispatching `Fused`. A
    /// VALID flash-arm timing leg must read `0` here (contract v5 §3.8:
    /// "`declined > 0` on any bench leg -> INVALID -- bench masks are
    /// prefix by construction").
    pub attention_block_flash_declined_dispatches: u64,
    /// Whether THIS BUILD compiled the vendored FlashAttention-2 kernels
    /// (`jammi_kernels::admission::FLASH_COMPILED`) -- always present so a
    /// `attention_block_flash_fused_dispatches == 0` reading is
    /// distinguishable between "this build cannot run flash at all" and
    /// "flash was compiled in but declined/disabled this run".
    pub flash_compiled: bool,
    /// This tier's own echo of [`Provenance::build_features`] — sorted,
    /// deduplicated linked-crate feature names this binary was compiled
    /// with (`crate::report::build_features`, the SAME function
    /// `Provenance::baked` calls, never a second copy). Carried directly on
    /// the tier (not only via the wrapping `Report.provenance`) because a
    /// STACKED/raw leg (unification contract C7 row 3a) IS this tier's own
    /// JSON sub-object with a stamp — it has no `report.provenance` wrapper
    /// to fall back on, so the tier stays self-describing on its own.
    pub build_features: Vec<&'static str>,
    /// The `JAMMI_KERNELS_DISABLE` op keys this process REQUESTED (sorted,
    /// empty when the env var was unset or empty) —
    /// `jammi_kernels::admission::disabled_ops_requested`. Always present,
    /// even on an ordinary run with nothing disabled: an omitted key would
    /// read as "this report predates the field", which is false.
    pub kernels_disabled_requested: Vec<String>,
    /// The `JAMMI_KERNELS_DISABLE` op keys that actually FIRED (disabled at
    /// least one live dispatch) this run (sorted) —
    /// `jammi_kernels::admission::disabled_ops_fired`. A run whose intended
    /// `JAMMI_KERNELS_DISABLE` never reached this process at all (a
    /// var-NAME typo, an unforwarded ssh/`docker -e` environment) reads
    /// BOTH this field and `kernels_disabled_requested` as `[]` —
    /// indistinguishable, on this pair alone, from a run that genuinely
    /// requested nothing. This is the field a downstream A/B harness
    /// compares against its OWN recorded intent (the op key(s) it meant to
    /// pass) to catch that drop: a non-empty intended request paired with
    /// an empty `kernels_disabled_requested` here is exactly the failure
    /// mode this pair exists to make visible — the eager leg of a
    /// forced-eager A/B silently measuring the fused arm instead. See
    /// `jammi_kernels::admission`'s module doc's "safety property" section
    /// for the separate, narrower guarantee `run` already hard-errors on
    /// (an entry that WAS requested but never fired).
    pub kernels_disabled_fired: Vec<String>,
    pub s_per_step_p50: Measurement,
    pub s_per_step_mean: Measurement,
    pub steps_per_s: Measurement,
    pub triplets_per_s: Measurement,
    /// Peak resident set. Absent off Linux rather than faked.
    pub peak_rss_bytes: Measurement,
    /// Peak device memory growth during the measured steps, over a baseline read
    /// AFTER the model and optimizer are resident but BEFORE the untimed
    /// pre-step `finetune_step::run` takes (see that function's doc comment
    /// on `vram_baseline` for the full reasoning).
    ///
    /// This ordering matters because the underlying sample
    /// (`nvidia-smi --query-gpu=memory.used`) is a DRIVER-level allocator
    /// POOL high-water mark, not live-allocated bytes: once the pool grows
    /// to admit a tensor it does not shrink back down between steps (the
    /// same convention `crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-
    /// p1-softmax-fold-bf8e807-a100-sxm4.json` (unification contract C8: this
    /// RECORD moved here from its original pre-schema location under
    /// `crates/jammi-bench/`) reasons about in 32 MiB pool blocks). A
    /// baseline read AFTER the pre-step
    /// would already sit at (or near) the run's own high-water mark, and
    /// the peak-minus-baseline subtraction would then floor at (or near)
    /// zero regardless of how much the run actually allocates. So this is
    /// activation and workspace growth: it deliberately excludes the
    /// backbone weights and the optimizer moments, because those are constant
    /// for a configuration and would mask the term that actually moves. It is
    /// device-total minus that baseline rather than a per-process figure — exact
    /// on a dedicated pod, an over-report on a shared GPU. Absent when
    /// `nvidia-smi` is not present.
    pub peak_vram_bytes: Measurement,
}

impl FinetuneStepTier {
    /// K7-completeness: every field a downstream leg-premise check needs to
    /// establish that two `finetune-step` legs measured the "same" thing —
    /// a STRICT SUPERSET of `ci/scripts/perf/identity_fields.py`'s own
    /// `FINETUNE_IDENTITY_FIELDS` COMPARISON tuple -- 17 entries as of PR
    /// #381 landing `max_grad_norm`/`attention_arm`/`warmup`, moved out of
    /// `ab_merge.py` into the shared `identity_fields.py` module that PR
    /// (unification contract C4.1: the COMPARISON tuple itself is not grown
    /// BY THIS UNIT; #381 is independent upstream work with its own reason
    /// to grow it, and contract C3.4 explicitly names this rebase: "whichever
    /// of #381 and this phase lands second rebases and adds the entry"),
    /// growing to 18 with this unit's OWN `row_lengths` entry (contract v4
    /// §1 item 1, K7 audit -- `identity_fields.py`'s own tuple grows to
    /// match in a companion docs-ci change; this const is the SUPERSET side
    /// of the subset check either way, so it is correct to carry the 18th
    /// name here regardless of which side lands first).
    /// The 18 comparison entries, plus five K7-completeness additions the
    /// comparison tuple omits BY DESIGN (provenance never compared
    /// cross-producer — see `ab_merge.py:47-55`'s provenance rows): `device_name`,
    /// `kernels_disabled_requested`, `kernels_disabled_fired`,
    /// `flash_compiled`, `build_features`.
    ///
    /// `max_grad_norm` is `NullMeans("no clip")` — `None`/`null` means the
    /// step ran with clipping OFF, a legitimate, declared value (mirrors
    /// `identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`'s own framing on
    /// the Python side), never "this producer predates the field".
    /// `attention_arm`/`warmup` are `NonNull` — see `FinetuneStepTier`'s own
    /// field docs for what each records.
    ///
    /// `ci/scripts/perf/test_identity_fields_subset.py` (contract C4.2)
    /// parses `FINETUNE_IDENTITY_FIELDS` out of `identity_fields.py` (via
    /// `ab_merge`'s re-export) and this const out of this file's own source
    /// and asserts the Python tuple is a SUBSET; `finetune_step_identity_
    /// fields_are_emitted` (below) asserts every field named here is
    /// actually present on a real, serialized tier; `finetune_step_tier_
    /// emits_every_shared_identity_field` (PR #381 audit B1) is the SAME
    /// check run the other direction — straight off `identity_fields.py`'s
    /// own tuple, independent of this const — so a future drift between
    /// the two Rust-side mechanisms cannot both silently agree on the wrong
    /// thing.
    pub const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        // The 18 entries `identity_fields.py::FINETUNE_IDENTITY_FIELDS`
        // compares (17 as of PR #381, plus this unit's `row_lengths`).
        ("seed", Nullable::NonNull),
        ("batch", Nullable::NonNull),
        ("seq", Nullable::NonNull),
        ("lora_rank", Nullable::NonNull),
        ("lora_alpha", Nullable::NonNull),
        ("lora_dropout", Nullable::NonNull),
        ("margin", Nullable::NonNull),
        ("target_modules", Nullable::NonNull),
        ("batched_forward", Nullable::NonNull),
        ("backbone_dtype", Nullable::NonNull),
        ("steps_measured", Nullable::NonNull),
        ("checkpoint_config_sha256", Nullable::NonNull),
        ("checkpoint_weights_sha256", Nullable::NonNull),
        ("checkpoint_weights_size_bytes", Nullable::NonNull),
        ("max_grad_norm", Nullable::NullMeans("no clip")),
        ("attention_arm", Nullable::NonNull),
        ("warmup", Nullable::NonNull),
        // NEVER null (see this field's own doc) -- both producers always
        // emit a concrete vector, dense or padded.
        ("row_lengths", Nullable::NonNull),
        // K7-completeness additions beyond the comparison tuple.
        ("device_name", Nullable::NonNull),
        ("kernels_disabled_requested", Nullable::NonNull),
        ("kernels_disabled_fired", Nullable::NonNull),
        ("flash_compiled", Nullable::NonNull),
        ("build_features", Nullable::NonNull),
    ];
}

/// One epoch's held-out example-mean loss point, recorded during
/// [`crate::finetune_run`]'s resume-cycled multi-epoch drive (CONTRACT H4:
/// "full per-epoch trajectory recorded").
#[derive(Debug, Serialize)]
pub struct EpochHeldOut {
    /// 0-based epoch index this point was measured after.
    pub epoch: usize,
    /// `HeldOutLoss::mean` — the example-mean over the committed held-out
    /// fixture, measured via `TrainingLoop::evaluate_held_out` immediately
    /// after this epoch's `run()` leg returned.
    pub held_out_mean: f64,
    /// `HeldOutLoss::tie_fraction` at this epoch.
    pub held_out_tie_fraction: f64,
    /// `HeldOutLoss::batch_partition_sha256` at this epoch — recorded per
    /// point (not only at the final epoch) because the partition is
    /// deterministic from `(held-out ids, batch_size)` alone and should
    /// therefore read IDENTICAL at every point; a divergence here across
    /// epochs is itself a finding (the held-out id list or batch size
    /// silently changed mid-run).
    pub held_out_batch_partition_sha256: String,
}

/// The finetune-run tier (unit 63, CONTRACT H4 / PLAN (d)): one full
/// (seed, arm) fine-tune run driving the REAL
/// [`jammi_ai::fine_tune::trainer::TrainingLoopBuilder`] and the public
/// per-example held-out evaluation seam
/// ([`jammi_ai::fine_tune::trainer::TrainingLoop::evaluate_held_out`], unit
/// 63 H1) — see [`crate::finetune_run`]'s module doc for the full design
/// (resume-cycled per-epoch trajectory, arm-as-provenance, the disjoint
/// held-out/train-split convention).
///
/// ## The endpoint is `held_out_example_mean`, never `final_loss_diagnostic`
///
/// CONTRACT H4/Frame: `d_i` (the C16/H2 paired sign test's per-seed datum)
/// is the FINAL-epoch `evaluate_held_out().mean` — [`Self::held_out_example_mean`],
/// paired with [`Self::final_epoch`]. [`Self::final_loss_diagnostic`] is
/// `TrainingResult::final_loss` (`best_val_loss`, a MIN-over-epochs order
/// statistic, `trainer.rs:1038/1048`) — recorded for comparison ONLY, never
/// the quantity a downstream merger should read as `d_i`.
///
/// ## Identity vs provenance (E3's disjoint convention, followed here)
///
/// [`Self::IDENTITY_FIELDS`] and [`Self::PROVENANCE_FIELDS`] are DISJOINT
/// sets — [`EncodeStepTier`]'s convention (its own doc's "provenance never
/// compared cross-producer"), not [`FinetuneStepTier`]'s superset-that-
/// includes-provenance-as-K7-completeness-additions convention. This is a
/// DELIBERATE deviation from "superset of the existing 18" read literally:
/// `attention_arm` is one of `FinetuneStepTier`'s 18 comparison fields
/// there, but on THIS tier it moves to provenance alongside the new `arm`
/// field, because the arm is the C16/H2 experiment's INDEPENDENT VARIABLE —
/// a merger that paired legs only when `attention_arm` (or `arm`) agreed
/// could never pair a fused leg with an alloff leg at all, defeating the
/// A/B protocol the sign test exists to run (CONTRACT H4: "the arm is
/// provenance, never identity"). Every other one of `FinetuneStepTier`'s 18
/// carries over into [`Self::IDENTITY_FIELDS`] verbatim (by name; `warmup`
/// and `row_lengths` read a DIFFERENT `NullMeans` reason here than there,
/// because a full multi-epoch real-text run has no per-tier
/// "discard-before-timing" convention and no single fixed row-lengths vector
/// over variable-length real text) — EXCEPT `batched_forward` and
/// `steps_measured`, reclassified below.
///
/// ## Unit-63 adversarial-audit finding 5: identity-completeness fixes
///
/// A round-5 audit found this set incomplete in two ways and non-honest in
/// two others. Fixed here:
///
/// (a) The held-out fixture's TEXT — a total determinant of every per-
/// example loss `d_i`, since `evaluate_held_out` scores the actual
/// anchor/positive/negative strings, not merely their ids — was hashed
/// NOWHERE: only [`Self::heldout_ids_sha256`] (the id ORDER) was anchored. A
/// caller could swap every row's text under a constant id list and neither
/// committed digest would move. [`Self::heldout_pairs_sha256`] closes this:
/// sha256 of the `--heldout-jsonl` file's own bytes, measured at load
/// (`main.rs::load_heldout_fixture`), never transcribed.
///
/// (b) [`Self::train_pairs_file_sha256`] (formerly spelled `dataset_sha256`
/// on this struct) is the RAW BYTES of the `--train-jsonl` file this run
/// read — a DIFFERENT quantity from the committed fixture manifest's own
/// `dataset_sha256` (a Merkle digest over PER-PAIR content hashes, built
/// off-process by a producer script), which happened to share the same
/// name. Two different quantities under one name meant neither anchored the
/// other — renamed to `train_pairs_file_sha256` so the field states exactly
/// what it hashes, and documents that content-anchoring this run's train
/// file against the committed `train_ids_sha256.json` manifest is the
/// PRODUCER's pre-run provisioning check (docs-ci/cookbook domain, landing
/// separately), not something this tier verifies for itself.
///
/// (c) Four identity slots could not vary independently of an already-
/// admitted field or a build-time constant, so their presence in the
/// COMPARISON set implied a discriminating power they never had:
///   * `split_rule` — a hardcoded literal (`"positional_fraction_split"`),
///     the same string on every run this binary can ever produce. Moved to
///     [`Self::PROVENANCE_FIELDS`]: recorded for legibility, never compared
///     (a constant cannot fail to match).
///   * `split_seed` — DROPPED entirely. It was defined as `params.seed`
///     verbatim (`TrainingDataLoader::split` takes no separate seed
///     parameter), so it was a pure, unconditional duplicate of the
///     already-identity [`Self::seed`] field — not merely constant-valued
///     like `split_rule`, but LITERALLY the same number under a second
///     name. Keeping it would have implied a "split seed" knob distinct
///     from the run seed that does not exist; two legs agreeing on `seed`
///     always agreed on `split_seed` too, so it added a slot to the count
///     without adding an independent check.
///   * `batched_forward` — always `true` (see this tier's own doc comment
///     on the field: "production has no un-batched arm for this tier to
///     record `false` for"). A structural fact about what this binary's
///     `encode_chunk` call always does, not a per-run knob — moved to
///     [`Self::PROVENANCE_FIELDS`] alongside `flash_compiled`/
///     `build_features` (other always-same-shape build facts already
///     recorded there rather than compared).
///   * `heldout_batch_partition_sha256` — KEPT in [`Self::IDENTITY_FIELDS`],
///     unlike the three above, despite ALSO being a pure function of two
///     already-identity inputs (the held-out id order — now doubly anchored
///     by [`Self::heldout_ids_sha256`] + [`Self::heldout_pairs_sha256`] —
///     and [`Self::batch`]). The distinction: `split_rule`/`split_seed`/
///     `batched_forward` are constants or literal echoes with NO algorithm
///     in between input and value, so comparing them can never catch
///     anything a raw-input comparison would miss. `heldout_batch_partition_sha256`
///     is instead the output of a real CODE PATH — the trainer's own
///     partitioning algorithm inside `evaluate_held_out` — applied to those
///     inputs; CONTRACT H1 names this "the batch partition IS identity"
///     precisely because a future implementation (this producer's own code
///     changing, or a second cross-producer implementation under the same
///     CONTRACT) could partition the SAME inputs differently and silently
///     score a different comparison than the one `heldout_ids`/`batch`
///     alone would lead a reader to expect. Comparing the REALIZED
///     partition hash directly is therefore a genuine cross-arm equality
///     guard against that algorithmic divergence, not a redundant echo of
///     inputs already in the set.
///
/// (d) Advisory: [`Self::steps_measured`] is a MEASURED OUTCOME of running
/// (`TrainingResult::total_steps` summed across the resume-cycle) — not a
/// premise the run was configured under — so it does not belong in the
/// comparison-identity set at all (contrast `FinetuneStepTier`'s own
/// `steps_measured`, which genuinely is identity there because two
/// `finetune-step` legs at a different measured step count computed a
/// different amount of work by definition of that tier's design). Moved to
/// [`Self::PROVENANCE_FIELDS`]: still recorded on every run, never a
/// comparison key here.
///
/// (e) Round-7 audit, finding 1: [`Self::mutant_id`]/[`Self::mutant_base_sha`]/
/// [`Self::mutant_patch_sha256`] are OMITTED from BOTH [`Self::IDENTITY_FIELDS`]
/// and [`Self::PROVENANCE_FIELDS`] entirely — not merely moved between them
/// like (c)'s three fields. A mutant leg's identity/provenance tuples are, by
/// the mutant campaign's own design (`mutants/README.md`'s "what M1 does NOT
/// touch"), IDENTICAL to a clean `fused` leg's: the patch changes which
/// binary produced the numbers, never what the run claims to have measured.
/// Naming the mutant on either tuple would make a mutant leg permanently
/// unpairable with the clean legs it exists to be diffed against — the same
/// reason [`Self::arm`] itself is provenance rather than identity (struct
/// doc, above), taken one step further: these three are neither. They are a
/// third, honest-labeling category — a caller's self-report of which patch
/// produced this leg, checked for internal completeness (all-or-none) by the
/// producer and cross-checked against the dose column's own claim by
/// `ci/scripts/perf/ab_merge.py`, never compared leg-to-leg the way
/// [`Self::IDENTITY_FIELDS`] is.
///
/// ## `margin`/`temperature`: objective-selected nullness (H4a-delta, CONTRACT
/// amendment 2026-08-28)
///
/// Unlike `FinetuneStepTier` (which always trains a hardcoded-margin Triplet
/// and so declares `margin` unconditionally `NonNull`), this tier runs
/// EITHER objective over the SAME committed fixture — H4a found the
/// committed H3 fixture TRIPLET-shaped, while the Frame's own
/// "embedding_loss+temp" phrasing anticipated MNRL, and H5 step 0's
/// dynamic-range probe needs BOTH to choose between them. So `margin` and
/// `temperature` are BOTH `Option`, and exactly one is `Some` per run,
/// selected by [`crate::finetune_run::Objective`]: `Objective::Triplet` →
/// `margin` non-null (the real, configured Triplet margin — never
/// `FinetuneStepTier`'s hardcoded `0.3`), `temperature` null
/// (`NullMeans("objective is triplet")`); `Objective::Mnrl` → `temperature`
/// non-null, `margin` null (`NullMeans("objective is mnrl")`). See each
/// field's own doc below.
#[derive(Debug, Serialize)]
pub struct FinetuneRunTier {
    // ── Identity: FinetuneStepTier's 18 (minus attention_arm — see struct
    //    doc), carried over by name ────────────────────────────────────
    pub seed: u64,
    pub batch: usize,
    /// `--max-seq-length` — the tokenizer truncation cap this run's config
    /// used (NOT a per-batch measured width: real text pairs vary in
    /// length row to row, unlike `finetune-step`'s fixed synthetic `seq`).
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    /// The Triplet objective's margin — `Some` only when
    /// [`crate::finetune_run::Objective::Triplet`] was selected for this
    /// run; `null` (`NullMeans("objective is mnrl")`) when
    /// [`crate::finetune_run::Objective::Mnrl`] was selected instead — see
    /// the struct doc's "objective-selected nullness" section and
    /// [`Self::temperature`]'s doc for MNRL's own scale knob.
    pub margin: Option<f64>,
    pub target_modules: Vec<String>,
    pub backbone_dtype: String,
    pub checkpoint_config_sha256: String,
    pub checkpoint_weights_sha256: String,
    pub checkpoint_weights_size_bytes: u64,
    /// `null` means clipping was off (`FineTuneConfig::max_grad_norm ==
    /// 0.0`) — same `NullMeans` reason `FinetuneStepTier::max_grad_norm`
    /// carries.
    pub max_grad_norm: Option<f64>,
    /// `null`: a full run has no "discard before timing" pre-step
    /// convention (unlike `finetune-step`'s own `warmup`, a per-tier
    /// micro-benchmark concept) — see `warmup_steps` for this tier's real
    /// LR-schedule warmup analogue.
    pub warmup: Option<usize>,
    /// `null`: real text is variable-length per micro-batch, so no single
    /// fixed vector describes "the" row lengths across a whole multi-epoch
    /// run the way `finetune-step`'s fixed synthetic batch can.
    pub row_lengths: Option<Vec<usize>>,

    // ── Identity: new fields (PLAN (d) / CONTRACT H4 v1+v2) ─────────────
    pub epochs: usize,
    pub lr: f64,
    pub schedule: String,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub grad_accum: usize,
    pub validation_fraction: f64,
    /// sha256 (hex) of the `--train-jsonl` file's own raw bytes — see this
    /// struct's own doc, finding 5(b), for why this is named distinctly
    /// from the committed fixture manifest's `dataset_sha256` (a different
    /// quantity: a Merkle over per-pair digests, not this file's bytes).
    pub train_pairs_file_sha256: String,
    pub heldout_ids_sha256: String,
    /// sha256 (hex) of the `--heldout-jsonl` file's own raw bytes — see this
    /// struct's own doc, finding 5(a): the held-out TEXT is a total
    /// determinant of every per-example loss `d_i`, so (like
    /// [`Self::heldout_ids_sha256`]'s id-order anchor) it must be content-
    /// anchored, never merely trusted by filename.
    pub heldout_pairs_sha256: String,
    /// `HeldOutLoss::batch_partition_sha256` at the FINAL epoch — the
    /// partition the reported [`Self::held_out_example_mean`] was scored
    /// under (CONTRACT H1 v2 delta 9: a property of `(model, partition)`).
    /// KEPT in identity despite being derivable from
    /// `(heldout_ids_sha256, heldout_pairs_sha256, batch)` — see this
    /// struct's own doc, finding 5(c), for why this one (unlike
    /// `split_rule`/`split_seed`/`batched_forward`) earns its own
    /// comparison slot: it is the REALIZED OUTPUT of a partitioning
    /// algorithm, not a constant or a literal echo of another field.
    pub heldout_batch_partition_sha256: String,
    /// `"triplet"` or `"mnrl"` — [`crate::finetune_run::Objective::as_str`],
    /// selected by the run's `--objective` flag (CONTRACT amendment
    /// 2026-08-28). Named in CONTRACT H4 as "embedding_loss+temperature";
    /// this tier trains BOTH objectives it names (H4a-delta), over the SAME
    /// committed fixture, so this field is genuinely NonNull either way.
    pub embedding_loss: String,
    /// MNRL's similarity-scale knob — `Some` only when
    /// [`crate::finetune_run::Objective::Mnrl`] was selected for this run;
    /// `null` (`NullMeans("objective is triplet")`) when
    /// [`crate::finetune_run::Objective::Triplet`] was selected instead —
    /// see the struct doc's "objective-selected nullness" section and
    /// [`Self::margin`]'s doc for the Triplet objective's own scale knob.
    pub temperature: Option<f64>,
    pub matryoshka_dims: Vec<usize>,
    pub early_stopping_patience: usize,
    pub early_stopping_metric: String,
    /// How often (in epochs) `evaluate_held_out` was called against the
    /// held-out fixture during this run's resume-cycle — always including
    /// the final epoch regardless of this cadence (see
    /// [`crate::finetune_run::run`]'s doc).
    pub eval_cadence: usize,

    // ── Provenance: PROVENANCE_FIELDS, disjoint from identity ───────────
    /// The CALLER-declared arm (`--arm`) — see this struct's own doc for
    /// why this is provenance, never identity.
    pub arm: String,
    pub device_name: String,
    pub kernels_disabled_requested: Vec<String>,
    pub kernels_disabled_fired: Vec<String>,
    pub flash_compiled: bool,
    pub build_features: Vec<&'static str>,
    /// The attention REFERENCE CLASS this process's `JAMMI_KERNELS_DISABLE`
    /// resolved to ASK for ([`crate::finetune_step::attention_arm`]) —
    /// `"eager"` iff an attention base (`attention_block`,
    /// `attention_block_flash`, or the `"all"` wildcard) is in
    /// `kernels_disabled_requested`, else `"fused"`. Deliberately NOT a
    /// claim about what actually dispatched (unit 63 re-audit round-2
    /// finding 2: this field's doc previously read "what actually ran",
    /// which was false — it is derived purely from the REQUESTED env var,
    /// the same as `FinetuneStepTier::attention_arm`'s own doc states of
    /// itself). Distinct from the caller's declared `arm`
    /// (`--arm`/[`Self::arm`], the higher-level fused-vs-alloff intent);
    /// PROVENANCE here (not identity) for the same reason `arm` is — see
    /// this struct's own doc. Whether the fused arm actually dispatched is
    /// the `*_fused_dispatches`/`*_eager_dispatches` counter fields' and a
    /// downstream merger's fused-proof job, exactly as
    /// `FinetuneStepTier::attention_arm`'s own doc states of that tier
    /// (mirrored here verbatim): "Deliberately NOT derived from the
    /// `attention_block_*_dispatches` deltas ... whether the fused arm
    /// actually ran stays where it already lives: `fused_proof` and the
    /// counters themselves."
    pub attention_arm: String,
    /// How `run()`'s internal early-stopping validation slice was carved
    /// out of the TRAIN rows this tier fed it — `TrainingDataLoader::split`
    /// is a plain positional (unshuffled) fraction split, never RNG-based,
    /// so this is a fixed constant across every leg this tier can ever
    /// produce. PROVENANCE, not identity (struct doc, finding 5(c)): a
    /// hardcoded literal has no discriminating power, so comparing it can
    /// never catch anything.
    pub split_rule: String,
    /// Always `true`: `TrainingLoop::encode_chunk`'s `Pairs`/`Triplet` arms
    /// always encode anchor+positive(+negative) in ONE joined forward via
    /// `encode_groups` — production has no un-batched arm for this tier to
    /// record `false` for (unlike `finetune-step`, which offers both as a
    /// within-run A/B). PROVENANCE, not identity (struct doc, finding
    /// 5(c)): a build-time structural constant, never a per-run knob.
    pub batched_forward: bool,
    /// Cumulative optimizer steps (`TrainingResult::total_steps` summed)
    /// across every resume-cycled epoch leg this run took. PROVENANCE, not
    /// identity (struct doc, advisory (d)): a MEASURED OUTCOME of running,
    /// not a premise the run was configured under — unlike
    /// `FinetuneStepTier::steps_measured`, where two legs at a different
    /// measured step count computed a different amount of work by that
    /// tier's own design.
    pub steps_measured: usize,

    // ── Fused-dispatch proof (unit 63 re-audit round-2 finding 2) ───────
    //
    // The SAME positive-proof channel `FinetuneStepTier` carries (identical
    // field names, identical semantics, identical read APIs — see each
    // sibling field's own doc there for the full rationale this block does
    // not repeat) — a before/after delta over the process-wide dispatch
    // counters taken around this run's WHOLE resume-cycled epoch loop (see
    // `finetune_run::run`'s own comment on where the snapshots are taken).
    // Like the counters on `FinetuneStepTier`, these are RECORDED
    // measurements, never identity or provenance (not in
    // `Self::IDENTITY_FIELDS` or `Self::PROVENANCE_FIELDS` — mirrors that
    // struct's own convention of leaving its counters out of
    // `FinetuneStepTier::IDENTITY_FIELDS` too): a downstream merger's
    // fused-proof gate reads these directly, by name, rather than through
    // either comparison tuple.
    pub ln_fused_dispatches: u64,
    pub ln_eager_dispatches: u64,
    pub rope_fused_dispatches: u64,
    pub rope_eager_dispatches: u64,
    pub softmax_fused_dispatches: u64,
    pub softmax_eager_dispatches: u64,
    pub geglu_fused_dispatches: u64,
    pub geglu_eager_dispatches: u64,
    pub lora_epilogue_fused_dispatches: u64,
    pub lora_epilogue_eager_dispatches: u64,
    pub lora_linear_fused_dispatches: u64,
    pub lora_linear_eager_dispatches: u64,
    /// The positive-proof channel for THIS finding: how many times
    /// ModernBERT's training-mode fused whole-attention-block kernel
    /// actually dispatched across this run's whole resume-cycle. On a
    /// `bert`-arch leg (this tier's generic CPU smoke fixture; ModernBert
    /// is the C16 gate's real checkpoint family — see
    /// `finetune_run::build_encoder_adapters`'s error message) this is
    /// legitimately `0` forever: classic BERT has no fused
    /// whole-attention-block kernel at all. On a `modernbert` leg that took
    /// at least one optimizer step, this and the three sibling counters
    /// below reading all-zero-at-once is the exact failure mode this
    /// finding fixed — see `finetune_run::run`'s own belt-and-braces typed
    /// refusal, which reads these same four counters before ever
    /// constructing this tier.
    pub attention_block_fused_dispatches: u64,
    pub attention_block_eager_dispatches: u64,
    pub adamw_fused_dispatches: u64,
    pub adamw_eager_dispatches: u64,
    pub attention_block_flash_fused_dispatches: u64,
    pub attention_block_flash_declined_dispatches: u64,

    // ── Premise legs (CONTRACT H4: "recorded per run, conjunctive, for
    //    the merger to refuse on") ───────────────────────────────────────
    /// The caller-declared premise (`--expect-dense`, default `false`) for
    /// whether this arm's real-text forward path took the dense transport —
    /// CALLER-DECLARED AND MERGER-CHECKED, never measured: this tier's
    /// real-text path drives `encode_chunk`'s plain `encoder.forward`, which
    /// never reaches `jammi_encoders::ModernBert::forward_with_lengths`'s
    /// dense-vs-padded fork (the one place `admission.is_dense` is actually
    /// decided) at all, so there is no live signal on this tier's admission
    /// path to read back and check the claim against. The committed
    /// fixture's variable-length arxiv pairs take the PADDED transport, so
    /// the default (`false`) matches the fixture's own known shape — see
    /// [`crate::finetune_run::run`]'s own doc and
    /// [`crate::finetune_run::FinetuneRunParams::expect_dense`]'s doc for
    /// why this tier's real-text path never reaches `forward_with_lengths`'s
    /// dense/padded fork at all.
    pub admission_is_dense: bool,
    /// `HeldOutLoss::tie_fraction` at the final epoch — the "tie cap"
    /// premise leg.
    pub tie_fraction: f64,

    // ── Measurements: recorded, never gated here (the merger gates) ─────
    /// 0-based index of the final epoch this run reached (`epochs - 1`).
    pub final_epoch: usize,
    /// THE endpoint (see struct doc): `evaluate_held_out().mean` at
    /// `final_epoch`.
    pub held_out_example_mean: f64,
    pub held_out_count: usize,
    /// `TrainingResult::final_loss` at the LAST epoch leg — DIAGNOSTIC
    /// ONLY, explicitly never `d_i` (see struct doc).
    pub final_loss_diagnostic: f64,
    /// The full per-epoch held-out trajectory (CONTRACT H4/v2 delta 4),
    /// one point per epoch this run actually evaluated (every
    /// `eval_cadence`th epoch, plus the final epoch unconditionally).
    pub trajectory: Vec<EpochHeldOut>,
    /// The RAW "learning-happened" train-side probe series (CONTRACT
    /// amendment 2026-08-29b, replacing the removed, producer-derived
    /// `learning_happened_delta` scalar): index 0 is the UNTRAINED model's
    /// probe — one `evaluate_held_out` call over the fixed train-probe
    /// batch, taken BEFORE the first epoch's `run()` leg (LoRA init is
    /// `ZerosB`, so this is a deterministic function of `(seed,
    /// target_modules)` alone) — then one entry per epoch thereafter, in
    /// epoch order, with the LAST entry the final epoch's probe. Always
    /// `params.epochs + 1` entries long. This producer never derives the
    /// "learning happened" premise itself: a downstream merger computes it
    /// from this series (`series[0] - series[series.len() - 1] > floor`) —
    /// the prior bug was exactly this producer pre-deriving that scalar
    /// from a baseline taken AFTER epoch 0 had already trained, silently
    /// excluding the largest-learning epoch from the window this field's
    /// old doc claimed ("over the run"); see
    /// [`crate::finetune_run::run`]'s own doc, amendment 2026-08-29b, for
    /// the full correction and the (deliberately absent) contract citation
    /// this replaces.
    pub train_probe_series: Vec<f64>,

    // ── Mutant provenance (unit 63 round-7 audit, finding 1) — honest
    //    labeling, NOT identity or provenance ───────────────────────────
    //
    // These three mirror [`crate::finetune_run::FinetuneRunParams::mutant_id`]/
    // `mutant_base_sha`/`mutant_patch_sha256` verbatim (see that struct's own
    // doc for the full "why not identity/provenance" rationale, repeated
    // here in short): a mutant leg's own IDENTITY_FIELDS/PROVENANCE_FIELDS
    // are IDENTICAL to a clean `fused` leg's (the mutant only patches which
    // binary produced the numbers, never what the run was configured to
    // measure), so a mutant's name belongs to neither comparison tuple —
    // deliberately absent from both [`Self::IDENTITY_FIELDS`] and
    // [`Self::PROVENANCE_FIELDS`] below. They are a CALLER-DECLARED
    // self-report (a mutant leg names itself; this process cannot verify
    // from inside itself which patch it was actually built from), not a
    // measured or derived fact — closer to a signature than a
    // premise/provenance leg. All three are `None` for an ordinary
    // (non-mutant) leg, and `#[serde(skip_serializing_if =
    // "Option::is_none")]` omits the keys entirely in that case, so a normal
    // leg's emitted JSON (and every committed golden built from one) is
    // byte-for-byte unchanged by this addition. `ci/scripts/perf/ab_merge.py`'s
    // mutant-dose-ladder merge mode reads these three keys BY THESE EXACT
    // NAMES to attribute a dose column's legs to a specific, auditable
    // mutant patch.
    /// `--mutant-id`: the mutant's own label (e.g. `"eps-0.10"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutant_id: Option<String>,
    /// `--mutant-base-sha`: the git commit sha this mutant's patch was cut
    /// against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutant_base_sha: Option<String>,
    /// `--mutant-patch-sha256`: sha256 (hex) of the mutant patch's own
    /// content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutant_patch_sha256: Option<String>,
}

impl FinetuneRunTier {
    /// The comparison identity: `FinetuneStepTier`'s 18 (minus
    /// `attention_arm`, moved to provenance — see struct doc) plus the 18
    /// new fields PLAN (d) / CONTRACT H4 name, THEN the unit-63 adversarial-
    /// audit finding-5 fixes applied: `split_rule`, `split_seed`,
    /// `batched_forward`, and `steps_measured` removed (`split_seed` dropped
    /// entirely; the other three reclassified to [`Self::PROVENANCE_FIELDS`]
    /// — see struct doc for the full per-field rationale), `dataset_sha256`
    /// renamed to `train_pairs_file_sha256`, and `heldout_pairs_sha256`
    /// added. 17 + 18 − 4 + 1 = 32.
    ///
    /// DISJOINT from [`Self::PROVENANCE_FIELDS`] (E3's convention, not
    /// `FinetuneStepTier`'s superset one) — see struct doc.
    pub const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        // FinetuneStepTier's 18, minus attention_arm (17 entries), minus
        // `batched_forward`/`steps_measured` (finding 5(c)/advisory (d) —
        // both reclassified to PROVENANCE_FIELDS below).
        ("seed", Nullable::NonNull),
        ("batch", Nullable::NonNull),
        ("seq", Nullable::NonNull),
        ("lora_rank", Nullable::NonNull),
        ("lora_alpha", Nullable::NonNull),
        ("lora_dropout", Nullable::NonNull),
        // H4a-delta (CONTRACT amendment 2026-08-28): unlike
        // `FinetuneStepTier::margin` (always NonNull, hardcoded Triplet),
        // this tier's `margin` is null exactly when `Objective::Mnrl` was
        // selected — see the struct doc's "objective-selected nullness"
        // section.
        ("margin", Nullable::NullMeans("objective is mnrl")),
        ("target_modules", Nullable::NonNull),
        ("backbone_dtype", Nullable::NonNull),
        ("checkpoint_config_sha256", Nullable::NonNull),
        ("checkpoint_weights_sha256", Nullable::NonNull),
        ("checkpoint_weights_size_bytes", Nullable::NonNull),
        ("max_grad_norm", Nullable::NullMeans("no clip")),
        (
            "warmup",
            Nullable::NullMeans(
                "a full run has no discard-before-timing convention; see warmup_steps",
            ),
        ),
        (
            "row_lengths",
            Nullable::NullMeans(
                "real text is variable-length; no single fixed row_lengths applies across a \
                 whole multi-epoch run",
            ),
        ),
        // New, minus `split_rule`/`split_seed` (finding 5(c): a constant and
        // a literal duplicate of `seed`, neither a genuine determinant),
        // `dataset_sha256` renamed to `train_pairs_file_sha256` (finding
        // 5(b): distinct from the fixture manifest's own `dataset_sha256`),
        // plus `heldout_pairs_sha256` added (finding 5(a): the held-out
        // TEXT was hashed nowhere before this fix).
        ("epochs", Nullable::NonNull),
        ("lr", Nullable::NonNull),
        ("schedule", Nullable::NonNull),
        ("warmup_steps", Nullable::NonNull),
        ("weight_decay", Nullable::NonNull),
        ("grad_accum", Nullable::NonNull),
        ("validation_fraction", Nullable::NonNull),
        ("train_pairs_file_sha256", Nullable::NonNull),
        ("heldout_ids_sha256", Nullable::NonNull),
        ("heldout_pairs_sha256", Nullable::NonNull),
        ("heldout_batch_partition_sha256", Nullable::NonNull),
        ("embedding_loss", Nullable::NonNull),
        ("temperature", Nullable::NullMeans("objective is triplet")),
        ("matryoshka_dims", Nullable::NonNull),
        ("early_stopping_patience", Nullable::NonNull),
        ("early_stopping_metric", Nullable::NonNull),
        ("eval_cadence", Nullable::NonNull),
    ];

    /// Provenance — recorded, present on every run, but NEVER a comparison
    /// key (see struct doc for why `arm`/`attention_arm` live here rather
    /// than in [`Self::IDENTITY_FIELDS`]). Grew 7 -> 10 with the unit-63
    /// adversarial-audit finding-5(c)/advisory-(d) reclassifications:
    /// `split_rule` (a hardcoded constant), `batched_forward` (a build-time
    /// structural fact), and `steps_measured` (a measured outcome, not a
    /// premise) — none of the three is a genuine comparison determinant;
    /// see struct doc for the full per-field rationale.
    pub const PROVENANCE_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("arm", Nullable::NonNull),
        ("device_name", Nullable::NonNull),
        ("kernels_disabled_requested", Nullable::NonNull),
        ("kernels_disabled_fired", Nullable::NonNull),
        ("flash_compiled", Nullable::NonNull),
        ("build_features", Nullable::NonNull),
        ("attention_arm", Nullable::NonNull),
        ("split_rule", Nullable::NonNull),
        ("batched_forward", Nullable::NonNull),
        ("steps_measured", Nullable::NonNull),
    ];
}

/// The on-GPU throughput/latency tier: the engine's two GPU-model serving
/// verbs — [`generate_text_embeddings`](jammi_ai::session::InferenceSession::generate_text_embeddings)
/// (embed) and [`infer`](jammi_ai::session::InferenceSession::infer)
/// (classification) — measured on `gpu.device = 0` over their own tiny
/// committed bundles. The GPU peer of [`ModelInferenceTier`] — where that tier
/// is CPU-hermetic and gates determinism, this one runs on the real device and
/// *records* the perf a GPU optimization would move, tagged with the device
/// that produced it.
///
/// Throughput and tail latency ride as measurements, not gates: the prove lane
/// runs on an ephemeral heterogeneous rented fleet (SXM4 / PCIe A100s, no
/// pinning), so an absolute rate is a property of `code × device ×
/// pod-conditions`, not of the code alone — a committed absolute floor would
/// gate pod variance, not a regression. The device-independent correctness
/// contracts this tier DOES hard-gate: (1) the session resolved to a real CUDA
/// device, never a silent CPU fallback (a serve error exits the tier
/// non-zero); (2) the classification lane's scored row count matches the
/// corpus row count on every serve (row conservation — [`GpuLane::rows`]).
/// Cross-repeat determinism and CPU↔GPU parity are hard-gated separately, in
/// the `gpu_capability` suite.
///
/// The report is emitted as one stable JSON document (`Report` →
/// `tiers.gpu_inference`, printed by `emit()`): `device`, `device_name`, and
/// one [`GpuLane`] per verb (`embed`, `infer`), each carrying `rows`,
/// `rows_per_s`, `p50_ms`, `p99_ms`, ALONGSIDE every declared
/// [`Self::IDENTITY_FIELDS`]/[`Self::PROVENANCE_FIELDS`] entry (see the
/// "Identity contract" section below — `iters` in particular is an IDENTITY
/// field, not a bare observability count). Every key is present on every run
/// (no field is conditionally omitted and no record carries a timestamp), so
/// two runs' JSON diffs cleanly — the groundwork the issue-#335 within-run A/B
/// perf comparator (`ci/scripts/perf/gpu_inference_ab.py`) reads.
///
/// ## Identity contract (issue #335, the esc-057/K7 class)
///
/// A within-run A/B ratio (parent-HEAD vs a PR change, measured back to back
/// on the SAME pod) is only a comparison of the SAME MEASUREMENT if both legs
/// agree on every output-affecting parameter — the exact esc-057/K7 gap
/// [`EncodeStepTier`]/[`FinetuneStepTier`] already close one layer down, at
/// the encode/finetune surfaces. This tier closes the analogous gap at the
/// on-GPU serving surface: [`Self::IDENTITY_FIELDS`] names the complete
/// comparison tuple —
///
///   * `corpus_seed`/`row_count`/`corpus_sha256`: the corpus-generation
///     premise. `row_count` in particular closes the "manufactured-2x
///     attack" (round-1 adversarial audit B1): `p50_ms` moves LINEARLY with
///     `row_count` (more rows per `generate_text_embeddings` call is a
///     longer call), so a leg that silently served a different row count
///     would print a ratio that reflects nothing about the code under
///     comparison. `corpus_sha256` (`model_inference::corpus_sha256`) is a
///     sha256 content hash over every committed sentence plus
///     `corpus_seed`/`row_count` — the belt-and-suspenders closing move: a
///     PR that merely REWORDS a sentence (holding `corpus_seed`/`row_count`
///     fixed) moves NEITHER of those two scalars, so without a content hash
///     that edit would silently slip through as "the same corpus".
///   * `warmup`/`iters`: the discarded-vs-measured serve counts that bound
///     what the percentiles actually fold over. `iters` was already emitted
///     pre-#335 but never admitted to identity (round-1 adversarial audit
///     B1) — a leg run with a different `iters` count computes a p50/p99
///     over a differently-sized sample, which is not the same measurement.
///   * `compute_precision` and BOTH served bundles' content identity — embed
///     and classifier, each hashed independently since a two-verb tier has
///     two checkpoints, not one.
///
/// [`Self::PROVENANCE_FIELDS`] names the recorded-but-never-compared facts
/// (the resolved hardware name, the kernel-disable/flash-compiled build
/// facts). `run()` asserts both sets present via
/// [`assert_identity_fields_present`] on every real invocation — the SAME
/// posture [`crate::encode_step::run`]/[`crate::finetune_step::run`] already
/// enforce, following [`EncodeStepTier`]'s disjoint (never
/// superset-folding) identity/provenance split (CONTRACT.md §E3's shape),
/// not [`FinetuneStepTier`]'s.
///
/// `compute_precision` is read off the LOADED EMBED model
/// ([`jammi_ai::model::LoadedModel::compute_precision`], the SAME accessor
/// [`EncodeStepTier::compute_precision`] reads) — the precision the tier's
/// pre-registered primary endpoint (embed `p50_ms`, see
/// `ci/scripts/perf/gpu_inference_ab.py`'s own module doc) actually served
/// at. The classifier bundle's own precision is not separately identity-
/// admitted: this tier states ONE primary endpoint (embed), and a second
/// identity field for a workload nothing gates would be a false
/// determinant of the exact kind esc-057/K7 forbid (a field two legs could
/// disagree on with no effect on what the endpoint measures).
///
/// `corpus_seed`/`row_count`/`warmup`/`corpus_sha256` were previously either
/// compile-time constants ([`crate::main`]'s `GPU_INFERENCE_PARAMS` literal)
/// or never computed at all, never emitted on the report — a within-run A/B
/// comparator reading two JSON reports has no way to state "both legs used
/// the same corpus/warmup convention" without them actually being on the
/// report; this tier now emits all four.
#[derive(Debug, Serialize)]
pub struct GpuInferenceTier {
    /// The device the serve resolved to (e.g. `cuda:0`) — proof it was not a CPU
    /// fallback.
    pub device: String,
    /// The concrete device sub-class the ordinal resolved to (e.g.
    /// `NVIDIA A100-SXM4-80GB`) — the provenance tag that makes the recorded
    /// throughput/latency interpretable across the heterogeneous fleet.
    /// PROVENANCE (see [`Self::PROVENANCE_FIELDS`]): only knowable after the
    /// device resolved.
    pub device_name: String,

    // ── Comparison identity: `Self::IDENTITY_FIELDS` ────────────────────
    /// The corpus-generation seed (mirrors `ModelInferenceSpec::corpus_seed`'s
    /// own rotation) — previously a compile-time-only constant, now emitted
    /// so a within-run A/B comparator can state both legs drew the same
    /// corpus.
    pub corpus_seed: u64,
    /// The synthetic corpus row count — identity (round-1 adversarial audit
    /// B1, the "manufactured-2x attack"): `p50_ms` moves LINEARLY with this
    /// value, so two legs at a different `row_count` are not comparable at
    /// all, regardless of anything else they agree on.
    pub row_count: usize,
    /// Serves discarded before the measured iterations (pays the one-time
    /// model-load/PTX-JIT cost so it does not land in a measured tail) —
    /// changes what the measured percentiles actually bound.
    pub warmup: usize,
    /// Measured serves (after warmup) the percentiles folded over — identity
    /// (round-1 adversarial audit B1): this field was already emitted
    /// pre-#335 but never admitted to [`Self::IDENTITY_FIELDS`], so two legs
    /// at a different `iters` (hence a differently-sized measured sample)
    /// could silently compare as "the same measurement".
    pub iters: usize,
    /// sha256 (hex) content hash of the corpus-generation premise
    /// ([`crate::model_inference::corpus_sha256`]): every committed sentence
    /// this tier's corpus generator can draw from, plus `corpus_seed` and
    /// `row_count`. The belt-and-suspenders closing move alongside
    /// `corpus_seed`/`row_count` above: a PR that REWORDS a committed
    /// sentence (holding the seed/row-count scalars fixed) moves neither of
    /// those two fields, so without a content hash that edit would silently
    /// slip through as "the same corpus" while actually serving different
    /// text.
    pub corpus_sha256: String,
    /// The compute precision (`f32`/`f16`/`bf16`) the LOADED embed model
    /// actually resolved to before the serve — read off
    /// [`jammi_ai::model::LoadedModel::compute_precision`], the SAME
    /// accessor [`EncodeStepTier::compute_precision`] reads. See this
    /// struct's own doc for why only the embed bundle's precision is
    /// admitted to identity.
    pub compute_precision: String,
    /// sha256 (hex) of the embed bundle's `config.json` bytes — a third of
    /// that checkpoint's content identity, the SAME `sha256_and_len` helper
    /// [`EncodeStepTier::checkpoint_config_sha256`] uses.
    pub embed_checkpoint_config_sha256: String,
    /// sha256 (hex) of the embed bundle's `model.safetensors` bytes.
    pub embed_checkpoint_weights_sha256: String,
    /// sha256 (hex) of the embed bundle's `tokenizer.json` bytes.
    pub embed_checkpoint_tokenizer_sha256: String,
    /// sha256 (hex) of the classifier bundle's `config.json` bytes — the
    /// SAME three-file content identity as the embed bundle above, hashed
    /// independently: this tier serves two DIFFERENT checkpoints (embed +
    /// classifier), so one hash triple cannot stand in for both.
    pub infer_checkpoint_config_sha256: String,
    /// sha256 (hex) of the classifier bundle's `model.safetensors` bytes.
    pub infer_checkpoint_weights_sha256: String,
    /// sha256 (hex) of the classifier bundle's `tokenizer.json` bytes.
    pub infer_checkpoint_tokenizer_sha256: String,

    // ── Provenance: `Self::PROVENANCE_FIELDS`, NEVER identity ───────────
    /// The `JAMMI_KERNELS_DISABLE` op keys this process REQUESTED (sorted;
    /// empty when unset) — `jammi_kernels::admission::disabled_ops_requested`.
    /// Mirrors `EncodeStepTier::kernels_disabled_requested`.
    pub kernels_disabled_requested: Vec<String>,
    /// Whether THIS BUILD compiled the vendored FlashAttention-2 kernels
    /// (`jammi_kernels::admission::FLASH_COMPILED`). The encode/infer serve
    /// path never dispatches flash regardless (fused arms are
    /// training-only), so this records a build fact, not a per-leg
    /// determinant — mirrors `EncodeStepTier::flash_compiled`.
    pub flash_compiled: bool,
    /// This tier's own echo of [`Provenance::build_features`]
    /// (`crate::report::build_features`, the SAME function every other
    /// tier's own `build_features` field reads).
    pub build_features: Vec<&'static str>,

    /// The embed verb's lane.
    pub embed: GpuLane,
    /// The classification (`infer`) verb's lane.
    pub infer: GpuLane,
}

impl GpuInferenceTier {
    /// The comparison identity for a within-run A/B ratio (issue #335): the
    /// complete output-affecting parameter set for the on-GPU embed/infer
    /// serving surface. DISJOINT from [`Self::PROVENANCE_FIELDS`]
    /// ([`EncodeStepTier`]'s own E3 convention, never
    /// [`FinetuneStepTier`]'s superset-folding one — see this struct's own
    /// doc). `ci/scripts/perf/identity_fields.py`'s
    /// `GPU_INFERENCE_IDENTITY_FIELDS` mirrors this list EXACTLY.
    pub const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("corpus_seed", Nullable::NonNull),
        ("row_count", Nullable::NonNull),
        ("warmup", Nullable::NonNull),
        ("iters", Nullable::NonNull),
        ("corpus_sha256", Nullable::NonNull),
        ("compute_precision", Nullable::NonNull),
        ("embed_checkpoint_config_sha256", Nullable::NonNull),
        ("embed_checkpoint_weights_sha256", Nullable::NonNull),
        ("embed_checkpoint_tokenizer_sha256", Nullable::NonNull),
        ("infer_checkpoint_config_sha256", Nullable::NonNull),
        ("infer_checkpoint_weights_sha256", Nullable::NonNull),
        ("infer_checkpoint_tokenizer_sha256", Nullable::NonNull),
    ];

    /// The provenance fields this tier records but NEVER admits to
    /// [`Self::IDENTITY_FIELDS`] — recorded so a downstream reader has the
    /// SAME `assert_identity_fields_present` presence/non-null guarantee on
    /// these fields without them ever being eligible as a cross-leg
    /// comparison key. Mirrors the corresponding entries of
    /// [`EncodeStepTier::PROVENANCE_FIELDS`].
    pub const PROVENANCE_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("device_name", Nullable::NonNull),
        ("kernels_disabled_requested", Nullable::NonNull),
        ("flash_compiled", Nullable::NonNull),
        ("build_features", Nullable::NonNull),
    ];
}

/// The identity-audited encode-step tier (unit 62, K7/E3): drives the
/// engine's real text-embedding serving surface —
/// [`generate_text_embeddings`](jammi_ai::session::InferenceSession::generate_text_embeddings),
/// the SAME `resolve -> tokenize -> forward -> pool -> normalize` path a
/// serving request walks — over a small deterministic corpus and a
/// committed-shape fixture model directory, so a step's report is a real
/// measurement of the shipped path, never a synthetic loop that bypasses the
/// engine's own resolve/tokenize/pool/normalize sequence.
///
/// ## Why this tier exists: K7 completeness at the bench-comparison seam
///
/// esc-057 (closed at the `ModelIdentity` layer by this unit's E1/E2) was a
/// silent-identity defect: pooling/tokenizer/weights could mutate under a
/// constant `model_id` with no `DefinitionHash` change. This tier closes the
/// analogous gap one layer up, at BENCH comparison: two `encode-step` legs
/// are "the same measurement" only if every field in
/// [`EncodeStepTier::IDENTITY_FIELDS`] agrees — the complete
/// output-affecting parameter set for this surface (seed, batch shape, the
/// resolved sequence/row lengths, the compute precision, the checkpoint's
/// content identity (config/weights/tokenizer/pooling-config bytes), the
/// pooling strategy actually applied, whether the output is normalized, the
/// requested device, and the warmup/measured-iteration counts that bound
/// what was actually timed).
///
/// ## `pooling` is READ OFF THE LOADED MODEL, never transcribed (unit-62 F-5')
///
/// The fixture model directory this tier builds carries an EXPLICIT
/// `1_Pooling/config.json` (mean pooling) rather than the bare `tiny_bert`
/// fixture, which ships with no `1_Pooling/` folder at all and would
/// silently resolve through `candle.rs`'s own mean-pooling fallback
/// (`pooling_from_config`'s documented default for a repo that ships no
/// pooling config) — the exact ambiguity esc-057 is about. [`Self::pooling`]
/// itself is read straight off
/// [`jammi_ai::model::LoadedModel::resolved_pooling`] (the SAME accessor
/// pattern [`Self::compute_precision`]'s own fix already established) —
/// the pooling strategy the LOADED model's text-embedding wrapper actually
/// pools with, never a constant mirroring the fixture-writer function. A
/// round-3 audit (F-5') found the prior constant `"mean"` literal
/// byte-identical across a flip of the fixture to `Cls` while every other
/// identity field (including the config/weights/tokenizer checksums) stayed
/// put — reading the LOADED model closes that gap: [`Self::checkpoint_pooling_sha256`]
/// closes the companion gap of the pooling-CONFIG BYTES never entering any
/// identity field at all.
///
/// ## `attention_arm` and the memeff `chunk_size` are PROVENANCE, never identity
///
/// [`Self::attention_arm`]/[`Self::chunk_size`]/[`Self::device_name`]/
/// [`Self::kernels_disabled_requested`]/[`Self::kernels_disabled_fired`]/
/// [`Self::flash_compiled`]/[`Self::build_features`] are recorded on this
/// tier but deliberately excluded from [`Self::IDENTITY_FIELDS`] — see
/// [`Self::PROVENANCE_FIELDS`]'s own doc for the full per-field rationale.
/// `attention_arm` in particular is FORBIDDEN from identity (v2 reshape 3 of
/// the unit-62 plan): a dispatched arm is a POST-HOC fact about what ran,
/// and K7's own `definition_of` requires an identity hash be computable
/// BEFORE compute (memoization soundness) — a field only knowable after the
/// forward completed can never be a memoization key. It is also constant on
/// this surface by construction (fused attention arms are training-only,
/// `modernbert.rs`'s `if self.training` gate — eval/serving stays eager),
/// which independently makes it a false determinant were it ever admitted:
/// two legs would always "agree" on it regardless of what else differed.
///
/// ## `device_requested` is identity; `device_name` stays provenance (round-3 audit ruling)
///
/// The original E3 "device is provenance" ruling rested on this tier being
/// CPU-only by construction — every leg necessarily agreed on device, so
/// admitting it into identity would have been a vacuous no-op determinant.
/// `--cuda` (`EncodeStepParams::gpu_device`) retired that premise: two real
/// legs can now genuinely differ on device. [`Self::device_requested`] — the
/// REQUESTED device, declared by the caller BEFORE compute (`"cpu"` or
/// `"cuda:<ordinal>"`) — is therefore identity field 15: it satisfies K7's
/// identity-computable-before-compute rule and two legs that asked for
/// different devices must never compare as the same measurement.
/// [`Self::device_name`] (the POST-HOC hardware string a real CUDA leg
/// queries off the driver, e.g. `"NVIDIA A100-SXM4-80GB"`, or the constant
/// `"cpu"` label for the CI-hermetic default) stays provenance: it is only
/// knowable AFTER the device resolved (never before compute), and two legs
/// that both requested `"cuda:0"` on two different physical GPUs are still
/// the "same measurement" for identity purposes provided every
/// [`Self::IDENTITY_FIELDS`] entry (including `device_requested`) agrees.
#[derive(Debug, Serialize)]
pub struct EncodeStepTier {
    // ── Comparison identity: [`Self::IDENTITY_FIELDS`] ──────────────────
    /// The corpus-generation seed — rotates which committed sentence each
    /// row draws (mirrors `ModelInferenceSpec::corpus_seed`'s own rotation),
    /// so a fixed seed is a fixed, reviewable input set.
    pub seed: u64,
    /// The number of rows one `generate_text_embeddings` call served —
    /// the corpus row count.
    pub batch: usize,
    /// The padded sequence length (columns) the real tokenizer produced for
    /// this batch (`BatchEncoding::seq_len`, batch-longest padding) — the
    /// widest row's real token count, MEASURED off the model's own
    /// `tokenizer.json` via the same [`jammi_ai::model::tokenizer::TokenizerWrapper`]
    /// the candle backend loads, never assumed.
    pub seq: usize,
    /// Each row's REAL (unpadded) token count — the sum of that row's
    /// attention-mask ones, off the same real tokenization [`Self::seq`]
    /// is measured from. Never a knob: the corpus sentences have genuinely
    /// different lengths, so this vector legitimately varies row to row
    /// (never `[seq; batch]` unless every row happens to tie the widest).
    pub row_lengths: Vec<usize>,
    /// The compute precision (`f32`/`f16`/`bf16`,
    /// [`jammi_numerics::ComputePrecision`]'s `Display`) the LOADED model
    /// actually resolved to before the serve — read straight off
    /// [`jammi_ai::model::LoadedModel::compute_precision`] via the tier's
    /// own session/model cache (the SAME accessor `InferenceSession::infer`
    /// reads before folding it into `ModelIdentity.compute_precision`,
    /// `compute_precision.rs`'s own materialization-identity contract),
    /// never a derived/default constant — this field previously recorded
    /// `ComputePrecision::default()` regardless of what the tier's own
    /// `GpuConfig`/per-model `config.json` override actually resolved,
    /// which is the exact false-determinant shape this struct's own doc
    /// forbids (unit-62 F-5). Output-affecting because a lower-precision
    /// forward is a different computation, not merely a faster one.
    pub compute_precision: String,
    /// sha256 (hex) of the fixture model dir's `config.json` bytes — a
    /// third of the checkpoint's content identity, the SAME `sha256_and_len`
    /// helper [`crate::finetune_step`]/[`crate::grad_oracle`] already use
    /// (never a second, independently-drifting hashing implementation).
    pub checkpoint_config_sha256: String,
    /// sha256 (hex) of the fixture model dir's `model.safetensors` bytes —
    /// another third of the checkpoint's content identity.
    pub checkpoint_weights_sha256: String,
    /// `model.safetensors`' byte length — a cheap, redundant cross-check
    /// alongside the sha256 above.
    pub checkpoint_weights_size_bytes: u64,
    /// sha256 (hex) of the fixture model dir's `tokenizer.json` bytes — the
    /// final third of the checkpoint's content identity. Output-affecting:
    /// this same unit's own E1/E2 legs prove tokenizer bytes move the
    /// encode surface's served output (a different vocabulary/merge table
    /// tokenizes the identical text to different token ids), so a
    /// tokenizer-bytes change with `checkpoint_config_sha256`/
    /// `checkpoint_weights_sha256` held fixed must never read as "the same
    /// leg" (unit-62 F-5). The SAME `sha256_and_len` helper the other two
    /// checkpoint hashes above use.
    pub checkpoint_tokenizer_sha256: String,
    /// The pooling strategy the LOADED model's text-embedding wrapper
    /// ACTUALLY pools with (unit-62 F-5') — read via
    /// [`jammi_ai::model::LoadedModel::resolved_pooling`] off the same
    /// session/model cache [`Self::compute_precision`] reads, then rendered
    /// through [`jammi_encoders::Pooling`]'s own `Display` (`"mean"`/
    /// `"cls"`/`"max"`/`"weighted_mean"`). Never a constant mirroring
    /// `encode_step::mean_pooling_flags` — see this struct's own
    /// doc for the F-5' finding this closes: flipping the fixture's
    /// `1_Pooling/config.json` to CLS must move this field.
    pub pooling: String,
    /// sha256 (hex) of the fixture model dir's `1_Pooling/config.json` bytes,
    /// `None` when the model dir carries no `1_Pooling/` folder at all
    /// (unit-62 F-5'(b)) — the SAME presence gate
    /// `backend::candle::all_candidate_paths` applies before hashing this
    /// file into the engine's own `content_digest` (`resolved.pooling_config
    /// .is_some()`), never a second, independently-drifting presence check.
    /// `NullMeans("no 1_Pooling/config.json in this model dir")`: `None`
    /// here means exactly that absence, never "this producer predates the
    /// field". This tier's own `build_encode_model_dir` always writes an
    /// explicit `1_Pooling/config.json` (see this struct's own doc), so a
    /// real run of THIS tier always reports `Some`; the `Option` exists so
    /// the schema honestly represents the presence-gated engine reality
    /// rather than assuming every model dir this surface could ever measure
    /// carries one. Output-affecting alongside [`Self::pooling`]: a
    /// differently-worded-but-equivalent pooling declaration (e.g. the
    /// `pooling_mode_mean_sqrt_len_tokens` alias `pooling_from_config` also
    /// maps to `Mean`) would report the same `pooling` string but a
    /// DIFFERENT `checkpoint_pooling_sha256`, correctly distinguishing the
    /// two source files as different measurements at the checkpoint-content
    /// layer even though they serve byte-identical output.
    pub checkpoint_pooling_sha256: Option<String>,
    /// Whether the served vector is L2-normalized. `jammi_encoders::pool_and_normalize`
    /// mandatorily normalizes on every reachable path today (there is no
    /// exposed toggle), so this reads `true` on every run — recorded
    /// honestly as the pipeline's current invariant, not a knob this tier
    /// can flip, so a future normalize-optional path has an identity slot
    /// already reserved rather than a silent addition. Pinned code-invariant
    /// (round-3 audit advisory, folded rather than deferred): the enforcing
    /// code is `jammi_encoders::pooling::pool_and_normalize`, whose `Result`
    /// signature has no toggle parameter at all — there is no code path in
    /// this crate's dependency graph that reaches a pooled embedding without
    /// going through it. `encode_step::tests::pool_and_normalize_is_mandatory_with_no_toggle`
    /// asserts this invariant directly (a hand-built, deliberately
    /// non-unit-norm hidden tensor still comes out unit-L2-norm), so a
    /// future normalize-optional signature change trips that test rather
    /// than silently leaving this field a stale, unmeasured constant.
    pub normalize: bool,
    /// Warmup serves (discarded, not folded into any measurement) before
    /// the measured iterations — pays the one-time model-load cost so it
    /// does not land in a measured wall-time.
    pub warmup: usize,
    /// The number of `generate_text_embeddings` calls actually folded into
    /// this tier's measured wall-time/throughput.
    pub iters_measured: usize,
    /// The REQUESTED device (identity field 15, round-3 audit lead ruling)
    /// — `"cpu"` for the CI-hermetic default, `"cuda:<ordinal>"` for the pod
    /// producer's `--cuda <ordinal>` — declared straight from
    /// [`crate::encode_step::EncodeStepParams::gpu_device`] BEFORE any
    /// compute runs (satisfies K7's identity-computable-before-compute
    /// rule), via `encode_step::requested_device_label` — the SAME cheap
    /// ordinal-derived `"cpu"`/`"cuda:<ordinal>"` label [`Self::device_name`]
    /// rendered unconditionally before this fix (see this struct's own doc
    /// for why the requested value is identity while the post-hoc hardware
    /// string stays provenance). Two legs that asked for different devices
    /// must never compare as the same measurement.
    pub device_requested: String,

    // ── Provenance: [`Self::PROVENANCE_FIELDS`], NEVER identity ─────────
    /// The device this run ACTUALLY served on, as a post-hoc hardware fact —
    /// the constant `"cpu"` label for the CI-hermetic default, or the real
    /// CUDA device sub-class name queried off the driver for a `--cuda` leg
    /// (e.g. `"NVIDIA A100-SXM4-80GB"`, the SAME in-process `cudarc` query
    /// `gpu_inference::cuda_device_name` already performs — never a second,
    /// independently-drifting hardware-name lookup). PROVENANCE (round-3
    /// audit lead ruling): only knowable AFTER the device resolved, so it
    /// can never be a memoization key; two legs that both requested
    /// `"cuda:0"` on two different physical GPUs are still the "same"
    /// measurement for identity purposes provided every
    /// [`Self::IDENTITY_FIELDS`] entry (including [`Self::device_requested`])
    /// agrees. See this struct's own doc for the full identity-vs-provenance
    /// split this field and `device_requested` now form.
    ///
    /// This value is trustworthy — never a hardware string attested for a
    /// run that actually executed on CPU — BECAUSE the silent-CPU-fallback
    /// state is structurally unrepresentable by the time it is computed
    /// (unit-62 audit round-4 F-C): `encode_step::run` threads `gpu_device`
    /// through `model_inference::corpus_session_on_device`, which sets
    /// `gpu.require_gpu = gpu_device >= 0` on the session's `GpuConfig` — the
    /// SAME convention `jammi-ai`'s `gpu_capability` harness pins
    /// (`config_for`: `require_gpu: device >= 0`). A `--cuda N` leg whose
    /// ordinal the box cannot actually satisfy therefore fails the FIRST
    /// model load (`CandleBackend::load`'s `select_device(device_config)?`,
    /// `backend/candle.rs`'s `gpu_unavailable` returning a typed
    /// `JammiError::Gpu`) inside `run()`'s warmup loop, well before this
    /// field is ever populated — `run()` returns `Err` and no
    /// `EncodeStepTier` (hence no report) is produced at all. So on every
    /// path that reaches this field, the requested ordinal and the actually-
    /// resolved device are the same device by construction.
    ///
    /// **Qualification (audit round 62, adversarial round 6, folded
    /// advisory)**: on a build compiled with `feature = "metal"` but not
    /// `"cuda"`, `select_device` can genuinely succeed for a `gpu_device >=
    /// 0` request via its metal branch, so the "first model load fails"
    /// argument above does not apply there — the guarantee holds on that
    /// build for a DIFFERENT reason instead: `encode_step::resolved_device_name`'s
    /// `cuda_device_name` call unconditionally errors on any
    /// `not(feature = "cuda")` build, aborting the run before a mismatched,
    /// Metal-resolved device name could ever reach this field. See that
    /// function's own doc for the full two-mechanism picture.
    pub device_name: String,
    /// The `JAMMI_KERNELS_DISABLE` op keys this process REQUESTED (sorted;
    /// empty when unset) — `jammi_kernels::admission::disabled_ops_requested`.
    /// PROVENANCE (mirrors `FinetuneStepTier::kernels_disabled_requested`).
    pub kernels_disabled_requested: Vec<String>,
    /// The `JAMMI_KERNELS_DISABLE` op keys that actually FIRED this run
    /// (sorted) — `jammi_kernels::admission::disabled_ops_fired`.
    /// PROVENANCE.
    pub kernels_disabled_fired: Vec<String>,
    /// Whether THIS BUILD compiled the vendored FlashAttention-2 kernels
    /// (`jammi_kernels::admission::FLASH_COMPILED`). PROVENANCE — the
    /// encode/eval path never dispatches flash regardless (fused arms are
    /// training-only), so this records a build fact, not a per-leg
    /// determinant.
    pub flash_compiled: bool,
    /// This tier's own echo of [`Provenance::build_features`]
    /// (`crate::report::build_features`, the SAME function
    /// `Provenance::baked` calls). PROVENANCE.
    pub build_features: Vec<&'static str>,
    /// The mem-efficient-attention op's chunk size, always `None` on this
    /// tier — the encode/eval path has no chunked-attention arm at all
    /// (`mem_efficient_attention.rs`'s own "`chunk_size` is provenance, not
    /// shared identity" doctrine: memeff is training-only, unreferenced
    /// outside `jammi-kernels`'s training call sites). `NullMeans` per
    /// [`Self::PROVENANCE_FIELDS`]'s `chunk_size` entry: `null` here means
    /// "this arm has no chunk size on this surface", never "this producer
    /// predates the field".
    pub chunk_size: Option<u64>,
    /// The attention reference class this leg ran — constant `"eager"` on
    /// this surface (fused arms are training-only), recorded as a
    /// provenance fact rather than a comparison key. See this struct's own
    /// doc for the full identity-forbidden rationale.
    pub attention_arm: String,

    // ── Measurements: recorded references, never gated ──────────────────
    /// Embed serving throughput at the mean measured serve, rows/s. A
    /// machine-dependent reference, mirrors `ModelInferenceTier::embed_rows_per_s`'s
    /// own "coarse code-path net, not the scaling SLO" framing.
    pub embed_rows_per_s: Measurement,
    /// The mean wall-clock of the `iters_measured` measured
    /// `generate_text_embeddings` calls, milliseconds.
    pub embed_serve_ms: Measurement,
}

impl EncodeStepTier {
    /// K7-completeness comparison tuple: the COMPLETE output-affecting
    /// parameter set for the encode surface (unit-62 CONTRACT.md §E3). This
    /// is the WHOLE identity set for this tier (unlike
    /// [`FinetuneStepTier::IDENTITY_FIELDS`]/
    /// [`crate::grad_oracle::GradOracleReport::IDENTITY_FIELDS`], which fold
    /// their own provenance fields in as "K7-completeness additions beyond
    /// the comparison tuple" — CONTRACT.md's E3 deliberately keeps this
    /// tier's provenance OUT of identity entirely; see
    /// [`Self::PROVENANCE_FIELDS`]). `ci/scripts/perf/identity_fields.py`'s
    /// future `ENCODE_IDENTITY_FIELDS` mirrors this list EXACTLY (unit-62
    /// E6, docs-ci domain) — the cardinality (15, round-3 audit F-5'/lead
    /// ruling: `checkpoint_pooling_sha256` + `device_requested` appended
    /// after the original 13, a position-stable addition rather than a
    /// re-order) and every name here is the pinned contract that mirror
    /// parses against.
    ///
    /// `attention_arm` is NOT a member (see this struct's own doc) — a
    /// negative-control test in `encode_step.rs` asserts this mechanically.
    pub const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
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
        // Round-3 audit additions (F-5'(b)/lead ruling), appended
        // position-stable rather than re-ordered into the original 13.
        (
            "checkpoint_pooling_sha256",
            Nullable::NullMeans("no 1_Pooling/config.json in this model dir"),
        ),
        ("device_requested", Nullable::NonNull),
    ];

    /// The provenance fields this tier records but NEVER admits to
    /// [`Self::IDENTITY_FIELDS`] — recorded (with a `NullMeans` reason where
    /// a field can legitimately read `null`) so a downstream reader has the
    /// SAME `assert_identity_fields_present` presence/non-null guarantee on
    /// these fields without them ever being eligible as a cross-leg
    /// comparison key. `chunk_size` is the one `NullMeans` entry (see its
    /// own field doc); every other entry here is `NonNull` (always
    /// populated, even when the value it carries is the empty/constant
    /// case — e.g. `kernels_disabled_requested: []` on an ordinary run).
    pub const PROVENANCE_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("device_name", Nullable::NonNull),
        ("kernels_disabled_requested", Nullable::NonNull),
        ("kernels_disabled_fired", Nullable::NonNull),
        ("flash_compiled", Nullable::NonNull),
        ("build_features", Nullable::NonNull),
        (
            "chunk_size",
            Nullable::NullMeans("this arm has no chunk size on the encode/eval surface"),
        ),
        ("attention_arm", Nullable::NonNull),
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

    /// A minimal, fully-populated [`FinetuneStepTier`] for the serialization
    /// tests below. Field VALUES are arbitrary except where a test itself
    /// varies them (`max_grad_norm`, for the null-vs-value policy tests);
    /// what is pinned is the emitted key SET and the present-vs-omitted /
    /// null-vs-value policy for `max_grad_norm`, so a field added or renamed
    /// on `FinetuneStepTier` is a visible, reviewed diff here rather than a
    /// silent addition/removal a downstream JSON-diffing perf gate would
    /// only notice indirectly.
    fn sample_finetune_step_tier(max_grad_norm: Option<f32>) -> FinetuneStepTier {
        FinetuneStepTier {
            device: "cpu".to_string(),
            device_name: "cpu".to_string(),
            seed: 42,
            backbone_dtype: "f32".to_string(),
            checkpoint_config_sha256: "a".repeat(64),
            checkpoint_weights_sha256: "b".repeat(64),
            checkpoint_weights_size_bytes: 1024,
            batch: 2,
            seq: 6,
            lora_rank: 2,
            lora_alpha: 4.0,
            lora_dropout: 0.0,
            margin: 0.3,
            target_modules: vec!["query".to_string()],
            batched_forward: true,
            max_grad_norm,
            // Dense-leg value: batch=2, seq=6 above, so `[6, 6]` (every row's
            // real length equals `seq`) -- see `FinetuneStepTier::row_lengths`'s
            // own doc.
            row_lengths: vec![6, 6],
            trainable_tensors: 2,
            warmup: 5,
            steps_measured: 1,
            losses: vec![0.5],
            loss_first: 0.5,
            loss_last: 0.5,
            ln_fused_dispatches: 0,
            ln_eager_dispatches: 0,
            rope_fused_dispatches: 0,
            rope_eager_dispatches: 0,
            softmax_fused_dispatches: 0,
            softmax_eager_dispatches: 0,
            geglu_fused_dispatches: 0,
            geglu_eager_dispatches: 0,
            lora_epilogue_fused_dispatches: 0,
            lora_epilogue_eager_dispatches: 0,
            lora_linear_fused_dispatches: 0,
            lora_linear_eager_dispatches: 0,
            attention_block_fused_dispatches: 0,
            attention_block_eager_dispatches: 0,
            adamw_fused_dispatches: 0,
            adamw_eager_dispatches: 0,
            clip_invocations: 0,
            attention_arm: "fused".to_string(),
            attention_block_flash_fused_dispatches: 0,
            attention_block_flash_declined_dispatches: 0,
            flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
            build_features: build_features(),
            kernels_disabled_requested: Vec::new(),
            kernels_disabled_fired: Vec::new(),
            s_per_step_p50: Measurement::measured(0.01, "s"),
            s_per_step_mean: Measurement::measured(0.01, "s"),
            steps_per_s: Measurement::measured(100.0, "steps/s"),
            triplets_per_s: Measurement::measured(200.0, "triplets/s"),
            peak_rss_bytes: Measurement::not_yet_measured("bytes"),
            peak_vram_bytes: Measurement::not_yet_measured("bytes"),
        }
    }

    /// Pins the null-when-absent policy: `max_grad_norm` is the field that
    /// tells a reader whether this row measured the shipped trainer's step
    /// (clip on) or the idealized no-clip step — an OMITTED key would read as
    /// "this report predates the field" (a build-provenance question) rather
    /// than "clipping was off for this row" (a run-provenance fact), so the
    /// key must be present, explicit `null`, when the flag was not supplied.
    #[test]
    fn finetune_step_tier_serializes_null_not_omitted_for_absent_max_grad_norm() {
        let tier = sample_finetune_step_tier(None);
        let value = serde_json::to_value(&tier).expect("serialize FinetuneStepTier");
        let obj = value.as_object().expect("object");
        assert!(
            obj.contains_key("max_grad_norm"),
            "max_grad_norm must be present (as null), not omitted, when absent: {obj:?}"
        );
        assert_eq!(obj["max_grad_norm"], serde_json::Value::Null);
    }

    /// The mirror case: a supplied `--max-grad-norm` serializes as the number,
    /// not as a string or a re-wrapped option shape.
    #[test]
    fn finetune_step_tier_serializes_number_for_present_max_grad_norm() {
        let tier = sample_finetune_step_tier(Some(1.0f32));
        let value = serde_json::to_value(&tier).expect("serialize FinetuneStepTier");
        let obj = value.as_object().expect("object");
        assert_eq!(
            obj["max_grad_norm"],
            serde_json::json!(1.0f32),
            "present max_grad_norm must serialize as the numeric value: {obj:?}"
        );
    }

    /// The full emitted key set, pinned so a field added or renamed on
    /// `FinetuneStepTier` — including the two `attention_block_*_dispatches`
    /// counters the fused whole-attention-block kernel needs for its own
    /// positive-proof channel, the two `adamw_*_dispatches` counters the
    /// fused multi-tensor AdamW kernel needs for the same reason, and
    /// `kernels_disabled_requested` /
    /// `kernels_disabled_fired` (contract K-aux, round 2 / B3: the RESOLVED
    /// `JAMMI_KERNELS_DISABLE` state a downstream A/B harness names the
    /// measured arm from), and `max_grad_norm` (the device-side clip's
    /// on/off flag for this row) — is a visible, reviewed diff here rather
    /// than a silent addition/removal a downstream JSON-diffing perf gate
    /// would only notice indirectly.
    #[test]
    fn finetune_step_tier_emits_the_full_pinned_key_set() {
        let tier = sample_finetune_step_tier(Some(1.0));
        let value = serde_json::to_value(&tier).expect("serialize FinetuneStepTier");
        let obj = value.as_object().expect("object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut expected = vec![
            "adamw_eager_dispatches",
            "adamw_fused_dispatches",
            "attention_arm",
            "attention_block_eager_dispatches",
            "attention_block_flash_declined_dispatches",
            "attention_block_flash_fused_dispatches",
            "attention_block_fused_dispatches",
            "batch",
            "batched_forward",
            "backbone_dtype",
            "build_features",
            "checkpoint_config_sha256",
            "checkpoint_weights_sha256",
            "checkpoint_weights_size_bytes",
            "clip_invocations",
            "device",
            "device_name",
            "flash_compiled",
            "geglu_eager_dispatches",
            "geglu_fused_dispatches",
            "kernels_disabled_fired",
            "kernels_disabled_requested",
            "ln_eager_dispatches",
            "ln_fused_dispatches",
            "lora_alpha",
            "lora_dropout",
            "lora_epilogue_eager_dispatches",
            "lora_epilogue_fused_dispatches",
            "lora_linear_eager_dispatches",
            "lora_linear_fused_dispatches",
            "lora_rank",
            "loss_first",
            "loss_last",
            "losses",
            "margin",
            "max_grad_norm",
            "peak_rss_bytes",
            "peak_vram_bytes",
            "rope_eager_dispatches",
            "rope_fused_dispatches",
            "row_lengths",
            "s_per_step_mean",
            "s_per_step_p50",
            "seed",
            "seq",
            "softmax_eager_dispatches",
            "softmax_fused_dispatches",
            "steps_measured",
            "steps_per_s",
            "target_modules",
            "trainable_tensors",
            "triplets_per_s",
            "warmup",
        ];
        expected.sort_unstable();
        assert_eq!(keys, expected);
    }

    /// Unification contract C3.1/C3.4: every field named in
    /// `FinetuneStepTier::IDENTITY_FIELDS` must actually be present on a
    /// real, serialized tier, and a field declared [`Nullable::NonNull`]
    /// must not read `null`. RED at base: `IDENTITY_FIELDS` does not exist
    /// on `main`, so this test fails to COMPILE (a `const` reference to a
    /// name that is not there), not merely to assert — the strongest RED
    /// shape this repo's own K7-completeness discipline can produce.
    #[test]
    fn finetune_step_identity_fields_are_emitted() {
        let tier = sample_finetune_step_tier(Some(1.0));
        let value = serde_json::to_value(&tier).expect("serialize FinetuneStepTier");
        let obj = value.as_object().expect("object");
        for (field, nullable) in FinetuneStepTier::IDENTITY_FIELDS {
            let entry = obj
                .get(*field)
                .unwrap_or_else(|| panic!("IDENTITY_FIELDS names {field:?}, absent on the tier"));
            if *nullable == Nullable::NonNull {
                assert!(
                    !entry.is_null(),
                    "{field:?} is declared NonNull but serialized as null"
                );
            }
        }
    }

    /// The report-level twin of the test above: `REPORT_IDENTITY_FIELDS`'
    /// three entries must all be present, non-null, under `report.provenance`
    /// of a real `Report`.
    #[test]
    fn report_identity_fields_are_emitted_under_provenance() {
        let report = Report::new("finetune-step", Tiers::default());
        let value = serde_json::to_value(&report).expect("serialize Report");
        let provenance = value
            .get("provenance")
            .and_then(|p| p.as_object())
            .expect("report.provenance object");
        for (field, nullable) in REPORT_IDENTITY_FIELDS {
            let entry = provenance.get(*field).unwrap_or_else(|| {
                panic!("REPORT_IDENTITY_FIELDS names {field:?}, absent under report.provenance")
            });
            if *nullable == Nullable::NonNull {
                assert!(
                    !entry.is_null(),
                    "provenance.{field} is declared NonNull but serialized as null"
                );
            }
        }
    }

    /// A minimal, fully-populated [`FinetuneRunTier`] for the tests below.
    fn sample_finetune_run_tier() -> FinetuneRunTier {
        FinetuneRunTier {
            seed: 42,
            batch: 4,
            seq: 64,
            lora_rank: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.05,
            margin: Some(0.3),
            target_modules: vec!["Wqkv".to_string()],
            backbone_dtype: "f32".to_string(),
            checkpoint_config_sha256: "a".repeat(64),
            checkpoint_weights_sha256: "b".repeat(64),
            checkpoint_weights_size_bytes: 1024,
            max_grad_norm: Some(1.0),
            warmup: None,
            row_lengths: None,
            epochs: 2,
            lr: 2e-4,
            schedule: "constant".to_string(),
            warmup_steps: 0,
            weight_decay: 0.01,
            grad_accum: 1,
            validation_fraction: 0.1,
            train_pairs_file_sha256: "c".repeat(64),
            heldout_ids_sha256: "d".repeat(64),
            heldout_pairs_sha256: "f".repeat(64),
            heldout_batch_partition_sha256: "e".repeat(64),
            embedding_loss: "triplet".to_string(),
            temperature: None,
            matryoshka_dims: Vec::new(),
            early_stopping_patience: 10_000,
            early_stopping_metric: "val_loss".to_string(),
            eval_cadence: 1,
            arm: "fused".to_string(),
            device_name: "cpu".to_string(),
            kernels_disabled_requested: Vec::new(),
            kernels_disabled_fired: Vec::new(),
            flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
            build_features: build_features(),
            attention_arm: "fused".to_string(),
            split_rule: "positional_fraction_split".to_string(),
            batched_forward: true,
            steps_measured: 3,
            ln_fused_dispatches: 0,
            ln_eager_dispatches: 0,
            rope_fused_dispatches: 0,
            rope_eager_dispatches: 0,
            softmax_fused_dispatches: 0,
            softmax_eager_dispatches: 0,
            geglu_fused_dispatches: 0,
            geglu_eager_dispatches: 0,
            lora_epilogue_fused_dispatches: 0,
            lora_epilogue_eager_dispatches: 0,
            lora_linear_fused_dispatches: 0,
            lora_linear_eager_dispatches: 0,
            attention_block_fused_dispatches: 3,
            attention_block_eager_dispatches: 0,
            adamw_fused_dispatches: 3,
            adamw_eager_dispatches: 0,
            attention_block_flash_fused_dispatches: 0,
            attention_block_flash_declined_dispatches: 0,
            admission_is_dense: false,
            tie_fraction: 0.0,
            final_epoch: 1,
            held_out_example_mean: 0.5,
            held_out_count: 4,
            final_loss_diagnostic: 0.5,
            trajectory: vec![EpochHeldOut {
                epoch: 1,
                held_out_mean: 0.5,
                held_out_tie_fraction: 0.0,
                held_out_batch_partition_sha256: "e".repeat(64),
            }],
            train_probe_series: vec![0.6, 0.55, 0.5],
            mutant_id: None,
            mutant_base_sha: None,
            mutant_patch_sha256: None,
        }
    }

    /// The MNRL twin of [`sample_finetune_run_tier`] (unit 63 H4a-delta,
    /// CONTRACT amendment 2026-08-28): the SAME sample values, except
    /// `margin`/`temperature` swap which one is `Some` and
    /// `embedding_loss` reads `"mnrl"` — exactly the flip
    /// [`crate::finetune_run::Objective::Mnrl`] produces on a real run.
    /// `20.0` is `MultipleNegativesRanking`'s standard default temperature
    /// (`jammi_wire::fine_tune::EmbeddingLoss::MultipleNegativesRanking`'s
    /// own doc).
    fn sample_finetune_run_tier_mnrl() -> FinetuneRunTier {
        FinetuneRunTier {
            margin: None,
            embedding_loss: "mnrl".to_string(),
            temperature: Some(20.0),
            ..sample_finetune_run_tier()
        }
    }

    /// Identity-value semantics per objective (unit 63 H4a-delta, task item
    /// 4): a Triplet-objective tier reads `margin: Some(_)`, `temperature:
    /// null`, `embedding_loss: "triplet"`.
    #[test]
    fn finetune_run_tier_triplet_objective_has_margin_nonnull_temperature_null() {
        let tier = sample_finetune_run_tier();
        assert!(tier.margin.is_some(), "Triplet run must report a margin");
        assert!(
            tier.temperature.is_none(),
            "Triplet run must report temperature: null"
        );
        assert_eq!(tier.embedding_loss, "triplet");
    }

    /// The MNRL mirror of the test above: `margin: null`, `temperature:
    /// Some(_)`, `embedding_loss: "mnrl"` — the nullness genuinely FLIPS
    /// between the two objectives, not merely "one of them happens to be
    /// null on this sample".
    #[test]
    fn finetune_run_tier_mnrl_objective_has_temperature_nonnull_margin_null() {
        let tier = sample_finetune_run_tier_mnrl();
        assert!(tier.margin.is_none(), "MNRL run must report margin: null");
        assert!(
            tier.temperature.is_some(),
            "MNRL run must report a temperature"
        );
        assert_eq!(tier.embedding_loss, "mnrl");
    }

    /// The MNRL sample must ALSO satisfy `IDENTITY_FIELDS`/`PROVENANCE_FIELDS`
    /// presence (the same self-check `finetune_run::run` performs before
    /// returning) — `margin: null` under `Nullable::NullMeans("objective is
    /// mnrl")` must NOT trip the `NonNull` panic branch now that `margin`'s
    /// own declared nullability changed (H4a-delta).
    #[test]
    fn finetune_run_tier_mnrl_sample_satisfies_identity_and_provenance_presence() {
        let tier = sample_finetune_run_tier_mnrl();
        let value = serde_json::to_value(&tier).expect("serialize MNRL FinetuneRunTier");
        assert_identity_fields_present(&value, FinetuneRunTier::IDENTITY_FIELDS);
        assert_identity_fields_present(&value, FinetuneRunTier::PROVENANCE_FIELDS);
    }

    /// Cardinality pin: `FinetuneStepTier`'s 18 minus `attention_arm` (17)
    /// plus the 18 new CONTRACT H4 fields, minus the unit-63 adversarial-
    /// audit finding-5(c)/advisory-(d) reclassifications (`split_rule`,
    /// `split_seed`, `batched_forward`, `steps_measured` — 4 removed), plus
    /// `heldout_pairs_sha256` (finding 5(a), 1 added) = 17 + 18 − 4 + 1 = 32.
    #[test]
    fn finetune_run_tier_identity_fields_cardinality_is_32() {
        assert_eq!(FinetuneRunTier::IDENTITY_FIELDS.len(), 32);
    }

    /// `PROVENANCE_FIELDS` carries `arm` + `attention_arm` (moved out of
    /// identity — see struct doc) plus the four fields every other tier's
    /// provenance carries (`device_name`, `kernels_disabled_requested`,
    /// `kernels_disabled_fired`, `flash_compiled`, `build_features`), plus
    /// the three unit-63 finding-5(c)/advisory-(d) reclassifications
    /// (`split_rule`, `batched_forward`, `steps_measured`) = 10.
    #[test]
    fn finetune_run_tier_provenance_fields_cardinality_is_10() {
        assert_eq!(FinetuneRunTier::PROVENANCE_FIELDS.len(), 10);
    }

    /// Unit 63 round-7 audit, finding 1: the three mutant-provenance fields
    /// are honest-labeling, NOT identity or provenance (struct doc, (e)) —
    /// pinning their absence from BOTH comparison consts so a future edit
    /// that reflexively adds a new field to one of these tuples cannot
    /// silently sweep the mutant fields in with it.
    #[test]
    fn mutant_fields_are_absent_from_both_identity_and_provenance_tuples() {
        for (field, _) in FinetuneRunTier::IDENTITY_FIELDS {
            assert!(
                !field.starts_with("mutant_"),
                "{field:?} is a mutant-provenance field but appears in IDENTITY_FIELDS"
            );
        }
        for (field, _) in FinetuneRunTier::PROVENANCE_FIELDS {
            assert!(
                !field.starts_with("mutant_"),
                "{field:?} is a mutant-provenance field but appears in PROVENANCE_FIELDS"
            );
        }
    }

    /// A normal (non-mutant) leg's `mutant_id`/`mutant_base_sha`/
    /// `mutant_patch_sha256` are all `None`, and `#[serde(skip_serializing_if
    /// = "Option::is_none")]` must OMIT all three keys entirely from the
    /// emitted JSON (never emit them as explicit `null`s) — the exact
    /// "normal legs' JSON is byte-unchanged" guarantee finding 1 requires,
    /// so a committed golden built before this change is unaffected.
    #[test]
    fn mutant_fields_are_omitted_entirely_when_none() {
        let tier = sample_finetune_run_tier();
        assert!(tier.mutant_id.is_none());
        let value = serde_json::to_value(&tier).expect("serialize FinetuneRunTier");
        let obj = value.as_object().expect("object");
        for field in ["mutant_id", "mutant_base_sha", "mutant_patch_sha256"] {
            assert!(
                !obj.contains_key(field),
                "{field:?} must be OMITTED (not merely null) when None, got {:?}",
                obj.get(field)
            );
        }
    }

    /// A fully-labeled mutant leg emits all three keys as plain strings —
    /// the shape `ci/scripts/perf/ab_merge.py`'s mutant-dose-ladder mode
    /// reads by these exact key names.
    #[test]
    fn mutant_fields_are_emitted_when_all_present() {
        let tier = FinetuneRunTier {
            mutant_id: Some("eps-0.10".to_string()),
            mutant_base_sha: Some("f".repeat(40)),
            mutant_patch_sha256: Some("a".repeat(64)),
            ..sample_finetune_run_tier()
        };
        let value = serde_json::to_value(&tier).expect("serialize FinetuneRunTier");
        let obj = value.as_object().expect("object");
        assert_eq!(obj["mutant_id"], serde_json::json!("eps-0.10"));
        assert_eq!(obj["mutant_base_sha"], serde_json::json!("f".repeat(40)));
        assert_eq!(
            obj["mutant_patch_sha256"],
            serde_json::json!("a".repeat(64))
        );
    }

    /// DISJOINTNESS: `IDENTITY_FIELDS` and `PROVENANCE_FIELDS` share no
    /// field name — E3's convention (see struct doc), never
    /// `FinetuneStepTier`'s superset-that-includes-provenance one. A field
    /// appearing in both would be a merger-facing ambiguity: is it a
    /// comparison key or not?
    #[test]
    fn finetune_run_tier_identity_and_provenance_are_disjoint() {
        let identity: std::collections::HashSet<&str> = FinetuneRunTier::IDENTITY_FIELDS
            .iter()
            .map(|(f, _)| *f)
            .collect();
        let provenance: std::collections::HashSet<&str> = FinetuneRunTier::PROVENANCE_FIELDS
            .iter()
            .map(|(f, _)| *f)
            .collect();
        let overlap: Vec<&&str> = identity.intersection(&provenance).collect();
        assert!(
            overlap.is_empty(),
            "IDENTITY_FIELDS and PROVENANCE_FIELDS overlap on {overlap:?} — a field cannot be \
             both a comparison key and provenance-only"
        );
    }

    /// `arm` and `attention_arm` are explicitly ABSENT from
    /// `IDENTITY_FIELDS` — the negative-control half of the disjointness
    /// pin above, naming the two fields the deviation from
    /// `FinetuneStepTier`'s convention is ABOUT (see struct doc: "the arm
    /// is provenance, never identity").
    #[test]
    fn finetune_run_tier_arm_and_attention_arm_are_not_identity() {
        let identity_names: Vec<&str> = FinetuneRunTier::IDENTITY_FIELDS
            .iter()
            .map(|(f, _)| *f)
            .collect();
        assert!(!identity_names.contains(&"arm"));
        assert!(!identity_names.contains(&"attention_arm"));
    }

    /// Unification contract C3.1 (this tier's own instance): every field
    /// named in `FinetuneRunTier::IDENTITY_FIELDS` and
    /// `FinetuneRunTier::PROVENANCE_FIELDS` must be present on a real,
    /// serialized tier, and a field declared `NonNull` must not read
    /// `null`.
    #[test]
    fn finetune_run_tier_identity_and_provenance_fields_are_emitted() {
        let tier = sample_finetune_run_tier();
        let value = serde_json::to_value(&tier).expect("serialize FinetuneRunTier");
        assert_identity_fields_present(&value, FinetuneRunTier::IDENTITY_FIELDS);
        assert_identity_fields_present(&value, FinetuneRunTier::PROVENANCE_FIELDS);
    }

    /// Determinism of the identity tuple across two constructions: building
    /// two `FinetuneRunTier`s from the SAME inputs (here, two identical
    /// calls to the sample builder — the tier-level analogue of the seam's
    /// own `two_calls_are_bitwise_identical` test) must serialize every
    /// `IDENTITY_FIELDS` entry to the SAME JSON value.
    #[test]
    fn finetune_run_tier_identity_tuple_is_deterministic_across_two_constructions() {
        let a = serde_json::to_value(sample_finetune_run_tier()).expect("serialize a");
        let b = serde_json::to_value(sample_finetune_run_tier()).expect("serialize b");
        for (field, _) in FinetuneRunTier::IDENTITY_FIELDS {
            assert_eq!(
                a.get(*field),
                b.get(*field),
                "identity field {field:?} differs across two constructions from identical inputs"
            );
        }
    }

    /// Round-2 audit (advisory A4): neither `IDENTITY_FIELDS` const has a
    /// field typed `Option` today, so the `NonNull`-declared-but-null
    /// PANIC branch inside `assert_identity_fields_present` was reachable
    /// in principle but never actually exercised by any existing test —
    /// and `Nullable::NullMeans` is genuinely unconstructed on this branch
    /// (contract C3.4: `max_grad_norm` enters only once PR #381 lands).
    /// `assert_identity_fields_present` is generic over `serde_json::Value`
    /// precisely so BOTH lattice arms can be exercised against a synthetic
    /// fixture object, without needing a real struct field typed `Option`
    /// (report.rs's own `device_name: String` — the field the round-2
    /// brief's literal "device_name None" suggestion named — is NOT
    /// constructible as `None`; this synthetic-`Value` fixture is what
    /// makes the branch testable without changing that field's type).
    #[test]
    fn assert_identity_fields_present_covers_both_nullable_arms() {
        // Arm 1: a NonNull-declared field that serialized as JSON `null`
        // MUST panic — this is the class B2/B4 exist to prevent (a field
        // that silently reads null with no declared meaning).
        let null_nonnull_fields: &[(&str, Nullable)] = &[("widget", Nullable::NonNull)];
        let result = std::panic::catch_unwind(|| {
            let value = serde_json::json!({ "widget": null });
            assert_identity_fields_present(&value, null_nonnull_fields);
        });
        assert!(
            result.is_err(),
            "a NonNull field serialized as null must panic — it did not"
        );

        // Arm 2: a NullMeans-declared field that serialized as JSON `null`
        // must NOT panic — this is the exact shape `max_grad_norm: null`
        // ("no clip") will take once PR #381 lands.
        let null_nullmeans_fields: &[(&str, Nullable)] =
            &[("widget", Nullable::NullMeans("no widget configured"))];
        let value = serde_json::json!({ "widget": null });
        assert_identity_fields_present(&value, null_nullmeans_fields); // must not panic

        // Control: a NonNull field that is genuinely present and non-null
        // passes cleanly (the "both fields present" baseline every other
        // call site in this module relies on).
        let present_fields: &[(&str, Nullable)] = &[("widget", Nullable::NonNull)];
        let value = serde_json::json!({ "widget": "present" });
        assert_identity_fields_present(&value, present_fields); // must not panic

        // Negative control on the assertion mechanism itself: an ABSENT
        // field (never even the key `null`) must ALSO panic, distinctly
        // from the null-but-present case above — `unwrap_or_else` in
        // `assert_identity_fields_present`'s own body is what fires here.
        let missing_field_fields: &[(&str, Nullable)] = &[("absent_field", Nullable::NonNull)];
        let result = std::panic::catch_unwind(|| {
            let value = serde_json::json!({ "other": "value" });
            assert_identity_fields_present(&value, missing_field_fields);
        });
        assert!(
            result.is_err(),
            "a field absent from the object entirely must panic — it did not"
        );

        // Round-3 audit (advisory 5): the two lattice cells the round-2
        // test above didn't cover — NullMeans×absent and NullMeans×present.
        // `Nullable` only ever discriminates NULL-vs-non-null; it says
        // NOTHING about whether the KEY may be omitted entirely — a
        // NullMeans field absent from the object is the SAME finding an
        // absent NonNull field is (contract C6.3's "presence for all"
        // half applies to every declared field regardless of nullability;
        // only the "non-null only for NonNull" half is nullability-gated).
        let nullmeans_fields: &[(&str, Nullable)] =
            &[("widget", Nullable::NullMeans("no widget configured"))];
        let result = std::panic::catch_unwind(|| {
            let value = serde_json::json!({ "other": "value" });
            assert_identity_fields_present(&value, nullmeans_fields);
        });
        assert!(
            result.is_err(),
            "a NullMeans field absent from the object entirely must STILL panic (presence is \
             required regardless of nullability) — it did not"
        );

        // NullMeans×present: a NullMeans field that is present AND
        // genuinely non-null (the `nvidia_driver_version` reading on a real
        // CUDA box, say) must pass cleanly — NullMeans widens what is
        // ACCEPTED, it never forbids a real value.
        let value = serde_json::json!({ "widget": "a real value, not null" });
        assert_identity_fields_present(&value, nullmeans_fields); // must not panic

        // Round-3 audit (advisory 5): an empty STRING on a NonNull field
        // must panic — `""` is not JSON `null`, so the null-only check
        // alone would have let this through (the exact shape `build.rs`'s
        // round-2 advisory A3 bug baked before it was fixed at the
        // source: `TARGET`/`PROFILE` defaulting to `""`).
        let nonnull_string_fields: &[(&str, Nullable)] = &[("widget", Nullable::NonNull)];
        let result = std::panic::catch_unwind(|| {
            let value = serde_json::json!({ "widget": "" });
            assert_identity_fields_present(&value, nonnull_string_fields);
        });
        assert!(
            result.is_err(),
            "a NonNull field serialized as an empty string must panic — it did not"
        );
        // Control: a non-string NonNull field (e.g. a number `0`) is NOT
        // caught by the empty-string check — `0` is a legitimate NonNull
        // value, never confused with `""`.
        let value = serde_json::json!({ "widget": 0 });
        assert_identity_fields_present(&value, nonnull_string_fields); // must not panic
    }

    /// Cross-language pin of the ONE shared identity declaration (PR #381
    /// audit B1): every name in `ci/scripts/perf/identity_fields.py`'s
    /// `FINETUNE_IDENTITY_FIELDS` tuple must be a key `FinetuneStepTier`
    /// actually serializes, read back out of THAT FILE — never a second
    /// hand-kept list here that could drift from it (the exact drift the
    /// audit found: `ab_merge.py`'s own tuple lacked `max_grad_norm`, so a
    /// clip-on jammi leg merged against a clip-off torch leg and PASSED).
    /// `ab_merge.leg_premise_violations` refuses a leg MISSING any member,
    /// so a member this struct does not emit would make every real A/B row
    /// INVALID; this test makes that a compile-time-adjacent failure
    /// instead of a pod-time one. The torch producer's side of the same
    /// pin is `test_ab_merge.py`'s `SharedIdentityDeclarationTests`.
    ///
    /// The tuple is parsed with a deliberately narrow scanner (the literal
    /// `FINETUNE_IDENTITY_FIELDS = (` ... `)` block, one double-quoted name
    /// per entry, `#` comments skipped) so a reshaped declaration fails
    /// loudly here (zero names parsed → panic) rather than silently pinning
    /// nothing — the execution-provenance rule that zero-matched is red.
    #[test]
    fn finetune_step_tier_emits_every_shared_identity_field() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<name>")
            .parent()
            .expect("workspace root")
            .join("ci")
            .join("scripts")
            .join("perf")
            .join("identity_fields.py");
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let start = src
            .find("FINETUNE_IDENTITY_FIELDS = (")
            .expect("identity_fields.py must declare `FINETUNE_IDENTITY_FIELDS = (`");
        let body = &src[start + "FINETUNE_IDENTITY_FIELDS = (".len()..];
        let end = body
            .find("\n)")
            .expect("FINETUNE_IDENTITY_FIELDS tuple must close with a `)` on its own line");
        let mut declared: Vec<String> = Vec::new();
        for line in body[..end].lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let name = line.trim_end_matches(',').trim_matches('"').to_string();
            assert!(
                line.starts_with('"') && line.ends_with("\","),
                "unrecognised FINETUNE_IDENTITY_FIELDS entry shape: {line:?}"
            );
            declared.push(name);
        }
        assert!(
            declared.len() >= 14,
            "parsed only {} identity names from {} — the scanner or the declaration changed shape",
            declared.len(),
            path.display()
        );
        for required in ["max_grad_norm", "attention_arm"] {
            assert!(
                declared.iter().any(|n| n == required),
                "{required} must be a shared identity field (PR #381 audit B1)"
            );
        }

        let tier = sample_finetune_step_tier(Some(1.0));
        let value = serde_json::to_value(&tier).expect("serialize FinetuneStepTier");
        let obj = value.as_object().expect("object");
        let missing: Vec<&String> = declared
            .iter()
            .filter(|n| !obj.contains_key(n.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "FinetuneStepTier does not emit shared identity field(s) {missing:?} declared in {}",
            path.display()
        );
    }
}
