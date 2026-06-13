//! The machine-readable report schema the harness emits.
//!
//! One JSON document per invocation, shaped so a downstream perf-gate can diff
//! a fresh run against committed goldens without parsing free text. The shape
//! covers every planned measurement tier up front — embed throughput, search
//! QPS, recall@k, propagate latency, peak RSS — so the schema is stable from
//! the first emit; tiers not yet measured serialize an explicit
//! [`Measurement::not_yet_measured`] marker rather than a zero that a gate
//! could mistake for a real datapoint.

use serde::Serialize;

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
    /// The measured tiers. Each tier is a named bag of measurements; a tier
    /// not exercised by this subcommand is omitted entirely (absent, not null).
    pub tiers: Tiers,
}

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
    /// The realistic quality tier (committed corpus): embed throughput, search
    /// QPS, recall@k, propagate latency, peak RSS. Not yet measured in this
    /// harness — present as a stub so the schema is stable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arxiv: Option<ArxivTier>,
    /// The binding-memory tier: the streamed exact-search RSS proof and its
    /// negative control. Populated by `search-rss`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binding: Option<BindingTier>,
}

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
    /// Recall of ANN-over-frozen-index vs the committed exact ground truth.
    pub recall_at_10: Measurement,
    /// Neighbor-graph propagation latency, milliseconds.
    pub propagate_latency_ms: Measurement,
    /// Process peak resident set, mebibytes.
    pub peak_rss_mib: Measurement,
}

impl ArxivTier {
    /// Every metric stubbed `not yet measured` — the schema exists, the numbers
    /// land in a later PR.
    pub fn stub() -> Self {
        Self {
            embed_per_s: Measurement::not_yet_measured("rows_per_s"),
            search_qps_exact: Measurement::not_yet_measured("queries_per_s"),
            search_qps_ann: Measurement::not_yet_measured("queries_per_s"),
            recall_at_10: Measurement::not_yet_measured("fraction"),
            propagate_latency_ms: Measurement::not_yet_measured("ms"),
            peak_rss_mib: Measurement::not_yet_measured("mib"),
        }
    }
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
}
