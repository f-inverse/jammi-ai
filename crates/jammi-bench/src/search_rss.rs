//! The bounded-RSS proof for streamed exact vector search.
//!
//! W2 rewrote `exact_vector_search` to stream a DataFusion scan through a
//! bounded top-`k` heap, replacing the old collect-every-vector pass. The bit-
//! identical equivalence is proven inside the engine; what that proof cannot
//! show — because it runs in-process under a hermetic test — is that the
//! streamed path actually *holds* a flat resident set while the old path's RSS
//! grows with the corpus. This module is that out-of-process proof.
//!
//! The shape of the proof:
//!
//! * A **negative control** — the old `O(N·d)` collect-all path, re-implemented
//!   here (it no longer exists in the engine) so the assertion has teeth: if the
//!   streamed path were secretly unbounded, the control would still grow, and a
//!   "both flat" outcome would correctly *fail* the proof. RC1: an RSS assertion
//!   must be able to fail.
//! * Two corpus sizes, `N₁ < N₂`, chosen so the naive control's growth is
//!   plainly visible (`N·d·4` bytes of vectors held at once) while staying well
//!   under the box's RAM. The assertion: the streamed delta stays under a small
//!   epsilon (flat), the naive delta exceeds a floor and tracks `N·d·4` (linear).
//!
//! ## Why each measurement runs in its own process
//!
//! RSS is sampled from `/proc/self/status` `VmHWM`, the kernel's whole-process
//! high-water mark. `VmHWM` is *monotonic* — it only ever rises and cannot be
//! reset from userspace — so two allocations of different size measured in one
//! process would both read the larger of the two: the high-water set by an
//! earlier multi-GiB naive run would contaminate every later streamed sample.
//!
//! The proof therefore measures each `(corpus_size, variant)` pair in a **fresh
//! child process** (`measure-once`, re-exec of this binary): each child starts
//! with a clean `VmHWM`, allocates only its own variant's working set, samples
//! its own peak, and reports it. The parent orchestrates the four children,
//! materializing each corpus on disk once and pointing both variants at the
//! identical Parquet so the contrast is over the same data. A per-process peak
//! is the reliable RSS number the design calls for.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use arrow::array::{Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use tempfile::tempdir;
use tokio::process::Command;

use jammi_db::index::exact::exact_vector_search;
use jammi_db::store::vectors::extend_with_fixed_size_list_f32;
use jammi_numerics::distance::cosine_distance;

use crate::report::{BindingTier, RssAssertion, RssPoint};
use crate::rss::{active_source, proc_peak_rss_mib};

/// The corpus dimensionality the proof runs at — the design's binding shape.
const DIM: usize = 384;

/// The two corpus sizes. The naive control holds `N·d·4` bytes of `f32` vectors
/// at once: at d=384 that is ~1.5 KiB/row, so 2M rows ≈ 2.9 GiB and 4M rows ≈
/// 5.9 GiB of vectors — a ~3 GiB swing that is unmistakable against a small
/// epsilon, yet leaves the 30 GiB box ample headroom (the box never approaches
/// OOM during the proof). The streamed path, holding only one batch plus the
/// `k` retained pairs, must stay flat across the same swing.
const ROWS_SMALL: usize = 2_000_000;
const ROWS_LARGE: usize = 4_000_000;

/// Rows per `RecordBatch` when writing and scanning the corpus. Many batches
/// force the streamed path to fold across batch boundaries — the per-batch drop
/// that bounds its footprint only bites when the scan yields more than one
/// batch.
const BATCH_ROWS: usize = 8_192;

/// Neighbors retrieved. Small relative to `N` so the streamed path's `O(k)`
/// retained set is negligible and the contrast isolates the `O(N·d)` vs
/// `O(batch·d)` difference.
const K: usize = 10;

/// LCG seed for the corpus. Distinct seed for the query so the query is not a
/// corpus row.
const CORPUS_SEED: u64 = 0x5151_2323_4747_8989;
const QUERY_SEED: u64 = 0x0102_0304_0506_0708;

/// The streamed search's footprint *above the scan baseline* must not grow by
/// more than this between `N₁` and `N₂`. The bounded heap holds only `k`
/// `(row_id, dist)` pairs and the per-batch vectors, so its overhead is `O(k +
/// batch·d)` — independent of `N`, and its delta across the two sizes should be
/// essentially zero. The epsilon is generous slack for `VmHWM` page granularity
/// and per-child allocator jitter, yet an order of magnitude below the naive
/// control's multi-GiB overhead swing — narrow enough that an actually-unbounded
/// accumulator would breach it.
const STREAMED_FLAT_EPSILON_MIB: f64 = 128.0;

/// The naive control must grow by at least this between `N₁` and `N₂` to count
/// as "grows with N". Set well below the ~3 GiB the `N·d·4` model predicts so
/// the floor is a clear lower bound, not a tight fit.
const NAIVE_GROWTH_FLOOR_MIB: f64 = 1_500.0;

/// Minimum free disk (MiB) on the corpus's filesystem before the proof will
/// run. Each corpus is ~`N·d·4` of uncompressed Parquet (LCG noise is
/// incompressible) and the parent holds at most one at a time; refuse below a
/// margin comfortably above the larger corpus's ~5.9 GiB.
const MIN_FREE_DISK_MIB: u64 = 20_000;

/// Which path a `measure-once` child exercises.
///
/// Three variants because the raw peak RSS of a search includes DataFusion's
/// own parquet-reader machinery (decompression buffers, batch prefetch, scan
/// metadata) whose footprint has a mild N-dependence of its own — independent
/// of the search algorithm under test. [`Variant::ScanOnly`] measures exactly
/// that reader baseline: stream the same scan and drop every batch unscored. The
/// proof then isolates the *search accumulator's* footprint as `search − scan`,
/// so the assertion is over the engine's algorithm, not the reader it rides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Drain the scan without scoring — DataFusion's parquet-reader baseline,
    /// the footprint both searches share and neither owns.
    ScanOnly,
    /// The streamed engine `exact_vector_search` — bounded accumulator,
    /// `O(k + batch·d)` above the scan baseline.
    Streamed,
    /// The bench-only naive collect-all baseline — unbounded accumulator,
    /// `O(N·d)` above the scan baseline.
    Naive,
}

impl Variant {
    /// The CLI token for this variant, used on the `measure-once` child command.
    pub fn as_str(self) -> &'static str {
        match self {
            Variant::ScanOnly => "scan-only",
            Variant::Streamed => "streamed",
            Variant::Naive => "naive",
        }
    }

    /// Parse the CLI token back into a variant.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "scan-only" => Ok(Variant::ScanOnly),
            "streamed" => Ok(Variant::Streamed),
            "naive" => Ok(Variant::Naive),
            other => Err(format!(
                "unknown variant {other:?}, expected scan-only|streamed|naive"
            )),
        }
    }
}

/// Drain the DataFusion scan one batch at a time, touching each batch's columns
/// the way a real search does (cast `_row_id`, materialize the batch's vectors)
/// but accumulating *nothing* across batches. This is the reader baseline: the
/// resident footprint of the scan itself, which both searches pay and neither
/// search algorithm owns. The vectors `Vec` is reused and cleared per batch, so
/// only one batch is ever resident — exactly the streamed search's per-batch
/// handling, minus the bounded heap.
async fn scan_only_drain(
    ctx: &SessionContext,
    table_name: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let df = ctx
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{table_name}\""
        ))
        .await?;
    let mut stream = df.execute_stream().await?;

    let mut seen = 0usize;
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        let row_ids_col = batch
            .column_by_name("_row_id")
            .ok_or("missing _row_id in scan-only drain")?;
        let row_ids_utf8 = cast(row_ids_col, &DataType::Utf8)?;
        let ids = row_ids_utf8
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("_row_id is not a Utf8-castable string type")?;
        vectors.clear();
        extend_with_fixed_size_list_f32(&batch, table_name, "vector", &mut vectors)?;
        // Touch the data so the scan and per-batch materialization are not
        // optimized away, without retaining anything across the loop.
        seen += ids.len().min(vectors.len());
    }
    Ok(seen)
}

/// A bench-only re-implementation of the *old* collect-all exact search — the
/// negative control. It materializes every vector in the corpus into a single
/// `Vec<Vec<f32>>` before scoring, exactly the `O(N·d)` pass W2 removed from the
/// engine. Its result is bit-identical to the streamed path (proven in-engine);
/// here only its *resident footprint* is the point.
///
/// This deliberately duplicates the engine's pre-streaming shape rather than
/// calling the engine — the engine no longer has this path, and the proof needs
/// the unbounded baseline to exist somewhere to drive RSS against.
async fn naive_collect_all_search(
    ctx: &SessionContext,
    table_name: &str,
    query: &[f32],
    k: usize,
) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
    let df = ctx
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{table_name}\""
        ))
        .await?;
    let mut stream = df.execute_stream().await?;

    // The unbounded accumulation: every row's id and vector held at once.
    let mut row_ids: Vec<String> = Vec::new();
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        let row_ids_col = batch
            .column_by_name("_row_id")
            .ok_or("missing _row_id in naive baseline scan")?;
        let row_ids_utf8 = cast(row_ids_col, &DataType::Utf8)?;
        let ids = row_ids_utf8
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("_row_id is not a Utf8-castable string type")?;
        for i in 0..ids.len() {
            row_ids.push(ids.value(i).to_string());
        }
        extend_with_fixed_size_list_f32(&batch, table_name, "vector", &mut vectors)?;
    }

    let mut scored: Vec<(String, f32)> = row_ids
        .into_iter()
        .zip(vectors.iter())
        .map(|(id, v)| (id, cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    Ok(scored)
}

/// A single fingerprint over a search result, so the parent can confirm the
/// streamed and naive children agreed without shipping the whole result list
/// across the process boundary. Concatenates `(row_id, distance bits)` in order.
fn result_digest(result: &[(String, f32)]) -> String {
    let mut s = String::new();
    for (id, dist) in result {
        s.push_str(id);
        s.push(':');
        s.push_str(&dist.to_bits().to_string());
        s.push(';');
    }
    s
}

/// The child entrypoint: run one `variant` over the pre-materialized corpus at
/// `corpus_path` (already holding `rows` rows of width [`DIM`]), sample this
/// process's own fresh peak RSS, and return it with a digest of the result.
///
/// Runs in a process whose `VmHWM` started clean, so the sampled peak reflects
/// only this variant's working set — the contamination a single in-process
/// measurement would suffer is structurally avoided.
pub async fn measure_once(
    variant: Variant,
    rows: usize,
    corpus_path: &Path,
) -> Result<(f64, String), Box<dyn std::error::Error>> {
    let url = crate::corpus::storage_url(corpus_path)?;
    let table = "bench_corpus";
    let ctx = crate::corpus::register(&url, table).await?;
    let query = crate::corpus::lcg_query(QUERY_SEED, DIM);

    // The scan-only baseline produces no result to compare — its digest is the
    // sentinel `-`. The two searches produce a result whose digest the parent
    // cross-checks so the contrast is provably over identical neighbours.
    let digest = match variant {
        Variant::ScanOnly => {
            let seen = scan_only_drain(&ctx, table).await?;
            if seen == 0 {
                return Err(format!("scan-only over {rows} rows saw no rows").into());
            }
            "-".to_string()
        }
        Variant::Streamed | Variant::Naive => {
            let result = match variant {
                Variant::Streamed => exact_vector_search(&ctx, table, &query, K).await?,
                Variant::Naive => naive_collect_all_search(&ctx, table, &query, K).await?,
                Variant::ScanOnly => unreachable!("scan-only handled above"),
            };
            if result.len() != K {
                return Err(format!(
                    "{} search over {rows} rows returned {} results, expected k={K}",
                    variant.as_str(),
                    result.len()
                )
                .into());
            }
            result_digest(&result)
        }
    };
    let rss = proc_peak_rss_mib()?;
    Ok((rss, digest))
}

/// What a `measure-once` child prints to stdout: its peak RSS and result digest,
/// as two whitespace-separated fields the parent parses. A tiny, line-oriented
/// contract avoids a JSON dependency on the hot path between the processes.
fn format_child_line(rss_mib: f64, digest: &str) -> String {
    format!("{rss_mib} {digest}")
}

/// Parse a `measure-once` child's stdout line back into `(rss_mib, digest)`.
fn parse_child_line(line: &str) -> Result<(f64, String), Box<dyn std::error::Error>> {
    let line = line.trim();
    let (rss, digest) = line
        .split_once(' ')
        .ok_or("malformed measure-once output: expected '<rss> <digest>'")?;
    Ok((rss.parse::<f64>()?, digest.to_string()))
}

/// Print the child result line to stdout. Called by the `measure-once`
/// subcommand handler in `main` after [`measure_once`].
pub fn emit_child_result(rss_mib: f64, digest: &str) {
    println!("{}", format_child_line(rss_mib, digest));
}

/// Free disk space in MiB on the filesystem backing `path`, via `statvfs`.
fn free_disk_mib(path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())?;
    // SAFETY: `c_path` is a valid NUL-terminated C string for the lifetime of
    // the call, and `stat` is written only by `statvfs` on success.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return Err(format!("statvfs failed for {}", path.display()).into());
    }
    let bytes = stat.f_bavail * stat.f_frsize;
    Ok(bytes / (1024 * 1024))
}

/// Spawn a `measure-once` child of this same binary for one `(variant, rows)`
/// over `corpus_path`, returning its measured peak RSS and result digest.
///
/// Re-execs the current executable so the measurement runs with a fresh
/// `VmHWM`. The child's stderr is inherited so a failure surfaces in the parent
/// run's logs; only its single stdout line carries the measurement.
async fn spawn_measure(
    variant: Variant,
    rows: usize,
    corpus_path: &Path,
) -> Result<(f64, String), Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let output = Command::new(exe)
        .arg("measure-once")
        .arg("--variant")
        .arg(variant.as_str())
        .arg("--rows")
        .arg(rows.to_string())
        .arg("--corpus-path")
        .arg(corpus_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .await?;
    if !output.status.success() {
        return Err(format!(
            "measure-once child ({} @ {rows} rows) exited with {}",
            variant.as_str(),
            output.status
        )
        .into());
    }
    let line = String::from_utf8(output.stdout)?;
    parse_child_line(&line)
}

/// Materialize the corpus for `rows` once, then measure the scan baseline and
/// both search variants over it in separate child processes. Returns the point
/// with each algorithm's footprint isolated above the scan baseline, and
/// verifies the two searches agreed on the result (else the contrast would be
/// over different data).
async fn measure_at(rows: usize, dir: &Path) -> Result<RssPoint, Box<dyn std::error::Error>> {
    let corpus_path = dir.join(format!("corpus_{rows}.parquet"));
    crate::corpus::materialize(&corpus_path, rows, DIM, CORPUS_SEED, BATCH_ROWS).await?;

    let (scan_only_rss_mib, _) = spawn_measure(Variant::ScanOnly, rows, &corpus_path).await?;
    let (streamed_rss_mib, streamed_digest) =
        spawn_measure(Variant::Streamed, rows, &corpus_path).await?;
    let (naive_rss_mib, naive_digest) = spawn_measure(Variant::Naive, rows, &corpus_path).await?;

    if streamed_digest != naive_digest {
        return Err(format!(
            "negative control disagrees with streamed result at {rows} rows — \
             the two variants searched different data or scored differently"
        )
        .into());
    }
    // The corpus is no longer needed once all three children have read it; free
    // the disk before the next (larger) size is materialized.
    std::fs::remove_file(&corpus_path).ok();

    Ok(RssPoint {
        rows,
        scan_only_rss_mib,
        streamed_rss_mib,
        naive_rss_mib,
        streamed_search_overhead_mib: streamed_rss_mib - scan_only_rss_mib,
        naive_search_overhead_mib: naive_rss_mib - scan_only_rss_mib,
        k: K,
    })
}

/// Run the full bounded-RSS proof and return the populated tier.
///
/// Refuses up front if the corpus filesystem lacks headroom. Materializes and
/// measures both sizes (each variant in its own process), evaluates the
/// flat-vs-linear assertion over the per-process peak deltas, and packages the
/// verdict. Returns the tier whether or not the assertion passed — the caller
/// emits the JSON and sets the process exit code from `assertion.passed`, so a
/// failed proof surfaces as a non-zero exit with full numbers, never a faked
/// pass.
pub async fn run() -> Result<BindingTier, Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let scratch: PathBuf = dir.path().to_path_buf();
    let free = free_disk_mib(&scratch)?;
    if free < MIN_FREE_DISK_MIB {
        return Err(format!(
            "insufficient scratch disk on {}: {free} MiB free, need {MIN_FREE_DISK_MIB} MiB \
             for a synthetic corpus — refusing to run",
            scratch.display()
        )
        .into());
    }

    let small = measure_at(ROWS_SMALL, &scratch).await?;
    let large = measure_at(ROWS_LARGE, &scratch).await?;

    // The assertion is over each algorithm's footprint *above the shared scan
    // baseline* — the bounded-memory claim is about the search accumulator, not
    // the DataFusion reader both searches ride. Subtracting the scan baseline at
    // each size removes the reader's own (sub-linear) N-dependence from the
    // comparison.
    let streamed_overhead_delta =
        large.streamed_search_overhead_mib - small.streamed_search_overhead_mib;
    let naive_overhead_delta = large.naive_search_overhead_mib - small.naive_search_overhead_mib;
    let scan_baseline_delta = large.scan_only_rss_mib - small.scan_only_rss_mib;

    // The vectors-held model: the naive control holds `rows·d` f32 at once, so
    // its overhead delta should track `(N₂-N₁)·d·4` bytes. Express the observed
    // delta as a fraction of that prediction; near 1.0 confirms linear-in-N
    // growth — the negative control's teeth.
    let predicted_naive_delta_mib =
        ((ROWS_LARGE - ROWS_SMALL) * DIM * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);
    let naive_growth_vs_linear_ratio = if predicted_naive_delta_mib > 0.0 {
        naive_overhead_delta / predicted_naive_delta_mib
    } else {
        0.0
    };

    let streamed_flat = streamed_overhead_delta < STREAMED_FLAT_EPSILON_MIB;
    let naive_grows = naive_overhead_delta > NAIVE_GROWTH_FLOOR_MIB;
    let passed = streamed_flat && naive_grows;

    let detail = format!(
        "streamed search overhead delta {streamed_overhead_delta:.1} MiB ({}; < \
         {STREAMED_FLAT_EPSILON_MIB:.0} MiB epsilon), naive overhead delta \
         {naive_overhead_delta:.1} MiB ({}; > {NAIVE_GROWTH_FLOOR_MIB:.0} MiB floor, {:.0}% of \
         the {predicted_naive_delta_mib:.0} MiB N·d·4 prediction); scan baseline delta \
         {scan_baseline_delta:.1} MiB (reader-side, subtracted from both)",
        if streamed_flat { "FLAT" } else { "GREW" },
        if naive_grows { "GREW" } else { "FLAT" },
        naive_growth_vs_linear_ratio * 100.0,
    );

    Ok(BindingTier {
        rss_source: active_source(),
        dim: DIM,
        points: vec![small, large],
        assertion: RssAssertion {
            passed,
            streamed_overhead_delta_mib: streamed_overhead_delta,
            streamed_flat_epsilon_mib: STREAMED_FLAT_EPSILON_MIB,
            naive_overhead_delta_mib: naive_overhead_delta,
            naive_growth_floor_mib: NAIVE_GROWTH_FLOOR_MIB,
            naive_growth_vs_linear_ratio,
            scan_baseline_delta_mib: scan_baseline_delta,
            detail,
        },
    })
}
