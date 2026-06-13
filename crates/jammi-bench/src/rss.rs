//! Peak resident-set-size sampling.
//!
//! The harness measures the *peak* RSS a piece of work drove the process to.
//! Two sources exist, in descending order of tightness:
//!
//! 1. **jemalloc resident** — if the process links `tikv-jemallocator`, the
//!    allocator's own `stats::resident` (after an `epoch` advance) reports
//!    bytes resident to that allocator, sampled tightly around the call.
//! 2. **`/proc/self/status` `VmHWM`** — the kernel's whole-process high-water
//!    mark. It is monotonic (never falls), so it cannot bracket a single call;
//!    instead the proof reads it as a *running high-water* and relies on the
//!    *delta* between two corpus sizes, which cancels the constant process
//!    baseline (loaded code, the DataFusion runtime, the synthetic corpus held
//!    by the bench itself).
//!
//! The active source is detected once at startup and recorded in the report so
//! a reader knows which mechanism produced the numbers.

use crate::report::RssSource;

/// Which RSS source this build actually has available. Determined by whether a
/// jemalloc control surface is linked; this build does not link one, so the
/// process high-water mark is the source. The variant is recorded in the
/// report rather than assumed, so wiring jemalloc later changes one place.
pub fn active_source() -> RssSource {
    // This crate does not register `tikv-jemallocator` as the global allocator,
    // so the jemalloc control stats are unavailable. The process high-water
    // mark is the source; the delta assertion is robust to its constant
    // baseline.
    RssSource::ProcVmHwm
}

/// Read the process's peak resident set in mebibytes from `/proc/self/status`
/// `VmHWM`.
///
/// `VmHWM` is the high-water mark of the process's resident set: it only ever
/// rises. Sampling it *after* a piece of work therefore reports the largest the
/// process grew to up to that point. The proof never reads an absolute value as
/// truth — it takes the difference between two corpus sizes, where the constant
/// baseline cancels.
pub fn proc_peak_rss_mib() -> Result<f64, RssError> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| RssError(format!("reading /proc/self/status: {e}")))?;
    let kb = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|kb| kb.trim().parse::<u64>().ok())
        .ok_or_else(|| RssError("VmHWM not found in /proc/self/status".into()))?;
    Ok(kb as f64 / 1024.0)
}

/// A failure to sample RSS. The proof cannot proceed without a measurement, so
/// this is a hard error rather than a degraded value — a missing RSS reading
/// must fail the run, never silently pass the assertion.
#[derive(Debug)]
pub struct RssError(pub String);

impl std::fmt::Display for RssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RSS measurement failed: {}", self.0)
    }
}

impl std::error::Error for RssError {}
