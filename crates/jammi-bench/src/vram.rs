//! Peak device-memory sampling: a whole-device `nvidia-smi` poll on a
//! background thread, reported as the high-water mark above a baseline the
//! caller read before its measured work began.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::report::Measurement;

/// Poll total device memory in use, in bytes, via `nvidia-smi`.
///
/// Whole-device, not per-process: on a dedicated pod this session is the only
/// consumer, and the tier subtracts a baseline read after the model is resident,
/// so the reported figure is activation and workspace growth. On a shared GPU
/// it would over-report, so the field is documented as
/// device-total-minus-baseline rather than as a process measurement.
pub(crate) fn nvidia_smi_memory_used() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    parse_memory_used(&String::from_utf8(out.stdout).ok()?)
}

/// The first line of `nvidia-smi --query-gpu=memory.used
/// --format=csv,noheader,nounits` (MiB), in bytes.
fn parse_memory_used(stdout: &str) -> Option<u64> {
    stdout
        .lines()
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|mib| mib * 1024 * 1024)
}

/// Where a run reads total device memory in use, in bytes; `None` when the host
/// cannot say (no GPU, no `nvidia-smi`), and the peak is then reported as not
/// measured.
pub(crate) type DeviceMemoryProbe = fn() -> Option<u64>;

/// Sample device memory on a background thread for the duration of the measured
/// work, so the reported peak is the real high-water mark rather than whatever
/// happened to be allocated when the last step ended.
pub(crate) struct VramSampler {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VramSampler {
    pub(crate) fn start(probe: DeviceMemoryProbe) -> Option<Self> {
        probe()?;
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (Arc::clone(&peak), Arc::clone(&stop));
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(used) = probe() {
                    p.fetch_max(used, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        });
        Some(Self {
            peak,
            stop,
            handle: Some(handle),
        })
    }

    pub(crate) fn finish(mut self, baseline: u64) -> Measurement {
        self.stop.store(true, Ordering::Relaxed);
        // A sampler thread that panicked saw only part of the window, so its
        // high-water mark is not the window's.
        let sampled_whole_window = self.handle.take().is_none_or(|h| h.join().is_ok());
        if !sampled_whole_window {
            return Measurement::not_yet_measured("bytes");
        }
        let peak = self.peak.load(Ordering::Relaxed);
        Measurement::measured(peak.saturating_sub(baseline) as f64, "bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `peak_vram_bytes` is `VramSampler::finish`'s
    /// `peak.saturating_sub(baseline)`. `saturating_sub`
    /// FLOORS at zero rather than wrapping — so if `baseline` is ever
    /// captured AT (or above) the run's own high-water mark, the reported
    /// delta collapses to zero even though the run legitimately allocated
    /// many GB. This pins the arithmetic directly, independent of
    /// `nvidia-smi`/a real GPU (`VramSampler`'s fields are plain atomics
    /// this test constructs directly, bypassing `start()`'s `nvidia-smi`
    /// precheck), using magnitudes drawn from a real A100 measurement
    /// (b8-s512-d0.05, `peak_vram_bytes` = 14.98 GB) so the test is anchored to a
    /// production-scale number, not an arbitrary toy pair.
    ///
    /// That a caller reads its baseline BEFORE its untimed pre-step has no
    /// effect a CPU host can observe; this pins the arithmetic it relies on.
    #[test]
    fn vram_sampler_finish_reports_true_delta_not_floored_by_a_baseline_at_the_peak() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // `baseline`: model + optimizer resident, BEFORE any of this run's
        // allocation. `delta_bytes`: the exact real-measurement magnitude for
        // b8-s512-d0.05 (peak_vram_bytes = 14.98 GB) so the asserted delta is
        // traceable to a real measurement, not an invented one.
        let baseline = 3 * GIB;
        let delta_bytes = 14_980_000_000_u64;
        let peak = baseline + delta_bytes;
        let sampler = VramSampler {
            peak: Arc::new(AtomicU64::new(peak)),
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
        };
        let m = sampler.finish(baseline);
        assert_eq!(
            m.value,
            Some((peak - baseline) as f64),
            "a baseline captured BEFORE this run's allocation must report the FULL delta, \
             not a floored/near-zero one"
        );
        assert!(
            m.value.unwrap() > 1.0e10,
            "expected a multi-GB delta (a baseline mistakenly captured AT the peak \
             would floor this to ~0 via saturating_sub)"
        );

        // The failure mode, reproduced in the arithmetic
        // alone: a baseline captured AT (or above) the peak — i.e. AFTER
        // the pool has already been driven to its high-water mark by an
        // untimed pre-step — floors to zero via `saturating_sub`, silently,
        // with no panic and no `None`.
        let collapsed_sampler = VramSampler {
            peak: Arc::new(AtomicU64::new(peak)),
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
        };
        let collapsed = collapsed_sampler.finish(peak); // baseline == peak
        assert_eq!(
            collapsed.value,
            Some(0.0),
            "sanity: a same-or-later baseline silently reports zero, never an error, which \
             is precisely why the CALL-SITE ordering in run() matters and cannot be caught by \
             this arithmetic test alone"
        );
    }

    #[test]
    fn nvidia_smi_memory_used_is_read_in_mib() {
        assert_eq!(parse_memory_used("40536\n"), Some(40536 * 1024 * 1024));
        assert_eq!(parse_memory_used(" 7 \n81920\n"), Some(7 * 1024 * 1024));
        assert_eq!(parse_memory_used(""), None);
        assert_eq!(parse_memory_used("[N/A]\n"), None);
    }
}
