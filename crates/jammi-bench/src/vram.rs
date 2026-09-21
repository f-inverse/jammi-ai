//! Peak device-memory sampling: a whole-device `nvidia-smi` poll on a
//! background thread, reported as the high-water mark above a baseline the
//! caller read before its measured work began.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::report::Measurement;

/// Total memory in use on CUDA device `ordinal`, in bytes, via `nvidia-smi -i`.
///
/// Whole-device, not per-process: on a dedicated pod the measured leg is the
/// device's only consumer, and the sampler subtracts a baseline, so the
/// reported figure is what the leg added. On a shared GPU it would over-report,
/// so the field is documented as device-total-minus-baseline rather than as a
/// process measurement. The query names the ordinal: a bare query lists every
/// device, and its first line is device 0 whatever the leg runs on.
fn nvidia_smi_memory_used(ordinal: usize) -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["-i", &ordinal.to_string()])
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    parse_memory_used(&String::from_utf8(out.stdout).ok()?)
}

/// The one line `nvidia-smi -i <ordinal> --query-gpu=memory.used
/// --format=csv,noheader,nounits` prints (MiB), in bytes. More than one line
/// is a query that did not name its device, and is no reading.
fn parse_memory_used(stdout: &str) -> Option<u64> {
    let mut lines = stdout.lines().filter(|line| !line.trim().is_empty());
    let mib = lines.next()?.trim().parse::<u64>().ok()?;
    lines.next().is_none().then_some(mib * 1024 * 1024)
}

/// Where a run reads total device memory in use, in bytes; `None` when the host
/// cannot say (no GPU, no `nvidia-smi`), and the peak is then reported as not
/// measured.
pub(crate) type DeviceMemoryProbe = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

/// The probe of the device a leg runs on: CUDA device `ordinal`'s
/// `nvidia-smi` reading, or nothing for a CPU leg, whose work never touches a
/// device.
pub(crate) fn device_memory_probe(cuda_ordinal: Option<usize>) -> DeviceMemoryProbe {
    match cuda_ordinal {
        Some(ordinal) => Arc::new(move || nvidia_smi_memory_used(ordinal)),
        None => Arc::new(|| None),
    }
}

/// What [`run_sampled`] reports for a command run on a producer's behalf —
/// the one device-memory instrument, wrapped around a process that is not
/// this binary (a PyTorch leg).
#[derive(Debug, serde::Serialize)]
pub struct SampledRun {
    /// The device's high-water mark while the command lived, above the
    /// baseline read before it started.
    pub peak_vram_bytes: Measurement,
    /// The command's own standard output, whole.
    pub child_stdout: String,
    /// The command's exit code; `None` when a signal ended it.
    pub exit_code: Option<i32>,
}

/// Run `command` to completion under the sampler: the device's high-water mark
/// while the child lived, above a baseline read before it started. The ONE
/// device-memory instrument for a leg, whatever produced it — a child of this
/// binary or a PyTorch process — because it never asks the child anything.
/// The child's stdout is captured, its stderr inherited.
pub(crate) fn run_sampled(
    command: &mut std::process::Command,
    probe: DeviceMemoryProbe,
) -> std::io::Result<(std::process::Output, Measurement)> {
    let baseline = probe().unwrap_or(0);
    let sampler = VramSampler::start(probe);
    let output = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .output()?;
    let peak = sampler.map_or_else(
        || Measurement::not_yet_measured("bytes"),
        |sampler| sampler.finish(baseline),
    );
    Ok((output, peak))
}

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
        assert_eq!(parse_memory_used(" 7 \n\n"), Some(7 * 1024 * 1024));
        assert_eq!(parse_memory_used(""), None);
        assert_eq!(parse_memory_used("[N/A]\n"), None);
    }

    /// A reading that lists more than one device did not name the leg's, and
    /// its first line is device 0 whatever the leg ran on: no reading.
    #[test]
    fn a_reading_of_every_device_is_no_reading_of_one() {
        assert_eq!(parse_memory_used("7\n81920\n"), None);
    }

    /// The sampler wraps any child: a CPU leg's probe measures nothing and
    /// says so, and the child's output comes back whole.
    #[test]
    fn a_sampled_child_returns_its_output_and_an_unmeasured_peak_without_a_device() {
        let (output, peak) = run_sampled(
            std::process::Command::new("sh").args(["-c", "printf leg"]),
            device_memory_probe(None),
        )
        .expect("run sh");
        assert_eq!(output.stdout, b"leg");
        assert_eq!(peak.value, None);
    }
}
