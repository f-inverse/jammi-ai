//! CPU-local pre-measurement for the media front-end parallelization A/B's
//! machine model (issue #421 follow-on, "media front-end parallelization"
//! contract §"Measurement"): times EXACTLY the batch's final tensor build
//! (`Tensor::from_vec`, host-side) plus its device upload
//! (`Tensor::to_device`) — for a full per-step micro-batch shape, on the
//! CURRENT device. This is A serial cost the rayon change under test cannot
//! parallelize away, never claimed here to be the ONLY one: the mel
//! filterbank (`mel_filterbank_hz`) and analysis window
//! (`hann_periodic`) `preprocess_clap_fusion_indexed` computes ONCE per
//! batch, BEFORE its own parallel `par_chunks_mut` fan-out
//! (`crates/jammi-ai/src/inference/audio_preprocess.rs`), are a SECOND,
//! untimed serial term this example does not measure at all.
//!
//! `ci/scripts/perf/frontend_ab.sh`'s `--serial-tail-ratio` argument is
//! `r = t_s / T`, where `t_s` is this example's printed value for the
//! relevant task and `T` is the corresponding tier's measured
//! `train_run_wall_s / steps_measured` (the per-step wall this front-end
//! sits inside). This example does not compute `r` itself — it has no way
//! to know `T` — it only measures `t_s`; the operator divides. Because the
//! filterbank/window hoist is omitted, the `r` the operator supplies UNDERSTATES
//! the true serial tail. The bar arithmetic stays conservative under that
//! understatement rather than dangerous: both bounds are increasing in `r`
//! (`ci/scripts/perf/frontend_ab_merge.py`'s own module doc has the
//! formulas), so a smaller `r` only LOWERS `upper_bound` — the ceiling a
//! ratio must clear to read PASS — making PASS strictly harder to reach,
//! never easier. An omitted serial term can cost this bar a real PASS
//! (an UNRESOLVED or a borderline FAIL where the true, larger `r` would
//! have read PASS); it can never manufacture a PASS the true `r` would
//! have refused.
//!
//! Two shapes, matching the contract's pre-registered legs exactly (batch
//! 8, triplets, so 24 items/step):
//!   - HTSAT audio: `[24, 4, 1001, 64]` f32 (CLAP-fusion mel input shape).
//!   - CLIP-vision image: `[24, 3, 224, 224]` f32.
//!
//! Deliberately standalone: this crate has no library target (`jammi-bench`
//! is `[[bin]]`-only), so this example depends only on `candle-core`
//! (already a direct workspace dependency) and the standard library —
//! never any of `jammi-bench`'s own internal modules.
//!
//! Run with `cargo run -p jammi-bench --release --example frontend_serial_tail
//! [-- --cuda <ordinal>] [-- --reps <n>]` (defaults: CPU, 20 reps). Prints
//! one `t_s=<seconds>` line per shape to stdout; nothing else is written
//! there, so a caller can `grep` the exact `task=... t_s=...` line it wants.

use candle_core::{Device, Tensor};
use std::time::Instant;

/// A full per-step micro-batch shape this A/B's front end constructs and
/// uploads once per training step (issue #421 follow-on contract's
/// pre-registered legs: batch 8, triplets, 24 items/step).
struct Shape {
    task: &'static str,
    dims: [usize; 4],
}

const SHAPES: &[Shape] = &[
    Shape {
        task: "audio_embedding",
        dims: [24, 4, 1001, 64],
    },
    Shape {
        task: "image_embedding",
        dims: [24, 3, 224, 224],
    },
];

/// Parses `--cuda <ordinal>` and `--reps <n>` from `argv`, ignoring the
/// program name. Absent flags keep the stated default. A malformed value
/// (non-numeric, or `--cuda`/`--reps` with no following argument) is a
/// caller error — this refuses loudly rather than silently falling back,
/// since a silently-wrong device or rep count would corrupt the measurement
/// this example exists to produce.
fn parse_args(argv: &[String]) -> (Option<usize>, usize) {
    let mut cuda_ordinal = None;
    let mut reps = 20usize;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--cuda" => {
                let value = argv
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("--cuda requires an ordinal argument"));
                cuda_ordinal =
                    Some(value.parse::<usize>().unwrap_or_else(|e| {
                        panic!("--cuda ordinal {value:?} is not a usize: {e}")
                    }));
                i += 2;
            }
            "--reps" => {
                let value = argv
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("--reps requires a count argument"));
                reps = value
                    .parse::<usize>()
                    .unwrap_or_else(|e| panic!("--reps count {value:?} is not a usize: {e}"));
                assert!(reps > 0, "--reps must be > 0");
                i += 2;
            }
            other => panic!("frontend_serial_tail: unrecognized argument {other:?}"),
        }
    }
    (cuda_ordinal, reps)
}

/// Times `reps` back-to-back `Tensor::from_vec` (host construction) +
/// `Tensor::to_device` (upload — a no-op copy on `Device::Cpu`, a real H2D
/// transfer on `Device::Cuda`) calls for `dims`, and returns the MEAN
/// per-call wall time in seconds — never the total, since a caller
/// comparing this against a single training step's wall needs a per-step
/// number, not a per-`reps`-loop one.
///
/// A fresh `Vec<f32>` is built inside the timed region on every rep (never
/// hoisted outside the loop and reused): the real front end allocates a
/// fresh host buffer per training step too, and hoisting the allocation out
/// here would undercount the serial tail this measurement exists to
/// capture.
fn time_construct_and_upload(dims: [usize; 4], device: &Device, reps: usize) -> f64 {
    let n: usize = dims.iter().product();
    let started = Instant::now();
    for _ in 0..reps {
        let data = vec![0f32; n];
        let host = Tensor::from_vec(data, dims.as_slice(), &Device::Cpu)
            .expect("frontend_serial_tail: Tensor::from_vec on the host");
        let uploaded = host
            .to_device(device)
            .expect("frontend_serial_tail: Tensor::to_device");
        // Forces the upload to actually happen rather than being optimized
        // away as dead code — candle's own tensor ops are never no-ops at
        // the LLVM level (they cross an FFI/allocation boundary), but
        // `std::hint::black_box` costs nothing and removes any doubt.
        std::hint::black_box(&uploaded);
    }
    started.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cuda_ordinal, reps) = parse_args(&argv);
    let device = match cuda_ordinal {
        Some(ordinal) => Device::new_cuda(ordinal)
            .unwrap_or_else(|e| panic!("Device::new_cuda({ordinal}) failed: {e}")),
        None => Device::Cpu,
    };
    let device_label = match cuda_ordinal {
        Some(ordinal) => format!("cuda:{ordinal}"),
        None => "cpu".to_string(),
    };

    for shape in SHAPES {
        let t_s = time_construct_and_upload(shape.dims, &device, reps);
        println!(
            "task={} device={device_label} reps={reps} dims={:?} t_s={t_s}",
            shape.task, shape.dims
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_defaults_to_cpu_and_20_reps() {
        assert_eq!(parse_args(&[]), (None, 20));
    }

    #[test]
    fn parse_args_reads_cuda_and_reps() {
        let argv: Vec<String> = ["--cuda", "1", "--reps", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(parse_args(&argv), (Some(1), 5));
    }

    #[test]
    #[should_panic(expected = "--reps must be > 0")]
    fn parse_args_refuses_zero_reps() {
        let argv: Vec<String> = ["--reps", "0"].iter().map(|s| s.to_string()).collect();
        parse_args(&argv);
    }

    #[test]
    #[should_panic(expected = "unrecognized argument")]
    fn parse_args_refuses_an_unknown_flag() {
        let argv: Vec<String> = ["--bogus"].iter().map(|s| s.to_string()).collect();
        parse_args(&argv);
    }

    /// The mechanism actually times SOMETHING measurable and positive on
    /// CPU — a non-vacuous smoke check (never asserts an exact value, since
    /// wall time is machine-dependent by construction), for both shapes
    /// this example exists to measure.
    #[test]
    fn time_construct_and_upload_reports_a_positive_duration_on_cpu() {
        for shape in SHAPES {
            let t_s = time_construct_and_upload(shape.dims, &Device::Cpu, 3);
            assert!(
                t_s > 0.0,
                "task {} must report a positive t_s, got {t_s}",
                shape.task
            );
            assert!(
                t_s.is_finite(),
                "task {} must report a finite t_s, got {t_s}",
                shape.task
            );
        }
    }
}
