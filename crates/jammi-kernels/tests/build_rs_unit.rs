//! Unit tests for `build.rs`'s own pure functions, driven directly against
//! that file's REAL source via `#[path]` (never a hand-copied duplicate that
//! could silently drift) — `build.rs` is not a normal `cargo test` target
//! (it is a standalone binary Cargo compiles and RUNS once, as the build
//! script; `cargo test -p jammi-kernels --list` confirms it names nothing
//! from `build.rs`), so an integration test file is the only way to exercise
//! its logic under `cargo test`. This mirrors
//! `crates/jammi-bench/tests/build_rs_unit.rs`'s own established pattern for
//! this crate — see that file's doc for the fuller rationale.
//!
//! Every function this suite calls (`gencode_sm`, `parse_nvcc_release`,
//! `check_toolkit_floor`, `parse_max_rss_kb`) is top-level in `build.rs`
//! (never `#[cfg(feature = "flash-attn")]`-gated), specifically so this
//! suite compiles and runs under this crate's DEFAULT feature set — no
//! `cuda`/`flash-attn` feature, no nvcc, no CUDA toolkit anywhere on the
//! test machine. `main()`/`build_cuda()`/`build_flash_attn()` are never
//! called here — only the pure helpers they call, so this suite never
//! touches `cargo:rustc-env`/`cargo:rerun-if-changed` output or spawns a
//! subprocess.

#[path = "../build.rs"]
#[allow(dead_code)]
mod build_script;

use build_script::{check_toolkit_floor, gencode_sm, parse_max_rss_kb, parse_nvcc_release};

/// Pins the CURRENT `GENCODE_ARCHES` literal's four entries exactly — a
/// change to any one of them without updating this test is exactly the
/// drift `gencode_sm` exists to prevent silently (the same anti-drift
/// precedent `flash/mod.rs`'s `gencode_sms_parses_the_pinned_build_value`
/// pins on the Rust-crate side of the same env var).
#[test]
fn gencode_sm_parses_every_pinned_gencode_entry() {
    assert_eq!(gencode_sm("arch=compute_80,code=sm_80"), "80");
    assert_eq!(gencode_sm("arch=compute_86,code=sm_86"), "86");
    assert_eq!(gencode_sm("arch=compute_89,code=sm_89"), "89");
    assert_eq!(gencode_sm("arch=compute_90,code=sm_90"), "90");
}

#[test]
#[should_panic(expected = "does not end in")]
fn gencode_sm_panics_on_a_malformed_entry() {
    gencode_sm("arch=compute_80,code=sm_");
}

#[test]
fn parse_nvcc_release_reads_the_real_nvcc_version_shape() {
    // The exact stdout shape `nvcc --version` emits (upstream format,
    // reproduced as a literal fixture — no real nvcc needed to test the
    // parser against it).
    assert_eq!(
        parse_nvcc_release(
            "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2022 NVIDIA Corporation\n\
             Built on Wed_Sep_21_10:33:58_PDT_2022\nCuda compilation tools, release 11.8, \
             V11.8.89\nBuild cuda_11.8.r11.8/compiler.31833905_0\n"
        ),
        Some((11, 8))
    );
    assert_eq!(
        parse_nvcc_release("Cuda compilation tools, release 12.4, V12.4.131"),
        Some((12, 4))
    );
}

#[test]
fn parse_nvcc_release_returns_none_on_unrecognised_output() {
    assert_eq!(parse_nvcc_release(""), None);
    assert_eq!(parse_nvcc_release("nvcc: command not found"), None);
    assert_eq!(parse_nvcc_release("release not-a-version"), None);
}

#[test]
fn check_toolkit_floor_admits_at_or_above_and_refuses_below() {
    let floor = (11, 8);
    assert!(
        check_toolkit_floor((11, 8), floor).is_ok(),
        "exact floor admits"
    );
    assert!(
        check_toolkit_floor((12, 0), floor).is_ok(),
        "newer major admits"
    );
    assert!(
        check_toolkit_floor((11, 9), floor).is_ok(),
        "newer minor, same major admits"
    );
    let err = check_toolkit_floor((11, 7), floor).unwrap_err();
    assert!(
        err.contains("11.7") && err.contains("11.8"),
        "remedy must name both the detected and required versions: {err}"
    );
    let err = check_toolkit_floor((10, 2), floor).unwrap_err();
    assert!(err.contains("10.2") && err.contains("11.8"), "{err}");
}

#[test]
fn parse_max_rss_kb_reads_gnu_times_v_line() {
    let fixture = "\tCommand being timed: \"nvcc -c foo.cu\"\n\tUser time (seconds): 42.10\n\t\
                    Maximum resident set size (kbytes): 1234567\n\tExit status: 0\n";
    assert_eq!(parse_max_rss_kb(fixture), Some(1_234_567));
}

#[test]
fn parse_max_rss_kb_returns_none_without_the_gnu_v_line() {
    // BSD/macOS `time` (no `-v`) never emits this line — `None` is the
    // documented best-effort fallback, not a parse failure.
    assert_eq!(
        parse_max_rss_kb("        0.01 real         0.00 user         0.00 sys"),
        None
    );
    assert_eq!(parse_max_rss_kb(""), None);
}
