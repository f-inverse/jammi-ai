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
//!
//! Whole-file `#![cfg(not(feature = "cuda"))]` (pod-smoke fix, M3-defect
//! class): `build.rs::build_cuda` is itself `#[cfg(feature = "cuda")]`-gated
//! and does `use bindgen_cuda::Builder` internally — under the DEFAULT
//! feature set that whole function (import included) is stripped before
//! this file's `#[path]` include ever re-parses it, which is the only
//! reason this suite has ever compiled clean. `bindgen_cuda` is a
//! `[build-dependencies]` crate: invisible to a `[[test]]` target's own
//! crate graph regardless of features. Building THIS test target with
//! `--features cuda` turns `cfg(feature = "cuda")` on for the include too,
//! so `build_cuda` (and its now-unresolvable `use bindgen_cuda::Builder`,
//! E0432) gets pulled into a compilation unit that was never meant to see
//! it — the real build script (a SEPARATE compilation Cargo drives with
//! `[build-dependencies]` on its own crate graph) is where `build_cuda`
//! actually needs to exist and compile; this hermetic pin file is not that
//! compilation unit and has no business re-parsing a cuda-only fn at all.
//! Skipping the whole file under `--features cuda` keeps the pins exactly
//! where they run and matter (the default, hermetic lane) and leaves
//! nothing here for a cuda-feature test compile to trip over.
#![cfg(not(feature = "cuda"))]

#[path = "../build.rs"]
mod build_script;

use build_script::{
    check_toolkit_floor, gencode_sm, parse_max_rss_kb, parse_nvcc_release, GENCODE_ARCHES,
};

/// Pins the REAL `build.rs::GENCODE_ARCHES` constant's four entries
/// exactly — round-2 audit finding F1's fix: an EARLIER revision of this
/// test passed hand-typed literal strings (`"arch=compute_80,code=sm_80"`,
/// ...) to `gencode_sm` instead of reading `GENCODE_ARCHES` itself, which
/// meant a mutation to that array (the audit's own mutant: rewritten to a
/// pre-Ampere-inclusive, 89/90-dropping `sm_70/sm_80/sm_86` set) went
/// completely undetected here — this test would have stayed green against
/// its own stale literal copies regardless of what the real array said.
/// Reading `GENCODE_ARCHES` directly through the `#[path]` seam closes
/// that gap structurally: there is no longer a hand-typed copy for the
/// real array to drift away from.
#[test]
fn gencode_sm_parses_every_pinned_gencode_entry() {
    let smss: Vec<&str> = GENCODE_ARCHES.iter().map(|g| gencode_sm(g)).collect();
    assert_eq!(smss, vec!["80", "86", "89", "90"]);
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

/// Negative control (audit advisory): `"prerelease"` contains `"release"`
/// as a literal SUBSTRING (`p-r-e-r-e-l-e-a-s-e`, positions 3..10 spell
/// `release`), so an unanchored `.split("release ")`-style match would
/// wrongly fire on a `"...prerelease 12.0..."` token shape and report a
/// version that was never actually labelled `release`. `parse_nvcc_release`
/// matches the whole `"release"` TOKEN (bounded by whitespace/commas), not
/// a bare substring, specifically to make this collision impossible —
/// this test proves the token-boundary anchor actually holds, not just
/// that some other, unrelated input returns `None`.
#[test]
fn parse_nvcc_release_does_not_collide_with_the_prerelease_substring() {
    assert_eq!(
        parse_nvcc_release("Cuda compilation tools, prerelease 12.0, V12.0.1"),
        None,
        "\"prerelease\" is a DIFFERENT token than \"release\" and must never match"
    );
    // A genuine "release" token elsewhere in the SAME string still parses
    // correctly even with a "prerelease" token present earlier — proves
    // the anchor selects the right token, not merely "avoids the wrong
    // one" by refusing everything.
    assert_eq!(
        parse_nvcc_release("prerelease build, Cuda compilation tools, release 11.8, V11.8.89"),
        Some((11, 8))
    );
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
