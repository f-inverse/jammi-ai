//! Compiles `src/cuda/*.cu` to PTX — ONLY when the `cuda` feature is active —
//! and, ONLY when the `flash-attn` feature is active on top of it, the
//! vendored FlashAttention-2 kernels + jammi's C wrapper into
//! `libjammi_flash.a` (see `build_flash_attn` below).
//!
//! Every other build (the laptop/CI default across this workspace) must not
//! require nvcc, a CUDA toolkit, or any system requirement beyond a plain
//! `cargo build`. `CARGO_FEATURE_CUDA` is the env var Cargo sets for build
//! scripts exactly when a crate's own `cuda` feature is enabled; checking it
//! here (rather than relying on `#[cfg(feature = "cuda")]` alone) makes the
//! early-return the single, auditable gate a reviewer can point at.

use std::env;
#[cfg(any(feature = "cuda", feature = "flash-attn"))]
use std::path::{Path, PathBuf};

/// The full `-gencode` set this crate's vendored FlashAttention-2 build
/// compiles NATIVE cubins for — sm80 (Ampere baseline) through sm90
/// (Hopper), one pair per arch this crate admits (M3 plan D1: compile a
/// real, per-arch cubin for every admitted arch, never a PTX-JIT-forward
/// entry — `code=sm_XX` only, no bare `code=compute_XX`, on every line).
/// See `third_party/flash-attention/VENDORED.md`'s "Supported archs"
/// section for the per-arch VALIDATION status — compiled is necessary,
/// not sufficient, for admission (`crate::admission::flash_built_arches`'s
/// own doc has the compiled-vs-admitted distinction).
///
/// Top-level (not `#[cfg(feature = "flash-attn")]`-gated): `main()` emits
/// `JAMMI_FLASH_GENCODE_SMS` from this UNCONDITIONALLY, in every feature
/// configuration (see `main`'s own comment for why), and this same literal
/// also feeds `build_flash_attn`'s real `-gencode` flags — one array, two
/// readers, so they can never drift apart. Also reachable from the
/// `#[path = "../build.rs"]` unit-test seam (`tests/build_rs_unit.rs`)
/// without that suite needing the `flash-attn` feature turned on.
const GENCODE_ARCHES: &[&str] = &[
    "arch=compute_80,code=sm_80",
    "arch=compute_86,code=sm_86",
    "arch=compute_89,code=sm_89",
    "arch=compute_90,code=sm_90",
];

/// Parses the `code=sm_<digits>` suffix out of ONE `-gencode` literal — the
/// same anti-drift pattern the deleted singular `GENCODE_ARCH`/`gencode_sm`
/// pair used, now applied per entry of [`GENCODE_ARCHES`] so
/// `JAMMI_FLASH_GENCODE_SMS` (`main`'s emission) and the actual `-gencode`
/// flags `build_flash_attn` passes to nvcc read the SAME literal array,
/// never an independently-retyped copy. Panics on a malformed entry — this
/// is compile-time-pinned build.rs source, not untrusted input, so a
/// malformed value can only mean a hand-edit broke [`GENCODE_ARCHES`]
/// itself, which must fail loud rather than silently drop an arch.
pub(crate) fn gencode_sm(entry: &str) -> &str {
    entry
        .rsplit("sm_")
        .next()
        .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
        .unwrap_or_else(|| {
            panic!(
                "jammi-kernels build.rs: gencode entry {entry:?} does not end in \
                 \"sm_<digits>\" — the code=sm_XX suffix JAMMI_FLASH_GENCODE_SMS derives from \
                 is missing or malformed"
            )
        })
}

/// Parses the `release <major>.<minor>` token out of `nvcc --version`'s
/// stdout (e.g. `"Cuda compilation tools, release 11.8, V11.8.89"` ->
/// `Some((11, 8))`). Pure string parsing — no filesystem, no subprocess —
/// so it is directly unit-testable against literal fixture strings
/// (`tests/build_rs_unit.rs`) without a real `nvcc` anywhere on the test
/// machine; `check_toolkit_floor` (below) is the sibling pure function that
/// turns the parsed pair into a pass/fail-with-remedy verdict.
///
/// `#[allow(dead_code)]`: this function's only call site is inside
/// `build_flash_attn`, which is `#[cfg(feature = "flash-attn")]`-gated —
/// under this crate's DEFAULT feature set (no `cuda`/`flash-attn`), that
/// call site does not exist, so the build script binary itself never
/// reaches this function and rustc's own dead-code lint is right that it
/// is unreachable FROM THAT BINARY. It is reachable from two OTHER,
/// legitimate places: `build_flash_attn` under `--features flash-attn`
/// (the real production path), and `tests/build_rs_unit.rs`'s `#[path]`
/// seam (every feature configuration) — the allow documents that this is
/// an intentional cross-cfg surface, not dead code left behind.
#[allow(dead_code)]
pub(crate) fn parse_nvcc_release(version_stdout: &str) -> Option<(u32, u32)> {
    let after = version_stdout.split("release ").nth(1)?;
    let token = after.split([',', ' ', '\n', '\r']).next()?;
    let mut parts = token.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// `Ok(())` iff `detected >= floor` (lexicographic major.minor comparison —
/// the same ordering CUDA's own release numbering uses), else an `Err`
/// remedy string naming both versions. Pure — split from
/// [`parse_nvcc_release`] so both halves are independently unit-testable.
///
/// This positive version check (detect the ACTUAL toolkit release, compare
/// against a stated floor, and name the remedy) replaces this crate's
/// earlier, now-inaccurate "requires CUDA 12.x with sm_80 support" prose:
/// [`GENCODE_ARCHES`]'s `sm_90` pair (added the same commit as this
/// function) needs CUDA >= 11.8, not 12.x — CUDA 11.8's release notes are
/// the toolkit version that first added `sm_90`/Hopper `-gencode` support;
/// sm_80/86/89 have been buildable since CUDA 11.1/11.1/11.4 respectively,
/// so 11.8 is the binding floor for THIS crate's combined gencode set, not
/// any individual arch's own older floor.
///
/// `#[allow(dead_code)]`: see [`parse_nvcc_release`]'s doc — same cross-cfg
/// reachability (only `build_flash_attn`, feature-gated, and the
/// `tests/build_rs_unit.rs` seam call this under the default feature set).
#[allow(dead_code)]
pub(crate) fn check_toolkit_floor(
    detected: (u32, u32),
    floor: (u32, u32),
) -> std::result::Result<(), String> {
    if detected >= floor {
        Ok(())
    } else {
        Err(format!(
            "detected CUDA toolkit release {}.{} is below the {}.{} floor this crate's \
             `flash-attn` feature needs for its sm_90 gencode pair — upgrade the CUDA toolkit \
             (nvcc --version must report release >= {}.{}) or drop the sm_90 entry from \
             GENCODE_ARCHES in build.rs if you only need sm_80/86/89",
            detected.0, detected.1, floor.0, floor.1, floor.0, floor.1
        ))
    }
}

/// Parses GNU `time -v`'s `"Maximum resident set size (kbytes): N"` line out
/// of its stderr, when present. `None` when the wrapper's report line is
/// absent (BSD/macOS `time` has no `-v`, so this is best-effort: expected on
/// the CI flash-attn-compile lane's Linux container when
/// `JAMMI_FLASH_MEASURE_RSS` opts in, absent everywhere else — see
/// `build_flash_attn`'s own comment for why this is opt-in, not
/// autodetected). Pure, unit-tested against a literal fixture line.
///
/// `#[allow(dead_code)]`: see [`parse_nvcc_release`]'s doc — same cross-cfg
/// reachability (only `build_flash_attn`, feature-gated, and the
/// `tests/build_rs_unit.rs` seam call this under the default feature set).
#[allow(dead_code)]
pub(crate) fn parse_max_rss_kb(gnu_time_stderr: &str) -> Option<u64> {
    for line in gnu_time_stderr.lines() {
        if let Some(rest) = line
            .trim()
            .strip_prefix("Maximum resident set size (kbytes): ")
        {
            return rest.trim().parse().ok();
        }
    }
    None
}

/// Every FILE (recursively) under `dir`, sorted for a deterministic
/// iteration order — used for TWO independent purposes `build_cuda`'s own
/// comment details in full: emitting `cargo:rerun-if-changed=<path>` PER
/// FILE, and feeding the newest-source-mtime computation that force-deletes
/// a stale `.ptx` output before `bindgen_cuda`'s `build_ptx()` runs.
///
/// Round-3 audit reconciliation: an EARLIER revision of THIS doc comment
/// attributed the per-file `rerun-if-changed` emission to Cargo's OWN
/// directory-level `cargo:rerun-if-changed=<dir>` line being unreliable
/// ("tracks the directory entry's own mtime, which an existing file's
/// content edit does not touch"). That claim was WRONG about Cargo's own
/// mechanism specifically: Cargo resolves a directory `rerun-if-changed`
/// target with a RECURSIVE mtime walk (`paths::mtime_recursive` in Cargo's
/// own source) that DOES pick up a nested file's content edit or deletion —
/// independently confirmed both by `crates/jammi-bench/build.rs`'s own
/// `edited_tracked_file_forces_dirty_on_rebuild` regression test (unit 61
/// phase 1's round-2 audit) and by that unit's round-3 re-audit reproducing
/// the directory-watch behavior directly. A single directory-level line
/// WOULD have correctly re-run THIS SCRIPT on a header-only edit.
///
/// The genuinely load-bearing reason `walk_files` is still required is the
/// SEPARATE mtime computation `build_cuda` uses to force-delete a stale
/// `.ptx` before calling `build_ptx()`: `bindgen_cuda` 0.1.6's OWN internal
/// per-kernel skip check (comparing that kernel's `.cu` mtime against its
/// existing `.ptx` output) runs regardless of whether THIS script just
/// reran, and never consults its own `watch: Vec<PathBuf>` field when
/// deciding whether to skip nvcc (only to print more `rerun-if-changed`
/// lines) — a header-only edit that never touches any kernel's OWN `.cu`
/// mtime silently serves stale PTX otherwise. See `build_cuda`'s own
/// comment for the full two-mechanism breakdown and the reference-pod
/// reproduction. Gated: unused (and correctly so — no `.cu` compiles) on
/// the plain, no-CUDA default build every other workspace crate takes.
#[cfg(any(feature = "cuda", feature = "flash-attn"))]
fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            out.extend(walk_files(&path));
        } else {
            out.push(path);
        }
    }
    out
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FLASH_ATTN");

    // Emitted UNCONDITIONALLY — every feature configuration this crate can
    // be built with, not only inside the `flash-attn` branch below (where
    // the deleted singular `JAMMI_FLASH_GENCODE_SM` used to live). This is
    // a pure string-parse over `GENCODE_ARCHES` (no nvcc, no CUTLASS, no
    // filesystem probe involved), so it costs nothing to run on every
    // build, and it makes `env!("JAMMI_FLASH_GENCODE_SMS")` compile in
    // EVERY cfg this crate ships — every stale reader of the deleted
    // singular name now fails to COMPILE instead of silently resolving to
    // a value from a different build (M3 plan D2's "loud migration").
    // `crate::admission::flash_built_arches()` is this crate's own reader
    // for callers outside `crate::flash` (which stays `#[cfg(feature =
    // "flash-attn")]`-gated); it gates the ANSWER on `FLASH_COMPILED`, not
    // on whether this env var exists — see that function's own doc.
    let gencode_sms: Vec<&str> = GENCODE_ARCHES.iter().map(|g| gencode_sm(g)).collect();
    println!(
        "cargo:rustc-env=JAMMI_FLASH_GENCODE_SMS={}",
        gencode_sms.join(",")
    );

    // Default build: no cuda feature, no nvcc, no CUDA toolkit. Nothing
    // below this line ever runs unless the `cuda` feature set this env var.
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        #[cfg(feature = "cuda")]
        {
            build_cuda();
        }
    }

    // `flash-attn` implies `cuda` (Cargo.toml) but is NOT implied by it:
    // the vendored FlashAttention-2 build is a separate, opt-in gate with
    // its own env var, so the `cuda` lane (the release image,
    // `--features cuda` through jammi-server) never compiles CUTLASS.
    if env::var("CARGO_FEATURE_FLASH_ATTN").is_ok() {
        #[cfg(feature = "flash-attn")]
        {
            build_flash_attn();
        }
    }
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    use bindgen_cuda::Builder;

    // rerun-if-changed, PER FILE under `src/cuda` (see `walk_files`'s own
    // doc for why a single directory-level line is not reliable here):
    // `Builder::build_ptx()` itself emits `cargo:rerun-if-changed=<path>`
    // for every KERNEL (`.cu`) path the glob below resolves to
    // (bindgen_cuda 0.1.6, `build_ptx`'s `kernel_paths.par_iter()` loop),
    // but that does NOT cover a shared HEADER (`.cuh`) a kernel
    // `#include`s — bindgen_cuda 0.1.6 does not parse `.cu`/`.cuh`
    // dependencies, so `src/cuda/rope_common.cuh` (shared by `rope.cu` AND
    // `rope_positions.cu`) editing it alone would neither re-run this
    // build script NOR (see `.watch` below) re-invoke nvcc for either
    // kernel that includes it — a stale-PTX build silently serving the
    // OLD kernel body (confirmed live on the reference pod before this
    // fix: a header-only edit needed two manual `touch`es to force a
    // rebuild). Walking `src/cuda` ourselves and emitting one line per
    // file closes the Cargo-rerun half; `.watch` below closes the second,
    // independent half (`build_ptx()`'s OWN internal staleness check).
    for path in walk_files(Path::new("src/cuda")) {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    // `Builder::default()` itself needs nvcc/nvidia-smi info BEFORE we get
    // a chance to call `.compute_cap(80)` below. Its internal `compute_cap()`
    // helper does TWO things: (a) resolves the compute-cap NUMBER from the
    // env var `CUDA_COMPUTE_CAP`, else `nvidia-smi`; (b) UNCONDITIONALLY —
    // even when (a) took the env-var branch — also shells out to `nvcc
    // --list-gpu-code` to build its supported-codes table. Setting
    // `CUDA_COMPUTE_CAP` here neutralizes ONLY (a): it removes the
    // `nvidia-smi` dependency, which is what matters for a real, common
    // build shape — a Docker build stage with the CUDA toolkit (nvcc)
    // installed for compilation but no GPU driver / `nvidia-smi` present
    // at build time, because the image is built once and deployed to GPU
    // nodes (the Dockerfile:193 precedent). It does NOT remove (b)'s nvcc
    // requirement, and cannot: nvcc itself is unavoidable for this
    // crate's `cuda` feature by design (there is no PTX without it). On a
    // machine with NEITHER nvcc nor `nvidia-smi` (verified locally on this
    // Mac), `Builder::default()` still panics — inside its own
    // `compute_cap()` helper, before `build_ptx()` below ever runs — for
    // the unavoidable "no nvcc" reason, not because of anything this
    // env-var fix could or should paper over.
    env::set_var("CUDA_COMPUTE_CAP", "80");

    // PINNED FLAGS:
    //
    //   - `compute_cap(80)`: sm_80 (Ampere) baseline, pinned both via the
    //     env var above (so `Builder::default()`'s own construction-time
    //     probe never touches `nvidia-smi`) AND via this explicit override
    //     (belt-and-suspenders — the two must agree), so the emitted PTX is
    //     identical regardless of which machine builds it. The shipped
    //     image ships this same single-arch PTX (Dockerfile:193
    //     precedent); the driver JIT-forwards it to 8.6 / 8.9 / 9.0 devices
    //     at first load.
    //   - NO `-use_fast_math`. bindgen_cuda's nvcc invocation does not add it
    //     by default and this build never adds it via `.arg(..)` either.
    //     This is NOT the same as bit-exact float parity with the CPU
    //     arm: nvcc's `--fmad=true` default (a separate flag from
    //     `-use_fast_math`, on regardless of it) may still contract an
    //     expression like `alpha*x + y` into a single-rounding hardware
    //     FMA, differing from the CPU's two separately-rounded operations
    //     by up to ~1 ULP. This build deliberately does NOT pin
    //     `--fmad=false` globally (that would cost real performance on
    //     every kernel, not just the ones that need bit-exact parity), so
    //     every fused-op oracle that compares CPU and CUDA output states a
    //     TOLERANCE (`tests/cuda_parity.rs`'s `F32_TOL` / bf16-ULP bounds),
    //     not bit-exact equality — fmad contraction is accepted within
    //     those stated bounds. A future kernel that genuinely needs
    //     bit-exact float parity on one specific expression (the C7
    //     device-side-dropout plan's Philox-derived scale, where the
    //     KEEP/DROP decision itself must match host-side f64 exactly) pins
    //     that with explicitly-rounded intrinsics in the expression itself
    //     (`__fmul_rn` / `__fadd_rn` / `__fmaf_rn`), not with a global
    //     `--fmad=false`.
    // Force nvcc to actually RE-RUN for a kernel whenever ANY file under
    // `src/cuda` (that kernel's own `.cu`, OR a header like
    // `rope_common.cuh` it `#include`s) is newer than that kernel's
    // EXISTING `.ptx` output — closing a gap that survived two earlier
    // (insufficient) fix attempts, both confirmed live on the reference
    // pod with a touch-free header-only edit:
    //   1. A `cargo:rerun-if-changed` line per file under `src/cuda`
    //      (still needed, kept above) makes CARGO re-run this whole build
    //      script on a header edit — necessary but not sufficient.
    //   2. `bindgen_cuda` 0.1.6's `Builder::build_ptx()` (below) has its
    //      OWN, SEPARATE per-kernel skip check
    //      (`out_modified.duration_since(in_modified)` on the OUT_DIR PTX
    //      vs THAT KERNEL'S OWN `.cu` mtime — read directly from
    //      `bindgen_cuda`'s source) that runs regardless of whether this
    //      build script just re-ran. It has a `watch: Vec<PathBuf>`
    //      field, but `build_ptx()` (unlike its sibling `build_lib()`,
    //      which DOES fold `watch_modified` into its own skip decision)
    //      only ever uses `self.watch` to emit MORE `cargo:rerun-if-changed`
    //      lines — never consults it when deciding whether to skip nvcc.
    //      So even after step 1 re-runs this script, `build_ptx()` still
    //      sees `rope.cu`/`rope_positions.cu`'s OWN (unchanged) mtimes and
    //      skips nvcc for both — a header-only edit silently served STALE
    //      PTX for every kernel that includes it.
    // The fix bindgen_cuda 0.1.6 leaves available: make its skip check
    // itself see a MISSING output (which its `else { false }` branch
    // always treats as "not ignored," i.e. compile) by deleting a
    // kernel's existing `.ptx` whenever the source tree has a file newer
    // than it. This is a coarse, SAFE over-approximation (it does not try
    // to track which kernel includes which header — any kernel could,
    // and the crate is small enough that recompiling all of them on any
    // `src/cuda` change costs seconds, not minutes), and it is exactly
    // what makes nvcc actually re-run — confirmed live: a touch-free
    // `rope_common.cuh` edit now produces a FRESH `rope.ptx`/
    // `rope_positions.ptx` mtime, where both prior attempts still served
    // the stale file.
    let out_dir = std::path::PathBuf::from(
        env::var("OUT_DIR").expect("OUT_DIR must be set inside a build script"),
    );
    let cuda_sources = walk_files(Path::new("src/cuda"));
    if let Some(newest_source) = cuda_sources
        .iter()
        .filter_map(|p| p.metadata().ok()?.modified().ok())
        .max()
    {
        for cu in cuda_sources
            .iter()
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("cu"))
        {
            let stem = cu.file_stem().expect("kernel path must have a filename");
            let ptx = out_dir.join(stem).with_extension("ptx");
            let ptx_is_stale = ptx
                .metadata()
                .and_then(|m| m.modified())
                .is_ok_and(|ptx_modified| newest_source > ptx_modified);
            if ptx_is_stale {
                let _ = std::fs::remove_file(&ptx);
            }
        }
    }

    let builder = Builder::default()
        .kernel_paths_glob("src/cuda/*.cu")
        .compute_cap(80);

    // `build_ptx()` invokes nvcc (a SEPARATE nvcc invocation per kernel
    // file, distinct from `Builder::default()`'s own `compute_cap()`-time
    // probe above) and writes one `<kernel-stem>.ptx` file per kernel into
    // `OUT_DIR` (e.g. `axpy.ptx` for `src/cuda/axpy.cu`). Each op's Rust
    // glue embeds its own PTX with
    // `include_str!(concat!(env!("OUT_DIR"), "/<name>.ptx"))` directly, so
    // the `Bindings` helper-file generator (`.write(..)`) is not needed here.
    let _bindings = builder.build_ptx().expect(
        "nvcc PTX build failed for jammi-kernels/src/cuda/*.cu — the `cuda` \
         feature requires a CUDA toolkit (nvcc) on PATH",
    );
}

/// Compiles the vendored FlashAttention-2 hdim64/bf16 forward + backward
/// translation units — ONE native cubin per entry of [`GENCODE_ARCHES`]
/// embedded in the SAME three object files, `nvcc` fans a `-gencode` list
/// out into one device-code section per arch within one TU — and jammi's
/// torch-free C wrapper into `$OUT_DIR/libjammi_flash.a` with a hand-rolled
/// `nvcc` invocation, then links it (plus `cudart` and `stdc++`) into this
/// crate.
///
/// Hand-rolled rather than `bindgen_cuda`: the 0.1.6 `Builder` has no
/// CUTLASS include hook, `build_lib` emits no `-I`, and `.arg` takes
/// `&'static str` — it cannot express this flag group. The existing PTX
/// path above is untouched.
///
/// CUBINS FOR THE ENUMERATED, VALIDATED SET; NO PTX (M3 plan D1/D2): this
/// build compiles a REAL, native `code=sm_XX` cubin for every arch in
/// [`GENCODE_ARCHES`] — never a bare `code=compute_XX` PTX entry, on any
/// line. Embedding PTX would let a device outside the enumerated set JIT
/// an unvalidated kernel variant at RUNTIME (a genuinely different tile
/// config on a smaller-smem arch, `flash_bwd_launch_template.h:160-193`) —
/// a second shipped code path with zero oracles. Adding a NEW arch to this
/// crate's admitted set is a THREE-part change in one PR: a `-gencode` pair
/// here, the matching entry in `jammi-encoders`/`jammi-bench`'s fence sites
/// (which read `crate::admission::flash_built_arches()`, not this file),
/// and a green pod parity artifact run on THAT exact arch (VENDORED.md's
/// "Supported archs" table) — never a `-gencode` addition alone, and never
/// admission via SASS minor-version forward-compat (real, but a compiled-
/// vs-validated distinction this crate deliberately does not lean on — see
/// `VENDORED.md`'s "Supported archs" section for the corrected mechanism
/// note and why jammi validates every arch it ships natively instead).
///
/// FLAG GROUP — upstream `setup.py`'s nvcc group, as measured in the build
/// spike (`third_party/flash-attention/VENDORED.md`), widened from a single
/// `-gencode` to one pair per [`GENCODE_ARCHES`] entry:
///
/// - `-O3 -std=c++17`
/// - `-gencode arch=compute_XX,code=sm_XX`, once per [`GENCODE_ARCHES`]
///   entry (native cubin ONLY per arch, matching upstream `setup.py`'s own
///   per-arch `code=sm_XX`-only convention — NOT ALSO `code=compute_XX`/
///   embedded PTX for any of them).
/// - `--threads <N>` (`N` = `$NVCC_THREADS` if set and `> 0`, else `4`) —
///   nvcc's own internal parallel-compilation flag (CUDA >= 11.5, which
///   this build's toolkit floor below already exceeds), splitting each of
///   the three TU compiles across `N` host threads; mitigates the ~4x
///   device-code-section cost [`GENCODE_ARCHES`]'s four entries add over
///   the single-arch build this replaced.
/// - `--expt-relaxed-constexpr --expt-extended-lambda`
/// - `--use_fast_math` — THE ONE-TU DIVERGENCE from this crate's
///   no-fast-math rule (`build_cuda` above pins it off for `src/cuda/*.cu`).
///   Upstream ships every FlashAttention-2 wheel with it; the kernels'
///   numerics (`exp2f` via `ex2.approx`, `__expf`, fused mul-adds in the
///   online softmax) are what every cross-stack parity oracle is calibrated
///   against. Turning it off here would produce a kernel no upstream user
///   runs and move every bf16 rounding decision away from the reference
///   the oracles compare with. Scoped to these three TUs only.
/// - `-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
///   -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__`
/// - `-DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI
///   -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_UNEVEN_K` —
///   each forces the template branch this crate's ABI takes anyway
///   (`p_dropout == 0`, no alibi, `softcap == 0`, `head_dim == 64` is a
///   multiple of 32), so they are bit-neutral and cut the instantiation
///   tree ~16x. NOT `DISABLE_LOCAL`: the sliding window is the product.
/// - `-Xcompiler -fPIC` — the one ADDITION to the spike's group, host-side
///   only: Rust links test binaries as PIE on Linux, and a non-PIC static
///   archive fails that link with a relocation error. It changes the host
///   object's relocation model, not the device code.
/// - `-Xptxas -v` — the SECOND addition (M3 plan v2 delta 4): `ptxas`'s
///   verbose flag, printed to stderr per TU and already captured verbatim
///   into `jammi_flash_build_times.txt` by the timing loop below (every
///   TU's stderr is written unconditionally, not only on failure). This is
///   the one failure mode native per-arch SASS uniquely introduces that
///   values-parity testing cannot see on its own: a register spill on a
///   smaller-smem arch (sm86/89's 64×128 bwd tile) is a real perf/
///   correctness-adjacent regression that still produces numerically
///   correct output, so a values oracle passing is not evidence against
///   it. This agent's own hermetic pass has no nvcc to produce real
///   register/spill counts with — see `VENDORED.md`'s "ptxas -v register/
///   spill counts" section for the placeholder table a pod-phase run
///   fills in from this exact stderr capture, never fabricated here.
///
/// Include order: `shim/` FIRST (it provides the `c10/…` and `ATen/…`
/// headers the unmodified upstream files include), then CUTLASS, then
/// `src/`. The wrapper TU also sees `jammi/`.
#[cfg(feature = "flash-attn")]
fn build_flash_attn() {
    use std::process::Command;
    use std::time::Instant;

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let fa_dir = manifest_dir.join("third_party").join("flash-attention");
    let src_dir = fa_dir.join("src");
    let shim_dir = fa_dir.join("shim");
    let jammi_dir = fa_dir.join("jammi");
    let cutlass_include = manifest_dir
        .join("third_party")
        .join("cutlass")
        .join("include");

    // ---- Fail loudly, with the remedy, before spending a minute in nvcc.
    if !cutlass_include.join("cutlass").join("cutlass.h").is_file() {
        panic!(
            "jammi-kernels `flash-attn`: CUTLASS submodule is not checked out at {} — run \
             `git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass` \
             (pinned commit: see crates/jammi-kernels/third_party/flash-attention/VENDORED.md)",
            cutlass_include.display()
        );
    }
    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "jammi-kernels `flash-attn`: no working `nvcc` found — set `NVCC=<path/to/nvcc>` or \
             `CUDA_HOME=<toolkit root>`, or put the CUDA toolkit's `bin` on PATH (the feature \
             needs a CUDA toolkit whose `nvcc` accepts every arch in GENCODE_ARCHES — currently \
             CUDA >= 11.8, for the sm_90 gencode pair; checked for real just below)"
        )
    });
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=NVCC_THREADS");
    println!("cargo:rerun-if-env-changed=JAMMI_FLASH_MEASURE_RSS");

    // ---- Toolkit floor: GENCODE_ARCHES's sm_90 pair needs CUDA >= 11.8
    // (see `check_toolkit_floor`'s own doc for the version-floor rationale
    // and why this is a POSITIVE check — detect the real toolkit release,
    // compare, name the remedy — rather than the crate's old, inaccurate
    // "requires CUDA 12.x" prose). Checked BEFORE spending a minute in the
    // real nvcc compiles below, so a too-old toolkit fails in milliseconds
    // with the exact detected/required versions, not a cryptic nvcc error
    // three TUs deep.
    const TOOLKIT_FLOOR: (u32, u32) = (11, 8);
    let version_output = Command::new(&nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "jammi-kernels `flash-attn`: failed to run `{} --version`: {e}",
                nvcc.display()
            )
        });
    let version_stdout = String::from_utf8_lossy(&version_output.stdout).into_owned();
    match parse_nvcc_release(&version_stdout) {
        Some(detected) => {
            if let Err(remedy) = check_toolkit_floor(detected, TOOLKIT_FLOOR) {
                panic!("jammi-kernels `flash-attn`: {remedy}");
            }
        }
        None => panic!(
            "jammi-kernels `flash-attn`: could not find a \"release <major>.<minor>\" token in \
             `{} --version`'s output:\n{version_stdout}\nThis crate's flash-attn feature needs \
             CUDA >= {}.{} for its sm_90 gencode pair — verify the toolkit manually",
            nvcc.display(),
            TOOLKIT_FLOOR.0,
            TOOLKIT_FLOOR.1
        ),
    }

    // ---- rerun-if-changed for EVERY vendored file (upstream sources, the
    // shims, the wrapper, the CUTLASS include tree — a directory entry
    // makes cargo scan it recursively).
    for dir in [&src_dir, &shim_dir, &jammi_dir] {
        for path in walk_files(dir) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    println!("cargo:rerun-if-changed={}", cutlass_include.display());
    println!(
        "cargo:rerun-if-changed={}",
        fa_dir.join("VENDORED.md").display()
    );

    // ---- `-gencode` flags, one pair per [`GENCODE_ARCHES`] entry (see the
    // doc comment above): the SAME top-level literal array `main()` already
    // parsed to emit `JAMMI_FLASH_GENCODE_SMS`, so the compiled cubin set
    // and the env var `crate::flash::check_arch`/`crate::admission::
    // flash_built_arches()` read can never drift apart — one array, two
    // readers.
    let nvcc_threads: u32 = env::var("NVCC_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4);
    let common_flags: Vec<String> = ["-O3".to_string(), "-std=c++17".to_string()]
        .into_iter()
        .chain(["--threads".to_string(), nvcc_threads.to_string()])
        .chain(
            GENCODE_ARCHES
                .iter()
                .flat_map(|g| ["-gencode".to_string(), g.to_string()]),
        )
        .chain(
            [
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DFLASHATTENTION_DISABLE_ALIBI",
                "-DFLASHATTENTION_DISABLE_SOFTCAP",
                "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                "-Xcompiler",
                "-fPIC",
                "-Xptxas",
                "-v",
            ]
            .iter()
            .map(|s| s.to_string()),
        )
        .chain([
            format!("-I{}", shim_dir.display()),
            format!("-I{}", cutlass_include.display()),
            format!("-I{}", src_dir.display()),
            format!("-I{}", jammi_dir.display()),
        ])
        .collect();

    let tus: [(&str, PathBuf); 3] = [
        (
            "flash_fwd_hdim64_bf16_sm80",
            src_dir.join("flash_fwd_hdim64_bf16_sm80.cu"),
        ),
        (
            "flash_bwd_hdim64_bf16_sm80",
            src_dir.join("flash_bwd_hdim64_bf16_sm80.cu"),
        ),
        ("flash_api_jammi", jammi_dir.join("flash_api_jammi.cu")),
    ];

    // ---- Compile the three TUs concurrently (they are independent; the
    // bwd TU alone is ~70 s on an A100 pod, the fwd ~45 s, the wrapper ~5 s
    // — measured on the OLD single-arch build; four gencodes each cost more
    // wall, see `VENDORED.md`'s re-measured build-times table once a pod
    // run records it, per the M3 plan's D1 cost note).
    //
    // Peak-RSS instrumentation (M3 plan v2 delta 5) is OPT-IN via
    // `JAMMI_FLASH_MEASURE_RSS=1`, NOT autodetected from `/usr/bin/time`'s
    // mere presence: unconditionally wrapping every nvcc invocation with
    // `time -v` would change how this feature compiles on any machine that
    // happens to HAVE a `/usr/bin/time` binary without GNU's `-v` support
    // (BSD/macOS `time` has no `-v` and exits before ever running the
    // wrapped command — this build has no cheap way to probe that ahead of
    // spawning it) — that would turn a build that is hermetic-elsewhere
    // into one that can newly fail on an unrelated machine shape. This PR
    // does not flip the env var on in `ci.yml`; a maintainer confirms the
    // CI flash-attn-compile lane's `time -v` behaviour first (see the
    // hand-off report).
    let measure_rss = env::var_os("JAMMI_FLASH_MEASURE_RSS").is_some();
    let started = Instant::now();
    let handles: Vec<_> = tus
        .iter()
        .map(|(stem, cu)| {
            let nvcc = nvcc.clone();
            let flags = common_flags.clone();
            let obj = out_dir.join(format!("{stem}.o"));
            let cu = cu.clone();
            let stem = stem.to_string();
            std::thread::spawn(move || {
                let t0 = Instant::now();
                let mut cmd = if measure_rss {
                    let mut c = Command::new("/usr/bin/time");
                    c.arg("-v").arg(&nvcc);
                    c
                } else {
                    Command::new(&nvcc)
                };
                let output = cmd
                    .args(&flags)
                    .arg("-c")
                    .arg(&cu)
                    .arg("-o")
                    .arg(&obj)
                    .output()
                    .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", nvcc.display()));
                let secs = t0.elapsed().as_secs_f64();
                if !output.status.success() {
                    panic!(
                        "nvcc failed on {} ({}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
                        cu.display(),
                        output.status,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                let peak_rss_kb = if measure_rss {
                    parse_max_rss_kb(&stderr)
                } else {
                    None
                };
                (stem, obj, secs, stderr, peak_rss_kb)
            })
        })
        .collect();
    let mut objs = Vec::new();
    let mut timing = String::new();
    for h in handles {
        let (stem, obj, secs, stderr, peak_rss_kb) = h.join().expect("nvcc worker thread panicked");
        timing.push_str(&format!("{stem}: {secs:.1} s\n"));
        if let Some(kb) = peak_rss_kb {
            timing.push_str(&format!("{stem} peak_rss_kb: {kb}\n"));
        }
        if !stderr.trim().is_empty() {
            timing.push_str(&format!("{stem} stderr:\n{stderr}\n"));
        }
        objs.push(obj);
    }
    timing.push_str(&format!(
        "wall ({} TUs concurrent, --threads {nvcc_threads}): {:.1} s\n",
        tus.len(),
        started.elapsed().as_secs_f64()
    ));
    std::fs::write(out_dir.join("jammi_flash_build_times.txt"), &timing)
        .expect("write jammi_flash_build_times.txt");

    // ---- Archive.
    let archive = out_dir.join("libjammi_flash.a");
    let _ = std::fs::remove_file(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = Command::new(&ar)
        .arg("rcs")
        .arg(&archive)
        .args(&objs)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn `{ar}`: {e}"));
    assert!(
        status.success(),
        "`{ar} rcs {}` failed: {status}",
        archive.display()
    );

    // ---- Link lines.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=jammi_flash");
    if let Some(lib64) = cuda_lib_dir(&nvcc) {
        println!("cargo:rustc-link-search=native={}", lib64.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    /// `$NVCC`, then `$CUDA_HOME/bin/nvcc`, `$CUDA_PATH/bin/nvcc`, then
    /// `nvcc` on PATH, then `/usr/local/cuda/bin/nvcc` — the first one that
    /// answers `--version`.
    fn find_nvcc() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(p) = env::var("NVCC") {
            candidates.push(PathBuf::from(p));
        }
        for var in ["CUDA_HOME", "CUDA_PATH"] {
            if let Ok(root) = env::var(var) {
                candidates.push(Path::new(&root).join("bin").join("nvcc"));
            }
        }
        candidates.push(PathBuf::from("nvcc"));
        candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
        candidates.into_iter().find(|c| {
            Command::new(c)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        })
    }

    /// `<toolkit>/lib64` derived from the nvcc that was found (or from
    /// `CUDA_HOME` / `CUDA_PATH` / `/usr/local/cuda`), if it exists.
    fn cuda_lib_dir(nvcc: &Path) -> Option<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(bin) = nvcc.parent() {
            if let Some(root) = bin.parent() {
                roots.push(root.to_path_buf());
            }
        }
        for var in ["CUDA_HOME", "CUDA_PATH"] {
            if let Ok(root) = env::var(var) {
                roots.push(PathBuf::from(root));
            }
        }
        roots.push(PathBuf::from("/usr/local/cuda"));
        // The predicate this function's name promises is "the directory
        // that HAS libcudart.so" — check only that. A candidate `lib64`
        // that merely EXISTS (every OTHER `.so` present but not
        // `libcudart.so`) is not a match; it falls through to the next
        // candidate root rather than winning on directory existence alone.
        roots
            .into_iter()
            .map(|r| r.join("lib64"))
            .find(|p| p.join("libcudart.so").is_file())
    }
}
