//! Compiles `src/cuda/*.cu` to PTX — ONLY when the `cuda` feature is active.
//!
//! Every other build (the laptop/CI default across this workspace) must not
//! require nvcc, a CUDA toolkit, or any system requirement beyond a plain
//! `cargo build`. `CARGO_FEATURE_CUDA` is the env var Cargo sets for build
//! scripts exactly when a crate's own `cuda` feature is enabled; checking it
//! here (rather than relying on `#[cfg(feature = "cuda")]` alone) makes the
//! early-return the single, auditable gate a reviewer can point at.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");

    // Default build: no cuda feature, no nvcc, no CUDA toolkit. Nothing
    // below this line ever runs unless the `cuda` feature set this env var.
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        #[cfg(feature = "cuda")]
        {
            build_cuda();
        }
    }
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    use bindgen_cuda::Builder;

    // NOTE on rerun-if-changed: we do NOT emit our own line for
    // `src/cuda/*.cu` here — `Builder::build_ptx()` already emits
    // `cargo:rerun-if-changed=<path>` for every kernel path the glob below
    // actually resolves to (bindgen_cuda 0.1.6, `build_ptx`'s
    // `kernel_paths.par_iter()` loop), so it is automatically
    // glob-consistent: a new `src/cuda/whatever.cu` added later is
    // rerun-tracked with no edit needed here. A hardcoded
    // `rerun-if-changed=src/cuda/axpy.cu` line (this file's previous
    // version) would silently stop covering a second kernel file.

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
