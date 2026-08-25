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

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FLASH_ATTN");

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

/// Compiles the vendored FlashAttention-2 hdim64/bf16/sm80 forward +
/// backward translation units and jammi's torch-free C wrapper into
/// `$OUT_DIR/libjammi_flash.a` with a hand-rolled `nvcc` invocation, then
/// links it (plus `cudart` and `stdc++`) into this crate.
///
/// Hand-rolled rather than `bindgen_cuda`: the 0.1.6 `Builder` has no
/// CUTLASS include hook, `build_lib` emits no `-I`, and `.arg` takes
/// `&'static str` — it cannot express this flag group. The existing PTX
/// path above is untouched.
///
/// FLAG GROUP — upstream `setup.py`'s nvcc group for sm80, WITH ONE
/// DELIBERATE OMISSION (see below), as measured in the build spike
/// (`third_party/flash-attention/VENDORED.md`):
///
/// - `-O3 -std=c++17`
/// - `-gencode arch=compute_80,code=sm_80` (sm_80 cubin ONLY — upstream
///   `setup.py` appends only `code=sm_80` for this arch too; an earlier
///   revision of this file ALSO appended `code=compute_80` (embedded PTX),
///   which is NOT in upstream's group and was never validated: shipping
///   that PTX would mean an 8.6/8.9/9.0 device JITs it at RUNTIME and takes
///   a DIFFERENT `Flash_bwd_kernel_traits` branch (the 64×128 config,
///   `flash_bwd_launch_template.h:178-190`, vs the 128×128 this crate's
///   smoke test actually exercises on the A100) — a second shipped code
///   path with zero oracles. Dropped rather than validated because no jammi
///   deployment target has been identified that needs a non-sm80 GPU; if
///   one is, the PTX gencode returns WITH its own oracle run on that arch,
///   not silently. Every jammi deployment target for this feature is sm80
///   (A100) until stated otherwise — see `VENDORED.md`'s "Supported archs"
///   note.
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
///
/// Include order: `shim/` FIRST (it provides the `c10/…` and `ATen/…`
/// headers the unmodified upstream files include), then CUTLASS, then
/// `src/`. The wrapper TU also sees `jammi/`.
#[cfg(feature = "flash-attn")]
fn build_flash_attn() {
    use std::path::{Path, PathBuf};
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
             requires CUDA 12.x with sm_80 support)"
        )
    });
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AR");

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

    // ---- The flag group (see the doc comment above).
    let common_flags: Vec<String> = [
        "-O3",
        "-std=c++17",
        "-gencode",
        "arch=compute_80,code=sm_80",
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
    ]
    .iter()
    .map(|s| s.to_string())
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
    // bwd TU alone is ~70 s on an A100 pod, the fwd ~45 s, the wrapper ~5 s).
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
                let output = Command::new(&nvcc)
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
                (
                    stem,
                    obj,
                    secs,
                    String::from_utf8_lossy(&output.stderr).into_owned(),
                )
            })
        })
        .collect();
    let mut objs = Vec::new();
    let mut timing = String::new();
    for h in handles {
        let (stem, obj, secs, stderr) = h.join().expect("nvcc worker thread panicked");
        timing.push_str(&format!("{stem}: {secs:.1} s\n"));
        if !stderr.trim().is_empty() {
            timing.push_str(&format!("{stem} stderr:\n{stderr}\n"));
        }
        objs.push(obj);
    }
    timing.push_str(&format!(
        "wall (3 TUs concurrent): {:.1} s\n",
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
        // F6: the previous predicate was `p.join("libcudart.so").is_file()
        // || p.is_dir()` — since a candidate `lib64` directory almost
        // always exists once CUDA is installed at all, `p.is_dir()` wins
        // for the FIRST candidate root regardless of whether `libcudart.so`
        // is actually in it, making the specific library check inert (a
        // `lib64` with every OTHER `.so` but not `libcudart.so` would still
        // "find" a match). The predicate this function's name promises is
        // "the directory that HAS libcudart.so" — check only that; a
        // directory existing without the library is not a match, it falls
        // through to the next candidate root.
        roots
            .into_iter()
            .map(|r| r.join("lib64"))
            .find(|p| p.join("libcudart.so").is_file())
    }
}
