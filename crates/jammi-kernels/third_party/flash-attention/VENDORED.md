# Vendored FlashAttention-2 (Dao-AILab/flash-attention)

Upstream: <https://github.com/Dao-AILab/flash-attention>, BSD-3-Clause
(`LICENSE` in this directory is the upstream file, verbatim; its notice at
line 3, "as shown by the AUTHORS file", refers to the sibling `AUTHORS`
file in this same directory — also vendored verbatim, see the file table
below).

## Licensing (jammi-kernels is Apache-2.0; this directory is BSD-3)

`crates/jammi-kernels/Cargo.toml` inherits `license = "Apache-2.0"` from
the workspace (`license.workspace = true`). Everything under
`third_party/flash-attention/{src,jammi,shim}` and this file's sibling
`LICENSE`/`AUTHORS` is BSD-3-Clause (Dao-AILab) — a DIFFERENT license than
the crate that vendors it, which is legal (Apache-2.0 code may embed BSD-3
code with its notice retained, as here) but must not be silently erased by
packaging. Two consequences, flagged to the lead as a shared-declaration
(`Cargo.toml`) change this crate does not make unilaterally:
- `cargo publish -p jammi-kernels` runs on release tags via
  `.github/workflows/crates.yml:344` with NO `include`/`exclude` in
  `Cargo.toml` today — a published crate tarball would carry BSD-3 sources
  under an `Apache-2.0`-declared `license`, with the LICENSE/AUTHORS text
  present but the top-level `license` field not disclosing the mix.
- Proposed `Cargo.toml` shape (NOT applied here — shared-declaration file,
  routed to the lead):
  ```toml
  [package]
  license = "Apache-2.0"
  # third_party/flash-attention/{src,jammi,shim,LICENSE,AUTHORS,VENDORED.md}
  # is Dao-AILab's BSD-3-Clause code, vendored (not re-licensed); the
  # crate's own code stays Apache-2.0. Cargo has no per-subtree license
  # field, so this is surfaced two ways: (1) `license-file` conventions
  # cannot express "mostly Apache-2.0, one vendored subtree BSD-3-Clause" —
  # the crate-level `license` stays "Apache-2.0" (accurate for the code
  # jammi-ai wrote) and the vendored subtree's own LICENSE/AUTHORS travel
  # WITH it in the published tarball (do NOT `exclude` them — that would
  # strip attribution while still shipping the compiled kernels built from
  # that source); (2) this note, and a one-line pointer in the crate's own
  # top-level doc comment / README, name the exception explicitly for
  # anyone auditing the published crate.
  include = [
      "src/**", "build.rs", "README.md",
      "third_party/flash-attention/src/**",
      "third_party/flash-attention/jammi/**",
      "third_party/flash-attention/shim/**",
      "third_party/flash-attention/LICENSE",
      "third_party/flash-attention/AUTHORS",
      "third_party/flash-attention/VENDORED.md",
      # third_party/cutlass is a git submodule, NOT vendored into this
      # crate's published tarball at all today (build.rs panics without it
      # checked out) — `flash-attn` was never a publishable-from-crates.io
      # feature; if that changes, CUTLASS's own license (BSD-3) needs the
      # same treatment.
  ]
  ```
  This is a PROPOSAL for the lead/docs-ci lane to apply; this round does
  not edit `Cargo.toml` (shared-declaration class).

| item | value |
|---|---|
| tag | `v2.8.3.post1` |
| commit | `a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2` |
| CUTLASS submodule | `dc4817921edda44a549197ff3a9dcf5df0636e7b` (the tag's `csrc/cutlass` gitlink), vendored as the git submodule `crates/jammi-kernels/third_party/cutlass` |
| upstream directory | `csrc/flash_attn/src/` → `src/` here |
| local edits to upstream files | **none** (see "Shims") |
| kernels compiled | `run_mha_fwd_<cutlass::bfloat16_t, 64, false>`, `run_mha_bwd_<cutlass::bfloat16_t, 64, false>` (head dim 64, bf16, non-causal, sm80 cubin ONLY — no PTX, see "Supported archs") |
| wrapper | `jammi/flash_api_jammi.{h,cu}` — jammi's own, torch-free, not upstream |

Why this tag: it is the newest `2.8.x` release at vendoring time and the
version whose `flash_api.cpp` / kernel headers the C wrapper's line citations
refer to. (The pod's PyTorch reference venv had no `flash_attn` wheel
installed — `ModuleNotFoundError` — so no wheel version pinned the choice;
the cross-stack oracle installs the matching wheel when it runs.)

## Checkout

The CUTLASS headers are a git submodule and are NOT in the repository
snapshot. Before building with `--features flash-attn`:

```sh
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass
```

`.gitmodules` marks it `shallow = true`; the pinned commit is fetched by
SHA (GitHub serves reachable SHAs). `build.rs` fails with this exact
command in its message when `include/cutlass/cutlass.h` is missing.
Nothing else is fetched at build time.

## Files (sha256 of the vendored copy; every one byte-identical to the tag)

| file | sha256 |
|---|---|
| `src/alibi.h` | `ba61a17e2d3a22d81a27e6f69302d27c887f79f369b288ab1fa1c6ff8a02450d` |
| `src/block_info.h` | `89cb30fd392372f0313db50cde8a93632ae8dc8c02a5902a245d985fc3fa609b` |
| `src/dropout.h` | `a1d14c0c10227d67bfd36bada415565f4deefa604e6eb1863d7dcc0fd8f12098` |
| `src/flash.h` | `8aff12bf7cab089990eb4a0b87349df8445c04515b7e64a47a934e7478fb2130` |
| `src/flash_bwd_hdim64_bf16_sm80.cu` | `10a9230f4b2474468a978170e5a18730a3bd9f263f7e496f8aea46488f371627` |
| `src/flash_bwd_kernel.h` | `bbfa29e50d509c08de34a9766adda45fc6edba884ef00afc5ad21a36255fd58e` |
| `src/flash_bwd_launch_template.h` | `2369366e77210f7705d58e1e8aea4559322ca2fb2263954ea636841f6a09cf4a` |
| `src/flash_bwd_preprocess_kernel.h` | `7f35a67c2576c3b810199c7881b3660059353cff02c370d0ace2bb7e85937ff7` |
| `src/flash_fwd_hdim64_bf16_sm80.cu` | `74aa25c8c9a655af59082976dfc8f65b7417ff501b6f0fba88760d672143230f` |
| `src/flash_fwd_kernel.h` | `765dd3ef217bc9d79c9c0494ba52ea63767099be737c14604bec748d85f0dde3` |
| `src/flash_fwd_launch_template.h` | `d9e9f4b92cb731d7955b514449e59b8e411bf7a0c929aafb454f2402d41fe976` |
| `src/hardware_info.h` | `deef803b86266ec3f3c6f3f3a4c414ad91cc92d637ba9be02eec520db712cfe3` |
| `src/kernel_traits.h` | `880a6f432613dc01979a605d304da75985f9ed34ff66f1f4ded85fc90b7047ac` |
| `src/mask.h` | `4ea99928f81a7861894a6fbeb9ab127abccce49fdb81fdca0b8e570b3f6f1a22` |
| `src/namespace_config.h` | `2f50a22748285224426a7d374715fc63123ad4b4153e1f46b455a87a8a07f4a5` |
| `src/philox.cuh` | `81ef706f740783b4e035b8a78d7f579ee5750d4defdb2068781b3b7cb027abee` |
| `src/philox_unpack.cuh` | `14d35e2e51b5f248d7007d4f783c46c29f5803444ba7d4e30204a50fd6811ac5` |
| `src/rotary.h` | `f000eba29dade79f629849ed5dbce654d475d61826142a0f94011506be3d2427` |
| `src/softmax.h` | `77e70b8d0dbe72f459ced29199a554a0d1e0559cad2370a2e5b5f1fa5faf73f1` |
| `src/static_switch.h` | `52da74dec74f8b9a26ef34bf4a311950c28c0c77b73eab8b9a169f8e65ae1616` |
| `src/utils.h` | `abc6d4c73522af35f6c31be3938830b9474f620d7c8083168d80dab226a0fdc9` |
| `LICENSE` | `8c9ccb96c065e706135b6cbad279b721da6156e51f3a5f27c6b3329af9416d73` |
| `AUTHORS` | `b102042819a44c39dd7af31e8f29358db944bd6eb372459fba3529c2e0584523` |

`src/philox_unpack.cuh` is one file beyond the planned list: `flash_fwd_kernel.h:8`
includes it (quoted, same directory) and it is a two-line forward to
`<ATen/cuda/detail/UnpackRaw.cuh>` — vendored unmodified rather than shimmed
under upstream's own name. Every other quoted include among the vendored
files resolves inside `src/`; the only angle-bracket includes outside CUDA
and the C++ standard library are the three torch/ATen headers below.

Re-verify against a fresh clone with
`git clone --depth 1 --branch v2.8.3.post1 https://github.com/Dao-AILab/flash-attention && (cd flash-attention/csrc/flash_attn/src && sha256sum <file>)`.

## Shims (`shim/`, first on the include path — how the upstream files stay unmodified)

The vendored headers include three PyTorch/ATen headers for two macros and
one struct. `shim/` provides same-named replacements so no upstream line
changes; `build.rs` puts `-Ishim` before `-I<cutlass>/include -Isrc`.

| shim | replaces | provides |
|---|---|---|
| `shim/c10/cuda/CUDAException.h` | `<c10/cuda/CUDAException.h>` (`flash_{fwd,bwd}_launch_template.h:7-8`) | `C10_CUDA_CHECK(expr)` / `C10_CUDA_KERNEL_LAUNCH_CHECK()` → `cudaGetErrorString` to stderr + `abort()`. These guard `cudaFuncSetAttribute` and the kernel LAUNCH (configuration errors, or a sticky error from an earlier asynchronous fault) — programming errors, not data; the wrapper's own status codes cover every data-level refusal before a launch is attempted. |
| `shim/ATen/cuda/CUDAGeneratorImpl.h` | `<ATen/cuda/CUDAGeneratorImpl.h>` (`flash.h:12`) | `at::PhiloxCudaState { seed_, offset_ }` (the field `Flash_fwd_params::philox_args`, `flash.h:122`) and an empty `at::Generator`. Dropout is compiled out, so the state is never read. |
| `shim/ATen/cuda/PhiloxCudaState.h` | (not included by the vendored set; kept for completeness with the spike) | forwards to the above |
| `shim/ATen/cuda/detail/UnpackRaw.cuh` | `<ATen/cuda/detail/UnpackRaw.cuh>` (via `src/philox_unpack.cuh`) | `at::cuda::philox::unpack(PhiloxCudaState) -> (seed, offset)` |

sha256 of the shims: `c10/cuda/CUDAException.h`
`d3d96dbba904af92ad9c4cc8c7e5abee17d9ff5e5b36b70b3c1e3360d1843c45`,
`ATen/cuda/CUDAGeneratorImpl.h`
`113bdcb2b441c376cea1c4e9030daef57aaf049c42c9c1a2b3351fe65df8450b`,
`ATen/cuda/PhiloxCudaState.h`
`7db9ae3933e942882623e6da0ba9bd6b6ba2fe1fefd9ec25565b6c55fdd22ce1`,
`ATen/cuda/detail/UnpackRaw.cuh`
`60f2251b0761146fe39ee08520a91b5e84b8e74f4d941c8d2b058ec13daff195`.

## Build (`crates/jammi-kernels/build.rs::build_flash_attn`, feature `flash-attn`)

Three translation units — `src/flash_fwd_hdim64_bf16_sm80.cu`,
`src/flash_bwd_hdim64_bf16_sm80.cu`, `jammi/flash_api_jammi.cu` — compiled
concurrently by `nvcc` with EXACTLY this flag group (upstream `setup.py`'s
sm80 group), archived by `ar` into `$OUT_DIR/libjammi_flash.a`, linked with
`static=jammi_flash`, `dylib=cudart`, `dylib=stdc++`:

```
-O3 -std=c++17
-gencode arch=compute_80,code=sm_80
--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
-DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI
-DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_UNEVEN_K
-Xcompiler -fPIC
-Ishim -I<cutlass>/include -Isrc -Ijammi
```

- `--use_fast_math` is the crate's ONE fast-math translation-unit group.
  `src/cuda/*.cu` (the crate's own PTX kernels) are built without it
  (`build.rs::build_cuda`). Upstream ships every FlashAttention-2 wheel with
  it; the kernels' numerics (`exp2f`/`__expf`, contracted mul-adds in the
  online softmax) are what every parity oracle is calibrated against —
  compiling without it would produce a kernel no upstream user runs.
- The four `FLASHATTENTION_DISABLE_*` defines each force the template
  branch the wrapper's ABI takes anyway (`p_dropout == 0`, no alibi,
  `softcap == 0`, `head_dim == 64` is a multiple of 32) — bit-neutral,
  ~16× fewer instantiations. `FLASHATTENTION_DISABLE_LOCAL` is NOT set:
  the sliding window is the product.
- `-Xcompiler -fPIC` is the ONE addition to upstream `setup.py`'s sm80
  group (host-code relocation model only; Rust links PIE executables on
  Linux). An EARLIER revision of this file also documented a SECOND
  addition — `-gencode arch=compute_80,code=compute_80` (embedded PTX) —
  that was wrong on two counts: it was not in upstream's group (`setup.py`
  appends only `code=sm_80` for the sm80 arch too), and it was never
  validated (see "Supported archs" below). Dropped.
- Both traits configs of the backward are instantiated; the launch picks
  128×128 (8 warps, 144 KB dynamic smem) when the device's opt-in smem
  allows it (A100/H100), else 64×128 (sm86/sm89, 99 KB)
  (`src/flash_bwd_launch_template.h:178-190`). The forward uses 128×128,
  4 warps (`src/flash_fwd_launch_template.h:188`).

### Supported archs

This build emits ONLY an sm_80 cubin (no PTX) — the ABI hard-refuses
(`JAMMI_FLASH_ERR_COMPUTE_CAPABILITY`) below compute capability 8.0, and a
device above sm80 (8.6/8.9/9.0) simply cannot LOAD this module at all
(no PTX to JIT), rather than silently loading the wrong-tile-config kernel.
This is deliberate: shipping the PTX would let 8.6/8.9/9.0 devices JIT a
DIFFERENT `Flash_bwd_kernel_traits` branch (the 64×128 config vs. this
crate's 128×128, `flash_bwd_launch_template.h:178-190`) with ZERO oracles
run against it — `tests/flash_smoke.rs`'s landing proof runs on an A100
(sm80) only. If a jammi deployment target needs sm86/89/90, the PTX
gencode returns together with its own build-and-parity run on that arch,
recorded here — not silently re-added.

### Measured compile time

A100-SXM4-80GB pod, CUDA 12.6 (`V12.6.85`), g++ 13.3.1, 128 vCPU shared with
a concurrent `cargo mutants` run (see the landing commit for the raw
`jammi_flash_build_times.txt`):

| TU | wall |
|---|---|
| `flash_fwd_hdim64_bf16_sm80.cu` | 48.3 s |
| `flash_bwd_hdim64_bf16_sm80.cu` | 76.7 s |
| `flash_api_jammi.cu` | 4.7 s |
| the three, concurrent (what `build.rs` does) | 76.7 s |
| `cargo build -p jammi-kernels --features flash-attn` from a cold target dir (candle + cudarc + this) | 121 s |

`libjammi_flash.a` = 1,325,232 bytes (sm_80 cubin + compute_80 PTX for both
kernels' instantiation trees). `nvcc` emitted 0 warnings.

The build spike (same flags, single-threaded, otherwise idle pod) measured
fwd TU 44 s / 2.9 GB RSS, bwd TU 70 s, 0 warnings, 0 ptxas spills.

## Feature isolation

`flash-attn = ["cuda"]` in `Cargo.toml`: it needs candle's CUDA device and
therefore implies `cuda`, but `cuda` never implies it and it is not a
default feature. Two guards:

- `ci/scripts/check_flash_attn_closure.py` — walks `cargo metadata
  --no-deps` from `jammi-server` under `default`, the release lane's
  `cuda,jetstream-broker,storage-cloud` and `--all-features`, and fails
  if `jammi-kernels/flash-attn` is reached (positive control: the cuda
  lane MUST reach `jammi-kernels/cuda`). `--self-test` exercises a leaked
  edge, a self-implication and a weak-dep edge.
- `crates/jammi-kernels/tests/feature_table.rs` — asserts this crate's own
  `default` and `cuda` entries do not name `flash-attn`; runs in every
  default `cargo test -p jammi-kernels`.

## What the wrapper serves

`jammi/flash_api_jammi.h` documents the ABI: the qkv-PACKED varlen case
(`flash_attn_varlen_qkvpacked_func` upstream), `[total_q, 3, H, 64]` bf16
in, `o` `[total_q, H, 64]` + `lse` `[H, total_q]` (`unpadded_lse`) out,
backward into a packed `d_qkv` with scratch `softmax_d`
`[H, total_q + 128·B]` and `dq_accum` `[splits, total_q + 128·B, H, 64]`.
Every `Flash_{fwd,bwd}_params` field is set in `flash_api_jammi.cu` with a
citation to the `flash_api.cpp` line it mirrors; every applicable
`TORCH_CHECK` is a returned status code (`jammi_flash_status`) with a
static message (`jammi_flash_strerror`). `p_dropout != 0` is a hard error
(the build would otherwise silently ignore it), `params = {}` zero-init
is kept, and `num_splits <= 1` is asserted — the split-KV forward and the
causal instantiation are neither compiled nor reachable.

The Rust side is `crates/jammi-kernels/src/flash/mod.rs`; the pod-run
proof is `crates/jammi-kernels/tests/flash_smoke.rs`.
