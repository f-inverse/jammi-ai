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
packaging. Cargo has no per-subtree license field, so the crate-level
`license` stays Apache-2.0 (accurate for the code jammi-ai wrote) and the
`include` list in `Cargo.toml` ships this subtree's `LICENSE`/`AUTHORS`
WITH it in the published tarball (never `exclude` them — that would strip
attribution while still shipping the kernels built from that source); the
comment beside `license` in `Cargo.toml` and the crate doc in `src/lib.rs`
name the exception. `third_party/cutlass` is a git submodule and is NOT in
the published tarball: `flash-attn` is not buildable from crates.io (see
"Checkout"); if it ever is, CUTLASS's own BSD-3 license needs the same
treatment.

| item | value |
|---|---|
| tag | `v2.8.3.post1` |
| commit | `a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2` |
| CUTLASS submodule | `dc4817921edda44a549197ff3a9dcf5df0636e7b` (the tag's `csrc/cutlass` gitlink), vendored as the git submodule `crates/jammi-kernels/third_party/cutlass` |
| upstream directory | `csrc/flash_attn/src/` → `src/` here |
| local edits to upstream files | **none** (see "Shims") |
| kernels compiled | `run_mha_fwd_<cutlass::bfloat16_t, 64, false>`, `run_mha_bwd_<cutlass::bfloat16_t, 64, false>` (head dim 64, bf16, non-causal; native cubins for sm80/86/89/90 — no PTX, see "Supported archs") |
| wrapper | `jammi/flash_api_jammi.{h,cu}` — jammi's own, torch-free, not upstream |

Why this tag: it is the newest `2.8.x` release at vendoring time and the
version whose `flash_api.cpp` / kernel headers the C wrapper's line citations
refer to. No installed `flash_attn` wheel pinned the choice; the
cross-stack oracle installs the matching wheel when it runs.

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

`src/philox_unpack.cuh` is vendored because `flash_fwd_kernel.h:8`
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
| `shim/ATen/cuda/PhiloxCudaState.h` | (not included by the vendored set) | forwards to the above |
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
group, with one native `-gencode` pair per compiled arch instead of a
single one), archived by `ar` into
`$OUT_DIR/libjammi_flash.a`, linked with `static=jammi_flash`,
`dylib=cudart`, `dylib=stdc++`:

```
-O3 -std=c++17
--threads <N>              (N = $NVCC_THREADS if set and > 0, else 4)
-gencode arch=compute_80,code=sm_80
-gencode arch=compute_86,code=sm_86
-gencode arch=compute_89,code=sm_89
-gencode arch=compute_90,code=sm_90
--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
-DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI
-DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_UNEVEN_K
-Xcompiler -fPIC
-Xptxas -v
-Ishim -I<cutlass>/include -Isrc -Ijammi
```

- `--threads <N>` is nvcc's own internal flag (CUDA >= 11.2; this build's
  toolkit floor, CUDA >= 11.8 for the `sm_89`/`sm_90` pairs, exceeds it)
  that parallelizes nvcc's PER-ARCHITECTURE compilation STEPS *within*
  one TU. This is a WALL-TIME flag, NOT a memory optimization: this build
  ALSO spawns its TUs as concurrent processes, so a flat per-TU `N`
  multiplies into `TUs × N` simultaneous nvcc front-ends, each with its
  own footprint — `3 × 4 = 12` is enough to OOM a 16 GB `ubuntu-latest`
  CI runner. `N` defaults to `available_parallelism() / <TU count>`,
  bounding TOTAL front-end concurrency to roughly the machine's own core
  count. Overridable via `$NVCC_THREADS` (a caller who has measured their
  own headroom keeps full control).
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
- `-Xcompiler -fPIC` is an addition to upstream `setup.py`'s per-arch
  groups (host-code relocation model only; Rust links PIE executables on
  Linux). This build does NOT add a bare `code=compute_XX` (embedded PTX)
  entry for ANY of its four `-gencode` pairs: each one appends only
  `code=sm_XX`, matching upstream's own per-arch convention — embedding
  PTX for any arch would ship a second, unvalidated code path an unknown
  future device could JIT into (see "Supported archs" below).
- `-Xptxas -v` is the SECOND addition: `ptxas`'s verbose register/spill
  report, captured into `jammi_flash_build_times.txt` per TU (see "`ptxas
  -v` register/spill counts" below).
- Both traits configs of the backward are instantiated in EVERY compiled
  cubin (a runtime branch, not a build-time choice per arch); the launch
  picks 128×128 (8 warps, 144 KB dynamic smem) when the device's opt-in
  smem allows it (sm80/sm90), else 64×128 (sm86/sm89, 99 KB)
  (`src/flash_bwd_launch_template.h:178-190`). The forward uses 128×128,
  4 warps, on every arch (`src/flash_fwd_launch_template.h:188` has no
  arch branch for the no-dropout path this crate's ABI takes).

### Supported archs

This build compiles NATIVE cubins for the enumerated set
`build.rs::GENCODE_ARCHES` names — sm80/86/89/90 (compute capability
`(8, 0)`/`(8, 6)`/`(8, 9)`/`(9, 0)`) — one `-gencode
arch=compute_XX,code=sm_XX` pair per arch, never a bare `code=compute_XX`
(embedded PTX) entry for any of them. The ABI hard-refuses
(`JAMMI_FLASH_ERR_COMPUTE_CAPABILITY`) below compute capability 8.0.

**COMPILED is not ADMITTED:** `build.rs` ALSO tracks a separate,
NARROWER `VALIDATED_SMS` const — the subset of `GENCODE_ARCHES` with an
actual green per-arch GPU parity leg. Every fence site
(`crate::flash::check_arch`, `jammi-encoders::modernbert`'s
`flash_arch_ok`, `jammi-bench`'s `flash_capable_cuda`) reads
`crate::admission::flash_validated_arches()` — the VALIDATED set — never
`flash_built_arches()`'s merely-compiled one; without that split, a
`-gencode arch=compute_100,code=sm_100` pair added to `GENCODE_ARCHES`
alone would be ADMITTED the instant it compiled. Adding a `-gencode` pair
leaves that arch compiled-but-REFUSED until its OWN entry lands in
`VALIDATED_SMS`, in the SAME commit as its per-arch GPU parity artifact
(table below). Refusal is set membership against the VALIDATED set —
never `>=`, never major-compat.

**The compatibility mechanism:** SASS is minor-version FORWARD-COMPATIBLE
within a major compute-capability generation — the CUDA C++ Programming
Guide's "Binary Compatibility" section (SASS/cubin forward-compat within a
major), a DIFFERENT guarantee from the separate CUDA Compatibility Guide's
driver/runtime-version pairing (which governs whether an OLDER driver can
run a binary built against a NEWER toolkit — not relevant to this crate's
own per-arch cubin question). A device with compute capability `(8, 6)` or
`(8, 9)` CAN load and run `sm_80` SASS under the Binary Compatibility
guarantee; only a genuinely newer MAJOR (`(9, 0)` on an `sm_80`-only
build, or vice versa) truly cannot load the module. Upstream's own
default build (`Dao-AILab/flash-attention`'s `setup.py`, its
`FLASH_ATTN_CUDA_ARCHS` env var / `cuda_archs()` helper) targets
`sm_80;sm_90;sm_100;sm_120` — notably NOT native `sm_86`/`sm_89` entries:
upstream relies on exactly the SASS forward-compat path above to serve RTX
30-series (8.6) and RTX 40-series (8.9) consumer GPUs `sm_80` bytes rather
than compiling for them natively.

jammi does **not** rely on that forward-compat path, deliberately: every
arch in the compiled set gets its OWN native `-gencode` entry and its OWN
GPU parity leg (table below), rather than admitting 8.6/8.9 on the
strength of an untested "SASS should still work here" argument. This
keeps the fence a single equality (device capability ∈ VALIDATED set) that
the Rust check, the encoders/bench fence sites, and this table all read
the same way, and it means every byte the fleet actually executes has a
committed values-parity run on that EXACT silicon, not "upstream says this
should work". Still no `code=compute_XX` PTX anywhere in this build: an
unknown future arch (e.g. a hypothetical `sm_100`) gets a typed refusal
(`FlashError::Arch` / `jammi-encoders`'s `"arch_in_flash_validated_set"`
decline), never unvalidated JIT.

Bwd tile selection (the per-arch table's "bwd tile path" column): both
`Flash_bwd_kernel_traits` configs are instantiated in EVERY compiled cubin
regardless of gencode count — the launch picks 128×128 (8 warps, 144 KB
dynamic smem) when the device's opt-in smem budget allows it (sm80/sm90:
≥144 KB), else 64×128 (sm86/sm89: 99 KB opt-in budget),
`src/flash_bwd_launch_template.h:178-190`. A100 and H100 execute IDENTICAL
bwd tile configs; L40S and A40 execute the smaller 64×128 instantiation —
the one genuinely different code path across the compiled set, and it is
a RUNTIME branch inside a single compiled cubin (present in every arch's
object code), not a build-time choice. The forward uses one 128×128,
4-warp config on every arch (`src/flash_fwd_launch_template.h:181-198`
has no arch branch for the no-dropout path this crate's ABI takes).

Per-arch validation status (**compiled ≠ validated** until a green GPU
parity leg exists on that exact arch; `build.rs::VALIDATED_SMS` is the
actual admitted set every fence reads, and this table is the evidence
pointer for each of its entries — the single source to cross-check when
reasoning about what has ACTUALLY been proven, as opposed to merely
compiled). Per-arch evidence lives in
`crates/jammi-kernels/artifacts/cuda-runs/*-arch-set-*-{a100-sxm4,a40,l40s,h100-hbm3}.json`
(e.g. `2026-09-02-446-arch-set-7567b25-a40.json`).

| arch | compute cap | bwd tile path | GPU parity leg | status |
|---|---|---|---|---|
| sm80 (A100) | `(8, 0)` | 128×128 | Full four-gencode-build suite, ALL legs (`flash_smoke.rs`, `flash_op_oracles.rs`, the padded 8-seed encoder-level oracle + all three negative controls, `cuda_parity.rs`'s flash-relevant legs, bench legs) | **VALIDATED** — fully green on the 4-gencode object |
| sm86 (A40) | `(8, 6)` | 64×128 | `flash_smoke.rs` + `flash_op_oracles.rs` + bench legs + the padded 8-seed encoder-level oracle (all three negative controls) — green. The FOUR 80GB-class encoder-level real-checkpoint tests (`flash_arm_encoder_level_three_way_oracle_dense_cuda_bf16` + its three negative controls) capability-SKIP here (named VRAM-floor reason, `vram_capable_or_skip`) rather than running — that fixture's own doc states its footprint is 80GB-class BY DESIGN (`flash_oracle_measure_arm`'s "holding more than one arm's graph alive at once OOM'd on an 80GB A100, confirmed live"), so a 48GB SKU structurally cannot run it | **VALIDATED for flash-attn admission**, with the coverage caveat above — the 80GB-class legs are proven ONLY on sm80/sm90. `cuda_parity.rs`'s OWN `lora_linear` bf16-backward parity bound has a separate, arch-aware widening on this tile class (unrelated to flash-attn/`GENCODE_ARCHES` admission — a different jammi kernel entirely; see that test's own doc); it does NOT gate this cell |
| sm89 (L40S) | `(8, 9)` | 64×128 | Same leg set/coverage as sm86 (A40) — identical 64×128 tile class | **VALIDATED for flash-attn admission**, same coverage caveat and `lora_linear` cross-reference as sm86 |
| sm90 (H100) | `(9, 0)` | 128×128 | Full four-gencode-build suite, ALL legs, INCLUDING the 80GB-class encoder-level tests (H100 SKUs used are 80GB-class, so no VRAM-floor skip fires here) | **VALIDATED** — fully green, also proves the `sm_90` gencode loads at all (a genuinely different major, not merely a forward-compat question) |

This table's own truth is enforced standing, not merely asserted once and trusted forever: `ci/scripts/check_arch_validation_freshness.py` re-demands the evidence above on every CI run — it requires each `VALIDATED_SMS` entry to have a committed, GREEN, ancestor-sha artifact under `crates/jammi-kernels/artifacts/cuda-runs/` (the discriminator is exactly the `compute_cap <major>.<minor>` substring each `box` field above carries) whose evidence predates no later change to `build.rs`, `src/flash/`, this directory (including this file), or `src/admission.rs`; a later change to any of those without a fresh GPU pass turns a VALIDATED cell red on the next CI run rather than leaving it silently stale.

### `ptxas -v` register/spill counts

**Not yet measured.** `build.rs`'s `common_flags` passes `-Xptxas -v`
unconditionally (every TU, every arch — see the flag group above), whose
report lands in each TU's own stderr and is captured verbatim into
`jammi_flash_build_times.txt` by the per-TU timing loop — filling the
table below is a transcription of that file from a GPU build:

| arch | fwd registers | fwd spill stores/loads | bwd registers | bwd spill stores/loads |
|---|---|---|---|---|
| sm80 | TBD | TBD | TBD | TBD |
| sm86 | TBD | TBD | TBD | TBD |
| sm89 | TBD | TBD | TBD | TBD |
| sm90 | TBD | TBD | TBD | TBD |

Rule: if any arch's TU reports a nonzero spill count, drop that arch's
native `-gencode` entry from `GENCODE_ARCHES` and admit it (if at all)
only via the sm80/sm90-cubin-plus-SASS-minor-compat shape ("The
compatibility mechanism" above describes upstream using exactly this for
8.6/8.9) — never ship a spilling native cubin silently on the theory that
"it still passes values parity" (a spill is a real perf/correctness-
adjacent regression a values oracle cannot see, per the flag's own doc in
`build.rs`).

### Measured compile time

**A100 single-arch baseline** (one `-gencode` pair, bf16 TUs only),
A100-SXM4-80GB, CUDA 12.6 (`V12.6.85`), g++ 13.3.1, 128 vCPU shared with a
concurrent `cargo mutants` run:

| TU | wall |
|---|---|
| `flash_fwd_hdim64_bf16_sm80.cu` | 48.3 s |
| `flash_bwd_hdim64_bf16_sm80.cu` | 76.7 s |
| `flash_api_jammi.cu` | 4.7 s |
| the three, concurrent (what `build.rs` does) | 76.7 s |
| `cargo build -p jammi-kernels --features flash-attn` from a cold target dir (candle + cudarc + this) | 121 s |

`libjammi_flash.a` = 1,325,232 bytes (sm_80 cubin for both kernels'
instantiation trees; NO embedded PTX — the single `-gencode
arch=compute_80,code=sm_80` pair never passes a bare `code=compute_80`
entry, consistent with this crate's no-PTX rule). `nvcc` emitted 0
warnings.

The same flags, single-threaded, on an otherwise idle A100 measured fwd
TU 44 s / 2.9 GB RSS, bwd TU 70 s, 0 warnings, **0 ptxas spills on sm_80
ONLY** (a single-`-gencode` build: sm80's spill count carries no
information about sm86/89/90's own, independently-compiled SASS, so it is
not evidence for the `ptxas -v` table above).

**Multi-arch build: not yet measured.** The expected cost is ~4× the
device-code-section wall above per TU before `--threads`'s own front-end
parallelism absorbs some of it (see `--threads` above for why it is a
WALL-TIME knob, not a memory one). `build.rs` prints per-TU wall to its
own stderr unconditionally; per-TU peak RSS is recorded only when
`JAMMI_FLASH_MEASURE_RSS=1` is set on a machine with GNU `time` installed,
and is diagnostic/per-child ONLY, not the all-TU aggregate a machine's
real RAM headroom depends on. `libjammi_flash.a` grows with the
device-code-section count.

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

The Rust side is `crates/jammi-kernels/src/flash/mod.rs`; the on-GPU
proof is `crates/jammi-kernels/tests/flash_smoke.rs`.
