# jammi-kernels

Candle `CustomOp` scaffolding and a feature-gated CUDA build path for fused
training kernels. Part of [Jammi AI](https://github.com/f-inverse/jammi-ai).

Leaf crate: depends only on `candle-core` / `candle-nn` (plus `cudarc`,
transitively, via candle's own `cuda` feature). No `jammi-*` dependencies —
this crate names no consumer and is not yet depended on by any other
workspace crate.

## What's here

- `admission` — the runtime scaffolding every fused op's call site uses to
  decide fused-vs-eager: a CUDA compute-capability probe (fused kernels need
  `sm_80`+), per-op dispatch counters, a log-once-per-process WARN helper,
  and a `Strict` mode that turns a failed domain check into a hard error
  instead of a silent fallback.
- `ops::Axpy` — `y' = alpha * x + y`, the proof op. A `CustomOp2` with real
  CPU forward/backward (F32, F64, BF16) and, behind the `cuda` feature, a
  CUDA forward loaded from build-time-compiled PTX via
  `CudaDevice::get_or_load_custom_func`. Stateless: `alpha` is construction
  data, nothing is saved between forward and backward.
- `build.rs` — early-returns unless the `cuda` feature is active. Under the
  feature, compiles `src/cuda/*.cu` to PTX via `bindgen_cuda`, pinned to the
  `sm_80` baseline (the driver JIT-forwards to 8.6/8.9/9.0) with no
  `-use_fast_math`. This is not a bit-exact-parity guarantee: nvcc's
  separate, on-by-default `--fmad` contraction can still fuse a
  multiply-add into one rounding on the GPU where the CPU arm does two —
  every fused kernel's CPU↔CUDA parity oracle states a tolerance for that,
  it does not assert bit-exact equality (see `build.rs`'s PINNED FLAGS
  comment for the full reasoning, including the one case — the C7
  device-side-dropout plan — that does need bit-exact parity on a specific
  expression, and pins that locally with rounding intrinsics rather than a
  global flag).

- `flash` (feature `flash-attn`, implies `cuda`, never implied by it) — the
  vendored FlashAttention-2 hdim64/bf16/sm80 varlen forward + backward
  kernels (`third_party/flash-attention/`, tag v2.8.3.post1, unmodified,
  behind jammi's torch-free C wrapper `flash_api_jammi.cu`) and the safe
  Rust boundary over them: `flash_varlen_fwd` → `(o, lse)`,
  `flash_varlen_bwd` → `d_qkv`, packed `[total_q, 3, H, 64]` layout,
  `cu_seqlens` varlen, symmetric window, deterministic backward. Built by
  `build.rs` with a hand-rolled `nvcc` (upstream's flag group, the crate's
  one `--use_fast_math` unit) into `libjammi_flash.a`. Needs the CUDA
  toolkit AND the CUTLASS submodule:

  ```sh
  git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass
  cargo build -p jammi-kernels --features flash-attn
  JAMMI_REQUIRE_CUDA=1 cargo test -p jammi-kernels --features flash-attn --test flash_smoke
  ```

  Provenance, file hashes, shims, flags and measured compile time:
  `third_party/flash-attention/VENDORED.md`. No consumer's feature closure
  reaches `flash-attn` (`ci/scripts/check_flash_attn_closure.py`); it is
  not part of `cuda` and not default.

## Non-goals of this crate

No architecture-specific code, no fusion policy for *which* op runs where —
that lives at each consumer's call site. This crate owns mechanism
(the `CustomOp` machinery, the build path, the admission primitives), not
which model uses it or when.
