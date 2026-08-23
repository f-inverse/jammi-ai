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

## Non-goals of this crate

No architecture-specific code, no fusion policy for *which* op runs where —
that lives at each consumer's call site. This crate owns mechanism
(the `CustomOp` machinery, the build path, the admission primitives), not
which model uses it or when.
