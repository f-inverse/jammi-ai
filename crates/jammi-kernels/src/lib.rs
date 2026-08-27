//! Candle `CustomOp` scaffolding and a feature-gated CUDA build path for
//! fused training kernels — elementwise / normalization / attention ops
//! whose CPU implementation is real (not a stub) and whose CUDA
//! implementation shares a single call path with it.
//!
//! This crate is a leaf: it depends on `candle-core` / `candle-nn` only, no
//! `jammi-*` crate, and it names no consumer.
//!
//! Licensing: the crate is Apache-2.0, with one exception —
//! `third_party/flash-attention/` is Dao-AILab's FlashAttention-2 source,
//! vendored verbatim under its own BSD-3-Clause license (its `LICENSE` and
//! `AUTHORS` ship in the published tarball; see that directory's
//! `VENDORED.md` for provenance and the exact pinned upstream version).
//!
//! [`ops::Axpy`] is the proof op
//! establishing the pattern every later fused op copies (real CPU fwd/bwd, a
//! feature-gated CUDA fwd loaded from build-time PTX, statelessness enforced
//! structurally — every op is required to be `Copy`, see `ops`'s module
//! doc); [`admission`] is the runtime scaffolding — a CUDA
//! compute-capability probe, per-op dispatch counters, a log-once WARN, and
//! a `Strict` mode — every later fused op's call site uses to decide
//! fused-vs-eager and make that decision observable.

pub mod admission;
pub mod error;
mod layout_walk;
pub mod ops;
/// Philox4x32-10 (ported from Random123, BSD-3-Clause — see the module's
/// own doc for the full provenance/citation) and the `(seed, layer,
/// forward#, index) -> u32` counter mapping [`ops::DropoutFused`] builds
/// on. Public so the crate's own test suite (`tests/*.rs`, a separate
/// compilation unit) and the CUDA parity suite can both exercise the
/// published known-answer test vectors directly.
pub mod philox;

#[cfg(feature = "cuda")]
mod cuda;

/// Vendored FlashAttention-2 varlen forward/backward behind a torch-free C
/// ABI — the FFI declarations, the safe Rust entry points, and the scratch
/// allocation rules. Feature `flash-attn` only (implies `cuda`); no
/// `KernelOp` here yet — this is the kernel boundary a later op composes.
#[cfg(feature = "flash-attn")]
pub mod flash;
