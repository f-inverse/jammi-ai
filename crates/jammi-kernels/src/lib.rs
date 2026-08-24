//! Candle `CustomOp` scaffolding and a feature-gated CUDA build path for
//! fused training kernels — elementwise / normalization / attention ops
//! whose CPU implementation is real (not a stub) and whose CUDA
//! implementation shares a single call path with it.
//!
//! This crate is a leaf: it depends on `candle-core` / `candle-nn` only, no
//! `jammi-*` crate, and it names no consumer. [`ops::Axpy`] is the proof op
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
