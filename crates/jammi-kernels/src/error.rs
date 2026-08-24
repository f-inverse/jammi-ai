//! This crate's own error type.
//!
//! Errors a `CustomOp`'s `cpu_fwd` / `cuda_fwd` / `bwd` raise inside candle's
//! call graph are `candle_core::Error` — every domain-validity refusal
//! inside an op (shape mismatch, unsupported dtype, a non-contiguous view on
//! the raw-pointer CUDA path) reuses candle's own error variants
//! (`ShapeMismatchBinaryOp`, `DTypeMismatchBinaryOp`,
//! `UnsupportedDTypeForOp`, `RequiresContiguous`) rather than wrapping them,
//! since `CustomOp2`'s trait methods are fixed to return `candle_core::Result`.
//! `KernelError` is for the admission scaffolding, which is not bound to that
//! trait signature.
use thiserror::Error;

/// Errors surfaced by the admission scaffolding (`crate::admission`).
#[derive(Debug, Error)]
pub enum KernelError {
    /// STRICT admission mode (scope-6 / K2 in the fused-kernels plan): a
    /// caller explicitly requested the fused path and the op's domain
    /// predicate failed. A silent fallback here would hide a real
    /// perf-or-correctness question from a strict-mode caller, so STRICT
    /// mode errors instead of falling back.
    #[error("fused op `{op}` refused in STRICT admission mode: predicate `{predicate}` failed")]
    StrictModeFallback {
        op: &'static str,
        predicate: &'static str,
    },
}

/// Crate-local `Result` alias for the admission scaffolding.
pub type Result<T> = std::result::Result<T, KernelError>;
