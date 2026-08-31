//! This crate's own error type.
//!
//! Errors a `CustomOp`'s `cpu_fwd` / `cuda_fwd` / `bwd` raise inside candle's
//! call graph are `candle_core::Error` — every domain-validity refusal
//! inside an op (shape mismatch, unsupported dtype, a non-contiguous view on
//! the raw-pointer CUDA path) reuses candle's own error variants
//! (`ShapeMismatchBinaryOp`, `DTypeMismatchBinaryOp`,
//! `UnsupportedDTypeForOp`, `RequiresContiguous`) rather than wrapping them,
//! since `CustomOp2`'s trait methods are fixed to return `candle_core::Result`.
//! `KernelError` is for errors raised OUTSIDE that trait boundary — both the
//! admission scaffolding (`crate::admission`) and an op's own
//! construction-time domain validation (e.g.
//! `ops::softmax::SoftmaxLastDimFused::with_scale`, which runs before any
//! `CustomOp2` method and so is not bound to `candle_core::Result` either).
use thiserror::Error;

/// Errors surfaced outside the `CustomOp` trait boundary: the admission
/// scaffolding (`crate::admission`) and an op's own construction-time
/// domain validation.
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
    /// Domain-validity refusal (family D) for a multiplicative scale
    /// construction argument (e.g. `ops::softmax::SoftmaxLastDimFused::with_scale`):
    /// `scale` must be finite and strictly positive. `0.0` would silently
    /// yield a uniform-attention forward (`scale * scores == 0` everywhere,
    /// so only `mask` survives the reduction) rather than an error, and a
    /// negative or non-finite value has no meaning as an attention scale at
    /// all — refused at construction rather than propagated as a confident
    /// wrong number.
    #[error("invalid scale {scale}: must be finite and > 0.0")]
    InvalidScale { scale: f32 },
    /// Domain-validity refusal (family D / K2): `crate::quantized_cuda_canary`'s
    /// load-time known-answer check failed on BOTH the quantized fast-kernel
    /// CUDA path AND the legacy PTX-JIT'd DMMV fallback `set_force_dmmv(true)`
    /// routes to (see that module's own doc for the full failure class this
    /// guards against — issue #434). Refusal beats a confident wrong number:
    /// this is returned instead of letting a quantized CUDA matmul run when
    /// neither known-correct path could be proven trustworthy on this device.
    #[error(
        "quantized-CUDA canary failed on both the fast kernel path and the legacy DMMV \
         fallback (see issue #434) -- refusing to run a quantized matmul on this device \
         rather than risk silent garbage"
    )]
    QuantizedCudaCanaryFailed,
}

/// Crate-local `Result` alias for the admission scaffolding.
pub type Result<T> = std::result::Result<T, KernelError>;

/// The ONE conversion from this crate's own [`KernelError`] to
/// `candle_core::Error`, for the handful of call sites that must cross that
/// boundary (a `CustomOp` trait method fixed to `candle_core::Result`, or a
/// `pub fn` whose own signature is `candle_core::Result` for symmetry with
/// the rest of this crate's public surface — `crate::quantized_cuda_canary`'s
/// `ensure_quantized_cuda_admitted` and
/// `ops::low_rank_residual_linear::admit_cast_boundary`'s `bwd` call sites,
/// as of this writing).
///
/// Wraps via `Error::Cuda(Box::new(err))` — NOT `Error::Msg(err.to_string())`
/// — deliberately: `Error::Cuda`'s payload is `Box<dyn std::error::Error +
/// Send + Sync>`, so a caller downstream can
/// `std::error::Error::downcast_ref::<KernelError>()` back to the ORIGINAL
/// typed value (e.g. distinguishing [`KernelError::QuantizedCudaCanaryFailed`]
/// from [`KernelError::StrictModeFallback`] programmatically) rather than
/// being reduced to a `String` only a caller could pattern-match by text.
/// The variant name `Cuda` is candle-core's own naming for "an opaque
/// downstream error boxed through," not a claim that `err` is CUDA-specific
/// — `KernelError::InvalidScale`, raised on every device, would flow through
/// the identical channel. See `quantized_cuda_canary::tests` for the
/// round-trip proof (construct a `KernelError`, convert, downcast back).
impl From<KernelError> for candle_core::Error {
    fn from(err: KernelError) -> Self {
        candle_core::Error::Cuda(Box::new(err))
    }
}
