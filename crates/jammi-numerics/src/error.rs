use thiserror::Error;

/// Errors surfaced by the numerical kernels.
#[derive(Debug, Error)]
pub enum NumericsError {
    /// Cholesky factorisation failed after the full jitter ladder. The
    /// matrix is too ill-conditioned for numerically stable inversion at
    /// the current scale.
    #[error(
        "matrix is ill-conditioned after {attempts} jitter retries (final jitter = {final_jitter:e})"
    )]
    IllConditioned { attempts: usize, final_jitter: f64 },

    /// Caller-supplied input violated a precondition (empty slice where
    /// a non-empty one is required, invalid range, NaN where finite,
    /// etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Two slices that must agree on length disagreed.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Crate-local `Result` alias.
pub type Result<T> = std::result::Result<T, NumericsError>;
