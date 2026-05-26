//! Cholesky factorisation with progressive diagonal jitter retry.
//!
//! GP covariance matrices are PSD but numerically near-singular when
//! training points cluster — direct Cholesky can fail. The standard
//! remedy is to add a small multiple of the identity to the diagonal
//! ("jitter") before factorising. We try a ladder of jitter values
//! scaled by the matrix's mean diagonal entry so the perturbation is
//! always relative to the matrix scale, not an absolute value that is
//! either negligible (large kernels) or destructive (small kernels).

use nalgebra::{Cholesky, DMatrix, Dyn};

use crate::error::{NumericsError, Result};

/// Multipliers tried in order; each is scaled by `max(mean_diag, 1.0)`
/// before being added to the diagonal.
const JITTER_LADDER: &[f64] = &[1e-10, 1e-9, 1e-8, 1e-7, 1e-6];

/// Cholesky-factorise `k`, falling back to jittered factorisation when
/// the direct call returns `None`. Returns `NumericsError::IllConditioned`
/// once the full ladder is exhausted.
pub fn cholesky_with_jitter(k: DMatrix<f64>) -> Result<Cholesky<f64, Dyn>> {
    if let Some(c) = k.clone().cholesky() {
        return Ok(c);
    }
    let n = k.nrows();
    let mean_diag: f64 = (0..n).map(|i| k[(i, i)]).sum::<f64>() / n as f64;
    let scale = mean_diag.abs().max(1.0);

    for &j in JITTER_LADDER {
        let mut perturbed = k.clone();
        let jitter = j * scale;
        for i in 0..n {
            perturbed[(i, i)] += jitter;
        }
        if let Some(c) = perturbed.cholesky() {
            return Ok(c);
        }
    }
    Err(NumericsError::IllConditioned {
        attempts: JITTER_LADDER.len(),
        final_jitter: JITTER_LADDER[JITTER_LADDER.len() - 1] * scale,
    })
}
