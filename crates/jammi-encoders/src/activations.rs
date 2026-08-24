//! Activation functions shared by more than one tower's MLP.

use candle_core::Tensor;

use crate::error::EncoderError;

/// QuickGelu activation: `x * sigmoid(1.702 * x)`. OpenCLIP uses this in
/// both the text ([`crate::clip_text`]) and vision
/// ([`crate::open_clip_vision`]) tower MLPs (not the standard erf-based
/// GELU) — was duplicated verbatim between the two files; this is the one
/// copy both now call.
pub(crate) fn quick_gelu(xs: &Tensor) -> Result<Tensor, EncoderError> {
    Ok((xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?)?)
}
