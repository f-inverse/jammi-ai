//! The compute-precision vocabulary shared by every trainable/inference
//! backbone in the workspace: a candle-free enum naming the floating-point
//! dtype a model's weights and activations run at.
//!
//! This is the single Rust type for the concept — the fine-tune frozen-backbone
//! dtype and the inference compute precision are the same knob (both answer
//! "what dtype does this backbone's matmuls run in"), so they share one
//! vocabulary rather than each crate defining its own enum. It lives here
//! (not in `jammi-lora` or `jammi-encoders`, both of which optionally carry
//! the `candle` tensor stack) so a crate that only needs the config vocabulary
//! — `jammi-db`, `jammi-wire` — never pulls a tensor library to name a dtype.
//! The `ComputePrecision -> candle_core::DType` conversion is a plain function
//! at the candle boundary (`jammi_encoders::compute_precision_to_dtype`),
//! never a `From` impl here — neither type is local to this crate, so an
//! orphan-rule-legal `From` impl cannot live in `jammi-numerics` at all.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::NumericsError;

/// The floating-point precision a backbone's weights and activations run at.
///
/// `F32` is the default: maximally compatible, and the byte-parity oracle's
/// baseline. `F16` halves memory and roughly doubles throughput on hardware
/// that supports it. `BF16` carries `F32`'s 8-bit exponent range in half the
/// bytes; it is a GPU-tier precision — the candle load boundary admits it on a
/// CUDA device of compute capability >= 8.0 (Ampere+) and fails loud below,
/// keyed off [`ComputePrecision::min_cuda_capability`]. Which precisions a
/// given device admits is a runtime property, decided at the candle boundary;
/// this vocabulary only names them and states each one's CUDA floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputePrecision {
    /// Full precision — maximally compatible.
    #[default]
    F32,
    /// Half-precision float — compatible with most CUDA devices; ~2x
    /// throughput/memory over `F32`.
    F16,
    /// BFloat16 — `F32`'s 8-bit exponent range in half the bytes. A GPU-tier
    /// precision: its bf16 tensor-core kernels are an Ampere innovation, so the
    /// candle load boundary admits it only on a CUDA device of compute
    /// capability >= 8.0 (sm_80+) and fails loud below. Fine-tune's frozen
    /// backbone may also persist this as its training-time dtype.
    BF16,
}

impl ComputePrecision {
    /// Minimum CUDA compute capability `(major, minor)` this precision needs to
    /// run natively, or `None` if it has no CUDA floor. `BF16` requires Ampere
    /// (8.0)+ — its tensor-core/bf16 kernels do not exist below sm_80. `F16` and
    /// `F32` carry no floor here (`F16` already runs on the CUDA devices Jammi
    /// targets).
    ///
    /// This is the pure, candle-free decision the candle-boundary loader keys
    /// its runtime gate off of; the loader acquires the device's actual
    /// capability and builds the typed error, but the allow/reject rule lives
    /// here so it is unit-testable without a GPU.
    pub fn min_cuda_capability(self) -> Option<(i32, i32)> {
        match self {
            Self::BF16 => Some((8, 0)),
            Self::F16 | Self::F32 => None,
        }
    }

    /// Whether a CUDA device of compute capability `(major, minor)` runs this
    /// precision natively. Lexicographic tuple order is the capability order
    /// (major dominates, minor breaks ties).
    pub fn is_supported_on_cuda(self, major: i32, minor: i32) -> bool {
        match self.min_cuda_capability() {
            Some(min) => (major, minor) >= min,
            None => true,
        }
    }
}

impl fmt::Display for ComputePrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
        }
    }
}

impl FromStr for ComputePrecision {
    type Err = NumericsError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            other => Err(NumericsError::InvalidInput(format!(
                "unknown compute precision '{other}'. Expected: f32, f16, bf16"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_f32() {
        assert_eq!(ComputePrecision::default(), ComputePrecision::F32);
    }

    #[test]
    fn display_and_from_str_round_trip() {
        for precision in [
            ComputePrecision::F32,
            ComputePrecision::F16,
            ComputePrecision::BF16,
        ] {
            let s = precision.to_string();
            assert_eq!(s.parse::<ComputePrecision>().unwrap(), precision);
        }
    }

    #[test]
    fn from_str_rejects_unknown_value() {
        assert!("fp8".parse::<ComputePrecision>().is_err());
    }

    #[test]
    fn bf16_requires_ampere_sm80() {
        // Below sm_80 the bf16 tensor-core kernels do not exist: reject
        // Turing (T4, sm_75) and Volta (V100, sm_70).
        assert!(!ComputePrecision::BF16.is_supported_on_cuda(7, 5));
        assert!(!ComputePrecision::BF16.is_supported_on_cuda(7, 0));
        // sm_80 and above: admit A100 (sm_80), A10/A10G (sm_86), H100 (sm_90).
        assert!(ComputePrecision::BF16.is_supported_on_cuda(8, 0));
        assert!(ComputePrecision::BF16.is_supported_on_cuda(8, 6));
        assert!(ComputePrecision::BF16.is_supported_on_cuda(9, 0));
        assert_eq!(ComputePrecision::BF16.min_cuda_capability(), Some((8, 0)));
    }

    #[test]
    fn f16_f32_have_no_cuda_floor() {
        for precision in [ComputePrecision::F16, ComputePrecision::F32] {
            assert_eq!(precision.min_cuda_capability(), None);
            assert!(precision.is_supported_on_cuda(5, 0)); // Maxwell
            assert!(precision.is_supported_on_cuda(7, 5)); // Turing
            assert!(precision.is_supported_on_cuda(9, 0)); // Hopper
        }
    }

    #[test]
    fn serde_uses_lowercase_tokens() {
        assert_eq!(
            serde_json::to_value(ComputePrecision::F16).unwrap(),
            serde_json::json!("f16")
        );
        assert_eq!(
            serde_json::from_value::<ComputePrecision>(serde_json::json!("bf16")).unwrap(),
            ComputePrecision::BF16
        );
    }
}
