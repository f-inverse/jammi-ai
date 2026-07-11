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
/// that supports it. `BF16` is a valid persisted/config value (fine-tune's
/// frozen backbone may have trained at it) but inference callers reject it at
/// the load boundary until a runtime compute-capability gate exists — that
/// refusal is the candle-boundary caller's concern, not this vocabulary's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputePrecision {
    /// Full precision — maximally compatible.
    #[default]
    F32,
    /// Half-precision float — compatible with most CUDA devices; ~2x
    /// throughput/memory over `F32`.
    F16,
    /// BFloat16 — recommended for CUDA/Metal training; cuts backbone VRAM by
    /// ~half. Fine-tune's frozen backbone may persist this; inference does
    /// not yet serve it.
    BF16,
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
