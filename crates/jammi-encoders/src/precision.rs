//! The candle boundary for the workspace's candle-free `ComputePrecision`
//! vocabulary (`jammi_numerics::ComputePrecision`).
//!
//! Neither `ComputePrecision` nor `candle_core::DType` is a local type in any
//! workspace crate but `jammi-numerics` (candle-free by contract) and
//! `candle-core` itself, so an orphan-rule-legal `From<ComputePrecision> for
//! DType` cannot be implemented anywhere in this workspace. This crate — the
//! encoder implementation layer, already the candle-native home for every
//! BERT-family builder's `backbone_dtype(DType)` setter — is the natural
//! place for the plain conversion function every caller (the inference
//! backend, the fine-tune worker) uses instead of a trait impl.

use candle_core::DType;
use jammi_numerics::ComputePrecision;

/// Map the compute-precision vocabulary onto the candle dtype it selects.
pub fn compute_precision_to_dtype(precision: ComputePrecision) -> DType {
    match precision {
        ComputePrecision::F32 => DType::F32,
        ComputePrecision::F16 => DType::F16,
        ComputePrecision::BF16 => DType::BF16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_precision_to_its_dtype() {
        assert_eq!(
            compute_precision_to_dtype(ComputePrecision::F32),
            DType::F32
        );
        assert_eq!(
            compute_precision_to_dtype(ComputePrecision::F16),
            DType::F16
        );
        assert_eq!(
            compute_precision_to_dtype(ComputePrecision::BF16),
            DType::BF16
        );
    }
}
