use jammi_engine::error::Result;

use super::{DeviceConfig, ModelBackend};
use crate::model::{LoadedModel, ModelDimensions, ResolvedModel};

/// Candle backend — loads safetensors models via candle.
pub struct CandleBackend;

/// A candle-loaded model ready for inference.
pub struct CandleModel {
    pub dimensions: ModelDimensions,
}

impl ModelBackend for CandleBackend {
    fn load(&self, _resolved: &ResolvedModel, _device: &DeviceConfig) -> Result<LoadedModel> {
        // Full implementation in Phase 04 (InferenceExec) with live model loading.
        todo!("CandleBackend::load — requires live model weights")
    }

    fn estimate_memory(&self, resolved: &ResolvedModel) -> usize {
        resolved
            .weights_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .sum()
    }
}
