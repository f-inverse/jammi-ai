use jammi_db::error::{JammiError, Result};

use super::{DeviceConfig, ModelBackend};
use crate::model::{LoadedModel, ModelDimensions, ResolvedModel};

/// ORT backend for ONNX models. This build carries no onnxruntime
/// dependency, so `load` always refuses; only memory estimation works.
pub struct OrtBackend;

/// An ORT-loaded model ready for inference.
pub struct OrtModel {
    /// Architecture dimensions for memory estimation and output sizing.
    pub dimensions: ModelDimensions,
}

impl ModelBackend for OrtBackend {
    fn load(&self, resolved: &ResolvedModel, _device: &DeviceConfig) -> Result<LoadedModel> {
        Err(JammiError::Model {
            model_id: resolved.model_id.0.clone(),
            message: "ORT backend is not available in this build. \
                      Use the Candle backend (safetensors weights)."
                .into(),
        })
    }

    fn estimate_memory(&self, resolved: &ResolvedModel) -> usize {
        let file_size: usize = resolved
            .weights_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .sum();
        (file_size as f64 * 1.3) as usize
    }
}
