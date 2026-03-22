use jammi_engine::error::{JammiError, Result};

use super::{DeviceConfig, ModelBackend};
use crate::model::{LoadedModel, ModelDimensions, ResolvedModel};

/// ORT backend — loads ONNX models via onnxruntime.
/// Full ORT integration lands in Phase 12a (External Sources & Backends).
pub struct OrtBackend;

/// An ORT-loaded model ready for inference.
/// Extended with `ort::Session` in Phase 12a when the ORT dependency is resolved.
pub struct OrtModel {
    /// Architecture dimensions for memory estimation and output sizing.
    pub dimensions: ModelDimensions,
}

impl ModelBackend for OrtBackend {
    fn load(&self, resolved: &ResolvedModel, _device: &DeviceConfig) -> Result<LoadedModel> {
        Err(JammiError::Model {
            model_id: resolved.model_id.0.clone(),
            message: "ORT backend is not available in this build. \
                      Use Candle backend (safetensors) or enable ORT in Phase 12a."
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
