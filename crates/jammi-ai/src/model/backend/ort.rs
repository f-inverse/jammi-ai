use jammi_engine::error::Result;

use super::{DeviceConfig, ModelBackend};
use crate::model::{LoadedModel, ModelDimensions, ResolvedModel};

/// ORT backend — loads ONNX models via onnxruntime.
pub struct OrtBackend;

/// An ORT-loaded model ready for inference.
pub struct OrtModel {
    pub dimensions: ModelDimensions,
}

impl ModelBackend for OrtBackend {
    fn load(&self, _resolved: &ResolvedModel, _device: &DeviceConfig) -> Result<LoadedModel> {
        // Full implementation in Phase 04 (InferenceExec) with live model loading.
        todo!("OrtBackend::load — requires live ONNX model")
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
