use std::sync::Arc;

use jammi_db::error::{JammiError, Result};

use super::{DeviceConfig, ModelBackend};
use crate::model::{LoadedModel, ModelDescription, ResolvedModel};

/// ORT backend for ONNX models. This build carries no onnxruntime
/// dependency, so describing or materializing always refuses; only memory
/// estimation works.
pub struct OrtBackend;

fn unavailable(resolved: &ResolvedModel) -> JammiError {
    JammiError::Model {
        model_id: resolved.model_id.0.clone(),
        message: "ORT backend is not available in this build. \
                  Use the Candle backend (safetensors weights)."
            .into(),
    }
}

impl ModelBackend for OrtBackend {
    fn describe(
        &self,
        resolved: &ResolvedModel,
        _device: &DeviceConfig,
    ) -> Result<ModelDescription> {
        Err(unavailable(resolved))
    }

    fn materialize(
        &self,
        resolved: &ResolvedModel,
        _description: Arc<ModelDescription>,
        _device: &DeviceConfig,
    ) -> Result<LoadedModel> {
        Err(unavailable(resolved))
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
