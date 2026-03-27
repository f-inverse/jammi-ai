pub mod candle;
pub mod http;
pub mod open_clip_vit;
pub mod ort;

use jammi_engine::error::Result;

use super::{LoadedModel, ResolvedModel};

/// Abstraction over model inference backends.
pub trait ModelBackend: Send + Sync {
    /// Load a resolved model into memory on the target device.
    fn load(&self, resolved: &ResolvedModel, device: &DeviceConfig) -> Result<LoadedModel>;

    /// Estimated GPU memory in bytes for a loaded model.
    fn estimate_memory(&self, resolved: &ResolvedModel) -> usize;
}

/// Device configuration derived from JammiConfig.
pub struct DeviceConfig {
    /// GPU device ordinal (-1 for CPU-only).
    pub gpu_device: i32,
    /// Fraction of GPU memory available for model loading.
    pub memory_fraction: f64,
}

impl DeviceConfig {
    /// Derive device configuration from the application config.
    pub fn from_config(config: &jammi_engine::config::JammiConfig) -> Self {
        Self {
            gpu_device: config.gpu.device,
            memory_fraction: config.gpu.memory_fraction,
        }
    }
}
