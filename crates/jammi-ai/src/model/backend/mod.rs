pub mod candle;
/// GGUF/k-quant loading and residency-estimation helpers shared by
/// [`super::resolver`] and [`candle`] — see the module's own doc.
pub(crate) mod gguf;
pub mod http;
pub mod open_clip_text;
pub mod ort;
/// Safetensors header-parsed residency-estimation helper shared by
/// [`super::resolver`] and [`candle`] — see the module's own doc.
pub(crate) mod safetensors_residency;

use jammi_db::error::Result;

use super::{LoadedModel, ResolvedModel};

/// Abstraction over model inference backends.
pub trait ModelBackend: Send + Sync {
    /// Load a resolved model into memory on the target device.
    fn load(&self, resolved: &ResolvedModel, device: &DeviceConfig) -> Result<LoadedModel>;

    /// Estimated GPU memory in bytes for a loaded model.
    fn estimate_memory(&self, resolved: &ResolvedModel) -> usize;
}

/// Device configuration derived from JammiConfig.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// The PRIMARY device ordinal (-1 for CPU-only): the first entry of
    /// [`Self::devices`], and the one a caller that names no device gets.
    pub gpu_device: i32,
    /// Every device this deployment placed work on, in order — the resolved
    /// `[gpu] devices` list, which is `[device]` when the deployment names
    /// only the singular. A process opens per-device resources (a GPU
    /// scheduler, a model-cache slot) for each of them, so the list may be
    /// wider than any one gang.
    ///
    /// Always non-empty, and `devices[0] == gpu_device`: both are guaranteed
    /// by `jammi_db::config::GpuConfig::validate`, which refuses an empty
    /// list and a list that does not lead with the primary. A
    /// hand-constructed value that violates either states a deployment no
    /// loaded configuration can produce.
    pub devices: Vec<i32>,
    /// Fraction of GPU memory available for model loading.
    pub memory_fraction: f64,
    /// When `true`, refuse to fall back to CPU if the requested GPU is
    /// unavailable: device selection returns an error so the server fails
    /// fast instead of silently serving on CPU. When `false` (the default),
    /// an unavailable GPU degrades to CPU with a loud warning.
    pub require_gpu: bool,
    /// Global default inference compute precision (`GpuConfig::compute_precision`).
    /// A per-model override in the resolved model's `config.json` wins over
    /// this at load time; both default to `F32`.
    pub compute_precision: jammi_numerics::ComputePrecision,
}

impl DeviceConfig {
    /// This configuration restricted to one of its devices: the same knobs
    /// with `device` as the primary.
    ///
    /// What a per-device consumer (a backend load for one rank, a per-device
    /// cache entry) is handed, so nothing has to carry "the config, plus
    /// separately, which device". A `device` that is not in
    /// [`Self::devices`] is refused: placing work on a device the deployment
    /// never declared is how a CPU-pinned session ends up on a GPU.
    pub fn for_device(&self, device: i32) -> Result<Self> {
        if !self.devices.contains(&device) {
            return Err(jammi_db::error::JammiError::Config(format!(
                "device {device} is not one of the configured [gpu] devices {:?}",
                self.devices
            )));
        }
        Ok(Self {
            gpu_device: device,
            ..self.clone()
        })
    }

    /// Derive device configuration from the application config.
    pub fn from_config(config: &jammi_db::config::JammiConfig) -> Self {
        Self {
            gpu_device: config.gpu.device,
            // The ONE reconciliation of the two arities lives on
            // `GpuConfig`: an absent plural and a one-entry plural are the
            // same deployment, and this reads the resolved answer rather
            // than re-deriving it.
            devices: config.gpu.device_list(),
            memory_fraction: config.gpu.memory_fraction,
            require_gpu: config.gpu.require_gpu,
            compute_precision: config.gpu.compute_precision,
        }
    }
}
