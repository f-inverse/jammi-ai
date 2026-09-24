pub mod candle;
/// GGUF/k-quant loading and residency-estimation helpers shared by
/// [`super::resolver`] and [`candle`] — see the module's own doc.
pub(crate) mod gguf;
pub mod open_clip_text;
pub mod remote;
/// The rows of one forward, shared by every backend — see the module's own
/// doc.
pub(crate) mod rows;
/// Safetensors header-parsed residency-estimation helper shared by
/// [`super::resolver`] and [`candle`] — see the module's own doc.
pub(crate) mod safetensors_residency;

use std::sync::Arc;

use jammi_db::error::Result;

use super::{LoadedModel, ModelDescription, ResolvedModel};

/// Abstraction over model inference backends: a resolved model is
/// described from its files, then materialized on a device.
pub trait ModelBackend: Send + Sync {
    /// Describe a resolved model for `device` from its files and
    /// configuration — everything planning its run needs
    /// ([`ModelDescription`]) — without allocating a tensor.
    fn describe(&self, resolved: &ResolvedModel, device: &DeviceConfig)
        -> Result<ModelDescription>;

    /// Materialize the weights of a described model on the device it was
    /// described for. The loaded model reports `description` unchanged.
    fn materialize(
        &self,
        resolved: &ResolvedModel,
        description: Arc<ModelDescription>,
        device: &DeviceConfig,
    ) -> Result<LoadedModel>;

    /// Estimated GPU memory in bytes for a loaded model.
    fn estimate_memory(&self, resolved: &ResolvedModel) -> usize;

    /// Describe, then materialize: the composition a caller with no
    /// description of its own takes. [`super::cache::ModelCache`] never
    /// calls this — it memoizes the description so the content digest is
    /// hashed once per resolved directory, and a submitter that only plans
    /// reads the description alone.
    fn load(&self, resolved: &ResolvedModel, device: &DeviceConfig) -> Result<LoadedModel> {
        let description = Arc::new(self.describe(resolved, device)?);
        self.materialize(resolved, description, device)
    }
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
    /// over `device` alone.
    ///
    /// What a per-device consumer (a backend load for one rank, a per-device
    /// cache entry) is handed, so nothing has to carry "the config, plus
    /// separately, which device". A `device` that is not in
    /// [`Self::devices`] is refused: placing work on a device the deployment
    /// never declared is how a CPU-pinned session ends up on a GPU.
    ///
    /// The result names `device` and nothing else, which is what keeps
    /// [`Self::devices`]'s documented invariant (`devices[0] == gpu_device`)
    /// true of every value of this type: carrying the deployment's whole
    /// list forward would leave the primary naming one card while the list
    /// led with another, and a reader that resolved a device from the pair
    /// would get two different answers. A consumer that needs the
    /// deployment's full list holds the session's own unrestricted
    /// configuration.
    pub fn for_device(&self, device: i32) -> Result<Self> {
        if !self.devices.contains(&device) {
            return Err(jammi_db::error::JammiError::Config(format!(
                "device {device} is not one of the configured [gpu] devices {:?}",
                self.devices
            )));
        }
        Ok(Self {
            gpu_device: device,
            devices: vec![device],
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

#[cfg(test)]
mod device_config_tests {
    use super::*;

    /// A restriction to one device SATISFIES the invariant the type's own
    /// documentation states — `devices` non-empty with `devices[0] ==
    /// gpu_device` — rather than producing a value the doc says cannot
    /// exist.
    ///
    /// The restricted value is what a backend load for one rank is handed;
    /// a reader that resolved a device from it (a second
    /// [`DeviceConfig::for_device`], a per-device budget lookup) would
    /// otherwise be told the primary is one card while the list leads with
    /// another.
    #[test]
    fn restricting_to_one_device_leaves_the_primary_leading_the_list() {
        let config = DeviceConfig {
            gpu_device: 0,
            devices: vec![0, 1, 2],
            memory_fraction: 0.9,
            require_gpu: true,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        };

        for device in [0, 1, 2] {
            let restricted = config.for_device(device).expect("a declared device");
            assert_eq!(restricted.gpu_device, device);
            assert_eq!(
                restricted.devices,
                vec![device],
                "the restriction names exactly the one device it is a restriction to"
            );
            assert_eq!(
                restricted.devices.first().copied(),
                Some(restricted.gpu_device),
                "`devices[0] == gpu_device` is the invariant this type documents"
            );
            // The knobs that are not about WHICH device travel unchanged.
            assert_eq!(restricted.memory_fraction, config.memory_fraction);
            assert_eq!(restricted.require_gpu, config.require_gpu);
            assert_eq!(restricted.compute_precision, config.compute_precision);
        }

        // A restriction is a restriction: the devices it dropped are no
        // longer reachable from it.
        let restricted = config.for_device(1).expect("a declared device");
        assert!(restricted.for_device(1).is_ok());
        let error = restricted
            .for_device(2)
            .expect_err("device 2 is not a device of a configuration restricted to device 1");
        assert!(
            error.to_string().contains("not one of the configured"),
            "unexpected message: {error}"
        );
    }

    /// The refusal is on the FULL list, so a device the deployment never
    /// declared is refused before anything is restricted to it.
    #[test]
    fn a_device_outside_the_configured_list_is_refused() {
        let config = DeviceConfig {
            gpu_device: 0,
            devices: vec![0, 1],
            memory_fraction: 0.9,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        };
        let error = config
            .for_device(7)
            .expect_err("device 7 was never declared by this deployment");
        assert!(
            error.to_string().contains("[gpu] devices [0, 1]"),
            "the refusal names the list it checked: {error}"
        );
    }
}
