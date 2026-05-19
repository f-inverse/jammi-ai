//! On-disk persistence helpers for a LoRA adapter directory.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::adapter::AdapterConfig;
use crate::error::LoraError;

/// Adapter directory file layout (constants kept private — the directory shape
/// is an internal invariant of this module).
const ADAPTER_WEIGHTS_FILE: &str = "adapter.safetensors";
const ADAPTER_CONFIG_FILE: &str = "adapter_config.json";

/// Write a LoRA adapter to `dir`:
///
/// - `adapter.safetensors` — the supplied `tensors` map.
/// - `adapter_config.json` — `config` serialised pretty.
///
/// Creates `dir` (and parents) if it does not exist.
pub fn save_adapter(
    dir: &Path,
    tensors: &HashMap<String, Tensor>,
    config: &AdapterConfig,
) -> Result<(), LoraError> {
    std::fs::create_dir_all(dir)?;
    candle_core::safetensors::save(tensors, dir.join(ADAPTER_WEIGHTS_FILE))?;
    let cfg_json = serde_json::to_string_pretty(config)?;
    std::fs::write(dir.join(ADAPTER_CONFIG_FILE), cfg_json)?;
    Ok(())
}

/// Read a LoRA adapter directory: parses `adapter_config.json` and loads
/// `adapter.safetensors` onto `device`.
pub fn load_adapter(
    dir: &Path,
    device: &Device,
) -> Result<(AdapterConfig, HashMap<String, Tensor>), LoraError> {
    let cfg_bytes = std::fs::read(dir.join(ADAPTER_CONFIG_FILE))?;
    let config: AdapterConfig = serde_json::from_slice(&cfg_bytes)?;
    let tensors = candle_core::safetensors::load(dir.join(ADAPTER_WEIGHTS_FILE), device)?;
    Ok((config, tensors))
}
