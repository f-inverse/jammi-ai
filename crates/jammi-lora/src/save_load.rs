//! On-disk persistence helpers for a LoRA adapter directory.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::LoraError;

/// Adapter directory file layout (constants kept private — the directory shape
/// is an internal invariant of this module).
const ADAPTER_WEIGHTS_FILE: &str = "adapter.safetensors";
const ADAPTER_CONFIG_FILE: &str = "adapter_config.json";

/// Write a LoRA adapter to `dir`:
///
/// - `adapter.safetensors` — the supplied `tensors` map.
/// - `adapter_config.json` — `config` serialised pretty as JSON.
///
/// Creates `dir` (and parents) if it does not exist. `config` can be any
/// [`Serialize`]-able type — typically [`AdapterConfig`](crate::AdapterConfig)
/// when the adapter wraps encoder-internal linears, but callers with
/// different adapter shapes can pass their own metadata struct or an
/// enum that discriminates between shapes.
pub fn save_adapter<C: Serialize>(
    dir: &Path,
    tensors: &HashMap<String, Tensor>,
    config: &C,
) -> Result<(), LoraError> {
    std::fs::create_dir_all(dir)?;
    candle_core::safetensors::save(tensors, dir.join(ADAPTER_WEIGHTS_FILE))?;
    let cfg_json = serde_json::to_string_pretty(config)?;
    std::fs::write(dir.join(ADAPTER_CONFIG_FILE), cfg_json)?;
    Ok(())
}

/// Read a LoRA adapter directory: parses `adapter_config.json` into `C` and
/// loads `adapter.safetensors` onto `device`.
pub fn load_adapter<C: DeserializeOwned>(
    dir: &Path,
    device: &Device,
) -> Result<(C, HashMap<String, Tensor>), LoraError> {
    let cfg_bytes = std::fs::read(dir.join(ADAPTER_CONFIG_FILE))?;
    let config: C = serde_json::from_slice(&cfg_bytes)?;
    let tensors = candle_core::safetensors::load(dir.join(ADAPTER_WEIGHTS_FILE), device)?;
    Ok((config, tensors))
}
