//! On-disk persistence helpers for a LoRA adapter directory.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::LoraError;

/// The adapter's weights file inside an adapter directory — the safetensors
/// map [`save_adapter`] writes and [`load_adapter`] reads. The one place the
/// name is spelled: a caller that packages, digests or ships an adapter
/// directory names its files through these constants.
pub const ADAPTER_WEIGHTS_FILE: &str = "adapter.safetensors";
/// The adapter's metadata file inside an adapter directory — the JSON
/// [`save_adapter`] writes beside [`ADAPTER_WEIGHTS_FILE`].
pub const ADAPTER_CONFIG_FILE: &str = "adapter_config.json";

/// Write a LoRA adapter to `dir`:
///
/// - [`ADAPTER_WEIGHTS_FILE`] — the supplied `tensors` map.
/// - [`ADAPTER_CONFIG_FILE`] — `config` serialised pretty as JSON.
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

/// Read a LoRA adapter directory: parses [`ADAPTER_CONFIG_FILE`] into `C`
/// and loads [`ADAPTER_WEIGHTS_FILE`] onto `device`.
pub fn load_adapter<C: DeserializeOwned>(
    dir: &Path,
    device: &Device,
) -> Result<(C, HashMap<String, Tensor>), LoraError> {
    let cfg_bytes = std::fs::read(dir.join(ADAPTER_CONFIG_FILE))?;
    let config: C = serde_json::from_slice(&cfg_bytes)?;
    let tensors = candle_core::safetensors::load(dir.join(ADAPTER_WEIGHTS_FILE), device)?;
    Ok((config, tensors))
}
