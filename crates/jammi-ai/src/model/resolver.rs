use std::path::{Path, PathBuf};
use std::sync::Arc;

use jammi_engine::catalog::Catalog;
use jammi_engine::error::{JammiError, Result};

use super::{BackendType, ModelId, ModelSource, ModelTask, ResolvedModel};

/// Resolves a `ModelSource` to file paths and backend selection.
pub struct ModelResolver {
    catalog: Arc<Catalog>,
    hf_api: hf_hub::api::sync::Api,
}

impl ModelResolver {
    /// Create a resolver backed by the given catalog and HuggingFace Hub API.
    pub fn new(catalog: Arc<Catalog>) -> Result<Self> {
        let hf_api = hf_hub::api::sync::Api::new()
            .map_err(|e| JammiError::Config(format!("HF Hub init failed: {e}")))?;
        Ok(Self { catalog, hf_api })
    }

    /// Access the catalog (for model registration after loading).
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Resolve a model source to file paths and backend selection.
    pub fn resolve(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ResolvedModel> {
        // Check catalog first — if this model was previously resolved and
        // registered, reuse the stored metadata instead of re-downloading.
        if let Some(resolved) = self.try_catalog_lookup(source, task, backend_hint)? {
            return Ok(resolved);
        }

        match source {
            ModelSource::Local(path) => self.resolve_local(path, source, task, backend_hint),
            ModelSource::HuggingFace(repo_id) => {
                self.resolve_hf_hub(repo_id, source, task, backend_hint)
            }
        }
    }

    /// Check the catalog for an existing model record matching this source.
    /// Returns `Some(ResolvedModel)` if found and files still exist on disk.
    fn try_catalog_lookup(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<Option<ResolvedModel>> {
        let model_id = ModelId::from(source);
        let record = match self.catalog.get_model(&model_id.0)? {
            Some(r) => r,
            None => return Ok(None),
        };

        // For fine-tuned models: resolve via the base model, set adapter_path.
        // The artifact_path for fine-tuned models points to the adapter directory,
        // not a full model directory — skip the config.json/weights checks.
        if record.model_type == "fine-tuned" {
            if let Some(ref base_id) = record.base_model_id {
                let base_source = ModelSource::parse(base_id);
                let base_resolved = self.resolve(&base_source, task, backend_hint)?;

                return Ok(Some(ResolvedModel {
                    model_id,
                    backend: base_resolved.backend,
                    task,
                    config_path: base_resolved.config_path,
                    weights_paths: base_resolved.weights_paths,
                    tokenizer_path: base_resolved.tokenizer_path,
                    model_config: base_resolved.model_config,
                    base_model_id: Some(ModelId(base_id.clone())),
                    adapter_path: record.artifact_path.map(PathBuf::from),
                    estimated_memory: base_resolved.estimated_memory,
                }));
            }
        }

        // Only use the catalog hit if artifact_path is set and still exists
        let artifact_dir = match &record.artifact_path {
            Some(p) => {
                let path = PathBuf::from(p);
                if path.exists() {
                    path
                } else {
                    return Ok(None);
                }
            }
            None => return Ok(None),
        };

        let config_path = artifact_dir.join("config.json");
        if !config_path.exists() {
            return Ok(None);
        }

        // Use stored config_json if available, otherwise re-read from disk
        let model_config: serde_json::Value = match &record.config_json {
            Some(json_str) => serde_json::from_str(json_str)?,
            None => serde_json::from_reader(std::fs::File::open(&config_path)?)?,
        };

        let backend = backend_hint.unwrap_or_else(|| {
            serde_json::from_str::<BackendType>(&format!("\"{}\"", record.backend))
                .unwrap_or(BackendType::Candle)
        });

        // Reconstruct weights paths from the artifact directory
        let weights_paths: Vec<PathBuf> = match backend {
            BackendType::Candle => {
                let p = artifact_dir.join("model.safetensors");
                if p.exists() {
                    vec![p]
                } else {
                    return Ok(None);
                }
            }
            BackendType::Ort => {
                let p = artifact_dir.join("model.onnx");
                if p.exists() {
                    vec![p]
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        let tokenizer_path = {
            let p = artifact_dir.join("tokenizer.json");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        };

        let estimated_memory: usize = weights_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .sum();

        Ok(Some(ResolvedModel {
            model_id,
            backend,
            task,
            config_path,
            weights_paths,
            tokenizer_path,
            model_config,
            base_model_id: record.base_model_id.map(ModelId),
            adapter_path: None,
            estimated_memory,
        }))
    }

    fn resolve_local(
        &self,
        path: &Path,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ResolvedModel> {
        if !path.exists() {
            return Err(JammiError::Model {
                model_id: source.to_string(),
                message: format!("Model directory does not exist: {}", path.display()),
            });
        }

        let config_path = path.join("config.json");
        if !config_path.exists() {
            return Err(JammiError::Model {
                model_id: source.to_string(),
                message: "Missing config.json in model directory".into(),
            });
        }
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        let has_safetensors = path.join("model.safetensors").exists();
        let has_onnx = path.join("model.onnx").exists();

        if !has_safetensors && !has_onnx {
            return Err(JammiError::Model {
                model_id: source.to_string(),
                message: "No model weights found (need model.safetensors or model.onnx)".into(),
            });
        }

        let backend = backend_hint.unwrap_or(if has_onnx {
            BackendType::Ort
        } else {
            BackendType::Candle
        });

        let weights_paths = match backend {
            BackendType::Candle => {
                let p = path.join("model.safetensors");
                if p.exists() {
                    vec![p]
                } else {
                    return Err(JammiError::Model {
                        model_id: source.to_string(),
                        message: "No safetensors weights found for Candle backend".into(),
                    });
                }
            }
            BackendType::Ort => {
                let p = path.join("model.onnx");
                if p.exists() {
                    vec![p]
                } else {
                    return Err(JammiError::Model {
                        model_id: source.to_string(),
                        message: "No ONNX weights found for ORT backend".into(),
                    });
                }
            }
            other => {
                return Err(JammiError::Model {
                    model_id: source.to_string(),
                    message: format!("Backend {other:?} not supported for local resolution"),
                })
            }
        };

        let tokenizer_path = {
            let p = path.join("tokenizer.json");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        };

        let estimated_memory: usize = weights_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .sum();

        Ok(ResolvedModel {
            model_id: ModelId::from(source),
            backend,
            task,
            config_path,
            weights_paths,
            tokenizer_path,
            model_config: config,
            base_model_id: None,
            adapter_path: None,
            estimated_memory,
        })
    }

    fn resolve_hf_hub(
        &self,
        repo_id: &str,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ResolvedModel> {
        let repo = self.hf_api.model(repo_id.to_string());

        let config_path = repo.get("config.json").map_err(|e| JammiError::Model {
            model_id: source.to_string(),
            message: format!("Failed to download config.json: {e}"),
        })?;
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        let backend = backend_hint.unwrap_or_else(|| self.select_backend_hf(&repo));

        let weights_paths = match backend {
            BackendType::Candle => self.download_safetensors(&repo, source)?,
            BackendType::Ort => self.download_onnx(&repo, source)?,
            other => {
                return Err(JammiError::Model {
                    model_id: source.to_string(),
                    message: format!("Backend {other:?} not supported in resolve"),
                })
            }
        };

        let tokenizer_path = repo.get("tokenizer.json").ok();

        let estimated_memory: usize = weights_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .sum();

        Ok(ResolvedModel {
            model_id: ModelId::from(source),
            backend,
            task,
            config_path,
            weights_paths,
            tokenizer_path,
            model_config: config,
            base_model_id: None,
            adapter_path: None,
            estimated_memory,
        })
    }

    fn select_backend_hf(&self, repo: &hf_hub::api::sync::ApiRepo) -> BackendType {
        if let Ok(info) = repo.info() {
            if info.siblings.iter().any(|s| s.rfilename == "model.onnx") {
                return BackendType::Ort;
            }
        }
        BackendType::Candle
    }

    fn download_safetensors(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        source: &ModelSource,
    ) -> Result<Vec<PathBuf>> {
        if let Ok(path) = repo.get("model.safetensors") {
            return Ok(vec![path]);
        }
        if let Ok(info) = repo.info() {
            let shards: Vec<PathBuf> = info
                .siblings
                .iter()
                .filter(|s| s.rfilename.ends_with(".safetensors"))
                .filter_map(|s| repo.get(&s.rfilename).ok())
                .collect();
            if !shards.is_empty() {
                return Ok(shards);
            }
        }
        Err(JammiError::Model {
            model_id: source.to_string(),
            message: "No safetensors weights found".into(),
        })
    }

    fn download_onnx(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        source: &ModelSource,
    ) -> Result<Vec<PathBuf>> {
        repo.get("model.onnx")
            .map(|p| vec![p])
            .map_err(|e| JammiError::Model {
                model_id: source.to_string(),
                message: format!("No ONNX model found: {e}"),
            })
    }
}
