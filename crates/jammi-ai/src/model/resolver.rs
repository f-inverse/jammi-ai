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
        match source {
            ModelSource::Local(path) => self.resolve_local(path, source, task, backend_hint),
            ModelSource::HuggingFace(repo_id) => {
                self.resolve_hf_hub(repo_id, source, task, backend_hint)
            }
        }
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
