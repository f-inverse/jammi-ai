use std::path::PathBuf;
use std::sync::Arc;

use jammi_engine::catalog::Catalog;
use jammi_engine::error::{JammiError, Result};

use super::{BackendType, ModelId, ModelTask, ResolvedModel};

/// Resolves a model ID to file paths and backend selection.
/// Resolution chain: catalog → fine-tuned → local path → HTTP → HF Hub.
pub struct ModelResolver {
    catalog: Arc<Catalog>,
    hf_api: hf_hub::api::sync::Api,
}

impl ModelResolver {
    pub fn new(catalog: Arc<Catalog>) -> Result<Self> {
        let hf_api = hf_hub::api::sync::Api::new()
            .map_err(|e| JammiError::Config(format!("HF Hub init failed: {e}")))?;
        Ok(Self { catalog, hf_api })
    }

    pub fn resolve(
        &self,
        model_id: &str,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ResolvedModel> {
        // 1. Catalog lookup
        if let Some(_record) = self.catalog.get_model(model_id)? {
            return Err(JammiError::Model {
                model_id: model_id.into(),
                message: "Catalog model loading not yet implemented".into(),
            });
        }

        // 2. Jammi fine-tuned model
        if model_id.starts_with("jammi:fine-tuned:") {
            return Err(JammiError::Model {
                model_id: model_id.into(),
                message: "Fine-tuned model loading not yet implemented".into(),
            });
        }

        // 3. Local path
        if model_id.starts_with('/') || model_id.starts_with("./") {
            return self.resolve_local(model_id, task, backend_hint);
        }

        // 4. HTTP endpoint
        if model_id.starts_with("http://") || model_id.starts_with("https://") {
            return Err(JammiError::Model {
                model_id: model_id.into(),
                message: "HTTP endpoint loading not yet implemented".into(),
            });
        }

        // 5. HuggingFace Hub
        self.resolve_hf_hub(model_id, task, backend_hint)
    }

    fn resolve_local(
        &self,
        path: &str,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ResolvedModel> {
        let model_dir = PathBuf::from(path);
        if !model_dir.exists() {
            return Err(JammiError::Model {
                model_id: path.into(),
                message: format!("Model directory does not exist: {path}"),
            });
        }

        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Err(JammiError::Model {
                model_id: path.into(),
                message: "Missing config.json in model directory".into(),
            });
        }
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        let has_safetensors = model_dir.join("model.safetensors").exists();
        let has_onnx = model_dir.join("model.onnx").exists();

        if !has_safetensors && !has_onnx {
            return Err(JammiError::Model {
                model_id: path.into(),
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
                let p = model_dir.join("model.safetensors");
                if p.exists() {
                    vec![p]
                } else {
                    return Err(JammiError::Model {
                        model_id: path.into(),
                        message: "No safetensors weights found for Candle backend".into(),
                    });
                }
            }
            BackendType::Ort => {
                let p = model_dir.join("model.onnx");
                if p.exists() {
                    vec![p]
                } else {
                    return Err(JammiError::Model {
                        model_id: path.into(),
                        message: "No ONNX weights found for ORT backend".into(),
                    });
                }
            }
            other => {
                return Err(JammiError::Model {
                    model_id: path.into(),
                    message: format!("Backend {other:?} not supported for local resolution"),
                })
            }
        };

        let tokenizer_path = {
            let p = model_dir.join("tokenizer.json");
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
            model_id: ModelId(path.to_string()),
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
        model_id: &str,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ResolvedModel> {
        let repo = self.hf_api.model(model_id.to_string());

        let config_path = repo.get("config.json").map_err(|e| JammiError::Model {
            model_id: model_id.into(),
            message: format!("Failed to download config.json: {e}"),
        })?;
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        let backend = backend_hint.unwrap_or_else(|| self.select_backend_hf(&repo));

        let weights_paths = match backend {
            BackendType::Candle => self.download_safetensors(&repo, model_id)?,
            BackendType::Ort => self.download_onnx(&repo, model_id)?,
            other => {
                return Err(JammiError::Model {
                    model_id: model_id.into(),
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
            model_id: ModelId(model_id.to_string()),
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
            if info.siblings.iter().any(|s| s.rfilename.ends_with(".onnx")) {
                return BackendType::Ort;
            }
        }
        BackendType::Candle
    }

    fn download_safetensors(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        model_id: &str,
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
            model_id: model_id.into(),
            message: "No safetensors weights found".into(),
        })
    }

    fn download_onnx(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        model_id: &str,
    ) -> Result<Vec<PathBuf>> {
        repo.get("model.onnx")
            .map(|p| vec![p])
            .map_err(|e| JammiError::Model {
                model_id: model_id.into(),
                message: format!("No ONNX model found: {e}"),
            })
    }
}
