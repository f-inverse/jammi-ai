use std::path::{Path, PathBuf};
use std::sync::Arc;

use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::ArtifactStore;

use super::{BackendType, ModelId, ModelSource, ModelTask, ResolvedModel, TokenizerSource};

/// Resolves a `ModelSource` to file paths and backend selection.
pub struct ModelResolver {
    catalog: Arc<Catalog>,
    /// Reloads a fine-tuned model's adapter: its catalog `artifact_path` is the
    /// object-store prefix the training worker wrote, fetched into a local cache
    /// dir candle loads from — so a cross-host worker fleet shares adapters.
    artifact_store: Arc<ArtifactStore>,
    hf_api: hf_hub::api::sync::Api,
}

impl ModelResolver {
    /// Create a resolver backed by the given catalog, artifact store, and
    /// HuggingFace Hub API.
    pub fn new(catalog: Arc<Catalog>, artifact_store: Arc<ArtifactStore>) -> Result<Self> {
        let hf_api = hf_hub::api::sync::Api::new()
            .map_err(|e| JammiError::Config(format!("HF Hub init failed: {e}")))?;
        Ok(Self {
            catalog,
            artifact_store,
            hf_api,
        })
    }

    /// Access the catalog (for model registration after loading).
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Resolve a model source to file paths and backend selection.
    pub async fn resolve(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ResolvedModel> {
        // Check catalog first — if this model was previously resolved and
        // registered, reuse the stored metadata instead of re-downloading.
        if let Some(resolved) =
            Box::pin(self.try_catalog_lookup(source, task, backend_hint)).await?
        {
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
    async fn try_catalog_lookup(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<Option<ResolvedModel>> {
        let model_id = ModelId::from(source);
        let record = match self.catalog.get_model(&model_id.0).await? {
            Some(r) => r,
            None => return Ok(None),
        };

        // For fine-tuned models: resolve via the base model, set adapter_path.
        // The artifact_path for a fine-tuned model is the object-store prefix the
        // training worker wrote the adapter under — fetch it into a local cache
        // dir candle can mmap (an in-place no-op for a `file://` root), and point
        // `adapter_path` at that dir. The base model resolves through its own
        // path, so this only routes the *adapter* through the artifact store.
        if record.model_type == "fine-tuned" {
            if let Some(ref base_id) = record.base_model_id {
                let base_source = ModelSource::parse(base_id);
                let base_resolved =
                    Box::pin(self.resolve(&base_source, task, backend_hint)).await?;

                let adapter_path = match &record.artifact_path {
                    Some(prefix) => {
                        let prefix_url = jammi_db::storage::StorageUrl::parse(prefix)?;
                        Some(
                            self.artifact_store
                                .fetch_artifact(&prefix_url)
                                .await?
                                .dir()
                                .to_path_buf(),
                        )
                    }
                    None => None,
                };

                return Ok(Some(ResolvedModel {
                    model_id,
                    backend: base_resolved.backend,
                    task,
                    config_path: base_resolved.config_path,
                    weights_paths: base_resolved.weights_paths,
                    tokenizer: base_resolved.tokenizer,
                    model_config: base_resolved.model_config,
                    preprocessor_config: base_resolved.preprocessor_config,
                    base_model_id: Some(ModelId(base_id.clone())),
                    adapter_path,
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

        // Try standard config.json first, then OpenCLIP open_clip_config.json
        let config_path = {
            let standard = artifact_dir.join("config.json");
            let open_clip = artifact_dir.join("open_clip_config.json");
            if standard.exists() {
                standard
            } else if open_clip.exists() {
                open_clip
            } else {
                return Ok(None);
            }
        };

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
                let standard = artifact_dir.join("model.safetensors");
                let open_clip = artifact_dir.join("open_clip_model.safetensors");
                if standard.exists() {
                    vec![standard]
                } else if open_clip.exists() {
                    vec![open_clip]
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

        let tokenizer = discover_local_tokenizer(&artifact_dir);

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
            tokenizer,
            model_config,
            preprocessor_config: read_local_preprocessor_config(&artifact_dir),
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

        // Try standard config.json first, then OpenCLIP open_clip_config.json
        let config_path = {
            let standard = path.join("config.json");
            let open_clip = path.join("open_clip_config.json");
            if standard.exists() {
                standard
            } else if open_clip.exists() {
                open_clip
            } else {
                return Err(JammiError::Model {
                    model_id: source.to_string(),
                    message: "Missing config.json or open_clip_config.json in model directory"
                        .into(),
                });
            }
        };
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        let has_safetensors = path.join("model.safetensors").exists()
            || path.join("open_clip_model.safetensors").exists();
        let has_onnx = path.join("model.onnx").exists();

        if !has_safetensors && !has_onnx {
            return Err(JammiError::Model {
                model_id: source.to_string(),
                message: "No model weights found (need model.safetensors, \
                          open_clip_model.safetensors, or model.onnx)"
                    .into(),
            });
        }

        let backend = backend_hint.unwrap_or(if has_onnx {
            BackendType::Ort
        } else {
            BackendType::Candle
        });

        let weights_paths = match backend {
            BackendType::Candle => {
                let standard = path.join("model.safetensors");
                let open_clip = path.join("open_clip_model.safetensors");
                if standard.exists() {
                    vec![standard]
                } else if open_clip.exists() {
                    vec![open_clip]
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

        let tokenizer = discover_local_tokenizer(path);

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
            tokenizer,
            model_config: config,
            preprocessor_config: read_local_preprocessor_config(path),
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

        // Try standard config.json first, then OpenCLIP open_clip_config.json
        let config_path = repo
            .get("config.json")
            .or_else(|_| repo.get("open_clip_config.json"))
            .map_err(|e| JammiError::Model {
                model_id: source.to_string(),
                message: format!("Failed to download config: {e}"),
            })?;
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        // Feature-extractor geometry for the audio (CLAP fusion) front-end.
        // Optional: text/vision repos don't ship it, so a missing file is not
        // an error — only audio models read it downstream.
        let preprocessor_config: Option<serde_json::Value> = repo
            .get("preprocessor_config.json")
            .ok()
            .and_then(|p| std::fs::File::open(p).ok())
            .and_then(|f| serde_json::from_reader(f).ok());

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

        // Prefer the HF-converted tokenizer.json if it exists; otherwise
        // fall back to the OpenCLIP native vocab file for stock OpenCLIP
        // repos that ship `bpe_simple_vocab_16e6.txt.gz` instead.
        let tokenizer = repo
            .get("tokenizer.json")
            .ok()
            .map(TokenizerSource::HuggingFaceJson)
            .or_else(|| {
                repo.get("bpe_simple_vocab_16e6.txt.gz")
                    .ok()
                    .map(TokenizerSource::OpenClipBpe)
            });

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
            tokenizer,
            model_config: config,
            preprocessor_config,
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
        // Try standard naming first, then OpenCLIP naming
        if let Ok(path) = repo.get("model.safetensors") {
            return Ok(vec![path]);
        }
        if let Ok(path) = repo.get("open_clip_model.safetensors") {
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

/// Read and parse `preprocessor_config.json` from a local model directory,
/// if present. This is the feature-extractor geometry the audio (CLAP fusion)
/// front-end is driven by; absent for text/vision models, which don't use it.
fn read_local_preprocessor_config(dir: &Path) -> Option<serde_json::Value> {
    let path = dir.join("preprocessor_config.json");
    std::fs::File::open(path)
        .ok()
        .and_then(|f| serde_json::from_reader(f).ok())
}

/// Locate a tokenizer artifact inside a local model directory, preferring
/// an HF-shape `tokenizer.json` and falling back to OpenCLIP's native
/// `bpe_simple_vocab_16e6.txt.gz`.
fn discover_local_tokenizer(dir: &Path) -> Option<TokenizerSource> {
    let hf = dir.join("tokenizer.json");
    if hf.exists() {
        return Some(TokenizerSource::HuggingFaceJson(hf));
    }
    let bpe = dir.join("bpe_simple_vocab_16e6.txt.gz");
    if bpe.exists() {
        return Some(TokenizerSource::OpenClipBpe(bpe));
    }
    None
}
