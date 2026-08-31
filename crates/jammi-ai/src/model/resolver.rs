use std::path::{Path, PathBuf};
use std::sync::Arc;

use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::ArtifactStore;

use super::backend::gguf::estimate_gguf_residency;
use super::{
    BackendType, ModelId, ModelSource, ModelTask, ResolvedModel, TokenizerSource, WeightsFormat,
};

/// The canonical GGUF weight filename (issue #351): mirrors
/// `model.safetensors`/`model.onnx`'s own literal-filename convention — the
/// digest-slot machinery (`backend::candle::all_candidate_paths`) stats
/// known names only, so a GGUF checkpoint must be named exactly this, never
/// sniffed by extension. A directory carrying some OTHER `*.gguf` filename
/// is a typed refusal naming the file(s) found and this convention — see
/// `ModelResolver::resolve_local`/`ModelResolver::resolve_hf_hub`.
const GGUF_WEIGHTS_FILENAME: &str = "model.gguf";

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
                    weights_format: base_resolved.weights_format,
                    task,
                    config_path: base_resolved.config_path,
                    weights_paths: base_resolved.weights_paths,
                    tokenizer: base_resolved.tokenizer,
                    model_config: base_resolved.model_config,
                    preprocessor_config: base_resolved.preprocessor_config,
                    pooling_config: base_resolved.pooling_config,
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
        let (weights_paths, weights_format): (Vec<PathBuf>, WeightsFormat) = match backend {
            BackendType::Candle => {
                let standard = artifact_dir.join("model.safetensors");
                let open_clip = artifact_dir.join("open_clip_model.safetensors");
                let gguf = artifact_dir.join(GGUF_WEIGHTS_FILENAME);
                if standard.exists() {
                    (vec![standard], WeightsFormat::Safetensors)
                } else if open_clip.exists() {
                    (vec![open_clip], WeightsFormat::Safetensors)
                } else if gguf.exists() {
                    (vec![gguf], WeightsFormat::Gguf)
                } else {
                    return Ok(None);
                }
            }
            BackendType::Ort => {
                let p = artifact_dir.join("model.onnx");
                if p.exists() {
                    (vec![p], WeightsFormat::Onnx)
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        let tokenizer = discover_local_tokenizer(&artifact_dir);

        let estimated_memory: usize = if weights_format == WeightsFormat::Gguf {
            estimate_gguf_residency(&weights_paths[0], &model_config, &model_id.0)?
        } else {
            weights_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len() as usize)
                .sum()
        };

        let pooling_config = read_local_pooling_config(&artifact_dir, &model_id.0)?;

        Ok(Some(ResolvedModel {
            model_id,
            backend,
            weights_format,
            task,
            config_path,
            weights_paths,
            tokenizer,
            model_config,
            preprocessor_config: read_local_preprocessor_config(&artifact_dir),
            pooling_config,
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
        let has_gguf = path.join(GGUF_WEIGHTS_FILENAME).exists();

        // Precedence FROZEN (issue #351): safetensors-or-onnx wins, byte-for-
        // byte, exactly as before this feature existed. Only when NEITHER is
        // present does a `model.gguf` file (or the typed "found *.gguf but
        // not model.gguf" refusal) enter the picture at all.
        if !has_safetensors && !has_onnx && !has_gguf {
            if let Some(other) = other_gguf_refusal(source, path) {
                return Err(other);
            }
            return Err(JammiError::Model {
                model_id: source.to_string(),
                message: "No model weights found (need model.safetensors, \
                          open_clip_model.safetensors, model.onnx, or model.gguf)"
                    .into(),
            });
        }

        let backend = backend_hint.unwrap_or(if has_onnx {
            BackendType::Ort
        } else {
            BackendType::Candle
        });

        let (weights_paths, weights_format) = match backend {
            BackendType::Candle => {
                let standard = path.join("model.safetensors");
                let open_clip = path.join("open_clip_model.safetensors");
                if standard.exists() {
                    (vec![standard], WeightsFormat::Safetensors)
                } else if open_clip.exists() {
                    (vec![open_clip], WeightsFormat::Safetensors)
                } else if has_gguf {
                    (vec![path.join(GGUF_WEIGHTS_FILENAME)], WeightsFormat::Gguf)
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
                    (vec![p], WeightsFormat::Onnx)
                } else {
                    // Reached when the directory carries `model.gguf` (or a
                    // non-canonical `*.gguf` file) but the caller pinned the
                    // ORT backend — GGUF is a Candle-only weight-storage
                    // format (issue #351), so this is the correct typed
                    // refusal, unmodified from today's ONNX-missing case.
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

        let estimated_memory: usize = if weights_format == WeightsFormat::Gguf {
            estimate_gguf_residency(&weights_paths[0], &config, &source.to_string())?
        } else {
            weights_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len() as usize)
                .sum()
        };

        Ok(ResolvedModel {
            model_id: ModelId::from(source),
            weights_format,
            backend,
            task,
            config_path,
            weights_paths,
            tokenizer,
            model_config: config,
            preprocessor_config: read_local_preprocessor_config(path),
            pooling_config: read_local_pooling_config(path, &source.to_string())?,
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

        // Sentence-transformers pooling declaration. Optional: bare BERT repos
        // don't ship it — the repo not having the file is not an error, only
        // the mean fallback applies downstream. But once the file HAS been
        // downloaded, it must be readable and parseable JSON — a corrupt
        // pooling declaration must never collapse into the same "absent"
        // case that drives the mean fallback.
        let pooling_config: Option<serde_json::Value> = match repo.get("1_Pooling/config.json") {
            Err(_) => None,
            Ok(downloaded_path) => {
                let file = std::fs::File::open(&downloaded_path)?;
                Some(
                    serde_json::from_reader(file).map_err(|e| JammiError::Model {
                        model_id: source.to_string(),
                        message: format!("1_Pooling/config.json is present but unparseable: {e}"),
                    })?,
                )
            }
        };

        // Fetch the repo listing at most ONCE for the two DECISIONS that can
        // consume it — backend auto-selection and the Candle weights-format
        // plan (issue #351 wave 13 audit advisory 1) — never two separate
        // live `info()` calls that could observe two different snapshots of
        // a repo being pushed to concurrently. A failed fetch becomes
        // `None`, not a fatal error here — hf-hub 0.5's `ApiRepo::get` is
        // CACHE-FIRST and network-free on a hit (sync.rs:758-764) while
        // `ApiRepo::info` is network-only (sync.rs:860-878), so a warm-cache
        // repo must keep resolving exactly as it did before this feature
        // existed even with no network at all. What `None` means to each
        // decision is owned by that decision's own pure function
        // (`select_backend_from_listing`, `hub_candle_weights_plan`), never
        // decided at this call site.
        //
        // Lazy (issue #351 wave 14 round-8 audit advisory 2): a HINTED
        // resolve for a non-Candle backend (e.g. `Some(BackendType::Ort)`)
        // consumes neither decision above, so it must make NO listing call
        // at all — restores the base (pre-#351) behavior for hinted
        // non-Candle resolves, where `repo.info()` was never invoked.
        let listing: Option<Vec<String>> =
            if backend_hint.is_none() || backend_hint == Some(BackendType::Candle) {
                repo.info()
                    .ok()
                    .map(|info| info.siblings.into_iter().map(|s| s.rfilename).collect())
            } else {
                None
            };

        let backend =
            backend_hint.unwrap_or_else(|| select_backend_from_listing(listing.as_deref()));

        // Precedence FROZEN (issue #351): safetensors wins, byte-for-byte,
        // exactly as before this feature existed. The choice between
        // safetensors/gguf/refusal/attempt is made from the repo LISTING
        // (`hub_candle_weights_plan` over `repo.info()`'s siblings) alone,
        // never from a download outcome — a transient download failure on a
        // repo that lists BOTH formats must propagate as the failure it is,
        // not silently substitute the other weight format (issue #351 wave
        // 12; this is the confident-wrong-number class this unit exists to
        // make fail-loud). `repo.info()` failing outright is NOT itself a
        // typed error: a missing listing is the `SafetensorsOnlyAttempt`
        // arm below — the frozen pre-#351 path, never a guessed gguf format
        // and never the listing-failure or rename-refusal error (issue
        // #351 wave 13).
        let (weights_paths, weights_format) = match backend {
            BackendType::Candle => {
                match hub_candle_weights_plan(listing.as_deref(), GGUF_WEIGHTS_FILENAME) {
                    HubCandleWeightsPlan::Safetensors(listed_safetensors) => {
                        // The listing PROVED at least one safetensors
                        // sibling is present, so a download failure here is
                        // worth naming that fact (issue #351 wave 13 audit
                        // advisory 2): distinguishes "listed but every
                        // download failed" from the bare message
                        // `SafetensorsOnlyAttempt` below also returns when
                        // there was no listing to make that claim from. The
                        // names come straight off the plan arm that decided
                        // this branch (issue #351 wave 14 round-8 audit
                        // advisory 1), not re-derived from `listing` here.
                        (
                            self.download_safetensors(&repo, source).map_err(|e| {
                                annotate_listed_safetensors_download_failure(e, &listed_safetensors)
                            })?,
                            WeightsFormat::Safetensors,
                        )
                    }
                    HubCandleWeightsPlan::SafetensorsOnlyAttempt => (
                        self.download_safetensors(&repo, source)?,
                        WeightsFormat::Safetensors,
                    ),
                    HubCandleWeightsPlan::Gguf => {
                        let p = repo
                            .get(GGUF_WEIGHTS_FILENAME)
                            .map_err(|e| JammiError::Model {
                                model_id: source.to_string(),
                                message: format!("Failed to download {GGUF_WEIGHTS_FILENAME}: {e}"),
                            })?;
                        (vec![p], WeightsFormat::Gguf)
                    }
                    HubCandleWeightsPlan::NonCanonicalGguf(others) => {
                        return Err(gguf_rename_refusal(source, others));
                    }
                    HubCandleWeightsPlan::Neither => {
                        return Err(JammiError::Model {
                            model_id: source.to_string(),
                            message: "No safetensors weights found".into(),
                        });
                    }
                }
            }
            BackendType::Ort => (self.download_onnx(&repo, source)?, WeightsFormat::Onnx),
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

        let estimated_memory: usize = if weights_format == WeightsFormat::Gguf {
            estimate_gguf_residency(&weights_paths[0], &config, &source.to_string())?
        } else {
            weights_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len() as usize)
                .sum()
        };

        Ok(ResolvedModel {
            model_id: ModelId::from(source),
            backend,
            weights_format,
            task,
            config_path,
            weights_paths,
            tokenizer,
            model_config: config,
            preprocessor_config,
            pooling_config,
            base_model_id: None,
            adapter_path: None,
            estimated_memory,
        })
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

/// `Some(typed error)` when `dir` contains at least one `*.gguf` file OTHER
/// than the canonical `GGUF_WEIGHTS_FILENAME` — names every such file and
/// points at the convention. `None` when the directory listing itself fails
/// (best-effort — the caller falls back to the generic "no weights found"
/// message) or lists no `*.gguf` file at all.
fn other_gguf_refusal(source: &ModelSource, dir: &Path) -> Option<JammiError> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut others: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|name| name.ends_with(".gguf") && name != GGUF_WEIGHTS_FILENAME)
        .collect();
    if others.is_empty() {
        return None;
    }
    others.sort();
    Some(gguf_rename_refusal(source, others))
}

/// The outcome of [`decide_hub_weights_format`]: which weight format a Hub
/// repo's file LISTING selects, decided once and up front — never inferred
/// from a download outcome (issue #351 wave 12). This is the mechanism that
/// keeps a transient download failure on a repo carrying BOTH formats from
/// silently switching the served weight format: the format is fixed by the
/// listing before any `repo.get(..)` is attempted, so a download failure in
/// either the `Safetensors` or `Gguf` arm propagates as the failure it is.
#[derive(Debug, PartialEq, Eq)]
enum HubWeightsDecision {
    /// At least one `*.safetensors` sibling is listed — safetensors wins,
    /// byte-for-byte, exactly as the local-directory precedence (frozen
    /// since before issue #351).
    Safetensors,
    /// No safetensors sibling listed, but the canonical GGUF filename is.
    Gguf,
    /// Neither safetensors nor the canonical GGUF filename listed, but at
    /// least one OTHER `*.gguf` sibling is — the typed "rename to the
    /// canonical name" refusal, naming every such file (sorted).
    NonCanonicalGguf(Vec<String>),
    /// No safetensors sibling and no `*.gguf` sibling of any name.
    Neither,
}

/// Decide the Hub-path weight format from a repo's sibling filename LISTING
/// alone — a pure function over names, deliberately independent of
/// `hf_hub`/network types so it is unit-testable without a live repo (issue
/// #351 wave 12). `canonical_gguf` is `GGUF_WEIGHTS_FILENAME` in production;
/// parameterized here only so tests can assert the exact constant is what
/// production passes.
fn decide_hub_weights_format(siblings: &[String], canonical_gguf: &str) -> HubWeightsDecision {
    if siblings.iter().any(|name| name.ends_with(".safetensors")) {
        return HubWeightsDecision::Safetensors;
    }
    if siblings.iter().any(|name| name == canonical_gguf) {
        return HubWeightsDecision::Gguf;
    }
    let mut others: Vec<String> = siblings
        .iter()
        .filter(|name| name.ends_with(".gguf"))
        .cloned()
        .collect();
    if others.is_empty() {
        return HubWeightsDecision::Neither;
    }
    others.sort();
    HubWeightsDecision::NonCanonicalGguf(others)
}

/// The FULL decision the Candle-backend Hub-path resolve makes about which
/// weight format to load, including the case the repo LISTING itself is
/// unavailable — a strict superset of [`HubWeightsDecision`]'s four
/// listing-present arms plus a fifth (issue #351 wave 13 audit block). `None`
/// listing here means `repo.info()` itself failed, which hf-hub 0.5 makes a
/// perfectly ordinary outcome: `ApiRepo::get` is CACHE-FIRST and
/// network-free on a hit (sync.rs:758-764), while `ApiRepo::info` is
/// network-only (sync.rs:860-878) — so a warm-cache safetensors-only repo
/// that resolved fine offline before this feature existed must keep doing
/// so. `hub_candle_weights_plan` is the single pure function that owns this
/// decision; every arm below is unit-tested without a live repo.
#[derive(Debug, PartialEq, Eq)]
enum HubCandleWeightsPlan {
    /// Listing available, safetensors sibling(s) listed — download them; a
    /// failure here PROPAGATES (never falls back to gguf). Carries the
    /// LISTED `*.safetensors` sibling name(s), taken from the same listing
    /// that produced this arm (round-8 audit advisory, issue #351 wave
    /// 14) — the call site needs these names to annotate a download
    /// failure, and previously re-derived them from `listing` a second
    /// time; carrying them here makes that re-derivation impossible to
    /// drift from the decision that actually selected this arm.
    Safetensors(Vec<String>),
    /// Listing available, no safetensors but the canonical `model.gguf` IS
    /// listed — download it; a failure here PROPAGATES too.
    Gguf,
    /// Listing available, no safetensors, no canonical `model.gguf`, but some
    /// OTHER `*.gguf` sibling is listed — the typed rename refusal.
    NonCanonicalGguf(Vec<String>),
    /// Listing available, no safetensors and no `*.gguf` sibling at all.
    Neither,
    /// Listing UNAVAILABLE (`repo.info()` failed). Never decide gguf and
    /// never emit the listing-failure or rename-refusal error from a failed
    /// listing — this is the frozen pre-#351 path: attempt the cache-first
    /// safetensors download exactly as the pre-feature code did, and
    /// propagate whatever error THAT returns.
    SafetensorsOnlyAttempt,
}

/// Decide the Candle-backend Hub-path weights plan from an OPTIONAL repo
/// listing. `None` means the listing itself is unavailable (`repo.info()`
/// failed) — NOT "no siblings"; `Some(&[])`/a listing with no weight
/// siblings is the ordinary `Neither` arm. Pure and independent of
/// `hf_hub`/network types, so every one of the five arms is unit-testable
/// without a live repo (issue #351 wave 13).
fn hub_candle_weights_plan(
    listing: Option<&[String]>,
    canonical_gguf: &str,
) -> HubCandleWeightsPlan {
    let Some(siblings) = listing else {
        return HubCandleWeightsPlan::SafetensorsOnlyAttempt;
    };
    match decide_hub_weights_format(siblings, canonical_gguf) {
        HubWeightsDecision::Safetensors => {
            let listed: Vec<String> = siblings
                .iter()
                .filter(|name| name.ends_with(".safetensors"))
                .cloned()
                .collect();
            HubCandleWeightsPlan::Safetensors(listed)
        }
        HubWeightsDecision::Gguf => HubCandleWeightsPlan::Gguf,
        HubWeightsDecision::NonCanonicalGguf(others) => {
            HubCandleWeightsPlan::NonCanonicalGguf(others)
        }
        HubWeightsDecision::Neither => HubCandleWeightsPlan::Neither,
    }
}

/// Decide Hub backend auto-selection (issue #351's ONNX arm) from an
/// OPTIONAL repo listing — pure, and deliberately shares the SAME listing
/// `hub_candle_weights_plan` decides the weights format from (issue #351
/// wave 13 audit advisory 1: one live `repo.info()` fetch per resolve, not
/// two separate snapshots of a repo that could be pushed to concurrently
/// between them). `None` (listing unavailable) falls back to `Candle`,
/// exactly as the prior `if let Ok(info) = repo.info()` did on a failed
/// fetch.
fn select_backend_from_listing(listing: Option<&[String]>) -> BackendType {
    if let Some(siblings) = listing {
        if siblings.iter().any(|s| s == "model.onnx") {
            return BackendType::Ort;
        }
    }
    BackendType::Candle
}

/// Wrap a failed [`ModelResolver::download_safetensors`] error with context
/// naming the SPECIFIC safetensors sibling(s) the repo listing had already
/// proven present (issue #351 wave 13 audit advisory 2) — distinguishes "the
/// listing said safetensors were there and every download of them failed"
/// from the bare "No safetensors weights found" message
/// `download_safetensors` also returns on the `SafetensorsOnlyAttempt` arm
/// (no listing at all, so no such proof to name).
fn annotate_listed_safetensors_download_failure(
    err: JammiError,
    listed_safetensors: &[String],
) -> JammiError {
    match err {
        JammiError::Model { model_id, message } => JammiError::Model {
            model_id,
            message: format!(
                "{message} (repo listing named safetensors sibling(s) {} but every \
                 download attempt failed)",
                listed_safetensors.join(", ")
            ),
        },
        other => other,
    }
}

/// The typed "found GGUF file(s) but not the canonical name" refusal, shared
/// verbatim between the local-directory (`other_gguf_refusal`) and Hub
/// (`decide_hub_weights_format`'s `NonCanonicalGguf` arm) paths. `others`
/// need not be pre-sorted; this sorts for deterministic message text
/// (family J).
fn gguf_rename_refusal(source: &ModelSource, mut others: Vec<String>) -> JammiError {
    others.sort();
    JammiError::Model {
        model_id: source.to_string(),
        message: format!(
            "Found GGUF file(s) {} but no '{GGUF_WEIGHTS_FILENAME}' — the canonical \
             quantized-weights filename (mirrors model.safetensors/model.onnx); rename to \
             '{GGUF_WEIGHTS_FILENAME}' to load it",
            others.join(", ")
        ),
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

/// Read and parse `1_Pooling/config.json` from a local model directory.
///
/// Returns `Ok(None)` iff the file is genuinely absent — the historical bare
/// BERT repo shape with no `1_Pooling/` subfolder at all, which falls back to
/// mean pooling downstream. A file that is *present* but cannot be opened or
/// parsed as JSON is a hard error: collapsing "corrupt" into the same `None`
/// that drives the mean fallback would let a truncated pooling declaration
/// silently produce a confident-wrong embedding.
fn read_local_pooling_config(dir: &Path, model_id: &str) -> Result<Option<serde_json::Value>> {
    let path = dir.join("1_Pooling/config.json");
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(JammiError::Model {
                model_id: model_id.to_string(),
                message: format!("1_Pooling/config.json is present but could not be opened: {e}"),
            })
        }
    };
    serde_json::from_reader(file)
        .map(Some)
        .map_err(|e| JammiError::Model {
            model_id: model_id.to_string(),
            message: format!("1_Pooling/config.json is present but unparseable: {e}"),
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Arm 1: a repo listing carrying BOTH `model.safetensors` and
    /// `model.gguf` decides `Safetensors` — the frozen precedence, decided
    /// from the LISTING alone, never from a download attempt (issue #351
    /// wave 12's defect: this is the exact shape a failed-download fallback
    /// would get wrong).
    #[test]
    fn decide_hub_weights_format_prefers_safetensors_when_both_are_listed() {
        let siblings = names(&["config.json", "model.safetensors", "model.gguf"]);
        assert_eq!(
            decide_hub_weights_format(&siblings, GGUF_WEIGHTS_FILENAME),
            HubWeightsDecision::Safetensors
        );
    }

    /// Arm 2: no safetensors sibling, canonical `model.gguf` listed.
    #[test]
    fn decide_hub_weights_format_selects_gguf_when_only_gguf_is_listed() {
        let siblings = names(&["config.json", "model.gguf", "tokenizer.json"]);
        assert_eq!(
            decide_hub_weights_format(&siblings, GGUF_WEIGHTS_FILENAME),
            HubWeightsDecision::Gguf
        );
    }

    /// Arm 3: no safetensors, no canonical `model.gguf`, but a
    /// differently-named `*.gguf` sibling — the typed rename refusal,
    /// naming every non-canonical file found (sorted).
    #[test]
    fn decide_hub_weights_format_refuses_non_canonical_gguf_filenames() {
        let siblings = names(&["config.json", "weights.q4.gguf", "other.gguf"]);
        assert_eq!(
            decide_hub_weights_format(&siblings, GGUF_WEIGHTS_FILENAME),
            HubWeightsDecision::NonCanonicalGguf(vec![
                "other.gguf".to_string(),
                "weights.q4.gguf".to_string(),
            ])
        );
    }

    /// Arm 4: no safetensors and no `*.gguf` sibling of any name at all.
    #[test]
    fn decide_hub_weights_format_is_neither_when_no_weights_are_listed() {
        let siblings = names(&["config.json", "tokenizer.json"]);
        assert_eq!(
            decide_hub_weights_format(&siblings, GGUF_WEIGHTS_FILENAME),
            HubWeightsDecision::Neither
        );
    }

    /// The defect this unit fixes, pinned directly at the decision level: a
    /// repo listing carrying both formats never depends on which download
    /// happens to succeed — `decide_hub_weights_format` is a pure function
    /// of the listing, so calling it twice for the same listing (standing
    /// in for "network attempt #1 failed transiently, #2 would have
    /// succeeded") always returns the same, safetensors-preferring answer.
    #[test]
    fn decide_hub_weights_format_is_deterministic_over_a_dual_format_listing() {
        let siblings = names(&["model.safetensors", "model.gguf"]);
        let first = decide_hub_weights_format(&siblings, GGUF_WEIGHTS_FILENAME);
        let second = decide_hub_weights_format(&siblings, GGUF_WEIGHTS_FILENAME);
        assert_eq!(first, HubWeightsDecision::Safetensors);
        assert_eq!(second, HubWeightsDecision::Safetensors);
    }

    // ─────────────────────────────────────────────────────────────────────
    // `hub_candle_weights_plan` (issue #351 wave 13): the FULL five-arm
    // decision, including the `None`-listing arm `decide_hub_weights_format`
    // alone can't express. Plan arm 1: `Some` + safetensors-listed.
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn plan_arm_1_some_listing_with_safetensors_selects_safetensors() {
        let siblings = names(&["config.json", "model.safetensors", "model.gguf"]);
        assert_eq!(
            hub_candle_weights_plan(Some(&siblings), GGUF_WEIGHTS_FILENAME),
            HubCandleWeightsPlan::Safetensors(vec!["model.safetensors".to_string()])
        );
    }

    /// Plan arm 2: `Some` + only the canonical `model.gguf` listed.
    #[test]
    fn plan_arm_2_some_listing_with_only_canonical_gguf_selects_gguf() {
        let siblings = names(&["config.json", "model.gguf"]);
        assert_eq!(
            hub_candle_weights_plan(Some(&siblings), GGUF_WEIGHTS_FILENAME),
            HubCandleWeightsPlan::Gguf
        );
    }

    /// Plan arm 3: `Some` + only a non-canonical `*.gguf` sibling — the
    /// typed rename refusal, naming every such file (sorted).
    #[test]
    fn plan_arm_3_some_listing_with_non_canonical_gguf_only_refuses_with_names() {
        let siblings = names(&["config.json", "weights.q4.gguf", "other.gguf"]);
        assert_eq!(
            hub_candle_weights_plan(Some(&siblings), GGUF_WEIGHTS_FILENAME),
            HubCandleWeightsPlan::NonCanonicalGguf(vec![
                "other.gguf".to_string(),
                "weights.q4.gguf".to_string(),
            ])
        );
    }

    /// Plan arm 4: `Some` + neither safetensors nor any `*.gguf` sibling.
    #[test]
    fn plan_arm_4_some_listing_with_neither_format_is_neither() {
        let siblings = names(&["config.json", "tokenizer.json"]);
        assert_eq!(
            hub_candle_weights_plan(Some(&siblings), GGUF_WEIGHTS_FILENAME),
            HubCandleWeightsPlan::Neither
        );
    }

    /// Plan arm 5 (the RED oracle this wave closes): `None` — the listing
    /// itself is unavailable (`repo.info()` failed, e.g. no network against
    /// a warm cache-first hit) — must NEVER decide gguf and must NEVER
    /// return the listing-failure or rename-refusal error; it is
    /// `SafetensorsOnlyAttempt`, the frozen pre-#351 path. At commit
    /// 61b7bc7e (before this wave), the equivalent call site (resolver.rs,
    /// prior `repo.info().map_err(..)?`) made a failed listing a HARD,
    /// fatal `JammiError::Model` naming "Failed to list repo files" —
    /// exactly the shape this arm's existence forbids. Asserting `None`
    /// lands on `SafetensorsOnlyAttempt` (never `NonCanonicalGguf`/`Neither`
    /// and never a listing-failure error) is what would have caught that
    /// regression: a warm-cache safetensors-only repo resolving offline,
    /// which the pre-feature code supported and 61b7bc7e broke.
    #[test]
    fn plan_arm_5_none_listing_attempts_safetensors_only_never_decides_gguf() {
        assert_eq!(
            hub_candle_weights_plan(None, GGUF_WEIGHTS_FILENAME),
            HubCandleWeightsPlan::SafetensorsOnlyAttempt
        );
    }

    /// The plan's `None` arm must agree with `decide_hub_weights_format`'s
    /// `Some` arms whenever a listing IS actually available — `None` is not
    /// a generic "give up" branch, it activates only when the listing
    /// itself could not be fetched at all.
    #[test]
    fn plan_agrees_with_decide_hub_weights_format_when_listing_is_present() {
        let siblings = names(&["model.safetensors", "model.gguf"]);
        assert_eq!(
            hub_candle_weights_plan(Some(&siblings), GGUF_WEIGHTS_FILENAME),
            HubCandleWeightsPlan::Safetensors(vec!["model.safetensors".to_string()])
        );
        assert_ne!(
            hub_candle_weights_plan(Some(&siblings), GGUF_WEIGHTS_FILENAME),
            HubCandleWeightsPlan::SafetensorsOnlyAttempt
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // `select_backend_from_listing`: shares the SAME optional listing
    // (advisory 1) — `None` falls back to `Candle`, exactly as the prior
    // `if let Ok(info) = repo.info()` did on a failed fetch.
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn select_backend_from_listing_picks_ort_when_model_onnx_is_listed() {
        let siblings = names(&["config.json", "model.onnx"]);
        assert_eq!(
            select_backend_from_listing(Some(&siblings)),
            BackendType::Ort
        );
    }

    #[test]
    fn select_backend_from_listing_falls_back_to_candle_when_listing_is_none() {
        assert_eq!(select_backend_from_listing(None), BackendType::Candle);
    }

    #[test]
    fn select_backend_from_listing_falls_back_to_candle_when_onnx_absent() {
        let siblings = names(&["config.json", "model.safetensors"]);
        assert_eq!(
            select_backend_from_listing(Some(&siblings)),
            BackendType::Candle
        );
    }

    /// Advisory 2: a listing that PROVED safetensors present but every
    /// download failed gets the original error message wrapped with the
    /// specific sibling name(s) the listing had already named — never
    /// silently substitutes the bare message.
    #[test]
    fn annotate_listed_safetensors_download_failure_names_the_listed_siblings() {
        let original = JammiError::Model {
            model_id: "some/repo".to_string(),
            message: "No safetensors weights found".to_string(),
        };
        let wrapped = annotate_listed_safetensors_download_failure(
            original,
            &["model.safetensors".to_string()],
        );
        let msg = wrapped.to_string();
        assert!(
            msg.contains("No safetensors weights found") && msg.contains("model.safetensors"),
            "expected the original message plus the listed sibling name, got: {msg}"
        );
    }
}
