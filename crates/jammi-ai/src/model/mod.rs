pub mod arch;
pub mod backend;
pub mod cache;
pub mod clip_bpe;
/// The single Hugging Face Hub client: `[models]` -> `HubSource`,
/// built once at the `jammi-ai` session choke point and shared by every
/// resolver/worker call site. See [`hub`]'s module docs for the precedence
/// chain and the `offline` promise.
pub mod hub;
/// The keyed single-flight memo both halves of [`cache::ModelCache`] are.
pub(crate) mod memo;
/// The shared "is this error message OOM-shaped" home. Neutral ground
/// between `inference` (the batch-halving retry) and `fine_tune` (the
/// training OOM guidance classifier) — neither reaches into the other's
/// module for this; both import from here.
pub(crate) mod oom;
pub mod resolver;
pub mod tokenizer;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::ArrayRef;
use backend::candle::CandleModel;
use jammi_db::error::Result;

use jammi_datafusion::BackendOutput;

/// Unique identifier for a loaded model, used as cache key.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&ModelSource> for ModelId {
    fn from(source: &ModelSource) -> Self {
        ModelId(source.to_string())
    }
}

/// The on-disk STORAGE format of a resolved model's weight files — an
/// explicit marker so a downstream consumer (the candle backend's load
/// dispatch, the digest/fingerprint machinery) branches on THIS, never on
/// sniffing `weights_paths`' file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsFormat {
    /// One or more `*.safetensors` files (the default).
    Safetensors,
    /// A single GGUF file (`model.gguf`, the literal canonical name — see
    /// [`resolver::ModelResolver`]'s module doc) carrying k-quant and/or
    /// dense tensors, loaded by the `Candle` backend's GGUF path.
    Gguf,
}

/// The model vocabulary this crate's public API takes — where a model is
/// loaded from and what it computes — defined by `jammi-datafusion`, whose
/// operators run them, and re-exported so a caller of this crate names them
/// from one place.
pub use jammi_datafusion::{ModelSource, ModelTask};

/// Where the tokenizer for a resolved model lives, and what shape it is.
///
/// Most checkpoints carry an HF-converted `tokenizer.json`; stock OpenCLIP
/// repos instead ship the legacy gzipped BPE vocab. The resolver picks
/// whichever is present and the loader dispatches on the variant.
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// HuggingFace-shape `tokenizer.json` (works for BERT-family, ModernBERT,
    /// DistilBERT, and OpenCLIP repos that ship a pre-converted file).
    HuggingFaceJson(std::path::PathBuf),
    /// OpenCLIP-native `bpe_simple_vocab_16e6.txt.gz` — built directly into a
    /// BPE tokenizer at load time, no HF pre-conversion required.
    OpenClipBpe(std::path::PathBuf),
}

impl TokenizerSource {
    /// Filesystem path of the tokenizer artifact.
    pub fn path(&self) -> &std::path::Path {
        match self {
            Self::HuggingFaceJson(p) | Self::OpenClipBpe(p) => p,
        }
    }
}

/// A resolved model — files located, NOT yet loaded.
pub struct ResolvedModel {
    /// HuggingFace or local identifier for this model.
    pub model_id: ModelId,
    /// Weight-file storage format — see [`WeightsFormat`]'s own doc.
    pub weights_format: WeightsFormat,
    /// ML task this model performs.
    pub task: ModelTask,
    /// Path to the model's `config.json`.
    pub config_path: std::path::PathBuf,
    /// Paths to weight files (safetensors shards or one GGUF file).
    pub weights_paths: Vec<std::path::PathBuf>,
    /// Tokenizer source (HF JSON or OpenCLIP BPE), if present.
    pub tokenizer: Option<TokenizerSource>,
    /// Parsed contents of `config.json`.
    pub model_config: serde_json::Value,
    /// Parsed contents of `preprocessor_config.json`, if present. Carries the
    /// feature-extractor geometry (CLAP fusion front-end: sample rate, FFT
    /// window, hop, mel-filter band, max length) the audio path needs so the
    /// bytes-to-spectrogram transform is config-driven, not hardcoded.
    pub preprocessor_config: Option<serde_json::Value>,
    /// Parsed contents of `1_Pooling/config.json`, if present. Carries the
    /// sentence-transformers pooling declaration (`pooling_mode_cls_token`,
    /// `pooling_mode_mean_tokens`, etc.) so the text-embedding path pools the
    /// way the model actually declares. Absent for bare BERT repos that ship
    /// no `1_Pooling/` subfolder, in which case the mean default applies.
    pub pooling_config: Option<serde_json::Value>,
    /// Parent model ID for fine-tuned variants.
    pub base_model_id: Option<ModelId>,
    /// Path to LoRA adapter directory (for fine-tuned models).
    pub adapter_path: Option<std::path::PathBuf>,
    /// Estimated GPU memory in bytes (sum of weight file sizes).
    pub estimated_memory: usize,
}

impl ResolvedModel {
    /// The directory the weights live in — what a catalog row records as
    /// the model's location — or `None` when it is not a UTF-8 path.
    pub(crate) fn weights_dir(&self) -> Option<&str> {
        self.weights_paths.first()?.parent()?.to_str()
    }
}

/// Model architecture dimensions used for memory estimation and output sizing.
#[derive(Debug, Clone)]
pub struct ModelDimensions {
    /// Size of the hidden representation (embedding dimension).
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub num_attention_heads: usize,
    /// Feed-forward intermediate layer size.
    pub intermediate_size: usize,
}

impl ModelDimensions {
    /// Parse from HuggingFace config.json or OpenCLIP open_clip_config.json.
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        // HF-CLAP audio tower (`ClapAudioModelWithProjection`): top-level
        // `clap_audio_model` config (or a nested `audio_config` block under a
        // top-level `ClapConfig`). Its embedding dimensionality is
        // `projection_dim`; `num_attention_heads`/`depths` are per-stage arrays,
        // so the standard scalar-`num_attention_heads` text branch cannot parse
        // it — detect it first off `model_type`.
        if let Some(dims) = Self::from_hf_clap_config(config) {
            return Some(dims);
        }

        // Standard text model format (BERT, ModernBERT, etc.)
        if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
            let hidden_size = hidden_size as usize;
            let num_layers = config
                .get("num_hidden_layers")
                .or(config.get("num_layers"))?
                .as_u64()? as usize;
            let num_attention_heads = config["num_attention_heads"].as_u64()? as usize;
            let intermediate_size = config
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(hidden_size as u64 * 4) as usize;
            return Some(Self {
                hidden_size,
                num_layers,
                num_attention_heads,
                intermediate_size,
            });
        }

        // OpenCLIP format: `model_cfg.vision_cfg` with `embed_dim` at top
        // level. The "is this OpenCLIP?" question is answered by the SHARED
        // predicate (`arch::EncoderFamily::from_config`), never by a local
        // `model_cfg` probe that could drift from the serving loader's own
        // branch; only the dimension MATH below is this module's.
        if arch::EncoderFamily::from_config(config) == Some(arch::EncoderFamily::OpenClip) {
            let model_cfg = config.get("model_cfg")?;
            let vision_cfg = model_cfg.get("vision_cfg")?;
            let embed_dim = model_cfg.get("embed_dim").and_then(|v| v.as_u64())? as usize;
            let width = vision_cfg.get("width").and_then(|v| v.as_u64())? as usize;
            let num_layers = vision_cfg.get("layers").and_then(|v| v.as_u64())? as usize;
            // heads may be absent in OpenCLIP configs — default to width/64 (ViT convention)
            let num_attention_heads = vision_cfg
                .get("heads")
                .and_then(|v| v.as_u64())
                .unwrap_or((width / 64) as u64) as usize;
            let mlp_ratio = vision_cfg
                .get("mlp_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0);
            let intermediate_size = (width as f64 * mlp_ratio) as usize;
            return Some(Self {
                hidden_size: embed_dim,
                num_layers,
                num_attention_heads,
                intermediate_size,
            });
        }

        None
    }

    /// Parse the HF-CLAP audio-tower geometry (`ClapAudioModelWithProjection`),
    /// returning `None` for any non-CLAP config.
    ///
    /// Accepts both the flat `clap_audio_model` config and a top-level
    /// `ClapConfig` carrying a nested `audio_config`. The reported
    /// `hidden_size` is the tower's output embedding dimensionality
    /// (`projection_dim`, the shared cross-modal latent); `num_layers` is the
    /// number of hierarchical Swin stages (`depths.len()`); attention heads and
    /// the intermediate FFN size are taken from the final stage (the widest),
    /// which bounds the per-batch activation footprint.
    fn from_hf_clap_config(config: &serde_json::Value) -> Option<Self> {
        // Same shared predicate as the OpenCLIP arm above: the CLAP-detection
        // RULES live in `arch`, this function owns only the geometry read.
        // Widening note: `arch` also honours the `architectures`-listing
        // signal the serving loader has always applied, so a `ClapConfig`
        // that names `ClapModel` without a nested `model_type` now reaches
        // the geometry read here instead of falling through to the text arm.
        // It still returns `None` unless the CLAP geometry fields are
        // actually present (`?` below), so no non-CLAP config can be
        // mis-parsed into CLAP dimensions.
        if arch::EncoderFamily::from_config(config) != Some(arch::EncoderFamily::ClapAudio) {
            return None;
        }
        let audio = config.get("audio_config").unwrap_or(config);
        let projection_dim = config
            .get("projection_dim")
            .or_else(|| audio.get("projection_dim"))
            .and_then(|v| v.as_u64())? as usize;
        let final_width = audio.get("hidden_size").and_then(|v| v.as_u64())? as usize;
        let depths = audio.get("depths").and_then(|v| v.as_array())?;
        let num_layers = depths.len();
        let num_attention_heads = audio
            .get("num_attention_heads")
            .and_then(|v| v.as_array())
            .and_then(|h| h.last())
            .and_then(|v| v.as_u64())? as usize;
        let mlp_ratio = audio
            .get("mlp_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(4.0);
        let intermediate_size = (final_width as f64 * mlp_ratio) as usize;
        Some(Self {
            hidden_size: projection_dim,
            num_layers,
            num_attention_heads,
            intermediate_size,
        })
    }

    /// Peak activation memory for one inference batch (encoder-only, no gradients).
    pub fn estimate_activation_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        let bytes_per_elem = 4; // f32
        let attention_scores =
            batch_size * self.num_attention_heads * seq_len * seq_len * bytes_per_elem;
        let ffn_intermediate = batch_size * seq_len * self.intermediate_size * bytes_per_elem;
        attention_scores.max(ffn_intermediate)
    }
}

/// The saved fine-tune adapter a resolved model carries: its
/// `adapter_config.json`, parsed, and the path of its `adapter.safetensors`.
/// Read once, when the model is described; materializing installs it.
pub(crate) struct SavedAdapterFiles {
    pub(crate) config: crate::fine_tune::target::SavedAdapter,
    pub(crate) weights: PathBuf,
}

/// What planning a model's run needs to know about it, read without
/// allocating a tensor or sending a request: the identity a
/// materialization's environment records, the width of its output, and the
/// form of its regression head. A backend computes it before it
/// materializes the model, and a loaded model is materialized FROM a
/// description and reports it unchanged — so a submitter that plans against
/// a description and the executing process that records what ran can never
/// disagree.
pub struct ModelDescription {
    /// The id a materialization's environment records the model under.
    pub(crate) model_id: String,
    /// What the backend that runs the model described.
    pub(crate) backing: DescribedBacking,
}

/// What a description holds for the backend that runs the model.
pub(crate) enum DescribedBacking {
    /// A model this engine runs over local weights.
    Local(LocalDescription),
    /// A model a remote endpoint runs: its declaration is its description.
    Remote(jammi_db::store::manifest::RemoteRun),
}

/// A model described from its local files and configuration.
pub(crate) struct LocalDescription {
    /// The run of the model on the device it was described for — every
    /// output-affecting fact a materialization's environment records.
    pub(crate) run: jammi_db::store::manifest::LocalRun,
    /// The architecture's geometry, read from `config.json`.
    pub(crate) dimensions: ModelDimensions,
    /// The precision the configuration resolves for this model on the
    /// described device — `config.json`'s own `compute_precision`, else the
    /// device's default — which every head materializes at. The run's
    /// precision is the backbone's: the same, unless a saved encoder
    /// adapter's persisted `backbone_dtype` won.
    pub(crate) configured_precision: jammi_numerics::ComputePrecision,
    /// The saved fine-tune adapter, when the resolved model carries one.
    pub(crate) saved_adapter: Option<SavedAdapterFiles>,
    /// The stat-only staleness fingerprint of the files the run's content
    /// digest was hashed from. See [`backend::candle::ModelFingerprint`].
    pub(crate) fingerprint: backend::candle::ModelFingerprint,
}

impl ModelDescription {
    /// This model's identity as a materialization's environment records it
    /// — the one construction a submitter's prediction and the executing
    /// process's record both use.
    pub fn identity(&self) -> jammi_db::store::manifest::ModelIdentity {
        use jammi_db::store::manifest::ModelRun;
        jammi_db::store::manifest::ModelIdentity {
            model_id: self.model_id.clone(),
            run: match &self.backing {
                DescribedBacking::Local(local) => ModelRun::Local(local.run.clone()),
                DescribedBacking::Remote(remote) => ModelRun::Remote(remote.clone()),
            },
        }
    }

    /// The backend that runs this model.
    pub fn backend(&self) -> jammi_db::catalog::model_repo::ModelBackendKind {
        match &self.backing {
            DescribedBacking::Local(local) => local.run.backend.into(),
            DescribedBacking::Remote(_) => jammi_db::catalog::model_repo::ModelBackendKind::Remote,
        }
    }

    /// The run of a model this engine runs over local weights — its
    /// precision, content digest and weight format — or `None` for a model
    /// a remote endpoint runs.
    pub fn local_run(&self) -> Option<&jammi_db::store::manifest::LocalRun> {
        match &self.backing {
            DescribedBacking::Local(local) => Some(&local.run),
            DescribedBacking::Remote(_) => None,
        }
    }

    /// The architecture's geometry, for memory estimation, or `None` for a
    /// model a remote endpoint runs.
    pub fn dimensions(&self) -> Option<&ModelDimensions> {
        match &self.backing {
            DescribedBacking::Local(local) => Some(&local.dimensions),
            DescribedBacking::Remote(_) => None,
        }
    }

    /// Output dimensionality of the model's embedding head.
    ///
    /// For BERT-family encoders this is the transformer's `hidden_size`.
    /// For OpenCLIP-family models (vision and text towers) this is the
    /// projected shared-latent `embed_dim` — the dimension the emitted
    /// vectors carry and cross-modal cosine similarity is computed in, not
    /// the per-tower hidden `width`. For a remote model it is the width its
    /// declaration states and every response is held to.
    pub fn embedding_dim(&self) -> usize {
        match &self.backing {
            DescribedBacking::Local(local) => local.dimensions.hidden_size,
            DescribedBacking::Remote(remote) => remote.dimensions as usize,
        }
    }

    /// The persisted predictive-distribution form of a regression head
    /// (`Gaussian` or `Quantile { levels }`), or `None` for a model that is
    /// not a regression head. Serving selects the `Infer` output adapter on
    /// it, so a quantile-trained head is served as quantile points, never
    /// silently mis-decoded as a Gaussian `(mean, std)`.
    pub fn regression_form(
        &self,
    ) -> Option<&jammi_datafusion::inference::adapter::DistributionForm> {
        let DescribedBacking::Local(local) = &self.backing else {
            return None;
        };
        match local.saved_adapter.as_ref().map(|adapter| &adapter.config) {
            Some(crate::fine_tune::target::SavedAdapter::ProjectionHead(cfg)) => {
                cfg.regression_form.as_ref()
            }
            _ => None,
        }
    }

    /// Stat-only staleness probe of the files a local model's identity was
    /// computed from, re-`stat`ing (never re-reading) the same file set the
    /// digest was hashed from and comparing `(len, mtime)` against the
    /// snapshot taken when the description was computed.
    ///
    /// - `Ok(true)` — unchanged: the description still describes the files.
    ///   A remote model is always this: its declaration is read once, when
    ///   the session opens, and has no files to go stale.
    /// - `Ok(false)` — at least one fingerprinted file diverged: the caller
    ///   must discard this description and describe again.
    /// - `Err` — a fingerprinted file vanished or became unreadable: a
    ///   typed refusal, never a silent "treat as fresh".
    ///
    /// `(len, mtime)` is a staleness TRIPWIRE, not a cryptographic
    /// guarantee — a same-length, same-mtime content swap is invisible to
    /// it. The digest, recomputed fresh on every re-description, remains
    /// the sole attestation of the bytes that were hashed; this probe only
    /// decides WHEN a re-description is triggered. The guarantee is BOUNDED
    /// STALENESS, never per-hit freshness: `Ok(true)` proves the file set
    /// was unchanged at the instant the probe ran, not that it stays so
    /// while the caller goes on to use the description. See
    /// [`backend::candle::ModelFingerprint`] for the narrow scope this
    /// bound sits within.
    pub(crate) fn probe_freshness(&self) -> Result<bool> {
        match &self.backing {
            DescribedBacking::Local(local) => local.fingerprint.probe(),
            DescribedBacking::Remote(_) => Ok(true),
        }
    }

    /// The local parts a local backend materializes from, refused for a
    /// description of a model a remote endpoint runs — the two backends'
    /// halves never mix.
    pub(crate) fn local_parts(&self) -> Result<&LocalDescription> {
        match &self.backing {
            DescribedBacking::Local(local) => Ok(local),
            DescribedBacking::Remote(_) => Err(jammi_db::error::JammiError::Model {
                model_id: self.model_id.clone(),
                message: "a remote model's description reached a local backend".into(),
            }),
        }
    }
}

/// A model materialized, ready for inference: the weights of a described
/// model resident on a device, or a remote model's endpoint ready to take
/// requests. Every fact a materialization records about it is its
/// [`ModelDescription`], unchanged from the one it was materialized from.
pub struct LoadedModel {
    runner: Runner,
}

/// What runs a loaded model's forwards.
enum Runner {
    Candle(Box<CandleModel>),
    Remote {
        model: Arc<backend::remote::RemoteModel>,
        description: Arc<ModelDescription>,
    },
}

/// One forward's input, prepared for the runner that will take it.
pub enum PreparedInput {
    /// Tensors uploaded for a local model.
    Candle(backend::candle::PreparedInput),
    /// The rows of one request to a remote model.
    Remote(backend::remote::PreparedRequest),
}

impl LoadedModel {
    pub(crate) fn candle(model: CandleModel) -> Self {
        Self {
            runner: Runner::Candle(Box::new(model)),
        }
    }

    pub(crate) fn remote(
        model: Arc<backend::remote::RemoteModel>,
        description: Arc<ModelDescription>,
    ) -> Self {
        Self {
            runner: Runner::Remote { model, description },
        }
    }

    /// The description this model was materialized from.
    pub fn description(&self) -> &Arc<ModelDescription> {
        match &self.runner {
            Runner::Candle(candle) => candle.description(),
            Runner::Remote { description, .. } => description,
        }
    }

    /// The local backend's own model, for a consumer inside this crate that
    /// drives a tower directly (the fine-tune trainer). A remote model has
    /// no tower here to drive.
    pub(crate) fn backend_model(&self) -> Result<&CandleModel> {
        match &self.runner {
            Runner::Candle(candle) => Ok(candle),
            Runner::Remote { description, .. } => Err(jammi_db::error::JammiError::Model {
                model_id: description.model_id.clone(),
                message: "a remote model is served by its endpoint; this engine holds no \
                          tower of it to train or drive"
                    .into(),
            }),
        }
    }

    fn candle_model(&self) -> Option<&CandleModel> {
        match &self.runner {
            Runner::Candle(candle) => Some(candle),
            Runner::Remote { .. } => None,
        }
    }

    /// The tokenizer the text forward turns content into token ids with,
    /// or `None` for a model with no local text tower.
    pub fn tokenizer(&self) -> Option<&tokenizer::TokenizerWrapper> {
        self.candle_model()?.tokenizer.as_ref()
    }

    /// The pooling strategy the loaded text-embedding forward path ACTUALLY
    /// resolved to and applies — the SAME strategy
    /// `backend::candle::CandleTextForward::forward_pooled` pools with, read
    /// via `CandleModel::resolved_pooling`: a bench/report consumer must
    /// read this off the loaded model, never transcribe a fixture-declared
    /// constant that could silently drift from what actually served. `None`
    /// when this model has no local pooling (a CLAP audio tower, an OpenCLIP
    /// text tower whose output is already pooled-and-projected, a
    /// classification head, a remote model).
    pub fn resolved_pooling(&self) -> Option<jammi_encoders::Pooling> {
        self.candle_model()?.resolved_pooling()
    }

    /// The token-sequence bound the loaded text forward truncates its
    /// tokenization to (`backend::candle::CandleTextForward::max_sequence_length`).
    /// A consumer that counts the tokens a serve actually forwards must
    /// truncate at this bound, read off the loaded model, never at a value
    /// re-derived from `config.json`. `None` when the loaded model has no
    /// local text forward (a CLAP audio tower, a remote model).
    pub fn max_sequence_length(&self) -> Option<usize> {
        self.candle_model()?.max_sequence_length()
    }

    /// Every kernel admission decision this model's forwards have taken
    /// since it was loaded — `CandleModel::kernel_admission`. A remote
    /// model runs no kernel here, so its ledger is empty.
    pub fn kernel_admission(&self) -> jammi_kernels::admission::AdmissionLedger {
        match self.candle_model() {
            Some(candle) => candle.kernel_admission(),
            None => jammi_kernels::admission::AdmissionLedger::default(),
        }
    }

    /// Estimate device memory for one inference batch. A remote model's
    /// batch occupies no memory here.
    pub fn estimate_batch_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        self.description()
            .dimensions()
            .map_or(0, |d| d.estimate_activation_memory(batch_size, seq_len))
    }

    /// The persisted scaler's σ_y for a reloaded regression head, or `None` for a
    /// non-regression / no-scaler model. Serving reads this to scale a
    /// Gaussian head's served σ from the z-space the loss trained (σ_z ≈ 1) back
    /// to raw units (`σ_y·σ_z`) — the σ-axis half of the de-standardise contract
    /// (the mean/quantile axes carry σ_y in the backend's affine).
    pub fn regression_std_scale(&self) -> Option<f32> {
        self.candle_model()?.regression_std_scale()
    }

    /// TEST-ONLY non-vacuity seam: zero a loaded regression head's trained LoRA
    /// `B` factor so it regresses to its zero-initialised base and emits the
    /// scaler offset `μ_y` for every input (the untrained-head behaviour). No-op
    /// for a non-regression model. Used by the regression-surface tests to
    /// prove their group-separation assertion collapses to ≈0 when the head
    /// carries no learned signal. See
    /// [`backend::candle::CandleModel::zero_distribution_head_for_test`].
    #[doc(hidden)]
    pub fn zero_distribution_head_for_test(&mut self) {
        if let Runner::Candle(candle) = &mut self.runner {
            candle.zero_distribution_head_for_test();
        }
    }

    /// The cost of every row of `content` under `task`: its length along the
    /// axis a forward pads.
    pub fn row_costs(&self, content: &[ArrayRef], task: ModelTask) -> Result<Vec<u32>> {
        match &self.runner {
            Runner::Candle(candle) => candle.row_costs(content, task),
            Runner::Remote { model, .. } => model.row_costs(content, task),
        }
    }

    /// The ladder a forward under `task` pads its rows on.
    pub fn shape_ladder(&self, task: ModelTask) -> Result<jammi_numerics::ShapeLadder> {
        match &self.runner {
            Runner::Candle(candle) => candle.shape_ladder(task),
            Runner::Remote { model, .. } => model.shape_ladder(task),
        }
    }

    /// The host half of a forward: prepare `content` for the runner.
    pub fn prepare(&self, content: &[ArrayRef], task: ModelTask) -> Result<PreparedInput> {
        match &self.runner {
            Runner::Candle(candle) => candle.prepare(content, task).map(PreparedInput::Candle),
            Runner::Remote { model, .. } => model.prepare(content, task).map(PreparedInput::Remote),
        }
    }

    /// The device half of a forward: run the model over a prepared input —
    /// a local model's device operation, or a remote model's request. An
    /// input prepared by the other runner is refused.
    pub async fn forward_prepared(&self, input: PreparedInput) -> Result<BackendOutput> {
        match (&self.runner, input) {
            (Runner::Candle(candle), PreparedInput::Candle(input)) => {
                candle.forward_prepared(input)
            }
            (Runner::Remote { model, .. }, PreparedInput::Remote(request)) => {
                model.forward(request).await
            }
            _ => Err(jammi_db::error::JammiError::Inference(
                "an input prepared for another model's runner reached this one".into(),
            )),
        }
    }

    /// [`Self::prepare`] then [`Self::forward_prepared`], with no device
    /// admission between: the single-row query encoders' call.
    pub async fn forward(&self, content: &[ArrayRef], task: ModelTask) -> Result<BackendOutput> {
        self.forward_prepared(self.prepare(content, task)?).await
    }
}

/// RAII guard that decrements ref count on drop.
pub struct ModelGuard {
    /// Shared handle to the loaded model.
    pub model: Arc<LoadedModel>,
    ref_count: Arc<AtomicUsize>,
    /// A clone of the SAME `Arc<GpuPermit>` the owning `CacheEntry` holds. A
    /// `GpuPermit` releases its reservation (`GpuScheduler::reserved_memory
    /// -= bytes`) only when its LAST `Arc` clone drops (`GpuPermit`'s own
    /// `Drop`, via `Arc`'s refcounting) — so evicting/removing the
    /// `CacheEntry` (e.g. `ModelCache::get_or_load`'s stale-fingerprint
    /// path, or `evict_one`) can never decrement `reserved_memory` while a
    /// `ModelGuard` still holds this model's device tensors resident across
    /// a forward pass. A permit owned solely by the `CacheEntry` would be
    /// released on removal regardless of outstanding guards — freeing
    /// budget for memory still occupied (double-booking).
    ///
    /// `Option`-wrapped so `Drop::drop` can release this clone — via
    /// [`Option::take`] — strictly BEFORE the `ref_count` decrement below,
    /// rather than relying on Rust's field-declaration-order drop (which
    /// runs field drops only AFTER the `Drop` impl's body returns).
    /// Without this ordering, the body's `fetch_sub` could make
    /// `ref_count == 0` visible to a concurrent `evict_one` while this
    /// guard's permit clone was still outstanding (the struct field hadn't
    /// dropped yet) — `evict_one` would then remove the `CacheEntry`, drop
    /// only ITS clone, and claim `true` (progress) even though the
    /// reservation was NOT actually released (this clone was still alive).
    /// Releasing the permit first makes "permit released" happen-before
    /// "ref_count == 0 is observable", so any `evict_one` that observes
    /// `ref_count == 0` for this entry is guaranteed the permit clone
    /// backing this guard is already gone.
    _gpu_permit: Option<Arc<crate::concurrency::GpuPermit>>,
    /// A clone of `ModelCache`'s cache-level admission [`tokio::sync::Notify`].
    ///
    /// **Why this cannot simply reuse `GpuScheduler`'s own release notify**
    /// (`GpuPermit::drop`'s `scheduler.notify.notify_waiters()`): that fires
    /// only when a `GpuPermit`'s LAST `Arc` clone actually drops. But this
    /// guard's `_gpu_permit` clone is never the last one while its
    /// `CacheEntry` is still present in the cache — the `CacheEntry` always
    /// retains its own clone (see `_gpu_permit`'s doc) until something
    /// explicitly removes the entry. So `ref_count` reaching zero for a
    /// STILL-CACHED entry — the transition that makes `evict_one` newly
    /// eligible to reclaim it — is invisible to `GpuScheduler`'s notify: no
    /// `Arc<GpuPermit>` clone drop event happens at all, only an atomic
    /// `ref_count` decrement. A waiter parked purely on
    /// `GpuScheduler::acquire`'s internal wait (or its notify directly) is
    /// then a permanent liveness hole: budget sized to one resident copy, A
    /// holds M1's guard, B's `do_load` fails to admit M2, evicts nothing (M1
    /// has `ref_count == 1`), and parks; A drops its guard, `ref_count` hits
    /// zero, M1 becomes evictable — but no `GpuPermit` clone ever dropped
    /// (the `CacheEntry`'s own clone is still live), so `notify_waiters()`
    /// is never called and B hangs forever.
    ///
    /// This field is the wake for that transition: `Drop` notifies it, unconditionally,
    /// AFTER the `ref_count` decrement below. `ModelCache::do_load`'s
    /// admission loop registers a `Notified` future on this notify (plus the
    /// `GpuScheduler`-level one) BEFORE each `try_acquire`/`evict_one` pass,
    /// so it wakes and re-runs the full admission loop on either transition.
    /// See `ModelCache::do_load`'s admission loop for the full wake-set
    /// enumeration.
    admission_notify: Arc<tokio::sync::Notify>,
    /// The device this model is resident on: where its forwards are admitted.
    device: Arc<crate::concurrency::GpuScheduler>,
}

impl ModelGuard {
    /// Construct a guard from its shared handles. Centralised so every
    /// caller gets the `Some(..)`-wrapped permit uniformly (see
    /// `_gpu_permit`'s doc for why the wrapping exists).
    pub(crate) fn new(
        model: Arc<LoadedModel>,
        ref_count: Arc<AtomicUsize>,
        gpu_permit: Arc<crate::concurrency::GpuPermit>,
        admission_notify: Arc<tokio::sync::Notify>,
    ) -> Self {
        Self {
            model,
            ref_count,
            device: Arc::clone(gpu_permit.device()),
            _gpu_permit: Some(gpu_permit),
            admission_notify,
        }
    }

    /// The device this model is resident on. Its forward admission
    /// ([`GpuScheduler::admit_forward`](crate::concurrency::GpuScheduler::admit_forward))
    /// is shared by every guard of every model there, so it holds across
    /// plans and across partitions alike.
    pub fn device(&self) -> &Arc<crate::concurrency::GpuScheduler> {
        &self.device
    }
}

impl Drop for ModelGuard {
    fn drop(&mut self) {
        // Release the permit clone BEFORE the ref_count decrement becomes
        // visible (see `_gpu_permit`'s doc) — the ordering is load-bearing:
        // it establishes a happens-before between "this guard's permit clone
        // is gone" and "ref_count == 0 may be observed by a concurrent
        // evict_one".
        drop(self._gpu_permit.take());
        self.ref_count.fetch_sub(1, Ordering::Release);
        // Signal the admission-wake notify
        // AFTER `ref_count` is visibly decremented, so any admission loop
        // this wakes observes the up-to-date `ref_count` when it re-checks
        // `evict_one`'s eligibility condition. Unconditional (every guard
        // drop signals, not just the one that happens to reach zero) —
        // spurious wakes only cost a cheap re-check of `try_acquire` +
        // `evict_one`'s idle scan, never a correctness problem, and a
        // conditional signal here would have to re-derive the same
        // "is this really the transition that matters" answer `evict_one`
        // already computes authoritatively.
        self.admission_notify.notify_waiters();
    }
}
