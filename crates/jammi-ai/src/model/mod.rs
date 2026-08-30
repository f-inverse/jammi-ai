pub mod backend;
pub mod cache;
pub mod clip_bpe;
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
use backend::ort::OrtModel;
use jammi_db::error::{JammiError, Result};
use serde::{Deserialize, Serialize};

use crate::inference::adapter::BackendOutput;

/// Unique identifier for a loaded model, used as cache key.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Explicit model source — the user declares where to load from, no fallback.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelSource {
    /// A HuggingFace Hub repository (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).
    HuggingFace(String),
    /// A local directory containing model files (config.json + weights).
    Local(PathBuf),
}

impl ModelSource {
    /// Create a HuggingFace Hub source.
    pub fn hf(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace(repo_id.into())
    }

    /// Create a local filesystem source.
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local(path.into())
    }

    /// Parse a user-provided model ID string into a ModelSource.
    ///
    /// Local filesystem forms follow the same convention as source URLs
    /// ([`StorageUrl`](jammi_db::storage::StorageUrl)): a `file://` URI or a
    /// filesystem path is local; a bare `owner/repo` is a Hub id.
    ///
    /// - `"local:/path/to/model"` or `"file:///path/to/model"` → `Local(path)`
    /// - a filesystem path — `"/abs/model"`, `"./model"`, `"../model"` → `Local(path)`
    /// - `"hf://owner/repo"` → `HuggingFace("owner/repo")` (strips `hf://`)
    /// - `"owner/repo"` → `HuggingFace("owner/repo")`
    ///
    /// A local path is resolved against the filesystem of the host running the
    /// engine (the server, for a remote client), so it must exist there.
    pub fn parse(id: &str) -> Self {
        if let Some(path) = id.strip_prefix("local:") {
            Self::Local(PathBuf::from(path))
        } else if let Some(path) = id.strip_prefix("file://") {
            Self::Local(PathBuf::from(path))
        } else if let Some(repo_id) = id.strip_prefix("hf://") {
            Self::HuggingFace(repo_id.to_string())
        } else if id.starts_with('/') || id.starts_with("./") || id.starts_with("../") {
            Self::Local(PathBuf::from(id))
        } else {
            Self::HuggingFace(id.to_string())
        }
    }

    /// Reconstruct a ModelSource from a canonical name (as stored in result_tables).
    /// Absolute paths that exist on disk → Local, everything else → HuggingFace.
    pub fn from_canonical(canonical_name: &str) -> Self {
        let path = std::path::Path::new(canonical_name);
        if path.is_absolute() && path.exists() {
            Self::Local(path.to_path_buf())
        } else {
            Self::HuggingFace(canonical_name.to_string())
        }
    }
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace(repo_id) => write!(f, "{repo_id}"),
            Self::Local(path) => write!(f, "{}", path.display()),
        }
    }
}

impl From<&ModelSource> for ModelId {
    fn from(source: &ModelSource) -> Self {
        ModelId(source.to_string())
    }
}

/// Which backend to use for this model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// Candle — native Rust inference via safetensors weights.
    Candle,
    /// ONNX Runtime — cross-platform inference via ONNX models.
    Ort,
    /// HTTP — remote model endpoint (REST/gRPC).
    Http,
}

/// What task this model performs.
///
/// Re-exported from `jammi_db` so the engine — which owns the catalog
/// tables that persist this — and `jammi_ai` agree on the variant set and
/// on-disk spelling without `jammi_db` depending on `jammi_ai`.
pub use jammi_db::ModelTask;

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

/// A resolved model — files located, backend determined, NOT yet loaded.
pub struct ResolvedModel {
    /// HuggingFace or local identifier for this model.
    pub model_id: ModelId,
    /// Selected inference backend.
    pub backend: BackendType,
    /// ML task this model performs.
    pub task: ModelTask,
    /// Path to the model's `config.json`.
    pub config_path: std::path::PathBuf,
    /// Paths to weight files (safetensors shards or ONNX).
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

        // OpenCLIP format: model_cfg.vision_cfg with embed_dim at top level
        if let Some(model_cfg) = config.get("model_cfg") {
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
        let audio = config.get("audio_config").unwrap_or(config);
        if audio.get("model_type").and_then(|v| v.as_str()) != Some("clap_audio_model") {
            return None;
        }
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

/// A model loaded into memory, ready for inference.
pub enum LoadedModel {
    /// Loaded via the Candle backend (safetensors weights).
    Candle(Box<CandleModel>),
    /// Loaded via the ORT backend (ONNX weights).
    Ort(OrtModel),
}

impl LoadedModel {
    /// The backend kind that loaded this model, as the canonical lowercase token
    /// the materialization contract records in `ModelIdentity.backend`. A loaded
    /// model is always a native backend (`candle` / `ort`); the `http` backend
    /// serves remotely and is never a `LoadedModel`.
    pub fn backend_kind(&self) -> &'static str {
        match self {
            LoadedModel::Candle(_) => "candle",
            LoadedModel::Ort(_) => "ort",
        }
    }

    /// The effective inference compute precision this model was loaded at —
    /// the resolved per-model `config.json` override, or the global
    /// `GpuConfig::compute_precision` default. Output-affecting (an `F16`
    /// backbone emits different embedding/logit bytes than `F32`), so the
    /// materialization contract folds it into `ModelIdentity.compute_precision`
    /// alongside `backend_kind`. The ORT backend does not yet select a compute
    /// precision, so it always reports `F32`.
    pub fn compute_precision(&self) -> jammi_numerics::ComputePrecision {
        match self {
            LoadedModel::Candle(m) => m.compute_precision,
            LoadedModel::Ort(_) => jammi_numerics::ComputePrecision::F32,
        }
    }

    /// The pooling strategy the loaded text-embedding forward path ACTUALLY
    /// resolved to and applies — the SAME strategy
    /// `backend::candle::CandleTextForward::forward_pooled` pools with, read
    /// via `CandleModel::resolved_pooling` (unit-62 F-5': a bench/report
    /// consumer must read this off the loaded model, never transcribe a
    /// fixture-declared constant that could silently drift from what actually
    /// served). `None` when this loaded model has no pooling concept at all
    /// (a CLAP audio tower, an OpenCLIP text tower whose output is already
    /// pooled-and-projected, a classification head) or for the ORT backend
    /// (which does not yet resolve a text-embedding pooling wrapper).
    pub fn resolved_pooling(&self) -> Option<jammi_encoders::Pooling> {
        match self {
            LoadedModel::Candle(m) => m.resolved_pooling(),
            LoadedModel::Ort(_) => None,
        }
    }

    /// The model's content digest (esc-057, K7): a SHA-256 fold of the
    /// resolved model directory's config / `1_Pooling/config.json` /
    /// tokenizer / weights bytes, computed once at load time by
    /// `backend::candle::compute_model_content_digest`. Output-affecting
    /// (two directories that share one `model_id` but differ in any of those
    /// bytes must never collide on one `DefinitionHash`), so the
    /// materialization contract folds it into `ModelIdentity.content_digest`
    /// alongside `backend_kind` / `compute_precision`.
    ///
    /// The ORT backend never actually reaches a loaded state today
    /// (`OrtBackend::load` unconditionally errors — see `forward`'s identical
    /// stance below) — there is no local-directory digest to report for it,
    /// and `ModelContentDigest::Unavailable` is reserved for the
    /// external-producer import path (a categorically different "no local
    /// files at all" case), so this returns a typed refusal rather than
    /// misusing that reason.
    pub fn content_digest(&self) -> Result<jammi_db::store::manifest::ModelContentDigest> {
        match self {
            LoadedModel::Candle(m) => Ok(m.content_digest.clone()),
            LoadedModel::Ort(_) => Err(JammiError::Inference(
                "ORT content digest not available in this build".into(),
            )),
        }
    }

    /// Stat-only warm-cache staleness probe (esc-058). `ModelCache::get_or_load`'s
    /// fast path calls this before handing out the cached `Arc<LoadedModel>` —
    /// re-`stat`ing (never re-reading) the same file set `content_digest` was
    /// hashed from at load time and comparing `(len, mtime)` against the
    /// load-time snapshot.
    ///
    /// - `Ok(true)` — unchanged (or nothing local to check — see below):
    ///   serve from cache.
    /// - `Ok(false)` — at least one fingerprinted file diverged: the caller
    ///   must evict the entry and reload rather than serve.
    /// - `Err` — a fingerprinted file vanished or became unreadable between
    ///   load and this probe: a typed refusal (K2), never a silent "treat as
    ///   fresh".
    ///
    /// **Honest residual** (see `backend::candle::ModelFingerprint`'s own
    /// doc): `(len, mtime)` is a staleness TRIPWIRE, not a cryptographic
    /// guarantee — a same-length, same-mtime content swap is invisible to
    /// it. `content_digest`, recomputed fresh on every actual reload, remains
    /// the sole authoritative attestation of the bytes that were hashed;
    /// this probe only decides WHEN a reload is triggered.
    ///
    /// **The guarantee this provides is BOUNDED STALENESS, never per-hit
    /// freshness (unit-62 design pressure-test, corrected framing).** A call
    /// that reports `Ok(true)` proves this file set was unchanged AT THE
    /// INSTANT this probe ran — not that the `Arc<LoadedModel>` the caller
    /// then goes on to use stays fresh for the duration of that use.
    /// `ModelCache::get_or_load`'s returned [`ModelGuard`] is never
    /// revalidated again after this call returns: a mutation landing between
    /// this probe and the guard's actual forward pass (or landing during a
    /// long-held guard) is a TOCTOU window this type does not — and
    /// structurally cannot, being `stat`-only and synchronous with a single
    /// call — close. Treat every guard as "fresh as of load or last warm-hit
    /// probe", never "fresh for as long as I hold it." See
    /// `backend::candle::ModelFingerprint`'s doc for the narrow-contract
    /// scope this bound additionally sits within (unit 65's classes are
    /// entirely outside even this bounded guarantee).
    ///
    /// The ORT backend never actually reaches a loaded state today (see
    /// `content_digest`'s doc), and more generally a backend whose
    /// `content_digest` is `ModelContentDigest::Unavailable` (an
    /// external-producer model with no local directory at all) has nothing
    /// on disk that could go stale — for either, this vacuously reports
    /// fresh rather than refusing, since "no local files to check" is not a
    /// staleness condition.
    pub(crate) fn probe_freshness(&self) -> Result<bool> {
        match self {
            LoadedModel::Candle(m) => m.fingerprint.probe(),
            LoadedModel::Ort(_) => Ok(true),
        }
    }

    /// Estimate GPU memory for one inference batch.
    pub fn estimate_batch_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        match self {
            LoadedModel::Candle(m) => m.dimensions.estimate_activation_memory(batch_size, seq_len),
            LoadedModel::Ort(m) => m.dimensions.estimate_activation_memory(batch_size, seq_len),
        }
    }

    /// Output dimensionality of the model's embedding head, if known.
    ///
    /// For BERT-family encoders this is the transformer's `hidden_size`.
    /// For OpenCLIP-family models (vision and text towers) this is the
    /// projected shared-latent `embed_dim` — the dimension that vectors
    /// emitted by `generate_text_embeddings`, `generate_image_embeddings`,
    /// `encode_text_query`, and `encode_image_query` carry, and the
    /// dimension that cross-modal cosine similarity is computed in. It is
    /// not the per-tower hidden `width`; the in-tower hidden size is
    /// projected through `visual.proj` / `text_projection` before the
    /// embedding is exposed.
    pub fn embedding_dim(&self) -> Option<usize> {
        match self {
            LoadedModel::Candle(m) => Some(m.dimensions.hidden_size),
            LoadedModel::Ort(m) => Some(m.dimensions.hidden_size),
        }
    }

    /// The persisted predictive-distribution form of a reloaded regression head
    /// (`Gaussian` or `Quantile { levels }`), or `None` for a non-regression
    /// model. Serving reads this to select the `Infer` output adapter so a
    /// quantile-trained head is served as quantile points, never silently
    /// mis-decoded as a Gaussian `(mean, std)`. The ORT backend has no
    /// regression head, so it always reports `None`.
    pub fn regression_form(&self) -> Option<&crate::inference::adapter::DistributionForm> {
        match self {
            LoadedModel::Candle(m) => m.regression_form(),
            LoadedModel::Ort(_) => None,
        }
    }

    /// The persisted scaler's σ_y for a reloaded regression head, or `None` for a
    /// non-regression / no-scaler / ORT model. Serving reads this to scale a
    /// Gaussian head's served σ from the z-space the loss trained (σ_z ≈ 1) back
    /// to raw units (`σ_y·σ_z`) — the σ-axis half of the de-standardise contract
    /// (the mean/quantile axes carry σ_y in the backend's affine).
    pub fn regression_std_scale(&self) -> Option<f32> {
        match self {
            LoadedModel::Candle(m) => m.regression_std_scale(),
            LoadedModel::Ort(_) => None,
        }
    }

    /// TEST-ONLY non-vacuity seam: zero a loaded regression head's trained LoRA
    /// `B` factor so it regresses to its zero-initialised base and emits the
    /// scaler offset `μ_y` for every input (the untrained-head behaviour). No-op
    /// for a non-regression / ORT model. Used by the regression-surface tests to
    /// prove their group-separation assertion collapses to ≈0 when the head
    /// carries no learned signal. See
    /// [`super::backend::candle::CandleModel::zero_distribution_head_for_test`].
    #[doc(hidden)]
    pub fn zero_distribution_head_for_test(&mut self) {
        match self {
            LoadedModel::Candle(m) => m.zero_distribution_head_for_test(),
            LoadedModel::Ort(_) => {}
        }
    }

    /// Run forward pass on Arrow content columns. Returns raw output.
    pub fn forward(&self, content: &[ArrayRef], task: ModelTask) -> Result<BackendOutput> {
        match self {
            LoadedModel::Candle(m) => m.forward(content, task),
            LoadedModel::Ort(_) => Err(JammiError::Inference(
                "ORT forward pass not available in this build".into(),
            )),
        }
    }
}

/// RAII guard that decrements ref count on drop.
pub struct ModelGuard {
    /// Shared handle to the loaded model.
    pub model: Arc<LoadedModel>,
    ref_count: Arc<AtomicUsize>,
    /// Audit round 62, F-3 (reshaped by F-A in round 4): a clone of the SAME
    /// `Arc<GpuPermit>` the owning `CacheEntry` holds. A `GpuPermit` releases
    /// its reservation (`GpuScheduler::reserved_memory -= bytes`) only when
    /// its LAST `Arc` clone drops (`GpuPermit`'s own `Drop`, via `Arc`'s
    /// refcounting) — so evicting/removing the `CacheEntry` (e.g.
    /// `ModelCache::get_or_load`'s stale-fingerprint path, or `evict_one`)
    /// can never decrement `reserved_memory` while a `ModelGuard` still holds
    /// this model's device tensors resident across a forward pass. The
    /// pre-F-3 `GpuPermit` was owned solely by `CacheEntry`, so removing the
    /// entry released the permit unconditionally regardless of any
    /// outstanding guard — freeing budget for memory that was, in fact,
    /// still occupied (double-booking).
    ///
    /// `Option`-wrapped (F-A, round 4) so `Drop::drop` can release this
    /// clone — via [`Option::take`] — strictly BEFORE the `ref_count`
    /// decrement below, rather than relying on Rust's field-declaration-order
    /// drop (which runs field drops only AFTER the `Drop` impl's body
    /// returns). Without this reordering, the body's `fetch_sub` could make
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
    /// Unit 62, closure-audit BLOCK 1 (admission-wake liveness hole): a
    /// clone of `ModelCache`'s cache-level admission [`tokio::sync::Notify`].
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
    /// This field closes that hole: `Drop` notifies it, unconditionally,
    /// AFTER the `ref_count` decrement below. `ModelCache::do_load`'s
    /// admission loop registers a `Notified` future on this notify (plus the
    /// `GpuScheduler`-level one) BEFORE each `try_acquire`/`evict_one` pass,
    /// so it wakes and re-runs the full admission loop on either transition.
    /// See `ModelCache::do_load`'s admission loop for the full wake-set
    /// enumeration.
    admission_notify: Arc<tokio::sync::Notify>,
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
            _gpu_permit: Some(gpu_permit),
            admission_notify,
        }
    }
}

impl Drop for ModelGuard {
    fn drop(&mut self) {
        // Release the permit clone BEFORE the ref_count decrement becomes
        // visible (see `_gpu_permit`'s doc) — this ordering is the actual
        // fix for F-A: it is not merely presentational, it establishes a
        // happens-before between "this guard's permit clone is gone" and
        // "ref_count == 0 may be observed by a concurrent evict_one".
        drop(self._gpu_permit.take());
        self.ref_count.fetch_sub(1, Ordering::Release);
        // Unit 62, closure-audit BLOCK 1: signal the admission-wake notify
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hub_ids() {
        assert_eq!(
            ModelSource::parse("sentence-transformers/all-MiniLM-L6-v2"),
            ModelSource::HuggingFace("sentence-transformers/all-MiniLM-L6-v2".into())
        );
        assert_eq!(
            ModelSource::parse("hf://owner/repo"),
            ModelSource::HuggingFace("owner/repo".into())
        );
        // A bare name and a bare relative `a/b` stay Hub ids (ambiguous with
        // `owner/repo`); use `./` or `file://` to force a local relative path.
        assert_eq!(
            ModelSource::parse("bert-base-uncased"),
            ModelSource::HuggingFace("bert-base-uncased".into())
        );
        assert_eq!(
            ModelSource::parse("models/bert"),
            ModelSource::HuggingFace("models/bert".into())
        );
    }

    #[test]
    fn parse_local_paths() {
        let cases = [
            ("local:/opt/models/bert", "/opt/models/bert"),
            ("file:///opt/models/bert", "/opt/models/bert"),
            ("/opt/models/bert", "/opt/models/bert"),
            ("./models/bert", "./models/bert"),
            ("../models/bert", "../models/bert"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                ModelSource::parse(input),
                ModelSource::Local(PathBuf::from(expected)),
                "parsing {input:?}"
            );
        }
    }
}
