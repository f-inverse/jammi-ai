//! GGUF/k-quant loading and residency-estimation helpers shared by the
//! resolver (header-only residency estimation, issue #351 pin V5) and the
//! candle backend's GGUF load path (pin V6) — ONE definition of which
//! tensor names are "matmul-site" for a supported architecture and which
//! GGML dtypes this workspace can represent, so the two call sites can
//! never silently drift on either question.
//!
//! # The three supported architectures
//!
//! GGUF/quantized serving is threaded only through the three text towers
//! `jammi_encoders::FrozenWeightLookup` was wired into at wave 1: BERT-family
//! (`bert`/`roberta`/`camembert`/`xlm-roberta`), DistilBERT, and ModernBERT.
//! Every other `model_type` (OpenCLIP, HF-CLAP) is a typed refusal at the
//! candle backend's load dispatch — see `CandleBackend::load`'s own GGUF
//! branch.
//!
//! # Matmul-site vs. everything else
//!
//! A "matmul-site" tensor is a weight (or bias) that
//! `jammi_encoders::{bert,distilbert,modernbert}`'s own `LoraSite`/
//! `resolve_base` construction routes through `FrozenWeightLookup` — the six
//! (BERT/DistilBERT) or four (ModernBERT, bias-free) per-layer linear
//! modules. A matmul-site tensor stored at a genuinely block-quantized GGML
//! dtype loads as [`jammi_lora::FrozenBase::Quantized`] — an `Arc<QTensor>`
//! that NEVER gets dequantized at load, staying resident in its compressed
//! form. Every OTHER tensor — embeddings, LayerNorms, classifier/NER heads,
//! and a matmul-site tensor that happens to be stored densely (`F32`/`F16`/
//! `BF16`, not k-quantized) — is dequantized to the model's compute dtype at
//! load and fed to the encoder's ordinary `VarBuilder`-backed Dense path via
//! a synthesized in-memory safetensors file (see [`load_gguf_backbone`]).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use jammi_db::error::{JammiError, Result};
use jammi_lora::{FrozenBase, QuantizedLinear};
use jammi_numerics::WeightQuantization;

/// The row-padding candle's own quantized-weight CUDA loader applies to
/// every resident matmul-site `QTensor` (`candle-core` 0.11.0
/// `src/quantized/cuda.rs:38`, `pub const MATRIX_ROW_PADDING: usize = 512;`
/// — verified against that source directly, matching `load_quantized`'s own
/// `data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size()`
/// formula at the same file's `load_quantized`). NOT a re-export: that
/// constant lives inside `candle_core::quantized::cuda`, which candle-core
/// only compiles `pub` under candle-core's OWN `cuda` feature (a private
/// `dummy_cuda`-backed stub otherwise) — residency has to estimate
/// correctly in a CPU-only (non-`cuda`-feature) build too, so this is an
/// explicit in-code copy, the same "verified against source, not
/// re-exported" convention `jammi_numerics::WeightQuantization::gguf_wire_id`
/// documents for the analogous GGML wire-ID table.
const MATRIX_ROW_PADDING: usize = 512;

/// The three text-tower architectures issue #351 threads GGUF loading
/// through (module doc). Every OTHER `model_type` is a typed refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufArchitecture {
    Bert,
    DistilBert,
    ModernBert,
}

impl GgufArchitecture {
    /// `None` for any `model_type` this workspace does not thread GGUF
    /// loading through — the caller must refuse rather than guess.
    pub(crate) fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "bert" | "roberta" | "camembert" | "xlm-roberta" => Some(Self::Bert),
            "distilbert" => Some(Self::DistilBert),
            "modernbert" => Some(Self::ModernBert),
            _ => None,
        }
    }
}

/// Byte width of one element at `precision` — the SAME sizes
/// `jammi_encoders::compute_precision_to_dtype` maps to candle `DType`s (`F32`
/// = 4, `F16`/`BF16` = 2), named here as a plain `usize` so [`estimate_gguf_residency`]
/// (which runs at RESOLVE time, before a `Device`/`DType` is ever selected)
/// stays candle-`DType`-free.
pub(crate) fn compute_precision_byte_size(precision: jammi_numerics::ComputePrecision) -> usize {
    match precision {
        jammi_numerics::ComputePrecision::F32 => 4,
        jammi_numerics::ComputePrecision::F16 | jammi_numerics::ComputePrecision::BF16 => 2,
    }
}

/// BERT-family checkpoints wrap every tensor under an optional `"bert."`
/// prefix (`BertForX` vs. a raw `BertModel` checkpoint) — mirrors
/// `jammi_encoders::bert::BertBuilder::build`'s own
/// `frozen_vb.contains_tensor("bert.embeddings.word_embeddings.weight")`
/// probe, applied here against the GGUF tensor-name set instead of a
/// safetensors `VarBuilder`.
fn bert_prefix(tensor_infos: &HashMap<String, gguf_file::TensorInfo>) -> &'static str {
    if tensor_infos.contains_key("bert.embeddings.word_embeddings.weight") {
        "bert."
    } else {
        ""
    }
}

/// Every matmul-site linear module's fully-qualified tensor-name PREFIX
/// (mirrors `jammi_encoders::{bert,distilbert,modernbert}`'s own
/// `LoraSite`/`resolve_base` per-layer module names exactly — the string
/// `module_vb.prefix()` returns at each of the six/four sites) plus whether
/// that site carries a bias tensor, for `num_layers` transformer layers of
/// `arch`. `.weight`/`.bias` are NOT appended — callers append the suffix
/// themselves, matching `candle_nn::VarBuilder`'s own convention (a
/// `module_vb.prefix()` never carries the leaf tensor name).
pub(crate) fn matmul_site_names(
    arch: GgufArchitecture,
    tensor_infos: &HashMap<String, gguf_file::TensorInfo>,
    num_layers: usize,
) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    match arch {
        GgufArchitecture::Bert => {
            let prefix = bert_prefix(tensor_infos);
            for n in 0..num_layers {
                let layer = format!("{prefix}encoder.layer.{n}");
                out.push((format!("{layer}.attention.self.query"), true));
                out.push((format!("{layer}.attention.self.key"), true));
                out.push((format!("{layer}.attention.self.value"), true));
                out.push((format!("{layer}.attention.output.dense"), true));
                out.push((format!("{layer}.intermediate.dense"), true));
                out.push((format!("{layer}.output.dense"), true));
            }
        }
        GgufArchitecture::DistilBert => {
            for n in 0..num_layers {
                let layer = format!("distilbert.transformer.layer.{n}");
                out.push((format!("{layer}.attention.q_lin"), true));
                out.push((format!("{layer}.attention.k_lin"), true));
                out.push((format!("{layer}.attention.v_lin"), true));
                out.push((format!("{layer}.attention.out_lin"), true));
                out.push((format!("{layer}.ffn.lin1"), true));
                out.push((format!("{layer}.ffn.lin2"), true));
            }
        }
        GgufArchitecture::ModernBert => {
            for n in 0..num_layers {
                let layer = format!("model.layers.{n}");
                out.push((format!("{layer}.attn.Wqkv"), false));
                out.push((format!("{layer}.attn.Wo"), false));
                out.push((format!("{layer}.mlp.Wi"), false));
                out.push((format!("{layer}.mlp.Wo"), false));
            }
        }
    }
    out
}

/// [`WeightQuantization`] for a genuinely block-quantized GGML dtype;
/// `None` for a dense-stored (`F32`/`F16`/`BF16`) tensor or any other GGML
/// dtype this workspace does not name (`Q8_1`/`Q8K` — module doc of
/// `jammi_numerics::WeightQuantization`). A plain function at the candle
/// boundary (never a `From` impl — the orphan rule, per that module's own
/// doc: neither type is local to either crate).
pub(crate) fn weight_quantization_from_ggml(dtype: GgmlDType) -> Option<WeightQuantization> {
    match dtype {
        GgmlDType::Q4_0 => Some(WeightQuantization::Q4_0),
        GgmlDType::Q4_1 => Some(WeightQuantization::Q4_1),
        GgmlDType::Q5_0 => Some(WeightQuantization::Q5_0),
        GgmlDType::Q5_1 => Some(WeightQuantization::Q5_1),
        GgmlDType::Q8_0 => Some(WeightQuantization::Q8_0),
        GgmlDType::Q2K => Some(WeightQuantization::Q2K),
        GgmlDType::Q3K => Some(WeightQuantization::Q3K),
        GgmlDType::Q4K => Some(WeightQuantization::Q4K),
        GgmlDType::Q5K => Some(WeightQuantization::Q5K),
        GgmlDType::Q6K => Some(WeightQuantization::Q6K),
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8_1 | GgmlDType::Q8K => {
            None
        }
    }
}

/// Whether `dtype` is a dense-stored (non-block-quantized) GGML tensor
/// format this workspace can load as a plain candle `Tensor`.
fn is_dense_stored(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
}

fn refusal(model_id: &str, message: String) -> JammiError {
    JammiError::Model {
        model_id: model_id.to_string(),
        message,
    }
}

/// Open `path` and parse ONLY its GGUF header (magic / metadata /
/// tensor_infos) — `gguf_file::Content::read` never reads tensor DATA
/// (candle-core 0.11.0: the byte-reading work happens in `TensorInfo::read`,
/// a SEPARATE call `Content::tensor` makes per-name, on demand — verified at
/// source). A parse failure here is a typed RESOLVER refusal (issue #351,
/// pin V5) — never a silent fallback to the raw file size, which would
/// under-report residency for a file whose header lies about its own tensor
/// count/shapes.
fn read_gguf_header(path: &Path, model_id: &str) -> Result<gguf_file::Content> {
    let file = std::fs::File::open(path)
        .map_err(|e| refusal(model_id, format!("failed to open {path:?}: {e}")))?;
    let mut reader = std::io::BufReader::new(file);
    gguf_file::Content::read(&mut reader).map_err(|e| {
        refusal(
            model_id,
            format!("failed to parse GGUF header of {path:?}: {e}"),
        )
    })
}

/// RESOLVE-TIME residency estimation (issue #351, pin V5): parses ONLY
/// `path`'s GGUF header (never tensor data — see [`read_gguf_header`]) and
/// returns a CONSERVATIVE (>= true residency) byte figure:
///
/// - a matmul-site tensor stored at a genuinely block-quantized dtype stays
///   resident as a `QTensor` — its own storage size (`elem_count /
///   block_size * type_size`) plus the CUDA row-padding candle's quantized
///   CUDA loader always applies ([`MATRIX_ROW_PADDING`], this module's own
///   doc) — applied unconditionally (not gated on the actual target
///   device), because this estimate must stay conservative regardless of
///   which device the model eventually loads on;
/// - every OTHER tensor (a non-matmul-site tensor of any stored dtype, or a
///   matmul-site tensor stored densely) is dequantized to `target_dtype_bytes`
///   at load — resident as a plain `Tensor`, `elem_count * target_dtype_bytes`;
/// - PLUS the single largest per-tensor DEQUANTIZE transient
///   (`stored_bytes + 4·N + target_dtype_bytes·N` — candle's own
///   `QTensor::dequantize` always produces an `F32` tensor FIRST, `4·N`,
///   before any narrowing cast to `target_dtype`, so both buffers can be
///   momentarily resident together) across every dequantized tensor — the
///   worst-case peak overlap during load, added ONCE (not per-tensor).
///
/// A malformed header (an element count that is not a multiple of its
/// dtype's own block size — the same domain check `TensorInfo::read` itself
/// applies when it actually reads bytes) or an unrepresented GGML dtype
/// (neither a `WeightQuantization` k-quant format nor `F32`/`F16`/`BF16`) is
/// a typed refusal here too, at RESOLVE time — never silently estimated
/// past a shape this workspace cannot actually load.
pub(crate) fn estimate_gguf_residency(
    path: &Path,
    model_config: &serde_json::Value,
    target_dtype_bytes: usize,
    model_id: &str,
) -> Result<usize> {
    let content = read_gguf_header(path, model_id)?;

    let model_type = model_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("bert");
    let num_layers = model_config
        .get("num_hidden_layers")
        .or_else(|| model_config.get("num_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let matmul_site: HashSet<String> = match GgufArchitecture::from_model_type(model_type) {
        Some(arch) => matmul_site_names(arch, &content.tensor_infos, num_layers)
            .into_iter()
            .flat_map(|(name, has_bias)| {
                let mut names = vec![format!("{name}.weight")];
                if has_bias {
                    names.push(format!("{name}.bias"));
                }
                names
            })
            .collect(),
        // An unsupported architecture never actually loads (`CandleBackend::load`
        // refuses it outright) — every tensor is conservatively costed as
        // dense-resident here rather than special-cased, since no matmul
        // site distinction is load-bearing for an estimate that will never
        // back a real load.
        None => HashSet::new(),
    };

    let mut total: u128 = 0;
    let mut max_transient: u128 = 0;
    for (name, info) in &content.tensor_infos {
        let dtype = info.ggml_dtype;
        let elem_count = info.shape.elem_count() as u128;
        let block_size = dtype.block_size() as u128;
        let type_size = dtype.type_size() as u128;
        if block_size == 0 || !elem_count.is_multiple_of(block_size) {
            return Err(refusal(
                model_id,
                format!(
                    "GGUF tensor '{name}' element count {elem_count} is not a multiple of its \
                     dtype's block size {block_size} in {path:?} — malformed header"
                ),
            ));
        }
        let is_kquant = weight_quantization_from_ggml(dtype).is_some();
        if !is_kquant && !is_dense_stored(dtype) {
            return Err(refusal(
                model_id,
                format!(
                    "GGUF tensor '{name}' in {path:?} has unsupported dtype {dtype:?} — expected \
                     a k-quant format (q4_0..q6k) or F32/F16/BF16"
                ),
            ));
        }
        let stored_bytes = elem_count / block_size * type_size;
        if is_kquant && matmul_site.contains(name) {
            let padding = MATRIX_ROW_PADDING as u128 * type_size / block_size;
            total += stored_bytes + padding;
        } else {
            let dense_bytes = elem_count * target_dtype_bytes as u128;
            total += dense_bytes;
            let transient = stored_bytes + elem_count * 4 + dense_bytes;
            if transient > max_transient {
                max_transient = transient;
            }
        }
    }
    total += max_transient;
    Ok(total as usize)
}

/// Whether a GGUF weight tensor is resident as a compressed `QTensor`
/// (matmul-site, genuinely quantized) or a dense candle `Tensor` (every
/// other case — module doc).
enum SiteWeight {
    Quantized(Arc<QTensor>),
    Dense(Tensor),
}

/// Everything the candle backend's GGUF load path needs, built ONCE per
/// load from the GGUF file's actual tensor DATA (unlike
/// [`estimate_gguf_residency`], which reads ONLY the header): the
/// per-matmul-site weight/bias map [`GgufBackbone::lookup`] exposes as a
/// [`jammi_encoders::FrozenWeightLookup`]-shaped closure, a temp safetensors
/// file carrying every OTHER tensor densified to the compute dtype
/// (embeddings, norms, non-matmul-site biases, classifier/NER heads —
/// whatever else the checkpoint carries), and the MODAL
/// [`WeightQuantization`] among the matmul-site tensors — the value
/// `ModelIdentity.quantization` reports (issue #351, pin Δ2).
pub(crate) struct GgufBackbone {
    sites: HashMap<String, (SiteWeight, Option<Tensor>)>,
    /// Kept alive for the duration of the `Bert`/`DistilBert`/`ModernBert`
    /// `build()` call: `VarBuilder::from_mmaped_safetensors` mmaps
    /// [`Self::densified_path`], but candle-core 0.11.0's own
    /// `st::TensorView::load` COPIES each tensor's bytes into owned storage
    /// at read time (never a zero-copy view over the mmap) — so this
    /// tempdir is safe to drop once `build()` returns, but MUST outlive
    /// that call.
    _scratch: tempfile::TempDir,
    pub(crate) densified_path: PathBuf,
    pub(crate) modal_quantization: Option<WeightQuantization>,
}

impl GgufBackbone {
    /// A [`jammi_encoders::FrozenWeightLookup`]-shaped closure over
    /// `self.sites` — `Ok(Some(..))` for every matmul-site name this GGUF
    /// checkpoint covers, `Ok(None)` for anything else (the encoder builder
    /// falls through to its own Dense-from-`VarBuilder` load against
    /// [`Self::densified_path`], per `FrozenWeightLookup`'s own module doc).
    /// Constructs a FRESH `FrozenBase` per call rather than storing one
    /// (`FrozenBase` carries no `Clone`) — `Arc<QTensor>::clone`/
    /// `Tensor::clone` are both cheap (refcount/view clones), and each site
    /// name is looked up exactly once per load, so this is not a
    /// meaningful cost.
    pub(crate) fn lookup(
        &self,
    ) -> impl Fn(&str) -> std::result::Result<Option<FrozenBase>, jammi_encoders::EncoderError> + '_
    {
        move |name: &str| match self.sites.get(name) {
            None => Ok(None),
            Some((SiteWeight::Quantized(w), bias)) => {
                let ql = QuantizedLinear::new(w.clone(), bias.clone())?;
                Ok(Some(FrozenBase::Quantized(ql)))
            }
            Some((SiteWeight::Dense(w), bias)) => Ok(Some(FrozenBase::Dense(
                candle_nn::Linear::new(w.clone(), bias.clone()),
            ))),
        }
    }
}

/// Build a [`GgufBackbone`] for `arch` from `weights_path`'s GGUF tensor
/// DATA (unlike [`estimate_gguf_residency`], this reads and — for
/// non-matmul-site tensors — dequantizes every tensor's bytes).
///
/// Refuses (typed, LISTING every missing name) when any matmul-site tensor
/// this architecture's `num_layers` requires is absent from the GGUF file,
/// and refuses (typed, naming tensor + dtype) when any tensor carries a
/// GGML dtype this workspace does not represent (neither a k-quant
/// [`WeightQuantization`] format nor `F32`/`F16`/`BF16`).
pub(crate) fn load_gguf_backbone(
    weights_path: &Path,
    arch: GgufArchitecture,
    num_layers: usize,
    target_dtype: DType,
    device: &Device,
    model_id: &str,
) -> Result<GgufBackbone> {
    let file = std::fs::File::open(weights_path)
        .map_err(|e| refusal(model_id, format!("failed to open {weights_path:?}: {e}")))?;
    let mut reader = std::io::BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader).map_err(|e| {
        refusal(
            model_id,
            format!("failed to parse GGUF header of {weights_path:?}: {e}"),
        )
    })?;

    // Dtype-support pre-flight over EVERY tensor in the file, not only
    // matmul sites — the densified path below must be able to dequantize
    // whatever it finds, and a typed refusal naming the offending
    // tensor+dtype up front is clearer than a deep candle panic/error
    // surfacing mid-densify.
    for (name, info) in &content.tensor_infos {
        let dtype = info.ggml_dtype;
        if weight_quantization_from_ggml(dtype).is_none() && !is_dense_stored(dtype) {
            return Err(refusal(
                model_id,
                format!(
                    "GGUF tensor '{name}' in {weights_path:?} has unsupported dtype {dtype:?} — \
                     expected a k-quant format (q4_0..q6k) or F32/F16/BF16"
                ),
            ));
        }
    }

    let site_specs = matmul_site_names(arch, &content.tensor_infos, num_layers);
    let mut missing = Vec::new();
    for (prefix, has_bias) in &site_specs {
        if !content
            .tensor_infos
            .contains_key(&format!("{prefix}.weight"))
        {
            missing.push(format!("{prefix}.weight"));
        }
        if *has_bias && !content.tensor_infos.contains_key(&format!("{prefix}.bias")) {
            missing.push(format!("{prefix}.bias"));
        }
    }
    if !missing.is_empty() {
        return Err(refusal(
            model_id,
            format!(
                "GGUF checkpoint {weights_path:?} is missing {} required tensor(s) for a \
                 {arch:?} backbone with {num_layers} layers: {}",
                missing.len(),
                missing.join(", ")
            ),
        ));
    }

    let site_names: HashSet<String> = site_specs
        .iter()
        .flat_map(|(prefix, has_bias)| {
            let mut names = vec![format!("{prefix}.weight")];
            if *has_bias {
                names.push(format!("{prefix}.bias"));
            }
            names
        })
        .collect();

    let mut sites: HashMap<String, (SiteWeight, Option<Tensor>)> = HashMap::new();
    let mut modal_counts: HashMap<WeightQuantization, usize> = HashMap::new();
    for (prefix, has_bias) in &site_specs {
        let weight_name = format!("{prefix}.weight");
        let qtensor = content
            .tensor(&mut reader, &weight_name, device)
            .map_err(|e| {
                refusal(
                    model_id,
                    format!("failed to read GGUF tensor '{weight_name}': {e}"),
                )
            })?;
        let bias = if *has_bias {
            let bias_name = format!("{prefix}.bias");
            let bias_q = content
                .tensor(&mut reader, &bias_name, device)
                .map_err(|e| {
                    refusal(
                        model_id,
                        format!("failed to read GGUF tensor '{bias_name}': {e}"),
                    )
                })?;
            let bias_dense = bias_q
                .dequantize(device)
                .map_err(|e| refusal(model_id, format!("failed to dequantize '{bias_name}': {e}")))?
                .to_dtype(target_dtype)
                .map_err(|e| refusal(model_id, format!("failed to cast '{bias_name}': {e}")))?;
            Some(bias_dense)
        } else {
            None
        };
        let ggml_dtype = qtensor.dtype();
        let weight = if let Some(wq) = weight_quantization_from_ggml(ggml_dtype) {
            *modal_counts.entry(wq).or_insert(0) += 1;
            SiteWeight::Quantized(Arc::new(qtensor))
        } else {
            let dense = qtensor
                .dequantize(device)
                .map_err(|e| {
                    refusal(
                        model_id,
                        format!("failed to dequantize '{weight_name}': {e}"),
                    )
                })?
                .to_dtype(target_dtype)
                .map_err(|e| refusal(model_id, format!("failed to cast '{weight_name}': {e}")))?;
            SiteWeight::Dense(dense)
        };
        sites.insert(prefix.clone(), (weight, bias));
    }

    // MODAL quantized dtype among matmul-site tensors, ties broken by
    // `WeightQuantization`'s own `Ord` (that type's module doc — the GGUF
    // wire ID).
    let modal_quantization = modal_counts
        .into_iter()
        .max_by(|(a_wq, a_n), (b_wq, b_n)| a_n.cmp(b_n).then_with(|| a_wq.cmp(b_wq)))
        .map(|(wq, _)| wq);

    // Densify every OTHER tensor (embeddings, norms, classifier/NER heads,
    // any bias not already claimed above) into a temp safetensors file the
    // encoder's `VarBuilder::from_mmaped_safetensors` reads unmodified —
    // module doc's "byte-identical when unset" seam applies transitively
    // here: every construction site that never consults `weight_source`
    // (embeddings, LayerNorms, classifier heads) is untouched, reading this
    // densified file exactly the way it reads a real safetensors checkpoint.
    let scratch = tempfile::tempdir()
        .map_err(|e| refusal(model_id, format!("failed to create GGUF scratch dir: {e}")))?;
    let densified_path = scratch.path().join("densified.safetensors");
    let mut dense_map: HashMap<String, Tensor> = HashMap::new();
    for name in content.tensor_infos.keys() {
        if site_names.contains(name) {
            continue;
        }
        let qtensor = content.tensor(&mut reader, name, device).map_err(|e| {
            refusal(
                model_id,
                format!("failed to read GGUF tensor '{name}': {e}"),
            )
        })?;
        let dense = qtensor
            .dequantize(device)
            .map_err(|e| refusal(model_id, format!("failed to dequantize '{name}': {e}")))?
            .to_dtype(target_dtype)
            .map_err(|e| refusal(model_id, format!("failed to cast '{name}': {e}")))?;
        dense_map.insert(name.clone(), dense);
    }
    candle_core::safetensors::save(&dense_map, &densified_path).map_err(|e| {
        refusal(
            model_id,
            format!("failed to write GGUF-derived densified safetensors: {e}"),
        )
    })?;

    Ok(GgufBackbone {
        sites,
        _scratch: scratch,
        densified_path,
        modal_quantization,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `weight_quantization_from_ggml` is a BIJECTION between the ten
    /// block-quantized `GgmlDType` variants and `WeightQuantization::ALL`
    /// (every variant covered exactly once — `GgmlDType::to_u32`/`from_u32`
    /// are `pub(crate)` inside candle-core, so this cannot round-trip
    /// through candle's own wire-ID encoding directly; the wire-ID table
    /// ITSELF is exhaustively pinned against candle's source by
    /// `jammi_numerics::WeightQuantization`'s own
    /// `gguf_wire_id_matches_the_verified_candle_table` test), and reports
    /// `None` for every dense-stored / unnamed dtype.
    #[test]
    fn weight_quantization_from_ggml_covers_every_named_variant() {
        let ggml_variants = [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ];
        let mut mapped: Vec<WeightQuantization> = ggml_variants
            .into_iter()
            .map(|ggml| weight_quantization_from_ggml(ggml).unwrap_or_else(|| panic!("{ggml:?}")))
            .collect();
        mapped.sort();
        let mut expected = WeightQuantization::ALL.to_vec();
        expected.sort();
        assert_eq!(mapped, expected);
        assert_eq!(weight_quantization_from_ggml(GgmlDType::F32), None);
        assert_eq!(weight_quantization_from_ggml(GgmlDType::F16), None);
        assert_eq!(weight_quantization_from_ggml(GgmlDType::BF16), None);
        assert_eq!(weight_quantization_from_ggml(GgmlDType::Q8_1), None);
        assert_eq!(weight_quantization_from_ggml(GgmlDType::Q8K), None);
    }

    #[test]
    fn compute_precision_byte_size_matches_encoder_dtype_widths() {
        assert_eq!(
            compute_precision_byte_size(jammi_numerics::ComputePrecision::F32),
            4
        );
        assert_eq!(
            compute_precision_byte_size(jammi_numerics::ComputePrecision::F16),
            2
        );
        assert_eq!(
            compute_precision_byte_size(jammi_numerics::ComputePrecision::BF16),
            2
        );
    }

    #[test]
    fn architecture_from_model_type_covers_the_three_supported_families() {
        for name in ["bert", "roberta", "camembert", "xlm-roberta"] {
            assert_eq!(
                GgufArchitecture::from_model_type(name),
                Some(GgufArchitecture::Bert)
            );
        }
        assert_eq!(
            GgufArchitecture::from_model_type("distilbert"),
            Some(GgufArchitecture::DistilBert)
        );
        assert_eq!(
            GgufArchitecture::from_model_type("modernbert"),
            Some(GgufArchitecture::ModernBert)
        );
        assert_eq!(GgufArchitecture::from_model_type("clip_audio_model"), None);
        assert_eq!(GgufArchitecture::from_model_type("open_clip"), None);
    }

    #[test]
    fn matmul_site_names_bert_uses_bare_prefix_without_the_bert_wrapper_tensor() {
        let infos: HashMap<String, gguf_file::TensorInfo> = HashMap::new();
        let names = matmul_site_names(GgufArchitecture::Bert, &infos, 1);
        assert!(names
            .iter()
            .any(|(n, has_bias)| n == "encoder.layer.0.attention.self.query" && *has_bias));
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn matmul_site_names_distilbert_and_modernbert_have_the_expected_counts_and_bias_flags() {
        let infos: HashMap<String, gguf_file::TensorInfo> = HashMap::new();
        let db = matmul_site_names(GgufArchitecture::DistilBert, &infos, 2);
        assert_eq!(db.len(), 12);
        assert!(db.iter().all(|(_, has_bias)| *has_bias));

        let mb = matmul_site_names(GgufArchitecture::ModernBert, &infos, 2);
        assert_eq!(mb.len(), 8);
        assert!(mb.iter().all(|(_, has_bias)| !*has_bias));
        assert!(mb.iter().any(|(n, _)| n == "model.layers.0.attn.Wqkv"));
    }
}
